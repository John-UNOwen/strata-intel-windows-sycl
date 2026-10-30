// src/core/gguf_expert_source.cpp - see the header. Plain C++, no device code.
#include "strata/core/gguf_expert_source.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <utility>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <basetsd.h>
using ssize_t = SSIZE_T;   // MSVC has no ssize_t; the read loops are shared with the POSIX branch
#else
#include <fcntl.h>
#include <unistd.h>
#endif
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#include <thread>

namespace strata::core {

namespace {
constexpr size_t kRing = 512;   // blobs alive at once: the prompt path holds a layer's worth of streamed experts

// Windows port: fds_ carries a HANDLE as an int (handles are 32-bit safe). The handle is opened
// FILE_FLAG_OVERLAPPED: on a synchronous handle Windows serializes every ReadFile on the file object,
// so mirror()'s pool and the prompt stager threads read one at a time. Overlapped, each thread's read
// is its own request; the caller still waits for it (a per-thread event), so fd_pread stays a pread.
// STRATA_SYNC_PREAD=1 opens the handles synchronous again (the A/B arm).
#if defined(_WIN32)
int fd_open(const std::string& path) {
    static const bool sync_read = [] { const char* e = std::getenv("STRATA_SYNC_PREAD"); return e && *e == '1'; }();
    const HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                 FILE_ATTRIBUTE_NORMAL | (sync_read ? 0 : FILE_FLAG_OVERLAPPED), nullptr);
    return h == INVALID_HANDLE_VALUE ? -1 : (int) (intptr_t) h;
}
void fd_close(int fd) { if (fd >= 0) CloseHandle((HANDLE) (intptr_t) fd); }
ssize_t fd_pread(int fd, void* dst, size_t n, int64_t off) {
    struct Event {   // one manual-reset event per thread, reused for every read it makes
        HANDLE h = CreateEventA(nullptr, TRUE, FALSE, nullptr);
        ~Event() { if (h) CloseHandle(h); }
    };
    thread_local Event ev;
    if (ev.h == nullptr) return -1;
    const HANDLE h = (HANDLE) (intptr_t) fd;
    OVERLAPPED ov{};
    ov.Offset = (DWORD) (off & 0xffffffffll);
    ov.OffsetHigh = (DWORD) ((off >> 32) & 0xffffffffll);
    ov.hEvent = ev.h;   // without an event, completion would signal the handle every thread shares
    const DWORD want = (DWORD) std::min<size_t>(n, (size_t) 1 << 30);   // callers loop on short reads
    DWORD got = 0;
    if (!ReadFile(h, dst, want, nullptr, &ov)) {
        const DWORD e = GetLastError();
        if (e == ERROR_HANDLE_EOF) return 0;
        if (e != ERROR_IO_PENDING) return -1;
    }
    if (!GetOverlappedResult(h, &ov, &got, TRUE)) return GetLastError() == ERROR_HANDLE_EOF ? 0 : -1;
    return (ssize_t) got;
}
#else
int fd_open(const std::string& path) { return ::open(path.c_str(), O_RDONLY | O_CLOEXEC); }
void fd_close(int fd) { if (fd >= 0) ::close(fd); }
ssize_t fd_pread(int fd, void* dst, size_t n, int64_t off) { return ::pread(fd, dst, n, (off_t) off); }
#endif
}

GgufExpertSource::~GgufExpertSource() { close(); }

void GgufExpertSource::close() {
    for (auto& [c, odev] : mirror_chunks_)
        if (c) { const OnDevice on(odev); sycl::free(c, dpct::get_in_order_queue()); }   // a card frees what its own queue mapped
    mirror_chunks_.clear();
    mirror_bytes_ = 0; mirror_ptr_.clear(); layer_first_.clear();
    for (int fd : fds_) fd_close(fd);
    fds_.clear(); names_.clear(); layer_fd_.clear(); ring_.clear(); ring_key_.clear(); where_.clear();
    ring_next_ = 0;
}

bool GgufExpertSource::open(const std::string& shard1, int64_t n_layers, int64_t n_expert, std::string& err) {
    close();
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native || lay.gguf_off.size() < (size_t) (3 * n_layers)) {
        err = "--stream-experts needs a native (IQ) pack whose native_experts.txt carries the GGUF tensor offsets";
        return false;
    }
    for (int64_t l = 0; l < n_layers; ++l)
        for (int r = 0; r < 3; ++r)
            if (lay.gguf_off[(size_t) (3 * l + r)] == 0) {
                err = "--stream-experts: layer " + std::to_string(l) + " has no GGUF offset for its experts";
                return false;
            }
    shard_ = shard1;
    const size_t cut = shard1.find_last_of("/\\");
    dir_ = cut == std::string::npos ? std::string() : shard1.substr(0, cut + 1);
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    layer_fd_.assign((size_t) (3 * n_layers), -1);
    for (int64_t l = 0; l < n_layers; ++l)
        for (int r = 0; r < 3; ++r)
            if (fd_of(l, r, err) < 0) { close(); return false; }
    ring_.resize(kRing);
    for (auto& b : ring_) b.resize((size_t) lay.max_blob + 4096);   // tail slack: the host gemv reads past the last block (windows page-tight blocks)
    ring_key_.assign(kRing, -1);
    return true;
}

// The file of role `role` (0 gate, 1 up, 2 down) of `layer`. ExpertLayout::gguf_file is per layer AND role
// (`3 * layer + role`) since upstream 0.1.31; indexing it by layer alone read a shard-2 layer from shard 1 (Swift 1.5,
// whose layers 13-47 are in shard 2: garbage IQ1_M scales, NaN logits - 2026-10-01).
int GgufExpertSource::fd_of(int64_t layer, int role, std::string& err) {
    const size_t i = (size_t) (3 * layer + role);
    if (layer_fd_[i] >= 0) return fds_[(size_t) layer_fd_[i]];
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::string name = shard_;
    if (lay.gguf_file.size() > i && !lay.gguf_file[i].empty()) name = dir_ + lay.gguf_file[i];
    for (size_t k = 0; k < names_.size(); ++k)
        if (names_[k] == name) { layer_fd_[i] = (int) k; return fds_[k]; }
    const int fd = fd_open(name);
    if (fd < 0) { err = "--stream-experts: cannot open " + name; return -1; }
    names_.push_back(name);
    fds_.push_back(fd);
    layer_fd_[i] = (int) fds_.size() - 1;
    return fd;
}

const uint8_t* GgufExpertSource::blob(int64_t layer, int64_t expert) {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_ || ring_.empty()) return nullptr;
    if (!mirror_ptr_.empty()) {
        const uint8_t* p = mirror_ptr_[(size_t) (layer * n_expert_ + expert)];
        if (p != nullptr) return p;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    const auto& fm = lay.fmt[(size_t) layer];
    const uint64_t blob = lay.bytes[(size_t) layer];
    // the arena loader's gather, for one expert: [gate | up | down] from the three tensors
    const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
    const uint64_t at[3] = {0, fm.up_off, fm.down_off};
    const int64_t key = ((int64_t) layer << 20) | expert;
    size_t slot;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = where_.find(key);
        if (it != where_.end()) { ++reads_; return ring_[it->second].data(); }   // still resident
        slot = ring_next_;
        ring_next_ = (ring_next_ + 1) % ring_.size();
        if (ring_key_[slot] >= 0) where_.erase(ring_key_[slot]);
        ring_key_[slot] = key;
        where_[key] = slot;
    }
    std::vector<uint8_t>& buf = ring_[slot];
    for (int r = 0; r < 3; ++r) {
        const int fd = fds_[(size_t) layer_fd_[(size_t) (3 * layer + r)]];
        const uint64_t src = lay.gguf_off[(size_t) (3 * layer + r)] + per[r] * (uint64_t) expert;
        uint64_t done = 0;
        while (done < per[r]) {
            const ssize_t n = fd_pread(fd, buf.data() + at[r] + done, (size_t) (per[r] - done), (int64_t) (src + done));
            if (n <= 0) { std::lock_guard<std::mutex> lk(mu_); where_.erase(key); ring_key_[slot] = -1; return nullptr; }
            done += (uint64_t) n;
        }
    }
    ++reads_;
    return buf.data();
}

int64_t GgufExpertSource::mirror(const std::vector<std::pair<int64_t, int64_t>>& pairs, uint64_t cap, int threads,
                                 std::string& err) {
    return mirror_into(pairs, cap, threads, err, true);
}

// The mirror lane (2026-10-30): append a segment without disturbing what is already mirrored - a later layer-split
// stage's misses, pinned through THAT card's context (the caller's OnDevice scopes malloc_host below; each chunk's
// recorded device id is its free ticket).  Pairs a segment already serves are skipped; the cap limits the added bytes.
int64_t GgufExpertSource::mirror_more(const std::vector<std::pair<int64_t, int64_t>>& pairs, uint64_t cap, int threads,
                                      std::string& err) {
    return mirror_into(pairs, cap, threads, err, false);
}

int64_t GgufExpertSource::mirror_into(const std::vector<std::pair<int64_t, int64_t>>& pairs, uint64_t cap, int threads,
                                      std::string& err, bool replace) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::vector<std::pair<int64_t, int64_t>> take;
    uint64_t total = 0;
    for (const auto& [l, e] : pairs) {
        if (l < 0 || l >= n_layers_ || e < 0 || e >= n_expert_) continue;
        if (!replace && !mirror_ptr_.empty() && mirror_ptr_[(size_t) (l * n_expert_ + e)] != nullptr) continue;   // already served
        const uint64_t b = (lay.bytes[(size_t) l] + 255) / 256 * 256;
        if (total + b > cap) break;
        take.push_back({l, e});
        total += b;
    }
    if (take.empty()) return 0;
    // Alchemist (Arc A-series) refuses a single USM host allocation past ~3 GiB (docs/INTEL.md, A750), so the mirror
    // is built from chunks under that. Battlemage and the CUDA/HIP builds never hit the limit, but chunking is neutral
    // for them. One chunk per (up to) 2 GiB of experts.
    constexpr uint64_t kChunk = 2ull << 30;
    const int owner = (int) dpct::get_current_device_id();   // the context that maps these chunks; its own queue frees them
    std::vector<std::pair<uint8_t*, int>> chunks;
    // Free every chunk unless the mirror is handed over to the object: covers an allocation failure, a failed read and
    // any exception (a container allocation, a std::thread that cannot be created) on the way out.
    struct ChunkCleanup {
        std::vector<std::pair<uint8_t*, int>>& v;
        bool armed = true;
        ~ChunkCleanup() { if (armed) for (auto& [c, d] : v) if (c) { const OnDevice on(d); sycl::free(c, dpct::get_in_order_queue()); } }
    } cleanup{chunks};
    std::vector<uint8_t*> ptrs(take.size(), nullptr);
    uint64_t placed = 0;
    uint8_t* cur = nullptr;
    uint64_t cur_off = 0, cur_size = 0;
    for (size_t j = 0; j < take.size(); ++j) {
        const uint64_t b = (lay.bytes[(size_t) take[j].first] + 255) / 256 * 256;
        if (cur == nullptr || cur_off + b > cur_size) {
            // never smaller than one blob, so a single blob past kChunk still fits its chunk
            const uint64_t want = std::max(b, std::min<uint64_t>(kChunk, total - placed));
            uint8_t* nc = nullptr;
            try {
                nc = (uint8_t*) sycl::malloc_host((size_t) want + 4096, dpct::get_in_order_queue());   // +4 KiB past the last blob: same read-ahead slack; cur_size stays 'want' so placement is unchanged
            } catch (const sycl::exception& ex) {
                err = std::string("mirror: ") + ex.what();
            }
            if (nc == nullptr) {
                if (err.empty())
                    err = "mirror: no pinned host memory for " + std::to_string(want >> 20) + " MiB";
                return -1;
            }
            chunks.push_back({nc, owner});
            cur = nc; cur_off = 0; cur_size = want;
        }
        ptrs[j] = cur + cur_off;
        cur_off += b;
        placed += b;
    }
    std::atomic<size_t> next{0};
    std::atomic<bool> bad{false};
    std::vector<std::thread> ts;
    try {
        for (int i = 0; i < std::max(1, threads); ++i)
            ts.emplace_back([&] {
                for (size_t j; (j = next.fetch_add(1)) < take.size() && !bad.load();) {
                    const auto [l, e] = take[j];
                    if (!read_into(l, e, ptrs[j], (size_t) lay.bytes[(size_t) l])) bad.store(true);
                }
            });
    } catch (...) {
        for (auto& th : ts) if (th.joinable()) th.join();   // join before the cleanup frees, and before unwinding
        throw;
    }
    for (auto& th : ts) th.join();
    if (bad.load()) {
        err = "mirror: reading an expert from the GGUF failed";
        return -1;
    }
    if (replace)
        for (auto& [c, d] : mirror_chunks_) if (c) { const OnDevice on(d); sycl::free(c, dpct::get_in_order_queue()); }
    if (replace) mirror_chunks_ = std::move(chunks);
    else mirror_chunks_.insert(mirror_chunks_.end(), chunks.begin(), chunks.end());
    cleanup.armed = false;                                  // the chunks now belong to the object
    if (replace) {
        mirror_bytes_ = total;
        mirror_ptr_.assign((size_t) (n_layers_ * n_expert_), nullptr);
        layer_first_.assign((size_t) n_layers_, nullptr);
    } else {
        if (mirror_ptr_.empty()) {   // the first segment through the append path: fresh tables
            mirror_ptr_.assign((size_t) (n_layers_ * n_expert_), nullptr);
            layer_first_.assign((size_t) n_layers_, nullptr);
        }
        mirror_bytes_ += total;
    }
    for (size_t j = 0; j < take.size(); ++j) {
        const auto [l, e] = take[j];
        mirror_ptr_[(size_t) (l * n_expert_ + e)] = ptrs[j];
        if (layer_first_[(size_t) l] == nullptr) layer_first_[(size_t) l] = ptrs[j];
    }
    return (int64_t) take.size();
}

bool GgufExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (mirror_ptr_.empty() || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return false;
    return mirror_ptr_[(size_t) (layer * n_expert_ + expert)] != nullptr;
}

const uint8_t* GgufExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (mirror_ptr_.empty() || layer < 0 || layer >= n_layers_) return nullptr;
    if (expert >= 0 && expert < n_expert_) {
        const uint8_t* p = mirror_ptr_[(size_t) (layer * n_expert_ + expert)];
        if (p != nullptr) return p;
    }
    return layer_first_[(size_t) layer];
}

bool GgufExpertSource::read_into(int64_t layer, int64_t expert, uint8_t* dst, size_t bytes) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_ || dst == nullptr) return false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const auto& fm = lay.fmt[(size_t) layer];
    const uint64_t blob = lay.bytes[(size_t) layer];
    if (bytes < blob) return false;
    const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
    const uint64_t at[3] = {0, fm.up_off, fm.down_off};
    for (int r = 0; r < 3; ++r) {
        const int fd = fds_[(size_t) layer_fd_[(size_t) (3 * layer + r)]];
        const uint64_t src = lay.gguf_off[(size_t) (3 * layer + r)] + per[r] * (uint64_t) expert;
        uint64_t done = 0;
        while (done < per[r]) {
            const ssize_t n = fd_pread(fd, dst + at[r] + done, (size_t) (per[r] - done), (int64_t) (src + done));
            if (n <= 0) return false;
            done += (uint64_t) n;
        }
    }
    return true;
}

}  // namespace strata::core
