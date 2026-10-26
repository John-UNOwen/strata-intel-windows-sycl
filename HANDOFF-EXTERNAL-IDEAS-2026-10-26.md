# Handoff: ideas from StrataFast and lean-qwenflash for the Arc fork (2026-10-26)

What two outside projects did that this fork (B580 + A770, Windows, IQ2_XS pack, k22 split) can use, in the order
to try them. Read the "Fault search" and "Speed" sections of `OPTIMIZATIONS-2026-10-16.md` first: storms still gate
every speed arm, so the two storm controls below come first.

**What this is based on.** Reading code and docs only. Nothing here was built or run on the PC. Every number below
was measured by the other projects on their hardware (named next to each number), not on this PC.

- **StrataFast** (`github.com/eddoursul/StrataFast`, branch `custom`, head `dd0390f`). A fork of upstream Strata that
  carries upstream's SYCL work from 0.1.45 to 0.1.48. Our `sycl/` tree is based on 0.1.44.1 (`296b59d`), so most of
  this is upstream SYCL code we have not merged yet. Its own additions are CUDA-side and mostly do not apply.
- **lean-qwenflash** (`github.com/MaxDevv/lean-qwenflash`). A CUDA / ExLlamaV3 engine for the same model: experts on
  NVMe, a RAM tier and a VRAM cache. Its base, `qwen38-flash-offload`, has an optional second Intel GPU as an expert
  tier (`qwen38-flash-offload/docs/b70-tier.md`). No code transfers; one design lesson does.

## Summary

| # | Item | Kind | Cost | What it can tell or give us |
|---|---|---|---|---|
| 1 | `--adapt-every 0` storm control arm | config | one arm | whether adaptive tier swaps cause the storms |
| 2 | Staged prompt DMA (port of upstream's A750 deadlock fix) as a control arm | small code, opt-in | ~40 lines | whether direct DMA from the unpinned arena causes the storms |
| 3 | Window hashes (`--window-hashes`, already in our tree) for a deterministic repro | tooling | none | localizes a storm to one window without perturbing it |
| 4 | Per-card `STRATA_PF_XMX`, then mode 1 on the A770 stage | code + arm | small | A770 prompt experts on XMX; today the knob does nothing on a split |
| 5 | Per-card `STRATA_MMVQ_LARGE_GRF` (port) | code + arm | medium | A770 decode windows (2-6 columns) |
| 6 | Planar Q6_K output head (port) | code + arm | medium | LM head per token, if the head is Q6_K |
| 7 | lean-qwenflash lesson: take misses off the CPU | confirms existing plan | - | backs speed items 1-2 of the optimization file |
| - | Not usable here | - | - | see the last section |

## 1. Storm control: `--adapt-every 0` (try first, config only)

**Evidence from elsewhere.**
- Upstream's `docs/DETAILS.md` (in StrataFast) reports the adaptive tier plus memory pressure resetting a Radeon 780M
  GPU in 7 of 9 runs, and about 40 clean runs with `--adapt-every 100000`.
- Our own engine's help text (`sycl/src/program/generate.cpp:904-906`) says reproducible arithmetic needs
  `--adapt-every 0`: the adaptive tier is the one thing that changes the expert placement while serving.

**Why it fits our storms.**
- The adaptive tier is on by default (`adapt_every = 4`, `generate.cpp:672`): every 4 windows it swaps hot missed
  experts into VRAM slots and rewrites the residency table (`adapt_tick`, `generate.cpp:8469-8530`).
- The prompt path reads that same host table live (`Prefill::init(..., host_res.data(), ...)`, `generate.cpp:6958`;
  `prefill.cpp:2565, 3845, 3866`). An expert it sees as resident is **not copied**; its slot is used in place.
- So a table entry that points at a slot holding a different expert (or one being overwritten) gives exactly what
  the storming 4K request showed: wrong numbers and **less copy work**, i.e. a faster prompt (539-541 against
  425-445 tok/s in `OPTIMIZATIONS-2026-10-16.md`).
- Not proven: the swap steps look ordered on the host (`CopyBack` marks the old expert not resident before the copy,
  `SwapIn` marks the new one after it). What I could not settle from the code is the device side on a split: the
  table upload (`res_put`, `generate.cpp:5932`) goes through `dpct::get_in_order_queue()` of the current device, the
  swap copies through `h.stream`, and the stage's window runs on its own queue. If a stage's queue still has a window
  in flight at the tick, it can read the new table with the old slot contents or the reverse. The arm answers this
  faster than more reading.

**The arm.** One line in `bench/arms/make_arms_k22.py`, against `k22ctx32k` like the other controls (keep
`--max-context 32768`):

| Arm | Change | A change in the rate proves |
|---|---|---|
| `k22ctx32kstatic` | `--adapt-every 0` | the adaptive tier's swaps (table, slot copies or their ordering) light the storm |

Run it with the existing recipe (`BENCH_SUITES=4K`, `BENCH_RUNS=6`, `BENCH_REPEATS=3`). Expect slightly lower decode
(no hot experts promoted), so read it as a storm rate, not a speed. If it is clean where `k22ctx32k` storms, the fix
is in the swap path on a split (barrier each stage's queue before the table upload and the slot copies), and
`--adapt-every 0` is a safe interim setting for VicGPT.

## 2. Storm control: staged prompt DMA (port of upstream 0.1.48's A750 fix)

**What upstream found** (StrataFast `sycl/src/prefill/prefill.cpp:110-136`, regression test
`sycl/tools/test_prompt_deadlock.sh`): SYCL has no `cudaHostRegister`, so the expert arena that the engine calls
"pinned" is ordinary pageable memory. A DMA straight from it (`copy->memcpy(stage, arena_ptr, 2.3 MB)`) makes the
Level Zero driver copy through its own staging buffers, and on some prompts that never finished on an A750 (Linux).
Their fix: only USM host memory is DMA'd directly; every other expert is first copied into the stager's real pinned
buffers by its threads, the path unpinned experts already take. `STRATA_PF_DMA_PAGEABLE=1` restores the old path.

**Our tree has the old path.** The arena is `VirtualAlloc` memory (`sycl/src/core/pinned.dp.cpp:114`), registration
is a no-op (`DPCT1027`, `pinned.dp.cpp:451`), and the prompt DMAs straight from it whenever `src->pinned(l, e)` is
true (`prefill.cpp:3871-3890`, and the stream-all list at `prefill.cpp:2587`). On Windows this has not been seen to
hang, but a driver-staged copy from pageable memory is the one prompt copy whose timing we do not control, and the
storm ignites in a prompt.

**The port, kept opt-in** (AGENTS.md: default path byte-identical):
- New switch `STRATA_PF_DMA_STAGED=1` (inverse of upstream's default). When set, treat every arena expert as
  "not pinned" **for the DMA decision only**, at all three places that decide it, so the job list and the issue loop
  agree: `prefill.cpp:2587` (stream-all job list), `prefill.cpp:3846` (staged job list) and `prefill.cpp:3871` (issue).
- Do not copy upstream's USM test as is: it asks `dpct::get_in_order_queue().get_context()`, which is the B580's
  context. On a split, ask with the stage's own queue (`m.copy` / the stage's context). With the Windows mirror off
  (size 0 today), no expert is USM host anyway, so the switch can simply skip the test.
- Cost: every miss gets one extra host memcpy by the stager threads. Expect a slower prompt.
- Arm: `k22ctx32kstaged` = `STRATA_PF_DMA_STAGED=1`. Same reading as item 1.
- Build both builds the file touches (JIT and AOT) before running, per AGENTS.md.

## 3. Deterministic repro with window hashes (already in our tree)

StrataFast added `STRATA_FOLLOW` (`generate.cpp:10694-10710` there): in serve mode, force a given token sequence
through the real verify windows and log each row's log-probability. We do not need to port it first: our tree already
has `--spec-follow PATH` plus `--window-hashes PATH` (`generate.cpp:901-906`), which writes one 64-bit hash of each
window's final rows. Not available in `--serve`.

How to use it on the storm:
- Take the storming 4K prompt and its clean continuation (token ids) from a clean session.
- Run the CLI (not serve) N times at the `k22ctx32k` shape with `--spec-follow cont.txt --window-hashes hN.txt
  --adapt-every 0`. Clean runs should write identical files.
- The first window whose hash differs between runs is where the arithmetic changed. Hashing costs no sync inside
  the window, unlike `STRATA_DBG_NAN`.
- If the CLI never storms with `--adapt-every 0` but serve does, that is item 1's answer from a second angle.
- Port `STRATA_FOLLOW` only if the storm will not reproduce outside `--serve`.

The user starts every model load; this is a recipe for them, not something to run from an agent shell.

## 4. Grouped XMX prompt experts on the A770 stage

**Upstream result** (StrataFast `sycl/src/prefill/xmx_moe.dp.cpp:147-175`, owner-approved 2026-10-18): mode 2 (fused
grouped XMX) is the default on Alchemist: A750 prompts +30-39% at 4K, +32-35% at 1-2K tokens, first-token KL 0.002
against a 0.0005 floor. Battlemage was 7-11% slower, so it stays off there. Upstream's own A770 measurements of mode
1/2 are not in either repo.

**Why it does nothing on our split today.**
- `strata::prefill::xmx::mode()` is a function `static` decided once, by `q_of(nullptr)`, i.e. the first card. In the
  JIT build (non-SG16) the B580 makes it print "ignored" and return 0, and the A770 inherits that 0
  (`xmx_moe.dp.cpp:160`). So `STRATA_PF_XMX` is off for both cards no matter what is set.
- `stage_slots()` and `dq_slots()` (`prefill.cpp:233-236`) size the staging rings from the same `mode()`.

**Our pack and mode 2.** Mode 2 only takes gate/up IQ3_XXS, IQ3_S, IQ2_S or IQ4_XS with down IQ4_NL or Q2_0
(`xmx2_fmt`, `prefill.cpp:3977-3981`; the same list upstream). Our expert gate/up is IQ2_XXS (type 16), so mode 2 is
not taken. The log line "STRATA_PF_XMX=2 not taken at layer ... formats a/b" (`prefill.cpp:3987`) confirms the types
on the PC. **Mode 1** (dequantize to f16 + grouped XMX GEMM, `prefill.cpp:4116-4150`) goes through
`iq_dequant_expert_f16`, which handles type 16 and 42 (`iq_kernels.dp.cpp:2930-2942`), so mode 1 is the one to try.

**Change.**
- Make `mode()` take the queue and cache per device with `strata::intel_gpu_gen_of(dev)`, like `gdn_xe_on`
  (`sycl/src/prefill/kernels.dp.cpp:2615-2628`) already does. A770 → the env value; B580 in the JIT build → 0.
- Make `stage_slots()` / `dq_slots()` per stage (each stage's `Prefill` has its own queue).
- Keep the default 0 on both cards (opt-in). Upstream's Alchemist default needs our own A770 numbers first.
- Arm: `k22pfxmx1` = `STRATA_PF_XMX=1`, prompt suites 4K and 16K. Report the A770 stage's prompt time, not just the
  total; the B580 stage is unchanged by design.

## 5. Large-GRF decode kernels on the A770 (port)

**Upstream result** (StrataFast `sycl/src/kernels/cuda/native_mmvq.dp.cpp:1718-1730`): the wide MMVQ kernels at 2-8
columns spill at 128 registers on the A770. At 256 (large GRF), `mmvq_bench` Q6_K 4 columns went from 70-80 to
150-237 GB/s, same checksum. Default on Alchemist only; Battlemage and 1-column launches unchanged. Upstream's
end-to-end decode figure is +3.5-4.2% on a B70 run with it forced (from StrataFast's notes; not an A770 number).

**Why it matters here.** Our decode is all verify windows with T up to 6, which are the 2-8 column kernels, and 26 of
48 layers run on the A770.

**Port notes.**
- Upstream's `mmvq_large_grf(s)` is again a function `static` captured from the first queue. Port it per device
  (`intel_gpu_gen_of(s->get_device())`) or the B580 decides for the A770.
- Large GRF is a kernel property; the port brings the second kernel variants with it. Check the dense weight types of
  our pack first (`strata --families` / the load log): the measured gain is for Q6_K; other types run the generic
  wide kernel, which upstream also switches but did not measure separately.
- Arm: `k22grf` = `STRATA_MMVQ_LARGE_GRF=1`, decode suites. Opt-in until measured.

## 6. Planar Q6_K output head (port)

StrataFast `native_mmvq.dp.cpp`, from line 3062 (`STRATA_Q6_PLANAR`, on by default there): converts a Q6_K matrix to a
planar layout once at load; upstream measured the LM head about 14% faster, bit-identical output.
- Applies only if our pack's output head is Q6_K. Check the load log first.
- Upstream allocates the planar copy through `dpct::get_in_order_queue()` (the B580). On a split the head lives on
  the last stage (the A770): allocate and free it through the head's own queue, or the A770 reads another context's
  memory (the T3 failure).
- The head is one matrix per token; even 14% of it is a small part of 40-50 ms per token. Do it after 4 and 5.

## 7. lean-qwenflash: take the misses off the CPU

Their measured result (RTX 3090 + Arc Pro B70, 2026-10-09, `b70-tier.md`): a second Intel GPU holding 3,000 extra
experts took 8K decode from 17.48 to 26.40-26.61 tok/s (+51%) and 32K from 16.28 to 24.60-24.92, prompt unchanged.
Three design points:
- **No duplicates across tiers.** Experts moved to the second GPU are dropped from RAM, and RAM refills with colder ones.
- **The helper runs beside the CPU**, not after it: the CPU posts the helper's picks, computes its own, then adds the
  helper's rows. The CPU waited only 0.4 ms per token for the helper.
- The gain came because the CPU lane was the long one.

What this means here:
- It supports speed items 1 and 2 of `OPTIMIZATIONS-2026-10-16.md` (mirror the A770 stage's misses, then let the A770
  read them over PCIe): both move work out of the CPU lane. Do those, not a new tier.
- Strata's own version of their layout is `--expert-cache-device1` (RemoteExperts, `docs/SECOND_GPU.md`): A770 runs
  all 48 layers, B580 holds extra experts. Not recommended for VicGPT: the A770 alone measured 90-130 tok/s prompt on
  this PC against 260-320 at k22, and the map step is prompt-bound (8-18K tokens). It also needs the A770 to be the
  main card (UR lists the B580 first) and the B580 to report free memory (it refused earlier with free memory 0).
- Before any of it, check that the CPU lane is the long one at k22: the per-token decode breakdown (`--stats`) should
  show the host expert time against the two stages' GPU time.

## Not usable here (checked)

- **Fused int8 ESIMD DPAS prompt experts on Battlemage** (`esimd_moe.dp.cpp`, `docs/INTEL_INT8_EXPERTS.md`,
  `STRATA_PREFILL_MMQ` / `STRATA_PF_I8`): gate/up IQ3_S, IQ3_XXS or IQ2_S only. Our IQ2_XXS gate/up would need a new
  decoder, and it would help only the B580 stage. Large job, low priority.
- **`STRATA_IQ3S_WIDE`**: IQ3_S only, not our type.
- **`STRATA_ATTN_RS=2`** (attention layout): +0.8-0.9% prompt on a B70. Too small to measure through our run-to-run
  drift.
- **Bigger prompt chunks on Battlemage** (B70 2048 → 4096: 618 → 1002 tok/s): already covered by our `own4096` arms,
  and chunk size is currently a storm suspect.
- **Stager event-wait hangs** (Level Zero v2 adapter, 2026-10-01/02): our tree already has both fixes
  (`prefill.cpp:218`, `prefill.cpp:507`).
- **GDN recurrence sub-group kernel on Alchemist**: already in our tree and already per device (`gdn_xe_on`).
- **StrataFast's CUDA-host features** (`--prompt-split`, `--arena-place`, `--no-file-cache`, PcieAuto): CUDA or
  single-card only.
- **Pictures**: upstream made them opt-in after a B70 GPU hang on the first picture. Matches our split16 picture NaN;
  keep pictures off on the split.
- **lean-qwenflash's NVMe O_DIRECT tier and ExLlamaV3 kernels**: a different engine and format; nothing to port.

## Suggested order

1. Add `k22ctx32kstatic` (item 1) to the arms and run the storm recipe. No code.
2. Port item 2 behind `STRATA_PF_DMA_STAGED=1`, build JIT and AOT, add `k22ctx32kstaged`.
3. If a storm still lights with both controls on, use item 3 to find the window.
4. Then the speed ports in order 4, 5, 6, each opt-in, each with its own arm and the pollution gate on.

## Running the two controls (done 2026-10-26)

Both arms are in `bench/arms/make_arms_k22.py`: `k22ctx32kstatic` (`--adapt-every 0`) and `k22ctx32kstaged`
(`STRATA_PF_DMA_STAGED=1` in the env; the code landed behind that switch in `sycl/src/prefill/prefill.cpp` at
four `pinned()`-keyed DMA decision points, the peer `ps_frac` branch included, and both builds rebuilt after it).
The name prefix auto-arms both with `STRATA_DBG_WINDOW`/`STRATA_DBG_NAN_DEFER` like the baseline.

Read the storm rate, not speed: baseline `k22ctx32k` polluted all 3 sessions run, and a polluted session skips
that arm's rest, so clean here means 3/3 sessions finished; one storm is a failure. Both controls cost speed by
design.

Close VicGPT's llama-server first and leave ports 8191-8240 free. From this folder's cmd window:

    set BENCH_SUITES=4K & set BENCH_RUNS=6 & BENCH-ARMS.bat k22ctx32kstatic
    set BENCH_SUITES=4K & set BENCH_RUNS=6 & BENCH-ARMS.bat k22ctx32kstaged

6-12 min per session plus 1-3 min engine load; keep Task Manager's GPU page open and stop on shared GPU memory
over ~0.2 GB, B580 dedicated near ~11.3 GB, or a flicker.

What to do with the answers: static clean and staged storming - the swap/table lane is implicated, design the
per-stage queue barrier for the swap path (item 1's stated fix) and keep `--adapt-every 0` as the interim VicGPT
setting. Staged clean and static storming - the DMA-from-pageable path is implicated: promote the staged copy with
numbers and keep the adaptive tier. Both clean - adopt `--adapt-every 0` interim, run a ~40-session soak before
trusting it in production, and the staged promotion waits for its own speed numbers. Both storming - item 3's CLI
recipe (`--spec-follow` + `--window-hashes`, `--adapt-every 0`, same shape, several starts; first differing window
hash names where the arithmetic changed; if the CLI will not storm at all, that is item 1's answer from a second
angle and serve-mode `STRATA_FOLLOW` becomes worth porting).

### Results (evening of 2026-10-26, both arms run on this PC)

`k22ctx32kstatic`: **3/3 sessions finished clean** - 18 fresh 4K requests, prompt medians 444.8-447.4 tok/s
(the baseline's clean band is 425-445, its storming reads were 539-541), zero suffix-draft accepts, zero
non-finite lines across 789 log lines. Baseline `k22ctx32k`: 0/3 clean. With the tier parked, nothing else in
this shape's traffic changed, so the adaptive tier's live rewrite of the host residency table is the implicated
lane. Evidence, not proof - the race itself is still unwitnessed in the code. Interim adopted: `--adapt-every 0`
added to `strata-iq2_xs-2card-k22-ctx64k.json` (the daily default; note it doubles as the arms base, so future
baseline arms inherit the interim too).

`k22ctx32kstaged`: polluted in session 1 with the usual fingerprint (suffix drafts 3/3 accepted, fast-side
prompt) - **but as run it tested nothing.** The daily shape uses `--stream-experts`, and that source reports
`pinned()` true only for mirror-backed experts, with the mirror off on Windows: every expert already took the
stager's pinned-buffer path, so the switch was inert and the session was the baseline restated. The A750
mechanism is real for the arena and mirror sources (`--mmap-experts` shapes, or `STRATA_MIRROR_MIB > 0`), not
for this stream shape; a staged test needs one of those. The port itself stays default-identical and ready.

Next step, suspect narrowed: make the tier safe rather than parked - swaps must not land between a prompt path's
read of the residency table and the copies that read scheduled (tick the tier only between requests, or snapshot
the table per request; both keep the cross-request learning `--adapt-every 0` gives up). Needs its own arm
(same shape, same gate) and a soak before production. If anything still storms with the tier parked, item 3's
window hashes.
