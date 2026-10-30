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

### Paired 5+5 run (late 2026-10-26): the tie

`BENCH_REPEATS=5`, one ladder: `k22ctx32kstatic` 5/5 clean (8/8 cumulative for the parked tier on the current
engine, 48 fresh 4K requests) and `k22ctx32ktier4` (adaptive restored: `--adapt-every 4`, and `--adapt-swaps`
defaults to 96 with the profile present, so the tier was genuinely armed) also 5/5 clean - 30 prompts all in
the clean band (442-458 tok/s), no storm-side fast reads. The with-tier shape's record on the current engine is
therefore 1 polluted session in 6 (yesterday's staged restatement); on the 10-16 era engine it was 3/3 and
worse. Two readings survive: the storm's base rate fell below what 5 sessions can catch (something else in the
65 commits since 10-16 already helped, or the environment matters), or the parked tier's 0-in-8 clean record
was luck. Parked remains the only state that has never polluted; interim stays. To settle it: more shots at the
with-tier shape (BENCH_RUNS=12 raises ignition chances per minute), or build the between-requests tick gate as
cheap insurance - it preserves cross-request adaptation and can be validated against tier4 with these same arms.

Follow-up the same night: tier4 only, 5 more sessions at 4K + 8K (`BENCH_RUNS=5`) - all clean. The armed shape is
now 10 clean sessions of 11 on the current engine (the one pollution is yesterday's staged restatement), against
0 polluted in 8 parked sessions. First 8K-phase numbers on this PC: 8020 tokens read at 510 tok/s median
(4019 at 440) - faster per token than 4K by size effect (per-chunk overhead amortized), tight distribution, no
ignition at the extra chunk boundary. The storm's current-engine rate is ~1 session in 11 with the tier armed;
whatever cut it from the 10-16 era's 3-of-3 has already landed. Parked stays as interim; the between-requests
tick gate remains the path to running the tier again without the risk.

### The gate is implemented (2026-10-26, commit `929c21e0`)

`--adapt-idle 1` (default off, byte-identical otherwise): the tier keeps its every-N-windows schedule, but a due
round only runs at the request boundary - after the existing drains, before the prompt starts - so the residency
table and cache slots are frozen for a request's whole life. Covers every start site the serve paths have (the
async worker's Idle start, the pipeline `pl_adapt` kick, the serial blocking kick; batch decode steps never
start rounds, only drain, so they needed nothing). One design note: a boundary runs at most one round, so long
prompts adapt more slowly than mid-request tiering - commented at the block. Both builds rebuilt, `--help`
identical between them. Validation arm: `k22ctx32ktier4gated` (tier4 + `--adapt-idle 1`) - run it in one ladder
with `k22ctx32ktier4`; read gated's clean rate and its prompt medians against tier4 (a boundary round that
lands on a request slows that request's TTFT - the accepted cost). If it holds, the daily config takes
`--adapt-every 4 --adapt-idle 1` and the interim parks.

### Adopted (2026-10-26, end of day): the gate ran clean at zero measurable cost

`k22ctx32ktier4gated` 5/5 clean next to `k22ctx32ktier4` 5/5 (tier4 now 15 clean of 16 armed sessions on the
current engine). Prompt medians and TTFT ranges are statistically indistinguishable from mid-request tiering
(4K: 441-443 vs 440-443 median, TTFT 9.06-9.44 vs 9.12-9.39 s; 8K likewise) - boundary rounds are sub-100 ms
on this pack and traffic, so the table freeze comes free here. The daily config now runs
`--adapt-every 4 --adapt-idle 1`; the interim (`--adapt-every 0`) is retired. Honest scope of the claim: the
gate removes the only known ignition window BY DESIGN (no table or slot mutation can land mid-request), and the
arm run proves it runs clean at zero cost, but the engine's current storm rate (~1 session in 16 armed) is too
low for the run to demonstrate prevention statistically. Overnight VicGPT use is the acceptance test; if a
storm still appears with the gate on, the shape has a mechanism the tier theory does not explain and item 3's
window hashes are the next tool.

## Progress (2026-10-30)

### Merged

`storm-controls` fast-forwarded onto `main` at `be4f63a8`: main now carries the staged-DMA switch, the
`--adapt-idle` gate, the 8K bench phase, `tools/dsh_probe.py`, and this file.

### 128K context measured (`k22ctx128k`, 1 session)

Loads and serves 131072 tokens clean (3/3 runs, gate live), at a real cost: 4K prompt median 335.6 tok/s
(318.7-335.9) with TTFT 12.05 s, against the same-week 32K band of 441-443 median at ~9.1 s - a ~24% prompt
tax from the KV squeeze (A770 expert cache trimmed to 5118 slots; "CUDA1 reports no free memory" warnings in
the load log). Context headroom is available and not free. The long-prompt/agent comparison against `k22ref`
in one ladder is still open before adopting it for anything.

### DSH as a client

`serve\server.py` already speaks OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`), stream and
non-stream, tools included - a harness can point at this port with no glue. `tools/dsh_probe.py` (commit
`e2583836`) tests what a harness needs: `/health` and `/v1/models`, openai `tool_calls`, anthropic
`tool_use`, and a tool call made from a needle buried 70% into a ~45K-token prompt. Point the harness
endpoint only after three PASS lines. Known cost: the server serializes everyone behind one sequence - a
10-16 s map chunk pauses harness turns and vice versa - so overnight windows are the quiet ones. Note:
`MockServer.exe` (the mfg-decomp mock) had been squatting 8080/8081; stopped 2026-10-26, it should rebind
to another port (owner plans the change).

### MTP retry - the gate makes the parked measurement finally cheap

The 2026-10-16 park ("unless a clean k22 run beats 18-24 tok/s", KB decision) named its own blocker: no k22
mtp run ever finished clean, because storms tainted every fast number. With the gate in the base, the retry
is cheap. Arm `k22ctx32kmtp` reproduces `strata-iq2_xs-2card-mtp-k22.json` (drafter `C:\strata-data\mtp\rt`,
A770 squeezed to 13000 free, `STRATA_MTP_PREFILL_SYNC=1` - the drafter's VRAM bill is part of the shape).
Pair it with `k22ctx32k` (same shape, no drafter, gated) in one ladder, chat suite on: the chat-phase decode
medians are the verdict against the same-day control (the note's 18-24 bar is the 10-16 baseline; today's
control median is the real bar), and the engine auto-disabling drafts mid-run is itself a verdict. Run:
`BENCH_SUITES=chat,4K`, `BENCH_RUNS=4`, `BENCH_REPEATS=3`, arms `k22ctx32k k22ctx32kmtp`. A clean loss
completes the park's ledger honestly; a win reopens it with the k22-mtp NaN story.

### MTP retry result (2026-10-30, 05:42-05:50): the drafter relights pollution, the gate does not cover it

- Control `k22ctx32k` (gated, no drafter): 3/3 clean sessions; chat decode medians 18.1 / 18.0 / 19.2 tok/s,
  4K medians 442.6 / 411.4 / 441.1 - ordinary bands.
- `k22ctx32kmtp` (gated): first session polluted in the chat phase; the gate aborted its remaining sessions.
  Its pre-pollution replies decoded 14.1-17.2 tok/s - below the control band, with decode cache hits at 50.5%
  (the drafter's 839 MiB squeezed the cache) - then the loop run: 28.4 tok/s with **48 of 48 windows' suffix
  drafts accepted**, the classic fingerprint, now with the gate live and the tier schedule frozen at every
  request boundary.

The park stands, on its own terms, and this retry adds the sharper fact: **every k22 session ever measured
with `--mtp` enabled has polluted** (2026-10-16 and 2026-10-30 alike), while same-day sessions without it run
clean under the same gate. The adaptive-tier gate closes one ignition window; nothing in `--adapt-idle`
addresses the drafter's own mid-request work (draft K/V updates, its side-branch queue on the A770). The
park's bar - "a clean k22 mtp run beats the baseline" - has now been tested under the cleanest conditions
that ever existed on this PC, and the run polluted before a single speed number could land. If anyone wants
the drafter on this split anyway, the next tools are item 3's window hashes on exactly this shape (it is
deterministic) or a `STRATA_DBG_WINDOW` diff against today's pair; both would hunt the drafter's pollution
lane, not the tier's.

### Review found a gate bug - async path only

External review of the gate: with `--adapt-async 1`, the request-boundary call `adapt_tick(false, true)`
returned as soon as it *started* the round (the Idle step posts the choose job and returned), leaving the
round's marks, copies and flip to finish via the mid-request polls - during the request, the exact thing
`--adapt-idle` forbids. Fix: the Idle step continues the state machine when the start is reached with `fin`
set (the boundary case) - `a_ready(fin)` waits at every following step, so one boundary call runs the round
through its flip; mid-request starts (`fin` false) return exactly where they always did. Scope: the blocking
and pipeline boundaries were always correct (`adapt(); apply_pending(true)` finishes before the prompt), and
the async worker only exists with `--adapt-async 1` plus the resident RAM mode - the daily shape runs neither
(default off; `--stream-experts` on 32 GB cannot provide resident mode), so every measurement in this file
was produced on the correctly-gated path. The fix rebuilt in both builds (JIT + AOT) as commit `14aa7933`;
default, blocking and pipeline behavior are byte-identical - it only engages with `--adapt-async 1`.

### B580 reserve tuning (2026-10-30)

The "spare ~1 GB" on the B580 is the engine's own `--vram-reserve-mib 1536` holding it back by design (late
allocations + driver churn; the machine has a no-room shutdown crash and an ~11.3 GB dedicated watch-line in
its history). The daily config now runs `--vram-reserve-mib 1024`: ~512 MiB returns to the B580 expert cache
(~+340 of the 5420 slots at the 32K/64K fills) and Task Manager should settle near 10.4/12 GB dedicated.
The user is watching the card; anything pushing B580 dedicated past ~11.3 GB or shared GPU memory past 0.2 GB
goes back toward 1300 or 1536.

## Review of `14aa7933` and the open improvements (2026-10-30)

### The async boundary fix is correct

`14aa7933` makes the `Idle` step continue instead of return when the boundary call (`fin`) starts a round, so
`a_ready(fin)` waits through `CopyBack`, `SwapIn` and `Commit` and the flip lands before the prompt. After `Commit`
the loop returns to `Idle` with `a_win - a_last == 0` (`a_last` was just set and the boundary call does not
advance `a_win`), so it stops there instead of starting a second round. Mid-request calls (`fin` false) take the
old `return true`. Default, blocking and pipeline paths are unchanged. Not built here, only read.

### Correction to an earlier remark: two-stage prompts already exist on the split

Lamina's `--prompt-split` (one card on chunk N's later layers while the other starts chunk N+1) gave it 1.24-1.55x
on 32K-128K prompts. A layer split already overlaps the stages of multi-chunk prompts: `docs/MULTI_GPU.md` says the
idle-card help is off from ~3.3K tokens "where the stages overlap anyway". VicGPT's 8-18K prompts are multi-chunk,
so there is nothing of Lamina's to port there. What Windows does lose is the **one-chunk** help (next item).

### Open improvements, in the order to try them

Each is opt-in or config-only, needs its own arm in `bench/arms/make_arms_k22.py` against a same-ladder control
with the gate on, and none has been measured on this PC yet. (That generator is not in the repo yet; push it with
the next arms.)

**Prompt (VicGPT's map step is prompt-bound)**

1. **Rerun the chunk-size arms under the gate.** `k22prefill16k`, `k22ctx32kchunk8k` and the 32K arms were void
   because they stormed, not because they lost. Today's data says bigger reads pay: the 8K phase reads 510 tok/s
   against 440 at 4K, and `own4096` was the one gate-checked prompt win (+35% at 4K). With the storm rate now
   about 1 in 16 armed sessions, these arms can finally produce numbers. Start with `k22own4096` and
   `--prefill auto:16384` on the 8K and 16K phases, gate on, `BENCH_REPEATS=3`.
2. **Grouped XMX prompt experts on the A770 stage (item 4 above).** Still blocked by the first-caller `static` in
   `xmx::mode()`: on the JIT split the B580 turns it off for both cards. Fix `mode()`, `stage_slots()` and
   `dq_slots()` per device (`intel_gpu_gen_of`, as `gdn_xe_on` does), then arm `k22pfxmx1` = `STRATA_PF_XMX=1`
   (mode 1 dequantizes, so it takes our IQ2_XXS gate/up; mode 2 does not). Upstream's +30-39% was mode 2 on an
   A750; mode 1 on the A770's 26 layers is unmeasured.
3. **A smaller KV cache at long context.** 128K cost ~24% prompt (335.6 against 441-443 tok/s at 4K), because the
   KV squeezed the A770's expert cache to 5118 slots. Our tree has `--kv k8v4` (int8 K, rotated 4-bit V, 816 B per
   cell against int8's 1,056, -23%) and `--kv q4_0` (half of int8). Upstream on a B70 at 128K: k8v4 with the KV
   in VRAM read prompts fastest (1,006 tok/s, `docs/INTEL.md:344`). Arms: `k22ctx128kk8v4` against `k22ctx128k`,
   and the same at 64K if the daily shape runs int8 KV. Read the storm gate first and the output's quality second
   (4-bit V changes rounding: Lamina measured its k8v4 1.5-4.5x as far from llama.cpp as llama.cpp's own CPU and
   CUDA backends). Not `--kv-resident` (KV streamed from "pinned" RAM): on Windows that RAM is pageable.

**Decode**

4. **Mirror the A770 stage's misses, then let it read them over PCIe** (speed items 1-2 of
   `OPTIMIZATIONS-2026-10-16.md`). Still the largest decode lever: the A770 stage's misses all go to the CPU today.
   lean-qwenflash's +51% decode came from exactly this (moving work out of the CPU lane). The mirror segment has
   to be allocated through the A770's own queue, or its kernels read another context's memory (T3).
5. **Large-GRF wide decode kernels on the A770 (item 5 above).** Our decode is all verify windows (T up to 6), which
   is what these kernels serve. Port per device.
6. **Optional: B580 main + A770 helper tier, for chat only.** No `--layer-split`; `--expert-cache-device1 auto
   --remote-expert-opt`; keep `STRATA_VRAM_FREE_MIB` for the A770. The B580 is already device 0. Expect better
   decode and much slower prompts (the helper tier serves decode only, so the prompt runs on the B580 alone:
   96 tok/s measured at 3.4K tokens). Not for VicGPT. The helper path has not run across two Windows contexts yet.

**MTP / pollution**

7. **Separate the suffix drafter from the MTP drafter.** The 10-30 MTP storm fired with "48 of 48 windows' suffix
   drafts accepted". None of the controls so far turned suffix drafting off (`k22ctx32kspec1p` explicitly does
   not). `--suffix-draft 0` does (`generate.cpp:1949`, default 3). Arms: `k22ctx32kmtpnosfx` (the mtp arm plus
   `--suffix-draft 0`) and `k22ctx32knosfx` (control plus `--suffix-draft 0`). If the mtp arm then runs clean,
   the drafter is fine and the suffix path on the A770 stage is the lane. If it still pollutes, use item 3's
   window hashes on the mtp shape.
8. **Draft-head reservation on CUDA0** (robustness item 3 of `OPTIMIZATIONS-2026-10-16.md`): `mtp_bind` reserves
   the drafter's VRAM on the B580 although the drafter lives on the last stage (the A770). Relevant again now
   that the mtp arm is being run: it costs the B580 cache slots for nothing, and the A770 was squeezed to 13000
   free MiB by hand to make room.

**Short prompts (chat, agent turns)**

9. **Re-enable the idle card's one-chunk prompt help across contexts.** Off on Windows today ("stage N reads its
   prompts alone", `generate.cpp:7094`), because the helper's rows are mapped host memory both cards read in
   place. Port the row exchange to copies, the way `Verifier::forward_handoff` already does for the window
   hand-off. Upstream on 2x RTX 3090: 1.1K-token prompts 496 -> 723 tok/s, 2K 878 -> 1,128, 3K 1,260 -> 1,440;
   no gain from ~3.3K tokens. Helps chat and agent turns, not VicGPT's map step. Medium-size port.

**Housekeeping**

10. **Merge upstream SYCL 0.1.45-0.1.48 in small steps.** It holds the large-GRF kernels and the planar Q6_K
    head (sections 5 and 6 of this file), the pageable-DMA fix (already ported) and upstream's Alchemist defaults. One step per arm round, with
    the gate, so a new storm can be tied to a step.

## XeStrata (2026-10-30)

`github.com/MistVVK/XeStrata` (head `721afe5`, xe0.2.8-1). An Intel-first port of upstream Strata, separate from
upstream's own `sycl/` port that this fork is built on. Its GPU code was rewritten to SYCL + Level Zero from
Strata 0.1.24 and has taken upstream's changes through 0.2.8 (our `sycl/` tree is 0.1.44.1). Japanese-first docs,
English translations in `docs/*.md`. Read, not built or run here.

**Why it matters.** On their Arc Pro B70 (Linux) it reads prompts 2.1-2.9x as fast and decodes 35-66% faster than
upstream's `sycl/` port, i.e. our engine's base (`docs/XE.md`, 0.2.8 section; generate and serve). Those are their
numbers on a 32 GB card, not ours.

**What it has not tested.** Windows only on an RTX 3050 Laptop; Arc A-series only for arithmetic (an A380, no
model run); no B580 or A770; no two-Arc split on Windows. Its two-GPU split was measured on a B70 + RTX 4070,
which already runs every GPU in its own context (`Runtime::at(device)`), so the per-context problem we fixed
after the fact is its starting design.

**License.** XeStrata is LGPL-3.0-or-later (plus MIT for upstream's untouched files); this repo is MIT. Copying
its code into a file makes that file LGPL. Reimplement the ideas below from the description, or decide on the
license first.

### A. Try XeStrata itself on this PC (largest possible gain, unknown risk)

A same-day comparison is cheaper than porting 0.1.44 -> 0.2.8 by hand. It ships a Windows 7z
(`XeStrata-<version>-windows-x64.7z`, GitHub Releases, with `SHA256SUMS`; needs the .NET 10 runtime and the VC++
redistributable). Ask the user before any download and name the file and its size.

- Point it at the existing IQ2_XS files instead of downloading again (setup takes a `--gguf-dir`; it never
  deletes files there).
- Check RAM first: its table asks ~48 GB RAM for IQ2_XS. With 32 GB it should fall into its "low-RAM mode" (VRAM
  makes up for RAM); check what `xestrata --check` / setup says before loading.
- Order: B580 alone, then A770 alone, then both with the split fixed at 22 (`--gpus` with B580 first,
  `--layer-split 22`; its `auto` may pick a different split or the helper role). Same stop signals as always.
- Compare against our k22 on the same day and the same prompts (4K, 8K, chat); its API is OpenAI-compatible on
  port 8095, so `tools/strata_bench.py` should reach it. Check the output for storms the same way.
- Known Arc-on-Windows risks it has not met: the A770 misreading allocations past 4 GiB (our
  `STRATA_SYCL_JIT_STATELESS` fix), the two Arcs as separate Level Zero platforms, the WDDM shared budget.

If it runs clean and faster, the fork's question becomes "what does XeStrata lack for VicGPT", not "what do we port".

### B. Ideas to reimplement in our fork

1. **Make the "pinned" arena actually pinned, per card.** XeStrata registers its ordinary-memory arena with
   `sycl::ext::oneapi::experimental::prepare_for_device_copy` (the SYCL counterpart of `cudaHostRegister`; our
   `pinned.dp.cpp:451` has a DPCT no-op there), in pieces no larger than the device's largest allocation, **once
   per GPU context** (`PinnedArena::register_on(device)`, `src/core/pinned.cpp`). After registering it copies both
   ends of every piece to the card and back and compares, because a UHD 770 accepted a registration and then
   copied wrong bytes. On Windows it first checks the card's WDDM shared-memory budget and skips registration
   past it (copies stay correct, just slower).
   - Check first: is `SYCL_EXT_ONEAPI_COPY_OPTIMIZE` defined in oneAPI 2026.1's headers? (A different extension
     from `SYCL_EXT_ONEAPI_USM_IMPORT`, which section 16 found missing.) A probe like `graph_probe.cpp` can
     register 256 MiB on both cards' contexts and time a copy.
   - What it buys: real DMA for arena and mirror shapes, and it is the base for the A770 mirror (improvement 4 in
     the 10-30 list). The daily `--stream-experts` shape already copies through the stager's pinned buffers, so
     expect little there.
   - **Task Manager:** registered host memory may count as "Shared GPU memory". That would trip the 0.2 GB stop
     signal while being normal. Agree on how to read it before the first run.
2. **A per-stage check that a running kernel sees host writes.** XeStrata's verifier tests at start that a
   kernel spinning on a host flag sees the host's write (`doorbell_visible`); where it fails it runs the window
   as host-launched segments instead (`STRATA_VERIFY_SEGMENTED`). Our verify windows spin on host flags on both
   cards, and the A770's flags must live in the A770's own context. A start-up check per stage is cheap
   insurance and rules a whole class of silent wrong-data faults in or out for the storm hunt.
3. **"Nothing queued on a card behind a window that waits for the host."** On the B70, a graph or copy submitted
   while a verify window spun on the host did not return, and the engine stalled. XeStrata queues a window's
   commit only after the window completes, and runs the adaptive tier only with no window in flight on either
   card. For our MTP pollution lane: list what the drafter queues on the A770 while a window there waits on the
   host, and test the serial ordering.
4. **Check the B580 stage's PCIe share; try `--pcie-frac 0`.** XeStrata now splits misses between the CPU and
   PCIe from times measured at start. Their RTX 4070 on PCIe 4.0 x4 decoded 15% faster once every miss went to
   the CPU (IQ3_XXS 32K 49.9 -> 57-58.6 tok/s). Our B580 is on PCIe 3.0 x4, slower still. Read the B580 stage's
   PCIe share in the start log; if it is above 0, arm `k22pcie0` (`--pcie-frac 0`) and read decode.
5. **Bigger prompt chunks when VRAM is short.** XeStrata's `--prefill auto` (from upstream #583) falls back to a
   96-slot ring, then sizes the int8 products' group scratch for one group's rows instead of the whole chunk. On
   an RTX 4070 (IQ3_XXS, 32K) that raised the chunk from 8192 to 16384 and the prompt from 1,194 to 2,058 tok/s.
   Our chunk is limited by VRAM too (2048 forced without borrowing, 4096 with own buffers). Pairs with the
   chunk-size arms in the 10-30 list.
6. **Concurrent map chunks: `--batch 2 --batch-groups 2`.** VicGPT's map chunks are independent. With two slots
   in two groups on a layer split, each card works on one group while the other card works on the other.
   XeStrata/upstream: 2x RTX A4000 ran four requests 25-30% faster in groups than in one group; B70 + 4070
   batch 8 went 73.1 -> 77-79 tok/s. Our tree has `--batch-groups`. Costs VRAM per slot (fewer cached experts),
   needs VicGPT to send two requests at once, and batching across two Windows contexts is untested.
7. **Prompt hand-off in pieces** (`STRATA_HANDOFF_PIPE`, upstream `1d9758f8`). A chunk's rows go to the next card
   in eight pieces so both cards' PCIe copies overlap. A 4096-token chunk's rows are ~210 MB, about 60 ms out
   of the B580 over PCIe 3.0 x4: a 1-2% estimate for a 4K prompt. Low priority.
8. **Use `STRATA_DECODE_TIMING=1`** (already in our tree) to see whether the GPU waits on the CPU's experts. That
   decides between items 4 and 6 above and improvement 4 (the A770 mirror) in the 10-30 list.

### Not usable here

- `--pipeline-windows 2` by default on two Xe2 cards (+17-28% decode on two Arc Pro B65): needs the MTP drafter
  (it pollutes on our split) and two Battlemage cards.
- Split serving (one card reads prompts, the other decodes): needs `--mtp`, two whole-model engines and RAM for
  two expert copies on Windows.
- Route prior (`STRATA_ROUTE_PRIOR`): upstream and XeStrata took it back out of the defaults (it repeats itself in
  long agent sessions).
- `joint_matrix_prefetch` removal (IGC 2.28: B70 1K prompt 468 -> 751 tok/s): our tree does not call it.
