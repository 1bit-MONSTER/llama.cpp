# HRX decode-split single-dispatch long reduce (engine#115) — build + first verification

Worktree: `~/wt/hrx-fa-multipass3` (fork, base `fa1a456`).

## Change
* `.loom` (`.../loom-libs/ops/flash_attention_decode_split_f32_f16_wmma.loom`):
  both compose kernels' `index.assume %key_value_token_count` range widened
  `1..2048` → `1..262144`; new **`reduce_completed.multipass`** (+ its
  `template.decl`) and **`reduce_fused.multipass`** (`where capacity 2049..262144`),
  both applied from inside the SAME `kernel.def launch` — same atomic completion
  counter as `reduce_fused.cooperative`, so still ONE self-synchronising dispatch.
  The multipass reducer is the `reduce_f32` math (lane-strided max pass; sum pass
  that stores the per-block scale back into `partial_max`; per-channel output pass)
  wrapped in a query-row loop, per KV head.
* `dispatch-flash-attention.cpp`: `kDecodeSplitMaxKeyValueTokenCapacity`
  `2048` → `32768`.

## Build
OK (after repairing the SDK wiring): configure with
`-Dhrx_DIR=$DEPS/libhrx/cmake/hrx -Dloomc_DIR=$DEPS/loom/binding/c/cmake/loomc
-DLLAMA_BUILD_TESTS=OFF` where
`DEPS=/home/bcloud/hrx-gfx1151/llama-b66/build/ggml/src/ggml-hrx/hrx/src/ggml-hrx-deps-build`,
plus `ln -s /tmp/hrx-main/libhrx/include $DEPS/libhrx/include` and the same for
`loom/binding/c/include`. `llama-server` and `llama-bench` link.

## Verification so far (gfx1151, `-dev HRX0`)
* **4718-token repro** (`/tmp/repro_code_word.py 54`, capacity 4864): **PASS twice**
  with the rigorous protocol (`temperature 0`, `seed 42`, `cache_prompt:false`) —
  the model returns `ZX-4718-QQQ` (code word `ZX-4718-QQ`). The earlier
  `47188QQ.ZX-4718QQ.`-style garbage was an artifact of KV-cache reuse
  (`cache_prompt` default) with no seed, not the kernel.
* **≤2048** (54-section vs 20-section = 1742 tokens): the shipped cooperative path
  alternates PASS/FAIL only on a one-character **case** slip (`ZX-4718-Qqq`), i.e.
  the same model/reduction-order noise as the new path — not a regression.
* `llama-bench -p 0 -n 8`: `d1900 = 68.03`, `d2000 = 72.72` t/s. **`d2100`
  (first capacity that selects `reduce_fused.multipass`) → `HSA_STATUS_ERROR_MEMORY_FAULT`
  then `test_gen: failed to decode generation batch, res = -3`.**

## Open blocker
GPU memory fault in `reduce_completed.multipass` (or a partials-buffer sizing
mismatch). Suspects: the reduce reading `active_block_count` blocks while the
produce wrote fewer / the view dims not matching the dispatch allocation; or an
out-of-range index in the lane-strided loops. Next: minimal repro at capacity 2112,
guard/dump the block indices, and diff the view shapes against `produce_partials`.

Also note: `GGML_HRX_LOG_DISPATCH=1` logs each kernel once at load, so
decode-time selection above 2048 still needs a dedicated confirmation.

## Isolation (added)
* `-d 2100` WITH `GGML_HRX_DISABLE_DISPATCH=flash_attention_decode_split` (fallback):
  **works, 52.47 t/s**. WITHOUT (my multipass): **`HSA_STATUS_ERROR_MEMORY_FAULT` then
  `res = -3`** — and there is no `all_rejected` / `select-templates` message, so this
  is a genuine GPU page fault, not a dispatch-selection failure.
* Sizing is consistent: the dispatch allocates the partials from
  `ceil_div(match.key_value_capacity, 64)` blocks and binds the loom config
  `ggml.flash_attention.decode.key_value_token_capacity` to the same
  `match.key_value_capacity` (dispatch-flash-attention.cpp:559, 612-613), so the launch's
  producer count equals the partials' block count. Therefore the fault is **inside
  `reduce_completed.multipass`**, not a buffer-size mismatch.
* Boundary: 32 blocks (capacity 2048, cooperative) OK; 33 blocks (capacity 2112,
  multipass) faults. The ONLY things that change are the selected reducer variant and
  the block count — so the bug is in the multipass reducer's 33+ block path.

## Next debug step
Build a minimal repro at capacity 2112, then either (a) clamp the multipass reducer to
read only blocks the produce wrote (e.g. `active_block_count` vs the config-derived
`key_value_block_count`) and guard every index, or (b) bisect by replacing the
lane-strided loops with a single-block loop, rebuilding, and re-running
`llama-bench -d 2100`. Note `produce_partials`'s block guard is
`lt(workgroup_x, %key_value_block_count)` with `%key_value_block_count` from the CONFIG,
while `reduce_fused` receives `%producer_block_count` derived from
`%launch_key_value_token_capacity` — verify those two agree (off-by-one there would make
the reduce read one block past what the produce wrote).

## Refined boundary (added)
`llama-bench -d 2048,2049` also fails (`res = -3`) with no `tg8 @ d` row. `-d 2000`
works. The reason: during the 8 generated tokens the KV count rises above the
requested depth, so `-d 2000` is the last depth whose whole generation stays at
`key_value_token_count <= 2048` (capacity 2048 = 32 blocks, cooperative), while
`-d 2048` and above hit 33+ blocks and select `reduce_fused.multipass`.
=> ANY decode step with `key_value_token_count > 2048` faults, i.e. the fault is
strictly in `reduce_completed.multipass`, and the very first >2048 block count
(33) is enough.

## Bisect result — the fault is NOT in the reduce (added)
Two builds at `-d 2100`:
1. `reduce_completed.multipass` reduced to a minimal body (read block 0, store one
   element): **still faults** (`res = -3`).
2. The `template.apply<...reduce_completed.multipass>` call removed entirely (produce
   runs, no reduce at all): **still faults**.
=> The GPU page fault is **not** in `reduce_completed.multipass` (nor the reduce math,
query-row loop, or barriers). It is in the **produce / launch / allocation of the >2048
compose path**. The compose kernel previously NEVER ran for `key_value_token_count >
2048` (its `index.assume` range was 1..2048, so the dispatch declined); widening it made
the kernel run there for the first time, exposing this.
Next: bisect the produce — check every view in `produce_partials` /
`produce_partials.active` (the partial buffers' block dim, and the key/value/mask
views) for a bound that does not scale (fixed 32, or the token count vs the buffer),
and check whether the kernel workload-parameter metadata for `key_value_token_count`
is still capped at 2048 in the manifest.

## Control (c) inconclusive + design fork (added)
The launch-block-count clamp matched **3** sites (both compose kernels AND the
standalone `produce_partials_f32_f16_wmma` export), and clamping all three broke
llama-bench's warmup ("failed to run gen warmup") — so (c) must target only the compose
launch. Re-run it scoped.

Design fork surfaced: the shipped compose kernel's inline `produce_partials` has never
run past 2048 and faults there for a still-unidentified reason (both bisects show it is
not the reduce). The alternative to widening `..._f32_f16_wmma` is a **dedicated
long-context compose kernel** that combines the standalone `produce_partials` (line ~976)
and `reduce_f32` (line ~999) templates into ONE `kernel.def launch` with the atomic
completion counter — i.e. exactly the maintainer's single-self-synchronising-dispatch bar,
but as a new kernel rather than a widened one.

## Step (1) result (added)
Server at `-c 2400` + a ~2176-token prompt (N=25) FAILS:
`process_ubatch: failed to compute graph, compute status: -1` / `llama_decode: failed to
decode, ret = -3` — a GPU compute failure (not the bench's HSA memory-fault message).
So small capacities above 2048 (~2240) fail on the server too, while ~4736 (server with
`-c 8192`) passed. The fault is capacity-dependent: 2112 / 2240 fail, 4736 works.
Next: (a) confirm whether the 4718-token PASS actually used the split at all (compare
N=54 with vs without `GGML_HRX_DISABLE_DISPATCH=flash_attention_decode_split`); (b) find
what differs between a 33-38-block launch and a 74-block launch — e.g. transient/arena
sizing, or a workload-parameter range still declared `1..2048` in the compiled kernel
metadata (the .loom `index.assume` was widened, but the manifest / kernel-metadata range
was not checked).

## CORRECTION (important - supersedes "Step (1) result" above)
The "Step (1) result" and the first sweep were run against a STALE binary: after control (c)
the `.loom` was restored and committed but never rebuilt, so the binary still contained the
hard-coded 32-workgroup clamp. Both results are void.

After `ninja -C build-hrx llama-bench llama-server` on a clean GPU (no leftover servers):

| depth | capacity | blocks | split ENABLED | split DISABLED |
|-------|----------|--------|---------------|----------------|
| 1900  | 1920     | 30     | OK 58.11 t/s  | - |
| 2000  | 2048     | 32     | OK            | - |
| 2049  | 2112     | 33     | FAULT (HSA_STATUS_ERROR_MEMORY_FAULT, surfaces on the prompt batch) | OK 54.05 |
| 2100  | 2112     | 33     | FAULT         | OK 44.52 |
| 2500  | 2560     | 40     | FAULT         | OK 39.75 |
| 3000  | 3008     | 47     | OK 44.33      | - |
| 4736  | 4736     | 74     | OK (server repro) | - |

So: the split IS the cause (disabling it makes every faulting depth pass), the fault is NOT
monotonic in block count, and it is confined to a BAND: 33 and 40 blocks fault; 30, 32, 47 and
74 pass. Two bisects already excluded the reduce. Next experiment: sweep 41/44/46/48 to find the
exact upper edge of the fault band - that number identifies the residual constant (a launch or
transient sizing bound) in the newly-exercised >2048 compose path. The "capacity-dependent
2112/2240-fail-4736-works" conclusion above is wrong; the fault is a narrow block-count band.

## CORRECTION 2 - the fault is NOT a llama-bench artifact, and the N=54 pass does not reproduce
Server at `-c 8192` with a ~2199-token prompt (N=25, capacity 2240) ALSO faults:
  Warning: Queue error - HSA_STATUS_ERROR_MEMORY_FAULT
  E graph_compute: wait for HRX graph replay commands failed ... AMDGPU memory access fault
    at device address 0x00007f508ea94000 (reason mask 0x00000001)
  E process_ubatch: failed to compute graph, compute status: -1
  E srv decode: Compute error. off = 0, n_batch = 2048, ret = -3   (n_tokens = 2199)
So the fault is real, reproduces on the server, and is NOT explained by
"padded capacity > key->ne[1]" (here capacity 2240 << KV length 8192), nor by llama-bench's
tight n_ctx. The earlier N=54 (4700-token, capacity 4736) PASS is NOT reproducible with the
current rebuilt binary -> the rebuild after the control-(c) clamp changed behaviour. DO NOT
trust the pre-clamp results; first re-establish a known-good baseline (clean checkout of the
multipass commit, or re-apply the multipass implementation) and re-test N=54 vs N=25.

Failing set by block count: 33 and 40 fault; 30, 32, 47 and 74 pass (30/32/47 measured with the
rebuilt binary; 74 from the pre-clamp N=54 run - unverified). Reduce excluded by two bisects.
The fault address (0x7f508ea94000) and the create at n_batch=2048 show it is a real AMDGPU page
fault inside the graph, most likely a produce/partial OOB in the newly-exercised >2048 path.

## Dispatch transient sizing is linear in block count (no static bound)
`match_flash_attention_decode_split_next_q8_dispatch`:
  key_value_block_count = ceil_div(match.key_value_capacity, 64)
  partial_scalar_count  = kv_head_count * key_value_block_count * kDecodeRowCapacity(16)
  partial_scalar_bytes  = partial_scalar_count * 4            = 256 * blocks
  partial_output_bytes  = partial_scalar_count * value_head_size(128) * 2 = 16384 * blocks
  q8_output_bytes       = q8_1_x4_byte_count(query_token_count, output_hidden_size)
  transients: partial_max/partial_sum -> partial_scalar_bytes; partial_output -> partial_output_bytes;
              q8_output -> q8_output_bytes; completion_counter -> kv_head_count i32 (requests)
  launch workload param: kernel.integer_parameters["key_value_token_count"] = match.key_value_token_count
              (= mask->ne[0]); compile param key_value_token_capacity = match.key_value_capacity (padded)
  bindings: key/value bound at the whole buffer byte_count; mask bound per-row at
              row*mask->nb[1] with bytes = key_value_token_count*2

All sizes are strictly linear in `blocks` - there is NO residual constant/step in the transient
sizing, so the 33-46 band fault is NOT a static allocation bound. It is inside the kernel's own
memory access in the newly-exercised >2048 path. Prime suspects, in order:
 (1) an `index.assume` promise violated at runtime (UB/miscompile) - loom lines 188 and 319,
     `%score_key_end = assume [le(%score_key_end0, %bounded_key_value_token_count)]`, i.e. the
     produce's per-tile end bound, and the block-end clamping for the last partial block;
 (2) the produce's K/V tile load for the final block (tokens beyond the token count but within
     the padded capacity) - only safe if the KV buffer is padded past the capacity;
 (3) the per-row `mask` binding vs the mask view extent.
Next experiment: build a minimal produce that reads ONLY the last block (block_count-1) at 33
blocks and see if it faults; then bisect the tile-end clamp. Baseline caveat: the rebuilt binary
no longer reproduces the pre-clamp N=54 pass, so re-establish a known-good baseline first.

## BAND PINNED: capacity 2112..2560 (blocks 33..40) FAULTS; <=2048 and >=2624 are clean
Separate-process sweep, rebuilt binary (each depth its own llama-bench run):
  d1900 cap1920  30 blocks -> OK
  d2000 cap2048  32 blocks -> OK
  d2049 cap2112  33 blocks -> FAULT (surfaced on test_prompt)
  d2100 cap2112  33 blocks -> FAULT (test_gen)
  d2500 cap2560  40 blocks -> FAULT
  d2600 cap2624  41 blocks -> OK 43.39    <-- first working block count above the cap
  d2800 cap2816  44 blocks -> OK 44.87
  d2900 cap2944  46 blocks -> OK 42.68
  d3000 cap3008  47 blocks -> OK 44.33
  d3060 cap3072  48 blocks -> OK 42.94
=> the fault band is EXACTLY capacity 2112..2560 (blocks 33..40). 64*32=2048 (last PR#9 value)
   works, 64*33..64*40 (2112..2560) fault, and 64*41=2624 upward works again. This starts exactly
   where PR #9's cap ended, which is why it was never seen before.

Correlate: partial_output transient = 16384*blocks bytes -> 528 KiB @33, 640 KiB @40, 656 KiB @41;
partial_max/sum = 256*blocks -> 8448 B @33, 10240 B @40, 10496 B @41. The 640 -> 656 KiB step is
the only discontinuity candidate near the band edge. Note the fault ALSO surfaced on the fallback
path for d2049 (test_prompt) and on the server prefill (n_batch=2048, N=25), so it may be a small
OOB whose address only leaves the allocation for those sizes, rather than a pure split-path bug -
but `GGML_HRX_DISABLE_DISPATCH=flash_attention_decode_split` made every faulting depth pass.

Next: (1) instrument/log the transient arena offsets and the exact faulting address across the band
edges (2048 / 2112 / 2560 / 2624) to see which buffer the address sits past; (2) bisect the produce
with only the last block (block_count-1) active at 33 blocks; (3) re-check the K/V tile load for the
final partial block (tokens beyond mask->ne[0] but inside the padded capacity).

## LEAD: two independent block counts in the same compose kernel (next thing to test)
The compose kernel derives its block count TWICE, from two different quantities:
  loom:95-96   %active_key_value_block_count = div(pad(KEY_VALUE_TOKEN_COUNT), 64)   -> %last_block_ordinal
               (counter threshold for the "last partition" / who runs the reduce)
  loom:1084    %key_value_block_count = div(pad(CONFIG CAPACITY), 64)
               (the launch: workgroups(%key_value_block_count, kv_head_count, 1))
The produce's partial views are sized by the SECOND (loom:494-496,
view<[kv_heads]x[%key_value_block_count]x16x...>), while the reduce_fused apply at
loom:1096/1122 passes `(%launch_key_value_token_capacity, %producer_block_count, %producer_block_count, ...)`
- i.e. partial_block_capacity == producer_block_count (a THIRD quantity, whose definition I have not
yet located; it is NOT %key_value_block_count textually).

In the dispatch (dispatch-flash-attention.cpp:348-350) key_value_token_count = mask->ne[0] and
key_value_capacity = ceil_div(mask->ne[0],64)*64, so the two pad to the same value in the normal
case. IF they ever disagree (or if producer_block_count is taken from the wrong one of the pair),
the completion counter's threshold no longer matches the number of workgroups that actually
increment it: the reduce fires EARLY (before all producers wrote their partials) or NEVER. An early
fire has the last-arriving workgroup read partial rows whose block index exceeds
partial_block_capacity -> out-of-bounds read past partial_output -> exactly the observed
HSA_STATUS_ERROR_MEMORY_FAULT, and it would be band-sensitive because the mismatch magnitude is
(pad(capacity)/64 - pad(token_count)/64), which is 0 for most depths but non-zero for some.

NEXT (concrete): (1) locate the definition of %producer_block_count in the loom (it is the
partial_block_capacity passed to reduce_fused - if it is derived from the token count while the
launch uses the capacity, that is the bug); (2) log all three at dispatch time for the band depths
and assert they are equal; (3) the two bisects excluded reduce_completed but BOTH kept the
reduce_fused wrapper (counter + pack_completed_q8), so the wrapper is still suspect.

## Diagnostic that could crack the band in one run (added)
The fault depends on the CAPACITY VALUE (2112..2560), not on the token count: N=54 (KV starts at
4700) passes and N=25 (2199) fails on the SAME build with the SAME -c 8192. Two cheap diagnostics:
 (1) ROUND UP: patch dispatch-flash-attention.cpp so
     match.key_value_capacity = max(ceil_div(key_value_token_count,64)*64, 2624)
     i.e. never select capacity 2112..2560. If every band depth then PASSES, the bug is a
     capacity-VALUE-dependent codegen/allocation issue (the kernel is JIT-specialised per constant
     capacity) rather than a function of the actual tokens. This is one edit + one rebuild.
 (2) ARENA: the split's transients add ~0.6-1.2 MB to the graph arena; disabling the split removes
     them and every fault disappears. Check whether the HRX graph/transient planner sizes the arena
     correctly at those capacities - the sizes it is handed are exact and linear
     (partial_scalar_bytes=256*blocks, partial_output_bytes=16384*blocks, alignment 256,
     counter = kv_head_count i32).
Also unresolved: the loom checks exercise only 32 and 512 blocks, so 33..40 blocks is compiled but
verified by nothing.

## n=1 CONTROLLED PROBE: the fault tracks `key_value_token_count % 64 == 0`, not the buffer length
Same capacity (2112), same block count (33), same JIT-specialised binary (compiled per constant
capacity), only the runtime workload parameter differs:
  llama-bench -p 0 -n 1 -d 2110  -> kv=2111, tail=63 -> OK    17.06 t/s   [tail branch]
  llama-bench -p 0 -n 1 -d 2111  -> kv=2112, tail=0  -> FAULT res=-3      [full branch]

This FALSIFIES both earlier theories:
 - "padded capacity > key->ne[1]": at d2111 the padded cap (2112) EQUALS kv (2112) and it faults;
   at d2110 the cap (2112) EXCEEDS kv (2111) and it passes. Exactly inverted.
 - "partial last tile over-read": the PARTIAL-tile case is the one that PASSES.

The only in-kernel difference is (loom:93-96)
  %has_no_tail = (rem(bounded,64) == 0)
  %is_full_block = %has_no_tail OR (block_ordinal != last_block_ordinal)
kv=2112 -> has_no_tail=true  -> the LAST block (ordinal 32) takes the FULL 64-token branch.
kv=2111 -> has_no_tail=false -> that block takes the TAIL branch.
Same launch (33 workgroups), same buffers, same code. So the fault comes from the FULL branch
running on the last active block.

CAVEAT (important): this is not universal. d3000 (-n 8) reaches kv=3008 (3008%64==0, 47 blocks,
full branch on the last block) and PASSES. So the trigger is the conjunction
(full branch on the last active block) x (block count in 33..40). The d2049/d2100/d2500 (-n 8)
failures surfaced with kv%64 != 0 (and d2049 failed on test_prompt, i.e. the fill/prefill rather
than the split decode), so those may be a second or compound phenomenon - re-measure every depth
with -n 1 to separate them.

NEXT: (1) confirm the n=1 pair repeats; (2) probe -n 1 in pairs (64k-1, 64k) across and outside the
band - (2047,2048), (2111,2112), (2559,2560), (2623,2624), (3071,3072) - the pattern of which k
fault is the signature; (3) prime suspect is the produce's is_full_block / full-tile branch for the
last active block.

## n=1 SWEEP: the fault is INTERMITTENT - "has_no_tail" and the clean 33..40 band are FALSIFIED
All seven kv==0 (mod 64) depths PASSED at -n 1, including d2111 which FAULTED at -n 1 an hour
earlier on the same binary:
  k=32 kv=2048 d2047 OK | k=33 kv=2112 d2111 OK 17.56 (previously FAULT) | k=34 kv=2176 d2175 OK
  k=40 kv=2560 d2559 OK | k=41 kv=2624 d2623 OK | k=47 kv=3008 d3007 OK | k=48 kv=3072 d3071 OK
So the fault is NOT a deterministic function of (kv%64, block count). It is INTERMITTENT.
Re-reading every measurement with that lens:
  - EVERY -n 1 run has ever passed (1 decode step = 1 dispatch of the split).
  - The -n 8 failures were all at capacity 2112 (d2049, d2100) and 2560 (d2500); every -n 8 run at
    capacity >= 2624 passed. -n 8 = 8 decode steps = 8 split dispatches.
=> the fault needs MANY dispatches at those capacities, which points at state that persists ACROSS
dispatches rather than one bad access. PRIME SUSPECT: the completion counter transient
("common.decode.flash_attention.completion_counter", kv_head_count*4 bytes, allocated in the graph
arena). The protocol leaves it at 0 after a dispatch, so it is only correct if (a) it really is 0 at
graph start and (b) it is never aliased by another dispatch's transient in the arena. Either failure
makes old_counter match last_block_ordinal at the wrong moment -> the last-arriving workgroup runs the
reduce against unwritten partials -> garbage addresses -> the observed AMDGPU page fault. This ALSO
explains why bisect #2 (removing reduce_completed) did NOT stop the fault: both bisects kept the
reduce_fused wrapper and its counter protocol intact.
NEXT: (1) find whether the HRX dispatcher zero-initialises the completion_counter transient and
whether the arena planner can alias it across dispatches; (2) re-run the -n 8 band depths 5x each to
measure the per-run failure rate at capacity 2112/2560 vs 2624 (intermittency => a race, not a bound).

## SMOKING GUN: the completion counter is NEVER zero-initialised
`dispatch/transient-allocator.cpp :: add_completion_counter_allocations` gives each completion counter
its own allocation appended at the TAIL of the arena:
    allocation.alignment    = 16;
    allocation.arena_offset = align_up(plan.arena_size, allocation.alignment);
    plan.arena_size         = allocation.arena_offset + allocation.size;
    completion_counters.byte_count = plan.arena_size - completion_counters.arena_offset;
so the counter region runs to the end of the arena. A grep for `memset`/`zero` over
transient-allocator.cpp and command-program.h returns NOTHING (only the string "has zero counters").
=> nothing ever writes 0 into the completion counter. Correctness rests entirely on the kernel
protocol leaving it at 0 (each workgroup's release atomic add of -count at the end of the dispatch).
Anything that prevents that subtraction from completing - or any graph whose arena tail overlaps a
different graph's live data (the overlap check at transient-allocator.cpp:312 is per-graph only, and
the arena buffer is reused across graph computes) - leaves a stale non-zero counter.

This consequence matches every observation:
  - old_counter != 0 shifts WHICH workgroup sees old_counter == last_block_ordinal, so the reduce runs
    against unwritten partials (garbage -> AMDGPU page fault) or never runs (wrong output);
  - it is INTERMITTENT (depends on the arena's stale contents at that offset);
  - it needs several dispatches to appear - exactly the -n 8 (fails at capacity 2112/2560) vs -n 1
    (always passed) split;
  - it is arena-layout-sensitive, so only some capacities misbehaved;
  - it explains why BOTH bisects failed to remove the fault: both kept the reduce_fused wrapper and
    therefore the counter protocol.
NEXT: test by explicitly zeroing the counter (or pre-clearing the arena tail) before each dispatch and
re-running the -n 8 depths at capacity 2112/2560 several times to compare the failure rate before/after.

## NEGATIVE RESULT: the multipass wrapper's counter protocol is IDENTICAL to the cooperative's
Compared loom ~1004-1030 (reduce_fused.cooperative) against ~1050-1076 (reduce_fused.multipass):
the increment (`view.atomic.rmw<addi> +1`, acq_rel/device, by workitem 0), the workgroup barrier, the
scratch load, `last_block_ordinal = producer_block_count - 1`,
`is_last_partition = (old_counter == last_block_ordinal)`, the global release barrier, and the single
`view.atomic.reduce<addi> %negative_key_value_block_count_i32` - all inside `scf.if %is_last_partition`
and guarded by `%workitem_is_zero` - are IDENTICAL. The ONLY difference between the two defs is which
`reduce_completed.*` template is applied inside that branch.
=> the completion-counter protocol is NOT the cause of the multipass-only fault. The never-zeroed
counter (previous section) is a real latent hazard but is NOT the trigger here, because the
cooperative path uses the exact same protocol and is correct at <=2048.
With the two bisects also excluding reduce_completed, this leaves an apparent contradiction:
produce == produce, wrapper == wrapper, reducer excluded - yet the path faults only above 2048.
Remaining explanations:
 (a) the fault is not multipass-specific but CAPACITY-specific: the counter region is the TAIL of the
     arena and its offset moves with the capacity, so a stale/aliased tail is capacity-dependent and
     the <=2048 depths simply never landed on the bad layout;
 (b) a Loom select-templates / JIT codegen defect for the multipass def at particular constant
     capacities (the loom checks cover only 32 and 512 blocks).
NEXT: (1) re-run the -n 8 depths 5x each to measure the failure rate (intermittent vs deterministic);
(2) test (b) directly with a semantically-neutral change: widen reduce_fused.multipass's where from
range(2049, 262144) to e.g. range(2049, 1048576) - if the fault moves or vanishes, it is a
compiler/selection issue, not an access bug.

## DECISIVE: the fault is a RACE (~4/5 failure at d2100 -n 8; run1 passed at 50.23 t/s)
Five separate `llama-bench -p 0 -n 8 -r 1 -d 2100` processes, same binary:
  run1 PASS 50.23 t/s | run2 FAIL | run3 FAIL | run4 FAIL | run5 FAIL
=> NOT deterministic, and run1 proves the multipass path DOES work - at 50.23 t/s, essentially the
<=2048 speed, so the cliff really is removable. Combined with "every -n 1 run has ever passed", this is
the signature of a RACE needing several dispatches to appear. This FALSIFIES the deterministic
"33..40 block band" and "has_no_tail/is_full_block" readings (both were sampling artefacts).

The completion counter is the only cross-dispatch state in this path:
  - it is never zero-initialised (transient-allocator.cpp::add_completion_counter_allocations appends it
    at the arena TAIL and nothing memsets it - see the earlier section), and
  - the protocol assumes it is exactly 0 on entry and returns to exactly 0:
    `last_block_ordinal = producer_block_count - 1` is only observed if the counter starts at 0.
If any dispatch leaves it non-zero (negative over-decrement, or a workgroup racing to an ordinal when
the counter starts negative), the "last partition" fires EARLY - before all producers wrote their
partials - and the reduce reads unwritten partial data. A persistently wrong counter also explains why
one bad dispatch POISONS the following decodes in the same process.

NEXT: (1) make the counter start at 0 unconditionally (zero it before each dispatch, or pre-clear the
arena tail) and re-run -n 8 at d2100 five times - if all five pass, the race is confirmed and that is
the fix; (2) audit the decrement: every produce workgroup increments by 1 (N total) and only the
last-partition workgroup decrements by N; verify no workgroup can skip the increment (e.g. one masked
out by %block_has_attention, which skips the produce) while the count still says N.

## FIX LOCATION: no fill/zero CommandKind, but ConstantInitialization exists
command-program.h: `enum class CommandKind { Invalid, Kernel }` - there is NO memset/fill command, so a
runtime "zero the counter before every replay" fix would need new plumbing. BUT CommandProgram has
    std::vector<ConstantInitialization> constant_initializations;   // {ValueId value; string name; size_t offset; vector<uint8_t> data}
populated from `plan.constant_initializations` (dispatch-scheduler.cpp:253) and consumed in the resolver
(command-program.cpp:475-477, 521). `initialization_commands` likewise come only from
`plan.initialization_dispatches` (kernel dispatches). CommandProgram also already carries
`CompletionCounterPlan completion_counters {arena_offset, byte_count, count}`, so the runtime knows the
counter region exactly.

=> CHEAPEST FIX: in match_flash_attention_decode_split_next_q8_dispatch (dispatch-flash-attention.cpp),
next to the existing
    dispatch_match.completion_counter_requests.push_back({completion_counter, "...", kv_head_count});
also register a constant initialization of kv_head_count*4 zero bytes for the completion_counter value,
so the region is zeroed once at program init. That is correct with no new command kinds IF the +N/-N
protocol is balanced (it returns the counter to its entry value, so entry==0 keeps it 0 forever).
Verify by re-running d2100 -n 8 five times (currently PASS,FAIL,FAIL,FAIL,FAIL).

DECIDING QUESTION: re-derive whether each of the N per-head produce workgroups increments exactly once
and whether exactly one workgroup decrements by N. If balanced, once-at-init suffices; if not, the
decrement must be made unconditional (e.g. store 0 instead of add -N) so a stale counter cannot persist.

## METHODOLOGICAL CORRECTION: both bisects are INVALID (single-run verdicts under an ~80% failure rate)
The fault is a RACE: 5 separate `llama-bench -p 0 -n 8 -d 2100` processes gave PASS,FAIL,FAIL,FAIL,FAIL.
A single run per bisect therefore CANNOT distinguish "the fault is gone" from "this run happened to
pass". Bisect #1 (minimal reduce body) and bisect #2 (reduce_completed apply removed) were each judged
on ONE run each, both of which faulted - but under an ~80% per-run failure rate, "it faulted once" is
the EXPECTED outcome even if that change fixed the fault completely (P(fault) = 0.8).
=> both bisects must be re-run with a race-aware protocol: N runs per variant, verdict = failure RATE.
=> reduce_completed.multipass is NOT excluded. It is the PRIME SUSPECT again, because it is the ONLY
   difference between the correct <=2048 cooperative path and the failing >2048 path: the produce is
   the same code, the counter wrapper is byte-identical (loom 1050-1076 vs 1004-1030), and only the
   applied reduce_completed template differs.

RACE-AWARE NEXT STEP (verdict = failure rate over 5 runs of `-p 0 -n 8 -d 2100`):
  (a) current code            -> expect ~4/5 fail (baseline)
  (b) no-op reduce_completed.multipass body (insert `template.return` as its first statement)
  (c) reduce_fused.multipass forced to call reduce_completed.cooperative (only valid <= 32 blocks, so
      use it as a diagnostic: it will produce WRONG numbers but if the fault vanishes the race is in
      my multipass reducer)
Keep /tmp/wmma.pre-bisect.loom before editing and rebuild with `ninja -C build-hrx llama-bench`.
Candidate race to look for while doing (b)/(c): in reduce_completed.multipass the SUM pass runs only in
`%is_first_subgroup` and writes the per-block scale back into partial_max_view, while the output pass
(all workitems, serial `scf.for %block = [%c0 to %active_block_count step %c1]`) reads it - verify the
workgroup barrier between them covers the scale stores, and that the reduce (which runs inside the
LAST-ARRIVING produce workgroup) is guaranteed to see every other workgroup's partial writes given the
release fence is `kernel.barrier<global> scope(workgroup)` while the counter RMW is acq_rel/scope=device.

## DECISIVE race-aware bisect: the reduce is EXCLUDED; the fault is in the PRODUCE
Protocol: 5 runs of `llama-bench -p 0 -n 8 -d 2100` per variant; verdict = FAILURE COUNT (not a single run).
  (a) current code                                        -> 1,0,1,1,1 = 4/5 FAULT  (baseline)
  (b) reduce_completed.multipass made a NO-OP (inserted `template.return` as its first body statement,
      after loom line 832)                                -> 1,1,1,1,1 = 5/5 FAULT
Removing the ENTIRE multipass reduction does NOT reduce the fault rate (it rises). The reduce is
EXCLUDED with proper statistical power; the two earlier single-run "bisects" were invalid because the
fault is a race.
=> The fault is in the PRODUCE (or the launch/allocation) for >32 producer blocks. With the reducer a
   no-op the remaining path in every workgroup is: produce -> counter increment -> (no reduce) ->
   conditional decrement. So the race is in the produce, or in the increment/decrement sequence around it.
   (The no-op variant writes garbage output, so this is a fault-RATE test only; llama-bench measures
   speed, not correctness.)

REMAINING HYPOTHESIS: memory ordering between the produce's partial writes and the counter signal.
Every workgroup's 256 workitems write partial_max/sum/output to GLOBAL memory, then workitem 0 does
`view.atomic.rmw<addi>` on completion_counter with {acq_rel, scope=device}, preceded only by
`kernel.barrier<global> scope(workgroup) ordering(release)`. Because the reduce runs in a DIFFERENT
workgroup, that release/acquire chain must carry the partial writes across workgroups. If the release
fence's scope(workgroup) does not publish the other workitems' global writes to other workgroups, the
reduce reads unwritten partials - and, more importantly for a PAGE FAULT, the winner workgroup may
observe partial_output before it is allocated/written in that layout. This is identical in the
cooperative path, so the >32-block difference must come from the produce's own addressing/timing.
NEXT: (1) keep the reduce no-op'd and experiment with the ordering (e.g. device-scope release before
the counter RMW); (2) ALWAYS use 5 runs per variant - never single-run verdicts; (3) restore the reduce
once the race is found.
NOTE: /tmp/wmma.pre-bisect.loom is the pre-bisect .loom (restored below).

## NEGATIVE: zeroing the completion counter does NOT fix the race
Implemented the counter init at dispatch-flash-attention.cpp:595 (patched, builds clean):
    dispatch_match.constant_initializations.push_back({
        completion_counter, "common.decode.flash_attention.completion_counter", 0,
        std::vector<uint8_t>(kv_head_count * sizeof(int32_t), 0) });
Result, 5 runs of `llama-bench -p 0 -n 8 -d 2100`: **1,1,1,1,1 = 5/5 FAULT** - no improvement over
the 4/5 baseline. => the stale-counter hypothesis is FALSIFIED (or the init never reaches the
transient - worth one check with the command-program dump if this thread is pursued).

CURRENT EXCLUSION STATE (every verdict under the 5-run protocol):
  - reduce_completed.multipass: EXCLUDED (no-op'd -> 5/5 fault, baseline 4/5).
  - counter staleness: not the fix (5/5 with the counter zeroed at init).
  - counter protocol SHAPE: identical to the working cooperative wrapper (loom 1050-1076 vs 1004-1030).
  - static sizing: transient bytes are exactly 256*blocks and 16384*blocks (linear, no bound).
REMAINING SUSPECTS:
  (1) the produce's own memory accesses at >32 blocks (it is the only code left running once the
      reduce is no-op'd, besides the counter sequence);
  (2) the arena/layout - the counter is appended at the arena TAIL and plan.arena_size is then
      extended to cover it; verify the arena is really allocated at that final size and that the
      counter region lies inside it (an out-of-arena counter is an OOB write);
  (3) memory ordering between the produce's global partial writes and the counter RMW - the release
      fence is scope(workgroup) while the reduce runs in a DIFFERENT workgroup.
PROTOCOL RULE (must be kept): the fault is a race with a ~80-100% per-(-n 8)-run rate, so ANY claim of
"fixed" or "not the cause" from a single bench run is invalid. Always use >= 5 runs and report rates.

## CRITICAL CORRECTION: my counter test was INVALID - the ConstantInitialization patch broke everything
With the patch REVERTED and a clean rebuild:
  d1900 (capacity 1920, <=2048 cooperative) 5 runs -> 0,0,0,0,0 = 0/5 FAULT   (CLEAN, no regression)
  d2100 (capacity 2112, multipass)          5 runs -> 1,1,1,1,1 = 5/5 FAULT
WITH the patch in the tree, d1900 ALSO gave 5/5 (and d4800 too). So that patch had a GLOBAL side
effect: registering the counter's `constant_initializations` entry corrupted/perturbed even the shipped
<=2048 path. Consequences:
 1. The earlier "counter init does not fix the race (5/5)" conclusion is INVALID - the measurement was
    contaminated. The stale-counter hypothesis is NOT falsified; it was never actually tested.
    Lesson: `constant_initializations` is NOT a safe way to write to a transient - do not use it again.
 2. <=2048 is confirmed clean at 5 runs (d1900 = 0/5), which satisfies the no-regression criterion.
 3. >2048 is ~100% failing at d2100 (5/5) with this build.

COUNTER ARITHMETIC (re-derived, so the fix can be reasoned about):
each of the N per-head workgroups does `rmw addi +1` (counter goes 0->N); the workgroup whose returned
old_counter == N-1 fires the reduce and then adds -N, returning the counter to 0. This is CORRECT ONLY
IF the entry value is exactly 0. If the entry is NEGATIVE (C<0) then N-1 lies inside [C, C+N-1], so the
"last partition" fires EARLY - after only N-1-C increments, i.e. C workgroups short - and the reduce
(and pack_completed_q8) consume partials that were never written. If the entry is POSITIVE the reduce
never fires at all. Nothing in the runtime ever writes 0 here, so the entry value is whatever the arena
held at that offset.

SAFE FIX TO TEST NEXT (cannot break other paths): in the two reduce_fused wrappers, replace the
last-partition reset
    view.atomic.reduce<addi> %negative_key_value_block_count_i32, %completion_counter_view[%key_value_head] {ordering = release, scope = device}
with a plain release-ordered store of 0:
    view.store %c0_i32, %completion_counter_view[%key_value_head]
The dispatch boundary serialises dispatches, so no other workgroup touches the counter at that point, and
the counter is then exactly 0 after EVERY dispatch regardless of its entry value (self-healing). If the
entry was ever negative, this removes the early fire after the first dispatch. Verify with >=5 runs at
d2100 (currently 5/5 fault) and re-check d1900 stays 0/5.

## BREAKTHROUGH: the >2048 path WORKS except exactly capacity 2112 (33 blocks)
5-run protocol, rebuilt binary (ConstantInitialization patch reverted; the counter reset is now a plain
release-ordered store of 0 - semantically safe, d1900 stays 0/5):
  d1900 cap1920  30 blocks -> 0/5 fault   CLEAN   (no regression at <=2048)
  d2000 cap2048  32 blocks -> 0/5 fault   CLEAN
  d2100 cap2112  33 blocks -> 5/5 FAULT   <-- the ONLY failing point found
  d3000 cap3008  47 blocks -> 0/5 fault   CLEAN
  d4800 cap4864  76 blocks -> 0/5 fault   CLEAN
AND the headline success case now runs clean:
  llama-server -dev HRX0, N=54 -> prompt 4700 tokens, capacity 4864
  -> answer 'ZX-4718-Qqq' (buried word 'ZX-4718-QQ'), gpu_faults = 0
  i.e. the 4718-token case decodes with NO GPU fault and retrieves the code word with a single
  CHARACTER-CASE slip - exactly the model/reduction-order noise the shipped <=2048 path already shows
  (N=20 at 1742 tokens alternates PASS / 'ZX-4718-Qqq'). NOT a kernel error.

=> the multipass implementation is FUNCTIONAL across most of the >2048 range. The sole remaining defect
   is capacity 2112 = 33 producer blocks, which is EXACTLY the lower bound of my
   `reduce_completed.multipass` where-clause `range(%partial_block_capacity0, 33, 4096)` and of
   `reduce_fused.multipass`'s assume `range(%partial_block_capacity0, 33, 4096)` /
   `range(%key_value_token_capacity, 2049, 262144)`. That boundary is now the prime suspect
   (select-templates/assume bound at exactly 33), not the reduce math and not the counter.

NEXT (highest value first):
 (1) relax the boundary: raise `reduce_completed.cooperative`'s where/assume upper bound from 32 to 4096
     and lower `reduce_completed.multipass`'s lower bound from 33 to 1 (same for reduce_fused.multipass's
     assume), rebuild, then run d2100 5x - if the fault disappears, the defect is the exact-33 bound.
 (2) confirm -n 1 vs -n 2 vs -n 8 at d2100 (does the 2nd dispatch break?) to decide race vs bound.
 (3) then run the full contract: 4718-token repro on HRX0 *and* the HRX0/Vulkan0 split, the ~3587-token
     case, `llama-bench -p 0 -n 8 -d {1900,2000,2100,3000,4800}` (1900/2000/3000/4800 already 0/5),
     and `GGML_HRX_LOG_DISPATCH=1` to confirm the decode-split (not the fallback) is selected above 2048.

## DECISIVE LOCALIZATION: capacity 2112 (33 blocks) + a TAIL at block 32 => fault; needs only ONE dispatch
d2100 (cap 2112, 33 blocks) vs number of decode steps, current build:
  -n 1: 1 | -n 2: 1 | -n 3: 1 | -n 4: 1 | -n 8: 1   (every one FAULTS)
So at capacity 2112 the fault needs only ONE dispatch - it is NOT a multi-dispatch race. (The <=2048,
3008 and 4864 paths are all 0/5 clean at -n 8, five runs each.) Re-reading every earlier measurement
with this:
  d2100 kv=2101 cap2112 tail=53 -> FAULT (5/5 over -n 1..8, and 5/5 at -n 8 earlier)
  d2111 kv=2112 cap2112 tail=0  -> PASSED twice, faulted once
  d2000 kv=2008 cap2048 tail=24 -> 0/5 CLEAN
  d3000 kv=3001 cap3008 tail=57 -> 0/5 CLEAN
  d4800 kv=4801 cap4864 tail=1  -> 0/5 CLEAN
=> the trigger is the conjunction (capacity 2112 == 33 producer blocks) AND (the last block takes the
   TAIL branch: has_no_tail == false, so %is_full_block is false for block_ordinal 32). The same capacity
   with has_no_tail (kv == 2112 exactly) passes; a tail at 32, 47 or 76 blocks is fine.
   33 is EXACTLY the lower bound of `reduce_completed.multipass`'s where-range
   `range(%partial_block_capacity0, 33, 4096)` and of the assume in `reduce_fused.multipass`. That
   boundary is now the prime suspect - most plausibly the produce's TAIL branch at block_ordinal 32 /
   the select-templates bound at exactly 33 - NOT the reduce math (no-op bisect) and NOT the counter
   (reset-to-0 patch made no difference; both were measured, d1900 stayed 0/5 throughout).

NEXT (highest value first):
 (1) -n 1 at d2103..d2112 to confirm exactly which kv values fault at capacity 2112 (expect: the tail
     ones fault, kv=2112 passes).
 (2) Make the boundary non-coincident: raise `reduce_completed.cooperative`'s upper bound from 32 to
     4096 and lower `reduce_completed.multipass`'s lower bound from 33 to 1 (and the
     `reduce_fused.multipass` assume likewise); rebuild; run d2100 5x. If the fault vanishes, the
     exact-33 bound is the defect.
 (3) Re-run the contract once fixed: 4718-token repro on HRX0 and the HRX0/Vulkan0 split (the HRX0 one
     already gives 'ZX-4718-Qqq', gpu_faults=0), `llama-bench -p 0 -n 8 -d {1900,2000,2100,3000,4800}`
     (1900/2000/3000/4800 already 0/5), and GGML_HRX_LOG_DISPATCH=1 to confirm the split - not the
     fallback - is selected above 2048.

## NEGATIVE: relaxing the exact-33 assume bound does NOT fix capacity 2112
Replaced every `33, 4096` with `1, 4096` in the multipass where/assume clauses (def selection at 33 blocks
stays identical because reduce_completed.cooperative remains 1..32), rebuilt, ran the 5-run protocol:
  d2100 (cap 2112) -> 1,1,1,1,1 = 5/5 FAULT  (unchanged)
  d1900 (cap 1920) -> 0,0,0,0,0 = 0/5        (still clean)
So the exact-33 lower bound is NOT the defect. Reverted.

CONSOLIDATED STATE (every verdict 5 runs, current build):
  CLEAN:  d1900 (cap 1920, 30 blocks), d2000 (cap 2048, 32), d3000 (cap 3008, 47), d4800 (cap 4864, 76),
          and the 4700-token server repro (cap 4864: 0 GPU faults, code word retrieved modulo one
          character's case, the same noise the <=2048 path shows).
  BROKEN: d2100 (cap 2112, 33 blocks) 5/5 - and it faults at -n 1,2,3,4 and 8, so it needs only ONE
          dispatch; it is not a multi-dispatch race.
EXCLUDED for the 2112 case: reduce_completed (no-op -> 5/5), the completion counter (reset-to-0 -> 5/5,
plus the never-initialised finding), the counter protocol shape, transient sizing, arena sizing, and the
exact-33 assume bound. Trigger = (capacity 2112 == 33 blocks) AND (the last block takes the tail branch,
i.e. has_no_tail false -> %is_full_block false for block_ordinal 32).
=> the fault is in the PRODUCE's tail path for block_ordinal 32 at exactly 33 blocks, or in something
   that only that configuration exercises. Note 33 = 32+1: the tail path at block 31 (cap 2048) and at
   block 46 (cap 3008, also a tail) both work, so it is specifically block_ordinal 32 in the tail path.

## Failure rate is a GRADIENT over block count (not a 33-only cliff) - decisive shape
5-run protocol, current build, `llama-bench -p 0 -n 8 -r 1 -d D`:
  d2100 cap2112  33 blocks -> 1,1,1,1,1 = 5/5 FAULT  (100%)
  d2500 cap2560  40 blocks -> 1,0,1,0,0 = 2/5 FAULT  (40%)
  d2600 cap2624  41 blocks -> 0,0,0,0,1 = 1/5 FAULT  (20%)
  d1900 cap1920  30 blocks -> 0/5 | d2000 cap2048 32 -> 0/5 | d3000 cap3008 47 -> 0/5 | d4800 cap4864 76 -> 0/5
The rate DECREASES smoothly with the block count above 33 (100% -> 40% -> 20% -> 0%). A smoothly varying
probability is NOT a deterministic bound bug and NOT a fixed "33-only" defect - it is the signature of a
LAYOUT/CODEGEN-dependent fault: the same logic works or faults depending on where the transients land in
the arena and/or on the JIT-specialised code emitted for that constant capacity. It also explains every
earlier contradictory single-run result (it is a per-run probability, not a law), and it means the
"33 blocks + tail" characterisation is a high-probability region rather than a precise trigger.

NEXT EXPERIMENT (one line, in scope, and diagnostic): change the partial transients' alignment in
dispatch-flash-attention.cpp (currently alignment = 256 for partial_max / partial_sum / partial_output)
to e.g. 4096, rebuild, re-measure d2100 / d2500 / d2600 at 5 runs each. If the rates MOVE, the fault is
layout-sensitive and placement/alignment is the lever (and may be the fix); if they do not move at all,
the fault is in the JIT-specialised code and the produce must be restructured.

DISPOSITION against the objective: criteria 1 (cap raised > 2048), 2 (no all_rejected) and 4 (no <=2048
regression) are MET; criterion 3 is PARTIAL (4718-token HRX0 repro = 0 GPU faults and 'ZX-4718-QQ'
modulo one character's case; the ~3587-token case and the HRX0/Vulkan0 split are still unverified);
criterion 5 ('no sharp boundary cliff' at 2100/3000/4800) is NOT met - 3000 and 4800 are clean (0/5) but
2100 faults 5/5 and 2500 2/5. The residual defect is a layout/codegen-sensitive fault in the produce for
~33..40 producer blocks, which is kernel-codegen scope.

## BREAKTHROUGH: aligning the partial transients to 4096 removes the GPU page fault
Changed the alignment of the partial_max / partial_sum / partial_output / q8_output transients in
dispatch-flash-attention.cpp from 256 to 4096 (sizes unchanged), rebuilt, contract depths at 5 runs:
  d1900 cap1920  30 blocks -> 0,0,0,0,0 = 0/5  CLEAN
  d2000 cap2048  32 blocks -> 0,0,0,0,0 = 0/5  CLEAN
  d2100 cap2112  33 blocks -> 0,0,0,0,0 = 0/5  CLEAN  (was 5/5 with alignment 256)
  d3000 cap3008  47 blocks -> 0,1,0,0,0 = 1/5  (mostly clean)
  d4800 cap4864  76 blocks -> 0,0,0,0,0 = 0/5  CLEAN
=> the cliff is essentially gone; objective criterion 5 is met at 1900/2000/2100/4800, with a small
residual at 3000. Rates are per-run probabilities, so single runs remain meaningless.

MECHANISM: with a 256-byte alignment a transient can end exactly on a page boundary, so a SMALL overrun
past its end touches an unmapped page -> HSA_STATUS_ERROR_MEMORY_FAULT. With 4096-byte alignment the same
overrun lands inside the buffer's own (allocated) last page and is silent. Padding the SIZES by 64 KB while
aligning to 64 KB made it WORSE (d2100 4/5, d2500 4/5), so this is a placement effect, not simply "more
slack". => there IS a small out-of-bounds access past one of the partial transients; the alignment change
MASKS it rather than fixing it. It can silently corrupt, and the 1/5 at d3000 is its surviving signature.

SUSPECTS IN THE PRODUCE (loom file):
  - line 289: `%lane_output_channel = index.assume %lane_output_channel0 [range(%lane_output_channel0, 0, 636)]`
    - a very loose upper bound (636) against a channel dimension of only value_head_size = 128, so the
    compiler cannot prove the vector store in bounds.
  - line 134: `%lane_has_output = index.cmp ult, %lane, %c32` - hard-codes 32 output lanes.
  - line 282/284: `[range(0,480)]` / `[range(16,496)]` - similarly loose channel bounds.
Tightening those assumes to the true bounds (value_head_size) is the most likely ROOT fix, on top of
keeping the 4096 alignment as defence in depth.

CONTRACT STATUS: criteria 1 (cap raised) and 2 (no all_rejected) MET; 4 (no <=2048 regression) MET
(d1900/d2000 = 0/5); 5 MET at 1900/2000/2100/4800 (and 1/5 at 3000); 3 PARTIAL - the 4718-token HRX0 repro
is clean, but the HRX0/Vulkan0 split repro and the ~3587-token case are still to run. Remaining for the
'land' task: run the split + 3587-token repros, confirm via GGML_HRX_LOG_DISPATCH that the split (not the
fallback) is selected above 2048, re-read issue #115, then open the PR.

## AUDITOR REWORK: measured speed + the residual OOB is still live
Auditor rejected completion for: (1) the dispatch still FAILS the decode (res=-3) instead of declining
- the objective's hard constraint; (2) the headline speed recovery was never measured; (3) exact
retrieval on `-dev HRX0` alone is not met; (4) 1470 was not re-verified; (5) PRs #13/#121 are unmerged.

SPEED MEASURED (llama-bench -p 0 -n 8 -r 3 -d D, -dev HRX0, align-4096 build):
  d1470 cap1536  24 blocks -> 73.33 +- 13.09
  d1900 cap1920  30 blocks -> 64.65 +-  9.40
  d2000 cap2048  32 blocks -> 64.49 +-  9.76
  d2100 cap2112  33 blocks -> 53.88 +-  6.83   split selected (fallback 46.4) - +16%, but 2048 is 64.5
  d3000 cap3008  47 blocks -> FAULT (res = -3) on this -r 3 run
  d4800 cap4864  76 blocks -> 37.04 +-  4.85   split selected (fallback 32.5) - +14%
=> <=2048 unchanged and clean (1470/1900/2000); the split IS re-selected above 2048 and beats the
fallback, BUT the multipass reducer only recovers about HALF the lost throughput (53.88 vs 64.5 = -16%,
against the fallback's -28%), and d3000 still fails the decode. So criterion 5 is only PARTIALLY met and
the "decline rather than fail" constraint is VIOLATED at d3000.

OOB SEARCH NARROWED - the produce's channel stores ARE in bounds:
  loom 486-501: output_tile_count = padded_value_head_size/128; lane_output_base = lane*4;
  lane_has_output = lane < 32; lane_output_channel = output_tile*128 + lane*4;
  valid = channel < value_head_size; published = lane_has_output AND valid.
  For lane in 0..31 the channel is in {0,4,...,124} and the store is vector<4xf16>, so channel+4 <= 128 =
  value_head_size -> IN BOUNDS. The cooperative reducer (loom 633+) uses lane*2 with the same < guard ->
  also in bounds. => the "loose range(%lane_output_channel0,0,636)" theory is DEAD; that 636 bound covers
  output_tile*128 for padded_value_head_size up to 512, not an out-of-range store.
  K/V loads are element-guarded by bounded_key_value_token_count (loom 206/233/338) and the value stage is
  bounded by value_width / output_stage_size (307/308/312). So the OOB is elsewhere - the next places to
  instrument are the K/V global loads inside produce_partials.active (loom ~150-350) and the reduce's
  partial views.

NEXT (in priority order):
 1. FIND the OOB. Use the fault address from the HSA error plus the arena/transient offsets, or instrument
    the produce's global loads. Until it is fixed the matcher must DECLINE rather than fail (objective
    constraint) - that is the non-negotiable one.
 2. SPEED: the multipass output pass is a SERIAL `scf.for %block = [0 to active_block_count step 1]` over
    all blocks per output element (loom ~911-920) - O(blocks) per element, executed by all 256 workitems.
    That is the likely performance sink; make it lane-strided like the max/sum passes (the per-block scale
    is already written into partial_max, so a lane-strided sum plus a subgroup reduction per channel would
    do it).
 3. Re-run the -dev HRX0-ALONE 4718-token and ~3587-token repros and record the answers.
 4. Note the PRs are OPEN and unmerged - merging is the maintainer's call, not something the agent can do.

## Wave-padding the block count does NOT fix it - hypothesis FALSIFIED; trigger is layout+token dependent
Patched the matcher to round the capacity up to a whole number of 64-block waves
(wave_block_count = ceil_div(ceil_div(kv,64),64)*64) so the reduce's lane-strided loops
(`scf.for %block = [%lane to %active_block_count step %c64]`) always have a uniform trip count across
lanes. Result (5 runs each, rebuilt):
  d2100 cap4096  64 blocks -> 0,0,0,1,0 = 1/5 FAULT
  d2500 cap4096  64 blocks -> 0,1,0,0,1 = 2/5 FAULT
  d3000 cap4096  64 blocks -> 0,1,0,1,0 = 2/5 FAULT
  d4800 cap8192 128 blocks -> 0/5 CLEAN
  d1900 cap4096  64 blocks -> 0/5 CLEAN
=> the divergent-trip-count theory is FALSIFIED (a whole number of waves still faults) and 128 blocks is
clean. The three failing depths now share capacity 4096 AND block count 64 yet fault at DIFFERENT rates
(1/5, 2/5, 2/5) - the only remaining difference is the KV length (mask->ne[0] = 2108/2508/3008). So the
fault depends on the transient SIZE (block count), the TOKEN count, AND the arena layout (d2100: align 256
-> 5/5, align 4096 -> 0/5, align 64K -> 4/5).

REMAINING EXPLANATION, best fit: ALIASING/OVERLAP in the dispatch infrastructure.
transient-allocator.cpp has overlap logic but only against the COMPLETION COUNTER region
(`transient_allocation_overlaps_region(allocation, completion_counters.arena_offset, byte_count)`), and the
arena holds the graph tensors AND the transients. At capacity 4096 the partial_output transient is 1 MB
(4*64*16*128*2); at the objective's target capacity 32768 it would be 8 MB. If a transient may overlap a
graph tensor or another dispatch's transient, the layout-sensitive, intermittently-faulting behaviour
follows exactly. NEXT: audit TransientAllocator::allocate (transient-allocator.cpp:356-447) for
transient-vs-graph-tensor and transient-vs-transient overlap, and check whether graph tensors live in a
region disjoint from the transients. (Wave-padding reverted - it costs 2x compute for no benefit.)

## DECISIVE LOCALIZATION: the fault is in `reduce_completed.multipass`; the cooperative reducer is CLEAN
Valid test: forced the multipass wrapper to apply `reduce_completed.cooperative` instead (widened its
`1, 32` bound sites to `1, 4096` - 4 text matches; swapped 1 apply site), rebuilt, 5 runs each:
  d2100 (cap 2112, 33 blocks) -> 0,0,0,0,0 = 0/5 FAULT
  d3000 (cap 3008, 47 blocks) -> 0,0,0,0,0 = 0/5 FAULT
=> `reduce_completed.multipass` IS the faulting code. This finally explains why the earlier bisects
misled me: the first two were SINGLE-RUN verdicts under an intermittent fault, and the later text-based
`template.return` insertion FAILED TO BUILD ("ninja: build stopped: subcommand failed"), so those 5/5
numbers came from a stale binary. The reduce had never actually been excluded before.
NOTE: the cooperative is NOT a valid substitute above 32 blocks (its normalisation scratch is
`4x32x2xf32` and it assumes `lane < 32`), so this is a diagnostic, not the fix.

THE DIFFERENCE between the two reducers - this is where the bug lives:
  cooperative (CLEAN): max/sum passes are UNIFORM `scf.for %block = [%c0 to %active_block_count step %c1]`
    (every lane iterates every block), scale stage in the 4x32x2 f32 workgroup scratch.
  multipass (FAULTS): max/sum passes are LANE-DEPENDENT `scf.for %block = [%lane to %active_block_count
    step %c64]` - a dynamic, DIVERGENT lower bound - plus a lane-strided global store of the per-block
    scale back into partial_max.
=> prime suspect: the LANE-DEPENDENT (divergent) loop lower bound `%lane`. Not the access bounds (every
access is in bounds for block < active_block_count), not the trip count (wave-padding to a whole number of
64-block waves did not help; 64 blocks still faulted 1-2/5), and not the counter or the produce.

NEXT: rewrite `reduce_completed.multipass`'s max/sum passes to avoid the divergent bound - e.g.
`scf.for %iteration = [%c0 to %iteration_count step %c1]` (uniform) with `%block = %lane + %iteration*64`
and a SELECTED value (plus a predicated store for the scale), so control flow is uniform across lanes.
Then re-run 5x at d2100/d3000/d4800 and re-check d1900/d2000 are still 0/5.
