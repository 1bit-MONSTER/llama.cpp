## The guard, measured — and it changes nothing that runs

Follows the reachability comment above. **PR: <https://github.com/1bit-MONSTER/llama.cpp/pull/174>** (`1bit/i674-split-guard`
@ `8fa0223188a`, base `1bit/hrx-vulkan-patched`, OPEN / non-draft / MERGEABLE / CLEAN, +30 lines in
`host-weight-placement.{h,cpp}`, no knob, no new file, nothing merged by me).

The one real defect the sitting found: `host_weight_exceeds_residency()` decides **placement** inside
`device_supports_op` and was **silent**, so a budget that reshaped the graph looked exactly like one that changed
nothing — the `GGML_HRX_REPORT` census was the only witness. The PR emits one bounded `GGML_LOG_WARN` per distinct
(weight, op) pair and says so when it stops logging.

| arm (pin `d8bd0ff6052`, 4B `-ngl 0`) | guard lines | census | output |
|---|---|---|---|
| budget **unset** | **0** | 288 off HRX | `Okay, the user is asking`, **+4055M** GTT |
| budget **128 MiB** | **2** (`GET_ROWS`, `MUL_MAT` on `token_embd`, 304 > 128 MiB) | 290 off HRX | `Error: Compute error.` |
| budget **8 MiB** | **8 + a suppression notice** (…`blk.0-2.ffn_down` 19 MiB, `ffn_gate`/`ffn_up` 13 MiB…) | **469** off HRX (`MUL_MAT` 181, `ADD` 35, `GLU` 36) | `Error: Compute error.` |

Two things worth taking from the third row, because it is the strongest form of the reachability answer:

1. **The widest split this knob can express is 469 nodes off HRX** — 181 GEMMs, 35 `ADD`s and 36 `GLU`s on the CPU while
   all **144 `RMS_NORM` stay on HRX**. That is the falsified experiment's shape minus the norm-scale `MUL`, which no
   whole-MiB budget can decline (norm scales are 512 B – 10 KiB; the knob's floor is 1 MiB).
2. **It still refuses** (`a 13 MiB weight does not fit ...`) and produces no output and no garbage. So the band that
   declines is inside the band that refuses, at every value I measured: 8, 32, 128, 256, 300, 304 MiB decline and
   refuse; 305 MiB and above decline nothing; unbudgeted answers correctly.

Also recorded because it was my own harness defect, caught before it became evidence: the first version of the guard
capped lines **without de-duplicating**, and the scheduler queries each node repeatedly, so the 128 MiB arm printed 8
lines for 2 nodes and the 8 MiB arm would have hidden all 108 GEMM declines behind duplicates. Same failure mode as
the three earlier ones in this thread — an instrument whose output *looks* like a measurement. The line budget is now
per distinct pair plus an explicit suppression notice.

**Row status:** recommendation unchanged — **close as falsified**, or re-scope to the invariant. Either way this PR is
the instrument that keeps the invariant checkable; it is not a fix and it does not change any behaviour.
