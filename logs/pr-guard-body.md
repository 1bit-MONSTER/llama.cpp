## What

`host_weight_exceeds_residency()` decides **placement** in `device_supports_op`, and it was silent: a budget that moved
nodes off HRX produced exactly the same log as one that changed nothing. The only witness was the
`GGML_HRX_REPORT` placement census. This PR makes the decision greppable.

One `GGML_LOG_WARN` per **distinct** (weight, op) pair, eight pairs, then one line saying the rest are suppressed.
De-duplication is the point, not a detail: `device_supports_op` is queried repeatedly per node (reserve and compute
passes), so a plain cap spent its whole budget on the same decision - measured 8 lines for **2** nodes at a 128 MiB
budget, and at 8 MiB (109 declined tensors) the GEMMs moving would never have appeared at all.

```
W hrx placement: MUL_MAT is not run on HRX because its weight 'token_embd.weight' (304 MiB) exceeds the 128 MiB
  residency budget (GGML_HRX_HOST_WEIGHT_LIMIT_MB), so the CPU serves this node
```

The predicate stays pure; no behaviour change, no knob, no new file.

## Why now

Engine #674's row was re-scoped to "no placement change may reintroduce the weightless-op/weight-op split". That
invariant is only testable if the placement decision is visible, so this is the guard half; the reachability evidence
(why the row's headline is falsified at this pin) is on
<https://github.com/1bit-MONSTER/engine/issues/674#issuecomment-6586220294>.

## Measured on the pin (`d8bd0ff6052`), `Qwen3-4B-Q4_K_M`, `-dev HRX0 -fit off -ngl 0 -c 256 --temp 0`, #715 trio, one `gpu-hold` ticket per arm set

| arm | guard lines | `placement_observer` census | output |
|---|---|---|---|
| budget **unset** | **0** | 288 off HRX (`MUL_MAT` 72, `RMS_NORM` 144, `CONT` 36, `SOFT_MAX` 36) | `Okay, the user is asking` |
| budget **128 MiB** | **2** (`GET_ROWS`, `MUL_MAT` on `token_embd`) | 290 off HRX (same + `GET_ROWS` 1, `MUL_MAT` 73) | `Error: Compute error.` |
| budget **8 MiB** | **8 + suppression notice** (`GET_ROWS`, `MUL_MAT` on `token_embd`, `blk.0-2.ffn_down`/`ffn_gate`/`ffn_up` …) | **469** off HRX (`MUL_MAT` 181, `ADD` 35, `GLU` 36, …) | `Error: Compute error.` |

Unbudgeted is byte-for-byte the old behaviour (same census, same answer, `+4055M` GTT), and every declining arm still
refuses - the instrument does not change what runs.

## Not in this PR

No fix, no placement change, no merge by me. Engine #674's recommendation is **close as falsified**; this lands the
instrument either way, so a future placement change cannot reshape the graph silently.
