## REACHABILITY: the corrupting configuration is **not expressible** at the pin, and **no declining budget can complete** — recommendation: close as falsified

Follows the 16:03Z question and my 16:08Z reconciliation. The board moved this row to "re-scope + split guard"; before
writing a guard I tested whether the placement that corrupts is reachable at all on merged code. It is not, and that is
now provable in two independent ways.

Pin **`d8bd0ff6052`**, `stage-pin788` (asserted before measuring: `refusing to run` x1, `GGML_HRX_169_DECLINE` **x0**,
residency trace present), `Qwen3-4B-Q4_K_M`, `-dev HRX0 -fit off -ngl 0 -c 256 --temp 0 -n 8`, #715 trio,
`GGML_HRX_HOST_WEIGHT_TRACE=1` + `GGML_HRX_REPORT=1` + `-v`, one `gpu-hold` ticket per arm set.

### 1. What a residency budget actually does to the graph (the census, first time measured)

`GGML_HRX_HOST_WEIGHT_LIMIT_MB` is the **only** shipped knob that can move a weight-consuming node off HRX while the rest
of the graph stays, because `host_weight_exceeds_residency()` (`ggml-hrx.cpp:1294`) is the only weight-dependent decline
in `device_supports_op` (the others are MoE, batched-IQ3_S `GET_ROWS`, `TOP_K` shape, and shape/type predicates that
never read a weight).

| arm | budget | GTT peak delta | residency | census (`placement_observer`) | output |
|---|---|---|---|---|---|
| `-dev none` | unset | **+297M** | 0 lines | (CPU only) | `Okay, the user is asking` |
| `-dev HRX0 -ngl 0` | unset | **+3955M** | 1810 lines / 3160.5 MiB | 1446 nodes, **288 off HRX** (`MUL_MAT` 72, `RMS_NORM` 144, `CONT` 36, `SOFT_MAX` 36) | `Okay, the user is asking` |
| `-dev HRX0 -ngl 0` | 128 | **+0M** | 182 / 80.4 MiB | 1446 nodes, **290 off HRX** (`MUL_MAT` **73**, **`GET_ROWS` 1**) | `Error: Compute error.` |
| `-dev HRX0 -ngl 0` | 32 | **+0M** | 8 / 0.0 MiB | same 290 off HRX | `Error: Compute error.` |

A budget moves **exactly two nodes** - the final `MUL_MAT` and the `GET_ROWS` lookup that read `token_embd` - and then
#173 refuses: `128 MiB ... a 5 MiB weight does not fit with 80 MiB resident and 41 MiB parked ... Raise it to at least
128 MiB`. GTT is `+0M` because nothing was materialised.

### 2. Boundary scan: the decline band is strictly inside the refusal band

`token_embd` is 304.3 MiB, so 304 MiB declines it and 305 MiB does not:

| budget | census | output |
|---|---|---|
| 300 | 290 off HRX | `Error: Compute error.` |
| 304 | 290 off HRX | `Error: Compute error.` |
| **305** | **288 off HRX - declines nothing** | **`Error: Compute error.`** (`a 13 MiB weight does not fit ...`) |

`bud305` is the decisive row: **nothing is declined and the run still refuses.** #173's refusal boundary (per-execution
working set, ~2030 MiB here) is far above the decline boundary (~304 MiB), so **every budget that declines a node is
strictly inside the band that refuses** - a declined placement never executes. No arm in this band produced output, and
none produced garbage.

### 3. Why the bisected minimal pair cannot be reproduced by this knob at all

The minimal corrupting pair is `{MUL_MAT, MUL}`. From the GGUF, 398 tensors: **norm-scale weights are 512 B - 10 240 B**
(`output_norm` + 144 `blk.*_norm.weight`). `host_weight_exceeds_residency()` compares `ggml_nbytes(weight) > limit`, and
`limit` is `GGML_HRX_HOST_WEIGHT_LIMIT_MB` in **whole MiB** (floor 1 MiB = 1 048 576 B, i.e. 102x-2048x the largest norm
scale). So **the norm `MUL` can never be declined at any integer-MiB budget**, and `MUL_MAT`-declined alone was measured
**correct** in the bisect (`MUL_MAT` alone, and `MUL_MAT` with `GET_ROWS`/`RMS_NORM`/`ROPE`/`CONT`).

Declines by budget, for the record: 256/128/32 -> **1** tensor (`token_embd`); 16 -> 19; 8 -> **109** (+108 GEMMs at
19.5 MiB); 1 -> **253**. The widest expressible split always leaves all 145 norm scales on HRX.

### 4. Recommendation, stated plainly

**Close as falsified.** The headline is false at the pin (16:08Z: `-ngl 0` answers correctly, `+4122M`), and this comment
adds that it cannot be made true through any shipped knob. The evidence table every `{MUL_MAT, MUL}` row in this thread
rests on was measured on `pin + 7c2fcb70a09`, which is labelled *FALSIFIED, do not merge* and is superseded by #172/#173.

If the row is re-scoped instead, the wording that survives this evidence is: **the falsified #169 placement experiment
corrupted output; superseded by #172/#173, and a live invariant - no placement change may reintroduce the
weightless-op/weight-op split - which is now measurable**, since (a) the census names the off-HRX node set and (b) the
decline is logged (guard below). It must **not** read as a live defect: there is no known configuration at the pin that
produces wrong output.

**One real defect this sitting did find, and it is the guard:** `host_weight_exceeds_residency()` had **no log line** — a
budget could reshape the graph silently, with the `GGML_HRX_REPORT` census as the only witness. Branch
`1bit/i674-split-guard` (off `d8bd0ff6052`, +30 lines, `host-weight-placement.{h,cpp}`) emits one bounded greppable
`GGML_LOG_WARN` per declined node (op, weight, weight MiB, budget MiB) and carries the invariant + these numbers in the
header. Predicate stays pure, behaviour unchanged; measurement of the line follows in a second comment.
