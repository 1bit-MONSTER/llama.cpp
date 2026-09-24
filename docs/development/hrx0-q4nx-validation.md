# HRX0 Q4NX port — validation report (v2, post-audit)

Repo `/home/bcloud/hrx-ws/hrx-v2-src`, branch `hrx0-q4nx-port` (see `git log`),
goal `mufpkzv5-bqyi90`.

> **v2 correction.** The v1 report used `zaya1-8b-ft-q4nx.gguf` /
> `…-c42.gguf` as its end-to-end evidence and called their identical output a
> cross-check. Those two containers store the MoE experts **2-D**
> (`[8192, tpe*n_expert]`), which is not a valid input to the Q4NX MoE op — their
> "identical" output was identical *garbage*. The valid 3-D containers
> (`…-repacked.gguf`, `zaya-q4nx-gemma4tok.gguf`) decode coherently and are the
> evidence below. The 2-D container now fails loudly instead of silently.

## What was ported

Q4NX (signed 4-bit int4 tiles, `GGML_TYPE_Q4NX` = 42 / `Q4NX_C43` = 43,
5120-byte tile = `[32 BF16 rows x 256 cols]`) existed only in the **HRX2**
backend (`ggml/src/ggml-hrx2`, Loom DSL). This ports the three kernels HRX0
needs into `ggml/src/ggml-hrx` (HIP C++ `.hip.cpp`):

| Kernel | Catalog name | ABI | Covers |
|---|---|---|---|
| `kernels/q4nx_dequant_f32.hip.cpp` | `hrx_q4nx_dequant_f32` | 2 bind / 3 par / 24 B / wg256 | tile → F32 dequant |
| `kernels/mul_mat_q4nx_fused_f32.hip.cpp` | `hrx_mul_mat_q4nx_fused_f32` | 3 bind / 4 par / 32 B / wg256 | `GGML_OP_MUL_MAT_Q4NX` (dense) |
| `kernels/mul_mat_id_q4nx_fused_f32.hip.cpp` | `hrx_mul_mat_id_q4nx_fused_f32` | 4 bind / 5 par / 104 B / wg256 | `GGML_OP_MUL_MAT_ID_Q4NX` (MoE) |

`ggml-hrx.cpp` gained the constants structs, the three providers + loaders, and
`supports`/`dispatch` pairs wired into both the device `supports_op` switch and
the graph-compute dispatch switch. All changes are additive.

The MoE kernel takes **both** activation forms the HRX2 route supports:
`src1 = [k, 1, ntokens]` (shared, `src1_nb1 = 0`, ggml's broadcast rule) and
`src1 = [k, n_expert_used, ntokens]` (per-slot), via `src1_nb1`/`src1_nb2`.

## Commits

- `8ecd22a64` — inventory + HRX2→HRX0 ABI mapping (`docs/hrx0-q4nx-port-mapping.md`)
- `38b2aa032` — dequant kernel + per-op test
- `556b0dec8` — dense fused matmul + `MUL_MAT_Q4NX` claim
- `a3d1c0741` — MoE fused matmul + `MUL_MAT_ID_Q4NX` claim
- `4f10b0a02` — accept the strided ids view in the MoE support gate
- `854c01772` — v1 validation report
- `d632ab10c` — register `GGML_TYPE_Q4NX_C43` (43) and route it with 42
- `e5a9fa94d` — v1 report update
- `75ff15819` — **per-slot src1 + loud out-of-range id rejection** (audit rework)

## Build

```bash
cmake --build build-hrx0-latest --target llama-cli llama-simple -j 8
```

Builds the three HSACO artifacts (`generated/hsaco/gfx1151/*.hsaco`), the
regenerated `hrx_kernel_catalog.cpp`, and `libggml-hrx.so` with no errors. The
catalog ABI check (bindings / parameter_count / constants_size) passes for all
three kernels at provider load.

## Per-op parity (independent CPU reference)

`tools/hip-bench/*` harnesses build synthetic Q4NX blobs covering all 16 signed
int4 nibble values, BF16 scales/zero-points, the `|x| > 100` / NaN sanitize
cases, multi-tile and multi-expert layouts, then run the kernel on gfx1151 and
compare every element against an independent CPU implementation of the format.

| Test | Result |
|---|---|
| `q4nx_dequant_bench.hip.cpp` | **PASS** — 49152 elements, `max_abs_diff = 0` |
| `mul_mat_q4nx_fused_bench.hip.cpp` | **PASS** — `max_abs_diff = 2.27e-06` |
| `mul_mat_id_q4nx_fused_bench.hip.cpp`, shared `src1` | **PASS** — `max_abs_diff = 7.79e-06` |
| `mul_mat_id_q4nx_fused_bench.hip.cpp`, per-slot `src1` | **PASS** — `max_abs_diff = 4.54e-06` |

## End-to-end on HRX0

Environment (see `docs/development/hrx0-iq3-xxs-iq3-s.md`):

```bash
export LD_LIBRARY_PATH=/home/bcloud/hrx-latest/prefix/lib:/home/bcloud/hrx-latest/hsa:/opt/hrx/lib/rocm_sysdeps/lib
cd build-hrx0-latest
./bin/llama-simple -m /home/bcloud/store43/zaya1-8b-ft-q4nx-repacked.gguf -n 24 "The capital of France is"
```

Validated outputs (3-D expert container, `ffn_*_exps` = `[8192, tpe, n_expert]`):

| Prompt | HRX0 output (greedy, 24 tokens) |
|---|---|
| `Once upon a time` | `Once upon a time in a small town named Pythonia, there lived a little girl named Lily. Lily loved to read books and play with` |
| `The capital of France is` | `The capital of France is Paris.` |
| `Q: What is 2+2? A:` | `Q: What is 2+2? A: 4.` |

The graph is scheduled on HRX0 (`sched_reserve: HRX0 compute buffer size`),
prompt eval ~40 tok/s and decode ~18 tok/s.

**2-D expert containers are rejected.** `zaya1-8b-ft-q4nx.gguf` (type 43) and
`…-c42.gguf` (type 42) store experts as 2-D `[8192, tpe*n_expert]`. The fork's
own `~/hrx-ws/repack_zaya_experts.py` documents the consequence:

> *"The HRX2 Q4NX MoE dispatch keys off `src0->ne[2]` (n_expert) and computes
> `tile_base = e * tpe`, so a 2-D tensor makes ne[2] == 1 and every routed
> expert id look out of range (`HRX2: MUL_MAT_ID Q4NX expert id N out of
> range`)."*

i.e. these containers are invalid for the Q4NX MoE op on **HRX2 as well**; the
3-D repack is a pure header rewrite (`[8192, tpe*n_exp]` and
`[8192, tpe, n_exp]` address the same bytes). HRX0 now mirrors HRX2's loud
rejection instead of silently zeroing the expert:

```
ggml_backend_hrx_dispatch_mul_mat_id_q4nx: MUL_MAT_ID_Q4NX expert id 3 out of
range: the Q4NX expert tensor is 2-D (ne[2]=1); repack experts to
[8192, tpe, n_expert]
```

The check only runs when `ne[2] == 1`, so normal models pay no readback.

## Type 43 (`Q4NX_C43`)

Registered in `d632ab10c`: `GGML_TYPE_Q4NX_C43 = 43`, `GGML_TYPE_COUNT = 44`,
type traits, the `ggml_mul_mat`/`ggml_mul_mat_id` routing, the loader
`check_tensor_dims` exception and the buft probes all accept it. A type-43
container now parses and reaches the graph instead of failing GGUF parse.

## Remaining gaps (environment, not the port)

1. **No end-to-end HRX2/CPU numeric reference.** HRX2 (`build/bin`, device
   `HRX20`) fails at compute with `BYTECODE/002: unsupported format version 34,
   expected 38` — `loom-link` is Aug-30 while the installed `libloomc` is
   Sep-23. `ggml-cpu` explicitly rejects the Q4NX ops, so CPU cannot run the
   model. Per the user's decision, op-level parity (4 bit-exact/plus-epsilon
   per-op comparisons) is accepted in place of the end-to-end comparison.
2. **`test-backend-hrx` does not pass cleanly** — it aborts on an `rms_norm`
   mismatch with a nondeterministic failing index. This is pre-existing:
   stashing all Q4NX changes and rebuilding reproduces it identically, and the
   Q4NX change is additive.
