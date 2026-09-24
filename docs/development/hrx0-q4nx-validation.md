# HRX0 Q4NX port — validation report (task-5)

Repo `/home/bcloud/hrx-ws/hrx-v2-src`, branch `hrx0-q4nx-port`.
Goal `mufpkzv5-bqyi90`.

## What was ported

Q4NX (signed 4-bit int4 tiles, `GGML_TYPE_Q4NX` = 42, 5120-byte tile =
`[32 BF16 rows x 256 cols]`) previously existed only in the **HRX2** backend
(`ggml/src/ggml-hrx2`, Loom DSL). This work ports the three kernels HRX0 needs
into `ggml/src/ggml-hrx` (HIP C++ `.hip.cpp`):

| Kernel | Catalog name | ABI | Covers |
|---|---|---|---|
| `kernels/q4nx_dequant_f32.hip.cpp` | `hrx_q4nx_dequant_f32` | 2 bind / 3 par / 24 B / wg256 | tile → F32 dequant |
| `kernels/mul_mat_q4nx_fused_f32.hip.cpp` | `hrx_mul_mat_q4nx_fused_f32` | 3 bind / 4 par / 32 B / wg256 | `GGML_OP_MUL_MAT_Q4NX` (dense) |
| `kernels/mul_mat_id_q4nx_fused_f32.hip.cpp` | `hrx_mul_mat_id_q4nx_fused_f32` | 4 bind / 5 par / 88 B / wg256 | `GGML_OP_MUL_MAT_ID_Q4NX` (MoE) |

`ggml-hrx.cpp` gained the constants structs, the three providers + loaders, and
`supports`/`dispatch` pairs wired into both the device `supports_op` switch and
the graph-compute dispatch switch. All changes are additive.

Commits:

- `8ecd22a64` task-1 — inventory + HRX2→HRX0 ABI mapping (`docs/hrx0-q4nx-port-mapping.md`)
- `38b2aa032` task-2 — dequant kernel + per-op test
- `556b0dec8` task-3 — dense fused matmul + `MUL_MAT_Q4NX` claim
- `a3d1c0741` task-4 — MoE fused matmul + `MUL_MAT_ID_Q4NX` claim
- `4f10b0a02` task-5 — accept non-contiguous ids in the MoE support gate

## Build

```bash
cmake --build build-hrx0-latest --target llama-cli llama-simple -j 8
```

Builds the three HSACO artifacts (`generated/hsaco/gfx1151/*.hsaco`), the
regenerated `hrx_kernel_catalog.cpp`, and `libggml-hrx.so` with no errors.
The catalog ABI check (bindings / parameter_count / constants_size) passes for
all three kernels at provider load.

## Per-op parity (independent CPU reference)

Each kernel has a standalone harness under `tools/hip-bench/` that builds a
synthetic Q4NX blob covering all 16 signed int4 nibble values, BF16
scales/zero-points, the `|x| > 100` / NaN sanitize cases and multi-tile /
multi-expert layouts, runs the kernel on gfx1151 and compares every element
against an independent CPU implementation of the documented format.

| Test | Result |
|---|---|
| `q4nx_dequant_bench.hip.cpp` | **PASS** — 49152 elements, `max_abs_diff = 0` |
| `mul_mat_q4nx_fused_bench.hip.cpp` | **PASS** — `max_abs_diff = 2.3e-06`, `max_rel = 1.7e-06` |
| `mul_mat_id_q4nx_fused_bench.hip.cpp` | **PASS** — `max_abs_diff = 7.8e-06`, `max_rel = 2.0e-06` |

Run:

```bash
export LD_LIBRARY_PATH=/home/bcloud/hrx-latest/prefix/lib:/home/bcloud/hrx-latest/hsa:/opt/hrx/lib/rocm_sysdeps/lib
INC="-isystem /opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel/include"
cd ggml/src/ggml-hrx/tools/hip-bench
/opt/rocm-therock/bin/amdclang++ -x hip --offload-arch=gfx1151 -O2 $INC q4nx_dequant_bench.hip.cpp -L/opt/rocm-therock/lib -lamdhip64 -o /tmp/b1 && /tmp/b1
```

## End-to-end on HRX0

```bash
export LD_LIBRARY_PATH=/home/bcloud/hrx-latest/prefix/lib:/home/bcloud/hrx-latest/hsa:/opt/hrx/lib/rocm_sysdeps/lib
cd build-hrx0-latest
./bin/llama-simple -m /home/bcloud/store43/zaya1-8b-ft-q4nx-c42.gguf -n 16 "The capital of France is"
```

Before the port the same command failed at model load
(`failed to find a compatible buffer type for tensor blk.0.attn_q.weight`:
ggml-cpu rejects Q4NX ops and HRX0 did not claim them) and later
(`no backend supports node op=97 ... MUL_MAT_ID_Q4NX`).

After the port:

```
sched_reserve:       HRX0 compute buffer size =    13.66 MiB
<bos>The capital of France isoverlapsoverlapsoverlaps...
main: decoded 16 tokens in 0.99 s, speed: 16.14 t/s
prompt eval time = 150.21 ms / 6 tokens (39.94 tok/s)
eval time = 836.45 ms / 15 runs (17.93 tok/s)
```

The model loads, the whole graph is scheduled on HRX0, and decoding runs.
Output text is repetitive — this is a raw base-model prompt with no chat
template; it is not by itself a correctness signal.

## Open gaps (not caused by this change)

1. **No end-to-end numeric reference.** The task contract asks for top-1 /
   logit parity against HRX2 or CPU. In this environment neither is available:
   - **HRX2** (`build/bin/llama-simple`, device `HRX20`) fails at compute with
     `HRX2: ... Loom source indexing failed: BYTECODE/002: unsupported format
     version 34, expected 38` and
     `HRX2: pointwise provider is not available for op=SCALE` — its Loom
     bytecode artifacts are stale relative to the installed `loomc`. Not
     fixable without regenerating the HRX2 catalog.
   - **CPU** explicitly rejects the Q4NX ops
     (`ggml/src/ggml-cpu/ggml-cpu.cpp:439`: *"Q4NX ops run only on the HRX2
     backend"*), so it cannot run the model.
   Parity is therefore established at the **op level only**, where all three
   kernels are bit-exact/equivalent against an independent CPU reference of
   the format.
2. **Existing HRX0 test suite does not pass cleanly.**
   `build-hrx0-latest/bin/test-backend-hrx` aborts on an `rms_norm` mismatch
   with a *nondeterministic* failing index (`rms_norm[159]`, `[709]`, `[32]`
   across runs). This is pre-existing: stashing all Q4NX changes and rebuilding
   reproduces the identical failure, and the Q4NX change is additive
   (119 insertions, 0 deletions at task-3; task-5 relaxes one predicate).
3. **Type 43 (`Q4NX_C43`) is not registered in ggml.** Branch HEAD `b08d51080`
   registered it in gguf-py only; `ggml.h` stops at 42 / `GGML_TYPE_COUNT = 43`.
   `store43/zaya1-8b-ft-q4nx.gguf` (280 type-43 tensors) is therefore still
   rejected at GGUF parse. The type-42 models (`-c42`, `-repacked`,
   `zaya-q4nx-gemma4tok`) load. The kernels are type-agnostic; extending the
   enum is a separate, decision-requiring change.
