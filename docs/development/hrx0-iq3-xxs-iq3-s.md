# HRX0 IQ3_XXS and IQ3_S on-device matvec

This note records the HRX0 (HIP) work that makes sub-4-bit "UD" GGUFs execute
their IQ3_XXS and IQ3_S matmuls on the GPU instead of CPU-falling-back, with
the exact environment, reproducible commands and measured results.

## Scope

- Backend: `ggml/src/ggml-hrx/` (the HIP-kernel HRX0 backend) in this tree.
- Kernels: `hrx_mul_mat_vec_iq3_xxs_f32` (commit `1c098eeb2`) and
  `hrx_mul_mat_vec_iq3_s_f32` (commit `537f0499f`).
- Target: `gfx1151` (Strix Halo / Radeon 8060S), device `HRX0`.

Only IQ3_XXS and IQ3_S are added here. Other sub-4-bit types present in the UD
containers (`iq1_s`, `iq1_m`, `iq2_xxs`, `iq2_s`, `iq2_xs`, `iq4_xs`) have no
HRX0 kernel and still fall back to CPU (see *Remaining CPU fallback* below).

## Environment

| Component | Value |
| --- | --- |
| hrx-system | `ROCm/hrx-system` `main` @ `c702461ed` |
| Install prefix | `/home/bcloud/hrx-latest/prefix` |
| ROCm toolchain | `/opt/rocm-therock` (AMD clang 23) |
| HSA runtime | TheRock `_rocm_sdk_core` `libhsa-runtime64.so.1` |

The current hrx-system requires the newer HSA entry point `hsa_amd_queue_create`,
which the ROCm-TheRock HSA runtime exports but the older `libhsa-runtime64`
under `/opt/hrx/lib` does not. The runtime also needs the ROCm sysdeps on the
loader path. A working loader path is therefore:

```bash
export LD_LIBRARY_PATH=/home/bcloud/hrx-latest/prefix/lib:/home/bcloud/hrx-latest/hsa:/opt/hrx/lib/rocm_sysdeps/lib
```

`/home/bcloud/hrx-latest/hsa` contains symlinks to TheRock's
`libhsa-runtime64.so.1` (plus the aqlprofile and rocprofiler-register shims).

## Build

hrx-system (runtime + loomc) into a fresh prefix:

```bash
cd /home/bcloud/amd-oss/hrx-system   # main @ c702461ed
python3 dev.py --cmake-build-dir build-c702461e cmake configure --fresh \
  -DCMAKE_BUILD_TYPE=Release -DIREE_HAL_DRIVER_AMDGPU=ON \
  -DCMAKE_C_COMPILER=/opt/rocm-therock/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-therock/bin/amdclang++ \
  -DCMAKE_ASM_COMPILER=/opt/rocm-therock/bin/amdclang \
  -DCMAKE_C_COMPILER_AR=/usr/bin/ar -DCMAKE_C_COMPILER_RANLIB=/usr/bin/ranlib \
  -DCMAKE_CXX_COMPILER_AR=/usr/bin/ar -DCMAKE_CXX_COMPILER_RANLIB=/usr/bin/ranlib \
  -DIREE_ENABLE_WERROR_FLAG=OFF -DCMAKE_INSTALL_PREFIX=/home/bcloud/hrx-latest/prefix \
  -DIREE_BUILD_TESTS=OFF -DLIBHRX_BUILD_CTS=OFF
python3 dev.py --cmake-build-dir build-c702461e cmake build hrx
# HRX0 kernels: kernels/mul_mat_vec_iq3_{xxs,s}.hip.cpp are compiled to HSACO by
# kernels/generate_hrx_kernels.py during this build.
```

HRX0 backend against that prefix:

```bash
cd /home/bcloud/hrx-ws/hrx-v2-src
cmake -S . -B build-hrx0-latest -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_HRX=ON -DGGML_HRX2=OFF \
  -DGGML_HRX_AMDGPU_TARGETS=gfx1151 \
  -DGGML_HRX_ROCM_PATH=/opt/rocm-therock \
  -DGGML_HRX_CLANGXX=/opt/rocm-therock/bin/amdclang++ \
  -DGGML_VULKAN=OFF -DGGML_NATIVE=OFF -DGGML_BUILD_TESTS=OFF \
  -Dhrx_DIR=/home/bcloud/hrx-latest/prefix/lib/cmake/hrx
ninja -C build-hrx0-latest test-backend-ops llama-perplexity llama-cli -j30
```

## Kernel design

Both kernels use one workgroup per `(row, col)` output with 256 threads;
64 threads cooperate on one 256-weight super-block and 4 blocks are in flight
per workgroup. Each thread produces 4 of the 32 weights in its block and the
workgroup reduces at the end.

- `hrx_mul_mat_vec_iq3_xxs_f32` decodes the 98-byte IQ3_XXS block using the
  256-entry `iq3xxs_grid` and the `ksigns_iq2xs` table; the grid magnitudes are
  stored in quarters, hence the final `0.25 * sum`.
- `hrx_mul_mat_vec_iq3_s_f32` decodes the 110-byte IQ3_S block
  (`d | qs[64] | qh[8] | signs[32] | scales[4]`): the 9-bit grid index is the
  `qs` low byte plus one `qh` bit, `iq3s_grid` is the 512-entry table, each
  32-weight group has a 4-bit scale (`db = d * (1 + 2*nibble)`) and four sign
  bytes (bits 0..3 for the even grid entry, 4..7 for the odd one). The grid
  stores true magnitudes, so there is no 1/4 factor.

## Verification

Dispatch/route trace (`GGML_HRX_TRACE_ROUTES=1`) confirms the kernels run
in-model, e.g. for `Qwen3-0.6B-UD-IQ2_M`:

```
HRX route MUL_MAT provider=hrx_mul_mat_vec_iq3_xxs_f32 type=iq3_xxs ... dst=Qcur-0
HRX route MUL_MAT provider=hrx_mul_mat_vec_iq3_s_f32   type=iq3_s   ... dst=Vcur-0
```

Op tests (`test-backend-ops -b HRX0`):

| Op | Result |
| --- | --- |
| `-o MUL_MAT` | 489/489 (incl. 22 `iq3_xxs`, 11 `iq3_s`) |
| `-o ADD` | 30/30 |

Perplexity, `bash llama-perplexity -f wiki.test.raw -c 512 --chunks 20`,
wikitext-2 (CPU reference is `-dev none`; `HRX0` is `-dev HRX0 -ngl 99`):

| Model | HRX0 | CPU | Δ / σ | iq3_s routes | iq3_xxs routes |
| --- | --- | --- | --- | --- | --- |
| UD-IQ1_S | 10197.78 ± 674.45 | 9248.60 ± 604.27 | 1.05σ | 95 | 105 |
| UD-IQ2_M | 59.09 ± 3.08 | 59.00 ± 3.07 | 0.02σ | 1250 | 400 |
| UD-Q2_K_XL | 39.76 ± 1.84 | 39.62 ± 1.83 | 0.05σ | 295 | 210 |
| UD-Q4_K_XL | 24.45 ± 1.16 | 24.55 ± 1.16 | 0.07σ | 0 | 0 |

All four agree with the CPU reference within statistical noise. The absolute
values are corpus-specific (e.g. BF16 on the same corpus/settings is ≈21.7);
the metric of interest is HRX0 ≈ CPU.

## Remaining CPU fallback

The UD containers mix quant types. HRX0 kernels exist for `iq3_xxs` and
`iq3_s` only, so the following still run on CPU (tensor counts per file):

| Model | CPU-fallback types |
| --- | --- |
| UD-IQ1_S | `iq1_s` 84, `iq2_xxs` 43, `iq1_m` 36, `iq2_s` 5 |
| UD-IQ2_M | `iq2_s` 110, `iq2_xs` 5, `iq4_xs` 1 |
| UD-Q2_K_XL | `iq2_s` 10, `iq2_xs` 10, `iq4_xs` 5 |
| UD-Q4_K_XL | `iq4_xs` 20 |

UD-Q4_K_XL contains no IQ3 types at all, so it exercises no HRX0 matvec and is
a parity control.

## Integrity

A pristine worktree at the committed HEAD (`git worktree add --detach <dir> HEAD`)
configured and built with the commands above reproduces both HSACOs
(`build/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_iq3_{xxs,s}.hsaco`)
and `test-backend-ops -b HRX0 -o MUL_MAT` 489/489 (11 `iq3_xxs` + 11 `iq3_s`
cases), so the committed support is self-contained and does not depend on any
other working-tree edits.
