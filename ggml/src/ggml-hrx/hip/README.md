<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# HIP kernels on the HRX backend

HRX can run code objects built by hipcc/amdclang++. The HRX runtime loads them
(`hrx_executable_load_data`, the same call the Loom JIT result goes through) and dispatches
them. The HIP runtime and ggml-hip are not involved. This directory lets a ggml-hrx matcher
dispatch a HIP kernel the same way it dispatches a Loom kernel.

## Layout

| file | role |
|---|---|
| `kernels/*.hip` | kernel sources; every file here is built for each target in `GGML_HRX_HIP_TARGETS` |
| `ggml-hrx-hip.cmake` | compiles each `.hip` to a raw code object (`--cuda-device-only --no-gpu-bundle-output`), embeds the code objects, adds the sources below |
| `embed_hip_code_objects.py` | writes `ggml-hrx-hip-code-objects.inc` (bytes + digest per stem/target) |
| `hip-code-objects.{h,cpp}` | lookup of embedded code objects by (stem, target) |
| `hip-kernel-registry.{h,cpp}` | `HipKernel` description -> `KernelDefinition` (family `hip`), found by `resolve_kernel_definition` |
| `hip-kernel-loader.{h,cpp}` | gives the kernel executable cache a finished "compile" (embedded code object + launch geometry) in place of a Loom JIT |
| `hip-dispatches.{h,cpp}` | **the one place HIP matchers are registered** |
| `dispatch-hip-scale.cpp`, `kernels/hip_scale_f32.hip` | worked example (GGML_OP_SCALE, opt-in `GGML_HRX_HIP_EXAMPLE_SCALE=1`) |
| `hip-smoke.cpp`, `kernels/hip_smoke.hip` | `ggml-hrx-hip-smoke`: libhrx-only load/dispatch/check of a code object |
| `drafts/` | kernels that are not built yet |

Hooks in upstream ggml-hrx files are one line each: the `include()` in `CMakeLists.txt`, the
HIP lookup in `resolve_kernel_definition` (kernel-corpus.cpp), the HIP branch in
`KernelExecutableCache::get_or_compile`, the `friend struct HipCodeObjectLoader` in
loom-kernel-jit.h and `register_hip_dispatches` in dispatch-common.cpp.

## Adding a kernel

1. **Kernel**: `kernels/<stem>.hip`, with the Apache header. The arguments are all pointers first
   (each one is an HRX binding, in order), then `uint32_t` by-value arguments (the launch
   parameters, in order). Pass floats as `uint32_t` bit patterns (`__builtin_bit_cast`). Use
   `extern "C" __global__` with `__launch_bounds__`. `blockDim`/`gridDim` work, because HRX fills
   the hidden kernargs. gfx11 is wave32: use `__shfl_xor(v, o, 32)` and
   `__builtin_amdgcn_wmma_*_w32`.
2. **Matcher**: `dispatch-hip-<name>.cpp` (our header). In its `register_hip_<name>_dispatch`:
   - `register_hip_kernel({ "<symbol>", "<stem>", bindings, launch_parameters, workload_parameters, launch_fn })`;
   - `registry.add({ "hip.<name>", GGML_OP_..., kind, priority, DispatchSource::Common, matcher })`.
     Name registrations `hip.*` so that `GGML_HRX_DISABLE_DISPATCH=hip.` turns every HIP matcher off.
   The matcher builds `Dispatch` as for Loom: `make_kernel_specialization(hip_kernel_ref("<symbol>"))`,
   `integer_parameters` for every launch and workload parameter, and bindings in pointer order.
   `launch_fn(params, geometry)` sets the workgroup count and size. The executable cache key
   holds only the workload parameters, so list the parameters that change the grid there and
   nothing else.
3. **Register**: add one line to `register_hip_dispatches` in `hip-dispatches.cpp` and the
   matcher source to `target_sources(ggml-hrx ...)` in `ggml-hrx-hip.cmake`.
4. **Check** (loom worker rules): `test-backend-ops -o <OP> -b HRX0`, run several times, with
   `GGML_HRX_LOG_DISPATCH=1` to confirm `hip.<name>` matched. Add a known-answer probe, a model
   KLD or teacher-forced comparison, and repeated identical requests. Keep a HIP kernel only
   if an interleaved A/B shows it at least as fast as the path it replaces.

At load, a missing (stem, target) code object fails with `no <target> code object '<stem>'`.
A kernel ABI that differs from the registration (binding count or constant bytes) fails with
`compiled ABI does not match manifest`.

## Build

`GGML_HRX_HIP_COMPILER` (default: `amdclang++` from `CMAKE_CXX_COMPILER` or /opt/rocm-therock/bin),
`GGML_HRX_HIP_TARGETS` (default `gfx1151`, semicolon list), `GGML_HRX_HIP_FLAGS` (default `-O3`).
Look at the ISA with
`llvm-objdump -d --mcpu=gfx1151 build/ggml/src/ggml-hrx/hip-code-objects/<stem>.gfx1151.hsaco`.
