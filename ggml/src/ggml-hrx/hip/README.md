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
| `GGML_HRX_HIP_ADDON_DIR` | optional out-of-tree kernel add-on, built in the same way (see "Kernel add-ons") |

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
   nothing else. Leaving one out does not fall back to the right grid: the first case that
   compiles the kernel fixes the geometry for every later case with the same key, and the
   kernel then runs too few workgroups and silently leaves the rest of the output untouched
   (see "Trap: a workload parameter left out is a stale grid").
3. **Register**: add one line to `register_hip_dispatches` in `hip-dispatches.cpp` and the
   matcher source to `target_sources(ggml-hrx ...)` in `ggml-hrx-hip.cmake`. (In an add-on,
   the line goes in its `ggml_hrx_hip_addon_register` and the source is picked up by the glob.)
4. **Check** (loom worker rules): `test-backend-ops -o <OP> -b HRX0`, run several times, with
   `GGML_HRX_LOG_DISPATCH=1` to confirm `hip.<name>` matched. Add a known-answer probe, a model
   KLD or teacher-forced comparison, and repeated identical requests. Keep a HIP kernel only
   if an interleaved A/B shows it at least as fast as the path it replaces.

**Trap: a workload parameter left out is a stale grid, and it is silent.** `launch_fn` runs once
per executable cache entry, not once per dispatch, so the grid it computes is reused for every
later dispatch whose key matches. Because the key is the workload-parameter list and nothing
else (not the geometry), a launch function that reads a parameter the list omits gets the first
case's grid for every case: no error, no log line, and the kernel simply does not cover the
space it was asked to. Measured on the DeepSeek-V4.1 indexer (`n_stream` left out, published as
gpu-kernels #46): `test-backend-ops -o LIGHTNING_INDEXER -b HRX0` gave **4/12**, with every
`ns=1` case passing and every `ns=4` case wrong by a whole stream plane, because streams 1-3
were never written: `inf mismatch: HRX0=0.943650 CPU=-inf` where an untouched element met a
`-inf` mask row, and `ERR = 2.909230024 > 0.000001000` on a case whose untouched elements never
happened to meet one. The same kernel driven directly
through `hipModuleLaunchKernel` on the same code object is correct at 4 streams to 3.9e-03, so a
standalone harness cannot see this: only a suite that runs the op at two different grid sizes
can. `test-backend-ops` does exactly that for ops with a stream/batch axis; keep those cases.

**Trap: fused registrations always come first.** For a root op the registry tries every
`DispatchMatchKind::Fused` registration before any `SingleOp` one; `priority` only orders
registrations within the same kind. A `SingleOp` HIP matcher with a high priority still loses
to any fused Loom matcher that takes the node, and a fused HIP matcher wins over every single-op
matcher whatever its priority. Check `GGML_HRX_LOG_DISPATCH=1` rather than reasoning from priorities.

At load, a missing (stem, target) code object fails with `no <target> code object '<stem>'`.
A kernel ABI that differs from the registration (binding count or constant bytes) fails with
`compiled ABI does not match manifest`.

## Build

`GGML_HRX_HIP_COMPILER` (default: `amdclang++` from `CMAKE_CXX_COMPILER` or /opt/rocm-therock/bin),
`GGML_HRX_HIP_TARGETS` (default `gfx1151`, semicolon list), `GGML_HRX_HIP_FLAGS` (default `-O3`).
Look at the ISA with
`llvm-objdump -d --mcpu=gfx1151 build/ggml/src/ggml-hrx/hip-code-objects/<stem>.gfx1151.hsaco`.

## Kernel add-ons

Kernels and matchers can live outside this tree. Configure with
`-DGGML_HRX_HIP_ADDON_DIR=<dir>` (absolute, or relative to the top source directory):

| add-on path | what the build does with it |
|---|---|
| `<dir>/kernels/*.hip` | compiled and embedded with `kernels/*.hip` (same targets, flags and lookup by stem; `-I` the kernel's own directory) |
| `<dir>/*.cpp`, `<dir>/*.h` | compiled into ggml-hrx, with `GGML_HRX_HIP_ADDON` defined |
| `<dir>/addon.cmake` | included at the end of `ggml-hrx-hip.cmake` if present (extra tools; relative source paths resolve against `ggml/src/ggml-hrx`) |

The add-on defines `ggml::hrx::ggml_hrx_hip_addon_register(DispatchRegistryBuilder &)`
(declared in `hip-dispatches.h`); `register_hip_dispatches` calls it after the matchers in this
directory, once per registry build. Add-on sources include `hip/hip-dispatches.h` and
`hip/hip-kernel-registry.h` like the example does. Kernel stems must not repeat a stem from
`kernels/` (configure fails). An add-on can also take the attention-sink rescale that follows
FlashAttention through `set_hip_attention_sink_hook`. A matcher for an op that ggml-hrx's
`supports_op` does not list by itself declares the op from its `register_*` function with
`hip_declare_eager_op(op)` (`hip/hip-capabilities.h`); `eager_capability_declared` consults that
set for every op outside its built-in list, so the node reaches the dispatcher instead of another
backend. Without the option nothing changes: only the kernels and matchers in this directory are
built, and the declared-op set stays empty.

## Worked example 2: `GGML_OP_TOP_K` (engine #679)

`dispatch-hip-top-k.cpp` + `kernels/hip_top_k_f32.hip` are the first non-example use of this recipe, and the
shortest path to the same thing again:

1. The kernel takes `(src, dst, column_count, top_k)` — two bindings, then the by-value arguments in order.
   `row_count` is a *workload* parameter only, because it sizes the grid (`workgroup_count = { rows, 1, 1 }`)
   and is not passed to the kernel. A launch parameter that the kernel does not declare fails at load with
   `compiled ABI does not match manifest`.
2. The matcher is a `SingleOp` registration at priority 10. It does not opt in behind an env var: this is a
   real dispatch, so `register_hip_top_k_dispatch` also calls `hip_declare_eager_op(GGML_OP_TOP_K)` to get the
   node out of the CPU backend in the first place.
3. Because the op is now *claimed*, every TOP_K node in every model reaches the dispatcher, and a node the
   matcher declines would become `unsupported HRX node` and fail the graph. The claim is therefore bounded per
   node in `ggml-hrx.cpp`'s `device_supports_op` through `hip_top_k_tensor_supported`, which checks exactly what
   the matcher checks (`f32` contiguous source, `i32` output, `1 <= k <= n`, `n <= 262144`); anything outside
   that keeps the CPU fallback. Mirroring a matcher's predicate in that function is the pattern to copy.

Why it exists: Qwen3.8-Flash-Next's QSA layers call `ggml_top_k` with `n = n_kv` (up to 262144) and
`k = indexer_top_k + r - 1 = 2051`, once per full-attention layer per decode step. On the CPU that was a
`std::partial_sort` per row with an indirect comparator plus a graph split around the node: measured +9.7 ms per
decode step when `n_kv` crossed 2051 and another ~3.8 us per KV token after that (9.70 tok/s at 18,782 tokens
against 31.15 for a GPU TOP_K).

Check it:

```
build/bin/test-backend-ops -o TOP_K -b HRX0            # set semantics, ties variant; run it more than once
GGML_HRX_LOG_DISPATCH=1 ...                           # must print hip.top_k_f32
```
