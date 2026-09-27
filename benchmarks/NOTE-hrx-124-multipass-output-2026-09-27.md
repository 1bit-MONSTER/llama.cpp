# HRX decode-split multipass output pass: drop the 128x redundant expf and the scalar block loop (engine#124)

Worktree: `~/1bit-engine-176/third_party/llama.cpp`, branch `1bit/hrx-124-output-pass` (base `00adc2b`).
Box: gfx1151 (Strix Halo), `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M`, `-dev HRX0`.

## What was wrong

`@ggml.flash_attention.decode_split.reduce_completed.multipass`
(`ggml/src/ggml-hrx/kernel-corpus/kernels/loom-libs/ops/flash_attention_decode_split_f32_f16_wmma.loom`)
finished with a scalar output pass: `workitem -> one output channel`, then a serial
`scf.for %block = [%c0 to %active_block_count]` over **every** KV block, run by all 256
workitems. Only 128 are live at `value_head_size = 128`, and each one recomputed
`scale = expf(partial_max[block] - maximum)` for every block, so `expf` ran once per
`(block, output element)` instead of once per `(block)`, and every element walked the
block dimension alone. That is the residual ~16% gap above capacity 2048 (issue #124).

## The change (bit-exact)

1. **LDS scale stage.** The lane-strided sum pass already computes
   `scale = expf(partial_max[block] - maximum)` for every block; store it into a
   per-row workgroup stage (`scale_stage_view[row, block]`) and also publish the row
   `sum` (`sum_stage_view[row]`). The output pass then just reloads the exact f32
   value. Removes ~128x redundant `expf` and the redundant `partial_max` reloads.
2. **Vectorised all-rows output pass.** After the per-row max/sum phase, one pass
   covers all query rows at once: workitem `w` owns a 4-channel *quad*
   (`quad = tile*256 + w`, `row = quad / quads_per_row`, `channel = (quad %
   quads_per_row)*4`), loads `vector<4xf16>` from `partial_output`, and accumulates
   `vector<4xf32>` over blocks with `unroll(%c4) schedule(interleaved)`. For the
   production GQA shape (8 query heads per KV head, `value_head_size = 128`) there are
   exactly 256 quads, so the whole workgroup (all four subgroups) is busy and each
   channel still accumulates its blocks in the same order as before.

Because the per-channel accumulation order is unchanged and the vector ops are
elementwise, the reduce output is **bit-identical** to the previous multipass reducer
(verified end-to-end below).

## Result

`llama-bench -p 0 -n 8 -r 5 -d 1900,2000,2100,3000,4800`, 6 interleaved A/B rounds,
median of the 6 run means:

| depth | capacity | blocks | base t/s | patched t/s | delta | path |
|---|---:|---:|---:|---:|---:|---|
| 1900 | 1920 | 30 | 70.72 | 71.48 | +1.1% | cooperative (untouched control) |
| 2000 | 2048 | 32 | 66.06 | 68.82 | +4.2% | cooperative (untouched control) |
| 2100 | 2112 | 33 | 58.19 | 67.15 | **+15.4%** | multipass |
| 3000 | 3008 | 47 | 50.84 | 60.89 | **+19.8%** | multipass |
| 4800 | 4864 | 76 | 41.83 | 51.02 | **+22.0%** | multipass |

The boundary cliff is gone: `d2100/d2000` moves **0.881 -> 0.976** (and
`d4800/d2000` 0.633 -> 0.741). The two depths the reducer does not touch moved by
+1.1% / +4.2%, i.e. run-to-run noise.

A throwaway diagnostic that deleted the output block loop entirely (keeping the
max/sum passes) measured **+21.6% / +50.0% / +30.5%** at d2100 / d3000 / d4800, which
is how the output pass was identified as the whole gap rather than the reduction math
(issue #124's diagnosis).

## Evidence

- **Bit-exact.** `llama-perplexity -c 2049 -b 1 -f <6600-token text> --save-all-logits`
  (token-by-token decode, so the decode-split kernel is dispatched; `GGML_HRX_LOG_DISPATCH=1`
  confirms `flash_attention_decode_split`). The 1,244,725,284-byte logits dumps for base
  and patched are **byte-identical** (`cmp`), PPL 21.9760 +/- 0.81813 on both.
- **Correctness.** Buried code word `ZX-4718-QQ` at 4700 prompt tokens (capacity 4864,
  multipass), `llama-server`, temperature 0, seed 42, `cache_prompt:false` -> answer
  `ZX-4718-QQ`. PASS.
- **Faults.** 6 A/B rounds x 5-rep sweeps at all five depths on the patched build:
  0 `HSA_STATUS_ERROR_MEMORY_FAULT`, 0 `res = -3`. The cooperative/direct paths are
  untouched, so <= 2048 (d1900/d2000) shows no tok/s or fault regression.

## Reproduce

```
# build (engine worktree ~/1bit-engine-176, HRX llama.cpp build dir)
cmake --build build/hrx/llama --target llama-bench llama-perplexity -j 16

# dispatch needs libhsa from the HRX toolchain
export IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1

# performance
build/hrx/llama/bin/llama-bench -m models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  -dev HRX0 -ngl 99 -p 0 -n 8 -r 5 -d 1900,2000,2100,3000,4800

# bit-exact reference vs the previous build
build/hrx/llama/bin/llama-perplexity -m models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  -f ref-text.txt -c 2049 -b 1 -dev HRX0 -ngl 99 --save-all-logits logits.bin
```

Raw runs and the A/B driver used here live under the session scratch dir
(`ab5-*.json`, `ab5-summary.txt`, `diag-noloop.json`).
