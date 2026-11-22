#!/usr/bin/env bash
set -u
BIN=/home/bcloud/wt/tprs-176/build-176/bin/llama
OUT=/home/bcloud/wt/tprs-176/smoke
export IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1
export GGML_HRX_HOST_STAGING_TRACE=1
"$BIN" cli -m /home/bcloud/models/Qwen3-4B-Q4_K_M.gguf -ngl 0 -c 4096 -n 8 -p "The capital of France is" -no-cnv > "$OUT/out.log" 2> "$OUT/err.log"
echo "rc=$?" > "$OUT/rc"
