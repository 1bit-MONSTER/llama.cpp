#!/usr/bin/env bash
# one arm of the #176 staging A/B: run the model, trace staging, sample GTT peak
set -u
OUT=$1
mkdir -p "$OUT"
export IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1
export GGML_HRX_HOST_STAGING_TRACE=1
GTT=$(ls /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | head -1)
: > "$OUT/gtt.log"
( while :; do cat "$GTT" 2>/dev/null; sleep 0.2; done ) > "$OUT/gtt.log" &
SAMP=$!
/home/bcloud/wt/tprs-176/build-176/bin/llama cli -m /home/bcloud/models/Qwen3-4B-Q4_K_M.gguf -ngl 0 -c 4096 -n 8 -p "The capital of France is" -no-cnv > "$OUT/out.log" 2> "$OUT/err.log"
rc=$?
kill $SAMP 2>/dev/null
echo "rc=$rc" > "$OUT/rc"
echo "arm $(basename "$OUT") rc=$rc"
