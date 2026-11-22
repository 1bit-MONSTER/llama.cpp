#!/usr/bin/env bash
# one arm of the #176 A/B: same model/args, staging trace on, GTT peak sampled
set -u
BIN=$1; TAG=$2
OUT=/home/bcloud/wt/tprs-176/arm-$TAG
mkdir -p "$OUT"
export IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1
export GGML_HRX_HOST_STAGING_TRACE=1
GTT=$(ls /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | head -1)
: > "$OUT/gtt.log"
( while :; do v=$(cat "$GTT" 2>/dev/null || echo 0); echo "$v"; sleep 0.2; done ) > "$OUT/gtt.log" &
SAMP=$!
"$BIN" -m /home/bcloud/models/Qwen3-4B-Q4_K_M.gguf -ngl 0 -c 4096 -n 8 \
    -p "The capital of France is" -no-cnv > "$OUT/stdout.log" 2> "$OUT/stderr.log"
rc=$?
kill $SAMP 2>/dev/null
echo "rc=$rc" > "$OUT/rc"
echo "arm $TAG rc=$rc"
