#!/usr/bin/env bash
# pi-hrx-sched, overnight #674: one decisive question the thread never asked.
#
# The corrupting configuration needs EVERY weight-consuming op on the CPU (GEMMs *and* the norm-scale MUL; removing
# either one is correct). What is left on HRX then is a graph of pure activation ops, and the thread's own sched_reserve
# line says what that costs: "HRX0_HOST compute buffer size = 13.26 MiB", i.e. the transients land in HRX's *host* buffer
# and are bound through the host-staging round trip instead of device memory. So the hypothesis this arm set tests is that
# #674 is not about weights at all but about how HRX executes a graph whose values are host-visible, and the knob that
# changes exactly that is GGML_HRX_USE_UNIFIED_MEMORY (direct_host_binding: a host buffer binds the HRX buffer handle
# instead of taking the staging path).
#
# PREDICTION, written before the run:
#   * unified memory ON fixes the garbage  -> the corruption lives in the non-direct host-binding / host-staging path for
#     transients, which is code that merged builds still run for any host-visible value;
#   * unified memory ON is still garbage   -> the corruption is in how HRX executes host-visible values at all (dispatch
#     side), independent of the binding route;
#   * the control arm (decline MUL_MAT only + unified memory) stays correct either way, or the knob itself is doing
#     something unrelated and nothing here is a #674 statement.
# Every arm is one 8-token greedy completion; the reference text is "Okay, the user is asking".
set -u
TIP=/home/bcloud/wt/i674-tip/build-tip/bin
M=/home/bcloud/models/Qwen3-0.6B-Q4_K_M.gguf
THROCK=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib
export HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1
export IREE_HAL_AMDGPU_LIBHSA_PATH=$THROCK/libhsa-runtime64.so.1
export LD_LIBRARY_PATH=$TIP:$THROCK
die() { local f; for f in /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input; do [ -r "$f" ] && echo "$(($(<"$f")/1000))C"; done | head -1; }
text() { python3 - "$1" <<'PY'
import sys
lines=open(sys.argv[1],encoding='utf-8',errors='replace').read().splitlines()
start=None
for i,l in enumerate(lines):
    if l.startswith('> What is the capital'): start=i+1
body=[]
for l in lines[start or 0:]:
    if l.startswith('[ Prompt:'): break
    s=l.strip()
    if s and s!='[Start thinking]': body.append(s)
print('|'.join(body) if body else '<no completion>')
PY
}

echo "provenance: $(git -C /home/bcloud/wt/i674-tip log --oneline -1)"
echo "knob strings: $(strings $TIP/libggml-hrx.so | grep -c GGML_HRX_169_DECLINE)"
arm() { # $1 tag, $2 env string, $3.. cli
  local tag=$1 envs=$2; shift 2
  local log=/tmp/uni-$tag.log
  env $envs $TIP/llama-cli -m $M -fit off -c 256 --temp 0 -no-cnv --no-warmup --no-perf -n 8 \
      -p "What is the capital of France?" -dev HRX0 -ngl 0 "$@" >"$log" 2>&1
  printf '%-26s %s  rc=%d  text=%s\n' "$tag" "$envs" "$?" "$(text "$log")"
}

arm baseline-garbage        "GGML_HRX_169_DECLINE=MUL_MAT,MUL"
arm unified                 "GGML_HRX_169_DECLINE=MUL_MAT,MUL GGML_HRX_USE_UNIFIED_MEMORY=1"
arm unified-serialized      "GGML_HRX_169_DECLINE=MUL_MAT,MUL GGML_HRX_USE_UNIFIED_MEMORY=1 GGML_HRX_DEBUG_SERIAL_EXECUTION=1"
arm control-gemm-only       "GGML_HRX_169_DECLINE=MUL_MAT GGML_HRX_USE_UNIFIED_MEMORY=1"
arm control-no-decline      "GGML_HRX_USE_UNIFIED_MEMORY=1"
arm no-decline-plain        ""
echo "die after: $(die)"
