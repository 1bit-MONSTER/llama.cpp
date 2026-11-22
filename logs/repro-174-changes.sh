#!/usr/bin/env bash
# pi-i669: answer PR #174's CHANGES review - the reviewer got 0 guard lines and an arena_reserve refusal, I got
# 2 and 8 lines with "Error: Compute error.". Same builder head, same knob, same models: reproduce both tables here.
set -u
BIN=/home/bcloud/wt/i674-repro/build-priv/bin
THROCK=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib
TRIO="HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1"
export HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1
export IREE_HAL_AMDGPU_LIBHSA_PATH=$THROCK/libhsa-runtime64.so.1
export LD_LIBRARY_PATH=$BIN:$THROCK

echo "### identity: guard strings in the library this binary will load"
strings $BIN/libggml-hrx.so.0.18.0 | grep -c "hrx placement: %s is not run on HRX"
sha256sum $BIN/libggml-hrx.so.0.18.0 | cut -c1-16

run() { # $1 tag $2 model $3 budget(empty=unset)
    local tag=$1 m=$2 b=$3 log=/tmp/repro-174-$1.log
    if [ -n "$b" ]; then export GGML_HRX_HOST_WEIGHT_LIMIT_MB=$b; else unset GGML_HRX_HOST_WEIGHT_LIMIT_MB; fi
    $BIN/llama-cli -m "$m" -dev HRX0 -fit off -ngl 0 -c 256 --temp 0 -no-cnv --no-warmup --no-perf -n 8 \
        -p "What is the capital of France?" >"$log" 2>&1
    local rc=$?
    printf '%-22s budget=%-6s rc=%-3d guard=%-2s arena=%s\n' "$tag" "${b:-unset}" "$rc" \
        "$(grep -c 'hrx placement:' "$log")" "$(grep -c 'arena_reserve' "$log")"
    grep -m3 'hrx placement:' "$log" | sed 's/^/     | /'
    grep -m1 'arena_reserve:' "$log" | cut -c1-160 | sed 's/^/     ! /'
    grep -m1 'refusing to run' "$log" | cut -c1-200 | sed 's/^/     ! /'
    tail -3 "$log" | sed 's/^/     > /'
    echo
}

for m in /home/bcloud/models/Qwen3-4B-Q4_K_M.gguf /home/bcloud/models/Qwen3-0.6B-Q4_K_M.gguf; do
    echo "== $(basename "$m")"
    run "$(basename "$m" .gguf)-unset" "$m" ""
    run "$(basename "$m" .gguf)-128"   "$m" 128
    run "$(basename "$m" .gguf)-8"     "$m" 8
done
for f in /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input; do [ -r "$f" ] && echo "die after: $(($(<"$f")/1000))C"; done | head -1
