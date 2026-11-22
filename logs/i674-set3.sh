#!/usr/bin/env bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# i674-set3.sh BINDIR_CONTROL BINDIR_DOWNLOAD_ALWAYS
#
# Engine #674, two questions in one ticket:
#   A  name the tensors behind the 512 KiB staging buffers: the stock ggml scheduler dump (GGML_SCHED_DEBUG=2)
#      prints every split input with its tensor name and size, so the value ids in #165's staging trace can be
#      matched to real tensors - no code needed.
#   B  does the experiment build (always download a value HRX wrote, instead of skipping GGML_TENSOR_FLAG_INPUT)
#      make the declined-weight graph correct?
# 0.6B, 8 tokens, ~1 s of GPU per arm.
set -u
bctl=${1:?usage: i674-set3.sh BINDIR_CONTROL BINDIR_DOWNLOAD_ALWAYS}
bdl=${2:?usage: i674-set3.sh BINDIR_CONTROL BINDIR_DOWNLOAD_ALWAYS}
out=${OUT:-$HOME/wt/i674-repro/logs/set3}
model=$HOME/models/Qwen3-0.6B-Q4_K_M.gguf
GTT=/sys/class/drm/card0/device/mem_info_gtt_used
: > "$out.txt"

run() { # run NAME BINDIR DECLINE "--device args" [env...]
    local name=$1 bindir=$2 decline=$3 dev=$4
    shift 4
    local start_gtt end_gtt rc
    start_gtt=$(cat $GTT)
    env -u HSA_ENABLE_SDMA \
        HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1 \
        GGML_HRX_POOL_LIMIT_MB=${LIMIT:-97280} ${decline:+GGML_HRX_169_DECLINE=$decline} "$@" \
        IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1 \
        LD_LIBRARY_PATH=$bindir \
        $bindir/llama-cli -m "$model" $dev -fit off -ngl 0 -c 256 --temp 0 -no-cnv -n 8 \
            -p "What is the capital of France?" > "$out-$name.log" 2>&1
    rc=$?
    end_gtt=$(cat $GTT)
    {
        printf '=== %s rc=%s gtt %sM -> %sM\n' "$name" "$rc" "$((start_gtt / 1048576))" "$((end_gtt / 1048576))"
        awk '/^> What is the capital/{f=1;next} /\[ Prompt:/{f=0} f' "$out-$name.log" \
            | grep -av "hrx debug serial\|hrx host staging" | tr -d '\r' | sed 's/^/OUT| /'
    } | tee -a "$out.txt"
}

run declsched "$bctl" "all" "-dev HRX0" GGML_SCHED_DEBUG=2
run decl_dl   "$bdl"  "all" "-dev HRX0"
echo "SET3 DONE" | tee -a "$out.txt"

# A: the tensors, grouped by the size the staging trace shows.
{
    echo "--- split inputs by size (from the schedule dump) ---"
    grep -a "KiB\|MiB" "$out-declsched.log" | grep -a "\[" | sed 's/^ *//' | sort | uniq -c | sort -rn | head -25
} | tee -a "$out.txt"
