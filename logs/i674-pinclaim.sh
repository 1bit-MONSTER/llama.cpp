#!/usr/bin/env bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
# The issue's own claim, at the merged pin: "any graph whose weight ops all run on the CPU computes garbage".
# -ngl 0 puts every layer on the CPU (the debug dump shows layer 0 on CPU), so this is that graph. One ticket.
set -u
bindir=${1:-$HOME/wt/i674-repro/stage-pin788}
model=$HOME/models/Qwen3-4B-Q4_K_M.gguf
out=$HOME/wt/i674-repro/logs/pinclaim
env -u HSA_ENABLE_SDMA HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1 \
    GGML_HRX_POOL_LIMIT_MB=97280 ${BUD:+GGML_HRX_HOST_WEIGHT_LIMIT_MB=$BUD} \
    IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1 \
    LD_LIBRARY_PATH=$bindir \
    ~/.local/bin/gpu-hold -l i669-674-pinclaim -t 20 -- \
    $bindir/llama-cli -m "$model" ${DEV:--dev HRX0} -fit off -ngl 0 -c 256 --temp 0 -no-cnv -n 8 \
        -p "What is the capital of France?" > "$out-${DEV_NAME:-hrx}${BUD:+-bud}.log" 2>&1
echo "rc=$?"; grep -aE 'assigned to device|host weight|^> |Paris|Ok' "$out-${DEV_NAME:-hrx}${BUD:+-bud}.log" | tail -6
