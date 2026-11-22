#!/usr/bin/env bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# i674-dump.sh [FILTER]
#
# Engine #674, the one experiment the 13:55 entry says is left: print the CONTENT at the layer-0 norm boundary
# (RMS_NORM "norm-0" -> MUL "attn_norm-0") for a single prefill, on three arms, and diff them:
#   cpu  -dev none        everything on CPU, known correct
#   mm   MUL_MAT declined the GEMM on CPU, the MUL on HRX  - known correct
#   pair MUL_MAT,MUL      both declined - reproduces the corruption
# If `norm-0` content already differs on the pair arm, the norm product is read from a different copy than the one
# written; if only `attn_norm-0` differs, the multiply reads a stale `mw`. No code, no rebuild: llama-debug is the
# fork's own examples/debug tool wired to ggml_backend_sched_set_eval_callback.
set -u
bindir=${1:-$HOME/wt/i674-repro/build-priv/bin}
# the tool anchors every pattern with "^" itself
filter=${2:-(norm-0|attn_norm-0)$}
model=$HOME/models/Qwen3-0.6B-Q4_K_M.gguf
out=$HOME/wt/i674-repro/logs/dump
[ -n "${ARMS:-}" ] || : > "$out.txt"

arm() { # arm NAME DECLINE DEV [BUDGET_MB]
    local name=$1 decline=$2 budget=${4:-} gtt0 gtt1
    gtt0=$(cat /sys/class/drm/card0/device/mem_info_gtt_used)
    env -u HSA_ENABLE_SDMA \
        HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1 \
        GGML_HRX_POOL_LIMIT_MB=${LIMIT:-97280} ${decline:+GGML_HRX_169_DECLINE=$decline} \
        ${budget:+GGML_HRX_HOST_WEIGHT_LIMIT_MB=$budget} \
        IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1 \
        LD_LIBRARY_PATH=$bindir \
        ~/.local/bin/gpu-hold -l "i669-674-dump-$name" -t 20 -- \
        "$bindir/llama-debug" -m "$model" -dev "$3" -fit off -ngl 0 -c 256 \
            --temp 0 -p "${PROMPT:-What is the capital of France?}" --tensor-filter "$filter" \
            > "$out-$name.log" 2>&1
    gtt1=$(cat /sys/class/drm/card0/device/mem_info_gtt_used)
    printf '=== %s (decline=%s, dev=%s, budget=%s) gtt %sM -> %sM\n' "$name" "${decline:-none}" "$3" \
        "${budget:-unset}" "$((gtt0 / 1048576))" "$((gtt1 / 1048576))" | tee -a "$out.txt"
    grep -aE "common_debug_cb_eval|sum = " "$out-$name.log" | tee -a "$out.txt" | sed 's/^/    /'
}

for a in ${ARMS:-cpu plain bud}; do
    case $a in
        cpu)   arm cpu   ""            none ;;
        plain) arm plain ""            HRX0 ;;
        # merged-code equivalent of the falsified placement: weights host resident, ops on the CPU
        bud)   arm bud   ""            HRX0 4096 ;;
    esac
done
echo "I674 DUMP DONE" | tee -a "$out.txt"
