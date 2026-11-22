#!/usr/bin/env bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# i674-split-arms.sh
#
# Engine #674 (lane i669, 15:10 AST entry): is the placement that corrupts - a weight-consuming op declined to the CPU
# while the rest of the graph stays on HRX - reachable at the merged pin d8bd0ff6052?
#
# host_weight_exceeds_residency() (ggml-hrx.cpp:1294 -> host-weight-placement.cpp:48) fires in device_supports_op,
# BEFORE any acquire, so #173's latched refusal never sees a declined weight. A budget under the largest weights but
# above the small ones therefore creates the mixed placement WITHOUT a refusal. 4096 (declines nothing, correct) and
# 256 (refused on the arena, 11-01 22:40) do not cover that band; 128 and 32 MiB do:
#   128 -> declines token_embd/output (~250 MiB), keeps the ~40 MiB per-layer GEMMs
#    32 -> declines those GEMMs too, keeps only the small ones
#
# Four arms in ONE gpu-hold ticket so they share one ambient GTT baseline. GGML_HRX_REPORT=1 on every arm so the
# declines are COUNTED from the placement census instead of inferred from the answer.
set -u
bindir=${BINDIR:-$HOME/wt/i674-repro/stage-pin788}   # llama-cli at pin d8bd0ff6052
model=$HOME/models/Qwen3-4B-Q4_K_M.gguf
logs=$HOME/wt/i674-repro/logs
out=$logs/split-arms.summary
samples=$HOME/wt/dsv41-testdata/pools-sample.sh
watchdog=$HOME/wt/dsv41-testdata/mem-watchdog.sh
G=/sys/class/drm/card0/device/mem_info_gtt_used

arm() { # arm NAME DEV BUDGET
    local name=$1 dev=$2 budget=${3:-} log=$logs/split-arm-$1.log trace=$logs/split-arm-$1.pools std=$logs/split-arm-$1.out
    env -u HSA_ENABLE_SDMA HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1 \
        GGML_HRX_POOL_LIMIT_MB=97280 GGML_HRX_HOST_WEIGHT_TRACE=1 GGML_HRX_REPORT=1 \
        ${budget:+GGML_HRX_HOST_WEIGHT_LIMIT_MB=$budget} \
        IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1 \
        LD_LIBRARY_PATH=$bindir \
        "$watchdog" 4 "$samples" "$trace" \
        "$bindir/llama-cli" -m "$model" -dev "$dev" -fit off -ngl 0 -c 256 -v \
            --temp 0 -no-cnv -n 8 -p "What is the capital of France?" > "$std" 2> "$log"
    local rc=$? start peak end floor lines resident declines refusals
    # pools-sample pads with %6d, so parse the gtt= token, never the column (the 12:07 runner's first bug).
    gttcol() { sed -n 's/^ *[0-9][0-9]* gtt= *\([0-9][0-9]*\)M.*/\1/p' "$trace" |
               { if [ "$1" = first ]; then head -1; elif [ "$1" = last ]; then tail -1; else sort -n | tail -1; fi; }; }
    start=$(gttcol first); peak=$(gttcol max); end=$(gttcol last)
    floor=$(sed -n 's/.*avail= *\([0-9][0-9]*\)M.*/\1/p' "$trace" | sort -n | head -1)
    lines=$(grep -ac 'hrx host weights: refilled=' "$log" || true)
    resident=$(grep -a 'hrx host weights: refilled=' "$log" | tail -1 | sed -n 's/.*resident=\([0-9.]*\) MiB.*/\1/p')
    refusals=$(grep -ac 'refusing to run' "$log" || true)
    # census: MUL_MAT nodes the scheduler kept OFF HRX. Presence proof that a decline happened, not a guess.
    declines=$(grep -ac 'using CPU instead\|declined' "$log" || true)
    {
        printf '=== %-8s dev=%-5s budget=%-6s rc=%d\n' "$name" "$dev" "${budget:-unset}" "$rc"
        printf '    gtt %sM -> peak %sM -> end %sM   peak_delta=+%sM  end_delta=+%sM\n' \
            "$start" "$peak" "$end" "$((peak - start))" "$((end - start))"
        printf '    MemAvailable floor=%sM  residency_lines=%s  resident=%s MiB  refusals=%s  decline_lines=%s\n' \
            "$floor" "$lines" "${resident:-none}" "$refusals" "$declines"
        printf '    answer: %s\n' "$(tr -d '\r' < "$std" | tail -2 | tr '\n' ' ' | cut -c1-120)"
    } | tee -a "$out"
}

if [ "${IN_HOLD:-}" = "1" ]; then
    printf '# split-arms %s bindir=%s arms=%s\n' "$(date '+%F %T %Z')" "$bindir" "${ARMS:-default}" >> "$out"
    # GGUF census (Qwen3-4B-Q4_K_M, 398 tensors): norm-scale weights are 512..10240 B, so a budget of any whole
    # MiB never declines the norm MUL. Declines by budget: 128/32 -> token_embd only (304 MiB); 8 -> +108 GEMMs;
    # 1 -> +252 GEMMs. ARMS= selects which of those bands a ticket measures.
    for a in ${ARMS:-cpu none bud128 bud32}; do
        case $a in
            cpu)    arm cpu    none ;;   # floor to aim at (#669: +162M)
            none)   arm none   HRX0 ;;   # the row in question, budget unset
            bud*)   arm "$a"   HRX0 "${a#bud}" ;;
        esac
    done
    echo "SPLIT ARMS DONE" | tee -a "$out"
    exit 0
fi

if [ "$(strings "$bindir/libggml-hrx.so.0.18.0" | grep -c 'refusing to run')" -lt 1 ] ||
   [ "$(strings "$bindir/libggml-hrx.so.0.18.0" | grep -c 'GGML_HRX_169_DECLINE')" -ne 0 ] ||
   ! strings "$bindir/libggml-hrx.so.0.18.0" | grep -q 'hrx host weights: refilled='; then
    echo "ABORT: $bindir is not the pinned build with the residency trace" >&2
    exit 2
fi
: >> "$out"   # append: a second ticket must not erase the first ticket's arms
printf '# gtt_before_hold=%sM  avail_before=%sM\n' "$(( $(cat $G) / 1048576 ))" \
    "$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)" >> "$out"
IN_HOLD=1 exec ~/.local/bin/gpu-hold -l "i669-split-arms" -t 20 -- env IN_HOLD=1 ARMS="${ARMS:-cpu none bud128 bud32}" bash "$0"
