#!/usr/bin/env bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# i674-budget-arms.sh
#
# Engine #674 (orchestrator 12:10 AST): reconcile `Qwen3-4B -ngl 0` +4121M GTT (pin 1d32c22609d6, budget unset)
# against 0 MiB delta + zero `host weight` lines (pin d8bd0ff6052, budget 4096). Same model, same flags, budget
# unset AND set, GTT delta and `host weight` line count for each.
#
# Three arms inside ONE gpu-hold ticket, so all three see the same ambient GTT baseline:
#   cpu       -dev none                       floor to aim at (#669: +162M)
#   none      -dev HRX0 -ngl 0, budget unset  the row in question
#   bud4096   -dev HRX0 -ngl 0, budget 4096   the arm that reported 0 MiB / zero lines
#
# The residency trace is GGML_LOG_INFO behind GGML_HRX_HOST_WEIGHT_TRACE and the CLI sink drops INFO without -v,
# so every arm sets the variable AND -v: a zero line count is then evidence, not an unasked question.
set -u
bindir=${BINDIR:-$HOME/wt/i674-repro/stage-pin788}   # llama-cli at pin d8bd0ff6052
model=$HOME/models/Qwen3-4B-Q4_K_M.gguf
logs=$HOME/wt/i674-repro/logs
out=${OUT:-$logs/budget-arms.summary}
samples=$HOME/wt/dsv41-testdata/pools-sample.sh
watchdog=$HOME/wt/dsv41-testdata/mem-watchdog.sh
G=/sys/class/drm/card0/device/mem_info_gtt_used

arm() { # arm NAME DEV BUDGET [EXTRA_ENV]
    local name=$1 dev=$2 budget=${3:-} extra=${4:-} log=$logs/budget-arm-$1.log trace=$logs/budget-arm-$1.pools std=$logs/budget-arm-$1.out
    env -u HSA_ENABLE_SDMA HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1 \
        GGML_HRX_POOL_LIMIT_MB=97280 GGML_HRX_HOST_WEIGHT_TRACE=1 \
        ${budget:+GGML_HRX_HOST_WEIGHT_LIMIT_MB=$budget} ${extra:+$extra} \
        IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1 \
        LD_LIBRARY_PATH=$bindir \
        "$watchdog" 4 "$samples" "$trace" \
        "$bindir/llama-cli" -m "$model" -dev "$dev" -fit off -ngl 0 -c 256 -v \
            --temp 0 -no-cnv -n 8 -p "What is the capital of France?" > "$std" 2> "$log"
    local rc=$? start peak end floor lines resident
    # pools-sample pads the number with %6d, so `gtt=  31518M` is NOT two fields reliably - parse the line, not
    # the columns. (The first version split on whitespace and printed +0M for every arm.)
    gttcol() { sed -n 's/^ *[0-9][0-9]* gtt= *\([0-9][0-9]*\)M.*/\1/p' "$trace" |
               { if [ "$1" = first ]; then head -1; elif [ "$1" = last ]; then tail -1; else sort -n | tail -1; fi; }; }
    start=$(gttcol first); peak=$(gttcol max); end=$(gttcol last)
    floor=$(sed -n 's/.*avail= *\([0-9][0-9]*\)M.*/\1/p' "$trace" | sort -n | head -1)
    # the per-execution residency line this issue's thread counts
    lines=$(grep -ac 'hrx host weights: refilled=' "$log" || true)
    resident=$(grep -a 'hrx host weights: refilled=' "$log" | tail -1 | sed -n 's/.*resident=\([0-9.]*\) MiB.*/\1/p')
    # the model's answer, anchored on the stats line llama-cli prints after it - never `tail`, which returned the
    # watchdog's own stderr line and cost this lane a re-read of three summaries (16:33, 15:20, and 12:12's +0M parse)
    answer=$(tr -d '\r' < "$std" | awk '/^\[ Prompt:/{exit} !/^mem-watchdog:/ && NF' | tail -2 | tr '\n' ' ' | cut -c1-100)
    {
        printf '=== %-8s dev=%-5s budget=%-6s %-28s rc=%d\n' "$name" "$dev" "${budget:-unset}" "${extra:-}" "$rc"
        printf '    gtt %sM -> peak %sM -> end %sM   peak_delta=+%sM  end_delta=+%sM\n' \
            "$start" "$peak" "$end" "$((peak - start))" "$((end - start))"
        printf '    MemAvailable floor=%sM   residency_lines=%s   last resident=%s MiB\n' \
            "$floor" "$lines" "${resident:-none}"
        printf '    answer: %s\n' "$answer"
    } | tee -a "$out"
}

if [ "${IN_HOLD:-}" = "1" ]; then
    printf '# budget-arms %s bindir=%s\n' "$(date '+%F %T %Z')" "$bindir" >> "$out"
    arm cpu     none
    arm none    HRX0
    arm uni     HRX0 "" GGML_HRX_USE_UNIFIED_MEMORY=1
    echo "BUDGET ARMS DONE" | tee -a "$out"
    exit 0
fi

# pre-flight: an arm is only meaningful on the pinned build, so assert it before measuring anything
if [ "$(strings "$bindir/libggml-hrx.so.0.18.0" | grep -c 'refusing to run')" -lt 1 ] ||
   [ "$(strings "$bindir/libggml-hrx.so.0.18.0" | grep -c 'GGML_HRX_169_DECLINE')" -ne 0 ] ||
   ! strings "$bindir/libggml-hrx.so.0.18.0" | grep -q 'hrx host weights: refilled='; then
    echo "ABORT: $bindir is not the pinned build with the residency trace" >&2
    exit 2
fi
: > "$out"
printf '# gtt_before_hold=%sM\n' "$(( $(cat $G) / 1048576 ))" >> "$out"
IN_HOLD=1 exec ~/.local/bin/gpu-hold -l "i669-budget-arms" -t 20 -- env IN_HOLD=1 bash "$0"
