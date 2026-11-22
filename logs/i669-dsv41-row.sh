#!/usr/bin/env bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# i669-dsv41-row.sh — engine #669 (brief ~/briefs/TISSUES-669.md, work item 4), the 217 GiB file-level row.
#
#   arm cpu      -dev none   -ngl 0   CPU floor (the 4B +162M analogue)
#   arm hrx0     -dev HRX0   -ngl 0   the issue's own configuration, no budget: the 107 GiB ramp of 30 Oct
#   arm hrx0bud8 -dev HRX0   -ngl 0   + GGML_HRX_HOST_WEIGHT_LIMIT_MB=8192: the GGUF census says 8192 MiB declines
#                                     exactly one tensor, per_layer_token_embd.weight (103.00 GiB Engram), and the
#                                     30 Oct probe says that table WAS the whole 107 GiB ramp - so this is the only
#                                     arm that can hold a 100 GiB floor
#   arm hrx8     -dev HRX0   -ngl 8   one ladder run, so the dsv41 lane can continue from it
#
# GATE, and it is the point of this script: the row's acceptance is a MemAvailable floor of >= 100 GiB, so a run
# that starts below 100 GiB cannot meet it and is not taken; and the window is the orchestrator's to give, not
# mine to assume (BOX-REQUEST-669-READY in ~/briefs/TISSUES-669-STATE.md), so it refuses without the marker file.
# Both refusals are cheap to see: they print the number they measured.
#
# Every arm runs under gpu-hold (one ticket for all three, so they share an ambient baseline), mem-watchdog 4,
# GGML_HRX_POOL_LIMIT_MB=97280 (the lane's 95 GiB cap) and the #715 no-SVM trio.
set -u
FLOOR_GIB=${FLOOR_GIB:-100}
MARKER=${MARKER:-$HOME/briefs/WINDOW-669-OPEN}
bindir=${BINDIR:-$HOME/wt/i674-repro/stage-pin788}
model=$HOME/models/dsv41/UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf
logs=$HOME/wt/i674-repro/logs
out=${OUT:-$logs/dsv41-row.summary}
samples=$HOME/wt/dsv41-testdata/pools-sample.sh
watchdog=$HOME/wt/dsv41-testdata/mem-watchdog.sh
G=/sys/class/drm/card0/device/mem_info_gtt_used
avail() { awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo; }
# Who is holding the floor. GTT is pinned system memory on this APU, so it is not reclaimable and it is exactly
# what a granted window has to give back; naming the holders turns "MemAvailable too low" into a decision.
gtt_holders() {
    local d b
    for d in /proc/[0-9]*; do
        b=$(grep -h '^drm-memory-gtt:' $d/fdinfo/* 2>/dev/null | awk '{s+=$2} END{print s+0}')
        [ "${b:-0}" -gt 0 ] && printf '%s %s %s\n' "$b" "${d#/proc/}" "$(tr '\0' ' ' < $d/cmdline 2>/dev/null | cut -c1-56)"
    done 2>/dev/null | sort -rn | head -4
}

arm() { # arm NAME DEV NGL [extra env assignments...]
    local name=$1 dev=$2 ngl=$3 log=$logs/dsv41-$1.log trace=$logs/dsv41-$1.pools std=$logs/dsv41-$1.out
    shift 3
    env -u HSA_ENABLE_SDMA HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1 "$@" \
        GGML_HRX_POOL_LIMIT_MB=97280 GGML_HRX_HOST_WEIGHT_TRACE=1 \
        IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1 \
        LD_LIBRARY_PATH=$bindir \
        "$watchdog" 4 "$samples" "$trace" \
        "$bindir/llama-cli" -m "$model" -dev "$dev" -fit off -ngl "$ngl" -c 256 -v \
            --temp 0 -no-cnv -n 8 -p "What is the capital of France?" > "$std" 2> "$log"
    local rc=$? start peak floor lines resident answer
    gttcol() { sed -n 's/^ *[0-9][0-9]* gtt= *\([0-9][0-9]*\)M.*/\1/p' "$trace" |
               { if [ "$1" = first ]; then head -1; else sort -n | tail -1; fi; }; }
    start=$(gttcol first); peak=$(gttcol max)
    floor=$(sed -n 's/.*avail= *\([0-9][0-9]*\)M.*/\1/p' "$trace" | sort -n | head -1)
    lines=$(grep -ac 'hrx host weights: refilled=' "$log" || true)
    resident=$(grep -a 'hrx host weights: refilled=' "$log" | tail -1 | sed -n 's/.*resident=\([0-9.]*\) MiB.*/\1/p')
    # the model's answer, anchored on the stats line llama-cli prints after it - never `tail`, which returns
    # the watchdog's own stderr line (the defect that cost this lane a re-read of three summaries)
    answer=$(tr -d '\r' < "$std" | awk '/^\[ Prompt:/{exit} !/^mem-watchdog:/ && NF' | tail -2 | tr '\n' ' ' | cut -c1-100)
    printf '=== %-5s dev=%-5s ngl=%-2s  rc=%d\n    gtt %sM -> peak %sM  peak_delta=+%sM\n    MemAvailable floor=%sM (%.1f GiB)   residency_lines=%s  resident=%s MiB\n    answer: %s\n' \
        "$name" "$dev" "$ngl" "$rc" "$start" "$peak" "$((peak - start))" "$floor" \
        "$(echo "$floor" | awk '{print $1/1024}')" "$lines" "${resident:-none}" "$answer" | tee -a "$out"
}

if [ "${IN_HOLD:-}" = "1" ]; then
    printf '# dsv41-row %s bindir=%s gtt_before=%sM\n' "$(date '+%F %T %Z')" "$bindir" "$(( $(cat $G) / 1048576 ))" >> "$out"
    arm cpu      none 0
    arm hrx0     HRX0 0
    arm hrx0bud8 HRX0 0 GGML_HRX_HOST_WEIGHT_LIMIT_MB=8192
    arm hrx8     HRX0 8
    echo "DSV41 ROW DONE" | tee -a "$out"
    exit 0
fi

a=$(avail)
[ -f "$MARKER" ] || { echo "REFUSING: no window marker $MARKER — the box is not idle for me (BOX-REQUEST-669-READY)" >&2; exit 2; }
[ "$a" -ge "$FLOOR_GIB" ] || {
    echo "REFUSING: MemAvailable ${a} GiB < the ${FLOOR_GIB} GiB floor this row must hold — a run that starts below it cannot meet its own acceptance" >&2
    echo "  a granted window must return $(( FLOOR_GIB - a )) GiB; the GTT holders (pinned, not reclaimable):" >&2
    gtt_holders | while read -r kb pid cmd; do
        printf '    %6.2f GiB  pid %-7s %s\n' "$(awk -v k="$kb" 'BEGIN{print k/1048576}')" "$pid" "$cmd" >&2
    done
    exit 3
}
[ -s "$model" ] || { echo "REFUSING: $model missing" >&2; exit 4; }
if [ "$(strings "$bindir/libggml-hrx.so.0.18.0" | grep -c 'refusing to run')" -lt 1 ] ||
   ! strings "$bindir/libggml-hrx.so.0.18.0" | grep -q 'hrx host weights: refilled='; then
    echo "ABORT: $bindir is not the pinned build with the #173 refusal and the residency trace" >&2; exit 5
fi
: > "$out"
IN_HOLD=1 exec ~/.local/bin/gpu-hold -l "i669-dsv41-row" -t 40 -- env IN_HOLD=1 bash "$0"
