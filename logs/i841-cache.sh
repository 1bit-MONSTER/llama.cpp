#!/usr/bin/env bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# i841-cache.sh — engine #841: does GGML_HRX_GRAPH_PROGRAM_CACHE_MB bound the cache at all?
#
# Same reproducer shape as the issue (zaya1-8b, -ngl 99 -fit off -c 4096 -np 1, one request per new prompt shape,
# cache_prompt=false), plus the cheap way to make the trim line print on a short run: the COUNT bound at 2, so
# `plan_evictions` runs immediately and the line states the bytes the budget is looking at.
#
#   ARM=default   both bounds at their defaults (64 programs, 8192 MiB)
#   ARM=count2    GGML_HRX_GRAPH_PROGRAM_CACHE=2: forces the trim line, which prints the accounted bytes
#   ARM=cachemb   GGML_HRX_GRAPH_PROGRAM_CACHE_MB=256: the knob that is supposed to bite
#
# SRV=<bindir> selects the build (before/after), OUT=<summary> the output file. GTT is read box-wide, as the issue
# did, so every number below is a delta against the same process's own baseline and the ambient is printed once.
set -u
R=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core
SRV=${SRV:-$HOME/wt/i674-repro/stage-pin788-srv}
# DT_RUNPATH is consulted AFTER LD_LIBRARY_PATH, and every stage binary's RUNPATH is the absolute build-priv/bin,
# so without $SRV first the stage dir is a decoy and both arms load whatever build-priv currently holds.
export LD_LIBRARY_PATH=$SRV:$R/lib
M=${M:-$HOME/models/zaya1-8b-Q4_K_M.gguf}
GTT=/sys/class/drm/card0/device/mem_info_gtt_used
PORT=${PORT:-39831}
ARM=${ARM:-default}
OUT=${OUT:-$PWD/i841.summary}
MIB() { echo $(( $1 / 1048576 )); }

req() { # $1 = words
    python3 - "$PORT" "$1" <<'PY'
import json,sys,urllib.request
port,n=int(sys.argv[1]),int(sys.argv[2])
body=json.dumps({"prompt":" ".join(["word%d"%i for i in range(n)]),"n_predict":1,
                 "temperature":0,"cache_prompt":False}).encode()
r=json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion",body,
    {"Content-Type":"application/json"}),timeout=900))
t=r.get("timings",{}); print("    evaluated=%s tokens at %.1f tok/s"%(t.get("prompt_n"), t.get("prompt_per_second") or 0))
PY
}

run() {
    local base srv
    base=$(cat $GTT)
    case $ARM in
        default) local envs=() ;;
        count2)  local envs=(GGML_HRX_GRAPH_PROGRAM_CACHE=2) ;;
        cachemb) local envs=(GGML_HRX_GRAPH_PROGRAM_CACHE_MB=256) ;;
        cachemb2) local envs=(GGML_HRX_GRAPH_PROGRAM_CACHE_MB=2048) ;;
        *) echo "unknown ARM=$ARM" >&2; return 2 ;;
    esac
    env "${envs[@]}" HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1 \
        "$SRV/llama-server" -m "$M" -ngl 99 -fit off -c 4096 -np 1 --port $PORT \
        > /tmp/i841-$ARM-$(basename "$SRV").srv.log 2>&1 &
    srv=$!
    for _ in $(seq 1 240); do
        curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null && break
        kill -0 $srv 2>/dev/null || { echo "server died"; tail -5 /tmp/i841-$ARM-$(basename "$SRV").srv.log; return 1; }
        sleep 1
    done
    printf '=== ARM %s  env=[%s]  srv=%s\n' "$ARM" "${envs[*]:-defaults}" "$SRV"
    printf '    loaded lib: %s (accounting strings: retained=%s arenas=%s)\n' "$SRV/libggml-hrx.so.0.18.0" \
        "$(strings "$SRV/libggml-hrx.so.0.18.0" | grep -c 'MiB retained')" \
        "$(strings "$SRV/libggml-hrx.so.0.18.0" | grep -c 'MiB of arenas')"
    printf '    GTT at start %s MiB (box-wide)\n' "$(MIB "$base")"
    printf '    after load: +%s MiB\n' "$(( ($(cat $GTT)-base)/1048576 ))"
    for n in 64 128 256 384 512 768 1024; do
        local tps
        tps=$(req "$n" | sed -n 's/.* at \(.*\) tok\/s.*/\1/p')
        printf '    shape %-5s -> +%-6s MiB | trims %-2s | %s tok/s\n' "$n" "$(( ($(cat $GTT)-base)/1048576 ))" \
            "$(grep -ci 'graph program cache trimmed' /tmp/i841-$ARM-$(basename "$SRV").srv.log)" "${tps:-n/a}"
    done
    echo "    --- trim lines:"
    grep -i 'graph program cache trimmed' /tmp/i841-$ARM-$(basename "$SRV").srv.log | tail -3 | sed 's/^/      /'
    echo "    --- budget/accounting lines:"
    grep -iE 'program cache (budget|trimmed)' /tmp/i841-$ARM-$(basename "$SRV").srv.log | tail -2 | sed 's/^/      /'
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; sleep 2
    printf '    after exit: +%s MiB\n' "$(( ($(cat $GTT)-base)/1048576 ))"
    echo "    peak peak_gtt=$(MIB "$(cat $GTT)") MiB"
}

run 2>&1 | tee -a "$OUT"
