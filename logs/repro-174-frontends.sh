#!/usr/bin/env bash
# pi-i669: PR #174 CHANGES - the reviewer ran llama-cli and saw 0 guard lines + a load-time arena refusal. My table's
# numbers came from llama-server arms. Test both front-ends at the PR head, same library, same model, same budgets.
set -u
BIN=/home/bcloud/wt/i674-repro/build-priv/bin
THROCK=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib
export HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1
export IREE_HAL_AMDGPU_LIBHSA_PATH=$THROCK/libhsa-runtime64.so.1
export LD_LIBRARY_PATH=$BIN:$THROCK
M4=/home/bcloud/models/Qwen3-4B-Q4_K_M.gguf

echo "### identity"
printf '  build line: %s\n' "$($BIN/llama-cli --version 2>/dev/null | head -1)"
printf '  libggml-hrx: %s  guard=%s\n' "$(sha256sum $BIN/libggml-hrx.so.0.18.0 | cut -c1-16)" \
    "$(strings $BIN/libggml-hrx.so.0.18.0 | grep -c 'hrx placement: %s is not run on HRX')"

port() { python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])'; }

report() { # $1 tag $2 log
    local tag=$1 log=$2
    printf '%-34s guard=%-2s suppress=%-2s arena=%-2s\n' "$tag" \
        "$(grep -ac 'hrx placement: .* is not run on HRX' "$log")" \
        "$(grep -ac 'further declines are not logged' "$log")" \
        "$(grep -ac 'arena_reserve:' "$log")"
    grep -a 'hrx placement:' "$log" | head -2 | sed 's/^/     | /'
}

srv_arm() { # $1 tag $2 budget
    local tag=$1 b=$2 log=/tmp/p174-srv-$1.log p
    if [ -n "$b" ]; then export GGML_HRX_HOST_WEIGHT_LIMIT_MB=$b; else unset GGML_HRX_HOST_WEIGHT_LIMIT_MB; fi
    p=$(port)
    $BIN/llama-server -m $M4 -dev HRX0 -fit off -ngl 0 -c 256 --host 127.0.0.1 --port $p --no-webui >"$log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 300); do curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1 && break; kill -0 $pid 2>/dev/null || break; sleep 0.2; done
    curl -s -m 120 "http://127.0.0.1:$p/completion" -H 'content-type: application/json' \
        -d '{"prompt":"What is the capital of France?","n_predict":8,"temperature":0,"cache_prompt":false}' \
        | head -c 300 >>"$log"
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    echo >>"$log"
    report "server budget=${b:-unset}" "$log"
    printf '     text: %s\n' "$(python3 - "$log" <<'PY'
import json,sys
t=open(sys.argv[1],encoding='utf-8',errors='replace').read()
for l in t.splitlines():
    l=l.strip()
    if l.startswith('{"'):
        try:
            d=json.loads(l.split('\n')[0])
            if 'content' in d: print(repr(d['content'][:60])); break
            if 'error' in d: print('ERROR', d['error'].get('message','')[:80]); break
        except Exception: pass
else: print('<no body>')
PY
)"
    echo
}

cli_arm() { # $1 tag $2 budget
    local tag=$1 b=$2 log=/tmp/p174-cli-$1.log
    if [ -n "$b" ]; then export GGML_HRX_HOST_WEIGHT_LIMIT_MB=$b; else unset GGML_HRX_HOST_WEIGHT_LIMIT_MB; fi
    $BIN/llama-cli -m $M4 -dev HRX0 -fit off -ngl 0 -c 256 --temp 0 -no-cnv --no-warmup --no-perf -n 8 \
        -p "What is the capital of France?" >"$log" 2>&1
    report "cli    budget=${b:-unset}" "$log"
    grep -a 'refusing to run' "$log" | head -1 | cut -c1-140 | sed 's/^/     ! /'
    echo
}

cli_arm unset ""
cli_arm 128 128
srv_arm unset ""
srv_arm 128 128
srv_arm 8 8
for f in /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input; do [ -r "$f" ] && echo "die after: $(($(<"$f")/1000))C"; done | head -1
