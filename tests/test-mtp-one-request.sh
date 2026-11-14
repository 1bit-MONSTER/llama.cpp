#!/usr/bin/env bash
# engine#2433 regression arm: one request through llama-server with MTP drafting (--spec-type draft-mtp).
# CPU only (-ngl 0), so it needs no GPU. Needs a GGUF that carries its own MTP head:
#   ONEBIT_MTP_TEST_MODEL=/path/to/model.gguf  (e.g. Qwen3.8-27B UD-Q4_K_XL); unset or missing -> skip (77).
# Pass: HTTP 200 from /completion with non-empty content, the server still alive afterwards, and the server log
# reporting a draft acceptance line (so the MTP drafter really ran). Fail: anything else, including a crash.
set -u
BIN_DIR=${1:?usage: test-mtp-one-request.sh <bin dir>}
M=${ONEBIT_MTP_TEST_MODEL:-}
if [ -z "$M" ] || [ ! -f "$M" ]; then
    echo "test-mtp-one-request: ONEBIT_MTP_TEST_MODEL unset or missing, SKIP"
    exit 77
fi
P=${ONEBIT_MTP_TEST_PORT:-$((20000 + RANDOM % 20000))}
LOG=$(mktemp -p "${TMPDIR:-$BIN_DIR}" mtp-one-request.XXXXXX.log)
"$BIN_DIR/llama-server" -m "$M" -ngl 0 -c 2048 -np 1 --host 127.0.0.1 --port "$P" \
    --spec-type draft-mtp --spec-draft-n-max 3 > "$LOG" 2>&1 &
SRV=$!
cleanup() { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; }
trap cleanup EXIT
code=000
for _ in $(seq 1 200); do
    code=$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$P/health" || true)
    [ "$code" = 200 ] && break
    kill -0 "$SRV" 2>/dev/null || break
    sleep 3
done
if [ "$code" != 200 ]; then echo "test-mtp-one-request: server not healthy (last $code), FAIL"; tail -20 "$LOG"; exit 1; fi
out=$(curl -s -m 600 -w '\n%{http_code}' "http://127.0.0.1:$P/completion" \
    -d '{"prompt":"The capital of France is","n_predict":16,"temperature":0}')
http=${out##*$'\n'}; body=${out%$'\n'*}
sleep 1
alive=yes; kill -0 "$SRV" 2>/dev/null || alive=no
content=$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("content",""))' 2>/dev/null || true)
acc=$(grep -m1 "draft acceptance" "$LOG" || true)
echo "test-mtp-one-request: http=$http alive=$alive content=$(printf '%q' "${content:0:60}")"
echo "  ${acc:-no draft acceptance line}"
if [ "$http" = 200 ] && [ -n "$content" ] && [ "$alive" = yes ] && [ -n "$acc" ]; then
    echo "test-mtp-one-request: OK"; rm -f "$LOG"; exit 0
fi
echo "test-mtp-one-request: FAIL (log $LOG)"; grep -E "SIGSEGV|signal|error" "$LOG" | tail -5
exit 1
