#!/usr/bin/env bash
# engine#2433 regression arm: one request through llama-server with MTP drafting (--spec-type draft-mtp).
# CPU only (-ngl 0), so it needs no GPU. Needs a GGUF that carries its own MTP head:
#   ONEBIT_MTP_TEST_MODEL=/path/to/model.gguf  (e.g. Qwen3.8-27B UD-Q4_K_XL); unset or missing -> SKIP (77).
# CI does not set ONEBIT_MTP_TEST_MODEL, so this row reports SKIP there; it runs on a box that has the GGUF.
# Pass: HTTP 200 from /completion with non-empty content, the server still alive afterwards, and the server log
# reporting a draft acceptance line (so the MTP drafter really ran). Distinct exit codes per failure:
#   1 server never became healthy   2 the server died (e.g. the #2433 SIGSEGV)   3 the request timed out or HTTP != 200
#   4 empty content                 5 no draft acceptance line, or 0 drafts accepted (the drafter did not work)
# Worst case of this script: health 200 x (curl -m 2 + sleep 3) = 1000 s + the request (-m 600) = ~1600 s; the ctest
# TIMEOUT must stay above that or these exit codes never get to run.
set -u
BIN_DIR=${1:?usage: test-mtp-one-request.sh <bin dir>}
M=${ONEBIT_MTP_TEST_MODEL:-}
if [ -z "$M" ] || [ ! -f "$M" ]; then
    echo "test-mtp-one-request: ONEBIT_MTP_TEST_MODEL unset or missing, SKIP"
    exit 77
fi
LOG=$(mktemp -p "${TMPDIR:-$BIN_DIR}" mtp-one-request.XXXXXX.log)
SRV=
cleanup() { [ -n "$SRV" ] && { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; }; }
trap cleanup EXIT
free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'; }

# A port the OS reports free can still be taken before llama-server binds it, so a bind failure retries on a new one.
code=000
for attempt in 1 2 3; do
    P=${ONEBIT_MTP_TEST_PORT:-$(free_port)}
    "$BIN_DIR/llama-server" -m "$M" -ngl 0 -c 2048 -np 1 --host 127.0.0.1 --port "$P" \
        --spec-type draft-mtp --spec-draft-n-max 3 > "$LOG" 2>&1 &
    SRV=$!
    for _ in $(seq 1 200); do
        code=$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$P/health" || true)
        [ "$code" = 200 ] && break
        kill -0 "$SRV" 2>/dev/null || break
        sleep 3
    done
    [ "$code" = 200 ] && break
    if grep -qiE "address already in use|couldn't bind|failed to bind" "$LOG" && [ -z "${ONEBIT_MTP_TEST_PORT:-}" ]; then
        echo "test-mtp-one-request: port $P taken, retrying ($attempt)"; cleanup; SRV=; continue
    fi
    break
done
if [ "$code" != 200 ]; then
    if [ -n "$SRV" ] && ! kill -0 "$SRV" 2>/dev/null; then
        echo "test-mtp-one-request: server DIED before it was healthy, FAIL"; tail -20 "$LOG"; exit 2
    fi
    echo "test-mtp-one-request: server never healthy (last $code), FAIL"; tail -20 "$LOG"; exit 1
fi
out=$(curl -s -m 600 -w '\n%{http_code}' "http://127.0.0.1:$P/completion" \
    -d '{"prompt":"The capital of France is","n_predict":16,"temperature":0}')
http=${out##*$'\n'}; body=${out%$'\n'*}
sleep 1
alive=yes; kill -0 "$SRV" 2>/dev/null || alive=no
content=$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("content",""))' 2>/dev/null || true)
acc=$(grep -m1 "draft acceptance" "$LOG" || true)
# "draft acceptance = 0.64286 (    9 accepted /    14 generated)": the accepted count is what says the drafter worked
acc_n=$(printf '%s' "$acc" | sed -nE 's/.*\( *([0-9]+) accepted.*/\1/p')
echo "test-mtp-one-request: http=$http alive=$alive content=$(printf '%q' "${content:0:60}")"
echo "  ${acc:-no draft acceptance line}"
if [ "$alive" != yes ]; then
    echo "test-mtp-one-request: the server DIED during the request, FAIL (log $LOG)"; grep -E "SIGSEGV|signal|error" "$LOG" | tail -5; exit 2
fi
if [ "$http" != 200 ]; then echo "test-mtp-one-request: request failed or timed out (http=$http), FAIL (log $LOG)"; exit 3; fi
if [ -z "$content" ]; then echo "test-mtp-one-request: empty content, FAIL (log $LOG)"; exit 4; fi
if [ -z "$acc" ]; then echo "test-mtp-one-request: no draft acceptance line, the MTP drafter did not run, FAIL (log $LOG)"; exit 5; fi
if [ -z "$acc_n" ] || [ "$acc_n" -lt 1 ]; then echo "test-mtp-one-request: 0 drafts accepted (${acc_n:-unparsed}), the MTP drafter did not work, FAIL (log $LOG)"; exit 5; fi
echo "test-mtp-one-request: OK"; rm -f "$LOG"; exit 0
