#!/usr/bin/env bash
# #179 test: does the q/k dump capture EVERY ubatch, or only the ones that rebuild the graph?
set -u
BIN=/home/bcloud/wt/tprs-179/build-179/bin/llama-cli
MODEL=/home/bcloud/models/Qwen3-0.6B-Q4_K_M.gguf
OUT=/tmp/zha179
rm -rf "$OUT"; mkdir -p "$OUT"
python3 - "$OUT/prompt.txt" <<'EOF'
import sys
words = ("the tidal record shows a slow oscillation in the basin whose amplitude depends on the "
         "geometry of the coast and the phase of the moon ").split()
open(sys.argv[1], "w").write(" ".join(words * 60))
EOF
export LLAMA_ZHA_DUMP_DIR="$OUT/dump"
mkdir -p "$LLAMA_ZHA_DUMP_DIR"
"$BIN" -m "$MODEL" -f "$OUT/prompt.txt" -c 2048 -n 1 -b 512 -ub 512 \
    > "$OUT/run.log" 2> "$OUT/run.err"
echo "rc=$?"
echo "--- prompt tokens / ubatches the run says it did"
grep -iE 'n_tokens|ubatch|prompt' "$OUT/run.err" | head -5
echo "--- dump manifest"
wc -l < "$LLAMA_ZHA_DUMP_DIR/ubatches.jsonl" 2>/dev/null | sed 's/^/ubatch lines: /'
head -3 "$LLAMA_ZHA_DUMP_DIR/ubatches.jsonl" 2>/dev/null
echo "--- tokens captured"
[ -f "$LLAMA_ZHA_DUMP_DIR/tokens.i32" ] && echo "tokens.i32 bytes: $(stat -c %s "$LLAMA_ZHA_DUMP_DIR/tokens.i32") (= $(( $(stat -c %s "$LLAMA_ZHA_DUMP_DIR/tokens.i32") / 4 )) tokens)"
echo "--- rows per dumped tensor (first 4 files) and file count"
ls "$LLAMA_ZHA_DUMP_DIR" | grep -c '\.f16$' | sed 's/^/f16 files: /'
python3 - "$LLAMA_ZHA_DUMP_DIR" <<'PY'
import json, os, sys
d = sys.argv[1]
for line in open(os.path.join(d, "tensors.jsonl")):
    t = json.loads(line)
    ne = t["ne"]
    path = os.path.join(d, t["key"] + ".f16")
    rows = os.path.getsize(path) // (2 * ne[0] * ne[1])
    print(f'{t["key"]}: ne={ne} rows={rows}')
PY
echo "--- warnings about skipped ubatches"
grep -c 'skipped' "$OUT/run.err" 2>/dev/null
echo "--- dump line from the log"
grep -E 'dumping q/k|tensors per ubatch' "$OUT/run.err" 2>/dev/null | head -3
