#!/usr/bin/env bash
# pi-hrx-sched, overnight #674 step 3: the content at one node — the next step the issue thread named and nobody ran.
#
# The thread's last comment left exactly this open: "dump the layer-0 MUL's two inputs and its output (first 8 floats +
# checksum) for one prefill token on -dev none, MUL_MAT alone (correct) and MUL_MAT,MUL (corrupt), and diff". Generalised
# to the whole graph, because with no --tensor-filter the fork's llama-debug prints every tensor's dims and first values,
# so the diff names the FIRST value that diverges instead of another mechanism excluded.
#
# Arms (knob present only in this worktree — asserted below):
#   cpu     : -dev none                              (reference)
#   mul     : decline MUL alone                      (control: thread measured this correct)
#   gemm    : decline MUL_MAT alone                  (control: thread measured this correct)
#   pair    : decline MUL_MAT,MUL                    (the corrupting arm)
set -u
W=/home/bcloud/wt/i674-tip
BIN=${BIN:-$W/build-tip/bin/llama-debug}
LIBS=${LIBS:-$W/build-tip/bin}
M=${MODEL:-/home/bcloud/models/Qwen3-0.6B-Q4_K_M.gguf}
THROCK=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib
OUT=${OUT:-$W/logs}
mkdir -p "$OUT" || exit 1
cd /tmp || exit 1
export HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1
export IREE_HAL_AMDGPU_LIBHSA_PATH=$THROCK/libhsa-runtime64.so.1
export LD_LIBRARY_PATH=$LIBS:$THROCK${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
die() { local f; for f in /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input; do [ -r "$f" ] && echo "$(($(<"$f")/1000))C"; done | head -1; }

echo "=== provenance"
git -C "$W" log --oneline -1
echo "  169_DECLINE strings: $(strings "$LIBS/libggml-hrx.so" | grep -c GGML_HRX_169_DECLINE)   (must be 1: this arm's control variable)"
"$BIN" --list-devices 2>&1 | grep -E 'HRX0|none' | sed 's/^/  /' || true

# Everything after the last tensor print is chat noise; timings are non-deterministic and must not be diffed.
clean() { grep -E 'common_debug_cb_eval:|^\s+\[|^\s+-?[0-9]' "$1" | sed 's/[[:space:]]*$//'; }

arm() { # $1 tag, $2 DECLINE value or "", rest = cli args
  local tag=$1 decline=$2; shift 2
  local log="$OUT/dump-$tag.log"
  echo "=== ARM $tag  decline='${decline}'  args: $*  die $(die)"
  env ${decline:+GGML_HRX_169_DECLINE=$decline} "$BIN" -m "$M" -fit off -c 256 --temp 0 -n 1 \
      -p "What is the capital of France?" "$@" >"$log" 2>&1
  echo "  rc=$?  tensor lines=$(grep -c 'common_debug_cb_eval' "$log")"
}

arm cpu  ""          -dev none
arm mul  MUL         -dev HRX0 -ngl 0
arm gemm MUL_MAT     -dev HRX0 -ngl 0
arm pair MUL_MAT,MUL -dev HRX0 -ngl 0
echo "=== die after: $(die)"

for a in mul gemm pair; do
  echo "=== first divergence: cpu vs $a"
  diff <(clean "$OUT/dump-cpu.log") <(clean "$OUT/dump-$a.log") > "$OUT/diff-cpu-$a.txt" 2>&1
  if [ ! -s "$OUT/diff-cpu-$a.txt" ]; then echo "  identical (no divergent value)"; else
    head -12 "$OUT/diff-cpu-$a.txt"
    echo "  ... last tensor printed by the $a arm:"
    grep 'common_debug_cb_eval' "$OUT/dump-$a.log" | tail -1
  fi
done
