#!/usr/bin/env bash
# engine#2452 / llama.cpp#391 regression arm: the fast greedy path's log line is the only way to know whether the
# optimisation was TAKEN. A run whose eligibility silently stops holding keeps producing correct tokens and loses the
# gain, so every output-based test still passes -- this row asserts the two conditions are distinguishable instead.
#
#   ONEBIT_FAST_GREEDY_MODEL=/path/to/small.gguf   unset or missing -> SKIP (77); CI leaves it unset.
#   BIN_DIR                                        a build's bin/ (llama-cli)
#
# Exit codes: 0 all arms hold; 1 an arm failed; 77 no model (skip).
set -uo pipefail
BIN_DIR=${1:?usage: test-fast-greedy-log.sh <bin dir>}
M=${ONEBIT_FAST_GREEDY_MODEL:-}
if [ -z "$M" ] || [ ! -f "$M" ]; then
    echo "test-fast-greedy-log: ONEBIT_FAST_GREEDY_MODEL unset or missing, SKIP"
    exit 77
fi
CLI="$BIN_DIR/llama-cli"
[ -x "$CLI" ] || { echo "test-fast-greedy-log: no llama-cli in $BIN_DIR"; exit 77; }

run() {  # <env assignments...> -- <cli args...>; prints the fast_greedy line
    env "$@" 2>&1 | grep -oE 'fast_greedy=[01] \(eligible=[01], LLAMA_NO_FAST_GREEDY=(set|unset)\)' | head -1
}
fail=0
chk() { if [ "$2" = "$3" ]; then echo "  ok   $1"; else echo "  FAIL $1: want '$3' got '$2'"; fail=1; fi; }

# 1. greedy: the path is taken
chk "a greedy run takes the fast path" \
    "$(run LLAMA_LOG_FAST_GREEDY=1 "$CLI" -m "$M" -ngl 0 -p hi -n 2 --temp 0)" \
    "fast_greedy=1 (eligible=1, LLAMA_NO_FAST_GREEDY=unset)"
# 2. not greedy: not eligible
chk "a sampled run is not eligible" \
    "$(run LLAMA_LOG_FAST_GREEDY=1 "$CLI" -m "$M" -ngl 0 -p hi -n 2 --temp 0.7)" \
    "fast_greedy=0 (eligible=0, LLAMA_NO_FAST_GREEDY=unset)"
# 3. the override: eligible but NOT taken -- the arm a single boolean could not express
chk "the override is distinguishable from ineligibility" \
    "$(run LLAMA_NO_FAST_GREEDY=1 LLAMA_LOG_FAST_GREEDY=1 "$CLI" -m "$M" -ngl 0 -p hi -n 2 --temp 0)" \
    "fast_greedy=0 (eligible=1, LLAMA_NO_FAST_GREEDY=set)"

echo "test-fast-greedy-log: $([ $fail = 0 ] && echo OK || echo FAILED)"
exit $fail
