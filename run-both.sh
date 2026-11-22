#!/usr/bin/env bash
# engine#2774, ORCH's ordered step (1): run BOTH suites at the engine pin
# (llama.cpp 84ab5dae8, engine main 6381982) under ONE gpu-hold ticket, and print
# pass/fail/abort per suite.
#
# Rules this script keeps, each learned the hard way tonight:
#   * IREE_HAL_AMDGPU_LIBHSA_PATH is the lever that makes the HRX runtime initialise --
#     LD_LIBRARY_PATH alone gave either a symbol error or the PM4_EMULATION abort;
#   * a device is only real when `--list-devices` names HRX0 (`--list` enumerates cases
#     without touching the device);
#   * one log per run, stamped, so a killed run cannot be read as a live one;
#   * both suites in the same ticket: the second one costs no extra queue position.
set -u
cd "$(dirname "$0")"
TH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib
HSA=$TH/libhsa-runtime64.so.1
BIN=./build-run/bin

probe() { IREE_HAL_AMDGPU_LIBHSA_PATH=$HSA timeout 120 "$BIN/test-hrx-ops" --list-devices 2>&1; }

if ! probe | grep -q "HRX0"; then
    echo "no HRX0 from --list-devices, so neither suite can run"
    probe | tail -2
    exit 3
fi
echo "device: HRX0 present (libhsa $HSA)"

STAMP=$(date -u +%Y%m%dT%H%M%S)
run_suite() {  # <name> <binary> <extra args...>
    local name=$1 bin=$2; shift 2
    local log=$HOME/.cache/scratch-i679/i679-$name-$STAMP.log
    echo "=== $name: $bin $* (log $log)"
    IREE_HAL_AMDGPU_LIBHSA_PATH=$HSA timeout 900 "$BIN/$bin" "$@" > "$log" 2>&1
    echo "  rc=$?"
    grep -E "^test skipped:|summary:" "$log" | tail -2 | sed 's/^/  /'
    # the two suites word their counts differently; count rows either way
    printf '  rows: PASS=%s FAIL=%s ABORT=%s\n' \
        "$(grep -c '^\[PASS\]' "$log")" \
        "$(grep -c '^\[FAIL\]' "$log")" \
        "$(grep -c 'missing expected kernel' "$log")"
    echo "  log: $log"
}

set -o pipefail
run_suite test-hrx-ops test-hrx-ops --hrx-ops-test-test-jobs 2
run_suite hrx-ops-test hrx-ops-test --hrx-test-jobs 1
echo "=== both suites finished; logs in $HOME/.cache/scratch-i679/i679-*-$STAMP.log"
