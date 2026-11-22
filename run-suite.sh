#!/usr/bin/env bash
# The owed hrx-ops-test run at pin 42b544465 (engine#2774 / engine#2870).
# Runs INSIDE a gpu-hold ticket.
#
# Lesson from the first attempt (2026-11-20 13:20Z): `--hrx-ops-test-test-list` prints the case list
# without touching the device, so "290 cases listed" was NOT initialisation -- the run then skipped all
# 288 device cases and still printed "2 passed, 0 failed ... rc 0". The criterion here is `--list-devices`
# naming HRX0, and the counts come from the summary line with skips read separately (#2912).
#
# libhrx resolves its HSA runtime through IREE_HAL_AMDGPU_LIBHSA_PATH (the engine maps
# ONEBIT_HRX_LIBHSA onto it); LD_LIBRARY_PATH alone gave either a symbol error or the
# HSA_AMD_AGENT_INFO_PM4_EMULATION abort.
set -u
cd "$(dirname "$0")"
BIN=./build-run/bin/test-hrx-ops

for hsa in /opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1 \
           /opt/1bit/libhsa/libhsa-runtime64.so.1 \
           /home/bcloud/amd-oss/toolchains/hrx-stock-b142-install/lib/libhsa-runtime64.so.1; do
    [ -f "$hsa" ] || continue
    out=$(IREE_HAL_AMDGPU_LIBHSA_PATH="$hsa" timeout 120 "$BIN" --list-devices 2>&1)
    if printf '%s' "$out" | grep -q "HRX0"; then
        echo "libhsa: $hsa -- HRX0 present"
        echo "=== running the suite (jobs 1) ==="
        # 2 workers, because the whole suite at jobs 1 (290 cases) ran ~842 s and was cut by the 14-minute
        # hold cap at case ~250 -- a capped run has no summary line and no counts (#2965 territory).
        LOG=/tmp/i679-suite-$(date -u +%H%M%S).log
        echo "log: $LOG"
        IREE_HAL_AMDGPU_LIBHSA_PATH="$hsa" timeout 800 "$BIN" --hrx-ops-test-test-jobs 2 > "$LOG" 2>&1
        echo "suite rc=$?"
        grep -E "^test skipped:|^test-hrx-ops summary:" "$LOG"
        # a summary that counts only the device-free cases is NOT a run of the suite (#2912)
        grep -qE "test skipped: no device available" "$LOG" && echo "VERDICT: device absent, counts are not a suite result"
        exit 0
    fi
    echo "libhsa: $hsa -- no HRX0 ($(printf '%s' "$out" | tail -1 | cut -c1-80))"
done
echo "NO libhsa candidate gave a device -- suite not run"
exit 3
