#!/usr/bin/env bash
# Build the #674 reproducer at the rebased tip once the box is genuinely cool, at -j2, retrying a thermal ABORT.
# Two attempts were killed at 93-94 C while other lanes' GPU work held the die hot (12:14, 12:17), so this waits
# for a sustained cool window instead of racing thermal-run's freeze/thaw.
set -u
T=$(grep -l k10temp /sys/class/hwmon/*/name | head -n 1 | xargs dirname)/temp1_input
export HIP_DEVICE_LIB_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/llvm/amdgcn/bitcode
cd ~/wt/i674-repro
cool() { echo $(( $(cat "$T") / 1000 )); }

for attempt in $(seq 10); do
  # a cool window is 72 C or less, held for a full minute
  for _ in $(seq 180); do
    if [ "$(cool)" -le 74 ] && sleep 15 && [ "$(cool)" -le 76 ]; then break; fi
    echo "waiting for cool window: $(cool)C $(date +%T)"
    sleep 20
  done
  echo "attempt $attempt at $(cool)C $(date +%T)"
  ~/lb/thermal-run.sh ninja -C build-priv -j1 llama-cli
  rc=$?
  echo "attempt $attempt rc=$rc $(date +%T)"
  [ "$rc" -eq 0 ] && { echo "BUILD OK $(date +%T)"; exit 0; }
  sleep 30
done
echo "BUILD GAVE UP $(date +%T)"
exit 1
