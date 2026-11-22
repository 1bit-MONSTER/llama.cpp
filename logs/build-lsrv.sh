#!/usr/bin/env bash
set -u
T=$(grep -l k10temp /sys/class/hwmon/*/name | head -n 1 | xargs dirname)/temp1_input
cool() { echo $(( $(cat "$T") / 1000 )); }
cd ~/wt/i674-repro
export HIP_DEVICE_LIB_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/llvm/amdgcn/bitcode
for attempt in 1 2 3 4; do
  for _ in $(seq 120); do
    [ "$(cool)" -le 78 ] && sleep 15 && [ "$(cool)" -le 80 ] && break
    echo "waiting for cool: $(cool)C $(date +%T)"; sleep 20
  done
  echo "attempt $attempt at $(cool)C $(date +%T)"
  ~/lb/thermal-run.sh ninja -C build-priv -j2 llama-server
  rc=$?
  echo "attempt $attempt rc=$rc"
  [ "$rc" -eq 0 ] && { echo "LSRV OK"; exit 0; }
  sleep 30
done
echo "LSRV GAVE UP"; exit 1
