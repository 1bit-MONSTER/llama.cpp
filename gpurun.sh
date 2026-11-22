#!/usr/bin/env bash
# review-450 device run: before/after test-hrx-fa-streams x3 each, then test-backend-ops -o FLASH_ATTN_EXT (after).
set -u
W=$HOME/wt/review-450
OUT=$W/gpurun
mkdir -p $OUT
exec 9> ~/.cache/lax-decode/box.lock
flock -w 900 9 || { echo "box lock busy"; exit 75; }
export IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1
export GGML_HRX_JIT_CACHE_DIR=$W/jit-cache
mes() { sudo -n dmesg 2>/dev/null | grep -cE 'MES.*(fail|failed|failure)'; }
temp() { echo $(( $(cat $(grep -l k10temp /sys/class/hwmon/*/name | head -n1 | xargs dirname)/temp1_input) / 1000 )); }
M0=$(mes); echo "power_mode=$(cat /sys/class/ec_su_axb35/apu/power_mode) Tctl=$(temp)C MES_before=$M0"
check() { local m=$(mes); echo "  MES now=$m Tctl=$(temp)C"; if [ "$m" -gt "$M0" ]; then echo "MES increase: stop"; exit 3; fi; if [ "$(temp)" -ge 93 ]; then echo "Tctl>=93 stop"; exit 4; fi; }
for arm in before after; do
  for r in 1 2 3; do
    echo "== $arm run $r"
    if [ $arm = before ]; then LP=$W/bin-before; BIN=$W/bin-before/test-hrx-fa-streams; else LP=$W/build-hrx/bin; BIN=$W/build-hrx/bin/test-hrx-fa-streams; fi
    LD_LIBRARY_PATH=$LP timeout 240 $BIN > $OUT/$arm-$r.log 2>&1; echo "  rc=$?"
    grep -E "^(ok|FAIL)|^PASS|^FAIL" $OUT/$arm-$r.log | sed 's/^/  /'
    check
  done
done
echo "== after, dispatch log"
LD_LIBRARY_PATH=$W/build-hrx/bin GGML_HRX_LOG_DISPATCH=1 timeout 240 $W/build-hrx/bin/test-hrx-fa-streams > $OUT/after-dispatch.log 2>&1; echo "  rc=$?"
grep -o "onebit.flash_attention_streams_f32_f16_wmma" $OUT/after-dispatch.log | wc -l | sed 's/^/  streams matcher mentions: /'
check
echo "== test-backend-ops -o FLASH_ATTN_EXT -b HRX0 (after)"
LD_LIBRARY_PATH=$W/build-hrx/bin timeout 420 $W/build-hrx/bin/test-backend-ops -o FLASH_ATTN_EXT -b HRX0 > $OUT/tbo-after.log 2>&1; echo "  rc=$?"
tail -4 $OUT/tbo-after.log | sed 's/^/  /'
grep -c "FAIL" $OUT/tbo-after.log | sed 's/^/  FAIL lines: /'
check
echo done
