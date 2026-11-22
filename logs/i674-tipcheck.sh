#!/usr/bin/env bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# i674-tipcheck.sh BINDIR   [ARM_SET=1|2]
#
# Engine #674, 0.6B, 8 tokens, every arm a gpu-hold ticket's worth of ~1 s of GPU.
#   set 1: cpu reference, control, the broken arm, and the broken arm fully serialized
#   set 2: staging bytes/flags per value, prepared-program cache off, and a 64-token context
set -u
bindir=${1:?usage: i674-tipcheck.sh BINDIR}
out=${OUT:-$HOME/wt/i674-repro/logs/tipcheck-set${ARM_SET:-1}}
model=$HOME/models/Qwen3-0.6B-Q4_K_M.gguf
GTT=/sys/class/drm/card0/device/mem_info_gtt_used
: > "$out.txt"

arm() { # arm NAME DECLINE "--device args" [extra env...]
    local name=$1 decline=$2 dev=$3
    shift 3
    local start_gtt end_gtt rc
    start_gtt=$(cat $GTT)
    env -u HSA_ENABLE_SDMA \
        HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1 \
        GGML_HRX_POOL_LIMIT_MB=${LIMIT:-97280} ${decline:+GGML_HRX_169_DECLINE=$decline} "$@" \
        IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1 \
        LD_LIBRARY_PATH=$bindir \
        $bindir/llama-cli -m "$model" $dev -fit off -ngl 0 -c ${CTX:-256} ${FA0:+-fa 0} --temp 0 -no-cnv -n 8 \
            -p "What is the capital of France?" > "$out-$name.log" 2>&1
    rc=$?
    end_gtt=$(cat $GTT)
    {
        printf '=== %s rc=%s gtt %sM -> %sM ctx=%s\n' "$name" "$rc" "$((start_gtt / 1048576))" \
            "$((end_gtt / 1048576))" "${CTX:-256}"
        grep -aE "hrx host staging: value |hrx host staging: [0-9]+ buffers" "$out-$name.log" | head -16
        awk '/^> What is the capital/{f=1;next} /\[ Prompt:/{f=0} f' "$out-$name.log" \
            | grep -av "hrx debug serial" | tr -d '\r' | sed 's/^/OUT| /'
    } | tee -a "$out.txt"
}

set -u
case ${ARM_SET:-1} in
1)
    arm cpu  ""    "-dev none"
    arm ctrl ""    "-dev HRX0"
    arm decl "all" "-dev HRX0"
    # Answered at the 12:41 run: byte-identical garbage, so #674 is deterministic content, not an ordering race.
    arm declser "all" "-dev HRX0" GGML_HRX_DEBUG_SERIAL_EXECUTION=1
    ;;
2)
    # What the staging round trip carries, per value (fork #165's instrument - merged, no rebuild needed).
    arm decltr "all" "-dev HRX0" GGML_HRX_HOST_STAGING_TRACE=1
    # Is a cached prepared program reused with another shape's staging set?
    arm declnc "all" "-dev HRX0" GGML_HRX_GRAPH_PROGRAM_CACHE=0
    # Does the corruption need a big KV? 64-token context, same one-token prompt.
    CTX=64 arm declc64 "all" "-dev HRX0"
    ;;
3)
    # Answered 12:52: the schedule dump is compiled out of a Release build (GGML_LOG_DEBUG), so no names there.
    # What is left is op selection. Attention off HRX, the fused dispatch off, and the placement census.
    arm declfa0  "all" "-dev HRX0" FA0=1
    arm declqwen "all" "-dev HRX0" GGML_HRX_DISABLE_QWEN_DISPATCH=1
    arm declrep  "all" "-dev HRX0" GGML_HRX_REPORT=1
    ;;
4)
    # Answered 13:01: -fa 0, fused dispatch off and the census arm are all the same garbage, so it is neither
    # attention nor a fused kernel. Minimal corrupting set: pairs were already correct on 30 Oct, so triples.
    arm t1 "MUL_MAT,GET_ROWS,RMS_NORM" "-dev HRX0"
    arm t2 "MUL_MAT,GET_ROWS,ROPE"     "-dev HRX0"
    arm t3 "MUL_MAT,RMS_NORM,ROPE"     "-dev HRX0"
    arm t4 "MUL_MAT,GET_ROWS,CONT"     "-dev HRX0"
    arm t5 "MUL_MAT,MUL,ADD"           "-dev HRX0"
    arm call "all"                     "-dev HRX0"
    ;;
5)
    # Answered 13:25: of the triples only {MUL_MAT,MUL,ADD} corrupts, and it reproduces the full-decline text
    # byte for byte. So the pairs are the next question, and they were never tested on 30 Oct.
    arm p_mulmat_mul   "MUL_MAT,MUL"   "-dev HRX0"
    arm p_mulmat_add   "MUL_MAT,ADD"   "-dev HRX0"
    arm p_mul_add      "MUL,ADD"       "-dev HRX0"
    arm p_all          "all"           "-dev HRX0"
    ;;
6)
    # Answered 13:31: {MUL_MAT,MUL} is the minimal corrupting pair (MUL_MAT,ADD and MUL,ADD are both correct),
    # and build_norm is RMS_NORM -> MUL(norm weight), so the suspected pattern is "RMS_NORM stays on HRX while its
    # consumer MUL and the GEMM move to the CPU". Test: add RMS_NORM to the decline - if taking the norm off HRX
    # fixes it, the bug is in the RMS_NORM-output-at-a-split-boundary path and nowhere else.
    arm q_norm_too  "MUL_MAT,MUL,RMS_NORM" "-dev HRX0"
    arm q_rope_too  "MUL_MAT,MUL,ROPE"     "-dev HRX0"
    arm q_cont_too  "MUL_MAT,MUL,CONT"     "-dev HRX0"
    arm q_pair      "MUL_MAT,MUL"          "-dev HRX0"
    ;;
esac
echo "TIPCHECK SET ${ARM_SET:-1} DONE" | tee -a "$out.txt"
