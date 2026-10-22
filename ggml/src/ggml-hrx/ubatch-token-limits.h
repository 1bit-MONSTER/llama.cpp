// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// Token-count bounds shared by the matchers that take ubatches above their kernels' token range
// and by the per-op claim in fused-context-claim.h. Keeping them in one place keeps the claim
// (which decides placement) from promising a node that no matcher can execute: that mismatch
// made every qwen35 prompt batch above 512 tokens fail with GGML_STATUS_FAILED (engine #389,
// "unsupported HRX node 2: L2_NORM ... f32[128,16,1024,1]").

#include <cstdint>

namespace ggml::hrx {

// Largest token count of one llm_ssm_conv_dconv4_silu_rollback_f32 dispatch: its
// llm.ssm_conv.rollback.n_t config range is [1, 512]. Larger convolutions run as a chain of
// rollback dispatches of at most this many tokens (dispatch-ssm-conv-chunked.cpp).
inline constexpr int64_t kSsmConvRollbackMaxTokens = 512;

// Largest token count of one ggml_set_rows dispatch: its ggml.set_rows.token_capacity config range is
// [1, 2048]. A KV-cache write cannot leave HRX (the cache lives there), so larger writes run as several
// dispatches (dispatch-row-chunks.cpp).
inline constexpr int64_t kSetRowsKernelMaxTokens = 2048;

// Largest ubatch the chunked paths and the qwen35 linear-attention matchers accept. The
// gated-delta-net WMMA kernels walk tokens in 16-token chunks with fixed LDS
// (llm.gated_delta_net.token_count range [1, 1048576]); the chained SSM_CONV and SET_ROWS have no
// bound of their own. 65536 is above any ubatch llama.cpp builds here.
inline constexpr int64_t kUbatchMaxTokens = 65536;

// Largest number of gated-delta-net rollback snapshots one node may carry. Nothing is allocated for this
// number: llama_memory_recurrent sizes (1 + n_rs_seq) state planes, and
// common_params_speculative::need_n_rs_seq() sets n_rs_seq to the draft n_max, so --spec-draft-n-max N
// already commits N + 1 planes at context creation whether or not a graph writes them. The matcher checks
// that the cache has exactly as many planes as the graph writes (cache->ne[2] == written_snapshot_count) and
// that every plane it touches is inside it, so the useful bound is the ubatch; a hard 5 refused the 6th to
// 8th snapshot of Qwen3.8-27B --mtp --spec-draft-n-max 5 and 7 for no reason (2026-10-21).
inline constexpr int64_t kMaximumSnapshots = kUbatchMaxTokens;

}  // namespace ggml::hrx
