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

// Where a graph's host time goes (engine #572). GGML_HRX_HOST_GAP=1 turns on two cheap
// steady_clock reads per phase and an atexit summary on stderr: the totals per phase and the
// per-graph replay accounting the executor already computes (build, launch, dispatches). Device
// kernel time is not in here; take it from rocprofv3 and subtract. Off, this compiles to an
// `if (enabled)` on a cached static and costs nothing.

#ifndef GGML_HRX_HOST_GAP_PROFILE_H
#define GGML_HRX_HOST_GAP_PROFILE_H

#include <cstdint>

namespace ggml::hrx {

bool host_gap_enabled();

// Times one named phase of the host path while in scope.
class HostGapScope {
  public:
    explicit HostGapScope(const char * phase);
    ~HostGapScope();

    HostGapScope(const HostGapScope &)             = delete;
    HostGapScope & operator=(const HostGapScope &) = delete;

  private:
    const char * phase_    = nullptr;
    uint64_t     start_ns_ = 0;
};

// One graph compute, as the executor already accounts it.
void host_gap_begin_graph();
void host_gap_end_graph(uint64_t wall_ns, uint64_t build_ns, uint64_t launch_ns, uint64_t dispatches);

}  // namespace ggml::hrx

#endif  // GGML_HRX_HOST_GAP_PROFILE_H
