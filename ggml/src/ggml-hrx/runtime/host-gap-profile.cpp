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

#include "host-gap-profile.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

namespace ggml::hrx {

namespace {

using Clock = std::chrono::steady_clock;

uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}

struct PhaseTotal {
    uint64_t ns    = 0;
    uint64_t count = 0;
};

struct GraphTotal {
    uint64_t ns         = 0;
    uint64_t build_ns   = 0;
    uint64_t launch_ns  = 0;
    uint64_t dispatches = 0;
    uint64_t count      = 0;
};

struct HostGapState {
    std::map<std::string, PhaseTotal> current;  // phases of the graph being timed
    std::map<std::string, PhaseTotal> totals;
    GraphTotal                        graphs;
    bool                              verbose = false;
};

// Leaked on purpose: the atexit summary must not race this object's destruction.
HostGapState & state() {
    static HostGapState * s = new HostGapState();
    return *s;
}

void print_graph(const char * tag, const GraphTotal & graph, const std::map<std::string, PhaseTotal> & phases) {
    std::fprintf(stderr, "hrx host gap %s: wall=%.3f ms build=%.3f ms launch=%.3f ms dispatches=%llu\n", tag,
                 graph.ns / 1e6, graph.build_ns / 1e6, graph.launch_ns / 1e6, (unsigned long long) graph.dispatches);
    for (const auto & entry : phases) {
        std::fprintf(stderr, "hrx host gap %s:   %-14s %6llu x %9.3f ms\n", tag, entry.first.c_str(),
                     (unsigned long long) entry.second.count, entry.second.ns / 1e6);
    }
}

void print_summary() {
    HostGapState &     s = state();
    const GraphTotal & g = s.graphs;
    std::fprintf(stderr, "hrx host gap TOTAL: graphs=%llu wall=%.3f ms build=%.3f ms launch=%.3f ms dispatches=%llu\n",
                 (unsigned long long) g.count, g.ns / 1e6, g.build_ns / 1e6, g.launch_ns / 1e6,
                 (unsigned long long) g.dispatches);
    for (const auto & entry : s.totals) {
        std::fprintf(stderr, "hrx host gap TOTAL:   %-14s %6llu x %9.3f ms\n", entry.first.c_str(),
                     (unsigned long long) entry.second.count, entry.second.ns / 1e6);
    }
}

}  // namespace

bool host_gap_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_HRX_HOST_GAP");
        const bool   on    = value != nullptr && value[0] != '\0' && value[0] != '0';
        if (on) {
            const char * verbose = std::getenv("GGML_HRX_HOST_GAP_VERBOSE");
            state().verbose      = verbose != nullptr && verbose[0] == '1';
            std::atexit(print_summary);
        }
        return on;
    }();
    return enabled;
}

void host_gap_begin_graph() {
    if (!host_gap_enabled()) {
        return;
    }
    state().current.clear();
}

void host_gap_end_graph(uint64_t wall_ns, uint64_t build_ns, uint64_t launch_ns, uint64_t dispatches) {
    if (!host_gap_enabled()) {
        return;
    }
    HostGapState & s = state();
    GraphTotal     graph{wall_ns, build_ns, launch_ns, dispatches, 1};
    if (s.verbose) {
        print_graph("graph", graph, s.current);
    }
    s.graphs.ns += wall_ns;
    s.graphs.build_ns += build_ns;
    s.graphs.launch_ns += launch_ns;
    s.graphs.dispatches += dispatches;
    s.graphs.count++;
    for (const auto & entry : s.current) {
        PhaseTotal & total = s.totals[entry.first];
        total.ns += entry.second.ns;
        total.count += entry.second.count;
    }
    s.current.clear();
}

HostGapScope::HostGapScope(const char * phase) {
    if (host_gap_enabled()) {
        phase_    = phase;
        start_ns_ = now_ns();
    }
}

HostGapScope::~HostGapScope() {
    if (phase_ == nullptr) {
        return;
    }
    PhaseTotal & total = state().current[phase_];
    total.ns += now_ns() - start_ns_;
    total.count++;
}

}  // namespace ggml::hrx
