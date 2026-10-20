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

#include "server-device-lost.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

namespace server_device_lost {

bool line_says_lost(const char * text) {
    if (text == nullptr) {
        return false;
    }
    for (const char * mark : { "AMDGPU hardware exception", "ErrorDeviceLost", "VK_ERROR_DEVICE_LOST" }) {
        if (std::strstr(text, mark) != nullptr) {
            return true;
        }
    }
    return false;
}

bool enabled(const char * value) {
    return value == nullptr || std::strcmp(value, "0") != 0;
}

namespace {

ggml_log_callback g_next      = nullptr;
void *            g_next_data = nullptr;

void on_log(ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (g_next != nullptr) {
        g_next(level, text, g_next_data);
    }
    if (line_says_lost(text)) {
        // the logger may print from its own thread: say it here, unbuffered, before leaving
        std::fprintf(stderr, "%s", text);
        std::fprintf(stderr, "llama-server: the GPU is lost; exiting so the supervisor starts the model again "
                             "(LLAMA_SERVER_EXIT_ON_DEVICE_LOST=0 keeps the server up)\n");
        std::fflush(stderr);
        std::_Exit(1);
    }
}

}  // namespace

void install() {
    if (!enabled(std::getenv("LLAMA_SERVER_EXIT_ON_DEVICE_LOST"))) {
        return;
    }
    ggml_log_get(&g_next, &g_next_data);
    ggml_log_set(on_log, nullptr);
}

}  // namespace server_device_lost
