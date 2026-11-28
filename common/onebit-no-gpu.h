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
//
// engine#3590: a run that asks for no GPU device must not open the GPU driver (our own file).
// The HRX build links libggml-hrx into every tool, and HRX registration opens /dev/kfd. So the command line is read
// before anything builds the backend registry, and GGML_HRX_DISABLE is set when no GPU device is asked for.

#pragma once

#include <cstdlib>
#include <string>

// True when the command line asks for no GPU device: --device none (or LLAMA_ARG_DEVICE=none with no --device),
// and no other option selects a device (-mmdev, -devd/--device-draft/--spec-draft-device other than none,
// -ot/--override-tensor, --list-devices).
inline bool common_args_request_no_gpu(int argc, char ** argv) {
    bool main_none = false, main_given = false, other_device = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i] ? argv[i] : "", name = arg, value;
        bool has_value = false;
        if (const auto eq = arg.find('='); arg.rfind("-", 0) == 0 && eq != std::string::npos) {
            name = arg.substr(0, eq); value = arg.substr(eq + 1); has_value = true;
        }
        const auto next = [&]() -> std::string {
            if (has_value) return value;
            return (i + 1 < argc && argv[i + 1]) ? std::string(argv[++i]) : std::string();
        };
        if (name == "-dev" || name == "--device") {
            main_given = true;
            main_none  = next() == "none";
        } else if (name == "-mmdev" || name == "--mmproj-device" || name == "-devd" || name == "--device-draft" ||
                   name == "--spec-draft-device") {
            if (next() != "none") other_device = true;
        } else if (name == "-ot" || name == "--override-tensor" || name == "--list-devices") {
            other_device = true;   // may name a device buffer type / wants to see the devices: keep every backend
        }
    }
    if (!main_given) {
        const char * env = std::getenv("LLAMA_ARG_DEVICE");
        main_none = env != nullptr && std::string(env) == "none";
    }
    return main_none && !other_device;
}

// Set GGML_HRX_DISABLE=1 when the command line asks for no GPU device. A value the caller set already wins.
inline void onebit_no_gpu_disable_hrx(int argc, char ** argv) {
    if (!common_args_request_no_gpu(argc, argv)) {
        return;
    }
#ifdef _WIN32
    if (!std::getenv("GGML_HRX_DISABLE")) {
        _putenv_s("GGML_HRX_DISABLE", "1");
    }
#else
    setenv("GGML_HRX_DISABLE", "1", 0);
#endif
}
