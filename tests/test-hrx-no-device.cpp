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
// engine#3590: a run that asks for NO GPU device must not open the GPU driver. The HRX build links libggml-hrx into
// every tool, and HRX's registration called hrx_gpu_initialize(), which opens /dev/kfd and creates an HSA context
// outside the box's device serialisation - measured on strixhalo 2026-11-28 with `llama-perplexity --device none`.
//
// Two parts:
//   1. common_args_request_no_gpu(): which command lines count as "no GPU" (pure; every case below);
//   2. in THIS process, after common_params_parse() with `--device none`: GGML_HRX_DISABLE is set, the backend
//      registry is built (ggml_backend_dev_count), HRX contributes no device, and no fd of this process points at
//      /dev/kfd or /dev/dri/*. In a build without HRX or on a machine without a GPU part 2 still runs (the fd check is
//      then a weak one); on the HRX build with the GPU present it is the real check: without the fix, the registry's
//      HRX registration opens /dev/kfd right here.

#include "arg.h"
#include "onebit-no-gpu.h"
#include "common.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <unistd.h>
#endif

static int failures = 0;

static void check(const char * what, bool ok) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) {
        failures++;
    }
}

static bool no_gpu(std::vector<const char *> args) {
    args.insert(args.begin(), "test");
    return common_args_request_no_gpu((int) args.size(), const_cast<char **>(args.data()));
}

#if defined(__linux__)
// the fds of this process that point at the GPU driver (/dev/kfd, /dev/dri/*)
static std::vector<std::string> gpu_fds() {
    std::vector<std::string> out;
    DIR * d = opendir("/proc/self/fd");
    if (d == nullptr) {
        return out;
    }
    while (dirent * e = readdir(d)) {
        if (e->d_name[0] == '.') {
            continue;
        }
        const std::string link = std::string("/proc/self/fd/") + e->d_name;
        char target[512];
        const ssize_t n = readlink(link.c_str(), target, sizeof(target) - 1);
        if (n <= 0) {
            continue;
        }
        target[n] = '\0';
        if (std::strcmp(target, "/dev/kfd") == 0 || std::strncmp(target, "/dev/dri/", 9) == 0) {
            out.push_back(std::string(e->d_name) + " -> " + target);
        }
    }
    closedir(d);
    return out;
}
#endif

int main() {
    // 1. which command lines are "no GPU"
    unsetenv("LLAMA_ARG_DEVICE");
    check("`--device none` is no GPU", no_gpu({ "--device", "none", "-m", "x.gguf" }));
    check("`-dev none` is no GPU", no_gpu({ "-dev", "none" }));
    check("`--device=none` is no GPU", no_gpu({ "--device=none" }));
    check("no --device at all is NOT no GPU (the default offloads)", !no_gpu({ "-m", "x.gguf" }));
    check("`-ngl 0` alone is NOT no GPU (other ops can still run on a device)", !no_gpu({ "-ngl", "0" }));
    check("a named device is NOT no GPU", !no_gpu({ "--device", "HRX0" }));
    check("a llama-bench sweep `-dev none,HRX0` is NOT no GPU", !no_gpu({ "-dev", "none,HRX0" }));
    check("the last --device wins: none then HRX0 is NOT no GPU", !no_gpu({ "-dev", "none", "-dev", "HRX0" }));
    check("the last --device wins: HRX0 then none IS no GPU", no_gpu({ "-dev", "HRX0", "-dev", "none" }));
    check("--device none with a GPU draft model is NOT no GPU", !no_gpu({ "-dev", "none", "-devd", "HRX0" }));
    check("--device none with a GPU projector is NOT no GPU", !no_gpu({ "-dev", "none", "-mmdev", "HRX0" }));
    check("--device none with draft and projector none IS no GPU",
          no_gpu({ "-dev", "none", "--device-draft", "none", "--mmproj-device", "none" }));
    check("--device none with --override-tensor is NOT no GPU (it may name a device buffer)",
          !no_gpu({ "-dev", "none", "-ot", "blk.*=HRX0" }));
    check("--device none with --list-devices is NOT no GPU (it wants to see them)",
          !no_gpu({ "-dev", "none", "--list-devices" }));
    setenv("LLAMA_ARG_DEVICE", "none", 1);
    check("LLAMA_ARG_DEVICE=none with no --device IS no GPU", no_gpu({ "-m", "x.gguf" }));
    check("an explicit --device beats LLAMA_ARG_DEVICE=none", !no_gpu({ "--device", "HRX0" }));
    unsetenv("LLAMA_ARG_DEVICE");

    // 2. this process: parse `--device none`, then build the registry; nothing may open the GPU driver
    unsetenv("GGML_HRX_DISABLE");
    common_params params;
    std::vector<const char *> argv = { "test", "--device", "none", "-m", "unused.gguf" };   // parse only records -m
    const bool parsed = common_params_parse((int) argv.size(), const_cast<char **>(argv.data()), params,
                                            LLAMA_EXAMPLE_COMMON);
    check("common_params_parse accepts `--device none`", parsed);
    const char * flag = std::getenv("GGML_HRX_DISABLE");
    check("common_params_parse set GGML_HRX_DISABLE for `--device none`", flag != nullptr && std::strcmp(flag, "1") == 0);

    const size_t n_dev = ggml_backend_dev_count();   // builds the registry: every linked backend registers now
    check("the registry still has a device (the CPU)", n_dev >= 1);
    size_t hrx_devices = 0;
    if (ggml_backend_reg_t hrx = ggml_backend_reg_by_name("HRX")) {
        hrx_devices = ggml_backend_reg_dev_count(hrx);
    }
    check("HRX contributes no device in a `--device none` run", hrx_devices == 0);
#if defined(__linux__)
    const auto fds = gpu_fds();
    for (const auto & f : fds) {
        std::printf("     open GPU fd: %s\n", f.c_str());
    }
    check("no fd of this process points at /dev/kfd or /dev/dri/*", fds.empty());
#endif

    std::printf("test-hrx-no-device: %s\n", failures == 0 ? "ok" : "FAIL");
    return failures == 0 ? 0 : 1;
}
