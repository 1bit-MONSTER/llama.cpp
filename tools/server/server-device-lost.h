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

// llama-server exits when the GPU is lost, so its supervisor starts it again.
//
// After a GPU reset (an MES hang, a MODE2 reset) the HRX backend cannot use the device again: ggml-hrx
// logs "DATA_LOSS; AMDGPU hardware exception" and every request after that fails with "Compute error"
// while /health still answers 200, so nothing restarts the server. The 1bit engine's `1bit serve`
// watched for that line and exited (engine #466); when Lemonade starts llama-server itself (engine
// "one hop"), the server does it: the line is passed on, then the process exits with status 1.
// Lemonade's BackendWatchdog sees the exit and loads the model again on the next request.
//
// LLAMA_SERVER_EXIT_ON_DEVICE_LOST: 0 keeps the server up (the old behaviour); unset or 1, it exits.

namespace server_device_lost {

// True for a log line that says the device is gone for good.
bool line_says_lost(const char * text);

// The switch: off only for "0".
bool enabled(const char * value);

// Wraps the current ggml log callback (call it after common_init()) when enabled.
void install();

}  // namespace server_device_lost
