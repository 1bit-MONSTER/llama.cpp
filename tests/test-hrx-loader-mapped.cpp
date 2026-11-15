// engine #2500: on HRX the loader's file mapping must become MAPPED weight buffers (HRX0_Mapped, host), never imported
// ones. llama.cpp#371 routed every mmap-backed weight through the import path (non-host), and zaya1-8b then aborted at
// load because a weight view HRX declines had no backend; llama.cpp#393 restored the mapped path. This loads a model on
// HRX and reads the loader's own "model buffer size" lines, so a revert of the loader hunk fails here, not only in the
// hw-gate. Model: argv[1] or LLAMACPP_TEST_MODELFILE; skipped (exit 0) without a model or an HRX device.
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<std::string> g_buffer_lines;

static void collect(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    if (text != nullptr && std::strstr(text, "model buffer size") != nullptr) {
        g_buffer_lines.emplace_back(text);
    }
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : std::getenv("LLAMACPP_TEST_MODELFILE");
    if (path == nullptr || path[0] == '\0') {
        std::printf("skipped: no model (argv[1] or LLAMACPP_TEST_MODELFILE)\n");
        return EXIT_SUCCESS;
    }
    llama_backend_init();
    ggml_backend_dev_t hrx = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)), "HRX") == 0) {
            hrx = d;
            break;
        }
    }
    if (hrx == nullptr) {
        std::printf("skipped: no HRX device\n");
        llama_backend_free();
        return EXIT_SUCCESS;
    }
    llama_log_set(collect, nullptr);
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers       = 99;
    ggml_backend_dev_t devs[] = { hrx, nullptr };
    params.devices            = devs;
    llama_model * model       = llama_model_load_from_file(path, params);
    llama_log_set(nullptr, nullptr);
    int fails = 0;
    if (model == nullptr) {
        std::fprintf(stderr, "FAIL: the model did not load\n");
        fails++;
    }
    size_t mapped = 0;
    for (const std::string & line : g_buffer_lines) {
        if (line.find("HRX0_Mapped") != std::string::npos) {
            mapped++;
        }
        // an imported weight buffer, or a nameless buffer type: nothing but spaces between "load_tensors:" and
        // "model buffer size" (the empty "()" of the #2500 abort)
        bool nameless = false;
        if (const size_t at = line.find("load_tensors:"); at != std::string::npos) {
            const size_t name = line.find_first_not_of(' ', at + std::strlen("load_tensors:"));
            nameless = name != std::string::npos && line.compare(name, std::strlen("model buffer size"), "model buffer size") == 0;
        }
        if (line.find("_Imported") != std::string::npos || nameless) {
            std::fprintf(stderr, "FAIL: a weight buffer is not a mapped one: %s", line.c_str());
            fails++;
        }
    }
    if (mapped == 0) {
        std::fprintf(stderr, "FAIL: no HRX0_Mapped weight buffer among %zu buffer lines\n", g_buffer_lines.size());
        fails++;
    }
    std::printf("%zu weight buffer lines, %zu HRX0_Mapped, %d failure(s)\n", g_buffer_lines.size(), mapped, fails);
    if (model != nullptr) {
        llama_model_free(model);
    }
    llama_backend_free();
    return fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
