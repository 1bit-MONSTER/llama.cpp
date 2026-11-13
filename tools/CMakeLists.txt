# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    add_subdirectory(batched-bench)
    add_subdirectory(gguf-split)
    add_subdirectory(imatrix)
    add_subdirectory(llama-bench)
    add_subdirectory(completion)
    add_subdirectory(perplexity)
    add_subdirectory(quantize)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(ui)
        add_subdirectory(cli)
        add_subdirectory(server)
    endif()
    add_subdirectory(tokenize)
    add_subdirectory(kvshare-prefill) # [1bit] target 1: reference provider for a shared KV region
    add_subdirectory(kvshare-tensor-map) # [1bit] target 1: the tensor table a provider writes into (engine#753)
    add_subdirectory(kvshare-layout-roundtrip) # [1bit] target 1: the provider's import path with no GPU (engine#753)
    add_subdirectory(kvshare-adopt-check) # [1bit] target 1: a decoder working from NPU-written K/V (engine#753)
    add_subdirectory(tts)
    add_subdirectory(mtmd)
    if (GGML_RPC)
        add_subdirectory(rpc)
    endif()
    if (LLAMA_CONNECT)
        add_subdirectory(connect)
    endif()
    if (NOT GGML_BACKEND_DL AND GGML_CPU)
        # these tools use backends directly (no dynamic loading) and depend on CPU backend symbols
        add_subdirectory(cvector-generator)
        add_subdirectory(export-lora)
    endif()
    add_subdirectory(fit-params)
    if (GGML_METAL)
        add_subdirectory(tuning)
    endif()
    add_subdirectory(results)
endif()
