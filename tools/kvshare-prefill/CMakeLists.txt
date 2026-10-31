# [1bit] TZEROCOPY target 1: the reference prefill provider for a shared KV region (tools/kvshare-prefill).
set(TARGET llama-kvshare-prefill)
add_executable(${TARGET} kvshare-prefill.cpp)
# nlohmann/json is the vendored header the rest of the tree uses (vendor/nlohmann)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/vendor)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
