set(TARGET kvshare-layout-roundtrip)

add_executable(${TARGET} kvshare-layout-roundtrip.cpp)
target_link_libraries(${TARGET} PRIVATE llama)
target_include_directories(${TARGET} PRIVATE ../../src)
install(TARGETS ${TARGET} RUNTIME)

# engine#753: the provider's import path, device-free. A context has to exist before it can report the layout of its cache,
# so the check is model-dependent: with KVSHARE_TEST_GGUF set it runs in the ordinary suite, and without one it skips (77)
# rather than reding CI.
if (LLAMA_BUILD_TESTS AND LLAMA_BUILD_COMMON)   # the fork's own ctest gate (top-level CMakeLists.txt)
    # the tool reads KVSHARE_TEST_GGUF at run time (a configure-time $ENV{} would freeze it)
    add_test(NAME kvshare_layout_roundtrip COMMAND ${TARGET})
    set_tests_properties(kvshare_layout_roundtrip PROPERTIES SKIP_RETURN_CODE 77)
endif()
