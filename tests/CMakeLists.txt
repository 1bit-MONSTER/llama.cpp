llama_add_compile_flags()

# Optional args:
# - LINK: libraries to link against (defaults to llama-common, which also brings llama)
function(llama_build source)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs LINK)
    cmake_parse_arguments(LLAMA_BUILD "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_BUILD_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_BUILD_LINK)
        set(LLAMA_BUILD_LINK llama-common)
    endif()

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_BUILD_LINK})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
# - LINK: libraries to link against (defaults to llama-common, which also brings llama)
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS LINK)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_TEST_LINK)
        set(LLAMA_TEST_LINK llama-common)
    endif()

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_TEST_LINK})

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge        LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r       LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm    LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon          LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4         LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf --check-decode)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2           LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe       LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm       LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt             LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3           LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2           LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen35          LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact          LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder       LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

# engine#2433 regression arm: one llama-server request with MTP drafting, CPU only; needs ONEBIT_MTP_TEST_MODEL
# (a GGUF with its own MTP head), else it reports SKIP (77)
if (NOT WIN32 AND LLAMA_BUILD_SERVER)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-mtp-one-request.sh
        NAME test-mtp-one-request
        LABEL behavior
        ARGS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    )
    # CI leaves ONEBIT_MTP_TEST_MODEL unset, so the row is SKIP there. TIMEOUT must exceed the script's own worst
    # case (health 200 x 5 s + the 600 s request = ~1600 s), or a slow load is cut off before its exit code prints
    set_tests_properties(test-mtp-one-request PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)

    # llama.cpp#391: the fast greedy path's log line is the only observable of whether the optimisation was TAKEN.
    # A round of review pointed out that the line had no consumer, which is the same defect one level up from the
    # regression it exists to catch; this row is the consumer. Needs a small GGUF (CPU only), else SKIP (77).
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-fast-greedy-log.sh
        NAME test-fast-greedy-log
        LABEL behavior
        ARGS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    )
    set_tests_properties(test-fast-greedy-log PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
endif()

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        LABEL behavior
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp  LABEL "main behavior"  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

# llama_context state tests: KV and recurrent state save, load, copy, rollback
set(LLAMA_CONTEXT_TEST_SRC
    test-llama-context.cpp
    llama-context/cases-kv.cpp
    llama-context/cases-rs.cpp
    llama-context/test-case.cpp
    llama-context/test-log.cpp
    llama-context/test-runner.cpp
    llama-context/test-util.cpp
)
llama_build(${LLAMA_CONTEXT_TEST_SRC})

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-kv-zero.cpp)
    target_include_directories(test-kv-zero PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-unicode.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-sampling.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-sampling-fast-greedy.cpp LABEL behavior)
    llama_build_and_test(test-reasoning-budget.cpp  LABEL behavior)
    llama_build_and_test(test-onebit-mtp-budget.cpp LABEL behavior) # engine #2433
    llama_build_and_test(test-grammar-parser.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-grammar-integration.cpp  LABEL behavior)
    llama_build_and_test(test-llama-grammar.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-batch-alloc.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-chat.cpp  LABEL behavior  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    llama_build_and_test(test-server-component.cpp  LABEL behavior)
    target_include_directories(test-server-component PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-component PRIVATE server-context)
    llama_build_and_test(test-json-schema-to-grammar.cpp  LABEL behavior)

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp LINK llama)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila     LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon     LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2      LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox   LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt        LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact     LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

    llama_build_and_test(test-llama-archs.cpp)

    llama_build_and_test(test-lora-inplace.cpp)
    target_include_directories(test-lora-inplace PRIVATE ${PROJECT_SOURCE_DIR}/src)

    set(MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models/")
    file(MAKE_DIRECTORY "${MODEL_DIR}")

    llama_test(
        test-llama-archs
        NAME test-generate-models
        LABEL main
        ARGS -o "${MODEL_DIR}"
    )
    set_tests_properties(test-generate-models PROPERTIES
        FIXTURES_SETUP generate-models
    )

endif()



# Target model for test-speculative-rejection-sampling
set(SPEC_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-speculative-models/target")
file(MAKE_DIRECTORY "${SPEC_TARGET_DIR}")
llama_test(
    test-llama-archs
    NAME test-generate-speculative-target
    LABEL main
    ARGS -a llama -s 999 -o "${SPEC_TARGET_DIR}"
)
set_tests_properties(
    test-generate-speculative-target
    PROPERTIES FIXTURES_SETUP speculative-model
)

# To avoid recreating the model use the fixture exclude any option:
# -FA "^speculative-model$"
llama_build_and_test(test-speculative-rejection-sampling.cpp ARGS
    -m "${SPEC_TARGET_DIR}/llama-dense.gguf")
set_tests_properties(test-speculative-rejection-sampling PROPERTIES FIXTURES_REQUIRED speculative-model)

# Test context state handling across all architectures, using the generated dummy models
llama_test(
    test-llama-context
    LABEL "main behavior backend"
    ARGS --models "${MODEL_DIR}"
)
# abort on graph reallocations that the initial reserve should have covered
set_tests_properties(test-llama-context PROPERTIES
    FIXTURES_REQUIRED generate-models
    ENVIRONMENT "GGML_SCHED_DEBUG_REALLOC=1"
)

llama_build(test-fusion.cpp)

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp  LABEL behavior)
llama_build_and_test(test-jinja.cpp  LABEL behavior)
llama_test(test-jinja  NAME test-jinja-py  ARGS -py  LABEL "python behavior")
llama_build_and_test(test-json-schema.cpp  LABEL behavior)
llama_build_and_test(test-chat-auto-parser.cpp  LABEL behavior  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp  LABEL behavior)
# debug tool for chat template differential analysis (not registered as a test, run it manually)
llama_build(test-chat-analysis.cpp)
llama_build_and_test(test-log.cpp  LABEL behavior)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
    LABEL behavior
)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${PROJECT_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

# the context state cases against a real model, the sweep above uses the generated dummy ones
llama_test(test-llama-context  NAME test-llama-context-model  LABEL "model behavior backend"  ARGS -m "${MODEL_DEST}")
set_tests_properties(test-llama-context-model PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-thread-safety.cpp  LABEL "main backend"  ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp  LABEL behavior)
llama_build_and_test(test-model-resolution.cpp  LABEL behavior)
# the test serves its repos from an httplib server, and the library links it privately
target_link_libraries(test-model-resolution PRIVATE cpp-httplib)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp  LABEL "main behavior backend"  LINK ggml)
endif()
if (GGML_METAL AND NOT GGML_BACKEND_DL)
    # calls the Metal backend's graph optimizer and fusion table directly
    llama_build_and_test(test-metal-graph-optimize.cpp  LABEL "main behavior backend"  LINK ggml ggml-metal)
    target_include_directories(test-metal-graph-optimize PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-metal)
endif()

llama_build(test-backend-ops.cpp LINK ggml)

llama_build_and_test(test-tiled-mulmat.cpp  LABEL "main behavior backend"  LINK ggml)
llama_build_and_test(test-model-load-cancel.cpp LABEL "model behavior")
llama_build_and_test(test-autorelease.cpp       LABEL "model behavior")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model behavior backend")

if (APPLE)
    llama_build(test-rset-release.cpp)
endif()

if (NOT GGML_BACKEND_DL AND GGML_CPU)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp  LABEL "main behavior"  LINK ggml)
    llama_build_and_test(test-quantize-fns.cpp  LABEL "main behavior"  LINK ggml)
    llama_build_and_test(test-cpu-repack-strided.cpp  LABEL "main behavior backend"  LINK ggml)  # engine#3531
    llama_build_and_test(test-quantize-perf.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-rope.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-col2im-1d.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-conv-1d.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-gelu-quick.cpp  LABEL "main behavior backend"  LINK ggml)
endif()

# libmtmd
llama_build_and_test(test-mtmd-c-api.c  LABEL behavior  LINK mtmd)

llama_build_and_test(test-mtmd-impl.cpp  LABEL behavior  LINK mtmd)

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE llama-common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data llama-common)
        llama_test(test-gguf-model-data  LABEL "model behavior")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp  LABEL "model behavior"  LINK llama)
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp  LABEL behavior  LINK ggml)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp  LABEL "behavior backend"  LINK ggml)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()

# the stall-aware /health decision: pure, no server and no model
add_executable(test-server-health test-server-health.cpp ../tools/server/server-health.cpp ../tools/server/server-device-lost.cpp)
target_link_libraries(test-server-health PRIVATE ggml)
target_include_directories(test-server-health PRIVATE ../tools/server)
add_test(NAME test-server-health COMMAND test-server-health)

# engine#2912: a device case that cannot see a device must not answer success. The caller runs one case to test a
# hypothesis and reads the exit code; on 2026-11-20 three device cases came back rc 0 in one second inside a held
# gpu-hold ticket, because another tenant was initialising into the GPU. The runner answers 77 now (the same "skipped"
# the engine's ctest rows use), and this row is the arm: no device, no model, ~0 s, and it fails on the old behaviour.
add_executable(test-testing-suite-skip test-testing-suite-skip.cpp)
add_test(NAME test-testing-suite-skip COMMAND test-testing-suite-skip)

if (TARGET ggml-hrx)
    # engine#2774: every "<corpus>:<kernel>" literal hrx-ops-test.cpp expects must be one a GGML_HRX_KERNEL_REF
    # registers, and the old suppression table must not come back. From the source: no GPU, no build, ~1 s.
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_Interpreter_FOUND)
        add_test(NAME test-hrx-stale-expectations
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-hrx/tools/check_stale_expectations.py
                         --check --root ${CMAKE_CURRENT_SOURCE_DIR}/..)
        add_test(NAME test-hrx-stale-expectations-self
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-hrx/tools/check_stale_expectations.py
                         --self-test)
    endif()
    # the program cache's byte budget and LRU order: pure policy, no device and no HRX link
    # Which rotations the fused registrations accept, i.e. what the claim may assert (engine #1467).
    # Header-only: no device, no model, and the vision rotation must be rejected here.
    # engine#2986: the GTT reserve is a cap, not a constant -- a flat 16 GiB gave a 16 GB machine (about 8 GiB of GTT) a
    # ceiling of zero, so this refused the device at registration for every model. The arithmetic of one function: no
    # device, no runtime, no model.
    add_executable(test-hrx-gtt-reserve test-hrx-gtt-reserve.cpp ../ggml/src/ggml-hrx/hrx-1bit-safety.cpp)
    target_include_directories(test-hrx-gtt-reserve PRIVATE ../ggml/include ../ggml/src)
    target_link_libraries(test-hrx-gtt-reserve PRIVATE ggml)
    add_test(NAME test-hrx-gtt-reserve COMMAND test-hrx-gtt-reserve)

    # engine#3382: the dynamic workload specialization the generic ops select, and the one-program-per-class key.
    # Plans and compiles only, no device: one compile of scale_f32 and two of concat_dim0_f32.
    add_executable(test-hrx-dynamic-workload test-hrx-dynamic-workload.cpp)
    target_link_libraries(test-hrx-dynamic-workload PRIVATE ggml-hrx ggml hrx::hrx loomc::loomc)
    target_include_directories(test-hrx-dynamic-workload PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-dynamic-workload COMMAND test-hrx-dynamic-workload)

    add_executable(test-hrx-rope-claim-params test-hrx-rope-claim-params.cpp)
    target_include_directories(test-hrx-rope-claim-params PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-rope-claim-params COMMAND test-hrx-rope-claim-params)
    target_link_libraries(test-hrx-rope-claim-params PRIVATE ggml)

    add_executable(test-hrx-graph-program-budget test-hrx-graph-program-budget.cpp
        ../ggml/src/ggml-hrx/runtime/graph-program-budget.cpp)
    target_include_directories(test-hrx-graph-program-budget PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-graph-program-budget COMMAND test-hrx-graph-program-budget)

    # the host-weight arena refusal's wording: pure helper, no device and no HRX link (engine #900)
    # the backend's failure classification: resource failure vs compute failure, pure (engine #1618)
    add_executable(test-hrx-failure-status test-hrx-failure-status.cpp
        ../ggml/src/ggml-hrx/runtime/failure-status.cpp)
    target_include_directories(test-hrx-failure-status PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-failure-status COMMAND test-hrx-failure-status)

    add_executable(test-hrx-pool-refusal-budget test-hrx-pool-refusal-budget.cpp
                   ../ggml/src/ggml-hrx/hrx-1bit-safety.cpp)
    target_include_directories(test-hrx-pool-refusal-budget PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    target_link_libraries(test-hrx-pool-refusal-budget PRIVATE ggml)
    add_test(NAME test-hrx-pool-refusal-budget COMMAND test-hrx-pool-refusal-budget)
    # both sides of GGML_HRX_STRICT: the named by-design classes and the unexplained case that still aborts
    # (pure helper, no device and no ggml link -- engine#2450)
    add_executable(test-hrx-placement-classification test-hrx-placement-classification.cpp
                   ../ggml/src/ggml-hrx/placement-classification.cpp)
    # engine#2494: residence is not usage. Links ggml because it needs a real backend buffer to mark USAGE_WEIGHTS,
    # and the CPU device is the only one guaranteed present -- no GPU, no HRX runtime, no model.
    add_executable(test-hrx-host-resident test-hrx-host-resident.cpp
                   ../ggml/src/ggml-hrx/host-weight-placement.cpp)
    # backend-context.h pulls ggml-backend-impl.h and kernel-executable-cache.h pulls hrx_runtime.h, so the unit
    # needs ggml/src and the HRX headers; it still links no HRX library, which is the point of the test.
    target_include_directories(test-hrx-host-resident PRIVATE ../ggml/src/ggml-hrx ../ggml/include ../ggml/src)
    if(HRX_SOURCE_DIR)
        target_include_directories(test-hrx-host-resident PRIVATE "${HRX_SOURCE_DIR}/libhrx/include")
    elseif(TARGET hrx::hrx)
        get_target_property(_hrx_header_dirs hrx::hrx INTERFACE_INCLUDE_DIRECTORIES)
        if(_hrx_header_dirs)
            target_include_directories(test-hrx-host-resident PRIVATE ${_hrx_header_dirs})
        endif()
    endif()
    target_link_libraries(test-hrx-host-resident PRIVATE ggml)
    add_test(NAME test-hrx-host-resident COMMAND test-hrx-host-resident)
    target_include_directories(test-hrx-placement-classification PRIVATE ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-placement-classification COMMAND test-hrx-placement-classification)
    # engine#3521: binary launches past the Loom kernels' declared ranges are refused, and the host caps are the
    # kernels' own numbers (header-only helper plus the kernel sources as text: no device, no ggml link)
    add_executable(test-hrx-binary-span-cap test-hrx-binary-span-cap.cpp)
    target_include_directories(test-hrx-binary-span-cap PRIVATE ../ggml/src/ggml-hrx)
    target_compile_definitions(test-hrx-binary-span-cap PRIVATE
        HRX_LOOM_OPS_DIR="${PROJECT_SOURCE_DIR}/ggml/src/ggml-hrx/kernel-corpus/kernels/loom-libs/ops")
    add_test(NAME test-hrx-binary-span-cap COMMAND test-hrx-binary-span-cap)
    # engine#2704: the GET_ROWS accept predicate both the placement path and the matchers call. Compiles the predicate's
    # own translation unit only -- no device, no libhrx, no graph -- because the defect it prevents is the two callers
    # disagreeing, which neither caller's own test can catch.
    add_executable(test-hrx-get-rows-facts test-hrx-get-rows-facts.cpp
                   ../ggml/src/ggml-hrx/dispatch_registration/common/dispatch-get-rows-support.cpp)
    target_include_directories(test-hrx-get-rows-facts PRIVATE ../ggml/include ../ggml/src/ggml-hrx)
    # ggml.h is included for the GGML_TYPE_IQ3_S enum check only; nothing from the library is called.
    target_link_libraries(test-hrx-get-rows-facts PRIVATE ggml)
    add_test(NAME test-hrx-get-rows-facts COMMAND test-hrx-get-rows-facts)
    # engine#2861: the MoE partition table is written with one descriptor layout and read with another. Header-only --
    # no device, no libhrx, no graph -- because the defect is two separately-compiled sides disagreeing about bit
    # positions, and a device run can only report the fault it causes, not which pair of constants disagreed.
    add_executable(test-hrx-moe-routing-layout test-hrx-moe-routing-layout.cpp)
    target_include_directories(test-hrx-moe-routing-layout PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-moe-routing-layout COMMAND test-hrx-moe-routing-layout)
    add_executable(test-hrx-arena-refusal test-hrx-arena-refusal.cpp)
    target_include_directories(test-hrx-arena-refusal PRIVATE ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-arena-refusal COMMAND test-hrx-arena-refusal)
    # what the transient arena owes the pool when it grows: pure helper, no device and no HRX link (engine #1711)
    add_executable(test-hrx-transient-arena-charge test-hrx-transient-arena-charge.cpp)
    target_include_directories(test-hrx-transient-arena-charge PRIVATE ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-transient-arena-charge COMMAND test-hrx-transient-arena-charge)
    # the host staging buffer's charge: the ledger is the only thing under test, no device and no libhrx (engine #1711)
    add_executable(test-hrx-staging-ledger test-hrx-staging-ledger.cpp
                   ../ggml/src/ggml-hrx/hrx-1bit-safety.cpp)
    target_include_directories(test-hrx-staging-ledger PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    target_link_libraries(test-hrx-staging-ledger PRIVATE ggml)
    add_test(NAME test-hrx-staging-ledger COMMAND test-hrx-staging-ledger)

    # engine#1856: the same conservation for the HOST-WEIGHT arena's charge, which had no ledger arm while the staging
    # buffer has had one since #270. Device-free for the same reason: the charge is simulated on the object and the paths
    # that must return it are exercised directly.
    add_executable(test-hrx-arena-ledger test-hrx-arena-ledger.cpp
                   ../ggml/src/ggml-hrx/hrx-1bit-safety.cpp)
    target_include_directories(test-hrx-arena-ledger PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    target_link_libraries(test-hrx-arena-ledger PRIVATE ggml)
    add_test(NAME test-hrx-arena-ledger COMMAND test-hrx-arena-ledger)

    # MemAvailable parsing for the unbounded-residency warning: pure, no device and no /proc (engine #669)
    add_executable(test-hrx-host-memory-available test-hrx-host-memory-available.cpp
        ../ggml/src/ggml-hrx/runtime/host-memory-available.cpp)
    target_include_directories(test-hrx-host-memory-available PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-host-memory-available COMMAND test-hrx-host-memory-available)

    # the one number a bounded run is judged on: the materialized bytes the model's host weights need (engine #669/#900)
    add_executable(test-hrx-host-weight-working-set test-hrx-host-weight-working-set.cpp
        ../ggml/src/ggml-hrx/runtime/host-weight-working-set.cpp)
    target_include_directories(test-hrx-host-weight-working-set PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-host-weight-working-set COMMAND test-hrx-host-weight-working-set)

    add_executable(test-hrx-buffer test-hrx-buffer.cpp)
    target_link_libraries(test-hrx-buffer PRIVATE ggml-hrx ggml hrx::hrx)
    target_include_directories(test-hrx-buffer PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-buffer COMMAND test-hrx-buffer)
    add_executable(hrx-buffer-test hrx-buffer-test.cpp)
    target_link_libraries(hrx-buffer-test PRIVATE ggml-hrx ggml hrx::hrx)
    target_include_directories(hrx-buffer-test PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME hrx-buffer-test COMMAND hrx-buffer-test
        --hrx-buffer-test-jobs 1 --hrx-buffer-test-output all --hrx-buffer-test-filter ".*")
    # engine #2500: the loader's mapping becomes HRX0_Mapped weight buffers (skips without LLAMACPP_TEST_MODELFILE)
    llama_build_and_test(test-hrx-loader-mapped.cpp LABEL "hrx")
    # 77 stays for the model skip; the device skip exits 1 and prints this string (test-hrx-ops' convention).
    # SKIP_RETURN_CODE wins over FAIL_REGULAR_EXPRESSION on ctest 4.2.3, so the exit code is what makes it loud.
    set_tests_properties(test-hrx-loader-mapped PROPERTIES SKIP_RETURN_CODE 77
        FAIL_REGULAR_EXPRESSION "test skipped: no device available")

    add_executable(hrx-backend-test hrx-backend-test.cpp)
    target_link_libraries(hrx-backend-test PRIVATE ggml-hrx ggml hrx::hrx loomc::loomc)
    target_include_directories(hrx-backend-test PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    # our line's hrx-backend-test takes case names as its only arguments (no test_runner::Suite flags)
    add_test(NAME hrx-backend-test COMMAND hrx-backend-test)
    # engine#3126 review: the capacity-class contract on ITS OWN row, because the suite's own row is a baselined known
    # failure (engine's tools/ci/ctest_known_fail.txt) and a case inside it cannot turn CI red however it fails. This
    # case is device-free -- import_ggml_graph + DispatchScheduler, no backend, no add-on -- so the row costs seconds,
    # and the name is not in the baseline, which is what makes a regression here a NEW failure.
    add_test(NAME hrx_backend_get_rows_capacity COMMAND hrx-backend-test get_rows_token_capacity_class)
    if (GGML_HRX_HIP_ADDON_DIR)
        # engine#2392: hrx-backend-test labels the cases the add-on's own matchers take (hip_addon_known_fail)
        target_compile_definitions(hrx-backend-test PRIVATE GGML_HRX_HIP_ADDON)
        # engine#2392: with a HIP add-on built in, its matchers claim nodes whose Loom-path plan many host cases
        # assert, so an add-on build otherwise never checks the Loom path at all. This runs the same suite with every
        # add-on registration (all named hip.*) switched off; measured 87/0 host, 14/14 device at 5006d9308.
        add_test(NAME hrx-backend-test-loom COMMAND hrx-backend-test)
        set_tests_properties(hrx-backend-test-loom PROPERTIES
            ENVIRONMENT "GGML_HRX_DISABLE_DISPATCH=hip."
            LABELS hrx
            TIMEOUT 900
            FAIL_REGULAR_EXPRESSION "test skipped: no device available")
    endif()

    # our line's Loom JIT suite (the base's ctest row); AMD's hrx-loom-jit-test is a different file
    add_executable(test-hrx-loom-jit test-hrx-loom-jit.cpp)
    target_link_libraries(test-hrx-loom-jit PRIVATE ggml-hrx ggml-hrx-kernel-corpus hrx::hrx loomc::loomc)
    target_include_directories(test-hrx-loom-jit PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-loom-jit COMMAND test-hrx-loom-jit)

    add_executable(hrx-loom-jit-test hrx-loom-jit-test.cpp)
    target_link_libraries(hrx-loom-jit-test PRIVATE ggml-hrx ggml-hrx-kernel-corpus hrx::hrx loomc::loomc)
    target_include_directories(hrx-loom-jit-test PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME hrx-loom-jit-test COMMAND hrx-loom-jit-test
        --hrx-loom-jit-test-jobs 2 --hrx-loom-jit-test-output all --hrx-loom-jit-test-filter ".*")

    # How a layout-alias node is judged when its input is PRODUCED vs EXTERNAL: engine #1467's projector is
    # refused at a no-op VIEW that the standalone oracle calls supported because it feeds the view from a leaf.
    add_executable(test-hrx-view-alias-elision test-hrx-view-alias-elision.cpp)
    target_link_libraries(test-hrx-view-alias-elision PRIVATE ggml-hrx ggml hrx::hrx loomc::loomc)
    target_include_directories(test-hrx-view-alias-elision PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-view-alias-elision COMMAND test-hrx-view-alias-elision)

    # a view into storage the graph does not hold (the deepstack window of Qwen3-VL, engine #1467): the plan on the
    # host, the values on HRX0 when there is one
    add_executable(test-hrx-external-window-view test-hrx-external-window-view.cpp)
    target_link_libraries(test-hrx-external-window-view PRIVATE ggml-hrx ggml hrx::hrx loomc::loomc)
    target_include_directories(test-hrx-external-window-view PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-external-window-view COMMAND test-hrx-external-window-view)
    # the device skip exits 1 (see the file); the regex is belt and braces, as on its siblings
    set_tests_properties(test-hrx-external-window-view PROPERTIES SKIP_RETURN_CODE 77
        FAIL_REGULAR_EXPRESSION "test skipped: no device available")

    # the fused GATED_DELTA_NET pattern when the scheduler has already elided one of its layout aliases (the state
    # RESHAPE visited before the L2_NORM root, engine #1925): host only
    add_executable(test-hrx-gdn-passed-alias test-hrx-gdn-passed-alias.cpp)
    target_link_libraries(test-hrx-gdn-passed-alias PRIVATE ggml-hrx ggml hrx::hrx loomc::loomc)
    target_include_directories(test-hrx-gdn-passed-alias PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-gdn-passed-alias COMMAND test-hrx-gdn-passed-alias)

    # our line's HRX op suite (#2150's q6_k shortlist .repeat among its cases); AMD's hrx-ops-test is a different file
    add_executable(test-hrx-ops test-hrx-ops.cpp)
    target_link_libraries(test-hrx-ops PRIVATE ggml-hrx ggml hrx::hrx loomc::loomc)
    target_include_directories(test-hrx-ops PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-ops COMMAND test-hrx-ops)
    # engine#2393 / #355: the same suite with every add-on registration disabled, so the LOOM path is exercised at
    # all -- until now no row ran it, and the add-on's own kernels hid whatever the loom path does. Measured at
    # 7e91e98a7 with GGML_HRX_DISABLE_DISPATCH=hip.: 270 passed, 0 failed, 1 known-fail (a labelled one),
    # 0 unexpected-pass, 271 total, rc 0. The no-device guard below is not optional: without it this row passes on
    # a machine whose runtime never initialised, which is the defect #375 fixed for its two siblings.
    add_test(NAME test-hrx-ops-loom COMMAND test-hrx-ops)
    set_tests_properties(test-hrx-ops-loom PROPERTIES
        ENVIRONMENT "GGML_HRX_DISABLE_DISPATCH=hip.")


    # engine#2918: multi-stream FLASH_ATTN_EXT (n_seq_max > 1 without a unified KV cache) on HRX0 vs the CPU
    add_executable(test-hrx-fa-streams test-hrx-fa-streams.cpp)
    target_link_libraries(test-hrx-fa-streams PRIVATE ggml)
    add_test(NAME test-hrx-fa-streams COMMAND test-hrx-fa-streams)
    set_tests_properties(test-hrx-fa-streams PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(test-hrx-hadamard test-hrx-hadamard.cpp)
    target_link_libraries(test-hrx-hadamard PRIVATE ggml)
    add_test(NAME test-hrx-hadamard COMMAND test-hrx-hadamard)
    set_tests_properties(test-hrx-hadamard PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(test-hrx-mxfp4 test-hrx-mxfp4.cpp)
    target_link_libraries(test-hrx-mxfp4 PRIVATE ggml-hrx ggml hrx::hrx loomc::loomc)
    target_include_directories(test-hrx-mxfp4 PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-mxfp4 COMMAND test-hrx-mxfp4)
    set_tests_properties(test-hrx-mxfp4 PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(test-hrx-nvfp4 test-hrx-nvfp4.cpp)
    target_link_libraries(test-hrx-nvfp4 PRIVATE ggml-hrx ggml hrx::hrx loomc::loomc)
    target_include_directories(test-hrx-nvfp4 PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-nvfp4 COMMAND test-hrx-nvfp4)
    set_tests_properties(test-hrx-nvfp4 PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(test-hrx-q2_0 test-hrx-q2_0.cpp)
    target_link_libraries(test-hrx-q2_0 PRIVATE ggml-hrx ggml hrx::hrx loomc::loomc)
    target_include_directories(test-hrx-q2_0 PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-q2_0 COMMAND test-hrx-q2_0)
    set_tests_properties(test-hrx-q2_0 PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(test-hrx-get-rows-iq test-hrx-get-rows-iq.cpp)
    target_link_libraries(test-hrx-get-rows-iq PRIVATE ggml-hrx ggml hrx::hrx loomc::loomc)
    target_include_directories(test-hrx-get-rows-iq PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-get-rows-iq COMMAND test-hrx-get-rows-iq)
    set_tests_properties(test-hrx-get-rows-iq PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(test-hrx-decode-stride test-hrx-decode-stride.cpp)
    target_link_libraries(test-hrx-decode-stride PRIVATE ggml-hrx ggml hrx::hrx loomc::loomc)
    target_include_directories(test-hrx-decode-stride PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-decode-stride COMMAND test-hrx-decode-stride)
    set_tests_properties(test-hrx-decode-stride PROPERTIES SKIP_RETURN_CODE 77)

    # prefill FLASH_ATTN_EXT through the transposed V at 256+ query tokens: values against the CPU and the launch plan
    # (engine #1989); skips without HRX0
    add_executable(test-hrx-fa-transposed-value test-hrx-fa-transposed-value.cpp)
    target_link_libraries(test-hrx-fa-transposed-value PRIVATE ggml-hrx ggml hrx::hrx loomc::loomc)
    target_include_directories(test-hrx-fa-transposed-value PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-fa-transposed-value COMMAND test-hrx-fa-transposed-value)
    set_tests_properties(test-hrx-fa-transposed-value PROPERTIES SKIP_RETURN_CODE 77)

    # the fused Q projection at one token on head sizes that are not a power of two (engine #1926); skips without HRX0
    add_executable(test-hrx-attention-q-decode-head-size test-hrx-attention-q-decode-head-size.cpp)
    target_link_libraries(test-hrx-attention-q-decode-head-size PRIVATE ggml-hrx ggml hrx::hrx loomc::loomc)
    target_include_directories(test-hrx-attention-q-decode-head-size PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-attention-q-decode-head-size COMMAND test-hrx-attention-q-decode-head-size)
    set_tests_properties(test-hrx-attention-q-decode-head-size PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(test-hrx-mul-mat-id-k32 test-hrx-mul-mat-id-k32.cpp)
    target_link_libraries(test-hrx-mul-mat-id-k32 PRIVATE ggml-hrx ggml hrx::hrx loomc::loomc)
    target_include_directories(test-hrx-mul-mat-id-k32 PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-mul-mat-id-k32 COMMAND test-hrx-mul-mat-id-k32)
    set_tests_properties(test-hrx-mul-mat-id-k32 PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(test-hrx-moe-split test-hrx-moe-split.cpp)
    target_link_libraries(test-hrx-moe-split PRIVATE ggml)
    add_test(NAME test-hrx-moe-split COMMAND test-hrx-moe-split)
    set_tests_properties(test-hrx-moe-split PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(test-hrx-fa-masked-v test-hrx-fa-masked-v.cpp)
    target_link_libraries(test-hrx-fa-masked-v PRIVATE ggml-hrx ggml hrx::hrx loomc::loomc)
    target_include_directories(test-hrx-fa-masked-v PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-fa-masked-v COMMAND test-hrx-fa-masked-v)
    set_tests_properties(test-hrx-fa-masked-v PROPERTIES SKIP_RETURN_CODE 77)

    # the routed-down hook on llm.routed_ffn.decode_down_next_q8: plan and device kernel names, an fp64 oracle and the
    # CPU backend, with the add-on's kernel off (loom) and on (hip, only when a HIP add-on is built in)
    add_executable(test-hrx-routed-down-hip test-hrx-routed-down-hip.cpp)
    target_link_libraries(test-hrx-routed-down-hip PRIVATE ggml-hrx ggml hrx::hrx loomc::loomc)
    target_include_directories(test-hrx-routed-down-hip PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-routed-down-loom COMMAND test-hrx-routed-down-hip --arm loom)
    set_tests_properties(test-hrx-routed-down-loom PROPERTIES SKIP_RETURN_CODE 77)
    if (GGML_HRX_HIP_ADDON_DIR)
        add_test(NAME test-hrx-routed-down-hip COMMAND test-hrx-routed-down-hip --arm hip)
        set_tests_properties(test-hrx-routed-down-hip PROPERTIES SKIP_RETURN_CODE 77)
        # Q4_K gate on: returns 77 when the add-on has no Q4_K kernel
        add_test(NAME test-hrx-routed-down-hip-q4k COMMAND test-hrx-routed-down-hip --arm hip-q4k)
        set_tests_properties(test-hrx-routed-down-hip-q4k PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    add_executable(test-hrx-attention-sink test-hrx-attention-sink.cpp)
    target_link_libraries(test-hrx-attention-sink PRIVATE ggml-hrx ggml hrx::hrx loomc::loomc)
    target_include_directories(test-hrx-attention-sink PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    add_test(NAME test-hrx-attention-sink COMMAND test-hrx-attention-sink)
    add_executable(hrx-ops-test hrx-ops-test.cpp)
    target_link_libraries(hrx-ops-test PRIVATE ggml-hrx ggml hrx::hrx loomc::loomc)
    target_include_directories(hrx-ops-test PRIVATE ../ggml/include ../ggml/src ../ggml/src/ggml-hrx)
    if (GGML_HRX_HIP_ADDON_DIR)
        # engine#3488: five native-weight rows are known-fail in a no-add-on build and must pass here, so the test
        # needs the build's own answer rather than a runtime probe.
        target_compile_definitions(hrx-ops-test PRIVATE GGML_HRX_HIP_ADDON)
    endif()
    add_test(NAME hrx-ops-test COMMAND hrx-ops-test
        --hrx-test-jobs 4 --hrx-test-output all --hrx-test-filter ".*")

    # engine#2393: test-hrx-ops and test-hrx-buffer were missing from this list, and both print the same
    # "test skipped: no device available" line. Measured on a machine with the runtime unset: test-hrx-ops exited
    # **0** while skipping 267 of its 269 cases -- a green row for a suite that never touched a device. Their four
    # siblings fail in that state, which is the right answer for a row whose subject is the hardware.
    set_tests_properties(hrx-buffer-test hrx-backend-test hrx-loom-jit-test hrx-ops-test
                         test-hrx-ops test-hrx-buffer test-hrx-ops-loom PROPERTIES
        LABELS hrx
        TIMEOUT 900
        FAIL_REGULAR_EXPRESSION "test skipped: no device available")

    # engine #2146: TOP_K on HRX gives the same indices in the same order every run
    add_executable(test-hrx-top-k-repeat test-hrx-top-k-repeat.cpp)
    target_link_libraries(test-hrx-top-k-repeat PRIVATE ggml)
    target_include_directories(test-hrx-top-k-repeat PRIVATE ../ggml/include)
    add_test(NAME test-hrx-top-k-repeat COMMAND test-hrx-top-k-repeat)
    set_tests_properties(test-hrx-top-k-repeat PROPERTIES SKIP_RETURN_CODE 77)
    # the same cases with the kernel switched off: TOP_K must fall back to the CPU, not fail the graph (engine #2146)
    add_test(NAME test-hrx-top-k-declined COMMAND test-hrx-top-k-repeat)
    set_tests_properties(test-hrx-top-k-declined PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(test-hrx-top-k-declined PROPERTIES ENVIRONMENT "GGML_HRX_DISABLE_DISPATCH=top_k")
endif()
