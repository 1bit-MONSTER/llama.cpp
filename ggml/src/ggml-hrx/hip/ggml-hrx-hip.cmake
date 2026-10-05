# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# HIP kernels for the HRX backend: every hip/kernels/*.hip is compiled by amdclang++ into one raw
# code object per target in GGML_HRX_HIP_TARGETS and embedded in libggml-hrx. Nothing here links
# the HIP runtime: HRX loads and dispatches the code objects. See hip/README.md.

set(GGML_HRX_HIP_TARGETS "gfx1151" CACHE STRING "GPU targets for the HRX HIP kernels (semicolon list)")
set(GGML_HRX_HIP_FLAGS "-O3" CACHE STRING "Extra amdclang++ flags for the HRX HIP kernels")
set(GGML_HRX_HIP_ADDON_DIR "" CACHE PATH
    "Optional HIP kernel add-on directory: <dir>/kernels/*.hip join the code objects, <dir>/*.cpp join ggml-hrx, <dir>/addon.cmake is included if present (hip/README.md)")

set(_ggml_hrx_hip_default_compiler "")
if (CMAKE_CXX_COMPILER MATCHES "amdclang\\+\\+$")
    set(_ggml_hrx_hip_default_compiler "${CMAKE_CXX_COMPILER}")
endif()
find_program(GGML_HRX_HIP_COMPILER NAMES amdclang++
    HINTS ${_ggml_hrx_hip_default_compiler} /opt/rocm-therock/bin $ENV{ROCM_PATH}/bin /opt/rocm/bin
    DOC "amdclang++ used to compile the HRX HIP kernels")
if (_ggml_hrx_hip_default_compiler AND NOT GGML_HRX_HIP_COMPILER)
    set(GGML_HRX_HIP_COMPILER "${_ggml_hrx_hip_default_compiler}")
endif()
if (NOT GGML_HRX_HIP_COMPILER)
    message(FATAL_ERROR "GGML_HRX needs amdclang++ for its HIP kernels; set GGML_HRX_HIP_COMPILER")
endif()

file(GLOB GGML_HRX_HIP_KERNEL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/hip/kernels/*.hip")

# Add-on: its kernels are built and embedded exactly like the ones above, its matchers are compiled into
# ggml-hrx, and register_hip_dispatches calls the ggml_hrx_hip_addon_register it defines.
set(GGML_HRX_HIP_ADDON_SOURCES)
if (GGML_HRX_HIP_ADDON_DIR)
    get_filename_component(GGML_HRX_HIP_ADDON_PATH "${GGML_HRX_HIP_ADDON_DIR}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    if (NOT IS_DIRECTORY "${GGML_HRX_HIP_ADDON_PATH}/kernels")
        message(FATAL_ERROR "GGML_HRX_HIP_ADDON_DIR=${GGML_HRX_HIP_ADDON_DIR}: no kernels/ directory")
    endif()
    file(GLOB _ggml_hrx_hip_addon_kernels CONFIGURE_DEPENDS "${GGML_HRX_HIP_ADDON_PATH}/kernels/*.hip")
    file(GLOB GGML_HRX_HIP_ADDON_SOURCES CONFIGURE_DEPENDS "${GGML_HRX_HIP_ADDON_PATH}/*.cpp" "${GGML_HRX_HIP_ADDON_PATH}/*.h")
    list(APPEND GGML_HRX_HIP_KERNEL_SOURCES ${_ggml_hrx_hip_addon_kernels})
    list(LENGTH _ggml_hrx_hip_addon_kernels _ggml_hrx_hip_addon_kernel_count)
    message(STATUS "GGML_HRX: HIP kernel add-on ${GGML_HRX_HIP_ADDON_PATH} (${_ggml_hrx_hip_addon_kernel_count} kernels)")
endif()

# Code objects are looked up by file stem, so stems must be unique across this directory and the add-on.
set(_ggml_hrx_hip_stems)
foreach(_source ${GGML_HRX_HIP_KERNEL_SOURCES})
    get_filename_component(_stem "${_source}" NAME_WE)
    if (_stem IN_LIST _ggml_hrx_hip_stems)
        message(FATAL_ERROR "GGML_HRX: two HIP kernels named '${_stem}' (${_source})")
    endif()
    list(APPEND _ggml_hrx_hip_stems "${_stem}")
endforeach()

separate_arguments(_ggml_hrx_hip_flags UNIX_COMMAND "${GGML_HRX_HIP_FLAGS}")
set(_ggml_hrx_hip_dir "${CMAKE_CURRENT_BINARY_DIR}/hip-code-objects")
file(MAKE_DIRECTORY "${_ggml_hrx_hip_dir}")

set(_ggml_hrx_hip_objects)
set(_ggml_hrx_hip_entries)
foreach(_source ${GGML_HRX_HIP_KERNEL_SOURCES})
    get_filename_component(_stem "${_source}" NAME_WE)
    get_filename_component(_source_dir "${_source}" DIRECTORY)
    foreach(_target ${GGML_HRX_HIP_TARGETS})
        set(_object "${_ggml_hrx_hip_dir}/${_stem}.${_target}.hsaco")
        add_custom_command(
            OUTPUT "${_object}"
            COMMAND "${GGML_HRX_HIP_COMPILER}" -x hip -std=c++17 --offload-arch=${_target} --cuda-device-only
                    --no-gpu-bundle-output ${_ggml_hrx_hip_flags}
                    -I "${_source_dir}" -I "${CMAKE_CURRENT_SOURCE_DIR}/hip/kernels"
                    -MD -MF "${_object}.d" -o "${_object}" "${_source}"
            DEPENDS "${_source}"
            DEPFILE "${_object}.d"
            COMMENT "HIP code object ${_stem} (${_target})"
            VERBATIM)
        list(APPEND _ggml_hrx_hip_objects "${_object}")
        list(APPEND _ggml_hrx_hip_entries "${_stem}:${_target}:${_object}")
    endforeach()
endforeach()

set(GGML_HRX_HIP_CODE_OBJECTS_INC "${CMAKE_CURRENT_BINARY_DIR}/ggml-hrx-hip-code-objects.inc")
add_custom_command(
    OUTPUT "${GGML_HRX_HIP_CODE_OBJECTS_INC}"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/hip/embed_hip_code_objects.py"
            "${GGML_HRX_HIP_CODE_OBJECTS_INC}" ${_ggml_hrx_hip_entries}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/hip/embed_hip_code_objects.py" ${_ggml_hrx_hip_objects}
    COMMENT "Embedding HRX HIP code objects"
    VERBATIM)
add_custom_target(ggml-hrx-hip-code-objects DEPENDS "${GGML_HRX_HIP_CODE_OBJECTS_INC}")

# Code objects + kernel registry live with the kernel corpus (resolve_kernel_definition finds HIP
# kernels there); the loader that feeds the executable cache lives in ggml-hrx.
target_sources(ggml-hrx-kernel-corpus PRIVATE
    hip/hip-code-objects.cpp
    hip/hip-code-objects.h
    hip/hip-kernel-registry.cpp
    hip/hip-kernel-registry.h
    "${GGML_HRX_HIP_CODE_OBJECTS_INC}")
add_dependencies(ggml-hrx-kernel-corpus ggml-hrx-hip-code-objects)
target_sources(ggml-hrx PRIVATE
    hip/hip-kernel-loader.cpp
    hip/hip-kernel-loader.h
    hip/hip-capabilities.cpp
    hip/hip-capabilities.h
    hip/hip-dispatches.cpp
    hip/hip-dispatches.h
    hip/dispatch-hip-scale.cpp
    ${GGML_HRX_HIP_ADDON_SOURCES})
if (GGML_HRX_HIP_ADDON_DIR)
    target_compile_definitions(ggml-hrx PRIVATE GGML_HRX_HIP_ADDON)
endif()

# Standalone check that HRX loads and runs a hipcc code object (hip/kernels/hip_smoke.hip).
add_executable(ggml-hrx-hip-smoke hip/hip-smoke.cpp hip/hip-code-objects.cpp)
target_link_libraries(ggml-hrx-hip-smoke PRIVATE hrx::hrx)
target_include_directories(ggml-hrx-hip-smoke PRIVATE . "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(ggml-hrx-hip-smoke PRIVATE cxx_std_17)
add_dependencies(ggml-hrx-hip-smoke ggml-hrx-hip-code-objects)

# Add-on extras (standalone harnesses, benches); paths in it resolve as in this file.
if (GGML_HRX_HIP_ADDON_DIR)
    include("${GGML_HRX_HIP_ADDON_PATH}/addon.cmake" OPTIONAL)
endif()
