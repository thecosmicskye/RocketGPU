cmake_minimum_required(VERSION 3.21)
project(RocketGPU VERSION 1.3.0 LANGUAGES CXX)

option(ROCKETGPU_USE_CUDA "Build CUDA kernels" OFF)
option(ROCKETGPU_BUILD_PYTHON "Build Python bindings" OFF)
option(ROCKETGPU_BUILD_BENCH "Build tools/bench_cpp (throughput benchmark)" OFF)
if (MSVC)
    set(_rocketgpu_cuda_split_default OFF)
else()
    set(_rocketgpu_cuda_split_default ON)
endif()
option(ROCKETGPU_CUDA_SPLIT "CUDA builds: compile the kernels as seven translation units (tools/split_cuda.py, needs Python 3 at configure time); OFF: one file. Off by default with MSVC, where the split has not been built" ${_rocketgpu_cuda_split_default})
set(ROCKETGPU_OBJECT_CACHE "" CACHE PATH "Dev builds: directory of reusable objects (tools/obj_cache.py); empty for none")
option(ROCKETGPU_NATIVE_ARCH "CPU builds: tune for this machine (-march=native / -mcpu=native); OFF for binaries that run on other machines" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (ROCKETGPU_OBJECT_CACHE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(_rocketgpu_obj_cache ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/obj_cache.py
        --root ${CMAKE_CURRENT_SOURCE_DIR} --cache ${ROCKETGPU_OBJECT_CACHE} --)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${_rocketgpu_obj_cache})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${_rocketgpu_obj_cache})
endif()
find_program(ROCKETGPU_CCACHE_EXECUTABLE ccache)
if (ROCKETGPU_CCACHE_EXECUTABLE AND NOT ROCKETGPU_OBJECT_CACHE)
    if (NOT DEFINED CMAKE_CXX_COMPILER_LAUNCHER)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${ROCKETGPU_CCACHE_EXECUTABLE}" CACHE STRING "CXX compiler launcher")
    endif()
    if (ROCKETGPU_USE_CUDA AND NOT DEFINED CMAKE_CUDA_COMPILER_LAUNCHER)
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${ROCKETGPU_CCACHE_EXECUTABLE}" CACHE STRING "CUDA compiler launcher")
    endif()
endif()

if (ROCKETGPU_USE_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    find_program(ROCKETGPU_NVIDIA_SMI_EXECUTABLE nvidia-smi)
    if (ROCKETGPU_NVIDIA_SMI_EXECUTABLE)
        execute_process(
            COMMAND ${ROCKETGPU_NVIDIA_SMI_EXECUTABLE} --query-gpu=compute_cap --format=csv,noheader,nounits
            OUTPUT_VARIABLE _rocketgpu_cuda_compute_caps
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        string(REGEX MATCH "[0-9]+\\.[0-9]+" _rocketgpu_first_cuda_compute_cap "${_rocketgpu_cuda_compute_caps}")
        if (_rocketgpu_first_cuda_compute_cap)
            string(REPLACE "." "" _rocketgpu_cuda_arch "${_rocketgpu_first_cuda_compute_cap}")
            set(CMAKE_CUDA_ARCHITECTURES "${_rocketgpu_cuda_arch}" CACHE STRING "CUDA architectures for RocketGPU" FORCE)
        endif()
    endif()
endif()

if (ROCKETGPU_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit 12.1 REQUIRED)
endif()

set(ROCKETGPU_PGO OFF CACHE STRING "CPU builds: profile-guided optimization (OFF, GENERATE, USE); see tools/pgo_build.sh")
set_property(CACHE ROCKETGPU_PGO PROPERTY STRINGS OFF GENERATE USE)
set(ROCKETGPU_PGO_PROFILE "" CACHE PATH "GENERATE: profile output directory. USE: .profdata file (Clang) or that directory (GCC)")
if (ROCKETGPU_PGO AND NOT ROCKETGPU_PGO STREQUAL "OFF")
    if (ROCKETGPU_USE_CUDA OR NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" OR NOT ROCKETGPU_PGO_PROFILE)
        message(FATAL_ERROR "ROCKETGPU_PGO needs a CPU build with GCC or Clang and ROCKETGPU_PGO_PROFILE")
    endif()
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(_rocketgpu_pgo_common -fprofile-prefix-path=${CMAKE_BINARY_DIR})
        if (ROCKETGPU_PGO STREQUAL "GENERATE")
            set(_rocketgpu_pgo -fprofile-generate=${ROCKETGPU_PGO_PROFILE} -fprofile-update=atomic)
        else()
            set(_rocketgpu_pgo -fprofile-use=${ROCKETGPU_PGO_PROFILE} -fprofile-correction -Wno-missing-profile -Wno-error=coverage-mismatch
                -fno-unroll-loops -fno-peel-loops -fno-tracer)
        endif()
    elseif (ROCKETGPU_PGO STREQUAL "GENERATE")
        set(_rocketgpu_pgo -fprofile-generate=${ROCKETGPU_PGO_PROFILE})
    else()
        set(_rocketgpu_pgo -fprofile-use=${ROCKETGPU_PGO_PROFILE} -Wno-profile-instr-unprofiled -Wno-profile-instr-out-of-date)
    endif()
    add_link_options(${_rocketgpu_pgo})
endif()

add_library(rocketgpu STATIC)
if (_rocketgpu_pgo)
    target_compile_options(rocketgpu PRIVATE ${_rocketgpu_pgo_common} ${_rocketgpu_pgo})
endif()
if (APPLE AND NOT ROCKETGPU_USE_CUDA AND NOT CMAKE_CROSSCOMPILING
    AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64"
    AND NOT CMAKE_OSX_ARCHITECTURES MATCHES ";|x86_64"
    AND NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION
    AND NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE)
    set_property(TARGET rocketgpu PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()
target_include_directories(rocketgpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_sources(rocketgpu PRIVATE
    src/arena/mesh_loader.cpp
    src/rocketsim_compat.cpp
    src/rocketsim_compat_serialize.cpp
    src/rocketsim_compat_dev.cpp
    src/rocketsim_compat_devbatch.cpp
    src/rocketgpu.cpp)
set_target_properties(rocketgpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)
target_link_libraries(rocketgpu PUBLIC Threads::Threads)

target_compile_options(rocketgpu PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -ffp-contract=off>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Wpedantic>
)
if (NOT ROCKETGPU_USE_CUDA AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(rocketgpu PRIVATE -fno-math-errno)
endif()
set(_rocketgpu_cxx_flags "${CMAKE_CXX_FLAGS}")
foreach(_rocketgpu_config ${CMAKE_BUILD_TYPE} ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER "${_rocketgpu_config}" _rocketgpu_config_upper)
    string(APPEND _rocketgpu_cxx_flags " ${CMAKE_CXX_FLAGS_${_rocketgpu_config_upper}}")
endforeach()
if (ROCKETGPU_NATIVE_ARCH AND NOT ROCKETGPU_USE_CUDA AND NOT CMAKE_CROSSCOMPILING
    AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"
    AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT _rocketgpu_cxx_flags MATCHES "-march=|-mcpu=")
    target_compile_options(rocketgpu PRIVATE -march=native)
elseif (ROCKETGPU_NATIVE_ARCH AND NOT ROCKETGPU_USE_CUDA AND NOT CMAKE_CROSSCOMPILING
    AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64"
    AND NOT CMAKE_OSX_ARCHITECTURES MATCHES ";|x86_64"
    AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang" AND NOT _rocketgpu_cxx_flags MATCHES "-mcpu=|-march=")
    target_compile_options(rocketgpu PRIVATE -mcpu=native)
endif()
if (NOT ROCKETGPU_USE_CUDA AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(rocketgpu PRIVATE -fno-semantic-interposition
        --param=max-inline-insns-single=2000 --param=max-inline-insns-auto=300
        --param=large-function-growth=800 --param=inline-unit-growth=400)
elseif (NOT ROCKETGPU_USE_CUDA AND NOT APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(rocketgpu PRIVATE -fno-semantic-interposition)
endif()

if (NOT ROCKETGPU_USE_CUDA)
    find_program(ROCKETGPU_ISPC_EXECUTABLE ispc)
endif()
option(ROCKETGPU_CPU_ISPC "CPU builds: SIMD kernels across cars (needs ispc)" ON)
set(_rocketgpu_ispc_portable_x86 OFF)
if (NOT ROCKETGPU_NATIVE_ARCH AND NOT ROCKETGPU_ISPC_TARGET AND
    (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" OR CMAKE_OSX_ARCHITECTURES MATCHES "x86_64"))
    set(_rocketgpu_ispc_portable_x86 ON)
endif()
if (NOT ROCKETGPU_USE_CUDA AND ROCKETGPU_CPU_ISPC AND ROCKETGPU_ISPC_EXECUTABLE AND
    (_rocketgpu_ispc_portable_x86 OR CMAKE_OSX_ARCHITECTURES MATCHES ";"))
    message(STATUS "RocketGPU CPU ISPC skipped for portable x86 or universal build")
endif()
if (NOT ROCKETGPU_USE_CUDA AND ROCKETGPU_CPU_ISPC AND ROCKETGPU_ISPC_EXECUTABLE AND
    NOT _rocketgpu_ispc_portable_x86 AND NOT CMAKE_OSX_ARCHITECTURES MATCHES ";")
    if (NOT ROCKETGPU_ISPC_TARGET)
        if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            set(ROCKETGPU_ISPC_TARGET neon-i32x4)
        elseif (ROCKETGPU_NATIVE_ARCH AND EXISTS /proc/cpuinfo)
            file(STRINGS /proc/cpuinfo _rocketgpu_avx512 REGEX "avx512f" LIMIT_COUNT 1)
            if (_rocketgpu_avx512)
                set(ROCKETGPU_ISPC_TARGET avx512skx-x16)
            else()
                set(ROCKETGPU_ISPC_TARGET avx2-i32x8)
            endif()
        else()
            set(ROCKETGPU_ISPC_TARGET avx2-i32x8)
        endif()
    endif()
    file(REAL_PATH "${ROCKETGPU_ISPC_EXECUTABLE}" _rocketgpu_ispc_real)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ROCKETGPU_ISPC_EXECUTABLE}" "${_rocketgpu_ispc_real}")
    execute_process(COMMAND "${ROCKETGPU_ISPC_EXECUTABLE}" --help
                    RESULT_VARIABLE _rocketgpu_ispc_help_result
                    OUTPUT_VARIABLE _rocketgpu_ispc_help ERROR_VARIABLE _rocketgpu_ispc_help_error)
    set(_rocketgpu_ispc_export_flag)
    set(_rocketgpu_ispc_suffix)
    if(_rocketgpu_ispc_help_result STREQUAL "0" AND
       "${_rocketgpu_ispc_help}${_rocketgpu_ispc_help_error}" MATCHES "(^|[^A-Za-z0-9_-])--no-internal-export-functions([^A-Za-z0-9_-]|$)")
        set(_rocketgpu_ispc_export_flag --no-internal-export-functions)
        set(_rocketgpu_ispc_suffix _external_only)
    endif()
    set(_rocketgpu_ispc_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_ispc)
    set(_rocketgpu_ispc_obj ${CMAKE_CURRENT_BINARY_DIR}/rg_pretick_ispc${_rocketgpu_ispc_suffix}${CMAKE_CXX_OUTPUT_EXTENSION})
    add_custom_command(
        OUTPUT ${_rocketgpu_ispc_obj}
        COMMAND ${ROCKETGPU_ISPC_EXECUTABLE} --target=${ROCKETGPU_ISPC_TARGET} -O2 --opt=disable-fma
                --enable-llvm-intrinsics ${_rocketgpu_ispc_export_flag} --pic -I ${_rocketgpu_ispc_dir} ${_rocketgpu_ispc_dir}/rg_pretick.ispc
                -o ${_rocketgpu_ispc_obj}
        DEPENDS "${ROCKETGPU_ISPC_EXECUTABLE}" "${_rocketgpu_ispc_real}" ${_rocketgpu_ispc_dir}/rg_pretick.ispc ${_rocketgpu_ispc_dir}/rg_math.isph ${_rocketgpu_ispc_dir}/rg_types.isph ${_rocketgpu_ispc_dir}/rg_layout.isph
        VERBATIM)
    target_sources(rocketgpu PRIVATE ${_rocketgpu_ispc_obj})
    target_compile_definitions(rocketgpu PUBLIC ROCKETGPU_CPU_ISPC=1)
    message(STATUS "RocketGPU CPU ISPC kernels: ${ROCKETGPU_ISPC_TARGET}")
endif()

if (ROCKETGPU_USE_CUDA)
    target_compile_definitions(rocketgpu PUBLIC ROCKETGPU_USE_CUDA=1)
    set(_rocketgpu_cuda_split ${ROCKETGPU_CUDA_SPLIT})
    if (_rocketgpu_cuda_split)
        find_package(Python3 COMPONENTS Interpreter)
        if (NOT Python3_Interpreter_FOUND)
            message(WARNING "ROCKETGPU_CUDA_SPLIT needs Python 3 at configure time (tools/split_cuda.py); "
                            "building the kernels as one file")
            set(_rocketgpu_cuda_split OFF)
        endif()
    endif()
    if (_rocketgpu_cuda_split)
        set(_rocketgpu_cuda_split_dir ${CMAKE_CURRENT_BINARY_DIR}/cuda_split)
        set(_rocketgpu_cuda_split_srcs)
        foreach(_m host step post world rows rays misc)
            list(APPEND _rocketgpu_cuda_split_srcs ${_rocketgpu_cuda_split_dir}/rocketgpu_cuda_${_m}.cu)
        endforeach()
        add_custom_command(
            OUTPUT ${_rocketgpu_cuda_split_srcs}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${_rocketgpu_cuda_split_dir}
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/split_cuda.py
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/rocketgpu_cuda.cu ${_rocketgpu_cuda_split_dir}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/rocketgpu_cuda.cu ${CMAKE_CURRENT_SOURCE_DIR}/tools/split_cuda.py
            VERBATIM)
        target_sources(rocketgpu PRIVATE ${_rocketgpu_cuda_split_srcs})
    else()
        target_sources(rocketgpu PRIVATE src/rocketgpu_cuda.cu)
    endif()
    target_link_libraries(rocketgpu PUBLIC CUDA::cudart)
    target_compile_options(rocketgpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    set_target_properties(rocketgpu PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON)
endif()

if (ROCKETGPU_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if (ROCKETGPU_BUILD_BENCH)
    add_executable(bench_cpp tools/bench_cpp.cpp)
    target_link_libraries(bench_cpp PRIVATE rocketgpu)
    set_target_properties(bench_cpp PROPERTIES CXX_STANDARD 20 CUDA_STANDARD 20 CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    if (ROCKETGPU_USE_CUDA)
        set_source_files_properties(tools/bench_cpp.cpp PROPERTIES LANGUAGE CUDA)
    endif()
endif()
