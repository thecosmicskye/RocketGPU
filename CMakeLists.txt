cmake_minimum_required(VERSION 3.21)
project(RocketGPU LANGUAGES CXX)

option(ROCKETGPU_USE_CUDA "Build CUDA kernels" OFF)
option(ROCKETGPU_BUILD_PYTHON "Build Python bindings" OFF)
option(ROCKETGPU_NATIVE_ARCH "CPU builds: tune for this machine (-march=native / -mcpu=native); OFF for binaries that run on other machines" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_program(ROCKETGPU_CCACHE_EXECUTABLE ccache)
if (ROCKETGPU_CCACHE_EXECUTABLE)
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
    find_package(CUDAToolkit REQUIRED)
endif()

add_library(rocketgpu STATIC)
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

if (ROCKETGPU_USE_CUDA)
    target_compile_definitions(rocketgpu PUBLIC ROCKETGPU_USE_CUDA=1)
    target_sources(rocketgpu PRIVATE src/rocketgpu_cuda.cu)
    target_link_libraries(rocketgpu PUBLIC CUDA::cudart)
    target_compile_options(rocketgpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    set_target_properties(rocketgpu PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

if (ROCKETGPU_BUILD_PYTHON)
    add_subdirectory(python)
endif()
