cmake_minimum_required(VERSION 3.21)
project(RocketGPU LANGUAGES CXX)

option(ROCKETGPU_USE_CUDA "Build CUDA kernels" OFF)
option(ROCKETGPU_BUILD_PYTHON "Build Python bindings" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
    src/rocketgpu.cpp)
set_target_properties(rocketgpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_options(rocketgpu PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -ffp-contract=off>
)


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
