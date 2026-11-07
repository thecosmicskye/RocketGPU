find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
if (ROCKETGPU_PGO STREQUAL "GENERATE")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/pgo-python.txt" "${Python_EXECUTABLE}\n${CMAKE_CXX_COMPILER_ID}\n${CMAKE_CXX_COMPILER}\n")
endif()
if (NOT DEFINED pybind11_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _rocketgpu_pybind11_cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if (_rocketgpu_pybind11_cmakedir)
        set(pybind11_DIR "${_rocketgpu_pybind11_cmakedir}" CACHE PATH "pybind11 CMake directory")
    endif()
endif()
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(rocketgpu_py NO_EXTRAS rocketgpu_bindings.cpp)
target_link_libraries(rocketgpu_py PRIVATE rocketgpu)
target_compile_definitions(rocketgpu_py PRIVATE ROCKETGPU_VERSION="${PROJECT_VERSION}")
target_include_directories(rocketgpu_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_compile_features(rocketgpu_py PRIVATE cxx_std_17)
set_target_properties(rocketgpu_py PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION OFF
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON)

add_custom_command(TARGET rocketgpu_py POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/rocketsim
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/rocketsim/__init__.py
        ${CMAKE_CURRENT_BINARY_DIR}/rocketsim/__init__.py
    COMMENT "Copying the rocketsim top-level package next to rocketgpu_py")
add_custom_command(TARGET rocketgpu_py POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/rocketgpu_py_cuda.py
        ${CMAKE_CURRENT_BINARY_DIR}/rocketgpu_py_cuda.py
    COMMENT "Copying rocketgpu_py_cuda.py next to rocketgpu_py")
