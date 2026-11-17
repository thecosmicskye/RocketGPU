add_library(rocketgpu_rs STATIC shim.cpp)
target_link_libraries(rocketgpu_rs PRIVATE rocketgpu)

add_executable(rocketgpu_rs_gen gen_presets.cpp)
target_link_libraries(rocketgpu_rs_gen PRIVATE rocketgpu)
set(_rocketgpu_rs_presets ${CMAKE_CURRENT_BINARY_DIR}/presets.rs)
add_custom_command(
    OUTPUT ${_rocketgpu_rs_presets}
    COMMAND rocketgpu_rs_gen ${_rocketgpu_rs_presets}
    DEPENDS rocketgpu_rs_gen
    VERBATIM)
set_target_properties(rocketgpu_rs rocketgpu_rs_gen PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CUDA_STANDARD 20 CUDA_RESOLVE_DEVICE_SYMBOLS ON)
if (ROCKETGPU_USE_CUDA)
    set_source_files_properties(shim.cpp gen_presets.cpp PROPERTIES LANGUAGE CUDA)
endif()
add_custom_target(rocketgpu_rs_presets DEPENDS ${_rocketgpu_rs_presets})
add_dependencies(rocketgpu_rs rocketgpu_rs_presets)
