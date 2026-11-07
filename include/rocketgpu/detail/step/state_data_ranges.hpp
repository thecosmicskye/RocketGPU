#pragma once
#include <cstdint>
#include <cstring>
namespace rocketgpu::detail {
struct StateDataRange { uint16_t off, len; };
static_assert(sizeof(rocketgpu::CarState) == 880, "run tools/gen_ispc_types.py");
inline constexpr StateDataRange kCarStateData[] = {{0, 139}, {140, 40}, {184, 79}, {264, 148}, {416, 74}, {492, 117}, {612, 50}, {664, 93}, {760, 113}};
static_assert(sizeof(rocketgpu::BallState) == 288, "run tools/gen_ispc_types.py");
inline constexpr StateDataRange kBallStateData[] = {{0, 12}, {16, 272}};
static_assert(sizeof(rocketgpu::detail::WheelRuntimeState) == 292, "run tools/gen_ispc_types.py");
inline constexpr StateDataRange kWheelRuntimeStateData[] = {{0, 253}, {256, 34}};
template <int N> inline bool StateDataEqual(const void* a, const void* b, const StateDataRange (&r)[N]) {
    for (const StateDataRange& x : r)
        if (std::memcmp(static_cast<const char*>(a) + x.off, static_cast<const char*>(b) + x.off, x.len) != 0) return false;
    return true;
}
}
