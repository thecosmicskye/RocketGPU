#pragma once

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(ROCKETGPU_USE_CUDA)
#include <cuda_runtime.h>
#define ROCKETGPU_HD __host__ __device__ inline
#define ROCKETGPU_DEVICE __device__ inline
#define ROCKETGPU_GLOBAL __global__
#define ROCKETGPU_FORCEINLINE __forceinline__
#else
using cudaStream_t = void*;
#define ROCKETGPU_HD inline
#define ROCKETGPU_DEVICE inline
#define ROCKETGPU_GLOBAL inline
#define ROCKETGPU_FORCEINLINE inline
#endif

// Added to a function already declared ROCKETGPU_HD: inline into every device caller.
#if defined(__CUDACC__)
#define ROCKETGPU_ALWAYS_INLINE __attribute__((always_inline))
#else
#define ROCKETGPU_ALWAYS_INLINE
#endif

// Index of the lowest set bit of a nonzero mask.
ROCKETGPU_HD int CountTrailingZeros64(uint64_t mask) {
#if defined(__CUDA_ARCH__)
    return __ffsll(static_cast<long long>(mask)) - 1;
#else
    int bit = 0;
    while (((mask >> bit) & 1u) == 0) {
        ++bit;
    }
    return bit;
#endif
}

#if defined(__CUDA_ARCH__)
#define ROCKETGPU_WARP_SYNC() __syncwarp()
#else
#define ROCKETGPU_WARP_SYNC() do { } while (0)
#endif

namespace rocketgpu {
#if defined(ROCKETGPU_USE_CUDA)

inline void CheckCuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

#else

inline void CheckCuda(int           , const char*         ) {}

#endif

ROCKETGPU_HD constexpr float Pi() { return 3.14159265358979323846f; }
ROCKETGPU_HD constexpr float HalfPi() { return 1.57079632679489661923f; }
ROCKETGPU_HD constexpr float SqrtHalf() { return 0.7071067811865475244f; }

ROCKETGPU_HD constexpr float Clamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Local array whose slots are not default-constructed: for scratch whose every read slot is assigned first.
template <typename T, int N>
struct UninitArray {
    alignas(T) unsigned char bytes[N * sizeof(T)];
    ROCKETGPU_HD T* data() { return reinterpret_cast<T*>(bytes); }
    ROCKETGPU_HD operator T*() { return data(); }
};

ROCKETGPU_HD constexpr float Sign(float v) {
    return (v > 0.0f) - (v < 0.0f);
}

ROCKETGPU_HD constexpr float Abs(float v) {
    return v >= 0.0f ? v : -v;
}

ROCKETGPU_HD constexpr float Min(float a, float b) {
    return a < b ? a : b;
}

ROCKETGPU_HD constexpr float Max(float a, float b) {
    return a > b ? a : b;
}

ROCKETGPU_HD constexpr float Lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

ROCKETGPU_HD constexpr float WrapAngle(float a) {
    while (a > Pi()) a -= 2.0f * Pi();
    while (a < -Pi()) a += 2.0f * Pi();
    return a;
}
}

