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
#if defined(_MSC_VER)
#include <intrin.h>
#endif

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

#if defined(__CUDACC__) && defined(_MSC_VER)
#define ROCKETGPU_ALWAYS_INLINE __forceinline__
#elif defined(__CUDACC__) || (!defined(ROCKETGPU_USE_CUDA) && defined(__GNUC__))
#define ROCKETGPU_ALWAYS_INLINE __attribute__((always_inline))
#else
#define ROCKETGPU_ALWAYS_INLINE
#endif

#if !defined(ROCKETGPU_USE_CUDA) && defined(__GNUC__)
#define ROCKETGPU_HOST_INLINE __attribute__((always_inline))
#else
#define ROCKETGPU_HOST_INLINE
#endif

#if !defined(ROCKETGPU_USE_CUDA) && defined(__GNUC__)
#define ROCKETGPU_HOST_NOINLINE __attribute__((noinline))
#define ROCKETGPU_HOST_COLD __attribute__((noinline, cold))
#else
#define ROCKETGPU_HOST_NOINLINE
#define ROCKETGPU_HOST_COLD
#endif

#if defined(__CUDACC__)
#define ROCKETGPU_RESTRICT __restrict__
#else
#define ROCKETGPU_RESTRICT
#endif

#ifndef ROCKETGPU_NOINLINE
#if defined(__CUDA_ARCH__)
#define ROCKETGPU_NOINLINE __noinline__
#else
#define ROCKETGPU_NOINLINE
#endif
#endif

ROCKETGPU_HD int CountTrailingZeros32(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __ffs(static_cast<int>(x)) - 1;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(x);
#else
    int n = 0;
    while ((x & 1u) == 0u) {
        x >>= 1;
        ++n;
    }
    return n;
#endif
}

ROCKETGPU_HD int CountTrailingZeros64(uint64_t x) {
#if defined(__CUDA_ARCH__)
    return __ffsll(static_cast<long long>(x)) - 1;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#else
    int n = 0;
    while ((x & 1u) == 0u) {
        x >>= 1;
        ++n;
    }
    return n;
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

ROCKETGPU_HD constexpr float WrapAngle(float a) {
    while (a > Pi()) a -= 2.0f * Pi();
    while (a < -Pi()) a += 2.0f * Pi();
    return a;
}
}
