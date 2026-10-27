#pragma once
#include "rocketgpu/detail/math.hpp"

namespace rocketgpu::detail {
struct NativeSseSphereNormalizer {
    const uint32_t* table;
    int shift;

    ROCKETGPU_HD Vec3 operator()(Vec3 value) const {
        constexpr float eps = 1.1920929e-7f;
        float length2 = Dot(value, value);
        if (length2 < eps * eps) { value = Vec3(-1, -1, -1); length2 = 3; }
        const uint32_t bits = GlibcFloatBits(length2);
        const int exponent = (bits >> 23) & 255;
        const int parity = 1 - (exponent & 1);
        const auto index = (parity << (23 - shift)) + ((bits & 0x7fffff) >> shift);
        const int adjustment = (exponent - (127 + parity)) / 2;
        const float estimate = GlibcBitsToFloat(static_cast<uint32_t>(
            int64_t(table[index]) - adjustment * int64_t(1 << 23)));
#if defined(__CUDA_ARCH__)
        const float first = __fmul_rn(length2, 0.5f);
        const float second = __fmul_rn(first, estimate);
        const float third = __fmul_rn(second, estimate);
        const float scale = __fmul_rn(estimate, __fsub_rn(1.5f, third));
        return Vec3(__fmul_rn(value.x, scale), __fmul_rn(value.y, scale), __fmul_rn(value.z, scale));
#else
        const float first = length2 * 0.5f;
        const float second = first * estimate;
        const float third = second * estimate;
        return value * (estimate * (1.5f - third));
#endif
    }
};

ROCKETGPU_HD Vec3 NativeSseClampLength(Vec3 value, float maximum,
    const uint32_t* table, int shift) {
    if (!table) return ClampLength(value, maximum);
    if (LengthSq(value) <= maximum * maximum) return value;
    return NativeSseSphereNormalizer{table, shift}(value) * maximum;
}

ROCKETGPU_HD bool NativeSseMath(const MutatorConfig& mutator) {
    return mutator.native_sphere_rsqrt_table != nullptr;
}
}
