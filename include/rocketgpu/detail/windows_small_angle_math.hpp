#pragma once
#include "rocketgpu/detail/common.hpp"

namespace rocketgpu { namespace detail {
ROCKETGPU_HD float WindowsUcrtFmaSmallSin(float x) {
    const double d = x;
    const float ax = fabsf(x);
    if (ax < 0x1p-13f) return x;
    const double square = d * d;
    if (ax < 0x1p-7f) {
        return static_cast<float>(fma(-(square * d), 0x1.5555555555555p-3, d));
    }
    double polynomial = fma(square, 0x1.71de3a556c734p-19, -0x1.a01a01a01a01ap-13);
    polynomial = fma(square, polynomial, 0x1.1111111111111p-7);
    polynomial = fma(square, polynomial, -0x1.5555555555555p-3);
    return static_cast<float>(fma(polynomial, d * square, d));
}

ROCKETGPU_HD float WindowsUcrtFmaSmallCos(float x) {
    const double d = x;
    const float ax = fabsf(x);
    if (ax < 0x1p-13f) return 1.0f;
    if (ax < 0x1p-7f) return static_cast<float>(fma(-d, d * 0.5, 1.0));
    const double square = d * d;
    const double base = 1.0 - square * 0.5;
    double polynomial = fma(square, -0x1.27e4fb7789f5cp-22, 0x1.a01a01a01a019p-16);
    polynomial = fma(square, polynomial, -0x1.6c16c16c16c16p-10);
    polynomial = fma(square, polynomial, 0x1.5555555555555p-5);
    return static_cast<float>(fma(polynomial, square * square, base));
}
}}
