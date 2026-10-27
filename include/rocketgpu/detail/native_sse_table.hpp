#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

namespace rocketgpu::detail {
struct NativeSseTable {
    int shift = 23;
    std::vector<uint32_t> values;
    uint64_t checked = 0;
};
inline uint32_t NativeSseEstimate(uint32_t bits) {
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    value = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(value)));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
#else
    throw std::runtime_error("Native SSE normalization requires an x86 SSE host");
#endif
}
inline uint32_t NativeSseLookup(const NativeSseTable& table, uint32_t bits) {
    const int exponent = (bits >> 23) & 255;
    const int parity = 1 - (exponent & 1);
    const auto index = (parity << (23 - table.shift)) + ((bits & 0x7fffff) >> table.shift);
    const int adjustment = (exponent - (127 + parity)) / 2;
    return static_cast<uint32_t>(int64_t(table.values[index]) - adjustment * int64_t(1 << 23));
}
inline const NativeSseTable& NativeSseHostTable() {
    static const NativeSseTable result = [] {
        NativeSseTable table;
        auto previous = NativeSseEstimate(0x3f800000);
        for (uint32_t bits = 0x3f800001; bits < 0x40800000; ++bits) {
            const auto value = NativeSseEstimate(bits);
            if (value != previous) {
                int zeros = 0;
                for (uint32_t n = bits; !(n & 1); n >>= 1) ++zeros;
                table.shift = std::min(table.shift, zeros);
            }
            previous = value;
        }
        if (table.shift < 8) throw std::runtime_error("Native SSE table exceeds bounded allocation");
        table.values.resize(size_t(1) << (24 - table.shift));
        for (size_t i = 0; i < table.values.size(); ++i)
            table.values[i] = NativeSseEstimate(0x3f800000 + (uint32_t(i) << table.shift));
        for (uint32_t bits = 0x3f800000; bits < 0x40800000; ++bits, ++table.checked)
            if (NativeSseLookup(table, bits) != NativeSseEstimate(bits))
                throw std::runtime_error("Native SSE mantissa mismatch");
        for (uint32_t exponent = 1; exponent < 255; ++exponent)
            for (uint32_t mantissa = 0; mantissa < 0x800000; mantissa += uint32_t(1) << table.shift)
                for (uint32_t suffix : {0u, (uint32_t(1) << table.shift) - 1}) {
                    const auto bits = (exponent << 23) | mantissa | suffix;
                    if (NativeSseLookup(table, bits) != NativeSseEstimate(bits))
                        throw std::runtime_error("Native SSE exponent mismatch");
                    ++table.checked;
                }
        return table;
    }();
    return result;
}
}
