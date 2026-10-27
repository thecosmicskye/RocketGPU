#pragma once

#include "rocketgpu/detail/common.hpp"

namespace rocketgpu::detail {
enum class RespawnRngKind : uint32_t { MinStd0, MinStd, Mt19937 };

struct RespawnRngState {
    uint32_t words[624] = {};
    uint32_t index = 624;
    RespawnRngKind kind = RespawnRngKind::MinStd0;
};

ROCKETGPU_HD void SeedRespawnRng(RespawnRngState& state, RespawnRngKind kind, uint32_t seed) {
    state.kind = kind;
    if (kind != RespawnRngKind::Mt19937) {
        state.words[0] = seed % 2147483647u;
        if (state.words[0] == 0) state.words[0] = 1;
        state.index = 0;
        return;
    }
    state.words[0] = seed;
    for (uint32_t i = 1; i < 624; ++i) {
        const uint32_t previous = state.words[i - 1];
        state.words[i] = 1812433253u * (previous ^ (previous >> 30)) + i;
    }
    state.index = 624;
}

ROCKETGPU_HD uint32_t NextRespawnRng(RespawnRngState& state) {
    if (state.kind != RespawnRngKind::Mt19937) {
        const uint64_t multiplier = state.kind == RespawnRngKind::MinStd0 ? 16807u : 48271u;
        state.words[0] = static_cast<uint32_t>((multiplier * state.words[0]) % 2147483647u);
        return state.words[0];
    }
    if (state.index == 624) {
        for (uint32_t i = 0; i < 624; ++i) {
            const uint32_t y = (state.words[i] & 0x80000000u) |
                (state.words[(i + 1) % 624] & 0x7fffffffu);
            state.words[i] = state.words[(i + 397) % 624] ^ (y >> 1) ^
                ((y & 1u) ? 0x9908b0dfu : 0u);
        }
        state.index = 0;
    }
    uint32_t y = state.words[state.index++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

ROCKETGPU_HD constexpr uint32_t RespawnRngMinValue(RespawnRngKind kind) {
    return kind == RespawnRngKind::Mt19937 ? 0u : 1u;
}
ROCKETGPU_HD constexpr uint32_t RespawnRngMaxValue(RespawnRngKind kind) {
    return kind == RespawnRngKind::Mt19937 ? 0xffffffffu : 2147483646u;
}

ROCKETGPU_HD void RespawnRngSwapInt(int* a, int* b) {
    const int t = *a;
    *a = *b;
    *b = t;
}

ROCKETGPU_HD uint64_t LibstdcxxUniformInt(RespawnRngState& state, uint64_t urange) {
    const uint64_t urngmin = RespawnRngMinValue(state.kind);
    const uint64_t urngrange = static_cast<uint64_t>(RespawnRngMaxValue(state.kind)) - urngmin;
    if (urngrange > urange) {
        const uint64_t uerange = urange + 1u;
        const uint64_t scaling = urngrange / uerange;
        const uint64_t past = uerange * scaling;
        uint64_t ret;
        do {
            ret = static_cast<uint64_t>(NextRespawnRng(state)) - urngmin;
        } while (ret >= past);
        return ret / scaling;
    }
    if (urngrange == urange) {
        return static_cast<uint64_t>(NextRespawnRng(state)) - urngmin;
    }
    return 0;
}

ROCKETGPU_HD void ShuffleLikeLibstdcxx(int* order, RespawnRngState& state) {
    const uint64_t urngmin = RespawnRngMinValue(state.kind);
    const uint64_t urngrange = static_cast<uint64_t>(RespawnRngMaxValue(state.kind)) - urngmin;
    const uint64_t urange = static_cast<uint64_t>(5);
    if (urngrange / urange >= urange) {
        int i = 1;
        if ((urange % 2u) == 0u) {
            const int x = static_cast<int>(LibstdcxxUniformInt(state, 1u));
            RespawnRngSwapInt(&order[i], &order[x]);
            ++i;
        }
        while (i != 5) {
            const uint64_t b0 = static_cast<uint64_t>(i) + 1u;
            const uint64_t b1 = b0 + 1u;
            const uint64_t x = LibstdcxxUniformInt(state, b0 * b1 - 1u);
            RespawnRngSwapInt(&order[i], &order[static_cast<int>(x / b1)]);
            ++i;
            RespawnRngSwapInt(&order[i], &order[static_cast<int>(x % b1)]);
            ++i;
        }
        return;
    }
    for (int i = 1; i < 5; ++i) {
        const int x = static_cast<int>(LibstdcxxUniformInt(state, static_cast<uint64_t>(i)));
        RespawnRngSwapInt(&order[i], &order[x]);
    }
}

ROCKETGPU_HD uint64_t MsvcRngFromUrng(RespawnRngState& state, uint64_t index) {
    const uint64_t bmask = 0xffffffffull;
    for (;;) {
        uint64_t ret = 0u;
        uint64_t mask = 0u;
        while (mask < index - 1u) {
            ret <<= 31; ret <<= 1;
            ret |= static_cast<uint64_t>(NextRespawnRng(state));
            mask <<= 31; mask <<= 1;
            mask |= bmask;
        }
        if (ret / index < mask / index || mask % index == index - 1u) {
            return ret % index;
        }
    }
}

ROCKETGPU_HD void ShuffleLikeMsvc(int* order, RespawnRngState& state) {
    for (int target = 1; target < 5; ++target) {
        const int off = static_cast<int>(MsvcRngFromUrng(state, static_cast<uint64_t>(target) + 1u));
        if (off != target) {
            RespawnRngSwapInt(&order[target], &order[off]);
        }
    }
}

ROCKETGPU_HD int RespawnRngFloorLog2(uint64_t v) {
    int r = -1;
    while (v != 0u) { v >>= 1; ++r; }
    return r;
}

ROCKETGPU_HD uint64_t LibcxxUniformInt(RespawnRngState& state, uint64_t b) {
    const uint64_t rp = b - 0 + 1u;
    if (rp == 1u) return 0;
    const uint64_t engine_min = RespawnRngMinValue(state.kind);
    const uint64_t Rp = static_cast<uint64_t>(RespawnRngMaxValue(state.kind)) - engine_min + 1u;
    const int dt = 64;
    const int edt = 32;
    int w = RespawnRngFloorLog2(rp);
    if (w < 64 && (rp & (0xffffffffffffffffull >> (dt - w))) != 0u) ++w;
    const int m = RespawnRngFloorLog2(Rp);
    int n = w / m + ((w % m) != 0 ? 1 : 0);
    int w0 = w / n;
    uint64_t y0 = (w0 < dt) ? ((Rp >> w0) << w0) : 0u;
    if (Rp - y0 > y0 / static_cast<uint64_t>(n)) {
        ++n;
        w0 = w / n;
        y0 = (w0 < dt) ? ((Rp >> w0) << w0) : 0u;
    }
    const int n0 = n - (w % n);
    const uint64_t y1 = (w0 < dt - 1) ? ((Rp >> (w0 + 1)) << (w0 + 1)) : 0u;
    const uint32_t mask0 = w0 > 0 ? (0xffffffffu >> (edt - w0)) : 0u;
    const uint32_t mask1 = (w0 < edt - 1) ? (0xffffffffu >> (edt - (w0 + 1))) : 0xffffffffu;
    uint64_t u;
    do {
        uint64_t sp = 0u;
        for (int k = 0; k < n0; ++k) {
            uint32_t uu;
            do { uu = NextRespawnRng(state) - static_cast<uint32_t>(engine_min); } while (uu >= y0);
            sp = (w0 < dt) ? (sp << w0) : 0u;
            sp += uu & mask0;
        }
        for (int k = n0; k < n; ++k) {
            uint32_t uu;
            do { uu = NextRespawnRng(state) - static_cast<uint32_t>(engine_min); } while (uu >= y1);
            sp = (w0 < dt - 1) ? (sp << (w0 + 1)) : 0u;
            sp += uu & mask1;
        }
        u = sp;
    } while (u >= rp);
    return u + 0;
}

ROCKETGPU_HD void ShuffleLikeLibcxx(int* order, RespawnRngState& state) {
    int d = 5 - 1;
    for (int first = 0; first < 5 - 1; ++first, --d) {
        const int i = static_cast<int>(LibcxxUniformInt(state, static_cast<uint64_t>(d)));
        if (i != 0) {
            RespawnRngSwapInt(&order[first], &order[first + i]);
        }
    }
}

ROCKETGPU_HD void ShuffleLikePlatformStd(int* order, RespawnRngState& state) {
    switch (state.kind) {
        case RespawnRngKind::Mt19937: ShuffleLikeMsvc(order, state); break;
        case RespawnRngKind::MinStd: ShuffleLikeLibcxx(order, state); break;
        default: ShuffleLikeLibstdcxx(order, state); break;
    }
}
}
