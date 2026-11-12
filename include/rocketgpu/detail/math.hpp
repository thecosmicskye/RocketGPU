#pragma once

#include "rocketgpu/rocketgpu.h"
#include "rocketgpu/detail/car_ref.hpp"

constexpr unsigned int kRocketGpuUcrtSinfLimit = 0x4b800456u;
constexpr unsigned int kRocketGpuUcrtCosfLimit = 0x4f490fdbu;
ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE unsigned int RocketGpuUcrtAbsBits(float x) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(x) & 0x7fffffffu;
#else
    unsigned int u;
    memcpy(&u, &x, sizeof(u));
    return u & 0x7fffffffu;
#endif
}
ROCKETGPU_HD double RocketGpuUcrtSinPoly(double r) {
    const double r2 = r * r;
    double p = fma(r2, 0x1.71de3a556c734p-19, -0x1.a01a01a01a01ap-13);
    p = fma(p, r2, 0x1.1111111111111p-7);
    p = fma(p, r2, -0x1.5555555555555p-3);
    return fma(p, r * r2, r);
}
ROCKETGPU_HD double RocketGpuUcrtCosPoly(double r) {
    const double r2 = r * r;
    const double base = 1.0 - r2 * 0.5;
    double p = fma(r2, -0x1.27e4fb7789f5cp-22, 0x1.a01a01a01a019p-16);
    p = fma(p, r2, -0x1.6c16c16c16c16p-10);
    p = fma(p, r2, 0x1.5555555555555p-5);
    return fma(p, r2 * r2, base);
}
ROCKETGPU_HD double RocketGpuUcrtReduce(double a, int* quadrant) {
    const int n = static_cast<int>(fma(a, 0x1.45f306dc9c883p-1, 0.5));
    const double nd = static_cast<double>(n);
    const double r1 = fma(-nd, 0x1.921fb54400000p+0, a);
    *quadrant = n & 3;
    return r1 - nd * 0x1.0b4611a626331p-34;
}
ROCKETGPU_HD float RocketGpuUcrtSinf(float x) {
    const unsigned int ax = RocketGpuUcrtAbsBits(x);
    if (ax < 0x39000000u) return x;
    const double d = static_cast<double>(x);
    if (ax <= 0x3f490fdbu) {
        if (ax < 0x3c000000u) return static_cast<float>(fma(-((d * d) * d), 0x1.5555555555555p-3, d));
        return static_cast<float>(RocketGpuUcrtSinPoly(d));
    }
    int q = 0;
    const double r = RocketGpuUcrtReduce(fabs(d), &q);
    const double y = (q & 1) ? RocketGpuUcrtCosPoly(r) : RocketGpuUcrtSinPoly(r);
    return static_cast<float>(((q >= 2) != (x < 0.0f)) ? -y : y);
}
ROCKETGPU_HD float RocketGpuUcrtCosf(float x) {
    const unsigned int ax = RocketGpuUcrtAbsBits(x);
    if (ax < 0x39000000u) return 1.0f;
    const double d = static_cast<double>(x);
    if (ax <= 0x3f490fdbu) {
        if (ax < 0x3c000000u) return static_cast<float>(fma(-(d * 0.5), d, 1.0));
        return static_cast<float>(RocketGpuUcrtCosPoly(d));
    }
    int q = 0;
    const double r = RocketGpuUcrtReduce(fabs(d), &q);
    const double y = (q & 1) ? RocketGpuUcrtSinPoly(r) : RocketGpuUcrtCosPoly(r);
    return static_cast<float>((q == 1 || q == 2) ? -y : y);
}
ROCKETGPU_HD float RocketGpuUcrtAtan2f(float yf, float xf) {
    static const double kAtanTab[241] = {
        0x1.ff55bb72cfde9p-5, 0x1.0f99ea71d52a6p-4, 0x1.1f86dbf082d58p-4, 0x1.2f719318a4a9ap-4,
        0x1.3f59f0e7c559dp-4, 0x1.4f3fd677292fbp-4, 0x1.5f2324fd2d7b2p-4, 0x1.6f03bdcea4b0cp-4,
        0x1.7ee182602f10ep-4, 0x1.8ebc54478fb28p-4, 0x1.9e94153cfdcf1p-4, 0x1.ae68a71c722b8p-4,
        0x1.be39ebe6f07c3p-4, 0x1.ce07c5c3cca32p-4, 0x1.ddd21701eba6ep-4, 0x1.ed98c2190043ap-4,
        0x1.fd5ba9aac2f6dp-4, 0x1.068d584212b3dp-3, 0x1.0e6adccf40881p-3, 0x1.1646541060850p-3,
        0x1.1e1fafb043726p-3, 0x1.25f6e171a535cp-3, 0x1.2dcbdb2fba1ffp-3, 0x1.359e8edeb99a3p-3,
        0x1.3d6eee8c6626cp-3, 0x1.453cec6092a9ep-3, 0x1.4d087a9da4f17p-3, 0x1.54d18ba11570ap-3,
        0x1.5c9811e3ec269p-3, 0x1.645bfffb3aa73p-3, 0x1.6c1d4898933d8p-3, 0x1.73dbde8a7d201p-3,
        0x1.7b97b4bce5b02p-3, 0x1.8350be398ebc7p-3, 0x1.8b06ee2879c28p-3, 0x1.92ba37d050271p-3,
        0x1.9a6a8e96c8626p-3, 0x1.a217e601081a5p-3, 0x1.a9c231b403279p-3, 0x1.b1696574d780bp-3,
        0x1.b90d7529260a2p-3, 0x1.c0ae54d768466p-3, 0x1.c84bf8a742e6dp-3, 0x1.cfe654e1d5395p-3,
        0x1.d77d5df205736p-3, 0x1.df110864c9d9dp-3, 0x1.e6a148e96ec4dp-3, 0x1.ee2e1451d980cp-3,
        0x1.f5b75f92c80ddp-3, 0x1.fd3d1fc40dbe4p-3, 0x1.025fa510665b5p-2, 0x1.061eea03d6290p-2,
        0x1.09dc597d86362p-2, 0x1.0d97ee509acb3p-2, 0x1.1151a362431c9p-2, 0x1.150973a9ce546p-2,
        0x1.18bf5a30bf178p-2, 0x1.1c735212dd883p-2, 0x1.2025567e47c95p-2, 0x1.23d562b381041p-2,
        0x1.278372057ef45p-2, 0x1.2b2f7fd9b5fe2p-2, 0x1.2ed987a823cfep-2, 0x1.328184fb58951p-2,
        0x1.362773707ebcbp-2, 0x1.39cb4eb76157bp-2, 0x1.3d6d129271134p-2, 0x1.410cbad6c7d32p-2,
        0x1.44aa436c2af09p-2, 0x1.4845a84d0c21bp-2, 0x1.4bdee586890e6p-2, 0x1.4f75f73869978p-2,
        0x1.530ad9951cd49p-2, 0x1.569d88e1b4cd7p-2, 0x1.5a2e0175e0f4ep-2, 0x1.5dbc3fbbe768dp-2,
        0x1.614840309cfe1p-2, 0x1.64d1ff635c1c5p-2, 0x1.685979f5fa6fdp-2, 0x1.6bdeac9cbd76cp-2,
        0x1.6f61941e4def0p-2, 0x1.72e22d53aa2a9p-2, 0x1.7660752817501p-2, 0x1.79dc6899118d1p-2,
        0x1.7d5604b63b3f7p-2, 0x1.80cd46a14b1d0p-2, 0x1.84422b8df95d7p-2, 0x1.87b4b0c1ebedbp-2,
        0x1.8b24d394a1b25p-2, 0x1.8e92916f5cde8p-2, 0x1.91fde7cd0c662p-2, 0x1.9566d43a34907p-2,
        0x1.98cd5454d6b18p-2, 0x1.9c3165cc58107p-2, 0x1.9f93066168001p-2, 0x1.a2f233e5e530bp-2,
        0x1.a64eec3cc23fcp-2, 0x1.a9a92d59e98cfp-2, 0x1.ad00f5422058bp-2, 0x1.b056420ae9343p-2,
        0x1.b3a911da65c6cp-2, 0x1.b6f962e737efbp-2, 0x1.ba473378624a5p-2, 0x1.bd9281e528191p-2,
        0x1.c0db4c94ec9efp-2, 0x1.c42191ff11eb6p-2, 0x1.c76550aad71f8p-2, 0x1.caa6872f3631bp-2,
        0x1.cde53432c1350p-2, 0x1.d121566b7f2adp-2, 0x1.d45aec9ec862bp-2, 0x1.d791f5a1226f4p-2,
        0x1.dac670561bb4fp-2, 0x1.ddf85bb026974p-2, 0x1.e127b6b0744afp-2, 0x1.e4548066cf51ap-2,
        0x1.e77eb7f175a34p-2, 0x1.eaa65c7cf28c4p-2, 0x1.edcb6d43f8434p-2, 0x1.f0ede98f393cfp-2,
        0x1.f40dd0b541417p-2, 0x1.f72b221a4e495p-2, 0x1.fa45dd3029258p-2, 0x1.fd5e0175fdf83p-2,
        0x1.0039c73c1a40bp-1, 0x1.01c341e82422dp-1, 0x1.034b709250488p-1, 0x1.04d25314342e5p-1,
        0x1.0657e94db30cfp-1, 0x1.07dc3324e9b38p-1, 0x1.095f30861a58fp-1, 0x1.0ae0e1639866cp-1,
        0x1.0c6145b5b43dap-1, 0x1.0de05d7aa6f7cp-1, 0x1.0f5e28b67e295p-1, 0x1.10daa77307a0dp-1,
        0x1.1255d9bfbd2a8p-1, 0x1.13cfbfb1b056ep-1, 0x1.1548596376469p-1, 0x1.16bfa6f5137e1p-1,
        0x1.1835a88be7c13p-1, 0x1.19aa5e5299f99p-1, 0x1.1b1dc87904284p-1, 0x1.1c8fe7341f64fp-1,
        0x1.1e00babdefeb3p-1, 0x1.1f7043557138ap-1, 0x1.20de813e823b1p-1, 0x1.224b74c1d192ap-1,
        0x1.23b71e2cc9e6ap-1, 0x1.25217dd17e501p-1, 0x1.268a940696da6p-1, 0x1.27f261273d1b3p-1,
        0x1.2958e59308e30p-1, 0x1.2abe21aded073p-1, 0x1.2c2215e024465p-1, 0x1.2d84c2961e48bp-1,
        0x1.2ee628406cbcap-1, 0x1.30464753b090ap-1, 0x1.31a52048874bep-1, 0x1.3302b39b78856p-1,
        0x1.345f01cce37bbp-1, 0x1.35ba0b60ecccep-1, 0x1.3713d0df6c503p-1, 0x1.386c52d3db11ep-1,
        0x1.39c391cd41719p-1, 0x1.3b198e5e2564ap-1, 0x1.3c6e491c78dc4p-1, 0x1.3dc1c2a188504p-1,
        0x1.3f13fb89e96f4p-1, 0x1.4064f47569f48p-1, 0x1.41b4ae06fea41p-1, 0x1.430328e4b26d5p-1,
        0x1.445065b795b55p-1, 0x1.459c652badc7fp-1, 0x1.46e727efe4715p-1, 0x1.4830aeb5f7bfdp-1,
        0x1.4978fa3269ee1p-1, 0x1.4ac00b1c71762p-1, 0x1.4c05e22de94e4p-1, 0x1.4d4a8023414e8p-1,
        0x1.4e8de5bb6ec04p-1, 0x1.4fd013b7dd17ep-1, 0x1.51110adc5ed81p-1, 0x1.5250cbef1e9fap-1,
        0x1.538f57b89061ep-1, 0x1.54ccaf0362c8fp-1, 0x1.5608d29c70c34p-1, 0x1.5743c352b33b9p-1,
        0x1.587d81f732fbap-1, 0x1.59b60f5cfab9dp-1, 0x1.5aed6c5909517p-1, 0x1.5c2399c244260p-1,
        0x1.5d58987169b18p-1, 0x1.5e8c6941043cfp-1, 0x1.5fbf0d0d5cc49p-1, 0x1.60f084b46e05ep-1,
        0x1.6220d115d7b8dp-1, 0x1.634ff312d1f3bp-1, 0x1.647deb8e20b8fp-1, 0x1.65aabb6c07b02p-1,
        0x1.66d663923e086p-1, 0x1.6800e4e7e2857p-1, 0x1.692a40556fb6ap-1, 0x1.6a5276c4b0575p-1,
        0x1.6b798920b3d98p-1, 0x1.6c9f7855c3198p-1, 0x1.6dc44551553aep-1, 0x1.6ee7f10204aefp-1,
        0x1.700a7c5784633p-1, 0x1.712be84295198p-1, 0x1.724c35b4fae7bp-1, 0x1.736b65a172dffp-1,
        0x1.748978fba8e0fp-1, 0x1.75a670b82d8d8p-1, 0x1.76c24dcc6c6c0p-1, 0x1.77dd112ea22c7p-1,
        0x1.78f6bbd5d315ep-1, 0x1.7a0f4eb9c19a2p-1, 0x1.7b26cad2e50fdp-1, 0x1.7c3d311a6092bp-1,
        0x1.7d528289fa093p-1, 0x1.7e66c01c114fdp-1, 0x1.7f79eacb97898p-1, 0x1.808c03940694ap-1,
        0x1.819d0b7158a4cp-1, 0x1.82ad036000005p-1, 0x1.83bbec5cdee22p-1, 0x1.84c9c7653f7eap-1,
        0x1.85d69576cc2c5p-1, 0x1.86e2578f87ae5p-1, 0x1.87ed0eadc5a2ap-1, 0x1.88f6bbd023118p-1,
        0x1.89ff5ff57f1f7p-1, 0x1.8b06fc1cf3dfep-1, 0x1.8c0d9145cf49dp-1, 0x1.8d13206f8c4cap-1,
        0x1.8e17aa99cc05dp-1, 0x1.8f1b30c44f167p-1, 0x1.901db3eeef187p-1, 0x1.911f35199833bp-1,
        0x1.921fb54442d18p-1};
    const double y = static_cast<double>(yf), x = static_cast<double>(xf);
    unsigned long long xb, yb;
    memcpy(&xb, &x, sizeof(xb));
    memcpy(&yb, &y, sizeof(yb));
    const unsigned long long ax = xb & 0x7fffffffffffffffull, ay = yb & 0x7fffffffffffffffull;
    const unsigned long long inf = 0x7ff0000000000000ull;
    const bool yneg = (yb >> 63) != 0, xneg = (xb >> 63) != 0;
    const int diff = static_cast<int>((yb >> 52) & 0x7ff) - static_cast<int>((xb >> 52) & 0x7ff);
    const auto bits_to_float = [](unsigned int u) { float f; memcpy(&f, &u, sizeof(f)); return f; };
    const auto quiet = [&](float f) { unsigned int u; memcpy(&u, &f, sizeof(u)); return bits_to_float(u | 0x400000u); };
    if (ax > inf) return quiet(xf);
    if (ay > inf) return quiet(yf);
    if (ay == 0) return xneg ? bits_to_float(yneg ? 0xc0490fdbu : 0x40490fdbu) : yf;
    if (ax == 0 || diff > 26) return bits_to_float(yneg ? 0xbfc90fdbu : 0x3fc90fdbu);
    if (diff < -13 && !xneg) {
        if (diff < -150) return yneg ? -0.0f : 0.0f;
        return static_cast<float>(y / x);
    }
    if (diff < -26) return bits_to_float(yneg ? 0xc0490fdbu : 0x40490fdbu);
    if (ay == inf) {
        if (!xneg) return bits_to_float(yneg ? 0xbf490fdbu : 0x3f490fdbu);
        return bits_to_float(yneg ? 0xc016cbe4u : 0x4016cbe4u);
    }
    double u = fabs(y), v = fabs(x);
    const bool swapped = u > v;
    if (swapped) {
        const double t = u;
        u = v;
        v = t;
    }
    const double ratio = u / v;
    double res;
    if (ratio > 0x1p-4) {
        const int i = static_cast<int>(fma(256.0, ratio, 0.5));
        const double id = static_cast<double>(i);
        const double r = fma(u, 256.0, -(id * v)) / fma(u, id, v * 256.0);
        res = fma(-((r * r) * r), 0x1.5555555550877p-2, r + kAtanTab[i - 16]);
    } else if (ratio < 0x1.a36e2eb1c432dp-14) {
        res = ratio;
    } else {
        const double r2 = ratio * ratio;
        double p = fma(-0x1.2492482bd6be1p-3, r2, 0x1.99999999643a3p-3);
        p = fma(-p, r2, 0x1.5555555555538p-2);
        res = fma(-(r2 * ratio), p, ratio);
    }
    if (swapped) res = 0x1.921fb54442d18p+0 - res;
    if (xneg) res = 0x1.921fb54442d18p+1 - res;
    return static_cast<float>(yneg ? -res : res);
}
ROCKETGPU_HD float RocketGpuUcrtPowfExp(double t, unsigned int sign, bool fused) {
    static const double kPowExp2[64] = {
        0x1.0000000000000p+0, 0x1.02c9a3e778061p+0, 0x1.059b0d3158574p+0,
        0x1.0874518759bc8p+0, 0x1.0b5586cf9890fp+0, 0x1.0e3ec32d3d1a2p+0,
        0x1.11301d0125b51p+0, 0x1.1429aaea92de0p+0, 0x1.172b83c7d517bp+0,
        0x1.1a35beb6fcb75p+0, 0x1.1d4873168b9aap+0, 0x1.2063b88628cd6p+0,
        0x1.2387a6e756238p+0, 0x1.26b4565e27cddp+0, 0x1.29e9df51fdee1p+0,
        0x1.2d285a6e4030bp+0, 0x1.306fe0a31b715p+0, 0x1.33c08b26416ffp+0,
        0x1.371a7373aa9cbp+0, 0x1.3a7db34e59ff7p+0, 0x1.3dea64c123422p+0,
        0x1.4160a21f72e2ap+0, 0x1.44e086061892dp+0, 0x1.486a2b5c13cd0p+0,
        0x1.4bfdad5362a27p+0, 0x1.4f9b2769d2ca7p+0, 0x1.5342b569d4f82p+0,
        0x1.56f4736b527dap+0, 0x1.5ab07dd485429p+0, 0x1.5e76f15ad2148p+0,
        0x1.6247eb03a5585p+0, 0x1.6623882552225p+0, 0x1.6a09e667f3bcdp+0,
        0x1.6dfb23c651a2fp+0, 0x1.71f75e8ec5f74p+0, 0x1.75feb564267c9p+0,
        0x1.7a11473eb0187p+0, 0x1.7e2f336cf4e62p+0, 0x1.82589994cce13p+0,
        0x1.868d99b4492edp+0, 0x1.8ace5422aa0dbp+0, 0x1.8f1ae99157736p+0,
        0x1.93737b0cdc5e5p+0, 0x1.97d829fde4e50p+0, 0x1.9c49182a3f090p+0,
        0x1.a0c667b5de565p+0, 0x1.a5503b23e255dp+0, 0x1.a9e6b5579fdbfp+0,
        0x1.ae89f995ad3adp+0, 0x1.b33a2b84f15fbp+0, 0x1.b7f76f2fb5e47p+0,
        0x1.bcc1e904bc1d2p+0, 0x1.c199bdd85529cp+0, 0x1.c67f12e57d14bp+0,
        0x1.cb720dcef9069p+0, 0x1.d072d4a07897cp+0, 0x1.d5818dcfba487p+0,
        0x1.da9e603db3285p+0, 0x1.dfc97337b9b5fp+0, 0x1.e502ee78b3ff6p+0,
        0x1.ea4afa2a490dap+0, 0x1.efa1bee615a27p+0, 0x1.f50765b6e4540p+0,
        0x1.fa7c1819e90d8p+0,
    };
    const auto bits_to_float = [](unsigned int u) { float f; memcpy(&f, &u, sizeof(f)); return f; };
    if (t > 0x1.62e43p6) return bits_to_float(0x7f800000u | sign);
    if (!(t > -0x1.9d1dap6)) return bits_to_float(sign);
    const int n = static_cast<int>(rint(t * 0x1.71547652b82fep+6));
    const double tab = kPowExp2[n & 63];
    double res;
    if (fused) {
        const double r = fma(-static_cast<double>(n), 0x1.62e42fefa39efp-7, t);
        res = fma(r * fma(fma(0x1.5555555555555p-3, r, 0.5), r, 1.0), tab, tab);
    } else {
        const double r = t - static_cast<double>(n) * 0x1.62e42fefa39efp-7;
        res = ((r * r) * (0x1.5555555555555p-3 * r + 0.5) + r) * tab + tab;
    }
    unsigned long long rb;
    memcpy(&rb, &res, sizeof(rb));
    rb += static_cast<unsigned long long>(static_cast<long long>(n >> 6)) << 52;
    memcpy(&res, &rb, sizeof(res));
    const float f = static_cast<float>(res);
    unsigned int u;
    memcpy(&u, &f, sizeof(u));
    return bits_to_float(u | sign);
}
ROCKETGPU_HD float RocketGpuUcrtPowf(float x, float y) {
    static const double kPowInvC[257] = {
        0x1.0000000000000p+1, 0x1.fe01fe01fe020p+0, 0x1.fc07f01fc07f0p+0,
        0x1.fa11caa01fa12p+0, 0x1.f81f81f81f820p+0, 0x1.f6310aca0dbb5p+0,
        0x1.f44659e4a4271p+0, 0x1.f25f644230ab5p+0, 0x1.f07c1f07c1f08p+0,
        0x1.ee9c7f8458e02p+0, 0x1.ecc07b301ecc0p+0, 0x1.eae807aba01ebp+0,
        0x1.e9131abf0b767p+0, 0x1.e741aa59750e4p+0, 0x1.e573ac901e574p+0,
        0x1.e3a9179dc1a73p+0, 0x1.e1e1e1e1e1e1ep+0, 0x1.e01e01e01e01ep+0,
        0x1.de5d6e3f8868ap+0, 0x1.dca01dca01dcap+0, 0x1.dae6076b981dbp+0,
        0x1.d92f2231e7f8ap+0, 0x1.d77b654b82c34p+0, 0x1.d5cac807572b2p+0,
        0x1.d41d41d41d41dp+0, 0x1.d272ca3fc5b1ap+0, 0x1.d0cb58f6ec074p+0,
        0x1.cf26e5c44bfc6p+0, 0x1.cd85689039b0bp+0, 0x1.cbe6d9601cbe7p+0,
        0x1.ca4b3055ee191p+0, 0x1.c8b265afb8a42p+0, 0x1.c71c71c71c71cp+0,
        0x1.c5894d10d4986p+0, 0x1.c3f8f01c3f8f0p+0, 0x1.c26b5392ea01cp+0,
        0x1.c0e070381c0e0p+0, 0x1.bf583ee868d8bp+0, 0x1.bdd2b899406f7p+0,
        0x1.bc4fd65883e7bp+0, 0x1.bacf914c1bad0p+0, 0x1.b951e2b18ff23p+0,
        0x1.b7d6c3dda338bp+0, 0x1.b65e2e3beee05p+0, 0x1.b4e81b4e81b4fp+0,
        0x1.b37484ad806cep+0, 0x1.b2036406c80d9p+0, 0x1.b094b31d922a4p+0,
        0x1.af286bca1af28p+0, 0x1.adbe87f94905ep+0, 0x1.ac5701ac5701bp+0,
        0x1.aaf1d2f87ebfdp+0, 0x1.a98ef606a63bep+0, 0x1.a82e65130e159p+0,
        0x1.a6d01a6d01a6dp+0, 0x1.a574107688a4ap+0, 0x1.a41a41a41a41ap+0,
        0x1.a2c2a87c51ca0p+0, 0x1.a16d3f97a4b02p+0, 0x1.a01a01a01a01ap+0,
        0x1.9ec8e951033d9p+0, 0x1.9d79f176b682dp+0, 0x1.9c2d14ee4a102p+0,
        0x1.9ae24ea5510dap+0, 0x1.999999999999ap+0, 0x1.9852f0d8ec0ffp+0,
        0x1.970e4f80cb872p+0, 0x1.95cbb0be377aep+0, 0x1.948b0fcd6e9e0p+0,
        0x1.934c67f9b2ce6p+0, 0x1.920fb49d0e229p+0, 0x1.90d4f120190d5p+0,
        0x1.8f9c18f9c18fap+0, 0x1.8e6527af1373fp+0, 0x1.8d3018d3018d3p+0,
        0x1.8bfce8062ff3ap+0, 0x1.8acb90f6bf3aap+0, 0x1.899c0f601899cp+0,
        0x1.886e5f0abb04ap+0, 0x1.87427bcc092b9p+0, 0x1.8618618618618p+0,
        0x1.84f00c2780614p+0, 0x1.83c977ab2beddp+0, 0x1.82a4a0182a4a0p+0,
        0x1.8181818181818p+0, 0x1.8060180601806p+0, 0x1.7f405fd017f40p+0,
        0x1.7e225515a4f1dp+0, 0x1.7d05f417d05f4p+0, 0x1.7beb3922e017cp+0,
        0x1.7ad2208e0ecc3p+0, 0x1.79baa6bb6398bp+0, 0x1.78a4c8178a4c8p+0,
        0x1.77908119ac60dp+0, 0x1.767dce434a9b1p+0, 0x1.756cac201756dp+0,
        0x1.745d1745d1746p+0, 0x1.734f0c541fe8dp+0, 0x1.724287f46debcp+0,
        0x1.713786d9c7c09p+0, 0x1.702e05c0b8170p+0, 0x1.6f26016f26017p+0,
        0x1.6e1f76b4337c7p+0, 0x1.6d1a62681c861p+0, 0x1.6c16c16c16c17p+0,
        0x1.6b1490aa31a3dp+0, 0x1.6a13cd1537290p+0, 0x1.691473a88d0c0p+0,
        0x1.6816816816817p+0, 0x1.6719f3601671ap+0, 0x1.661ec6a5122f9p+0,
        0x1.6524f853b4aa3p+0, 0x1.642c8590b2164p+0, 0x1.63356b88ac0dep+0,
        0x1.623fa77016240p+0, 0x1.614b36831ae94p+0, 0x1.6058160581606p+0,
        0x1.5f66434292dfcp+0, 0x1.5e75bb8d015e7p+0, 0x1.5d867c3ece2a5p+0,
        0x1.5c9882b931057p+0, 0x1.5babcc647fa91p+0, 0x1.5ac056b015ac0p+0,
        0x1.59d61f123ccaap+0, 0x1.58ed2308158edp+0, 0x1.5805601580560p+0,
        0x1.571ed3c506b3ap+0, 0x1.56397ba7c52e2p+0, 0x1.5555555555555p+0,
        0x1.54725e6bb82fep+0, 0x1.5390948f40febp+0, 0x1.52aff56a8054bp+0,
        0x1.51d07eae2f815p+0, 0x1.50f22e111c4c5p+0, 0x1.5015015015015p+0,
        0x1.4f38f62dd4c9bp+0, 0x1.4e5e0a72f0539p+0, 0x1.4d843bedc2c4cp+0,
        0x1.4cab88725af6ep+0, 0x1.4bd3edda68fe1p+0, 0x1.4afd6a052bf5bp+0,
        0x1.4a27fad76014ap+0, 0x1.49539e3b2d067p+0, 0x1.4880522014880p+0,
        0x1.47ae147ae147bp+0, 0x1.46dce34596066p+0, 0x1.460cbc7f5cf9ap+0,
        0x1.453d9e2c776cap+0, 0x1.446f86562d9fbp+0, 0x1.43a2730abee4dp+0,
        0x1.42d6625d51f87p+0, 0x1.420b5265e5951p+0, 0x1.4141414141414p+0,
        0x1.40782d10e6566p+0, 0x1.3fb013fb013fbp+0, 0x1.3ee8f42a5af07p+0,
        0x1.3e22cbce4a902p+0, 0x1.3d5d991aa75c6p+0, 0x1.3c995a47babe7p+0,
        0x1.3bd60d9232955p+0, 0x1.3b13b13b13b14p+0, 0x1.3a524387ac822p+0,
        0x1.3991c2c187f63p+0, 0x1.38d22d366088ep+0, 0x1.3813813813814p+0,
        0x1.3755bd1c945eep+0, 0x1.3698df3de0748p+0, 0x1.35dce5f9f2af8p+0,
        0x1.3521cfb2b78c1p+0, 0x1.34679ace01346p+0, 0x1.33ae45b57bcb2p+0,
        0x1.32f5ced6a1dfap+0, 0x1.323e34a2b10bfp+0, 0x1.3187758e9ebb6p+0,
        0x1.30d190130d190p+0, 0x1.301c82ac40260p+0, 0x1.2f684bda12f68p+0,
        0x1.2eb4ea1fed14bp+0, 0x1.2e025c04b8097p+0, 0x1.2d50a012d50a0p+0,
        0x1.2c9fb4d812ca0p+0, 0x1.2bef98e5a3711p+0, 0x1.2b404ad012b40p+0,
        0x1.2a91c92f3c105p+0, 0x1.29e4129e4129ep+0, 0x1.293725bb804a5p+0,
        0x1.288b01288b013p+0, 0x1.27dfa38a1ce4dp+0, 0x1.27350b8812735p+0,
        0x1.268b37cd60127p+0, 0x1.25e22708092f1p+0, 0x1.2539d7e9177b2p+0,
        0x1.2492492492492p+0, 0x1.23eb79717605bp+0, 0x1.23456789abcdfp+0,
        0x1.22a0122a0122ap+0, 0x1.21fb78121fb78p+0, 0x1.21579804855e6p+0,
        0x1.20b470c67c0d9p+0, 0x1.2012012012012p+0, 0x1.1f7047dc11f70p+0,
        0x1.1ecf43c7fb84cp+0, 0x1.1e2ef3b3fb874p+0, 0x1.1d8f5672e4abdp+0,
        0x1.1cf06ada2811dp+0, 0x1.1c522fc1ce059p+0, 0x1.1bb4a4046ed29p+0,
        0x1.1b17c67f2bae3p+0, 0x1.1a7b9611a7b96p+0, 0x1.19e0119e0119ep+0,
        0x1.19453808ca29cp+0, 0x1.18ab083902bdbp+0, 0x1.1811811811812p+0,
        0x1.1778a191bd684p+0, 0x1.16e0689427379p+0, 0x1.1648d50fc3201p+0,
        0x1.15b1e5f75270dp+0, 0x1.151b9a3fdd5c9p+0, 0x1.1485f0e0acd3bp+0,
        0x1.13f0e8d344724p+0, 0x1.135c81135c811p+0, 0x1.12c8b89edc0acp+0,
        0x1.12358e75d3033p+0, 0x1.11a3019a74826p+0, 0x1.1111111111111p+0,
        0x1.107fbbe011080p+0, 0x1.0fef010fef011p+0, 0x1.0f5edfab325a2p+0,
        0x1.0ecf56be69c90p+0, 0x1.0e40655826011p+0, 0x1.0db20a88f4696p+0,
        0x1.0d24456359e3ap+0, 0x1.0c9714fbcda3bp+0, 0x1.0c0a7868b4171p+0,
        0x1.0b7e6ec259dc8p+0, 0x1.0af2f722eecb5p+0, 0x1.0a6810a6810a7p+0,
        0x1.09ddba6af8360p+0, 0x1.0953f39010954p+0, 0x1.08cabb37565e2p+0,
        0x1.0842108421084p+0, 0x1.07b9f29b8eae2p+0, 0x1.073260a47f7c6p+0,
        0x1.06ab59c7912fbp+0, 0x1.0624dd2f1a9fcp+0, 0x1.059eea0727586p+0,
        0x1.05197f7d73404p+0, 0x1.04949cc1664c5p+0, 0x1.0410410410410p+0,
        0x1.038c6b78247fcp+0, 0x1.03091b51f5e1ap+0, 0x1.02864fc7729e9p+0,
        0x1.0204081020408p+0, 0x1.0182436517a37p+0, 0x1.0101010101010p+0,
        0x1.0080402010080p+0, 0x1.0000000000000p+0,
    };
    static const double kPowLogC[257] = {
        0x0.0p+0, 0x1.ff00aa2b10bc0p-9, 0x1.fe02a6b106789p-8,
        0x1.7dc475f810a77p-7, 0x1.fc0a8b0fc03e4p-7, 0x1.3cea44346a575p-6,
        0x1.7b91b07d5b11bp-6, 0x1.b9fc027af9198p-6, 0x1.f829b0e783300p-6,
        0x1.1b0d98923d980p-5, 0x1.39e87b9febd60p-5, 0x1.58a5bafc8e4d5p-5,
        0x1.77458f632dcfcp-5, 0x1.95c830ec8e3ebp-5, 0x1.b42dd711971bfp-5,
        0x1.d276b8adb0b52p-5, 0x1.f0a30c01162a6p-5, 0x1.075983598e471p-4,
        0x1.16536eea37ae1p-4, 0x1.253f62f0a1417p-4, 0x1.341d7961bd1d1p-4,
        0x1.42edcbea646f0p-4, 0x1.51b073f06183fp-4, 0x1.60658a93750c4p-4,
        0x1.6f0d28ae56b4cp-4, 0x1.7da766d7b12cdp-4, 0x1.8c345d6319b21p-4,
        0x1.9ab42462033adp-4, 0x1.a926d3a4ad563p-4, 0x1.b78c82bb0eda1p-4,
        0x1.c5e548f5bc743p-4, 0x1.d4313d66cb35dp-4, 0x1.e27076e2af2e6p-4,
        0x1.f0a30c01162a6p-4, 0x1.fec9131dbeabbp-4, 0x1.0671512ca596ep-3,
        0x1.0d77e7cd08e59p-3, 0x1.14785846742acp-3, 0x1.1b72ad52f67a0p-3,
        0x1.2266f190a5acbp-3, 0x1.29552f81ff523p-3, 0x1.303d718e47fd3p-3,
        0x1.371fc201e8f74p-3, 0x1.3dfc2b0ecc62ap-3, 0x1.44d2b6ccb7d1ep-3,
        0x1.4ba36f39a55e5p-3, 0x1.526e5e3a1b438p-3, 0x1.59338d9982086p-3,
        0x1.5ff3070a793d4p-3, 0x1.66acd4272ad51p-3, 0x1.6d60fe719d21dp-3,
        0x1.740f8f54037a5p-3, 0x1.7ab890210d909p-3, 0x1.815c0a14357ebp-3,
        0x1.87fa06520c911p-3, 0x1.8e928de886d41p-3, 0x1.9525a9cf456b4p-3,
        0x1.9bb362e7dfb83p-3, 0x1.a23bc1fe2b563p-3, 0x1.a8becfc882f19p-3,
        0x1.af3c94e80bff3p-3, 0x1.b5b519e8fb5a4p-3, 0x1.bc286742d8cd6p-3,
        0x1.c2968558c18c1p-3, 0x1.c8ff7c79a9a22p-3, 0x1.cf6354e09c5dcp-3,
        0x1.d5c216b4fbb91p-3, 0x1.dc1bca0abec7dp-3, 0x1.e27076e2af2e6p-3,
        0x1.e8c0252aa5a60p-3, 0x1.ef0adcbdc5936p-3, 0x1.f550a564b7b37p-3,
        0x1.fb9186d5e3e2bp-3, 0x1.00e6c45ad501dp-2, 0x1.0402594b4d041p-2,
        0x1.071b85fcd590dp-2, 0x1.0a324e27390e3p-2, 0x1.0d46b579ab74bp-2,
        0x1.1058bf9ae4ad5p-2, 0x1.136870293a8b0p-2, 0x1.1675cababa60ep-2,
        0x1.1980d2dd4236fp-2, 0x1.1c898c16999fbp-2, 0x1.1f8ff9e48a2f3p-2,
        0x1.22941fbcf7966p-2, 0x1.2596010df763ap-2, 0x1.2895a13de86a3p-2,
        0x1.2b9303ab89d25p-2, 0x1.2e8e2bae11d31p-2, 0x1.31871c9544185p-2,
        0x1.347dd9a987d55p-2, 0x1.3772662bfd85bp-2, 0x1.3a64c556945eap-2,
        0x1.3d54fa5c1f710p-2, 0x1.404308686a7e4p-2, 0x1.432ef2a04e814p-2,
        0x1.4618bc21c5ec2p-2, 0x1.49006804009d1p-2, 0x1.4be5f957778a1p-2,
        0x1.4ec973260026ap-2, 0x1.51aad872df82dp-2, 0x1.548a2c3add263p-2,
        0x1.5767717455a6cp-2, 0x1.5a42ab0f4cfe2p-2, 0x1.5d1bdbf5809cap-2,
        0x1.5ff3070a793d4p-2, 0x1.62c82f2b9c795p-2, 0x1.659b57303e1f3p-2,
        0x1.686c81e9b14afp-2, 0x1.6b3bb2235943ep-2, 0x1.6e08eaa2ba1e4p-2,
        0x1.70d42e2789236p-2, 0x1.739d7f6bbd007p-2, 0x1.7664e1239dbcfp-2,
        0x1.792a55fdd47a2p-2, 0x1.7bede0a37afc0p-2, 0x1.7eaf83b82afc3p-2,
        0x1.816f41da0d496p-2, 0x1.842d1da1e8b17p-2, 0x1.86e919a330ba0p-2,
        0x1.89a3386c1425bp-2, 0x1.8c5b7c858b48bp-2, 0x1.8f11e873662c7p-2,
        0x1.91c67eb45a83ep-2, 0x1.947941c2116fbp-2, 0x1.972a341135158p-2,
        0x1.99d958117e08bp-2, 0x1.9c86b02dc0863p-2, 0x1.9f323ecbf984cp-2,
        0x1.a1dc064d5b995p-2, 0x1.a484090e5bb0ap-2, 0x1.a72a4966bd9eap-2,
        0x1.a9cec9a9a084ap-2, 0x1.ac718c258b0e4p-2, 0x1.af1293247786bp-2,
        0x1.b1b1e0ebdfc5bp-2, 0x1.b44f77bcc8f63p-2, 0x1.b6eb59d3cf35ep-2,
        0x1.b9858969310fbp-2, 0x1.bc1e08b0dad0ap-2, 0x1.beb4d9da71b7cp-2,
        0x1.c149ff115f027p-2, 0x1.c3dd7a7cdad4dp-2, 0x1.c66f4e3ff6ff8p-2,
        0x1.c8ff7c79a9a22p-2, 0x1.cb8e0744d7acap-2, 0x1.ce1af0b85f3ebp-2,
        0x1.d0a63ae721e64p-2, 0x1.d32fe7e00ebd5p-2, 0x1.d5b7f9ae2c684p-2,
        0x1.d83e7258a2f3ep-2, 0x1.dac353e2c5954p-2, 0x1.dd46a04c1c4a1p-2,
        0x1.dfc859906d5b5p-2, 0x1.e24881a7c6c26p-2, 0x1.e4c71a8687704p-2,
        0x1.e744261d68788p-2, 0x1.e9bfa659861f5p-2, 0x1.ec399d2468cc0p-2,
        0x1.eeb20c640ddf4p-2, 0x1.f128f5faf06edp-2, 0x1.f39e5bc811e5cp-2,
        0x1.f6123fa7028acp-2, 0x1.f884a36fe9ec2p-2, 0x1.faf588f78f31fp-2,
        0x1.fd64f20f61572p-2, 0x1.ffd2e0857f498p-2, 0x1.011fab125ff8ap-1,
        0x1.02552a5a5d0ffp-1, 0x1.0389eefce633bp-1, 0x1.04bdf9da926d2p-1,
        0x1.05f14bd26459cp-1, 0x1.0723e5c1cdf40p-1, 0x1.0855c884b450ep-1,
        0x1.0986f4f573521p-1, 0x1.0ab76bece14d2p-1, 0x1.0be72e4252a83p-1,
        0x1.0d163ccb9d6b8p-1, 0x1.0e44985d1cc8cp-1, 0x1.0f7241c9b497dp-1,
        0x1.109f39e2d4c97p-1, 0x1.11cb81787ccf8p-1, 0x1.12f719593efbcp-1,
        0x1.1422025243d45p-1, 0x1.154c3d2f4d5eap-1, 0x1.1675cababa60ep-1,
        0x1.179eabbd899a1p-1, 0x1.18c6e0ff5cf06p-1, 0x1.19ee6b467c96fp-1,
        0x1.1b154b57da29fp-1, 0x1.1c3b81f713c25p-1, 0x1.1d610fe677003p-1,
        0x1.1e85f5e7040d0p-1, 0x1.1faa34b87094cp-1, 0x1.20cdcd192ab6ep-1,
        0x1.21f0bfc65beecp-1, 0x1.23130d7bebf43p-1, 0x1.2434b6f483934p-1,
        0x1.2555bce98f7cbp-1, 0x1.26762013430e0p-1, 0x1.2795e1289b11bp-1,
        0x1.28b500df60783p-1, 0x1.29d37fec2b08bp-1, 0x1.2af15f02640adp-1,
        0x1.2c0e9ed448e8cp-1, 0x1.2d2b4012edc9ep-1, 0x1.2e47436e40268p-1,
        0x1.2f62a99509546p-1, 0x1.307d7334f10bep-1, 0x1.3197a0fa7fe6ap-1,
        0x1.32b1339121d71p-1, 0x1.33ca2ba328995p-1, 0x1.34e289d9ce1d3p-1,
        0x1.35fa4edd36ea0p-1, 0x1.37117b54747b6p-1, 0x1.38280fe58797fp-1,
        0x1.393e0d3562a1ap-1, 0x1.3a5373e7ebdfap-1, 0x1.3b68449fffc23p-1,
        0x1.3c7c7fff73206p-1, 0x1.3d9026a7156fbp-1, 0x1.3ea33936b2f5cp-1,
        0x1.3fb5b84d16f42p-1, 0x1.40c7a4880dce9p-1, 0x1.41d8fe84672aep-1,
        0x1.42e9c6ddf80bfp-1, 0x1.43f9fe2f9ce67p-1, 0x1.4509a5133bb0ap-1,
        0x1.4618bc21c5ec2p-1, 0x1.472743f33aaadp-1, 0x1.48353d1ea88dfp-1,
        0x1.4942a83a2fc07p-1, 0x1.4a4f85db03ebbp-1, 0x1.4b5bd6956e274p-1,
        0x1.4c679afccee3ap-1, 0x1.4d72d3a39fd00p-1, 0x1.4e7d811b75bb1p-1,
        0x1.4f87a3f5026e9p-1, 0x1.50913cc01686bp-1, 0x1.519a4c0ba3446p-1,
        0x1.52a2d265bc5abp-1, 0x1.53aad05b99b7dp-1, 0x1.54b2467999498p-1,
        0x1.55b9354b40bcdp-1, 0x1.56bf9d5b3f399p-1, 0x1.57c57f336f191p-1,
        0x1.58cadb5cd7989p-1, 0x1.59cfb25fae87ep-1, 0x1.5ad404c359f2dp-1,
        0x1.5bd7d30e71c73p-1, 0x1.5cdb1dc6c1765p-1, 0x1.5ddde57149923p-1,
        0x1.5ee02a9241675p-1, 0x1.5fe1edad18919p-1, 0x1.60e32f44788d9p-1,
        0x1.61e3efda46467p-1, 0x1.62e42fefa39efp-1,
    };
    const auto bits_to_float = [](unsigned int u) { float f; memcpy(&f, &u, sizeof(f)); return f; };
    unsigned int ux, uy;
    memcpy(&ux, &x, sizeof(ux));
    memcpy(&uy, &y, sizeof(uy));
    const unsigned int ax = ux & 0x7fffffffu, ay = uy & 0x7fffffffu;
    const auto odd_int = [&]() {
        if ((ay & 0x7f800000u) > 0x4b000000u) return false;
        const float yr = rintf(y);
        return yr == y && (static_cast<int>(yr) & 1) != 0;
    };
    if (ay >= 0x7f800000u) {
        if (ay > 0x7f800000u) return ux == 0x3f800000u ? (ay < 0x7fc00000u ? 1.5f : 1.0f) : bits_to_float((ax > 0x7f800000u ? ux : uy) | 0x400000u);
        if (ax > 0x7f800000u) return bits_to_float(ux | 0x400000u);
        if (ax == 0x3f800000u) return 1.0f;
        return bits_to_float((static_cast<int>(uy) < 0) == (ax < 0x3f800000u) ? 0x7f800000u : 0u);
    }
    if (ay == 0) return (ax > 0x7f800000u && ax < 0x7fc00000u) ? bits_to_float(ux | 0x400000u) : 1.0f;
    if (uy == 0x3f800000u) return ax > 0x7f800000u ? bits_to_float(ux | 0x400000u) : x;
    if (ax > 0x7f800000u) return bits_to_float(ux | 0x400000u);
    if (ax == 0x7f800000u || ax == 0) {
        const unsigned int mag = ((static_cast<int>(uy) < 0) == (ax == 0)) ? 0x7f800000u : 0u;
        return bits_to_float(mag | (odd_int() ? (ux & 0x80000000u) : 0u));
    }
    unsigned int sign = 0;
    if (static_cast<int>(ux) < 0 && (ay & 0x7f800000u) <= 0x4b000000u) {
        if (rintf(y) != y) return bits_to_float(0xffc00000u);
        if (odd_int()) sign = 0x80000000u;
    }
    const double dx = static_cast<double>(x), dy = static_cast<double>(y);
    if (static_cast<int>(ux) < 0x3f880000) {
        const double f = dx - 1.0;
        if (fabs(f) < 0x1p-4) {
            const double u = f / (f + 2.0);
            const double w = u + u;
            const double v = w * w;
            const double w3 = w * v;
            const double a = v * 0x1.9999999bac6d4p-7 + 0x1.55555555554e6p-4;
            const double b = v * 0x1.c8034c85dfff0p-12 + 0x1.2492307f1519fp-9;
            const double s = (((v * v) * w3) * b + w3 * a) - u * f;
            return RocketGpuUcrtPowfExp(dy * (f + s), sign, false);
        }
    }
    unsigned long long bx;
    memcpy(&bx, &dx, sizeof(bx));
    const unsigned long long mant = bx & 0x000fffffffffffffull;
    const unsigned int idx = static_cast<unsigned int>(mant >> 44) + static_cast<unsigned int>((mant >> 43) & 1);
    const unsigned long long cb = static_cast<unsigned long long>(0x3fe00u | idx) << 44;
    const unsigned long long mb = mant | 0x3fe0000000000000ull;
    double c, m;
    memcpy(&c, &cb, sizeof(c));
    memcpy(&m, &mb, sizeof(m));
    const int e = static_cast<int>((bx >> 52) & 0x7ff) - 0x3ff;
    const double r = (c - m) * kPowInvC[idx];
    const double rq = r * fma(fma(0x1.5555555555555p-2, r, 0.5), r, 1.0);
    const double log_x = (static_cast<double>(e) * 0x1.62e42fefa39efp-1 + kPowLogC[idx]) - rq;
    return RocketGpuUcrtPowfExp(dy * log_x, sign, true);
}
ROCKETGPU_HD float RocketGpuUcrtAcosf(float x) {
    const auto bits_to_float = [](unsigned int u) { float f; memcpy(&f, &u, sizeof(f)); return f; };
    unsigned int ux;
    memcpy(&ux, &x, sizeof(ux));
    const unsigned int ax = ux & 0x7fffffffu, ex = (ux >> 23) & 0xff;
    if (ax > 0x7f800000u) return bits_to_float(ux | 0x400000u);
    if (ex < 0x65) return bits_to_float(0x3fc90fdbu);
    if (ex >= 0x7f) {
        if (x == 1.0f) return 0.0f;
        if (x == -1.0f) return bits_to_float(0x40490fdbu);
        return bits_to_float(0xffc00000u);
    }
    const float a = fabsf(x);
    const bool big = ex >= 0x7e;
    const float z = big ? (1.0f - a) * 0.5f : a * a;
    const float s = big ? sqrtf(z) : 0.0f;
    float t = bits_to_float(0x3b81ce6bu);
    t = fmaf(-t, z, bits_to_float(0xbc5b3fe1u));
    t = fmaf(t, z, -bits_to_float(0x3d678bddu));
    t = fmaf(t, z, bits_to_float(0x3e3c94dcu));
    const float r = (t * z) / fmaf(-bits_to_float(0x3f561f0du), z, bits_to_float(0x3f8d6fa5u));
    if (!big) {
        const double lo = 0x1.1a62633145c07p-54 - static_cast<double>(r * x);
        return static_cast<float>(0x1.921fb54442d18p+0 - (static_cast<double>(x) - lo));
    }
    if (static_cast<int>(ux) < 0) {
        const double h = (static_cast<double>(s * r) - 0x1.1a62633145c07p-54) + static_cast<double>(s);
        return static_cast<float>(0x1.921fb54442d18p+1 - (h + h));
    }
    unsigned int sb;
    memcpy(&sb, &s, sizeof(sb));
    const float hi = bits_to_float(sb & 0xffff0000u);
    const float c = fmaf(-hi, hi, z) / (s + hi);
    return fmaf(r, s + s, c + c) + (hi + hi);
}

#if defined(__CUDA_ARCH__)

struct RocketGpuGlibcSincosTable {
    double sign[4];
    double hpi_inv;
    double hpi;
    double c0, c1, c2, c3, c4;
    double s1, s2, s3;
};
__device__ __forceinline__ const RocketGpuGlibcSincosTable& RocketGpuGlibcSincosEntry(int idx) {
    static const RocketGpuGlibcSincosTable kTable[2] = {
        {{1.0, -1.0, -1.0, 1.0}, 0x1.45F306DC9C883p+23, 0x1.921FB54442D18p0,
         0x1p0, -0x1.ffffffd0c621cp-2, 0x1.55553e1068f19p-5, -0x1.6c087e89a359dp-10, 0x1.99343027bf8c3p-16,
         -0x1.555545995a603p-3, 0x1.1107605230bc4p-7, -0x1.994eb3774cf24p-13},
        {{1.0, -1.0, -1.0, 1.0}, 0x1.45F306DC9C883p+23, 0x1.921FB54442D18p0,
         -0x1p0, 0x1.ffffffd0c621cp-2, -0x1.55553e1068f19p-5, 0x1.6c087e89a359dp-10, -0x1.99343027bf8c3p-16,
         -0x1.555545995a603p-3, 0x1.1107605230bc4p-7, -0x1.994eb3774cf24p-13}};
    return kTable[idx];
}
__device__ __forceinline__ unsigned int RocketGpuGlibcAbstop12(float x) {
    return (__float_as_uint(x) >> 20) & 0x7ffu;
}
__device__ __forceinline__ float RocketGpuGlibcSinfPoly(double x, double x2, const RocketGpuGlibcSincosTable& p, int n) {
    if ((n & 1) == 0) {
        const double x3 = x * x2;
        const double s1 = fma(x2, p.s3, p.s2);
        const double x7 = x3 * x2;
        const double s = fma(x3, p.s1, x);
        return static_cast<float>(fma(x7, s1, s));
    }
    const double x4 = x2 * x2;
    const double c2 = fma(x2, p.c4, p.c3);
    const double c1 = fma(x2, p.c1, p.c0);
    const double x6 = x4 * x2;
    const double c = fma(x4, p.c2, c1);
    return static_cast<float>(fma(x6, c2, c));
}
__device__ __forceinline__ double RocketGpuGlibcReduceLarge(unsigned int xi, int* np) {
    static const unsigned int kInvPio4[24] = {
        0xa2u,       0xa2f9u,     0xa2f983u,   0xa2f9836eu, 0xf9836e4eu, 0x836e4e44u, 0x6e4e4415u, 0x4e441529u,
        0x441529fcu, 0x1529fc27u, 0x29fc2757u, 0xfc2757d1u, 0x2757d1f5u, 0x57d1f534u, 0xd1f534ddu, 0xf534ddc0u,
        0x34ddc0dbu, 0xddc0db62u, 0xc0db6295u, 0xdb629599u, 0x6295993cu, 0x95993c43u, 0x993c4390u, 0x3c439041u};
    const unsigned int* arr = &kInvPio4[(xi >> 26) & 15];
    const int shift = (xi >> 23) & 7;
    xi = (xi & 0xffffffu) | 0x800000u;
    xi <<= shift;
    unsigned long long res0 = static_cast<unsigned int>(xi * arr[0]);
    const unsigned long long res1 = static_cast<unsigned long long>(xi) * arr[4];
    const unsigned long long res2 = static_cast<unsigned long long>(xi) * arr[8];
    res0 = (res2 >> 32) | (res0 << 32);
    res0 += res1;
    const unsigned long long n = (res0 + (1ull << 61)) >> 62;
    res0 -= n << 62;
    *np = static_cast<int>(n);
    return static_cast<double>(static_cast<long long>(res0)) * 0x1.921FB54442D18p-62;
}
__device__ inline float RocketGpuSinf(float y) {
    if (RocketGpuUcrtAbsBits(y) < kRocketGpuUcrtSinfLimit) {
        return RocketGpuUcrtSinf(y);
    }
    const unsigned int top = RocketGpuGlibcAbstop12(y);
    if (top < RocketGpuGlibcAbstop12(__int_as_float(0x7f800000))) {
        const unsigned int xi = __float_as_uint(y);
        int n = 0;
        const double xr = RocketGpuGlibcReduceLarge(xi, &n);
        const int ns = n + static_cast<int>(xi >> 31);
        const double sgn = RocketGpuGlibcSincosEntry(0).sign[ns & 3];
        const RocketGpuGlibcSincosTable& p = RocketGpuGlibcSincosEntry((ns & 2) ? 1 : 0);
        return RocketGpuGlibcSinfPoly(xr * sgn, xr * xr, p, n);
    }
    return static_cast<float>(sin(y));
}
__device__ inline float RocketGpuCosf(float y) {
    if (RocketGpuUcrtAbsBits(y) < kRocketGpuUcrtCosfLimit) {
        return RocketGpuUcrtCosf(y);
    }
    const unsigned int top = RocketGpuGlibcAbstop12(y);
    if (top < RocketGpuGlibcAbstop12(__int_as_float(0x7f800000))) {
        int n = 0;
        const double xr = RocketGpuGlibcReduceLarge(__float_as_uint(y), &n);
        const double sgn = RocketGpuGlibcSincosEntry(0).sign[n & 3];
        const RocketGpuGlibcSincosTable& p = RocketGpuGlibcSincosEntry((n & 2) ? 1 : 0);
        return RocketGpuGlibcSinfPoly(xr * sgn, xr * xr, p, n ^ 1);
    }
    return static_cast<float>(cos(y));
}
#else
inline float RocketGpuSinf(float x) {
    return RocketGpuUcrtAbsBits(x) < kRocketGpuUcrtSinfLimit ? RocketGpuUcrtSinf(x) : sinf(x);
}
inline float RocketGpuCosf(float x) {
    return RocketGpuUcrtAbsBits(x) < kRocketGpuUcrtCosfLimit ? RocketGpuUcrtCosf(x) : cosf(x);
}
#endif

namespace rocketgpu {
namespace detail {
ROCKETGPU_HD float DivideBy50(float x) {
#if defined(__CUDA_ARCH__)
    constexpr float kHi = 1.0f / 50.0f;
    constexpr float kLo = static_cast<float>(1.0 / 50.0 - static_cast<double>(kHi));
    const float ax = fabsf(x);
    if (ax != 0.0f && ax < 0x1p-64f) {
        return x / 50.0f;
    }
    return __fmaf_rn(x, kHi, __fmul_rn(x, kLo));
#else
    return x / 50.0f;
#endif
}

ROCKETGPU_HD constexpr Vec3 operator+(const Vec3& a, const Vec3& b) {
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

#if !defined(ROCKETGPU_USE_CUDA) && defined(__GNUC__)
#define ROCKETGPU_HOST_LANES 1
namespace lanes {
typedef float F4 __attribute__((vector_size(16)));
typedef int32_t I4 __attribute__((vector_size(16)));
inline F4 Load(const Vec3& v) { return F4{v.x, v.y, v.z, 0.0f}; }
inline F4 Splat(float s) { return F4{s, s, s, s}; }
inline Vec3 ToVec3(F4 v) { return Vec3(v[0], v[1], v[2]); }
inline F4 LoadPadded(const Vec3& v) { F4 r; memcpy(&r, &v, sizeof(r)); return r; }
inline void StorePadded(Vec3& v, F4 x) { memcpy(&v, &x, sizeof(x)); }
inline F4 Select(I4 m, F4 a, F4 b) { return reinterpret_cast<F4>((reinterpret_cast<I4>(a) & m) | (reinterpret_cast<I4>(b) & ~m)); }
inline F4 Abs(F4 v) { return Select(v >= Splat(0.0f), v, -v); }
inline F4 Min(F4 a, F4 b) { return Select(a < b, a, b); }
inline F4 Max(F4 a, F4 b) { return Select(a > b, a, b); }
template <int A, int B, int C>
inline F4 Turn(F4 v) {
#if defined(__clang__)
    return __builtin_shufflevector(v, v, A, B, C, 3);
#else
    return __builtin_shuffle(v, I4{A, B, C, 3});
#endif
}
}
#endif

ROCKETGPU_HD constexpr Vec3 operator-(const Vec3& a, const Vec3& b) {
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

ROCKETGPU_HD constexpr Vec3 operator-(const Vec3& v) {
    return Vec3(-v.x, -v.y, -v.z);
}

ROCKETGPU_HD constexpr Vec3 operator*(const Vec3& a, float s) {
    return Vec3(a.x * s, a.y * s, a.z * s);
}

ROCKETGPU_HD constexpr Vec3 operator*(float s, const Vec3& a) {
    return a * s;
}

ROCKETGPU_HD constexpr Vec3 operator/(const Vec3& a, float s) {
    return Vec3(a.x / s, a.y / s, a.z / s);
}

ROCKETGPU_HD Vec3& operator+=(Vec3& a, const Vec3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z; return a;
}

ROCKETGPU_HD Vec3& operator-=(Vec3& a, const Vec3& b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z; return a;
}

ROCKETGPU_HD Vec3& operator*=(Vec3& a, float s) {
    a.x *= s; a.y *= s; a.z *= s; return a;
}

ROCKETGPU_HD Vec3& operator/=(Vec3& a, float s) {
    a.x /= s; a.y /= s; a.z /= s; return a;
}

ROCKETGPU_HD constexpr float Dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

ROCKETGPU_HD constexpr Vec3 Cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

ROCKETGPU_HD float LengthSq(const Vec3& v) {
    return Dot(v, v);
}

ROCKETGPU_HD float SqrtScalar(float v) {
    return sqrtf(v);
}

ROCKETGPU_HD float Length(const Vec3& v) {
    return sqrtf(LengthSq(v));
}

ROCKETGPU_HD float RsqrtScalar(float v) {
    return 1.0f / sqrtf(v);
}

ROCKETGPU_HD Vec3 Normalize(const Vec3& v, const Vec3& fallback = Vec3(0.0f, 0.0f, 0.0f)) {
    const float len_sq = LengthSq(v);
    if (len_sq <= 1e-12f) {
        return fallback;
    }
    const float inv_len = RsqrtScalar(len_sq);
    return v * inv_len;
}

ROCKETGPU_HD Vec3 RocketSimVecNormalized(const Vec3& v) {
    const float len_sq = LengthSq(v);
    const float length = len_sq > 0.0f ? SqrtScalar(len_sq) : 0.0f;

    constexpr float kFltEpsilonSquared = 1.1920928955078125e-7f * 1.1920928955078125e-7f;
    if (length > kFltEpsilonSquared) {
        return Vec3(v.x / length, v.y / length, v.z / length);
    }
    return Vec3(0.0f, 0.0f, 0.0f);
}

ROCKETGPU_HD Vec3 Ue3SafeNormal(const Vec3& v) {
    const float len_sq = LengthSq(v);
    if (len_sq == 1.0f) {
        return v;
    }
    if (len_sq < 1e-8f) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }
    const float scale = 1.0f / SqrtScalar(len_sq);
    return Vec3(v.x * scale, v.y * scale, v.z * scale);
}

ROCKETGPU_HD Vec3 ClampLength(const Vec3& v, float max_len) {
    const float len_sq = LengthSq(v);
    const float max_sq = max_len * max_len;
    if (len_sq <= max_sq) {
        return v;
    }
    const float inv_len = RsqrtScalar(len_sq);

    return (v * inv_len) * max_len;
}

ROCKETGPU_HD constexpr Vec3 Hadamard(const Vec3& a, const Vec3& b) {
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

ROCKETGPU_HD constexpr Vec3 ClampVec(const Vec3& v, const Vec3& lo, const Vec3& hi) {
    return Vec3(
        Clamp(v.x, lo.x, hi.x),
        Clamp(v.y, lo.y, hi.y),
        Clamp(v.z, lo.z, hi.z));
}

ROCKETGPU_HD constexpr Quat operator+(const Quat& a, const Quat& b) {
    return Quat(a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z);
}

ROCKETGPU_HD constexpr Quat operator*(const Quat& q, float s) {
    return Quat(q.w * s, q.x * s, q.y * s, q.z * s);
}

ROCKETGPU_HD Quat& operator+=(Quat& a, const Quat& b) {
    a.w += b.w; a.x += b.x; a.y += b.y; a.z += b.z; return a;
}

ROCKETGPU_HD constexpr Quat Mul(const Quat& a, const Quat& b) {
    return Quat(
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w);
}

ROCKETGPU_HD constexpr Quat Conjugate(const Quat& q) {
    return Quat(q.w, -q.x, -q.y, -q.z);
}

ROCKETGPU_HD float LengthSq(const Quat& q) {
    return q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
}

ROCKETGPU_HD Quat Inverse(const Quat& q) {
    const float len_sq = LengthSq(q);
    if (len_sq <= 1e-12f) {
        return Quat(1.0f, 0.0f, 0.0f, 0.0f);
    }
    const float inv_len_sq = 1.0f / len_sq;
    return Conjugate(q) * inv_len_sq;
}

ROCKETGPU_HD Quat Normalize(const Quat& q) {
    const float len_sq = LengthSq(q);
    if (len_sq <= 1e-12f) {
        return Quat(1.0f, 0.0f, 0.0f, 0.0f);
    }
    const float inv_len = RsqrtScalar(len_sq);
    return Quat(q.w * inv_len, q.x * inv_len, q.y * inv_len, q.z * inv_len);
}

ROCKETGPU_HD float BulletQuatLength2(const Quat& q) {
    return (q.x * q.x + q.z * q.z) + (q.y * q.y + q.w * q.w);
}

ROCKETGPU_HD Quat BulletQuatMul(const Quat& q1, const Quat& q2) {
    const float a0x = q1.w * q2.x;
    const float a0y = q1.w * q2.y;
    const float a0z = q1.w * q2.z;
    const float a0w = q1.w * q2.w;

    const float bx = q1.z * q2.y;
    const float by = q1.x * q2.z;
    const float bz = q1.y * q2.x;
    const float bw = q1.z * q2.z;

    const float a1x = q1.x * q2.w;
    const float a1y = q1.y * q2.w;
    const float a1z = q1.z * q2.w;
    const float a1w = q1.x * q2.x;

    const float a2x = q1.y * q2.z;
    const float a2y = q1.z * q2.x;
    const float a2z = q1.x * q2.y;
    const float a2w = q1.y * q2.y;
    return Quat(
        (a0w - bw) - (a1w + a2w),
        (a0x - bx) + (a1x + a2x),
        (a0y - by) + (a1y + a2y),
        (a0z - bz) + (a1z + a2z));
}

ROCKETGPU_HD Quat BulletQuatSafeNormalize(const Quat& q) {
    constexpr float kBulletSimdEpsilon = 1.1920928955078125e-7f;
    const float len_sq = BulletQuatLength2(q);
    if (!(len_sq > kBulletSimdEpsilon)) {
        return q;
    }
    const float inv_len = RsqrtScalar(len_sq);
    return Quat(q.w * inv_len, q.x * inv_len, q.y * inv_len, q.z * inv_len);
}

ROCKETGPU_HD Quat FromYaw(float yaw) {
    constexpr float kYawSpecialCaseEps = 1e-6f;
    constexpr float kSinPiOver8 = 0.38268343236508977173f;
    constexpr float kCosPiOver8 = 0.92387953251128675613f;

    if (Abs(yaw - (Pi() * 0.25f)) <= kYawSpecialCaseEps) {
        return Quat(kCosPiOver8, 0.0f, 0.0f, kSinPiOver8);
    }
    if (Abs(yaw + (Pi() * 0.25f)) <= kYawSpecialCaseEps) {
        return Quat(kCosPiOver8, 0.0f, 0.0f, -kSinPiOver8);
    }
    if (Abs(yaw - (Pi() * 0.75f)) <= kYawSpecialCaseEps) {
        return Quat(kSinPiOver8, 0.0f, 0.0f, kCosPiOver8);
    }
    if (Abs(yaw + (Pi() * 0.75f)) <= kYawSpecialCaseEps) {
        return Quat(kSinPiOver8, 0.0f, 0.0f, -kCosPiOver8);
    }
    if (Abs(yaw - HalfPi()) <= kYawSpecialCaseEps) {
        return Quat(SqrtHalf(), 0.0f, 0.0f, SqrtHalf());
    }
    if (Abs(yaw + HalfPi()) <= kYawSpecialCaseEps) {
        return Quat(SqrtHalf(), 0.0f, 0.0f, -SqrtHalf());
    }

    const float h = 0.5f * yaw;
    return Quat(RocketGpuCosf(h), 0.0f, 0.0f, RocketGpuSinf(h));
}

ROCKETGPU_HD Vec3 Rotate(const Quat& q, const Vec3& v) {
    const Quat p(0.0f, v.x, v.y, v.z);
    const Quat r = Mul(Mul(q, p), Inverse(q));
    return Vec3(r.x, r.y, r.z);
}

ROCKETGPU_HD Vec3 InverseRotate(const Quat& q, const Vec3& v) {
    return Rotate(Inverse(q), v);
}

ROCKETGPU_HD Quat BulletQuatFromAxisAngle(const Vec3& axis, float angle) {
    const float d = Length(axis);
    const float s = RocketGpuSinf(angle * 0.5f) / d;
    return Quat(RocketGpuCosf(angle * 0.5f), axis.x * s, axis.y * s, axis.z * s);
}

ROCKETGPU_HD Vec3 BulletQuatRotate(const Quat& q, const Vec3& v) {
    const float rx = q.w * v.x + q.y * v.z - q.z * v.y;
    const float ry = q.w * v.y + q.z * v.x - q.x * v.z;
    const float rz = q.w * v.z + q.x * v.y - q.y * v.x;
    const float rw = -q.x * v.x - q.y * v.y - q.z * v.z;
    const float cx = -q.x;
    const float cy = -q.y;
    const float cz = -q.z;
    const float cw = q.w;
    return Vec3(
        rw * cx + rx * cw + ry * cz - rz * cy,
        rw * cy + ry * cw + rz * cx - rx * cz,
        rw * cz + rz * cw + rx * cy - ry * cx);
}

ROCKETGPU_HD void GetBasisVectorsSetRotationForm(const Quat& q, Vec3& forward, Vec3& right, Vec3& up);

ROCKETGPU_HD void GameSetRotationScalarColumns(const Quat& q, Vec3& fwd, Vec3& right, Vec3& up);

ROCKETGPU_HD void GetBasisVectors(const Quat& q, Vec3& forward, Vec3& right, Vec3& up) {
    const float len_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    if (len_sq <= 1e-12f) {
        forward = Vec3(1.0f, 0.0f, 0.0f);
        right = Vec3(0.0f, 1.0f, 0.0f);
        up = Vec3(0.0f, 0.0f, 1.0f);
        return;
    }

    const float s = 2.0f / len_sq;
    const float xs = q.x * s;
    const float ys = q.y * s;
    const float zs = q.z * s;
    const float wx = q.w * xs;
    const float wy = q.w * ys;
    const float wz = q.w * zs;
    const float xx = q.x * xs;
    const float xy = q.x * ys;
    const float xz = q.x * zs;
    const float yy = q.y * ys;
    const float yz = q.y * zs;
    const float zz = q.z * zs;

    forward = Vec3(1.0f - (yy + zz), xy + wz, xz - wy);
    right = Vec3(xy - wz, 1.0f - (xx + zz), yz + wx);
    up = Vec3(xz + wy, yz - wx, 1.0f - (xx + yy));
}

ROCKETGPU_HD Vec3 BulletMatrixRotate(const Quat& q, const Vec3& v) {
    Vec3 col0(1.0f, 0.0f, 0.0f);
    Vec3 col1(0.0f, 1.0f, 0.0f);
    Vec3 col2(0.0f, 0.0f, 1.0f);
    GameSetRotationScalarColumns(q, col0, col1, col2);
    return Vec3(
        col0.x * v.x + col1.x * v.y + col2.x * v.z,
        col0.y * v.x + col1.y * v.y + col2.y * v.z,
        col0.z * v.x + col1.z * v.y + col2.z * v.z);
}

ROCKETGPU_HD void GetBasisVectorsSetRotationForm(const Quat& q, Vec3& forward, Vec3& right, Vec3& up) {
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float d = (x * x + z * z) + (y * y + w * w);
    if (d <= 1e-12f) {
        forward = Vec3(1.0f, 0.0f, 0.0f);
        right = Vec3(0.0f, 1.0f, 0.0f);
        up = Vec3(0.0f, 0.0f, 1.0f);
        return;
    }
    const float s = 2.0f / d;

    forward = Vec3(1.0f - (y * y + z * z) * s, (x * y + z * w) * s + 0.0f, (z * x - w * y) * s + 0.0f);
    right = Vec3((x * y - w * z) * s + 0.0f, 1.0f - (x * x + z * z) * s, (y * z + w * x) * s + 0.0f);
    up = Vec3((z * x + y * w) * s + 0.0f, (y * z - w * x) * s + 0.0f, 1.0f - (x * x + y * y) * s);
}

ROCKETGPU_HD void GetOfficialUploadBasisVectors(const Quat& q, Vec3& forward, Vec3& right, Vec3& up) {
    const float len_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (len_sq <= 1e-12f) {
        forward = Vec3(1.0f, 0.0f, 0.0f);
        right = Vec3(0.0f, 1.0f, 0.0f);
        up = Vec3(0.0f, 0.0f, 1.0f);
        return;
    }

    const float s = 2.0f / len_sq;
    const float xs = q.x * s;
    const float ys = q.y * s;
    const float zs = q.z * s;
    const float wx = q.w * xs;
    const float wy = q.w * ys;
    const float wz = q.w * zs;
    const float xx = q.x * xs;
    const float xy = q.x * ys;
    const float xz = q.x * zs;
    const float yy = q.y * ys;
    const float yz = q.y * zs;
    const float zz = q.z * zs;

    forward = Vec3(1.0f - (yy + zz), xy + wz, xz - wy);
    right = Vec3(xy - wz, 1.0f - (xx + zz), yz + wx);
    up = Vec3(xz + wy, yz - wx, 1.0f - (xx + yy));
}

ROCKETGPU_HD Vec3 RotateViaBasis(const Vec3& forward, const Vec3& right, const Vec3& up, const Vec3& v) {
    return forward * v.x + right * v.y + up * v.z;
}

ROCKETGPU_HD Quat BulletQuatFromBasisColumns(const Vec3& fwd, const Vec3& right, const Vec3& up) {
    const float m00 = fwd.x, m01 = right.x, m02 = up.x;
    const float m10 = fwd.y, m11 = right.y, m12 = up.y;
    const float m20 = fwd.z, m21 = right.z, m22 = up.z;

    const float trace = m00 + m11 + m22;
    float raw0, raw1, raw2, raw3;
    float x;
    if (trace > 0.0f) {
        x = trace + 1.0f;
        raw0 = m21 - m12;
        raw1 = m02 - m20;
        raw2 = m10 - m01;
        raw3 = x;
    } else {
        const int i = m00 < m11 ? (m11 < m22 ? 2 : 1) : (m00 < m22 ? 2 : 0);
        if (i == 0) {
            x = m00 - m11 - m22 + 1.0f;
            raw0 = x;
            raw1 = m10 + m01;
            raw2 = m20 + m02;
            raw3 = m21 - m12;
        } else if (i == 1) {
            x = m11 - m22 - m00 + 1.0f;
            raw0 = m01 + m10;
            raw1 = x;
            raw2 = m21 + m12;
            raw3 = m02 - m20;
        } else {
            x = m22 - m00 - m11 + 1.0f;
            raw0 = m02 + m20;
            raw1 = m12 + m21;
            raw2 = x;
            raw3 = m10 - m01;
        }
    }

    const float s = sqrtf(x);
    const float scale = 0.5f / s;
    return Quat(raw3 * scale, raw0 * scale, raw1 * scale, raw2 * scale);
}

ROCKETGPU_HD Quat BulletQuatFromSetRotationBasis(const Quat& q) {
    Vec3 fwd, right, up;
    GetBasisVectors(q, fwd, right, up);
    return BulletQuatFromBasisColumns(fwd, right, up);
}

ROCKETGPU_HD void IntegrateOrientation(Quat& q, const Vec3& ang_vel_world, float dt) {
    constexpr float kAngularMotionThreshold = 0.5f * Pi() * 0.5f;

    const float angle_sq = LengthSq(ang_vel_world);
    float angle = 0.0f;
    if (angle_sq > 1e-12f) {
        angle = SqrtScalar(angle_sq);
    }

    if (angle * dt > kAngularMotionThreshold) {
        angle = kAngularMotionThreshold / dt;
    }

    Vec3 axis(0.0f, 0.0f, 0.0f);
    if (angle < 0.001f) {
        const float angle_dt_sq = angle * angle;
        axis = ang_vel_world * (0.5f * dt - (dt * dt * dt) * 0.020833333333f * angle_dt_sq);
    } else {
        axis = ang_vel_world * (RocketGpuSinf(0.5f * angle * dt) / angle);
    }

    const Quat delta(RocketGpuCosf(angle * dt * 0.5f), axis.x, axis.y, axis.z);
    const Quat orn0 = BulletQuatFromSetRotationBasis(q);
    const Quat predicted = BulletQuatSafeNormalize(BulletQuatMul(delta, orn0));

    constexpr float kBulletSimdEpsilon = 1.1920928955078125e-7f;
    if (!(BulletQuatLength2(predicted) > kBulletSimdEpsilon)) {
        return;
    }
    q = predicted;
}

ROCKETGPU_HD void BulletSetRotationColumns(const Quat& q, Vec3& fwd, Vec3& right, Vec3& up, bool native_sse_math = false) {
    if (native_sse_math) {
        GetBasisVectorsSetRotationForm(q, fwd, right, up);
        return;
    }
    const float d = BulletQuatLength2(q);
    const float s = 2.0f / d;
    const float xs = q.x * s;
    const float ys = q.y * s;
    const float zs = q.z * s;
    const float wx = q.w * xs;
    const float wy = q.w * ys;
    const float wz = q.w * zs;
    const float xx = q.x * xs;
    const float xy = q.x * ys;
    const float xz = q.x * zs;
    const float yy = q.y * ys;
    const float yz = q.y * zs;
    const float zz = q.z * zs;
    fwd = Vec3(1.0f - (yy + zz), xy + wz, xz - wy);
    right = Vec3(xy - wz, 1.0f - (xx + zz), yz + wx);
    up = Vec3(xz + wy, yz - wx, 1.0f - (xx + yy));
}

ROCKETGPU_HD Quat GameGetRotationScalar(const Vec3& fwd, const Vec3& right, const Vec3& up) {
    const float m00 = fwd.x, m01 = right.x, m02 = up.x;
    const float m10 = fwd.y, m11 = right.y, m12 = up.y;
    const float m20 = fwd.z, m21 = right.z, m22 = up.z;
    const float trace = (m00 + m11) + m22;
    if (trace > 0.0f) {
        float s = sqrtf(trace + 1.0f);
        const float w = s * 0.5f;
        s = 0.5f / s;
        return Quat(w, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s);
    }
    if (m00 < m11 ? m11 < m22 : m00 < m22) {
        float s = sqrtf(((m22 - m00) - m11) + 1.0f);
        const float z = s * 0.5f;
        s = 0.5f / s;
        return Quat((m10 - m01) * s, (m02 + m20) * s, (m12 + m21) * s, z);
    }
    if (m00 < m11) {
        float s = sqrtf(((m11 - m22) - m00) + 1.0f);
        const float y = s * 0.5f;
        s = 0.5f / s;
        return Quat((m02 - m20) * s, (m01 + m10) * s, y, (m21 + m12) * s);
    }
    float s = sqrtf(((m00 - m11) - m22) + 1.0f);
    const float x = s * 0.5f;
    s = 0.5f / s;
    return Quat((m21 - m12) * s, x, (m10 + m01) * s, (m20 + m02) * s);
}

ROCKETGPU_HD void GameSetRotationScalarColumns(const Quat& q, Vec3& fwd, Vec3& right, Vec3& up) {
    const float d = ((q.x * q.x + q.y * q.y) + q.z * q.z) + q.w * q.w;
    const float s = 2.0f / d;
    const float xs = q.x * s, ys = q.y * s, zs = q.z * s;
    const float wx = q.w * xs, wy = q.w * ys, wz = q.w * zs;
    const float xx = q.x * xs, xy = q.x * ys, xz = q.x * zs;
    const float yy = q.y * ys, yz = q.y * zs, zz = q.z * zs;
    fwd = Vec3(1.0f - (yy + zz), xy + wz, xz - wy);
    right = Vec3(xy - wz, 1.0f - (xx + zz), yz + wx);
    up = Vec3(xz + wy, yz - wx, 1.0f - (xx + yy));
}

ROCKETGPU_HD void GameUeQuatAxes(const Quat& q, Vec3& fwd, Vec3& right, Vec3& up) {
    const float z2 = q.z + q.z, y2 = q.y + q.y;
    const float x2x = (q.x + q.x) * q.x, x2w = (q.x + q.x) * q.w;
    fwd = Vec3(1.0f - (z2 * q.z + y2 * q.y), z2 * q.w + y2 * q.x, z2 * q.x - y2 * q.w);
    right = Vec3(y2 * q.x - z2 * q.w, 1.0f - (z2 * q.z + x2x), x2w + z2 * q.y);
    up = Vec3(y2 * q.w + z2 * q.x, z2 * q.y - x2w, 1.0f - (y2 * q.y + x2x));
}

ROCKETGPU_HD Vec3 GameUeQuatRotateVector(const Quat& q, const Vec3& v) {
    const float qq = (q.y * q.y + q.x * q.x) + q.z * q.z;
    const float d = q.w * q.w - qq;
    const float w2 = q.w + q.w;
    const float dot = (v.x * q.x + v.y * q.y) + v.z * q.z;
    const float t = dot + dot;
    const float cx = v.z * q.y - v.y * q.z, cy = v.x * q.z - v.z * q.x, cz = v.y * q.x - v.x * q.y;
    return Vec3((cx * w2 + d * v.x) + t * q.x, (cy * w2 + d * v.y) + t * q.y, (cz * w2 + d * v.z) + t * q.z);
}

ROCKETGPU_HD Quat GameQuatNormalizeScalar(const Quat& q, float min_len_sq) {
    const float len_sq = ((q.x * q.x + q.y * q.y) + q.z * q.z) + q.w * q.w;
    if (!(len_sq > min_len_sq)) {
        return Quat(1.0f, 0.0f, 0.0f, 0.0f);
    }
    const float s = 1.0f / sqrtf(len_sq);
    return Quat(q.w * s, q.x * s, q.y * s, q.z * s);
}

ROCKETGPU_HD float GameQuatQuantizeComponent(float c) {
    constexpr int kMax = (1 << 18) - 1;
    const float max_f = static_cast<float>(kMax);
    float v = ((c * 1.41421354f) * 0.5f + 0.5f) * max_f;
    v = (v + v) + 0.5f;
    int k = static_cast<int>(rintf(v));
    k = k < 0 ? (k - 1) / 2 : k / 2;
    k = k < 0 ? 0 : (k < kMax ? k : kMax);
    const float r = static_cast<float>(k) / max_f - 0.5f;
    return (r + r) * 0.707106769f;
}

ROCKETGPU_HD ROCKETGPU_HOST_INLINE Quat GameRbStateQuantizeQuat(const Quat& q_in) {
    const Quat q = GameQuatNormalizeScalar(q_in, 1e-8f);
    const float a0 = fabsf(q.x), a1 = fabsf(q.y), a2 = fabsf(q.z), a3 = fabsf(q.w);
    const int i01 = a0 < a1 ? 1 : 0;
    const int i23 = a2 < a3 ? 3 : 2;
    const int big = (i01 == 1 ? a1 : a0) >= (i23 == 3 ? a3 : a2) ? i01 : i23;
    const float c_big = big == 0 ? q.x : big == 1 ? q.y : big == 2 ? q.z : q.w;
    const float sign = c_big < 0.0f ? -1.0f : 1.0f;
    const float d0 = GameQuatQuantizeComponent((big == 0 ? q.y : q.x) * sign);
    const float d1 = GameQuatQuantizeComponent((big <= 1 ? q.z : q.y) * sign);
    const float d2 = GameQuatQuantizeComponent((big <= 2 ? q.w : q.z) * sign);
    const float rest = ((1.0f - d0 * d0) - d1 * d1) - d2 * d2;
#if defined(__CUDA_ARCH__)
    const float dropped = __fsqrt_rn(rest);
#else
    const float dropped = sqrtf(rest);
#endif
    const float ox = big == 0 ? dropped : d0;
    const float oy = big == 1 ? dropped : (big == 0 ? d0 : d1);
    const float oz = big == 2 ? dropped : (big <= 1 ? d1 : d2);
    const float ow = big == 3 ? dropped : d2;
    return GameQuatNormalizeScalar(Quat(ow, ox, oy, oz), 1e-8f);
}

ROCKETGPU_HD ROCKETGPU_HOST_INLINE Quat GameQuantizeBasisColumns(Vec3& fwd, Vec3& right, Vec3& up) {
    const Quat q = GameRbStateQuantizeQuat(GameGetRotationScalar(fwd, right, up));
    GameSetRotationScalarColumns(q, fwd, right, up);
    return q;
}

ROCKETGPU_HD void GameIntegrateOrientationBasisColumns(
    Vec3& fwd, Vec3& right, Vec3& up, Quat& q_mirror, const Vec3& ang_vel_world, float dt,
    const Quat* basis_rot = nullptr) {
    constexpr float kAngularMotionThreshold = 0.5f * Pi() * 0.5f;
    float angle = sqrtf((ang_vel_world.x * ang_vel_world.x + ang_vel_world.y * ang_vel_world.y) +
                        ang_vel_world.z * ang_vel_world.z);
    if (angle * dt > kAngularMotionThreshold) {
        angle = kAngularMotionThreshold / dt;
    }
    float f;
    if (angle < 0.001f) {
        f = dt * 0.5f - ((((dt * dt) * dt) * 0.020833333333f) * angle) * angle;
    } else {
        const float half_angle = (angle * 0.5f) * dt;
        f = RocketGpuSinf(half_angle) / angle;
    }
    const float ax = ang_vel_world.x * f, ay = ang_vel_world.y * f, az = ang_vel_world.z * f;
    const float c = RocketGpuCosf((angle * dt) * 0.5f);
    const Quat o = basis_rot != nullptr ? *basis_rot : GameGetRotationScalar(fwd, right, up);
    const Quat p(
        ((c * o.w - ax * o.x) - ay * o.y) - az * o.z,
        ((ax * o.w + c * o.x) + ay * o.z) - az * o.y,
        ((c * o.y + ay * o.w) + az * o.x) - ax * o.z,
        ((c * o.z + az * o.w) + ax * o.y) - ay * o.x);
    q_mirror = GameQuatNormalizeScalar(p, -1.0f);
    GameSetRotationScalarColumns(q_mirror, fwd, right, up);
}
#if defined(ROCKETGPU_HOST_LANES)

struct GameIntegrateLane {
    Quat o;
    Vec3 w;
    Vec3* fwd;
    Vec3* right;
    Vec3* up;
    Quat* q;
};

inline void RocketGpuUcrtSinCos4(lanes::F4 x, lanes::F4 y, lanes::F4& sin_x, lanes::F4& cos_y) {
    using namespace lanes;
    typedef double D4 __attribute__((vector_size(32)));
    const auto fma4 = [](D4 a, D4 b, D4 c) {
#if defined(__clang__)
        return __builtin_elementwise_fma(a, b, c);
#else
        return D4{fma(a[0], b[0], c[0]), fma(a[1], b[1], c[1]), fma(a[2], b[2], c[2]), fma(a[3], b[3], c[3])};
#endif
    };
    const auto splat_d = [](double v) { return D4{v, v, v, v}; };
    const I4 abs_mask = I4{0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
    const I4 tiny = I4{0x39000000, 0x39000000, 0x39000000, 0x39000000};
    const I4 small = I4{0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000};
    const D4 hd = __builtin_convertvector(x, D4);
    const D4 hd2 = hd * hd;
    D4 sp = fma4(hd2, splat_d(0x1.71de3a556c734p-19), splat_d(-0x1.a01a01a01a01ap-13));
    sp = fma4(sp, hd2, splat_d(0x1.1111111111111p-7));
    sp = fma4(sp, hd2, splat_d(-0x1.5555555555555p-3));
    sp = fma4(sp, hd * hd2, hd);
    const D4 s1 = fma4(-((hd * hd) * hd), splat_d(0x1.5555555555555p-3), hd);
    const I4 hb = reinterpret_cast<I4>(x) & abs_mask;
    sin_x = Select(hb < tiny, x, Select(hb < small, __builtin_convertvector(s1, F4), __builtin_convertvector(sp, F4)));
    const D4 cd = __builtin_convertvector(y, D4);
    const D4 cd2 = cd * cd;
    const D4 cbase = splat_d(1.0) - cd2 * splat_d(0.5);
    D4 cp = fma4(cd2, splat_d(-0x1.27e4fb7789f5cp-22), splat_d(0x1.a01a01a01a019p-16));
    cp = fma4(cp, cd2, splat_d(-0x1.6c16c16c16c16p-10));
    cp = fma4(cp, cd2, splat_d(0x1.5555555555555p-5));
    cp = fma4(cp, cd2 * cd2, cbase);
    const D4 c1 = fma4(-(cd * splat_d(0.5)), cd, splat_d(1.0));
    const I4 cb = reinterpret_cast<I4>(y) & abs_mask;
    cos_y = Select(cb < tiny, Splat(1.0f), Select(cb < small, __builtin_convertvector(c1, F4), __builtin_convertvector(cp, F4)));
}

inline uint32_t GameIntegrateOrientationBasisColumns4(int n, const GameIntegrateLane* in, float dt) {
    using namespace lanes;
    constexpr float kAngularMotionThreshold = 0.5f * Pi() * 0.5f;
    const auto sqrt4 = [](F4 v) {
#if defined(__clang__)
        return __builtin_elementwise_sqrt(v);
#else
        return F4{sqrtf(v[0]), sqrtf(v[1]), sqrtf(v[2]), sqrtf(v[3])};
#endif
    };
    F4 wx = Splat(0.0f), wy = Splat(0.0f), wz = Splat(0.0f);
    F4 ow = Splat(1.0f), ox = Splat(0.0f), oy = Splat(0.0f), oz = Splat(0.0f);
    for (int k = 0; k < n; ++k) {
        wx[k] = in[k].w.x;
        wy[k] = in[k].w.y;
        wz[k] = in[k].w.z;
        ow[k] = in[k].o.w;
        ox[k] = in[k].o.x;
        oy[k] = in[k].o.y;
        oz[k] = in[k].o.z;
    }
    const F4 dt4 = Splat(dt);
    F4 angle = sqrt4((wx * wx + wy * wy) + wz * wz);
    angle = Select(angle * dt4 > Splat(kAngularMotionThreshold), Splat(kAngularMotionThreshold / dt), angle);
    const F4 half = (angle * Splat(0.5f)) * dt4;
    const F4 carg = (angle * dt4) * Splat(0.5f);
    const I4 poly_limit = I4{0x3f490fdb, 0x3f490fdb, 0x3f490fdb, 0x3f490fdb};
    const I4 abs_mask = I4{0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
    const I4 bad = ((reinterpret_cast<I4>(half) & abs_mask) > poly_limit) | ((reinterpret_cast<I4>(carg) & abs_mask) > poly_limit);
    F4 sine, c;
    RocketGpuUcrtSinCos4(half, carg, sine, c);
    const F4 f_small = Splat(dt * 0.5f) - ((Splat(((dt * dt) * dt) * 0.020833333333f) * angle) * angle);
    const F4 f = Select(angle < Splat(0.001f), f_small, sine / angle);
    const F4 ax = wx * f, ay = wy * f, az = wz * f;
    const F4 pw = ((c * ow - ax * ox) - ay * oy) - az * oz;
    const F4 px = ((ax * ow + c * ox) + ay * oz) - az * oy;
    const F4 py = ((c * oy + ay * ow) + az * ox) - ax * oz;
    const F4 pz = ((c * oz + az * ow) + ax * oy) - ay * ox;
    const F4 len_sq = ((px * px + py * py) + pz * pz) + pw * pw;
    const I4 ok = len_sq > Splat(-1.0f);
    const F4 inv = Splat(1.0f) / sqrt4(len_sq);
    const F4 qw = Select(ok, pw * inv, Splat(1.0f)), qx = Select(ok, px * inv, Splat(0.0f));
    const F4 qy = Select(ok, py * inv, Splat(0.0f)), qz = Select(ok, pz * inv, Splat(0.0f));
    const F4 d = ((qx * qx + qy * qy) + qz * qz) + qw * qw;
    const F4 s = Splat(2.0f) / d;
    const F4 xs = qx * s, ys = qy * s, zs = qz * s;
    const F4 wxs = qw * xs, wys = qw * ys, wzs = qw * zs;
    const F4 xx = qx * xs, xy = qx * ys, xz = qx * zs;
    const F4 yy = qy * ys, yz = qy * zs, zz = qz * zs;
    const F4 one = Splat(1.0f);
    const F4 f0 = one - (yy + zz), f1 = xy + wzs, f2 = xz - wys;
    const F4 r0 = xy - wzs, r1 = one - (xx + zz), r2 = yz + wxs;
    const F4 u0 = xz + wys, u1 = yz - wxs, u2 = one - (xx + yy);
    uint32_t left = 0u;
    for (int k = 0; k < n; ++k) {
        if (bad[k] != 0) {
            left |= 1u << k;
            continue;
        }
        *in[k].fwd = Vec3(f0[k], f1[k], f2[k]);
        *in[k].right = Vec3(r0[k], r1[k], r2[k]);
        *in[k].up = Vec3(u0[k], u1[k], u2[k]);
        if (in[k].q != nullptr) {
            *in[k].q = Quat(qw[k], qx[k], qy[k], qz[k]);
        }
    }
    return left;
}

#endif

ROCKETGPU_HD bool IntegrateOrientationBasisColumns(
    Vec3& fwd, Vec3& right, Vec3& up, Quat& q_mirror, const Vec3& ang_vel_world, float dt, bool native_sse_math = false) {
    constexpr float kBulletSimdEpsilon = 1.1920928955078125e-7f;
    constexpr float kAngularMotionThreshold = 0.5f * Pi() * 0.5f;
    const float angle_sq = LengthSq(ang_vel_world);
    float angle = 0.0f;
    if (angle_sq > kBulletSimdEpsilon) {
        angle = SqrtScalar(angle_sq);
    }
    if (angle * dt > kAngularMotionThreshold) {
        angle = kAngularMotionThreshold / dt;
    }
    Vec3 axis(0.0f, 0.0f, 0.0f);
    if (angle < 0.001f) {
        axis = ang_vel_world * (0.5f * dt - (dt * dt * dt) * 0.020833333333f * angle * angle);
    } else {
        axis = ang_vel_world * (RocketGpuSinf(0.5f * angle * dt) / angle);
    }
    const float cosine = RocketGpuCosf(angle * dt * 0.5f);
    const Quat delta(cosine, axis.x, axis.y, axis.z);
    const Quat orn0 = BulletQuatFromBasisColumns(fwd, right, up);
    const Quat predicted_raw = BulletQuatMul(delta, orn0);
    const Quat predicted = BulletQuatSafeNormalize(predicted_raw);
    if (!(BulletQuatLength2(predicted) > kBulletSimdEpsilon)) {
        return false;
    }
    BulletSetRotationColumns(predicted, fwd, right, up, native_sse_math);
    q_mirror = predicted;
    return true;
}

ROCKETGPU_HD void IntegrateCarOrientation(CarRefT car, const Vec3& ang_vel_world, float dt, bool native_sse_math = false,
                                          const Quat* basis_rot = nullptr) {
    if (!car.basis_authoritative) {
        IntegrateOrientation(car.rot, ang_vel_world, dt);
        return;
    }
    if (!native_sse_math) {
        GameIntegrateOrientationBasisColumns(
            car.basis_forward, car.basis_right, car.basis_up, car.rot, ang_vel_world, dt, basis_rot);
        return;
    }
    IntegrateOrientationBasisColumns(
        car.basis_forward, car.basis_right, car.basis_up, car.rot, ang_vel_world, dt, native_sse_math);
}

ROCKETGPU_HD void CarBasisColumns(CarCRefT car, Vec3& fwd, Vec3& right, Vec3& up) {
    if (car.basis_authoritative) {
        fwd = car.basis_forward;
        right = car.basis_right;
        up = car.basis_up;
        return;
    }
    GetBasisVectors(car.rot, fwd, right, up);
}

ROCKETGPU_HD Vec3 RotateCarBasis(CarCRefT car, const Vec3& v) {
    if (!car.basis_authoritative) return Rotate(car.rot, v);
    return RotateViaBasis(car.basis_forward, car.basis_right, car.basis_up, v);
}

ROCKETGPU_HD Vec3 BulletPlaneSpace1First(const Vec3& n) {
    constexpr float kSimdSqrt12 = 0.7071067811865475244008443621048490f;
    if (Abs(n.z) > kSimdSqrt12) {
        const float a = n.y * n.y + n.z * n.z;
        const float k = 1.0f / SqrtScalar(a);
        return Vec3(0.0f, -n.z * k, n.y * k);
    }
    const float a = n.x * n.x + n.y * n.y;
    const float k = 1.0f / SqrtScalar(a);
    return Vec3(-n.y * k, n.x * k, 0.0f);
}
ROCKETGPU_HD Vec3 InverseRotateCarBasis(CarCRefT car, const Vec3& v) {
    if (!car.basis_authoritative) return InverseRotate(car.rot, v);
    return Vec3(Dot(car.basis_forward, v), Dot(car.basis_right, v), Dot(car.basis_up, v));
}

ROCKETGPU_HD void CanonicalizeImportedCarOrientation(CarRefT car) {
    if (!car.basis_authoritative) {
        GetOfficialUploadBasisVectors(car.rot, car.basis_forward, car.basis_right, car.basis_up);
        car.basis_authoritative = 1;
    }
    car.rot = BulletQuatFromBasisColumns(car.basis_forward, car.basis_right, car.basis_up);
}

ROCKETGPU_HD uint32_t GlibcFloatBits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}
ROCKETGPU_HD float GlibcBitsToFloat(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

ROCKETGPU_HD float Ue3SinTab(int j) {
    const float a = ((static_cast<float>(j) + static_cast<float>(j)) * 3.14159274f) * 6.103515625e-05f;
    return RocketGpuUcrtSinf(a);
}

struct Ue3Rotator {
    int pitch;
    int yaw;
    int roll;
};

ROCKETGPU_HD float Ue3RotatorSin(int a) { return Ue3SinTab((a >> 2) & 16383); }

ROCKETGPU_HD Quat Ue3QuatFromYawRotator(int yaw) {
    const float sy = Ue3RotatorSin(yaw), cy = Ue3RotatorSin(yaw + 16384);
    const float cp = Ue3RotatorSin(16384);
    const float m[3][3] = {{cy * cp, cp * sy, 0.0f}, {-(cp * sy), cy * cp, -0.0f}, {-0.0f, 0.0f, cp * cp}};
    const float tr = (m[1][1] + m[0][0]) + m[2][2];
    if (tr > 0.0f) {
        const float inv_s = 1.0f / sqrtf(tr + 1.0f);
        const float s = inv_s * 0.5f;
        return Quat(0.5f / inv_s, (m[1][2] - m[2][1]) * s, (m[2][0] - m[0][2]) * s, (m[0][1] - m[1][0]) * s);
    }
    int i = m[0][0] < m[1][1] ? 1 : 0;
    if (m[i][i] <= m[2][2] && m[2][2] != m[i][i]) i = 2;
    const int j = i == 2 ? 0 : i + 1, k = j == 2 ? 0 : j + 1;
    const float inv_s = 1.0f / sqrtf(((m[i][i] - m[j][j]) - m[k][k]) + 1.0f);
    const float s = inv_s * 0.5f;
    float v[3];
    v[i] = 0.5f / inv_s;
    v[j] = (m[i][j] + m[j][i]) * s;
    v[k] = (m[i][k] + m[k][i]) * s;
    return Quat((m[j][k] - m[k][j]) * s, v[0], v[1], v[2]);
}

ROCKETGPU_HD int SpawnYawRotatorUnits(float yaw) { return static_cast<int>(yaw * (32768.0f / 3.14159274f) + 0.5f); }

ROCKETGPU_HD void SeedCarBasisFromSpawnRotator(CarRefT car, int blue_yaw, bool orange) {
    Quat q = Ue3QuatFromYawRotator(blue_yaw);
    if (orange) {
        const Quat h = Ue3QuatFromYawRotator(32768);
        q = Quat(((h.w * q.w - h.x * q.x) - h.y * q.y) - h.z * q.z,
                 ((h.w * q.x + h.x * q.w) + h.y * q.z) - h.z * q.y,
                 ((h.w * q.y + h.y * q.w) + h.z * q.x) - h.x * q.z,
                 ((h.w * q.z + h.z * q.w) + h.x * q.y) - h.y * q.x);
    }
    GameSetRotationScalarColumns(q, car.basis_forward, car.basis_right, car.basis_up);
    car.basis_authoritative = 1;
}

ROCKETGPU_HD Ue3Rotator Ue3QuatToRotator(const Quat& q) {
    const float x2 = q.x + q.x, y2 = q.y + q.y, z2 = q.z + q.z;
    const float xx = q.x * x2, xy = q.x * y2, xz = q.x * z2;
    const float yy = q.y * y2, yz = q.y * z2, zz = q.z * z2;
    const float wx = q.w * x2, wy = q.w * y2, wz = q.w * z2;
    const Vec3 fwd(1.0f - (yy + zz), xy + wz, xz - wy);
    const Vec3 right(xy - wz, 1.0f - (xx + zz), yz + wx);
    const Vec3 up(xz + wy, yz - wx, 1.0f - (xx + yy));
    const auto ue3_round = [](float v) { return static_cast<int>(rintf(v + v + 0.5f)) >> 1; };
    Ue3Rotator r;
    r.yaw = ue3_round(RocketGpuUcrtAtan2f(fwd.y, fwd.x) * 32768.0f / 3.14159274f);
    const float fwd_xy = static_cast<float>(sqrt(static_cast<double>(fwd.y * fwd.y + fwd.x * fwd.x)));
    r.pitch = ue3_round(RocketGpuUcrtAtan2f(fwd.z, fwd_xy) * 32768.0f / 3.14159274f);
    const float sy = Ue3RotatorSin(r.yaw);
    const float cy = Ue3RotatorSin(r.yaw + 16384);
    const float z_dot = (up.y * cy + up.x * -sy) + up.z * -0.0f;
    const float y_dot = (right.y * cy + right.x * -sy) + right.z * -0.0f;
    r.roll = ue3_round(RocketGpuUcrtAtan2f(z_dot, y_dot) * 32768.0f / 3.14159274f);
    return r;
}

ROCKETGPU_HD void SpawnInertiaBasis(const Quat& spawn_rot, Vec3& forward, Vec3& right, Vec3& up) {
    Vec3 f, r, u;
    GameUeQuatAxes(Ue3QuatFromYawRotator(Ue3QuatToRotator(spawn_rot).yaw), f, r, u);
    GameSetRotationScalarColumns(GameGetRotationScalar(f, r, u), forward, right, up);
}

ROCKETGPU_HD Vec3 Ue3RotatorForwardAxis(const Ue3Rotator& r) {
    const float cp = Ue3RotatorSin(r.pitch + 16384);
    return Vec3(cp * Ue3RotatorSin(r.yaw + 16384), cp * Ue3RotatorSin(r.yaw), Ue3RotatorSin(r.pitch));
}

ROCKETGPU_HD Vec3 Ue3RotatorUpAxis(const Ue3Rotator& r) {
    const float sy = Ue3RotatorSin(r.yaw);
    const float cy = Ue3RotatorSin(r.yaw + 16384);
    const float sp = Ue3RotatorSin(r.pitch);
    const float cp = Ue3RotatorSin(r.pitch + 16384);
    const float sr = Ue3RotatorSin(r.roll);
    const float cr = Ue3RotatorSin(r.roll + 16384);
    const float cr_sp = cr * sp;
    return Vec3(-(cr_sp * cy + sy * sr), cy * sr - cr_sp * sy, cp * cr);
}

ROCKETGPU_HD void Ue3RotatorAxes(const Ue3Rotator& r, Vec3& x, Vec3& y, Vec3& z) {
    const float sp = Ue3RotatorSin(r.pitch), cp = Ue3RotatorSin(r.pitch + 16384);
    const float sy = Ue3RotatorSin(r.yaw), cy = Ue3RotatorSin(r.yaw + 16384);
    const float sr = Ue3RotatorSin(r.roll), cr = Ue3RotatorSin(r.roll + 16384);
    const float sp_sr = sp * sr;
    const float cr_sp = cr * sp;
    x = Vec3(cy * cp, cp * sy, sp);
    y = Vec3(sp_sr * cy - cr * sy, cy * cr + sp_sr * sy, -sr * cp);
    z = Vec3(-(cr_sp * cy + sy * sr), cy * sr - cr_sp * sy, cp * cr);
}

ROCKETGPU_HD float SafeAtan2(float y, float x) {
#if defined(ROCKETGPU_USE_CUDA)
    return static_cast<float>(atan2(static_cast<double>(y), static_cast<double>(x)));
#else
    return static_cast<float>(std::atan2(static_cast<double>(y), static_cast<double>(x)));
#endif
}

ROCKETGPU_HD Vec3 BulletHitboxSize(const CarConfig& cfg) {
    const Vec3 sized = cfg.hitbox_published + Vec3(2.5f, 2.5f, 2.5f);
    const bool has_published = sized.x == cfg.hitbox_size.x && sized.y == cfg.hitbox_size.y && sized.z == cfg.hitbox_size.z;
    const Vec3 published = has_published ? cfg.hitbox_published : cfg.hitbox_size - Vec3(2.5f, 2.5f, 2.5f);
    const Vec3 half_bt = (published * 0.02f) * 0.5f + Vec3(0.025f, 0.025f, 0.025f);
    return (half_bt * 50.0f) * 2.0f;
}

ROCKETGPU_HD float BoundingSphereRadius(const CarConfig& cfg) {
    const Vec3 e = cfg.hitbox_size * 0.5f;

    return Length(e) + Length(cfg.hitbox_offset) + 5.0f;
}

ROCKETGPU_HD float ApproximateGoalHalfWidth() {
    return 892.755f;
}

ROCKETGPU_HD float ApproximateGoalHeight() {
    return 642.775f;
}

ROCKETGPU_HD float ApproximateGoalDepth() {
    return 880.0f;
}
}
}
