#pragma once

#include "rocketgpu/rocketgpu.h"
#if defined(_WIN32)
#include "rocketgpu/detail/windows_small_angle_math.hpp"
#endif

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
__device__ __forceinline__ double RocketGpuGlibcReduceFast(double x, const RocketGpuGlibcSincosTable& p, int* np) {
    const double r = x * p.hpi_inv;
    const int n = (static_cast<int>(r) + 0x800000) >> 24;
    *np = n;
    return fma(-static_cast<double>(n), p.hpi, x);
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
__device__ __forceinline__ float RocketGpuSinf(float y) {
    const unsigned int top = RocketGpuGlibcAbstop12(y);
    const unsigned int top_pio4 = RocketGpuGlibcAbstop12(0x1.921FB6p-1f);
    const unsigned int top_120 = RocketGpuGlibcAbstop12(120.0f);
    const double x = static_cast<double>(y);
    if (top < top_pio4) {
        const double x2 = x * x;
        if (top < RocketGpuGlibcAbstop12(0x1p-12f)) {
            return y;
        }
        return RocketGpuGlibcSinfPoly(x, x2, RocketGpuGlibcSincosEntry(0), 0);
    }
    if (top < top_120) {
        int n = 0;
        const double xr = RocketGpuGlibcReduceFast(x, RocketGpuGlibcSincosEntry(0), &n);
        const double sgn = RocketGpuGlibcSincosEntry(0).sign[n & 3];
        const RocketGpuGlibcSincosTable& p = RocketGpuGlibcSincosEntry((n & 2) ? 1 : 0);
        return RocketGpuGlibcSinfPoly(xr * sgn, xr * xr, p, n);
    }
    return static_cast<float>(sin(x));
}
__device__ __forceinline__ float RocketGpuCosf(float y) {
    const unsigned int top = RocketGpuGlibcAbstop12(y);
    const unsigned int top_pio4 = RocketGpuGlibcAbstop12(0x1.921FB6p-1f);
    const unsigned int top_120 = RocketGpuGlibcAbstop12(120.0f);
    const double x = static_cast<double>(y);
    if (top < top_pio4) {
        const double x2 = x * x;
        if (top < RocketGpuGlibcAbstop12(0x1p-12f)) {
            return 1.0f;
        }
        return RocketGpuGlibcSinfPoly(x, x2, RocketGpuGlibcSincosEntry(0), 1);
    }
    if (top < top_120) {
        int n = 0;
        const double xr = RocketGpuGlibcReduceFast(x, RocketGpuGlibcSincosEntry(0), &n);
        const double sgn = RocketGpuGlibcSincosEntry(0).sign[n & 3];
        const RocketGpuGlibcSincosTable& p = RocketGpuGlibcSincosEntry((n & 2) ? 1 : 0);
        return RocketGpuGlibcSinfPoly(xr * sgn, xr * xr, p, n ^ 1);
    }
    return static_cast<float>(cos(x));
}
#else
inline float RocketGpuSinf(float x) { return sinf(x); }
inline float RocketGpuCosf(float x) { return cosf(x); }
#endif

namespace rocketgpu {
namespace detail {
ROCKETGPU_HD constexpr Vec3 operator+(const Vec3& a, const Vec3& b) {
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

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

ROCKETGPU_HD Vec3 NormalizeBulletStyle(const Vec3& v, const Vec3& fallback = Vec3(0.0f, 0.0f, 0.0f)) {
    const float len_sq = LengthSq(v);
    if (len_sq <= 1e-12f) {
        return fallback;
    }
    return v * (1.0f / SqrtScalar(len_sq));
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

ROCKETGPU_HD Quat AxisAngle(const Vec3& axis_unit, float angle) {
    const float h = 0.5f * angle;
    const float s = RocketGpuSinf(h);
    const float c = RocketGpuCosf(h);
    return Normalize(Quat(c, axis_unit.x * s, axis_unit.y * s, axis_unit.z * s));
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

ROCKETGPU_HD void GetBasisVectorsBulletQuatDot(const Quat& q, Vec3& forward, Vec3& right, Vec3& up) {
    const float len_sq = (q.x * q.x + q.z * q.z) + (q.y * q.y + q.w * q.w);
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
    GetBasisVectorsBulletQuatDot(q, col0, col1, col2);
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
    float raw[4];
    float x;
    if (trace > 0.0f) {
        x = trace + 1.0f;
        raw[0] = m21 - m12;
        raw[1] = m02 - m20;
        raw[2] = m10 - m01;
        raw[3] = x;
    } else {
        int i = m00 < m11 ? (m11 < m22 ? 2 : 1) : (m00 < m22 ? 2 : 0);
        int j = (i + 1) % 3;
        int k = (i + 2) % 3;
        auto mel = [&](int r, int c) -> float {
            const float rows[3][3] = {
                {m00, m01, m02},
                {m10, m11, m12},
                {m20, m21, m22}
            };
            return rows[r][c];
        };
        x = mel(i, i) - mel(j, j) - mel(k, k) + 1.0f;
        raw[3] = mel(k, j) - mel(j, k);
        raw[j] = mel(j, i) + mel(i, j);
        raw[k] = mel(k, i) + mel(i, k);
        raw[i] = x;
    }

    const float s = sqrtf(x);
    const float scale = 0.5f / s;
    return Quat(raw[3] * scale, raw[0] * scale, raw[1] * scale, raw[2] * scale);
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

// Bullet's scalar quaternion forms, which the game uses. Bullet's SSE forms above group sums
// differently and round differently, so they are not bit-exact against the game.
// btMatrix3x3::getRotation, scalar build: w = sqrt(trace + 1) * 0.5, the rest times 0.5 / s.
ROCKETGPU_HD Quat GameGetRotationScalar(const Vec3& fwd, const Vec3& right, const Vec3& up) {
    // m[r][c]: fwd, right, up are the columns. Bullet's i/j/k index form is spelled out per i (no dynamic
    // indexing, which puts the matrix in local memory on the GPU); the arithmetic is the same.
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
        // i = 2, j = 0, k = 1
        float s = sqrtf(((m22 - m00) - m11) + 1.0f);
        const float z = s * 0.5f;
        s = 0.5f / s;
        return Quat((m10 - m01) * s, (m02 + m20) * s, (m12 + m21) * s, z);
    }
    if (m00 < m11) {
        // i = 1, j = 2, k = 0
        float s = sqrtf(((m11 - m22) - m00) + 1.0f);
        const float y = s * 0.5f;
        s = 0.5f / s;
        return Quat((m02 - m20) * s, (m01 + m10) * s, y, (m21 + m12) * s);
    }
    // i = 0, j = 1, k = 2
    float s = sqrtf(((m00 - m11) - m22) + 1.0f);
    const float x = s * 0.5f;
    s = 0.5f / s;
    return Quat((m21 - m12) * s, x, (m10 + m01) * s, (m20 + m02) * s);
}

// btMatrix3x3::setRotation, scalar build: s = 2 / (((x*x + y*y) + z*z) + w*w).
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

// q * 1 / sqrt(((x*x + y*y) + z*z) + w*w), as FQuat::Normalize and integrateTransform do; identity at or below
// min_len_sq (FQuat uses 1e-8; integrateTransform has no guard, pass a negative value).
ROCKETGPU_HD Quat GameQuatNormalizeScalar(const Quat& q, float min_len_sq) {
    const float len_sq = ((q.x * q.x + q.y * q.y) + q.z * q.z) + q.w * q.w;
    if (!(len_sq > min_len_sq)) {
        return Quat(1.0f, 0.0f, 0.0f, 0.0f);
    }
    const float s = 1.0f / sqrtf(len_sq);
    return Quat(q.w * s, q.x * s, q.y * s, q.z * s);
}

// The game's replicated-rigid-body quaternion quantization (FRigidBodyState quantize, which writes then
// reads the state through the net serializer with 18 bits). Encode:
// normalize; drop the component of largest magnitude (ties to the lower index of x, y, z, w); if it is negative,
// negate the other three; each other component c in order goes to clamp(RoundToInt(((c * sqrt2) * 0.5 + 0.5) *
// (2^18 - 1)), 0, 2^18 - 1), where RoundToInt(v) = cvtss2si(v + v + 0.5) >> 1. Decode: ((k / (2^18 - 1)) - 0.5) * 2
// * (1 / sqrt2); the dropped component is sqrt(((1 - a*a) - b*b) - c*c), taken in double; normalize again.
ROCKETGPU_HD float GameQuatQuantizeComponent(float c) {
    constexpr int kMax = (1 << 18) - 1;
    const float max_f = static_cast<float>(kMax);
    float v = ((c * 1.41421354f) * 0.5f + 0.5f) * max_f;
    v = (v + v) + 0.5f;
    int k = static_cast<int>(rintf(v));
    k = k < 0 ? (k - 1) / 2 : k / 2;  // arithmetic shift right by one
    k = k < 0 ? 0 : (k < kMax ? k : kMax);
    const float r = static_cast<float>(k) / max_f - 0.5f;
    return (r + r) * 0.707106769f;
}

ROCKETGPU_HD Quat GameRbStateQuantizeQuat(const Quat& q_in) {
    const Quat q = GameQuatNormalizeScalar(q_in, 1e-8f);
    const float c[4] = {q.x, q.y, q.z, q.w};
    const float a[4] = {fabsf(c[0]), fabsf(c[1]), fabsf(c[2]), fabsf(c[3])};
    const int i01 = a[0] < a[1] ? 1 : 0;
    const int i23 = a[2] < a[3] ? 3 : 2;
    const int big = a[i01] >= a[i23] ? i01 : i23;
    const float sign = c[big] < 0.0f ? -1.0f : 1.0f;
    float d[3];
    for (int i = 0, n = 0; i < 4; ++i) {
        if (i != big) d[n++] = GameQuatQuantizeComponent(c[i] * sign);
    }
    const float rest = ((1.0f - d[0] * d[0]) - d[1] * d[1]) - d[2] * d[2];
    // The game takes this root in double and rounds to float; for rest >= 0 that equals the correctly rounded sqrtf
    // (53 >= 2 * 24 + 2 bits, so double rounding is innocuous), which keeps FP64 off the GPU.
#if defined(__CUDA_ARCH__)
    const float dropped = __fsqrt_rn(rest);
#else
    const float dropped = sqrtf(rest);
#endif
    float o[4];
    for (int i = 0, n = 0; i < 4; ++i) o[i] = i == big ? dropped : d[n++];
    return GameQuatNormalizeScalar(Quat(o[3], o[0], o[1], o[2]), 1e-8f);
}

// Quantize an orientation given as basis columns, as the game's tick-start state quantize does; returns the quat.
ROCKETGPU_HD Quat GameQuantizeBasisColumns(Vec3& fwd, Vec3& right, Vec3& up) {
    const Quat q = GameRbStateQuantizeQuat(GameGetRotationScalar(fwd, right, up));
    GameSetRotationScalarColumns(q, fwd, right, up);
    return q;
}

// btTransformUtil::integrateTransform orientation part, scalar build: dorn * getRotation(basis),
// normalized without a guard, then setRotation.
ROCKETGPU_HD void GameIntegrateOrientationBasisColumns(
    Vec3& fwd, Vec3& right, Vec3& up, Quat& q_mirror, const Vec3& ang_vel_world, float dt,
    bool soccar_platform_math = false) {
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
#if defined(__CUDA_ARCH__) && defined(_WIN32)
        const float sine = soccar_platform_math ? WindowsUcrtFmaSmallSin(half_angle) : RocketGpuSinf(half_angle);
#else
        const float sine = RocketGpuSinf(half_angle);
#endif
        f = sine / angle;
    }
    const float ax = ang_vel_world.x * f, ay = ang_vel_world.y * f, az = ang_vel_world.z * f;
    const float cosine_angle = (angle * dt) * 0.5f;
#if defined(__CUDA_ARCH__) && defined(_WIN32)
    const float c = soccar_platform_math ? WindowsUcrtFmaSmallCos(cosine_angle) : RocketGpuCosf(cosine_angle);
#else
    (void)soccar_platform_math;
    const float c = RocketGpuCosf(cosine_angle);
#endif
    const Quat o = GameGetRotationScalar(fwd, right, up);
    const Quat p(
        ((c * o.w - ax * o.x) - ay * o.y) - az * o.z,
        ((ax * o.w + c * o.x) + ay * o.z) - az * o.y,
        ((c * o.y + ay * o.w) + az * o.x) - ax * o.z,
        ((c * o.z + az * o.w) + ax * o.y) - ay * o.x);
    q_mirror = GameQuatNormalizeScalar(p, -1.0f);
    GameSetRotationScalarColumns(q_mirror, fwd, right, up);
}

ROCKETGPU_HD bool IntegrateOrientationBasisColumns(
    Vec3& fwd, Vec3& right, Vec3& up, Quat& q_mirror, const Vec3& ang_vel_world, float dt, bool soccar_platform_math = false,
    bool native_sse_math = false) {
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
        const float half_angle = 0.5f * angle * dt;
#if defined(__CUDA_ARCH__) && defined(_WIN32)

        const float sine = soccar_platform_math
            ? WindowsUcrtFmaSmallSin(half_angle) : RocketGpuSinf(half_angle);
#else
        (void)soccar_platform_math;
        const float sine = RocketGpuSinf(half_angle);
#endif
        axis = ang_vel_world * (sine / angle);
    }
    const float cosine_angle = angle * dt * 0.5f;
#if defined(__CUDA_ARCH__) && defined(_WIN32)
    const float cosine = soccar_platform_math
        ? WindowsUcrtFmaSmallCos(cosine_angle) : RocketGpuCosf(cosine_angle);
#else
    const float cosine = RocketGpuCosf(cosine_angle);
#endif
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

ROCKETGPU_HD void IntegrateCarOrientation(CarState& car, const Vec3& ang_vel_world, float dt, const char* site = "final", bool soccar_platform_math = false, bool native_sse_math = false) {
    if (!car.basis_authoritative) {
        IntegrateOrientation(car.rot, ang_vel_world, dt);
        return;
    }
    if (!native_sse_math) {
        GameIntegrateOrientationBasisColumns(
            car.basis_forward, car.basis_right, car.basis_up, car.rot, ang_vel_world, dt, soccar_platform_math);
        return;
    }
    IntegrateOrientationBasisColumns(
        car.basis_forward, car.basis_right, car.basis_up, car.rot, ang_vel_world, dt, soccar_platform_math, native_sse_math);
}

ROCKETGPU_HD void CarBasisColumns(const CarState& car, Vec3& fwd, Vec3& right, Vec3& up) {
    if (car.basis_authoritative) {
        fwd = car.basis_forward;
        right = car.basis_right;
        up = car.basis_up;
        return;
    }
    GetBasisVectors(car.rot, fwd, right, up);
}

ROCKETGPU_HD Vec3 CarUp(const CarState& car) {
    if (car.basis_authoritative) return car.basis_up;
    Vec3 f, r, u; GetBasisVectors(car.rot, f, r, u); return u;
}

ROCKETGPU_HD Vec3 RotateCarBasis(const CarState& car, const Vec3& v) {
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
ROCKETGPU_HD Vec3 InverseRotateCarBasis(const CarState& car, const Vec3& v) {
    if (!car.basis_authoritative) return InverseRotate(car.rot, v);
    return Vec3(Dot(car.basis_forward, v), Dot(car.basis_right, v), Dot(car.basis_up, v));
}

ROCKETGPU_HD void CanonicalizeImportedCarOrientation(CarState& car) {
    if (!car.basis_authoritative) {
        GetOfficialUploadBasisVectors(car.rot, car.basis_forward, car.basis_right, car.basis_up);
        car.basis_authoritative = 1;
    }
    car.rot = BulletQuatFromBasisColumns(car.basis_forward, car.basis_right, car.basis_up);
}

ROCKETGPU_HD void SeedCarBasisFromSetEulerYPR(CarState& car, float yaw) {
    const float euler_x = -0.0f;
    const float euler_y = -0.0f;
    const float euler_z = yaw;
    const float ci = RocketGpuCosf(euler_x);
    const float cj = RocketGpuCosf(euler_y);
    const float ch = RocketGpuCosf(euler_z);
    const float si = RocketGpuSinf(euler_x);
    const float sj = RocketGpuSinf(euler_y);
    const float sh = RocketGpuSinf(euler_z);
    const float cc = ci * ch;
    const float cs = ci * sh;
    const float sc = si * ch;
    const float ss = si * sh;
    const float m00 = cj * ch, m01 = sj * sc - cs, m02 = sj * cc + ss;
    const float m10 = cj * sh, m11 = sj * ss + cc, m12 = sj * cs - sc;
    const float m20 = -sj, m21 = cj * si, m22 = cj * ci;
    car.basis_forward = Vec3(m00, m10, m20);
    car.basis_right = Vec3(m01, m11, m21);
    car.basis_up = Vec3(m02, m12, m22);
    car.basis_authoritative = 1;
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
ROCKETGPU_HD float GlibcAtanf(float x) {
    const float atanhi[4] = {GlibcBitsToFloat(0x3eed6338u), GlibcBitsToFloat(0x3f490fdau),
                             GlibcBitsToFloat(0x3f7b985eu), GlibcBitsToFloat(0x3fc90fdau)};
    const float atanlo[4] = {GlibcBitsToFloat(0x31ac3769u), GlibcBitsToFloat(0x33222168u),
                             GlibcBitsToFloat(0x33140fb4u), GlibcBitsToFloat(0x33a22168u)};
    const float aT[11] = {GlibcBitsToFloat(0x3eaaaaabu), GlibcBitsToFloat(0xbe4ccccdu), GlibcBitsToFloat(0x3e124925u),
                          GlibcBitsToFloat(0xbde38e38u), GlibcBitsToFloat(0x3dba2e6eu), GlibcBitsToFloat(0xbd9d8795u),
                          GlibcBitsToFloat(0x3d886b35u), GlibcBitsToFloat(0xbd6ef16bu), GlibcBitsToFloat(0x3d4bda59u),
                          GlibcBitsToFloat(0xbd15a221u), GlibcBitsToFloat(0x3c8569d7u)};
    constexpr float one = 1.0f;
    constexpr float huge = 1.0e30f;
    const int32_t hx = static_cast<int32_t>(GlibcFloatBits(x));
    const int32_t ix = hx & 0x7fffffff;
    int id;
    if (ix >= 0x4c000000) {
        if (ix > 0x7f800000) return x + x;
        if (hx > 0) return atanhi[3] + atanlo[3];
        return -atanhi[3] - atanlo[3];
    }
    if (ix < 0x3ee00000) {
        if (ix < 0x31000000) {
            if (huge + x > one) return x;
        }
        id = -1;
    } else {
        x = fabsf(x);
        if (ix < 0x3f980000) {
            if (ix < 0x3f300000) {
                id = 0;
                x = (2.0f * x - one) / (2.0f + x);
            } else {
                id = 1;
                x = (x - one) / (x + one);
            }
        } else {
            if (ix < 0x401c0000) {
                id = 2;
                x = (x - 1.5f) / (one + 1.5f * x);
            } else {
                id = 3;
                x = -1.0f / x;
            }
        }
    }
    const float z0 = x * x;
    const float w = z0 * z0;
    const float s1 = z0 * (aT[0] + w * (aT[2] + w * (aT[4] + w * (aT[6] + w * (aT[8] + w * aT[10])))));
    const float s2 = w * (aT[1] + w * (aT[3] + w * (aT[5] + w * (aT[7] + w * aT[9]))));
    if (id < 0) return x - x * (s1 + s2);
    const float z = atanhi[id] - ((x * (s1 + s2) - atanlo[id]) - x);
    return (hx < 0) ? -z : z;
}
ROCKETGPU_HD float GlibcAtan2f(float y, float x) {
    constexpr float tiny = 1.0e-30f;
    constexpr float zero = 0.0f;
    const float pi_o_4 = GlibcBitsToFloat(0x3f490fdbu);
    const float pi_o_2 = GlibcBitsToFloat(0x3fc90fdbu);
    const float pi = GlibcBitsToFloat(0x40490fdbu);
    const float pi_lo = GlibcBitsToFloat(0xb3bbbd2eu);
    const int32_t hx = static_cast<int32_t>(GlibcFloatBits(x));
    const int32_t ix = hx & 0x7fffffff;
    const int32_t hy = static_cast<int32_t>(GlibcFloatBits(y));
    const int32_t iy = hy & 0x7fffffff;
    if ((ix > 0x7f800000) || (iy > 0x7f800000)) return x + y;
    if (hx == 0x3f800000) return GlibcAtanf(y);
    const int m = ((hy >> 31) & 1) | ((hx >> 30) & 2);
    if (iy == 0) {
        switch (m) {
            case 0:
            case 1: return y;
            case 2: return pi + tiny;
            default: return -pi - tiny;
        }
    }
    if (ix == 0) return (hy < 0) ? -pi_o_2 - tiny : pi_o_2 + tiny;
    if (ix == 0x7f800000) {
        if (iy == 0x7f800000) {
            switch (m) {
                case 0: return pi_o_4 + tiny;
                case 1: return -pi_o_4 - tiny;
                case 2: return 3.0f * pi_o_4 + tiny;
                default: return -3.0f * pi_o_4 - tiny;
            }
        } else {
            switch (m) {
                case 0: return zero;
                case 1: return -zero;
                case 2: return pi + tiny;
                default: return -pi - tiny;
            }
        }
    }
    if (iy == 0x7f800000) return (hy < 0) ? -pi_o_2 - tiny : pi_o_2 + tiny;
    const int32_t k = (iy - ix) >> 23;
    float z;
    if (k > 26) {
        z = pi_o_2 + 0.5f * pi_lo;
    } else if (k < -26 && hx < 0) {
        z = 0.0f;
    } else {
        z = GlibcAtanf(fabsf(y / x));
    }
    switch (m) {
        case 0: return z;
        case 1: return -z;
        case 2: return pi - (z - pi_lo);
        default: return (z - pi_lo) - pi;
    }
}

ROCKETGPU_HD float SafeAtan2(float y, float x) {
#if defined(ROCKETGPU_USE_CUDA)
    return static_cast<float>(atan2(static_cast<double>(y), static_cast<double>(x)));
#else
    return static_cast<float>(std::atan2(static_cast<double>(y), static_cast<double>(x)));
#endif
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
