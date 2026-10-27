/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2008 Erwin Coumans  https://bulletphysics.org

This file is a device-safe local port of Bullet's btGjkEpa2.h/.cpp, adapted
to use rocketgpu math/types while keeping the control flow and structure
as close as practical to the original source.
*/

#pragma once

#include <cstdlib>

#include "rocketgpu/detail/math.hpp"

namespace rocketgpu {
namespace detail {
namespace bullet_port {
#if defined(__CUDACC__) && defined(_MSC_VER)
#define ROCKETGPU_BULLET_PORT_NOINLINE __noinline__
#elif defined(__CUDACC__)
#define ROCKETGPU_BULLET_PORT_NOINLINE __noinline__ __attribute__((noinline))
#elif defined(__GNUC__) || defined(__clang__)
#define ROCKETGPU_BULLET_PORT_NOINLINE __attribute__((noinline))
#else
#define ROCKETGPU_BULLET_PORT_NOINLINE
#endif

using btScalar = float;

ROCKETGPU_HD btScalar btSseStyleDot3(
    btScalar ax,
    btScalar ay,
    btScalar az,
    btScalar bx,
    btScalar by,
    btScalar bz) {
#if defined(__CUDA_ARCH__)

    return __fadd_rn(
        __fadd_rn(__fmul_rn(ax, bx), __fmul_rn(ay, by)),
        __fmul_rn(az, bz));
#else
    return ax * bx + ay * by + az * bz;
#endif
}

#ifndef BT_LARGE_FLOAT

#define BT_LARGE_FLOAT 1.0e18f
#endif

#ifndef SIMD_EPSILON
#define SIMD_EPSILON 1.1920929e-7f
#endif

#ifndef SIMD_INFINITY

#define SIMD_INFINITY 3.402823466e+38f
#endif

#ifndef btAssert
#define btAssert(x) ((void)0)
#endif

struct btQuaternion {
    btScalar m_x = 0.0f;
    btScalar m_y = 0.0f;
    btScalar m_z = 0.0f;
    btScalar m_w = 1.0f;

    ROCKETGPU_HD btQuaternion() = default;
    ROCKETGPU_HD btQuaternion(btScalar x, btScalar y, btScalar z, btScalar w)
        : m_x(x), m_y(y), m_z(z), m_w(w) {}

    ROCKETGPU_HD btScalar x() const { return m_x; }
    ROCKETGPU_HD btScalar y() const { return m_y; }
    ROCKETGPU_HD btScalar z() const { return m_z; }
    ROCKETGPU_HD btScalar w() const { return m_w; }
    ROCKETGPU_HD btScalar dot(const btQuaternion& q) const {
        return m_x * q.x() + m_y * q.y() + m_z * q.z() + m_w * q.w();
    }
    ROCKETGPU_HD btScalar length2() const { return dot(*this); }
};

struct btVector3 {
    btScalar m_x = 0.0f;
    btScalar m_y = 0.0f;
    btScalar m_z = 0.0f;

    ROCKETGPU_HD btVector3() = default;
    ROCKETGPU_HD btVector3(btScalar x, btScalar y, btScalar z) : m_x(x), m_y(y), m_z(z) {}

    ROCKETGPU_HD btScalar x() const { return m_x; }
    ROCKETGPU_HD btScalar y() const { return m_y; }
    ROCKETGPU_HD btScalar z() const { return m_z; }
    ROCKETGPU_HD btScalar getX() const { return m_x; }
    ROCKETGPU_HD btScalar getY() const { return m_y; }
    ROCKETGPU_HD btScalar getZ() const { return m_z; }
    ROCKETGPU_HD void setValue(btScalar x, btScalar y, btScalar z) { m_x = x; m_y = y; m_z = z; }
    ROCKETGPU_HD void setZero() { setValue(0.0f, 0.0f, 0.0f); }
    ROCKETGPU_HD btScalar& operator[](int idx) { return idx == 0 ? m_x : (idx == 1 ? m_y : m_z); }
    ROCKETGPU_HD btScalar operator[](int idx) const { return idx == 0 ? m_x : (idx == 1 ? m_y : m_z); }
    ROCKETGPU_HD btVector3 operator-() const { return btVector3(-m_x, -m_y, -m_z); }
    ROCKETGPU_HD btVector3& operator+=(const btVector3& o) {
        m_x += o.m_x;
        m_y += o.m_y;
        m_z += o.m_z;
        return *this;
    }
    ROCKETGPU_HD btVector3& operator-=(const btVector3& o) {
        m_x -= o.m_x;
        m_y -= o.m_y;
        m_z -= o.m_z;
        return *this;
    }
    ROCKETGPU_HD btVector3& operator*=(btScalar s) {
        m_x *= s;
        m_y *= s;
        m_z *= s;
        return *this;
    }
    ROCKETGPU_HD btVector3& operator/=(btScalar s) {
        *this *= btScalar(1.0) / s;
        return *this;
    }
    ROCKETGPU_HD btScalar dot(const btVector3& o) const {
        return btSseStyleDot3(m_x, m_y, m_z, o.m_x, o.m_y, o.m_z);
    }
    ROCKETGPU_HD btVector3 cross(const btVector3& o) const {
        return btVector3(
            m_y * o.m_z - m_z * o.m_y,
            m_z * o.m_x - m_x * o.m_z,
            m_x * o.m_y - m_y * o.m_x);
    }
    ROCKETGPU_HD btScalar length2() const { return dot(*this); }
    ROCKETGPU_HD btScalar length() const { return SqrtScalar(length2()); }
    ROCKETGPU_HD void normalize() {
        const btScalar l2 = length2();
        if (l2 > 1.0e-20f) {
            *this /= SqrtScalar(l2);
        }
    }
    ROCKETGPU_HD btVector3 safeNormalize() const {
        btVector3 out = *this;
        const btScalar l2 = out.length2();
        if (l2 > 1.0e-20f) {
            out /= SqrtScalar(l2);
        } else {
            out.setZero();
        }
        return out;
    }
    ROCKETGPU_HD btVector3 dot3(const btVector3& a, const btVector3& b, const btVector3& c) const {
        return btVector3(dot(a), dot(b), dot(c));
    }
    ROCKETGPU_HD int maxAxis() const {
        return m_x < m_y ? (m_y < m_z ? 2 : 1) : (m_x < m_z ? 2 : 0);
    }
};

ROCKETGPU_HD btVector3 operator+(const btVector3& a, const btVector3& b) {
    return btVector3(a.x() + b.x(), a.y() + b.y(), a.z() + b.z());
}

ROCKETGPU_HD btVector3 operator-(const btVector3& a, const btVector3& b) {
    return btVector3(a.x() - b.x(), a.y() - b.y(), a.z() - b.z());
}

ROCKETGPU_HD btVector3 operator*(const btVector3& a, btScalar s) {
    return btVector3(a.x() * s, a.y() * s, a.z() * s);
}

ROCKETGPU_HD btVector3 operator*(btScalar s, const btVector3& a) {
    return a * s;
}

ROCKETGPU_HD btVector3 operator/(const btVector3& a, btScalar s) {
    const btScalar inv = btScalar(1.0) / s;
    return btVector3(a.x() * inv, a.y() * inv, a.z() * inv);
}

ROCKETGPU_HD btScalar btDot(const btVector3& a, const btVector3& b) {
    return a.dot(b);
}

ROCKETGPU_HD btVector3 btCross(const btVector3& a, const btVector3& b) {
    return a.cross(b);
}

ROCKETGPU_HD btScalar btSqrt(btScalar v) {
    return SqrtScalar(v);
}

ROCKETGPU_HD btScalar btFabs(btScalar v) {
    return Abs(v);
}

template <typename T>
ROCKETGPU_HD T btMax(const T& a, const T& b) {
    return a > b ? a : b;
}

template <typename T>
ROCKETGPU_HD void btSwap(T& a, T& b) {
    T tmp = a;
    a = b;
    b = tmp;
}

struct btMatrix3x3 {
    btVector3 m_el[3];

    ROCKETGPU_HD btMatrix3x3()
        : m_el{btVector3(1.0f, 0.0f, 0.0f), btVector3(0.0f, 1.0f, 0.0f), btVector3(0.0f, 0.0f, 1.0f)} {}

    ROCKETGPU_HD btMatrix3x3(const btVector3& row0, const btVector3& row1, const btVector3& row2)
        : m_el{row0, row1, row2} {}

    ROCKETGPU_HD explicit btMatrix3x3(const btQuaternion& q) {
        const btScalar d = q.length2();
        const btScalar s = btScalar(2.0) / d;
        const btScalar xs = q.x() * s;
        const btScalar ys = q.y() * s;
        const btScalar zs = q.z() * s;
        const btScalar wx = q.w() * xs;
        const btScalar wy = q.w() * ys;
        const btScalar wz = q.w() * zs;
        const btScalar xx = q.x() * xs;
        const btScalar xy = q.x() * ys;
        const btScalar xz = q.x() * zs;
        const btScalar yy = q.y() * ys;
        const btScalar yz = q.y() * zs;
        const btScalar zz = q.z() * zs;
        m_el[0] = btVector3(btScalar(1.0) - (yy + zz), xy - wz, xz + wy);
        m_el[1] = btVector3(xy + wz, btScalar(1.0) - (xx + zz), yz - wx);
        m_el[2] = btVector3(xz - wy, yz + wx, btScalar(1.0) - (xx + yy));
    }

    ROCKETGPU_HD const btVector3& operator[](int i) const { return m_el[i]; }
    ROCKETGPU_HD btVector3& operator[](int i) { return m_el[i]; }

    ROCKETGPU_HD btVector3 getColumn(int i) const {
        return btVector3(m_el[0][i], m_el[1][i], m_el[2][i]);
    }

    ROCKETGPU_HD btMatrix3x3 transpose() const {
        return btMatrix3x3(getColumn(0), getColumn(1), getColumn(2));
    }

    ROCKETGPU_HD btMatrix3x3 absolute() const {
        return btMatrix3x3(
            btVector3(btFabs(m_el[0].x()), btFabs(m_el[0].y()), btFabs(m_el[0].z())),
            btVector3(btFabs(m_el[1].x()), btFabs(m_el[1].y()), btFabs(m_el[1].z())),
            btVector3(btFabs(m_el[2].x()), btFabs(m_el[2].y()), btFabs(m_el[2].z())));
    }

    ROCKETGPU_HD btScalar tdotx(const btVector3& v) const {
        return m_el[0].x() * v.x() + m_el[1].x() * v.y() + m_el[2].x() * v.z();
    }

    ROCKETGPU_HD btScalar tdoty(const btVector3& v) const {
        return m_el[0].y() * v.x() + m_el[1].y() * v.y() + m_el[2].y() * v.z();
    }

    ROCKETGPU_HD btScalar tdotz(const btVector3& v) const {
        return m_el[0].z() * v.x() + m_el[1].z() * v.y() + m_el[2].z() * v.z();
    }

    ROCKETGPU_HD btMatrix3x3 transposeTimes(const btMatrix3x3& m) const {
        return btMatrix3x3(
            btVector3(
                m_el[0].x() * m[0].x() + m_el[1].x() * m[1].x() + m_el[2].x() * m[2].x(),
                m_el[0].x() * m[0].y() + m_el[1].x() * m[1].y() + m_el[2].x() * m[2].y(),
                m_el[0].x() * m[0].z() + m_el[1].x() * m[1].z() + m_el[2].x() * m[2].z()),
            btVector3(
                m_el[0].y() * m[0].x() + m_el[1].y() * m[1].x() + m_el[2].y() * m[2].x(),
                m_el[0].y() * m[0].y() + m_el[1].y() * m[1].y() + m_el[2].y() * m[2].y(),
                m_el[0].y() * m[0].z() + m_el[1].y() * m[1].z() + m_el[2].y() * m[2].z()),
            btVector3(
                m_el[0].z() * m[0].x() + m_el[1].z() * m[1].x() + m_el[2].z() * m[2].x(),
                m_el[0].z() * m[0].y() + m_el[1].z() * m[1].y() + m_el[2].z() * m[2].y(),
                m_el[0].z() * m[0].z() + m_el[1].z() * m[1].z() + m_el[2].z() * m[2].z()));
    }

    ROCKETGPU_HD btVector3 operator*(const btVector3& v) const {
        return btVector3(btDot(m_el[0], v), btDot(m_el[1], v), btDot(m_el[2], v));
    }
};

ROCKETGPU_HD btVector3 operator*(const btVector3& v, const btMatrix3x3& m) {
    return btVector3(m.tdotx(v), m.tdoty(v), m.tdotz(v));
}

struct btTransform {
    btMatrix3x3 m_basis;
    btVector3 m_origin = btVector3(0.0f, 0.0f, 0.0f);

    ROCKETGPU_HD btTransform() = default;
    ROCKETGPU_HD explicit btTransform(const btVector3& origin) : m_basis(), m_origin(origin) {}
    ROCKETGPU_HD btTransform(const btQuaternion& q, const btVector3& origin)
        : m_basis(btMatrix3x3(q)), m_origin(origin) {}

    ROCKETGPU_HD const btMatrix3x3& getBasis() const { return m_basis; }
    ROCKETGPU_HD btMatrix3x3& getBasis() { return m_basis; }
    ROCKETGPU_HD const btVector3& getOrigin() const { return m_origin; }
    ROCKETGPU_HD btVector3& getOrigin() { return m_origin; }
    ROCKETGPU_HD void setBasis(const btMatrix3x3& basis) { m_basis = basis; }
    ROCKETGPU_HD void setOrigin(const btVector3& origin) { m_origin = origin; }

    ROCKETGPU_HD btTransform inverseTimes(const btTransform& other) const {
        btTransform out;
        out.m_basis = m_basis.transposeTimes(other.m_basis);
        out.m_origin = (other.m_origin - m_origin) * m_basis;
        return out;
    }

    ROCKETGPU_HD btVector3 invXform(const btVector3& inVec) const {
        const btVector3 v = inVec - m_origin;
        return m_basis.transpose() * v;
    }

    ROCKETGPU_HD btVector3 operator*(const btVector3& v) const {
        return v.dot3(m_basis[0], m_basis[1], m_basis[2]) + m_origin;
    }

    ROCKETGPU_HD btVector3 operator()(const btVector3& v) const {
        return (*this) * v;
    }
};

enum btShapeKind : uint8_t {
    BT_SHAPE_BOX = 0,
    BT_SHAPE_TRIANGLE = 1,
    BT_SHAPE_SPHERE = 2,

    BT_SHAPE_CONVEX_HULL = 3,
};

struct btConvexShape {
    btShapeKind m_kind = BT_SHAPE_BOX;
    btVector3 m_implicitShapeDimensions = btVector3(0.0f, 0.0f, 0.0f);
    btVector3 m_vertices1[3];
    btScalar m_margin = 0.0f;

    const btVector3* m_hull_points = nullptr;
    int m_hull_point_count = 0;
    btVector3 m_hull_local_aabb_min = btVector3(1.0f, 1.0f, 1.0f);
    btVector3 m_hull_local_aabb_max = btVector3(-1.0f, -1.0f, -1.0f);

    ROCKETGPU_HD int hullSupportIndex(const btVector3& vec) const {
        btScalar max_dot = -3.402823466e+38f;
        int max_index = -1;
        for (int i = 0; i < m_hull_point_count; ++i) {
            const btVector3& p = m_hull_points[i];
            const btScalar dot = (p.x() * vec.x() + p.y() * vec.y()) + p.z() * vec.z();
            if (dot > max_dot) {
                max_dot = dot;
                max_index = i;
            }
        }
        return max_index;
    }

    ROCKETGPU_HD btVector3 localGetSupportVertexWithoutMarginNonVirtual(const btVector3& vec) const {
        if (m_kind == BT_SHAPE_BOX) {
            return btVector3(
                vec.x() >= 0.0f ? m_implicitShapeDimensions.x() : -m_implicitShapeDimensions.x(),
                vec.y() >= 0.0f ? m_implicitShapeDimensions.y() : -m_implicitShapeDimensions.y(),
                vec.z() >= 0.0f ? m_implicitShapeDimensions.z() : -m_implicitShapeDimensions.z());
        }
        if (m_kind == BT_SHAPE_TRIANGLE) {
            const btVector3 dots = vec.dot3(m_vertices1[0], m_vertices1[1], m_vertices1[2]);
            return m_vertices1[dots.maxAxis()];
        }
        if (m_kind == BT_SHAPE_CONVEX_HULL) {
            const int index = hullSupportIndex(vec);
            if (index < 0) {
                return btVector3(0.0f, 0.0f, 0.0f);
            }

            return m_hull_points[index];
        }
        return btVector3(0.0f, 0.0f, 0.0f);
    }

    ROCKETGPU_HD void recalcHullLocalAabb() {
        const btVector3 axes[6] = {
            btVector3(1.0f, 0.0f, 0.0f), btVector3(0.0f, 1.0f, 0.0f), btVector3(0.0f, 0.0f, 1.0f),
            btVector3(-1.0f, 0.0f, 0.0f), btVector3(0.0f, -1.0f, 0.0f), btVector3(0.0f, 0.0f, -1.0f)};
        btVector3 supporting[6];
        for (int d = 0; d < 6; ++d) {
            supporting[d] = localGetSupportVertexWithoutMarginNonVirtual(axes[d]);
        }
        for (int i = 0; i < 3; ++i) {
            m_hull_local_aabb_max[i] = supporting[i][i] + m_margin;
            m_hull_local_aabb_min[i] = supporting[i + 3][i] - m_margin;
        }
    }

    ROCKETGPU_HD btVector3 localGetSupportVertexNonVirtual(const btVector3& vec) const {
        btVector3 vecnorm = vec;
        if (vecnorm.length2() < (SIMD_EPSILON * SIMD_EPSILON)) {
            vecnorm.setValue(-1.0f, -1.0f, -1.0f);
        }
        vecnorm.normalize();
        return localGetSupportVertexWithoutMarginNonVirtual(vecnorm) + m_margin * vecnorm;
    }

    ROCKETGPU_HD btScalar getMarginNonVirtual() const {
        return m_margin;
    }

    ROCKETGPU_HD btVector3 localGetSupportingVertexVirtual(const btVector3& vec) const {
        btVector3 supVertex = localGetSupportVertexWithoutMarginNonVirtual(vec);
        if (m_margin != btScalar(0.)) {
            btVector3 vecnorm = vec;
            if (vecnorm.length2() < (SIMD_EPSILON * SIMD_EPSILON)) {
                vecnorm.setValue(btScalar(-1.), btScalar(-1.), btScalar(-1.));
            }
            vecnorm.normalize();
            supVertex += m_margin * vecnorm;
        }
        return supVertex;
    }

    ROCKETGPU_HD void getAabb(const btTransform& t, btVector3& aabb_min, btVector3& aabb_max) const {
        if (m_kind == BT_SHAPE_CONVEX_HULL) {
            btVector3 local_half_extents = btScalar(0.5) * (m_hull_local_aabb_max - m_hull_local_aabb_min);
            local_half_extents += btVector3(m_margin, m_margin, m_margin);
            const btVector3 local_center = btScalar(0.5) * (m_hull_local_aabb_max + m_hull_local_aabb_min);
            const btMatrix3x3 abs_b = t.getBasis().absolute();
            const btVector3 center = t(local_center);
            const btVector3 extent = local_half_extents.dot3(abs_b[0], abs_b[1], abs_b[2]);
            aabb_min = center - extent;
            aabb_max = center + extent;
            return;
        }
        if (m_kind == BT_SHAPE_BOX) {
            const btVector3 half_extents_with_margin =
                m_implicitShapeDimensions + btVector3(m_margin, m_margin, m_margin);
            const btMatrix3x3 abs_b = t.getBasis().absolute();
            const btVector3 center = t.getOrigin();
            const btVector3 extent =
                half_extents_with_margin.dot3(abs_b[0], abs_b[1], abs_b[2]);
            aabb_min = center - extent;
            aabb_max = center + extent;
            return;
        }

        if (m_kind == BT_SHAPE_TRIANGLE) {
            for (int i = 0; i < 3; ++i) {
                btVector3 axis(0.0f, 0.0f, 0.0f);
                axis[i] = 1.0f;
                btVector3 support = localGetSupportVertexNonVirtual(axis * t.getBasis());
                btVector3 world_point = t(support);
                aabb_max[i] = world_point[i] + m_margin;

                axis[i] = -1.0f;
                support = localGetSupportVertexNonVirtual(axis * t.getBasis());
                world_point = t(support);
                aabb_min[i] = world_point[i] - m_margin;
            }
            return;
        }

        const btScalar radius = m_margin;
        const btVector3 center = t.getOrigin();
        const btVector3 extent(radius, radius, radius);
        aabb_min = center - extent;
        aabb_max = center + extent;
    }
};

struct btSphereShape : public btConvexShape {
    ROCKETGPU_HD explicit btSphereShape(btScalar radius) : btConvexShape() {
        m_kind = BT_SHAPE_SPHERE;
        m_margin = radius;
    }
};

struct btConvexHullShape : public btConvexShape {
    ROCKETGPU_HD btConvexHullShape(const btVector3* points, int count) : btConvexShape() {
        m_kind = BT_SHAPE_CONVEX_HULL;
        m_margin = btScalar(0.04);
        m_hull_points = points;
        m_hull_point_count = count;
        recalcHullLocalAabb();
    }

    ROCKETGPU_HD btVector3 calculateLocalInertia(btScalar mass) const {
        const btScalar margin = m_margin;

        const btTransform ident(btVector3(0.0f, 0.0f, 0.0f));
        btVector3 aabb_min, aabb_max;
        getAabb(ident, aabb_min, aabb_max);
        const btVector3 half_extents = (aabb_max - aabb_min) * btScalar(0.5);
        const btScalar lx = btScalar(2.) * (half_extents.x() + margin);
        const btScalar ly = btScalar(2.) * (half_extents.y() + margin);
        const btScalar lz = btScalar(2.) * (half_extents.z() + margin);
        const btScalar x2 = lx * lx;
        const btScalar y2 = ly * ly;
        const btScalar z2 = lz * lz;
        const btScalar scaledmass = mass * btScalar(0.08333333);
        return scaledmass * (btVector3(y2 + z2, x2 + z2, x2 + y2));
    }
};

struct btGjkEpaScratch;

struct btGjkEpaDebugStop {
    int32_t requested_phase = 0;
    int32_t hit_phase = 0;
    int32_t stage = 0;
    int32_t status = 0;
    int32_t rank = 0;
    btScalar detail0 = 0.0f;
    btScalar detail1 = 0.0f;
    btVector3 detail_vec0 = btVector3(0.0f, 0.0f, 0.0f);
    btVector3 detail_vec1 = btVector3(0.0f, 0.0f, 0.0f);

    ROCKETGPU_HD bool Stop(int32_t phase, int32_t stage_value, int32_t status_value, int32_t rank_value, btScalar detail0_value, btScalar detail1_value) {
        return StopRequested(requested_phase, phase, stage_value, status_value, rank_value, detail0_value, detail1_value);
    }

    ROCKETGPU_HD bool StopRequested(int32_t requested, int32_t phase, int32_t stage_value, int32_t status_value, int32_t rank_value, btScalar detail0_value, btScalar detail1_value) {
        if (requested != phase) {
            return false;
        }
        requested_phase = requested;
        hit_phase = phase;
        stage = stage_value;
        status = status_value;
        rank = rank_value;
        detail0 = detail0_value;
        detail1 = detail1_value;
        detail_vec0 = btVector3(0.0f, 0.0f, 0.0f);
        detail_vec1 = btVector3(0.0f, 0.0f, 0.0f);
        return true;
    }

    ROCKETGPU_HD bool StopVec(
        int32_t requested,
        int32_t phase,
        int32_t stage_value,
        int32_t status_value,
        int32_t rank_value,
        btScalar detail0_value,
        btScalar detail1_value,
        const btVector3& detail_vec0_value,
        const btVector3& detail_vec1_value) {
        if (!StopRequested(requested, phase, stage_value, status_value, rank_value, detail0_value, detail1_value)) {
            return false;
        }
        detail_vec0 = detail_vec0_value;
        detail_vec1 = detail_vec1_value;
        return true;
    }
};

struct btGjkEpaSolver2
{
    struct sResults
    {
        enum eStatus
        {
            Separated,
            Penetrating,
            GJK_Failed,
            EPA_Failed
        } status;
        btVector3 witnesses[2];
        btVector3 normal;
        btScalar distance;
    };

    static int StackSizeRequirement();

    static bool Distance(const btConvexShape* shape0, const btTransform& wtrs0,
                         const btConvexShape* shape1, const btTransform& wtrs1,
                         const btVector3& guess,
                         sResults& results);

    static bool DistanceWithScratch(const btConvexShape* shape0, const btTransform& wtrs0,
                                    const btConvexShape* shape1, const btTransform& wtrs1,
                                    const btVector3& guess,
                                    sResults& results,
                                    btGjkEpaScratch& scratch);

    static bool Penetration(const btConvexShape* shape0, const btTransform& wtrs0,
                            const btConvexShape* shape1, const btTransform& wtrs1,
                            const btVector3& guess,
                            sResults& results,
                            bool usemargins = true);

    static bool PenetrationWithScratch(const btConvexShape* shape0, const btTransform& wtrs0,
                                       const btConvexShape* shape1, const btTransform& wtrs1,
                                       const btVector3& guess,
                                       sResults& results,
                                       btGjkEpaScratch& scratch,
                                       bool usemargins = true,
                                       btGjkEpaDebugStop* debug_stop = nullptr);

    static btScalar SignedDistance(const btVector3& position,
                                   btScalar margin,
                                   const btConvexShape* shape,
                                   const btTransform& wtrs,
                                   sResults& results);

    static bool SignedDistance(const btConvexShape* shape0, const btTransform& wtrs0,
                               const btConvexShape* shape1, const btTransform& wtrs1,
                               const btVector3& guess,
                               sResults& results);
};

namespace gjkepa2_impl
{
#define GJK_MAX_ITERATIONS 128
#define GJK_ACCURACY ((btScalar)0.0001)
#define GJK_MIN_DISTANCE ((btScalar)0.0001)
#define GJK_DUPLICATED_EPS ((btScalar)0.0001)
#define GJK_SIMPLEX2_EPS ((btScalar)0.0)
#define GJK_SIMPLEX3_EPS ((btScalar)0.0)
#define GJK_SIMPLEX4_EPS ((btScalar)0.0)

#define EPA_MAX_VERTICES 128
#define EPA_MAX_ITERATIONS 255
#define EPA_ACCURACY ((btScalar)0.0001)
#define EPA_PLANE_EPS ((btScalar)0.00001)
#define EPA_INSIDE_EPS ((btScalar)0.01)
#define EPA_FALLBACK (10 * EPA_ACCURACY)
#define EPA_MAX_FACES (EPA_MAX_VERTICES * 2)

typedef unsigned int U;
typedef unsigned char U1;

struct MinkowskiDiff
{
	const btConvexShape* m_shapes[2];
	btMatrix3x3 m_toshape1;
	btTransform m_toshape0;
    bool m_enableMargin;

    ROCKETGPU_HD void EnableMargin(bool enable)
    {
        m_enableMargin = enable;
    }

    ROCKETGPU_HD btVector3 Support0(const btVector3& d) const
    {
        if (m_enableMargin)
            return m_shapes[0]->localGetSupportVertexNonVirtual(d);
        else
            return m_shapes[0]->localGetSupportVertexWithoutMarginNonVirtual(d);
    }

    ROCKETGPU_HD btVector3 Support1(const btVector3& d) const
    {
        if (m_enableMargin)
            return m_toshape0 * (m_shapes[1]->localGetSupportVertexNonVirtual(m_toshape1 * d));
        else
            return m_toshape0 * (m_shapes[1]->localGetSupportVertexWithoutMarginNonVirtual(m_toshape1 * d));
    }

    ROCKETGPU_HD btVector3 Support(const btVector3& d) const
    {
        return (Support0(d) - Support1(-d));
    }

    ROCKETGPU_HD btVector3 Support(const btVector3& d, U index) const
    {
        if (index)
            return (Support1(d));
        else
            return (Support0(d));
    }
};

typedef MinkowskiDiff tShape;

inline void TraceDebugEpaWitnessReconstruct(
    const btVector3 support0_dir_norm[4],
    const btVector3 support0_without_margin[4],
    const btVector3 support0[4],
    const btVector3& w0,
    const btTransform& wtrs0,
    const btVector3& normal,
    btScalar depth) {
#if defined(ROCKETGPU_CAN_COMPILE_OFFICIAL_HOST_HELPERS)
#else
    (void)support0_dir_norm;
    (void)support0_without_margin;
    (void)support0;
    (void)w0;
    (void)wtrs0;
    (void)normal;
    (void)depth;
#endif
}

struct GJK
{
    struct sSV
    {
        btVector3 d, w;
    };
    struct sSimplex
    {
        sSV* c[4];
        btScalar p[4];
        U rank;
    };
    struct eStatus
    {
        enum _
        {
            Valid,
            Inside,
            Failed
        };
    };

    tShape m_shape;
    btVector3 m_ray;
    btScalar m_distance;
    sSimplex m_simplices[2];
    sSV m_store[4];
    sSV* m_free[4];
    U m_nfree;
    U m_current;
    sSimplex* m_simplex;
    eStatus::_ m_status;
    btVector3 m_debug_seed_query_dir;
    btVector3 m_debug_seed_support_d;
    btVector3 m_debug_seed_support_w;
    btVector3 m_debug_seed_support0;
    btVector3 m_debug_seed_support1_dir_local;
    btVector3 m_debug_seed_support1_local;
    btVector3 m_debug_seed_support1;

    ROCKETGPU_HD GJK()
    {
        Initialize();
    }

    ROCKETGPU_HD void Initialize()
    {
        m_ray = btVector3(0, 0, 0);
        m_nfree = 0;
        m_status = eStatus::Failed;
        m_current = 0;
        m_distance = 0;
        m_debug_seed_query_dir = btVector3(0, 0, 0);
        m_debug_seed_support_d = btVector3(0, 0, 0);
        m_debug_seed_support_w = btVector3(0, 0, 0);
        m_debug_seed_support0 = btVector3(0, 0, 0);
        m_debug_seed_support1_dir_local = btVector3(0, 0, 0);
        m_debug_seed_support1_local = btVector3(0, 0, 0);
        m_debug_seed_support1 = btVector3(0, 0, 0);
    }

    ROCKETGPU_HD eStatus::_ Evaluate(
        const tShape& shapearg,
        const btVector3& guess,
        const btGjkEpaDebugStop* debug_stop = nullptr)
    {
        U iterations = 0;
        btScalar sqdist = 0;
        btScalar alpha = 0;
        btVector3 lastw[4];
        U clastw = 0;
        m_free[0] = &m_store[0];
        m_free[1] = &m_store[1];
        m_free[2] = &m_store[2];
        m_free[3] = &m_store[3];
        m_nfree = 4;
        m_current = 0;
        m_status = eStatus::Valid;
        m_shape = shapearg;
        m_distance = 0;
        m_simplices[0].rank = 0;
        m_ray = guess;
        const btScalar sqrl = m_ray.length2();
        const btVector3 seed_query_dir = sqrl > 0 ? -m_ray : btVector3(1, 0, 0);
        appendvertice(m_simplices[0], seed_query_dir);
        m_simplices[0].p[0] = 1;
        m_debug_seed_query_dir = seed_query_dir;
        m_debug_seed_support_d = m_simplices[0].c[0]->d;
        m_debug_seed_support_w = m_simplices[0].c[0]->w;
#if defined(ROCKETGPU_CAN_COMPILE_OFFICIAL_HOST_HELPERS)
        constexpr bool kTraceSeedSupport = true;
#else
        constexpr bool kTraceSeedSupport = false;
#endif
        const bool capture_seed_support =
            kTraceSeedSupport ||
            (debug_stop && debug_stop->requested_phase >= 229 && debug_stop->requested_phase <= 231);
        if (capture_seed_support) {
            const btVector3 seed_support0 = m_shape.Support0(m_simplices[0].c[0]->d);
            const btVector3 seed_support1_dir_local = m_shape.m_toshape1 * (-m_simplices[0].c[0]->d);
            const btVector3 seed_support1_local = m_shape.m_enableMargin
                ? m_shape.m_shapes[1]->localGetSupportVertexNonVirtual(seed_support1_dir_local)
                : m_shape.m_shapes[1]->localGetSupportVertexWithoutMarginNonVirtual(seed_support1_dir_local);
            const btVector3 seed_support1 = m_shape.Support1(-m_simplices[0].c[0]->d);
            m_debug_seed_support0 = seed_support0;
            m_debug_seed_support1_dir_local = seed_support1_dir_local;
            m_debug_seed_support1_local = seed_support1_local;
            m_debug_seed_support1 = seed_support1;
        }
        m_ray = m_simplices[0].c[0]->w;
        sqdist = sqrl;
        lastw[0] =
            lastw[1] =
                lastw[2] =
                    lastw[3] = m_ray;
        do
        {
            const U next = 1 - m_current;
            sSimplex& cs = m_simplices[m_current];
            sSimplex& ns = m_simplices[next];
            const btScalar rl = m_ray.length();
            if (rl < GJK_MIN_DISTANCE)
            {
                m_status = eStatus::Inside;
                break;
            }
            appendvertice(cs, -m_ray);
            const btVector3& w = cs.c[cs.rank - 1]->w;
#if defined(ROCKETGPU_CAN_COMPILE_OFFICIAL_HOST_HELPERS)
            const btVector3 iter_support1_dir_local = m_shape.m_toshape1 * (-cs.c[cs.rank - 1]->d);
            const btVector3 iter_support1_local = m_shape.m_enableMargin
                ? m_shape.m_shapes[1]->localGetSupportVertexNonVirtual(iter_support1_dir_local)
                : m_shape.m_shapes[1]->localGetSupportVertexWithoutMarginNonVirtual(iter_support1_dir_local);
#endif
            bool found = false;
            for (U i = 0; i < 4; ++i)
            {
                if ((w - lastw[i]).length2() < GJK_DUPLICATED_EPS)
                {
                    found = true;
                    break;
                }
            }
            if (found)
            {
                removevertice(m_simplices[m_current]);
                break;
            }
            else
            {
                lastw[clastw = (clastw + 1) & 3] = w;
            }
            const btScalar omega = btDot(m_ray, w) / rl;
            alpha = btMax(omega, alpha);
            if (((rl - alpha) - (GJK_ACCURACY * rl)) <= 0)
            {
                removevertice(m_simplices[m_current]);
                break;
            }
            btScalar weights[4];
            U mask = 0;
            switch (cs.rank)
            {
                case 2:
                    sqdist = projectorigin(cs.c[0]->w,
                                           cs.c[1]->w,
                                           weights, mask);
                    break;
                case 3:
                    sqdist = projectorigin(cs.c[0]->w,
                                           cs.c[1]->w,
                                           cs.c[2]->w,
                                           weights, mask);
                    break;
                case 4:
                    sqdist = projectorigin(cs.c[0]->w,
                                           cs.c[1]->w,
                                           cs.c[2]->w,
                                           cs.c[3]->w,
                                           weights, mask);
                    break;
            }
            if (sqdist >= 0)
            {
                ns.rank = 0;
                m_ray = btVector3(0, 0, 0);
                m_current = next;
                for (U i = 0, ni = cs.rank; i < ni; ++i)
                {
                    if (mask & (1 << i))
                    {
                        ns.c[ns.rank] = cs.c[i];
                        ns.p[ns.rank++] = weights[i];
                        m_ray += cs.c[i]->w * weights[i];
                    }
                    else
                    {
                        m_free[m_nfree++] = cs.c[i];
                    }
                }
                if (mask == 15) {
                    m_status = eStatus::Inside;
                }
            }
            else
            {
                removevertice(m_simplices[m_current]);
                break;
            }
            if ((++iterations) >= GJK_MAX_ITERATIONS) {
                m_status = eStatus::Failed;
            }
        } while (m_status == eStatus::Valid);
        m_simplex = &m_simplices[m_current];
        switch (m_status)
        {
            case eStatus::Valid:
                m_distance = m_ray.length();
                break;
            case eStatus::Inside:
                m_distance = 0;
                break;
            default:
            {
            }
        }
        return (m_status);
    }

    ROCKETGPU_HD bool EncloseOriginRank4()
    {
        if (btFabs(det(m_simplex->c[0]->w - m_simplex->c[3]->w,
                       m_simplex->c[1]->w - m_simplex->c[3]->w,
                       m_simplex->c[2]->w - m_simplex->c[3]->w)) > 0)
            return true;
        return false;
    }
    ROCKETGPU_HD bool EncloseOriginRank3()
    {
        const btVector3 n = btCross(m_simplex->c[1]->w - m_simplex->c[0]->w,
                                    m_simplex->c[2]->w - m_simplex->c[0]->w);
        if (n.length2() > 0)
        {
            appendvertice(*m_simplex, n);
            if (EncloseOriginRank4()) return true;
            removevertice(*m_simplex);
            appendvertice(*m_simplex, -n);
            if (EncloseOriginRank4()) return true;
            removevertice(*m_simplex);
        }
        return false;
    }
    ROCKETGPU_HD bool EncloseOriginRank2()
    {
        const btVector3 d = m_simplex->c[1]->w - m_simplex->c[0]->w;
        for (U i = 0; i < 3; ++i)
        {
            btVector3 axis = btVector3(0, 0, 0);
            axis[i] = 1;
            const btVector3 p = btCross(d, axis);
            if (p.length2() > 0)
            {
                appendvertice(*m_simplex, p);
                if (EncloseOriginRank3()) return true;
                removevertice(*m_simplex);
                appendvertice(*m_simplex, -p);
                if (EncloseOriginRank3()) return true;
                removevertice(*m_simplex);
            }
        }
        return false;
    }
    ROCKETGPU_HD bool EncloseOriginRank1()
    {
        for (U i = 0; i < 3; ++i)
        {
            btVector3 axis = btVector3(0, 0, 0);
            axis[i] = 1;
            appendvertice(*m_simplex, axis);
            if (EncloseOriginRank2()) return true;
            removevertice(*m_simplex);
            appendvertice(*m_simplex, -axis);
            if (EncloseOriginRank2()) return true;
            removevertice(*m_simplex);
        }
        return false;
    }
    ROCKETGPU_HD bool EncloseOrigin()
    {
        switch (m_simplex->rank)
        {
            case 1:
                return EncloseOriginRank1();
            case 2:
                return EncloseOriginRank2();
            case 3:
                return EncloseOriginRank3();
            case 4:
                return EncloseOriginRank4();
        }
        return false;
    }

    ROCKETGPU_HD void getsupport(const btVector3& d, sSV& sv) const
    {
        sv.d = d / d.length();
        sv.w = m_shape.Support(sv.d);
    }

    ROCKETGPU_HD void removevertice(sSimplex& simplex)
    {
        m_free[m_nfree++] = simplex.c[--simplex.rank];
    }

    ROCKETGPU_HD void appendvertice(sSimplex& simplex, const btVector3& v)
    {
        simplex.p[simplex.rank] = 0;
        simplex.c[simplex.rank] = m_free[--m_nfree];
        getsupport(v, *simplex.c[simplex.rank++]);
    }

    ROCKETGPU_HD static btScalar det(const btVector3& a, const btVector3& b, const btVector3& c)
    {
        return (a.y() * b.z() * c.x() + a.z() * b.x() * c.y() -
                a.x() * b.z() * c.y() - a.y() * b.x() * c.z() +
                a.x() * b.y() * c.z() - a.z() * b.y() * c.x());
    }

    ROCKETGPU_HD static btScalar projectorigin(const btVector3& a,
                                           const btVector3& b,
                                           btScalar* w, U& m)
    {
        const btVector3 d = b - a;
        const btScalar l = d.length2();
        if (l > GJK_SIMPLEX2_EPS)
        {
            const btScalar t(l > 0 ? -btDot(a, d) / l : 0);
            if (t >= 1)
            {
                w[0] = 0;
                w[1] = 1;
                m = 2;
                return (b.length2());
            }
            else if (t <= 0)
            {
                w[0] = 1;
                w[1] = 0;
                m = 1;
                return (a.length2());
            }
            else
            {
                w[0] = 1 - (w[1] = t);
                m = 3;
                return ((a + d * t).length2());
            }
        }
        return (-1);
    }

    ROCKETGPU_HD static btScalar projectorigin(const btVector3& a,
                                           const btVector3& b,
                                           const btVector3& c,
                                           btScalar* w, U& m)
    {
        static const U imd3[] = {1, 2, 0};
        const btVector3* vt[] = {&a, &b, &c};
        const btVector3 dl[] = {a - b, b - c, c - a};
        const btVector3 n = btCross(dl[0], dl[1]);
        const btScalar l = n.length2();
        if (l > GJK_SIMPLEX3_EPS)
        {
            btScalar mindist = -1;
            btScalar subw[2] = {0.f, 0.f};
            U subm(0);
            for (U i = 0; i < 3; ++i)
            {
                if (btDot(*vt[i], btCross(dl[i], n)) > 0)
                {
                    const U j = imd3[i];
                    const btScalar subd(projectorigin(*vt[i], *vt[j], subw, subm));
                    if ((mindist < 0) || (subd < mindist))
                    {
                        mindist = subd;
                        m = static_cast<U>(((subm & 1) ? 1 << i : 0) + ((subm & 2) ? 1 << j : 0));
                        w[i] = subw[0];
                        w[j] = subw[1];
                        w[imd3[j]] = 0;
                    }
                }
            }
            if (mindist < 0)
            {
                const btScalar d = btDot(a, n);
                const btScalar s = btSqrt(l);
                const btVector3 p = n * (d / l);
                mindist = p.length2();
                m = 7;
                w[0] = (btCross(dl[1], b - p)).length() / s;
                w[1] = (btCross(dl[2], c - p)).length() / s;
                w[2] = 1 - (w[0] + w[1]);
            }
            return (mindist);
        }
        return (-1);
    }

    ROCKETGPU_HD static btScalar projectorigin(const btVector3& a,
                                           const btVector3& b,
                                           const btVector3& c,
                                           const btVector3& d,
                                           btScalar* w, U& m)
    {
        static const U imd3[] = {1, 2, 0};
        const btVector3* vt[] = {&a, &b, &c, &d};
        const btVector3 dl[] = {a - d, b - d, c - d};
        const btScalar vl = det(dl[0], dl[1], dl[2]);
        const bool ng = (vl * btDot(a, btCross(b - c, a - b))) <= 0;
        if (ng && (btFabs(vl) > GJK_SIMPLEX4_EPS))
        {
            btScalar mindist = -1;
            btScalar subw[3] = {0.f, 0.f, 0.f};
            U subm(0);
            for (U i = 0; i < 3; ++i)
            {
                const U j = imd3[i];
                const btScalar s = vl * btDot(d, btCross(dl[i], dl[j]));
                if (s > 0)
                {
                    const btScalar subd = projectorigin(*vt[i], *vt[j], d, subw, subm);
                    if ((mindist < 0) || (subd < mindist))
                    {
                        mindist = subd;
                        m = static_cast<U>((subm & 1 ? 1 << i : 0) +
                                           (subm & 2 ? 1 << j : 0) +
                                           (subm & 4 ? 8 : 0));
                        w[i] = subw[0];
                        w[j] = subw[1];
                        w[imd3[j]] = 0;
                        w[3] = subw[2];
                    }
                }
            }
            if (mindist < 0)
            {
                mindist = 0;
                m = 15;
                w[0] = det(c, b, d) / vl;
                w[1] = det(a, c, d) / vl;
                w[2] = det(b, a, d) / vl;
                w[3] = 1 - (w[0] + w[1] + w[2]);
            }
            return (mindist);
        }
        return (-1);
    }
};

struct EPA
{
    typedef GJK::sSV sSV;
    struct sFace
    {
        btVector3 n;
        btScalar d;
        sSV* c[3];
        sFace* f[3];
        sFace* l[2];
        U1 e[3];
        U1 pass;
    };
    struct sList
    {
        sFace* root;
        U count;
        ROCKETGPU_HD sList() : root(0), count(0) {}
    };
    struct sHorizon
    {
        sFace* cf;
        sFace* ff;
        U nf;
        ROCKETGPU_HD sHorizon() : cf(0), ff(0), nf(0) {}
    };
    struct eStatus
    {
        enum _
        {
            Valid,
            Touching,
            Degenerated,
            NonConvex,
            InvalidHull,
            OutOfFaces,
            OutOfVertices,
            AccuraryReached,
            FallBack,
            Failed
        };
    };

    eStatus::_ m_status;
    GJK::sSimplex m_result;
    btVector3 m_normal;
    btScalar m_depth;
    sSV m_sv_store[EPA_MAX_VERTICES];
    sFace m_fc_store[EPA_MAX_FACES];
    U m_nextsv;
    sList m_hull;
    sList m_stock;
    int32_t m_debug_first_iteration;
    int32_t m_debug_last_iteration;
    int32_t m_debug_first_support1_max_axis;
    int32_t m_debug_last_support1_max_axis;
    btScalar m_debug_first_best_d;
    btScalar m_debug_first_wdist;
    btScalar m_debug_last_best_d;
    btScalar m_debug_last_wdist;
    btVector3 m_debug_first_best_n;
    btVector3 m_debug_first_best_w0;
    btVector3 m_debug_first_best_w1;
    btVector3 m_debug_first_best_w2;
    btVector3 m_debug_first_support_vertex_d;
    btVector3 m_debug_first_support0;
    btVector3 m_debug_first_support1_dir_local;
    btVector3 m_debug_first_support1_toshape1_row0;
    btVector3 m_debug_first_support1_toshape1_row1;
    btVector3 m_debug_first_support1_toshape1_row2;
    btVector3 m_debug_first_support1_toshape0_row0;
    btVector3 m_debug_first_support1_toshape0_row1;
    btVector3 m_debug_first_support1_toshape0_row2;
    btVector3 m_debug_first_support1_toshape0_origin;
    btVector3 m_debug_first_support1_dots;
    btVector3 m_debug_first_support1_local;
    btVector3 m_debug_first_support1;
    btVector3 m_debug_first_support_w;
    btVector3 m_debug_last_best_n;
    btVector3 m_debug_last_best_w0;
    btVector3 m_debug_last_best_w1;
    btVector3 m_debug_last_best_w2;
    btVector3 m_debug_last_support_vertex_d;
    btVector3 m_debug_last_support0;
    btVector3 m_debug_last_support1_dir_local;
    btVector3 m_debug_last_support1_toshape1_row0;
    btVector3 m_debug_last_support1_toshape1_row1;
    btVector3 m_debug_last_support1_toshape1_row2;
    btVector3 m_debug_last_support1_toshape0_row0;
    btVector3 m_debug_last_support1_toshape0_row1;
    btVector3 m_debug_last_support1_toshape0_row2;
    btVector3 m_debug_last_support1_toshape0_origin;
    btVector3 m_debug_last_support1_dots;
    btVector3 m_debug_last_support1_local;
    btVector3 m_debug_last_support1;
    btVector3 m_debug_last_support_w;

    ROCKETGPU_HD EPA()
    {
        Initialize();
    }

    ROCKETGPU_HD static void bind(sFace* fa, U ea, sFace* fb, U eb)
    {
        fa->e[ea] = (U1)eb;
        fa->f[ea] = fb;
        fb->e[eb] = (U1)ea;
        fb->f[eb] = fa;
    }

    ROCKETGPU_HD static void append(sList& list, sFace* face)
    {
        face->l[0] = 0;
        face->l[1] = list.root;
        if (list.root) list.root->l[0] = face;
        list.root = face;
        ++list.count;
    }

    ROCKETGPU_HD static void remove(sList& list, sFace* face)
    {
        if (face->l[1]) face->l[1]->l[0] = face->l[0];
        if (face->l[0]) face->l[0]->l[1] = face->l[1];
        if (face == list.root) list.root = face->l[1];
        --list.count;
    }

    ROCKETGPU_HD void Initialize()
    {
        m_status = eStatus::Failed;
        m_normal = btVector3(0, 0, 0);
        m_depth = 0;
        m_nextsv = 0;
        m_hull = sList();
        m_stock = sList();
        m_debug_first_iteration = -1;
        m_debug_last_iteration = -1;
        m_debug_first_support1_max_axis = -1;
        m_debug_last_support1_max_axis = -1;
        m_debug_first_best_d = 0.0f;
        m_debug_first_wdist = 0.0f;
        m_debug_last_best_d = 0.0f;
        m_debug_last_wdist = 0.0f;
        m_debug_first_best_n = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_first_best_w0 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_first_best_w1 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_first_best_w2 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_first_support_vertex_d = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_first_support0 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_first_support1_dir_local = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_first_support1_toshape1_row0 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_first_support1_toshape1_row1 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_first_support1_toshape1_row2 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_first_support1_toshape0_row0 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_first_support1_toshape0_row1 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_first_support1_toshape0_row2 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_first_support1_toshape0_origin = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_first_support1_dots = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_first_support1_local = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_first_support1 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_first_support_w = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_last_best_n = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_last_best_w0 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_last_best_w1 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_last_best_w2 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_last_support_vertex_d = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_last_support0 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_last_support1_dir_local = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_last_support1_toshape1_row0 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_last_support1_toshape1_row1 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_last_support1_toshape1_row2 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_last_support1_toshape0_row0 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_last_support1_toshape0_row1 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_last_support1_toshape0_row2 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_last_support1_toshape0_origin = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_last_support1_dots = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_last_support1_local = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_last_support1 = btVector3(0.0f, 0.0f, 0.0f);
        m_debug_last_support_w = btVector3(0.0f, 0.0f, 0.0f);
        for (U i = 0; i < EPA_MAX_FACES; ++i)
        {
            append(m_stock, &m_fc_store[EPA_MAX_FACES - i - 1]);
        }
    }

    ROCKETGPU_HD void CaptureIterationDebug(
        int32_t iteration,
        const sFace* best,
        const sSV* w,
        const btVector3& support0,
        const btVector3& support1_dir_local,
        const btVector3& support1_toshape1_row0,
        const btVector3& support1_toshape1_row1,
        const btVector3& support1_toshape1_row2,
        const btVector3& support1_toshape0_row0,
        const btVector3& support1_toshape0_row1,
        const btVector3& support1_toshape0_row2,
        const btVector3& support1_toshape0_origin,
        const btVector3& support1_dots,
        int32_t support1_max_axis,
        const btVector3& support1_local,
        const btVector3& support1,
        btScalar wdist)
    {
        const bool capture_first = m_debug_first_iteration < 0;
        if (capture_first) {
            m_debug_first_iteration = iteration;
            m_debug_first_support1_max_axis = support1_max_axis;
            m_debug_first_best_d = best->d;
            m_debug_first_wdist = wdist;
            m_debug_first_best_n = best->n;
            m_debug_first_best_w0 = best->c[0]->w;
            m_debug_first_best_w1 = best->c[1]->w;
            m_debug_first_best_w2 = best->c[2]->w;
            m_debug_first_support_vertex_d = w->d;
            m_debug_first_support0 = support0;
            m_debug_first_support1_dir_local = support1_dir_local;
            m_debug_first_support1_toshape1_row0 = support1_toshape1_row0;
            m_debug_first_support1_toshape1_row1 = support1_toshape1_row1;
            m_debug_first_support1_toshape1_row2 = support1_toshape1_row2;
            m_debug_first_support1_toshape0_row0 = support1_toshape0_row0;
            m_debug_first_support1_toshape0_row1 = support1_toshape0_row1;
            m_debug_first_support1_toshape0_row2 = support1_toshape0_row2;
            m_debug_first_support1_toshape0_origin = support1_toshape0_origin;
            m_debug_first_support1_dots = support1_dots;
            m_debug_first_support1_local = support1_local;
            m_debug_first_support1 = support1;
            m_debug_first_support_w = w->w;
        }
        m_debug_last_iteration = iteration;
        m_debug_last_support1_max_axis = support1_max_axis;
        m_debug_last_best_d = best->d;
        m_debug_last_wdist = wdist;
        m_debug_last_best_n = best->n;
        m_debug_last_best_w0 = best->c[0]->w;
        m_debug_last_best_w1 = best->c[1]->w;
        m_debug_last_best_w2 = best->c[2]->w;
        m_debug_last_support_vertex_d = w->d;
        m_debug_last_support0 = support0;
        m_debug_last_support1_dir_local = support1_dir_local;
        m_debug_last_support1_toshape1_row0 = support1_toshape1_row0;
        m_debug_last_support1_toshape1_row1 = support1_toshape1_row1;
        m_debug_last_support1_toshape1_row2 = support1_toshape1_row2;
        m_debug_last_support1_toshape0_row0 = support1_toshape0_row0;
        m_debug_last_support1_toshape0_row1 = support1_toshape0_row1;
        m_debug_last_support1_toshape0_row2 = support1_toshape0_row2;
        m_debug_last_support1_toshape0_origin = support1_toshape0_origin;
        m_debug_last_support1_dots = support1_dots;
        m_debug_last_support1_local = support1_local;
        m_debug_last_support1 = support1;
        m_debug_last_support_w = w->w;
    }

    ROCKETGPU_HD eStatus::_ Evaluate(GJK& gjk, const btVector3& guess)
    {
        GJK::sSimplex& simplex = *gjk.m_simplex;
        if ((simplex.rank > 1) && gjk.EncloseOrigin())
        {
            while (m_hull.root)
            {
                sFace* f = m_hull.root;
                remove(m_hull, f);
                append(m_stock, f);
            }
            m_status = eStatus::Valid;
            m_nextsv = 0;
            const btVector3 simplex_w0_pre = simplex.c[0]->w;
            const btVector3 simplex_w1_pre = simplex.c[1]->w;
            const btVector3 simplex_w2_pre = simplex.c[2]->w;
            const btVector3 simplex_w3_pre = simplex.c[3]->w;
            const btScalar orientation_det = gjk.det(
                simplex_w0_pre - simplex_w3_pre,
                simplex_w1_pre - simplex_w3_pre,
                simplex_w2_pre - simplex_w3_pre);
            const bool swapped = orientation_det < 0;
            if (swapped)
            {
                btSwap(simplex.c[0], simplex.c[1]);
                btSwap(simplex.p[0], simplex.p[1]);
            }
            sFace* tetra[] = {newface(simplex.c[0], simplex.c[1], simplex.c[2], true),
                              newface(simplex.c[1], simplex.c[0], simplex.c[3], true),
                              newface(simplex.c[2], simplex.c[1], simplex.c[3], true),
                              newface(simplex.c[0], simplex.c[2], simplex.c[3], true)};
            if (m_hull.count == 4)
            {
                sFace* best = findbest();
                sFace outer = *best;
                U pass = 0;
                U iterations = 0;
                bind(tetra[0], 0, tetra[1], 0);
                bind(tetra[0], 1, tetra[2], 0);
                bind(tetra[0], 2, tetra[3], 0);
                bind(tetra[1], 1, tetra[3], 2);
                bind(tetra[1], 2, tetra[2], 1);
                bind(tetra[2], 2, tetra[3], 1);
                m_status = eStatus::Valid;
                for (; iterations < EPA_MAX_ITERATIONS; ++iterations)
                {
                    if (m_nextsv < EPA_MAX_VERTICES)
                    {
                        sHorizon horizon;
                        sSV* w = &m_sv_store[m_nextsv++];
                        bool valid = true;
                        best->pass = (U1)(++pass);
                        gjk.getsupport(best->n, *w);
                        const btVector3 support0 = gjk.m_shape.Support0(w->d);
                        const btVector3 support1_dir_local = gjk.m_shape.m_toshape1 * (-w->d);
                        const btVector3 support1_toshape1_row0 = gjk.m_shape.m_toshape1[0];
                        const btVector3 support1_toshape1_row1 = gjk.m_shape.m_toshape1[1];
                        const btVector3 support1_toshape1_row2 = gjk.m_shape.m_toshape1[2];
                        const btVector3 support1_toshape0_row0 = gjk.m_shape.m_toshape0.getBasis()[0];
                        const btVector3 support1_toshape0_row1 = gjk.m_shape.m_toshape0.getBasis()[1];
                        const btVector3 support1_toshape0_row2 = gjk.m_shape.m_toshape0.getBasis()[2];
                        const btVector3 support1_toshape0_origin = gjk.m_shape.m_toshape0.getOrigin();
                        const btVector3 support1_vertex0 = gjk.m_shape.m_shapes[1]->m_vertices1[0];
                        const btVector3 support1_vertex1 = gjk.m_shape.m_shapes[1]->m_vertices1[1];
                        const btVector3 support1_vertex2 = gjk.m_shape.m_shapes[1]->m_vertices1[2];
                        const btVector3 support1_dots =
                            support1_dir_local.dot3(support1_vertex0, support1_vertex1, support1_vertex2);
                        const int support1_max_axis = support1_dots.maxAxis();
                        btVector3 support1_dir_norm = support1_dir_local;
                        if (support1_dir_norm.length2() < (SIMD_EPSILON * SIMD_EPSILON))
                        {
                            support1_dir_norm.setValue(-1.0f, -1.0f, -1.0f);
                        }
                        support1_dir_norm.normalize();
                        const btVector3 support1_norm_dots =
                            support1_dir_norm.dot3(support1_vertex0, support1_vertex1, support1_vertex2);
                        const btVector3 support1_local = gjk.m_shape.m_enableMargin
                            ? gjk.m_shape.m_shapes[1]->localGetSupportVertexNonVirtual(support1_dir_local)
                            : gjk.m_shape.m_shapes[1]->localGetSupportVertexWithoutMarginNonVirtual(support1_dir_local);
                        const btVector3 support1 = gjk.m_shape.Support1(-w->d);
                        const btScalar wdist = btDot(best->n, w->w) - best->d;
                        CaptureIterationDebug(
                            static_cast<int32_t>(iterations),
                            best,
                            w,
                            support0,
                            support1_dir_local,
                            support1_toshape1_row0,
                            support1_toshape1_row1,
                            support1_toshape1_row2,
                            support1_toshape0_row0,
                            support1_toshape0_row1,
                            support1_toshape0_row2,
                            support1_toshape0_origin,
                            support1_dots,
                            support1_max_axis,
                            support1_local,
                            support1,
                            wdist);
                        if (wdist > EPA_ACCURACY)
                        {
                            for (U j = 0; (j < 3) && valid; ++j)
                            {
                                valid &= expand(pass, w,
                                                best->f[j], best->e[j],
                                                horizon);
                            }
                            if (valid && (horizon.nf >= 3))
                            {
                                bind(horizon.cf, 1, horizon.ff, 2);
                                remove(m_hull, best);
                                append(m_stock, best);
                                best = findbest();
                                outer = *best;
                            }
                            else
                            {
                                m_status = eStatus::InvalidHull;
                                break;
                            }
                        }
                        else
                        {
                            m_status = eStatus::AccuraryReached;
                            break;
                        }
                    }
                    else
                    {
                        m_status = eStatus::OutOfVertices;
                        break;
                    }
                }
                const btVector3 projection = outer.n * outer.d;
                m_normal = outer.n;
                m_depth = outer.d;
                m_result.rank = 3;
                m_result.c[0] = outer.c[0];
                m_result.c[1] = outer.c[1];
                m_result.c[2] = outer.c[2];
                m_result.p[0] = btCross(outer.c[1]->w - projection,
                                        outer.c[2]->w - projection)
                                    .length();
                m_result.p[1] = btCross(outer.c[2]->w - projection,
                                        outer.c[0]->w - projection)
                                    .length();
                m_result.p[2] = btCross(outer.c[0]->w - projection,
                                        outer.c[1]->w - projection)
                                    .length();
                const btScalar sum = m_result.p[0] + m_result.p[1] + m_result.p[2];
                m_result.p[0] /= sum;
                m_result.p[1] /= sum;
                m_result.p[2] /= sum;
                return (m_status);
            }
        }
        m_status = eStatus::FallBack;
        m_normal = -guess;
        const btScalar nl = m_normal.length();
        if (nl > 0)
            m_normal = m_normal / nl;
        else
            m_normal = btVector3(1, 0, 0);
        m_depth = 0;
        m_result.rank = 1;
        m_result.c[0] = simplex.c[0];
        m_result.p[0] = 1;
        return (m_status);
    }

    ROCKETGPU_HD bool getedgedist(sFace* face, sSV* a, sSV* b, btScalar& dist)
    {
        const btVector3 ba = b->w - a->w;
        const btVector3 n_ab = btCross(ba, face->n);
        const btScalar a_dot_nab = btDot(a->w, n_ab);

        if (a_dot_nab < 0)
        {
            const btScalar ba_l2 = ba.length2();
            const btScalar a_dot_ba = btDot(a->w, ba);
            const btScalar b_dot_ba = btDot(b->w, ba);

            if (a_dot_ba > 0)
            {
                dist = a->w.length();
            }
            else if (b_dot_ba < 0)
            {
                dist = b->w.length();
            }
            else
            {
                const btScalar a_dot_b = btDot(a->w, b->w);
                dist = btSqrt(btMax((a->w.length2() * b->w.length2() - a_dot_b * a_dot_b) / ba_l2, (btScalar)0));
            }

            return true;
        }

        return false;
    }

    ROCKETGPU_HD sFace* newface(sSV* a, sSV* b, sSV* c, bool forced)
    {
        if (m_stock.root)
        {
            sFace* face = m_stock.root;
            remove(m_stock, face);
            append(m_hull, face);
            face->pass = 0;
            face->c[0] = a;
            face->c[1] = b;
            face->c[2] = c;
            face->n = btCross(b->w - a->w, c->w - a->w);
            const btScalar l = face->n.length();
            const bool v = l > EPA_ACCURACY;

            if (v)
            {
                int edge_index = -1;
                bool edge_hit = getedgedist(face, a, b, face->d);
                if (edge_hit)
                {
                    edge_index = 0;
                }
                if (!edge_hit)
                {
                    edge_hit = getedgedist(face, b, c, face->d);
                    if (edge_hit) edge_index = 1;
                }
                if (!edge_hit)
                {
                    edge_hit = getedgedist(face, c, a, face->d);
                    if (edge_hit) edge_index = 2;
                }
                if (!edge_hit)
                {
                    face->d = btDot(a->w, face->n) / l;
                }

                face->n /= l;
                const bool accepted = forced || (face->d >= -EPA_PLANE_EPS);
                if (accepted)
                {
                    return face;
                }
                else
                    m_status = eStatus::NonConvex;
            }
            else
            {
                m_status = eStatus::Degenerated;
            }

            remove(m_hull, face);
            append(m_stock, face);
            return 0;
        }
        m_status = m_stock.root ? eStatus::OutOfVertices : eStatus::OutOfFaces;
        return 0;
    }

    ROCKETGPU_HD sFace* findbest()
    {
        sFace* minf = m_hull.root;
        btScalar mind = minf->d * minf->d;
        for (sFace* f = minf->l[1]; f; f = f->l[1])
        {
            const btScalar sqd = f->d * f->d;
            if (sqd < mind)
            {
                minf = f;
                mind = sqd;
            }
        }
        return (minf);
    }

    struct sExpandFrame
    {
        U1 f;
        U1 e;
        U1 stage;
    };
    ROCKETGPU_HD bool expand(U pass, sSV* w, sFace* f, U e, sHorizon& horizon)
    {
        static const U i1m3[] = {1, 2, 0};
        static const U i2m3[] = {2, 0, 1};
        sExpandFrame frames[EPA_MAX_FACES];
        U sp = 0;
        frames[sp].f = (U1)(f - m_fc_store);
        frames[sp].e = (U1)e;
        frames[sp].stage = 0;
        ++sp;
        bool ret = false;
        while (sp > 0)
        {
            const U top = sp - 1;
            sFace* cf = &m_fc_store[frames[top].f];
            const U ce = frames[top].e;
            if (frames[top].stage == 0)
            {
                if (cf->pass != pass)
                {
                    const U e1 = i1m3[ce];
                    if ((btDot(cf->n, w->w) - cf->d) < -EPA_PLANE_EPS)
                    {
                        sFace* nf = newface(cf->c[e1], cf->c[ce], w, false);
                        if (nf)
                        {
                            bind(nf, 0, cf, ce);
                            if (horizon.cf)
                                bind(horizon.cf, 1, nf, 2);
                            else
                                horizon.ff = nf;
                            horizon.cf = nf;
                            ++horizon.nf;
                            ret = true;
                        }
                        else
                        {
                            ret = false;
                        }
                        --sp;
                        continue;
                    }
                    cf->pass = (U1)pass;
                    frames[top].stage = 1;
                    if (sp < EPA_MAX_FACES)
                    {
                        frames[sp].f = (U1)(cf->f[e1] - m_fc_store);
                        frames[sp].e = cf->e[e1];
                        frames[sp].stage = 0;
                        ++sp;
                        continue;
                    }
                    ret = false;
                    --sp;
                    continue;
                }
                ret = false;
                --sp;
                continue;
            }
            if (frames[top].stage == 1)
            {
                if (!ret)
                {
                    --sp;
                    continue;
                }
                const U e2 = i2m3[ce];
                frames[top].stage = 2;
                if (sp < EPA_MAX_FACES)
                {
                    frames[sp].f = (U1)(cf->f[e2] - m_fc_store);
                    frames[sp].e = cf->e[e2];
                    frames[sp].stage = 0;
                    ++sp;
                    continue;
                }
                ret = false;
                --sp;
                continue;
            }

            if (ret)
            {
                remove(m_hull, cf);
                append(m_stock, cf);
            }
            --sp;
        }
        return (ret);
    }
};

ROCKETGPU_HD void Initialize(const btConvexShape* shape0, const btTransform& wtrs0,
                         const btConvexShape* shape1, const btTransform& wtrs1,
                         btGjkEpaSolver2::sResults& results,
                         tShape& shape,
                         bool withmargins)
{
    results.witnesses[0] =
        results.witnesses[1] = btVector3(0, 0, 0);
    results.status = btGjkEpaSolver2::sResults::Separated;
    shape.m_shapes[0] = shape0;
    shape.m_shapes[1] = shape1;
    shape.m_toshape1 = wtrs1.getBasis().transposeTimes(wtrs0.getBasis());
    shape.m_toshape0 = wtrs0.inverseTimes(wtrs1);
    shape.EnableMargin(withmargins);
}
}

using namespace gjkepa2_impl;

struct btGjkEpaScratch {
    GJK gjk;
    EPA epa;
};

ROCKETGPU_HD int btGjkEpaSolver2::StackSizeRequirement()
{
    return (sizeof(GJK) + sizeof(EPA));
}

ROCKETGPU_HD ROCKETGPU_BULLET_PORT_NOINLINE bool btGjkEpaSolver2::Distance(const btConvexShape* shape0,
                                        const btTransform& wtrs0,
                                        const btConvexShape* shape1,
                                        const btTransform& wtrs1,
                                        const btVector3& guess,
                                        sResults& results)
{
    tShape shape;
    Initialize(shape0, wtrs0, shape1, wtrs1, results, shape, false);
    GJK gjk;
    GJK::eStatus::_ gjk_status = gjk.Evaluate(shape, guess);
    if (gjk_status == GJK::eStatus::Valid)
    {
        btVector3 w0 = btVector3(0, 0, 0);
        btVector3 w1 = btVector3(0, 0, 0);
        for (U i = 0; i < gjk.m_simplex->rank; ++i)
        {
            const btScalar p = gjk.m_simplex->p[i];
            w0 += shape.Support(gjk.m_simplex->c[i]->d, 0) * p;
            w1 += shape.Support(-gjk.m_simplex->c[i]->d, 1) * p;
        }
        results.witnesses[0] = wtrs0 * w0;
        results.witnesses[1] = wtrs0 * w1;
        results.normal = w0 - w1;
        results.distance = results.normal.length();
        results.normal /= results.distance > GJK_MIN_DISTANCE ? results.distance : 1;
        return true;
    }
    else
    {
        results.status = gjk_status == GJK::eStatus::Inside ? sResults::Penetrating : sResults::GJK_Failed;
        return false;
    }
}

ROCKETGPU_HD ROCKETGPU_BULLET_PORT_NOINLINE bool btGjkEpaSolver2::DistanceWithScratch(const btConvexShape* shape0,
                                                       const btTransform& wtrs0,
                                                       const btConvexShape* shape1,
                                                       const btTransform& wtrs1,
                                                       const btVector3& guess,
                                                       sResults& results,
                                                       btGjkEpaScratch& scratch)
{
    tShape shape;
    Initialize(shape0, wtrs0, shape1, wtrs1, results, shape, false);
    scratch.gjk.Initialize();
    GJK::eStatus::_ gjk_status = scratch.gjk.Evaluate(shape, guess);
    if (gjk_status == GJK::eStatus::Valid)
    {
        btVector3 w0 = btVector3(0, 0, 0);
        btVector3 w1 = btVector3(0, 0, 0);
        for (U i = 0; i < scratch.gjk.m_simplex->rank; ++i)
        {
            const btScalar p = scratch.gjk.m_simplex->p[i];
            w0 += shape.Support(scratch.gjk.m_simplex->c[i]->d, 0) * p;
            w1 += shape.Support(-scratch.gjk.m_simplex->c[i]->d, 1) * p;
        }
        results.witnesses[0] = wtrs0 * w0;
        results.witnesses[1] = wtrs0 * w1;
        results.normal = w0 - w1;
        results.distance = results.normal.length();
        results.normal /= results.distance > GJK_MIN_DISTANCE ? results.distance : 1;
        return true;
    }
    else
    {
        results.status = gjk_status == GJK::eStatus::Inside ? sResults::Penetrating : sResults::GJK_Failed;
        return false;
    }
}

ROCKETGPU_HD ROCKETGPU_BULLET_PORT_NOINLINE bool btGjkEpaSolver2::Penetration(const btConvexShape* shape0,
                                           const btTransform& wtrs0,
                                           const btConvexShape* shape1,
                                           const btTransform& wtrs1,
                                           const btVector3& guess,
                                           sResults& results,
                                           bool usemargins)
{
    tShape shape;
    Initialize(shape0, wtrs0, shape1, wtrs1, results, shape, usemargins);
    GJK gjk;
    GJK::eStatus::_ gjk_status = gjk.Evaluate(shape, -guess);
    switch (gjk_status)
    {
        case GJK::eStatus::Inside:
        {
            EPA epa;
            EPA::eStatus::_ epa_status = epa.Evaluate(gjk, -guess);
            if (epa_status != EPA::eStatus::Failed)
            {
                btVector3 debug_dirs[4] = {
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                };
                btVector3 debug_ws[4] = {
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                };
                for (U i = 0; i < epa.m_result.rank; ++i)
                {
                    debug_dirs[i] = epa.m_result.c[i] ? epa.m_result.c[i]->d : btVector3(0, 0, 0);
                    debug_ws[i] = epa.m_result.c[i] ? epa.m_result.c[i]->w : btVector3(0, 0, 0);
                }
                btVector3 w0 = btVector3(0, 0, 0);
                btVector3 support0_dir_norm[4] = {
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                };
                btVector3 support0_without_margin[4] = {
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                };
                btVector3 support0[4] = {
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                };
                for (U i = 0; i < epa.m_result.rank; ++i)
                {
                    support0_dir_norm[i] = epa.m_result.c[i]->d;
                    if (support0_dir_norm[i].length2() < (SIMD_EPSILON * SIMD_EPSILON)) {
                        support0_dir_norm[i].setValue(-1.0f, -1.0f, -1.0f);
                    }
                    support0_dir_norm[i].normalize();
                    support0_without_margin[i] =
                        shape.m_shapes[0]->localGetSupportVertexWithoutMarginNonVirtual(
                            support0_dir_norm[i]);
                    support0[i] =
                        support0_without_margin[i] +
                        shape.m_shapes[0]->getMarginNonVirtual() * support0_dir_norm[i];
                    w0 += support0[i] * epa.m_result.p[i];
                }
#if !defined(__CUDA_ARCH__)
                TraceDebugEpaWitnessReconstruct(
                    support0_dir_norm,
                    support0_without_margin,
                    support0,
                    w0,
                    wtrs0,
                    epa.m_normal,
                    epa.m_depth);
#endif
                results.status = sResults::Penetrating;
                results.witnesses[0] = wtrs0 * w0;
                results.witnesses[1] = wtrs0 * (w0 - epa.m_normal * epa.m_depth);
                results.normal = -epa.m_normal;
                results.distance = -epa.m_depth;
                return true;
            }
            else
                results.status = sResults::EPA_Failed;
        }
        break;
        case GJK::eStatus::Failed:
            results.status = sResults::GJK_Failed;
            break;
        default:
        {
        }
    }
    return false;
}

ROCKETGPU_HD ROCKETGPU_BULLET_PORT_NOINLINE bool btGjkEpaSolver2::PenetrationWithScratch(const btConvexShape* shape0,
                                                          const btTransform& wtrs0,
                                                          const btConvexShape* shape1,
                                                          const btTransform& wtrs1,
                                                          const btVector3& guess,
                                                          sResults& results,
                                                          btGjkEpaScratch& scratch,
                                                          bool usemargins,
                                                          btGjkEpaDebugStop* debug_stop)
{
    tShape shape;
    Initialize(shape0, wtrs0, shape1, wtrs1, results, shape, usemargins);
    scratch.gjk.Initialize();
    GJK::eStatus::_ gjk_status = scratch.gjk.Evaluate(shape, -guess, debug_stop);
    switch (gjk_status)
    {
        case GJK::eStatus::Inside:
        {
            scratch.epa.Initialize();
            EPA::eStatus::_ epa_status = scratch.epa.Evaluate(scratch.gjk, -guess);
            if (epa_status != EPA::eStatus::Failed)
            {
                btVector3 debug_dirs[4] = {
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                };
                btVector3 debug_ws[4] = {
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                };
                for (U i = 0; i < scratch.epa.m_result.rank; ++i)
                {
                    debug_dirs[i] = scratch.epa.m_result.c[i] ? scratch.epa.m_result.c[i]->d : btVector3(0, 0, 0);
                    debug_ws[i] = scratch.epa.m_result.c[i] ? scratch.epa.m_result.c[i]->w : btVector3(0, 0, 0);
                }
                btVector3 w0 = btVector3(0, 0, 0);
                btVector3 support0_dir_norm[4] = {
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                };
                btVector3 support0_without_margin[4] = {
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                };
                btVector3 support0[4] = {
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                    btVector3(0, 0, 0),
                };
                for (U i = 0; i < scratch.epa.m_result.rank; ++i)
                {
                    support0_dir_norm[i] = scratch.epa.m_result.c[i]->d;
                    if (support0_dir_norm[i].length2() < (SIMD_EPSILON * SIMD_EPSILON)) {
                        support0_dir_norm[i].setValue(-1.0f, -1.0f, -1.0f);
                    }
                    support0_dir_norm[i].normalize();
                    support0_without_margin[i] =
                        shape.m_shapes[0]->localGetSupportVertexWithoutMarginNonVirtual(
                            support0_dir_norm[i]);
                    support0[i] =
                        support0_without_margin[i] +
                        shape.m_shapes[0]->getMarginNonVirtual() * support0_dir_norm[i];
                    w0 += support0[i] * scratch.epa.m_result.p[i];
                }
#if !defined(__CUDA_ARCH__)
                TraceDebugEpaWitnessReconstruct(
                    support0_dir_norm,
                    support0_without_margin,
                    support0,
                    w0,
                    wtrs0,
                    scratch.epa.m_normal,
                    scratch.epa.m_depth);
#endif
                results.status = sResults::Penetrating;
                results.witnesses[0] = wtrs0 * w0;
                results.witnesses[1] = wtrs0 * (w0 - scratch.epa.m_normal * scratch.epa.m_depth);
                results.normal = -scratch.epa.m_normal;
                results.distance = -scratch.epa.m_depth;
                return true;
            }
            else
                results.status = sResults::EPA_Failed;
        }
        break;
        case GJK::eStatus::Failed:
            results.status = sResults::GJK_Failed;
            break;
        default:
        {
        }
    }
    return false;
}

ROCKETGPU_HD btScalar btGjkEpaSolver2::SignedDistance(const btVector3& position,
                                                  btScalar margin,
                                                  const btConvexShape* shape0,
                                                  const btTransform& wtrs0,
                                                  sResults& results)
{
    tShape shape;
    btSphereShape shape1(margin);
    btTransform wtrs1(btQuaternion(0, 0, 0, 1), position);
    Initialize(shape0, wtrs0, &shape1, wtrs1, results, shape, false);
    GJK gjk;
    GJK::eStatus::_ gjk_status = gjk.Evaluate(shape, btVector3(1, 1, 1));
    if (gjk_status == GJK::eStatus::Valid)
    {
        btVector3 w0 = btVector3(0, 0, 0);
        btVector3 w1 = btVector3(0, 0, 0);
        for (U i = 0; i < gjk.m_simplex->rank; ++i)
        {
            const btScalar p = gjk.m_simplex->p[i];
            w0 += shape.Support(gjk.m_simplex->c[i]->d, 0) * p;
            w1 += shape.Support(-gjk.m_simplex->c[i]->d, 1) * p;
        }
        results.witnesses[0] = wtrs0 * w0;
        results.witnesses[1] = wtrs0 * w1;
        const btVector3 delta = results.witnesses[1] -
                                results.witnesses[0];
        const btScalar margin = shape0->getMarginNonVirtual() +
                                shape1.getMarginNonVirtual();
        const btScalar length = delta.length();
        results.normal = delta / length;
        results.witnesses[0] += results.normal * margin;
        results.distance = length - margin;
        return results.distance;
    }
    else
    {
        if (gjk_status == GJK::eStatus::Inside)
        {
            if (Penetration(shape0, wtrs0, &shape1, wtrs1, gjk.m_ray, results))
            {
                const btVector3 delta = results.witnesses[0] -
                                        results.witnesses[1];
                const btScalar length = delta.length();
                if (length >= SIMD_EPSILON)
                    results.normal = delta / length;
                return (-length);
            }
        }
    }
    return (SIMD_INFINITY);
}

ROCKETGPU_HD bool btGjkEpaSolver2::SignedDistance(const btConvexShape* shape0,
                                              const btTransform& wtrs0,
                                              const btConvexShape* shape1,
                                              const btTransform& wtrs1,
                                              const btVector3& guess,
                                              sResults& results)
{
    if (!Distance(shape0, wtrs0, shape1, wtrs1, guess, results))
        return (Penetration(shape0, wtrs0, shape1, wtrs1, guess, results, false));
    else
        return true;
}

#undef GJK_MAX_ITERATIONS
#undef GJK_ACCURACY
#undef GJK_MIN_DISTANCE
#undef GJK_DUPLICATED_EPS
#undef GJK_SIMPLEX2_EPS
#undef GJK_SIMPLEX3_EPS
#undef GJK_SIMPLEX4_EPS

#undef EPA_MAX_VERTICES
#undef EPA_MAX_FACES
#undef EPA_MAX_ITERATIONS
#undef EPA_ACCURACY
#undef EPA_FALLBACK
#undef EPA_PLANE_EPS
#undef EPA_INSIDE_EPS
}
}
}

#undef ROCKETGPU_BULLET_PORT_NOINLINE
