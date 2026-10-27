

ROCKETGPU_HD constexpr int SnowdayPuckCirclePointAmount() { return 20; }
ROCKETGPU_HD constexpr int SnowdayPuckPointCount() { return 2 * SnowdayPuckCirclePointAmount(); }
ROCKETGPU_HD constexpr float SnowdayPuckRadiusUu() { return 114.25f; }
ROCKETGPU_HD constexpr float SnowdayPuckHeightUu() { return 62.5f; }
ROCKETGPU_HD constexpr float SnowdayPuckMassBt() { return 50.0f; }
ROCKETGPU_HD constexpr float SnowdayPuckFriction() { return 0.1f; }
ROCKETGPU_HD constexpr float SnowdayPuckRestitution() { return 0.3f; }

#if !defined(__CUDA_ARCH__)

inline void BuildSnowdayPuckPointsBt(float ball_radius_uu, Vec3* out) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    const float angStep = static_cast<float>((3.14159265358979323846 * 2) / SnowdayPuckCirclePointAmount());
    float curAng = 0;
    for (int i = 0; i < SnowdayPuckCirclePointAmount(); i++) {
        Vec3 point(
            cosf(curAng) * ball_radius_uu * kUUToBT,
            sinf(curAng) * ball_radius_uu * kUUToBT,
            SnowdayPuckHeightUu() / 2 * kUUToBT);
        out[2 * i] = point;
        point.z *= -1;
        out[2 * i + 1] = point;
        curAng += angStep;
    }
}
#endif

#if !defined(__CUDA_ARCH__)

inline Vec3 ComputeSnowdayPuckLocalInertiaBt(const Vec3* points_bt, float mass_bt) {
    bullet_port::btVector3 pts[40];
    for (int i = 0; i < SnowdayPuckPointCount(); ++i) {
        pts[i] = bullet_port::btVector3(points_bt[i].x, points_bt[i].y, points_bt[i].z);
    }
    const bullet_port::btConvexHullShape hull(pts, SnowdayPuckPointCount());
    const bullet_port::btVector3 inertia = hull.calculateLocalInertia(mass_bt);
    return Vec3(inertia.x(), inertia.y(), inertia.z());
}
#endif

#if !defined(__CUDA_ARCH__)

inline float ComputeSnowdayPuckContactBreakingThresholdBt(const Vec3* points_bt) {
    bullet_port::btVector3 pts[40];
    for (int i = 0; i < SnowdayPuckPointCount(); ++i) {
        pts[i] = bullet_port::btVector3(points_bt[i].x, points_bt[i].y, points_bt[i].z);
    }
    const bullet_port::btConvexHullShape hull(pts, SnowdayPuckPointCount());
    const bullet_port::btTransform ident(bullet_port::btVector3(0.0f, 0.0f, 0.0f));
    bullet_port::btVector3 aabb_min, aabb_max;
    hull.getAabb(ident, aabb_min, aabb_max);
    const float radius = (aabb_max - aabb_min).length() * 0.5f;
    const bullet_port::btVector3 center = (aabb_min + aabb_max) * 0.5f;
    const float disc = radius + center.length();
    return disc * 0.02f;
}
#endif

