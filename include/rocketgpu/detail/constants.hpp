#pragma once

#include "rocketgpu/detail/math.hpp"

namespace rocketgpu {
namespace detail {
ROCKETGPU_HD constexpr float ArenaExtentX() { return 4096.0f; }
ROCKETGPU_HD constexpr float ArenaExtentY() { return 5120.0f; }
ROCKETGPU_HD constexpr float ArenaHeight() { return 2048.0f; }
ROCKETGPU_HD constexpr float CornerCenterX() { return 3072.0f; }
ROCKETGPU_HD constexpr float CornerCenterY() { return 4096.0f; }
ROCKETGPU_HD constexpr float CornerRadius() { return 1024.0f; }
ROCKETGPU_HD constexpr float FloorRampRadius() { return 256.0f; }
ROCKETGPU_HD constexpr float CarSpawnRestZ() { return 17.0f; }
ROCKETGPU_HD constexpr float BallRestZ() { return 93.15f; }
ROCKETGPU_HD constexpr int BoostPadCountSoccar() { return 34; }

ROCKETGPU_HD constexpr float CarTorqueScale() { return (2.0f * Pi() / 65536.0f) * 1000.0f; }

ROCKETGPU_HD constexpr float CarBallCollisionFriction() { return 2.0f; }
ROCKETGPU_HD constexpr float CarBallCollisionRestitution() { return 0.0f; }
ROCKETGPU_HD constexpr float CarCarCollisionFriction() { return 0.09f; }
ROCKETGPU_HD constexpr float CarCarCollisionRestitution() { return 0.1f; }
ROCKETGPU_HD constexpr float BoostPadCylinderHeight() { return 95.0f; }

struct SpawnPos {
    float x;
    float y;
    float yaw;
};

ROCKETGPU_HD constexpr SpawnPos HeatseekerKickoffSpawn(int index) {
    return index == 0 ? SpawnPos{-1000.0f, -4620.0f, 3.14159265358979323846f / 2.0f}
         : index == 1 ? SpawnPos{1000.0f, -4620.0f, 3.14159265358979323846f / 2.0f}
         : index == 2 ? SpawnPos{-2000.0f, -4620.0f, 3.14159265358979323846f / 2.0f}
                      : SpawnPos{2000.0f, -4620.0f, 3.14159265358979323846f / 2.0f};
}
ROCKETGPU_HD constexpr SpawnPos SoccarKickoffSpawn(int index) {
    switch (index) {
    case 0: return {-2048.0f, -2560.0f, Pi() * 0.25f};
    case 1: return { 2048.0f, -2560.0f, Pi() * 0.75f};
    case 2: return { -256.0f, -3840.0f, Pi() * 0.50f};
    case 3: return {  256.0f, -3840.0f, Pi() * 0.50f};
    default: return {   0.0f, -4608.0f, Pi() * 0.50f};
    }
}

ROCKETGPU_HD constexpr SpawnPos SoccarRespawnSpawn(int index) {
    switch (index) {
    case 0: return {-2304.0f, -4608.0f, HalfPi()};
    case 1: return {-2688.0f, -4608.0f, HalfPi()};
    case 2: return { 2304.0f, -4608.0f, HalfPi()};
    default: return { 2688.0f, -4608.0f, HalfPi()};
    }
}

struct ArenaRules {
    Vec3 kickoff_pos[2][5];
    float kickoff_yaw[2][5];
    Vec3 respawn_pos[2][4];
    float respawn_yaw[2][4];
    Vec3 ball_spawn;
    Vec3 goal_point[2];
    Vec3 goal_normal[2];
    Vec3 goal_min[2];
    Vec3 goal_max[2];
};

ROCKETGPU_HD constexpr Vec3 BoostPadLocSoccar(int index) {
    switch (index) {
    case 0: return Vec3(0.0f, -4240.0f, 70.0f);
    case 1: return Vec3(-1792.0f, -4184.0f, 70.0f);
    case 2: return Vec3(1792.0f, -4184.0f, 70.0f);
    case 3: return Vec3(-940.0f, -3308.0f, 70.0f);
    case 4: return Vec3(940.0f, -3308.0f, 70.0f);
    case 5: return Vec3(0.0f, -2816.0f, 70.0f);
    case 6: return Vec3(-3584.0f, -2484.0f, 70.0f);
    case 7: return Vec3(3584.0f, -2484.0f, 70.0f);
    case 8: return Vec3(-1788.0f, -2302.0f, 70.0f);
    case 9: return Vec3(1788.0f, -2302.0f, 70.0f);
    case 10: return Vec3(-2048.0f, -1036.0f, 70.0f);
    case 11: return Vec3(0.0f, -1024.0f, 70.0f);
    case 12: return Vec3(2048.0f, -1036.0f, 70.0f);
    case 13: return Vec3(-1024.0f, 0.0f, 70.0f);
    case 14: return Vec3(1024.0f, 0.0f, 70.0f);
    case 15: return Vec3(-2048.0f, 1036.0f, 70.0f);
    case 16: return Vec3(0.0f, 1024.0f, 70.0f);
    case 17: return Vec3(2048.0f, 1036.0f, 70.0f);
    case 18: return Vec3(-1788.0f, 2302.0f, 70.0f);
    case 19: return Vec3(1788.0f, 2302.0f, 70.0f);
    case 20: return Vec3(-3584.0f, 2484.0f, 70.0f);
    case 21: return Vec3(3584.0f, 2484.0f, 70.0f);
    case 22: return Vec3(0.0f, 2816.0f, 70.0f);
    case 23: return Vec3(-940.0f, 3308.0f, 70.0f);
    case 24: return Vec3(940.0f, 3308.0f, 70.0f);
    case 25: return Vec3(-1792.0f, 4184.0f, 70.0f);
    case 26: return Vec3(1792.0f, 4184.0f, 70.0f);
    case 27: return Vec3(0.0f, 4240.0f, 70.0f);
    case 28: return Vec3(-3584.0f, 0.0f, 73.0f);
    case 29: return Vec3(3584.0f, 0.0f, 73.0f);
    case 30: return Vec3(-3072.0f, 4096.0f, 73.0f);
    case 31: return Vec3(3072.0f, 4096.0f, 73.0f);
    case 32: return Vec3(-3072.0f, -4096.0f, 73.0f);
    default: return Vec3(3072.0f, -4096.0f, 73.0f);
    }
}

ROCKETGPU_HD constexpr uint8_t BoostPadIsBigSoccar(int index) {
    return index >= 28 ? 1 : 0;
}

ROCKETGPU_HD constexpr float SteerDegToRad() { return 0.01745329238474369f; }

template <int I, int N>
ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE float EvalInterpCurveLinearStep(
    const float (&in)[N], const float (&out)[N], float x) {
    if constexpr (I < N) {
        if (in[I] > x) {
            return (out[I] - out[I - 1]) * ((x - in[I - 1]) / (in[I] - in[I - 1])) + out[I - 1];
        }
        return EvalInterpCurveLinearStep<I + 1, N>(in, out, x);
    } else {
        return out[N - 1];
    }
}

template <int N>
ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE float EvalInterpCurveLinear(
    const float (&in)[N], const float (&out)[N], float x) {
    if (N < 2 || in[0] >= x) return out[0];
    if (x >= in[N - 1]) return out[N - 1];
    return EvalInterpCurveLinearStep<1, N>(in, out, x);
}

ROCKETGPU_HD float EvalSteerAngleFromSpeed(const float (&c)[6], float speed_abs) {
    const float in[6] = {0.0f, 500.0f, 1000.0f, 1500.0f, 1750.0f, 3000.0f};
    return EvalInterpCurveLinear(in, c, Abs(speed_abs));
}

ROCKETGPU_HD float EvalPowerslideSteerAngleFromSpeed(const float (&c)[2], float speed_abs) {
    const float in[2] = {0.0f, 2500.0f};
    return EvalInterpCurveLinear(in, c, Abs(speed_abs));
}

ROCKETGPU_HD float EvalSteerAngleFromSpeedThreeWheel(float speed_abs) {
    const float in[2] = {0.0f, 2300.0f};
    const float out[2] = {19.622270584106445f, 1.9959354400634766f};
    return EvalInterpCurveLinear(in, out, Abs(speed_abs));
}

ROCKETGPU_HD float EvalDriveTorqueFactor(float forward_speed) {
    const float in[5] = {-1410.0f, -1400.0f, 0.0f, 1400.0f, 1410.0f};
    const float out[5] = {0.0f, 0.1f, 1.0f, 0.1f, 0.0f};
    return EvalInterpCurveLinear(in, out, forward_speed);
}

ROCKETGPU_HD float EvalNonStickyFrictionFactor(float contact_normal_z) {
    const float in[3] = {0.0f, 0.707f, 1.0f};
    const float out[3] = {0.1f, 0.5f, 1.0f};
    return EvalInterpCurveLinear(in, out, contact_normal_z);
}

ROCKETGPU_HD float EvalLatFriction(float input) {
    const float in[2] = {0.0f, 1.0f};
    const float out[2] = {1.0f, 0.2f};
    return EvalInterpCurveLinear(in, out, input);
}

ROCKETGPU_HD float EvalLatFrictionThreeWheel(float input) {
    const float in[2] = {0.0f, 1.0f};
    const float out[2] = {0.30f, 0.25f};
    return EvalInterpCurveLinear(in, out, input);
}

ROCKETGPU_HD float EvalLongFriction(float          ) {
    return 1.0f;
}

ROCKETGPU_HD float EvalHandbrakeLatFrictionFactor(float          ) {
    return 0.1f;
}

ROCKETGPU_HD float EvalHandbrakeLongFrictionFactor(float input) {
    const float in[2] = {0.0f, 1.0f};
    const float out[2] = {0.5f, 0.9f};
    return EvalInterpCurveLinear(in, out, input);
}

ROCKETGPU_HD float EvalBallCarExtraImpulseFactor(float speed_abs) {
    const float in[5] = {0.0f, 500.0f, 1400.0f, 2300.0f, 4600.0f};
    const float out[5] = {0.65f, 0.65f, 0.6f, 0.55f, 0.30f};
    return EvalInterpCurveLinear(in, out, Abs(speed_abs));
}

ROCKETGPU_HD float EvalLinearPieceCurve3(
    float x0, float y0, float x1, float y1, float x2, float y2, float input) {
    if (input <= x0) return y0;
    if (x1 > input) {
        const float range_between = x1 - x0;
        const float val_diff_between = y1 - y0;
        const float linear_interp_factor = (input - x0) / range_between;
        return y0 + val_diff_between * linear_interp_factor;
    }
    if (x2 > input) {
        const float range_between = x2 - x1;
        const float val_diff_between = y2 - y1;
        const float linear_interp_factor = (input - x1) / range_between;
        return y1 + val_diff_between * linear_interp_factor;
    }
    return y2;
}

ROCKETGPU_HD float EvalBumpVelGround(float speed) {
    return EvalLinearPieceCurve3(0.0f, 5.0f / 6.0f, 1400.0f, 1100.0f, 2200.0f, 1530.0f, speed);
}

ROCKETGPU_HD float EvalBumpVelAir(float speed) {
    return EvalLinearPieceCurve3(0.0f, 5.0f / 6.0f, 1400.0f, 1390.0f, 2200.0f, 1945.0f, speed);
}

ROCKETGPU_HD float EvalBumpUpwardVel(float speed) {
    return EvalLinearPieceCurve3(0.0f, 2.0f / 6.0f, 1400.0f, 278.0f, 2200.0f, 417.0f, speed);
}

ROCKETGPU_HD float EvalGameBumpVelGround(float speed) {
    return EvalLinearPieceCurve3(0.0f, 0.75f / 0.9f, 1400.0f, 1000.0f / 0.9f, 2200.0f, 1375.0f / 0.9f, speed);
}

ROCKETGPU_HD float EvalGameBumpVelAir(float speed) {
    return EvalLinearPieceCurve3(0.0f, 0.75f / 0.9f, 1400.0f, 1250.0f / 0.9f, 2200.0f, 1750.0f / 0.9f, speed);
}

ROCKETGPU_HD float EvalGameBumpUpwardVel(float speed) {
    return EvalLinearPieceCurve3(0.0f, 0.25f / 0.9f, 1400.0f, 250.0f / 0.9f, 2200.0f, 375.0f / 0.9f, speed);
}
}
}
