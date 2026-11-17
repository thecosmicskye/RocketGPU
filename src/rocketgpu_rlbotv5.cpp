#include <rocketgpu/rlbotv5.h>

#include <rocketgpu/detail/native_sse_normalization.hpp>

namespace rocketgpu {
namespace {
using namespace detail;

float RlbotV5RoundPos(float x) {
    const float scaled = x * 100.0f;
    return floorf(scaled + 0.5f) * (1.0f / 100.0f);
}

float RlbotV5TruncOffset(float x, float scale) {
    const float inv_scale = 1.0f / scale;
    const float k = static_cast<float>(static_cast<int32_t>(x * scale));
    const float rounded = k * inv_scale;
    const float offset_mag = 0.1f * inv_scale;
    const float sgn = rounded == 0.0f ? 0.0f : (rounded > 0.0f ? 1.0f : -1.0f);
    const float offset = sgn * offset_mag;
    return rounded + offset;
}

float RlbotV5Radians(int rotator_units) {
    const float degrees = static_cast<float>(rotator_units) * (360.0f / 65536.0f);
    return degrees * (3.14159274f / 180.0f);
}

RlbotV5Physics RlbotV5PhysicsOfBody(const Vec3& pos_bt, const Vec3& vel, const Vec3& ang_vel, const Vec3& fwd,
                                const Vec3& right, const Vec3& up) {
    RlbotV5Physics out;
    const Vec3 pos = pos_bt * 50.0f;
    out.location = Vec3(RlbotV5RoundPos(pos.x), RlbotV5RoundPos(pos.y), RlbotV5RoundPos(pos.z));
    out.velocity = Vec3(RlbotV5TruncOffset(vel.x, 100.0f), RlbotV5TruncOffset(vel.y, 100.0f),
                        RlbotV5TruncOffset(vel.z, 100.0f));
    out.angular_velocity = Vec3(RlbotV5TruncOffset(ang_vel.x, 10000.0f), RlbotV5TruncOffset(ang_vel.y, 10000.0f),
                                RlbotV5TruncOffset(ang_vel.z, 10000.0f));
    const Ue3Rotator rot = Ue3QuatToRotator(GameRbStateQuantizeQuat(GameGetRotationScalar(fwd, right, up)));
    out.pitch = RlbotV5Radians(rot.pitch);
    out.yaw = RlbotV5Radians(rot.yaw);
    out.roll = RlbotV5Radians(rot.roll);
    return out;
}
}

RlbotV5Physics RlbotV5CarPhysics(const CarState& state, const MutatorConfig& mutator) {
    using namespace detail;
    CarState car = state;
    if (car.tick_count_since_update != 0) {
        car.RestoreWorkingFields();
    }
    Vec3 vel = car.vel_bt * 50.0f;
    if (!car.is_demoed) {
        const float len_sq = LengthSq(vel);
        if (len_sq > mutator.car_max_speed * mutator.car_max_speed) {
            const Vec3 dir = mutator.native_sphere_rsqrt_table
                ? NativeSseSphereNormalizer{mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift}(vel)
                : vel * RsqrtScalar(len_sq);
            vel = dir * mutator.car_max_speed;
        }
        car.ang_vel = NativeSseClampLength(car.ang_vel, mutator.car_max_ang_speed, mutator.native_sphere_rsqrt_table,
                                           mutator.native_sphere_rsqrt_shift);
    }
    Vec3 fwd = car.basis_forward, right = car.basis_right, up = car.basis_up;
    if (!car.basis_authoritative) {
        GetBasisVectors(car.rot, fwd, right, up);
    }
    return RlbotV5PhysicsOfBody(car.pos_bt, vel, car.ang_vel, fwd, right, up);
}

namespace detail {
void CarVelAfterCaps(const CarState& car, const MutatorConfig& mutator, Vec3& vel, Vec3& ang_vel) {
    vel = car.vel;
    ang_vel = car.ang_vel;
    if (car.is_demoed) {
        return;
    }
    const float max_speed_bt = mutator.car_max_speed * (1.0f / 50.0f);
    if (LengthSq(car.vel_bt) > max_speed_bt * max_speed_bt) {
        const Vec3 vel_bt_normalized = mutator.native_sphere_rsqrt_table
            ? NativeSseSphereNormalizer{mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift}(car.vel_bt)
            : car.vel_bt * RsqrtScalar(LengthSq(car.vel_bt));
        vel = (vel_bt_normalized * max_speed_bt) * 50.0f;
    }
    ang_vel = NativeSseClampLength(car.ang_vel, mutator.car_max_ang_speed, mutator.native_sphere_rsqrt_table,
                                   mutator.native_sphere_rsqrt_shift);
}
}

bool RlbotV5IsSupersonic(const CarState& car, const MutatorConfig& mutator) {
    const Vec3 vel = RlbotV5CarPhysics(car, mutator).velocity;
    return detail::LengthSq(vel) >= mutator.supersonic_start_speed * mutator.supersonic_start_speed;
}

RlbotV5Physics RlbotV5BallPhysics(const BallState& ball, const MutatorConfig& mutator) {
    using namespace detail;
    Vec3 vel = ball.vel_bt * 50.0f;
    if (!mutator.native_sphere_rsqrt_table) {
        const float len_sq = LengthSq(vel);
        if (len_sq > mutator.ball_max_speed * mutator.ball_max_speed) {
            vel = (vel * RsqrtScalar(len_sq)) * mutator.ball_max_speed;
        }
    } else {
        const float max_speed_bt = mutator.ball_max_speed * (1.0f / 50.0f);
        if (LengthSq(ball.vel_bt) > max_speed_bt * max_speed_bt) {
            const Vec3 dir = NativeSseSphereNormalizer{mutator.native_sphere_rsqrt_table,
                                                       mutator.native_sphere_rsqrt_shift}(ball.vel_bt);
            vel = (dir * max_speed_bt) * 50.0f;
        }
    }
    const Vec3 ang_vel = NativeSseClampLength(ball.ang_vel, mutator.ball_max_ang_speed,
                                              mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift);
    Vec3 fwd = ball.basis_forward, right = ball.basis_right, up = ball.basis_up;
    if (!ball.basis_authoritative) {
        GetOfficialUploadBasisVectors(ball.rot, fwd, right, up);
    }
    return RlbotV5PhysicsOfBody(ball.pos_bt, vel, ang_vel, fwd, right, up);
}

}
