

ROCKETGPU_HD constexpr float HeatseekerTargetSpeedIncrement() { return 85.0f; }
ROCKETGPU_HD constexpr float HeatseekerMinSpeedupInterval() { return 1.0f; }
ROCKETGPU_HD constexpr float HeatseekerTargetY() { return 5120.0f; }
ROCKETGPU_HD constexpr float HeatseekerTargetZ() { return 320.0f; }
ROCKETGPU_HD constexpr float HeatseekerHorizontalBlend() { return 1.45f; }
ROCKETGPU_HD constexpr float HeatseekerVerticalBlend() { return 0.78f; }
ROCKETGPU_HD constexpr float HeatseekerSpeedBlend() { return 0.3f; }
ROCKETGPU_HD constexpr float HeatseekerMaxTurnPitch() { return 7000.0f * 3.14159265358979323846f / static_cast<float>(1 << 15); }
ROCKETGPU_HD constexpr float HeatseekerMaxSpeed() { return 4600.0f; }
ROCKETGPU_HD constexpr float HeatseekerWallBounceChangeYThresh() { return 300.0f; }
ROCKETGPU_HD constexpr float HeatseekerWallBounceChangeYNormal() { return 0.5f; }
ROCKETGPU_HD constexpr float HeatseekerWallBounceForceScale() { return 1.0f / 3.0f; }
ROCKETGPU_HD constexpr float HeatseekerWallBounceUpFrac() { return 0.3f; }
ROCKETGPU_HD constexpr float HeatseekerArenaExtentY() { return 5120.0f; }

struct RsAngle {
    float yaw = 0.0f;
    float pitch = 0.0f;
    float roll = 0.0f;
};

ROCKETGPU_HD float RsWrapNormalizeFloat(float val, float minmax) {
    float result = fmodf(val, minmax * 2.0f);
    if (result > minmax) {
        result -= minmax * 2.0f;
    } else if (result < -minmax) {
        result += minmax * 2.0f;
    }
    return result;
}

ROCKETGPU_HD void RsAngleNormalizeFix(RsAngle& a) {
    a.yaw = RsWrapNormalizeFloat(a.yaw, 3.14159265358979323846f);
    a.pitch = RsWrapNormalizeFloat(a.pitch, 3.14159265358979323846f / 2.0f);
    a.roll = RsWrapNormalizeFloat(a.roll, 3.14159265358979323846f);
}

ROCKETGPU_HD RsAngle RsAngleFromVec(const Vec3& forward) {
    RsAngle out;
    constexpr float kFltEpsilon = 1.1920929e-07f;
    if (fabsf(forward.y) > kFltEpsilon || fabsf(forward.x) > kFltEpsilon) {
        out.yaw = GlibcAtan2f(forward.y, forward.x);
        const float dist2d = sqrtf(forward.x * forward.x + forward.y * forward.y);
        out.pitch = -GlibcAtan2f(-forward.z, dist2d);
    } else {
        out.yaw = 0.0f;
        if (forward.z > kFltEpsilon) {
            out.pitch = 3.14159265358979323846f / 2.0f;
        } else if (forward.z < -kFltEpsilon) {
            out.pitch = -3.14159265358979323846f / 2.0f;
        } else {
            out.pitch = 0.0f;
        }
    }
    out.roll = 0.0f;
    return out;
}

ROCKETGPU_HD Vec3 RsAngleForwardVec(const RsAngle& a) {
    const float cp = RocketGpuCosf(-a.pitch);
    const float cy = RocketGpuCosf(a.yaw);
    const float sy = RocketGpuSinf(a.yaw);
    const float sp = RocketGpuSinf(-a.pitch);
    return Vec3(cp * cy, cp * sy, -sp);
}

ROCKETGPU_HD RsAngle RsAngleDelta(const RsAngle& from, const RsAngle& to) {
    RsAngle d;
    d.yaw = to.yaw - from.yaw;
    d.pitch = to.pitch - from.pitch;
    d.roll = to.roll - from.roll;
    RsAngleNormalizeFix(d);
    return d;
}

ROCKETGPU_HD RsAngle RsRoundAngleUE3(RsAngle ang) {
    constexpr float kToInts = static_cast<float>(1 << 15) / 3.14159265358979323846f;
    constexpr float kBackToRadians = (1.0f / kToInts) * 4.0f;
    constexpr int kRoundingMask = 0x4000 - 1;
    const int r_yaw = (static_cast<int>(ang.yaw * kToInts) >> 2) & kRoundingMask;
    const int r_pitch = (static_cast<int>(ang.pitch * kToInts) >> 2) & kRoundingMask;
    ang.yaw = static_cast<float>(r_yaw) * kBackToRadians;
    ang.pitch = static_cast<float>(r_pitch) * kBackToRadians;
    return ang;
}

ROCKETGPU_HD float RsVecLength(const Vec3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

template <typename SetBallVelBtFn>
ROCKETGPU_HD void HeatseekerBallPreTick(BallState& ball, float dt, SetBallVelBtFn set_ball_linear_velocity_bt) {
    const float y_target_dir = ball.hs_info.y_target_dir;
    if (y_target_dir == 0.0f) {
        return;
    }
    const RsAngle vel_angle = RsAngleFromVec(ball.vel);
    const Vec3 goal_target_pos(0.0f, HeatseekerTargetY() * y_target_dir, HeatseekerTargetZ());
    const RsAngle angle_to_goal = RsAngleFromVec(goal_target_pos - ball.pos);
    const RsAngle delta_angle = RsAngleDelta(vel_angle, angle_to_goal);
    const float cur_speed = RsVecLength(ball.vel);
    const float speed_ratio = cur_speed / HeatseekerMaxSpeed();
    RsAngle new_angle = vel_angle;
    const float base_interp_factor = speed_ratio * dt;
    new_angle.yaw += delta_angle.yaw * base_interp_factor * HeatseekerHorizontalBlend();
    new_angle.pitch += delta_angle.pitch * base_interp_factor * HeatseekerVerticalBlend();
    RsAngleNormalizeFix(new_angle);
    new_angle.pitch = Clamp(new_angle.pitch, -HeatseekerMaxTurnPitch(), HeatseekerMaxTurnPitch());
    new_angle = RsRoundAngleUE3(new_angle);
    const float new_speed = cur_speed + ((ball.hs_info.cur_target_speed - cur_speed) * HeatseekerSpeedBlend());
    const Vec3 new_dir = RsAngleForwardVec(new_angle);
    const Vec3 new_vel = new_dir * new_speed;
    set_ball_linear_velocity_bt(ball, new_vel * (1.0f / 50.0f));
    ball.hs_info.time_since_hit += dt;
}

ROCKETGPU_HD void HeatseekerOnBallHit(BallState& ball, int car_team) {
    const bool can_increase = (ball.hs_info.time_since_hit > HeatseekerMinSpeedupInterval()) || (ball.hs_info.y_target_dir == 0.0f);
    const float new_target_dir = (car_team == 0) ? 1.0f : -1.0f;
    if (can_increase && (new_target_dir != ball.hs_info.y_target_dir)) {
        ball.hs_info.time_since_hit = 0.0f;
        ball.hs_info.cur_target_speed = Min(ball.hs_info.cur_target_speed + HeatseekerTargetSpeedIncrement(), HeatseekerMaxSpeed());
    }
    ball.hs_info.y_target_dir = new_target_dir;
}

ROCKETGPU_HD bool HeatseekerOnWorldContact(BallState& ball, const Vec3& normal) {
    if (ball.hs_info.y_target_dir == 0.0f) {
        return false;
    }
    const float dir = ball.hs_info.y_target_dir;
    const float rel_normal_y = normal.y * dir;
    const float rel_y = ball.pos.y * dir;
    if (rel_normal_y <= -HeatseekerWallBounceChangeYNormal() &&
        rel_y >= HeatseekerArenaExtentY() - HeatseekerWallBounceChangeYThresh()) {
        ball.hs_info.y_target_dir *= -1.0f;
        const Vec3 goal_target_pos(0.0f, HeatseekerTargetY() * ball.hs_info.y_target_dir, HeatseekerTargetZ());
        const Vec3 dir_to_goal = RocketSimVecNormalized(goal_target_pos - ball.pos);
        const Vec3 bounce_dir = dir_to_goal * (1.0f - HeatseekerWallBounceUpFrac()) + Vec3(0.0f, 0.0f, 1.0f) * HeatseekerWallBounceUpFrac();
        const Vec3 bounce_impulse = bounce_dir * RsVecLength(ball.vel) * HeatseekerWallBounceForceScale();
        ball.vel_impulse_cache += bounce_impulse;
        ball.vel_impulse_cache_bt += bounce_impulse * (1.0f / 50.0f);
        return true;
    }
    return false;
}

