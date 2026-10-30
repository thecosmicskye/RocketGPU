

struct WheelStepState {
    Vec3 hard_point_ws = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 ray_source_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 wheel_dir_ws = Vec3(0.0f, 0.0f, -1.0f);
    Vec3 wheel_axle_ws = Vec3(0.0f, -1.0f, 0.0f);
    Vec3 contact_point_ws = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 contact_point_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 contact_normal_ws = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 vel_at_contact = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 impulse = Vec3(0.0f, 0.0f, 0.0f);
    float suspension_length = 0.0f;
    float wheel_trace_length = 0.0f;
    float unclamped_wheel_trace_length = 0.0f;

    float wheel_trace_len_bt = 0.0f;

    float suspension_len_bt = 0.0f;
    float suspension_relative_velocity_bt = 0.0f;
    float debug_hit_fraction = 0.0f;
    float suspension_relative_velocity = 0.0f;
    float clipped_inv_contact_dot_suspension = 1.0f;
    float extra_pushback = 0.0f;
    float debug_pushback_dist_bt = 0.0f;
    float wheel_radius = 0.0f;
    float suspension_rest_length = 0.0f;
    float max_suspension_travel = 12.0f;
    float suspension_stiffness = 500.0f;
    float wheels_damping_compression = 25.0f;
    float wheels_damping_relaxation = 40.0f;
    float suspension_force_scale = 1.0f;
    float wheels_suspension_force = 0.0f;
    float steer_angle = 0.0f;
    float engine_force = 0.0f;
    float brake = 0.0f;
    float lat_friction = 1.0f;
    float long_friction = 1.0f;
    float prev_steer_angle = 0.0f;
    float prev_engine_force = 0.0f;
    float prev_brake = 0.0f;
    float prev_lat_friction = 0.0f;
    float prev_long_friction = 0.0f;

    Vec3 cal_axle_dir = Vec3(0.0f, -1.0f, 0.0f);
    Vec3 cal_forward_dir = Vec3(1.0f, 0.0f, 0.0f);
    float cal_rolling_friction = 0.0f;
    float cal_lat_friction = 0.0f;
    float cal_long_friction = 0.0f;
    float cal_friction_scale = 0.0f;
    int32_t ground_car_idx = -1;
    int32_t ground_static_id = -1;
    uint8_t ground_kind = ROCKETGPU_ARENA_RAY_HIT_UNKNOWN;
    uint8_t is_front_wheel = 0;
    uint8_t is_in_contact = 0;
    uint8_t is_in_contact_with_world = 0;
};

struct StaticCollisionImpulseDebug {
    Vec3 rel_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 vel_at_contact_bt = Vec3(0.0f, 0.0f, 0.0f);
    float rel_vel_bt = 0.0f;
    Vec3 denom_basis_row0_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 denom_basis_row1_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 denom_basis_row2_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 denom_inv_inertia_local_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 denom_c0_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 denom_inv_inertia_row0_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 denom_inv_inertia_row1_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 denom_inv_inertia_row2_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 denom_angular_component_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 denom_vec_bt = Vec3(0.0f, 0.0f, 0.0f);
    float denom_inv_mass_bt = 0.0f;
    float denom_angular_dot_bt = 0.0f;
    float denom_bt = 0.0f;
    float positional_error_bt = 0.0f;
    float velocity_error_bt = 0.0f;
    float normal_impulse_bt = 0.0f;
};

struct WheelRuntimeState {
    int32_t event_world_ids[4] = {-1, -1, -1, -1};
    Vec3 event_world_normals[4] = {};
    float steer_angle[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float engine_force[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float brake[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float lat_friction[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float long_friction[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float extra_pushback[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float solver_time_step = 1.0f / 60.0f;

    Vec3 accumulated_force = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 accumulated_ang_accel = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 accumulated_torque_bt = Vec3(0.0f, 0.0f, 0.0f);

    ::rocketgpu::Quat pre_solve_rot = ::rocketgpu::Quat(1.0f, 0.0f, 0.0f, 0.0f);

    Vec3 pre_solve_basis_forward = Vec3(1.0f, 0.0f, 0.0f);
    Vec3 pre_solve_basis_right = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 pre_solve_basis_up = Vec3(0.0f, 0.0f, 1.0f);
    uint8_t pre_solve_basis_authoritative = 0;

    Vec3 cal_prev_sticky_dir = Vec3(0.0f, 0.0f, 0.0f);

    Vec3 cal_prev_wheel_normal_sum = Vec3(0.0f, 0.0f, 0.0f);
    uint8_t cal_prev_world_wheel_contacts = 0;

    uint8_t cal_game_jumped = 0;
    uint8_t cal_game_double_jumped = 0;
    uint8_t cal_cached_ground = 0;
    float cal_jump_inactive_time = 0.0f;
};

ROCKETGPU_HD Vec3 BulletBoxHalfExtentWithMargin(const CarConfig& cfg) {
    return (cfg.hitbox_size * 0.5f) - BallCarHitboxInset(cfg) + Vec3(BallCarShapeMargin(cfg), BallCarShapeMargin(cfg), BallCarShapeMargin(cfg));
}

ROCKETGPU_HD Vec3 BulletBoxHalfExtentWithMarginBt(const CarConfig& cfg) {
    constexpr float kDefaultMarginBt = 0.04f;
    constexpr float kUUToBT = 1.0f / 50.0f;
    const Vec3 half_extent_bt = (cfg.hitbox_size * kUUToBT) / 2.0f;
    const float min_half_extent_bt = Min(half_extent_bt.x, Min(half_extent_bt.y, half_extent_bt.z));
    const float safe_margin_bt = 0.1f * min_half_extent_bt;
    const float margin_bt = safe_margin_bt < kDefaultMarginBt ? safe_margin_bt : kDefaultMarginBt;

    const float implicit_margin_bt = margin_bt;
    return half_extent_bt - Vec3(implicit_margin_bt, implicit_margin_bt, implicit_margin_bt) +
        Vec3(margin_bt, margin_bt, margin_bt);
}

ROCKETGPU_HD constexpr float WheelSuspensionSubtraction() {
    return 2.5f;
}

ROCKETGPU_HD Vec3 BoxBodyInertiaDiag(const CarConfig& cfg, float mass) {
    constexpr float kBTToUU = 50.0f;
    const Vec3 half_extent = BulletBoxHalfExtentWithMarginBt(cfg) * kBTToUU;
    const float x = half_extent.x * 2.0f;
    const float y = half_extent.y * 2.0f;
    const float z = half_extent.z * 2.0f;
    const float x2 = x * x;
    const float y2 = y * y;
    const float z2 = z * z;
    const float scale = mass / 12.0f;
    return Vec3(scale * (y2 + z2), scale * (x2 + z2), scale * (x2 + y2));
}

ROCKETGPU_HD Vec3 ApplyInverseBodyInertiaWorld(const CarState& car, const CarConfig& cfg, float mass, const Vec3& torque_world) {
    const Vec3 inertia = BoxBodyInertiaDiag(cfg, mass);

    Vec3 forward_m, right_m, up_m;
    CarBasisColumns(car, forward_m, right_m, up_m);
    const Vec3 row0(forward_m.x, right_m.x, up_m.x);
    const Vec3 row1(forward_m.y, right_m.y, up_m.y);
    const Vec3 row2(forward_m.z, right_m.z, up_m.z);
    const Vec3 inv_inertia_local(
        inertia.x != 0.0f ? 1.0f / inertia.x : 0.0f,
        inertia.y != 0.0f ? 1.0f / inertia.y : 0.0f,
        inertia.z != 0.0f ? 1.0f / inertia.z : 0.0f);
    const float s00 = row0.x * inv_inertia_local.x;
    const float s01 = row0.y * inv_inertia_local.y;
    const float s02 = row0.z * inv_inertia_local.z;
    const float s10 = row1.x * inv_inertia_local.x;
    const float s11 = row1.y * inv_inertia_local.y;
    const float s12 = row1.z * inv_inertia_local.z;
    const float s20 = row2.x * inv_inertia_local.x;
    const float s21 = row2.y * inv_inertia_local.y;
    const float s22 = row2.z * inv_inertia_local.z;
    const Vec3 inv_inertia_world_row0(
        s00 * row0.x + s01 * row0.y + s02 * row0.z,
        s00 * row1.x + s01 * row1.y + s02 * row1.z,
        s00 * row2.x + s01 * row2.y + s02 * row2.z);
    const Vec3 inv_inertia_world_row1(
        s10 * row0.x + s11 * row0.y + s12 * row0.z,
        s10 * row1.x + s11 * row1.y + s12 * row1.z,
        s10 * row2.x + s11 * row2.y + s12 * row2.z);
    const Vec3 inv_inertia_world_row2(
        s20 * row0.x + s21 * row0.y + s22 * row0.z,
        s20 * row1.x + s21 * row1.y + s22 * row1.z,
        s20 * row2.x + s21 * row2.y + s22 * row2.z);
    return Vec3(
        Dot(inv_inertia_world_row0, torque_world),
        Dot(inv_inertia_world_row1, torque_world),
        Dot(inv_inertia_world_row2, torque_world));
}

ROCKETGPU_HD void ApplyCentralForce(CarState& car, float mass, const Vec3& force_world, float dt) {
    SetCarLinearVelocityUu(car, car.vel + force_world * (dt / mass));
}

ROCKETGPU_HD constexpr float UnitsUUToBT() {
    return 1.0f / 50.0f;
}

ROCKETGPU_HD constexpr float UnitsBTToUU() {
    return 50.0f;
}

ROCKETGPU_HD Vec3 ToBtLinearVec(const Vec3& value_uu) {
    return value_uu * UnitsUUToBT();
}

ROCKETGPU_HD Vec3 FromBtLinearVec(const Vec3& value_bt) {
    return value_bt * UnitsBTToUU();
}

ROCKETGPU_HD Vec3 BoxBodyInertiaDiagBt(const CarConfig& cfg, float mass) {
    const Vec3 half_extent = BulletBoxHalfExtentWithMarginBt(cfg);
    const float x = half_extent.x * 2.0f;
    const float y = half_extent.y * 2.0f;
    const float z = half_extent.z * 2.0f;
    const float x2 = x * x;
    const float y2 = y * y;
    const float z2 = z * z;
    const float scale = mass / 12.0f;
    return Vec3(scale * (y2 + z2), scale * (x2 + z2), scale * (x2 + y2));
}

ROCKETGPU_HD Vec3 ApplyInverseBodyInertiaWorldBtRowDot(
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    const Vec3& torque_world_bt) {
    Vec3 forward, right, up;
    CarBasisColumns(car, forward, right, up);
    const Vec3 row0(forward.x, right.x, up.x);
    const Vec3 row1(forward.y, right.y, up.y);
    const Vec3 row2(forward.z, right.z, up.z);
    const Vec3 inertia_bt = BoxBodyInertiaDiagBt(cfg, mass);
    const Vec3 inv_inertia_local(
        inertia_bt.x != 0.0f ? 1.0f / inertia_bt.x : 0.0f,
        inertia_bt.y != 0.0f ? 1.0f / inertia_bt.y : 0.0f,
        inertia_bt.z != 0.0f ? 1.0f / inertia_bt.z : 0.0f);

    const float s00 = row0.x * inv_inertia_local.x;
    const float s01 = row0.y * inv_inertia_local.y;
    const float s02 = row0.z * inv_inertia_local.z;
    const float s10 = row1.x * inv_inertia_local.x;
    const float s11 = row1.y * inv_inertia_local.y;
    const float s12 = row1.z * inv_inertia_local.z;
    const float s20 = row2.x * inv_inertia_local.x;
    const float s21 = row2.y * inv_inertia_local.y;
    const float s22 = row2.z * inv_inertia_local.z;
    const Vec3 inv_inertia_world_row0(
        s00 * row0.x + s01 * row0.y + s02 * row0.z,
        s00 * row1.x + s01 * row1.y + s02 * row1.z,
        s00 * row2.x + s01 * row2.y + s02 * row2.z);
    const Vec3 inv_inertia_world_row1(
        s10 * row0.x + s11 * row0.y + s12 * row0.z,
        s10 * row1.x + s11 * row1.y + s12 * row1.z,
        s10 * row2.x + s11 * row2.y + s12 * row2.z);
    const Vec3 inv_inertia_world_row2(
        s20 * row0.x + s21 * row0.y + s22 * row0.z,
        s20 * row1.x + s21 * row1.y + s22 * row1.z,
        s20 * row2.x + s21 * row2.y + s22 * row2.z);
    return Vec3(
        Dot(inv_inertia_world_row0, torque_world_bt),
        Dot(inv_inertia_world_row1, torque_world_bt),
        Dot(inv_inertia_world_row2, torque_world_bt));
}

inline void PopulateCarWorldVelocityTraceInertia(
    CarWorldVelocityTraceInputs& trace,
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    const Vec3& rel_contact,
    const Vec3& normal) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    const Vec3 rel_contact_bt =
        trace.has_rel_contact_bt ? trace.rel_contact_bt : rel_contact * kUUToBT;
    trace.torque_axis0_bt = Cross(rel_contact_bt, normal);

    Vec3 forward, right, up;
    CarBasisColumns(car, forward, right, up);
    trace.bodyA_basis_row0 = Vec3(forward.x, right.x, up.x);
    trace.bodyA_basis_row1 = Vec3(forward.y, right.y, up.y);
    trace.bodyA_basis_row2 = Vec3(forward.z, right.z, up.z);

    const Vec3 inertia_bt = BoxBodyInertiaDiagBt(cfg, mass);
    trace.bodyA_inv_inertia_local = Vec3(
        inertia_bt.x != 0.0f ? 1.0f / inertia_bt.x : 0.0f,
        inertia_bt.y != 0.0f ? 1.0f / inertia_bt.y : 0.0f,
        inertia_bt.z != 0.0f ? 1.0f / inertia_bt.z : 0.0f);

    const float b00 = trace.bodyA_basis_row0.x;
    const float b01 = trace.bodyA_basis_row0.y;
    const float b02 = trace.bodyA_basis_row0.z;
    const float b10 = trace.bodyA_basis_row1.x;
    const float b11 = trace.bodyA_basis_row1.y;
    const float b12 = trace.bodyA_basis_row1.z;
    const float b20 = trace.bodyA_basis_row2.x;
    const float b21 = trace.bodyA_basis_row2.y;
    const float b22 = trace.bodyA_basis_row2.z;
    const float s00 = b00 * trace.bodyA_inv_inertia_local.x;
    const float s01 = b01 * trace.bodyA_inv_inertia_local.y;
    const float s02 = b02 * trace.bodyA_inv_inertia_local.z;
    const float s10 = b10 * trace.bodyA_inv_inertia_local.x;
    const float s11 = b11 * trace.bodyA_inv_inertia_local.y;
    const float s12 = b12 * trace.bodyA_inv_inertia_local.z;
    const float s20 = b20 * trace.bodyA_inv_inertia_local.x;
    const float s21 = b21 * trace.bodyA_inv_inertia_local.y;
    const float s22 = b22 * trace.bodyA_inv_inertia_local.z;
    trace.bodyA_inv_inertia_world_row0 = Vec3(s00 * b00 + s01 * b01 + s02 * b02, s00 * b10 + s01 * b11 + s02 * b12, s00 * b20 + s01 * b21 + s02 * b22);
    trace.bodyA_inv_inertia_world_row1 = Vec3(s10 * b00 + s11 * b01 + s12 * b02, s10 * b10 + s11 * b11 + s12 * b12, s10 * b20 + s11 * b21 + s12 * b22);
    trace.bodyA_inv_inertia_world_row2 = Vec3(s20 * b00 + s21 * b01 + s22 * b02, s20 * b10 + s21 * b11 + s22 * b12, s20 * b20 + s21 * b21 + s22 * b22);
    trace.angular_component_a_pre_factor = Vec3(
        Dot(trace.bodyA_inv_inertia_world_row0, trace.torque_axis0_bt),
        Dot(trace.bodyA_inv_inertia_world_row1, trace.torque_axis0_bt),
        Dot(trace.bodyA_inv_inertia_world_row2, trace.torque_axis0_bt));
    trace.bodyA_angular_factor = Vec3(1.0f, 1.0f, 1.0f);
}

ROCKETGPU_HD Vec3 ApplyInverseBodyInertiaWorldBt(
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    const Vec3& torque_world_bt) {
    return ApplyInverseBodyInertiaWorldBtRowDot(car, cfg, mass, torque_world_bt);
}

ROCKETGPU_HD Vec3 ApplyInverseBodyInertiaWorldBtColumnDot(
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    const Vec3& v) {
    Vec3 forward, right, up;
    CarBasisColumns(car, forward, right, up);
    const Vec3 row0(forward.x, right.x, up.x);
    const Vec3 row1(forward.y, right.y, up.y);
    const Vec3 row2(forward.z, right.z, up.z);
    const Vec3 inertia_bt = BoxBodyInertiaDiagBt(cfg, mass);
    const Vec3 inv_inertia_local(
        inertia_bt.x != 0.0f ? 1.0f / inertia_bt.x : 0.0f,
        inertia_bt.y != 0.0f ? 1.0f / inertia_bt.y : 0.0f,
        inertia_bt.z != 0.0f ? 1.0f / inertia_bt.z : 0.0f);
    const float s00 = row0.x * inv_inertia_local.x, s01 = row0.y * inv_inertia_local.y, s02 = row0.z * inv_inertia_local.z;
    const float s10 = row1.x * inv_inertia_local.x, s11 = row1.y * inv_inertia_local.y, s12 = row1.z * inv_inertia_local.z;
    const float s20 = row2.x * inv_inertia_local.x, s21 = row2.y * inv_inertia_local.y, s22 = row2.z * inv_inertia_local.z;

    const float m00 = s00 * row0.x + s01 * row0.y + s02 * row0.z;
    const float m01 = s00 * row1.x + s01 * row1.y + s02 * row1.z;
    const float m02 = s00 * row2.x + s01 * row2.y + s02 * row2.z;
    const float m10 = s10 * row0.x + s11 * row0.y + s12 * row0.z;
    const float m11 = s10 * row1.x + s11 * row1.y + s12 * row1.z;
    const float m12 = s10 * row2.x + s11 * row2.y + s12 * row2.z;
    const float m20 = s20 * row0.x + s21 * row0.y + s22 * row0.z;
    const float m21 = s20 * row1.x + s21 * row1.y + s22 * row1.z;
    const float m22 = s20 * row2.x + s21 * row2.y + s22 * row2.z;
    return Vec3(
        m00 * v.x + m10 * v.y + m20 * v.z,
        m01 * v.x + m11 * v.y + m21 * v.z,
        m02 * v.x + m12 * v.y + m22 * v.z);
}

ROCKETGPU_HD Vec3 ApplyCarWorldContactAngularComponentBt(
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    const Vec3& rel_contact_bt,
    const Vec3& normal) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    return ApplyInverseBodyInertiaWorldBt(
        car,
        cfg,
        mass,
        Cross(rel_contact_bt, normal)) * kUUToBT;
}

ROCKETGPU_HD Vec3 CarWorldContactAngularComponentBtNative(
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    const Vec3& rel_contact_bt,
    const Vec3& normal) {
    return ApplyInverseBodyInertiaWorldBt(
        car,
        cfg,
        mass,
        Cross(rel_contact_bt, normal));
}

ROCKETGPU_HD Vec3 ApplyCarWorldContactAngularComponent(
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    const Vec3& rel_contact,
    const Vec3& normal) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    return ApplyCarWorldContactAngularComponentBt(
        car,
        cfg,
        mass,
        rel_contact * kUUToBT,
        normal);
}

ROCKETGPU_HD float BodyImpulseDenominatorBt(
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    const Vec3& pos_world_bt,
    const Vec3& normal_world,
    StaticCollisionImpulseDebug* debug = nullptr) {
    const Vec3 rel_bt = pos_world_bt - car.pos;
    const Vec3 c0_bt = Cross(rel_bt, normal_world);

    const Vec3 angular_component_bt = ApplyInverseBodyInertiaWorldBtColumnDot(car, cfg, mass, c0_bt);
    const Vec3 vec_bt = Cross(angular_component_bt, rel_bt);
    const float angular_dot_bt = Dot(normal_world, vec_bt);
    const float inv_mass_bt = 1.0f / mass;
    const float denom_bt = inv_mass_bt + angular_dot_bt;
    if (debug) {
        Vec3 forward, right, up;
        CarBasisColumns(car, forward, right, up);
        const Vec3 row0(forward.x, right.x, up.x);
        const Vec3 row1(forward.y, right.y, up.y);
        const Vec3 row2(forward.z, right.z, up.z);
        const Vec3 inertia_bt = BoxBodyInertiaDiagBt(cfg, mass);
        const Vec3 inv_inertia_local(
            inertia_bt.x != 0.0f ? 1.0f / inertia_bt.x : 0.0f,
            inertia_bt.y != 0.0f ? 1.0f / inertia_bt.y : 0.0f,
            inertia_bt.z != 0.0f ? 1.0f / inertia_bt.z : 0.0f);
        const float s00 = row0.x * inv_inertia_local.x;
        const float s01 = row0.y * inv_inertia_local.y;
        const float s02 = row0.z * inv_inertia_local.z;
        const float s10 = row1.x * inv_inertia_local.x;
        const float s11 = row1.y * inv_inertia_local.y;
        const float s12 = row1.z * inv_inertia_local.z;
        const float s20 = row2.x * inv_inertia_local.x;
        const float s21 = row2.y * inv_inertia_local.y;
        const float s22 = row2.z * inv_inertia_local.z;
        debug->denom_basis_row0_bt = row0;
        debug->denom_basis_row1_bt = row1;
        debug->denom_basis_row2_bt = row2;
        debug->denom_inv_inertia_local_bt = inv_inertia_local;
        debug->denom_c0_bt = c0_bt;
        debug->denom_inv_inertia_row0_bt = Vec3(
            s00 * row0.x + s01 * row0.y + s02 * row0.z,
            s00 * row1.x + s01 * row1.y + s02 * row1.z,
            s00 * row2.x + s01 * row2.y + s02 * row2.z);
        debug->denom_inv_inertia_row1_bt = Vec3(
            s10 * row0.x + s11 * row0.y + s12 * row0.z,
            s10 * row1.x + s11 * row1.y + s12 * row1.z,
            s10 * row2.x + s11 * row2.y + s12 * row2.z);
        debug->denom_inv_inertia_row2_bt = Vec3(
            s20 * row0.x + s21 * row0.y + s22 * row0.z,
            s20 * row1.x + s21 * row1.y + s22 * row1.z,
            s20 * row2.x + s21 * row2.y + s22 * row2.z);
        debug->denom_angular_component_bt = angular_component_bt;
        debug->denom_vec_bt = vec_bt;
        debug->denom_inv_mass_bt = inv_mass_bt;
        debug->denom_angular_dot_bt = angular_dot_bt;
    }
    return denom_bt;
}

ROCKETGPU_HD void ApplyImpulseAtPointBtRoundtripBtInputs(
    CarState& car,
    const CarConfig& cfg,
    float mass,
    const Vec3& impulse_world_bt,
    const Vec3& rel_point_world_bt);

ROCKETGPU_HD void ApplyImpulseAtPointBtRoundtripBtInputs(
    CarState& car,
    const CarConfig& cfg,
    float mass,
    const Vec3& impulse_world_bt,
    const Vec3& rel_point_world_bt) {
    const float official_inv_mass = 1.0f / mass;
    SetCarLinearVelocityBt(car, car.vel_bt + impulse_world_bt * official_inv_mass);

    car.vel_touched_solve_stage = 1;

    const Vec3 torque_bt = Cross(rel_point_world_bt, impulse_world_bt);
    car.ang_vel += ApplyInverseBodyInertiaWorldBtRowDot(car, cfg, mass, torque_bt);
}

ROCKETGPU_HD float ResolveStaticCollisionImpulseMagnitudeBtInputs(
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    const Vec3& contact_pos_world_bt,
    const Vec3& contact_normal_world,
    float distance_bt,
    float dt,
    StaticCollisionImpulseDebug* debug = nullptr) {
    if (debug) {
        *debug = StaticCollisionImpulseDebug{};
    }
    CarState car_bt = car;

    car_bt.pos = car.pos_bt;

    car_bt.vel = car.vel_bt;
    const Vec3 rel_bt = contact_pos_world_bt - car_bt.pos;
    const Vec3 vel_at_contact_bt = VelocityAtPoint(car_bt, rel_bt);
    const float rel_vel_bt = Dot(contact_normal_world, vel_at_contact_bt);
    const float denom_bt = BodyImpulseDenominatorBt(car_bt, cfg, mass, contact_pos_world_bt, contact_normal_world, debug);
    if (debug) {
        debug->rel_bt = rel_bt;
        debug->vel_at_contact_bt = vel_at_contact_bt;
        debug->rel_vel_bt = rel_vel_bt;
        debug->denom_bt = denom_bt;
    }
    if (denom_bt <= 0.0f || dt <= 0.0f) {
        return 0.0f;
    }
    const float jac_diag_ab_inv = 1.0f / denom_bt;
    const float positional_error_bt = 0.2f * -distance_bt / dt;
    const float velocity_error_bt = -rel_vel_bt;
    const float penetration_impulse = positional_error_bt * jac_diag_ab_inv;
    const float velocity_impulse = velocity_error_bt * jac_diag_ab_inv;
    const float normal_impulse_bt = penetration_impulse + velocity_impulse;
    if (debug) {
        debug->positional_error_bt = positional_error_bt;
        debug->velocity_error_bt = velocity_error_bt;
        debug->normal_impulse_bt = normal_impulse_bt > 0.0f ? normal_impulse_bt : 0.0f;
    }
    return normal_impulse_bt > 0.0f ? normal_impulse_bt : 0.0f;
}

// v3 contact_constraint.rs resolve_single_collision with a dynamic body2 (a wheel ray that hit another car):
// the relative normal velocity and the jacobian diagonal include the hit car.
ROCKETGPU_HD float ResolveCarCarWheelPushbackMagnitudeBtInputs(
    const CarState& car,
    const CarConfig& cfg,
    const CarState& other,
    const CarConfig& other_cfg,
    float mass,
    const Vec3& contact_pos_world_bt,
    const Vec3& contact_normal_world,
    float distance_bt,
    float dt) {
    CarState a_bt = car;
    a_bt.pos = car.pos_bt;
    a_bt.vel = car.vel_bt;
    CarState b_bt = other;
    b_bt.pos = other.pos_bt;
    b_bt.vel = other.vel_bt;
    const Vec3 vel1 = VelocityAtPoint(a_bt, contact_pos_world_bt - a_bt.pos);
    const Vec3 vel2 = VelocityAtPoint(b_bt, contact_pos_world_bt - b_bt.pos);
    const float rel_vel_bt = Dot(contact_normal_world, vel1 - vel2);
    const float denom0 = BodyImpulseDenominatorBt(a_bt, cfg, mass, contact_pos_world_bt, contact_normal_world, nullptr);
    const float denom1 = BodyImpulseDenominatorBt(b_bt, other_cfg, mass, contact_pos_world_bt, contact_normal_world, nullptr);
    if (dt <= 0.0f || denom0 + denom1 <= 0.0f) {
        return 0.0f;
    }
    const float jac_diag_ab_inv = 1.0f / (denom0 + denom1);
    const float penetration_impulse = (0.2f * -distance_bt / dt) * jac_diag_ab_inv;
    const float velocity_impulse = -rel_vel_bt * jac_diag_ab_inv;
    const float normal_impulse_bt = penetration_impulse + velocity_impulse;
    return normal_impulse_bt > 0.0f ? normal_impulse_bt : 0.0f;
}

ROCKETGPU_HD void ApplyCentralImpulseViaBtRoundtrip(CarState& car, const Vec3& delta_vel_world_uu, float car_mass) {
    if (!car.vel_touched_solve_stage && !car.is_on_ground) {
        const Vec3 impulse_bt = ToBtLinearVec(delta_vel_world_uu) * car_mass;
        const float inv_mass = 1.0f / car_mass;
        SetCarLinearVelocityBt(car, car.vel_bt + impulse_bt * inv_mass);
        return;
    }

    const Vec3 impulse_bt = ToBtLinearVec(delta_vel_world_uu) * car_mass;
    const float inv_mass = 1.0f / car_mass;
    const bool was_touched = car.vel_touched_solve_stage != 0;
    SetCarLinearVelocityBt(car, car.vel_bt + impulse_bt * inv_mass);
    if (was_touched || car.is_on_ground) {
        car.vel_touched_solve_stage = 1;
    }
}

ROCKETGPU_HD void ApplyAngularAccelWorld(CarState& car, const Vec3& ang_accel_world, float dt) {
    car.ang_vel += ang_accel_world * dt;
}

ROCKETGPU_HD void BuildInvInertiaTensorWorld(
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    Vec3& out_row0,
    Vec3& out_row1,
    Vec3& out_row2) {
    const Vec3 inertia_local = BoxBodyInertiaDiag(cfg, mass);
    const Vec3 inv_inertia_local(
        inertia_local.x != 0.0f ? (1.0f / inertia_local.x) : 0.0f,
        inertia_local.y != 0.0f ? (1.0f / inertia_local.y) : 0.0f,
        inertia_local.z != 0.0f ? (1.0f / inertia_local.z) : 0.0f);

    Vec3 forward_col(1.0f, 0.0f, 0.0f);
    Vec3 right_col(0.0f, 1.0f, 0.0f);
    Vec3 up_col(0.0f, 0.0f, 1.0f);
    CarBasisColumns(car, forward_col, right_col, up_col);

    const float b00 = forward_col.x, b01 = right_col.x, b02 = up_col.x;
    const float b10 = forward_col.y, b11 = right_col.y, b12 = up_col.y;
    const float b20 = forward_col.z, b21 = right_col.z, b22 = up_col.z;

    const float s00 = b00 * inv_inertia_local.x, s01 = b01 * inv_inertia_local.y, s02 = b02 * inv_inertia_local.z;
    const float s10 = b10 * inv_inertia_local.x, s11 = b11 * inv_inertia_local.y, s12 = b12 * inv_inertia_local.z;
    const float s20 = b20 * inv_inertia_local.x, s21 = b21 * inv_inertia_local.y, s22 = b22 * inv_inertia_local.z;

    out_row0 = Vec3(
        s00 * b00 + s01 * b01 + s02 * b02,
        s00 * b10 + s01 * b11 + s02 * b12,
        s00 * b20 + s01 * b21 + s02 * b22);
    out_row1 = Vec3(
        s10 * b00 + s11 * b01 + s12 * b02,
        s10 * b10 + s11 * b11 + s12 * b12,
        s10 * b20 + s11 * b21 + s12 * b22);
    out_row2 = Vec3(
        s20 * b00 + s21 * b01 + s22 * b02,
        s20 * b10 + s21 * b11 + s22 * b12,
        s20 * b20 + s21 * b21 + s22 * b22);
}

ROCKETGPU_HD void InverseMatrix3x3BulletStyle(
    const Vec3& r0,
    const Vec3& r1,
    const Vec3& r2,
    Vec3& out_row0,
    Vec3& out_row1,
    Vec3& out_row2) {
    auto elem = [](const Vec3& r, int c) -> float {
        return c == 0 ? r.x : (c == 1 ? r.y : r.z);
    };
    auto cofac = [&](int row1, int col1, int row2, int col2) -> float {
        const Vec3& mr1 = row1 == 0 ? r0 : (row1 == 1 ? r1 : r2);
        const Vec3& mr2 = row2 == 0 ? r0 : (row2 == 1 ? r1 : r2);
        return elem(mr1, col1) * elem(mr2, col2) - elem(mr1, col2) * elem(mr2, col1);
    };

    const float co_x = cofac(1, 1, 2, 2);
    const float co_y = cofac(1, 2, 2, 0);
    const float co_z = cofac(1, 0, 2, 1);
    const float det = r0.x * co_x + r0.y * co_y + r0.z * co_z;
    const float s = (det != 0.0f) ? (1.0f / det) : 0.0f;
    out_row0 = Vec3(co_x * s, cofac(0, 2, 2, 1) * s, cofac(0, 1, 1, 2) * s);
    out_row1 = Vec3(co_y * s, cofac(0, 0, 2, 2) * s, cofac(0, 2, 1, 0) * s);
    out_row2 = Vec3(co_z * s, cofac(0, 1, 2, 0) * s, cofac(0, 0, 1, 1) * s);
}

ROCKETGPU_HD Vec3 Mul3x3Vec(const Vec3& row0, const Vec3& row1, const Vec3& row2, const Vec3& v) {
    return Vec3(Dot(row0, v), Dot(row1, v), Dot(row2, v));
}

ROCKETGPU_HD Vec3 VecMul3x3(const Vec3& v, const Vec3& row0, const Vec3& row1, const Vec3& row2) {
    return Vec3(
        row0.x * v.x + row1.x * v.y + row2.x * v.z,
        row0.y * v.x + row1.y * v.y + row2.y * v.z,
        row0.z * v.x + row1.z * v.y + row2.z * v.z);
}

ROCKETGPU_HD void BuildInvInertiaTensorWorldBtFromQuat(
    const ::rocketgpu::Quat& rot,
    const CarConfig& cfg,
    float mass,
    Vec3& out_row0,
    Vec3& out_row1,
    Vec3& out_row2);

ROCKETGPU_HD void BuildInvInertiaTensorWorldBtFromColumns(
    const Vec3& forward_col,
    const Vec3& right_col,
    const Vec3& up_col,
    const CarConfig& cfg,
    float mass,
    Vec3& out_row0,
    Vec3& out_row1,
    Vec3& out_row2);

ROCKETGPU_HD void BuildInvInertiaTensorWorldBt(
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    Vec3& out_row0,
    Vec3& out_row1,
    Vec3& out_row2) {
    if (car.basis_authoritative) {
        BuildInvInertiaTensorWorldBtFromColumns(
            car.basis_forward, car.basis_right, car.basis_up, cfg, mass, out_row0, out_row1, out_row2);
        return;
    }
    BuildInvInertiaTensorWorldBtFromQuat(car.rot, cfg, mass, out_row0, out_row1, out_row2);
}

ROCKETGPU_HD void BuildPreSolveInvInertiaTensorWorldBt(
    const WheelRuntimeState& runtime,
    const CarConfig& cfg,
    float mass,
    Vec3& out_row0,
    Vec3& out_row1,
    Vec3& out_row2) {
    if (runtime.pre_solve_basis_authoritative) {
        BuildInvInertiaTensorWorldBtFromColumns(
            runtime.pre_solve_basis_forward, runtime.pre_solve_basis_right, runtime.pre_solve_basis_up,
            cfg, mass, out_row0, out_row1, out_row2);
        return;
    }
    BuildInvInertiaTensorWorldBtFromQuat(runtime.pre_solve_rot, cfg, mass, out_row0, out_row1, out_row2);
}

ROCKETGPU_HD void BuildInvInertiaTensorWorldBtFromColumns(
    const Vec3& forward_col,
    const Vec3& right_col,
    const Vec3& up_col,
    const CarConfig& cfg,
    float mass,
    Vec3& out_row0,
    Vec3& out_row1,
    Vec3& out_row2) {
    const Vec3 inertia_local = BoxBodyInertiaDiagBt(cfg, mass);
    const Vec3 inv_inertia_local(
        inertia_local.x != 0.0f ? (1.0f / inertia_local.x) : 0.0f,
        inertia_local.y != 0.0f ? (1.0f / inertia_local.y) : 0.0f,
        inertia_local.z != 0.0f ? (1.0f / inertia_local.z) : 0.0f);

    const float b00 = forward_col.x, b01 = right_col.x, b02 = up_col.x;
    const float b10 = forward_col.y, b11 = right_col.y, b12 = up_col.y;
    const float b20 = forward_col.z, b21 = right_col.z, b22 = up_col.z;
    const float s00 = b00 * inv_inertia_local.x, s01 = b01 * inv_inertia_local.y, s02 = b02 * inv_inertia_local.z;
    const float s10 = b10 * inv_inertia_local.x, s11 = b11 * inv_inertia_local.y, s12 = b12 * inv_inertia_local.z;
    const float s20 = b20 * inv_inertia_local.x, s21 = b21 * inv_inertia_local.y, s22 = b22 * inv_inertia_local.z;
    out_row0 = Vec3(s00 * b00 + s01 * b01 + s02 * b02,
                    s00 * b10 + s01 * b11 + s02 * b12,
                    s00 * b20 + s01 * b21 + s02 * b22);
    out_row1 = Vec3(s10 * b00 + s11 * b01 + s12 * b02,
                    s10 * b10 + s11 * b11 + s12 * b12,
                    s10 * b20 + s11 * b21 + s12 * b22);
    out_row2 = Vec3(s20 * b00 + s21 * b01 + s22 * b02,
                    s20 * b10 + s21 * b11 + s22 * b12,
                    s20 * b20 + s21 * b21 + s22 * b22);
}
ROCKETGPU_HD void BuildInvInertiaTensorWorldBtFromQuat(
    const ::rocketgpu::Quat& rot,
    const CarConfig& cfg,
    float mass,
    Vec3& out_row0,
    Vec3& out_row1,
    Vec3& out_row2) {
    Vec3 forward_col(1.0f, 0.0f, 0.0f);
    Vec3 right_col(0.0f, 1.0f, 0.0f);
    Vec3 up_col(0.0f, 0.0f, 1.0f);
    GetBasisVectors(rot, forward_col, right_col, up_col);
    BuildInvInertiaTensorWorldBtFromColumns(forward_col, right_col, up_col, cfg, mass, out_row0, out_row1, out_row2);
}

ROCKETGPU_HD Vec3 ComputeBulletStyleExternalTorqueInput(
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    const Vec3& ang_accel_world) {
    Vec3 inv_r0, inv_r1, inv_r2;
    BuildInvInertiaTensorWorld(car, cfg, mass, inv_r0, inv_r1, inv_r2);
    Vec3 i_r0, i_r1, i_r2;
    InverseMatrix3x3BulletStyle(inv_r0, inv_r1, inv_r2, i_r0, i_r1, i_r2);
    return Mul3x3Vec(i_r0, i_r1, i_r2, ang_accel_world);
}

ROCKETGPU_HD void QueueCentralForce(WheelRuntimeState* wheel_runtime_state, const Vec3& force_bt) {
    if (!wheel_runtime_state) {
        return;
    }
    wheel_runtime_state->accumulated_force += force_bt;
}

ROCKETGPU_HD void QueueAngularAccelWorld(
    WheelRuntimeState* wheel_runtime_state,
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    const Vec3& ang_accel_world) {
    if (!wheel_runtime_state) {
        return;
    }
    wheel_runtime_state->accumulated_ang_accel +=
        ComputeBulletStyleExternalTorqueInput(car, cfg, mass, ang_accel_world);
}

ROCKETGPU_HD void QueueOfficialTorqueBt(WheelRuntimeState* wheel_runtime_state, const Vec3& torque_bt) {
    if (!wheel_runtime_state) {
        return;
    }
    wheel_runtime_state->accumulated_torque_bt += torque_bt;
}

ROCKETGPU_HD void MatMul3x3BulletStyle(
    const Vec3& a0, const Vec3& a1, const Vec3& a2,
    const Vec3& b0, const Vec3& b1, const Vec3& b2,
    Vec3& o0, Vec3& o1, Vec3& o2) {
    o0 = Vec3(a0.x * b0.x + a0.y * b1.x + a0.z * b2.x,
              a0.x * b0.y + a0.y * b1.y + a0.z * b2.y,
              a0.x * b0.z + a0.y * b1.z + a0.z * b2.z);
    o1 = Vec3(a1.x * b0.x + a1.y * b1.x + a1.z * b2.x,
              a1.x * b0.y + a1.y * b1.y + a1.z * b2.y,
              a1.x * b0.z + a1.y * b1.z + a1.z * b2.z);
    o2 = Vec3(a2.x * b0.x + a2.y * b1.x + a2.z * b2.x,
              a2.x * b0.y + a2.y * b1.y + a2.z * b2.y,
              a2.x * b0.z + a2.y * b1.z + a2.z * b2.z);
}

ROCKETGPU_HD void BuildInertiaTensorWorldBtInverse(
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    Vec3& out_row0,
    Vec3& out_row1,
    Vec3& out_row2) {
    Vec3 inv_r0, inv_r1, inv_r2;
    BuildInvInertiaTensorWorldBt(car, cfg, mass, inv_r0, inv_r1, inv_r2);
    InverseMatrix3x3BulletStyle(inv_r0, inv_r1, inv_r2, out_row0, out_row1, out_row2);
}

ROCKETGPU_HD Vec3 VelocityAtPoint(const CarState& car, const Vec3& rel_point_world) {
    return car.vel + Cross(car.ang_vel, rel_point_world);
}

ROCKETGPU_HD Vec3 VelocityAtPoint(
    const CarState& car,
    const Vec3& rel_point_world,
    const Vec3& external_linear_delta,
    const Vec3& external_angular_delta) {
    return car.vel + external_linear_delta + Cross(car.ang_vel + external_angular_delta, rel_point_world);
}

ROCKETGPU_HD Vec3 ComputeDeferredLinearVelocityDelta(
    const CarState& car,
    const WheelRuntimeState* wheel_runtime_state,
    const MutatorConfig& mutator,
    float dt) {
    if (car.is_demoed || car.just_respawned) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }

    Vec3 delta(0.0f, 0.0f, mutator.gravity_z * dt);
    if (wheel_runtime_state) {
        delta += (wheel_runtime_state->accumulated_force * (dt / mutator.car_mass)) * UnitsBTToUU();
    }
    return delta;
}

ROCKETGPU_HD Vec3 ComputeDeferredAngularVelocityDelta(
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    const WheelRuntimeState* wheel_runtime_state,
    float dt) {
    if (car.is_demoed || car.just_respawned || !wheel_runtime_state) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }
    Vec3 inv_r0, inv_r1, inv_r2;
    BuildInvInertiaTensorWorld(car, cfg, mass, inv_r0, inv_r1, inv_r2);

    const Vec3 angular_accel = VecMul3x3(wheel_runtime_state->accumulated_ang_accel, inv_r0, inv_r1, inv_r2);
    return angular_accel * dt;
}

ROCKETGPU_HD Vec3 ComputeOfficialExternalForceImpulseBt(
    const WheelRuntimeState* wheel_runtime_state,
    const MutatorConfig& mutator,
    float dt) {
    const float inv_mass = 1.0f / mutator.car_mass;

    const Vec3 gravity_force_bt(
        0.0f, 0.0f, (mutator.gravity_z * UnitsUUToBT()) * mutator.car_mass);
    Vec3 total_force_bt = gravity_force_bt;
    if (wheel_runtime_state) {
        total_force_bt = wheel_runtime_state->accumulated_force + gravity_force_bt;
    }
    return (total_force_bt * inv_mass) * dt;
}

ROCKETGPU_HD void ApplyDeferredVelocityDeltas(
    CarState& car,
    const CarConfig& cfg,
    const Vec3& deferred_linear_delta,
    const Vec3& deferred_angular_delta,
    const WheelRuntimeState* wheel_runtime_state,
    const MutatorConfig& mutator,
    float dt) {
    if (car.just_respawned) {
        return;
    }

#if !defined(__CUDA_ARCH__)
    {
        Vec3 probe_ext_torque(0.0f, 0.0f, 0.0f);
        if (wheel_runtime_state) {
            Vec3 probe_inv_r0, probe_inv_r1, probe_inv_r2;
            BuildPreSolveInvInertiaTensorWorldBt(
                *wheel_runtime_state, cfg, mutator.car_mass, probe_inv_r0, probe_inv_r1, probe_inv_r2);
            probe_ext_torque = VecMul3x3(
                wheel_runtime_state->accumulated_torque_bt,
                probe_inv_r0, probe_inv_r1, probe_inv_r2) * dt;
        }
        if (wheel_runtime_state) {
            Vec3 probe_r0, probe_r1, probe_r2;
            BuildPreSolveInvInertiaTensorWorldBt(
                *wheel_runtime_state, cfg, mutator.car_mass, probe_r0, probe_r1, probe_r2);
        }
    }
#endif

    (void)car.vel_touched_solve_stage;
    {
        const Vec3 ext_f_bt = ComputeOfficialExternalForceImpulseBt(wheel_runtime_state, mutator, dt);
        SetCarLinearVelocityBt(car, car.vel_bt + ext_f_bt);
        {
            Vec3 extT(0.0f, 0.0f, 0.0f);
            if (wheel_runtime_state) {
                Vec3 inv_r0, inv_r1, inv_r2;
                BuildPreSolveInvInertiaTensorWorldBt(
                    *wheel_runtime_state, cfg, mutator.car_mass, inv_r0, inv_r1, inv_r2);
                extT = VecMul3x3(wheel_runtime_state->accumulated_torque_bt, inv_r0, inv_r1, inv_r2) * dt;
            }
            car.ang_vel += extT;
        }
        return;
    }
}

ROCKETGPU_HD void ApplyImpulseAtPoint(CarState& car, const CarConfig& cfg, float mass, const Vec3& impulse_world, const Vec3& rel_point_world) {
    SetCarLinearVelocityUu(car, car.vel + impulse_world / mass);
    const Vec3 torque_impulse = Cross(rel_point_world, impulse_world);
    car.ang_vel += ApplyInverseBodyInertiaWorld(car, cfg, mass, torque_impulse);
}

ROCKETGPU_HD float SphereBodyInertia(float mass, float radius) {
    return 0.4f * mass * radius * radius;
}

ROCKETGPU_HD Vec3 ApplyInverseBallInertiaWorld(float mass, float radius, const Vec3& torque_world) {
    const float inertia = SphereBodyInertia(mass, radius);
    return inertia > 0.0f ? (torque_world / inertia) : Vec3(0.0f, 0.0f, 0.0f);
}

ROCKETGPU_HD Vec3 ApplyInverseBallInertiaWorldBt(
    const ::rocketgpu::Quat& ball_rot,
    float mass,
    float radius_uu,
    const Vec3& torque_bt) {
    Vec3 forward_col(1.0f, 0.0f, 0.0f);
    Vec3 right_col(0.0f, 1.0f, 0.0f);
    Vec3 up_col(0.0f, 0.0f, 1.0f);
    GetBasisVectors(ball_rot, forward_col, right_col, up_col);
    return ApplyInverseBallInertiaWorldBtColumns(forward_col, right_col, up_col, mass, radius_uu, torque_bt);
}

ROCKETGPU_HD Vec3 ApplyInverseBallInertiaWorldBtBall(
    const ::rocketgpu::BallState& ball,
    float mass,
    float radius_uu,
    const Vec3& torque_bt) {
    Vec3 forward_col(1.0f, 0.0f, 0.0f);
    Vec3 right_col(0.0f, 1.0f, 0.0f);
    Vec3 up_col(0.0f, 0.0f, 1.0f);
    GetBallBasisColumns(ball, forward_col, right_col, up_col);
    return ApplyInverseBallInertiaWorldBtColumns(forward_col, right_col, up_col, mass, radius_uu, torque_bt);
}
ROCKETGPU_HD Vec3 ApplyInverseBallInertiaWorldBtColumns(
    const Vec3& forward_col,
    const Vec3& right_col,
    const Vec3& up_col,
    float mass,
    float radius_uu,
    const Vec3& torque_bt) {
    const float radius_bt = radius_uu * (1.0f / 50.0f);
    const float inertia_bt = 0.4f * mass * radius_bt * radius_bt;
    const float inv_inertia = inertia_bt != 0.0f ? (1.0f / inertia_bt) : 0.0f;
    const float b00 = forward_col.x, b01 = right_col.x, b02 = up_col.x;
    const float b10 = forward_col.y, b11 = right_col.y, b12 = up_col.y;
    const float b20 = forward_col.z, b21 = right_col.z, b22 = up_col.z;
    const float s00 = b00 * inv_inertia, s01 = b01 * inv_inertia, s02 = b02 * inv_inertia;
    const float s10 = b10 * inv_inertia, s11 = b11 * inv_inertia, s12 = b12 * inv_inertia;
    const float s20 = b20 * inv_inertia, s21 = b21 * inv_inertia, s22 = b22 * inv_inertia;
    const Vec3 row0(s00 * b00 + s01 * b01 + s02 * b02,
                    s00 * b10 + s01 * b11 + s02 * b12,
                    s00 * b20 + s01 * b21 + s02 * b22);
    const Vec3 row1(s10 * b00 + s11 * b01 + s12 * b02,
                    s10 * b10 + s11 * b11 + s12 * b12,
                    s10 * b20 + s11 * b21 + s12 * b22);
    const Vec3 row2(s20 * b00 + s21 * b01 + s22 * b02,
                    s20 * b10 + s21 * b11 + s22 * b12,
                    s20 * b20 + s21 * b21 + s22 * b22);
    return Mul3x3Vec(row0, row1, row2, torque_bt);
}

ROCKETGPU_HD Vec3 ApplyInverseBallInertiaWorldBtDiagColumns(
    const Vec3& forward_col,
    const Vec3& right_col,
    const Vec3& up_col,
    const Vec3& inertia_local_bt,
    const Vec3& torque_bt) {
    const float inv_x = inertia_local_bt.x != 0.0f ? (1.0f / inertia_local_bt.x) : 0.0f;
    const float inv_y = inertia_local_bt.y != 0.0f ? (1.0f / inertia_local_bt.y) : 0.0f;
    const float inv_z = inertia_local_bt.z != 0.0f ? (1.0f / inertia_local_bt.z) : 0.0f;
    const float b00 = forward_col.x, b01 = right_col.x, b02 = up_col.x;
    const float b10 = forward_col.y, b11 = right_col.y, b12 = up_col.y;
    const float b20 = forward_col.z, b21 = right_col.z, b22 = up_col.z;
    const float s00 = b00 * inv_x, s01 = b01 * inv_y, s02 = b02 * inv_z;
    const float s10 = b10 * inv_x, s11 = b11 * inv_y, s12 = b12 * inv_z;
    const float s20 = b20 * inv_x, s21 = b21 * inv_y, s22 = b22 * inv_z;
    const Vec3 row0(s00 * b00 + s01 * b01 + s02 * b02,
                    s00 * b10 + s01 * b11 + s02 * b12,
                    s00 * b20 + s01 * b21 + s02 * b22);
    const Vec3 row1(s10 * b00 + s11 * b01 + s12 * b02,
                    s10 * b10 + s11 * b11 + s12 * b12,
                    s10 * b20 + s11 * b21 + s12 * b22);
    const Vec3 row2(s20 * b00 + s21 * b01 + s22 * b02,
                    s20 * b10 + s21 * b11 + s22 * b12,
                    s20 * b20 + s21 * b21 + s22 * b22);
    return Mul3x3Vec(row0, row1, row2, torque_bt);
}

ROCKETGPU_HD Vec3 ApplyInverseBallInertiaWorldBtMutator(
    const BallState& ball,
    const MutatorConfig& mutator,
    const Vec3& torque_bt) {
    if (mutator.ball_shape == 1) {
        Vec3 forward_col(1.0f, 0.0f, 0.0f);
        Vec3 right_col(0.0f, 1.0f, 0.0f);
        Vec3 up_col(0.0f, 0.0f, 1.0f);
        GetBallBasisColumns(ball, forward_col, right_col, up_col);
        return ApplyInverseBallInertiaWorldBtDiagColumns(
            forward_col, right_col, up_col, mutator.ball_local_inertia_bt, torque_bt);
    }
    return ApplyInverseBallInertiaWorldBtBall(ball, mutator.ball_mass, mutator.ball_radius, torque_bt);
}

ROCKETGPU_HD Vec3 BallVelocityAtPoint(const BallState& ball, const Vec3& rel_point_world) {
    return ball.vel + Cross(ball.ang_vel, rel_point_world);
}

ROCKETGPU_HD void ApplyImpulseAtPoint(BallState& ball, float mass, float radius, const Vec3& impulse_world, const Vec3& rel_point_world) {
    SetBallLinearVelocityBt(ball, ball.vel_bt + UuVecToBtDiv(impulse_world / mass));
    const Vec3 torque_impulse = Cross(rel_point_world, impulse_world);
    ball.ang_vel += ApplyInverseBallInertiaWorld(mass, radius, torque_impulse);
}

ROCKETGPU_HD void MaybeUpdateBestHitFraction(
    bool valid,
    float fraction,
    float max_len,
    const Vec3& point,
    const Vec3& point_bt,
    const Vec3& normal,
    ArenaRayHit& best,
    uint8_t kind,
    int32_t mesh_index,
    int32_t tri_index) {
    if (!valid || fraction < 0.0f || fraction > 1.0f) {
        return;
    }
    if (best.cal_min_fraction_sq > 0.0f && fraction * fraction < best.cal_min_fraction_sq) {
        return;
    }

    const float t = fraction * max_len;
    if (!best.hit || fraction < best.ray_fraction) {
        best.hit = 1;
        best.t = t;
        best.point = point;
        best.point_bt = point_bt;
        best.normal = normal;
        best.ray_fraction = fraction;
        best.kind = kind;
        best.mesh_index = mesh_index;
        best.tri_index = tri_index;
    }
}

ROCKETGPU_HD void MaybeUpdateBestHit(
    bool valid,
    float t,
    const Vec3& point,
    const Vec3& point_bt,
    const Vec3& normal,
    float max_len,
    ArenaRayHit& best,
    uint8_t kind,
    int32_t mesh_index,
    int32_t tri_index) {
    if (!valid || max_len <= 0.0f || t < 0.0f || t > max_len) {
        return;
    }
    if (best.cal_min_fraction_sq > 0.0f && (t / max_len) * (t / max_len) < best.cal_min_fraction_sq) {
        return;
    }
    if (!best.hit || t < best.t) {
        best.hit = 1;
        best.t = t;
        best.point = point;
        best.point_bt = point_bt;
        best.normal = normal;
        best.ray_fraction = t / max_len;
        best.kind = kind;
        best.mesh_index = mesh_index;
        best.tri_index = tri_index;
    }
}

ROCKETGPU_HD void PlaneSpace1LikeBullet(const Vec3& n, Vec3& p, Vec3& q) {
    constexpr float kSqrtHalf = 0.70710678118654752440f;
    if (Abs(n.z) > kSqrtHalf) {
        const float a = n.y * n.y + n.z * n.z;
        const float k = RsqrtScalar(a);
        p = Vec3(0.0f, -n.z * k, n.y * k);
        q = Vec3(a * k, -n.x * p.z, n.x * p.y);
    } else {
        const float a = n.x * n.x + n.y * n.y;
        const float k = RsqrtScalar(a);
        p = Vec3(-n.y * k, n.x * k, 0.0f);
        q = Vec3(-n.z * p.y, n.z * p.x, a * k);
    }
}

ROCKETGPU_HD void RayCastTriangleLikeBullet(
    const Vec3& from,
    const Vec3& to,
    const Vec3& v0,
    const Vec3& v1,
    const Vec3& v2,
    float& hit_fraction,
    Vec3& hit_normal,
    bool& hit,
    const uint32_t* native_table = nullptr, int native_shift = 0) {
    const Vec3 v10 = v1 - v0;
    const Vec3 v20 = v2 - v0;
    Vec3 triangle_normal = Cross(v10, v20);

    const float dist = Dot(v0, triangle_normal);
    float dist_a = Dot(triangle_normal, from);
    dist_a -= dist;
    float dist_b = Dot(triangle_normal, to);
    dist_b -= dist;

    if (dist_a * dist_b >= 0.0f) {
        return;
    }

    const float proj_length = dist_a - dist_b;
    const float distance = dist_a / proj_length;
    if (distance >= hit_fraction) {
        return;
    }

    float edge_tolerance = LengthSq(triangle_normal);
    edge_tolerance *= -0.0001f;

    const Vec3 point = (1.0f - distance) * from + distance * to;
    const Vec3 v0p = v0 - point;
    const Vec3 v1p = v1 - point;
    const Vec3 cp0 = Cross(v0p, v1p);
    if (Dot(cp0, triangle_normal) < edge_tolerance) {
        return;
    }

    const Vec3 v2p = v2 - point;
    const Vec3 cp1 = Cross(v1p, v2p);
    if (Dot(cp1, triangle_normal) < edge_tolerance) {
        return;
    }

    const Vec3 cp2 = Cross(v2p, v0p);
    if (Dot(cp2, triangle_normal) < edge_tolerance) {
        return;
    }

    triangle_normal = native_table
        ? NativeSseSphereNormalizer{native_table, native_shift}(triangle_normal)
        : Normalize(triangle_normal, triangle_normal);
    hit_fraction = distance;
    hit_normal = dist_a <= 0.0f ? -triangle_normal : triangle_normal;
    hit = true;
}

// The arena planes are axis-aligned, so CastStaticPlaneLikeBullet's two triangles lie exactly in the plane and their
// normal is exactly along it: each triangle cast returns at dist_a * dist_b >= 0 unless the ray endpoints are strictly
// on opposite sides. False only when the cast cannot hit (NaN endpoints return true and take the full cast).
ROCKETGPU_HD bool StaticPlaneRayMayCross(
    const Vec3& source_bt, const Vec3& to_bt, const Vec3& plane_origin_bt, const Vec3& plane_normal) {
    const float side_from = Dot(plane_normal, source_bt - plane_origin_bt);
    const float side_to = Dot(plane_normal, to_bt - plane_origin_bt);
    return !((side_from >= 0.0f && side_to >= 0.0f) || (side_from <= 0.0f && side_to <= 0.0f));
}

ROCKETGPU_HD void CastStaticPlaneLikeBullet(
    const Vec3& source_bt,
    const Vec3& to_bt,
    const Vec3& plane_origin_bt,
    const Vec3& plane_normal,
    float max_len,
    ArenaRayHit& best,
    const uint32_t* native_table = nullptr, int native_shift = 0) {
    const Vec3 from_local = source_bt - plane_origin_bt;
    const Vec3 to_local = to_bt - plane_origin_bt;
    Vec3 aabb_min = from_local;
    aabb_min.x = Min(aabb_min.x, to_local.x);
    aabb_min.y = Min(aabb_min.y, to_local.y);
    aabb_min.z = Min(aabb_min.z, to_local.z);
    Vec3 aabb_max = from_local;
    aabb_max.x = Max(aabb_max.x, to_local.x);
    aabb_max.y = Max(aabb_max.y, to_local.y);
    aabb_max.z = Max(aabb_max.z, to_local.z);

    const Vec3 half_extents = (aabb_max - aabb_min) * 0.5f;
    const float radius = Length(half_extents);
    const Vec3 center = (aabb_max + aabb_min) * 0.5f;

    Vec3 tangent0(0.0f, 0.0f, 0.0f);
    Vec3 tangent1(0.0f, 0.0f, 0.0f);
    PlaneSpace1LikeBullet(plane_normal, tangent0, tangent1);
    const Vec3 projected_center = center - (Dot(plane_normal, center) * plane_normal);

    const Vec3 tri0[3] = {
        projected_center + tangent0 * radius + tangent1 * radius,
        projected_center + tangent0 * radius - tangent1 * radius,
        projected_center - tangent0 * radius - tangent1 * radius,
    };
    const Vec3 tri1[3] = {
        projected_center - tangent0 * radius - tangent1 * radius,
        projected_center - tangent0 * radius + tangent1 * radius,
        projected_center + tangent0 * radius + tangent1 * radius,
    };

    float fraction = best.hit ? best.ray_fraction : 1.0f;
    Vec3 normal(0.0f, 0.0f, 1.0f);
    bool hit = false;
    RayCastTriangleLikeBullet(from_local, to_local, tri0[0], tri0[1], tri0[2], fraction, normal, hit, native_table, native_shift);
    RayCastTriangleLikeBullet(from_local, to_local, tri1[0], tri1[1], tri1[2], fraction, normal, hit, native_table, native_shift);
    if (!hit) {
        return;
    }

    const float one_minus_fraction = 1.0f - fraction;
    const Vec3 point_bt(
        source_bt.x * one_minus_fraction + to_bt.x * fraction,
        source_bt.y * one_minus_fraction + to_bt.y * fraction,
        source_bt.z * one_minus_fraction + to_bt.z * fraction);

    MaybeUpdateBestHitFraction(
        true,
        fraction,
        max_len,
        point_bt * 50.0f,
        point_bt,
        normal,
        best,
        ROCKETGPU_ARENA_RAY_HIT_STATIC_PLANE);
}

ROCKETGPU_HD void CastArenaRayPlanesImpl(const Vec3& source_bt, const Vec3& ray_delta_bt, float max_len, ArenaRayHit& best,
    const uint32_t* native_table = nullptr, int native_shift = 0) {
    constexpr float kEps = 1e-5f;
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kBTToUU = 50.0f;
    const Vec3 to_bt = source_bt + ray_delta_bt;

    if (Abs(ray_delta_bt.z) > kEps) {
        if (StaticPlaneRayMayCross(source_bt, to_bt, Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f))) {
            CastStaticPlaneLikeBullet(source_bt, to_bt, Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f), max_len, best, native_table, native_shift);
        }
        if (StaticPlaneRayMayCross(source_bt, to_bt, Vec3(0.0f, 0.0f, ArenaHeight() * kUUToBT), Vec3(0.0f, 0.0f, -1.0f))) {
            CastStaticPlaneLikeBullet(source_bt, to_bt, Vec3(0.0f, 0.0f, ArenaHeight() * kUUToBT), Vec3(0.0f, 0.0f, -1.0f), max_len, best, native_table, native_shift);
        }
    }

    if (Abs(ray_delta_bt.x) > kEps) {
        if (StaticPlaneRayMayCross(source_bt, to_bt, Vec3(ArenaExtentX() * kUUToBT, 0.0f, ArenaHeight() * 0.5f * kUUToBT), Vec3(-1.0f, 0.0f, 0.0f))) {
            CastStaticPlaneLikeBullet(source_bt, to_bt, Vec3(ArenaExtentX() * kUUToBT, 0.0f, ArenaHeight() * 0.5f * kUUToBT), Vec3(-1.0f, 0.0f, 0.0f), max_len, best, native_table, native_shift);
        }
        if (StaticPlaneRayMayCross(source_bt, to_bt, Vec3(-ArenaExtentX() * kUUToBT, 0.0f, ArenaHeight() * 0.5f * kUUToBT), Vec3(1.0f, 0.0f, 0.0f))) {
            CastStaticPlaneLikeBullet(source_bt, to_bt, Vec3(-ArenaExtentX() * kUUToBT, 0.0f, ArenaHeight() * 0.5f * kUUToBT), Vec3(1.0f, 0.0f, 0.0f), max_len, best, native_table, native_shift);
        }
    }

    if (Abs(ray_delta_bt.y) > kEps) {
        (void)kBTToUU;
    }
}

ROCKETGPU_HD void CastArenaRayCornerCylindersImpl(const Vec3& source_bt, const Vec3& ray_delta_bt, float max_len, ArenaRayHit& best) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kBTToUU = 50.0f;
    const Vec3 centers[4] = {
        Vec3(CornerCenterX() * kUUToBT, CornerCenterY() * kUUToBT, 0.0f),
        Vec3(-CornerCenterX() * kUUToBT, CornerCenterY() * kUUToBT, 0.0f),
        Vec3(CornerCenterX() * kUUToBT, -CornerCenterY() * kUUToBT, 0.0f),
        Vec3(-CornerCenterX() * kUUToBT, -CornerCenterY() * kUUToBT, 0.0f),
    };
    const float dx = ray_delta_bt.x;
    const float dy = ray_delta_bt.y;
    const float a = dx * dx + dy * dy;
    if (a <= 1e-8f) {
        return;
    }

    for (int i = 0; i < 4; ++i) {
        const Vec3 center = centers[i];
        const float mx = source_bt.x - center.x;
        const float my = source_bt.y - center.y;
        const float b = 2.0f * (mx * dx + my * dy);
        const float c = mx * mx + my * my - (CornerRadius() * kUUToBT) * (CornerRadius() * kUUToBT);
        float disc = b * b - 4.0f * a * c;
        if (disc < 0.0f) {
            continue;
        }
#if defined(ROCKETGPU_USE_CUDA)
        disc = sqrtf(disc);
#else
        disc = std::sqrt(disc);
#endif
        const float inv_2a = 0.5f / a;
        const float t0 = (-b - disc) * inv_2a;
        const float t1 = (-b + disc) * inv_2a;
        float t = -1.0f;
        if (t0 >= 0.0f) {
            t = t0;
        }
        if (t1 >= 0.0f && (t < 0.0f || t1 < t)) {
            t = t1;
        }
        if (t < 0.0f || t > 1.0f) {
            continue;
        }
        const Vec3 point_bt = source_bt + ray_delta_bt * t;
        const Vec3 point = point_bt * kBTToUU;
        const bool valid = Abs(point.x) >= CornerCenterX() - 1e-3f && Abs(point.y) >= CornerCenterY() - 1e-3f && Abs(point.y) <= ArenaExtentY() + 1e-3f && point.z >= -1e-3f && point.z <= ArenaHeight() + 1e-3f;
        const Vec3 normal = Normalize((center - point_bt) * kBTToUU, Vec3(0.0f, 0.0f, 1.0f));
        MaybeUpdateBestHitFraction(
            valid,
            t,
            max_len,
            point,
            point_bt,
            normal,
            best,
            ROCKETGPU_ARENA_RAY_HIT_CORNER_CYLINDER);
    }
}

ROCKETGPU_HD bool RayCastArenaMeshTrianglesBtSource(
    const ArenaMeshTriangle* mesh_triangles,
    int32_t mesh_triangle_count,
    const Vec3& ray_source_bt,
    const Vec3& ray_delta_bt,
    float max_len,
    ArenaRayHit& best,
    const uint32_t* native_table = nullptr, int native_shift = 0) {
    constexpr float kBTToUU = 50.0f;
    if (mesh_triangles == nullptr || mesh_triangle_count <= 0 || max_len <= 0.0f) {
        return best.hit != 0;
    }

    const Vec3 to_bt = ray_source_bt + ray_delta_bt;
    const Vec3 ray_min_bt(
        Min(ray_source_bt.x, to_bt.x),
        Min(ray_source_bt.y, to_bt.y),
        Min(ray_source_bt.z, to_bt.z));
    const Vec3 ray_max_bt(
        Max(ray_source_bt.x, to_bt.x),
        Max(ray_source_bt.y, to_bt.y),
        Max(ray_source_bt.z, to_bt.z));
    float best_fraction = best.hit ? best.ray_fraction : 1.0f;

    for (int32_t tri_idx = 0; tri_idx < mesh_triangle_count; ++tri_idx) {
        const ArenaMeshTriangle& tri = mesh_triangles[tri_idx];
        if (tri.max.x < ray_min_bt.x || tri.min.x > ray_max_bt.x ||
            tri.max.y < ray_min_bt.y || tri.min.y > ray_max_bt.y ||
            tri.max.z < ray_min_bt.z || tri.min.z > ray_max_bt.z) {
            continue;
        }

        const Vec3 v10 = tri.b - tri.a;
        const Vec3 v20 = tri.c - tri.a;
        Vec3 triangle_normal = Cross(v10, v20);
        const float plane_dist = Dot(tri.a, triangle_normal);
        const float dist_a = Dot(triangle_normal, ray_source_bt) - plane_dist;
        const float dist_b = Dot(triangle_normal, to_bt) - plane_dist;
        if (dist_a * dist_b >= 0.0f) {
            continue;
        }

        const float proj_length = dist_a - dist_b;
        if (proj_length == 0.0f) {
            continue;
        }

        const float distance = dist_a / proj_length;
        if (distance >= best_fraction) {
            continue;
        }

        const float edge_tolerance = -0.0001f * LengthSq(triangle_normal);

        const Vec3 point_bt = (1.0f - distance) * ray_source_bt + distance * to_bt;
        const Vec3 v0p = tri.a - point_bt;
        const Vec3 v1p = tri.b - point_bt;
        if (Dot(Cross(v0p, v1p), triangle_normal) < edge_tolerance) {
            continue;
        }

        const Vec3 v2p = tri.c - point_bt;
        if (Dot(Cross(v1p, v2p), triangle_normal) < edge_tolerance) {
            continue;
        }
        if (Dot(Cross(v2p, v0p), triangle_normal) < edge_tolerance) {
            continue;
        }

        triangle_normal = native_table
            ? NativeSseSphereNormalizer{native_table, native_shift}(triangle_normal)
            : Normalize(triangle_normal, tri.normal);
        if (dist_a <= 0.0f) {
            triangle_normal = -triangle_normal;
        }

        MaybeUpdateBestHitFraction(
            true,
            distance,
            max_len,
            point_bt * kBTToUU,
            point_bt,
            triangle_normal,
            best,
            ROCKETGPU_ARENA_RAY_HIT_ARENA_MESH,
            static_cast<int32_t>(tri.mesh_index),
            tri_idx);
        best_fraction = best.hit ? best.ray_fraction : 1.0f;
    }

    return best.hit != 0;
}

ROCKETGPU_HD bool RayCastArenaMeshTriangleBtSource(
    const ArenaMeshTriangle& tri,
    int32_t tri_idx,
    const Vec3& ray_source_bt,
    const Vec3& to_bt,
    float max_len,
    float& best_fraction,
    ArenaRayHit& best,
    const uint32_t* native_table = nullptr, int native_shift = 0) {
    constexpr float kBTToUU = 50.0f;
    const Vec3 v10 = tri.b - tri.a;
    const Vec3 v20 = tri.c - tri.a;
    Vec3 triangle_normal = Cross(v10, v20);
    const float plane_dist = Dot(tri.a, triangle_normal);
    const float dist_a = Dot(triangle_normal, ray_source_bt) - plane_dist;
    const float dist_b = Dot(triangle_normal, to_bt) - plane_dist;
    if (dist_a * dist_b >= 0.0f) {
        return false;
    }

    const float proj_length = dist_a - dist_b;
    if (proj_length == 0.0f) {
        return false;
    }

    const float distance = dist_a / proj_length;
    if (distance >= best_fraction) {
        return false;
    }

    const float edge_tolerance = -0.0001f * LengthSq(triangle_normal);
    const Vec3 point_bt = (1.0f - distance) * ray_source_bt + distance * to_bt;
    const Vec3 v0p = tri.a - point_bt;
    const Vec3 v1p = tri.b - point_bt;
    if (Dot(Cross(v0p, v1p), triangle_normal) < edge_tolerance) {
        return false;
    }

    const Vec3 v2p = tri.c - point_bt;
    if (Dot(Cross(v1p, v2p), triangle_normal) < edge_tolerance) {
        return false;
    }
    if (Dot(Cross(v2p, v0p), triangle_normal) < edge_tolerance) {
        return false;
    }

    triangle_normal = native_table
        ? NativeSseSphereNormalizer{native_table, native_shift}(triangle_normal)
        : Normalize(triangle_normal, tri.normal);
    if (dist_a <= 0.0f) {
        triangle_normal = -triangle_normal;
    }

    MaybeUpdateBestHitFraction(
        true,
        distance,
        max_len,
        point_bt * kBTToUU,
        point_bt,
        triangle_normal,
        best,
        ROCKETGPU_ARENA_RAY_HIT_ARENA_MESH,
        static_cast<int32_t>(tri.mesh_index),
        tri_idx);
    best_fraction = best.hit ? best.ray_fraction : 1.0f;
    return true;
}

ROCKETGPU_HD bool RayCastArenaMeshBvhBtSource(
    const ArenaMeshTriangle* mesh_triangles,
    int32_t mesh_triangle_count,
    const ArenaMeshBvhQuantization* mesh_bvh_quantization,
    int32_t mesh_bvh_quantization_count,
    const ArenaMeshBvhMeshInfo* mesh_bvh_meshes,
    int32_t mesh_bvh_mesh_count,
    const ArenaMeshBvhNode* mesh_bvh_nodes,
    int32_t mesh_bvh_node_count,
    const ArenaMeshBvhSubtreeInfo* mesh_bvh_subtrees,
    int32_t mesh_bvh_subtree_count,
    const Vec3& ray_source_bt,
    const Vec3& ray_delta_bt,
    float max_len,
    ArenaRayHit& best,
    const uint32_t* native_table = nullptr, int native_shift = 0) {
    if (mesh_triangles == nullptr || mesh_triangle_count <= 0 || max_len <= 0.0f ||
        mesh_bvh_quantization == nullptr || mesh_bvh_meshes == nullptr || mesh_bvh_nodes == nullptr ||
        mesh_bvh_quantization_count <= 0 || mesh_bvh_mesh_count <= 0 || mesh_bvh_node_count <= 0) {
        return RayCastArenaMeshTrianglesBtSource(mesh_triangles, mesh_triangle_count, ray_source_bt, ray_delta_bt, max_len, best, native_table, native_shift);
    }

    const Vec3 to_bt = ray_source_bt + ray_delta_bt;
    const Vec3 ray_min_bt(
        Min(ray_source_bt.x, to_bt.x),
        Min(ray_source_bt.y, to_bt.y),
        Min(ray_source_bt.z, to_bt.z));
    const Vec3 ray_max_bt(
        Max(ray_source_bt.x, to_bt.x),
        Max(ray_source_bt.y, to_bt.y),
        Max(ray_source_bt.z, to_bt.z));
    // Inside the free box every mesh AABB misses the ray AABB (each lies beyond it in x or y), so each mesh is skipped
    // by the test below.
    if (ray_max_bt.x < mesh_bvh_meshes[0].free_half_x && ray_min_bt.x > -mesh_bvh_meshes[0].free_half_x &&
        ray_max_bt.y < mesh_bvh_meshes[0].free_half_y && ray_min_bt.y > -mesh_bvh_meshes[0].free_half_y) {
        return best.hit != 0;
    }
    float best_fraction = best.hit ? best.ray_fraction : 1.0f;

    uint16_t quantized_query_aabb_min[3] = {0, 0, 0};
    uint16_t quantized_query_aabb_max[3] = {0, 0, 0};

    for (int mesh_info_idx = 0; mesh_info_idx < mesh_bvh_mesh_count; ++mesh_info_idx) {
        const ArenaMeshBvhMeshInfo& mesh_info = mesh_bvh_meshes[mesh_info_idx];
        // aabb_min/max is the union of this mesh's triangle AABBs, so when it misses the ray AABB every
        // triangle fails the per-triangle AABB test below: skipping the walk can't change the hit.
        if (mesh_info.aabb_max.x < ray_min_bt.x || mesh_info.aabb_min.x > ray_max_bt.x ||
            mesh_info.aabb_max.y < ray_min_bt.y || mesh_info.aabb_min.y > ray_max_bt.y ||
            mesh_info.aabb_max.z < ray_min_bt.z || mesh_info.aabb_min.z > ray_max_bt.z) {
            continue;
        }
        if (mesh_info.mesh_index < 0 ||
            mesh_info.mesh_index >= mesh_bvh_quantization_count ||
            mesh_info.node_offset < 0 ||
            mesh_info.node_count <= 0 ||
            mesh_info.node_offset + mesh_info.node_count > mesh_bvh_node_count ||
            mesh_info.triangle_start < 0 ||
            mesh_info.triangle_count <= 0 ||
            mesh_info.triangle_start + mesh_info.triangle_count > mesh_triangle_count) {
            continue;
        }

        const ArenaMeshBvhQuantization& quantization = mesh_bvh_quantization[mesh_info.mesh_index];
        QuantizeArenaMeshBvhPoint(quantization, ray_min_bt, 0, quantized_query_aabb_min);
        QuantizeArenaMeshBvhPoint(quantization, ray_max_bt, 1, quantized_query_aabb_max);

        const ArenaMeshBvhSubtreeInfo* mesh_subtrees = nullptr;
        int32_t mesh_subtree_count = 0;
        if (mesh_bvh_subtrees != nullptr &&
            mesh_info.subtree_offset >= 0 &&
            mesh_info.subtree_count > 0 &&
            mesh_info.subtree_offset + mesh_info.subtree_count <= mesh_bvh_subtree_count) {
            mesh_subtrees = mesh_bvh_subtrees + mesh_info.subtree_offset;
            mesh_subtree_count = mesh_info.subtree_count;
        }
        // Bullet's reportRayOverlappingNodex walks the whole quantized tree stacklessly (node order), and the closest
        // hit keeps the first triangle on an exact fraction tie: walk the node-ordered headers, not the build-order ones.
        SelectArenaMeshBvhDfsOrderSubtrees(
            mesh_info, mesh_bvh_subtrees, mesh_bvh_subtree_count, mesh_subtrees, mesh_subtree_count);

        WalkArenaMeshBvhStacklessQuantizedTreeCacheFriendly(
            mesh_bvh_nodes,
            mesh_subtrees,
            mesh_subtree_count,
            quantized_query_aabb_min,
            quantized_query_aabb_max,
            mesh_info.node_offset,
            mesh_info.node_count,
            [&](int32_t local_tri_idx) {
                if (local_tri_idx < 0 || local_tri_idx >= mesh_info.triangle_count) {
                    return;
                }
                const int32_t global_tri_idx = mesh_info.triangle_start + local_tri_idx;
                const ArenaMeshTriangle& tri = mesh_triangles[global_tri_idx];
                if (tri.max.x < ray_min_bt.x || tri.min.x > ray_max_bt.x ||
                    tri.max.y < ray_min_bt.y || tri.min.y > ray_max_bt.y ||
                    tri.max.z < ray_min_bt.z || tri.min.z > ray_max_bt.z) {
                    return;
                }
                RayCastArenaMeshTriangleBtSource(
                    tri,
                    global_tri_idx,
                    ray_source_bt,
                    to_bt,
                    max_len,
                    best_fraction,
                    best,
                    native_table,
                    native_shift);
            });
    }

    return best.hit != 0;
}
