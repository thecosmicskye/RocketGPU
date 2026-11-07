constexpr float kWheelSuspensionStiffness = 500.0f;
constexpr float kWheelDampingCompression = 25.0f;
constexpr float kWheelDampingRelaxation = 40.0f;

struct WheelStepState {
    Vec3 hard_point_ws = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 ray_source_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 wheel_dir_ws = Vec3(0.0f, 0.0f, -1.0f);
    Vec3 wheel_axle_ws = Vec3(0.0f, -1.0f, 0.0f);
    Vec3 contact_point_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 contact_normal_ws = Vec3(0.0f, 0.0f, 1.0f);

    float wheel_trace_len_bt = 0.0f;

    float suspension_len_bt = 0.0f;
    float suspension_relative_velocity_bt = 0.0f;
    float suspension_relative_velocity = 0.0f;
    float clipped_inv_contact_dot_suspension = 1.0f;
    float extra_pushback = 0.0f;
    float wheel_radius = 0.0f;
    float suspension_rest_length = 0.0f;
    float max_suspension_travel = 12.0f;
    float suspension_force_scale = 1.0f;

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

ROCKETGPU_HD Vec3 BulletBoxHalfExtentWithMargin(const CarConfig& cfg) {
    return (BulletHitboxSize(cfg) * 0.5f) - BallCarHitboxInset(cfg) + Vec3(BallCarShapeMargin(cfg), BallCarShapeMargin(cfg), BallCarShapeMargin(cfg));
}

ROCKETGPU_HD Vec3 BulletBoxHalfExtentWithMarginBt(const CarConfig& cfg) {
    constexpr float kDefaultMarginBt = 0.04f;
    constexpr float kUUToBT = 1.0f / 50.0f;
    const Vec3 half_extent_bt = (BulletHitboxSize(cfg) * kUUToBT) / 2.0f;
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

struct CarInvInertiaRowsUu {
    Vec3 row0, row1, row2;
};

ROCKETGPU_HD Vec3 ApplyInverseBodyInertiaWorld(const CarInvInertiaRowsUu& r, const Vec3& torque_world) {
    return Vec3(Dot(r.row0, torque_world), Dot(r.row1, torque_world), Dot(r.row2, torque_world));
}

ROCKETGPU_HD Vec3 CarInvInertiaLocalUu(const CarConfig& cfg, float mass) {
    const Vec3 inertia = BoxBodyInertiaDiag(cfg, mass);
    return Vec3(
        inertia.x != 0.0f ? 1.0f / inertia.x : 0.0f,
        inertia.y != 0.0f ? 1.0f / inertia.y : 0.0f,
        inertia.z != 0.0f ? 1.0f / inertia.z : 0.0f);
}

ROCKETGPU_HD CarInvInertiaRowsUu MakeCarInvInertiaRowsUuWithLocal(const CarState& car, const Vec3& inv_inertia_local) {
    if (car.restored_inv_inertia_valid) {
        return CarInvInertiaRowsUu{car.restored_inv_inertia_bt[0] * 0.0004f,
                                  car.restored_inv_inertia_bt[1] * 0.0004f,
                                  car.restored_inv_inertia_bt[2] * 0.0004f};
    }
    const Vec3& forward_m = car.tick_start_basis_forward;
    const Vec3& right_m = car.tick_start_basis_right;
    const Vec3& up_m = car.tick_start_basis_up;
    const Vec3 row0(forward_m.x, right_m.x, up_m.x);
    const Vec3 row1(forward_m.y, right_m.y, up_m.y);
    const Vec3 row2(forward_m.z, right_m.z, up_m.z);
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
    return CarInvInertiaRowsUu{inv_inertia_world_row0, inv_inertia_world_row1, inv_inertia_world_row2};
}

ROCKETGPU_HD CarInvInertiaRowsUu MakeCarInvInertiaRowsUu(const CarState& car, const CarConfig& cfg, float mass) {
    if (car.restored_inv_inertia_valid) {
        return MakeCarInvInertiaRowsUuWithLocal(car, Vec3(0.0f, 0.0f, 0.0f));
    }
    return MakeCarInvInertiaRowsUuWithLocal(car, CarInvInertiaLocalUu(cfg, mass));
}

ROCKETGPU_HD Vec3 ApplyInverseBodyInertiaWorld(const CarState& car, const CarConfig& cfg, float mass, const Vec3& torque_world) {
    return ApplyInverseBodyInertiaWorld(MakeCarInvInertiaRowsUu(car, cfg, mass), torque_world);
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
    const Vec3 sized = cfg.hitbox_published + Vec3(2.5f, 2.5f, 2.5f);
    const bool has_published = sized.x == cfg.hitbox_size.x && sized.y == cfg.hitbox_size.y && sized.z == cfg.hitbox_size.z;
    const Vec3 published = has_published ? cfg.hitbox_published : cfg.hitbox_size - Vec3(2.5f, 2.5f, 2.5f);
    const Vec3 setup_half = (published * 0.02f) * 0.5f + Vec3(0.025f, 0.025f, 0.025f);
    const Vec3 box_half = (setup_half * 50.0f) * 0.02f;
    const float margin = Min(0.04f, 0.1f * Min(Min(box_half.x, box_half.y), box_half.z));
    const Vec3 margin_v(margin, margin, margin);
    const Vec3 half_extent = (box_half - margin_v) + margin_v;
    const float x = half_extent.x * 2.0f;
    const float y = half_extent.y * 2.0f;
    const float z = half_extent.z * 2.0f;
    const float x2 = x * x;
    const float y2 = y * y;
    const float z2 = z * z;
    const float scale = mass / 12.0f;
    return Vec3(scale * (y2 + z2), scale * (x2 + z2), scale * (x2 + y2));
}

#if !defined(ROCKETGPU_USE_CUDA)
struct CarControlsMemo {
    CarControls in;
    CarControls out;
    float raw_pitch = 2.0f;
    bool valid = false;
};
#endif

struct CarFrame {
    Vec3 fwd, right, up;
    Vec3 inv_inertia_local_bt;
    Vec3 inv_world_row0_bt, inv_world_row1_bt, inv_world_row2_bt;
#if !defined(__CUDA_ARCH__)
    mutable bool has_inertia_world_bt = false;
    mutable bool has_inertia_world_uu = false;
#if !defined(ROCKETGPU_USE_CUDA)
#if defined(ROCKETGPU_HOST_LANES)
    mutable UninitArray<lanes::F4, 3> inertia_world_bt_cols;
#else
    mutable UninitArray<Vec3, 3> inertia_world_bt_rows;
#endif
    mutable UninitArray<Vec3, 3> inertia_world_uu_rows;
#else
    mutable Vec3 inertia_world_bt_row0, inertia_world_bt_row1, inertia_world_bt_row2;
    mutable Vec3 inertia_world_uu_row0, inertia_world_uu_row1, inertia_world_uu_row2;
#endif
#endif
#if !defined(ROCKETGPU_USE_CUDA)
    mutable const CarRbSnapMemo* rb_snap = nullptr;
    mutable const CarShapeCacheEntry* shape = nullptr;
    mutable CarControlsMemo* ctl_memo = nullptr;
    mutable bool wheels_pre_inited = false;
    mutable struct WheelsX4* wx = nullptr;
#endif
    mutable bool queue_forces = false;
    mutable Vec3 queued_force = Vec3(0.0f, 0.0f, 0.0f);
    mutable Vec3 queued_ang_accel = Vec3(0.0f, 0.0f, 0.0f);
    mutable Vec3 queued_torque_bt = Vec3(0.0f, 0.0f, 0.0f);
    mutable bool has_queued_jump_force = false;
    mutable Vec3 queued_jump_force = Vec3(0.0f, 0.0f, 0.0f);
    static constexpr int kQueueItems = 4;
    mutable int n_force_items = 0;
    mutable int n_torque_items = 0;
#if !defined(ROCKETGPU_USE_CUDA)
    mutable UninitArray<Vec3, kQueueItems> force_items;
    mutable UninitArray<Vec3, kQueueItems> torque_items;
#else
    mutable Vec3 force_items[kQueueItems];
    mutable Vec3 torque_items[kQueueItems];
#endif
};

ROCKETGPU_HD void BuildInvInertiaTensorWorldBt(
    const CarState& car, const CarConfig& cfg, float mass,
    Vec3& out_row0, Vec3& out_row1, Vec3& out_row2);

ROCKETGPU_HD void BuildInvInertiaTensorWorldBtFromColumnsWithLocalInv(
    const Vec3& forward_col,
    const Vec3& right_col,
    const Vec3& up_col,
    const Vec3& inv_inertia_local,
    Vec3& out_row0,
    Vec3& out_row1,
    Vec3& out_row2) {
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

#if !defined(ROCKETGPU_USE_CUDA)
ROCKETGPU_HD Vec3 CarInvInertiaLocalBt(const CarConfig& cfg, float mass) {
    const Vec3 inertia_bt = BoxBodyInertiaDiagBt(cfg, mass);
    return Vec3(
        inertia_bt.x != 0.0f ? 1.0f / inertia_bt.x : 0.0f,
        inertia_bt.y != 0.0f ? 1.0f / inertia_bt.y : 0.0f,
        inertia_bt.z != 0.0f ? 1.0f / inertia_bt.z : 0.0f);
}
#endif

ROCKETGPU_HD CarFrame MakeCarFrame(const CarState& car, const CarConfig& cfg, float mass
#if !defined(ROCKETGPU_USE_CUDA)
                                   , const CarShapeCacheEntry* shape = nullptr
#endif
                                   ) {
    CarFrame f;
    f.fwd = Vec3(1.0f, 0.0f, 0.0f);
    f.right = Vec3(0.0f, 1.0f, 0.0f);
    f.up = Vec3(0.0f, 0.0f, 1.0f);
    CarBasisColumns(car, f.fwd, f.right, f.up);
#if !defined(ROCKETGPU_USE_CUDA)
    f.inv_inertia_local_bt = shape != nullptr ? shape->inv_inertia_local_bt : CarInvInertiaLocalBt(cfg, mass);
#else
    const Vec3 inertia_bt = BoxBodyInertiaDiagBt(cfg, mass);
    f.inv_inertia_local_bt = Vec3(
        inertia_bt.x != 0.0f ? 1.0f / inertia_bt.x : 0.0f,
        inertia_bt.y != 0.0f ? 1.0f / inertia_bt.y : 0.0f,
        inertia_bt.z != 0.0f ? 1.0f / inertia_bt.z : 0.0f);
#endif
    if (car.restored_inv_inertia_valid) {
        f.inv_world_row0_bt = car.restored_inv_inertia_bt[0];
        f.inv_world_row1_bt = car.restored_inv_inertia_bt[1];
        f.inv_world_row2_bt = car.restored_inv_inertia_bt[2];
    } else {
        BuildInvInertiaTensorWorldBtFromColumnsWithLocalInv(
            car.tick_start_basis_forward, car.tick_start_basis_right, car.tick_start_basis_up,
            f.inv_inertia_local_bt, f.inv_world_row0_bt, f.inv_world_row1_bt, f.inv_world_row2_bt);
    }
    return f;
}

ROCKETGPU_HD void CarBasisColumns(const CarFrame* frame, const CarState& car, Vec3& fwd, Vec3& right, Vec3& up) {
    if (frame) {
        fwd = frame->fwd;
        right = frame->right;
        up = frame->up;
        return;
    }
    CarBasisColumns(car, fwd, right, up);
}

ROCKETGPU_HD Vec3 ApplyInverseBodyInertiaWorldBtRowDot(const CarFrame& f, const Vec3& torque_world_bt) {
    return Vec3(
        Dot(f.inv_world_row0_bt, torque_world_bt),
        Dot(f.inv_world_row1_bt, torque_world_bt),
        Dot(f.inv_world_row2_bt, torque_world_bt));
}

ROCKETGPU_HD Vec3 ApplyInverseBodyInertiaWorldBtRowDot(
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    const Vec3& torque_world_bt) {
    Vec3 row0, row1, row2;
    BuildInvInertiaTensorWorldBt(car, cfg, mass, row0, row1, row2);
    return Vec3(Dot(row0, torque_world_bt), Dot(row1, torque_world_bt), Dot(row2, torque_world_bt));
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
    const Vec3& v,
    const CarFrame* frame = nullptr) {
    Vec3 row0, row1, row2;
    if (frame) {
        row0 = frame->inv_world_row0_bt;
        row1 = frame->inv_world_row1_bt;
        row2 = frame->inv_world_row2_bt;
    } else {
        BuildInvInertiaTensorWorldBt(car, cfg, mass, row0, row1, row2);
    }
    return Vec3(row0.x * v.x + row1.x * v.y + row2.x * v.z,
                row0.y * v.x + row1.y * v.y + row2.y * v.z,
                row0.z * v.x + row1.z * v.y + row2.z * v.z);
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
    const CarFrame* frame = nullptr,
    const Vec3* car_pos_override_bt = nullptr) {
    const Vec3 rel_bt = pos_world_bt - (car_pos_override_bt ? *car_pos_override_bt : car.pos);
    const Vec3 c0_bt = Cross(rel_bt, normal_world);

    const Vec3 angular_component_bt = ApplyInverseBodyInertiaWorldBtColumnDot(car, cfg, mass, c0_bt, frame);
    const Vec3 vec_bt = Cross(angular_component_bt, rel_bt);
    const float angular_dot_bt = Dot(normal_world, vec_bt);
    const float inv_mass_bt = 1.0f / mass;
    const float denom_bt = inv_mass_bt + angular_dot_bt;
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
    const CarFrame* frame = nullptr) {
    const Vec3 rel_bt = contact_pos_world_bt - car.pos_bt;
    const Vec3 vel_at_contact_bt = car.vel_bt + Cross(car.ang_vel, rel_bt);
    const float rel_vel_bt = Dot(contact_normal_world, vel_at_contact_bt);
    const float denom_bt = BodyImpulseDenominatorBt(
        car, cfg, mass, contact_pos_world_bt, contact_normal_world, frame, &car.pos_bt);
    if (denom_bt <= 0.0f || dt <= 0.0f) {
        return 0.0f;
    }
    const float jac_diag_ab_inv = 1.0f / denom_bt;
    const float positional_error_bt = 0.2f * -distance_bt / dt;
    const float velocity_error_bt = -rel_vel_bt;
    const float penetration_impulse = positional_error_bt * jac_diag_ab_inv;
    const float velocity_impulse = velocity_error_bt * jac_diag_ab_inv;
    const float normal_impulse_bt = penetration_impulse + velocity_impulse;
    return normal_impulse_bt > 0.0f ? normal_impulse_bt : 0.0f;
}

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
    const Vec3 vel1 = car.vel_bt + Cross(car.ang_vel, contact_pos_world_bt - car.pos_bt);
    const Vec3 vel2 = other.vel_bt + Cross(other.ang_vel, contact_pos_world_bt - other.pos_bt);
    const float rel_vel_bt = Dot(contact_normal_world, vel1 - vel2);
    const float denom0 = BodyImpulseDenominatorBt(
        car, cfg, mass, contact_pos_world_bt, contact_normal_world, nullptr, &car.pos_bt);
    const float denom1 = BodyImpulseDenominatorBt(
        other, other_cfg, mass, contact_pos_world_bt, contact_normal_world, nullptr, &other.pos_bt);
    if (dt <= 0.0f || denom0 + denom1 <= 0.0f) {
        return 0.0f;
    }
    const float jac_diag_ab_inv = 1.0f / (denom0 + denom1);
    const float penetration_impulse = (0.2f * -distance_bt / dt) * jac_diag_ab_inv;
    const float velocity_impulse = -rel_vel_bt * jac_diag_ab_inv;
    const float normal_impulse_bt = penetration_impulse + velocity_impulse;
    return normal_impulse_bt > 0.0f ? normal_impulse_bt : 0.0f;
}

ROCKETGPU_HD void ApplyCentralImpulseBt(CarState& car, const Vec3& impulse_bt, float car_mass) {
    const float inv_mass = 1.0f / car_mass;
    if (!car.vel_touched_solve_stage && !car.is_on_ground) {
        SetCarLinearVelocityBt(car, car.vel_bt + impulse_bt * inv_mass);
        return;
    }
    const bool was_touched = car.vel_touched_solve_stage != 0;
    SetCarLinearVelocityBt(car, car.vel_bt + impulse_bt * inv_mass);
    if (was_touched || car.is_on_ground) {
        car.vel_touched_solve_stage = 1;
    }
}

ROCKETGPU_HD void ApplyCentralImpulseViaBtRoundtrip(CarState& car, const Vec3& delta_vel_world_uu, float car_mass) {
    ApplyCentralImpulseBt(car, ToBtLinearVec(delta_vel_world_uu) * car_mass, car_mass);
}

ROCKETGPU_HD Vec3 GameJumpUpScaledBt(const CarState& car, float scale_uu) {
    Vec3 f, r, u;
    GameUeQuatAxes(car.rot, f, r, u);
    return Vec3((scale_uu * u.x) * 0.02f, (scale_uu * u.y) * 0.02f, (scale_uu * u.z) * 0.02f);
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
    Vec3& out_row2,
    const CarFrame* frame = nullptr) {
    (void)frame;
    const CarInvInertiaRowsUu rows = MakeCarInvInertiaRowsUu(car, cfg, mass);
    out_row0 = rows.row0;
    out_row1 = rows.row1;
    out_row2 = rows.row2;
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

#if defined(ROCKETGPU_HOST_LANES)
inline void InverseMatrix3x3BulletStyleColumns(const Vec3& r0, const Vec3& r1, const Vec3& r2, lanes::F4& c0,
                                               lanes::F4& c1, lanes::F4& c2) {
    using namespace lanes;
    const F4 a = Load(r0), b = Load(r1), c = Load(r2);
    const F4 a_yzx = Turn<1, 2, 0>(a), a_zxy = Turn<2, 0, 1>(a);
    const F4 b_yzx = Turn<1, 2, 0>(b), b_zxy = Turn<2, 0, 1>(b);
    const F4 c_yzx = Turn<1, 2, 0>(c), c_zxy = Turn<2, 0, 1>(c);
    const F4 x0 = b_yzx * c_zxy - b_zxy * c_yzx;
    const F4 x1 = c_yzx * a_zxy - c_zxy * a_yzx;
    const F4 x2 = a_yzx * b_zxy - a_zxy * b_yzx;
    const F4 p = a * x0;
    const float det = (p[0] + p[1]) + p[2];
    const F4 s = Splat((det != 0.0f) ? (1.0f / det) : 0.0f);
    c0 = x0 * s;
    c1 = x1 * s;
    c2 = x2 * s;
}

inline Vec3 Mul3x3ColumnsVec(lanes::F4 c0, lanes::F4 c1, lanes::F4 c2, const Vec3& v) {
    using namespace lanes;
    return ToVec3((c0 * Splat(v.x) + c1 * Splat(v.y)) + c2 * Splat(v.z));
}
#endif

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
    if (car.restored_inv_inertia_valid) {
        out_row0 = car.restored_inv_inertia_bt[0];
        out_row1 = car.restored_inv_inertia_bt[1];
        out_row2 = car.restored_inv_inertia_bt[2];
        return;
    }
    BuildInvInertiaTensorWorldBtFromColumns(
        car.tick_start_basis_forward, car.tick_start_basis_right, car.tick_start_basis_up, cfg, mass, out_row0, out_row1,
        out_row2);
}

ROCKETGPU_HD void BuildPreSolveInvInertiaTensorWorldBt(
    const CarState& car,
    const WheelRuntimeState& runtime,
    const CarConfig& cfg,
    float mass,
    Vec3& out_row0,
    Vec3& out_row1,
    Vec3& out_row2) {
    if (car.restored_inv_inertia_valid) {
        BuildInvInertiaTensorWorldBt(car, cfg, mass, out_row0, out_row1, out_row2);
        return;
    }
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
    BuildInvInertiaTensorWorldBtFromColumnsWithLocalInv(
        forward_col, right_col, up_col, inv_inertia_local, out_row0, out_row1, out_row2);
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
    const Vec3& ang_accel_world,
    const CarFrame* frame = nullptr) {
#if !defined(__CUDA_ARCH__)
    if (frame) {
#if !defined(ROCKETGPU_USE_CUDA)
        Vec3* const rows = frame->inertia_world_uu_rows;
        if (!frame->has_inertia_world_uu) {
            Vec3 inv_r0, inv_r1, inv_r2;
            if (frame->shape != nullptr) {
                const CarInvInertiaRowsUu r = MakeCarInvInertiaRowsUuWithLocal(car, frame->shape->inv_inertia_local_uu);
                inv_r0 = r.row0;
                inv_r1 = r.row1;
                inv_r2 = r.row2;
            } else {
                BuildInvInertiaTensorWorld(car, cfg, mass, inv_r0, inv_r1, inv_r2, frame);
            }
            InverseMatrix3x3BulletStyle(inv_r0, inv_r1, inv_r2, rows[0], rows[1], rows[2]);
            frame->has_inertia_world_uu = true;
        }
        return Mul3x3Vec(rows[0], rows[1], rows[2], ang_accel_world);
#else
        if (!frame->has_inertia_world_uu) {
            Vec3 inv_r0, inv_r1, inv_r2;
            BuildInvInertiaTensorWorld(car, cfg, mass, inv_r0, inv_r1, inv_r2, frame);
            InverseMatrix3x3BulletStyle(inv_r0, inv_r1, inv_r2, frame->inertia_world_uu_row0,
                                        frame->inertia_world_uu_row1, frame->inertia_world_uu_row2);
            frame->has_inertia_world_uu = true;
        }
        return Mul3x3Vec(frame->inertia_world_uu_row0, frame->inertia_world_uu_row1, frame->inertia_world_uu_row2,
                         ang_accel_world);
#endif
    }
#else
    (void)frame;
#endif
    Vec3 inv_r0, inv_r1, inv_r2;
    BuildInvInertiaTensorWorld(car, cfg, mass, inv_r0, inv_r1, inv_r2);
    Vec3 i_r0, i_r1, i_r2;
    InverseMatrix3x3BulletStyle(inv_r0, inv_r1, inv_r2, i_r0, i_r1, i_r2);
    return Mul3x3Vec(i_r0, i_r1, i_r2, ang_accel_world);
}

ROCKETGPU_HD void QueueCentralForce(WheelRuntimeState* wheel_runtime_state, const Vec3& force_bt,
                                   const CarFrame* frame = nullptr) {
    if (!wheel_runtime_state) {
        return;
    }
    if (frame && frame->queue_forces) {
        frame->queued_force += force_bt;
        if (frame->n_force_items == 0) {
            frame->force_items[0] = force_bt;
        } else if (frame->n_force_items == 1) {
            frame->force_items[1] = force_bt;
        } else if (frame->n_force_items == 2) {
            frame->force_items[2] = force_bt;
        } else if (frame->n_force_items == 3) {
            frame->force_items[3] = force_bt;
        }
        ++frame->n_force_items;
        return;
    }
    wheel_runtime_state->accumulated_force += force_bt;
}

ROCKETGPU_HD void QueueAngularAccelWorld(
    WheelRuntimeState* wheel_runtime_state,
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    const Vec3& ang_accel_world,
    const CarFrame* frame = nullptr) {
    if (!wheel_runtime_state) {
        return;
    }
    if (car.basis_authoritative && frame && frame->queue_forces) {
        return;
    }
    const Vec3 ang_accel = ComputeBulletStyleExternalTorqueInput(car, cfg, mass, ang_accel_world, frame);
    if (frame && frame->queue_forces) {
        frame->queued_ang_accel += ang_accel;
        return;
    }
    wheel_runtime_state->accumulated_ang_accel += ang_accel;
}

ROCKETGPU_HD void QueueOfficialTorqueBt(WheelRuntimeState* wheel_runtime_state, const Vec3& torque_bt,
                                       const CarFrame* frame = nullptr) {
    if (!wheel_runtime_state) {
        return;
    }
    if (frame && frame->queue_forces) {
        frame->queued_torque_bt += torque_bt;
        if (frame->n_torque_items == 0) {
            frame->torque_items[0] = torque_bt;
        } else if (frame->n_torque_items == 1) {
            frame->torque_items[1] = torque_bt;
        } else if (frame->n_torque_items == 2) {
            frame->torque_items[2] = torque_bt;
        } else if (frame->n_torque_items == 3) {
            frame->torque_items[3] = torque_bt;
        }
        ++frame->n_torque_items;
        return;
    }
    wheel_runtime_state->accumulated_torque_bt += torque_bt;
}

ROCKETGPU_HD void QueueFirstForceAndTorqueBt(WheelRuntimeState* wheel_runtime_state, const Vec3& force_bt,
                                            const Vec3& torque_bt, const CarFrame* frame) {
    if (!wheel_runtime_state) {
        return;
    }
    if (!frame || !frame->queue_forces || frame->n_force_items > CarFrame::kQueueItems ||
        frame->n_torque_items > CarFrame::kQueueItems) {
        QueueCentralForce(wheel_runtime_state, force_bt, frame);
        QueueOfficialTorqueBt(wheel_runtime_state, torque_bt, frame);
        return;
    }
    Vec3 force = force_bt;
    if (frame->n_force_items > 0) force += frame->force_items[0];
    if (frame->n_force_items > 1) force += frame->force_items[1];
    if (frame->n_force_items > 2) force += frame->force_items[2];
    if (frame->n_force_items > 3) force += frame->force_items[3];
    Vec3 torque = torque_bt;
    if (frame->n_torque_items > 0) torque += frame->torque_items[0];
    if (frame->n_torque_items > 1) torque += frame->torque_items[1];
    if (frame->n_torque_items > 2) torque += frame->torque_items[2];
    if (frame->n_torque_items > 3) torque += frame->torque_items[3];
    frame->queued_force = force;
    frame->queued_torque_bt = torque;
    frame->n_force_items = CarFrame::kQueueItems + 1;
    frame->n_torque_items = CarFrame::kQueueItems + 1;
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

#if defined(ROCKETGPU_HOST_LANES)
inline lanes::F4* InertiaWorldBtColumns(const CarFrame* frame) {
    lanes::F4* const cols = frame->inertia_world_bt_cols;
    if (!frame->has_inertia_world_bt) {
        InverseMatrix3x3BulletStyleColumns(frame->inv_world_row0_bt, frame->inv_world_row1_bt, frame->inv_world_row2_bt,
                                           cols[0], cols[1], cols[2]);
        frame->has_inertia_world_bt = true;
    }
    return cols;
}
#endif

ROCKETGPU_HD void BuildInertiaTensorWorldBtInverse(
    const CarFrame* frame,
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    Vec3& out_row0,
    Vec3& out_row1,
    Vec3& out_row2) {
    if (frame) {
#if defined(__CUDA_ARCH__)
        InverseMatrix3x3BulletStyle(frame->inv_world_row0_bt, frame->inv_world_row1_bt, frame->inv_world_row2_bt,
                                    out_row0, out_row1, out_row2);
        return;
#else
#if defined(ROCKETGPU_HOST_LANES)
        lanes::F4* const cols = InertiaWorldBtColumns(frame);
        out_row0 = Vec3(cols[0][0], cols[1][0], cols[2][0]);
        out_row1 = Vec3(cols[0][1], cols[1][1], cols[2][1]);
        out_row2 = Vec3(cols[0][2], cols[1][2], cols[2][2]);
#elif !defined(ROCKETGPU_USE_CUDA)
        Vec3* const rows = frame->inertia_world_bt_rows;
        if (!frame->has_inertia_world_bt) {
            InverseMatrix3x3BulletStyle(frame->inv_world_row0_bt, frame->inv_world_row1_bt, frame->inv_world_row2_bt,
                                        rows[0], rows[1], rows[2]);
            frame->has_inertia_world_bt = true;
        }
        out_row0 = rows[0];
        out_row1 = rows[1];
        out_row2 = rows[2];
#else
        if (!frame->has_inertia_world_bt) {
            InverseMatrix3x3BulletStyle(frame->inv_world_row0_bt, frame->inv_world_row1_bt, frame->inv_world_row2_bt,
                                        frame->inertia_world_bt_row0, frame->inertia_world_bt_row1,
                                        frame->inertia_world_bt_row2);
            frame->has_inertia_world_bt = true;
        }
        out_row0 = frame->inertia_world_bt_row0;
        out_row1 = frame->inertia_world_bt_row1;
        out_row2 = frame->inertia_world_bt_row2;
#endif
        return;
#endif
    }
    BuildInertiaTensorWorldBtInverse(car, cfg, mass, out_row0, out_row1, out_row2);
}

#if defined(ROCKETGPU_HOST_LANES)
inline Vec3 MulInertiaWorldBt(const CarFrame* frame, const CarState& car, const CarConfig& cfg, float mass,
                              const Vec3& v) {
    if (frame != nullptr) {
        const lanes::F4* const cols = InertiaWorldBtColumns(frame);
        return Mul3x3ColumnsVec(cols[0], cols[1], cols[2], v);
    }
    Vec3 i0, i1, i2;
    BuildInertiaTensorWorldBtInverse(car, cfg, mass, i0, i1, i2);
    return Mul3x3Vec(i0, i1, i2, v);
}
#endif

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
    float dt,
    const Vec3* inv_inertia_local_uu = nullptr) {
    if (car.is_demoed || car.just_respawned || !wheel_runtime_state) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }
    const Vec3& acc = wheel_runtime_state->accumulated_ang_accel;
    if (acc.x == 0.0f && acc.y == 0.0f && acc.z == 0.0f) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }
#if defined(ROCKETGPU_HOST_LANES)
    if (inv_inertia_local_uu != nullptr && !car.restored_inv_inertia_valid) {
        using namespace lanes;
        const Vec3& f = car.tick_start_basis_forward;
        const Vec3& r = car.tick_start_basis_right;
        const Vec3& u = car.tick_start_basis_up;
        const Vec3& il = *inv_inertia_local_uu;
        const F4 F = Load(f), R = Load(r), U = Load(u);
        const F4 row0 = (F * Splat(f.x * il.x) + R * Splat(r.x * il.y)) + U * Splat(u.x * il.z);
        const F4 row1 = (F * Splat(f.y * il.x) + R * Splat(r.y * il.y)) + U * Splat(u.y * il.z);
        const F4 row2 = (F * Splat(f.z * il.x) + R * Splat(r.z * il.y)) + U * Splat(u.z * il.z);
        return ToVec3(((row0 * Splat(acc.x) + row1 * Splat(acc.y)) + row2 * Splat(acc.z)) * Splat(dt));
    }
#endif
    Vec3 inv_r0, inv_r1, inv_r2;
    if (inv_inertia_local_uu != nullptr) {
        const CarInvInertiaRowsUu rows = MakeCarInvInertiaRowsUuWithLocal(car, *inv_inertia_local_uu);
        inv_r0 = rows.row0;
        inv_r1 = rows.row1;
        inv_r2 = rows.row2;
    } else {
        BuildInvInertiaTensorWorld(car, cfg, mass, inv_r0, inv_r1, inv_r2);
    }

    const Vec3 angular_accel = VecMul3x3(acc, inv_r0, inv_r1, inv_r2);
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
    (void)deferred_linear_delta;
    (void)deferred_angular_delta;
    if (car.just_respawned) {
        return;
    }

    (void)car.vel_touched_solve_stage;
    if (car.ext_force_impulse_bt_valid && !car.is_demoed) {
        SetCarLinearVelocityBt(car, car.vel_bt + car.ext_force_impulse_bt);
        car.ang_vel += car.ext_torque_impulse;
        return;
    }
    {
        const Vec3 ext_f_bt = ComputeOfficialExternalForceImpulseBt(wheel_runtime_state, mutator, dt);
        SetCarLinearVelocityBt(car, car.vel_bt + ext_f_bt);
        {
            Vec3 extT(0.0f, 0.0f, 0.0f);
            if (wheel_runtime_state) {
                const Vec3& t_bt = wheel_runtime_state->accumulated_torque_bt;
                if (t_bt.x != 0.0f || t_bt.y != 0.0f || t_bt.z != 0.0f) {
                    Vec3 inv_r0, inv_r1, inv_r2;
                    BuildPreSolveInvInertiaTensorWorldBt(
                        car, *wheel_runtime_state, cfg, mutator.car_mass, inv_r0, inv_r1, inv_r2);
                    extT = VecMul3x3(t_bt, inv_r0, inv_r1, inv_r2) * dt;
                }
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

ROCKETGPU_HD Vec3 ApplyInverseBallInertiaWorldBtBall(
    const ::rocketgpu::BallState& ball,
    float mass,
    float radius_uu,
    const Vec3& torque_bt) {
    if (ball.restored_inv_inertia_valid) {
        return Mul3x3Vec(
            ball.restored_inv_inertia_bt[0], ball.restored_inv_inertia_bt[1], ball.restored_inv_inertia_bt[2], torque_bt);
    }
    Vec3 forward_col(1.0f, 0.0f, 0.0f);
    Vec3 right_col(0.0f, 1.0f, 0.0f);
    Vec3 up_col(0.0f, 0.0f, 1.0f);
    GetBallInertiaBasisColumns(ball, forward_col, right_col, up_col);
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
    if (mutator.ball_shape == 1 && !ball.restored_inv_inertia_valid) {
        Vec3 forward_col(1.0f, 0.0f, 0.0f);
        Vec3 right_col(0.0f, 1.0f, 0.0f);
        Vec3 up_col(0.0f, 0.0f, 1.0f);
        GetBallInertiaBasisColumns(ball, forward_col, right_col, up_col);
        return ApplyInverseBallInertiaWorldBtDiagColumns(
            forward_col, right_col, up_col, mutator.ball_local_inertia_bt, torque_bt);
    }
    return ApplyInverseBallInertiaWorldBtBall(ball, mutator.ball_mass, mutator.ball_radius, torque_bt);
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

ROCKETGPU_HD void CastStaticPlaneLikeRL(
    const Vec3& source_bt,
    const Vec3& to_bt,
    const Vec3& plane_origin_bt,
    const Vec3& plane_normal,
    float max_len,
    ArenaRayHit& best,
    bool wins_ties = false) {
    constexpr float kFltEpsilon = 1.1920929e-07f;
    const Vec3 delta = to_bt - source_bt;
    const Vec3 offset = plane_origin_bt - source_bt;
    const float denom = -(delta.y * plane_normal.y + delta.x * plane_normal.x + delta.z * plane_normal.z);
    const float numer = -(offset.y * plane_normal.y + offset.x * plane_normal.x + offset.z * plane_normal.z);
    if (!(kFltEpsilon < denom && kFltEpsilon < numer && numer <= denom)) {
        return;
    }
    const float fraction = numer / denom;
    const bool tie = wins_ties && best.hit && fraction == best.ray_fraction;
    if (!(fraction < (best.hit ? best.ray_fraction : 1.0f) || tie)) {
        return;
    }
    if (tie) {
        best.hit = 0;
    }
    const float one_minus_fraction = 1.0f - fraction;
    const Vec3 point_bt(
        source_bt.x * one_minus_fraction + to_bt.x * fraction,
        source_bt.y * one_minus_fraction + to_bt.y * fraction,
        source_bt.z * one_minus_fraction + to_bt.z * fraction);
    MaybeUpdateBestHitFraction(
        true, fraction, max_len, point_bt * 50.0f, point_bt, plane_normal, best, ROCKETGPU_ARENA_RAY_HIT_STATIC_PLANE);
}

ROCKETGPU_HD void CastArenaRayPlanesImpl(const Vec3& source_bt, const Vec3& ray_delta_bt, float max_len, ArenaRayHit& best,
    const uint32_t* native_table = nullptr, int native_shift = 0, bool neg_x_wins_mesh_tie = false) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    const Vec3 to_bt = source_bt + ray_delta_bt;
    (void)native_table;
    (void)native_shift;

    constexpr float kWallOriginZ = 0x1.3d70a4p+4f;
    if (!(to_bt.x > -ArenaExtentX() * kUUToBT)) {
        CastStaticPlaneLikeRL(source_bt, to_bt, Vec3(-ArenaExtentX() * kUUToBT, 0.0f, kWallOriginZ), Vec3(1.0f, 0.0f, 0.0f), max_len, best, neg_x_wins_mesh_tie);
    }
    if (!(to_bt.z > 0.0f)) {
        CastStaticPlaneLikeRL(source_bt, to_bt, Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f), max_len, best);
    }
    if (!(to_bt.z < 40.8f && source_bt.z < 40.8f && Abs(source_bt.y) < 200.0f && Abs(to_bt.y) < 200.0f)) {
        CastStaticPlaneLikeRL(source_bt, to_bt, Vec3(0.0f, 0.0f, ArenaHeight() * kUUToBT),
                              Vec3(-0x1.135bd8p-47f, 0x1.777a5cp-24f, -1.0f), max_len, best);
    }
    if (!(to_bt.x < 81.9f && source_bt.x < 81.9f && Abs(source_bt.z) < 100.0f && Abs(to_bt.z) < 100.0f)) {
        CastStaticPlaneLikeRL(source_bt, to_bt, Vec3(ArenaExtentX() * kUUToBT, 0.0f, kWallOriginZ),
                              Vec3(-0x1.fffffep-1f, 0.0f, 0x1.8p-23f), max_len, best);
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

ROCKETGPU_HD bool RayCastArenaMeshTriangleBtSource(
    const ArenaMeshTriangle& tri,
    int32_t tri_idx,
    const Vec3& ray_source_bt,
    const Vec3& to_bt,
    float max_len,
    float& best_fraction,
    ArenaRayHit& best,
    const ArenaMeshTriangle* local_tri = nullptr) {
    constexpr float kBTToUU = 50.0f;
    const ArenaMeshTriangle& ray_tri = local_tri ? *local_tri : tri;
    const Vec3 origin(0.0f, local_tri ? static_cast<float>(ray_tri.cal_origin_sign) * kGoalMeshOriginYBt : 0.0f, 0.0f);
    const Vec3 from = ray_source_bt - origin;
    const Vec3 to = to_bt - origin;
    const Vec3 v10 = ray_tri.b - ray_tri.a;
    const Vec3 v20 = ray_tri.c - ray_tri.a;
    Vec3 triangle_normal = Cross(v10, v20);
    const float plane_dist = Dot(ray_tri.a, triangle_normal);
    const float dist_a = Dot(triangle_normal, from) - plane_dist;
    const float dist_b = Dot(triangle_normal, to) - plane_dist;
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
    const Vec3 point_bt = (1.0f - distance) * from + distance * to;
    const Vec3 v0p = ray_tri.a - point_bt;
    const Vec3 v1p = ray_tri.b - point_bt;
    if (Dot(Cross(v0p, v1p), triangle_normal) < edge_tolerance) {
        return false;
    }

    const Vec3 v2p = ray_tri.c - point_bt;
    if (Dot(Cross(v1p, v2p), triangle_normal) < edge_tolerance) {
        return false;
    }
    if (Dot(Cross(v2p, v0p), triangle_normal) < edge_tolerance) {
        return false;
    }

    triangle_normal = Normalize(triangle_normal, ray_tri.normal);
    if (dist_a <= 0.0f) {
        triangle_normal = -triangle_normal;
    }

    const Vec3 world_point_bt = (1.0f - distance) * ray_source_bt + distance * to_bt;
    MaybeUpdateBestHitFraction(
        true,
        distance,
        max_len,
        world_point_bt * kBTToUU,
        world_point_bt,
        triangle_normal,
        best,
        ROCKETGPU_ARENA_RAY_HIT_ARENA_MESH,
        static_cast<int32_t>(tri.mesh_index),
        tri_idx);
    best_fraction = best.hit ? best.ray_fraction : 1.0f;
    return true;
}

ROCKETGPU_HD bool RayCastArenaMeshTrianglesBtSource(
    const ArenaMeshTriangle* mesh_triangles,
    int32_t mesh_triangle_count,
    const Vec3& ray_source_bt,
    const Vec3& ray_delta_bt,
    float max_len,
    ArenaRayHit& best,
    const ArenaMeshTriangle* local_triangles = nullptr) {
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
        const ArenaMeshTriangle& tri = local_triangles ? local_triangles[tri_idx] : mesh_triangles[tri_idx];
        const Vec3 origin(0.0f, local_triangles ? static_cast<float>(tri.cal_origin_sign) * kGoalMeshOriginYBt : 0.0f, 0.0f);
        const Vec3 local_min = ray_min_bt - origin, local_max = ray_max_bt - origin;
        if (tri.max.x < local_min.x || tri.min.x > local_max.x ||
            tri.max.y < local_min.y || tri.min.y > local_max.y ||
            tri.max.z < local_min.z || tri.min.z > local_max.z) {
            continue;
        }

        RayCastArenaMeshTriangleBtSource(tri, tri_idx, ray_source_bt, to_bt, max_len, best_fraction, best,
                                        local_triangles ? local_triangles + tri_idx : nullptr);
    }

    return best.hit != 0;
}

#if !defined(ROCKETGPU_USE_CUDA) && (defined(__GNUC__) || defined(__clang__))
inline bool HostRayCastArenaMeshBlocks(
    const ArenaMeshTriangle* mesh_triangles,
    int32_t mesh_triangle_count,
    const ArenaMeshBvhQuantization* mesh_bvh_quantization,
    int32_t mesh_bvh_quantization_count,
    const ArenaMeshBvhMeshInfo* mesh_bvh_meshes,
    int32_t mesh_bvh_mesh_count,
    const ArenaMeshBvhNode* mesh_bvh_nodes,
    int32_t mesh_bvh_node_count,
    const Vec3& ray_source_bt,
    const Vec3& ray_delta_bt,
    float max_len,
    ArenaRayHit& best,
    int* best_mesh,
    const ArenaMeshLeafGrid* leaf_grid,
    const ArenaMeshTriangle* local_triangles,
    int32_t* walk_hint = nullptr) {
    typedef float F4 __attribute__((vector_size(16)));
    typedef int32_t I4 __attribute__((vector_size(16)));
    typedef uint16_t U16x8 __attribute__((vector_size(16)));
    typedef int16_t I16x8 __attribute__((vector_size(16)));
    typedef int16_t I16x4 __attribute__((vector_size(8)));
    const HostStepTables* tables = leaf_grid != nullptr ? leaf_grid->host_tables : nullptr;
    if (tables == nullptr || tables->ray_blocks == nullptr || tables->meshes != mesh_bvh_meshes ||
        tables->mesh_count != mesh_bvh_mesh_count || tables->ray_block_tris != local_triangles ||
        tables->ray_block_mesh != mesh_triangles || tables->ray_grid.box_quantization != mesh_bvh_quantization ||
        mesh_triangles == nullptr || mesh_triangle_count <= 0 || max_len <= 0.0f || mesh_bvh_quantization == nullptr ||
        mesh_bvh_meshes == nullptr || mesh_bvh_nodes == nullptr || mesh_bvh_quantization_count <= 0 ||
        mesh_bvh_mesh_count <= 0 || mesh_bvh_node_count <= 0) {
        return false;
    }
    const Vec3 to_bt = ray_source_bt + ray_delta_bt;
    const Vec3 ray_min_bt(Min(ray_source_bt.x, to_bt.x), Min(ray_source_bt.y, to_bt.y), Min(ray_source_bt.z, to_bt.z));
    const Vec3 ray_max_bt(Max(ray_source_bt.x, to_bt.x), Max(ray_source_bt.y, to_bt.y), Max(ray_source_bt.z, to_bt.z));
    if (ray_max_bt.x < mesh_bvh_meshes[0].free_half_x && ray_min_bt.x > -mesh_bvh_meshes[0].free_half_x &&
        ray_max_bt.y < mesh_bvh_meshes[0].free_half_y && ray_min_bt.y > -mesh_bvh_meshes[0].free_half_y) {
        return true;
    }
    const ArenaMeshLeafGrid& grid = tables->ray_grid;
    struct Walk {
        const ArenaMeshBvhMeshInfo* info;
        int pos;
        int begin;
        int end;
        uint16_t qmin[3];
        uint16_t qmax[3];
    };
    constexpr int kMaxWalks = 4;
    Walk walks[kMaxWalks];
    int walk_count = 0;
    for (uint32_t candidates = tables->MeshCandidates(ray_min_bt, ray_max_bt); candidates != 0u;
         candidates &= candidates - 1u) {
        const int pos = CountTrailingZeros32(candidates);
        if (pos >= mesh_bvh_mesh_count) {
            break;
        }
        const int ray_info = mesh_bvh_meshes[pos].ray_order;
        const int info_idx = ray_info >= 0 ? ray_info : pos;
        const ArenaMeshBvhMeshInfo& mesh_info = mesh_bvh_meshes[info_idx];
        if (mesh_info.aabb_max.x < ray_min_bt.x || mesh_info.aabb_min.x > ray_max_bt.x ||
            mesh_info.aabb_max.y < ray_min_bt.y || mesh_info.aabb_min.y > ray_max_bt.y ||
            mesh_info.aabb_max.z < ray_min_bt.z || mesh_info.aabb_min.z > ray_max_bt.z) {
            continue;
        }
        if (mesh_info.mesh_index < 0 || mesh_info.mesh_index >= mesh_bvh_quantization_count ||
            mesh_info.node_offset < 0 || mesh_info.node_count <= 0 ||
            mesh_info.node_offset + mesh_info.node_count > mesh_bvh_node_count || mesh_info.triangle_start < 0 ||
            mesh_info.triangle_count <= 0 || mesh_info.triangle_start + mesh_info.triangle_count > mesh_triangle_count) {
            continue;
        }
        const ArenaMeshLeafGridInfo& cell_info = grid.infos[info_idx];
        if (walk_count == kMaxWalks || cell_info.cell_offset < 0) {
            return false;
        }
        Walk& walk = walks[walk_count++];
        walk.info = &mesh_info;
        walk.pos = pos;
        const float origin_y = mesh_info.cal_origin_y;
        QuantizeArenaMeshBvhRayBox(mesh_bvh_quantization[mesh_info.mesh_index],
                                   Vec3(ray_source_bt.x, ray_source_bt.y - origin_y, ray_source_bt.z),
                                   Vec3(to_bt.x, to_bt.y - origin_y, to_bt.z), walk.qmin, walk.qmax);
        int cell[3];
        for (int a = 0; a < 3; ++a) {
            if (walk.qmax[a] - walk.qmin[a] > cell_info.cell_width[a]) {
                return false;
            }
            cell[a] = static_cast<int>((static_cast<uint64_t>(walk.qmin[a]) * cell_info.cell_magic[a]) >> 32);
            cell[a] = cell[a] < cell_info.cells[a] ? cell[a] : cell_info.cells[a] - 1;
        }
        const int cell_idx =
            cell_info.cell_offset + (cell[2] * cell_info.cells[1] + cell[1]) * cell_info.cells[0] + cell[0];
        walk.begin = grid.cell_start[cell_idx];
        walk.end = grid.cell_start[cell_idx + 1];
        if (walk_count == 1 && walk_hint != nullptr) {
            walk_hint[0] = cell_idx;
            walk_hint[1] = walk.begin;
        }
    }

    const auto sel = [](I4 m, F4 a, F4 b) {
        return reinterpret_cast<F4>((m & reinterpret_cast<I4>(a)) | (~m & reinterpret_cast<I4>(b)));
    };
    const auto load = [](const float* p) {
        F4 v;
        memcpy(&v, p, sizeof(v));
        return v;
    };
    const auto splat = [](float v) { return F4{v, v, v, v}; };
    const auto LaneMask4 = [](I4 m) {
#if defined(__SSE__)
        return static_cast<unsigned>(__builtin_ia32_movmskps(reinterpret_cast<F4>(m)));
#else
        const I4 bits = m & I4{1, 2, 4, 8};
        return static_cast<unsigned>(bits[0] + bits[1] + bits[2] + bits[3]);
#endif
    };

    const float cal_min_fraction_sq = best.cal_min_fraction_sq;
    float fraction = best.hit ? best.ray_fraction : 1.0f;
    bool hit = best.hit != 0;
    float hit_fraction = best.ray_fraction;
    int win = -1, win_pos = -1;
    float win_distance = 0.0f, win_dist_a = 0.0f;
    const F4 from[3] = {splat(ray_source_bt.x), splat(ray_source_bt.y), splat(ray_source_bt.z)};
    const F4 to[3] = {splat(to_bt.x), splat(to_bt.y), splat(to_bt.z)};
    for (int w = 0; w < walk_count; ++w) {
        const Walk& walk = walks[w];
        const float fraction_before = fraction;
        const bool hit_before = hit;
        const auto splat4 = [](uint16_t a, uint16_t b) { return U16x8{a, a, a, a, b, b, b, b}; };
        const U16x8 qtest[3] = {
            splat4(walk.qmax[0], walk.qmax[1]),
            splat4(walk.qmax[2], static_cast<uint16_t>(~walk.qmin[0])),
            splat4(static_cast<uint16_t>(~walk.qmin[1]), static_cast<uint16_t>(~walk.qmin[2]))};
        bool prepared = false;
        F4 box_src[3], box_inv[3], lambda_max;
        int near_hi[3];
        for (int e = walk.begin; e < walk.end; e += 4) {
            const ArenaMeshRayBlock& blk = tables->ray_blocks[e >> 2];
            U16x8 qa, qb, qc;
            memcpy(&qa, blk.quant[0], sizeof(qa));
            memcpy(&qb, blk.quant[1], sizeof(qb));
            memcpy(&qc, blk.quant[2], sizeof(qc));
            const I16x8 m = reinterpret_cast<I16x8>((qa <= qtest[0]) & (qb <= qtest[1]) & (qc <= qtest[2]));
            const I16x4 pass16 = __builtin_shufflevector(m, m, 0, 1, 2, 3) & __builtin_shufflevector(m, m, 4, 5, 6, 7);
            unsigned hits = LaneMask4(__builtin_convertvector(pass16, I4));
            if (hits == 0) {
                continue;
            }
            if (!prepared) {
                prepared = true;
                const float origin_y = walk.info->cal_origin_y;
                const ArenaMeshBvhRayLeaf ray_leaf(mesh_bvh_quantization[walk.info->mesh_index],
                                                   Vec3(ray_source_bt.x, ray_source_bt.y - origin_y, ray_source_bt.z),
                                                   Vec3(to_bt.x, to_bt.y - origin_y, to_bt.z));
                ray_leaf.Prepare();
                const float src[3] = {ray_leaf.source.x, ray_leaf.source.y, ray_leaf.source.z};
                const float inv[3] = {ray_leaf.inv_dir.x, ray_leaf.inv_dir.y, ray_leaf.inv_dir.z};
                for (int a = 0; a < 3; ++a) {
                    box_src[a] = splat(src[a]);
                    box_inv[a] = splat(inv[a]);
                    near_hi[a] = inv[a] < 0.0f;
                }
                lambda_max = splat(ray_leaf.lambda_max);
            }
            F4 tn[3], tf[3];
            for (int a = 0; a < 3; ++a) {
                const F4 lo = load(blk.lo[a]), hi = load(blk.hi[a]);
                tn[a] = ((near_hi[a] ? hi : lo) - box_src[a]) * box_inv[a];
                tf[a] = ((near_hi[a] ? lo : hi) - box_src[a]) * box_inv[a];
            }
            I4 ok = (tn[0] <= tf[1]) & (tn[1] <= tf[0]);
            const F4 t0 = sel(tn[1] <= tn[0], tn[0], tn[1]);
            const F4 t1 = sel(tf[0] <= tf[1], tf[0], tf[1]);
            ok &= (t0 <= tf[2]) & (tn[2] <= t1);
            ok &= (sel(tn[2] <= t0, t0, tn[2]) < lambda_max) & (splat(0.0f) < sel(t1 <= tf[2], t1, tf[2]));
            const F4 oy = load(blk.origin_y);
            const F4 nx = load(blk.n[0]), ny = load(blk.n[1]), nz = load(blk.n[2]);
            const F4 plane = load(blk.plane);
            const F4 fy = from[1] - oy, ty = to[1] - oy;
            const F4 dist_a = ((nx * from[0] + ny * fy) + nz * from[2]) - plane;
            const F4 dist_b = ((nx * to[0] + ny * ty) + nz * to[2]) - plane;
            ok &= ~(dist_a * dist_b >= splat(0.0f));
            const F4 proj_length = dist_a - dist_b;
            ok &= ~(proj_length == splat(0.0f));
            const F4 distance = dist_a / proj_length;
            const F4 keep = splat(1.0f) - distance;
            const F4 px = keep * from[0] + distance * to[0];
            const F4 py = keep * fy + distance * ty;
            const F4 pz = keep * from[2] + distance * to[2];
            const F4 tol = load(blk.tol);
            const F4 v0x = load(blk.a[0]) - px, v0y = load(blk.a[1]) - py, v0z = load(blk.a[2]) - pz;
            const F4 v1x = load(blk.b[0]) - px, v1y = load(blk.b[1]) - py, v1z = load(blk.b[2]) - pz;
            const F4 v2x = load(blk.c[0]) - px, v2y = load(blk.c[1]) - py, v2z = load(blk.c[2]) - pz;
            const auto edge = [&](F4 ux, F4 uy, F4 uz, F4 wx, F4 wy, F4 wz) {
                const F4 cx = uy * wz - uz * wy, cy = uz * wx - ux * wz, cz = ux * wy - uy * wx;
                return ~(((cx * nx + cy * ny) + cz * nz) < tol);
            };
            ok &= edge(v0x, v0y, v0z, v1x, v1y, v1z);
            ok &= edge(v1x, v1y, v1z, v2x, v2y, v2z);
            ok &= edge(v2x, v2y, v2z, v0x, v0y, v0z);
            hits &= LaneMask4(ok);
            while (hits != 0) {
                const int k = CountTrailingZeros32(hits);
                hits &= hits - 1;
                const float d = distance[k];
                if (d >= fraction) {
                    continue;
                }
                if (!(d < 0.0f || d > 1.0f) && !(cal_min_fraction_sq > 0.0f && d * d < cal_min_fraction_sq) &&
                    (!hit || d < hit_fraction)) {
                    hit = true;
                    hit_fraction = d;
                    win = e + k;
                    win_distance = d;
                    win_dist_a = dist_a[k];
                }
                fraction = hit ? hit_fraction : 1.0f;
            }
        }
        if (hit && (!hit_before || fraction < fraction_before)) {
            win_pos = walk.pos;
        }
    }
    if (win >= 0) {
        constexpr float kBTToUU = 50.0f;
        const ArenaMeshRayBlock& blk = tables->ray_blocks[win >> 2];
        const int k = win & 3;
        const int32_t global_tri_idx = blk.tri[k];
        const Vec3 n(blk.n[0][k], blk.n[1][k], blk.n[2][k]);
        const float len_sq = LengthSq(n);
        Vec3 triangle_normal = len_sq <= 1e-12f
            ? (local_triangles != nullptr ? local_triangles : mesh_triangles)[global_tri_idx].normal
            : n * RsqrtScalar(len_sq);
        if (win_dist_a <= 0.0f) {
            triangle_normal = -triangle_normal;
        }
        const Vec3 world_point_bt = (1.0f - win_distance) * ray_source_bt + win_distance * to_bt;
        best.hit = 1;
        best.t = win_distance * max_len;
        best.point = world_point_bt * kBTToUU;
        best.point_bt = world_point_bt;
        best.normal = triangle_normal;
        best.ray_fraction = win_distance;
        best.kind = ROCKETGPU_ARENA_RAY_HIT_ARENA_MESH;
        best.mesh_index = blk.mesh_index[k];
        best.tri_index = global_tri_idx;
    }
    if (best_mesh != nullptr && win_pos >= 0) {
        *best_mesh = win_pos;
    }
    return true;
}
#endif

template <bool kSplitHeaders = false>
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
    int mesh_first = 0, int mesh_step = 1, int* best_mesh = nullptr, const ArenaMeshLeafGrid* leaf_grid = nullptr,
    const ArenaMeshTriangle* local_triangles = nullptr) {
    constexpr bool split_headers = kSplitHeaders;
    if (mesh_triangles == nullptr || mesh_triangle_count <= 0 || max_len <= 0.0f ||
        mesh_bvh_quantization == nullptr || mesh_bvh_meshes == nullptr || mesh_bvh_nodes == nullptr ||
        mesh_bvh_quantization_count <= 0 || mesh_bvh_mesh_count <= 0 || mesh_bvh_node_count <= 0) {
        return RayCastArenaMeshTrianglesBtSource(mesh_triangles, mesh_triangle_count, ray_source_bt, ray_delta_bt, max_len, best, local_triangles);
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
    if (ray_max_bt.x < mesh_bvh_meshes[0].free_half_x && ray_min_bt.x > -mesh_bvh_meshes[0].free_half_x &&
        ray_max_bt.y < mesh_bvh_meshes[0].free_half_y && ray_min_bt.y > -mesh_bvh_meshes[0].free_half_y) {
        return best.hit != 0;
    }
    float best_fraction = best.hit ? best.ray_fraction : 1.0f;

    uint16_t quantized_query_aabb_min[3] = {0, 0, 0};
    uint16_t quantized_query_aabb_max[3] = {0, 0, 0};

    const int ray_pos_first = split_headers ? 0 : mesh_first;
    const int ray_pos_step = split_headers ? 1 : mesh_step;
    int next_ray_info = ray_pos_first < mesh_bvh_mesh_count ? mesh_bvh_meshes[ray_pos_first].ray_order : -1;
#if !defined(ROCKETGPU_USE_CUDA)
    const HostStepTables* host_tables = leaf_grid != nullptr ? leaf_grid->host_tables : nullptr;
    const bool host_tabled = !split_headers && mesh_first == 0 && mesh_step == 1 && host_tables != nullptr &&
                             host_tables->meshes == mesh_bvh_meshes && host_tables->mesh_count == mesh_bvh_mesh_count;
    const uint32_t host_meshes = host_tabled ? host_tables->MeshCandidates(ray_min_bt, ray_max_bt) : ~0u;
    const ArenaMeshLeafGrid* ray_grid =
        host_tabled && host_tables->ray_grid.infos != nullptr &&
                (host_tables->ray_grid.entry_boxes == nullptr ||
                 host_tables->ray_grid.box_quantization == mesh_bvh_quantization)
            ? &host_tables->ray_grid
            : nullptr;
#endif
#if !defined(ROCKETGPU_USE_CUDA)
    (void)next_ray_info;
    const auto host_next = [host_meshes](int pos) {
        if (pos >= 32) {
            return pos;
        }
        const uint32_t rest = host_meshes & (~0u << pos);
        return rest != 0u ? CountTrailingZeros64(rest) : 32;
    };
    for (int ray_pos = host_next(ray_pos_first); ray_pos < mesh_bvh_mesh_count;
         ray_pos = host_next(ray_pos + ray_pos_step)) {
        const int ray_info = mesh_bvh_meshes[ray_pos].ray_order;
#else
    for (int ray_pos = ray_pos_first; ray_pos < mesh_bvh_mesh_count; ray_pos += ray_pos_step) {
        const int ray_info = next_ray_info;
        if (ray_pos + ray_pos_step < mesh_bvh_mesh_count) {
            next_ray_info = mesh_bvh_meshes[ray_pos + ray_pos_step].ray_order;
        }
#endif
        const ArenaMeshBvhMeshInfo& mesh_info = mesh_bvh_meshes[ray_info >= 0 ? ray_info : ray_pos];
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
        const float bvh_origin_y = mesh_info.cal_origin_y;
        const Vec3 bvh_from(ray_source_bt.x, ray_source_bt.y - bvh_origin_y, ray_source_bt.z);
        const Vec3 bvh_to(to_bt.x, to_bt.y - bvh_origin_y, to_bt.z);
#if !defined(ROCKETGPU_USE_CUDA)
        QuantizeArenaMeshBvhRayBox(quantization, bvh_from, bvh_to, quantized_query_aabb_min, quantized_query_aabb_max);
#else
        QuantizeArenaMeshBvhPoint(quantization, MinComponents(bvh_from, bvh_to), 0, quantized_query_aabb_min);
        QuantizeArenaMeshBvhPoint(quantization, MaxComponents(bvh_from, bvh_to), 1, quantized_query_aabb_max);
#endif
        const ArenaMeshBvhRayLeaf ray_leaf(quantization, bvh_from, bvh_to);

        const ArenaMeshBvhSubtreeInfo* mesh_subtrees = nullptr;
        int32_t mesh_subtree_count = 0;
        if (mesh_bvh_subtrees != nullptr &&
            mesh_info.subtree_offset >= 0 &&
            mesh_info.subtree_count > 0 &&
            mesh_info.subtree_offset + mesh_info.subtree_count <= mesh_bvh_subtree_count) {
            mesh_subtrees = mesh_bvh_subtrees + mesh_info.subtree_offset;
            mesh_subtree_count = mesh_info.subtree_count;
        }
        SelectArenaMeshBvhDfsOrderSubtrees(
            mesh_info, mesh_bvh_subtrees, mesh_bvh_subtree_count, mesh_subtrees, mesh_subtree_count);
        const float origin_y = local_triangles != nullptr ? mesh_info.cal_origin_y : 0.0f;
        const ArenaMeshTriangle* mesh_tris = (local_triangles != nullptr && origin_y != 0.0f) ? local_triangles : mesh_triangles;
        auto leaf = [&](int32_t local_tri_idx) {
            if (local_tri_idx < 0 || local_tri_idx >= mesh_info.triangle_count) {
                return;
            }
            const int32_t global_tri_idx = mesh_info.triangle_start + local_tri_idx;
            const ArenaMeshTriangle& tri = mesh_tris[global_tri_idx];
            RayCastArenaMeshTriangleBtSource(
                tri,
                global_tri_idx,
                ray_source_bt,
                to_bt,
                max_len,
                best_fraction,
                best,
                local_triangles ? local_triangles + global_tri_idx : nullptr);
        };
        if constexpr (split_headers) {
            if (leaf_grid != nullptr && leaf_grid->infos != nullptr) {
#if defined(ROCKETGPU_USE_CUDA)
                const int grid_mesh = ray_info >= 0 ? ray_info : ray_pos;
                const auto fits = [&](const ArenaMeshLeafGridInfo& gi) {
                    return gi.cell_offset >= 0 &&
                           quantized_query_aabb_max[0] - quantized_query_aabb_min[0] <= gi.cell_width[0] &&
                           quantized_query_aabb_max[1] - quantized_query_aabb_min[1] <= gi.cell_width[1] &&
                           quantized_query_aabb_max[2] - quantized_query_aabb_min[2] <= gi.cell_width[2];
                };
                const ArenaMeshLeafGridInfo* lg = &leaf_grid->infos[grid_mesh];
                const int32_t* cell_start = leaf_grid->cell_start;
                const ArenaMeshLeafEntry* grid_entries = leaf_grid->entries;
#if defined(ROCKETGPU_USE_CUDA)
                if (leaf_grid->ray_infos != nullptr && fits(leaf_grid->ray_infos[grid_mesh])) {
                    lg = &leaf_grid->ray_infos[grid_mesh];
                    cell_start = leaf_grid->ray_cell_start;
                    grid_entries = leaf_grid->ray_entries;
                }
#endif
                const ArenaMeshLeafGridInfo& lg_info = *lg;
                if (fits(lg_info)) {
#else
                const ArenaMeshLeafGridInfo& lg_info = leaf_grid->infos[ray_info >= 0 ? ray_info : ray_pos];
                if (lg_info.cell_offset >= 0 &&
                    quantized_query_aabb_max[0] - quantized_query_aabb_min[0] <= lg_info.cell_width[0] &&
                    quantized_query_aabb_max[1] - quantized_query_aabb_min[1] <= lg_info.cell_width[1] &&
                    quantized_query_aabb_max[2] - quantized_query_aabb_min[2] <= lg_info.cell_width[2]) {
#endif
                    int cell[3];
                    for (int a = 0; a < 3; ++a) {
                        cell[a] = quantized_query_aabb_min[a] / lg_info.cell_width[a];
                        cell[a] = cell[a] < lg_info.cells[a] ? cell[a] : lg_info.cells[a] - 1;
                    }
                    const int cell_idx = lg_info.cell_offset + (cell[2] * lg_info.cells[1] + cell[1]) * lg_info.cells[0] + cell[0];
#if defined(ROCKETGPU_USE_CUDA)
                    const int e = cell_start[cell_idx];
                    const int end = cell_start[cell_idx + 1];
#else
                    const int e = leaf_grid->cell_start[cell_idx];
                    const int end = leaf_grid->cell_start[cell_idx + 1];
#endif
                    if (end - e <= 65536) {
                        for (int idx = e + mesh_first; idx < end; idx += mesh_step) {
                            ArenaMeshLeafEntry entry;
#if defined(__CUDA_ARCH__)
                            const uint4 raw = __ldg(reinterpret_cast<const uint4*>(&grid_entries[idx]));
                            memcpy(&entry, &raw, sizeof(entry));
#elif defined(ROCKETGPU_USE_CUDA)
                            entry = grid_entries[idx];
#else
                            entry = leaf_grid->entries[idx];
#endif
                            if (TestQuantizedAabbAgainstQuantizedAabb(
                                    quantized_query_aabb_min, quantized_query_aabb_max,
                                    entry.quantized_aabb_min, entry.quantized_aabb_max) &&
                                ray_leaf(entry.quantized_aabb_min, entry.quantized_aabb_max)) {
                                const float fraction_before = best_fraction;
                                const uint8_t hit_before = best.hit;
                                leaf(entry.triangle_index);
                                if (best_mesh != nullptr && best.hit && (!hit_before || best_fraction < fraction_before)) {
                                    *best_mesh = (ray_pos << 16) | (idx - e);
                                }
                            }
                        }
                        continue;
                    }
                }
            }
            if (mesh_subtree_count > 0) {
                for (int h = mesh_first; h < mesh_subtree_count; h += mesh_step) {
                    const ArenaMeshBvhSubtreeInfo& subtree = mesh_subtrees[h];
                    if (!TestQuantizedAabbAgainstQuantizedAabb(quantized_query_aabb_min, quantized_query_aabb_max,
                                                                subtree.quantized_aabb_min, subtree.quantized_aabb_max)) {
                        continue;
                    }
                    const float fraction_before = best_fraction;
                    const uint8_t hit_before = best.hit;
                    WalkArenaMeshBvhStacklessQuantizedTree(mesh_bvh_nodes, quantized_query_aabb_min,
                                                           quantized_query_aabb_max, subtree.root_node_index,
                                                           subtree.root_node_index + subtree.subtree_size, leaf,
                                                           ray_leaf);
                    if (best_mesh != nullptr && best.hit && (!hit_before || best_fraction < fraction_before)) {
                        *best_mesh = (ray_pos << 16) | h;
                    }
                }
            } else if (ray_pos % mesh_step == mesh_first) {
                const float fraction_before = best_fraction;
                const uint8_t hit_before = best.hit;
                WalkArenaMeshBvhStacklessQuantizedTree(mesh_bvh_nodes, quantized_query_aabb_min,
                                                       quantized_query_aabb_max, mesh_info.node_offset,
                                                       mesh_info.node_offset + mesh_info.node_count, leaf,
                                                       ray_leaf);
                if (best_mesh != nullptr && best.hit && (!hit_before || best_fraction < fraction_before)) {
                    *best_mesh = ray_pos << 16;
                }
            }
            continue;
        }
        const float fraction_before = best_fraction;
        const uint8_t hit_before = best.hit;

#if !defined(ROCKETGPU_USE_CUDA)
        if (ray_grid != nullptr && WalkArenaMeshLeafGrid(ray_grid, ray_info >= 0 ? ray_info : ray_pos,
                                                         quantized_query_aabb_min, quantized_query_aabb_max, leaf,
                                                         ray_leaf)) {
        } else
#endif
        if (!WalkArenaMeshLeafGrid(leaf_grid, ray_info >= 0 ? ray_info : ray_pos, quantized_query_aabb_min,
                                   quantized_query_aabb_max, leaf, ray_leaf)) {
            WalkArenaMeshBvhStacklessQuantizedTreeCacheFriendly(
                mesh_bvh_nodes,
                mesh_subtrees,
                mesh_subtree_count,
                quantized_query_aabb_min,
                quantized_query_aabb_max,
                mesh_info.node_offset,
                mesh_info.node_count,
                leaf,
                ray_leaf);
        }
        if (best_mesh != nullptr && best.hit && (!hit_before || best_fraction < fraction_before)) {
            *best_mesh = ray_pos;
        }
    }

    return best.hit != 0;
}
