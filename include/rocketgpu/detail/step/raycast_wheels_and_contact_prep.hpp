

template <typename WheelT>
ROCKETGPU_HD float WheelRealRayLengthBt(const WheelT& wheel) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    const float rest_bt = wheel.suspension_rest_length * kUUToBT;
    const float travel_cm = (wheel.max_suspension_travel * kUUToBT) * 100.0f;
    const float travel_bt = travel_cm / 100.0f;
    const float radius_bt = wheel.wheel_radius * kUUToBT;
    return ((rest_bt + travel_bt) + radius_bt) - 0.05f;
}
ROCKETGPU_HD ArenaRayHit CastArenaRayWithBtSource(
    const Vec3& source_uu,
    const Vec3& ray_dir,
    float max_len,
    const Vec3& ray_source_bt,
    const ArenaMeshTriangle* mesh_triangles = nullptr,
    int32_t mesh_triangle_count = 0,
    const ArenaMeshBvhQuantization* mesh_bvh_quantization = nullptr,
    int32_t mesh_bvh_quantization_count = 0,
    const ArenaMeshBvhMeshInfo* mesh_bvh_meshes = nullptr,
    int32_t mesh_bvh_mesh_count = 0,
    const ArenaMeshBvhNode* mesh_bvh_nodes = nullptr,
    int32_t mesh_bvh_node_count = 0,
    const ArenaMeshBvhSubtreeInfo* mesh_bvh_subtrees = nullptr,
    int32_t mesh_bvh_subtree_count = 0,
    float max_len_bt_override = -1.0f,
    const uint32_t* native_table = nullptr, int native_shift = 0,
    float cal_min_dist_sq_bt = 0.0f) {
    ArenaRayHit best;
    constexpr float kUUToBT = 1.0f / 50.0f;

    const float max_len_bt = max_len_bt_override > 0.0f ? max_len_bt_override : max_len * kUUToBT;
    const Vec3 ray_delta_bt = ray_dir * max_len_bt;
    if (cal_min_dist_sq_bt > 0.0f) {
        best.cal_min_fraction_sq = cal_min_dist_sq_bt / LengthSq((ray_source_bt + ray_delta_bt) - ray_source_bt);
    }
    bool mesh_ray_cast_bt = false;

    if (mesh_bvh_nodes != nullptr && mesh_bvh_node_count > 0) {
        mesh_ray_cast_bt = true;
        RayCastArenaMeshBvhBtSource(
            mesh_triangles,
            mesh_triangle_count,
            mesh_bvh_quantization,
            mesh_bvh_quantization_count,
            mesh_bvh_meshes,
            mesh_bvh_mesh_count,
            mesh_bvh_nodes,
            mesh_bvh_node_count,
            mesh_bvh_subtrees,
            mesh_bvh_subtree_count,
            ray_source_bt,
            ray_delta_bt,
            max_len,
            best,
            native_table,
            native_shift);
    }
    CastArenaRayPlanesImpl(ray_source_bt, ray_delta_bt, max_len, best, native_table, native_shift);

    if (!mesh_ray_cast_bt) {
        CastArenaRayCornerCylindersImpl(ray_source_bt, ray_delta_bt, max_len, best);
    }
    return best;
}

ROCKETGPU_HD Vec3 WheelRayCarHitboxHalfExtentWithBulletSafeMarginBT(const CarConfig& other_cfg);

ROCKETGPU_HD bool CastWheelRayAgainstCarHitboxGjkCast(
    const Vec3& ray_source_bt,
    const Vec3& ray_dir,
    float max_len,
    const CarState& other_car,
    const CarConfig& other_cfg,
    ArenaRayHit& out_hit,
    float max_len_bt_override);

// v3 SphereShape::perform_quad_raycast: the wheel ray is moved into ball-local space and intersected with the sphere
// analytically (nearest root in [0, dist]); the hit normal is the local hit point / radius rotated
// back to world, and the hit point is the ray interpolated at the hit fraction (ClosestQuadRayResultCallback).
ROCKETGPU_HD bool CastAnalyticRayVsSphereV3(
    const Vec3& ray_from_bt,
    const Vec3& ray_to_bt,
    const Vec3& ball_center_bt,
    float ball_radius_bt,
    const Vec3& ball_basis_forward,
    const Vec3& ball_basis_right,
    const Vec3& ball_basis_up,
    float& out_fraction,
    Vec3& out_normal) {
    const Vec3 df = ray_from_bt - ball_center_bt;
    const Vec3 dt = ray_to_bt - ball_center_bt;
    const Vec3 src(Dot(df, ball_basis_forward), Dot(df, ball_basis_right), Dot(df, ball_basis_up));
    const Vec3 tgt(Dot(dt, ball_basis_forward), Dot(dt, ball_basis_right), Dot(dt, ball_basis_up));
    const float r = ball_radius_bt;
    const float r_sq = r * r;
    const Vec3 closest(Clamp(0.0f, Min(src.x, tgt.x), Max(src.x, tgt.x)),
                       Clamp(0.0f, Min(src.y, tgt.y), Max(src.y, tgt.y)),
                       Clamp(0.0f, Min(src.z, tgt.z), Max(src.z, tgt.z)));
    if (!(Dot(closest, closest) <= r_sq)) {
        return false;
    }
    const Vec3 delta = tgt - src;
    const float dist = SqrtScalar(Dot(delta, delta));
    const Vec3 dir(delta.x / dist, delta.y / dist, delta.z / dist);
    const float b = 2.0f * Dot(src, dir);
    const float c = Dot(src, src) - r_sq;
    const float disc = b * b - 4.0f * c;
    if (!(disc >= 0.0f)) {
        return false;
    }
    const float sq = SqrtScalar(disc);
    float t0 = (-b - sq) / 2.0f;
    float t1 = (-b + sq) / 2.0f;
    if (t0 > t1) {
        const float tmp = t0; t0 = t1; t1 = tmp;
    }
    // A ray that starts inside the ball (t0 < 0 <= t1) does not hit it: Bullet's convex ray test (RocketSim v2, and
    // the game) reports no hit from inside a convex shape, while v3's far-root exit hit put a wheel on the ball
    // whenever the ball overlapped a wheel (deep kickoff hits, e.g. PartyCannon 15726 k18).
    if (!(t0 >= 0.0f && t0 <= dist)) {
        return false;
    }
    const float t = t0;
    const Vec3 hit_local = src + dir * t;
    const Vec3 n_local(hit_local.x / r, hit_local.y / r, hit_local.z / r);
    out_normal = ball_basis_forward * n_local.x + ball_basis_right * n_local.y + ball_basis_up * n_local.z;
    out_fraction = t / dist;
    return true;
}

ROCKETGPU_HD bool CastWheelRayAgainstBall(
    const Vec3& ray_source_bt,
    const Vec3& ray_dir,
    float max_len,
    const BallState& ball,
    float ball_radius,
    ArenaRayHit& out_hit,
    int32_t env_idx = -1,
    int32_t car_idx = -1,
    int32_t wheel_idx = -1,
    float max_len_bt_override = -1.0f,
    const uint32_t* native_table = nullptr, int native_shift = 0) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kBTToUU = 50.0f;
    const Vec3 ball_center_bt = ball.pos_bt;
    const float ball_radius_bt = ball_radius * kUUToBT;
    const float max_len_bt = max_len_bt_override > 0.0f ? max_len_bt_override : kUUToBT * max_len;
    {
        const Vec3 ball_delta_bt = ball_center_bt - ray_source_bt;
        const float reject_radius_bt = ball_radius_bt + max_len_bt + 1.0e-4f;
        if (Dot(ball_delta_bt, ball_delta_bt) > reject_radius_bt * reject_radius_bt) {
            return false;
        }
    }

    const Vec3 ray_from_bt = ray_source_bt;
    const Vec3 ray_to_bt = ray_source_bt + ray_dir * max_len_bt;

    Vec3 ball_basis_forward(1.0f, 0.0f, 0.0f);
    Vec3 ball_basis_right(0.0f, 1.0f, 0.0f);
    Vec3 ball_basis_up(0.0f, 0.0f, 1.0f);
    GetBallBasisColumns(ball, ball_basis_forward, ball_basis_right, ball_basis_up);

    float fraction = 0.0f;
    Vec3 normal(0.0f, 0.0f, 0.0f);
    Vec3 hit_point_bt(0.0f, 0.0f, 0.0f);
    const bool hit = CastAnalyticRayVsSphereV3(ray_from_bt, ray_to_bt, ball_center_bt, ball_radius_bt,
              ball_basis_forward, ball_basis_right, ball_basis_up, fraction, normal);

    const float normal_len2 = Dot(normal, normal);
    if (!hit || normal_len2 <= 0.0001f) {
        return false;
    }

    {
        {
            const float inv_len = 1.0f / SqrtScalar(normal_len2);
            normal = normal * inv_len;
        }
    }

    if (fraction >= 1.0f) {
        return false;
    }

    const float t_hit = fraction * max_len;
    const Vec3 interp_hit_bt = (1.0f - fraction) * ray_from_bt + fraction * ray_to_bt;
    const Vec3 hit_point_uu = interp_hit_bt * kBTToUU;

    MaybeUpdateBestHit(
        true,
        t_hit,
        hit_point_uu,
        interp_hit_bt,
        normal,
        max_len,
        out_hit,
        ROCKETGPU_ARENA_RAY_HIT_BALL);
    (void)hit_point_bt;
    return true;
}

ROCKETGPU_HD ArenaRayHit CastWheelRayAgainstCars(
    const Vec3& ray_source_bt,
    const Vec3& ray_dir,
    float max_len,
    const CarState* cars_in_env,
    const CarConfig* car_cfgs_in_env,
    int32_t cars_in_env_count,
    int32_t self_car_idx,
    int32_t env_idx = -1,
    int32_t wheel_idx = -1,
    int32_t* out_best_other_idx = nullptr,
    float max_len_bt_override = -1.0f,
    bool* out_best_no_response = nullptr,
    const uint32_t* native_table = nullptr, int native_shift = 0,

    const uint8_t* no_response_by_car = nullptr,
    float cal_min_hit_dist_sq_bt = 0.0f) {
    ArenaRayHit best;
    if (cars_in_env == nullptr || car_cfgs_in_env == nullptr || cars_in_env_count <= 0) {
        return best;
    }

    int best_other_idx = -1;
    bool best_no_response = false;

    for (int32_t i = 0; i < cars_in_env_count; ++i) {
        if (i == self_car_idx) {
            continue;
        }
        const CarState& other = cars_in_env[i];
        // Reject before the call, without the basis: the hitbox center is within |offset| of the car origin
        // (|R off| <= |off|) and the L1 norms bound the L2 ones, so this (0.01 bt looser) never rejects a ray the
        // box-center sphere test in CastWheelRayAgainstCarHitboxV3Exact keeps.
        {
            const CarConfig& other_cfg = car_cfgs_in_env[i];
            const float max_len_bt = max_len_bt_override > 0.0f ? max_len_bt_override : max_len * (1.0f / 50.0f);
            const float reach_uu = 0.5f * (Abs(other_cfg.hitbox_size.x) + Abs(other_cfg.hitbox_size.y) + Abs(other_cfg.hitbox_size.z)) +
                Abs(other_cfg.hitbox_offset.x) + Abs(other_cfg.hitbox_offset.y) + Abs(other_cfg.hitbox_offset.z);
            const float rr = reach_uu * (1.0f / 50.0f) + max_len_bt + 1.0e-2f;
            if (LengthSq(other.pos_bt - ray_source_bt) > rr * rr) {
                continue;
            }
        }

        ArenaRayHit hit;
        const bool hit_car = CastWheelRayAgainstCarHitboxGjkCast(
            ray_source_bt, ray_dir, max_len, other, car_cfgs_in_env[i], hit, max_len_bt_override);
        if (!hit_car) {
            continue;
        }

        if (cal_min_hit_dist_sq_bt > 0.0f &&
            LengthSq(hit.point_bt - ray_source_bt) < cal_min_hit_dist_sq_bt) {
            continue;
        }
        if (!best.hit || hit.t < best.t) {
            best = hit;
            best_other_idx = i;
            best_no_response = (no_response_by_car != nullptr ? no_response_by_car[i]
                                                              : other.demo_no_contact_response) != 0;
        }
    }

    if (out_best_other_idx != nullptr) {
        *out_best_other_idx = best_other_idx;
    }
    if (out_best_no_response != nullptr) {
        *out_best_no_response = best_no_response;
    }
    return best;
}

ROCKETGPU_HD WheelStepState MakeWheelStepState(const CarState& car, const CarConfig& cfg, int wheel_idx) {
    WheelStepState wheel;
    const bool front = wheel_idx < 2;
    const bool left = (wheel_idx % 2) != 0;
    Vec3 offset = front ? cfg.front_wheels_offset : cfg.back_wheels_offset;
    if (left) {
        offset.y = -offset.y;
    }
    Vec3 car_forward(1.0f, 0.0f, 0.0f);
    Vec3 car_right(0.0f, 1.0f, 0.0f);
    Vec3 car_up(0.0f, 0.0f, 1.0f);

    if (car.basis_authoritative) {
        CarBasisColumns(car, car_forward, car_right, car_up);
    } else {
        GetOfficialUploadBasisVectors(car.rot, car_forward, car_right, car_up);
    }
    wheel.hard_point_ws = car.pos + RotateViaBasis(car_forward, car_right, car_up, offset);

    {
        constexpr float kUUToBT = 1.0f / 50.0f;
        const Vec3 offset_bt = offset * kUUToBT;
        wheel.ray_source_bt = car.pos_bt + RotateViaBasis(car_forward, car_right, car_up, offset_bt);
    }
    wheel.wheel_dir_ws = -car_up;
    wheel.wheel_axle_ws = -car_right;
    wheel.wheel_radius = front ? cfg.front_wheel_radius : cfg.back_wheel_radius;
    wheel.suspension_rest_length = (front ? cfg.front_suspension_rest : cfg.back_suspension_rest) - 12.0f;

    wheel.max_suspension_travel = 14.5f;
    wheel.is_front_wheel = front ? 1 : 0;
    wheel.suspension_force_scale = front ? (36.0f - 0.25f) : (54.0f + 0.25f + 0.015f);
    wheel.contact_normal_ws = -wheel.wheel_dir_ws;
    wheel.contact_point_ws = wheel.hard_point_ws + wheel.wheel_dir_ws * (
        wheel.suspension_rest_length + wheel.max_suspension_travel + wheel.wheel_radius - WheelSuspensionSubtraction());
    return wheel;
}

ROCKETGPU_HD void SetWheelRayMiss(WheelStepState& wheel) {
    wheel.wheel_trace_length = wheel.suspension_rest_length + wheel.max_suspension_travel + wheel.wheel_radius;
    wheel.unclamped_wheel_trace_length = wheel.wheel_trace_length;
    wheel.suspension_length = wheel.suspension_rest_length + wheel.max_suspension_travel;
    wheel.suspension_relative_velocity = 0.0f;
    wheel.clipped_inv_contact_dot_suspension = 1.0f;
    wheel.extra_pushback = 0.0f;
    wheel.is_in_contact = 0;
    wheel.is_in_contact_with_world = 0;
    wheel.ground_kind = ROCKETGPU_ARENA_RAY_HIT_UNKNOWN;
    wheel.ground_car_idx = -1;
    wheel.contact_normal_ws = -wheel.wheel_dir_ws;
}

ROCKETGPU_HD void ComputeWheelRayVelocityTerms(const CarState& car, const Vec3& car_up, const Vec3& hit_point_bt,
                                           WheelStepState& wheel);

// The game (RLVehicle::rayCast, wheel setup 140f097b0) rejects wheel hits closer than (F - 1 uu) * 0.02 to the
// hardpoint, F = half the physics-asset box height. v3's live-measured constant 0.116684 bt^2 puts F at 18.08 uu for
// the Octane, whose hitbox_size.z (38.6591) is 2.5 uu taller than that box: hence hz/2 - 2.25. PartyCannon 28636 k14
// car 3 (Octane): the game keeps a wheel hit on car 5 at 0.3652 bt, which hz/2 - 1 (0.3666 bt) rejected.
ROCKETGPU_HD float CalMinWheelHitDistSqBt(const CarConfig& cfg) {
    const float d = (cfg.hitbox_size.z * 0.5f - 2.25f) * UnitsUUToBT();
    return d * d;
}

ROCKETGPU_HD void CalSuspensionForceScales(const CarConfig& cfg, float car_mass, float& front, float& back) {
    const float mass = 1.0f / (1.0f / car_mass);
    const float front_x = cfg.front_wheels_offset.x * UnitsUUToBT();
    const float back_x = -(cfg.back_wheels_offset.x * UnitsUUToBT());
    if (front_x < 0.0f || back_x < 0.0f) {
        front = back = (mass * 0.5f) * 0.5f;
    } else {
        const float back_share = (front_x / (back_x + front_x)) * mass;
        front = (mass - back_share) * 0.5f;
        back = back_share * 0.5f;
    }
}

ROCKETGPU_HD void RayCastWheelAgainstArena(
    const CarState& car,
    const CarConfig& cfg,
    WheelStepState& wheel,
    const ArenaMeshTriangle* mesh_triangles = nullptr,
    int32_t mesh_triangle_count = 0,
    const CarState* cars_in_env = nullptr,
    const CarConfig* car_cfgs_in_env = nullptr,
    int32_t cars_in_env_count = 0,
    int32_t self_car_idx = -1,
    int32_t env_idx = -1,
    int32_t wheel_idx = -1,
    const BallState* ball = nullptr,
    float ball_radius = 0.0f,
    const ArenaMeshBvhQuantization* mesh_bvh_quantization = nullptr,
    int32_t mesh_bvh_quantization_count = 0,
    const ArenaMeshBvhMeshInfo* mesh_bvh_meshes = nullptr,
    int32_t mesh_bvh_mesh_count = 0,
    const ArenaMeshBvhNode* mesh_bvh_nodes = nullptr,
    int32_t mesh_bvh_node_count = 0,
    const ArenaMeshBvhSubtreeInfo* mesh_bvh_subtrees = nullptr,
    int32_t mesh_bvh_subtree_count = 0,
    bool arena_static = true,
    const uint32_t* native_table = nullptr, int native_shift = 0,
    const uint8_t* no_response_by_car = nullptr,
    float cal_min_hit_dist_sq_bt = 0.0f) {
    (void)cfg;
    const float real_ray_length = wheel.suspension_rest_length + wheel.max_suspension_travel + wheel.wheel_radius - WheelSuspensionSubtraction();

    const float real_ray_length_bt = WheelRealRayLengthBt(wheel);

    ArenaRayHit hit = !arena_static ? ArenaRayHit{} : CastArenaRayWithBtSource(
        wheel.hard_point_ws,
        wheel.wheel_dir_ws,
        real_ray_length,
        wheel.ray_source_bt,
        mesh_triangles,
        mesh_triangle_count,
        mesh_bvh_quantization,
        mesh_bvh_quantization_count,
        mesh_bvh_meshes,
        mesh_bvh_mesh_count,
        mesh_bvh_nodes,
        mesh_bvh_node_count,
        mesh_bvh_subtrees,
        mesh_bvh_subtree_count,
        real_ray_length_bt, native_table, native_shift, cal_min_hit_dist_sq_bt);

    const auto cal_too_close = [&](const ArenaRayHit& h) {
        return cal_min_hit_dist_sq_bt > 0.0f && h.hit && LengthSq(h.point_bt - wheel.ray_source_bt) < cal_min_hit_dist_sq_bt;
    };
    bool hit_world = hit.hit != 0;
    int32_t car_hit_other_idx = -1;
    bool car_hit_no_response = false;
    ArenaRayHit car_hit = CastWheelRayAgainstCars(
        wheel.ray_source_bt,
        wheel.wheel_dir_ws,
        real_ray_length,
        cars_in_env,
        car_cfgs_in_env,
        cars_in_env_count,
        self_car_idx,
        env_idx,
        wheel_idx,
        &car_hit_other_idx,
        real_ray_length_bt,
        &car_hit_no_response, native_table, native_shift,
        no_response_by_car, cal_min_hit_dist_sq_bt);
    if (cal_too_close(car_hit)) {
        car_hit = ArenaRayHit{};
    }
    int32_t ground_car_idx = -1;
    bool selected_no_response = false;
    if (car_hit.hit && (!hit.hit || car_hit.t < hit.t)) {
        hit = car_hit;
        hit_world = false;
        ground_car_idx = car_hit_other_idx;
        selected_no_response = car_hit_no_response;
    }

    if (ball != nullptr && ball_radius > 0.0f) {
        ArenaRayHit ball_hit;
        if (CastWheelRayAgainstBall(wheel.ray_source_bt, wheel.wheel_dir_ws, real_ray_length, *ball, ball_radius, ball_hit, env_idx, self_car_idx, wheel_idx, real_ray_length_bt, native_table, native_shift)) {
            if (ball_hit.hit && !cal_too_close(ball_hit) && (!hit.hit || ball_hit.t < hit.t)) {
                hit = ball_hit;
                ground_car_idx = -1;
                hit_world = false;
                selected_no_response = false;
            }
        }
    }

    if (selected_no_response) {
        hit = ArenaRayHit{};
        hit_world = false;
        ground_car_idx = -1;
    }

    if (hit.hit && hit.kind == ROCKETGPU_ARENA_RAY_HIT_CAR) {
        hit.normal = native_table ? NativeSseSphereNormalizer{native_table, native_shift}(hit.normal)
                                  : NormalizeBulletStyle(hit.normal, hit.normal);
    }
    const Vec3 car_up = -wheel.wheel_dir_ws;

    if (hit.hit) {
        wheel.contact_point_ws = hit.point;
        wheel.contact_point_bt = hit.point_bt;

        wheel.contact_normal_ws = native_table
            ? NativeSseSphereNormalizer{native_table, native_shift}(hit.normal)
            : NormalizeBulletStyle(hit.normal, Vec3(0.0f, 0.0f, 1.0f));
        wheel.debug_hit_fraction = hit.ray_fraction;
        wheel.is_in_contact = 1;
        wheel.is_in_contact_with_world = hit_world ? 1 : 0;

        wheel.ground_kind = hit.kind;
        wheel.ground_car_idx = ground_car_idx;

        wheel.ground_static_id = hit_world
            ? (hit.kind == ROCKETGPU_ARENA_RAY_HIT_ARENA_MESH ? hit.mesh_index :
                (hit.normal.z > .5f ? 100 : hit.normal.z < -.5f ? 101 :
                    hit.normal.x > .5f ? 102 : 103)) : -1;

        const float trace_len_bt = Dot(wheel.ray_source_bt - hit.point_bt, car_up);
        const float radius_bt = wheel.wheel_radius * UnitsUUToBT();
        const float rest_bt = wheel.suspension_rest_length * UnitsUUToBT();
        const float travel_bt =
            ((wheel.max_suspension_travel * UnitsUUToBT()) * 100.0f) / 100.0f;
        wheel.unclamped_wheel_trace_length = trace_len_bt * UnitsBTToUU();
        wheel.wheel_trace_len_bt = trace_len_bt;
        float suspension_len_bt = trace_len_bt - radius_bt;
        suspension_len_bt = Clamp(suspension_len_bt, rest_bt - travel_bt, rest_bt + travel_bt);
        wheel.suspension_len_bt = suspension_len_bt;
        wheel.suspension_length = suspension_len_bt * UnitsBTToUU();
        wheel.wheel_trace_length = (suspension_len_bt + radius_bt) * UnitsBTToUU();

        ComputeWheelRayVelocityTerms(car, car_up, hit.point_bt, wheel);
    } else {
        SetWheelRayMiss(wheel);
    }
}

ROCKETGPU_HD void ComputeWheelRayVelocityTerms(const CarState& car, const Vec3& car_up, const Vec3& hit_point_bt,
                                           WheelStepState& wheel) {
        const float denominator = Dot(wheel.contact_normal_ws, car_up);

        const Vec3 relpos_bt = hit_point_bt - car.pos_bt;
        // VelocityAtPoint with vel = vel_bt, without copying the CarState.
        const Vec3 vel_at_contact_bt = car.vel_bt + Cross(car.ang_vel, relpos_bt);
        wheel.vel_at_contact = FromBtLinearVec(vel_at_contact_bt);
        const float proj_vel_bt = Dot(wheel.contact_normal_ws, vel_at_contact_bt);
        if (denominator > 0.1f) {
            wheel.clipped_inv_contact_dot_suspension = 1.0f / denominator;
            wheel.suspension_relative_velocity_bt = proj_vel_bt * wheel.clipped_inv_contact_dot_suspension;
            wheel.suspension_relative_velocity = wheel.suspension_relative_velocity_bt * UnitsBTToUU();
        } else {
            wheel.clipped_inv_contact_dot_suspension = 10.0f;
            wheel.suspension_relative_velocity = 0.0f;
            wheel.suspension_relative_velocity_bt = 0.0f;
        }
}

ROCKETGPU_HD void StoreWheelRayStage(const WheelStepState& wheel, WheelRayStage& stage) {
    stage.contact_point_ws = wheel.contact_point_ws;
    stage.contact_point_bt = wheel.contact_point_bt;
    stage.contact_normal_ws = wheel.contact_normal_ws;
    stage.vel_at_contact = wheel.vel_at_contact;
    stage.suspension_length = wheel.suspension_length;
    stage.wheel_trace_length = wheel.wheel_trace_length;
    stage.unclamped_wheel_trace_length = wheel.unclamped_wheel_trace_length;
    stage.wheel_trace_len_bt = wheel.wheel_trace_len_bt;
    stage.suspension_len_bt = wheel.suspension_len_bt;
    stage.suspension_relative_velocity_bt = wheel.suspension_relative_velocity_bt;
    stage.debug_hit_fraction = wheel.debug_hit_fraction;
    stage.suspension_relative_velocity = wheel.suspension_relative_velocity;
    stage.clipped_inv_contact_dot_suspension = wheel.clipped_inv_contact_dot_suspension;
    stage.extra_pushback = wheel.extra_pushback;
    stage.ground_car_idx = wheel.ground_car_idx;
    stage.ground_static_id = wheel.ground_static_id;
    stage.ground_kind = wheel.ground_kind;
    stage.is_in_contact = wheel.is_in_contact;
    stage.is_in_contact_with_world = wheel.is_in_contact_with_world;
}

ROCKETGPU_HD void LoadWheelRayStage(WheelStepState& wheel, const WheelRayStage& stage) {
    wheel.contact_point_ws = stage.contact_point_ws;
    wheel.contact_point_bt = stage.contact_point_bt;
    wheel.contact_normal_ws = stage.contact_normal_ws;
    wheel.vel_at_contact = stage.vel_at_contact;
    wheel.suspension_length = stage.suspension_length;
    wheel.wheel_trace_length = stage.wheel_trace_length;
    wheel.unclamped_wheel_trace_length = stage.unclamped_wheel_trace_length;
    wheel.wheel_trace_len_bt = stage.wheel_trace_len_bt;
    wheel.suspension_len_bt = stage.suspension_len_bt;
    wheel.suspension_relative_velocity_bt = stage.suspension_relative_velocity_bt;
    wheel.debug_hit_fraction = stage.debug_hit_fraction;
    wheel.suspension_relative_velocity = stage.suspension_relative_velocity;
    wheel.clipped_inv_contact_dot_suspension = stage.clipped_inv_contact_dot_suspension;
    wheel.extra_pushback = stage.extra_pushback;
    wheel.ground_car_idx = stage.ground_car_idx;
    wheel.ground_static_id = stage.ground_static_id;
    wheel.ground_kind = stage.ground_kind;
    wheel.is_in_contact = stage.is_in_contact;
    wheel.is_in_contact_with_world = stage.is_in_contact_with_world;
}

ROCKETGPU_HD Vec3 GetUpwardsDirFromWheelContacts(const CarState& car, const WheelStepState* wheels) {
    Vec3 sum(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < 4; ++i) {
        if (wheels[i].is_in_contact) {
            sum += wheels[i].contact_normal_ws;
        }
    }
    if (sum.x == 0.0f && sum.y == 0.0f && sum.z == 0.0f) {
        return CarUp(car);
    }
    constexpr float kBulletSimdEpsilon = 1.1920928955078125e-7f;
    const float len_sq = LengthSq(sum);
    if (len_sq >= kBulletSimdEpsilon * kBulletSimdEpsilon) {
        return sum * (1.0f / SqrtScalar(len_sq));
    }
    return Vec3(1.0f, 0.0f, 0.0f);
}

ROCKETGPU_HD float WheelForwardSpeed(const CarState& car) {
    Vec3 car_forward(1.0f, 0.0f, 0.0f);
    Vec3 unused_right(0.0f, 1.0f, 0.0f);
    Vec3 unused_up(0.0f, 0.0f, 1.0f);
    CarBasisColumns(car, car_forward, unused_right, unused_up);

    return Dot(car.vel_bt, car_forward) * UnitsBTToUU();
}

ROCKETGPU_HD Vec3 UpwardsDirFromWheelNormalSum(const CarState& car, const Vec3& sum) {
    if (sum.x == 0.0f && sum.y == 0.0f && sum.z == 0.0f) {
        return CarUp(car);
    }
    constexpr float kBulletSimdEpsilon = 1.1920928955078125e-7f;
    const float len_sq = LengthSq(sum);
    if (len_sq >= kBulletSimdEpsilon * kBulletSimdEpsilon) {
        return sum * (1.0f / SqrtScalar(len_sq));
    }
    return Vec3(1.0f, 0.0f, 0.0f);
}

ROCKETGPU_HD void ApplyAutoRollReference(
    CarState& car,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    int num_wheels_in_contact,
    const WheelStepState* wheels,
    float dt,
    WheelRuntimeState* wheel_runtime_state = nullptr,
    const Vec3* wheel_normal_sum_override = nullptr) {
    const Vec3 ground_up_dir = num_wheels_in_contact > 0
        ? (wheel_normal_sum_override ? UpwardsDirFromWheelNormalSum(car, *wheel_normal_sum_override)
                                     : GetUpwardsDirFromWheelContacts(car, wheels))
        : car.world_contact.contact_normal;
    const Vec3 ground_down_dir = -ground_up_dir;

    Vec3 forward_dir(1.0f, 0.0f, 0.0f);
    Vec3 right_dir(0.0f, 1.0f, 0.0f);
    Vec3 up_dir(0.0f, 0.0f, 1.0f);
    CarBasisColumns(car, forward_dir, right_dir, up_dir);
    (void)up_dir;
    const Vec3 cross_right_dir = Cross(ground_up_dir, forward_dir);
    const Vec3 cross_forward_dir = Cross(ground_down_dir, cross_right_dir);

    const float right_torque_factor = 1.0f - Clamp(Dot(right_dir, cross_right_dir), 0.0f, 1.0f);
    const float forward_torque_factor = 1.0f - Clamp(Dot(forward_dir, cross_forward_dir), 0.0f, 1.0f);

    const Vec3 torque_dir_right = forward_dir * (Dot(right_dir, ground_up_dir) >= 0.0f ? -1.0f : 1.0f);
    const Vec3 torque_dir_forward = right_dir * (Dot(forward_dir, ground_up_dir) >= 0.0f ? 1.0f : -1.0f);

    const Vec3 torque = torque_dir_forward * forward_torque_factor + torque_dir_right * right_torque_factor;

    const Vec3 autoroll_force_bt = ground_down_dir * 100.0f * UnitsUUToBT() * mutator.car_mass;
    const Vec3 autoroll_torque = torque * 80.0f;
    if (wheel_runtime_state) {
        QueueCentralForce(wheel_runtime_state, autoroll_force_bt);
        QueueAngularAccelWorld(wheel_runtime_state, car, cfg, mutator.car_mass, autoroll_torque);

        Vec3 ar_i0, ar_i1, ar_i2;
        BuildInertiaTensorWorldBtInverse(car, cfg, mutator.car_mass, ar_i0, ar_i1, ar_i2);
        const Vec3 official_torque_bt = Mul3x3Vec(ar_i0, ar_i1, ar_i2, torque) * 80.0f;
        QueueOfficialTorqueBt(wheel_runtime_state, official_torque_bt);
    } else {
        const Vec3 autoroll_force = autoroll_force_bt * UnitsBTToUU();
        ApplyCentralForce(car, mutator.car_mass, autoroll_force, dt);
        ApplyAngularAccelWorld(car, autoroll_torque, dt);
    }
}

// One wheel's chassis velocity change in the vehicle phase. A wheel's suspension and friction impulses depend on the
// chassis rotation, position and that wheel only, so each wheel (or GPU lane) computes its own, and
// CombineWheelImpulses applies them in wheel order 0..3, bit-identical to applying them wheel by wheel.
struct WheelImpulseOut {
    Vec3 dv_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 dw = Vec3(0.0f, 0.0f, 0.0f);
    int32_t applied = 0;
};

// ApplyImpulseAtPointBtRoundtripBtInputs split into its per-wheel part (this) and the chassis update (the combiner).
ROCKETGPU_HD WheelImpulseOut WheelImpulseAtPointBt(
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    const Vec3& impulse_world_bt,
    const Vec3& rel_point_world_bt) {
    WheelImpulseOut out;
    const float official_inv_mass = 1.0f / mass;
    out.dv_bt = impulse_world_bt * official_inv_mass;
    out.dw = ApplyInverseBodyInertiaWorldBtRowDot(car, cfg, mass, Cross(rel_point_world_bt, impulse_world_bt));
    out.applied = 1;
    return out;
}

ROCKETGPU_HD void CombineWheelImpulses(CarState& car, const WheelImpulseOut out[4]) {
    for (int i = 0; i < 4; ++i) {
        if (!out[i].applied) {
            continue;
        }
        SetCarLinearVelocityBt(car, car.vel_bt + out[i].dv_bt);
        car.vel_touched_solve_stage = 1;
        car.ang_vel += out[i].dw;
    }
}

ROCKETGPU_HD WheelImpulseOut WheelSuspensionPhase(
    const CarState& car,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    WheelStepState& wheel,
    float dt) {
    WheelImpulseOut out;
    if (!wheel.is_in_contact) {
        wheel.wheels_suspension_force = 0.0f;
        wheel.impulse = Vec3(0.0f, 0.0f, 0.0f);
        return out;
    }

    const float force_bt =
        (wheel.suspension_rest_length * UnitsUUToBT() - wheel.suspension_len_bt) *
        wheel.suspension_stiffness *
        wheel.clipped_inv_contact_dot_suspension;
    const float damping = wheel.suspension_relative_velocity < 0.0f ? wheel.wheels_damping_compression : wheel.wheels_damping_relaxation;
    float suspension_force_bt =
        (force_bt - damping * wheel.suspension_relative_velocity_bt) *
        wheel.suspension_force_scale;
    if (suspension_force_bt < 0.0f) {
        suspension_force_bt = 0.0f;
    }
    wheel.wheels_suspension_force = suspension_force_bt * UnitsBTToUU();

    if (wheel.wheels_suspension_force > 0.0f) {
        const Vec3 rel_bt = wheel.contact_point_bt - car.pos_bt;

        const float base_force_scale_bt = (suspension_force_bt * dt) + wheel.extra_pushback;
        const Vec3 impulse_bt = wheel.contact_normal_ws * base_force_scale_bt;

        out = WheelImpulseAtPointBt(car, cfg, mutator.car_mass, impulse_bt, rel_bt);
    }
    return out;
}

// The wheel's contact friction impulse (from the chassis velocity after all suspension impulses) and its velocity change.
ROCKETGPU_HD WheelImpulseOut WheelFrictionPhase(
    const CarState& car,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    WheelStepState& wheel,
    float dt);

ROCKETGPU_HD void ApplyWheelSuspensionAndFriction(
    CarState& car,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    const CarControls& controls,
    float dt,
    WheelStepState* wheels) {
    WheelImpulseOut out[4];
    for (int i = 0; i < 4; ++i) {
        out[i] = WheelSuspensionPhase(car, cfg, mutator, wheels[i], dt);
    }
    CombineWheelImpulses(car, out);
    for (int i = 0; i < 4; ++i) {
        out[i] = WheelFrictionPhase(car, cfg, mutator, wheels[i], dt);
    }
    CombineWheelImpulses(car, out);
    (void)controls;
}

ROCKETGPU_HD void ComputeWheelTransformBasisFromSteer(
    const WheelStepState& wheel,
    float steer_angle,
    Vec3& out_right_col,
    Vec3& out_forward_col,
    Vec3& out_up_col,
    Vec3* out_debug_steering = nullptr,
    bool soccar_platform_math = false,
    bool native_sse_math = false) {
    const Vec3 up = -wheel.wheel_dir_ws;
    const Vec3 right = wheel.wheel_axle_ws;
    const Vec3 forward = Normalize(Cross(up, right), Vec3(1.0f, 0.0f, 0.0f));

    const float axis_len = Length(up);
    const float half_angle = steer_angle * 0.5f;

#if defined(__CUDA_ARCH__) && defined(_WIN32)
    const float sine = soccar_platform_math
        ? static_cast<float>(sin(static_cast<double>(half_angle))) : RocketGpuSinf(half_angle);
#else
    (void)soccar_platform_math;
    const float sine = RocketGpuSinf(half_angle);
#endif
    const float sin_scale = sine / axis_len;
    const Quat steering_rot(
        RocketGpuCosf(half_angle), up.x * sin_scale, up.y * sin_scale, up.z * sin_scale);
    Vec3 mat_col0(1.0f, 0.0f, 0.0f);
    Vec3 mat_col1(0.0f, 1.0f, 0.0f);
    Vec3 mat_col2(0.0f, 0.0f, 1.0f);

    if (native_sse_math) {
        GetBasisVectorsSetRotationForm(steering_rot, mat_col0, mat_col1, mat_col2);
    } else {
        GetBasisVectorsBulletQuatDot(steering_rot, mat_col0, mat_col1, mat_col2);
    }
    if (out_debug_steering != nullptr) {
        out_debug_steering[0] = Vec3(mat_col0.x, mat_col1.x, mat_col2.x);
        out_debug_steering[1] = Vec3(mat_col0.y, mat_col1.y, mat_col2.y);
        out_debug_steering[2] = Vec3(mat_col0.z, mat_col1.z, mat_col2.z);
        out_debug_steering[3] = Vec3(steering_rot.x, steering_rot.y, steering_rot.z);
        out_debug_steering[4] = Vec3(steering_rot.w, 0.0f, 0.0f);
    }
    const Vec3 neg_right = -right;
    out_right_col = Vec3(
        mat_col0.x * neg_right.x + mat_col1.x * neg_right.y + mat_col2.x * neg_right.z,
        mat_col0.y * neg_right.x + mat_col1.y * neg_right.y + mat_col2.y * neg_right.z,
        mat_col0.z * neg_right.x + mat_col1.z * neg_right.y + mat_col2.z * neg_right.z);
    out_forward_col = Vec3(
        mat_col0.x * forward.x + mat_col1.x * forward.y + mat_col2.x * forward.z,
        mat_col0.y * forward.x + mat_col1.y * forward.y + mat_col2.y * forward.z,
        mat_col0.z * forward.x + mat_col1.z * forward.y + mat_col2.z * forward.z);
    out_up_col = up;
}

ROCKETGPU_HD void ComputeWheelBasisFromSteer(
    const WheelStepState& wheel,
    float steer_angle,
    const Vec3& fallback_forward,
    Vec3& out_axle_dir,
    Vec3& out_forward_dir,
    Vec3* out_debug_steering = nullptr,
    bool soccar_platform_math = false,
    bool native_sse_math = false,
    Vec3* out_raw_axle_dir = nullptr) {
    Vec3 world_right_col(0.0f, 0.0f, 0.0f);
    Vec3 world_forward_col(0.0f, 0.0f, 0.0f);
    Vec3 world_up_col(0.0f, 0.0f, 1.0f);
    ComputeWheelTransformBasisFromSteer(wheel, steer_angle, world_right_col, world_forward_col, world_up_col, out_debug_steering, soccar_platform_math, native_sse_math);

    Vec3 axle_dir = world_right_col;
    const float axle_proj = Dot(axle_dir, wheel.contact_normal_ws);
    axle_dir = Normalize(axle_dir - wheel.contact_normal_ws * axle_proj, axle_dir);
    const Vec3 forward_dir = Normalize(Cross(wheel.contact_normal_ws, axle_dir), world_forward_col);
    out_axle_dir = axle_dir;
    out_forward_dir = LengthSq(forward_dir) > 0.0f ? forward_dir : fallback_forward;
    if (out_raw_axle_dir != nullptr) {
        *out_raw_axle_dir = world_right_col;
    }
}

struct BilateralBodyBt {
    Vec3 com_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 lin_vel_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 ang_vel = Vec3(0.0f, 0.0f, 0.0f);

    Vec3 world2_row0 = Vec3(1.0f, 0.0f, 0.0f);
    Vec3 world2_row1 = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 world2_row2 = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 inv_inertia_local = Vec3(0.0f, 0.0f, 0.0f);
    float inv_mass = 0.0f;
};

ROCKETGPU_HD Vec3 BilateralWorld2Mul(const BilateralBodyBt& body, const Vec3& v) {
    return Vec3(
        Dot(body.world2_row0, v),
        Dot(body.world2_row1, v),
        Dot(body.world2_row2, v));
}

ROCKETGPU_HD Vec3 BilateralVelocityInLocalPoint(const BilateralBodyBt& body, const Vec3& rel_pos) {
    return body.lin_vel_bt + Cross(body.ang_vel, rel_pos);
}

ROCKETGPU_HD BilateralBodyBt MakeBilateralCarBodyBt(const CarState& car, const CarConfig& cfg, float mass) {
    BilateralBodyBt body;

    body.com_bt = car.pos_bt;
    body.lin_vel_bt = car.vel_bt;
    body.ang_vel = car.ang_vel;
    Vec3 forward(1.0f, 0.0f, 0.0f);
    Vec3 right(0.0f, 1.0f, 0.0f);
    Vec3 up(0.0f, 0.0f, 1.0f);
    CarBasisColumns(car, forward, right, up);

    body.world2_row0 = forward;
    body.world2_row1 = right;
    body.world2_row2 = up;
    const Vec3 inertia_bt = BoxBodyInertiaDiagBt(cfg, mass);

    body.inv_inertia_local = Vec3(
        inertia_bt.x != 0.0f ? 1.0f / inertia_bt.x : 0.0f,
        inertia_bt.y != 0.0f ? 1.0f / inertia_bt.y : 0.0f,
        inertia_bt.z != 0.0f ? 1.0f / inertia_bt.z : 0.0f);
    body.inv_mass = mass != 0.0f ? 1.0f / mass : 0.0f;
    return body;
}

ROCKETGPU_HD float ResolveSingleBilateralBt(
    const BilateralBodyBt& body1,
    const BilateralBodyBt& body2,
    const Vec3& pos_bt,
    const Vec3& normal,
    float& out_rel_vel_bt) {
    const Vec3 rel_pos1 = pos_bt - body1.com_bt;
    const Vec3 rel_pos2 = pos_bt - body2.com_bt;
    const Vec3 vel1 = BilateralVelocityInLocalPoint(body1, rel_pos1);
    const Vec3 vel2 = BilateralVelocityInLocalPoint(body2, rel_pos2);
    const Vec3 vel = vel1 - vel2;

    const Vec3 a_j = BilateralWorld2Mul(body1, Cross(rel_pos1, normal));
    const Vec3 b_j = BilateralWorld2Mul(body2, Cross(rel_pos2, -normal));
    const Vec3 minv_jt0 = Hadamard(body1.inv_inertia_local, a_j);
    const Vec3 minv_jt1 = Hadamard(body2.inv_inertia_local, b_j);
    const float a_diag =
        body1.inv_mass + Dot(minv_jt0, a_j) + body2.inv_mass + Dot(minv_jt1, b_j);

    out_rel_vel_bt = Dot(normal, vel);
    if (a_diag <= 0.0f) {
        return 0.0f;
    }
    const float jac_diag_ab_inv = 1.0f / a_diag;
    return -0.2f * out_rel_vel_bt * jac_diag_ab_inv;
}

ROCKETGPU_HD WheelImpulseOut WheelFrictionPhase(
    const CarState& car,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    WheelStepState& wheel,
    float dt) {
    WheelImpulseOut out;
    if (!wheel.is_in_contact) {
        return out;
    }
    const BilateralBodyBt body1 = MakeBilateralCarBodyBt(car, cfg, mutator.car_mass);
    const BilateralBodyBt fixed_body;
    float lateral_speed_bt = 0.0f;
    const float side_impulse = ResolveSingleBilateralBt(
        body1, fixed_body, wheel.contact_point_bt, wheel.cal_axle_dir, lateral_speed_bt);
    wheel.impulse = ((wheel.cal_forward_dir * wheel.cal_rolling_friction) * wheel.cal_long_friction +
                     (wheel.cal_axle_dir * side_impulse) * wheel.cal_lat_friction) * wheel.cal_friction_scale;
    if (LengthSq(wheel.impulse) == 0.0f) {
        return out;
    }

    Vec3 car_up(0.0f, 0.0f, 1.0f);
    Vec3 unused_forward(1.0f, 0.0f, 0.0f);
    Vec3 unused_right(0.0f, 1.0f, 0.0f);
    CarBasisColumns(car, unused_forward, unused_right, car_up);
    const Vec3 friction_rel_bt = wheel.contact_point_bt - car.pos_bt;
    const Vec3 friction_wheel_rel_pos_bt = friction_rel_bt - car_up * Dot(car_up, friction_rel_bt);
    const Vec3 friction_impulse_bt = wheel.impulse * dt;
    return WheelImpulseAtPointBt(car, cfg, mutator.car_mass, friction_impulse_bt, friction_wheel_rel_pos_bt);
}

// True when the wheel ray from hard_point along the car's down axis meets another car's hitbox on a face whose
// outward normal is the recorded wheel contact normal: a game wheel contact on that car, not on the world. The
// recorded normal is the previous tick's, so the box is grown by one tick of the two cars' relative motion; a ray that
// only grazes the grown box (the game's ray hit near an edge, where Bullet's convex ray cast returns a normal between
// the faces: PartyCannon 11445 k67 car 5 on car 4) counts when the normal points out of the box towards the ray.
ROCKETGPU_HD bool SetWheelNormalIsOnOtherCar(const CarState& car, const CarConfig& cfg, const Vec3& hard_point,
                                            const Vec3& normal, const CarState* cars, const CarConfig* cfgs,
                                            int32_t count, int32_t self_idx) {
    if (cfgs == nullptr) {
        return false;
    }
    Vec3 f(1.0f, 0.0f, 0.0f), r(0.0f, 1.0f, 0.0f), u(0.0f, 0.0f, 1.0f);
    CarBasisColumns(car, f, r, u);
    const Vec3 dir = -u;
    const float max_len = 2.0f * (Max(cfg.front_suspension_rest, cfg.back_suspension_rest) +
                                  Max(cfg.front_wheel_radius, cfg.back_wheel_radius));
    for (int32_t j = 0; j < count; ++j) {
        if (j == self_idx || cars[j].is_demoed) {
            continue;
        }
        Vec3 ax[3];
        CarBasisColumns(cars[j], ax[0], ax[1], ax[2]);
        const Vec3 half = BulletBoxHalfExtentWithMargin(cfgs[j]);
        const float grow = Length(car.vel - cars[j].vel) * (1.0f / 120.0f);
        const float h[3] = {half.x, half.y, half.z};
        const Vec3 center = cars[j].pos + RotateCarBasis(cars[j], cfgs[j].hitbox_offset);
        const Vec3 d = hard_point - center;
        float t_min = 0.0f, t_max = max_len;
        int face = -1;
        float face_sign = 0.0f;
        bool miss = false;
        for (int k = 0; k < 3 && !miss; ++k) {
            const float o = Dot(d, ax[k]);
            const float v = Dot(dir, ax[k]);
            const float hk = h[k] + grow;
            if (Abs(v) < 1.0e-8f) {
                miss = Abs(o) > hk;
                continue;
            }
            float t0 = (-hk - o) / v, t1 = (hk - o) / v;
            float s0 = -1.0f;
            if (t0 > t1) {
                const float tmp = t0; t0 = t1; t1 = tmp;
                s0 = 1.0f;
            }
            if (t0 > t_min) {
                t_min = t0;
                face = k;
                face_sign = s0;
            }
            t_max = Min(t_max, t1);
            miss = t_min > t_max;
        }
        if (miss || face < 0) {
            continue;
        }
        if (Dot(ax[face] * face_sign, normal) > 0.99f) {
            return true;
        }
        // Outward direction from the unexpanded box to the ray's entry point into the grown box.
        const Vec3 p = d + dir * t_min;
        Vec3 out(0.0f, 0.0f, 0.0f);
        for (int k = 0; k < 3; ++k) {
            const float o = Dot(p, ax[k]);
            out += ax[k] * (o - Clamp(o, -h[k], h[k]));
        }
        const float out_len = Length(out);
        if (out_len > 1.0e-4f && Dot(out, normal) > 0.9f * out_len) {
            return true;
        }
    }
    return false;
}

// The previous tick's world wheel contacts a set state supplied (CarState::set_wheel_normals_mask): sums their
// normals into normal_sum and returns their count. The recorded contacts include wheels on the ball, whose normal
// points from the ball centre to the wheel, and wheels resting on another car; the game's sticky force and auto-roll
// count neither (as v3's world-contact gate and the step's own raycast do).
// A wheel contact on the world has its surface within a wheel ray of the hard point, so a ray from the hard point
// against the normal meets the arena on a face with that normal. With the ball or another car in reach and no such
// face, the contact was on that body; this catches the ones the ball and car tests miss because the bodies have moved
// apart by the set pose (RLPR PartyCannon 3187 k50 wheel on the ball, 8575 k30 wheel on a car: the game applies no
// sticky force). A teleported car keeps its contacts (no body in reach), as the game does (cb_aerial, cw_ceiling).
ROCKETGPU_HD bool SetWheelNormalHasWorldFace(const WheelStepState& wheel, const Vec3& normal,
                                             const ArenaMeshTriangle* mesh_triangles, int32_t mesh_triangle_count,
                                             const ArenaMeshBvhQuantization* mesh_bvh_quantization,
                                             int32_t mesh_bvh_quantization_count,
                                             const ArenaMeshBvhMeshInfo* mesh_bvh_meshes, int32_t mesh_bvh_mesh_count,
                                             const ArenaMeshBvhNode* mesh_bvh_nodes, int32_t mesh_bvh_node_count,
                                             const ArenaMeshBvhSubtreeInfo* mesh_bvh_subtrees,
                                             int32_t mesh_bvh_subtree_count) {
    const float len = wheel.suspension_rest_length + wheel.max_suspension_travel + wheel.wheel_radius + 20.0f;
    const ArenaRayHit hit = CastArenaRayWithBtSource(
        wheel.hard_point_ws, -normal, len, wheel.ray_source_bt, mesh_triangles, mesh_triangle_count,
        mesh_bvh_quantization, mesh_bvh_quantization_count, mesh_bvh_meshes, mesh_bvh_mesh_count, mesh_bvh_nodes,
        mesh_bvh_node_count, mesh_bvh_subtrees, mesh_bvh_subtree_count);
    return hit.hit && Dot(hit.normal, normal) > 0.9f * Length(hit.normal);
}

ROCKETGPU_HD int SetStateWorldWheelNormals(const CarState& car, const CarConfig& cfg, const WheelStepState* wheels,
                                           const BallState* ball, float ball_radius, const CarState* cars,
                                           const CarConfig* cfgs, int32_t count, int32_t self_idx, Vec3& normal_sum,
                                           const ArenaMeshTriangle* mesh_triangles = nullptr,
                                           int32_t mesh_triangle_count = 0,
                                           const ArenaMeshBvhQuantization* mesh_bvh_quantization = nullptr,
                                           int32_t mesh_bvh_quantization_count = 0,
                                           const ArenaMeshBvhMeshInfo* mesh_bvh_meshes = nullptr,
                                           int32_t mesh_bvh_mesh_count = 0,
                                           const ArenaMeshBvhNode* mesh_bvh_nodes = nullptr,
                                           int32_t mesh_bvh_node_count = 0,
                                           const ArenaMeshBvhSubtreeInfo* mesh_bvh_subtrees = nullptr,
                                           int32_t mesh_bvh_subtree_count = 0) {
    normal_sum = Vec3(0.0f, 0.0f, 0.0f);
    int n = 0;
    for (int i = 0; i < 4; ++i) {
        if (!((car.set_wheel_normals_mask >> i) & 1)) {
            continue;
        }
        const Vec3& hp = wheels[i].hard_point_ws;
        const float reach = wheels[i].suspension_rest_length + wheels[i].max_suspension_travel +
                            wheels[i].wheel_radius + 30.0f;
        bool body_in_reach = ball != nullptr && Length(hp - ball->pos) < ball_radius + reach;
        for (int32_t j = 0; cars != nullptr && cfgs != nullptr && j < count && !body_in_reach; ++j) {
            if (j != self_idx && !cars[j].is_demoed) {
                const Vec3 center = cars[j].pos + RotateCarBasis(cars[j], cfgs[j].hitbox_offset);
                body_in_reach = Length(hp - center) < Length(BulletBoxHalfExtentWithMargin(cfgs[j])) + reach;
            }
        }
        if (body_in_reach &&
            !SetWheelNormalHasWorldFace(wheels[i], car.set_wheel_normals[i], mesh_triangles, mesh_triangle_count,
                                        mesh_bvh_quantization, mesh_bvh_quantization_count, mesh_bvh_meshes,
                                        mesh_bvh_mesh_count, mesh_bvh_nodes, mesh_bvh_node_count, mesh_bvh_subtrees,
                                        mesh_bvh_subtree_count)) {
            continue;
        }
        if (ball != nullptr) {
            const Vec3 to_wheel = wheels[i].hard_point_ws - ball->pos;
            const float dist = Length(to_wheel);
            if (dist < ball_radius + 80.0f && dist > 0.0f && Dot(car.set_wheel_normals[i], to_wheel) > 0.95f * dist) {
                continue;
            }
        }
        // The wheel's ray at the set pose lands on another car (v3's set-state refresh raycast, which its own gate
        // reads, counts no world contact there).
        if (wheels[i].is_in_contact && wheels[i].ground_kind == ROCKETGPU_ARENA_RAY_HIT_CAR) {
            continue;
        }
        if (cars != nullptr && SetWheelNormalIsOnOtherCar(car, cfg, wheels[i].hard_point_ws, car.set_wheel_normals[i],
                                                          cars, cfgs, count, self_idx)) {
            continue;
        }
        normal_sum += car.set_wheel_normals[i];
        ++n;
    }
    return n;
}

// The drive, brake, steer and friction-curve setup for one wheel (wheel index i), from the chassis state and the
// per-car values in WheelDriveShared. It writes only this wheel and its runtime slots.
struct WheelDriveShared {
    Vec3 car_forward = Vec3(1.0f, 0.0f, 0.0f);
    float next_steer_angle = 0.0f;
    float drive_engine_force = 0.0f;
    float drive_brake_force = 0.0f;
    float real_throttle = 0.0f;
    float friction_scale = 0.0f;
};

ROCKETGPU_HD void WheelDriveSetupPhase(
    const CarState& car,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    const WheelDriveShared& shared,
    WheelStepState& wheel,
    int i,
    WheelRuntimeState* wheel_runtime_state,
    float dt) {
    wheel.impulse = Vec3(0.0f, 0.0f, 0.0f);

    const float prev_steer_angle = wheel_runtime_state ? wheel_runtime_state->steer_angle[i] : 0.0f;
    const float prev_engine_force = wheel_runtime_state ? wheel_runtime_state->engine_force[i] : 0.0f;
    const float prev_brake = wheel_runtime_state ? wheel_runtime_state->brake[i] : 0.0f;
    const float prev_lat_friction = wheel_runtime_state ? wheel_runtime_state->lat_friction[i] : 0.0f;
    const float prev_long_friction = wheel_runtime_state ? wheel_runtime_state->long_friction[i] : 0.0f;
    wheel.prev_steer_angle = prev_steer_angle;
    wheel.prev_engine_force = prev_engine_force;
    wheel.prev_brake = prev_brake;
    wheel.prev_lat_friction = prev_lat_friction;
    wheel.prev_long_friction = prev_long_friction;

    const float current_steer_angle = wheel.is_front_wheel ? shared.next_steer_angle : 0.0f;

    const float applied_steer_angle = current_steer_angle;
    const float applied_engine_force = shared.drive_engine_force;
    const float applied_brake = shared.drive_brake_force;
    wheel.engine_force = shared.drive_engine_force;
    wheel.brake = shared.drive_brake_force;
    wheel.steer_angle = current_steer_angle;

    if (wheel_runtime_state) {
        wheel_runtime_state->engine_force[i] = shared.drive_engine_force;
        wheel_runtime_state->brake[i] = shared.drive_brake_force;
        wheel_runtime_state->steer_angle[i] = current_steer_angle;
    }

    float current_lat_friction = prev_lat_friction;
    float current_long_friction = prev_long_friction;
    float base_friction = 0.0f;
    float longitudinal_friction = 0.0f;
    float friction_curve_input = 0.0f;
    Vec3 friction_curve_lat_dir(0.0f, -1.0f, 0.0f);
    Vec3 friction_curve_long_dir(1.0f, 0.0f, 0.0f);
    Vec3 friction_curve_cross_vec(0.0f, 0.0f, 0.0f);
    if (wheel.is_in_contact) {
        Vec3 basis_right_col(0.0f, 0.0f, 0.0f);
        Vec3 basis_forward_col(0.0f, 0.0f, 0.0f);
        Vec3 basis_up_col(0.0f, 0.0f, 1.0f);
        ComputeWheelTransformBasisFromSteer(wheel, applied_steer_angle, basis_right_col, basis_forward_col, basis_up_col, nullptr, mutator.game_mode == static_cast<uint8_t>(Mode::Soccar), NativeSseMath(mutator));

        const Vec3 wheel_delta_bt = wheel.ray_source_bt - car.pos_bt;
        const Vec3 cross_vec = (Cross(car.ang_vel, wheel_delta_bt) + car.vel_bt) * 50.0f;
        friction_curve_cross_vec = cross_vec;
        const Vec3 lat_dir = basis_right_col;
        const Vec3 long_dir = Cross(lat_dir, wheel.contact_normal_ws);
        friction_curve_lat_dir = lat_dir;
        friction_curve_long_dir = long_dir;
        base_friction = Abs(Dot(cross_vec, lat_dir));
        longitudinal_friction = Abs(Dot(cross_vec, long_dir));
        if (base_friction > 5.0f) {
            friction_curve_input = base_friction / (longitudinal_friction + base_friction);
        }

        current_lat_friction = cfg.three_wheels ? EvalLatFrictionThreeWheel(friction_curve_input)
                                                : EvalLatFriction(friction_curve_input);
        current_long_friction = EvalLongFriction(friction_curve_input);

        if (car.handbrake_val > 0.0f) {
            current_lat_friction *= (EvalHandbrakeLatFrictionFactor(friction_curve_input) - 1.0f) * car.handbrake_val + 1.0f;
            current_long_friction *= (EvalHandbrakeLongFrictionFactor(friction_curve_input) - 1.0f) * car.handbrake_val + 1.0f;
        } else {
            current_long_friction = 1.0f;
        }

        const bool is_contact_sticky = shared.real_throttle != 0.0f &&
            !(!wheel.is_in_contact_with_world);
        if (!is_contact_sticky) {
            const float non_sticky = EvalNonStickyFrictionFactor(wheel.contact_normal_ws.z);
            current_lat_friction *= non_sticky;
            current_long_friction *= non_sticky;
        }
    }

    wheel.lat_friction = current_lat_friction;
    wheel.long_friction = current_long_friction;
    if (wheel_runtime_state) {
        wheel_runtime_state->lat_friction[i] = current_lat_friction;
        wheel_runtime_state->long_friction[i] = current_long_friction;
    }
    const float applied_lat_friction = current_lat_friction;
    const float applied_long_friction = current_long_friction;

    if (!wheel.is_in_contact) {
        return;
    }

    Vec3 axle_dir(0.0f, -1.0f, 0.0f);
    Vec3 forward_dir = shared.car_forward;
    Vec3 debug_steering[5];
    Vec3 raw_axle_dir = axle_dir;
    ComputeWheelBasisFromSteer(wheel, applied_steer_angle, shared.car_forward, axle_dir, forward_dir, debug_steering, mutator.game_mode == static_cast<uint8_t>(Mode::Soccar), NativeSseMath(mutator),
        (&raw_axle_dir));

    const Vec3 contact_point_bt = wheel.contact_point_bt;

    const BilateralBodyBt body1 = MakeBilateralCarBodyBt(car, cfg, mutator.car_mass);
    BilateralBodyBt body2;

    const float side_impulse = 0.0f;

    const Vec3 car_rel_contact_bt = contact_point_bt - body1.com_bt;
    const Vec3 contact_vel_bt =
        BilateralVelocityInLocalPoint(body1, car_rel_contact_bt) -
        BilateralVelocityInLocalPoint(body2, car_rel_contact_bt);
    const float forward_speed_bt = Dot(contact_vel_bt, forward_dir);

    float rolling_friction = 0.0f;
    if (applied_engine_force == 0.0f) {
        if (applied_brake != 0.0f) {
            float rel_vel = forward_speed_bt;
            if (dt > (1.0f / 80.0f)) {
                const float threshold = -(1.0f / (dt * 150.0f)) + 0.8f;
                if (Abs(rel_vel) < threshold) {
                    rel_vel = 0.0f;
                }
            }
            rolling_friction = Clamp(-rel_vel * 113.73963f, -applied_brake, applied_brake);
        }
    } else {
        rolling_friction = -applied_engine_force / shared.friction_scale;
    }

    wheel.impulse = ((forward_dir * rolling_friction) * applied_long_friction +
                     (axle_dir * side_impulse) * applied_lat_friction) * shared.friction_scale;
    wheel.cal_axle_dir = raw_axle_dir;
    wheel.cal_forward_dir = forward_dir;
    wheel.cal_rolling_friction = rolling_friction;
    wheel.cal_lat_friction = applied_lat_friction;
    wheel.cal_long_friction = applied_long_friction;
    wheel.cal_friction_scale = shared.friction_scale;
}

// One wheel's ray (or staged ray) and pushback for the vehicle phase. The summary fields the later all-wheel reads
// need are returned in WheelContactOut and reduced by CombineWheelContacts. A wheel resting on another car or the ball
// pushes that body (dyn_hit_*), which later wheels read, so such a car must run its wheels serially.
struct WheelContactOut {
    Vec3 normal = Vec3(0.0f, 0.0f, 0.0f);
    uint32_t flags = 0;  // kWheelContact* bits
};
constexpr uint32_t kWheelContactInContact = 1u;
constexpr uint32_t kWheelContactWithWorld = 2u;
constexpr uint32_t kWheelContactOnBall = 4u;

// The per-car reductions of the wheel contacts, the same on every lane of a car.
struct WheelContactSummary {
    int num_wheels_in_contact = 0;
    bool wheels_have_world_contact = false;
    Vec3 wheel_normal_sum = Vec3(0.0f, 0.0f, 0.0f);        // in-contact normals, game wheel order (sim i ^ 1)
    Vec3 contact_normal_sum_sim = Vec3(0.0f, 0.0f, 0.0f);  // in-contact normals, sim order 0..3 (GetUpwardsDirFromWheelContacts)
    Vec3 world_normal_sum = Vec3(0.0f, 0.0f, 0.0f);        // world-contact normals, sim order 0..3
    uint8_t world_contacts = 0;
};

ROCKETGPU_HD WheelContactOut WheelRayPhase(
    const CarState& car,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    WheelStepState& wheel,
    int i,
    float cal_front_scale,
    float cal_back_scale,
    float cal_min_hit_dist_sq_bt,
    WheelRuntimeState* wheel_runtime_state,
    const ArenaMeshTriangle* mesh_triangles,
    int32_t mesh_triangle_count,
    const CarState* cars_in_env,
    const CarConfig* car_cfgs_in_env,
    int32_t cars_in_env_count,
    int32_t self_car_idx,
    int32_t env_idx,
    const BallState* ball,
    float ball_radius,
    const ArenaMeshBvhQuantization* mesh_bvh_quantization,
    int32_t mesh_bvh_quantization_count,
    const ArenaMeshBvhMeshInfo* mesh_bvh_meshes,
    int32_t mesh_bvh_mesh_count,
    const ArenaMeshBvhNode* mesh_bvh_nodes,
    int32_t mesh_bvh_node_count,
    const ArenaMeshBvhSubtreeInfo* mesh_bvh_subtrees,
    int32_t mesh_bvh_subtree_count,
    const WheelRayStage* staged_rays,
    bool staged_rays_velocity_stale,
    CarState* dyn_hit_cars,
    BallState* dyn_hit_ball,
    uint32_t* dyn_hit_pushed_mask) {
    wheel = MakeWheelStepState(car, cfg, i);
    wheel.suspension_force_scale = wheel.is_front_wheel ? cal_front_scale : cal_back_scale;

    wheel.extra_pushback = wheel_runtime_state ? wheel_runtime_state->extra_pushback[i] : 0.0f;
    if (staged_rays != nullptr && staged_rays[i].valid == 2) {
        SetWheelRayMiss(wheel);
    } else if (staged_rays != nullptr && staged_rays[i].valid) {
        LoadWheelRayStage(wheel, staged_rays[i]);
        if (staged_rays_velocity_stale && wheel.is_in_contact) {
            ComputeWheelRayVelocityTerms(car, -wheel.wheel_dir_ws, wheel.contact_point_bt, wheel);
        }
    } else
    RayCastWheelAgainstArena(
        car,
        cfg,
        wheel,
        mesh_triangles,
        mesh_triangle_count,
        cars_in_env,
        car_cfgs_in_env,
        cars_in_env_count,
        self_car_idx,
        env_idx,
        i,
        ball,
        ball_radius,
        mesh_bvh_quantization,
        mesh_bvh_quantization_count,
        mesh_bvh_meshes,
        mesh_bvh_mesh_count,
        mesh_bvh_nodes,
        mesh_bvh_node_count,
        mesh_bvh_subtrees,
        mesh_bvh_subtree_count,
        mutator.arena_collision != 0, mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift,
        nullptr, cal_min_hit_dist_sq_bt);
    if (wheel.is_in_contact) {
        wheel.extra_pushback = 0.0f;
    }
    if (wheel.is_in_contact) {
        const float radius_bt = wheel.wheel_radius * UnitsUUToBT();
        wheel.suspension_len_bt = wheel.wheel_trace_len_bt - radius_bt;
        wheel.suspension_length = wheel.suspension_len_bt * UnitsBTToUU();
        wheel.wheel_trace_length = (wheel.suspension_len_bt + radius_bt) * UnitsBTToUU();
    }
    if (wheel.is_in_contact_with_world) {
        const float solver_dt = (1.0f / 60.0f);

        const float pb_rest_bt = wheel.suspension_rest_length * UnitsUUToBT();
        const float pb_radius_bt = wheel.wheel_radius * UnitsUUToBT();
        const float ray_pushback_thresh_bt = (pb_rest_bt + pb_radius_bt) - 0.05f;
        const float wheel_trace_len_bt = wheel.wheel_trace_len_bt;
        wheel.debug_pushback_dist_bt = wheel_trace_len_bt - ray_pushback_thresh_bt;
        if (wheel_trace_len_bt < ray_pushback_thresh_bt) {
            const float wheel_trace_dist_delta_bt = wheel.debug_pushback_dist_bt;
            StaticCollisionImpulseDebug pushback_debug;
            const float unscaled_pushback_bt = ResolveStaticCollisionImpulseMagnitudeBtInputs(
                car,
                cfg,
                mutator.car_mass,
                wheel.contact_point_bt,
                wheel.contact_normal_ws,
                wheel_trace_dist_delta_bt,
                solver_dt,
                &pushback_debug);
            wheel.debug_pushback_dist_bt = wheel_trace_dist_delta_bt;
            wheel.extra_pushback = unscaled_pushback_bt / 4.0f;
        }
    }
    if (wheel.is_in_contact && !wheel.is_in_contact_with_world) {
        // v3 wheel_info.rs apply_ray_cast + vehicle_rl.rs apply_single_wheel_dynamic_effects: a wheel on another
        // car gets resolve_single_collision pushback against that car; the hit body then takes ground_stick
        // (-normal, massed) and, for a car, -normal * full pushback, before the next wheel reads its velocity.
        const Vec3 n_bt = wheel.contact_normal_ws;
        const Vec3 cp_bt = wheel.contact_point_bt;
        if (wheel.ground_kind == ROCKETGPU_ARENA_RAY_HIT_CAR && dyn_hit_cars != nullptr && car_cfgs_in_env != nullptr &&
            wheel.ground_car_idx >= 0 && wheel.ground_car_idx < cars_in_env_count &&
            wheel.ground_car_idx != self_car_idx) {
            CarState& other = dyn_hit_cars[wheel.ground_car_idx];
            const CarConfig& other_cfg = car_cfgs_in_env[wheel.ground_car_idx];
            const float solver_dt = (1.0f / 60.0f);
            const float ray_pushback_thresh_bt =
                (wheel.suspension_rest_length * UnitsUUToBT() + wheel.wheel_radius * UnitsUUToBT()) - 0.05f;
            if (dyn_hit_pushed_mask != nullptr && wheel.ground_car_idx < 32 &&
                !((*dyn_hit_pushed_mask >> wheel.ground_car_idx) & 1u)) {
                other.cal_tick_start_ang_vel = other.ang_vel;
            }
            wheel.extra_pushback = 0.0f;
            if (wheel.wheel_trace_len_bt < ray_pushback_thresh_bt) {
                wheel.extra_pushback = ResolveCarCarWheelPushbackMagnitudeBtInputs(
                    car, cfg, other, other_cfg, mutator.car_mass, cp_bt, n_bt,
                    wheel.wheel_trace_len_bt - ray_pushback_thresh_bt, solver_dt) / 4.0f;
            }
            ApplyImpulseAtPointBtRoundtripBtInputs(other, other_cfg, mutator.car_mass, -n_bt, cp_bt - other.pos_bt);
            if (wheel.extra_pushback > 0.0f) {
                const float full_pushback = wheel.extra_pushback * 4.0f;
                ApplyImpulseAtPointBtRoundtripBtInputs(
                    other, other_cfg, mutator.car_mass, -n_bt * full_pushback, cp_bt - other.pos_bt);
            }
            if (dyn_hit_pushed_mask != nullptr && wheel.ground_car_idx < 32) {
                *dyn_hit_pushed_mask |= (1u << wheel.ground_car_idx);
            }
        } else if (wheel.ground_kind == ROCKETGPU_ARENA_RAY_HIT_BALL && dyn_hit_ball != nullptr) {
            BallState& b = *dyn_hit_ball;
            const Vec3 push_bt = -n_bt;
            SetBallLinearVelocityBt(b, b.vel_bt + push_bt * (1.0f / mutator.ball_mass));
            b.ang_vel += ApplyInverseBallInertiaWorldBt(b.rot, mutator.ball_mass, ball_radius, Cross(cp_bt - b.pos_bt, push_bt));
        }
    }
    if (wheel_runtime_state) {
        wheel_runtime_state->extra_pushback[i] = wheel.extra_pushback;
    }
    WheelContactOut out;
    out.normal = wheel.contact_normal_ws;
    out.flags = (wheel.is_in_contact ? kWheelContactInContact : 0u) |
                (wheel.is_in_contact_with_world ? kWheelContactWithWorld : 0u) |
                (wheel.ground_kind == ROCKETGPU_ARENA_RAY_HIT_BALL ? kWheelContactOnBall : 0u);
    return out;
}

// The wheel-contact reductions in the order the serial loop made them: game wheel order (sim i ^ 1) for the car's
// contact flags, count and normal sum, sim order for the world-contact sum.
ROCKETGPU_HD WheelContactSummary CombineWheelContacts(
    CarState& car,
    WheelRuntimeState* wheel_runtime_state,
    const WheelContactOut out[4]) {
    WheelContactSummary sum;
    for (int k = 0; k < 4; ++k) {
        const int i = k ^ 1;
        const bool in_contact = (out[i].flags & kWheelContactInContact) != 0;
        const bool with_world = (out[i].flags & kWheelContactWithWorld) != 0;
        const bool on_ball = (out[i].flags & kWheelContactOnBall) != 0;
        car.wheels_with_contact[i] = in_contact;
        car.world_contact.wheel_ball_mask |= static_cast<uint8_t>((in_contact && on_ball) << i);
        sum.num_wheels_in_contact += in_contact ? 1 : 0;
        sum.wheels_have_world_contact = sum.wheels_have_world_contact || with_world;
        if (in_contact) {
            sum.wheel_normal_sum += out[i].normal;
        }
    }

    car.is_on_ground = sum.num_wheels_in_contact >= 3 ? 1 : 0;

    for (int i = 0; i < 4; ++i) {
        const bool in_contact = (out[i].flags & kWheelContactInContact) != 0;
        if (in_contact) {
            sum.contact_normal_sum_sim += out[i].normal;
        }
        if (in_contact && (out[i].flags & kWheelContactWithWorld) != 0) {
            sum.world_normal_sum += out[i].normal;
            ++sum.world_contacts;
        }
    }
    if (wheel_runtime_state) {
        wheel_runtime_state->cal_prev_wheel_normal_sum = sum.world_normal_sum;
        wheel_runtime_state->cal_prev_world_wheel_contacts = sum.world_contacts;
    }
    return sum;
}

ROCKETGPU_HD int PreTickVehicleDynamics(
    CarState& car,
    CarControls controls,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    float dt,
    WheelStepState* wheels,
    float& forward_speed_uu,
    WheelRuntimeState* wheel_runtime_state,
    const ArenaMeshTriangle* mesh_triangles = nullptr,
    int32_t mesh_triangle_count = 0,
    const CarState* cars_in_env = nullptr,
    const CarConfig* car_cfgs_in_env = nullptr,
    int32_t cars_in_env_count = 0,
    int32_t self_car_idx = -1,
    int32_t env_idx = -1,
    const BallState* ball = nullptr,
    float ball_radius = 0.0f,
    const ArenaMeshBvhQuantization* mesh_bvh_quantization = nullptr,
    int32_t mesh_bvh_quantization_count = 0,
    const ArenaMeshBvhMeshInfo* mesh_bvh_meshes = nullptr,
    int32_t mesh_bvh_mesh_count = 0,
    const ArenaMeshBvhNode* mesh_bvh_nodes = nullptr,
    int32_t mesh_bvh_node_count = 0,
    const ArenaMeshBvhSubtreeInfo* mesh_bvh_subtrees = nullptr,
    int32_t mesh_bvh_subtree_count = 0,

    const WheelRayStage* staged_rays = nullptr,

    bool staged_rays_velocity_stale = false,
    CarState* dyn_hit_cars = nullptr,
    BallState* dyn_hit_ball = nullptr,
    uint32_t* dyn_hit_pushed_mask = nullptr) {
    car.world_contact.wheel_ball_mask = 0;

    // v3 Car::update_wheels sticky gate reads the previous tick's wheel contacts with the world (sticky_gate_prev and
    // get_upwards_dir_from_wheel_contacts); keep them before this tick's raycasts overwrite the autoroll cache.
    const uint8_t sticky_prev_world_contacts =
        wheel_runtime_state ? wheel_runtime_state->cal_prev_world_wheel_contacts : 0;
    const Vec3 sticky_prev_world_normal_sum =
        wheel_runtime_state ? wheel_runtime_state->cal_prev_wheel_normal_sum : Vec3(0.0f, 0.0f, 0.0f);
    const bool sticky_prev_contact = (car.wheels_with_contact[0] || car.wheels_with_contact[1] || car.wheels_with_contact[2] || car.wheels_with_contact[3]);

    const bool landing_prev_any_contact = car.tick_count_since_update != 0 &&
        (car.wheels_with_contact[0] || car.wheels_with_contact[1] || car.wheels_with_contact[2] || car.wheels_with_contact[3]);
    float cal_front_scale = 0.0f, cal_back_scale = 0.0f, cal_min_hit_dist_sq_bt = 0.0f;
    CalSuspensionForceScales(cfg, mutator.car_mass, cal_front_scale, cal_back_scale);
    cal_min_hit_dist_sq_bt = CalMinWheelHitDistSqBt(cfg);

    // v3 wheel order (front-left, front-right, back-left, back-right) is RocketGPU's i ^ 1. It matters when wheels
    // rest on another car: each wheel's pushback reads the velocity the previous wheels' pushes left on that car.
    WheelContactOut contact_out[4];
    for (int k = 0; k < 4; ++k) {
        const int i = k ^ 1;
        contact_out[i] = WheelRayPhase(car, cfg, mutator, wheels[i], i, cal_front_scale, cal_back_scale,
                                       cal_min_hit_dist_sq_bt, wheel_runtime_state, mesh_triangles, mesh_triangle_count,
                                       cars_in_env, car_cfgs_in_env, cars_in_env_count, self_car_idx, env_idx, ball,
                                       ball_radius, mesh_bvh_quantization, mesh_bvh_quantization_count, mesh_bvh_meshes,
                                       mesh_bvh_mesh_count, mesh_bvh_nodes, mesh_bvh_node_count, mesh_bvh_subtrees,
                                       mesh_bvh_subtree_count, staged_rays, staged_rays_velocity_stale, dyn_hit_cars,
                                       dyn_hit_ball, dyn_hit_pushed_mask);
    }
    const WheelContactSummary contact_summary = CombineWheelContacts(car, wheel_runtime_state, contact_out);
    const int num_wheels_in_contact = contact_summary.num_wheels_in_contact;
    const bool wheels_have_world_contact = contact_summary.wheels_have_world_contact;
    const Vec3 wheel_normal_sum = contact_summary.wheel_normal_sum;
    // A dodge or double-jump impulse this tick already landed (cal_flip_before_wheels): drive, brake and steer
    // read the post-impulse forward speed, as the game does (RLBot c3 car1 f96764: a landing dodge with throttle and
    // steer matches to 0.01 uu/s; the tick-start speed gives the drive force at 1235 uu/s instead of none past 1410).
    forward_speed_uu = WheelForwardSpeed(car);
    const float abs_forward_speed = Abs(forward_speed_uu);
    const float friction_scale = mutator.car_mass / 3.0f;
    Vec3 car_forward(1.0f, 0.0f, 0.0f);
    Vec3 unused_right(0.0f, 1.0f, 0.0f);
    Vec3 unused_up(0.0f, 0.0f, 1.0f);
    CarBasisColumns(car, car_forward, unused_right, unused_up);

    if (controls.handbrake) {
        car.handbrake_val = Clamp(car.handbrake_val + 5.0f * dt, 0.0f, 1.0f);
    } else {
        car.handbrake_val = Clamp(car.handbrake_val - 2.0f * dt, 0.0f, 1.0f);
    }

    float real_throttle = controls.throttle;
    float real_brake = 0.0f;

    float drive_speed_scale = EvalDriveTorqueFactor(abs_forward_speed);
    float engine_throttle = real_throttle;
    if (!controls.handbrake) {
        const float abs_throttle = Abs(real_throttle);

        if ((abs_throttle > 0.01f)) {
            if (abs_forward_speed > 25.0f && Sign(real_throttle) != Sign(forward_speed_uu)) {
                real_brake = 1.0f;
                if (abs_forward_speed > 0.01f) {
                    engine_throttle = 0.0f;
                }
            }
        } else {
            engine_throttle = 0.0f;
            real_brake = abs_forward_speed < 25.0f ? 1.0f : 0.15f;
        }
    }

    if (num_wheels_in_contact < 3) {
        drive_speed_scale *= 0.25f;
    }

    const float throttle_torque_bt = mutator.car_mass * 400.0f * UnitsUUToBT();
    const float brake_torque_bt = mutator.car_mass * (14.25f + (1.0f / 3.0f)) * UnitsUUToBT();
    const float drive_engine_force = engine_throttle * throttle_torque_bt * drive_speed_scale;
    const float drive_brake_force = real_brake * brake_torque_bt;
    const float steer_angle_base = cfg.three_wheels ? EvalSteerAngleFromSpeedThreeWheel(abs_forward_speed)
                                                    : EvalSteerAngleFromSpeed(cfg.steer_angle_curve, abs_forward_speed);
    float next_steer_angle = steer_angle_base;
    if (car.handbrake_val > 0.0f) {
        next_steer_angle += (EvalPowerslideSteerAngleFromSpeed(cfg.powerslide_steer_angle_curve, abs_forward_speed) - next_steer_angle) * car.handbrake_val;
    }
    next_steer_angle *= controls.steer;

    WheelDriveShared drive_shared;
    drive_shared.car_forward = car_forward;
    drive_shared.next_steer_angle = next_steer_angle;
    drive_shared.drive_engine_force = drive_engine_force;
    drive_shared.drive_brake_force = drive_brake_force;
    drive_shared.real_throttle = real_throttle;
    drive_shared.friction_scale = friction_scale;
    for (int i = 0; i < 4; ++i) {
        WheelDriveSetupPhase(car, cfg, mutator, drive_shared, wheels[i], i, wheel_runtime_state, dt);
    }

    if (wheel_runtime_state) {
        // v3: gate = a wheel touched the world last tick (on the tick after a state set, the refresh raycast at the
        // set pose, i.e. this tick's world contacts); direction = normalized sum of those world contact normals, else
        // the chassis up; full stick on raw throttle or forward speed above STOPPING_FORWARD_VEL.
        const bool restore_tick = car.tick_count_since_update == 0;
        Vec3 world_normal_sum = sticky_prev_world_normal_sum;
        bool gate = sticky_prev_world_contacts > 0;
        if (restore_tick) {
            world_normal_sum = Vec3(0.0f, 0.0f, 0.0f);
            gate = false;
            // A set state with the game's wheel contact normals carries the previous tick's contacts, which the game's
            // sticky force reads (RLPR cw_high_speed_wall: a teleported car still gets the wall's sticky force on its
            // first tick).
            if (car.set_wheel_normals_mask) {
                gate = SetStateWorldWheelNormals(car, cfg, wheels, ball, ball_radius, cars_in_env, car_cfgs_in_env,
                                                 cars_in_env_count, self_car_idx, world_normal_sum,
                                                 mesh_triangles, mesh_triangle_count, mesh_bvh_quantization,
                                                 mesh_bvh_quantization_count, mesh_bvh_meshes, mesh_bvh_mesh_count,
                                                 mesh_bvh_nodes, mesh_bvh_node_count, mesh_bvh_subtrees,
                                                 mesh_bvh_subtree_count) > 0;
            } else {
                world_normal_sum = contact_summary.world_normal_sum;
                gate = contact_summary.world_contacts > 0;
            }
        }
        if (gate) {
            const Vec3 upwards_dir = Normalize(world_normal_sum, unused_up);
            const bool full_stick = (controls.throttle != 0.0f) || (abs_forward_speed > 25.0f);
            float sticky_force_scale = cfg.three_wheels ? 0.0f : 0.5f;
            if (full_stick) {
                sticky_force_scale += 1.0f - Abs(upwards_dir.z);
            }
            const Vec3 sticky_force_bt =
                upwards_dir * sticky_force_scale * (-Abs(mutator.gravity_z) * UnitsUUToBT()) * mutator.car_mass;
            QueueCentralForce(wheel_runtime_state, sticky_force_bt);
        }
    } else {
    bool sticky_block = wheels_have_world_contact || sticky_prev_contact;
    bool apply_sticky = true;
    apply_sticky = landing_prev_any_contact || (car.tick_count_since_update == 0 && num_wheels_in_contact > 0);
    sticky_block = wheels_have_world_contact || apply_sticky;

    if (sticky_block) {
        Vec3 upwards_dir = wheels_have_world_contact ? Normalize(wheel_normal_sum, Vec3(0.0f, 0.0f, 1.0f)) : unused_up;
        if (wheel_runtime_state) {
            const Vec3 this_tick_dir = upwards_dir;
            const Vec3 prev = wheel_runtime_state->cal_prev_sticky_dir;
            if (car.tick_count_since_update != 0 && (prev.x != 0.0f || prev.y != 0.0f || prev.z != 0.0f)) {
                upwards_dir = prev;
            }
            wheel_runtime_state->cal_prev_sticky_dir = this_tick_dir;
        }
        const bool full_stick = (real_throttle != 0.0f) || (abs_forward_speed > 25.0f);
        float sticky_force_scale = cfg.three_wheels ? 0.0f : 0.5f;
        if (full_stick) {
            sticky_force_scale += 1.0f - Abs(upwards_dir.z);
        }

        const Vec3 sticky_force_bt =
            upwards_dir * sticky_force_scale * (mutator.gravity_z * UnitsUUToBT()) * mutator.car_mass;
        if (!apply_sticky) {
        } else if (wheel_runtime_state) {
            QueueCentralForce(wheel_runtime_state, sticky_force_bt);
        } else {
            const Vec3 sticky_force = sticky_force_bt * UnitsBTToUU();
            ApplyCentralForce(car, mutator.car_mass, sticky_force, dt);
        }
    }
    }

    return num_wheels_in_contact;
}

// RL's asymmetric 8-bit axis grid (negatives /128, positives /127), as in v3's quantize_axis_input.
ROCKETGPU_HD float QuantizeAxisInput(float x) {
    const float c = Clamp(x, -1.0f, 1.0f);
    const float y = c < 0.0f ? Max(c * 128.0f, -128.0f) : Min(c * 127.0f, 127.0f);
    const float w = ((y + 128.0f) + (y + 128.0f)) + 0.5f;
    const int byte = (static_cast<int>(rintf(w)) >> 1) & 0xFF;
    const float s = static_cast<float>(byte) - 128.0f;
    return byte < 0x80 ? s * (1.0f / 128.0f) : s / 127.0f;
}

ROCKETGPU_HD CarControls QuantizeAxisControls(CarControls controls) {
    controls.throttle = QuantizeAxisInput(controls.throttle);
    controls.steer = QuantizeAxisInput(controls.steer);
    controls.pitch = QuantizeAxisInput(controls.pitch);
    controls.yaw = QuantizeAxisInput(controls.yaw);
    controls.roll = QuantizeAxisInput(controls.roll);
    return controls;
}

ROCKETGPU_HD float DodgeCancelPitchRaw(const CarState& car, const CarControls& controls, const MutatorConfig& mutator, float dt,
                                      float raw_pitch = 2.0f) {
    {
        constexpr float kGameMinDodgeTorqueTime = 0.041f;
        // v3 base.rs:484-491 quantizes the negated raw pitch, not the step's quantized pitch.
        const float pitch = raw_pitch <= 1.0f ? raw_pitch : controls.pitch;
        return car.internal_flip_time >= kGameMinDodgeTorqueTime ? pitch : 0.0f;
    }
    return (car.internal_flip_time > (0.0f)) ? car.last_controls.pitch : controls.pitch;
}

ROCKETGPU_HD float DodgeCancelPitch(const CarState& car, const CarControls& controls, const MutatorConfig& mutator, float dt,
                                   float raw_pitch = 2.0f) {
    const float pitch = DodgeCancelPitchRaw(car, controls, mutator, dt, raw_pitch);
    // RL quantizes the negated pitch here.
    return (-QuantizeAxisInput(-pitch));
}

ROCKETGPU_HD void ApplyAirControl(
    CarState& car,
    const CarControls& controls,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    float dt,
    bool update_air_control,
    WheelRuntimeState* wheel_runtime_state = nullptr,
    bool air_throttle = true,
    float cal_pitchlock_restart_time = -1.0f,
    const Vec3* cal_damp_ang_vel = nullptr) {
    Vec3 basis_forward(1.0f, 0.0f, 0.0f);
    Vec3 basis_right(0.0f, 1.0f, 0.0f);
    Vec3 basis_up(0.0f, 0.0f, 1.0f);
    CarBasisColumns(car, basis_forward, basis_right, basis_up);
    const Vec3 dir_pitch_right(-basis_right.x, -basis_right.y, -basis_right.z);
    const Vec3 dir_yaw_up = basis_up;
    const Vec3 dir_roll_forward(-basis_forward.x, -basis_forward.y, -basis_forward.z);

    bool do_air_control = false;

    do_air_control = true;

    // The game applies air control while auto-flipping (RLPR AirControl impulse records on auto-flip ticks, e.g. Wisp
    // frame 1704 car 1); v3 update_air_torque has no auto-flip gate either.
    do_air_control = do_air_control && update_air_control;
    if (do_air_control) {
        float pitch_torque_scale = 1.0f;
        if (car.is_flipping) {
            pitch_torque_scale = 0.0f;
        } else if (car.has_flipped && car.internal_flip_time < (mutator.flip_torque_time + mutator.flip_pitchlock_extra_time)) {
            pitch_torque_scale = 0.0f;        } else if (cal_pitchlock_restart_time >= 0.0f && cal_pitchlock_restart_time < mutator.flip_pitchlock_extra_time) {
            pitch_torque_scale = 0.0f;
        }

        Vec3 torque_world(0.0f, 0.0f, 0.0f);
        if (controls.pitch != 0.0f || controls.yaw != 0.0f || controls.roll != 0.0f) {
            torque_world =
                (controls.pitch * dir_pitch_right * pitch_torque_scale * mutator.car_air_control_pitch) +
                (controls.yaw * dir_yaw_up * mutator.car_air_control_yaw) +
                (controls.roll * dir_roll_forward * mutator.car_air_control_roll);
        }

        const Vec3 damp_ang_vel = cal_damp_ang_vel != nullptr ? *cal_damp_ang_vel : car.ang_vel;
        const float damp_pitch = Dot(dir_pitch_right, damp_ang_vel) * mutator.car_air_damp_pitch *
            (1.0f - Min(Abs(controls.pitch * pitch_torque_scale), 1.0f));
        const float damp_yaw = Dot(dir_yaw_up, damp_ang_vel) * mutator.car_air_damp_yaw *
            (1.0f - Min(Abs(controls.yaw), 1.0f));
        const float damp_roll = Dot(dir_roll_forward, damp_ang_vel) * mutator.car_air_damp_roll;
        const Vec3 damping_world =
            dir_yaw_up * damp_yaw +
            dir_pitch_right * damp_pitch +
            dir_roll_forward * damp_roll;

        const Vec3 air_control_torque = (torque_world - damping_world) * CarTorqueScale();
        if (wheel_runtime_state) {
            QueueAngularAccelWorld(wheel_runtime_state, car, cfg, mutator.car_mass, air_control_torque);

            Vec3 ac_i0, ac_i1, ac_i2;
            BuildInertiaTensorWorldBtInverse(car, cfg, mutator.car_mass, ac_i0, ac_i1, ac_i2);
            QueueOfficialTorqueBt(wheel_runtime_state,
                Mul3x3Vec(ac_i0, ac_i1, ac_i2, torque_world - damping_world) * CarTorqueScale());
        } else {
            ApplyAngularAccelWorld(car, air_control_torque, dt);
        }
    }

    const float air_throttle_input = (car.is_boosting) ? 1.0f : controls.throttle;
    if (air_throttle_input != 0.0f && air_throttle) {
        Vec3 throttle_fwd(1.0f, 0.0f, 0.0f);
        Vec3 throttle_rgt(0.0f, 1.0f, 0.0f);
        Vec3 throttle_up(0.0f, 0.0f, 1.0f);
        CarBasisColumns(car, throttle_fwd, throttle_rgt, throttle_up);
        (void)throttle_rgt; (void)throttle_up;

        const Vec3 air_throttle_force_bt =
            throttle_fwd * air_throttle_input * mutator.car_air_throttle_accel * UnitsUUToBT() * mutator.car_mass;
        if (wheel_runtime_state) {
            QueueCentralForce(wheel_runtime_state, air_throttle_force_bt);
        } else {
            const Vec3 air_throttle_force = air_throttle_force_bt * UnitsBTToUU();
            ApplyCentralForce(car, mutator.car_mass, air_throttle_force, dt);
        }
    }
}

ROCKETGPU_HD void ApplyBoost(
    CarState& car,
    const CarControls& controls,
    const MutatorConfig& mutator,
    float dt,
    WheelRuntimeState* wheel_runtime_state = nullptr) {
    if (!car.is_demoed) {
        // RL order: the first boost tick is free; while boosting, pay first and stop without pushing once empty.
        bool push = false;
        if (car.is_boosting) {
            car.boost = Max(0.0f, car.boost - mutator.boost_used_per_second * dt);
            const bool latch_expired = !controls.boost && car.boosting_time >= mutator.boost_min_time;
            push = !(car.boost == 0.0f || latch_expired);
        } else {
            push = controls.boost && car.boost > 0.0f;
        }
        car.is_boosting = push ? 1 : 0;
        if (push) {
            Vec3 boost_fwd(1.0f, 0.0f, 0.0f);
            Vec3 boost_rgt(0.0f, 1.0f, 0.0f);
            Vec3 boost_up(0.0f, 0.0f, 1.0f);
            CarBasisColumns(car, boost_fwd, boost_rgt, boost_up);
            const float accel = mutator.boost_accel_ground;
            const Vec3 boost_force_bt = accel * UnitsUUToBT() * boost_fwd * mutator.car_mass;
            if (wheel_runtime_state) {
                QueueCentralForce(wheel_runtime_state, boost_force_bt);
            } else {
                ApplyCentralForce(car, mutator.car_mass, boost_force_bt * UnitsBTToUU(), dt);
            }
            car.boosting_time += dt;
            car.time_since_boosted = 0.0f;
        } else {
            car.boosting_time = 0.0f;
            car.time_since_boosted += dt;
            if (mutator.recharge_boost_enabled && car.time_since_boosted >= mutator.recharge_boost_delay) {
                car.boost += mutator.recharge_boost_per_second * dt;
            }
        }
        car.boost = Min(Max(car.boost, 0.0f), mutator.boost_max);
        return;
    }

    const bool has_boost = car.boost > 0.0f;
    if (has_boost) {
        if (car.is_boosting) {
            car.is_boosting = (controls.boost || car.boosting_time < mutator.boost_min_time) ? 1 : 0;
        } else if (controls.boost) {
            car.is_boosting = 1;
        }
    } else {
        car.is_boosting = 0;
    }

    if (car.is_boosting && !car.is_demoed) {
        Vec3 boost_fwd(1.0f, 0.0f, 0.0f);
        Vec3 boost_rgt(0.0f, 1.0f, 0.0f);
        Vec3 boost_up(0.0f, 0.0f, 1.0f);
        CarBasisColumns(car, boost_fwd, boost_rgt, boost_up);
        (void)boost_rgt; (void)boost_up;
        const Vec3 forward = boost_fwd;

        const float accel = mutator.boost_accel_ground;

        const float boost_accel_bt = accel * UnitsUUToBT();
        const Vec3 boost_force_bt = boost_accel_bt * forward * mutator.car_mass;
        if (wheel_runtime_state) {
            QueueCentralForce(wheel_runtime_state, boost_force_bt);
        } else {
            const Vec3 boost_force = boost_force_bt * UnitsBTToUU();
            ApplyCentralForce(car, mutator.car_mass, boost_force, dt);
        }
        car.boost = Max(0.0f, car.boost - mutator.boost_used_per_second * dt);
        car.boosting_time += dt;
        car.time_since_boosted = 0.0f;
    } else {
        car.boosting_time = 0.0f;
        car.time_since_boosted += dt;
        if (mutator.recharge_boost_enabled && car.time_since_boosted >= mutator.recharge_boost_delay) {
            car.boost = Min(mutator.boost_max, car.boost + mutator.recharge_boost_per_second * dt);
        }
    }
    car.boost = Min(car.boost, mutator.boost_max);
}

ROCKETGPU_HD void CapCarAngularSpeed(CarState& car, const MutatorConfig& mutator);

// v3 RigidBody::limit_vels(MAX_SPEED, MAX_ANG_SPEED), which v3 calls right after the jump-start, dodge and
// double-jump impulses (car/base.rs update_jump and update_double_jump_or_flip), so later pre-tick forces and the
// solver see the capped velocity instead of a velocity capped only at the end of the tick.
ROCKETGPU_HD void CalLimitCarVels(CarState& car, const MutatorConfig& mutator) {
    const float max_speed_bt = mutator.car_max_speed * UnitsUUToBT();
    const Vec3 vel_bt = car.vel_bt;
    const float vel_bt_len_sq = LengthSq(vel_bt);
    if (vel_bt_len_sq > max_speed_bt * max_speed_bt) {
        const Vec3 vel_bt_normalized = mutator.native_sphere_rsqrt_table
            ? NativeSseSphereNormalizer{mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift}(vel_bt)
            : vel_bt * RsqrtScalar(vel_bt_len_sq);
        SetCarLinearVelocityBt(car, vel_bt_normalized * max_speed_bt);
    }
    CapCarAngularSpeed(car, mutator);
}

ROCKETGPU_HD void StartJump(CarState& car, const MutatorConfig& mutator) {
    car.is_jumping = 1;
    car.internal_jump_time = 0.0f;

    Vec3 jump_fwd(1.0f, 0.0f, 0.0f);
    Vec3 jump_rgt(0.0f, 1.0f, 0.0f);
    Vec3 jump_up(0.0f, 0.0f, 1.0f);
    CarBasisColumns(car, jump_fwd, jump_rgt, jump_up);
    (void)jump_fwd; (void)jump_rgt;
    ApplyCentralImpulseViaBtRoundtrip(car, jump_up * mutator.jump_immediate_speed, mutator.car_mass);
    CalLimitCarVels(car, mutator);
}

ROCKETGPU_HD void ContinueJump(
    CarState& car,
    const CarControls& controls,
    const MutatorConfig& mutator,
    float dt,
    WheelRuntimeState* wheel_runtime_state) {
    if (car.is_jumping) {
        if (!(car.internal_jump_time < mutator.jump_min_time ||
              (controls.jump && !car.jump_hold_broken && car.internal_jump_time < mutator.jump_max_time))) {
            car.is_jumping = 0;
        }
    }
    if (car.is_jumping) {
        car.has_jumped = 1;

        Vec3 cjump_fwd(1.0f, 0.0f, 0.0f);
        Vec3 cjump_rgt(0.0f, 1.0f, 0.0f);
        Vec3 cjump_up(0.0f, 0.0f, 1.0f);
        CarBasisColumns(car, cjump_fwd, cjump_rgt, cjump_up);
        (void)cjump_fwd; (void)cjump_rgt;

        Vec3 total_jump_force(
            cjump_up.x * mutator.jump_accel,
            cjump_up.y * mutator.jump_accel,
            cjump_up.z * mutator.jump_accel);

        const Vec3 jump_force_bt(
            (total_jump_force.x * UnitsUUToBT()) * mutator.car_mass,
            (total_jump_force.y * UnitsUUToBT()) * mutator.car_mass,
            (total_jump_force.z * UnitsUUToBT()) * mutator.car_mass);

        if (wheel_runtime_state) {
            QueueCentralForce(wheel_runtime_state, jump_force_bt);
        } else {
            const Vec3 jump_force = jump_force_bt * UnitsBTToUU();
            ApplyCentralForce(car, mutator.car_mass, jump_force, dt);
        }
    }
    if (car.is_jumping || car.has_jumped) {
        car.internal_jump_time += dt;
    }
}

ROCKETGPU_HD void ApplyFlipTorque(
    CarState& car,
    const CarControls& controls,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    float dt,
    WheelRuntimeState* wheel_runtime_state = nullptr,
    float raw_pitch = 2.0f) {
    if (car.is_flipping) {
        car.is_flipping = (car.has_flipped && car.internal_flip_time < mutator.flip_torque_time) ? 1 : 0;
    }
    if (!car.is_flipping) {
        return;
    }

    // The game rounds the dodge torque to whole units before the pitch-cancel scale: its recorded DodgeTorque
    // impulses are (round(rel.x * 260), round(rel.y * 224) * cancel_scale, 0) * dt in the tick-start car basis on all
    // 65k dodge ticks of the Nexto, PartyCannon and Wisp recordings (rel (0.7071, 0.7071) gives 184 and 158, not
    // 183.85 and 158.39).
    const Vec3 rel_torque = car.flip_rel_torque;
    float torque_y = rintf(rel_torque.y * mutator.flip_torque_y);
    const float cancel_pitch = DodgeCancelPitch(car, controls, mutator, dt, raw_pitch);
    if (rel_torque.y != 0.0f && cancel_pitch != 0.0f && Sign(rel_torque.y) == Sign(cancel_pitch)) {
        torque_y *= (1.0f - Min(Abs(cancel_pitch), 1.0f));
    }

    const Vec3 dodge_torque_local(
        rintf(rel_torque.x * mutator.flip_torque_x),
        torque_y,
        0.0f);
    const Vec3 dodge_torque_world = RotateCarBasis(car, dodge_torque_local);
    if (wheel_runtime_state) {
        QueueAngularAccelWorld(wheel_runtime_state, car, cfg, mutator.car_mass, dodge_torque_world);

        Vec3 dg_i0, dg_i1, dg_i2;
        BuildInertiaTensorWorldBtInverse(car, cfg, mutator.car_mass, dg_i0, dg_i1, dg_i2);
        Vec3 dg_f(1.0f, 0.0f, 0.0f), dg_r(0.0f, 1.0f, 0.0f), dg_u(0.0f, 0.0f, 1.0f);
        CarBasisColumns(car, dg_f, dg_r, dg_u);
        const Vec3 basis_row0(dg_f.x, dg_r.x, dg_u.x);
        const Vec3 basis_row1(dg_f.y, dg_r.y, dg_u.y);
        const Vec3 basis_row2(dg_f.z, dg_r.z, dg_u.z);
        Vec3 dg_m0, dg_m1, dg_m2;
        MatMul3x3BulletStyle(dg_i0, dg_i1, dg_i2, basis_row0, basis_row1, basis_row2, dg_m0, dg_m1, dg_m2);
        QueueOfficialTorqueBt(wheel_runtime_state, Mul3x3Vec(dg_m0, dg_m1, dg_m2, dodge_torque_local));
    } else {
        ApplyAngularAccelWorld(car, dodge_torque_world, dt);
    }
}

ROCKETGPU_HD void ApplyAutoFlip(
    CarState& car,
    const CarControls& controls,
    const MutatorConfig& mutator,
    float dt) {
    const bool jump_pressed = controls.jump && !car.last_controls.jump;
    if (jump_pressed && car.world_contact.has_contact && car.world_contact.contact_normal.z > SqrtHalf()) {
        Vec3 rb_fwd(1.0f, 0.0f, 0.0f);
        Vec3 rb_rgt(0.0f, 1.0f, 0.0f);
        Vec3 rb_up(0.0f, 0.0f, 1.0f);
        CarBasisColumns(car, rb_fwd, rb_rgt, rb_up);
        (void)rb_fwd;
        const float roll = -GlibcAtan2f(rb_rgt.z, rb_up.z);
        const float abs_roll = Abs(roll);

        const bool flip_gate = (Dot(car.world_contact.contact_normal, rb_up) < -0.966f);
        if (flip_gate) {
            car.auto_flip_timer = static_cast<float>(
                static_cast<double>(0.4f) * (static_cast<double>(abs_roll) / 3.14159265358979323846));
            car.auto_flip_torque_scale = (roll > 0.0f) ? 1.0f : -1.0f;
            car.is_auto_flipping = 1;

            Vec3 af_fwd(1.0f, 0.0f, 0.0f);
            Vec3 af_rgt(0.0f, 1.0f, 0.0f);
            Vec3 af_up(0.0f, 0.0f, 1.0f);
            CarBasisColumns(car, af_fwd, af_rgt, af_up);
            (void)af_fwd; (void)af_rgt;
            ApplyCentralImpulseViaBtRoundtrip(car, (-af_up) * 200.0f, mutator.car_mass);
        }
    }

    (void)mutator;

    if (car.is_auto_flipping) {
        if (car.auto_flip_timer <= 0.0f) {
            car.is_auto_flipping = 0;
            car.auto_flip_timer = 0.0f;
        } else {
            Vec3 aft_fwd(1.0f, 0.0f, 0.0f);
            Vec3 aft_rgt(0.0f, 1.0f, 0.0f);
            Vec3 aft_up(0.0f, 0.0f, 1.0f);
            CarBasisColumns(car, aft_fwd, aft_rgt, aft_up);
            (void)aft_rgt; (void)aft_up;

            // v3 car_consts::autoflip::ANG_VEL_DELTA: 1000 rad/s per tick (no dt), so the clamp below leaves the car spinning
            // at the max angular speed about its forward axis. The recordings agree (Wisp frame 1704 car 1: truth
            // omega = clamp(omega + 1000 fwd) + the recorded AirControl impulse to 1e-5 rad/s; 50 rad/s^2 misses by 0.08).
            car.ang_vel += (aft_fwd * 1000.0f) * car.auto_flip_torque_scale;
            // v3 update_auto_flip calls limit_vels(MAX_SPEED, MAX_ANG_SPEED) right after the auto-flip increment.
            CalLimitCarVels(car, mutator);

            car.auto_flip_timer -= dt;
        }
    }
}

// Returns true when a directional flip starts.
ROCKETGPU_HD bool TryDoubleJumpOrFlip(
    CarState& car,
    const CarControls& controls_in,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    float forward_speed_uu,
    float raw_pitch = 2.0f) {
    // RL quantizes the dodge input with pitch negated first. The dodge uses the raw pitch, as v3
    // base.rs:684 (quantize_axis_inputs(pyr * (-1, 1, 1))); re-quantizing the step's quantized pitch moves it a step.
    CarControls controls = controls_in;
    controls.pitch = -QuantizeAxisInput(-(raw_pitch <= 1.0f ? raw_pitch : controls.pitch));

    Vec3 forward(1.0f, 0.0f, 0.0f);
    Vec3 right_unused(0.0f, 1.0f, 0.0f);
    Vec3 up(0.0f, 0.0f, 1.0f);
    CarBasisColumns(car, forward, right_unused, up);
    (void)right_unused;

    const float input_magnitude = Abs(controls.yaw) + Abs(controls.pitch) + Abs(controls.roll);
    const bool is_flip_input = input_magnitude >= cfg.dodge_deadzone;
    bool can_use = false;
    if (is_flip_input) {
        can_use = ((!car.has_double_jumped && !car.has_flipped) || mutator.unlimited_flips) && !car.is_auto_flipping;
    } else {
        can_use = ((!car.has_double_jumped && !car.has_flipped) || mutator.unlimited_double_jumps) && !car.is_auto_flipping;
    }
    if (!can_use) {
        return false;
    }

    if (is_flip_input) {
        car.internal_flip_time = 0.0f;
        car.last_directional_flip_start = car.tick_count_since_update;
        car.has_flipped = 1;
        car.is_flipping = 1;

        Vec3 dodge_dir(-controls.pitch, controls.yaw + controls.roll, 0.0f);
        if (Abs(controls.yaw + controls.roll) < 0.1f && Abs(controls.pitch) < 0.1f) {
            dodge_dir = Vec3(0.0f, 0.0f, 0.0f);
        } else {
            dodge_dir = Normalize(dodge_dir, Vec3(0.0f, 0.0f, 0.0f));
        }

        car.flip_rel_torque = Vec3(-dodge_dir.y, dodge_dir.x, 0.0f);
        if (Abs(dodge_dir.x) < 0.1f) dodge_dir.x = 0.0f;
        if (Abs(dodge_dir.y) < 0.1f) dodge_dir.y = 0.0f;

        if (LengthSq(dodge_dir) > 0.0f) {
            const float forward_speed_ratio = Abs(forward_speed_uu) / mutator.car_max_speed;
            bool should_dodge_backwards = false;
            if (Abs(forward_speed_uu) < 100.0f) {
                should_dodge_backwards = dodge_dir.x < 0.0f;
            } else {
                should_dodge_backwards = ((dodge_dir.x >= 0.0f) != (forward_speed_uu >= 0.0f));
            }

            Vec3 initial_dodge_vel = dodge_dir * mutator.flip_initial_vel_scale;
            const float max_speed_scale_x = should_dodge_backwards ? mutator.flip_backward_impulse_max_speed_scale
                                                                   : mutator.flip_forward_impulse_max_speed_scale;
            initial_dodge_vel.x *= ((max_speed_scale_x - 1.0f) * forward_speed_ratio) + 1.0f;
            initial_dodge_vel.y *= ((mutator.flip_side_impulse_max_speed_scale - 1.0f) * forward_speed_ratio) + 1.0f;
            if (should_dodge_backwards) {
                initial_dodge_vel.x *= mutator.flip_backward_impulse_scale_x;
            }

            // The game aims the dodge with the car's integer yaw (65536 units per turn, rounded) looked up in UE3's
            // 16384-entry sine table (index yaw >> 2), so the direction snaps to multiples of 4 units: the recorded
            // DodgeImpulse yaw is a multiple of 4 on 868 of 870 dodges (Nexto, PartyCannon, Wisp) and equals
            // (rint(yaw) >> 2) << 2 of the tick-start forward on 863 (the rest sit within 0.1 unit of a rounding edge).
            const int dodge_yaw = static_cast<int>(rintf(GlibcAtan2f(forward.y, forward.x) * (32768.0f / 3.14159265f)));
            const int dodge_yaw_idx = (dodge_yaw >> 2) & 16383;
            constexpr double kSinTabStep = 6.283185307179586476925 / 16384.0;
            const Vec3 forward_dir_2d(
                static_cast<float>(sin(static_cast<double>((dodge_yaw_idx + 4096) & 16383) * kSinTabStep)),
                static_cast<float>(sin(static_cast<double>(dodge_yaw_idx) * kSinTabStep)), 0.0f);
            Vec3 right_dir_2d(-forward_dir_2d.y, forward_dir_2d.x, 0.0f);
            Vec3 final_delta_vel = forward_dir_2d * initial_dodge_vel.x + right_dir_2d * initial_dodge_vel.y;
            ApplyCentralImpulseViaBtRoundtrip(car, final_delta_vel, mutator.car_mass);
            CalLimitCarVels(car, mutator);
        }
    } else {
        ApplyCentralImpulseViaBtRoundtrip(car, up * mutator.jump_immediate_speed, mutator.car_mass);
        CalLimitCarVels(car, mutator);

        car.has_double_jumped = 1;
    }
    return is_flip_input;
}

ROCKETGPU_HD void UpdateJumpState(
    CarState& car,
    const CarControls& controls,
    const MutatorConfig& mutator,
    bool jump_pressed,
    float dt,
    WheelRuntimeState* wheel_runtime_state = nullptr,
    bool cal_jump_blocked = false) {
    if (car.is_on_ground && !car.is_jumping) {
        if (!(car.has_jumped && car.internal_jump_time < mutator.jump_min_time + mutator.jump_reset_time_pad)) {
            car.has_jumped = 0;
            car.internal_jump_time = 0.0f;
        }
    }

    if (car.is_jumping) {
        if (!(car.internal_jump_time < mutator.jump_min_time ||
              (controls.jump && !car.jump_hold_broken && car.internal_jump_time < mutator.jump_max_time))) {
            car.is_jumping = 0;
        }
    }

    ContinueJump(car, controls, mutator, dt, wheel_runtime_state);
}

// v3's flip_time is a restart timer once no flip is active. Landing mid-flip restarts it
// at 0 (base.rs:975-979); otherwise it keeps counting on the ground (base.rs:670) and in the air (base.rs:784). RocketGPU
// counts a finished flip from its start (restart tick k <-> k + TORQUE_TIME/dt ticks, see rgpu_worker FLIP_CONV2), so
// convert that clock to v3's accumulated restart timer when has_flipped is cleared on landing.
// A zero timer with no flip active stays zero and means "unknown" (no pitch lock): RocketSim-convention set states
// write flip_time = 0 there, and the simulated timer is never exactly 0 at tick start.
ROCKETGPU_HD float CalFlipRestartTimeOnGround(const CarState& car, const MutatorConfig& mutator, float dt) {
    if (!car.has_flipped) {
        return car.internal_flip_time;
    }
    if (car.is_flipping) {
        return 0.0f;
    }
    if (car.internal_flip_time < mutator.flip_torque_time) {
        return 0.0f;  // no flip clock to convert (set state with has_flipped and a zero timer): leave the timer unknown
    }
    int k = static_cast<int>(rintf(car.internal_flip_time / dt)) - static_cast<int>(rintf(mutator.flip_torque_time / dt));
    if (k < 1) {
        k = 1;
    }
    if (k > 64) {
        return static_cast<float>(k) * dt;
    }
    float t = 0.0f;
    for (int i = 0; i < k; ++i) {
        t += dt;
    }
    return t;
}

// n additions of dt from 0: RocketGPU's float flip clock after n ticks.
ROCKETGPU_HD float CalAccumulatedTicks(int n, float dt) {
    if (n > 1024) {
        return static_cast<float>(n) * dt;
    }
    float t = 0.0f;
    for (int i = 0; i < n; ++i) {
        t += dt;
    }
    return t;
}

// A state set (tick_count_since_update == 0) carries jump_time and flip_time in v3's convention (car_state.rs). Rebuild
// the internal clocks the step runs on from them:
// - jump: the jump clock while is_jumping; after the jump, v3's time since it ended (it seeds the game's inactive-jump
//   timer in PreTickCar); 0 without a jump.
// - flip: during a flip the internal clock trails v3's by one tick (m v3 ticks -> m - 1 additions of dt). After the
//   torque ends it keeps counting from the flip start (m ticks since the end -> m + flip_torque_time / dt additions),
//   also on the landing tick, where the next update converts it back to the restart timer. With no flip it is v3's
//   restart timer.
ROCKETGPU_HD void CalInternalTimersFromSetState(CarState& car, const MutatorConfig& mutator, float dt) {
    car.internal_jump_time = (car.is_jumping || car.has_jumped) ? car.jump_time : 0.0f;
    const float flip_ticks = rintf(car.flip_time / dt);
    const int m = flip_ticks >= 0.0f ? static_cast<int>(Min(flip_ticks, 1.0e6f)) : 0;
    if (car.is_flipping) {
        car.internal_flip_time = CalAccumulatedTicks(m > 1 ? m - 1 : 0, dt);
    } else if (car.has_flipped) {
        car.internal_flip_time = CalAccumulatedTicks(m + static_cast<int>(rintf(mutator.flip_torque_time / dt)), dt);
    } else {
        car.internal_flip_time = car.flip_time;
    }
}

ROCKETGPU_HD void UpdateDoubleJumpOrFlipState(
    CarState& car,
    const CarControls& controls,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    bool jump_pressed,
    float forward_speed_uu,
    float dt,
    bool cal_press_as_air = false,
    float raw_pitch = 2.0f,
    bool* flip_started = nullptr) {
    if (car.is_on_ground && !cal_press_as_air) {
        car.has_double_jumped = 0;
        car.internal_flip_time = CalFlipRestartTimeOnGround(car, mutator, dt);
        car.has_flipped = 0;
        car.air_time = 0.0f;
        car.air_time_since_jump = 0.0f;
    } else {
        if (car.is_on_ground) {
            car.has_double_jumped = 0;
            car.internal_flip_time = CalFlipRestartTimeOnGround(car, mutator, dt);
            car.has_flipped = 0;
            car.air_time = 0.0f;
            car.air_time_since_jump = 0.0f;
        }
        car.air_time += dt;
        if (car.has_jumped && !car.is_jumping) {
            car.air_time_since_jump += dt;
        } else {
            car.air_time_since_jump = 0.0f;
        }

        if (jump_pressed && car.air_time_since_jump < mutator.doublejump_max_delay) {
            if (TryDoubleJumpOrFlip(car, controls, cfg, mutator, forward_speed_uu, raw_pitch) && flip_started) {
                *flip_started = true;
            }

            if (car.is_flipping && car.internal_flip_time == 0.0f) {
                car.internal_flip_time = -dt;
            }
        }
    }

    if (car.is_flipping) {
        car.internal_flip_time += dt;
    } else if (car.has_flipped || (car.internal_flip_time != 0.0f)) {
        car.internal_flip_time += dt;
    }
}

ROCKETGPU_HD void CapCarAngularSpeed(CarState& car, const MutatorConfig& mutator) {
    car.ang_vel = NativeSseClampLength(car.ang_vel, mutator.car_max_ang_speed,
        mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift);
}

ROCKETGPU_HD void CapBallSpeeds(BallState& ball, const MutatorConfig& mutator) {
    {
        const float max_speed_bt = mutator.ball_max_speed * UnitsUUToBT();
        const Vec3 vel_bt = ball.vel_bt;
        const float vel_bt_len_sq = LengthSq(vel_bt);
        if (vel_bt_len_sq > max_speed_bt * max_speed_bt) {
            const Vec3 vel_bt_normalized = mutator.native_sphere_rsqrt_table
                ? NativeSseSphereNormalizer{mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift}(vel_bt)
                : vel_bt * RsqrtScalar(vel_bt_len_sq);
            const Vec3 vel_bt_clamped = vel_bt_normalized * max_speed_bt;
            SetBallLinearVelocityBt(ball, vel_bt_clamped);
        }
    }

    ball.ang_vel = NativeSseClampLength(ball.ang_vel, mutator.ball_max_ang_speed,
        mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift);
}

ROCKETGPU_HD Vec3 BulletDampedLinearVelocityFromBt(const Vec3& vel_bt, float damping, float dt) {
    if (damping <= 0.0f) {
        return vel_bt;
    }
    const float damping_scale = powf(Max(0.0f, 1.0f - damping), dt);
    return vel_bt * damping_scale;
}

ROCKETGPU_HD Vec3 BulletDampedLinearVelocityFromBt(const Vec3& vel_bt, const MutatorConfig& mutator, float dt) {
    if (mutator.cached_ball_damping_drag == mutator.ball_drag && mutator.cached_ball_damping_dt == dt &&
        mutator.cached_ball_damping_scale >= 0.0f) {
        return vel_bt * mutator.cached_ball_damping_scale;
    }
    return BulletDampedLinearVelocityFromBt(vel_bt, mutator.ball_drag, dt);
}

ROCKETGPU_HD void ApplyBulletLinearDampingBall(BallState& ball, const MutatorConfig& mutator, float dt) {
    if (mutator.ball_drag <= 0.0f) {
        return;
    }
    SetBallLinearVelocityBt(ball, BulletDampedLinearVelocityFromBt(ball.vel_bt, mutator, dt));
}

ROCKETGPU_HD void IntegrateCarTransformOnly(CarState& car, float dt, bool soccar_platform_math = false, bool native_sse_math = false) {
    if (!(car.vel_bt.x == 0.0f && car.vel_bt.y == 0.0f && car.vel_bt.z == 0.0f)) {
        SetCarOriginBt(car, car.pos_bt + car.vel_bt * dt);
    }

    IntegrateCarOrientation(car, car.ang_vel, dt, "final2", soccar_platform_math, native_sse_math);
}

ROCKETGPU_HD void IntegrateBallTransformOnly(BallState& ball, float dt);
ROCKETGPU_HD void IntegrateBallTransformOnly(BallState& ball, const MutatorConfig& mutator, float dt);

ROCKETGPU_HD void EnsureBallBasisAuthoritative(BallState& ball) {
    if (!ball.basis_authoritative) {
        GetOfficialUploadBasisVectors(ball.rot, ball.basis_forward, ball.basis_right, ball.basis_up);
        ball.basis_authoritative = 1;
    }
}
ROCKETGPU_HD void IntegrateBallTransformOnly(BallState& ball, float dt) {
    IntegrateBallOriginBt(ball, ball.vel_bt, dt);

    EnsureBallBasisAuthoritative(ball);
}

ROCKETGPU_HD void IntegrateBallTransformOnly(BallState& ball, const MutatorConfig& mutator, float dt) {
    IntegrateBallTransformOnly(ball, dt);
    if (mutator.ball_shape == 1 || !mutator.ball_no_rot) {
        if (NativeSseMath(mutator)) {
            (void)IntegrateOrientationBasisColumns(
                ball.basis_forward, ball.basis_right, ball.basis_up, ball.rot, ball.ang_vel, dt, false, true);
        } else {
            GameIntegrateOrientationBasisColumns(ball.basis_forward, ball.basis_right, ball.basis_up, ball.rot, ball.ang_vel, dt);
        }
        ball.basis_authoritative = 1;
    }
}

ROCKETGPU_HD void ApplyBallVelocityImpulseCache(BallState& ball) {
    if (LengthSq(ball.vel_impulse_cache_bt) > 0.0f || LengthSq(ball.vel_impulse_cache) > 0.0f) {
        SetBallLinearVelocityBt(ball, ball.vel_bt + ball.vel_impulse_cache_bt);
        ball.vel_impulse_cache = Vec3(0.0f, 0.0f, 0.0f);
        ball.vel_impulse_cache_bt = Vec3(0.0f, 0.0f, 0.0f);
    }
}

ROCKETGPU_HD float QuantRoundTo(float x, float scale) {
    return rintf(x * scale) / scale;
}

ROCKETGPU_HD float QuantTruncOffset(float x, float scale) {
    const float k = truncf(x * scale);
    return k == 0.0f ? 0.0f : (k + copysignf(0.1f, k)) / scale;
}

// v3 shared/quantize.rs quantize_vec_ue3 bit for bit. Velocity mode:
// (i32)(v * scale) * (1 / scale) + sign * (0.1 * (1 / scale)); position mode: floor(p * scale + 0.5) / scale.
ROCKETGPU_HD float QuantTruncOffsetV3(float x, float scale) {
    const float inv_scale = 1.0f / scale;
    const float k = static_cast<float>(static_cast<int32_t>(x * scale));
    const float rounded = k * inv_scale;
    const float offset_mag = 0.1f * inv_scale;
    const float sgn = rounded == 0.0f ? 0.0f : (rounded > 0.0f ? 1.0f : -1.0f);
    const float offset = sgn * offset_mag;
    return rounded + offset;
}

// The game multiplies by the reciprocal (floor(p * 100 + 0.5) * 0.01f), where v3 divides by 100: RLPR Nexto car
// position steps p1 = q(p0) * 0.02f + v1 / 50 * (1 / 120.f), times 50, are bit exact on 98.9% of ticks with the
// product and 26.8% with the quotient.
ROCKETGPU_HD float QuantRoundPosV3(float x, float scale) {
    const float scaled = x * scale;
    return floorf(scaled + 0.5f) * (1.0f / scale);
}

ROCKETGPU_HD void QuantizeBallStateV3Exact(BallState& ball) {
    constexpr float kUuToBt = 1.0f / 50.0f;
    const Vec3 p = ball.pos_bt * 50.0f;
    SetBallOriginBt(ball, Vec3(QuantRoundPosV3(p.x, 100.0f), QuantRoundPosV3(p.y, 100.0f),
                               QuantRoundPosV3(p.z, 100.0f)) * kUuToBt);
    const Vec3 v = ball.vel_bt * 50.0f;
    SetBallLinearVelocityBt(ball, Vec3(QuantTruncOffsetV3(v.x, 100.0f), QuantTruncOffsetV3(v.y, 100.0f),
                                       QuantTruncOffsetV3(v.z, 100.0f)) * kUuToBt);
    ball.ang_vel = Vec3(QuantTruncOffsetV3(ball.ang_vel.x, 10000.0f),
                        QuantTruncOffsetV3(ball.ang_vel.y, 10000.0f),
                        QuantTruncOffsetV3(ball.ang_vel.z, 10000.0f));
    // The ball's RBState goes through the same 18-bit quaternion codec as the car's (recorded ball orientations step
    // bit exact from the quantized previous orientation on 98.5% of Nexto ticks, and on 0% without it).
    if (!ball.basis_authoritative) {
        GetOfficialUploadBasisVectors(ball.rot, ball.basis_forward, ball.basis_right, ball.basis_up);
        ball.basis_authoritative = 1;
    }
    ball.rot = GameQuantizeBasisColumns(ball.basis_forward, ball.basis_right, ball.basis_up);
}

ROCKETGPU_HD void QuantizeBallState(BallState& ball, bool v3_exact = false) {
    if (v3_exact) {
        QuantizeBallStateV3Exact(ball);
        return;
    }
    SetBallOriginUu(ball, Vec3(QuantRoundTo(ball.pos.x, 100.0f),
                               QuantRoundTo(ball.pos.y, 100.0f),
                               QuantRoundTo(ball.pos.z, 100.0f)));
    SetBallLinearVelocityUu(ball, Vec3(QuantTruncOffset(ball.vel.x, 100.0f),
                                       QuantTruncOffset(ball.vel.y, 100.0f),
                                       QuantTruncOffset(ball.vel.z, 100.0f)));
    ball.ang_vel = Vec3(QuantTruncOffset(ball.ang_vel.x, 10000.0f),
                        QuantTruncOffset(ball.ang_vel.y, 10000.0f),
                        QuantTruncOffset(ball.ang_vel.z, 10000.0f));
}

ROCKETGPU_HD void FinishBallTick(BallState& ball, const MutatorConfig& mutator) {
    ApplyBallVelocityImpulseCache(ball);
    ball.tick_count_since_update += 1;
    return;
}

#if defined(__CUDA_ARCH__)
#define ROCKETGPU_NOINLINE __noinline__
#else
#define ROCKETGPU_NOINLINE
#endif
ROCKETGPU_HD ROCKETGPU_NOINLINE void QuantizeCarState(CarState& car, bool v3_exact = false) {
    if (v3_exact) {
        constexpr float kUuToBt = 1.0f / 50.0f;
        const uint8_t touched_v3 = car.vel_touched_solve_stage;
        const Vec3 p = car.pos_bt * 50.0f;
        SetCarOriginBt(car, Vec3(QuantRoundPosV3(p.x, 100.0f), QuantRoundPosV3(p.y, 100.0f),
                                 QuantRoundPosV3(p.z, 100.0f)) * kUuToBt);
        const Vec3 v = car.vel_bt * 50.0f;
        SetCarLinearVelocityBt(car, Vec3(QuantTruncOffsetV3(v.x, 100.0f), QuantTruncOffsetV3(v.y, 100.0f),
                                         QuantTruncOffsetV3(v.z, 100.0f)) * kUuToBt);
        car.vel_touched_solve_stage = touched_v3;
        car.ang_vel = Vec3(QuantTruncOffsetV3(car.ang_vel.x, 10000.0f), QuantTruncOffsetV3(car.ang_vel.y, 10000.0f),
                           QuantTruncOffsetV3(car.ang_vel.z, 10000.0f));
        // The game's state quantize also sends the rotation through the 18-bit quaternion codec every tick.
        Vec3 fwd, right, up;
        CarBasisColumns(car, fwd, right, up);
        car.rot = GameQuantizeBasisColumns(fwd, right, up);
        car.basis_forward = fwd;
        car.basis_right = right;
        car.basis_up = up;
        car.basis_authoritative = 1;
        return;
    }

    const uint8_t touched = car.vel_touched_solve_stage;

    SetCarOriginUu(car, Vec3(floorf(car.pos.x * 100.0f + 0.5f) / 100.0f, floorf(car.pos.y * 100.0f + 0.5f) / 100.0f,
                             floorf(car.pos.z * 100.0f + 0.5f) / 100.0f));
    SetCarLinearVelocityUu(car, Vec3(QuantTruncOffset(car.vel.x, 100.0f), QuantTruncOffset(car.vel.y, 100.0f),
                                     QuantTruncOffset(car.vel.z, 100.0f)));
    car.vel_touched_solve_stage = touched;
    car.ang_vel = Vec3(QuantTruncOffset(car.ang_vel.x, 10000.0f), QuantTruncOffset(car.ang_vel.y, 10000.0f),
                       QuantTruncOffset(car.ang_vel.z, 10000.0f));
}

ROCKETGPU_HD void FinishCarTick(CarState& car, const MutatorConfig& mutator) {
    if (car.HasFlipOrJump() && !car.flip_or_jump_last_tick)
        car.last_flip_or_jump_start = car.tick_count_since_update;
    car.flip_or_jump_last_tick = car.HasFlipOrJump();
    car.ball_contact_last_tick = car.ball_contact_this_tick;
    if (car.is_demoed) {
        return;
    }

    if (LengthSq(car.vel_impulse_cache_bt) > 0.0f) {
        SetCarLinearVelocityBt(car, car.vel_bt + car.vel_impulse_cache_bt);
        car.vel_impulse_cache = Vec3(0.0f, 0.0f, 0.0f);
        car.vel_impulse_cache_bt = Vec3(0.0f, 0.0f, 0.0f);
    } else if (LengthSq(car.vel_impulse_cache) > 0.0f) {
        SetCarLinearVelocityUu(car, car.vel + car.vel_impulse_cache);
        car.vel_impulse_cache = Vec3(0.0f, 0.0f, 0.0f);
    }

    // v3 Arena::step_tick limits and quantizes at the start of the next tick (CalTickStartLimitAndQuantizeCar);
    // finish_physics_tick only adds the velocity impulse cache.
    car.tick_count_since_update += 1;
    return;
}

// v3 Arena::step_tick: limit_vels(car max speed, car max ang speed) then quantize::quantize on every car body
// before any pre-tick update, so the reported post-tick state is the unquantized integration result.
ROCKETGPU_HD void CalTickStartLimitAndQuantizeCar(CarState& car, const MutatorConfig& mutator) {
    if (car.is_demoed) {
        return;
    }
    const float max_speed_bt = mutator.car_max_speed * UnitsUUToBT();
    const Vec3 vel_bt = car.vel_bt;
    const float vel_bt_len_sq = LengthSq(vel_bt);
    if (vel_bt_len_sq > max_speed_bt * max_speed_bt) {
        const Vec3 vel_bt_normalized = mutator.native_sphere_rsqrt_table
            ? NativeSseSphereNormalizer{mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift}(vel_bt)
            : vel_bt * RsqrtScalar(vel_bt_len_sq);
        SetCarLinearVelocityBt(car, vel_bt_normalized * max_speed_bt);
    }
    CapCarAngularSpeed(car, mutator);
    // The game's RBState quantize: position k * (1 / 100), velocity and angular velocity
    // k * inv + sign * (inv * 0.1) at scales 100 and 10000; the v3-exact branch computes exactly this.
    QuantizeCarState(car, true);
}

// v3 Car::apply_flip_zdamp_prepass (run for every car after the tick-start limit/quantize, before any pre-tick update):
// an airborne flipping car with flip_time in [Z_DAMP_START, TORQUE_TIME] (not yet advanced this tick) has its
// vertical velocity scaled by 1 - Z_DAMP_120 when falling or before Z_DAMP_END. RocketGPU's flip_time trails v3's by
// one tick at tick start (it advances before the damping test in the flip update), so the window reads flip_time + dt.
ROCKETGPU_HD void CalFlipZDampPrepass(CarState& car, const MutatorConfig& mutator, float dt) {
    if (car.is_demoed || car.is_on_ground || !car.is_flipping) {
        return;
    }
    const float flip_time = car.internal_flip_time + dt;
    if (!(flip_time >= 0.15f && flip_time <= mutator.flip_torque_time)) {
        return;
    }
    if (car.vel_bt.z < 0.0f || flip_time < 0.21f) {
        const Vec3 v = car.vel_bt;
        SetCarLinearVelocityBt(car, Vec3(v.x, v.y, v.z * (1.0f - 0.35f)));
    }
}

ROCKETGPU_HD void CalTickStartLimitAndQuantizeBall(BallState& ball, const MutatorConfig& mutator) {
    CapBallSpeeds(ball, mutator);
    QuantizeBallState(ball, (true & 1u) != 0);
}

ROCKETGPU_HD void PostTickCar(CarState& car, CarControls controls, const MutatorConfig& mutator, float dt) {
    if (car.is_demoed) {
        return;
    }
    if (car.car_contact.cooldown_timer > 0.0f) {
        car.car_contact.cooldown_timer = Max(0.0f, car.car_contact.cooldown_timer - dt);
    }
    car.last_controls = controls;
}

struct GjkUsageBitfield {
    bool used_vertex_a = false;
    bool used_vertex_b = false;
    bool used_vertex_c = false;
    bool used_vertex_d = false;

    ROCKETGPU_HD void reset() {
        used_vertex_a = false;
        used_vertex_b = false;
        used_vertex_c = false;
        used_vertex_d = false;
    }
};

struct GjkSubSimplexClosestResult {
    Vec3 closest_point_on_simplex = Vec3(0.0f, 0.0f, 0.0f);
    GjkUsageBitfield used_vertices;
    float barycentric_coords[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    bool degenerate = false;

    ROCKETGPU_HD void reset() {
        closest_point_on_simplex = Vec3(0.0f, 0.0f, 0.0f);
        used_vertices.reset();
        set_barycentric_coordinates();
        degenerate = false;
    }

    ROCKETGPU_HD bool is_valid() const {
        return
            barycentric_coords[0] >= 0.0f &&
            barycentric_coords[1] >= 0.0f &&
            barycentric_coords[2] >= 0.0f &&
            barycentric_coords[3] >= 0.0f;
    }

    ROCKETGPU_HD void set_barycentric_coordinates(
        float a = 0.0f,
        float b = 0.0f,
        float c = 0.0f,
        float d = 0.0f) {
        barycentric_coords[0] = a;
        barycentric_coords[1] = b;
        barycentric_coords[2] = c;
        barycentric_coords[3] = d;
    }
};

struct GjkSimplexSolver {
    int num_vertices = 0;
    Vec3 simplex_w[4];
    Vec3 simplex_p[4];
    Vec3 simplex_q[4];
    Vec3 cached_p1 = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 cached_p2 = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 cached_v = Vec3(0.0f, 0.0f, 0.0f);

    Vec3 last_w = Vec3(1.0e18f, 1.0e18f, 1.0e18f);
    GjkSubSimplexClosestResult cached_bc;
    bool cached_valid_closest = false;
    bool needs_update = true;

    ROCKETGPU_HD void reset() {
        cached_valid_closest = false;
        num_vertices = 0;
        needs_update = true;
        last_w = Vec3(1.0e18f, 1.0e18f, 1.0e18f);
        cached_bc.reset();
    }

    ROCKETGPU_HD void add_vertex(const Vec3& w, const Vec3& p, const Vec3& q) {
        last_w = w;
        needs_update = true;
        simplex_w[num_vertices] = w;
        simplex_p[num_vertices] = p;
        simplex_q[num_vertices] = q;
        ++num_vertices;
    }

    ROCKETGPU_HD bool full_simplex() const {
        return num_vertices == 4;
    }

    ROCKETGPU_HD bool in_simplex(const Vec3& w) const {
        constexpr float kEqualVertexThresholdBT2 = 0.0001f;
        for (int i = 0; i < num_vertices; ++i) {
            if (LengthSq(simplex_w[i] - w) <= kEqualVertexThresholdBT2) {
                return true;
            }
        }
        return w.x == last_w.x && w.y == last_w.y && w.z == last_w.z;
    }

    ROCKETGPU_HD void remove_vertex(int index) {
        --num_vertices;
        simplex_w[index] = simplex_w[num_vertices];
        simplex_p[index] = simplex_p[num_vertices];
        simplex_q[index] = simplex_q[num_vertices];
    }

    ROCKETGPU_HD void reduce_vertices(const GjkUsageBitfield& used) {
        if ((num_vertices >= 4) && !used.used_vertex_d) {
            remove_vertex(3);
        }
        if ((num_vertices >= 3) && !used.used_vertex_c) {
            remove_vertex(2);
        }
        if ((num_vertices >= 2) && !used.used_vertex_b) {
            remove_vertex(1);
        }
        if ((num_vertices >= 1) && !used.used_vertex_a) {
            remove_vertex(0);
        }
    }

    ROCKETGPU_HD bool closest_pt_point_triangle(
        const Vec3& p,
        const Vec3& a,
        const Vec3& b,
        const Vec3& c,
        GjkSubSimplexClosestResult& result) {
        result.used_vertices.reset();

        const Vec3 ab = b - a;
        const Vec3 ac = c - a;
        const Vec3 ap = p - a;
        const float d1 = Dot(ab, ap);
        const float d2 = Dot(ac, ap);
        if (d1 <= 0.0f && d2 <= 0.0f) {
            result.closest_point_on_simplex = a;
            result.used_vertices.used_vertex_a = true;
            result.set_barycentric_coordinates(1.0f, 0.0f, 0.0f);
            return true;
        }

        const Vec3 bp = p - b;
        const float d3 = Dot(ab, bp);
        const float d4 = Dot(ac, bp);
        if (d3 >= 0.0f && d4 <= d3) {
            result.closest_point_on_simplex = b;
            result.used_vertices.used_vertex_b = true;
            result.set_barycentric_coordinates(0.0f, 1.0f, 0.0f);
            return true;
        }

        const float vc = d1 * d4 - d3 * d2;
        if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
            const float v = d1 / (d1 - d3);
            result.closest_point_on_simplex = a + ab * v;
            result.used_vertices.used_vertex_a = true;
            result.used_vertices.used_vertex_b = true;
            result.set_barycentric_coordinates(1.0f - v, v, 0.0f);
            return true;
        }

        const Vec3 cp = p - c;
        const float d5 = Dot(ab, cp);
        const float d6 = Dot(ac, cp);
        if (d6 >= 0.0f && d5 <= d6) {
            result.closest_point_on_simplex = c;
            result.used_vertices.used_vertex_c = true;
            result.set_barycentric_coordinates(0.0f, 0.0f, 1.0f);
            return true;
        }

        const float vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
            const float w = d2 / (d2 - d6);
            result.closest_point_on_simplex = a + ac * w;
            result.used_vertices.used_vertex_a = true;
            result.used_vertices.used_vertex_c = true;
            result.set_barycentric_coordinates(1.0f - w, 0.0f, w);
            return true;
        }

        const float va = d3 * d6 - d5 * d4;
        if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
            const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            result.closest_point_on_simplex = b + (c - b) * w;
            result.used_vertices.used_vertex_b = true;
            result.used_vertices.used_vertex_c = true;
            result.set_barycentric_coordinates(0.0f, 1.0f - w, w);
            return true;
        }

        const float denom = 1.0f / (va + vb + vc);
        const float v = vb * denom;
        const float w = vc * denom;
        result.closest_point_on_simplex = a + ab * v + ac * w;
        result.used_vertices.used_vertex_a = true;
        result.used_vertices.used_vertex_b = true;
        result.used_vertices.used_vertex_c = true;
        result.set_barycentric_coordinates(1.0f - v - w, v, w);
        return true;
    }

    ROCKETGPU_HD int point_outside_of_plane(
        const Vec3& p,
        const Vec3& a,
        const Vec3& b,
        const Vec3& c,
        const Vec3& d) {
        const Vec3 normal = Cross(b - a, c - a);
        const float signp = Dot(p - a, normal);
        const float signd = Dot(d - a, normal);
        if (signd * signd < (1.0e-4f * 1.0e-4f)) {
            return -1;
        }
        const int plane_test = signp * signd < 0.0f;
        return plane_test;
    }

    ROCKETGPU_HD bool closest_pt_point_tetrahedron(
        const Vec3& p,
        const Vec3& a,
        const Vec3& b,
        const Vec3& c,
        const Vec3& d,
        GjkSubSimplexClosestResult& final_result) {
        GjkSubSimplexClosestResult temp_result;
        final_result.closest_point_on_simplex = p;
        final_result.used_vertices.reset();
        final_result.used_vertices.used_vertex_a = true;
        final_result.used_vertices.used_vertex_b = true;
        final_result.used_vertices.used_vertex_c = true;
        final_result.used_vertices.used_vertex_d = true;

        const int outside_abc = point_outside_of_plane(p, a, b, c, d);
        const int outside_acd = point_outside_of_plane(p, a, c, d, b);
        const int outside_adb = point_outside_of_plane(p, a, d, b, c);
        const int outside_bdc = point_outside_of_plane(p, b, d, c, a);

        if (outside_abc < 0 || outside_acd < 0 || outside_adb < 0 || outside_bdc < 0) {
            final_result.degenerate = true;
            return false;
        }

        if (!outside_abc && !outside_acd && !outside_adb && !outside_bdc) {
            return false;
        }

        float best_sq_dist = 3.402823466e+38f;
        if (outside_abc) {
            closest_pt_point_triangle(p, a, b, c, temp_result);
            const Vec3 q = temp_result.closest_point_on_simplex;
            const float sq_dist = LengthSq(q - p);
            if (sq_dist < best_sq_dist) {
                best_sq_dist = sq_dist;
                final_result.closest_point_on_simplex = q;
                final_result.used_vertices.reset();
                final_result.used_vertices.used_vertex_a = temp_result.used_vertices.used_vertex_a;
                final_result.used_vertices.used_vertex_b = temp_result.used_vertices.used_vertex_b;
                final_result.used_vertices.used_vertex_c = temp_result.used_vertices.used_vertex_c;
                final_result.set_barycentric_coordinates(
                    temp_result.barycentric_coords[0],
                    temp_result.barycentric_coords[1],
                    temp_result.barycentric_coords[2],
                    0.0f);
            }
        }

        if (outside_acd) {
            closest_pt_point_triangle(p, a, c, d, temp_result);
            const Vec3 q = temp_result.closest_point_on_simplex;
            const float sq_dist = LengthSq(q - p);
            if (sq_dist < best_sq_dist) {
                best_sq_dist = sq_dist;
                final_result.closest_point_on_simplex = q;
                final_result.used_vertices.reset();
                final_result.used_vertices.used_vertex_a = temp_result.used_vertices.used_vertex_a;
                final_result.used_vertices.used_vertex_c = temp_result.used_vertices.used_vertex_b;
                final_result.used_vertices.used_vertex_d = temp_result.used_vertices.used_vertex_c;
                final_result.set_barycentric_coordinates(
                    temp_result.barycentric_coords[0],
                    0.0f,
                    temp_result.barycentric_coords[1],
                    temp_result.barycentric_coords[2]);
            }
        }

        if (outside_adb) {
            closest_pt_point_triangle(p, a, d, b, temp_result);
            const Vec3 q = temp_result.closest_point_on_simplex;
            const float sq_dist = LengthSq(q - p);
            if (sq_dist < best_sq_dist) {
                best_sq_dist = sq_dist;
                final_result.closest_point_on_simplex = q;
                final_result.used_vertices.reset();
                final_result.used_vertices.used_vertex_a = temp_result.used_vertices.used_vertex_a;
                final_result.used_vertices.used_vertex_b = temp_result.used_vertices.used_vertex_c;
                final_result.used_vertices.used_vertex_d = temp_result.used_vertices.used_vertex_b;
                final_result.set_barycentric_coordinates(
                    temp_result.barycentric_coords[0],
                    temp_result.barycentric_coords[2],
                    0.0f,
                    temp_result.barycentric_coords[1]);
            }
        }

        if (outside_bdc) {
            closest_pt_point_triangle(p, b, d, c, temp_result);
            const Vec3 q = temp_result.closest_point_on_simplex;
            const float sq_dist = LengthSq(q - p);
            if (sq_dist < best_sq_dist) {
                best_sq_dist = sq_dist;
                final_result.closest_point_on_simplex = q;
                final_result.used_vertices.reset();
                final_result.used_vertices.used_vertex_b = temp_result.used_vertices.used_vertex_a;
                final_result.used_vertices.used_vertex_c = temp_result.used_vertices.used_vertex_c;
                final_result.used_vertices.used_vertex_d = temp_result.used_vertices.used_vertex_b;
                final_result.set_barycentric_coordinates(
                    0.0f,
                    temp_result.barycentric_coords[0],
                    temp_result.barycentric_coords[2],
                    temp_result.barycentric_coords[1]);
            }
        }

        return true;
    }

    ROCKETGPU_HD bool update_closest_vector_and_points() {
        if (needs_update) {
            cached_bc.reset();
            needs_update = false;

            switch (num_vertices) {
        case 0:
            cached_valid_closest = false;
            break;
        case 1:
            cached_p1 = simplex_p[0];
            cached_p2 = simplex_q[0];
            cached_v = cached_p1 - cached_p2;
            cached_bc.reset();
            cached_bc.set_barycentric_coordinates(1.0f, 0.0f, 0.0f, 0.0f);
            cached_valid_closest = cached_bc.is_valid();
            break;
        case 2: {
            const Vec3& from = simplex_w[0];
            const Vec3& to = simplex_w[1];
            const Vec3 diff = -from;
            const Vec3 v = to - from;
            float t = Dot(v, diff);
            if (t > 0.0f) {
                const float dot_vv = Dot(v, v);
                if (t < dot_vv) {
                    t /= dot_vv;
                    cached_bc.used_vertices.used_vertex_a = true;
                    cached_bc.used_vertices.used_vertex_b = true;
                } else {
                    t = 1.0f;
                    cached_bc.used_vertices.used_vertex_b = true;
                }
            } else {
                t = 0.0f;
                cached_bc.used_vertices.used_vertex_a = true;
            }

            cached_bc.set_barycentric_coordinates(1.0f - t, t, 0.0f, 0.0f);
            cached_p1 = simplex_p[0] + (simplex_p[1] - simplex_p[0]) * t;
            cached_p2 = simplex_q[0] + (simplex_q[1] - simplex_q[0]) * t;
            cached_v = cached_p1 - cached_p2;
            reduce_vertices(cached_bc.used_vertices);
            cached_valid_closest = cached_bc.is_valid();
            break;
        }
        case 3:
            closest_pt_point_triangle(Vec3(0.0f, 0.0f, 0.0f), simplex_w[0], simplex_w[1], simplex_w[2], cached_bc);
            cached_p1 =
                simplex_p[0] * cached_bc.barycentric_coords[0] +
                simplex_p[1] * cached_bc.barycentric_coords[1] +
                simplex_p[2] * cached_bc.barycentric_coords[2];
            cached_p2 =
                simplex_q[0] * cached_bc.barycentric_coords[0] +
                simplex_q[1] * cached_bc.barycentric_coords[1] +
                simplex_q[2] * cached_bc.barycentric_coords[2];
            cached_v = cached_p1 - cached_p2;
            reduce_vertices(cached_bc.used_vertices);
            cached_valid_closest = cached_bc.is_valid();
            break;
        case 4: {
            const bool has_separation = closest_pt_point_tetrahedron(
                Vec3(0.0f, 0.0f, 0.0f),
                simplex_w[0],
                simplex_w[1],
                simplex_w[2],
                simplex_w[3],
                cached_bc);
            if (has_separation) {
                cached_p1 =
                    simplex_p[0] * cached_bc.barycentric_coords[0] +
                    simplex_p[1] * cached_bc.barycentric_coords[1] +
                    simplex_p[2] * cached_bc.barycentric_coords[2] +
                    simplex_p[3] * cached_bc.barycentric_coords[3];
                cached_p2 =
                    simplex_q[0] * cached_bc.barycentric_coords[0] +
                    simplex_q[1] * cached_bc.barycentric_coords[1] +
                    simplex_q[2] * cached_bc.barycentric_coords[2] +
                    simplex_q[3] * cached_bc.barycentric_coords[3];
                cached_v = cached_p1 - cached_p2;
                reduce_vertices(cached_bc.used_vertices);
                cached_valid_closest = cached_bc.is_valid();
            } else {
                if (cached_bc.degenerate) {
                    cached_valid_closest = false;
                } else {
                    cached_valid_closest = true;
                    cached_v = Vec3(0.0f, 0.0f, 0.0f);
                }
            }
            break;
        }
        default:
            cached_valid_closest = false;
            break;
            }
        }

        return cached_valid_closest;
    }

    ROCKETGPU_HD bool closest(Vec3& v) {
        const bool success = update_closest_vector_and_points();
        v = cached_v;
        return success;
    }

    ROCKETGPU_HD void compute_points(Vec3& p1, Vec3& p2) {
        update_closest_vector_and_points();
        p1 = cached_p1;
        p2 = cached_p2;
    }
};

struct ScalarSphereRayNormalizer {
    ROCKETGPU_HD Vec3 operator()(const Vec3& value) const {
        constexpr float eps = 1.1920929e-7f;
        const float length2 = Dot(value, value);
        return length2 < eps * eps
            ? Vec3(-1.0f, -1.0f, -1.0f) * (1.0f / SqrtScalar(3.0f))
            : value * (1.0f / SqrtScalar(length2));
    }
};

ROCKETGPU_HD Vec3 WheelRayCarHitboxHalfExtentWithBulletSafeMarginBT(const CarConfig& other_cfg) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kDefaultMarginBt = 0.04f;

    const Vec3 half_extent_bt = other_cfg.hitbox_size * (0.5f * kUUToBT);
    const float min_dimension_bt = Min(half_extent_bt.x, Min(half_extent_bt.y, half_extent_bt.z));
    const float safe_margin_bt = Min(kDefaultMarginBt, 0.1f * min_dimension_bt);

    const float implicit_margin_bt = safe_margin_bt;
    const Vec3 implicit_bt = half_extent_bt - Vec3(implicit_margin_bt, implicit_margin_bt, implicit_margin_bt);
    return implicit_bt + Vec3(safe_margin_bt, safe_margin_bt, safe_margin_bt);
}

// Closest point from p to the car's rounded hitbox (inner box + margin), as btGjkPairDetector reports it
// for a zero-margin point vs a btBoxShape: distance to the surface and the outward normal on the box.
ROCKETGPU_HD bool WheelRayCarHitboxClosestBT(const Vec3& p, const Vec3& center, const Vec3& f, const Vec3& r,
    const Vec3& u, const Vec3& inner, float margin, float& dist, Vec3& normal) {
    const Vec3 d = p - center;
    const Vec3 l(Dot(d, f), Dot(d, r), Dot(d, u));
    const Vec3 q(Max(-inner.x, Min(inner.x, l.x)), Max(-inner.y, Min(inner.y, l.y)), Max(-inner.z, Min(inner.z, l.z)));
    const Vec3 dl = l - q;
    const float len = Length(dl);
    if (!(len > 0.0f)) {
        return false;  // inside the core box: no penetration solver, so GJK reports no result
    }
    const Vec3 nl = dl * (1.0f / len);
    normal = f * nl.x + r * nl.y + u * nl.z;
    dist = len - margin;
    return true;
}

// Wheel ray vs car hitbox as the game's btCollisionWorld::rayTestSingle does it (Bullet 2.82, default flags):
// btGjkConvexCast of a point against the rounded box. Conservative advancement stops once the point is within
// 0.001 bt of the surface, so the fraction and normal come from that point, not the exact surface hit.
ROCKETGPU_HD bool CastWheelRayAgainstCarHitboxGjkCast(
    const Vec3& ray_source_bt,
    const Vec3& ray_dir,
    float max_len,
    const CarState& other_car,
    const CarConfig& other_cfg,
    ArenaRayHit& out_hit,
    float max_len_bt_override) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kBTToUU = 50.0f;
    constexpr float kRadius = 0.001f;
    constexpr int kMaxIterations = 32;
    const float max_len_bt = max_len_bt_override > 0.0f ? max_len_bt_override : max_len * kUUToBT;
    const Vec3 ray_to_bt = ray_source_bt + ray_dir * max_len_bt;
    const Vec3 half_raw = other_cfg.hitbox_size * (0.5f * kUUToBT);
    const float margin = Min(0.1f * Min(half_raw.x, Min(half_raw.y, half_raw.z)), 0.04f);  // btBoxShape setSafeMargin
    const Vec3 inner = half_raw - Vec3(margin, margin, margin);
    Vec3 f(1.0f, 0.0f, 0.0f), rt(0.0f, 1.0f, 0.0f), u(0.0f, 0.0f, 1.0f);
    CarBasisColumns(other_car, f, rt, u);
    const Vec3 center = other_car.pos_bt + RotateViaBasis(f, rt, u, other_cfg.hitbox_offset * kUUToBT);
    {
        const float rr = Length(half_raw) + max_len_bt + 1.0e-3f;
        if (LengthSq(center - ray_source_bt) > rr * rr) {
            return false;
        }
    }

    const Vec3 r = ray_to_bt - ray_source_bt;
    float lambda = 0.0f;
    float last_lambda = 0.0f;
    float dist = 0.0f;
    Vec3 n(0.0f, 0.0f, 0.0f);
    if (!WheelRayCarHitboxClosestBT(ray_source_bt, center, f, rt, u, inner, margin, dist, n)) {
        return false;
    }
    bool penetrated = false;
    int iter = 0;
    while (dist > kRadius) {
        if (++iter > kMaxIterations) {
            return false;
        }
        lambda = lambda - dist / Dot(r, n);
        if (lambda > 1.0f || lambda < 0.0f || lambda <= last_lambda) {
            return false;
        }
        last_lambda = lambda;
        float next_dist = 0.0f;
        if (!WheelRayCarHitboxClosestBT(ray_source_bt + r * lambda, center, f, rt, u, inner, margin, next_dist, n)) {
            return false;
        }
        if (next_dist < 0.0f) {
            penetrated = true;  // Bullet returns lastLambda (== lambda here) with this normal
            break;
        }
        dist = next_dist;
    }
    if (!penetrated && Dot(n, r) >= 0.0f) {
        return false;
    }

    out_hit.t = lambda * max_len;
    out_hit.normal = Normalize(n, Vec3(0.0f, 0.0f, 1.0f));
    out_hit.point_bt = ray_source_bt + r * lambda;
    out_hit.point = out_hit.point_bt * kBTToUU;
    out_hit.hit = 1;
    out_hit.kind = ROCKETGPU_ARENA_RAY_HIT_CAR;
    return true;
}

ROCKETGPU_HD float ArenaTriangleEdgeAngle(const ArenaMeshTriangle& tri, int edge_idx) {
    if (edge_idx == 0) {
        return tri.edge_v0v1_angle;
    }
    if (edge_idx == 1) {
        return tri.edge_v1v2_angle;
    }
    return tri.edge_v2v0_angle;
}

ROCKETGPU_HD uint8_t ArenaTriangleEdgeConvexFlag(int edge_idx) {
    return edge_idx == 0 ? ROCKETGPU_TRI_INFO_V0V1_CONVEX : (edge_idx == 1 ? ROCKETGPU_TRI_INFO_V1V2_CONVEX : ROCKETGPU_TRI_INFO_V2V0_CONVEX);
}

ROCKETGPU_HD uint8_t ArenaTriangleEdgeSwapFlag(int edge_idx) {
    return edge_idx == 0 ? ROCKETGPU_TRI_INFO_V0V1_SWAP_NORMALB : (edge_idx == 1 ? ROCKETGPU_TRI_INFO_V1V2_SWAP_NORMALB : ROCKETGPU_TRI_INFO_V2V0_SWAP_NORMALB);
}

ROCKETGPU_HD Vec3 ClosestPointOnSegmentBT(const Vec3& point, const Vec3& seg_a, const Vec3& seg_b) {
    const Vec3 seg = seg_b - seg_a;
    const float seg_len_sq = LengthSq(seg);
    if (seg_len_sq <= 1.0e-12f) {
        return seg_a;
    }

    const float t = Clamp(Dot(point - seg_a, seg) / seg_len_sq, 0.0f, 1.0f);
    return seg_a + seg * t;
}

ROCKETGPU_HD float InternalEdgeSignedAngle(const Vec3& ref_axis0, const Vec3& ref_axis1, const Vec3& swing_axis, bool soccar_platform_math = false) {
#if defined(_WIN32)

    if (soccar_platform_math) return SafeAtan2(Dot(swing_axis, ref_axis0), Dot(swing_axis, ref_axis1));
#else
    (void)soccar_platform_math;
#endif
    return GlibcAtan2f(Dot(swing_axis, ref_axis0), Dot(swing_axis, ref_axis1));
}

ROCKETGPU_HD bool ClampArenaTriangleInternalEdgeNormal(
    const Vec3& edge,
    const Vec3& tri_normal,
    const Vec3& contact_normal,
    float corrected_edge_angle,
    Vec3& out_clamped_normal,
    bool soccar_platform_math = false) {
    const Vec3 edge_cross = Normalize(Cross(edge, tri_normal), tri_normal);
    const float current_angle = InternalEdgeSignedAngle(edge_cross, tri_normal, contact_normal, soccar_platform_math);

    if (corrected_edge_angle < 0.0f) {
        if (current_angle < corrected_edge_angle) {
            const float diff_angle = corrected_edge_angle - current_angle;
            out_clamped_normal = BulletMatrixRotate(BulletQuatFromAxisAngle(edge, diff_angle), contact_normal);
            return true;
        }
    } else if (current_angle > corrected_edge_angle) {
        const float diff_angle = corrected_edge_angle - current_angle;
        out_clamped_normal = BulletMatrixRotate(BulletQuatFromAxisAngle(edge, diff_angle), contact_normal);
        return true;
    }

    return false;
}

ROCKETGPU_HD Vec3 ClosestPointOnSegmentBTClamped(const Vec3& point, const Vec3& seg_a, const Vec3& seg_b, float t_lo, float t_hi) {
    const Vec3 seg = seg_b - seg_a;
    const float seg_len_sq = LengthSq(seg);
    if (seg_len_sq <= 1.0e-12f) {
        return seg_a;
    }
    const float t = Clamp(Dot(point - seg_a, seg) / seg_len_sq, t_lo, t_hi);
    return seg_a + seg * t;
}

template <bool kGameMeshEdgeRules>
ROCKETGPU_HD bool MaybeAdjustArenaTriangleInternalEdgeContactT(
    const ArenaMeshTriangle& tri,
    float cal_origin_y,
    const Vec3& point_on_a_bt,
    float distance_bt,
    Vec3& io_point_on_b_bt,
    Vec3& io_normal,
    ArenaTriangleInternalEdgeDebug* debug,
    bool soccar_platform_math,
    const uint32_t* native_table, int native_shift) {
    constexpr float search_t_lo = kGameMeshEdgeRules ? 1.0e-5f : 0.0f;
    constexpr float search_t_hi = kGameMeshEdgeRules ? 1.0f - 1.0e-5f : 1.0f;

    constexpr float kEdgeDistanceThreshold = 0.1f;
    constexpr float kConvexEpsilon = 0.0f;
    constexpr float kMaxEdgeAngleThreshold = 2.0f * Pi();

    const Vec3 tri_normal = native_table
        ? NativeSseSphereNormalizer{native_table, native_shift}(Cross(tri.b - tri.a, tri.c - tri.a))
        : tri.normal;

    const float contact_len_sq = LengthSq(io_normal);
    Vec3 contact_normal = tri_normal;
    if (contact_len_sq > 1e-12f) {
        contact_normal = native_table
            ? NativeSseSphereNormalizer{native_table, native_shift}(io_normal)
            : io_normal * (1.0f / sqrtf(contact_len_sq));
    }

    const Vec3 raw_contact_normal = io_normal;
    const Vec3 contact = (kGameMeshEdgeRules && cal_origin_y != 0.0f)
        ? Vec3(io_point_on_b_bt.x, io_point_on_b_bt.y - cal_origin_y, io_point_on_b_bt.z)
        : io_point_on_b_bt;
    if (debug) {
        debug->point_on_b_in = io_point_on_b_bt;
        debug->normal_in = contact_normal;
        debug->tri_normal = tri_normal;
    }

    int best_edge = -1;
    float best_edge_distance = 1.0e30f;
    const Vec3 edge_start[3] = {tri.a, tri.b, tri.c};
    const Vec3 edge_end[3] = {tri.b, tri.c, tri.a};

    uint32_t on_edge_mask = 0u;
    for (int edge_idx = 0; edge_idx < 3; ++edge_idx) {
        const float edge_angle = ArenaTriangleEdgeAngle(tri, edge_idx);
        if (Abs(edge_angle) >= kMaxEdgeAngleThreshold) {
            continue;
        }
        const Vec3 nearest = ClosestPointOnSegmentBTClamped(
            contact, edge_start[edge_idx], edge_end[edge_idx], search_t_lo, search_t_hi);
        const float distance = Length(contact - nearest);
        if (kGameMeshEdgeRules && distance < 1.1920928955078125e-07f) {
            on_edge_mask |= 1u << edge_idx;
        }
        if (distance < best_edge_distance) {
            best_edge_distance = distance;
            best_edge = edge_idx;
        }
    }
    if (debug) {
        debug->best_edge = best_edge;
        debug->best_edge_distance = best_edge_distance;
        debug->best_edge_angle = best_edge >= 0 ? ArenaTriangleEdgeAngle(tri, best_edge) : 0.0f;
    }

    if (best_edge < 0 || best_edge_distance >= kEdgeDistanceThreshold) {
        if (debug) {
            debug->point_on_b_out = io_point_on_b_bt;
            debug->normal_out = io_normal;
        }
        return false;
    }

    const float edge_angle = ArenaTriangleEdgeAngle(tri, best_edge);
    if (Abs(edge_angle) >= kMaxEdgeAngleThreshold) {
        const Vec3 edge = edge_start[best_edge] - edge_end[best_edge];
        if (LengthSq(edge) > 1.0e-12f) {
            const Vec3 edge_unit = Normalize(edge, Vec3(1.0f, 0.0f, 0.0f));
            const Vec3 feature_normal = Normalize(Cross(edge_unit, tri_normal), contact_normal);
            Vec3 repaired_normal = contact_normal - edge_unit * Dot(contact_normal, edge_unit);
            if (LengthSq(repaired_normal) > 1.0e-12f) {
                repaired_normal = Normalize(repaired_normal, feature_normal);
            } else {
                repaired_normal = feature_normal;
            }
            if (Dot(repaired_normal, contact_normal) < 0.0f) {
                repaired_normal = -repaired_normal;
            }
            io_normal = repaired_normal;
            io_point_on_b_bt = point_on_a_bt - io_normal * distance_bt;
        }
        if (debug) {
            debug->point_on_b_out = io_point_on_b_bt;
            debug->normal_out = io_normal;
        }
        return true;
    }

    bool is_near_edge = false;
    int num_concave_edge_hits = 0;

    for (int edge_idx = 0; edge_idx < 3; ++edge_idx) {
        const float candidate_edge_angle = ArenaTriangleEdgeAngle(tri, edge_idx);
        if (Abs(candidate_edge_angle) >= kMaxEdgeAngleThreshold) {
            continue;
        }

        const Vec3 nearest = ClosestPointOnSegmentBT(contact, edge_start[edge_idx], edge_end[edge_idx]);
        const float edge_distance = Length(contact - nearest);
        if (edge_distance >= kEdgeDistanceThreshold || (best_edge != edge_idx && (on_edge_mask & (1u << edge_idx)) == 0u)) {
            continue;
        }

        is_near_edge = true;
        if (candidate_edge_angle == 0.0f) {
            ++num_concave_edge_hits;
            continue;
        }

        const bool is_edge_convex = (tri.edge_flags & ArenaTriangleEdgeConvexFlag(edge_idx)) != 0;
        const float swap_factor = is_edge_convex ? 1.0f : -1.0f;
        const Vec3 nA = tri_normal * swap_factor;
        const Vec3 edge = edge_start[edge_idx] - edge_end[edge_idx];

        Vec3 computed_normal_b =
            BulletQuatRotate(BulletQuatFromAxisAngle(edge, candidate_edge_angle), tri_normal);
        if ((tri.edge_flags & ArenaTriangleEdgeSwapFlag(edge_idx)) != 0) {
            computed_normal_b = -computed_normal_b;
        }
        const Vec3 nB = computed_normal_b * swap_factor;

        const float ndot_a = Dot(contact_normal, nA);
        const float ndot_b = Dot(contact_normal, nB);
        const bool back_facing = (ndot_a < kConvexEpsilon) && (ndot_b < kConvexEpsilon);
        if (debug) {
            debug->active_edge = edge_idx;
            debug->active_edge_distance = edge_distance;
            debug->active_edge_angle = candidate_edge_angle;
            debug->active_edge_convex = is_edge_convex ? 1 : 0;
            debug->back_facing = back_facing ? 1 : 0;
            debug->nA = nA;
            debug->nB = nB;
        }
        if (back_facing) {
            ++num_concave_edge_hits;
            continue;
        }

        Vec3 clamped_local_normal(0.0f, 0.0f, 1.0f);
        const bool clamped = ClampArenaTriangleInternalEdgeNormal(
                edge,
                tri_normal * swap_factor,
                edge_idx == 0 ? contact_normal : raw_contact_normal,
                candidate_edge_angle,
                clamped_local_normal,
                soccar_platform_math);
        const bool accepted_clamp = clamped && Dot(clamped_local_normal, tri_normal) > 0.0f;
        if (debug) {
            debug->clamped = clamped ? 1 : 0;
            debug->accepted_clamp = accepted_clamp ? 1 : 0;
            debug->clamped_local_normal = clamped_local_normal;
        }
        if (accepted_clamp) {
            io_normal = clamped_local_normal;
            io_point_on_b_bt = point_on_a_bt - io_normal * distance_bt;
            contact_normal = io_normal;
        }
    }

    if (is_near_edge && num_concave_edge_hits > 0) {
        const Vec3 new_normal = tri_normal;
        if (Dot(new_normal, contact_normal) < 0.0f) {
            if (debug) {
                debug->num_concave_edge_hits = num_concave_edge_hits;
                debug->point_on_b_out = io_point_on_b_bt;
                debug->normal_out = io_normal;
            }
            return false;
        }
        io_normal = new_normal;
        io_point_on_b_bt = point_on_a_bt - io_normal * distance_bt;
    }
    if (debug) {
        debug->num_concave_edge_hits = num_concave_edge_hits;
        debug->point_on_b_out = io_point_on_b_bt;
        debug->normal_out = io_normal;
    }
    return true;
}

ROCKETGPU_HD bool MaybeAdjustArenaTriangleInternalEdgeContact(
    const ArenaMeshTriangle& world_tri,
    const Vec3& point_on_a_bt,
    float distance_bt,
    Vec3& io_point_on_b_bt,
    Vec3& io_normal,
    ArenaTriangleInternalEdgeDebug* debug,
    bool soccar_platform_math,
    const uint32_t* native_table, int native_shift,
    const ArenaMeshTriangle* cal_game_tri) {
    if (cal_game_tri != nullptr) {
        const float cal_origin_y = cal_game_tri->cal_origin_sign != 0
            ? static_cast<float>(cal_game_tri->cal_origin_sign) * kGoalMeshOriginYBt : 0.0f;
        return MaybeAdjustArenaTriangleInternalEdgeContactT<true>(
            *cal_game_tri, cal_origin_y, point_on_a_bt, distance_bt, io_point_on_b_bt, io_normal, debug,
            soccar_platform_math, native_table, native_shift);
    }
    return MaybeAdjustArenaTriangleInternalEdgeContactT<false>(
        world_tri, 0.0f, point_on_a_bt, distance_bt, io_point_on_b_bt, io_normal, debug,
        soccar_platform_math, native_table, native_shift);
}

ROCKETGPU_HD constexpr bool EnableArenaTriangleInternalEdgeAdjust() {
    return true;
}
