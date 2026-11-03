template <typename WheelT>
ROCKETGPU_HD float WheelRealRayLengthBt(const WheelT& wheel) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    const float rest_bt = wheel.suspension_rest_length * kUUToBT;
    const float travel_bt = (wheel.max_suspension_travel - WheelSuspensionSubtraction()) * kUUToBT;
    const float radius_bt = wheel.wheel_radius * kUUToBT;
    return (rest_bt + radius_bt) + travel_bt;
}
struct ArenaRayMeshOneLane {
    static constexpr bool kSplitHeaders = false;
    int first = 0;
    int step = 1;
    ROCKETGPU_HD bool Leader() const { return true; }
    ROCKETGPU_HD ArenaRayHit Combine(const ArenaRayHit& hit, int& order, float) const { return hit; }
};

template <typename MeshLanes = ArenaRayMeshOneLane>
ROCKETGPU_HD ArenaRayHit CastArenaRayDeltaBt(
    ArenaRayHit best,
    const Vec3& ray_source_bt,
    const Vec3& ray_delta_bt,
    float max_len,
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
    const uint32_t* native_table = nullptr, int native_shift = 0,
    const MeshLanes& mesh_lanes = MeshLanes{}, const ArenaMeshLeafGrid* leaf_grid = nullptr,
    const ArenaMeshTriangle* local_triangles = nullptr) {
    bool mesh_ray_cast_bt = false;
    bool neg_x_wins_mesh_tie = false;

    const Vec3 to_bt = ray_source_bt + ray_delta_bt;
    const Vec3 ray_min_bt(Min(ray_source_bt.x, to_bt.x), Min(ray_source_bt.y, to_bt.y), Min(ray_source_bt.z, to_bt.z));
    const Vec3 ray_max_bt(Max(ray_source_bt.x, to_bt.x), Max(ray_source_bt.y, to_bt.y), Max(ray_source_bt.z, to_bt.z));
    if (mesh_bvh_nodes != nullptr && mesh_bvh_node_count > 0) {
        mesh_ray_cast_bt = true;
        const bool skip_meshes =
            mesh_triangles != nullptr && mesh_triangle_count > 0 && max_len > 0.0f &&
            mesh_bvh_quantization != nullptr && mesh_bvh_quantization_count > 0 &&
            mesh_bvh_meshes != nullptr && mesh_bvh_mesh_count > 0 &&
            ray_max_bt.x < mesh_bvh_meshes[0].free_half_x && ray_min_bt.x > -mesh_bvh_meshes[0].free_half_x &&
            ray_max_bt.y < mesh_bvh_meshes[0].free_half_y && ray_min_bt.y > -mesh_bvh_meshes[0].free_half_y;
        int best_mesh = -1;
        if (!skip_meshes) {
            RayCastArenaMeshBvhBtSource<MeshLanes::kSplitHeaders>(
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
                mesh_lanes.first,
                mesh_lanes.step,
                &best_mesh,
                leaf_grid,
                local_triangles);
            best = mesh_lanes.Combine(best, best_mesh, best.ray_fraction);
        }
        if (best.hit && best_mesh >= 0 && mesh_bvh_meshes != nullptr && mesh_bvh_mesh_count > 0) {
            const int pos = MeshLanes::kSplitHeaders ? (best_mesh >> 16) : best_mesh;
            neg_x_wins_mesh_tie = pos >= mesh_bvh_meshes[0].ray_neg_x_plane_pos;
        }
    }
    if (ray_min_bt.x > -81.9f && ray_max_bt.x < 81.9f && ray_max_bt.z < 40.8f &&
        ray_min_bt.y > -200.0f && ray_max_bt.y < 200.0f && ray_min_bt.z > -20.0f) {
        if (ray_min_bt.z <= 0.0f) {
            CastStaticPlaneLikeRL(ray_source_bt, to_bt, Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f), max_len, best);
        }
    } else {
        CastArenaRayPlanesImpl(ray_source_bt, ray_delta_bt, max_len, best, native_table, native_shift, neg_x_wins_mesh_tie);
    }

    if (!mesh_ray_cast_bt) {
        CastArenaRayCornerCylindersImpl(ray_source_bt, ray_delta_bt, max_len, best);
    }
    return best;
}

template <typename MeshLanes = ArenaRayMeshOneLane>
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
    float cal_min_dist_sq_bt = 0.0f,
    const MeshLanes& mesh_lanes = MeshLanes{},
    const ArenaMeshTriangle* local_triangles = nullptr) {
    (void)source_uu;
    ArenaRayHit best;
    constexpr float kUUToBT = 1.0f / 50.0f;

    const float max_len_bt = max_len_bt_override > 0.0f ? max_len_bt_override : max_len * kUUToBT;
    const Vec3 ray_delta_bt = ray_dir * max_len_bt;
    if (cal_min_dist_sq_bt > 0.0f) {
        best.cal_min_fraction_sq = cal_min_dist_sq_bt / LengthSq((ray_source_bt + ray_delta_bt) - ray_source_bt);
    }
    return CastArenaRayDeltaBt(best, ray_source_bt, ray_delta_bt, max_len, mesh_triangles, mesh_triangle_count,
                               mesh_bvh_quantization, mesh_bvh_quantization_count, mesh_bvh_meshes, mesh_bvh_mesh_count,
                               mesh_bvh_nodes, mesh_bvh_node_count, mesh_bvh_subtrees, mesh_bvh_subtree_count,
                               native_table, native_shift, mesh_lanes, nullptr, local_triangles);
}

ROCKETGPU_HD bool CastWheelRayAgainstCarHitboxGjkCast(
    const Vec3& ray_source_bt,
    const Vec3& ray_dir,
    float max_len,
    const CarState& other_car,
    const CarConfig& other_cfg,
    ArenaRayHit& out_hit,
    float max_len_bt_override);

ROCKETGPU_HD bool CastGjkRayVsSphere(
    const Vec3& ray_from_bt,
    const Vec3& ray_to_bt,
    const Vec3& ball_center_bt,
    float ball_radius_bt,
    float& out_fraction,
    Vec3& out_normal) {
    constexpr float kRadius = 0.001f;
    constexpr int kMaxIterations = 64;
    auto closest = [&](const Vec3& p, const Vec3& c, float& dist, Vec3& n) {
        const Vec3 offset = (p + c) * 0.5f;
        const Vec3 w = (p - offset) - (c - offset);
        const float len2 = Dot(w, w);
        if (!(len2 > 1.1920929e-7f * 1.1920929e-7f)) {
            return false;
        }
        const float rlen = 1.0f / SqrtScalar(len2);
        n = w * rlen;
        dist = (1.0f / rlen) - ball_radius_bt;
        return true;
    };
    const Vec3 r = ray_to_bt - ray_from_bt;
    float lambda = 0.0f;
    float last_lambda = 0.0f;
    float dist = 0.0f;
    Vec3 n(0.0f, 0.0f, 0.0f);
    if (!closest(ray_from_bt, ball_center_bt, dist, n)) {
        return false;
    }
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
        const Vec3 p = ray_from_bt * (1.0f - lambda) + ray_to_bt * lambda;
        const Vec3 c = ball_center_bt * lambda + ball_center_bt * (1.0f - lambda);
        float next_dist = 0.0f;
        if (!closest(p, c, next_dist, n)) {
            return false;
        }
        if (next_dist < 0.0f) {
            out_fraction = last_lambda;
            out_normal = n;
            return true;
        }
        dist = next_dist;
    }
    if (Dot(n, r) >= 0.0f) {
        return false;
    }
    out_fraction = lambda;
    out_normal = n;
    return true;
}

ROCKETGPU_HD bool CastWheelRayAgainstBall(
    const Vec3& ray_source_bt,
    const Vec3& ray_dir,
    float max_len,
    const BallState& ball,
    float ball_radius,
    ArenaRayHit& out_hit,
    float max_len_bt_override = -1.0f) {
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

    float fraction = 0.0f;
    Vec3 normal(0.0f, 0.0f, 0.0f);
    Vec3 hit_point_bt(0.0f, 0.0f, 0.0f);
    const bool hit = CastGjkRayVsSphere(ray_from_bt, ray_to_bt, ball_center_bt, ball_radius_bt, fraction, normal);

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

template <typename MeshLanes = ArenaRayMeshOneLane>
ROCKETGPU_HD ArenaRayHit CastWheelRayAgainstCars(
    const Vec3& ray_source_bt,
    const Vec3& ray_dir,
    float max_len,
    const CarState* cars_in_env,
    const CarConfig* car_cfgs_in_env,
    int32_t cars_in_env_count,
    int32_t self_car_idx,
    int32_t* out_best_other_idx = nullptr,
    float max_len_bt_override = -1.0f,
    bool* out_best_no_response = nullptr,

    int32_t no_response_mask = -1,
    float cal_min_hit_dist_sq_bt = 0.0f,
    const MeshLanes& lanes = MeshLanes{},
    uint32_t excluded_bodies = 0u) {
    ArenaRayHit best;
    if (cars_in_env == nullptr || car_cfgs_in_env == nullptr || cars_in_env_count <= 0) {
        return best;
    }
    if (cars_in_env_count < 31 && (lanes.step == 1 || (self_car_idx >= 0 && self_car_idx < 32))) {
        const uint32_t cand_mask = ((1u << cars_in_env_count) - 1u) & ~(1u << self_car_idx) & ~(excluded_bodies >> 1);
        if (cand_mask == 0u) {
            if (out_best_other_idx != nullptr) {
                *out_best_other_idx = -1;
            }
            if (out_best_no_response != nullptr) {
                *out_best_no_response = false;
            }
            return best;
        }
    }

    int best_other_idx = -1;
    bool best_no_response = false;

    for (int32_t i = lanes.first; i < cars_in_env_count; i += lanes.step) {
        if (i == self_car_idx || ((excluded_bodies >> (i + 1)) & 1u)) {
            continue;
        }
        const CarState& other = cars_in_env[i];
        if (other.is_demoed) {
            continue;
        }
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
            best_no_response = no_response_mask >= 0 ? ((no_response_mask >> i) & 1) != 0
                                                     : other.demo_no_contact_response != 0;
        }
    }
    if (lanes.step > 1) {
        best = lanes.Combine(best, best_other_idx, best.t);
        if (best.hit) {
            best_no_response = no_response_mask >= 0
                                   ? ((no_response_mask >> best_other_idx) & 1) != 0
                                   : cars_in_env[best_other_idx].demo_no_contact_response != 0;
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

ROCKETGPU_HD void InitWheelStepState(WheelStepState& wheel, const CarState& car, const CarConfig& cfg, int wheel_idx,
                                   const CarFrame* frame = nullptr) {
    const bool front = wheel_idx < 2;
    const bool left = (wheel_idx % 2) != 0;
    Vec3 offset = front ? cfg.front_wheels_offset : cfg.back_wheels_offset;
    if (left) {
        offset.y = -offset.y;
    }
    const float published_rest = (front ? cfg.front_suspension_rest : cfg.back_suspension_rest) - 12.0f;
    offset.z = cfg.hitbox_offset.z;
    const float bone_z = floorf((offset.z - published_rest) * 100.0f + 0.5f) / 100.0f;
    Vec3 car_forward(1.0f, 0.0f, 0.0f);
    Vec3 car_right(0.0f, 1.0f, 0.0f);
    Vec3 car_up(0.0f, 0.0f, 1.0f);

    if (car.basis_authoritative) {
        CarBasisColumns(frame, car, car_forward, car_right, car_up);
    } else {
        GetOfficialUploadBasisVectors(car.rot, car_forward, car_right, car_up);
    }
    if (car.set_wheel_normals_mask != 0) {
        wheel.hard_point_ws = car.pos + RotateViaBasis(car_forward, car_right, car_up, offset);
    }

    {
        constexpr float kUUToBT = 1.0f / 50.0f;
        const Vec3 offset_bt = offset * kUUToBT;
        wheel.ray_source_bt = car.pos_bt + RotateViaBasis(car_forward, car_right, car_up, offset_bt);
    }
    wheel.wheel_dir_ws = -car_up;
    wheel.wheel_axle_ws = -car_right;
    wheel.wheel_radius = front ? cfg.front_wheel_radius : cfg.back_wheel_radius;
    wheel.suspension_rest_length = offset.z - bone_z;

    wheel.max_suspension_travel = 14.5f;
    wheel.is_front_wheel = front ? 1 : 0;
    wheel.suspension_force_scale = front ? (36.0f - 0.25f) : (54.0f + 0.25f + 0.015f);
    wheel.contact_normal_ws = -wheel.wheel_dir_ws;
}

ROCKETGPU_HD WheelStepState MakeWheelStepState(const CarState& car, const CarConfig& cfg, int wheel_idx,
                                             const CarFrame* frame = nullptr) {
    WheelStepState wheel;
    InitWheelStepState(wheel, car, cfg, wheel_idx, frame);
    return wheel;
}

ROCKETGPU_HD void SetWheelRayMiss(WheelStepState& wheel) {
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

ROCKETGPU_HD uint32_t WheelRayExcludedBodies(const CarState& car, const CarConfig& cfg, const BulletSweepHistory* sweep,
                                             int self_car_idx, int car_count) {
    const int bodies = Min(car_count + 1, kBulletSweepMaxBodies);
    if (sweep == nullptr || bodies <= 0) {
        return 0u;
    }
    const uint32_t mask =
        sweep->sap_body_aabb_valid & ((1u << bodies) - 1u) & ~(1u << (self_car_idx + 1));
    if (mask == 0u) {
        return 0u;
    }
    Vec3 car_forward(1.0f, 0.0f, 0.0f);
    Vec3 car_right(0.0f, 1.0f, 0.0f);
    Vec3 car_up(0.0f, 0.0f, 1.0f);
    if (car.basis_authoritative) {
        CarBasisColumns(car, car_forward, car_right, car_up);
    } else {
        GetOfficialUploadBasisVectors(car.rot, car_forward, car_right, car_up);
    }
    const Vec3 wheel_dir_ws = -car_up;
    constexpr float kUUToBT = 1.0f / 50.0f;
    const float offset_z = cfg.hitbox_offset.z;
    const float front_bone_z = floorf((offset_z - (cfg.front_suspension_rest - 12.0f)) * 100.0f + 0.5f) / 100.0f;
    const float back_bone_z = floorf((offset_z - (cfg.back_suspension_rest - 12.0f)) * 100.0f + 0.5f) / 100.0f;
    const float front_len_bt =
        ((offset_z - front_bone_z) * kUUToBT + cfg.front_wheel_radius * kUUToBT) +
        ((14.5f - WheelSuspensionSubtraction()) * kUUToBT);
    const float back_len_bt =
        ((offset_z - back_bone_z) * kUUToBT + cfg.back_wheel_radius * kUUToBT) +
        ((14.5f - WheelSuspensionSubtraction()) * kUUToBT);
    Vec3 mn(0.0f, 0.0f, 0.0f), mx(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < 4; ++i) {
        Vec3 offset = (i < 2) ? cfg.front_wheels_offset : cfg.back_wheels_offset;
        if ((i % 2) != 0) {
            offset.y = -offset.y;
        }
        offset.z = offset_z;
        const Vec3 from = car.pos_bt + RotateViaBasis(car_forward, car_right, car_up, offset * kUUToBT);
        const Vec3 to = from + wheel_dir_ws * (i < 2 ? front_len_bt : back_len_bt);
        const Vec3 lo(Min(from.x, to.x), Min(from.y, to.y), Min(from.z, to.z));
        const Vec3 hi(Max(from.x, to.x), Max(from.y, to.y), Max(from.z, to.z));
        mn = i == 0 ? lo : Vec3(Min(mn.x, lo.x), Min(mn.y, lo.y), Min(mn.z, lo.z));
        mx = i == 0 ? hi : Vec3(Max(mx.x, hi.x), Max(mx.y, hi.y), Max(mx.z, hi.z));
    }
    uint32_t excluded = 0u;
    for (int body = 0; body < bodies; ++body) {
        if (!((mask >> body) & 1u)) {
            continue;
        }
        const float* a = sweep->sap_body_aabb[body];
        if (a[3] < mn.x || mx.x < a[0] || a[4] < mn.y || mx.y < a[1] || a[5] < mn.z || mx.z < a[2]) {
            excluded |= 1u << body;
        }
    }
    return excluded;
}

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

template <typename MeshLanes = ArenaRayMeshOneLane>
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
    int32_t no_response_mask = -1,
    float cal_min_hit_dist_sq_bt = 0.0f,
    const MeshLanes& mesh_lanes = MeshLanes{},
    const ArenaMeshTriangle* local_triangles = nullptr,
    uint32_t excluded_bodies = 0u) {
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
        real_ray_length_bt, native_table, native_shift, cal_min_hit_dist_sq_bt, mesh_lanes, local_triangles);

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
        &car_hit_other_idx,
        real_ray_length_bt,
        &car_hit_no_response,
        no_response_mask, cal_min_hit_dist_sq_bt, mesh_lanes, excluded_bodies);
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

    if (ball != nullptr && ball_radius > 0.0f && !(excluded_bodies & 1u)) {
        ArenaRayHit ball_hit;
        if (CastWheelRayAgainstBall(wheel.ray_source_bt, wheel.wheel_dir_ws, real_ray_length, *ball, ball_radius, ball_hit, real_ray_length_bt)) {
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

    const Vec3 car_up = -wheel.wheel_dir_ws;

    if (hit.hit) {
        wheel.contact_point_bt = hit.point_bt;

        wheel.contact_normal_ws = hit.normal;
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
        wheel.wheel_trace_len_bt = trace_len_bt;
        wheel.suspension_len_bt = trace_len_bt - radius_bt;

        ComputeWheelRayVelocityTerms(car, car_up, hit.point_bt, wheel);
    } else {
        SetWheelRayMiss(wheel);
    }
}

ROCKETGPU_HD void ComputeWheelRayVelocityTerms(const CarState& car, const Vec3& car_up, const Vec3& hit_point_bt,
                                           WheelStepState& wheel) {
        const float denominator = Dot(wheel.contact_normal_ws, car_up);

        const Vec3 relpos_bt = hit_point_bt - car.pos_bt;
        const Vec3 vel_at_contact_bt = car.vel_bt + Cross(car.ang_vel, relpos_bt);
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
    stage.contact_point_bt = wheel.contact_point_bt;
    stage.contact_normal_ws = wheel.contact_normal_ws;
    stage.wheel_trace_len_bt = wheel.wheel_trace_len_bt;
    stage.suspension_len_bt = wheel.suspension_len_bt;
    stage.suspension_relative_velocity_bt = wheel.suspension_relative_velocity_bt;
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
    wheel.contact_point_bt = stage.contact_point_bt;
    wheel.contact_normal_ws = stage.contact_normal_ws;
    wheel.wheel_trace_len_bt = stage.wheel_trace_len_bt;
    wheel.suspension_len_bt = stage.suspension_len_bt;
    wheel.suspension_relative_velocity_bt = stage.suspension_relative_velocity_bt;
    wheel.suspension_relative_velocity = stage.suspension_relative_velocity;
    wheel.clipped_inv_contact_dot_suspension = stage.clipped_inv_contact_dot_suspension;
    wheel.extra_pushback = stage.extra_pushback;
    wheel.ground_car_idx = stage.ground_car_idx;
    wheel.ground_static_id = stage.ground_static_id;
    wheel.ground_kind = stage.ground_kind;
    wheel.is_in_contact = stage.is_in_contact;
    wheel.is_in_contact_with_world = stage.is_in_contact_with_world;
}

ROCKETGPU_HD Vec3 UpwardsDirFromWheelNormalSum(const Vec3& sum) {
    const float len_sq = LengthSq(sum);
    if (len_sq >= 1e-8f) {
        return sum * (1.0f / SqrtScalar(len_sq));
    }
    return Vec3(0.0f, 0.0f, 0.0f);
}

ROCKETGPU_HD Vec3 GetUpwardsDirFromWheelContacts(const WheelStepState* wheels) {
    Vec3 sum(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < 4; ++i) {
        if (wheels[i].is_in_contact) {
            sum += wheels[i].contact_normal_ws;
        }
    }
    return UpwardsDirFromWheelNormalSum(sum);
}

ROCKETGPU_HD float QuantTruncOffsetV3(float x, float scale);

struct CarRbSnapshot {
    Quat rot;
    Vec3 vel_uu;
};

ROCKETGPU_HD CarRbSnapshot TickStartCarRbSnapshot(const CarState& car) {
    return CarRbSnapshot{car.rot, car.vel};
}

ROCKETGPU_HD CarRbSnapshot FreshCarRbSnapshot(const CarState& car, const CarFrame* frame, float max_speed) {
    Vec3 car_forward(1.0f, 0.0f, 0.0f);
    Vec3 car_right(0.0f, 1.0f, 0.0f);
    Vec3 car_up(0.0f, 0.0f, 1.0f);
    CarBasisColumns(frame, car, car_forward, car_right, car_up);
    Vec3 v = car.vel_bt * 50.0f;
    const float len2 = (v.y * v.y + v.x * v.x) + v.z * v.z;
    if (max_speed > 0.0f && max_speed * max_speed < len2) {
        const float inv = 1.0f / sqrtf(len2);
        v = Vec3(max_speed * (inv * v.x), max_speed * (inv * v.y), max_speed * (inv * v.z));
    }
    return CarRbSnapshot{GameRbStateQuantizeQuat(GameGetRotationScalar(car_forward, car_right, car_up)),
                         Vec3(QuantTruncOffsetV3(v.x, 100.0f), QuantTruncOffsetV3(v.y, 100.0f),
                              QuantTruncOffsetV3(v.z, 100.0f))};
}

ROCKETGPU_HD float SnapshotForwardSpeed(const CarRbSnapshot& snapshot) {
    const Quat& q = snapshot.rot;
    const float w2 = q.w + q.w;
    const float s = q.w * q.w - ((q.y * q.y + q.x * q.x) + q.z * q.z);
    float d = (1.0f * q.x + 0.0f * q.y) + 0.0f * q.z;
    d = d + d;
    const Vec3 fwd(((0.0f * q.y - 0.0f * q.z) * w2 + s * 1.0f) + d * q.x,
                   ((1.0f * q.z - 0.0f * q.x) * w2 + s * 0.0f) + d * q.y,
                   ((0.0f * q.x - 1.0f * q.y) * w2 + s * 0.0f) + d * q.z);
    const Vec3& v = snapshot.vel_uu;
    return (fwd.x * v.x + fwd.y * v.y) + fwd.z * v.z;
}

ROCKETGPU_HD float WheelForwardSpeed(const CarState& car) {
    return SnapshotForwardSpeed(CarRbSnapshot{car.rot, car.vel});
}

ROCKETGPU_HD void ApplyAutoRollReference(
    CarState& car,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    int num_wheels_in_contact,
    const WheelStepState* wheels,
    float dt,
    WheelRuntimeState* wheel_runtime_state = nullptr,
    const Vec3* wheel_normal_sum_override = nullptr,
    const CarFrame* frame = nullptr) {
    const Vec3 ground_up_dir = num_wheels_in_contact > 0
        ? (wheel_normal_sum_override ? UpwardsDirFromWheelNormalSum(*wheel_normal_sum_override)
                                     : GetUpwardsDirFromWheelContacts(wheels))
        : car.world_contact.contact_normal;
    const Vec3 ground_down_dir = -ground_up_dir;

    Vec3 forward_dir(1.0f, 0.0f, 0.0f);
    Vec3 right_dir(0.0f, 1.0f, 0.0f);
    Vec3 up_dir(0.0f, 0.0f, 1.0f);
    CarBasisColumns(frame, car, forward_dir, right_dir, up_dir);
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
        QueueAngularAccelWorld(wheel_runtime_state, car, cfg, mutator.car_mass, autoroll_torque, frame);

        Vec3 F, R, U;
        GameUeQuatAxes(car.rot, F, R, U);
        const Vec3& G = ground_up_dir;
        const float crx = F.z * G.y - F.y * G.z;
        const float cry = F.x * G.z - F.z * G.x;
        const float crz = F.y * G.x - F.x * G.y;
        const auto clamp01 = [](float v) { return 0.0f <= v ? (1.0f <= v ? 1.0f : v) : 0.0f; };
        const float r_dot = clamp01((crx * R.x + R.y * cry) + R.z * crz);
        const float nx = -G.x, ny = -G.y, nz = -G.z;
        const float f_dot = clamp01(((ny * crz - nz * cry) * F.x + (crx * nz - nx * crz) * F.y) + (nx * cry - crx * ny) * F.z);
        const Vec3 f_axis = 0.0f <= (R.y * G.y + R.x * G.x) + R.z * G.z ? Vec3(-F.x, -F.y, -F.z) : F;
        const Vec3 r_axis = (F.y * G.y + F.x * G.x) + F.z * G.z < 0.0f ? Vec3(-R.x, -R.y, -R.z) : R;
        const float rf = (1.0f - r_dot) * 80.0f;
        const float ff = (1.0f - f_dot) * 80.0f;
        const Vec3 t_game(ff * r_axis.x + rf * f_axis.x, ff * r_axis.y + rf * f_axis.y, ff * r_axis.z + rf * f_axis.z);
        Vec3 ar_i0, ar_i1, ar_i2;
        BuildInertiaTensorWorldBtInverse(frame, car, cfg, mutator.car_mass, ar_i0, ar_i1, ar_i2);
        QueueFirstForceAndTorqueBt(wheel_runtime_state, autoroll_force_bt, Mul3x3Vec(ar_i0, ar_i1, ar_i2, t_game),
                                   frame);
    } else {
        const Vec3 autoroll_force = autoroll_force_bt * UnitsBTToUU();
        ApplyCentralForce(car, mutator.car_mass, autoroll_force, dt);
        ApplyAngularAccelWorld(car, autoroll_torque, dt);
    }
}

struct WheelImpulseOut {
    Vec3 dv_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 dw = Vec3(0.0f, 0.0f, 0.0f);
    int32_t applied = 0;
};

ROCKETGPU_HD WheelImpulseOut WheelImpulseAtPointBt(
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    const Vec3& impulse_world_bt,
    const Vec3& rel_point_world_bt,
    const CarFrame* frame = nullptr) {
    WheelImpulseOut out;
    const float official_inv_mass = 1.0f / mass;
    out.dv_bt = impulse_world_bt * official_inv_mass;
    const Vec3 torque_bt = Cross(rel_point_world_bt, impulse_world_bt);
    out.dw = frame ? ApplyInverseBodyInertiaWorldBtRowDot(*frame, torque_bt)
                   : ApplyInverseBodyInertiaWorldBtRowDot(car, cfg, mass, torque_bt);
    out.applied = 1;
    return out;
}

ROCKETGPU_HD void CombineWheelImpulses(CarState& car, const WheelImpulseOut out[4]) {
    Vec3 vel_bt = car.vel_bt;
    Vec3 ang_vel = car.ang_vel;
    bool any = false;
    for (int k = 0; k < 4; ++k) {
        const int i = k ^ 1;
        if (!out[i].applied) {
            continue;
        }
        vel_bt = vel_bt + out[i].dv_bt;
        ang_vel += out[i].dw;
        any = true;
    }
    if (any) {
        SetCarLinearVelocityBt(car, vel_bt);
        car.vel_touched_solve_stage = 1;
        car.ang_vel = ang_vel;
    }
}

ROCKETGPU_HD WheelImpulseOut WheelSuspensionPhase(
    const CarState& car,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    WheelStepState& wheel,
    float dt,
    const CarFrame* frame = nullptr) {
    WheelImpulseOut out;
    if (!wheel.is_in_contact) {
        return out;
    }

    const float force_bt =
        (wheel.suspension_rest_length * UnitsUUToBT() - wheel.suspension_len_bt) *
        kWheelSuspensionStiffness *
        wheel.clipped_inv_contact_dot_suspension;
    const float damping = wheel.suspension_relative_velocity < 0.0f ? kWheelDampingCompression : kWheelDampingRelaxation;
    float suspension_force_bt =
        (force_bt - damping * wheel.suspension_relative_velocity_bt) *
        wheel.suspension_force_scale;
    if (suspension_force_bt < 0.0f) {
        suspension_force_bt = 0.0f;
    }
    if (suspension_force_bt * UnitsBTToUU() > 0.0f) {
        const Vec3 rel_bt = wheel.contact_point_bt - car.pos_bt;

        const float base_force_scale_bt = (suspension_force_bt * dt) + wheel.extra_pushback;
        const Vec3 impulse_bt = wheel.contact_normal_ws * base_force_scale_bt;

        out = WheelImpulseAtPointBt(car, cfg, mutator.car_mass, impulse_bt, rel_bt, frame);
    }
    return out;
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

ROCKETGPU_HD BilateralBodyBt MakeBilateralCarBodyBt(const CarState& car, const CarConfig& cfg, float mass,
                                                    const CarFrame* frame = nullptr) {
    BilateralBodyBt body;

    body.com_bt = car.pos_bt;
    body.lin_vel_bt = car.vel_bt;
    body.ang_vel = car.ang_vel;
    Vec3 forward(1.0f, 0.0f, 0.0f);
    Vec3 right(0.0f, 1.0f, 0.0f);
    Vec3 up(0.0f, 0.0f, 1.0f);
    CarBasisColumns(frame, car, forward, right, up);

    body.world2_row0 = forward;
    body.world2_row1 = right;
    body.world2_row2 = up;
    body.inv_mass = mass != 0.0f ? 1.0f / mass : 0.0f;
    if (frame) {
        body.inv_inertia_local = frame->inv_inertia_local_bt;
        return body;
    }
    const Vec3 inertia_bt = BoxBodyInertiaDiagBt(cfg, mass);

    body.inv_inertia_local = Vec3(
        inertia_bt.x != 0.0f ? 1.0f / inertia_bt.x : 0.0f,
        inertia_bt.y != 0.0f ? 1.0f / inertia_bt.y : 0.0f,
        inertia_bt.z != 0.0f ? 1.0f / inertia_bt.z : 0.0f);
    return body;
}

ROCKETGPU_HD float ResolveSingleBilateralAgainstFixedBt(
    const BilateralBodyBt& body1,
    const Vec3& pos_bt,
    const Vec3& normal) {
    const Vec3 rel_pos1 = pos_bt - body1.com_bt;
    const Vec3 vel = BilateralVelocityInLocalPoint(body1, rel_pos1);

    const Vec3 a_j = BilateralWorld2Mul(body1, Cross(rel_pos1, normal));
    const Vec3 minv_jt0 = Hadamard(body1.inv_inertia_local, a_j);
    const float a_diag = body1.inv_mass + Dot(minv_jt0, a_j);

    const float out_rel_vel_bt = Dot(normal, vel);
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
    const BilateralBodyBt& body1,
    WheelStepState& wheel,
    float dt,
    const CarFrame* frame) {
    WheelImpulseOut out;
    if (!wheel.is_in_contact) {
        return out;
    }
    const float side_impulse = ResolveSingleBilateralAgainstFixedBt(
        body1, wheel.contact_point_bt, wheel.cal_axle_dir);
    const float roll = wheel.cal_rolling_friction * wheel.cal_long_friction;
    const float side = (side_impulse * (dt * wheel.cal_friction_scale)) * wheel.cal_lat_friction;
    wheel.impulse = Vec3(roll * wheel.cal_forward_dir.x + side * wheel.cal_axle_dir.x,
                         roll * wheel.cal_forward_dir.y + side * wheel.cal_axle_dir.y,
                         roll * wheel.cal_forward_dir.z + side * wheel.cal_axle_dir.z);
    if (LengthSq(wheel.impulse) == 0.0f) {
        return out;
    }

    const Vec3& car_up = body1.world2_row2;
    const Vec3 friction_rel_bt = wheel.contact_point_bt - car.pos_bt;
    Vec3 friction_wheel_rel_pos_bt = friction_rel_bt - car_up * Dot(car_up, friction_rel_bt);
    if (cfg.three_wheels) {
        friction_wheel_rel_pos_bt = friction_wheel_rel_pos_bt - car_up * (4.0f * 0.02f);
    }
    const Vec3 friction_impulse_bt = wheel.impulse;
    return WheelImpulseAtPointBt(car, cfg, mutator.car_mass, friction_impulse_bt, friction_wheel_rel_pos_bt, frame);
}

ROCKETGPU_HD void ApplyWheelSuspensionAndFriction(
    CarState& car,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    const CarControls& controls,
    float dt,
    WheelStepState* wheels,
    const CarFrame* frame = nullptr) {
    if (!(wheels[0].is_in_contact || wheels[1].is_in_contact || wheels[2].is_in_contact || wheels[3].is_in_contact)) {
        return;
    }
    WheelImpulseOut out[4];
    for (int i = 0; i < 4; ++i) {
        out[i] = WheelSuspensionPhase(car, cfg, mutator, wheels[i], dt, frame);
    }
    CombineWheelImpulses(car, out);
    const BilateralBodyBt body1 = MakeBilateralCarBodyBt(car, cfg, mutator.car_mass, frame);
    for (int i = 0; i < 4; ++i) {
        out[i] = WheelFrictionPhase(car, cfg, mutator, body1, wheels[i], dt, frame);
    }
    CombineWheelImpulses(car, out);
    (void)controls;
}

ROCKETGPU_HD Vec3 ComputeWheelSteeredRightCol(
    const WheelStepState& wheel,
    float steer_angle,
    bool native_sse_math = false) {
    const Vec3 up = -wheel.wheel_dir_ws;
    const Vec3 right = wheel.wheel_axle_ws;

    const float axis_len = Length(up);
    const float half_angle = steer_angle * 0.5f;

    const float sin_scale = RocketGpuSinf(half_angle) / axis_len;
    const Quat steering_rot(
        RocketGpuCosf(half_angle), up.x * sin_scale, up.y * sin_scale, up.z * sin_scale);
    Vec3 mat_col0(1.0f, 0.0f, 0.0f);
    Vec3 mat_col1(0.0f, 1.0f, 0.0f);
    Vec3 mat_col2(0.0f, 0.0f, 1.0f);

    if (native_sse_math) {
        GetBasisVectorsSetRotationForm(steering_rot, mat_col0, mat_col1, mat_col2);
    } else {
        const Quat& q = steering_rot;
        const float s = 2.0f / (((q.x * q.x + q.y * q.y) + q.z * q.z) + q.w * q.w);
        const float xs = q.x * s, ys = q.y * s, zs = q.z * s;
        const float wx = q.w * xs, wy = q.w * ys, wz = q.w * zs;
        const float xx = q.x * xs, xy = q.x * ys, xz = q.x * zs;
        const float yy = q.y * ys, yz = q.y * zs, zz = q.z * zs;
        mat_col0 = Vec3(1.0f - (yy + zz), xy + wz, xz - wy);
        mat_col1 = Vec3(xy - wz, 1.0f - (xx + zz), yz + wx);
        mat_col2 = Vec3(xz + wy, yz - wx, 1.0f - (xx + yy));
    }
    const Vec3 neg_right = -right;
    return Vec3(
        mat_col0.x * neg_right.x + mat_col1.x * neg_right.y + mat_col2.x * neg_right.z,
        mat_col0.y * neg_right.x + mat_col1.y * neg_right.y + mat_col2.y * neg_right.z,
        mat_col0.z * neg_right.x + mat_col1.z * neg_right.y + mat_col2.z * neg_right.z);
}

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

ROCKETGPU_HD bool SetWheelNormalHasWorldFace(const WheelStepState& wheel, const Vec3& normal,
                                             const ArenaMeshTriangle* mesh_triangles, int32_t mesh_triangle_count,
                                             const ArenaMeshBvhQuantization* mesh_bvh_quantization,
                                             int32_t mesh_bvh_quantization_count,
                                             const ArenaMeshBvhMeshInfo* mesh_bvh_meshes, int32_t mesh_bvh_mesh_count,
                                             const ArenaMeshBvhNode* mesh_bvh_nodes, int32_t mesh_bvh_node_count,
                                             const ArenaMeshBvhSubtreeInfo* mesh_bvh_subtrees,
                                             int32_t mesh_bvh_subtree_count,
                                             const ArenaMeshTriangle* local_triangles = nullptr) {
    const float len = wheel.suspension_rest_length + wheel.max_suspension_travel + wheel.wheel_radius + 20.0f;
    const ArenaRayHit hit = CastArenaRayWithBtSource(
        wheel.hard_point_ws, -normal, len, wheel.ray_source_bt, mesh_triangles, mesh_triangle_count,
        mesh_bvh_quantization, mesh_bvh_quantization_count, mesh_bvh_meshes, mesh_bvh_mesh_count, mesh_bvh_nodes,
        mesh_bvh_node_count, mesh_bvh_subtrees, mesh_bvh_subtree_count, -1.0f, nullptr, 0, 0.0f,
        ArenaRayMeshOneLane{}, local_triangles);
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
                                           int32_t mesh_bvh_subtree_count = 0,
                                           const ArenaMeshTriangle* local_triangles = nullptr) {
    normal_sum = Vec3(0.0f, 0.0f, 0.0f);
    int n = 0;
    for (int k = 0; k < 4; ++k) {
        const int i = k ^ 1;
        if (!((car.set_wheel_normals_mask >> i) & 1)) {
            continue;
        }
        if ((car.set_wheel_normals_mask >> 5) & 1) {
            normal_sum += car.set_wheel_normals[i];
            ++n;
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
                                        mesh_bvh_subtree_count, local_triangles)) {
            continue;
        }
        if (ball != nullptr) {
            const Vec3 to_wheel = wheels[i].hard_point_ws - ball->pos;
            const float dist = Length(to_wheel);
            if (dist < ball_radius + 80.0f && dist > 0.0f && Dot(car.set_wheel_normals[i], to_wheel) > 0.95f * dist) {
                continue;
            }
        }
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

struct WheelDriveShared {
    Vec3 front_right_col = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 back_right_col = Vec3(0.0f, 0.0f, 0.0f);
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
    float dt,
    const CarFrame* frame = nullptr) {
    (void)mutator;
    (void)frame;
    const float prev_lat_friction = wheel_runtime_state ? wheel_runtime_state->lat_friction[i] : 0.0f;
    const float prev_long_friction = wheel_runtime_state ? wheel_runtime_state->long_friction[i] : 0.0f;

    const float current_steer_angle = wheel.is_front_wheel ? shared.next_steer_angle : 0.0f;

    const float applied_engine_force = shared.drive_engine_force;
    const float applied_brake = shared.drive_brake_force;

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
    Vec3 basis_right_col(0.0f, 0.0f, 0.0f);
    if (wheel.is_in_contact) {
        basis_right_col = wheel.is_front_wheel ? shared.front_right_col : shared.back_right_col;

        const Vec3 wheel_delta_bt = wheel.ray_source_bt - car.pos_bt;
        const Vec3 cross_vec = (Cross(car.ang_vel, wheel_delta_bt) + car.vel_bt) * 50.0f;
        const Vec3 lat_dir = basis_right_col;
        const Vec3 long_dir = Cross(lat_dir, wheel.contact_normal_ws);
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

    if (wheel_runtime_state) {
        wheel_runtime_state->lat_friction[i] = current_lat_friction;
        wheel_runtime_state->long_friction[i] = current_long_friction;
    }
    const float applied_lat_friction = current_lat_friction;
    const float applied_long_friction = current_long_friction;

    if (!wheel.is_in_contact) {
        return;
    }

    const Vec3 raw_axle_dir = basis_right_col;
    const Vec3 c = Cross(wheel.contact_normal_ws, raw_axle_dir);
    const Vec3 forward_dir = c * (1.0f / SqrtScalar(LengthSq(c)));

    float rolling_friction = 0.0f;
    if (applied_engine_force == 0.0f) {
        if (applied_brake != 0.0f) {
            const Vec3 car_rel_contact_bt = wheel.contact_point_bt - car.pos_bt;
            const Vec3 contact_vel_bt = car.vel_bt + Cross(car.ang_vel, car_rel_contact_bt);
            float rel_vel = Dot(contact_vel_bt, forward_dir);
            if (dt > (1.0f / 80.0f)) {
                const float threshold = -(1.0f / (dt * 150.0f)) + 0.8f;
                if (Abs(rel_vel) < threshold) {
                    rel_vel = 0.0f;
                }
            }
            rolling_friction = Clamp(-rel_vel * 113.73963f, -applied_brake, applied_brake) * (dt * shared.friction_scale);
        }
    } else {
        rolling_friction = -applied_engine_force * dt;
    }

    wheel.cal_axle_dir = raw_axle_dir;
    wheel.cal_forward_dir = forward_dir;
    wheel.cal_rolling_friction = rolling_friction;
    wheel.cal_lat_friction = applied_lat_friction;
    wheel.cal_long_friction = applied_long_friction;
    wheel.cal_friction_scale = shared.friction_scale;
}

struct WheelContactOut {
    Vec3 normal = Vec3(0.0f, 0.0f, 0.0f);
    uint32_t flags = 0;
};
constexpr uint32_t kWheelContactInContact = 1u;
constexpr uint32_t kWheelContactWithWorld = 2u;
constexpr uint32_t kWheelContactOnBall = 4u;

struct WheelContactSummary {
    int num_wheels_in_contact = 0;
    bool wheels_have_world_contact = false;
    Vec3 wheel_normal_sum = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 contact_normal_sum_sim = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 world_normal_sum = Vec3(0.0f, 0.0f, 0.0f);
    uint8_t world_contacts = 0;
};

ROCKETGPU_HD void SetWorldWheelPushback(const CarState& car, const CarConfig& cfg, const MutatorConfig& mutator,
                                        WheelStepState& wheel, const CarFrame* frame = nullptr) {
    const float solver_dt = (1.0f / 60.0f);

    const float pb_rest_bt = wheel.suspension_rest_length * UnitsUUToBT();
    const float pb_radius_bt = wheel.wheel_radius * UnitsUUToBT();
    const float ray_pushback_thresh_bt = (pb_rest_bt + pb_radius_bt) - WheelSuspensionSubtraction() * UnitsUUToBT();
    const float wheel_trace_len_bt = wheel.wheel_trace_len_bt;
    if (wheel_trace_len_bt < ray_pushback_thresh_bt) {
        const float wheel_trace_dist_delta_bt = wheel_trace_len_bt - ray_pushback_thresh_bt;
        const float unscaled_pushback_bt = ResolveStaticCollisionImpulseMagnitudeBtInputs(
            car,
            cfg,
            mutator.car_mass,
            wheel.contact_point_bt,
            wheel.contact_normal_ws,
            wheel_trace_dist_delta_bt,
            solver_dt,
            frame);
        wheel.extra_pushback = unscaled_pushback_bt / 4.0f;
    }
}

template <bool kStagedIndependent = false>
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
    uint32_t* dyn_hit_pushed_mask,
    const CarFrame* frame = nullptr,
    uint32_t excluded_bodies = 0u) {
    const float extra_pushback = wheel_runtime_state ? wheel_runtime_state->extra_pushback[i] : 0.0f;
    WheelRayStage stage;
    stage.valid = 0;
    if (staged_rays != nullptr) {
        stage = staged_rays[i];
    }
#if defined(__CUDA_ARCH__)
    WheelStepState made = MakeWheelStepState(car, cfg, i, frame);
#else
    WheelStepState& made = *new (&wheel) WheelStepState;
    InitWheelStepState(made, car, cfg, i, frame);
#endif
    made.suspension_force_scale = made.is_front_wheel ? cal_front_scale : cal_back_scale;

    made.extra_pushback = extra_pushback;
    if (!kStagedIndependent && stage.valid == 2) {
        SetWheelRayMiss(made);
    } else if (kStagedIndependent || stage.valid) {
        LoadWheelRayStage(made, stage);
    }
#if defined(__CUDA_ARCH__)
    wheel = made;
#endif
    if (!kStagedIndependent && stage.valid == 2) {
    } else if (kStagedIndependent || stage.valid) {
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
        -1, cal_min_hit_dist_sq_bt, ArenaRayMeshOneLane{}, mutator.cal_arena_local_tris, excluded_bodies);
    if (wheel.is_in_contact) {
        wheel.extra_pushback = 0.0f;
    }
    if (wheel.is_in_contact_with_world) {
        SetWorldWheelPushback(car, cfg, mutator, wheel, frame);
    }
    if (!kStagedIndependent && wheel.is_in_contact && !wheel.is_in_contact_with_world) {
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
            wheel.extra_pushback = 0.0f;
            if (wheel.wheel_trace_len_bt < ray_pushback_thresh_bt) {
                wheel.extra_pushback = ResolveCarCarWheelPushbackMagnitudeBtInputs(
                    car, cfg, other, other_cfg, mutator.car_mass, cp_bt, n_bt,
                    wheel.wheel_trace_len_bt - ray_pushback_thresh_bt, solver_dt) / 4.0f;
            }
            if (wheel.extra_pushback > 0.0f) {
                const float full_pushback = wheel.extra_pushback * 4.0f;
                ApplyImpulseAtPointBtRoundtripBtInputs(
                    other, other_cfg, mutator.car_mass, -n_bt * full_pushback, cp_bt - other.pos_bt);
            }
            ApplyImpulseAtPointBtRoundtripBtInputs(other, other_cfg, mutator.car_mass, -n_bt, cp_bt - other.pos_bt);
            if (dyn_hit_pushed_mask != nullptr && wheel.ground_car_idx < 32) {
                *dyn_hit_pushed_mask |= (1u << wheel.ground_car_idx);
            }
        } else if (wheel.ground_kind == ROCKETGPU_ARENA_RAY_HIT_BALL && dyn_hit_ball != nullptr) {
            BallState& b = *dyn_hit_ball;
            b.sleeping |= 4u;
            const Vec3 push_bt = -n_bt;
            SetBallLinearVelocityBt(b, b.vel_bt + push_bt * (1.0f / mutator.ball_mass));
            b.ang_vel += ApplyInverseBallInertiaWorldBtMutator(b, mutator, Cross(cp_bt - b.pos_bt, push_bt));
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
    }
    for (int k = 0; k < 4; ++k) {
        const int i = k ^ 1;
        if ((out[i].flags & kWheelContactInContact) != 0 && (out[i].flags & kWheelContactWithWorld) != 0) {
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

template <bool kStagedIndependent = false>
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
    uint32_t* dyn_hit_pushed_mask = nullptr,
    const CarFrame* frame = nullptr,
    uint32_t excluded_bodies = 0u) {
    car.world_contact.wheel_ball_mask = 0;

    const uint8_t sticky_prev_world_contacts =
        wheel_runtime_state ? wheel_runtime_state->cal_prev_world_wheel_contacts : 0;
    const Vec3 sticky_prev_world_normal_sum =
        wheel_runtime_state ? wheel_runtime_state->cal_prev_wheel_normal_sum : Vec3(0.0f, 0.0f, 0.0f);
    const bool sticky_prev_contact = (car.wheels_with_contact[0] || car.wheels_with_contact[1] || car.wheels_with_contact[2] || car.wheels_with_contact[3]);

    const bool landing_prev_any_contact = car.tick_count_since_update != 0 &&
        (car.wheels_with_contact[0] || car.wheels_with_contact[1] || car.wheels_with_contact[2] || car.wheels_with_contact[3]);
    float cal_front_scale = 0.0f, cal_back_scale = 0.0f, cal_min_hit_dist_sq_bt = 0.0f;
    CalSuspensionForceScales(cfg, mutator.car_mass, cal_front_scale, cal_back_scale);
    if constexpr (!kStagedIndependent) {
        cal_min_hit_dist_sq_bt = CalMinWheelHitDistSqBt(cfg);
    }

    WheelContactOut contact_out[4];
    for (int k = 0; k < 4; ++k) {
        const int i = k ^ 1;
        contact_out[i] = WheelRayPhase<kStagedIndependent>(car, cfg, mutator, wheels[i], i, cal_front_scale, cal_back_scale,
                                       cal_min_hit_dist_sq_bt, wheel_runtime_state, mesh_triangles, mesh_triangle_count,
                                       cars_in_env, car_cfgs_in_env, cars_in_env_count, self_car_idx, env_idx, ball,
                                       ball_radius, mesh_bvh_quantization, mesh_bvh_quantization_count, mesh_bvh_meshes,
                                       mesh_bvh_mesh_count, mesh_bvh_nodes, mesh_bvh_node_count, mesh_bvh_subtrees,
                                       mesh_bvh_subtree_count, staged_rays, staged_rays_velocity_stale, dyn_hit_cars,
                                       dyn_hit_ball, dyn_hit_pushed_mask, frame, excluded_bodies);
    }
    const WheelContactSummary contact_summary = CombineWheelContacts(car, wheel_runtime_state, contact_out);
    const int num_wheels_in_contact = contact_summary.num_wheels_in_contact;
    const bool wheels_have_world_contact = contact_summary.wheels_have_world_contact;
    const Vec3 wheel_normal_sum = contact_summary.wheel_normal_sum;
    const float abs_snapshot_forward_speed = Abs(forward_speed_uu);
    forward_speed_uu = SnapshotForwardSpeed(FreshCarRbSnapshot(car, frame, mutator.car_max_speed));
    const float abs_forward_speed = Abs(forward_speed_uu);
    const float friction_scale = mutator.car_mass / 3.0f;

    if (controls.handbrake) {
        car.handbrake_val = Clamp(car.handbrake_val + 5.0f * dt, 0.0f, 1.0f);
    } else {
        car.handbrake_val = Clamp(car.handbrake_val - 2.0f * dt, 0.0f, 1.0f);
    }

    float real_throttle = controls.throttle;
    float real_brake = 0.0f;

    float drive_speed_scale = EvalDriveTorqueFactor(forward_speed_uu);
    float engine_throttle = car.driving_disabled ? 0.0f : real_throttle;
    if (!controls.handbrake && !car.driving_disabled) {
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

    float drive_engine_force_uu = (drive_speed_scale * (engine_throttle * 288000.0f)) * 0.25f;
    if (num_wheels_in_contact < 3) {
        drive_engine_force_uu *= 0.25f;
    }
    const float drive_engine_force = drive_engine_force_uu * UnitsUUToBT();
    const float drive_brake_force = real_brake * 10500.0f * 0.25f * UnitsUUToBT();
    float next_steer_angle;
    {
        float steer_deg = cfg.three_wheels ? EvalSteerAngleFromSpeedThreeWheel(abs_forward_speed)
                                           : EvalSteerAngleFromSpeed(cfg.steer_angle_curve_deg, abs_forward_speed);
        if (car.handbrake_val > 0.0f) {
            steer_deg += (EvalPowerslideSteerAngleFromSpeed(cfg.powerslide_steer_angle_curve_deg, abs_forward_speed) - steer_deg) * car.handbrake_val;
        }
        next_steer_angle = steer_deg * controls.steer * SteerDegToRad();
    }

    WheelDriveShared drive_shared;
    drive_shared.next_steer_angle = next_steer_angle;
    drive_shared.drive_engine_force = drive_engine_force;
    drive_shared.drive_brake_force = drive_brake_force;
    drive_shared.real_throttle = real_throttle;
    drive_shared.friction_scale = friction_scale;
    const bool front_in_contact = wheels[0].is_in_contact || wheels[1].is_in_contact;
    const bool back_in_contact = wheels[2].is_in_contact || wheels[3].is_in_contact;
    const bool native_sse = NativeSseMath(mutator);
    if (front_in_contact) {
        drive_shared.front_right_col = ComputeWheelSteeredRightCol(wheels[0], next_steer_angle, native_sse);
    }
    if (back_in_contact) {
#if defined(__CUDA_ARCH__)
        const bool positive_zero_steer = next_steer_angle == 0.0f && !signbit(next_steer_angle);
#else
        const bool positive_zero_steer = next_steer_angle == 0.0f && !std::signbit(next_steer_angle);
#endif
        if (front_in_contact && positive_zero_steer) {
            drive_shared.back_right_col = drive_shared.front_right_col;
        } else {
            drive_shared.back_right_col = ComputeWheelSteeredRightCol(wheels[2], 0.0f, native_sse);
        }
    }
    for (int i = 0; i < 4; ++i) {
        WheelDriveSetupPhase(car, cfg, mutator, drive_shared, wheels[i], i, wheel_runtime_state, dt, frame);
    }

    if (wheel_runtime_state) {
        const bool restore_tick = car.tick_count_since_update == 0;
        Vec3 world_normal_sum = sticky_prev_world_normal_sum;
        bool gate = sticky_prev_world_contacts > 0;
        if (restore_tick) {
            world_normal_sum = Vec3(0.0f, 0.0f, 0.0f);
            gate = false;
            if (!kStagedIndependent && car.set_wheel_normals_mask) {
                gate = SetStateWorldWheelNormals(car, cfg, wheels, ball, ball_radius, cars_in_env, car_cfgs_in_env,
                                                 cars_in_env_count, self_car_idx, world_normal_sum,
                                                 mesh_triangles, mesh_triangle_count, mesh_bvh_quantization,
                                                 mesh_bvh_quantization_count, mesh_bvh_meshes, mesh_bvh_mesh_count,
                                                 mesh_bvh_nodes, mesh_bvh_node_count, mesh_bvh_subtrees,
                                                 mesh_bvh_subtree_count, mutator.cal_arena_local_tris) > 0;
            } else {
                world_normal_sum = contact_summary.world_normal_sum;
                gate = contact_summary.world_contacts > 0;
            }
        }
        if (gate) {
            const Vec3 upwards_dir = UpwardsDirFromWheelNormalSum(world_normal_sum);
            const bool full_stick = (controls.throttle != 0.0f) || (abs_snapshot_forward_speed > 25.0f);
            const float sticky_base = cfg.three_wheels ? 0.0f : 0.5f;
            const float sticky_full = cfg.three_wheels ? 1.5f : sticky_base + 1.0f;
            const float sticky_target = full_stick ? sticky_full : sticky_base;
            const float sticky_force_scale = Abs(upwards_dir.z) * (sticky_base - sticky_target) + sticky_target;
            const float sticky_accel =
                (mutator.gravity_z * sticky_force_scale) * (mutator.gravity_z > 0.0f ? -1.0f : 1.0f);
            const Vec3 sticky_force_bt = ((sticky_accel * upwards_dir) * UnitsUUToBT()) * mutator.car_mass;
            QueueCentralForce(wheel_runtime_state, sticky_force_bt, frame);
        }
    } else {
    bool sticky_block = wheels_have_world_contact || sticky_prev_contact;
    bool apply_sticky = true;
    apply_sticky = landing_prev_any_contact || (car.tick_count_since_update == 0 && num_wheels_in_contact > 0);
    sticky_block = wheels_have_world_contact || apply_sticky;

    if (sticky_block) {
        Vec3 upwards_dir = wheels_have_world_contact ? Normalize(wheel_normal_sum, Vec3(0.0f, 0.0f, 1.0f)) : -wheels[0].wheel_dir_ws;
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
        if (cfg.three_wheels) {
            const float wall = full_stick ? 1.5f : 0.0f;
            sticky_force_scale = Abs(upwards_dir.z) * (0.0f - wall) + wall;
        }

        const Vec3 sticky_force_bt =
            upwards_dir * sticky_force_scale * (mutator.gravity_z * UnitsUUToBT()) * mutator.car_mass;
        if (!apply_sticky) {
        } else if (wheel_runtime_state) {
            QueueCentralForce(wheel_runtime_state, sticky_force_bt, frame);
        } else {
            const Vec3 sticky_force = sticky_force_bt * UnitsBTToUU();
            ApplyCentralForce(car, mutator.car_mass, sticky_force, dt);
        }
    }
    }

    return num_wheels_in_contact;
}

ROCKETGPU_HD float QuantizeAxisInput(float x) {
    const float c = Clamp(x, -1.0f, 1.0f);
    const float y = c < 0.0f ? Max(c * 128.0f, -128.0f) : Min(c * 127.0f, 127.0f);
    const float w = ((y + 128.0f) + (y + 128.0f)) + 0.5f;
    const int byte = (static_cast<int>(rintf(w)) >> 1) & 0xFF;
    const float s = static_cast<float>(byte) - 128.0f;
    constexpr float kInv127 = 1.0f / 127.0f;
    const float q = s * kInv127;
    const float s_over_127 = fmaf(fmaf(-127.0f, q, s), kInv127, q);
    return byte < 0x80 ? s * (1.0f / 128.0f) : s_over_127;
}

ROCKETGPU_HD CarControls QuantizeAxisControls(CarControls controls) {
    controls.throttle = QuantizeAxisInput(controls.throttle);
    controls.steer = QuantizeAxisInput(controls.steer);
    controls.pitch = QuantizeAxisInput(controls.pitch);
    controls.yaw = QuantizeAxisInput(controls.yaw);
    controls.roll = QuantizeAxisInput(controls.roll);
    return controls;
}

ROCKETGPU_HD float DodgeCancelPitchRaw(const CarState& car, const CarControls& controls,
                                      float raw_pitch = 2.0f) {
    {
        constexpr float kGameMinDodgeTorqueTime = 0.041f;
        const float pitch = raw_pitch <= 1.0f ? raw_pitch : controls.pitch;
        return car.internal_flip_time >= kGameMinDodgeTorqueTime ? pitch : 0.0f;
    }
    return (car.internal_flip_time > (0.0f)) ? car.last_controls.pitch : controls.pitch;
}

ROCKETGPU_HD float DodgeCancelPitch(const CarState& car, const CarControls& controls,
                                   float raw_pitch = 2.0f) {
    const float pitch = DodgeCancelPitchRaw(car, controls, raw_pitch);
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
    const Vec3* cal_damp_ang_vel = nullptr,
    const CarFrame* frame = nullptr) {
    Vec3 basis_forward(1.0f, 0.0f, 0.0f);
    Vec3 basis_right(0.0f, 1.0f, 0.0f);
    Vec3 basis_up(0.0f, 0.0f, 1.0f);
    CarBasisColumns(frame, car, basis_forward, basis_right, basis_up);
    const Vec3 dir_pitch_right(-basis_right.x, -basis_right.y, -basis_right.z);
    const Vec3 dir_yaw_up = basis_up;
    const Vec3 dir_roll_forward(-basis_forward.x, -basis_forward.y, -basis_forward.z);

    bool do_air_control = false;

    do_air_control = true;

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
            QueueAngularAccelWorld(wheel_runtime_state, car, cfg, mutator.car_mass, air_control_torque, frame);

            Vec3 ue_f, ue_r, ue_u;
            GameUeQuatAxes(car.rot, ue_f, ue_r, ue_u);
            constexpr float kRot = 2.0f * Pi() / 65536.0f;
            constexpr float kGate = 1.0e-4f;
            const float in_pitch = controls.pitch * pitch_torque_scale;
            const float a_p = (mutator.car_air_control_pitch * -in_pitch) * kRot;
            const float a_y = (mutator.car_air_control_yaw * controls.yaw) * kRot;
            const float a_r = (-mutator.car_air_control_roll * controls.roll) * kRot;
            const Vec3 t_in(
                (((a_p * ue_r.x + 0.0f) + a_y * ue_u.x) + a_r * ue_f.x) * 1000.0f,
                (((a_p * ue_r.y + 0.0f) + a_y * ue_u.y) + a_r * ue_f.y) * 1000.0f,
                (((a_p * ue_r.z + 0.0f) + a_y * ue_u.z) + a_r * ue_f.z) * 1000.0f);
            const float d_p = (1.0f - Abs(in_pitch)) * (Dot(damp_ang_vel, ue_r) * (mutator.car_air_damp_pitch * kRot));
            const float d_y = (1.0f - Abs(controls.yaw)) * (Dot(damp_ang_vel, ue_u) * (mutator.car_air_damp_yaw * kRot));
            const float d_r = (1.0f * Dot(damp_ang_vel, ue_f)) * (mutator.car_air_damp_roll * kRot);
            const Vec3 t_damp(
                (((0.0f - d_p * ue_r.x) - d_y * ue_u.x) - d_r * ue_f.x) * 1000.0f,
                (((0.0f - d_p * ue_r.y) - d_y * ue_u.y) - d_r * ue_f.y) * 1000.0f,
                (((0.0f - d_p * ue_r.z) - d_y * ue_u.z) - d_r * ue_f.z) * 1000.0f);
            const bool apply_in = Abs(t_in.x) >= kGate || Abs(t_in.y) >= kGate || Abs(t_in.z) >= kGate;
            const bool apply_damp = Abs(t_damp.x) >= kGate || Abs(t_damp.y) >= kGate || Abs(t_damp.z) >= kGate;
            if (apply_in || apply_damp) {
                Vec3 i0, i1, i2;
                BuildInertiaTensorWorldBtInverse(frame, car, cfg, mutator.car_mass, i0, i1, i2);
                if (apply_in) {
                    QueueOfficialTorqueBt(wheel_runtime_state, Mul3x3Vec(i0, i1, i2, t_in), frame);
                }
                if (apply_damp) {
                    QueueOfficialTorqueBt(wheel_runtime_state, Mul3x3Vec(i0, i1, i2, t_damp), frame);
                }
            }
        } else {
            ApplyAngularAccelWorld(car, air_control_torque, dt);
        }
    }

    const float air_throttle_input = (car.is_boosting) ? 1.0f : controls.throttle;
    if (air_throttle_input != 0.0f && air_throttle) {
        Vec3 throttle_fwd, throttle_rgt, throttle_up;
        GameUeQuatAxes(car.rot, throttle_fwd, throttle_rgt, throttle_up);
        (void)throttle_rgt; (void)throttle_up;

        const float air_throttle_force = mutator.car_air_throttle_accel * mutator.car_mass;
        const Vec3 air_throttle_force_bt = ((throttle_fwd * air_throttle_force) * air_throttle_input) * UnitsUUToBT();
        if (wheel_runtime_state) {
            QueueCentralForce(wheel_runtime_state, air_throttle_force_bt, frame);
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
    WheelRuntimeState* wheel_runtime_state = nullptr,
    const CarFrame* frame = nullptr) {
    if (!car.is_demoed) {
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
            const Vec3 boost_dir = GameUeQuatRotateVector(car.rot, Vec3(1.0f, 0.0f, 0.0f));
            const float boost_force = mutator.boost_accel_ground * mutator.car_mass;
            const Vec3 boost_force_bt = (boost_force * boost_dir) * UnitsUUToBT();
            if (wheel_runtime_state) {
                QueueCentralForce(wheel_runtime_state, boost_force_bt, frame);
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
        CarBasisColumns(frame, car, boost_fwd, boost_rgt, boost_up);
        (void)boost_rgt; (void)boost_up;
        const Vec3 forward = boost_fwd;

        const float accel = mutator.boost_accel_ground;

        const float boost_accel_bt = accel * UnitsUUToBT();
        const Vec3 boost_force_bt = boost_accel_bt * forward * mutator.car_mass;
        if (wheel_runtime_state) {
            QueueCentralForce(wheel_runtime_state, boost_force_bt, frame);
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

ROCKETGPU_HD float QuantRoundPosV3(float x, float scale);

ROCKETGPU_HD bool RobinTorque(const CarState& car, float pitch_freq, float pitch_amp,
                              float yaw_freq, float yaw_amp, float roll_freq, float roll_amp, Vec3& t) {
    const Vec3 p = car.pos_bt * 50.0f;
    float m = fabsf(QuantRoundPosV3(p.x, 100.0f));
    const float ay = fabsf(QuantRoundPosV3(p.y, 100.0f));
    m = m <= ay ? ay : m;
    const float az = fabsf(QuantRoundPosV3(p.z, 100.0f));
    m = m <= az ? az : m;
    const float sp = RocketGpuSinf(m * pitch_freq) * pitch_amp;
    const float sy = RocketGpuSinf(m * yaw_freq) * yaw_amp;
    const float sr = RocketGpuSinf(m * roll_freq) * roll_amp;
    Vec3 fwd, rgt, up;
    GameUeQuatAxes(car.rot, fwd, rgt, up);
    t = (up * sy + rgt * sp) + fwd * sr;
    return fabsf(t.x) >= 1e-4f || fabsf(t.y) >= 1e-4f || fabsf(t.z) >= 1e-4f;
}

ROCKETGPU_HD void ApplyRobinJumpTorque(CarState& car, const CarConfig& cfg, float mass, const CarFrame* frame) {
    Vec3 t;
    if (!RobinTorque(car, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, t)) {
        return;
    }
    Vec3 i0, i1, i2;
    BuildInertiaTensorWorldBtInverse(frame, car, cfg, mass, i0, i1, i2);
    const Vec3 torque_impulse(Dot(i0, t), Dot(i1, t), Dot(i2, t));
    Vec3 v0, v1, v2;
    if (frame) {
        v0 = frame->inv_world_row0_bt; v1 = frame->inv_world_row1_bt; v2 = frame->inv_world_row2_bt;
    } else {
        BuildInvInertiaTensorWorldBt(car, cfg, mass, v0, v1, v2);
    }
    car.ang_vel += Vec3(Dot(v0, torque_impulse), Dot(v1, torque_impulse), Dot(v2, torque_impulse));
}

ROCKETGPU_HD void ApplyRobinAirTorque(CarState& car, const CarConfig& cfg, float mass, float dt,
                                      WheelRuntimeState* wheel_runtime_state, const CarFrame* frame) {
    Vec3 t;
    if (!RobinTorque(car, 1e-4f, 1.0f, 2e-4f, 1.0f, 3e-4f, 2.0f, t)) {
        return;
    }
    if (wheel_runtime_state) {
        QueueAngularAccelWorld(wheel_runtime_state, car, cfg, mass, t, frame);
        Vec3 i0, i1, i2;
        BuildInertiaTensorWorldBtInverse(frame, car, cfg, mass, i0, i1, i2);
        QueueOfficialTorqueBt(wheel_runtime_state, Mul3x3Vec(i0, i1, i2, t), frame);
    } else {
        ApplyAngularAccelWorld(car, t, dt);
    }
}

ROCKETGPU_HD void StartJump(CarState& car, const CarConfig& cfg, const MutatorConfig& mutator, const CarFrame* frame = nullptr) {
    car.is_jumping = 1;
    car.internal_jump_time = 0.0f;
    if (cfg.three_wheels) {
        ApplyRobinJumpTorque(car, cfg, mutator.car_mass, frame);
    }

    (void)frame;
    ApplyCentralImpulseBt(car, GameJumpUpScaledBt(car, mutator.jump_immediate_speed * mutator.car_mass), mutator.car_mass);
}

ROCKETGPU_HD void ContinueJump(
    CarState& car,
    const CarControls& controls,
    const MutatorConfig& mutator,
    float dt,
    WheelRuntimeState* wheel_runtime_state,
    const CarFrame* frame = nullptr) {
    if (car.is_jumping) {
        if (!(car.internal_jump_time < mutator.jump_min_time ||
              (controls.jump && !car.jump_hold_broken && car.internal_jump_time < mutator.jump_max_time))) {
            car.is_jumping = 0;
        }
    }
    if (car.is_jumping) {
        car.has_jumped = 1;

        const Vec3 jump_force_bt = GameJumpUpScaledBt(car, mutator.jump_accel * mutator.car_mass);

        if (wheel_runtime_state && frame && frame->queue_forces) {
            frame->queued_jump_force = jump_force_bt;
            frame->has_queued_jump_force = true;
        } else if (wheel_runtime_state) {
            QueueCentralForce(wheel_runtime_state, jump_force_bt, frame);
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
    float raw_pitch = 2.0f,
    const CarFrame* frame = nullptr,
    Vec3* deferred_torque_bt = nullptr) {
    if (car.is_flipping) {
        car.is_flipping = (car.has_flipped && car.internal_flip_time < mutator.flip_torque_time) ? 1 : 0;
    }
    if (!car.is_flipping) {
        return;
    }

    const Vec3 rel_torque = car.flip_rel_torque;
    float torque_y = rintf(rel_torque.y * mutator.flip_torque_y);
    const float cancel_pitch = DodgeCancelPitch(car, controls, raw_pitch);
    if (rel_torque.y != 0.0f && cancel_pitch != 0.0f && Sign(rel_torque.y) == Sign(cancel_pitch)) {
        torque_y *= (1.0f - Min(Abs(cancel_pitch), 1.0f));
    }

    const Vec3 dodge_torque_local(
        rintf(rel_torque.x * mutator.flip_torque_x),
        torque_y,
        0.0f);
    const Vec3 dodge_torque_world = RotateCarBasis(car, dodge_torque_local);
    if (wheel_runtime_state) {
        QueueAngularAccelWorld(wheel_runtime_state, car, cfg, mutator.car_mass, dodge_torque_world, frame);

        Vec3 ue_f, ue_r, ue_u;
        GameUeQuatAxes(car.rot, ue_f, ue_r, ue_u);
        const Vec3& t = dodge_torque_local;
        const Vec3 t_world(((t.y * ue_r.x + t.x * ue_f.x) + t.z * ue_u.x) + 0.0f * 0.0f,
                           ((t.y * ue_r.y + t.x * ue_f.y) + t.z * ue_u.y) + 0.0f * 0.0f,
                           ((t.y * ue_r.z + t.x * ue_f.z) + t.z * ue_u.z) + 0.0f * 0.0f);
        constexpr float kGate = 1.0e-4f;
        if (Abs(t_world.x) >= kGate || Abs(t_world.y) >= kGate || Abs(t_world.z) >= kGate) {
            Vec3 i0, i1, i2;
            BuildInertiaTensorWorldBtInverse(frame, car, cfg, mutator.car_mass, i0, i1, i2);
            if (deferred_torque_bt) {
                *deferred_torque_bt = Mul3x3Vec(i0, i1, i2, t_world);
            } else {
                QueueOfficialTorqueBt(wheel_runtime_state, Mul3x3Vec(i0, i1, i2, t_world), frame);
            }
        }
    } else {
        ApplyAngularAccelWorld(car, dodge_torque_world, dt);
    }
}

ROCKETGPU_HD void ApplyAutoFlip(
    CarState& car,
    const CarControls& controls,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    float dt,
    const CarFrame* frame = nullptr) {
    const bool jump_pressed = controls.jump && !car.last_controls.jump;
    if (jump_pressed && car.world_contact.has_contact && car.world_contact.contact_normal.z > 0.707f) {
        Vec3 rb_fwd(1.0f, 0.0f, 0.0f);
        Vec3 rb_rgt(0.0f, 1.0f, 0.0f);
        Vec3 rb_up(0.0f, 0.0f, 1.0f);
        CarBasisColumns(frame, car, rb_fwd, rb_rgt, rb_up);
        (void)rb_fwd;
        const float roll = -RocketGpuUcrtAtan2f(rb_rgt.z, rb_up.z);
        const float abs_roll = Abs(roll);

        const bool flip_gate =
            (Dot(car.world_contact.contact_normal, Ue3RotatorUpAxis(Ue3QuatToRotator(car.rot))) < -0.966f);
        if (flip_gate) {
            const float flip_time = cfg.three_wheels ? 0.2f : 0.4f;
            car.auto_flip_timer = static_cast<float>(
                static_cast<double>(flip_time) * (static_cast<double>(abs_roll) / 3.14159265358979323846));
            car.auto_flip_torque_scale = (roll > 0.0f) ? 1.0f : -1.0f;
            car.is_auto_flipping = 1;

            Vec3 af_fwd, af_rgt, af_up;
            GameUeQuatAxes(car.rot, af_fwd, af_rgt, af_up);
            (void)af_fwd; (void)af_rgt;
            ApplyCentralImpulseViaBtRoundtrip(car, (-af_up) * 200.0f, mutator.car_mass);
        }
    }

    if (car.is_auto_flipping) {
        if (car.auto_flip_timer <= 0.0f) {
            car.is_auto_flipping = 0;
            car.auto_flip_timer = 0.0f;
        } else {
            Vec3 aft_fwd, aft_rgt, aft_up;
            GameUeQuatAxes(car.rot, aft_fwd, aft_rgt, aft_up);
            (void)aft_rgt; (void)aft_up;

            const Vec3 t_flip = (aft_fwd * 1000.0f) * car.auto_flip_torque_scale;
            Vec3 m0, m1, m2, i0, i1, i2;
            if (frame) {
                m0 = frame->inv_world_row0_bt;
                m1 = frame->inv_world_row1_bt;
                m2 = frame->inv_world_row2_bt;
            } else {
                BuildInvInertiaTensorWorldBt(car, cfg, mutator.car_mass, m0, m1, m2);
            }
            BuildInertiaTensorWorldBtInverse(frame, car, cfg, mutator.car_mass, i0, i1, i2);
            car.ang_vel += Mul3x3Vec(m0, m1, m2, Mul3x3Vec(i0, i1, i2, t_flip));

            car.auto_flip_timer -= dt;
        }
    }
}

ROCKETGPU_HD bool TryDoubleJumpOrFlip(
    CarState& car,
    const CarControls& controls_in,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    float forward_speed_uu,
    float raw_pitch = 2.0f,
    const CarFrame* frame = nullptr) {
    CarControls controls = controls_in;
    controls.pitch = -QuantizeAxisInput(-(raw_pitch <= 1.0f ? raw_pitch : controls.pitch));

    Vec3 forward(1.0f, 0.0f, 0.0f);
    Vec3 right_unused(0.0f, 1.0f, 0.0f);
    Vec3 up(0.0f, 0.0f, 1.0f);
    CarBasisColumns(frame, car, forward, right_unused, up);
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
        car.flip_timer_live = 1;
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

            const float dodge_imp = mutator.flip_initial_vel_scale * mutator.car_mass;
            const float dodge_imp_x =
                should_dodge_backwards ? rintf(dodge_imp * mutator.flip_backward_impulse_scale_x) : dodge_imp;
            Vec3 initial_dodge_vel(dodge_dir.x * dodge_imp_x, dodge_dir.y * dodge_imp, 0.0f);
            const float max_speed_scale_x = should_dodge_backwards ? mutator.flip_backward_impulse_max_speed_scale
                                                                   : mutator.flip_forward_impulse_max_speed_scale;
            initial_dodge_vel.x *= ((max_speed_scale_x - 1.0f) * forward_speed_ratio) + 1.0f;
            initial_dodge_vel.y *= ((mutator.flip_side_impulse_max_speed_scale - 1.0f) * forward_speed_ratio) + 1.0f;

            const int dodge_yaw = Ue3QuatToRotator(car.rot).yaw;
            const int dodge_yaw_idx = (dodge_yaw >> 2) & 16383;
            const Vec3 forward_dir_2d(Ue3SinTab((dodge_yaw_idx + 4096) & 16383), Ue3SinTab(dodge_yaw_idx), 0.0f);
            Vec3 right_dir_2d(-forward_dir_2d.y, forward_dir_2d.x, 0.0f);
            Vec3 final_delta_vel = forward_dir_2d * initial_dodge_vel.x + right_dir_2d * initial_dodge_vel.y;
            ApplyCentralImpulseBt(car, final_delta_vel * 0.02f, mutator.car_mass);
        }
    } else {
        if (cfg.three_wheels) {
            ApplyRobinJumpTorque(car, cfg, mutator.car_mass, frame);
        }
        ApplyCentralImpulseBt(car, GameJumpUpScaledBt(car, mutator.jump_immediate_speed * mutator.car_mass), mutator.car_mass);

        car.has_double_jumped = 1;
    }
    return is_flip_input;
}

ROCKETGPU_HD void UpdateJumpState(
    CarState& car,
    const CarControls& controls,
    const MutatorConfig& mutator,
    float dt,
    WheelRuntimeState* wheel_runtime_state = nullptr,
    const CarFrame* frame = nullptr) {
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

    ContinueJump(car, controls, mutator, dt, wheel_runtime_state, frame);
}

ROCKETGPU_HD float CalFlipRestartTimeOnGround(const CarState& car, const MutatorConfig& mutator, float dt) {
    if (car.is_flipping) {
        return 0.0f;
    }
    if (!car.has_flipped) {
        return car.internal_flip_time;
    }
    if (car.internal_flip_time < mutator.flip_torque_time) {
        return 0.0f;
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
    bool* flip_started = nullptr,
    const CarFrame* frame = nullptr) {
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
            if (TryDoubleJumpOrFlip(car, controls, cfg, mutator, forward_speed_uu, raw_pitch, frame) && flip_started) {
                *flip_started = true;
            }

            if (car.is_flipping && car.internal_flip_time == 0.0f) {
                car.internal_flip_time = -dt;
            }
        }
    }

    if (car.is_flipping) {
        car.internal_flip_time += dt;
    } else if (car.has_flipped || car.flip_timer_live || (car.internal_flip_time != 0.0f)) {
        car.internal_flip_time += dt;
    }
}

ROCKETGPU_HD void CapCarAngularSpeed(CarState& car, const MutatorConfig& mutator) {
    car.ang_vel = NativeSseClampLength(car.ang_vel, mutator.car_max_ang_speed,
        mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift);
}

ROCKETGPU_HD Vec3 CapBallSpeeds(BallState& ball, const MutatorConfig& mutator) {
    Vec3 vel_uu = ball.vel_bt * 50.0f;
    if (!mutator.native_sphere_rsqrt_table) {
        const float vel_uu_len_sq = LengthSq(vel_uu);
        if (vel_uu_len_sq > mutator.ball_max_speed * mutator.ball_max_speed) {
            vel_uu = (vel_uu * RsqrtScalar(vel_uu_len_sq)) * mutator.ball_max_speed;
            SetBallLinearVelocityBt(ball, vel_uu * UnitsUUToBT());
        }
    } else {
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
        vel_uu = ball.vel_bt * 50.0f;
    }

    ball.ang_vel = NativeSseClampLength(ball.ang_vel, mutator.ball_max_ang_speed,
        mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift);
    return vel_uu;
}

ROCKETGPU_HD Vec3 BulletDampedLinearVelocityFromBt(const Vec3& vel_bt, float damping, float dt) {
    if (damping <= 0.0f) {
        return vel_bt;
    }
    const float damping_scale = RocketGpuUcrtPowf(Max(0.0f, 1.0f - damping), dt);
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

ROCKETGPU_HD void IntegrateCarTransformOnly(CarState& car, float dt, bool native_sse_math = false) {
    if (!(car.vel_bt.x == 0.0f && car.vel_bt.y == 0.0f && car.vel_bt.z == 0.0f)) {
        SetCarOriginBt(car, car.pos_bt + car.vel_bt * dt);
    }

    IntegrateCarOrientation(car, car.ang_vel, dt, native_sse_math);
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
    ball.restored_inv_inertia_valid = 0;
    if (mutator.ball_shape == 1 || !mutator.ball_no_rot) {
        if (NativeSseMath(mutator)) {
            (void)IntegrateOrientationBasisColumns(
                ball.basis_forward, ball.basis_right, ball.basis_up, ball.rot, ball.ang_vel, dt, true);
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

ROCKETGPU_HD float QuantTruncOffset(float x, float scale) {
    const float k = truncf(x * scale);
    return k == 0.0f ? 0.0f : (k + copysignf(0.1f, k)) / scale;
}

ROCKETGPU_HD float QuantTruncOffsetV3(float x, float scale) {
    const float inv_scale = 1.0f / scale;
    const float k = static_cast<float>(static_cast<int32_t>(x * scale));
    const float rounded = k * inv_scale;
    const float offset_mag = 0.1f * inv_scale;
    const float sgn = rounded == 0.0f ? 0.0f : (rounded > 0.0f ? 1.0f : -1.0f);
    const float offset = sgn * offset_mag;
    return rounded + offset;
}

ROCKETGPU_HD float QuantRoundPosV3(float x, float scale) {
    const float scaled = x * scale;
    return floorf(scaled + 0.5f) * (1.0f / scale);
}

ROCKETGPU_HD void QuantizeBallStateV3Exact(BallState& ball, Vec3 v) {
    constexpr float kUuToBt = 1.0f / 50.0f;
    const Vec3 p = ball.pos_bt * 50.0f;
    const Vec3 qp(QuantRoundPosV3(p.x, 100.0f), QuantRoundPosV3(p.y, 100.0f), QuantRoundPosV3(p.z, 100.0f));
    SetBallOriginBt(ball, qp * kUuToBt);
    const Vec3 qv(QuantTruncOffsetV3(v.x, 100.0f), QuantTruncOffsetV3(v.y, 100.0f), QuantTruncOffsetV3(v.z, 100.0f));
    SetBallLinearVelocityBt(ball, qv * kUuToBt);
    ball.pos = qp;
    ball.vel = qv;
    ball.ang_vel = Vec3(QuantTruncOffsetV3(ball.ang_vel.x, 10000.0f),
                        QuantTruncOffsetV3(ball.ang_vel.y, 10000.0f),
                        QuantTruncOffsetV3(ball.ang_vel.z, 10000.0f));
    if (!ball.basis_authoritative) {
        GetOfficialUploadBasisVectors(ball.rot, ball.basis_forward, ball.basis_right, ball.basis_up);
        ball.basis_authoritative = 1;
    }
    ball.inertia_basis_forward = ball.basis_forward;
    ball.inertia_basis_right = ball.basis_right;
    ball.inertia_basis_up = ball.basis_up;
    ball.inertia_basis_valid = 1;
    ball.rot = GameQuantizeBasisColumns(ball.basis_forward, ball.basis_right, ball.basis_up);
}

ROCKETGPU_HD void FinishBallTick(BallState& ball) {
    ApplyBallVelocityImpulseCache(ball);
    ball.inertia_basis_valid = 0;
    ball.tick_count_since_update += 1;
    if (ball.constrained_2d) {
        --ball.constrained_2d;
    }
    return;
}

#if defined(__CUDA_ARCH__)
#define ROCKETGPU_NOINLINE __noinline__
#else
#define ROCKETGPU_NOINLINE
#endif
ROCKETGPU_HD ROCKETGPU_NOINLINE void QuantizeCarState(CarState& car, bool v3_exact = false,
                                                      const Vec3* pre_scaled_vel_uu = nullptr) {
    if (v3_exact) {
        constexpr float kUuToBt = 1.0f / 50.0f;
        const uint8_t touched_v3 = car.vel_touched_solve_stage;
        const Vec3 p = car.pos_bt * 50.0f;
        const Vec3 qp(QuantRoundPosV3(p.x, 100.0f), QuantRoundPosV3(p.y, 100.0f), QuantRoundPosV3(p.z, 100.0f));
        SetCarOriginBt(car, qp * kUuToBt);
        car.pos = qp;
        const Vec3 v = pre_scaled_vel_uu ? *pre_scaled_vel_uu : car.vel_bt * 50.0f;
        const Vec3 q(QuantTruncOffsetV3(v.x, 100.0f), QuantTruncOffsetV3(v.y, 100.0f), QuantTruncOffsetV3(v.z, 100.0f));
        SetCarLinearVelocityBt(car, q * kUuToBt);
        car.vel = q;
        car.vel_touched_solve_stage = touched_v3;
        car.ang_vel = Vec3(QuantTruncOffsetV3(car.ang_vel.x, 10000.0f), QuantTruncOffsetV3(car.ang_vel.y, 10000.0f),
                           QuantTruncOffsetV3(car.ang_vel.z, 10000.0f));
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

ROCKETGPU_HD void FinishCarTick(CarState& car) {
    car.restored_inv_inertia_valid = 0;
    if (car.HasFlipOrJump() && !car.flip_or_jump_last_tick)
        car.last_flip_or_jump_start = car.tick_count_since_update;
    car.flip_or_jump_last_tick = car.HasFlipOrJump();
    car.ball_contact_last_tick = car.ball_contact_this_tick;
    if (car.is_demoed) {
        return;
    }

    if (car.pending_bump_count > 0) {
        Vec3 vel_bt = car.vel_bt;
        for (int i = 0; i < car.pending_bump_count; ++i) {
            vel_bt = vel_bt + car.pending_bump_bt[i];
        }
        SetCarLinearVelocityBt(car, vel_bt);
        car.pending_bump_count = 0;
        car.vel_impulse_cache = Vec3(0.0f, 0.0f, 0.0f);
        car.vel_impulse_cache_bt = Vec3(0.0f, 0.0f, 0.0f);
    } else if (LengthSq(car.vel_impulse_cache_bt) > 0.0f) {
        SetCarLinearVelocityBt(car, car.vel_bt + car.vel_impulse_cache_bt);
        car.vel_impulse_cache = Vec3(0.0f, 0.0f, 0.0f);
        car.vel_impulse_cache_bt = Vec3(0.0f, 0.0f, 0.0f);
    } else if (LengthSq(car.vel_impulse_cache) > 0.0f) {
        SetCarLinearVelocityUu(car, car.vel + car.vel_impulse_cache);
        car.vel_impulse_cache = Vec3(0.0f, 0.0f, 0.0f);
    }

    car.tick_count_since_update += 1;
    return;
}

ROCKETGPU_HD void CalTickStartLimitAndQuantizeCar(CarState& car, const MutatorConfig& mutator) {
    if (car.is_demoed) {
        return;
    }
    Vec3 vel_uu = car.vel_bt * 50.0f;
    const float vel_uu_len_sq = LengthSq(vel_uu);
    if (vel_uu_len_sq > mutator.car_max_speed * mutator.car_max_speed) {
        const Vec3 vel_uu_normalized = mutator.native_sphere_rsqrt_table
            ? NativeSseSphereNormalizer{mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift}(vel_uu)
            : vel_uu * RsqrtScalar(vel_uu_len_sq);
        vel_uu = vel_uu_normalized * mutator.car_max_speed;
        SetCarLinearVelocityBt(car, vel_uu * UnitsUUToBT());
    }
    CapCarAngularSpeed(car, mutator);
    CarBasisColumns(car, car.tick_start_basis_forward, car.tick_start_basis_right, car.tick_start_basis_up);
    QuantizeCarState(car, true, &vel_uu);
}

ROCKETGPU_HD void CalFlipZDampPrepass(CarState& car, const MutatorConfig& mutator, float dt) {
    if (car.is_demoed || car.is_on_ground || !car.is_flipping) {
        return;
    }
    const float flip_time = car.internal_flip_time + dt;
    if (!(flip_time >= 0.15f && flip_time <= mutator.flip_torque_time)) {
        return;
    }
    if (car.vel_bt.z < 0.0f || flip_time < 0.21f) {
        const float dv_uu = (-car.vel.z) * 0.35f;
        if (Abs(dv_uu) >= 1.0e-4f) {
            const float inv_mass = 1.0f / mutator.car_mass;
            const float impulse = (dv_uu * 0.02f) * (1.0f / inv_mass);
            const Vec3 v = car.vel_bt;
            SetCarLinearVelocityBt(car, Vec3(v.x, v.y, v.z + impulse * inv_mass));
        }
    }
}

ROCKETGPU_HD void CalTickStartLimitAndQuantizeBall(BallState& ball, const MutatorConfig& mutator) {
    QuantizeBallStateV3Exact(ball, CapBallSpeeds(ball, mutator));
}

ROCKETGPU_HD void PostTickCar(CarState& car, CarControls controls, float dt) {
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

struct WheelGjkSubSimplexResult {
    Vec3 closest = Vec3(0.0f, 0.0f, 0.0f);
    float bc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t used = 0;
    bool degenerate = false;
};

ROCKETGPU_HD void WheelGjkClosestPtPointTriangle(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c,
    WheelGjkSubSimplexResult& result) {
    result.used = 0;
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 ap = p - a;
    const float d1 = Dot(ab, ap);
    const float d2 = Dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        result.closest = a;
        result.used = 1u;
        result.bc[0] = 1.0f; result.bc[1] = 0.0f; result.bc[2] = 0.0f; result.bc[3] = 0.0f;
        return;
    }
    const Vec3 bp = p - b;
    const float d3 = Dot(ab, bp);
    const float d4 = Dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        result.closest = b;
        result.used = 2u;
        result.bc[0] = 0.0f; result.bc[1] = 1.0f; result.bc[2] = 0.0f; result.bc[3] = 0.0f;
        return;
    }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        result.closest = a + ab * v;
        result.used = 1u | 2u;
        result.bc[0] = 1.0f - v; result.bc[1] = v; result.bc[2] = 0.0f; result.bc[3] = 0.0f;
        return;
    }
    const Vec3 cp = p - c;
    const float d5 = Dot(ab, cp);
    const float d6 = Dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        result.closest = c;
        result.used = 4u;
        result.bc[0] = 0.0f; result.bc[1] = 0.0f; result.bc[2] = 1.0f; result.bc[3] = 0.0f;
        return;
    }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        result.closest = a + ac * w;
        result.used = 1u | 4u;
        result.bc[0] = 1.0f - w; result.bc[1] = 0.0f; result.bc[2] = w; result.bc[3] = 0.0f;
        return;
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        result.closest = b + (c - b) * w;
        result.used = 2u | 4u;
        result.bc[0] = 0.0f; result.bc[1] = 1.0f - w; result.bc[2] = w; result.bc[3] = 0.0f;
        return;
    }
    const float denom = 1.0f / ((va + vb) + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    result.closest = (a + ab * v) + ac * w;
    result.used = 1u | 2u | 4u;
    result.bc[0] = (1.0f - v) - w; result.bc[1] = v; result.bc[2] = w; result.bc[3] = 0.0f;
}

ROCKETGPU_HD int WheelGjkPointOutsideOfPlane(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) {
    const Vec3 normal = Cross(b - a, c - a);
    const float signp = Dot(p - a, normal);
    const float signd = Dot(d - a, normal);
    if (signd * signd < 1.0e-4f * 1.0e-4f) {
        return -1;
    }
    return signp * signd < 0.0f ? 1 : 0;
}

ROCKETGPU_HD uint32_t WheelGjkUsedBit(uint32_t used, int i) {
    return (used >> i) & 1u;
}

ROCKETGPU_HD bool WheelGjkClosestPtPointTetrahedron(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c,
    const Vec3& d, WheelGjkSubSimplexResult& final_result) {
    WheelGjkSubSimplexResult temp;
    final_result.closest = p;
    final_result.used = 1u | 2u | 4u | 8u;
    const int outside_abc = WheelGjkPointOutsideOfPlane(p, a, b, c, d);
    const int outside_acd = WheelGjkPointOutsideOfPlane(p, a, c, d, b);
    const int outside_adb = WheelGjkPointOutsideOfPlane(p, a, d, b, c);
    const int outside_bdc = WheelGjkPointOutsideOfPlane(p, b, d, c, a);
    if (outside_abc < 0 || outside_acd < 0 || outside_adb < 0 || outside_bdc < 0) {
        final_result.degenerate = true;
        return false;
    }
    if (!outside_abc && !outside_acd && !outside_adb && !outside_bdc) {
        return false;
    }
    float best_sq_dist = 3.402823466e+38f;
    if (outside_abc) {
        WheelGjkClosestPtPointTriangle(p, a, b, c, temp);
        const Vec3 q = temp.closest;
        const float sq_dist = Dot(q - p, q - p);
        if (sq_dist < best_sq_dist) {
            best_sq_dist = sq_dist;
            final_result.closest = q;
            final_result.used = WheelGjkUsedBit(temp.used, 0) | (WheelGjkUsedBit(temp.used, 1) << 1) | (WheelGjkUsedBit(temp.used, 2) << 2);
            final_result.bc[0] = temp.bc[0]; final_result.bc[1] = temp.bc[1];
            final_result.bc[2] = temp.bc[2]; final_result.bc[3] = 0.0f;
        }
    }
    if (outside_acd) {
        WheelGjkClosestPtPointTriangle(p, a, c, d, temp);
        const Vec3 q = temp.closest;
        const float sq_dist = Dot(q - p, q - p);
        if (sq_dist < best_sq_dist) {
            best_sq_dist = sq_dist;
            final_result.closest = q;
            final_result.used = WheelGjkUsedBit(temp.used, 0) | (WheelGjkUsedBit(temp.used, 1) << 2) | (WheelGjkUsedBit(temp.used, 2) << 3);
            final_result.bc[0] = temp.bc[0]; final_result.bc[1] = 0.0f;
            final_result.bc[2] = temp.bc[1]; final_result.bc[3] = temp.bc[2];
        }
    }
    if (outside_adb) {
        WheelGjkClosestPtPointTriangle(p, a, d, b, temp);
        const Vec3 q = temp.closest;
        const float sq_dist = Dot(q - p, q - p);
        if (sq_dist < best_sq_dist) {
            best_sq_dist = sq_dist;
            final_result.closest = q;
            final_result.used = WheelGjkUsedBit(temp.used, 0) | (WheelGjkUsedBit(temp.used, 2) << 1) | (WheelGjkUsedBit(temp.used, 1) << 3);
            final_result.bc[0] = temp.bc[0]; final_result.bc[1] = temp.bc[2];
            final_result.bc[2] = 0.0f; final_result.bc[3] = temp.bc[1];
        }
    }
    if (outside_bdc) {
        WheelGjkClosestPtPointTriangle(p, b, d, c, temp);
        const Vec3 q = temp.closest;
        const float sq_dist = Dot(q - p, q - p);
        if (sq_dist < best_sq_dist) {
            best_sq_dist = sq_dist;
            final_result.closest = q;
            final_result.used = (WheelGjkUsedBit(temp.used, 0) << 1) | (WheelGjkUsedBit(temp.used, 2) << 2) | (WheelGjkUsedBit(temp.used, 1) << 3);
            final_result.bc[0] = 0.0f; final_result.bc[1] = temp.bc[0];
            final_result.bc[2] = temp.bc[2]; final_result.bc[3] = temp.bc[1];
        }
    }
    return true;
}

struct WheelGjkSimplex {
    Vec3 w[4];
    Vec3 q[4];
    Vec3 p = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 cached_p1 = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 cached_p2 = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 cached_v = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 last_w = Vec3(0.0f, 0.0f, 0.0f);
    int n = 0;
    bool cached_valid = false;
    bool needs_update = true;

    ROCKETGPU_HD void Reset() {
        cached_valid = false;
        n = 0;
        needs_update = true;
        last_w = Vec3(1.0e18f, 1.0e18f, 1.0e18f);
    }
    ROCKETGPU_HD void RemoveVertex(int index) {
        --n;
        w[index] = w[n];
        q[index] = q[n];
    }
    ROCKETGPU_HD void ReduceVertices(uint32_t used) {
        if (n >= 4 && !(used & 8u)) RemoveVertex(3);
        if (n >= 3 && !(used & 4u)) RemoveVertex(2);
        if (n >= 2 && !(used & 2u)) RemoveVertex(1);
        if (n >= 1 && !(used & 1u)) RemoveVertex(0);
    }
    ROCKETGPU_HD bool InSimplex(const Vec3& v) const {
        bool found = false;
        for (int i = 0; i < n; ++i) {
            if (LengthSq(v - w[i]) <= 1.0e-4f) {
                found = true;
            }
        }
        if (v.x == last_w.x && v.y == last_w.y && v.z == last_w.z) {
            return true;
        }
        return found;
    }
    ROCKETGPU_HD void AddVertex(const Vec3& vw, const Vec3& vp, const Vec3& vq) {
        last_w = vw;
        needs_update = true;
        w[n] = vw;
        p = vp;
        q[n] = vq;
        ++n;
    }
    ROCKETGPU_HD static bool Valid(const WheelGjkSubSimplexResult& r) {
        return r.bc[0] >= 0.0f && r.bc[1] >= 0.0f && r.bc[2] >= 0.0f && r.bc[3] >= 0.0f;
    }
    ROCKETGPU_HD bool UpdateClosestVectorAndPoints() {
        if (!needs_update) {
            return cached_valid;
        }
        needs_update = false;
        WheelGjkSubSimplexResult bc;
        const Vec3 origin(0.0f, 0.0f, 0.0f);
        switch (n) {
            case 1: {
                cached_p1 = p;
                cached_p2 = q[0];
                cached_v = cached_p1 - cached_p2;
                bc.bc[0] = 1.0f;
                cached_valid = Valid(bc);
                break;
            }
            case 2: {
                const Vec3 diff = origin - w[0];
                const Vec3 v = w[1] - w[0];
                float t = Dot(v, diff);
                if (t > 0.0f) {
                    const float dot_vv = Dot(v, v);
                    if (t < dot_vv) {
                        t /= dot_vv;
                        bc.used = 1u | 2u;
                    } else {
                        t = 1.0f;
                        bc.used = 2u;
                    }
                } else {
                    t = 0.0f;
                    bc.used = 1u;
                }
                bc.bc[0] = 1.0f - t;
                bc.bc[1] = t;
                cached_p1 = p + (p - p) * t;
                cached_p2 = q[0] + (q[1] - q[0]) * t;
                cached_v = cached_p1 - cached_p2;
                ReduceVertices(bc.used);
                cached_valid = Valid(bc);
                break;
            }
            case 3: {
                WheelGjkClosestPtPointTriangle(origin, w[0], w[1], w[2], bc);
                cached_p1 = (p * bc.bc[0] + p * bc.bc[1]) + p * bc.bc[2];
                cached_p2 = (q[0] * bc.bc[0] + q[1] * bc.bc[1]) + q[2] * bc.bc[2];
                cached_v = cached_p1 - cached_p2;
                ReduceVertices(bc.used);
                cached_valid = Valid(bc);
                break;
            }
            case 4: {
                if (WheelGjkClosestPtPointTetrahedron(origin, w[0], w[1], w[2], w[3], bc)) {
                    cached_p1 = ((p * bc.bc[0] + p * bc.bc[1]) + p * bc.bc[2]) + p * bc.bc[3];
                    cached_p2 = ((q[0] * bc.bc[0] + q[1] * bc.bc[1]) + q[2] * bc.bc[2]) + q[3] * bc.bc[3];
                    cached_v = cached_p1 - cached_p2;
                    ReduceVertices(bc.used);
                } else {
                    if (bc.degenerate) {
                        cached_valid = false;
                    } else {
                        cached_valid = true;
                        cached_v = Vec3(0.0f, 0.0f, 0.0f);
                    }
                    break;
                }
                cached_valid = Valid(bc);
                break;
            }
            default:
                cached_valid = false;
        }
        return cached_valid;
    }
};

ROCKETGPU_HD bool WheelGjkPointBoxClosest(const Vec3& point_a, const Vec3& box_origin, const Vec3& f, const Vec3& r,
    const Vec3& u, const Vec3& box_core, float box_margin, bool& has_result, float& out_distance, Vec3& out_normal) {
    constexpr float kRelError2 = 1.0e-6f;
    constexpr float kSimdEpsilon = 1.1920929e-7f;
    const Vec3 offset = (point_a + box_origin) * 0.5f;
    const Vec3 local_a = point_a - offset;
    const Vec3 local_b = box_origin - offset;
    const float margin = 0.0f + box_margin;
    Vec3 axis(0.0f, 1.0f, 0.0f);
    float squared_distance = 1.0e18f;
    WheelGjkSimplex simplex;
    simplex.Reset();
    bool check_simplex = false;
    int cur_iter = 0;
    const Vec3 p_world = Vec3(0.0f, 0.0f, 0.0f) + local_a;
    while (true) {
        const Vec3 sep_b(Dot(f, axis), Dot(r, axis), Dot(u, axis));
        const Vec3 q_local(sep_b.x >= 0.0f ? box_core.x : -box_core.x, sep_b.y >= 0.0f ? box_core.y : -box_core.y,
                           sep_b.z >= 0.0f ? box_core.z : -box_core.z);
        const Vec3 q_world((f.x * q_local.x + r.x * q_local.y) + u.x * q_local.z + local_b.x,
                           (f.y * q_local.x + r.y * q_local.y) + u.y * q_local.z + local_b.y,
                           (f.z * q_local.x + r.z * q_local.y) + u.z * q_local.z + local_b.z);
        const Vec3 w = p_world - q_world;
        const float delta = Dot(axis, w);
        if (delta > 0.0f && delta * delta > squared_distance * 1.0e18f) {
            check_simplex = true;
            break;
        }
        if (simplex.InSimplex(w)) {
            check_simplex = true;
            break;
        }
        const float f0 = squared_distance - delta;
        const float f1 = squared_distance * kRelError2;
        if (f0 <= f1) {
            check_simplex = true;
            break;
        }
        simplex.AddVertex(w, p_world, q_world);
        if (!simplex.UpdateClosestVectorAndPoints()) {
            check_simplex = true;
            break;
        }
        const Vec3 new_axis = simplex.cached_v;
        if (LengthSq(new_axis) < kRelError2) {
            axis = new_axis;
            check_simplex = true;
            break;
        }
        const float previous_squared_distance = squared_distance;
        squared_distance = LengthSq(new_axis);
        if (previous_squared_distance - squared_distance <= kSimdEpsilon * previous_squared_distance) {
            check_simplex = true;
            break;
        }
        axis = new_axis;
        if (cur_iter++ > 1000) {
            break;
        }
        if (simplex.n == 4) {
            break;
        }
    }
    if (!check_simplex) {
        return has_result;
    }
    simplex.UpdateClosestVectorAndPoints();
    const float len_sq = LengthSq(axis);
    if (!(len_sq > kSimdEpsilon * kSimdEpsilon)) {
        return has_result;
    }
    const float rlen = 1.0f / sqrtf(len_sq);
    Vec3 normal = axis * rlen;
    const float distance = (1.0f / rlen) - margin;
    if (!(distance < 0.0f || distance * distance < 1.0e18f)) {
        return has_result;
    }
    const Vec3 half_with_margin = box_core + Vec3(box_margin, box_margin, box_margin);
    const Vec3 extent(
        (half_with_margin.x * Abs(f.x) + half_with_margin.y * Abs(r.x)) + half_with_margin.z * Abs(u.x),
        (half_with_margin.x * Abs(f.y) + half_with_margin.y * Abs(r.y)) + half_with_margin.z * Abs(u.y),
        (half_with_margin.x * Abs(f.z) + half_with_margin.y * Abs(r.z)) + half_with_margin.z * Abs(u.z));
    const Vec3 a_min = local_a - Vec3(0.0f, 0.0f, 0.0f);
    const Vec3 a_max = local_a + Vec3(0.0f, 0.0f, 0.0f);
    const Vec3 b_min = local_b - extent;
    const Vec3 b_max = local_b + extent;
    const Vec3 pos_a = (a_max + a_min) * 0.5f;
    const Vec3 pos_b = (b_min + b_max) * 0.5f;
    if (Dot(pos_a - pos_b, normal) < 0.0f) {
        normal = normal * -1.0f;
    }
    if (distance < out_distance) {
        has_result = true;
        out_normal = normal;
        out_distance = distance;
    }
    return has_result;
}

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
    constexpr int kMaxIterations = 64;
    const float max_len_bt = max_len_bt_override > 0.0f ? max_len_bt_override : max_len * kUUToBT;
    const Vec3 ray_to_bt = ray_source_bt + ray_dir * max_len_bt;
    const Vec3 half_raw = BulletHitboxSize(other_cfg) * (0.5f * kUUToBT);
    const float margin = Min(0.1f * Min(half_raw.x, Min(half_raw.y, half_raw.z)), 0.04f);
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
    {
        const Vec3 rel_from = ray_source_bt - other_car.pos_bt;
        const Vec3 rel_to = ray_to_bt - other_car.pos_bt;
        const Vec3 lf(Dot(rel_from, f), Dot(rel_from, rt), Dot(rel_from, u));
        const Vec3 lt(Dot(rel_to, f), Dot(rel_to, rt), Dot(rel_to, u));
        const Vec3 off_bt = other_cfg.hitbox_offset * kUUToBT;
        const Vec3 ext = inner + Vec3(margin, margin, margin);
        const Vec3 bmin = off_bt - ext;
        const Vec3 bmax = off_bt + ext;
        if ((lf.x < bmin.x && lt.x < bmin.x) || (lf.x > bmax.x && lt.x > bmax.x) ||
            (lf.y < bmin.y && lt.y < bmin.y) || (lf.y > bmax.y && lt.y > bmax.y) ||
            (lf.z < bmin.z && lt.z < bmin.z) || (lf.z > bmax.z && lt.z > bmax.z)) {
            return false;
        }
        Vec3 dir = lt - lf;
        const float dir_len = Length(dir);
        if (!(dir_len > 0.0f)) {
            return false;
        }
        dir = dir * (1.0f / dir_len);
        const float lambda_max = Dot(dir, lt - lf);
        const float inv[3] = {dir.x == 0.0f ? 1.0e30f : 1.0f / dir.x, dir.y == 0.0f ? 1.0e30f : 1.0f / dir.y,
                              dir.z == 0.0f ? 1.0e30f : 1.0f / dir.z};
        const float from[3] = {lf.x, lf.y, lf.z};
        const float lo[3] = {bmin.x, bmin.y, bmin.z};
        const float hi[3] = {bmax.x, bmax.y, bmax.z};
        float tmin = ((inv[0] < 0.0f ? hi[0] : lo[0]) - from[0]) * inv[0];
        float tmax = ((inv[0] < 0.0f ? lo[0] : hi[0]) - from[0]) * inv[0];
        for (int a = 1; a < 3; ++a) {
            const float t0 = ((inv[a] < 0.0f ? hi[a] : lo[a]) - from[a]) * inv[a];
            const float t1 = ((inv[a] < 0.0f ? lo[a] : hi[a]) - from[a]) * inv[a];
            if (tmin > t1 || t0 > tmax) {
                return false;
            }
            if (t0 > tmin) tmin = t0;
            if (t1 < tmax) tmax = t1;
        }
        if (!(tmin < lambda_max && tmax > 0.0f)) {
            return false;
        }
    }

    const Vec3 r(ray_to_bt.x - ray_source_bt.x - (center.x - center.x), ray_to_bt.y - ray_source_bt.y - (center.y - center.y),
                 ray_to_bt.z - ray_source_bt.z - (center.z - center.z));
    bool has_result = false;
    float dist = 1.0e18f;
    Vec3 n(0.0f, 0.0f, 0.0f);
    if (!WheelGjkPointBoxClosest(ray_source_bt, center, f, rt, u, inner, margin, has_result, dist, n)) {
        return false;
    }
    float lambda = 0.0f;
    int iter = 0;
    while (true) {
        if (dist <= kRadius) {
            if (-0.0f <= Dot(r, n)) {
                return false;
            }
            break;
        }
        if (++iter > kMaxIterations) {
            return false;
        }
        const float next_lambda = lambda - dist / Dot(r, n);
        if (next_lambda > 1.0f || next_lambda < 0.0f || next_lambda <= lambda) {
            return false;
        }
        const float s = 1.0f - next_lambda;
        const Vec3 point_a(next_lambda * ray_to_bt.x + s * ray_source_bt.x, next_lambda * ray_to_bt.y + s * ray_source_bt.y,
                           next_lambda * ray_to_bt.z + s * ray_source_bt.z);
        const Vec3 box_origin(next_lambda * center.x + s * center.x, next_lambda * center.y + s * center.y,
                              next_lambda * center.z + s * center.z);
        if (!WheelGjkPointBoxClosest(point_a, box_origin, f, rt, u, inner, margin, has_result, dist, n)) {
            return false;
        }
        lambda = next_lambda;
        if (!(0.0f <= dist)) {
            break;
        }
    }

    const float n_len_sq = LengthSq(n);
    if (!(n_len_sq > 1.0e-4f) || !(lambda < 1.0f)) {
        return false;
    }
    const float inv_len = 1.0f / sqrtf(n_len_sq);
    const float s = 1.0f - lambda;
    out_hit.t = lambda * max_len;
    out_hit.ray_fraction = lambda;
    out_hit.normal = Vec3(n.x * inv_len, n.y * inv_len, n.z * inv_len);
    out_hit.point_bt = Vec3(s * ray_source_bt.x + lambda * ray_to_bt.x, s * ray_source_bt.y + lambda * ray_to_bt.y,
                            s * ray_source_bt.z + lambda * ray_to_bt.z);
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

ROCKETGPU_HD bool ClampArenaTriangleInternalEdgeNormal(
    const Vec3& edge,
    const Vec3& tri_normal,
    const Vec3& contact_normal,
    float corrected_edge_angle,
    Vec3& out_clamped_normal) {
    const Vec3 edge_cross = Normalize(Cross(edge, tri_normal), tri_normal);
    const float current_angle = RocketGpuUcrtAtan2f(Dot(contact_normal, edge_cross), Dot(contact_normal, tri_normal));

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

    const Vec3 contact = (kGameMeshEdgeRules && cal_origin_y != 0.0f)
        ? Vec3(io_point_on_b_bt.x, io_point_on_b_bt.y - cal_origin_y, io_point_on_b_bt.z)
        : io_point_on_b_bt;

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

    if (best_edge < 0) {
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
        if (back_facing) {
            ++num_concave_edge_hits;
            continue;
        }

        Vec3 clamped_local_normal(0.0f, 0.0f, 1.0f);
        const bool clamped = ClampArenaTriangleInternalEdgeNormal(
                edge,
                tri_normal * swap_factor,
                edge_idx == 0 ? contact_normal : io_normal,
                candidate_edge_angle,
                clamped_local_normal);
        const bool accepted_clamp = clamped && Dot(clamped_local_normal, tri_normal) > 0.0f;
        if (accepted_clamp) {
            io_normal = clamped_local_normal;
            io_point_on_b_bt = point_on_a_bt - io_normal * distance_bt;
        }
    }

    if (!is_near_edge) {
        return false;
    }

    if (is_near_edge && num_concave_edge_hits > 0) {
        const Vec3 new_normal = tri_normal;
        if (Dot(new_normal, contact_normal) < 0.0f) {
            return false;
        }
        io_normal = new_normal;
        io_point_on_b_bt = point_on_a_bt - io_normal * distance_bt;
    }
    return true;
}

ROCKETGPU_HD bool MaybeAdjustArenaTriangleInternalEdgeContact(
    const ArenaMeshTriangle& world_tri,
    const Vec3& point_on_a_bt,
    float distance_bt,
    Vec3& io_point_on_b_bt,
    Vec3& io_normal,
    const uint32_t* native_table, int native_shift,
    const ArenaMeshTriangle* cal_game_tri) {
    if (cal_game_tri != nullptr) {
        const float cal_origin_y = cal_game_tri->cal_origin_sign != 0
            ? static_cast<float>(cal_game_tri->cal_origin_sign) * kGoalMeshOriginYBt : 0.0f;
        return MaybeAdjustArenaTriangleInternalEdgeContactT<true>(
            *cal_game_tri, cal_origin_y, point_on_a_bt, distance_bt, io_point_on_b_bt, io_normal,
            native_table, native_shift);
    }
    return MaybeAdjustArenaTriangleInternalEdgeContactT<false>(
        world_tri, 0.0f, point_on_a_bt, distance_bt, io_point_on_b_bt, io_normal,
        native_table, native_shift);
}

ROCKETGPU_HD constexpr bool EnableArenaTriangleInternalEdgeAdjust() {
    return true;
}
