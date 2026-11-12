template <typename WheelT>
ROCKETGPU_HD float WheelRealRayLengthBt(const WheelT& wheel) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    const float rest_bt = wheel.suspension_rest_length * kUUToBT;
    const float travel_bt = (wheel.max_suspension_travel - WheelSuspensionSubtraction()) * kUUToBT;
    const float radius_bt = wheel.wheel_radius * kUUToBT;
    return (rest_bt + radius_bt) + travel_bt;
}
struct WheelRaysWhole {
    static constexpr bool kAllRays = true;
    ROCKETGPU_HD void Merge(Vec3&, Vec3&) const {}
    ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE uint32_t Excluded(const BulletSweepHistory* sweep, uint32_t mask, int bodies,
                                                           const Vec3& mn, const Vec3& mx) const {
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
};

struct ArenaRayMeshOneLane {
    using WheelRays = WheelRaysWhole;
    static constexpr bool kSplitHeaders = false;
    int first = 0;
    int step = 1;
    ROCKETGPU_HD bool Leader() const { return true; }
    ROCKETGPU_HD ArenaRayHit Combine(const ArenaRayHit& hit, int& order, float) const { return hit; }
};

#if !defined(ROCKETGPU_USE_CUDA)
inline bool HostRayClearOfMeshes(const ArenaMeshLeafGrid* leaf_grid, const ArenaMeshTriangle* mesh_triangles,
                                 const Vec3& src, const Vec3& delta) {
    if (leaf_grid == nullptr || leaf_grid->host_tables == nullptr) {
        return false;
    }
    const ArenaMeshClearance& clearance = leaf_grid->host_tables->clearance;
    if (clearance.lb == nullptr || clearance.mesh != mesh_triangles) {
        return false;
    }
    const Vec3 half = delta * 0.5f;
    return clearance.Clearance(src + half) > SqrtScalar(Dot(half, half)) + 2e-3f;
}
#else
ROCKETGPU_HD bool DeviceRayClearOfMeshes(const ArenaMeshLeafGrid* leaf_grid,
                                         const ArenaMeshTriangle* mesh_triangles, const Vec3& src,
                                         const Vec3& delta) {
    if (leaf_grid == nullptr || leaf_grid->clearance == nullptr || leaf_grid->clearance->mesh != mesh_triangles) {
        return false;
    }
    const Vec3 half = delta * 0.5f;
    return leaf_grid->clearance->Clearance(src + half) > SqrtScalar(Dot(half, half)) + 2e-3f;
}
#endif

template <typename MeshLanes = ArenaRayMeshOneLane>
ROCKETGPU_HD ROCKETGPU_HOST_INLINE ArenaRayHit CastArenaRayDeltaBt(
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
    const ArenaMeshTriangle* local_triangles = nullptr, bool meshes_clear = false, int32_t* walk_hint = nullptr) {
    (void)walk_hint;
    bool mesh_ray_cast_bt = false;
    bool neg_x_wins_mesh_tie = false;

    const Vec3 to_bt = ray_source_bt + ray_delta_bt;
    const Vec3 ray_min_bt(Min(ray_source_bt.x, to_bt.x), Min(ray_source_bt.y, to_bt.y), Min(ray_source_bt.z, to_bt.z));
    const Vec3 ray_max_bt(Max(ray_source_bt.x, to_bt.x), Max(ray_source_bt.y, to_bt.y), Max(ray_source_bt.z, to_bt.z));
    if (mesh_bvh_nodes != nullptr && mesh_bvh_node_count > 0) {
        mesh_ray_cast_bt = true;
        const bool skip_meshes = meshes_clear ||
            mesh_triangles != nullptr && mesh_triangle_count > 0 && max_len > 0.0f &&
            mesh_bvh_quantization != nullptr && mesh_bvh_quantization_count > 0 &&
            mesh_bvh_meshes != nullptr && mesh_bvh_mesh_count > 0 &&
            ray_max_bt.x < mesh_bvh_meshes[0].free_half_x && ray_min_bt.x > -mesh_bvh_meshes[0].free_half_x &&
            ray_max_bt.y < mesh_bvh_meshes[0].free_half_y && ray_min_bt.y > -mesh_bvh_meshes[0].free_half_y;
        int best_mesh = -1;
#if !defined(ROCKETGPU_USE_CUDA)
        if (!skip_meshes && !(max_len > 0.0f && HostRayClearOfMeshes(leaf_grid, mesh_triangles, ray_source_bt, ray_delta_bt))) {
#else
        if (!skip_meshes && !(max_len > 0.0f && DeviceRayClearOfMeshes(leaf_grid, mesh_triangles, ray_source_bt, ray_delta_bt))) {
#endif
#if !defined(ROCKETGPU_USE_CUDA) && (defined(__GNUC__) || defined(__clang__))
            if (!(!MeshLanes::kSplitHeaders && mesh_lanes.first == 0 && mesh_lanes.step == 1 &&
                  HostRayCastArenaMeshBlocks(mesh_triangles, mesh_triangle_count, mesh_bvh_quantization,
                                             mesh_bvh_quantization_count, mesh_bvh_meshes, mesh_bvh_mesh_count,
                                             mesh_bvh_nodes, mesh_bvh_node_count, ray_source_bt, ray_delta_bt,
                                             max_len, best, &best_mesh, leaf_grid, local_triangles, walk_hint)))
#endif
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
        if (!mesh_lanes.Leader()) {
            return best;
        }
        if (best.hit && best_mesh >= 0 && mesh_bvh_meshes != nullptr && mesh_bvh_mesh_count > 0) {
            const int pos = MeshLanes::kSplitHeaders ? (best_mesh >> 16) : best_mesh;
            neg_x_wins_mesh_tie = pos >= mesh_bvh_meshes[0].ray_neg_x_plane_pos;
        }
    } else if (!mesh_lanes.Leader()) {
        return best;
    }
    if (mesh_ray_cast_bt && mesh_bvh_meshes != nullptr && mesh_bvh_mesh_count > 0 &&
        ROCKETGPU_HOST_UNLIKELY(mesh_bvh_meshes[0].ray_neg_x_plane_pos < 0)) {
        return best;
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
ROCKETGPU_HD ROCKETGPU_HOST_INLINE ArenaRayHit CastArenaRayWithBtSource(
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
    const ArenaMeshTriangle* local_triangles = nullptr,
    const ArenaMeshLeafGrid* leaf_grid = nullptr) {
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
                               native_table, native_shift, mesh_lanes, leaf_grid, local_triangles);
}

ROCKETGPU_HOST_COLD ROCKETGPU_HD bool CastWheelRayAgainstCarHitboxGjkCast(
    const Vec3& ray_source_bt,
    const Vec3& ray_dir,
    float max_len,
    CarCRefT other_car,
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
    CarCSpanT cars_in_env,
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
        CarCRefT other = cars_in_env[i];
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

ROCKETGPU_HD void InitWheelStepState(WheelStepState& wheel, CarCRefT car, const CarConfig& cfg, int wheel_idx,
                                   const CarFrame* frame, float bone_z) {
    const bool front = wheel_idx < 2;
    const bool left = (wheel_idx % 2) != 0;
    Vec3 offset = front ? cfg.front_wheels_offset : cfg.back_wheels_offset;
    if (left) {
        offset.y = -offset.y;
    }
    offset.z = cfg.hitbox_offset.z;
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

ROCKETGPU_HD void InitWheelStepState(WheelStepState& wheel, CarCRefT car, const CarConfig& cfg, int wheel_idx,
                                   const CarFrame* frame = nullptr) {
    const bool front = wheel_idx < 2;
    const float published_rest = (front ? cfg.front_suspension_rest : cfg.back_suspension_rest) - 12.0f;
    const float bone_z = floorf((cfg.hitbox_offset.z - published_rest) * 100.0f + 0.5f) / 100.0f;
    InitWheelStepState(wheel, car, cfg, wheel_idx, frame, bone_z);
}

ROCKETGPU_HD WheelStepState MakeWheelStepState(CarCRefT car, const CarConfig& cfg, int wheel_idx,
                                             const CarFrame* frame, float bone_z) {
    WheelStepState wheel;
    InitWheelStepState(wheel, car, cfg, wheel_idx, frame, bone_z);
    return wheel;
}

ROCKETGPU_HD WheelStepState MakeWheelStepState(CarCRefT car, const CarConfig& cfg, int wheel_idx,
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

ROCKETGPU_HD void ComputeWheelRayVelocityTerms(CarCRefT car, const Vec3& car_up, const Vec3& hit_point_bt,
                                           WheelStepState& wheel);

#if !defined(__CUDA_ARCH__)
inline float WheelRayReachBt(const CarConfig& cfg) {
    const float oz = cfg.hitbox_offset.z;
    const float front_bz = floorf((oz - (cfg.front_suspension_rest - 12.0f)) * 100.0f + 0.5f) / 100.0f;
    const float back_bz = floorf((oz - (cfg.back_suspension_rest - 12.0f)) * 100.0f + 0.5f) / 100.0f;
    const Vec3 fo = cfg.front_wheels_offset, bo = cfg.back_wheels_offset;
    return (Max(sqrtf(fo.x * fo.x + fo.y * fo.y + oz * oz), sqrtf(bo.x * bo.x + bo.y * bo.y + oz * oz)) +
            Max(Abs(oz - front_bz), Abs(oz - back_bz)) + Max(cfg.front_wheel_radius, cfg.back_wheel_radius) + 14.5f) *
               (1.01f / 50.0f) +
           0.01f;
}
#endif

template <typename WheelRays = WheelRaysWhole>
ROCKETGPU_HD uint32_t WheelRayExcludedBodies(CarCRefT car, const CarConfig& cfg, const BulletSweepHistory* sweep,
                                             int self_car_idx, int car_count, WheelStepState* wheel_out = nullptr,
                                             int wheel_idx = 0, const float* cached_reach_bt = nullptr,
                                             const WheelRays& rays = WheelRays{}) {
    const int bodies = Min(car_count + 1, kBulletSweepMaxBodies);
    const uint32_t mask = (sweep != nullptr && bodies > 0)
        ? (sweep->sap_body_aabb_valid & ((1u << bodies) - 1u) & ~(1u << (self_car_idx + 1)))
        : 0u;
    if (mask == 0u) {
        if (wheel_out != nullptr) {
            InitWheelStepState(*wheel_out, car, cfg, wheel_idx);
        }
        return 0u;
    }
#if !defined(__CUDA_ARCH__) && (defined(__GNUC__) || defined(__clang__))
    if (wheel_out == nullptr) {
        const float reach = cached_reach_bt != nullptr ? *cached_reach_bt : WheelRayReachBt(cfg);
        const Vec3 p = car.pos_bt;
        const float lx = p.x - reach, hx = p.x + reach, ly = p.y - reach, hy = p.y + reach, lz = p.z - reach,
                    hz = p.z + reach;
        typedef float F4 __attribute__((vector_size(16)));
        typedef int32_t I4 __attribute__((vector_size(16)));
        const float inf = __builtin_inff();
        const F4 lo4 = {lx, ly, lz, -inf}, hi4 = {hx, hy, hz, inf};
        uint32_t near = 0u;
        for (int body = 0; body < bodies; ++body) {
            const float* a = sweep->sap_body_aabb[body];
            F4 mn, mx;
            memcpy(&mn, a, sizeof(F4));
            memcpy(&mx, a + 3, sizeof(F4));
            const I4 sep = (mx < lo4) | (hi4 < mn);
            const bool far = ((sep[0] | sep[1]) | (sep[2] | sep[3])) != 0;
            near |= static_cast<uint32_t>(!far) << body;
        }
        if ((near & mask) == 0u) {
            return mask;
        }
    }
#endif
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
    const float back_bone_z = (cfg.back_suspension_rest == cfg.front_suspension_rest)
        ? front_bone_z
        : (floorf((offset_z - (cfg.back_suspension_rest - 12.0f)) * 100.0f + 0.5f) / 100.0f);
    if (wheel_out != nullptr) {
        const bool w_front = wheel_idx < 2;
        const bool w_left = (wheel_idx % 2) != 0;
        if (car.set_wheel_normals_mask != 0) {
            Vec3 w_offset = w_front ? cfg.front_wheels_offset : cfg.back_wheels_offset;
            if (w_left) {
                w_offset.y = -w_offset.y;
            }
            w_offset.z = offset_z;
            wheel_out->hard_point_ws = car.pos + RotateViaBasis(car_forward, car_right, car_up, w_offset);
        }
        wheel_out->wheel_dir_ws = wheel_dir_ws;
        wheel_out->wheel_axle_ws = -car_right;
        wheel_out->wheel_radius = w_front ? cfg.front_wheel_radius : cfg.back_wheel_radius;
        wheel_out->suspension_rest_length = offset_z - (w_front ? front_bone_z : back_bone_z);
        wheel_out->max_suspension_travel = 14.5f;
        wheel_out->is_front_wheel = w_front ? 1 : 0;
        wheel_out->suspension_force_scale = w_front ? (36.0f - 0.25f) : (54.0f + 0.25f + 0.015f);
        wheel_out->contact_normal_ws = car_up;
    }
    const float front_len_bt =
        ((offset_z - front_bone_z) * kUUToBT + cfg.front_wheel_radius * kUUToBT) +
        ((14.5f - WheelSuspensionSubtraction()) * kUUToBT);
    const float back_len_bt =
        ((offset_z - back_bone_z) * kUUToBT + cfg.back_wheel_radius * kUUToBT) +
        ((14.5f - WheelSuspensionSubtraction()) * kUUToBT);
    Vec3 mn(0.0f, 0.0f, 0.0f), mx(0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (!WheelRays::kAllRays && i != wheel_idx) {
            continue;
        }
        Vec3 offset = (i < 2) ? cfg.front_wheels_offset : cfg.back_wheels_offset;
        if ((i % 2) != 0) {
            offset.y = -offset.y;
        }
        offset.z = offset_z;
        const Vec3 from = car.pos_bt + RotateViaBasis(car_forward, car_right, car_up, offset * kUUToBT);
        if (wheel_out != nullptr && i == wheel_idx) {
            wheel_out->ray_source_bt = from;
        }
        const Vec3 to = from + wheel_dir_ws * (i < 2 ? front_len_bt : back_len_bt);
        const Vec3 lo(Min(from.x, to.x), Min(from.y, to.y), Min(from.z, to.z));
        const Vec3 hi(Max(from.x, to.x), Max(from.y, to.y), Max(from.z, to.z));
        mn = (i == 0 || !WheelRays::kAllRays) ? lo : Vec3(Min(mn.x, lo.x), Min(mn.y, lo.y), Min(mn.z, lo.z));
        mx = (i == 0 || !WheelRays::kAllRays) ? hi : Vec3(Max(mx.x, hi.x), Max(mx.y, hi.y), Max(mx.z, hi.z));
    }
    rays.Merge(mn, mx);
    return rays.Excluded(sweep, mask, bodies, mn, mx);
}

#if !defined(ROCKETGPU_USE_CUDA)
ROCKETGPU_HD void CalWheelBoneZ(const CarConfig& cfg, float& front, float& back) {
    front = floorf((cfg.hitbox_offset.z - (cfg.front_suspension_rest - 12.0f)) * 100.0f + 0.5f) / 100.0f;
    back = (cfg.back_suspension_rest == cfg.front_suspension_rest)
        ? front
        : (floorf((cfg.hitbox_offset.z - (cfg.back_suspension_rest - 12.0f)) * 100.0f + 0.5f) / 100.0f);
}
#endif

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
ROCKETGPU_HD ROCKETGPU_HOST_INLINE void RayCastWheelAgainstArena(
    CarCRefT car,
    const CarConfig& cfg,
    WheelStepState& wheel,
    const ArenaMeshTriangle* mesh_triangles = nullptr,
    int32_t mesh_triangle_count = 0,
    CarCSpanT cars_in_env = nullptr,
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
    uint8_t arena_static = 1,
    const uint32_t* native_table = nullptr, int native_shift = 0,
    int32_t no_response_mask = -1,
    float cal_min_hit_dist_sq_bt = 0.0f,
    const MeshLanes& mesh_lanes = MeshLanes{},
    const ArenaMeshTriangle* local_triangles = nullptr,
    uint32_t excluded_bodies = 0u,
    const ArenaMeshLeafGrid* leaf_grid = nullptr) {
    (void)cfg;
    const float real_ray_length = wheel.suspension_rest_length + wheel.max_suspension_travel + wheel.wheel_radius - WheelSuspensionSubtraction();

    const float real_ray_length_bt = WheelRealRayLengthBt(wheel);

#if !defined(ROCKETGPU_USE_CUDA)
    if constexpr (std::is_same<MeshLanes, ArenaRayMeshOneLane>::value) {
#else
    {
#endif
        const int active_bits = Min(cars_in_env_count, kBulletSweepMaxBodies - 1);
        const uint32_t dyn_mask =
            ((active_bits > 0 ? ((1u << (active_bits + 1)) - 1u) : 1u)) &
            ~(self_car_idx >= 0 && self_car_idx + 1 < kBulletSweepMaxBodies ? (1u << (self_car_idx + 1)) : 0u);
        if (arena_static == 1 && mesh_bvh_nodes != nullptr && mesh_bvh_node_count > 0 && mesh_triangles != nullptr &&
            real_ray_length > 0.0f && (excluded_bodies & dyn_mask) == dyn_mask) {
            const float max_len_bt = real_ray_length_bt > 0.0f ? real_ray_length_bt : real_ray_length * (1.0f / 50.0f);
            const Vec3 src = wheel.ray_source_bt;
            const Vec3 delta = wheel.wheel_dir_ws * max_len_bt;
            const Vec3 to = src + delta;
            const Vec3 lo(Min(src.x, to.x), Min(src.y, to.y), Min(src.z, to.z));
            const Vec3 hi(Max(src.x, to.x), Max(src.y, to.y), Max(src.z, to.z));
            if (lo.x > -81.9f && hi.x < 81.9f && hi.z < 40.8f && lo.y > -200.0f && hi.y < 200.0f && lo.z > -20.0f &&
#if !defined(ROCKETGPU_USE_CUDA)
                HostRayClearOfMeshes(leaf_grid, mesh_triangles, src, delta)) {
#else
                DeviceRayClearOfMeshes(leaf_grid, mesh_triangles, src, delta)) {
#endif
                bool floor_hit = false;
                Vec3 point_bt;
                if (lo.z <= 0.0f) {
                    constexpr float kFltEpsilon = 1.1920929e-07f;
                    const Vec3 d = to - src;
                    const Vec3 off = Vec3(0.0f, 0.0f, 0.0f) - src;
                    const float denom = -(d.y * 0.0f + d.x * 0.0f + d.z * 1.0f);
                    const float numer = -(off.y * 0.0f + off.x * 0.0f + off.z * 1.0f);
                    if (kFltEpsilon < denom && kFltEpsilon < numer && numer <= denom) {
                        const float fraction = numer / denom;
                        if (fraction < 1.0f && !(fraction < 0.0f)) {
                            const float cal_min_fraction_sq = cal_min_hit_dist_sq_bt > 0.0f
                                ? cal_min_hit_dist_sq_bt / LengthSq((src + delta) - src) : 0.0f;
                            if (!(cal_min_fraction_sq > 0.0f && fraction * fraction < cal_min_fraction_sq)) {
                                const float one_minus_fraction = 1.0f - fraction;
                                point_bt = Vec3(src.x * one_minus_fraction + to.x * fraction,
                                                src.y * one_minus_fraction + to.y * fraction,
                                                src.z * one_minus_fraction + to.z * fraction);
                                floor_hit = true;
                            }
                        }
                    }
                }
                if (floor_hit) {
                    const Vec3 car_up = -wheel.wheel_dir_ws;
                    wheel.contact_point_bt = point_bt;
                    wheel.contact_normal_ws = Vec3(0.0f, 0.0f, 1.0f);
                    wheel.is_in_contact = 1;
                    wheel.is_in_contact_with_world = 1;
                    wheel.ground_kind = ROCKETGPU_ARENA_RAY_HIT_STATIC_PLANE;
                    wheel.ground_car_idx = -1;
                    wheel.ground_static_id = 100;
                    const float trace_len_bt = Dot(wheel.ray_source_bt - point_bt, car_up);
                    const float radius_bt = wheel.wheel_radius * UnitsUUToBT();
                    wheel.wheel_trace_len_bt = trace_len_bt;
                    wheel.suspension_len_bt = trace_len_bt - radius_bt;
                    ComputeWheelRayVelocityTerms(car, car_up, point_bt, wheel);
                } else {
                    SetWheelRayMiss(wheel);
                }
                return;
            }
        }
    }

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
        real_ray_length_bt, native_table, native_shift, cal_min_hit_dist_sq_bt, mesh_lanes, local_triangles, leaf_grid);

    bool hit_world = hit.hit != 0;
    int32_t ground_car_idx = -1;
    const int active_car_bits = Min(cars_in_env_count, kBulletSweepMaxBodies - 1);
    const uint32_t all_dyn_mask =
        ((active_car_bits > 0 ? ((1u << (active_car_bits + 1)) - 1u) : 1u)) &
        ~(self_car_idx >= 0 && self_car_idx + 1 < kBulletSweepMaxBodies ? (1u << (self_car_idx + 1)) : 0u);
    if ((excluded_bodies & all_dyn_mask) != all_dyn_mask) {
        const auto cal_too_close = [&](const ArenaRayHit& h) {
            return cal_min_hit_dist_sq_bt > 0.0f && h.hit && LengthSq(h.point_bt - wheel.ray_source_bt) < cal_min_hit_dist_sq_bt;
        };
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
        if (!mesh_lanes.Leader()) {
            return;
        }
        if (cal_too_close(car_hit)) {
            car_hit = ArenaRayHit{};
        }
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
    } else if (!mesh_lanes.Leader()) {
        return;
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

ROCKETGPU_HD void ComputeWheelRayVelocityTerms(CarCRefT car, const Vec3& car_up, const Vec3& hit_point_bt,
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
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (wheels[i].is_in_contact) {
            sum += wheels[i].contact_normal_ws;
        }
    }
    return UpwardsDirFromWheelNormalSum(sum);
}

ROCKETGPU_HD float QuantTruncOffsetV3(float x, float scale);

#if defined(ROCKETGPU_HOST_LANES)
inline lanes::F4 QuantTruncOffsetV3Lanes(lanes::F4 x, float scale) {
    using lanes::F4;
    using lanes::Splat;
    const float inv_scale = 1.0f / scale;
    const F4 k = __builtin_convertvector(__builtin_convertvector(x * Splat(scale), lanes::I4), F4);
    const F4 rounded = k * Splat(inv_scale);
    const float offset_mag = 0.1f * inv_scale;
    const F4 sgn = lanes::Select(rounded == Splat(0.0f), Splat(0.0f),
                                 lanes::Select(rounded > Splat(0.0f), Splat(1.0f), Splat(-1.0f)));
    return rounded + sgn * Splat(offset_mag);
}
#endif

struct CarRbSnapshot {
    Quat rot;
    Vec3 vel_uu;
};

ROCKETGPU_HD CarRbSnapshot TickStartCarRbSnapshot(CarCRefT car) {
    return CarRbSnapshot{car.rot, car.vel};
}

ROCKETGPU_HD CarRbSnapshot FreshCarRbSnapshot(CarCRefT car, const CarFrame* frame, float max_speed) {
    Vec3 car_forward(1.0f, 0.0f, 0.0f);
    Vec3 car_right(0.0f, 1.0f, 0.0f);
    Vec3 car_up(0.0f, 0.0f, 1.0f);
    CarBasisColumns(frame, car, car_forward, car_right, car_up);
#if defined(ROCKETGPU_HOST_LANES)
    using lanes::F4;
    using lanes::Splat;
    F4 v = lanes::Load(car.vel_bt) * Splat(50.0f);
    const F4 v2 = v * v;
    const float len2 = (v2[1] + v2[0]) + v2[2];
    if (max_speed > 0.0f && max_speed * max_speed < len2) {
        const float inv = 1.0f / sqrtf(len2);
        v = Splat(max_speed) * (Splat(inv) * v);
    }
    const Vec3 snap_vel = lanes::ToVec3(QuantTruncOffsetV3Lanes(v, 100.0f));
#else
    Vec3 v = car.vel_bt * 50.0f;
    const float len2 = (v.y * v.y + v.x * v.x) + v.z * v.z;
    if (max_speed > 0.0f && max_speed * max_speed < len2) {
        const float inv = 1.0f / sqrtf(len2);
        v = Vec3(max_speed * (inv * v.x), max_speed * (inv * v.y), max_speed * (inv * v.z));
    }
    const Vec3 snap_vel(QuantTruncOffsetV3(v.x, 100.0f), QuantTruncOffsetV3(v.y, 100.0f), QuantTruncOffsetV3(v.z, 100.0f));
#endif
#if !defined(ROCKETGPU_USE_CUDA)
    if (frame != nullptr && frame->rb_snap != nullptr && frame->rb_snap->valid) {
        const CarRbSnapMemo& m = *frame->rb_snap;
        static_assert(offsetof(CarFrame, right) == offsetof(CarFrame, fwd) + 12 &&
                          offsetof(CarFrame, up) == offsetof(CarFrame, fwd) + 24 &&
                          offsetof(CarRbSnapMemo, right) == offsetof(CarRbSnapMemo, fwd) + 12 &&
                          offsetof(CarRbSnapMemo, up) == offsetof(CarRbSnapMemo, fwd) + 24,
                      "contiguous basis blocks");
        const char* const a = reinterpret_cast<const char*>(frame) + offsetof(CarFrame, fwd);
        const char* const b = reinterpret_cast<const char*>(&m) + offsetof(CarRbSnapMemo, fwd);
        uint64_t aw[4], bw[4];
        uint32_t a_tail, b_tail;
        std::memcpy(aw, a, 32);
        std::memcpy(bw, b, 32);
        std::memcpy(&a_tail, a + 32, 4);
        std::memcpy(&b_tail, b + 32, 4);
        if (((aw[0] ^ bw[0]) | (aw[1] ^ bw[1]) | (aw[2] ^ bw[2]) | (aw[3] ^ bw[3]) | (a_tail ^ b_tail)) == 0u) {
            return CarRbSnapshot{m.rot, snap_vel};
        }
    }
#endif
    return CarRbSnapshot{GameRbStateQuantizeQuat(GameGetRotationScalar(car_forward, car_right, car_up)), snap_vel};
}

ROCKETGPU_HD float SnapshotForwardSpeed(const CarRbSnapshot& snapshot) {
    const Quat& q = snapshot.rot;
#if defined(ROCKETGPU_HOST_LANES)
    using lanes::F4;
    using lanes::Splat;
    const float w2_l = q.w + q.w;
    const float s_l = q.w * q.w - ((q.y * q.y + q.x * q.x) + q.z * q.z);
    float d_l = (1.0f * q.x + 0.0f * q.y) + 0.0f * q.z;
    d_l = d_l + d_l;
    const F4 qv{q.x, q.y, q.z, 0.0f};
    const F4 q_yzx = lanes::Turn<1, 2, 0>(qv), q_zxy = lanes::Turn<2, 0, 1>(qv);
    const F4 a = F4{0.0f, 1.0f, 0.0f, 0.0f} * q_yzx - F4{0.0f, 0.0f, 1.0f, 0.0f} * q_zxy;
    const F4 fwd_l = (a * Splat(w2_l) + Splat(s_l) * F4{1.0f, 0.0f, 0.0f, 0.0f}) + Splat(d_l) * qv;
    const F4 p = fwd_l * lanes::Load(snapshot.vel_uu);
    return (p[0] + p[1]) + p[2];
#else
    const float w2 = q.w + q.w;
    const float s = q.w * q.w - ((q.y * q.y + q.x * q.x) + q.z * q.z);
    float d = (1.0f * q.x + 0.0f * q.y) + 0.0f * q.z;
    d = d + d;
    const Vec3 fwd(((0.0f * q.y - 0.0f * q.z) * w2 + s * 1.0f) + d * q.x,
                   ((1.0f * q.z - 0.0f * q.x) * w2 + s * 0.0f) + d * q.y,
                   ((0.0f * q.x - 1.0f * q.y) * w2 + s * 0.0f) + d * q.z);
    const Vec3& v = snapshot.vel_uu;
    return (fwd.x * v.x + fwd.y * v.y) + fwd.z * v.z;
#endif
}

ROCKETGPU_HD float WheelForwardSpeed(CarCRefT car) {
    return SnapshotForwardSpeed(CarRbSnapshot{car.rot, car.vel});
}

ROCKETGPU_HD void ApplyAutoRollReference(
    CarRefT car,
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
#if defined(ROCKETGPU_HOST_LANES)
        QueueFirstForceAndTorqueBt(wheel_runtime_state, autoroll_force_bt,
                                   MulInertiaWorldBt(frame, car, cfg, mutator.car_mass, t_game), frame);
#else
        Vec3 ar_i0, ar_i1, ar_i2;
        BuildInertiaTensorWorldBtInverse(frame, car, cfg, mutator.car_mass, ar_i0, ar_i1, ar_i2);
        QueueFirstForceAndTorqueBt(wheel_runtime_state, autoroll_force_bt, Mul3x3Vec(ar_i0, ar_i1, ar_i2, t_game),
                                   frame);
#endif
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
    CarCRefT car,
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

ROCKETGPU_HD void CombineWheelImpulses(CarRefT car, const WheelImpulseOut out[4]) {
    Vec3 vel_bt = car.vel_bt;
    Vec3 ang_vel = car.ang_vel;
    bool any = false;
#pragma unroll
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
    CarCRefT car,
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

ROCKETGPU_HD BilateralBodyBt MakeBilateralCarBodyBt(CarCRefT car, const CarConfig& cfg, float mass,
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
    CarCRefT car,
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
    const Vec3 friction_impulse_bt(roll * wheel.cal_forward_dir.x + side * wheel.cal_axle_dir.x,
                                   roll * wheel.cal_forward_dir.y + side * wheel.cal_axle_dir.y,
                                   roll * wheel.cal_forward_dir.z + side * wheel.cal_axle_dir.z);
    if (LengthSq(friction_impulse_bt) == 0.0f) {
        return out;
    }

    const Vec3& car_up = body1.world2_row2;
    const Vec3 friction_rel_bt = wheel.contact_point_bt - car.pos_bt;
    Vec3 friction_wheel_rel_pos_bt = friction_rel_bt - car_up * Dot(car_up, friction_rel_bt);
    if (cfg.three_wheels) {
        friction_wheel_rel_pos_bt = friction_wheel_rel_pos_bt - car_up * (4.0f * 0.02f);
    }
    return WheelImpulseAtPointBt(car, cfg, mutator.car_mass, friction_impulse_bt, friction_wheel_rel_pos_bt, frame);
}

#if !defined(ROCKETGPU_USE_CUDA) && (defined(__GNUC__) || defined(__clang__)) && !defined(ROCKETGPU_NO_WHEEL_X4)
#define ROCKETGPU_WHEEL_X4 1
#else
#define ROCKETGPU_WHEEL_X4 0
#endif

#if ROCKETGPU_WHEEL_X4
typedef float WheelF4 __attribute__((vector_size(16)));
typedef int32_t WheelI4 __attribute__((vector_size(16)));

ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE WheelF4 WheelF4Splat(float v) {
    return WheelF4{v, v, v, v};
}

ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE WheelF4 WheelF4Sel(WheelI4 m, WheelF4 a, WheelF4 b) {
    return (WheelF4)((m & (WheelI4)a) | (~m & (WheelI4)b));
}

template <int A, int B, int C, int D>
ROCKETGPU_ALWAYS_INLINE WheelF4 WheelF4Shuffle(WheelF4 a, WheelF4 b) {
#if defined(__clang__)
    return __builtin_shufflevector(a, b, A, B, C, D);
#else
    return __builtin_shuffle(a, b, WheelI4{A, B, C, D});
#endif
}

struct WheelImpulseX4 {
    WheelF4 dvx, dvy, dvz, dwx, dwy, dwz;
};

struct WheelsX4 {
    WheelF4 src_x, src_y, src_z;
    WheelF4 point_x, point_y, point_z;
    WheelF4 normal_x, normal_y, normal_z;
    WheelF4 susp_len_bt;
    WheelF4 rel_vel_bt;
    WheelF4 rel_vel;
    WheelF4 clipped;
    WheelF4 extra_pushback;
    WheelF4 rest_len;
    WheelF4 force_scale;
    WheelF4 axle_x, axle_y, axle_z;
    WheelF4 fwd_x, fwd_y, fwd_z;
    WheelF4 rolling, lat, lng;
    WheelI4 contact;
    WheelI4 world;
    WheelI4 static_id;
    float friction_scale;
    Vec3 dir;
    Vec3 axle;
};

ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE bool WheelI4Any(WheelI4 m) {
    return (m[0] | m[1] | m[2] | m[3]) != 0;
}

inline void WheelsX4FromStates(WheelsX4& w, const WheelStepState* s) {
#define ROCKETGPU_WHEEL_F4(field) WheelF4{s[0].field, s[1].field, s[2].field, s[3].field}
#define ROCKETGPU_WHEEL_I4(flag) WheelI4{s[0].flag ? -1 : 0, s[1].flag ? -1 : 0, s[2].flag ? -1 : 0, s[3].flag ? -1 : 0}
    w.src_x = ROCKETGPU_WHEEL_F4(ray_source_bt.x);
    w.src_y = ROCKETGPU_WHEEL_F4(ray_source_bt.y);
    w.src_z = ROCKETGPU_WHEEL_F4(ray_source_bt.z);
    w.point_x = ROCKETGPU_WHEEL_F4(contact_point_bt.x);
    w.point_y = ROCKETGPU_WHEEL_F4(contact_point_bt.y);
    w.point_z = ROCKETGPU_WHEEL_F4(contact_point_bt.z);
    w.normal_x = ROCKETGPU_WHEEL_F4(contact_normal_ws.x);
    w.normal_y = ROCKETGPU_WHEEL_F4(contact_normal_ws.y);
    w.normal_z = ROCKETGPU_WHEEL_F4(contact_normal_ws.z);
    w.susp_len_bt = ROCKETGPU_WHEEL_F4(suspension_len_bt);
    w.rel_vel_bt = ROCKETGPU_WHEEL_F4(suspension_relative_velocity_bt);
    w.rel_vel = ROCKETGPU_WHEEL_F4(suspension_relative_velocity);
    w.clipped = ROCKETGPU_WHEEL_F4(clipped_inv_contact_dot_suspension);
    w.extra_pushback = ROCKETGPU_WHEEL_F4(extra_pushback);
    w.rest_len = ROCKETGPU_WHEEL_F4(suspension_rest_length);
    w.force_scale = ROCKETGPU_WHEEL_F4(suspension_force_scale);
    w.contact = ROCKETGPU_WHEEL_I4(is_in_contact);
    w.world = ROCKETGPU_WHEEL_I4(is_in_contact_with_world);
    w.static_id = WheelI4{s[0].ground_static_id, s[1].ground_static_id, s[2].ground_static_id, s[3].ground_static_id};
    w.dir = s[0].wheel_dir_ws;
    w.axle = s[0].wheel_axle_ws;
#undef ROCKETGPU_WHEEL_I4
#undef ROCKETGPU_WHEEL_F4
}

ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE WheelImpulseX4 WheelImpulseAtPointBtX4(
    const CarFrame& f, float official_inv_mass,
    WheelF4 ix, WheelF4 iy, WheelF4 iz, WheelF4 rx, WheelF4 ry, WheelF4 rz) {
    WheelImpulseX4 o;
    const WheelF4 inv_mass = WheelF4Splat(official_inv_mass);
    o.dvx = ix * inv_mass;
    o.dvy = iy * inv_mass;
    o.dvz = iz * inv_mass;
    const WheelF4 tx = ry * iz - rz * iy;
    const WheelF4 ty = rz * ix - rx * iz;
    const WheelF4 tz = rx * iy - ry * ix;
    const Vec3& r0 = f.inv_world_row0_bt;
    const Vec3& r1 = f.inv_world_row1_bt;
    const Vec3& r2 = f.inv_world_row2_bt;
    o.dwx = WheelF4Splat(r0.x) * tx + WheelF4Splat(r0.y) * ty + WheelF4Splat(r0.z) * tz;
    o.dwy = WheelF4Splat(r1.x) * tx + WheelF4Splat(r1.y) * ty + WheelF4Splat(r1.z) * tz;
    o.dwz = WheelF4Splat(r2.x) * tx + WheelF4Splat(r2.y) * ty + WheelF4Splat(r2.z) * tz;
    return o;
}

ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE void CombineWheelImpulsesX4(CarRefT car, WheelI4 applied, const WheelImpulseX4& o) {
    if (!(applied[0] | applied[1] | applied[2] | applied[3])) {
        return;
    }
    Vec3 vel_bt = car.vel_bt;
    Vec3 ang_vel = car.ang_vel;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int i = k ^ 1;
        if (!applied[i]) {
            continue;
        }
        vel_bt = vel_bt + Vec3(o.dvx[i], o.dvy[i], o.dvz[i]);
        ang_vel += Vec3(o.dwx[i], o.dwy[i], o.dwz[i]);
    }
    SetCarLinearVelocityBt(car, vel_bt);
    car.vel_touched_solve_stage = 1;
    car.ang_vel = ang_vel;
}

ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE void ApplyWheelSuspensionAndFrictionX4(
    CarRefT car, const CarConfig& cfg, const MutatorConfig& mutator, float dt,
    const WheelsX4& w, const CarFrame& f) {
    const WheelI4 contact = w.contact;
    const WheelF4 zero = WheelF4Splat(0.0f);
    const float official_inv_mass = 1.0f / mutator.car_mass;

    const WheelF4 force_bt =
        ((w.rest_len * WheelF4Splat(UnitsUUToBT()) -
          w.susp_len_bt) *
         WheelF4Splat(kWheelSuspensionStiffness)) *
        w.clipped;
    const WheelF4 damping = WheelF4Sel(w.rel_vel < zero,
                                       WheelF4Splat(kWheelDampingCompression),
                                       WheelF4Splat(kWheelDampingRelaxation));
    WheelF4 suspension_force_bt =
        (force_bt - damping * w.rel_vel_bt) *
        w.force_scale;
    suspension_force_bt = WheelF4Sel(suspension_force_bt < zero, zero, suspension_force_bt);
    const WheelI4 susp_applied = contact & (suspension_force_bt * WheelF4Splat(UnitsBTToUU()) > zero);

    const WheelF4 rx = w.point_x - WheelF4Splat(car.pos_bt.x);
    const WheelF4 ry = w.point_y - WheelF4Splat(car.pos_bt.y);
    const WheelF4 rz = w.point_z - WheelF4Splat(car.pos_bt.z);
    const WheelF4 base_force_scale_bt = (suspension_force_bt * WheelF4Splat(dt)) + w.extra_pushback;
    CombineWheelImpulsesX4(
        car, susp_applied,
        WheelImpulseAtPointBtX4(f, official_inv_mass,
                                w.normal_x * base_force_scale_bt,
                                w.normal_y * base_force_scale_bt,
                                w.normal_z * base_force_scale_bt, rx, ry, rz));

    const Vec3 lin = car.vel_bt;
    const Vec3 ang = car.ang_vel;
    const float body_inv_mass = mutator.car_mass != 0.0f ? 1.0f / mutator.car_mass : 0.0f;
    const WheelF4 ax = w.axle_x;
    const WheelF4 ay = w.axle_y;
    const WheelF4 az = w.axle_z;
    const WheelF4 vx = WheelF4Splat(lin.x) + (WheelF4Splat(ang.y) * rz - WheelF4Splat(ang.z) * ry);
    const WheelF4 vy = WheelF4Splat(lin.y) + (WheelF4Splat(ang.z) * rx - WheelF4Splat(ang.x) * rz);
    const WheelF4 vz = WheelF4Splat(lin.z) + (WheelF4Splat(ang.x) * ry - WheelF4Splat(ang.y) * rx);
    const WheelF4 cx = ry * az - rz * ay;
    const WheelF4 cy = rz * ax - rx * az;
    const WheelF4 cz = rx * ay - ry * ax;
    const WheelF4 ajx = WheelF4Splat(f.fwd.x) * cx + WheelF4Splat(f.fwd.y) * cy + WheelF4Splat(f.fwd.z) * cz;
    const WheelF4 ajy = WheelF4Splat(f.right.x) * cx + WheelF4Splat(f.right.y) * cy + WheelF4Splat(f.right.z) * cz;
    const WheelF4 ajz = WheelF4Splat(f.up.x) * cx + WheelF4Splat(f.up.y) * cy + WheelF4Splat(f.up.z) * cz;
    const Vec3& iil = f.inv_inertia_local_bt;
    const WheelF4 a_diag = WheelF4Splat(body_inv_mass) +
        ((WheelF4Splat(iil.x) * ajx) * ajx + (WheelF4Splat(iil.y) * ajy) * ajy + (WheelF4Splat(iil.z) * ajz) * ajz);
    const WheelF4 out_rel_vel_bt = ax * vx + ay * vy + az * vz;
    const WheelI4 side_active = contact & ~(a_diag <= zero);
    const WheelF4 side_denom = WheelF4Sel(side_active, a_diag, WheelF4Splat(1.0f));
    const WheelF4 side_impulse = WheelF4Sel(side_active,
                                            (WheelF4Splat(-0.2f) * out_rel_vel_bt) * (WheelF4Splat(1.0f) / side_denom), zero);
    const WheelF4 roll = w.rolling * w.lng;
    const WheelF4 side = (side_impulse * (WheelF4Splat(dt) * WheelF4Splat(w.friction_scale))) *
                         w.lat;
    const WheelF4 fix = roll * w.fwd_x + side * ax;
    const WheelF4 fiy = roll * w.fwd_y + side * ay;
    const WheelF4 fiz = roll * w.fwd_z + side * az;
    const WheelI4 fric_applied = contact & (fix * fix + fiy * fiy + fiz * fiz != zero);

    const Vec3& up = f.up;
    const WheelF4 upx = WheelF4Splat(up.x), upy = WheelF4Splat(up.y), upz = WheelF4Splat(up.z);
    const WheelF4 d = upx * rx + upy * ry + upz * rz;
    WheelF4 wrx = rx - upx * d;
    WheelF4 wry = ry - upy * d;
    WheelF4 wrz = rz - upz * d;
    if (cfg.three_wheels) {
        const float s = 4.0f * 0.02f;
        wrx = wrx - WheelF4Splat(up.x * s);
        wry = wry - WheelF4Splat(up.y * s);
        wrz = wrz - WheelF4Splat(up.z * s);
    }
    CombineWheelImpulsesX4(car, fric_applied,
                           WheelImpulseAtPointBtX4(f, official_inv_mass, fix, fiy, fiz, wrx, wry, wrz));
}
#endif

ROCKETGPU_HD ROCKETGPU_HOST_INLINE void ApplyWheelSuspensionAndFriction(
    CarRefT car,
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
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[i] = WheelSuspensionPhase(car, cfg, mutator, wheels[i], dt, frame);
    }
    CombineWheelImpulses(car, out);
    const BilateralBodyBt body1 = MakeBilateralCarBodyBt(car, cfg, mutator.car_mass, frame);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[i] = WheelFrictionPhase(car, cfg, mutator, body1, wheels[i], dt, frame);
    }
    CombineWheelImpulses(car, out);
    (void)controls;
}

ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE Vec3 ComputeWheelSteeredRightCol(
    const WheelStepState& wheel,
    float steer_angle,
    bool native_sse_math = false) {
    const Vec3 up = -wheel.wheel_dir_ws;
    const Vec3 right = wheel.wheel_axle_ws;

    const float half_angle = steer_angle * 0.5f;
    const float sin_scale = (half_angle == 0.0f && LengthSq(up) > 0.0f)
        ? half_angle : (RocketGpuSinf(half_angle) / Length(up));
    const float cos_half = (half_angle == 0.0f) ? 1.0f : RocketGpuCosf(half_angle);
    const Quat steering_rot(
        cos_half, up.x * sin_scale, up.y * sin_scale, up.z * sin_scale);
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

ROCKETGPU_HD bool SetWheelNormalIsOnOtherCar(CarCRefT car, const CarConfig& cfg, const Vec3& hard_point,
                                            const Vec3& normal, CarCSpanT cars, const CarConfig* cfgs,
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

ROCKETGPU_HD int SetStateWorldWheelNormals(CarCRefT car, const CarConfig& cfg, const WheelStepState* wheels,
                                           const BallState* ball, float ball_radius, CarCSpanT cars,
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
    CarCRefT car,
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

#if !defined(ROCKETGPU_USE_CUDA)
#if ROCKETGPU_WHEEL_X4
inline void InitWheelsX4(WheelsX4& w, CarCRefT car, const CarFrame* frame, const CarShapeCacheEntry& shape) {
    Vec3 car_forward(1.0f, 0.0f, 0.0f);
    Vec3 car_right(0.0f, 1.0f, 0.0f);
    Vec3 car_up(0.0f, 0.0f, 1.0f);
    if (car.basis_authoritative) {
        CarBasisColumns(frame, car, car_forward, car_right, car_up);
    } else {
        GetOfficialUploadBasisVectors(car.rot, car_forward, car_right, car_up);
    }
    WheelF4 ox, oy, oz;
    memcpy(&ox, shape.wheel_offset_bt_x, sizeof(WheelF4));
    memcpy(&oy, shape.wheel_offset_bt_y, sizeof(WheelF4));
    memcpy(&oz, shape.wheel_offset_bt_z, sizeof(WheelF4));
    const WheelF4 rsx = WheelF4Splat(car.pos_bt.x) +
        ((WheelF4Splat(car_forward.x) * ox + WheelF4Splat(car_right.x) * oy) + WheelF4Splat(car_up.x) * oz);
    const WheelF4 rsy = WheelF4Splat(car.pos_bt.y) +
        ((WheelF4Splat(car_forward.y) * ox + WheelF4Splat(car_right.y) * oy) + WheelF4Splat(car_up.y) * oz);
    const WheelF4 rsz = WheelF4Splat(car.pos_bt.z) +
        ((WheelF4Splat(car_forward.z) * ox + WheelF4Splat(car_right.z) * oy) + WheelF4Splat(car_up.z) * oz);
    w.src_x = rsx;
    w.src_y = rsy;
    w.src_z = rsz;
    memcpy(&w.rest_len, shape.wheel_rest_len, sizeof(WheelF4));
    w.dir = -car_up;
    w.axle = -car_right;
}
#else
inline void InitWheelStepStates4(WheelStepState* wheels, CarCRefT car, const CarConfig& cfg,
                                 const CarFrame* frame, const CarShapeCacheEntry& shape) {
    Vec3 car_forward(1.0f, 0.0f, 0.0f);
    Vec3 car_right(0.0f, 1.0f, 0.0f);
    Vec3 car_up(0.0f, 0.0f, 1.0f);
    if (car.basis_authoritative) {
        CarBasisColumns(frame, car, car_forward, car_right, car_up);
    } else {
        GetOfficialUploadBasisVectors(car.rot, car_forward, car_right, car_up);
    }
    for (int i = 0; i < 4; ++i) {
        const bool front = i < 2;
        WheelStepState& wheel = *new (&wheels[i]) WheelStepState;
        if (car.set_wheel_normals_mask != 0) {
            Vec3 offset = front ? cfg.front_wheels_offset : cfg.back_wheels_offset;
            if ((i % 2) != 0) {
                offset.y = -offset.y;
            }
            offset.z = cfg.hitbox_offset.z;
            wheel.hard_point_ws = car.pos + RotateViaBasis(car_forward, car_right, car_up, offset);
        }
        wheel.ray_source_bt = car.pos_bt + RotateViaBasis(car_forward, car_right, car_up,
                                                          Vec3(shape.wheel_offset_bt_x[i], shape.wheel_offset_bt_y[i],
                                                               shape.wheel_offset_bt_z[i]));
        wheel.wheel_dir_ws = -car_up;
        wheel.wheel_axle_ws = -car_right;
        wheel.wheel_radius = front ? cfg.front_wheel_radius : cfg.back_wheel_radius;
        wheel.suspension_rest_length = shape.wheel_rest_len[i];
        wheel.max_suspension_travel = 14.5f;
        wheel.is_front_wheel = front ? 1 : 0;
        wheel.suspension_force_scale = front ? (36.0f - 0.25f) : (54.0f + 0.25f + 0.015f);
        wheel.contact_normal_ws = -wheel.wheel_dir_ws;
    }
}
#endif
#endif

#if ROCKETGPU_WHEEL_X4
ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE WheelF4 WheelF4Abs(WheelF4 v) {
    return WheelF4Sel(v >= WheelF4Splat(0.0f), v, -v);
}

template <int I, int N>
ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE WheelF4 EvalInterpCurveLinearStepX4(
    const float (&in)[N], const float (&out)[N], WheelF4 x) {
    if constexpr (I < N) {
        const WheelF4 v = WheelF4Splat(out[I] - out[I - 1]) *
                              ((x - WheelF4Splat(in[I - 1])) / WheelF4Splat(in[I] - in[I - 1])) +
                          WheelF4Splat(out[I - 1]);
        return WheelF4Sel(WheelF4Splat(in[I]) > x, v, EvalInterpCurveLinearStepX4<I + 1, N>(in, out, x));
    } else {
        return WheelF4Splat(out[N - 1]);
    }
}

template <int N>
ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE WheelF4 EvalInterpCurveLinearX4(
    const float (&in)[N], const float (&out)[N], WheelF4 x) {
    static_assert(N >= 2, "curve needs two points");
    return WheelF4Sel(WheelF4Splat(in[0]) >= x, WheelF4Splat(out[0]),
                      WheelF4Sel(x >= WheelF4Splat(in[N - 1]), WheelF4Splat(out[N - 1]),
                                 EvalInterpCurveLinearStepX4<1, N>(in, out, x)));
}

ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE void WheelDriveSetupPhaseX4(
    CarCRefT car,
    const CarConfig& cfg,
    const WheelDriveShared& shared,
    WheelsX4& w,
    WheelRuntimeState* wheel_runtime_state,
    float dt) {
    const WheelI4 contact = w.contact;
    const WheelI4 front{-1, -1, 0, 0};
    const WheelF4 zero = WheelF4Splat(0.0f);
    const WheelF4 one = WheelF4Splat(1.0f);

    WheelF4 prev_lat_friction = zero;
    WheelF4 prev_long_friction = zero;
    if (wheel_runtime_state) {
        const WheelF4 steer = WheelF4Sel(front, WheelF4Splat(shared.next_steer_angle), zero);
        memcpy(&prev_lat_friction, wheel_runtime_state->lat_friction, sizeof(WheelF4));
        memcpy(&prev_long_friction, wheel_runtime_state->long_friction, sizeof(WheelF4));
        const WheelF4 engine = WheelF4Splat(shared.drive_engine_force);
        const WheelF4 brake = WheelF4Splat(shared.drive_brake_force);
        memcpy(wheel_runtime_state->engine_force, &engine, sizeof(WheelF4));
        memcpy(wheel_runtime_state->brake, &brake, sizeof(WheelF4));
        memcpy(wheel_runtime_state->steer_angle, &steer, sizeof(WheelF4));
    }
    if (!WheelI4Any(contact)) {
        return;
    }

    const Vec3& fr = shared.front_right_col;
    const Vec3& br = shared.back_right_col;
    const WheelF4 lx = WheelF4Sel(front, WheelF4Splat(fr.x), WheelF4Splat(br.x));
    const WheelF4 ly = WheelF4Sel(front, WheelF4Splat(fr.y), WheelF4Splat(br.y));
    const WheelF4 lz = WheelF4Sel(front, WheelF4Splat(fr.z), WheelF4Splat(br.z));
    const Vec3& ang = car.ang_vel;
    const Vec3& vel = car.vel_bt;
    const WheelF4 angx = WheelF4Splat(ang.x), angy = WheelF4Splat(ang.y), angz = WheelF4Splat(ang.z);

    const WheelF4 wdx = w.src_x - WheelF4Splat(car.pos_bt.x);
    const WheelF4 wdy = w.src_y - WheelF4Splat(car.pos_bt.y);
    const WheelF4 wdz = w.src_z - WheelF4Splat(car.pos_bt.z);
    const WheelF4 s50 = WheelF4Splat(50.0f);
    const WheelF4 cvx = ((angy * wdz - angz * wdy) + WheelF4Splat(vel.x)) * s50;
    const WheelF4 cvy = ((angz * wdx - angx * wdz) + WheelF4Splat(vel.y)) * s50;
    const WheelF4 cvz = ((angx * wdy - angy * wdx) + WheelF4Splat(vel.z)) * s50;
    const WheelF4 nx = w.normal_x;
    const WheelF4 ny = w.normal_y;
    const WheelF4 nz = w.normal_z;
    const WheelF4 base_friction = WheelF4Abs(cvx * lx + cvy * ly + cvz * lz);
    const WheelI4 friction_active = contact & (base_friction > WheelF4Splat(5.0f));
    WheelF4 friction_curve_input = zero;
    if (WheelI4Any(friction_active)) {
        const WheelF4 long_x = ly * nz - lz * ny;
        const WheelF4 long_y = lz * nx - lx * nz;
        const WheelF4 long_z = lx * ny - ly * nx;
        const WheelF4 longitudinal_friction = WheelF4Abs(cvx * long_x + cvy * long_y + cvz * long_z);
        const WheelF4 friction_denom = WheelF4Sel(friction_active, longitudinal_friction + base_friction, one);
        friction_curve_input = WheelF4Sel(friction_active, base_friction / friction_denom, zero);
    }

    static constexpr float kIn01[2] = {0.0f, 1.0f};
    static constexpr float kLatOut[2] = {1.0f, 0.2f};
    static constexpr float kLatThreeWheelOut[2] = {0.30f, 0.25f};
    static constexpr float kHandbrakeLongOut[2] = {0.5f, 0.9f};
    static constexpr float kNonStickyIn[3] = {0.0f, 0.707f, 1.0f};
    static constexpr float kNonStickyOut[3] = {0.1f, 0.5f, 1.0f};
    WheelF4 lat = cfg.three_wheels ? EvalInterpCurveLinearX4(kIn01, kLatThreeWheelOut, friction_curve_input)
                                   : EvalInterpCurveLinearX4(kIn01, kLatOut, friction_curve_input);
    WheelF4 lng;
    if (car.handbrake_val > 0.0f) {
        const WheelF4 hb = WheelF4Splat(car.handbrake_val);
        lat = lat * ((WheelF4Splat(EvalHandbrakeLatFrictionFactor(0.0f)) - one) * hb + one);
        lng = WheelF4Splat(EvalLongFriction(0.0f)) *
              ((EvalInterpCurveLinearX4(kIn01, kHandbrakeLongOut, friction_curve_input) - one) * hb + one);
    } else {
        lng = one;
    }
    if (shared.real_throttle == 0.0f || !(w.world[0] & w.world[1] & w.world[2] & w.world[3])) {
        const WheelI4 non_sticky = shared.real_throttle != 0.0f ? ~w.world : WheelI4{-1, -1, -1, -1};
        const WheelF4 ns = EvalInterpCurveLinearX4(kNonStickyIn, kNonStickyOut, nz);
        lat = WheelF4Sel(non_sticky, lat * ns, lat);
        lng = WheelF4Sel(non_sticky, lng * ns, lng);
    }
    lat = WheelF4Sel(contact, lat, prev_lat_friction);
    lng = WheelF4Sel(contact, lng, prev_long_friction);
    if (wheel_runtime_state) {
        memcpy(wheel_runtime_state->lat_friction, &lat, sizeof(WheelF4));
        memcpy(wheel_runtime_state->long_friction, &lng, sizeof(WheelF4));
    }

    const WheelF4 cx = ny * lz - nz * ly;
    const WheelF4 cy = nz * lx - nx * lz;
    const WheelF4 cz = nx * ly - ny * lx;
    const WheelF4 len_sq = WheelF4Sel(contact, cx * cx + cy * cy + cz * cz, one);
    const WheelF4 inv_len =
        one / WheelF4{SqrtScalar(len_sq[0]), SqrtScalar(len_sq[1]), SqrtScalar(len_sq[2]), SqrtScalar(len_sq[3])};
    const WheelF4 fx = cx * inv_len;
    const WheelF4 fy = cy * inv_len;
    const WheelF4 fz = cz * inv_len;

    WheelF4 rolling = zero;
    const float applied_engine_force = shared.drive_engine_force;
    const float applied_brake = shared.drive_brake_force;
    if (applied_engine_force == 0.0f) {
        if (applied_brake != 0.0f) {
            const WheelF4 rx = w.point_x - WheelF4Splat(car.pos_bt.x);
            const WheelF4 ry = w.point_y - WheelF4Splat(car.pos_bt.y);
            const WheelF4 rz = w.point_z - WheelF4Splat(car.pos_bt.z);
            const WheelF4 vx = WheelF4Splat(vel.x) + (angy * rz - angz * ry);
            const WheelF4 vy = WheelF4Splat(vel.y) + (angz * rx - angx * rz);
            const WheelF4 vz = WheelF4Splat(vel.z) + (angx * ry - angy * rx);
            WheelF4 rel_vel = vx * fx + vy * fy + vz * fz;
            if (dt > (1.0f / 80.0f)) {
                const float threshold = -(1.0f / (dt * 150.0f)) + 0.8f;
                rel_vel = WheelF4Sel(WheelF4Abs(rel_vel) < WheelF4Splat(threshold), zero, rel_vel);
            }
            const WheelF4 t = -rel_vel * WheelF4Splat(113.73963f);
            const WheelF4 lo = WheelF4Splat(-applied_brake);
            const WheelF4 hi = WheelF4Splat(applied_brake);
            rolling = WheelF4Sel(t < lo, lo, WheelF4Sel(t > hi, hi, t)) * WheelF4Splat(dt * shared.friction_scale);
        }
    } else {
        rolling = WheelF4Splat(-applied_engine_force * dt);
    }

    w.axle_x = lx;
    w.axle_y = ly;
    w.axle_z = lz;
    w.fwd_x = fx;
    w.fwd_y = fy;
    w.fwd_z = fz;
    w.rolling = rolling;
    w.lat = lat;
    w.lng = lng;
    w.friction_scale = shared.friction_scale;
}
#endif

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

ROCKETGPU_HD void SetWorldWheelPushback(CarCRefT car, const CarConfig& cfg, const MutatorConfig& mutator,
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
ROCKETGPU_HD ROCKETGPU_HOST_INLINE WheelContactOut WheelRayPhase(
    CarCRefT car,
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
    CarCSpanT cars_in_env,
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
    CarSpanT dyn_hit_cars,
    BallState* dyn_hit_ball,
    uint32_t* dyn_hit_pushed_mask,
    const CarFrame* frame = nullptr,
    uint32_t excluded_bodies = 0u,
    float precomputed_bone_z = 0.0f,
    bool has_precomputed_bone_z = false,
    const ArenaMeshLeafGrid* leaf_grid = nullptr) {
    const float extra_pushback = wheel_runtime_state ? wheel_runtime_state->extra_pushback[i] : 0.0f;
    const uint8_t stage_valid = staged_rays != nullptr ? staged_rays[i].valid : 0;
#if defined(__CUDA_ARCH__)
    WheelStepState made = has_precomputed_bone_z
        ? MakeWheelStepState(car, cfg, i, frame, precomputed_bone_z)
        : MakeWheelStepState(car, cfg, i, frame);
#else
#if !defined(ROCKETGPU_USE_CUDA)
    WheelStepState& made = wheel;
    if (frame == nullptr || !frame->wheels_pre_inited) {
        new (&wheel) WheelStepState;
#else
    WheelStepState& made = *new (&wheel) WheelStepState;
    {
#endif
        if (has_precomputed_bone_z) {
            InitWheelStepState(made, car, cfg, i, frame, precomputed_bone_z);
        } else {
            InitWheelStepState(made, car, cfg, i, frame);
        }
    }
#endif
    made.suspension_force_scale = made.is_front_wheel ? cal_front_scale : cal_back_scale;

    made.extra_pushback = extra_pushback;
    if constexpr (kStagedIndependent) {
        if (stage_valid == 2) {
            SetWheelRayMiss(made);
        } else {
            LoadWheelRayStage(made, staged_rays[i]);
        }
    } else {
        if (stage_valid == 2) {
            SetWheelRayMiss(made);
        } else if (stage_valid) {
            LoadWheelRayStage(made, staged_rays[i]);
        }
    }
#if defined(__CUDA_ARCH__)
    wheel = made;
#endif
    if constexpr (kStagedIndependent) {
        if (staged_rays_velocity_stale && wheel.is_in_contact) {
            ComputeWheelRayVelocityTerms(car, -wheel.wheel_dir_ws, wheel.contact_point_bt, wheel);
        }
    } else {
        if (stage_valid == 2) {
        } else if (stage_valid) {
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
            mutator.arena_collision, mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift,
            -1, cal_min_hit_dist_sq_bt, ArenaRayMeshOneLane{}, mutator.cal_arena_local_tris, excluded_bodies,
            leaf_grid);
    }
    if (wheel.is_in_contact) {
        wheel.extra_pushback = 0.0f;
    }
    if (wheel.is_in_contact_with_world) {
        SetWorldWheelPushback(car, cfg, mutator, wheel, frame);
    }
    if constexpr (!kStagedIndependent) {
        if (wheel.is_in_contact && !wheel.is_in_contact_with_world) {
            const Vec3 n_bt = wheel.contact_normal_ws;
            const Vec3 cp_bt = wheel.contact_point_bt;
            if (wheel.ground_kind == ROCKETGPU_ARENA_RAY_HIT_CAR && dyn_hit_cars != nullptr && car_cfgs_in_env != nullptr &&
                wheel.ground_car_idx >= 0 && wheel.ground_car_idx < cars_in_env_count &&
                wheel.ground_car_idx != self_car_idx) {
                CarRefT other = dyn_hit_cars[wheel.ground_car_idx];
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
    CarRefT car,
    WheelRuntimeState* wheel_runtime_state,
    const WheelContactOut out[4]) {
    WheelContactSummary sum;
#pragma unroll
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

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool in_contact = (out[i].flags & kWheelContactInContact) != 0;
        if (in_contact) {
            sum.contact_normal_sum_sim += out[i].normal;
        }
    }
#pragma unroll
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

#if !defined(ROCKETGPU_USE_CUDA)
#if ROCKETGPU_WHEEL_X4
__attribute__((always_inline)) inline bool WheelRaysFloorOnlyHostX4(
    CarCRefT car, const CarConfig& cfg, const MutatorConfig& mutator, WheelsX4& w,
    float cal_front_scale, float cal_back_scale, float cal_min_hit_dist_sq_bt, WheelRuntimeState* wheel_runtime_state,
    const ArenaMeshTriangle* mesh_triangles, int32_t mesh_triangle_count,
    const ArenaMeshBvhQuantization* mesh_bvh_quantization, int32_t mesh_bvh_quantization_count,
    const ArenaMeshBvhMeshInfo* mesh_bvh_meshes, int32_t mesh_bvh_mesh_count, const ArenaMeshBvhNode* mesh_bvh_nodes,
    int32_t mesh_bvh_node_count, const ArenaMeshBvhSubtreeInfo* mesh_bvh_subtrees, int32_t mesh_bvh_subtree_count,
    const CarFrame* frame, WheelContactOut* contact_out, const ArenaMeshLeafGrid* leaf_grid) {
    const CarShapeCacheEntry& shape = *frame->shape;
    if (!shape.wheel_ray_len_positive) {
        return false;
    }
    const HostStepTables* hint_tables = leaf_grid != nullptr ? leaf_grid->host_tables : nullptr;
    int32_t* hints = nullptr;
    if (hint_tables != nullptr && hint_tables->ray_hints != nullptr && wheel_runtime_state != nullptr) {
        const uintptr_t off =
            reinterpret_cast<uintptr_t>(wheel_runtime_state) - reinterpret_cast<uintptr_t>(hint_tables->ray_hint_base);
        const uintptr_t slot = off / sizeof(WheelRuntimeState);
        if (off % sizeof(WheelRuntimeState) == 0 && slot < static_cast<uintptr_t>(hint_tables->ray_hint_cars)) {
            hints = hint_tables->ray_hints + slot * 8;
            for (int i = 0; i < 4; ++i) {
                const int32_t cell = hints[2 * i], first = hints[2 * i + 1];
                if (cell >= 0 && cell < hint_tables->ray_cell_count) {
                    __builtin_prefetch(&hint_tables->ray_grid.cell_start[cell]);
                }
                if (first >= 0 && (first >> 2) < hint_tables->ray_block_count) {
                    const char* block = reinterpret_cast<const char*>(&hint_tables->ray_blocks[first >> 2]);
                    for (size_t line = 0; line < sizeof(ArenaMeshRayBlock); line += 64) {
                        __builtin_prefetch(block + line);
                    }
                }
            }
        }
    }
    const auto sel = [](WheelI4 m, WheelF4 a, WheelF4 b) { return WheelF4Sel(m, a, b); };
    const auto all = [](WheelI4 m) { return (m[0] & m[1] & m[2] & m[3]) != 0; };
    const WheelF4 sx = w.src_x, sy = w.src_y, sz = w.src_z;
    WheelF4 len;
    memcpy(&len, shape.wheel_ray_len_bt, sizeof(WheelF4));
    const Vec3 dir = w.dir;
    const WheelF4 dx = WheelF4Splat(dir.x) * len, dy = WheelF4Splat(dir.y) * len, dz = WheelF4Splat(dir.z) * len;
    const WheelF4 tx = sx + dx, ty = sy + dy, tz = sz + dz;
    const WheelF4 lox = sel(sx < tx, sx, tx), loy = sel(sy < ty, sy, ty), loz = sel(sz < tz, sz, tz);
    const WheelF4 hix = sel(sx > tx, sx, tx), hiy = sel(sy > ty, sy, ty), hiz = sel(sz > tz, sz, tz);
    WheelI4 floor_ok = (lox > WheelF4Splat(-81.9f)) & (hix < WheelF4Splat(81.9f)) & (hiz < WheelF4Splat(40.8f)) &
                       (loy > WheelF4Splat(-200.0f)) & (hiy < WheelF4Splat(200.0f)) & (loz > WheelF4Splat(-20.0f));
    const WheelF4 free_x = WheelF4Splat(mesh_bvh_meshes[0].free_half_x);
    const WheelF4 free_y = WheelF4Splat(mesh_bvh_meshes[0].free_half_y);
    const WheelI4 in_free = (hix < free_x) & (lox > -free_x) & (hiy < free_y) & (loy > -free_y);
    if (!all(in_free)) {
        const HostStepTables* tables = leaf_grid != nullptr ? leaf_grid->host_tables : nullptr;
        if (tables == nullptr || tables->clearance.lb == nullptr || tables->clearance.mesh != mesh_triangles) {
            floor_ok &= in_free;
        } else {
            const WheelF4 half = WheelF4Splat(0.5f);
            const WheelF4 hx = dx * half, hy = dy * half, hz = dz * half;
            const WheelF4 mx = sx + hx, my = sy + hy, mz = sz + hz;
            const WheelF4 r2 = (hx * hx + hy * hy) + hz * hz;
            const ArenaMeshClearance& cl = tables->clearance;
            const WheelF4 zero = WheelF4Splat(0.0f);
            const WheelF4 inv = WheelF4Splat(cl.inv_cell);
            const WheelF4 fx = (mx - WheelF4Splat(cl.lo.x)) * inv;
            const WheelF4 fy = (my - WheelF4Splat(cl.lo.y)) * inv;
            const WheelF4 fz = (mz - WheelF4Splat(cl.lo.z)) * inv;
            const WheelI4 inside = (fx >= zero) & (fy >= zero) & (fz >= zero) &
                                   (fx < WheelF4Splat(static_cast<float>(cl.nx))) &
                                   (fy < WheelF4Splat(static_cast<float>(cl.ny))) &
                                   (fz < WheelF4Splat(static_cast<float>(cl.nz)));
            const WheelI4 ix = __builtin_convertvector(sel(inside, fx, zero), WheelI4);
            const WheelI4 iy = __builtin_convertvector(sel(inside, fy, zero), WheelI4);
            const WheelI4 iz = __builtin_convertvector(sel(inside, fz, zero), WheelI4);
            WheelF4 clear, dist;
            for (int i = 0; i < 4; ++i) {
                clear[i] = static_cast<float>(
                    cl.lb[(static_cast<size_t>(iz[i]) * cl.ny + iy[i]) * cl.nx + ix[i]]);
                dist[i] = SqrtScalar(r2[i]);
            }
            clear = sel(inside, clear * WheelF4Splat(cl.unit), zero);
            floor_ok &= in_free | (clear > dist + WheelF4Splat(2e-3f));
        }
    }
    const WheelF4 zero = WheelF4Splat(0.0f), one = WheelF4Splat(1.0f);
    const WheelF4 eps = WheelF4Splat(1.1920929e-07f);
    const WheelF4 ddx = tx - sx, ddy = ty - sy, ddz = tz - sz;
    const WheelF4 ox = zero - sx, oy = zero - sy, oz = zero - sz;
    const WheelF4 denom = -((ddy * zero + ddx * zero) + ddz * one);
    const WheelF4 numer = -((oy * zero + ox * zero) + oz * one);
    const WheelF4 fraction = numer / denom;
    WheelI4 hit = (loz <= zero) & (eps < denom) & (eps < numer) & (numer <= denom) & (fraction < one) &
                  ~(fraction < zero);
    if (cal_min_hit_dist_sq_bt > 0.0f) {
        const WheelF4 cal_min_fraction_sq = WheelF4Splat(cal_min_hit_dist_sq_bt) / ((ddx * ddx + ddy * ddy) + ddz * ddz);
        hit &= ~((cal_min_fraction_sq > zero) & (fraction * fraction < cal_min_fraction_sq));
    }
    const WheelF4 omf = one - fraction;
    WheelF4 px = sx * omf + tx * fraction, py = sy * omf + ty * fraction, pz = sz * omf + tz * fraction;
    WheelF4 nx = zero, ny = zero, nz = one;
    WheelI4 static_id = (hit & WheelI4{100, 100, 100, 100}) | ~hit;
    if (hints != nullptr) {
        for (int j = 0; j < 8; ++j) {
            hints[j] = -1;
        }
    }
    if (!all(floor_ok)) {
        for (int i = 0; i < 4; ++i) {
            if (floor_ok[i]) {
                continue;
            }
            const Vec3 src(sx[i], sy[i], sz[i]);
            const Vec3 delta = dir * shape.wheel_ray_len_bt[i];
            ArenaRayHit best;
            if (cal_min_hit_dist_sq_bt > 0.0f) {
                best.cal_min_fraction_sq = cal_min_hit_dist_sq_bt / LengthSq((src + delta) - src);
            }
            const ArenaRayHit h = CastArenaRayDeltaBt(
                best, src, delta, shape.wheel_ray_len_uu[i], mesh_triangles, mesh_triangle_count,
                mesh_bvh_quantization, mesh_bvh_quantization_count, mesh_bvh_meshes, mesh_bvh_mesh_count,
                mesh_bvh_nodes, mesh_bvh_node_count, mesh_bvh_subtrees, mesh_bvh_subtree_count,
                mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift, ArenaRayMeshOneLane{}, leaf_grid,
                mutator.cal_arena_local_tris, false, hints != nullptr ? hints + 2 * i : nullptr);
            hit[i] = h.hit ? -1 : 0;
            px[i] = h.point_bt.x;
            py[i] = h.point_bt.y;
            pz[i] = h.point_bt.z;
            nx[i] = h.normal.x;
            ny[i] = h.normal.y;
            nz[i] = h.normal.z;
            static_id[i] = !h.hit ? -1
                : h.kind == ROCKETGPU_ARENA_RAY_HIT_ARENA_MESH ? h.mesh_index
                : (h.normal.z > .5f ? 100 : h.normal.z < -.5f ? 101 : h.normal.x > .5f ? 102 : 103);
        }
    }
    const Vec3 car_up = -dir;
    const WheelF4 ux = WheelF4Splat(car_up.x), uy = WheelF4Splat(car_up.y), uz = WheelF4Splat(car_up.z);
    const WheelF4 trace = ((sx - px) * ux + (sy - py) * uy) + (sz - pz) * uz;
    const WheelF4 denominator = (nx * ux + ny * uy) + nz * uz;
    const WheelF4 rx = px - WheelF4Splat(car.pos_bt.x), ry = py - WheelF4Splat(car.pos_bt.y),
                  rz = pz - WheelF4Splat(car.pos_bt.z);
    const WheelF4 ax = WheelF4Splat(car.ang_vel.x), ay = WheelF4Splat(car.ang_vel.y), az = WheelF4Splat(car.ang_vel.z);
    const WheelF4 vx = WheelF4Splat(car.vel_bt.x) + (ay * rz - az * ry);
    const WheelF4 vy = WheelF4Splat(car.vel_bt.y) + (az * rx - ax * rz);
    const WheelF4 vz = WheelF4Splat(car.vel_bt.z) + (ax * ry - ay * rx);
    const WheelF4 proj = (nx * vx + ny * vy) + nz * vz;
    const WheelI4 upright = denominator > WheelF4Splat(0.1f);
    const WheelF4 clipped = sel(upright, one / denominator, WheelF4Splat(10.0f));
    const WheelF4 rel_vel_bt = sel(upright, proj * clipped, zero);
    const WheelF4 rel_vel = sel(upright, rel_vel_bt * WheelF4Splat(UnitsBTToUU()), zero);
    WheelF4 pb_thresh;
    memcpy(&pb_thresh, shape.wheel_pushback_thresh_bt, sizeof(WheelF4));
    const WheelI4 push = hit & (trace < pb_thresh);
    WheelF4 pushback = zero;
    if (push[0] | push[1] | push[2] | push[3]) {
        const WheelF4 c0x = ry * nz - rz * ny, c0y = rz * nx - rx * nz, c0z = rx * ny - ry * nx;
        const Vec3 r0 = frame->inv_world_row0_bt, r1 = frame->inv_world_row1_bt, r2 = frame->inv_world_row2_bt;
        const WheelF4 angx = (WheelF4Splat(r0.x) * c0x + WheelF4Splat(r1.x) * c0y) + WheelF4Splat(r2.x) * c0z;
        const WheelF4 angy = (WheelF4Splat(r0.y) * c0x + WheelF4Splat(r1.y) * c0y) + WheelF4Splat(r2.y) * c0z;
        const WheelF4 angz = (WheelF4Splat(r0.z) * c0x + WheelF4Splat(r1.z) * c0y) + WheelF4Splat(r2.z) * c0z;
        const WheelF4 cvx = angy * rz - angz * ry, cvy = angz * rx - angx * rz, cvz = angx * ry - angy * rx;
        const WheelF4 angular_dot = (nx * cvx + ny * cvy) + nz * cvz;
        const WheelF4 denom_bt = WheelF4Splat(1.0f / mutator.car_mass) + angular_dot;
        const WheelF4 jac = one / denom_bt;
        const float solver_dt = 1.0f / 60.0f;
        const WheelF4 positional_error = (WheelF4Splat(0.2f) * -(trace - pb_thresh)) / WheelF4Splat(solver_dt);
        const WheelF4 normal_impulse = positional_error * jac + (-proj) * jac;
        const WheelF4 impulse = sel(normal_impulse > zero, normal_impulse, zero);
        pushback = sel(push & (denom_bt > zero), impulse / WheelF4Splat(4.0f), zero);
    }
    const WheelF4 radius{cfg.front_wheel_radius, cfg.front_wheel_radius, cfg.back_wheel_radius, cfg.back_wheel_radius};
    w.force_scale = WheelF4{cal_front_scale, cal_front_scale, cal_back_scale, cal_back_scale};
    w.point_x = sel(hit, px, zero);
    w.point_y = sel(hit, py, zero);
    w.point_z = sel(hit, pz, zero);
    w.normal_x = sel(hit, nx, ux);
    w.normal_y = sel(hit, ny, uy);
    w.normal_z = sel(hit, nz, uz);
    w.susp_len_bt = sel(hit, trace - radius * WheelF4Splat(UnitsUUToBT()), zero);
    w.rel_vel_bt = sel(hit, rel_vel_bt, zero);
    w.rel_vel = sel(hit, rel_vel, zero);
    w.clipped = sel(hit, clipped, one);
    w.extra_pushback = pushback;
    w.contact = hit;
    w.world = hit;
    w.static_id = static_id;
    if (wheel_runtime_state) {
        memcpy(wheel_runtime_state->extra_pushback, &pushback, sizeof(WheelF4));
    }
    static_assert(sizeof(WheelContactOut) == sizeof(WheelF4), "WheelContactOut is (normal, flags)");
    const WheelF4 flags = (WheelF4)(hit & WheelI4{static_cast<int32_t>(kWheelContactInContact | kWheelContactWithWorld),
                                                  static_cast<int32_t>(kWheelContactInContact | kWheelContactWithWorld),
                                                  static_cast<int32_t>(kWheelContactInContact | kWheelContactWithWorld),
                                                  static_cast<int32_t>(kWheelContactInContact | kWheelContactWithWorld)});
    const WheelF4 xy01 = WheelF4Shuffle<0, 4, 1, 5>(w.normal_x, w.normal_y);
    const WheelF4 xy23 = WheelF4Shuffle<2, 6, 3, 7>(w.normal_x, w.normal_y);
    const WheelF4 zf01 = WheelF4Shuffle<0, 4, 1, 5>(w.normal_z, flags);
    const WheelF4 zf23 = WheelF4Shuffle<2, 6, 3, 7>(w.normal_z, flags);
    const WheelF4 rows[4] = {WheelF4Shuffle<0, 1, 4, 5>(xy01, zf01), WheelF4Shuffle<2, 3, 6, 7>(xy01, zf01),
                             WheelF4Shuffle<0, 1, 4, 5>(xy23, zf23), WheelF4Shuffle<2, 3, 6, 7>(xy23, zf23)};
    memcpy(static_cast<void*>(contact_out), rows, sizeof(rows));
    return true;
}
#endif

ROCKETGPU_ALWAYS_INLINE inline bool WheelRaysFloorOnlyHost(
    CarCRefT car, const CarConfig& cfg, const MutatorConfig& mutator, WheelStepState* wheels,
    float cal_front_scale, float cal_back_scale, float cal_min_hit_dist_sq_bt, WheelRuntimeState* wheel_runtime_state,
    const ArenaMeshTriangle* mesh_triangles, int32_t mesh_triangle_count, int32_t cars_in_env_count,
    int32_t self_car_idx, const ArenaMeshBvhQuantization* mesh_bvh_quantization, int32_t mesh_bvh_quantization_count,
    const ArenaMeshBvhMeshInfo* mesh_bvh_meshes, int32_t mesh_bvh_mesh_count, const ArenaMeshBvhNode* mesh_bvh_nodes,
    int32_t mesh_bvh_node_count, const ArenaMeshBvhSubtreeInfo* mesh_bvh_subtrees, int32_t mesh_bvh_subtree_count,
    const WheelRayStage* staged_rays, const CarFrame* frame, uint32_t excluded_bodies,
    float front_bone_z, float back_bone_z, WheelContactOut* contact_out, const ArenaMeshLeafGrid* leaf_grid
#if ROCKETGPU_WHEEL_X4
    , WheelsX4* wx
#endif
    ) {
    if (mutator.arena_collision != 1 || mesh_triangles == nullptr || mesh_triangle_count <= 0 ||
        mesh_bvh_quantization == nullptr || mesh_bvh_quantization_count <= 0 || mesh_bvh_meshes == nullptr ||
        mesh_bvh_mesh_count <= 0 || mesh_bvh_nodes == nullptr || mesh_bvh_node_count <= 0) {
        return false;
    }
    if (staged_rays != nullptr &&
        (staged_rays[0].valid | staged_rays[1].valid | staged_rays[2].valid | staged_rays[3].valid) != 0) {
        return false;
    }
    const int active_car_bits = Min(cars_in_env_count, kBulletSweepMaxBodies - 1);
    const uint32_t all_dyn_mask =
        ((active_car_bits > 0 ? ((1u << (active_car_bits + 1)) - 1u) : 1u)) &
        ~(self_car_idx >= 0 && self_car_idx + 1 < kBulletSweepMaxBodies ? (1u << (self_car_idx + 1)) : 0u);
    if ((excluded_bodies & all_dyn_mask) != all_dyn_mask) {
        return false;
    }
#if ROCKETGPU_WHEEL_X4
    if (wx != nullptr) {
        return WheelRaysFloorOnlyHostX4(car, cfg, mutator, *wx, cal_front_scale, cal_back_scale,
                                        cal_min_hit_dist_sq_bt, wheel_runtime_state, mesh_triangles,
                                        mesh_triangle_count, mesh_bvh_quantization, mesh_bvh_quantization_count,
                                        mesh_bvh_meshes, mesh_bvh_mesh_count, mesh_bvh_nodes, mesh_bvh_node_count,
                                        mesh_bvh_subtrees, mesh_bvh_subtree_count, frame, contact_out, leaf_grid);
    }
#endif
    constexpr float kUUToBT = 1.0f / 50.0f;
    const float free_x = mesh_bvh_meshes[0].free_half_x, free_y = mesh_bvh_meshes[0].free_half_y;
    float ray_len[4];
    Vec3 delta[4], to[4];
    bool floor_cast[4];
    for (int i = 0; i < 4; ++i) {
        WheelStepState& w = wheels[i];
        if (frame == nullptr || !frame->wheels_pre_inited) {
            new (&w) WheelStepState;
            InitWheelStepState(w, car, cfg, i, frame, i < 2 ? front_bone_z : back_bone_z);
        }
        ray_len[i] = w.suspension_rest_length + w.max_suspension_travel + w.wheel_radius - WheelSuspensionSubtraction();
        const float len_bt_override = WheelRealRayLengthBt(w);
        const float max_len_bt = len_bt_override > 0.0f ? len_bt_override : ray_len[i] * kUUToBT;
        const Vec3 src = w.ray_source_bt;
        delta[i] = w.wheel_dir_ws * max_len_bt;
        to[i] = src + delta[i];
        const Vec3 lo(Min(src.x, to[i].x), Min(src.y, to[i].y), Min(src.z, to[i].z));
        const Vec3 hi(Max(src.x, to[i].x), Max(src.y, to[i].y), Max(src.z, to[i].z));
        const bool skip_meshes = ray_len[i] > 0.0f &&
            ((hi.x < free_x && lo.x > -free_x && hi.y < free_y && lo.y > -free_y) ||
             HostRayClearOfMeshes(leaf_grid, mesh_triangles, src, delta[i]));
        const bool floor_region = lo.x > -81.9f && hi.x < 81.9f && hi.z < 40.8f && lo.y > -200.0f && hi.y < 200.0f &&
                                  lo.z > -20.0f;
        if (!(skip_meshes && floor_region)) {
            return false;
        }
        floor_cast[i] = lo.z <= 0.0f;
    }
    for (int k = 0; k < 4; ++k) {
        const int i = k ^ 1;
        WheelStepState& wheel = wheels[i];
        wheel.suspension_force_scale = wheel.is_front_wheel ? cal_front_scale : cal_back_scale;
        wheel.extra_pushback = wheel_runtime_state ? wheel_runtime_state->extra_pushback[i] : 0.0f;
        bool hit = false;
        Vec3 point_bt;
        if (floor_cast[i]) {
            constexpr float kFltEpsilon = 1.1920929e-07f;
            const Vec3 src = wheel.ray_source_bt;
            const Vec3 d = to[i] - src;
            const Vec3 off = Vec3(0.0f, 0.0f, 0.0f) - src;
            const float denom = -(d.y * 0.0f + d.x * 0.0f + d.z * 1.0f);
            const float numer = -(off.y * 0.0f + off.x * 0.0f + off.z * 1.0f);
            if (kFltEpsilon < denom && kFltEpsilon < numer && numer <= denom) {
                const float fraction = numer / denom;
                if (fraction < 1.0f && !(fraction < 0.0f)) {
                    const float cal_min_fraction_sq =
                        cal_min_hit_dist_sq_bt > 0.0f ? cal_min_hit_dist_sq_bt / LengthSq((src + delta[i]) - src) : 0.0f;
                    if (!(cal_min_fraction_sq > 0.0f && fraction * fraction < cal_min_fraction_sq)) {
                        const float one_minus_fraction = 1.0f - fraction;
                        point_bt = Vec3(src.x * one_minus_fraction + to[i].x * fraction,
                                        src.y * one_minus_fraction + to[i].y * fraction,
                                        src.z * one_minus_fraction + to[i].z * fraction);
                        hit = true;
                    }
                }
            }
        }
        const Vec3 car_up = -wheel.wheel_dir_ws;
        if (hit) {
            wheel.contact_point_bt = point_bt;
            wheel.contact_normal_ws = Vec3(0.0f, 0.0f, 1.0f);
            wheel.is_in_contact = 1;
            wheel.is_in_contact_with_world = 1;
            wheel.ground_kind = ROCKETGPU_ARENA_RAY_HIT_STATIC_PLANE;
            wheel.ground_car_idx = -1;
            wheel.ground_static_id = 100;
            const float trace_len_bt = Dot(wheel.ray_source_bt - point_bt, car_up);
            const float radius_bt = wheel.wheel_radius * UnitsUUToBT();
            wheel.wheel_trace_len_bt = trace_len_bt;
            wheel.suspension_len_bt = trace_len_bt - radius_bt;
            ComputeWheelRayVelocityTerms(car, car_up, point_bt, wheel);
            wheel.extra_pushback = 0.0f;
            SetWorldWheelPushback(car, cfg, mutator, wheel, frame);
        } else {
            SetWheelRayMiss(wheel);
        }
        if (wheel_runtime_state) {
            wheel_runtime_state->extra_pushback[i] = wheel.extra_pushback;
        }
        WheelContactOut out;
        out.normal = wheel.contact_normal_ws;
        out.flags = (wheel.is_in_contact ? kWheelContactInContact : 0u) |
                    (wheel.is_in_contact_with_world ? kWheelContactWithWorld : 0u) |
                    (wheel.ground_kind == ROCKETGPU_ARENA_RAY_HIT_BALL ? kWheelContactOnBall : 0u);
        contact_out[i] = out;
    }
    return true;
}
#endif

template <bool kStagedIndependent = false>
ROCKETGPU_HD ROCKETGPU_HOST_INLINE int PreTickVehicleDynamics(
    CarRefT car,
    CarControls controls,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    float dt,
    WheelStepState* wheels,
    float& forward_speed_uu,
    WheelRuntimeState* wheel_runtime_state,
    const ArenaMeshTriangle* mesh_triangles = nullptr,
    int32_t mesh_triangle_count = 0,
    CarCSpanT cars_in_env = nullptr,
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
    CarSpanT dyn_hit_cars = nullptr,
    BallState* dyn_hit_ball = nullptr,
    uint32_t* dyn_hit_pushed_mask = nullptr,
    const CarFrame* frame = nullptr,
    uint32_t excluded_bodies = 0u,
    const ArenaMeshLeafGrid* leaf_grid = nullptr) {
    car.world_contact.wheel_ball_mask = 0;

    const uint8_t sticky_prev_world_contacts =
        wheel_runtime_state ? wheel_runtime_state->cal_prev_world_wheel_contacts : 0;
    const Vec3 sticky_prev_world_normal_sum =
        wheel_runtime_state ? wheel_runtime_state->cal_prev_wheel_normal_sum : Vec3(0.0f, 0.0f, 0.0f);
    const bool sticky_prev_contact = (car.wheels_with_contact[0] || car.wheels_with_contact[1] || car.wheels_with_contact[2] || car.wheels_with_contact[3]);

    const bool landing_prev_any_contact = car.tick_count_since_update != 0 &&
        (car.wheels_with_contact[0] || car.wheels_with_contact[1] || car.wheels_with_contact[2] || car.wheels_with_contact[3]);
    float cal_front_scale = 0.0f, cal_back_scale = 0.0f, cal_min_hit_dist_sq_bt = 0.0f;
#if !defined(ROCKETGPU_USE_CUDA)
    float front_bone_z, back_bone_z;
    if (const CarShapeCacheEntry* const shape = frame != nullptr ? frame->shape : nullptr) {
        cal_front_scale = shape->susp_front_scale;
        cal_back_scale = shape->susp_back_scale;
        if constexpr (!kStagedIndependent) {
            cal_min_hit_dist_sq_bt = shape->min_wheel_hit_dist_sq_bt;
        }
        front_bone_z = shape->front_bone_z;
        back_bone_z = shape->back_bone_z;
    } else {
        CalSuspensionForceScales(cfg, mutator.car_mass, cal_front_scale, cal_back_scale);
        if constexpr (!kStagedIndependent) {
            cal_min_hit_dist_sq_bt = CalMinWheelHitDistSqBt(cfg);
        }
        CalWheelBoneZ(cfg, front_bone_z, back_bone_z);
    }
    const uint32_t active_excluded_bodies = kStagedIndependent ? 0u : excluded_bodies;
#else
    CalSuspensionForceScales(cfg, mutator.car_mass, cal_front_scale, cal_back_scale);
    if constexpr (!kStagedIndependent) {
        cal_min_hit_dist_sq_bt = CalMinWheelHitDistSqBt(cfg);
    }
    const uint32_t active_excluded_bodies = kStagedIndependent ? 0u : excluded_bodies;
    const float front_bone_z =
        floorf((cfg.hitbox_offset.z - (cfg.front_suspension_rest - 12.0f)) * 100.0f + 0.5f) / 100.0f;
    const float back_bone_z = (cfg.back_suspension_rest == cfg.front_suspension_rest)
        ? front_bone_z
        : (floorf((cfg.hitbox_offset.z - (cfg.back_suspension_rest - 12.0f)) * 100.0f + 0.5f) / 100.0f);
#endif

    WheelContactOut contact_out[4];
#if ROCKETGPU_WHEEL_X4
    WheelsX4& wx = *frame->wx;
    WheelsX4* const wx_rays =
        (!kStagedIndependent && frame->shape != nullptr && car.set_wheel_normals_mask == 0) ? &wx : nullptr;
    if (wx_rays != nullptr) {
        InitWheelsX4(wx, car, frame, *frame->shape);
    }
    bool wx_done = false;
#elif !defined(ROCKETGPU_USE_CUDA)
    if (frame != nullptr && frame->shape != nullptr) {
        InitWheelStepStates4(wheels, car, cfg, frame, *frame->shape);
        frame->wheels_pre_inited = true;
    }
#endif
    if constexpr (kStagedIndependent) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int i = k ^ 1;
            contact_out[i] = WheelRayPhase<true>(
                car, cfg, mutator, wheels[i], i, cal_front_scale, cal_back_scale,
                cal_min_hit_dist_sq_bt, wheel_runtime_state, mesh_triangles, mesh_triangle_count,
                cars_in_env, car_cfgs_in_env, cars_in_env_count, self_car_idx, env_idx, ball,
                ball_radius, mesh_bvh_quantization, mesh_bvh_quantization_count, mesh_bvh_meshes,
                mesh_bvh_mesh_count, mesh_bvh_nodes, mesh_bvh_node_count, mesh_bvh_subtrees,
                mesh_bvh_subtree_count, staged_rays, staged_rays_velocity_stale, dyn_hit_cars,
                dyn_hit_ball, dyn_hit_pushed_mask, frame, active_excluded_bodies,
                i < 2 ? front_bone_z : back_bone_z, true, leaf_grid);
        }
    } else {
#if ROCKETGPU_WHEEL_X4
        const bool floor_only = WheelRaysFloorOnlyHost(
            car, cfg, mutator, wheels, cal_front_scale, cal_back_scale, cal_min_hit_dist_sq_bt, wheel_runtime_state,
            mesh_triangles, mesh_triangle_count, cars_in_env_count, self_car_idx, mesh_bvh_quantization,
            mesh_bvh_quantization_count, mesh_bvh_meshes, mesh_bvh_mesh_count, mesh_bvh_nodes, mesh_bvh_node_count,
            mesh_bvh_subtrees, mesh_bvh_subtree_count, staged_rays, frame, active_excluded_bodies, front_bone_z,
            back_bone_z, contact_out, leaf_grid, wx_rays);
        wx_done = floor_only && wx_rays != nullptr;
        if (!floor_only)
#elif !defined(ROCKETGPU_USE_CUDA)
        if (!WheelRaysFloorOnlyHost(car, cfg, mutator, wheels, cal_front_scale, cal_back_scale, cal_min_hit_dist_sq_bt,
                                    wheel_runtime_state, mesh_triangles, mesh_triangle_count, cars_in_env_count,
                                    self_car_idx, mesh_bvh_quantization, mesh_bvh_quantization_count, mesh_bvh_meshes,
                                    mesh_bvh_mesh_count, mesh_bvh_nodes, mesh_bvh_node_count, mesh_bvh_subtrees,
                                    mesh_bvh_subtree_count, staged_rays, frame, active_excluded_bodies, front_bone_z,
                                    back_bone_z, contact_out, leaf_grid))
#endif
        for (int k = 0; k < 4; ++k) {
            const int i = k ^ 1;
            contact_out[i] = WheelRayPhase<false>(
                car, cfg, mutator, wheels[i], i, cal_front_scale, cal_back_scale,
                cal_min_hit_dist_sq_bt, wheel_runtime_state, mesh_triangles, mesh_triangle_count,
                cars_in_env, car_cfgs_in_env, cars_in_env_count, self_car_idx, env_idx, ball,
                ball_radius, mesh_bvh_quantization, mesh_bvh_quantization_count, mesh_bvh_meshes,
                mesh_bvh_mesh_count, mesh_bvh_nodes, mesh_bvh_node_count, mesh_bvh_subtrees,
                mesh_bvh_subtree_count, staged_rays, staged_rays_velocity_stale, dyn_hit_cars,
                dyn_hit_ball, dyn_hit_pushed_mask, frame, active_excluded_bodies,
                i < 2 ? front_bone_z : back_bone_z, true, leaf_grid);
        }
    }
#if ROCKETGPU_WHEEL_X4
    if (!wx_done) {
        WheelsX4FromStates(wx, wheels);
    }
#elif !defined(ROCKETGPU_USE_CUDA)
    if (frame != nullptr) {
        frame->wheels_pre_inited = false;
    }
#endif
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

#if defined(ROCKETGPU_USE_CUDA)
    float drive_speed_scale = EvalDriveTorqueFactor(forward_speed_uu);
#endif
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

#if !defined(ROCKETGPU_USE_CUDA)
    const float drive_speed_scale = engine_throttle != 0.0f ? EvalDriveTorqueFactor(forward_speed_uu) : 1.0f;
#endif
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
#if ROCKETGPU_WHEEL_X4
    const bool front_in_contact = (wx.contact[0] | wx.contact[1]) != 0;
    const bool back_in_contact = (wx.contact[2] | wx.contact[3]) != 0;
    const bool native_sse = NativeSseMath(mutator);
    WheelStepState steer_wheel;
    steer_wheel.wheel_dir_ws = wx.dir;
    steer_wheel.wheel_axle_ws = wx.axle;
    if (front_in_contact) {
        drive_shared.front_right_col = ComputeWheelSteeredRightCol(steer_wheel, next_steer_angle, native_sse);
    }
    if (back_in_contact) {
        const bool positive_zero_steer = next_steer_angle == 0.0f && !std::signbit(next_steer_angle);
        if (front_in_contact && positive_zero_steer) {
            drive_shared.back_right_col = drive_shared.front_right_col;
        } else {
            drive_shared.back_right_col = ComputeWheelSteeredRightCol(steer_wheel, 0.0f, native_sse);
        }
    }
#else
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
#endif
#if ROCKETGPU_WHEEL_X4
    WheelDriveSetupPhaseX4(car, cfg, drive_shared, wx, wheel_runtime_state, dt);
#else
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        WheelDriveSetupPhase(car, cfg, mutator, drive_shared, wheels[i], i, wheel_runtime_state, dt, frame);
    }
#endif

    if (wheel_runtime_state) {
        const bool restore_tick = car.tick_count_since_update == 0;
        Vec3 world_normal_sum = sticky_prev_world_normal_sum;
        bool gate = sticky_prev_world_contacts > 0;
        if (restore_tick) {
            world_normal_sum = Vec3(0.0f, 0.0f, 0.0f);
            gate = false;
            if constexpr (!kStagedIndependent) {
                if (car.set_wheel_normals_mask) {
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
#if ROCKETGPU_WHEEL_X4
        Vec3 upwards_dir = wheels_have_world_contact ? Normalize(wheel_normal_sum, Vec3(0.0f, 0.0f, 1.0f)) : -wx.dir;
#else
        Vec3 upwards_dir = wheels_have_world_contact ? Normalize(wheel_normal_sum, Vec3(0.0f, 0.0f, 1.0f)) : -wheels[0].wheel_dir_ws;
#endif
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

#if !defined(ROCKETGPU_USE_CUDA)
ROCKETGPU_HD float QuantizeAxisByteValue(int byte) {
    const float s = static_cast<float>(byte) - 128.0f;
    constexpr float kInv127 = 1.0f / 127.0f;
    const float q = s * kInv127;
    const float s_over_127 = fmaf(fmaf(-127.0f, q, s), kInv127, q);
    return byte < 0x80 ? s * (1.0f / 128.0f) : s_over_127;
}

struct QuantizeAxisTable {
    float v[256];
};
inline const QuantizeAxisTable kQuantizeAxisTable = [] {
    QuantizeAxisTable t{};
    for (int b = 0; b < 256; ++b) {
        t.v[b] = QuantizeAxisByteValue(b);
    }
    return t;
}();
#endif

ROCKETGPU_HD float QuantizeAxisInput(float x) {
    const float c = Clamp(x, -1.0f, 1.0f);
    const float y = c < 0.0f ? Max(c * 128.0f, -128.0f) : Min(c * 127.0f, 127.0f);
    const float w = ((y + 128.0f) + (y + 128.0f)) + 0.5f;
    const int byte = (static_cast<int>(rintf(w)) >> 1) & 0xFF;
#if !defined(ROCKETGPU_USE_CUDA)
    return kQuantizeAxisTable.v[byte];
#else
    const float s = static_cast<float>(byte) - 128.0f;
    constexpr float kInv127 = 1.0f / 127.0f;
    const float q = s * kInv127;
    const float s_over_127 = fmaf(fmaf(-127.0f, q, s), kInv127, q);
    return byte < 0x80 ? s * (1.0f / 128.0f) : s_over_127;
#endif
}

#if !defined(ROCKETGPU_USE_CUDA) && (defined(__GNUC__) || defined(__clang__))
ROCKETGPU_HD CarControls QuantizeAxisControls(CarControls controls, float* clamped_pitch = nullptr) {
    typedef float F4 __attribute__((vector_size(16)));
    typedef int32_t I4 __attribute__((vector_size(16)));
    const auto splat = [](float v) { return F4{v, v, v, v}; };
    const auto sel = [](I4 m, F4 a, F4 b) {
        return reinterpret_cast<F4>((m & reinterpret_cast<I4>(a)) | (~m & reinterpret_cast<I4>(b)));
    };
    const F4 x = {controls.throttle, controls.steer, controls.pitch, controls.yaw};
    const F4 c = sel(x < splat(-1.0f), splat(-1.0f), sel(x > splat(1.0f), splat(1.0f), x));
    if (clamped_pitch != nullptr) {
        *clamped_pitch = c[2];
    }
    const F4 neg = c * splat(128.0f);
    const F4 pos = c * splat(127.0f);
    const F4 y = sel(c < splat(0.0f), sel(neg > splat(-128.0f), neg, splat(-128.0f)),
                     sel(pos < splat(127.0f), pos, splat(127.0f)));
    const F4 t = y + splat(128.0f);
    const F4 w = (t + t) + splat(0.5f);
    const F4 r = (w + splat(8388608.0f)) - splat(8388608.0f);
    const I4 b = (__builtin_convertvector(r, I4) >> 1) & 0xFF;
    controls.throttle = kQuantizeAxisTable.v[b[0]];
    controls.steer = kQuantizeAxisTable.v[b[1]];
    controls.pitch = kQuantizeAxisTable.v[b[2]];
    controls.yaw = kQuantizeAxisTable.v[b[3]];
    controls.roll = QuantizeAxisInput(controls.roll);
    return controls;
}
#else
ROCKETGPU_HD CarControls QuantizeAxisControls(CarControls controls) {
    controls.throttle = QuantizeAxisInput(controls.throttle);
    controls.steer = QuantizeAxisInput(controls.steer);
    controls.pitch = QuantizeAxisInput(controls.pitch);
    controls.yaw = QuantizeAxisInput(controls.yaw);
    controls.roll = QuantizeAxisInput(controls.roll);
    return controls;
}
#endif

ROCKETGPU_HD float DodgeCancelPitchRaw(CarCRefT car, const CarControls& controls,
                                      float raw_pitch = 2.0f) {
    {
        constexpr float kGameMinDodgeTorqueTime = 0.041f;
        const float pitch = raw_pitch <= 1.0f ? raw_pitch : controls.pitch;
        return car.internal_flip_time >= kGameMinDodgeTorqueTime ? pitch : 0.0f;
    }
    return (car.internal_flip_time > (0.0f)) ? car.last_controls.pitch : controls.pitch;
}

ROCKETGPU_HD float DodgeCancelPitch(CarCRefT car, const CarControls& controls,
                                   float raw_pitch = 2.0f) {
    const float pitch = DodgeCancelPitchRaw(car, controls, raw_pitch);
    return (-QuantizeAxisInput(-pitch));
}

ROCKETGPU_HOST_NOINLINE ROCKETGPU_HD void ApplyAirControl(
    CarRefT car,
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
    if (!update_air_control && !air_throttle) {
        return;
    }

    const bool do_air_control = update_air_control;
    Vec3 ue_f(0.0f, 0.0f, 0.0f);
    bool have_ue_axes = false;
#if defined(ROCKETGPU_HOST_LANES)
    if (do_air_control) {
        float pitch_torque_scale = 1.0f;
        if (car.is_flipping) {
            pitch_torque_scale = 0.0f;
        } else if (car.has_flipped && car.internal_flip_time < (mutator.flip_torque_time + mutator.flip_pitchlock_extra_time)) {
            pitch_torque_scale = 0.0f;
        } else if (cal_pitchlock_restart_time >= 0.0f && cal_pitchlock_restart_time < mutator.flip_pitchlock_extra_time) {
            pitch_torque_scale = 0.0f;
        }

        const Vec3 damp_ang_vel = cal_damp_ang_vel != nullptr ? *cal_damp_ang_vel : car.ang_vel;
        if (!wheel_runtime_state || !(car.basis_authoritative && frame && frame->queue_forces)) {
            Vec3 basis_forward(1.0f, 0.0f, 0.0f);
            Vec3 basis_right(0.0f, 1.0f, 0.0f);
            Vec3 basis_up(0.0f, 0.0f, 1.0f);
            CarBasisColumns(frame, car, basis_forward, basis_right, basis_up);
            const Vec3 dir_pitch_right(-basis_right.x, -basis_right.y, -basis_right.z);
            const Vec3 dir_yaw_up = basis_up;
            const Vec3 dir_roll_forward(-basis_forward.x, -basis_forward.y, -basis_forward.z);
            Vec3 torque_world(0.0f, 0.0f, 0.0f);
            if (controls.pitch != 0.0f || controls.yaw != 0.0f || controls.roll != 0.0f) {
                torque_world =
                    (controls.pitch * dir_pitch_right * pitch_torque_scale * mutator.car_air_control_pitch) +
                    (controls.yaw * dir_yaw_up * mutator.car_air_control_yaw) +
                    (controls.roll * dir_roll_forward * mutator.car_air_control_roll);
            }

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
            } else {
                ApplyAngularAccelWorld(car, air_control_torque, dt);
            }
        }
        if (wheel_runtime_state) {
            using lanes::F4;
            using lanes::I4;
            using lanes::Splat;
            Vec3 ue_r, ue_u;
            GameUeQuatAxes(car.rot, ue_f, ue_r, ue_u);
            have_ue_axes = true;
            constexpr float kRot = 2.0f * Pi() / 65536.0f;
            constexpr float kGate = 1.0e-4f;
            const float in_pitch = controls.pitch * pitch_torque_scale;
            const float a_p = (mutator.car_air_control_pitch * -in_pitch) * kRot;
            const float a_y = (mutator.car_air_control_yaw * controls.yaw) * kRot;
            const float a_r = (-mutator.car_air_control_roll * controls.roll) * kRot;
            const F4 R = lanes::Load(ue_r), U = lanes::Load(ue_u), Fw = lanes::Load(ue_f);
            const F4 t_in = (((Splat(a_p) * R + Splat(0.0f)) + Splat(a_y) * U) + Splat(a_r) * Fw) * Splat(1000.0f);
            const float d_p = (1.0f - Abs(in_pitch)) * (Dot(damp_ang_vel, ue_r) * (mutator.car_air_damp_pitch * kRot));
            const float d_y = (1.0f - Abs(controls.yaw)) * (Dot(damp_ang_vel, ue_u) * (mutator.car_air_damp_yaw * kRot));
            const float d_r = (1.0f * Dot(damp_ang_vel, ue_f)) * (mutator.car_air_damp_roll * kRot);
            const F4 t_damp = (((Splat(0.0f) - Splat(d_p) * R) - Splat(d_y) * U) - Splat(d_r) * Fw) * Splat(1000.0f);
            const I4 g_in = lanes::Abs(t_in) >= Splat(kGate);
            const I4 g_damp = lanes::Abs(t_damp) >= Splat(kGate);
            const bool apply_in = (g_in[0] | g_in[1] | g_in[2]) != 0;
            const bool apply_damp = (g_damp[0] | g_damp[1] | g_damp[2]) != 0;
            if (apply_in) {
                QueueOfficialTorqueBt(wheel_runtime_state,
                                      MulInertiaWorldBt(frame, car, cfg, mutator.car_mass, lanes::ToVec3(t_in)), frame);
            }
            if (apply_damp) {
                QueueOfficialTorqueBt(wheel_runtime_state,
                                      MulInertiaWorldBt(frame, car, cfg, mutator.car_mass, lanes::ToVec3(t_damp)), frame);
            }
        }
    }
#else
    if (do_air_control) {
        Vec3 basis_forward(1.0f, 0.0f, 0.0f);
        Vec3 basis_right(0.0f, 1.0f, 0.0f);
        Vec3 basis_up(0.0f, 0.0f, 1.0f);
        CarBasisColumns(frame, car, basis_forward, basis_right, basis_up);
        const Vec3 dir_pitch_right(-basis_right.x, -basis_right.y, -basis_right.z);
        const Vec3 dir_yaw_up = basis_up;
        const Vec3 dir_roll_forward(-basis_forward.x, -basis_forward.y, -basis_forward.z);
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

            Vec3 ue_r, ue_u;
            GameUeQuatAxes(car.rot, ue_f, ue_r, ue_u);
            have_ue_axes = true;
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

#endif
    const float air_throttle_input = (car.is_boosting) ? 1.0f : controls.throttle;
    if (air_throttle_input != 0.0f && air_throttle) {
        if (!have_ue_axes) {
            Vec3 throttle_rgt, throttle_up;
            GameUeQuatAxes(car.rot, ue_f, throttle_rgt, throttle_up);
            (void)throttle_rgt; (void)throttle_up;
        }

        const float air_throttle_force = mutator.car_air_throttle_accel * mutator.car_mass;
        const Vec3 air_throttle_force_bt = ((ue_f * air_throttle_force) * air_throttle_input) * UnitsUUToBT();
        if (wheel_runtime_state) {
            QueueCentralForce(wheel_runtime_state, air_throttle_force_bt, frame);
        } else {
            const Vec3 air_throttle_force = air_throttle_force_bt * UnitsBTToUU();
            ApplyCentralForce(car, mutator.car_mass, air_throttle_force, dt);
        }
    }
}

ROCKETGPU_HD void ApplyBoost(
    CarRefT car,
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

ROCKETGPU_HD bool RobinTorque(CarCRefT car, float pitch_freq, float pitch_amp,
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

ROCKETGPU_HD void ApplyRobinJumpTorque(CarRefT car, const CarConfig& cfg, float mass, const CarFrame* frame) {
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

ROCKETGPU_HD void ApplyRobinAirTorque(CarRefT car, const CarConfig& cfg, float mass, float dt,
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

ROCKETGPU_HD void StartJump(CarRefT car, const CarConfig& cfg, const MutatorConfig& mutator, const CarFrame* frame = nullptr) {
    car.is_jumping = 1;
    car.internal_jump_time = 0.0f;
    if (cfg.three_wheels) {
        ApplyRobinJumpTorque(car, cfg, mutator.car_mass, frame);
    }

    (void)frame;
    ApplyCentralImpulseBt(car, GameJumpUpScaledBt(car, mutator.jump_immediate_speed * mutator.car_mass), mutator.car_mass);
}

ROCKETGPU_HD void ContinueJump(
    CarRefT car,
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
    CarRefT car,
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
    CarRefT car,
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
    CarRefT car,
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
    CarRefT car,
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

ROCKETGPU_HD float CalFlipRestartTimeOnGround(CarCRefT car, const MutatorConfig& mutator, float dt) {
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

ROCKETGPU_HD void CalInternalTimersFromSetState(CarRefT car, const MutatorConfig& mutator, float dt) {
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
    CarRefT car,
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

ROCKETGPU_HD void CapCarAngularSpeed(CarRefT car, const MutatorConfig& mutator) {
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

ROCKETGPU_HD void IntegrateCarTransformOnly(CarRefT car, float dt, bool native_sse_math = false,
                                            const Quat* basis_rot = nullptr) {
    if (!(car.vel_bt.x == 0.0f && car.vel_bt.y == 0.0f && car.vel_bt.z == 0.0f)) {
        SetCarOriginBt(car, car.pos_bt + car.vel_bt * dt);
    }

    IntegrateCarOrientation(car, car.ang_vel, dt, native_sse_math, basis_rot);
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

#ifndef ROCKETGPU_NOINLINE
#if defined(__CUDA_ARCH__)
#define ROCKETGPU_NOINLINE __noinline__
#else
#define ROCKETGPU_NOINLINE
#endif
#endif
#if defined(ROCKETGPU_NOINLINE_DEFAULTED) && !defined(ROCKETGPU_USE_CUDA) && !defined(__CUDACC__) && \
    (defined(__GNUC__) || defined(__clang__))
__attribute__((always_inline))
#endif
ROCKETGPU_HD ROCKETGPU_NOINLINE void QuantizeCarState(CarArgT car_arg, bool v3_exact = false,
                                                      const Vec3* pre_scaled_vel_uu = nullptr,
                                                      bool save_tick_start_basis = false) {
    CarRefT car = car_arg;
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
        if (save_tick_start_basis) {
            car.tick_start_basis_forward = fwd;
            car.tick_start_basis_right = right;
            car.tick_start_basis_up = up;
        }
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

ROCKETGPU_HD void FinishCarTick(CarRefT car) {
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

ROCKETGPU_HD void CalTickStartLimitAndQuantizeCar(CarRefT car, const MutatorConfig& mutator) {
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
    QuantizeCarState(car, true, &vel_uu, true);
}

ROCKETGPU_HD void CalFlipZDampPrepass(CarRefT car, const MutatorConfig& mutator, float dt) {
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

ROCKETGPU_HD void PostTickCar(CarRefT car, CarControls controls, float dt) {
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

ROCKETGPU_HOST_COLD ROCKETGPU_HD bool CastWheelRayAgainstCarHitboxGjkCast(
    const Vec3& ray_source_bt,
    const Vec3& ray_dir,
    float max_len,
    CarCRefT other_car,
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
#pragma unroll
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

    bool is_near_edge = false;
    int num_concave_edge_hits = 0;

#pragma unroll
    for (int edge_idx = 0; edge_idx < 3; ++edge_idx) {
        if (best_edge != edge_idx && (on_edge_mask & (1u << edge_idx)) == 0u) {
            continue;
        }
        const float candidate_edge_angle = ArenaTriangleEdgeAngle(tri, edge_idx);
        if (Abs(candidate_edge_angle) >= kMaxEdgeAngleThreshold) {
            continue;
        }

        const Vec3 nearest = ClosestPointOnSegmentBT(contact, edge_start[edge_idx], edge_end[edge_idx]);
        const float edge_distance = Length(contact - nearest);
        if (edge_distance >= kEdgeDistanceThreshold) {
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
