

struct ArenaMeshTriangle;

struct ArenaTriangleInternalEdgeDebug {
    int best_edge = -1;
    int active_edge = -1;
    int num_concave_edge_hits = 0;
    float best_edge_distance = 0.0f;
    float best_edge_angle = 0.0f;
    float active_edge_distance = 0.0f;
    float active_edge_angle = 0.0f;
    uint8_t active_edge_convex = 0;
    uint8_t back_facing = 0;
    uint8_t clamped = 0;
    uint8_t accepted_clamp = 0;
    Vec3 point_on_b_in = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 point_on_b_out = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 normal_in = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 normal_out = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 tri_normal = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 nA = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 nB = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 clamped_local_normal = Vec3(0.0f, 0.0f, 1.0f);
};

constexpr bool EnableArenaTriangleInternalEdgeAdjust();

ROCKETGPU_HD Vec3 ArenaTriangleFaceNormal(const ArenaMeshTriangle& tri);
ROCKETGPU_HD bool MaybeAdjustArenaTriangleInternalEdgeContact(
    const ArenaMeshTriangle& tri,
    const Vec3& point_on_a_bt,
    float distance_bt,
    Vec3& io_point_on_b_bt,
    Vec3& io_normal,
    ArenaTriangleInternalEdgeDebug* debug = nullptr,
    bool soccar_platform_math = false,
    const uint32_t* native_table = nullptr, int native_shift = 0,
    const ArenaMeshTriangle* cal_game_tri = nullptr);

struct CarWorldPlaneContactGeometryBt {
    Vec3 support_dir_local_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 support_local_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 child_origin_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 child_world_origin_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 position_world_on_a_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 local_point_a_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 rel_pos1_bt = Vec3(0.0f, 0.0f, 0.0f);
    float distance_bt = 0.0f;

    float add_distance_bt = 0.0f;
    Vec3 local_point_b_bt = Vec3(0.0f, 0.0f, 0.0f);
};

ROCKETGPU_HD float BulletStyleDot3Bt(const Vec3& a, const Vec3& b) {
    return bullet_port::btSseStyleDot3(a.x, a.y, a.z, b.x, b.y, b.z);
}

ROCKETGPU_HD Vec3 BulletStyleTransformPointBt(
    const Vec3& row0,
    const Vec3& row1,
    const Vec3& row2,
    const Vec3& origin,
    const Vec3& point) {
    return Vec3(
        BulletStyleDot3Bt(point, row0) + origin.x,
        BulletStyleDot3Bt(point, row1) + origin.y,
        BulletStyleDot3Bt(point, row2) + origin.z);
}

ROCKETGPU_HD Vec3 BulletStyleInvTransformPointBt(
    const Vec3& basis_forward,
    const Vec3& basis_right,
    const Vec3& basis_up,
    const Vec3& origin,
    const Vec3& point) {
    const Vec3 rel = point - origin;
    return Vec3(
        BulletStyleDot3Bt(rel, basis_forward),
        BulletStyleDot3Bt(rel, basis_right),
        BulletStyleDot3Bt(rel, basis_up));
}

ROCKETGPU_HD Vec3 BulletStaticPlaneOriginBt(const Vec3& inward_normal) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    if (inward_normal.z > 0.5f) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }
    if (inward_normal.z < -0.5f) {
        return Vec3(0.0f, 0.0f, ArenaHeight() * kUUToBT);
    }
    if (inward_normal.x > 0.5f) {
        return Vec3(-ArenaExtentX() * kUUToBT, 0.0f, ArenaHeight() * 0.5f * kUUToBT);
    }
    if (inward_normal.x < -0.5f) {
        return Vec3(ArenaExtentX() * kUUToBT, 0.0f, ArenaHeight() * 0.5f * kUUToBT);
    }
    return Vec3(0.0f, 0.0f, 0.0f);
}

ROCKETGPU_HD CarWorldPlaneContactGeometryBt ComputeBulletStyleCarWorldPlaneContactGeometryBt(
    const CarState& car,
    const CarConfig& cfg,
    const Vec3& inward_normal,
    float boundary) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    (void)boundary;
    CarWorldPlaneContactGeometryBt out;

    Vec3 basis_forward(1.0f, 0.0f, 0.0f);
    Vec3 basis_right(0.0f, 1.0f, 0.0f);
    Vec3 basis_up(0.0f, 0.0f, 1.0f);
    CarBasisColumns(car, basis_forward, basis_right, basis_up);

    const Vec3 body_origin_bt = car.pos_bt;
    const Vec3 body_row0(basis_forward.x, basis_right.x, basis_up.x);
    const Vec3 body_row1(basis_forward.y, basis_right.y, basis_up.y);
    const Vec3 body_row2(basis_forward.z, basis_right.z, basis_up.z);
    const Vec3 child_origin_bt = cfg.hitbox_offset * kUUToBT;
    const Vec3 child_world_origin_bt = BulletStyleTransformPointBt(
        body_row0,
        body_row1,
        body_row2,
        body_origin_bt,
        child_origin_bt);
    const Vec3 support_dir(
        BulletStyleDot3Bt(-inward_normal, basis_forward),
        BulletStyleDot3Bt(-inward_normal, basis_right),
        BulletStyleDot3Bt(-inward_normal, basis_up));
    const Vec3 plane_origin_bt = BulletStaticPlaneOriginBt(inward_normal);
    const Vec3 convex_origin_in_plane_bt = child_world_origin_bt - plane_origin_bt;

    const Vec3 half_extent_bt = BulletBoxHalfExtentWithMarginBt(cfg);
    const Vec3 vtx(
        support_dir.x >= 0.0f ? half_extent_bt.x : -half_extent_bt.x,
        support_dir.y >= 0.0f ? half_extent_bt.y : -half_extent_bt.y,
        support_dir.z >= 0.0f ? half_extent_bt.z : -half_extent_bt.z);
    const Vec3 vtx_in_plane = BulletStyleTransformPointBt(
        body_row0,
        body_row1,
        body_row2,
        convex_origin_in_plane_bt,
        vtx);
    const float distance =
        BulletStyleDot3Bt(inward_normal, vtx_in_plane);
    const Vec3 vtx_in_plane_projected = vtx_in_plane - inward_normal * distance;
    const Vec3 point_on_b = vtx_in_plane_projected + plane_origin_bt;
    const Vec3 point_on_a = point_on_b + inward_normal * distance;
    const Vec3 local_a = BulletStyleInvTransformPointBt(
        basis_forward,
        basis_right,
        basis_up,
        body_origin_bt,
        point_on_a);

    const Vec3 position_world_on_a = BulletStyleTransformPointBt(
        body_row0,
        body_row1,
        body_row2,
        body_origin_bt,
        local_a);
    const Vec3 local_b = point_on_b - plane_origin_bt;
    const Vec3 position_world_on_b = local_b + plane_origin_bt;
    const float distance_refreshed =
        BulletStyleDot3Bt(position_world_on_a - position_world_on_b, inward_normal);
    const Vec3 rel_pos1 = position_world_on_a - body_origin_bt;

    out.support_dir_local_bt = support_dir;
    out.support_local_bt = vtx;
    out.child_origin_bt = child_origin_bt;
    out.child_world_origin_bt = child_world_origin_bt;
    out.position_world_on_a_bt = position_world_on_a;
    out.local_point_a_bt = local_a;
    out.rel_pos1_bt = rel_pos1;
    out.distance_bt = distance_refreshed;
    out.add_distance_bt = distance;
    out.local_point_b_bt = local_b;
    return out;
}

// Retained car-world plane contacts true.
// v3 keeps one Bullet persistent manifold per (car, static body) pair across ticks:
// collision_dispatcher.rs near_callback_with_bounds refreshes the cached points with
// the current body transform, then merges the fresh convex-plane point
// (persistent_manifold.rs refresh_contact_points / merge_contact_points /
// sort_cached_points). Plane points keep local_point_a in the car body frame and
// local_point_b relative to the plane body origin (BulletStaticPlaneOriginBt) in
// point_on_b_bt.
ROCKETGPU_HD void CarWorldRetainedPlanePointWorld(
    const CarWorldPlaneRetainedPoint& pt,
    const Vec3& normal,
    const Vec3& body_row0,
    const Vec3& body_row1,
    const Vec3& body_row2,
    const Vec3& body_origin_bt,
    const Vec3& plane_origin_bt,
    Vec3& out_pwa_bt,
    Vec3& out_pwb_bt,
    float& out_distance_bt) {
    out_pwa_bt = BulletStyleTransformPointBt(body_row0, body_row1, body_row2, body_origin_bt, pt.local_point_a_bt);
    out_pwb_bt = pt.point_on_b_bt + plane_origin_bt;
    out_distance_bt = BulletStyleDot3Bt(out_pwa_bt - out_pwb_bt, normal);
}

// btPersistentManifold::removeContactPoint: the last point moves into the freed slot (with its cached world
// anchor). The vacated slot is past count and never read again.
ROCKETGPU_HD void CarWorldRetainedRemovePoint(CarWorldPlaneRetainedManifold& manifold, Vec3 world_a_bt[4], int idx) {
    const int last = static_cast<int>(manifold.count) - 1;
    if (idx != last) {
        manifold.points[idx] = manifold.points[last];
        world_a_bt[idx] = world_a_bt[last];
    }
    manifold.count = static_cast<uint8_t>(last);
}

// refresh_contact_points: one world transform per point. The cull pass reuses the anchors of the distance pass
// (same inputs, same operations), and world_a_bt[i] keeps the world anchor of the point left in slot i.
ROCKETGPU_HD void RefreshCarWorldRetainedPlaneManifold(
    CarWorldPlaneRetainedManifold& manifold,
    const Vec3& normal,
    const Vec3& body_row0,
    const Vec3& body_row1,
    const Vec3& body_row2,
    const Vec3& body_origin_bt,
    const Vec3& plane_origin_bt,
    float threshold_bt,
    Vec3 world_a_bt[4]) {
    Vec3 world_b_bt[4];
    for (int i = 0; i < manifold.count; ++i) {
        CarWorldRetainedPlanePointWorld(
            manifold.points[i], normal, body_row0, body_row1, body_row2, body_origin_bt, plane_origin_bt,
            world_a_bt[i], world_b_bt[i], manifold.points[i].distance_bt);
    }
    // Descending: a removal only moves an already-visited point into slot i, so slot i below is still point i.
    for (int i = static_cast<int>(manifold.count) - 1; i >= 0; --i) {
        const float distance_bt = manifold.points[i].distance_bt;
        if (distance_bt > threshold_bt) {
            CarWorldRetainedRemovePoint(manifold, world_a_bt, i);
            continue;
        }
        const Vec3 projected_point = world_a_bt[i] - normal * distance_bt;
        const Vec3 projected_difference = world_b_bt[i] - projected_point;
        if (Dot(projected_difference, projected_difference) > threshold_bt * threshold_bt) {
            CarWorldRetainedRemovePoint(manifold, world_a_bt, i);
        }
    }
}

ROCKETGPU_HD int CarWorldRetainedSortCachedPoints(
    const CarWorldPlaneRetainedManifold& manifold,
    const CarWorldPlaneRetainedPoint& pt) {
    int max_penetration_index = 4;
    float max_penetration = pt.distance_bt;
    for (int i = 0; i < 4; ++i) {
        if (manifold.points[i].distance_bt < max_penetration) {
            max_penetration_index = i;
            max_penetration = manifold.points[i].distance_bt;
        }
    }
    const auto res = [&](int a, int b, int c) {
        return LengthSq(Cross(
            pt.local_point_a_bt - manifold.points[a].local_point_a_bt,
            manifold.points[b].local_point_a_bt - manifold.points[c].local_point_a_bt));
    };
    const float res0 = max_penetration_index != 0 ? res(1, 3, 2) : 0.0f;
    const float res1 = max_penetration_index != 1 ? res(0, 3, 2) : 0.0f;
    const float res2 = max_penetration_index != 2 ? res(0, 3, 1) : 0.0f;
    const float res3 = max_penetration_index != 3 ? res(0, 2, 1) : 0.0f;
    int biggest_area = 0;
    float max_area = res0;
    if (res1 > max_area) {
        biggest_area = 1;
        max_area = res1;
    }
    if (res2 > max_area) {
        biggest_area = 2;
        max_area = res2;
    }
    if (res3 > max_area) {
        biggest_area = 3;
    }
    return biggest_area;
}

// Returns the slot the point was written to.
ROCKETGPU_HD int MergeCarWorldRetainedPlanePoint(
    CarWorldPlaneRetainedManifold& manifold,
    const CarWorldPlaneRetainedPoint& pt,
    float threshold_bt) {
    if (manifold.count == 0) {
        manifold.points[0] = pt;
        manifold.count = 1;
        return 0;
    }
    float shortest_dist = threshold_bt * threshold_bt;
    int nearest_point = -1;
    for (int i = 0; i < manifold.count; ++i) {
        const Vec3 diff = manifold.points[i].local_point_a_bt - pt.local_point_a_bt;
        const float distance_sq = Dot(diff, diff);
        if (distance_sq < shortest_dist) {
            shortest_dist = distance_sq;
            nearest_point = i;
        }
    }
    if (nearest_point >= 0) {
        manifold.points[nearest_point] = pt;
        return nearest_point;
    }
    if (manifold.count < 4) {
        const int slot = manifold.count;
        manifold.points[slot] = pt;
        ++manifold.count;
        return slot;
    }
    const int slot = CarWorldRetainedSortCachedPoints(manifold, pt);
    manifold.points[slot] = pt;
    return slot;
}

ROCKETGPU_HD void AppendCarWorldPlaneContactRows(
    const CarState& car,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    const Vec3& inward_normal,
    float boundary,
    const Vec3& external_linear_delta,
    const Vec3& external_angular_delta,
    ContactAccumulator& acc,
    CarWorldSplitContactRow* split_rows,
    int& split_row_count,
    CarWorldVelocityContactRow* normal_rows,
    int& normal_row_count,
    CarWorldFrictionContactRow* friction_rows,
    int& friction_row_count,
    float dt,
    bool has_bt_plane = false,
    EnvStridedArray<SharedContactOrderGroup> local_world_order_groups = nullptr,
    int* local_world_order_group_count = nullptr,
    int car_idx = -1,
    CarWorldPlaneRetainedManifold* retained_manifold = nullptr) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kBTToUU = 50.0f;
    Vec3 basis_forward(1.0f, 0.0f, 0.0f);
    Vec3 basis_right(0.0f, 1.0f, 0.0f);
    Vec3 basis_up(0.0f, 0.0f, 1.0f);
    CarBasisColumns(car, basis_forward, basis_right, basis_up);

    const Vec3 support_dir_world = -inward_normal;
    const Vec3 support_dir_local(
        Dot(support_dir_world, basis_forward),
        Dot(support_dir_world, basis_right),
        Dot(support_dir_world, basis_up));
    const Vec3 half_extent_bt = BulletBoxHalfExtentWithMarginBt(cfg);
    const Vec3 support_local_bt =
        cfg.hitbox_offset * kUUToBT +
        Vec3(
            support_dir_local.x >= 0.0f ? half_extent_bt.x : -half_extent_bt.x,
            support_dir_local.y >= 0.0f ? half_extent_bt.y : -half_extent_bt.y,
            support_dir_local.z >= 0.0f ? half_extent_bt.z : -half_extent_bt.z);
    const Vec3 rel_contact_bt =
        basis_forward * support_local_bt.x +
        basis_right * support_local_bt.y +
        basis_up * support_local_bt.z;
    const Vec3 support_world_bt = car.pos_bt + rel_contact_bt;
    const CarWorldPlaneContactGeometryBt bullet_plane_geom =
        has_bt_plane
            ? ComputeBulletStyleCarWorldPlaneContactGeometryBt(car, cfg, inward_normal, boundary)
            : CarWorldPlaneContactGeometryBt{};
    const Vec3 row_rel_contact_bt =
        has_bt_plane ? bullet_plane_geom.rel_pos1_bt : rel_contact_bt;
    const int source_tag = CarWorldPlaneSourceTag(inward_normal, boundary);
    const float margin_bt =
        has_bt_plane ? bullet_plane_geom.distance_bt : Dot(support_world_bt, inward_normal) - boundary * kUUToBT;

    const float threshold_bt = has_bt_plane ? CarContactBreakingThresholdUU(cfg) * kUUToBT : 0.0f;

    const float add_margin_bt = has_bt_plane ? bullet_plane_geom.add_distance_bt : margin_bt;
    CarWorldPlaneRetainedManifold* const retained =
        (has_bt_plane) ? retained_manifold : nullptr;
    const Vec3 retained_body_row0(basis_forward.x, basis_right.x, basis_up.x);
    const Vec3 retained_body_row1(basis_forward.y, basis_right.y, basis_up.y);
    const Vec3 retained_body_row2(basis_forward.z, basis_right.z, basis_up.z);
    const Vec3 retained_plane_origin_bt = BulletStaticPlaneOriginBt(inward_normal);
    // World anchors of the retained points after the refresh (slot-aligned), reused for the emitted rows.
    Vec3 retained_world_a_bt[4];
    if (retained && retained->count > 0) {
        RefreshCarWorldRetainedPlaneManifold(
            *retained, inward_normal, retained_body_row0, retained_body_row1, retained_body_row2,
            car.pos_bt, retained_plane_origin_bt, threshold_bt, retained_world_a_bt);
    }
    if (add_margin_bt >= threshold_bt) {
        if (retained) {
            // No fresh point: every retained point is a box vertex no closer than the
            // support vertex, so the refresh above culls it (distance > threshold).
            retained->count = 0;
        }

        constexpr float kBulletGContactBreakingThresholdBt = 0.02f;
        constexpr float kBulletStaticPlaneAabbHalfSlabBt = 0.2f;
        const float broadphase_window_bt =
            threshold_bt + kBulletStaticPlaneAabbHalfSlabBt + 2.0f * kBulletGContactBreakingThresholdBt;
        if (has_bt_plane && add_margin_bt < broadphase_window_bt &&
            local_world_order_groups && local_world_order_group_count && car_idx >= 0) {
            AppendSharedContactOrderGroup(
                local_world_order_groups,
                *local_world_order_group_count,
                SharedContactOrderBucket::World,
                car_idx,
                -1,
                source_tag,
                normal_row_count,
                0,
                friction_row_count,
                0,
                *local_world_order_group_count);
        }
        return;
    }

    const float penetration = -margin_bt * kBTToUU;

    acc.add(inward_normal, Max(penetration, 0.001f));

    if (has_bt_plane && margin_bt > threshold_bt && !retained) {
        return;
    }

    Vec3 emit_rel_contact_bt[4];
    float emit_distance_bt[4];
    int emit_count = 0;
    if (retained) {
        int fresh_slot = -1;
        if (!(margin_bt > threshold_bt)) {
            CarWorldPlaneRetainedPoint fresh_point;
            fresh_point.local_point_a_bt = bullet_plane_geom.local_point_a_bt;
            fresh_point.point_on_b_bt = bullet_plane_geom.local_point_b_bt;
            fresh_point.distance_bt = bullet_plane_geom.distance_bt;
            fresh_slot = MergeCarWorldRetainedPlanePoint(*retained, fresh_point, threshold_bt);
        }
        // Refreshed points already hold this transform's anchor and distance; only the merged point is new.
        for (int i = 0; i < retained->count; ++i) {
            Vec3 pwa;
            if (i == fresh_slot) {
                Vec3 pwb;
                CarWorldRetainedPlanePointWorld(
                    retained->points[i], inward_normal, retained_body_row0, retained_body_row1, retained_body_row2,
                    car.pos_bt, retained_plane_origin_bt, pwa, pwb, emit_distance_bt[i]);
            } else {
                pwa = retained_world_a_bt[i];
                emit_distance_bt[i] = retained->points[i].distance_bt;
            }
            emit_rel_contact_bt[i] = pwa - car.pos_bt;
        }
        emit_count = retained->count;
    } else {
        emit_rel_contact_bt[0] = row_rel_contact_bt;
        emit_distance_bt[0] = has_bt_plane ? bullet_plane_geom.distance_bt : margin_bt;
        emit_count = 1;
    }
    const int group_normal_row_start = normal_row_count;
    const int group_friction_row_start = friction_row_count;
    for (int emit_idx = 0; emit_idx < emit_count; ++emit_idx) {
    const Vec3 row_rel_contact_bt = emit_rel_contact_bt[emit_idx];
    const float penetration = retained ? -emit_distance_bt[emit_idx] * kBTToUU : -margin_bt * kBTToUU;
    const Vec3 rel_contact = row_rel_contact_bt * kBTToUU;
    if (split_rows && split_row_count < kMaxCarWorldSplitContactRows) {
        CarWorldSplitContactRow& split_row = split_rows[split_row_count];
        const Vec3 split_angular_bt_native = CarWorldContactAngularComponentBtNative(
            car, cfg, mutator.car_mass, row_rel_contact_bt, inward_normal);

        const float plane_split_pen_bt = -emit_distance_bt[emit_idx];
        InitializeCarWorldSplitContactRow(
            split_row,
            penetration,
            inward_normal,
            1.0f / mutator.car_mass,
            rel_contact,
            ApplyInverseBodyInertiaWorld(car, cfg, mutator.car_mass, Cross(rel_contact, inward_normal)),
            dt,
            &row_rel_contact_bt,
            &split_angular_bt_native,
            has_bt_plane ? &plane_split_pen_bt : nullptr);
        if (split_row.valid) {
            split_row.source_tag = source_tag;
            ++split_row_count;
        }
    }

    const Vec3 normal_restitution_rel_vel = VelocityAtPoint(car, rel_contact);
    const Vec3 normal_base_rel_vel = VelocityAtPoint(car, rel_contact, external_linear_delta, external_angular_delta);
    const Vec3 tangent_basis_rel_vel = VelocityAtPoint(car, rel_contact, external_linear_delta, external_angular_delta);
    const Vec3 friction_base_rel_vel = VelocityAtPoint(car, rel_contact, external_linear_delta, Vec3(0.0f, 0.0f, 0.0f));
    CarWorldVelocityTraceInputs velocity_trace_inputs{
        car.vel,
        car.ang_vel,
        external_linear_delta,
        external_angular_delta,
        car.pos,
    };
    velocity_trace_inputs.rel_contact_bt = row_rel_contact_bt;
    velocity_trace_inputs.has_rel_contact_bt = 1;
    velocity_trace_inputs.linear_velocity_bt = car.vel_bt;
    velocity_trace_inputs.car_pos_bt = car.pos_bt;
    velocity_trace_inputs.external_force_impulse_bt = car.ext_force_impulse_bt;
    velocity_trace_inputs.has_bt_inputs = car.ext_force_impulse_bt_valid;
    velocity_trace_inputs.external_torque_impulse = car.ext_torque_impulse;
    velocity_trace_inputs.restitution_no_velocity_threshold = true;
#if !defined(__CUDA_ARCH__)
    PopulateCarWorldVelocityTraceInertia(
        velocity_trace_inputs,
        car,
        cfg,
        mutator.car_mass,
        rel_contact,
        inward_normal);
#endif
    const Vec3 angular_component_bt_native = CarWorldContactAngularComponentBtNative(
        car,
        cfg,
        mutator.car_mass,
        row_rel_contact_bt,
        inward_normal);
    const Vec3 angular_component = angular_component_bt_native * (1.0f / 50.0f);
    Vec3 tangent(0.0f, 0.0f, 0.0f);
    if (car.ext_force_impulse_bt_valid && car.basis_authoritative) {
        const Vec3 vel_nd_bt = car.vel_bt + car.ext_force_impulse_bt + Cross(car.ang_vel + car.ext_torque_impulse, row_rel_contact_bt);
        const float rel_vel_n_bt = Dot(inward_normal, vel_nd_bt);
        const Vec3 lateral_bt = vel_nd_bt - inward_normal * rel_vel_n_bt;
        const float lat_rel_vel = LengthSq(lateral_bt);
        constexpr float kBulletSimdEpsilon = 1.1920928955078125e-7f;
        tangent = lat_rel_vel > kBulletSimdEpsilon
            ? lateral_bt * (1.0f / SqrtScalar(lat_rel_vel))
            : BulletPlaneSpace1First(inward_normal);
    } else {
        tangent = tangent_basis_rel_vel - inward_normal * Dot(tangent_basis_rel_vel, inward_normal);
        const float tangent_len = Length(tangent);
        tangent = tangent_len > 1e-6f ? (tangent / tangent_len) : OrthonormalTangent(inward_normal);
    }
    const Vec3 tangent_angular_component_bt_native = CarWorldContactAngularComponentBtNative(
        car,
        cfg,
        mutator.car_mass,
        row_rel_contact_bt,
        tangent);
    const Vec3 tangent_angular_component = tangent_angular_component_bt_native * (1.0f / 50.0f);
    const int normal_row_start = normal_row_count;
    const int friction_row_start = friction_row_count;
    AppendCarWorldVelocityContactRows(
        normal_rows,
        normal_row_count,
        friction_rows,
        friction_row_count,
        inward_normal,
        mutator.arena_restitution,
        mutator.arena_friction,
        1.0f / mutator.car_mass,
        rel_contact,
        angular_component,
        tangent_angular_component,
        normal_restitution_rel_vel,
        normal_base_rel_vel,
        tangent_basis_rel_vel,
        friction_base_rel_vel,
        &velocity_trace_inputs,
        &angular_component_bt_native,
        &tangent_angular_component_bt_native,
        &tangent);
    for (int row_idx = normal_row_start; row_idx < normal_row_count; ++row_idx) {
        normal_rows[row_idx].source_tag = source_tag;
    }
    for (int row_idx = friction_row_start; row_idx < friction_row_count; ++row_idx) {
        friction_rows[row_idx].source_tag = source_tag;
    }
    }
    const int normal_row_start = group_normal_row_start;
    const int friction_row_start = group_friction_row_start;
    const int appended_normal_count = normal_row_count - normal_row_start;
    const int appended_friction_count = friction_row_count - friction_row_start;
    bool record_order_group = true;
    if (local_world_order_groups && local_world_order_group_count && *local_world_order_group_count > 0 &&
        IsBtFloorCeilingPlaneSourceTag(source_tag)) {
        const SharedContactOrderGroup& previous_group =
            local_world_order_groups[*local_world_order_group_count - 1];

        record_order_group = previous_group.source_tag != 101;
    }
    if (record_order_group && appended_normal_count > 0 &&
        local_world_order_groups && local_world_order_group_count && car_idx >= 0) {
        AppendSharedContactOrderGroup(
            local_world_order_groups,
            *local_world_order_group_count,
            SharedContactOrderBucket::World,
            car_idx,
            -1,
            source_tag,
            normal_row_start,
            appended_normal_count,
            appended_friction_count > 0 ? friction_row_start : -1,
            appended_friction_count,
            *local_world_order_group_count);
    }
}

ROCKETGPU_HD bool CarWorldTriangleAgainstAabbBullet(
    const ArenaMeshTriangle& tri,
    const Vec3& aabb_min,
    const Vec3& aabb_max) {
    if (Min(Min(tri.a.z, tri.b.z), tri.c.z) > aabb_max.z) {
        return false;
    }
    if (Max(Max(tri.a.z, tri.b.z), tri.c.z) < aabb_min.z) {
        return false;
    }
    if (Min(Min(tri.a.x, tri.b.x), tri.c.x) > aabb_max.x) {
        return false;
    }
    if (Max(Max(tri.a.x, tri.b.x), tri.c.x) < aabb_min.x) {
        return false;
    }
    if (Min(Min(tri.a.y, tri.b.y), tri.c.y) > aabb_max.y) {
        return false;
    }
    if (Max(Max(tri.a.y, tri.b.y), tri.c.y) < aabb_min.y) {
        return false;
    }
    return true;
}

// A car-triangle GJK/EPA contact with the internal-edge adjust applied; raw_* are the values before it.
struct CarWorldTriangleHit {
    Vec3 normal;
    Vec3 raw_normal;
    Vec3 point_on_b_bt;
    Vec3 raw_point_on_b_bt;
    float distance_bt;
    uint8_t hit;
    uint8_t internal_edge_adjusted;
};

ROCKETGPU_HD CarWorldTriangleHit ComputeCarWorldTriangleHit(
    const CarState& car,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    const ArenaMeshTriangle* mesh_triangles,
    int32_t tri_idx,
    float contact_threshold_bt,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch) {
    const ArenaMeshTriangle& tri = mesh_triangles[tri_idx];
    CarWorldTriangleHit out;
    out.hit = 0;
    out.internal_edge_adjusted = 0;
    Vec3 contact_normal(0.0f, 0.0f, 1.0f);
    float distance_bt = 0.0f;
    Vec3 point_on_a_bt(0.0f, 0.0f, 0.0f);
    Vec3 point_on_b_bt(0.0f, 0.0f, 0.0f);
    const bool cal_frame_gjk = mutator.cal_arena_local_tris != nullptr;
    const ArenaMeshTriangle& gjk_tri = cal_frame_gjk ? mutator.cal_arena_local_tris[tri_idx] : tri;
    const float gjk_tri_origin_y_bt =
        cal_frame_gjk ? static_cast<float>(gjk_tri.cal_origin_sign) * kGoalMeshOriginYBt : 0.0f;
    if (!ComputeCarArenaTriangleContact(car, cfg, gjk_tri, contact_threshold_bt, contact_normal, distance_bt,
                                        &point_on_a_bt, &point_on_b_bt, gjk_epa_scratch, nullptr, nullptr, 0,
                                        gjk_tri_origin_y_bt)) {
        return out;
    }
    contact_normal = ArenaTriangleFaceNormal(gjk_tri);
    out.raw_normal = contact_normal;
    out.raw_point_on_b_bt = point_on_b_bt;
    const Vec3 cp_point_on_a_bt = point_on_b_bt + contact_normal * distance_bt;
    if (EnableArenaTriangleInternalEdgeAdjust()) {
        out.internal_edge_adjusted = MaybeAdjustArenaTriangleInternalEdgeContact(
            tri, cp_point_on_a_bt, distance_bt, point_on_b_bt, contact_normal, nullptr,
            mutator.game_mode == static_cast<uint8_t>(Mode::Soccar),
            mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift,
            cal_frame_gjk ? &mutator.cal_arena_local_tris[tri_idx] : nullptr) ? 1 : 0;
    }
    out.normal = contact_normal;
    out.point_on_b_bt = point_on_b_bt;
    out.distance_bt = distance_bt;
    out.hit = 1;
    return out;
}

// Car-triangle pairs staged for a thread each. The gather pass lists, in walk order, every triangle of a staged
// (car, mesh) pair that passes the plane test; the contact pass computes each pair's hit; the merge pass folds the
// hits into the pair's candidates in list order, as the inline walk does.
struct CarWorldTrianglePairs {
    int32_t* car_idx;
    int32_t* tri_idx;
    CarWorldTriangleHit* hits;
    int32_t* count;
    int32_t capacity;
    int32_t car_stride;  // global car index = env_idx * car_stride + slot
};
enum : int { kTrianglePairsNone = 0, kTrianglePairsGather = 1, kTrianglePairsMerge = 2 };
constexpr int kCarWorldMeshGatherTriangles = 128;

struct MeshContactCandidate {
    int32_t tri_idx = -1;
    int32_t mesh_world_idx = -1;
    Vec3 normal = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 tri_normal = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 rel_contact = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 rel_contact_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 local_point_a_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 cache_local_point_a_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 point_on_b_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 raw_normal = Vec3(0.0f, 0.0f, 1.0f);
    uint8_t internal_edge_adjusted = 0;
    float distance_bt = 0.0f;

    float add_distance_bt = 0.0f;
};

constexpr int kCarWorldMeshStageCandidates = 4;
constexpr int kCarWorldMeshStageMeshes = 16;
constexpr int kCarWorldMeshStageFingerprintFloats = 25;
struct CarWorldMeshCandidateStageMesh {
    MeshContactCandidate candidates[kCarWorldMeshStageCandidates];
    int32_t candidate_count = 0;
    int32_t broadphase_hits = 0;
    int32_t pair_start = 0;  // CarWorldTrianglePairs entries from the gather pass; -1 count: none
    int32_t pair_count = -1;
    uint8_t has_collected_normal = 0;
    Vec3 last_collected_normal = Vec3(0.0f, 0.0f, 1.0f);
};
struct CarWorldMeshCandidateStage {
    float fingerprint[kCarWorldMeshStageFingerprintFloats] = {};
    int32_t valid = 0;
    int32_t mesh_count = 0;
    uint32_t mesh_mask = 0;  // meshes with a staged entry; the others yield no candidates
    CarWorldMeshCandidateStageMesh meshes[kCarWorldMeshStageMeshes];
};

#if defined(__CUDACC__)
extern __device__ CarWorldMeshCandidateStage* g_rocketgpu_car_world_stage;
extern __device__ int g_rocketgpu_car_world_stage_stride;
#endif

ROCKETGPU_HD CarWorldMeshCandidateStage* CarWorldMeshCandidateStageFor(int env_idx, int car_slot) {
#if defined(__CUDA_ARCH__)
    CarWorldMeshCandidateStage* base = g_rocketgpu_car_world_stage;
    const int stride = g_rocketgpu_car_world_stage_stride;
    if (base == nullptr || stride <= 0 || car_slot < 0 || car_slot >= stride || env_idx < 0) {
        return nullptr;
    }
    return base + static_cast<size_t>(env_idx) * static_cast<size_t>(stride) + car_slot;
#else
    (void)env_idx;
    (void)car_slot;
    return nullptr;
#endif
}

ROCKETGPU_HD void FillCarWorldMeshCandidateFingerprint(const CarState& car, const CarConfig& cfg, float* fp) {
    Vec3 axis_x(1.0f, 0.0f, 0.0f);
    Vec3 axis_y(0.0f, 1.0f, 0.0f);
    Vec3 axis_z(0.0f, 0.0f, 1.0f);
    CarBasisColumns(car, axis_x, axis_y, axis_z);
    fp[0] = car.pos.x; fp[1] = car.pos.y; fp[2] = car.pos.z;
    fp[3] = car.pos_bt.x; fp[4] = car.pos_bt.y; fp[5] = car.pos_bt.z;
    fp[6] = car.rot.w; fp[7] = car.rot.x; fp[8] = car.rot.y; fp[9] = car.rot.z;
    fp[10] = axis_x.x; fp[11] = axis_x.y; fp[12] = axis_x.z;
    fp[13] = axis_y.x; fp[14] = axis_y.y; fp[15] = axis_y.z;
    fp[16] = axis_z.x; fp[17] = axis_z.y; fp[18] = axis_z.z;
    fp[19] = cfg.hitbox_size.x; fp[20] = cfg.hitbox_size.y; fp[21] = cfg.hitbox_size.z;
    fp[22] = cfg.hitbox_offset.x; fp[23] = cfg.hitbox_offset.y; fp[24] = cfg.hitbox_offset.z;
}

ROCKETGPU_HD bool CarWorldMeshCandidateStageMatches(const CarWorldMeshCandidateStage& stage,
                                                const CarState& car,
                                                const CarConfig& cfg) {
    float fp[kCarWorldMeshStageFingerprintFloats];
    FillCarWorldMeshCandidateFingerprint(car, cfg, fp);
    for (int i = 0; i < kCarWorldMeshStageFingerprintFloats; ++i) {
        union { float f; uint32_t u; } a, b;
        a.f = fp[i];
        b.f = stage.fingerprint[i];
        if (a.u != b.u) {
            return false;
        }
    }
    return true;
}

ROCKETGPU_HD int SelectCarWorldMeshContactReplaceIndexBulletStyle(
    const MeshContactCandidate* cache,
    int cache_count,
    const MeshContactCandidate& pt) {
    if (cache_count < 4) {
        return cache_count;
    }

    for (int i = 0; i < 4; ++i) {
    }

    int max_penetration_index = -1;
    float max_penetration = pt.add_distance_bt;
    for (int i = 0; i < 4; ++i) {
        if (cache[i].add_distance_bt < max_penetration) {
            max_penetration_index = i;
            max_penetration = cache[i].add_distance_bt;
        }
    }

    float res0 = 0.0f;
    float res1 = 0.0f;
    float res2 = 0.0f;
    float res3 = 0.0f;

    if (max_penetration_index != 0) {
        const Vec3 a0 = pt.cache_local_point_a_bt - cache[1].cache_local_point_a_bt;
        const Vec3 b0 = cache[3].cache_local_point_a_bt - cache[2].cache_local_point_a_bt;
        res0 = LengthSq(Cross(a0, b0));
    }
    if (max_penetration_index != 1) {
        const Vec3 a1 = pt.cache_local_point_a_bt - cache[0].cache_local_point_a_bt;
        const Vec3 b1 = cache[3].cache_local_point_a_bt - cache[2].cache_local_point_a_bt;
        res1 = LengthSq(Cross(a1, b1));
    }
    if (max_penetration_index != 2) {
        const Vec3 a2 = pt.cache_local_point_a_bt - cache[0].cache_local_point_a_bt;
        const Vec3 b2 = cache[3].cache_local_point_a_bt - cache[1].cache_local_point_a_bt;
        res2 = LengthSq(Cross(a2, b2));
    }
    if (max_penetration_index != 3) {
        const Vec3 a3 = pt.cache_local_point_a_bt - cache[0].cache_local_point_a_bt;
        const Vec3 b3 = cache[2].cache_local_point_a_bt - cache[1].cache_local_point_a_bt;
        res3 = LengthSq(Cross(a3, b3));
    }

    int biggest_area = 0;
    float biggest_value = res0;
    if (res1 > biggest_value) {
        biggest_area = 1;
        biggest_value = res1;
    }
    if (res2 > biggest_value) {
        biggest_area = 2;
        biggest_value = res2;
    }
    if (res3 > biggest_value) {
        biggest_area = 3;
    }
    return biggest_area;
}

// The car hitbox as the mesh query sees it (Bullet units): basis, box center and the rotated box's AABB with margin.
struct CarWorldMeshQueryFrame {
    Vec3 axis_x = Vec3(1.0f, 0.0f, 0.0f);
    Vec3 axis_y = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 axis_z = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 box_center_bt;
    Vec3 box_half_extent_with_margin_bt;
    Vec3 aabb_min_bt;
    Vec3 aabb_max_bt;
};

ROCKETGPU_HD CarWorldMeshQueryFrame ComputeCarWorldMeshQueryFrame(const CarState& car, const CarConfig& cfg) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    CarWorldMeshQueryFrame f;
    CarBasisColumns(car, f.axis_x, f.axis_y, f.axis_z);
    const Vec3& axis_x = f.axis_x;
    const Vec3& axis_y = f.axis_y;
    const Vec3& axis_z = f.axis_z;
    const Vec3 body_row0(axis_x.x, axis_y.x, axis_z.x);
    const Vec3 body_row1(axis_x.y, axis_y.y, axis_z.y);
    const Vec3 body_row2(axis_x.z, axis_y.z, axis_z.z);
    f.box_center_bt = BulletStyleTransformPointBt(
        body_row0,
        body_row1,
        body_row2,
        car.pos_bt,
        cfg.hitbox_offset * kUUToBT);
    const Vec3 h = BulletBoxHalfExtentWithMarginBt(cfg);
    f.box_half_extent_with_margin_bt = h;
    const Vec3 extent_bt(
        Abs(axis_x.x) * h.x + Abs(axis_y.x) * h.y + Abs(axis_z.x) * h.z,
        Abs(axis_x.y) * h.x + Abs(axis_y.y) * h.y + Abs(axis_z.y) * h.z,
        Abs(axis_x.z) * h.x + Abs(axis_y.z) * h.y + Abs(axis_z.z) * h.z);
    f.aabb_min_bt = f.box_center_bt - extent_bt;
    f.aabb_max_bt = f.box_center_bt + extent_bt;
    return f;
}

ROCKETGPU_HD bool CarWorldMeshInfoUsable(
    const ArenaMeshBvhMeshInfo& mesh_info,
    int32_t mesh_bvh_quantization_count,
    int32_t mesh_bvh_node_count) {
    return mesh_info.mesh_index >= 0 &&
           mesh_info.mesh_index < mesh_bvh_quantization_count &&
           mesh_info.node_offset >= 0 &&
           mesh_info.node_count > 0 &&
           mesh_info.node_offset + mesh_info.node_count <= mesh_bvh_node_count;
}

// A mesh whose bounds the query AABB misses yields no candidates.
ROCKETGPU_HD bool CarWorldMeshAabbMayOverlap(
    const ArenaMeshBvhMeshInfo& mesh_info,
    const Vec3& aabb_min_bt,
    const Vec3& aabb_max_bt) {
    constexpr float kMeshSkipMarginBt = 1.0e-3f;
    return aabb_min_bt.x <= mesh_info.aabb_max.x + kMeshSkipMarginBt &&
           aabb_max_bt.x >= mesh_info.aabb_min.x - kMeshSkipMarginBt &&
           aabb_min_bt.y <= mesh_info.aabb_max.y + kMeshSkipMarginBt &&
           aabb_max_bt.y >= mesh_info.aabb_min.y - kMeshSkipMarginBt &&
           aabb_min_bt.z <= mesh_info.aabb_max.z + kMeshSkipMarginBt &&
           aabb_max_bt.z >= mesh_info.aabb_min.z - kMeshSkipMarginBt;
}

ROCKETGPU_HD void AppendCarWorldMeshContactRows(
    const CarState& car,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    const Vec3& external_linear_delta,
    const Vec3& external_angular_delta,
    ContactAccumulator& acc,
    CarWorldSplitContactRow* split_rows,
    int& split_row_count,
    CarWorldVelocityContactRow* normal_rows,
    int& normal_row_count,
    CarWorldFrictionContactRow* friction_rows,
    int& friction_row_count,
    float dt,
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
    int env_idx = -1,
    int car_idx = -1,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch = nullptr,
    EnvStridedArray<SharedContactOrderGroup> local_world_order_groups = nullptr,
    int* local_world_order_group_count = nullptr,
    const ArenaBroadphaseGrid* broadphase_grid = nullptr,
    float broadphase_dt = 0.0f,
    CarWorldMeshCandidateStage* stage_out = nullptr,
    const CarWorldMeshCandidateStage* stage_in = nullptr,
    int stage_mesh = -1,
    const ArenaMeshLeafGrid* leaf_grid = nullptr,
    const CarWorldTrianglePairs* triangle_pairs = nullptr,
    int triangle_pairs_pass = kTrianglePairsNone) {
    if (mesh_triangles == nullptr || mesh_triangle_count <= 0) {
        return;
    }

    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kBTToUU = 50.0f;
    constexpr int kMaxMeshCandidates = 4;
    constexpr int kMaxMeshRows = 4;

    const float contact_threshold_bt = CarContactBreakingThresholdUU(cfg) * kUUToBT;
    const CarWorldMeshQueryFrame query_frame = ComputeCarWorldMeshQueryFrame(car, cfg);
    const Vec3 axis_x = query_frame.axis_x;
    const Vec3 axis_y = query_frame.axis_y;
    const Vec3 axis_z = query_frame.axis_z;
    const Vec3 box_center_bt = query_frame.box_center_bt;
    const Vec3 box_half_extent_with_margin_bt = query_frame.box_half_extent_with_margin_bt;
    const Vec3 aabb_min_bt = query_frame.aabb_min_bt;
    const Vec3 aabb_max_bt = query_frame.aabb_max_bt;
    const BulletBroadphaseProxy car_broadphase_proxy =
        broadphase_grid != nullptr
            ? ComputeBulletCarProxyAabb(car, cfg, broadphase_dt, *broadphase_grid, NativeSseMath(mutator))
            : BulletBroadphaseProxy{};

    const Vec3 car_pos_bt = car.pos_bt;

    Vec3 last_collected_normal(0.0f, 0.0f, 1.0f);
    bool has_collected_normal = false;

    uint16_t quantized_query_aabb_min[3] = {0, 0, 0};
    uint16_t quantized_query_aabb_max[3] = {0, 0, 0};

    int current_mesh_broadphase_hits = 0;
    auto append_local_world_order_group = [&](int source_tag,
                                              int normal_start,
                                              int normal_count_for_group,
                                              int friction_start,
                                              int friction_count_for_group) {
        if (!local_world_order_groups || !local_world_order_group_count) {
            return;
        }
        const int local_order = *local_world_order_group_count;
        AppendSharedContactOrderGroup(
            local_world_order_groups,
            *local_world_order_group_count,
            SharedContactOrderBucket::World,
            car_idx,
            -1,
            source_tag,
            normal_count_for_group > 0 ? normal_start : -1,
            normal_count_for_group,
            friction_count_for_group > 0 ? friction_start : -1,
            friction_count_for_group,
            local_order);
    };

    auto emit_candidates = [&](
        MeshContactCandidate* candidates,
        int candidate_count) {
        int emitted_rows = 0;
        for (int candidate_idx = 0; candidate_idx < candidate_count && emitted_rows < kMaxMeshRows; ++candidate_idx) {
            MeshContactCandidate candidate = candidates[candidate_idx];
            const float penetration_uu = Max(0.0f, -candidate.distance_bt) * kBTToUU;
            acc.add(candidate.normal, Max(penetration_uu, 0.001f));

            if (split_rows && split_row_count < kMaxCarWorldSplitContactRows) {
                CarWorldSplitContactRow& split_row = split_rows[split_row_count];
                const Vec3 mesh_split_angular_bt_native = CarWorldContactAngularComponentBtNative(
                    car, cfg, mutator.car_mass, candidate.rel_contact_bt, candidate.normal);
                const float mesh_split_pen_bt = -candidate.distance_bt;
                InitializeCarWorldSplitContactRow(
                    split_row,
                    penetration_uu,
                    candidate.normal,
                    1.0f / mutator.car_mass,
                    candidate.rel_contact,
                    ApplyInverseBodyInertiaWorld(car, cfg, mutator.car_mass, Cross(candidate.rel_contact, candidate.normal)),
                    dt,
                    &candidate.rel_contact_bt,
                    &mesh_split_angular_bt_native,
                    &mesh_split_pen_bt);
                if (split_row.valid) {
                    split_row.source_tag = 100 + Max(candidate.mesh_world_idx, 0);
                    ++split_row_count;
                }
            }

            const Vec3 normal_restitution_rel_vel = VelocityAtPoint(car, candidate.rel_contact);
            const Vec3 normal_base_rel_vel = VelocityAtPoint(car, candidate.rel_contact, external_linear_delta, external_angular_delta);
            const Vec3 tangent_basis_rel_vel = normal_base_rel_vel;
            const Vec3 friction_base_rel_vel = VelocityAtPoint(car, candidate.rel_contact, external_linear_delta, Vec3(0.0f, 0.0f, 0.0f));
            CarWorldVelocityTraceInputs velocity_trace_inputs{
                car.vel,
                car.ang_vel,
                external_linear_delta,
                external_angular_delta,
                car.pos,
            };
            velocity_trace_inputs.rel_contact_bt = candidate.rel_contact_bt;
            velocity_trace_inputs.has_rel_contact_bt = 1;
            velocity_trace_inputs.linear_velocity_bt = car.vel_bt;
            velocity_trace_inputs.car_pos_bt = car.pos_bt;
            velocity_trace_inputs.external_force_impulse_bt = car.ext_force_impulse_bt;
            velocity_trace_inputs.has_bt_inputs = car.ext_force_impulse_bt_valid;
            velocity_trace_inputs.external_torque_impulse = car.ext_torque_impulse;
            velocity_trace_inputs.restitution_no_velocity_threshold = true;
#if !defined(__CUDA_ARCH__)
            PopulateCarWorldVelocityTraceInertia(
                velocity_trace_inputs,
                car,
                cfg,
                mutator.car_mass,
                candidate.rel_contact,
                candidate.normal);
#endif
            const Vec3 angular_component_bt_native = CarWorldContactAngularComponentBtNative(
                car,
                cfg,
                mutator.car_mass,
                candidate.rel_contact_bt,
                candidate.normal);
            const Vec3 angular_component = angular_component_bt_native * (1.0f / 50.0f);
            Vec3 tangent(0.0f, 0.0f, 0.0f);
            if (car.ext_force_impulse_bt_valid && car.basis_authoritative) {
                const Vec3 vel_nd_bt = car.vel_bt + car.ext_force_impulse_bt + Cross(car.ang_vel + car.ext_torque_impulse, candidate.rel_contact_bt);
                const float rel_vel_n_bt = Dot(candidate.normal, vel_nd_bt);
                const Vec3 lateral_bt = vel_nd_bt - candidate.normal * rel_vel_n_bt;
                const float lat_rel_vel = LengthSq(lateral_bt);
                constexpr float kBulletSimdEpsilon = 1.1920928955078125e-7f;
                tangent = lat_rel_vel > kBulletSimdEpsilon
                    ? lateral_bt * (1.0f / SqrtScalar(lat_rel_vel))
                    : BulletPlaneSpace1First(candidate.normal);
            } else {
                tangent = tangent_basis_rel_vel - candidate.normal * Dot(tangent_basis_rel_vel, candidate.normal);
                const float tangent_len = Length(tangent);
                tangent = tangent_len > 1e-6f ? (tangent / tangent_len) : OrthonormalTangent(candidate.normal);
            }
            const Vec3 tangent_angular_component_bt_native = CarWorldContactAngularComponentBtNative(
                car,
                cfg,
                mutator.car_mass,
                candidate.rel_contact_bt,
                tangent);
            const Vec3 tangent_angular_component = tangent_angular_component_bt_native * (1.0f / 50.0f);
            const int normal_row_start = normal_row_count;
            const int friction_row_start = friction_row_count;
            const int mesh_source_tag = 100 + Max(candidate.mesh_world_idx, 0);
            AppendCarWorldVelocityContactRows(
                normal_rows,
                normal_row_count,
                friction_rows,
                friction_row_count,
                candidate.normal,
                mutator.arena_restitution,
                mutator.arena_friction,
                1.0f / mutator.car_mass,
                candidate.rel_contact,
                angular_component,
                tangent_angular_component,
                normal_restitution_rel_vel,
                normal_base_rel_vel,
                tangent_basis_rel_vel,
                friction_base_rel_vel,
                &velocity_trace_inputs,
                &angular_component_bt_native,
                &tangent_angular_component_bt_native,
                &tangent);
            for (int row_idx = normal_row_start; row_idx < normal_row_count; ++row_idx) {
                normal_rows[row_idx].source_tag = mesh_source_tag;
            }
            const auto debug_candidate_matches_target_triangle = [&]() {
                return false;
            };
            const bool debug_emit_target_triangle =
                debug_candidate_matches_target_triangle();
            for (int row_idx = friction_row_start; row_idx < friction_row_count; ++row_idx) {
                friction_rows[row_idx].source_tag = mesh_source_tag;
            }
            ++emitted_rows;
        }
    };

    // Gather pass: the triangles of this (car, mesh) pair that pass the plane test, in walk order.
    int32_t gathered_tris[kCarWorldMeshGatherTriangles];
    int gathered_count = 0;
    auto consider_triangle = [&](int32_t tri_idx, MeshContactCandidate* candidates, int& candidate_count,
                                 const CarWorldTriangleHit* precomputed = nullptr) {
        const ArenaMeshTriangle& tri = mesh_triangles[tri_idx];
        if (precomputed == nullptr) {
            if (!CarWorldTriangleAgainstAabbBullet(tri, aabb_min_bt, aabb_max_bt)) {
                return;
            }
            ++current_mesh_broadphase_hits;
            const Vec3 triangle_normal_world = Normalize(tri.normal, Vec3(0.0f, 0.0f, 1.0f));
            const float center_proj = Dot(triangle_normal_world, box_center_bt);
            const float support_extent =
                Abs(Dot(triangle_normal_world, axis_x)) * box_half_extent_with_margin_bt.x +
                Abs(Dot(triangle_normal_world, axis_y)) * box_half_extent_with_margin_bt.y +
                Abs(Dot(triangle_normal_world, axis_z)) * box_half_extent_with_margin_bt.z;
            const float tri_proj = Dot(triangle_normal_world, tri.a);
            const float separation_along_normal = tri_proj - (center_proj + support_extent);
            const float separation_against_normal = (center_proj - support_extent) - tri_proj;
            if (separation_along_normal > contact_threshold_bt ||
                separation_against_normal > contact_threshold_bt) {
                return;
            }
            if (triangle_pairs_pass == kTrianglePairsGather) {
                if (gathered_count < kCarWorldMeshGatherTriangles) {
                    gathered_tris[gathered_count] = tri_idx;
                }
                ++gathered_count;
                return;
            }
        }
        const CarWorldTriangleHit hit =
            precomputed != nullptr
                ? *precomputed
                : ComputeCarWorldTriangleHit(car, cfg, mutator, mesh_triangles, tri_idx, contact_threshold_bt, gjk_epa_scratch);
        if (hit.hit == 0) {
            return;
        }
        const Vec3 contact_normal = hit.normal;
        const float distance_bt = hit.distance_bt;
        const Vec3 point_on_b_bt = hit.point_on_b_bt;
        const Vec3 contact_normal_pre_adjust = hit.raw_normal;
        const Vec3 point_on_b_bt_pre_adjust = hit.raw_point_on_b_bt;
        const bool internal_edge_adjusted = hit.internal_edge_adjusted != 0;

        MeshContactCandidate candidate;
        candidate.tri_idx = tri_idx;
        candidate.mesh_world_idx = tri.mesh_index;
        candidate.normal = contact_normal;
        candidate.raw_normal = contact_normal_pre_adjust;
        candidate.internal_edge_adjusted = internal_edge_adjusted ? 1 : 0;
        candidate.tri_normal = tri.normal;

        candidate.distance_bt = distance_bt;
        candidate.add_distance_bt = distance_bt;
        Vec3 manifold_point_a_bt =
            point_on_b_bt_pre_adjust + contact_normal_pre_adjust * candidate.distance_bt;

        candidate.point_on_b_bt = point_on_b_bt;
        candidate.local_point_a_bt = CarBodyLocalPointBt(car, manifold_point_a_bt);

        candidate.rel_contact_bt =
            CarBodyWorldPointBt(car, candidate.local_point_a_bt) - car_pos_bt;
        candidate.rel_contact = candidate.rel_contact_bt * kBTToUU;
        candidate.cache_local_point_a_bt = candidate.local_point_a_bt;

        if (candidate.distance_bt > contact_threshold_bt) {
            return;
        }

        const Vec3 live_add_contact_local_point_a_bt = candidate.local_point_a_bt;
        Vec3 live_add_contact_body_forward(1.0f, 0.0f, 0.0f);
        Vec3 live_add_contact_body_right(0.0f, 1.0f, 0.0f);
        Vec3 live_add_contact_body_up(0.0f, 0.0f, 1.0f);
        GetBasisVectors(
            car.rot,
            live_add_contact_body_forward,
            live_add_contact_body_right,
            live_add_contact_body_up);
        const Vec3 live_add_contact_body_origin_bt = car.pos_bt;
        const Vec3 live_add_contact_roundtrip_point_a_bt =
            CarBodyWorldPointBt(car, live_add_contact_local_point_a_bt);
        const Vec3 live_add_contact_body_row2(
            live_add_contact_body_forward.z,
            live_add_contact_body_right.z,
            live_add_contact_body_up.z);

        const Vec3 refreshed_point_on_a_bt = live_add_contact_roundtrip_point_a_bt;
        const float refreshed_distance_bt =
            Dot(refreshed_point_on_a_bt - candidate.point_on_b_bt, candidate.normal);
        candidate.distance_bt = refreshed_distance_bt;
        candidate.rel_contact_bt = refreshed_point_on_a_bt - live_add_contact_body_origin_bt;
        candidate.rel_contact = candidate.rel_contact_bt * kBTToUU;

        int merge_idx = -1;
        {
            float shortest_dist = contact_threshold_bt * contact_threshold_bt;
            for (int i = 0; i < candidate_count; ++i) {
                const Vec3 diff_a = candidates[i].cache_local_point_a_bt - candidate.cache_local_point_a_bt;
                const float dist_to_point = Dot(diff_a, diff_a);
                if (dist_to_point < shortest_dist) {
                    shortest_dist = dist_to_point;
                    merge_idx = i;
                }
            }
        }
        if (candidate.distance_bt > contact_threshold_bt) {
            if (merge_idx >= 0) {
                for (int i = merge_idx; i + 1 < candidate_count; ++i) {
                    candidates[i] = candidates[i + 1];
                }
                --candidate_count;
            }

            last_collected_normal = contact_normal_pre_adjust;
            has_collected_normal = true;
            return;
        }

        last_collected_normal = contact_normal_pre_adjust;
        has_collected_normal = true;

        if (merge_idx >= 0) {
            candidates[merge_idx] = candidate;
            return;
        }
        if (candidate_count < kMaxMeshCandidates) {
            candidates[candidate_count] = candidate;
            ++candidate_count;
            return;
        }

        const int replace_idx = SelectCarWorldMeshContactReplaceIndexBulletStyle(candidates, candidate_count, candidate);
        candidates[replace_idx] = candidate;
    };

    const bool have_bvh_traversal =
        mesh_bvh_quantization != nullptr &&
        mesh_bvh_meshes != nullptr &&
        mesh_bvh_nodes != nullptr &&
        mesh_bvh_mesh_count > 0 &&
        mesh_bvh_node_count > 0;

    static_assert(kMaxMeshCandidates == kCarWorldMeshStageCandidates,
                  "CarWorldMeshCandidateStage holds every mesh candidate");
    // stage_mesh >= 0 stages that one mesh and leaves the stage header to the caller.
    if (stage_out != nullptr) {
        if (stage_mesh < 0) {
            stage_out->valid = 0;
            stage_out->mesh_count = 0;
        }
        if (!have_bvh_traversal || mesh_bvh_mesh_count > kCarWorldMeshStageMeshes) {
            return;
        }
        if (stage_mesh < 0) {
            stage_out->mesh_count = mesh_bvh_mesh_count;
            stage_out->mesh_mask = ~0u;
        }
    }
    if (stage_in != nullptr &&
        (stage_in->valid == 0 || !have_bvh_traversal || stage_in->mesh_count != mesh_bvh_mesh_count)) {
        stage_in = nullptr;
    }
    if (have_bvh_traversal) {
        UninitArray<MeshContactCandidate, kMaxMeshCandidates> candidates;
        const int mesh_begin = stage_mesh >= 0 ? stage_mesh : 0;
        const int mesh_end = stage_mesh >= 0 && stage_mesh < mesh_bvh_mesh_count ? stage_mesh + 1 : (stage_mesh >= 0 ? 0 : mesh_bvh_mesh_count);
        for (int mesh_info_idx = mesh_begin; mesh_info_idx < mesh_end; ++mesh_info_idx) {
            const ArenaMeshBvhMeshInfo& mesh_info = mesh_bvh_meshes[mesh_info_idx];
            if (!CarWorldMeshInfoUsable(mesh_info, mesh_bvh_quantization_count, mesh_bvh_node_count)) {
                continue;
            }
            // Not staged: the query AABB misses this mesh, so it has no candidates; only a Bullet manifold with it
            // still gives it an (empty) order group, as below.
            if (stage_in != nullptr && broadphase_grid != nullptr && ((stage_in->mesh_mask >> mesh_info_idx) & 1u) == 0) {
                if (BulletCarMeshManifoldExists(car_broadphase_proxy, *broadphase_grid, mesh_info)) {
                    append_local_world_order_group(
                        100 + Max(mesh_info.mesh_index, 0), normal_row_count, 0, friction_row_count, 0);
                }
                continue;
            }

            int candidate_count = 0;
            current_mesh_broadphase_hits = 0;
            gathered_count = 0;
            const ArenaMeshBvhQuantization& quantization = mesh_bvh_quantization[mesh_info.mesh_index];
            QuantizeArenaMeshBvhPoint(quantization, aabb_min_bt, 0, quantized_query_aabb_min);
            QuantizeArenaMeshBvhPoint(quantization, aabb_max_bt, 1, quantized_query_aabb_max);

            int mesh_subtree_overlap_count = 0;
            const ArenaMeshBvhSubtreeInfo* mesh_subtrees = nullptr;
            int mesh_subtree_count = 0;
            if (mesh_bvh_subtrees != nullptr &&
                mesh_info.subtree_offset >= 0 &&
                mesh_info.subtree_count > 0 &&
                mesh_info.subtree_offset + mesh_info.subtree_count <= mesh_bvh_subtree_count) {
                mesh_subtrees = mesh_bvh_subtrees + mesh_info.subtree_offset;
                mesh_subtree_count = mesh_info.subtree_count;

                // The overlap count only gates the order group, which staging never emits.
                for (int subtree_idx = 0; broadphase_grid == nullptr && stage_out == nullptr && subtree_idx < mesh_subtree_count; ++subtree_idx) {
                    const ArenaMeshBvhSubtreeInfo& subtree = mesh_subtrees[subtree_idx];
                    const bool subtree_overlap = TestQuantizedAabbAgainstQuantizedAabb(
                        quantized_query_aabb_min,
                        quantized_query_aabb_max,
                        subtree.quantized_aabb_min,
                        subtree.quantized_aabb_max);
                    if (subtree_overlap) {
                        ++mesh_subtree_overlap_count;
                    }
                }
            }

            const bool mesh_aabb_may_overlap = CarWorldMeshAabbMayOverlap(mesh_info, aabb_min_bt, aabb_max_bt);
            if (stage_in != nullptr) {
                if (((stage_in->mesh_mask >> mesh_info_idx) & 1u) == 0) {
                    // Not staged: the query AABB misses this mesh, so it has no candidates.
                } else {
                const CarWorldMeshCandidateStageMesh& staged = stage_in->meshes[mesh_info_idx];
                candidate_count = staged.candidate_count < kMaxMeshCandidates ? staged.candidate_count : kMaxMeshCandidates;
                for (int c = 0; c < candidate_count; ++c) {
                    candidates[c] = staged.candidates[c];
                }
                current_mesh_broadphase_hits = staged.broadphase_hits;
                if (staged.has_collected_normal != 0) {
                    has_collected_normal = true;
                    last_collected_normal = staged.last_collected_normal;
                }
                }
            } else if (mesh_aabb_may_overlap && triangle_pairs_pass == kTrianglePairsMerge &&
                       stage_out->meshes[mesh_info_idx].pair_count >= 0) {
                const CarWorldMeshCandidateStageMesh& gathered = stage_out->meshes[mesh_info_idx];
                for (int k = 0; k < gathered.pair_count; ++k) {
                    consider_triangle(triangle_pairs->tri_idx[gathered.pair_start + k], candidates, candidate_count,
                                      &triangle_pairs->hits[gathered.pair_start + k]);
                }
                current_mesh_broadphase_hits = gathered.broadphase_hits;
            } else if (mesh_aabb_may_overlap) {
            const auto visit_leaf = [&](int32_t local_triangle_index) {
                if (local_triangle_index < 0 || local_triangle_index >= mesh_info.triangle_count) {
                    return;
                }
                consider_triangle(mesh_info.triangle_start + local_triangle_index, candidates, candidate_count);
            };
            if (!WalkArenaMeshLeafGrid(
                    leaf_grid, mesh_info_idx, quantized_query_aabb_min, quantized_query_aabb_max, visit_leaf)) {
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
                    visit_leaf);
            }
            }
            if (stage_out != nullptr && triangle_pairs_pass == kTrianglePairsGather) {
                CarWorldMeshCandidateStageMesh& staged = stage_out->meshes[mesh_info_idx];
                staged.broadphase_hits = current_mesh_broadphase_hits;
                staged.pair_count = -1;  // the merge pass walks this pair itself
#if defined(__CUDA_ARCH__)
                if (gathered_count <= kCarWorldMeshGatherTriangles) {
                    int start = 0;
                    if (gathered_count > 0) {
                        start = atomicAdd(triangle_pairs->count, 0);
                        while (true) {
                            if (start > triangle_pairs->capacity - gathered_count) {
                                start = -1;
                                break;
                            }
                            const int observed = atomicCAS(triangle_pairs->count, start, start + gathered_count);
                            if (observed == start) {
                                break;
                            }
                            start = observed;
                        }
                    }
                    if (start >= 0) {
                        for (int k = 0; k < gathered_count; ++k) {
                            triangle_pairs->car_idx[start + k] = env_idx * triangle_pairs->car_stride + car_idx;
                            triangle_pairs->tri_idx[start + k] = gathered_tris[k];
                        }
                        staged.pair_start = start;
                        staged.pair_count = gathered_count;
                    }
                }
#endif
                continue;
            }
            if (stage_out != nullptr) {
                CarWorldMeshCandidateStageMesh& staged = stage_out->meshes[mesh_info_idx];
                staged.candidate_count = candidate_count;
                for (int c = 0; c < candidate_count && c < kMaxMeshCandidates; ++c) {
                    staged.candidates[c] = candidates[c];
                }
                staged.broadphase_hits = current_mesh_broadphase_hits;
                staged.has_collected_normal = has_collected_normal ? 1 : 0;
                staged.last_collected_normal = last_collected_normal;
                continue;
            }

            const int group_normal_start = normal_row_count;
            const int group_friction_start = friction_row_count;
            emit_candidates(candidates, candidate_count);
            const int group_normal_count = normal_row_count - group_normal_start;
            const int group_friction_count = friction_row_count - group_friction_start;

            const bool mesh_manifold_exists =
                broadphase_grid != nullptr &&
                BulletCarMeshManifoldExists(car_broadphase_proxy, *broadphase_grid, mesh_info);

            if (current_mesh_broadphase_hits > 0 ||
                group_normal_count > 0 ||
                (broadphase_grid != nullptr ? mesh_manifold_exists : (mesh_subtree_overlap_count > 0))) {
                append_local_world_order_group(
                    100 + Max(mesh_info.mesh_index, 0),
                    group_normal_start,
                    group_normal_count,
                    group_friction_start,
                    group_friction_count);
            }
        }
    } else {
        UninitArray<MeshContactCandidate, kMaxMeshCandidates> candidates;
        int candidate_count = 0;

        for (int32_t tri_idx = 0; tri_idx < mesh_triangle_count; ++tri_idx) {
            const ArenaMeshTriangle& tri = mesh_triangles[tri_idx];
            const bool have_quantized_mesh =
                mesh_bvh_quantization != nullptr &&
                tri.mesh_index < static_cast<uint8_t>(mesh_bvh_quantization_count);
            if (have_quantized_mesh) {
                const ArenaMeshBvhQuantization& quantization = mesh_bvh_quantization[tri.mesh_index];
                QuantizeArenaMeshBvhPoint(quantization, aabb_min_bt, 0, quantized_query_aabb_min);
                QuantizeArenaMeshBvhPoint(quantization, aabb_max_bt, 1, quantized_query_aabb_max);
                if (!TestQuantizedAabbAgainstQuantizedAabb(
                        quantized_query_aabb_min,
                        quantized_query_aabb_max,
                        tri.quantized_aabb_min,
                        tri.quantized_aabb_max)) {
                    continue;
                }
            } else if (tri.max.x < aabb_min_bt.x || tri.min.x > aabb_max_bt.x ||
                       tri.max.y < aabb_min_bt.y || tri.min.y > aabb_max_bt.y ||
                       tri.max.z < aabb_min_bt.z || tri.min.z > aabb_max_bt.z) {
                continue;
            }
            consider_triangle(tri_idx, candidates, candidate_count);
        }
        emit_candidates(candidates, candidate_count);
    }

    if (stage_out != nullptr) {
        if (stage_mesh < 0) {
            stage_out->valid = 1;
        }
        return;
    }
    if (has_collected_normal) {
        acc.hit = 1;
        acc.latest_normal = last_collected_normal;
    }
}
