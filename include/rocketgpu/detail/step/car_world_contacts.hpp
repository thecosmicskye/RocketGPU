struct ArenaMeshTriangle;

constexpr bool EnableArenaTriangleInternalEdgeAdjust();

ROCKETGPU_HD void BuildInvInertiaTensorWorldBt(
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    Vec3& out_row0,
    Vec3& out_row1,
    Vec3& out_row2);
ROCKETGPU_HD Vec3 Mul3x3Vec(const Vec3& row0, const Vec3& row1, const Vec3& row2, const Vec3& v);

ROCKETGPU_HD Vec3 ArenaTriangleFaceNormal(const ArenaMeshTriangle& tri);
ROCKETGPU_HD bool MaybeAdjustArenaTriangleInternalEdgeContact(
    const ArenaMeshTriangle& tri,
    const Vec3& point_on_a_bt,
    float distance_bt,
    Vec3& io_point_on_b_bt,
    Vec3& io_normal,
    const uint32_t* native_table = nullptr, int native_shift = 0,
    const ArenaMeshTriangle* cal_game_tri = nullptr);

ROCKETGPU_HD bool SoccarPlaneExtentContainsInPlaneBt(const Vec3& inward_normal, const Vec3& in_plane_bt, float threshold_bt) {
    const bool horizontal = Abs(inward_normal.z) > 0.5f;
    const float lim_x = threshold_bt + (horizontal ? 4358.144f / 50.0f : 1245.184f / 50.0f);
    const float lim_y = threshold_bt + 6225.92f / 50.0f;
    return !(in_plane_bt.x < -lim_x || in_plane_bt.y < -lim_y || lim_x < in_plane_bt.x || lim_y < in_plane_bt.y);
}

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
        return Vec3(-ArenaExtentX() * kUUToBT, 0.0f, 992.0f * kUUToBT);
    }
    if (inward_normal.x < -0.5f) {
        return Vec3(ArenaExtentX() * kUUToBT, 0.0f, 992.0f * kUUToBT);
    }
    return Vec3(0.0f, 0.0f, 0.0f);
}

struct GamePlaneXfBt {
    Vec3 r0 = Vec3(1.0f, 0.0f, 0.0f);
    Vec3 r1 = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 r2 = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 o = Vec3(0.0f, 0.0f, 0.0f);
};

ROCKETGPU_HD GamePlaneXfBt GameSoccarPlaneXfBt(const Vec3& axis_normal, const Vec3& origin_bt) {
    GamePlaneXfBt xf;
    xf.o = origin_bt;
    if (axis_normal.z < -0.5f) {
        xf.r0 = Vec3(-1.0f, -0x1.777a5cp-24f, -0x1.135bd8p-47f);
        xf.r1 = Vec3(-0x1.777a5cp-24f, 1.0f, 0x1.777a5cp-24f);
        xf.r2 = Vec3(-0x1p-71f, 0x1.777a5cp-24f, -1.0f);
    } else if (axis_normal.x > 0.5f) {
        xf.r0 = Vec3(0.0f, 0.0f, 1.0f);
        xf.r1 = Vec3(0.0f, 1.0f, 0.0f);
        xf.r2 = Vec3(-1.0f, 0.0f, 0.0f);
    } else if (axis_normal.x < -0.5f) {
        xf.r0 = Vec3(0x1.8p-23f, -0x1.8p-70f, -0x1.fffffep-1f);
        xf.r1 = Vec3(0x1.135bd8p-47f, 1.0f, 0.0f);
        xf.r2 = Vec3(0x1.fffffep-1f, -0x1.135bd6p-47f, 0x1.8p-23f);
    }
    return xf;
}

ROCKETGPU_HD Vec3 GamePlaneNormalBt(const GamePlaneXfBt& xf) {
    return Vec3(
        (xf.r0.y * 0.0f + xf.r0.x * 0.0f) + xf.r0.z,
        (xf.r1.y * 0.0f + xf.r1.x * 0.0f) + xf.r1.z,
        (xf.r2.y * 0.0f + xf.r2.x * 0.0f) + xf.r2.z);
}

ROCKETGPU_HD Vec3 GamePlanePointWorldBt(const GamePlaneXfBt& xf, const Vec3& offset_bt) {
    const Vec3 local_b(
        BulletStyleDot3Bt(Vec3(xf.r0.x, xf.r1.x, xf.r2.x), offset_bt),
        BulletStyleDot3Bt(Vec3(xf.r0.y, xf.r1.y, xf.r2.y), offset_bt),
        BulletStyleDot3Bt(Vec3(xf.r0.z, xf.r1.z, xf.r2.z), offset_bt));
    return BulletStyleTransformPointBt(xf.r0, xf.r1, xf.r2, xf.o, local_b);
}

ROCKETGPU_HD Vec3 GamePlaneVertexInPlaneBt(
    const GamePlaneXfBt& xf, const Vec3& c0, const Vec3& c1, const Vec3& c2, const Vec3& co, const Vec3& vtx) {
    const Vec3 neg_origin(-xf.o.x, -xf.o.y, -xf.o.z);
    const auto in_plane = [&](const Vec3& col) {
        const float m0 = BulletStyleDot3Bt(col, c0);
        const float m1 = BulletStyleDot3Bt(col, c1);
        const float m2 = BulletStyleDot3Bt(col, c2);
        const float mo = BulletStyleDot3Bt(col, co) + BulletStyleDot3Bt(col, neg_origin);
        return ((m0 * vtx.x + m1 * vtx.y) + m2 * vtx.z) + mo;
    };
    return Vec3(
        in_plane(Vec3(xf.r0.x, xf.r1.x, xf.r2.x)),
        in_plane(Vec3(xf.r0.y, xf.r1.y, xf.r2.y)),
        in_plane(Vec3(xf.r0.z, xf.r1.z, xf.r2.z)));
}

ROCKETGPU_HD Vec3 GamePlaneProjectedWorldBt(const GamePlaneXfBt& xf, const Vec3& in_plane) {
    return Vec3(
        ((xf.r0.x * in_plane.x + xf.r0.y * in_plane.y) + xf.r0.z * 0.0f) + xf.o.x,
        ((xf.r1.x * in_plane.x + xf.r1.y * in_plane.y) + xf.r1.z * 0.0f) + xf.o.y,
        ((xf.r2.x * in_plane.x + xf.r2.y * in_plane.y) + xf.r2.z * 0.0f) + xf.o.z);
}

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
    float& out_distance_bt,
    const GamePlaneXfBt* xf = nullptr) {
    out_pwa_bt = BulletStyleTransformPointBt(body_row0, body_row1, body_row2, body_origin_bt, pt.local_point_a_bt);
    out_pwb_bt = xf != nullptr ? GamePlanePointWorldBt(*xf, pt.point_on_b_bt) : pt.point_on_b_bt + plane_origin_bt;
    out_distance_bt = BulletStyleDot3Bt(out_pwa_bt - out_pwb_bt, normal);
}

ROCKETGPU_HD void CarWorldRetainedRemovePoint(CarWorldPlaneRetainedManifold& manifold, int idx) {
    const int last = static_cast<int>(manifold.count) - 1;
    if (idx != last) {
        manifold.points[idx] = manifold.points[last];
    }
    manifold.count = static_cast<uint8_t>(last);
}

ROCKETGPU_HD void RefreshCarWorldRetainedPlaneManifold(
    CarWorldPlaneRetainedManifold& manifold,
    const Vec3& normal,
    const Vec3& body_row0,
    const Vec3& body_row1,
    const Vec3& body_row2,
    const Vec3& body_origin_bt,
    const Vec3& plane_origin_bt,
    float threshold_bt,
    const GamePlaneXfBt* xf = nullptr) {
    for (int i = static_cast<int>(manifold.count) - 1; i >= 0; --i) {
        Vec3 world_a_bt(0.0f, 0.0f, 0.0f);
        Vec3 world_b_bt(0.0f, 0.0f, 0.0f);
        float distance_bt = 0.0f;
        CarWorldRetainedPlanePointWorld(
            manifold.points[i], normal, body_row0, body_row1, body_row2, body_origin_bt, plane_origin_bt,
            world_a_bt, world_b_bt, distance_bt, xf);
        manifold.points[i].distance_bt = distance_bt;
        if (distance_bt > threshold_bt) {
            CarWorldRetainedRemovePoint(manifold, i);
            continue;
        }
        const Vec3 projected_point = world_a_bt - normal * distance_bt;
        const Vec3 projected_difference = world_b_bt - projected_point;
        if (Dot(projected_difference, projected_difference) > threshold_bt * threshold_bt) {
            CarWorldRetainedRemovePoint(manifold, i);
        }
    }
}

template <typename Manifold, typename Point>
ROCKETGPU_HD int CarWorldRetainedSortCachedPoints(const Manifold& manifold, const Point& pt) {
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

template <typename Manifold, typename Point>
ROCKETGPU_HD int MergeCarWorldRetainedPlanePoint(Manifold& manifold, const Point& pt, float threshold_bt) {
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

#if defined(__CUDA_ARCH__) && defined(_MSC_VER)
#define ROCKETGPU_CAR_WORLD_ALWAYS_INLINE __forceinline__
#elif defined(__CUDA_ARCH__)
#define ROCKETGPU_CAR_WORLD_ALWAYS_INLINE __attribute__((always_inline))
#else
#define ROCKETGPU_CAR_WORLD_ALWAYS_INLINE
#endif
ROCKETGPU_HD void AppendCarWorldPlaneContactRows(
    const CarState& car,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    const Vec3& axis_normal,
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
    CarWorldPlaneRetainedManifold* retained_manifold = nullptr,
    bool defer_floor_ceiling_group = false,
    int goal_floor = -1,
    float precomputed_threshold_bt = -1.0f,
    const CarShapeCacheEntry* shape = nullptr) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kBTToUU = 50.0f;
    const bool game_plane_xf = has_bt_plane && mutator.game_mode == static_cast<uint8_t>(Mode::Soccar);
    const GamePlaneXfBt plane_xf = GameSoccarPlaneXfBt(
        axis_normal, goal_floor >= 0 ? GoalFloorOriginBt(goal_floor) : BulletStaticPlaneOriginBt(axis_normal));
    const GamePlaneXfBt* const xf = game_plane_xf ? &plane_xf : nullptr;
    const Vec3 inward_normal = game_plane_xf ? GamePlaneNormalBt(plane_xf) : axis_normal;
    Vec3 basis_forward(1.0f, 0.0f, 0.0f);
    Vec3 basis_right(0.0f, 1.0f, 0.0f);
    Vec3 basis_up(0.0f, 0.0f, 1.0f);
    CarBasisColumns(car, basis_forward, basis_right, basis_up);

    const Vec3 half_extent_bt = shape != nullptr ? shape->half_extent_bt : BulletBoxHalfExtentWithMarginBt(cfg);
    const Vec3 retained_plane_origin_bt =
        goal_floor >= 0 ? GoalFloorOriginBt(goal_floor) : BulletStaticPlaneOriginBt(inward_normal);
    const Vec3 retained_body_row0(basis_forward.x, basis_right.x, basis_up.x);
    const Vec3 retained_body_row1(basis_forward.y, basis_right.y, basis_up.y);
    const Vec3 retained_body_row2(basis_forward.z, basis_right.z, basis_up.z);

    Vec3 rel_contact_bt(0.0f, 0.0f, 0.0f);
    float non_bt_margin_bt = 0.0f;
    Vec3 bt_vtx_in_plane_frame(0.0f, 0.0f, 0.0f);
    float bt_add_distance = 0.0f;
    if (has_bt_plane) {
        const Vec3 child_origin_bt = cfg.hitbox_offset * kUUToBT;
        const Vec3 child_world_origin_bt = BulletStyleTransformPointBt(
            retained_body_row0,
            retained_body_row1,
            retained_body_row2,
            car.pos_bt,
            child_origin_bt);
        const Vec3 support_dir(
            BulletStyleDot3Bt(-inward_normal, basis_forward),
            BulletStyleDot3Bt(-inward_normal, basis_right),
            BulletStyleDot3Bt(-inward_normal, basis_up));
        const Vec3 vtx(
            support_dir.x >= 0.0f ? half_extent_bt.x : -half_extent_bt.x,
            support_dir.y >= 0.0f ? half_extent_bt.y : -half_extent_bt.y,
            support_dir.z >= 0.0f ? half_extent_bt.z : -half_extent_bt.z);
        if (xf != nullptr) {
            bt_vtx_in_plane_frame =
                GamePlaneVertexInPlaneBt(*xf, basis_forward, basis_right, basis_up, child_world_origin_bt, vtx);
            bt_add_distance = bt_vtx_in_plane_frame.z;
        } else {
            const Vec3 convex_origin_in_plane_bt = child_world_origin_bt - retained_plane_origin_bt;
            bt_vtx_in_plane_frame = BulletStyleTransformPointBt(
                retained_body_row0,
                retained_body_row1,
                retained_body_row2,
                convex_origin_in_plane_bt,
                vtx);
            bt_add_distance = BulletStyleDot3Bt(inward_normal, bt_vtx_in_plane_frame);
        }
    } else {
        const Vec3 support_dir_world = -inward_normal;
        const Vec3 support_dir_local(
            Dot(support_dir_world, basis_forward),
            Dot(support_dir_world, basis_right),
            Dot(support_dir_world, basis_up));
        const Vec3 support_local_bt =
            cfg.hitbox_offset * kUUToBT +
            Vec3(
                support_dir_local.x >= 0.0f ? half_extent_bt.x : -half_extent_bt.x,
                support_dir_local.y >= 0.0f ? half_extent_bt.y : -half_extent_bt.y,
                support_dir_local.z >= 0.0f ? half_extent_bt.z : -half_extent_bt.z);
        rel_contact_bt =
            basis_forward * support_local_bt.x +
            basis_right * support_local_bt.y +
            basis_up * support_local_bt.z;
        const Vec3 support_world_bt = car.pos_bt + rel_contact_bt;
        non_bt_margin_bt = Dot(support_world_bt, inward_normal) - boundary * kUUToBT;
    }

    const float threshold_bt = has_bt_plane
        ? (precomputed_threshold_bt >= 0.0f ? precomputed_threshold_bt : CarContactBreakingThresholdBt(cfg))
        : 0.0f;
    const float add_margin_bt = has_bt_plane ? bt_add_distance : non_bt_margin_bt;
    CarWorldPlaneRetainedManifold* const retained =
        (has_bt_plane) ? retained_manifold : nullptr;
    if (retained && retained->count > 0) {
        RefreshCarWorldRetainedPlaneManifold(
            *retained, inward_normal, retained_body_row0, retained_body_row1, retained_body_row2,
            car.pos_bt, retained_plane_origin_bt, threshold_bt, xf);
    }
    const bool within_extent =
        !(add_margin_bt >= threshold_bt) &&
        (!has_bt_plane || mutator.game_mode != static_cast<uint8_t>(Mode::Soccar) ||
         (goal_floor >= 0 ? GoalFloorExtentContainsBt(bt_vtx_in_plane_frame, threshold_bt)
                          : SoccarPlaneExtentContainsInPlaneBt(inward_normal, bt_vtx_in_plane_frame,
                                                               threshold_bt)));
    if (add_margin_bt >= threshold_bt || (!within_extent && (!retained || retained->count == 0))) {
        if (retained) {
            retained->count = 0;
        }

        constexpr float kBulletGContactBreakingThresholdBt = 0.02f;
        constexpr float kBulletStaticPlaneAabbHalfSlabBt = 0.2f;
        const float broadphase_window_bt =
            threshold_bt + kBulletStaticPlaneAabbHalfSlabBt + 2.0f * kBulletGContactBreakingThresholdBt;
        if (has_bt_plane && add_margin_bt < broadphase_window_bt &&
            local_world_order_groups && local_world_order_group_count && car_idx >= 0) {
            const int source_tag =
                goal_floor >= 0 ? kGoalFloorSourceTag + goal_floor : CarWorldPlaneSourceTag(inward_normal, boundary);
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

    const int source_tag =
        goal_floor >= 0 ? kGoalFloorSourceTag + goal_floor : CarWorldPlaneSourceTag(inward_normal, boundary);

    Vec3 bt_position_world_on_a(0.0f, 0.0f, 0.0f);
    Vec3 bt_local_point_a(0.0f, 0.0f, 0.0f);
    Vec3 bt_local_point_b(0.0f, 0.0f, 0.0f);
    Vec3 bt_rel_pos1(0.0f, 0.0f, 0.0f);
    float bt_distance_refreshed = 0.0f;
    if (has_bt_plane) {
        const Vec3 point_on_b = xf != nullptr
            ? GamePlaneProjectedWorldBt(*xf, bt_vtx_in_plane_frame)
            : (bt_vtx_in_plane_frame - inward_normal * bt_add_distance) + retained_plane_origin_bt;
        const Vec3 point_on_a = point_on_b + inward_normal * bt_add_distance;
        bt_local_point_a = BulletStyleInvTransformPointBt(
            basis_forward,
            basis_right,
            basis_up,
            car.pos_bt,
            point_on_a);
        bt_position_world_on_a = BulletStyleTransformPointBt(
            retained_body_row0,
            retained_body_row1,
            retained_body_row2,
            car.pos_bt,
            bt_local_point_a);
        bt_local_point_b = point_on_b - retained_plane_origin_bt;
        const Vec3 position_world_on_b =
            xf != nullptr ? GamePlanePointWorldBt(*xf, bt_local_point_b) : bt_local_point_b + retained_plane_origin_bt;
        bt_distance_refreshed =
            BulletStyleDot3Bt(bt_position_world_on_a - position_world_on_b, inward_normal);
        bt_rel_pos1 = bt_position_world_on_a - car.pos_bt;
    }

    const Vec3 row_rel_contact_bt_init = has_bt_plane ? bt_rel_pos1 : rel_contact_bt;
    const float margin_bt = has_bt_plane ? bt_distance_refreshed : non_bt_margin_bt;
    const float penetration = -margin_bt * kBTToUU;

    acc.add(inward_normal, Max(penetration, 0.001f));

    if (has_bt_plane && margin_bt > threshold_bt && !retained) {
        return;
    }

    int fresh_slot = -1;
    int emit_count = 1;
    if (retained) {
        if (!(margin_bt > threshold_bt) && within_extent) {
            CarWorldPlaneRetainedPoint fresh_point;
            fresh_point.local_point_a_bt = bt_local_point_a;
            fresh_point.point_on_b_bt = bt_local_point_b;
            fresh_point.distance_bt = bt_distance_refreshed;
            fresh_slot = MergeCarWorldRetainedPlanePoint(*retained, fresh_point, threshold_bt);
        }
        emit_count = retained->count;
    }
    const int group_normal_row_start = normal_row_count;
    const int group_friction_row_start = friction_row_count;
    Vec3 inv_r0, inv_r1, inv_r2;
    BuildInvInertiaTensorWorldBt(car, cfg, mutator.car_mass, inv_r0, inv_r1, inv_r2);
    const float inv_car_mass = 1.0f / mutator.car_mass;
    const bool bt_authoritative = car.ext_force_impulse_bt_valid && car.basis_authoritative;
    const Vec3 vel_plus_ext_bt = car.vel_bt + car.ext_force_impulse_bt;
    const Vec3 ang_vel_plus_ext = car.ang_vel + car.ext_torque_impulse;
    CarWorldVelocityTraceInputs velocity_trace_inputs{
        car.vel,
        car.ang_vel,
        external_linear_delta,
        external_angular_delta,
        car.pos,
    };
    velocity_trace_inputs.has_rel_contact_bt = 1;
    velocity_trace_inputs.linear_velocity_bt = car.vel_bt;
    velocity_trace_inputs.car_pos_bt = car.pos_bt;
    velocity_trace_inputs.external_force_impulse_bt = car.ext_force_impulse_bt;
    velocity_trace_inputs.has_bt_inputs = car.ext_force_impulse_bt_valid;
    velocity_trace_inputs.external_torque_impulse = car.ext_torque_impulse;
    velocity_trace_inputs.restitution_no_velocity_threshold = true;
    for (int emit_idx = 0; emit_idx < emit_count; ++emit_idx) {
    const float emit_dist_bt = !retained ? margin_bt
        : (emit_idx == fresh_slot ? bt_distance_refreshed : retained->points[emit_idx].distance_bt);
    const Vec3 row_rel_contact_bt = !retained ? row_rel_contact_bt_init
        : (emit_idx == fresh_slot ? bt_rel_pos1
            : (BulletStyleTransformPointBt(
                   retained_body_row0, retained_body_row1, retained_body_row2,
                   car.pos_bt, retained->points[emit_idx].local_point_a_bt) - car.pos_bt));
    const float penetration = retained ? -emit_dist_bt * kBTToUU : -margin_bt * kBTToUU;
    const Vec3 rel_contact = row_rel_contact_bt * kBTToUU;
    const Vec3 angular_component_bt_native = Mul3x3Vec(inv_r0, inv_r1, inv_r2, Cross(row_rel_contact_bt, inward_normal));
    if (split_rows && split_row_count < kMaxCarWorldSplitContactRows) {
        CarWorldSplitContactRow& split_row = split_rows[split_row_count];
        Vec3 split_angular_bt_native(0.0f, 0.0f, 0.0f);
        Vec3 split_angular(0.0f, 0.0f, 0.0f);
        if (!(penetration <= 0.0f || dt <= 0.0f)) {
            split_angular_bt_native = angular_component_bt_native;
            split_angular = ApplyInverseBodyInertiaWorld(car, cfg, mutator.car_mass, Cross(rel_contact, inward_normal));
        }

        const float plane_split_pen_bt = -emit_dist_bt;
        InitializeCarWorldSplitContactRow(
            split_row,
            penetration,
            inward_normal,
            inv_car_mass,
            rel_contact,
            split_angular,
            dt,
            &row_rel_contact_bt,
            &split_angular_bt_native,
            has_bt_plane ? &plane_split_pen_bt : nullptr);
        if (split_row.valid) {
            split_row.source_tag = source_tag;
            ++split_row_count;
        }
    }

    velocity_trace_inputs.rel_contact_bt = row_rel_contact_bt;
    const Vec3 angular_component = angular_component_bt_native * (1.0f / 50.0f);
    Vec3 normal_restitution_rel_vel(0.0f, 0.0f, 0.0f);
    Vec3 normal_base_rel_vel(0.0f, 0.0f, 0.0f);
    Vec3 tangent_basis_rel_vel(0.0f, 0.0f, 0.0f);
    Vec3 friction_base_rel_vel(0.0f, 0.0f, 0.0f);
    Vec3 tangent(0.0f, 0.0f, 0.0f);
    if (bt_authoritative) {
        const Vec3 vel_nd_bt = vel_plus_ext_bt + Cross(ang_vel_plus_ext, row_rel_contact_bt);
        const float rel_vel_n_bt = Dot(inward_normal, vel_nd_bt);
        const Vec3 lateral_bt = vel_nd_bt - inward_normal * rel_vel_n_bt;
        const float lat_rel_vel = LengthSq(lateral_bt);
        constexpr float kBulletSimdEpsilon = 1.1920928955078125e-7f;
        tangent = lat_rel_vel > kBulletSimdEpsilon
            ? lateral_bt * (1.0f / SqrtScalar(lat_rel_vel))
            : BulletPlaneSpace1First(inward_normal);
    } else {
        normal_restitution_rel_vel = VelocityAtPoint(car, rel_contact);
        normal_base_rel_vel = VelocityAtPoint(car, rel_contact, external_linear_delta, external_angular_delta);
        tangent_basis_rel_vel = normal_base_rel_vel;
        friction_base_rel_vel = VelocityAtPoint(car, rel_contact, external_linear_delta, Vec3(0.0f, 0.0f, 0.0f));
        tangent = tangent_basis_rel_vel - inward_normal * Dot(tangent_basis_rel_vel, inward_normal);
        const float tangent_len = Length(tangent);
        tangent = tangent_len > 1e-6f ? (tangent / tangent_len) : OrthonormalTangent(inward_normal);
    }
    const Vec3 tangent_angular_component_bt_native = Mul3x3Vec(inv_r0, inv_r1, inv_r2, Cross(row_rel_contact_bt, tangent));
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
        inv_car_mass,
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
    if (!defer_floor_ceiling_group && local_world_order_groups && local_world_order_group_count &&
        *local_world_order_group_count > 0 && IsBtFloorCeilingPlaneSourceTag(source_tag)) {
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
        if (defer_floor_ceiling_group && IsBtFloorCeilingPlaneSourceTag(source_tag)) {
            local_world_order_groups[*local_world_order_group_count - 1].valid = 2;
        }
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
    const ArenaMeshTriangle* mesh_triangles,
    const ArenaMeshTriangle* cal_arena_local_tris,
    const uint32_t* native_sphere_rsqrt_table,
    int native_sphere_rsqrt_shift,
    int32_t tri_idx,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch,
    float precomputed_threshold_bt = -1.0f) {
    const ArenaMeshTriangle& tri = mesh_triangles[tri_idx];
    CarWorldTriangleHit out;
    out.hit = 0;
    out.internal_edge_adjusted = 0;
    Vec3 contact_normal(0.0f, 0.0f, 1.0f);
    float distance_bt = 0.0f;
    Vec3 point_on_b_bt(0.0f, 0.0f, 0.0f);
    const bool cal_frame_gjk = cal_arena_local_tris != nullptr;
    const ArenaMeshTriangle& gjk_tri = cal_frame_gjk ? cal_arena_local_tris[tri_idx] : tri;
    const float gjk_tri_origin_y_bt =
        cal_frame_gjk ? static_cast<float>(gjk_tri.cal_origin_sign) * kGoalMeshOriginYBt : 0.0f;
    if (!ComputeCarArenaTriangleContact(car, cfg, gjk_tri, contact_normal, distance_bt,
                                        nullptr, &point_on_b_bt, gjk_epa_scratch, nullptr,
                                        gjk_tri_origin_y_bt) ||
        distance_bt > (precomputed_threshold_bt >= 0.0f ? precomputed_threshold_bt : CarContactBreakingThresholdBt(cfg))) {
        return out;
    }
    contact_normal = ArenaTriangleFaceNormal(gjk_tri);
    out.raw_normal = contact_normal;
    out.raw_point_on_b_bt = point_on_b_bt;
    const Vec3 cp_point_on_a_bt = point_on_b_bt + contact_normal * distance_bt;
    if (EnableArenaTriangleInternalEdgeAdjust()) {
        out.internal_edge_adjusted = MaybeAdjustArenaTriangleInternalEdgeContact(
            tri, cp_point_on_a_bt, distance_bt, point_on_b_bt, contact_normal,
            native_sphere_rsqrt_table, native_sphere_rsqrt_shift,
            cal_frame_gjk ? &cal_arena_local_tris[tri_idx] : nullptr) ? 1 : 0;
    }
    out.normal = contact_normal;
    out.point_on_b_bt = point_on_b_bt;
    out.distance_bt = distance_bt;
    out.hit = 1;
    return out;
}

ROCKETGPU_HD CarWorldTriangleHit ComputeCarWorldTriangleHit(
    const CarState& car,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    const ArenaMeshTriangle* mesh_triangles,
    int32_t tri_idx,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch,
    float precomputed_threshold_bt = -1.0f) {
    return ComputeCarWorldTriangleHit(
        car, cfg, mesh_triangles, mutator.cal_arena_local_tris,
        mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift,
        tri_idx, gjk_epa_scratch, precomputed_threshold_bt);
}

struct CarWorldTrianglePairs {
    int32_t* car_idx;
    int32_t* tri_idx;
    CarWorldTriangleHit* hits;
    int32_t* count;
    int32_t capacity;
    int32_t car_stride;
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
    int32_t pair_start = 0;
    int32_t pair_count = -1;
    uint8_t has_collected_normal = 0;
    Vec3 last_collected_normal = Vec3(0.0f, 0.0f, 1.0f);
    uint8_t pairs_replayable = 0;
    const int32_t* pair_tris = nullptr;
    const CarWorldTriangleHit* pair_hits = nullptr;
};
constexpr int kCarWorldMeshStageSlots = 4;
struct CarWorldMeshCandidateStage {
    float fingerprint[kCarWorldMeshStageFingerprintFloats] = {};
    int32_t valid = 0;
    int32_t mesh_count = 0;
    uint32_t mesh_mask = 0;
    CarWorldMeshCandidateStageMesh meshes[kCarWorldMeshStageSlots];
};

ROCKETGPU_HD int CarWorldMeshStageSlot(uint32_t mesh_mask, int mesh) {
    uint32_t v = mesh_mask & ((1u << mesh) - 1u);
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    return static_cast<int>((((v + (v >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24);
}

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
    uint32_t differ = 0u;
#pragma unroll
    for (int i = 0; i < kCarWorldMeshStageFingerprintFloats; ++i) {
        union { float f; uint32_t u; } a, b;
        a.f = fp[i];
        b.f = stage.fingerprint[i];
        differ |= a.u ^ b.u;
    }
    return differ == 0u;
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

struct CarWorldMeshQueryFrame {
    Vec3 axis_x = Vec3(1.0f, 0.0f, 0.0f);
    Vec3 axis_y = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 axis_z = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 box_center_bt;
    Vec3 box_half_extent_with_margin_bt;
    Vec3 extent_bt;
    Vec3 aabb_min_bt;
    Vec3 aabb_max_bt;
};

ROCKETGPU_HD CarWorldMeshQueryFrame ComputeCarWorldMeshQueryFrame(const CarState& car, const CarConfig& cfg,
                                                                  const CarShapeCacheEntry* shape = nullptr) {
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
    const Vec3 h = shape != nullptr ? shape->half_extent_bt : BulletBoxHalfExtentWithMarginBt(cfg);
    f.box_half_extent_with_margin_bt = h;
    const Vec3 extent_bt(
        Abs(axis_x.x) * h.x + Abs(axis_y.x) * h.y + Abs(axis_z.x) * h.z,
        Abs(axis_x.y) * h.x + Abs(axis_y.y) * h.y + Abs(axis_z.y) * h.z,
        Abs(axis_x.z) * h.x + Abs(axis_y.z) * h.y + Abs(axis_z.z) * h.z);
    f.extent_bt = extent_bt;
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

ROCKETGPU_HD float CarWorldTriangleBoxAxisGap(
    const ArenaMeshTriangle& tri, const Vec3& axis, const Vec3& box_center_bt, float half_extent_bt) {
    const float c = Dot(axis, box_center_bt);
    const float pa = Dot(axis, tri.a);
    const float pb = Dot(axis, tri.b);
    const float pc = Dot(axis, tri.c);
    return Max(Min(pa, Min(pb, pc)) - (c + half_extent_bt), (c - half_extent_bt) - Max(pa, Max(pb, pc)));
}

#if !defined(ROCKETGPU_USE_CUDA)
inline bool CarWorldTriangleEdgeAxesSeparated(const ArenaMeshTriangle& tri, const CarWorldMeshQueryFrame& f,
                                              float reject_bt) {
    const Vec3 e[3] = {tri.b - tri.a, tri.c - tri.b, tri.a - tri.c};
    const Vec3 ax[3] = {f.axis_x, f.axis_y, f.axis_z};
    const Vec3& h = f.box_half_extent_with_margin_bt;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const Vec3 u = Cross(e[i], ax[j]);
            const float len = SqrtScalar(Dot(u, u));
            if (!(len > 1e-6f)) {
                continue;
            }
            const float c = Dot(u, f.box_center_bt);
            const float r = Abs(Dot(u, f.axis_x)) * h.x + Abs(Dot(u, f.axis_y)) * h.y + Abs(Dot(u, f.axis_z)) * h.z;
            const float pa = Dot(u, tri.a), pb = Dot(u, tri.b), pc = Dot(u, tri.c);
            if (Max(Min(pa, Min(pb, pc)) - (c + r), (c - r) - Max(pa, Max(pb, pc))) > reject_bt * len) {
                return true;
            }
        }
    }
    return false;
}
#endif

ROCKETGPU_HD int CarWorldTriangleQueryTest(
    const ArenaMeshTriangle& tri, const CarWorldMeshQueryFrame& f, float contact_threshold_bt,
    const ArenaMeshTriangle* local_tri = nullptr) {
    if (local_tri != nullptr && local_tri->cal_origin_sign != 0) {
        const float center_y = f.box_center_bt.y - static_cast<float>(local_tri->cal_origin_sign) * kGoalMeshOriginYBt;
        const Vec3 local_min(f.aabb_min_bt.x, center_y - f.extent_bt.y, f.aabb_min_bt.z);
        const Vec3 local_max(f.aabb_max_bt.x, center_y + f.extent_bt.y, f.aabb_max_bt.z);
        if (!CarWorldTriangleAgainstAabbBullet(*local_tri, local_min, local_max)) {
            return 0;
        }
    } else if (!CarWorldTriangleAgainstAabbBullet(tri, f.aabb_min_bt, f.aabb_max_bt)) {
        return 0;
    }
    const Vec3 triangle_normal_world = Normalize(tri.normal, Vec3(0.0f, 0.0f, 1.0f));
    const float center_proj = Dot(triangle_normal_world, f.box_center_bt);
    const float support_extent =
        Abs(Dot(triangle_normal_world, f.axis_x)) * f.box_half_extent_with_margin_bt.x +
        Abs(Dot(triangle_normal_world, f.axis_y)) * f.box_half_extent_with_margin_bt.y +
        Abs(Dot(triangle_normal_world, f.axis_z)) * f.box_half_extent_with_margin_bt.z;
    const float tri_proj = Dot(triangle_normal_world, tri.a);
    const float separation_along_normal = tri_proj - (center_proj + support_extent);
    const float separation_against_normal = (center_proj - support_extent) - tri_proj;
    if (separation_along_normal > contact_threshold_bt ||
        separation_against_normal > contact_threshold_bt) {
        return 1;
    }
    constexpr float kBoxAxisGapSlackBt = 1.0e-3f;
    const float box_axis_reject_bt = contact_threshold_bt + kBoxAxisGapSlackBt;
    if (CarWorldTriangleBoxAxisGap(tri, f.axis_x, f.box_center_bt, f.box_half_extent_with_margin_bt.x) > box_axis_reject_bt ||
        CarWorldTriangleBoxAxisGap(tri, f.axis_y, f.box_center_bt, f.box_half_extent_with_margin_bt.y) > box_axis_reject_bt ||
        CarWorldTriangleBoxAxisGap(tri, f.axis_z, f.box_center_bt, f.box_half_extent_with_margin_bt.z) > box_axis_reject_bt) {
        return 1;
    }
#if !defined(ROCKETGPU_USE_CUDA)
    if (CarWorldTriangleEdgeAxesSeparated(tri, f, box_axis_reject_bt)) {
        return 1;
    }
#endif
    return 2;
}

struct CarWorldMeshReach {
    Vec3 lo;
    Vec3 hi;
};

ROCKETGPU_HD float CarWorldMeshReachBt(const CarConfig& cfg) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kSlackBt = 0.2f;
    const Vec3 half_bt = BulletBoxHalfExtentWithMarginBt(cfg);
    const Vec3 offset_bt = cfg.hitbox_offset * kUUToBT;
    return SqrtScalar(Dot(half_bt, half_bt)) + SqrtScalar(Dot(offset_bt, offset_bt)) + kSlackBt;
}

ROCKETGPU_HD CarWorldMeshReach ComputeCarWorldMeshReach(const CarState& car, const CarConfig& cfg, float dt,
                                                        const CarShapeCacheEntry* shape = nullptr) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    const float reach_bt = shape != nullptr ? shape->mesh_reach_bt : CarWorldMeshReachBt(cfg);
    const Vec3 pred_bt = car.pos_bt + car.vel * (kUUToBT * dt);
    CarWorldMeshReach reach;
    reach.lo = Vec3(Min(car.pos_bt.x, pred_bt.x) - reach_bt, Min(car.pos_bt.y, pred_bt.y) - reach_bt,
                    Min(car.pos_bt.z, pred_bt.z) - reach_bt);
    reach.hi = Vec3(Max(car.pos_bt.x, pred_bt.x) + reach_bt, Max(car.pos_bt.y, pred_bt.y) + reach_bt,
                    Max(car.pos_bt.z, pred_bt.z) + reach_bt);
    return reach;
}

ROCKETGPU_HD bool CarWorldMeshInReach(const CarWorldMeshReach& reach, const ArenaMeshBvhMeshInfo& info) {
    return reach.lo.x <= info.aabb_max.x && reach.hi.x >= info.aabb_min.x && reach.lo.y <= info.aabb_max.y &&
           reach.hi.y >= info.aabb_min.y && reach.lo.z <= info.aabb_max.z && reach.hi.z >= info.aabb_min.z;
}

ROCKETGPU_HD bool CarWorldBoxInsideFreeInterior(
    const Vec3& lo, const Vec3& hi, const ArenaMeshBvhMeshInfo* mesh_infos, int32_t mesh_count) {
    if (mesh_count <= 0 || mesh_infos == nullptr) {
        return true;
    }
    const float fx = mesh_infos[0].free_half_x;
    const float fy = mesh_infos[0].free_half_y;
    if (hi.x < fx && lo.x > -fx && hi.y < fy && lo.y > -fy) {
        return true;
    }
    return false;
}

ROCKETGPU_HD uint32_t CarWorldMeshReachMask(
    const CarWorldMeshReach& reach, const ArenaMeshBvhMeshInfo* mesh_infos, int32_t mesh_count) {
    if (CarWorldBoxInsideFreeInterior(reach.lo, reach.hi, mesh_infos, mesh_count)) {
        return 0u;
    }
    uint32_t mask = 0u;
    const auto test = [&](int m) {
        const Vec3 lo = mesh_infos[m].aabb_min;
        const Vec3 hi = mesh_infos[m].aabb_max;
#if defined(__CUDA_ARCH__)
        const bool in = (reach.lo.x <= hi.x) & (reach.hi.x >= lo.x) & (reach.lo.y <= hi.y) & (reach.hi.y >= lo.y) &
                        (reach.lo.z <= hi.z) & (reach.hi.z >= lo.z);
#else
        const bool in = (reach.lo.x <= hi.x) && (reach.hi.x >= lo.x) && (reach.lo.y <= hi.y) && (reach.hi.y >= lo.y) &&
                        (reach.lo.z <= hi.z) && (reach.hi.z >= lo.z);
#endif
        mask |= static_cast<uint32_t>(in) << m;
    };
    if (mesh_count <= 16) {
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            if (m < mesh_count) {
                test(m);
            }
        }
    } else {
        for (int m = 0; m < mesh_count && m < 32; ++m) {
            test(m);
        }
    }
    return mask;
}

ROCKETGPU_HD bool CarWorldMeshesOutOfReach(
    const CarState& car,
    const CarConfig& cfg,
    const ArenaMeshBvhMeshInfo* mesh_infos,
    int32_t mesh_count,
    float dt,
    uint32_t* reach_mask_out = nullptr,
    const CarShapeCacheEntry* shape = nullptr) {
    const CarWorldMeshReach reach = ComputeCarWorldMeshReach(car, cfg, dt, shape);
    if (mesh_count <= 32) {
        const uint32_t mask = CarWorldMeshReachMask(reach, mesh_infos, mesh_count);
        if (reach_mask_out != nullptr) {
            *reach_mask_out = mask;
        }
        return mask == 0u;
    }
    for (int m = 0; m < mesh_count; ++m) {
        if (CarWorldMeshInReach(reach, mesh_infos[m])) {
            return false;
        }
    }
    return true;
}

ROCKETGPU_HD void ReleaseCarWorldMeshManifoldsOutOfReach(
    const ArenaMeshBvhMeshInfo* mesh_bvh_meshes,
    int32_t mesh_bvh_mesh_count,
    int32_t mesh_bvh_quantization_count,
    int32_t mesh_bvh_node_count,
    CarWorldMeshRetainedManifold* retained_mesh_manifolds) {
    if (retained_mesh_manifolds == nullptr) {
        return;
    }
    static_assert(kCarWorldMeshRetainedManifoldsPerCar == 2, "the slot form below covers two slots");
    if (retained_mesh_manifolds[0].count == 0 && retained_mesh_manifolds[1].count == 0) {
        return;
    }
    CarWorldMeshRetainedManifold& s0 = retained_mesh_manifolds[0];
    CarWorldMeshRetainedManifold& s1 = retained_mesh_manifolds[1];
    const int m0 = s0.mesh;
    const int m1 = s1.mesh;
    const bool held0 = s0.count > 0;
    const bool held1 = s1.count > 0;
    const bool release0 = held0 && m0 < mesh_bvh_mesh_count &&
        CarWorldMeshInfoUsable(mesh_bvh_meshes[m0], mesh_bvh_quantization_count, mesh_bvh_node_count);
    const bool release1 = held1 && m1 < mesh_bvh_mesh_count && !(held0 && m0 == m1) &&
        CarWorldMeshInfoUsable(mesh_bvh_meshes[m1], mesh_bvh_quantization_count, mesh_bvh_node_count);
    if (release0) {
        s0.count = 0;
    }
    if (release1) {
        s1.count = 0;
    }
}

constexpr float kCarWorldPlaneReachSlackBt = 1.0e-3f;
ROCKETGPU_HD void CarWorldPlaneSupportDistances(
    const CarState& car, const CarConfig& cfg, float d[4], float& threshold_bt, float& window_bt,
    const CarShapeCacheEntry* shape = nullptr) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kBulletGContactBreakingThresholdBt = 0.02f;
    constexpr float kBulletStaticPlaneAabbHalfSlabBt = 0.2f;
    threshold_bt = shape != nullptr ? shape->contact_breaking_threshold_bt : CarContactBreakingThresholdBt(cfg);
    window_bt = threshold_bt + kBulletStaticPlaneAabbHalfSlabBt + 2.0f * kBulletGContactBreakingThresholdBt;
    Vec3 fwd(1.0f, 0.0f, 0.0f);
    Vec3 right(0.0f, 1.0f, 0.0f);
    Vec3 up(0.0f, 0.0f, 1.0f);
    CarBasisColumns(car, fwd, right, up);
    const Vec3 half_bt = shape != nullptr ? shape->half_extent_bt : BulletBoxHalfExtentWithMarginBt(cfg);
    const Vec3 offset_bt = cfg.hitbox_offset * kUUToBT;
    const Vec3 center_bt = car.pos_bt + fwd * offset_bt.x + right * offset_bt.y + up * offset_bt.z;
    const float ext_x = Abs(fwd.x) * half_bt.x + Abs(right.x) * half_bt.y + Abs(up.x) * half_bt.z;
    const float ext_z = Abs(fwd.z) * half_bt.x + Abs(right.z) * half_bt.y + Abs(up.z) * half_bt.z;
    const float wall_bt = ArenaExtentX() * kUUToBT;
    d[0] = center_bt.x + wall_bt - ext_x;
    d[1] = wall_bt - center_bt.x - ext_x;
    d[2] = center_bt.z - ext_z;
    d[3] = ArenaHeight() * kUUToBT - center_bt.z - ext_z;
}

ROCKETGPU_HD int CarWorldPlanesOutOfReach(const CarState& car, const CarConfig& cfg) {
    constexpr float kSlackBt = kCarWorldPlaneReachSlackBt;
    float d[4];
    float threshold_bt;
    float window_bt;
    CarWorldPlaneSupportDistances(car, cfg, d, threshold_bt, window_bt);
    int mask = 0;
    const auto classify = [&](float d, int bit) {
        if (mask < 0) {
            return;
        }
        if (d < threshold_bt + kSlackBt || Abs(d - window_bt) <= kSlackBt) {
            mask = -1;
        } else if (d < window_bt) {
            mask |= bit;
        }
    };
    classify(d[0], 1);
    classify(d[1], 2);
    classify(d[2], 4);
    classify(d[3], 8);
    return mask;
}

ROCKETGPU_HD int CarWorldPlanesInReachCount(const CarState& car, const CarConfig& cfg) {
    float d[4];
    float threshold_bt;
    float window_bt;
    CarWorldPlaneSupportDistances(car, cfg, d, threshold_bt, window_bt);
    int count = 0;
    for (int k = 0; k < 4; ++k) {
        count += (d[k] < threshold_bt + kCarWorldPlaneReachSlackBt ||
                  Abs(d[k] - window_bt) <= kCarWorldPlaneReachSlackBt) ? 1 : 0;
    }
    return count;
}

template <bool kStagedHitsOnly = false, bool kAllowTrianglePairsGather = false>
ROCKETGPU_HD ROCKETGPU_CAR_WORLD_ALWAYS_INLINE void AppendCarWorldMeshContactRowsBody(
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
    int triangle_pairs_pass = kTrianglePairsNone,
    CarWorldMeshRetainedManifold* retained_mesh_manifolds = nullptr,
    uint32_t mesh_reach_hint = 0u,
    const Vec3* predict_vel = nullptr,
    const Vec3* predict_ang_vel = nullptr,
    const BulletBroadphaseProxy* fresh_car_proxy = nullptr,
    const CarShapeCacheEntry* shape = nullptr,
    const CarListManifolds* list_manifolds = nullptr) {
    if (mesh_triangles == nullptr || mesh_triangle_count <= 0) {
        return;
    }
    if (list_manifolds != nullptr) {
        mesh_reach_hint = list_manifolds->meshes;
    }

    constexpr float kBTToUU = 50.0f;
    constexpr int kMaxMeshCandidates = 4;
    constexpr int kMaxMeshRows = 4;

    const float contact_threshold_bt =
        shape != nullptr ? shape->contact_breaking_threshold_bt : CarContactBreakingThresholdBt(cfg);
    const bool have_bvh_traversal =
        mesh_bvh_quantization != nullptr &&
        mesh_bvh_meshes != nullptr &&
        mesh_bvh_nodes != nullptr &&
        mesh_bvh_mesh_count > 0 &&
        mesh_bvh_node_count > 0;
    if (stage_in != nullptr &&
        (stage_in->valid == 0 || !have_bvh_traversal || stage_in->mesh_count != mesh_bvh_mesh_count)) {
        stage_in = nullptr;
    }
    const bool skip_out_of_reach = broadphase_grid != nullptr && stage_out == nullptr && mesh_bvh_meshes != nullptr;
    const bool reach_hinted =
        (mesh_reach_hint != 0u || list_manifolds != nullptr) && skip_out_of_reach && mesh_bvh_mesh_count <= 32;
    bool live_retained_mesh_walk = false;
    for (int k = 0; stage_in != nullptr && retained_mesh_manifolds != nullptr && k < kCarWorldMeshRetainedManifoldsPerCar; ++k) {
        const int m = retained_mesh_manifolds[k].mesh;
        if (retained_mesh_manifolds[k].count > 0 && m < 32 && ((stage_in->mesh_mask >> m) & 1u) != 0 &&
            stage_in->meshes[CarWorldMeshStageSlot(stage_in->mesh_mask, m)].pairs_replayable == 0) {
            live_retained_mesh_walk = true;
        }
    }
    const bool query_frame_used = stage_in == nullptr || broadphase_grid == nullptr || live_retained_mesh_walk;
    const CarWorldMeshQueryFrame query_frame =
        query_frame_used ? ComputeCarWorldMeshQueryFrame(car, cfg, shape) : CarWorldMeshQueryFrame{};
    const Vec3 box_center_bt = query_frame.box_center_bt;
    const Vec3 aabb_min_bt = query_frame.aabb_min_bt;
    const Vec3 aabb_max_bt = query_frame.aabb_max_bt;
    const CarWorldMeshReach mesh_reach =
        (skip_out_of_reach && !reach_hinted) ? ComputeCarWorldMeshReach(car, cfg, broadphase_dt, shape) : CarWorldMeshReach{};
    bool any_mesh_in_reach = !skip_out_of_reach;
    uint32_t mesh_visit_mask = ~0u;
    if (skip_out_of_reach && mesh_bvh_mesh_count <= 32) {
        const uint32_t in_reach =
            reach_hinted ? mesh_reach_hint : CarWorldMeshReachMask(mesh_reach, mesh_bvh_meshes, mesh_bvh_mesh_count);
        any_mesh_in_reach = in_reach != 0u;
        mesh_visit_mask = in_reach;
        for (int k = 0; retained_mesh_manifolds != nullptr && k < kCarWorldMeshRetainedManifoldsPerCar; ++k) {
            if (retained_mesh_manifolds[k].count > 0 && retained_mesh_manifolds[k].mesh < 32) {
                mesh_visit_mask |= 1u << retained_mesh_manifolds[k].mesh;
            }
        }
        if (have_bvh_traversal && stage_out == nullptr && mesh_visit_mask == 0u) {
            return;
        }
    } else {
        for (int m = 0; !any_mesh_in_reach && m < mesh_bvh_mesh_count; ++m) {
            any_mesh_in_reach = CarWorldMeshInReach(mesh_reach, mesh_bvh_meshes[m]);
        }
    }
#if !defined(ROCKETGPU_USE_CUDA)
    UninitArray<BulletBroadphaseProxy, 1> car_broadphase_proxy_slot;
    BulletBroadphaseProxy& car_broadphase_proxy = car_broadphase_proxy_slot.data()[0];
#else
    BulletBroadphaseProxy car_broadphase_proxy{};
#endif
    bool car_broadphase_proxy_ready = false;
    auto ensure_car_broadphase_proxy = [&]() {
        if (!car_broadphase_proxy_ready && broadphase_grid != nullptr && any_mesh_in_reach) {
#if !defined(ROCKETGPU_USE_CUDA)
            if (fresh_car_proxy != nullptr) {
                car_broadphase_proxy = BulletBroadphaseProxy{};
                car_broadphase_proxy.aabb_min = fresh_car_proxy->aabb_min;
                car_broadphase_proxy.aabb_max = fresh_car_proxy->aabb_max;
                car_broadphase_proxy.child_aabb_min = fresh_car_proxy->child_aabb_min;
                car_broadphase_proxy.child_aabb_max = fresh_car_proxy->child_aabb_max;
                if (broadphase_grid->valid()) {
                    broadphase_grid->cell_indices(car_broadphase_proxy.aabb_min, car_broadphase_proxy.cell_i,
                                                  car_broadphase_proxy.cell_j, car_broadphase_proxy.cell_k);
                }
                car_broadphase_proxy.valid = 1;
            } else
#endif
            {
                car_broadphase_proxy = ComputeBulletCarProxyAabb(
                    car, cfg, broadphase_dt, *broadphase_grid, NativeSseMath(mutator), predict_vel, predict_ang_vel);
            }
            car_broadphase_proxy_ready = true;
        }
    };
    auto check_car_mesh_manifold_exists = [&](const ArenaMeshBvhMeshInfo& info, int info_slot) -> bool {
        if (broadphase_grid == nullptr || !any_mesh_in_reach) {
            return false;
        }
        if (list_manifolds != nullptr) {
            return info_slot >= 0 && info_slot < 32 && ((list_manifolds->meshes >> info_slot) & 1u) != 0u;
        }
        ensure_car_broadphase_proxy();
        return BulletCarMeshManifoldExists(car_broadphase_proxy, *broadphase_grid, info);
    };

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
        if (candidate_count <= 0) {
            return;
        }
        Vec3 inv_r0, inv_r1, inv_r2;
        BuildInvInertiaTensorWorldBt(car, cfg, mutator.car_mass, inv_r0, inv_r1, inv_r2);
        const float inv_car_mass = 1.0f / mutator.car_mass;
        const bool bt_authoritative = car.ext_force_impulse_bt_valid && car.basis_authoritative;
        const Vec3 vel_plus_ext_bt = car.vel_bt + car.ext_force_impulse_bt;
        const Vec3 ang_vel_plus_ext = car.ang_vel + car.ext_torque_impulse;
        CarWorldVelocityTraceInputs velocity_trace_inputs{
            car.vel,
            car.ang_vel,
            external_linear_delta,
            external_angular_delta,
            car.pos,
        };
        velocity_trace_inputs.has_rel_contact_bt = 1;
        velocity_trace_inputs.linear_velocity_bt = car.vel_bt;
        velocity_trace_inputs.car_pos_bt = car.pos_bt;
        velocity_trace_inputs.external_force_impulse_bt = car.ext_force_impulse_bt;
        velocity_trace_inputs.has_bt_inputs = car.ext_force_impulse_bt_valid;
        velocity_trace_inputs.external_torque_impulse = car.ext_torque_impulse;
        velocity_trace_inputs.restitution_no_velocity_threshold = true;
        int emitted_rows = 0;
        for (int candidate_idx = 0; candidate_idx < candidate_count && emitted_rows < kMaxMeshRows; ++candidate_idx) {
            const MeshContactCandidate& candidate = candidates[candidate_idx];
            const float penetration_uu = Max(0.0f, -candidate.distance_bt) * kBTToUU;
            acc.add(candidate.normal, Max(penetration_uu, 0.001f));
            const Vec3 angular_component_bt_native =
                Mul3x3Vec(inv_r0, inv_r1, inv_r2, Cross(candidate.rel_contact_bt, candidate.normal));

            if (split_rows && split_row_count < kMaxCarWorldSplitContactRows) {
                CarWorldSplitContactRow& split_row = split_rows[split_row_count];
                Vec3 mesh_split_angular_bt_native(0.0f, 0.0f, 0.0f);
                Vec3 mesh_split_angular(0.0f, 0.0f, 0.0f);
                if (!(penetration_uu <= 0.0f || dt <= 0.0f)) {
                    mesh_split_angular_bt_native = angular_component_bt_native;
                    mesh_split_angular = ApplyInverseBodyInertiaWorld(
                        car, cfg, mutator.car_mass, Cross(candidate.rel_contact, candidate.normal));
                }
                const float mesh_split_pen_bt = -candidate.distance_bt;
                InitializeCarWorldSplitContactRow(
                    split_row,
                    penetration_uu,
                    candidate.normal,
                    inv_car_mass,
                    candidate.rel_contact,
                    mesh_split_angular,
                    dt,
                    &candidate.rel_contact_bt,
                    &mesh_split_angular_bt_native,
                    &mesh_split_pen_bt);
                if (split_row.valid) {
                    split_row.source_tag = 100 + Max(candidate.mesh_world_idx, 0);
                    ++split_row_count;
                }
            }

            velocity_trace_inputs.rel_contact_bt = candidate.rel_contact_bt;
            const Vec3 angular_component = angular_component_bt_native * (1.0f / 50.0f);
            Vec3 normal_restitution_rel_vel(0.0f, 0.0f, 0.0f);
            Vec3 normal_base_rel_vel(0.0f, 0.0f, 0.0f);
            Vec3 tangent_basis_rel_vel(0.0f, 0.0f, 0.0f);
            Vec3 friction_base_rel_vel(0.0f, 0.0f, 0.0f);
            Vec3 tangent(0.0f, 0.0f, 0.0f);
            if (bt_authoritative) {
                const Vec3 vel_nd_bt = vel_plus_ext_bt + Cross(ang_vel_plus_ext, candidate.rel_contact_bt);
                const float rel_vel_n_bt = Dot(candidate.normal, vel_nd_bt);
                const Vec3 lateral_bt = vel_nd_bt - candidate.normal * rel_vel_n_bt;
                const float lat_rel_vel = LengthSq(lateral_bt);
                constexpr float kBulletSimdEpsilon = 1.1920928955078125e-7f;
                tangent = lat_rel_vel > kBulletSimdEpsilon
                    ? lateral_bt * (1.0f / SqrtScalar(lat_rel_vel))
                    : BulletPlaneSpace1First(candidate.normal);
            } else {
                normal_restitution_rel_vel = VelocityAtPoint(car, candidate.rel_contact);
                normal_base_rel_vel = VelocityAtPoint(car, candidate.rel_contact, external_linear_delta, external_angular_delta);
                tangent_basis_rel_vel = normal_base_rel_vel;
                friction_base_rel_vel = VelocityAtPoint(car, candidate.rel_contact, external_linear_delta, Vec3(0.0f, 0.0f, 0.0f));
                tangent = tangent_basis_rel_vel - candidate.normal * Dot(tangent_basis_rel_vel, candidate.normal);
                const float tangent_len = Length(tangent);
                tangent = tangent_len > 1e-6f ? (tangent / tangent_len) : OrthonormalTangent(candidate.normal);
            }
            const Vec3 tangent_angular_component_bt_native =
                Mul3x3Vec(inv_r0, inv_r1, inv_r2, Cross(candidate.rel_contact_bt, tangent));
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
                inv_car_mass,
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
            for (int row_idx = friction_row_start; row_idx < friction_row_count; ++row_idx) {
                friction_rows[row_idx].source_tag = mesh_source_tag;
            }
            ++emitted_rows;
        }
    };

    UninitArray<int32_t, kAllowTrianglePairsGather ? kCarWorldMeshGatherTriangles : 1> gathered_tris;
    int gathered_count = 0;
    auto consider_triangle = [&](int32_t tri_idx, MeshContactCandidate* candidates, int& candidate_count,
                                 const CarWorldTriangleHit* precomputed = nullptr) {
        const ArenaMeshTriangle& tri = mesh_triangles[tri_idx];
        if (precomputed == nullptr) {
            const int query_test = CarWorldTriangleQueryTest(
                tri, query_frame, contact_threshold_bt,
                mutator.cal_arena_local_tris != nullptr ? &mutator.cal_arena_local_tris[tri_idx] : nullptr);
            if (query_test == 0) {
                return;
            }
            ++current_mesh_broadphase_hits;
            if (query_test == 1) {
                return;
            }
            if constexpr (kAllowTrianglePairsGather) {
                if (triangle_pairs_pass == kTrianglePairsGather) {
                    if (gathered_count < kCarWorldMeshGatherTriangles) {
                        gathered_tris[gathered_count] = tri_idx;
                    }
                    ++gathered_count;
                    return;
                }
            }
        }
        CarWorldTriangleHit hit;
        if constexpr (kStagedHitsOnly) {
            if (precomputed == nullptr) {
                return;
            }
            hit = *precomputed;
        } else {
            hit = precomputed != nullptr
                ? *precomputed
                : ComputeCarWorldTriangleHit(car, cfg, mutator, mesh_triangles, tri_idx, gjk_epa_scratch,
                                             contact_threshold_bt);
        }
        if (hit.hit == 0 || hit.distance_bt > contact_threshold_bt) {
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

        candidate.add_distance_bt = distance_bt;
        const Vec3 manifold_point_a_bt =
            point_on_b_bt_pre_adjust + contact_normal_pre_adjust * distance_bt;

        candidate.point_on_b_bt = point_on_b_bt;
        candidate.local_point_a_bt = CarBodyLocalPointBt(car, manifold_point_a_bt);
        candidate.cache_local_point_a_bt = candidate.local_point_a_bt;

        const Vec3 refreshed_point_on_a_bt = CarBodyWorldPointBt(car, candidate.local_point_a_bt);
        candidate.distance_bt =
            Dot(refreshed_point_on_a_bt - candidate.point_on_b_bt, candidate.normal);
        candidate.rel_contact_bt = refreshed_point_on_a_bt - car_pos_bt;
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

    const auto find_retained_mesh_manifold = [&](int mesh_info_idx) -> CarWorldMeshRetainedManifold* {
        if (retained_mesh_manifolds == nullptr || broadphase_grid == nullptr) {
            return nullptr;
        }
        for (int k = 0; k < kCarWorldMeshRetainedManifoldsPerCar; ++k) {
            if (retained_mesh_manifolds[k].count > 0 && retained_mesh_manifolds[k].mesh == mesh_info_idx) {
                return &retained_mesh_manifolds[k];
            }
        }
        return nullptr;
    };
    const auto merge_retained_mesh_manifold = [&](int mesh_info_idx, const ArenaMeshBvhMeshInfo& mesh_info,
                                                  bool manifold_exists, MeshContactCandidate* candidates,
                                                  int& candidate_count,
                                                  const CarWorldMeshRetainedPoint* seed_points = nullptr) {
        if (retained_mesh_manifolds == nullptr || broadphase_grid == nullptr) {
            return;
        }
        CarWorldMeshRetainedManifold* manifold = find_retained_mesh_manifold(mesh_info_idx);
        if (!manifold_exists) {
            if (manifold != nullptr) {
                manifold->count = 0;
            }
            return;
        }
        if (manifold == nullptr) {
            if (candidate_count == 0) {
                return;
            }
            for (int k = 0; k < kCarWorldMeshRetainedManifoldsPerCar && manifold == nullptr; ++k) {
                if (retained_mesh_manifolds[k].count == 0) {
                    manifold = &retained_mesh_manifolds[k];
                }
            }
            if (manifold == nullptr) {
                return;
            }
            manifold->mesh = static_cast<uint8_t>(mesh_info_idx);
        }
        const Vec3 mesh_origin_bt(0.0f, mesh_info.cal_origin_y, 0.0f);
        const auto point_distance = [&](const CarWorldMeshRetainedPoint& pt, Vec3& pwa, Vec3& pwb) {
            pwa = CarBodyWorldPointBt(car, pt.local_point_a_bt);
            pwb = mesh_info.cal_origin_y != 0.0f ? pt.local_point_b_bt + mesh_origin_bt : pt.local_point_b_bt;
            return Dot(pwa - pwb, pt.normal);
        };
        if (manifold->count > 0) {
            Vec3 world_a[4];
            Vec3 world_b[4];
            for (int i = 0; i < manifold->count; ++i) {
                manifold->points[i].distance_bt = point_distance(manifold->points[i], world_a[i], world_b[i]);
            }
            for (int i = static_cast<int>(manifold->count) - 1; i >= 0; --i) {
                const CarWorldMeshRetainedPoint& pt = manifold->points[i];
                const Vec3 projected_difference = world_b[i] - (world_a[i] - pt.normal * pt.distance_bt);
                if (pt.distance_bt > contact_threshold_bt ||
                    Dot(projected_difference, projected_difference) > contact_threshold_bt * contact_threshold_bt) {
                    const int last = static_cast<int>(manifold->count) - 1;
                    if (i != last) {
                        manifold->points[i] = manifold->points[last];
                        world_a[i] = world_a[last];
                        world_b[i] = world_b[last];
                    }
                    manifold->count = static_cast<uint8_t>(last);
                }
            }
        }
        const bool push_all = manifold->count == 0 || seed_points != nullptr;
        if (seed_points != nullptr) {
            manifold->count = 0;
        }
        for (int c = 0; c < candidate_count; ++c) {
            if (seed_points != nullptr && candidates[c].tri_idx <= -2) {
                manifold->points[manifold->count++] = seed_points[-2 - candidates[c].tri_idx];
                continue;
            }
            CarWorldMeshRetainedPoint pt;
            pt.local_point_a_bt = candidates[c].local_point_a_bt;
            pt.local_point_b_bt = mesh_info.cal_origin_y != 0.0f ? candidates[c].point_on_b_bt - mesh_origin_bt
                                                                 : candidates[c].point_on_b_bt;
            pt.normal = candidates[c].normal;
            Vec3 pwa;
            Vec3 pwb;
            pt.distance_bt = point_distance(pt, pwa, pwb);
            if (push_all) {
                manifold->points[manifold->count++] = pt;
            } else {
                MergeCarWorldRetainedPlanePoint(*manifold, pt, contact_threshold_bt);
            }
        }
        candidate_count = manifold->count;
        for (int i = 0; i < manifold->count; ++i) {
            const CarWorldMeshRetainedPoint& pt = manifold->points[i];
            Vec3 pwa;
            Vec3 pwb;
            point_distance(pt, pwa, pwb);
            MeshContactCandidate& candidate = candidates[i];
            candidate.mesh_world_idx = mesh_info.mesh_index;
            candidate.normal = pt.normal;
            candidate.distance_bt = pt.distance_bt;
            candidate.local_point_a_bt = pt.local_point_a_bt;
            candidate.cache_local_point_a_bt = pt.local_point_a_bt;
            candidate.point_on_b_bt = pwb;
            candidate.rel_contact_bt = pwa - car_pos_bt;
            candidate.rel_contact = candidate.rel_contact_bt * kBTToUU;
        }
    };

    static_assert(kMaxMeshCandidates == kCarWorldMeshStageCandidates,
                  "CarWorldMeshCandidateStage holds every mesh candidate");
    if (stage_out != nullptr) {
        if (stage_mesh < 0) {
            stage_out->valid = 0;
            stage_out->mesh_count = 0;
        }
        if (!have_bvh_traversal || mesh_bvh_mesh_count > kCarWorldMeshStageMeshes) {
            return;
        }
        if (stage_mesh < 0) {
            if (mesh_bvh_mesh_count > kCarWorldMeshStageSlots) {
                return;
            }
            stage_out->mesh_count = mesh_bvh_mesh_count;
            stage_out->mesh_mask = ~0u;
        }
    }
    if (have_bvh_traversal) {
        const int mesh_begin = stage_mesh >= 0 ? stage_mesh : 0;
        const int mesh_end = stage_mesh >= 0 && stage_mesh < mesh_bvh_mesh_count ? stage_mesh + 1 : (stage_mesh >= 0 ? 0 : mesh_bvh_mesh_count);
#if !defined(ROCKETGPU_USE_CUDA)
        const auto next_mesh = [mesh_visit_mask](int idx) {
            if (idx >= 32) {
                return idx;
            }
            const uint32_t rest = mesh_visit_mask & (~0u << idx);
            return rest != 0u ? CountTrailingZeros32(rest) : 32;
        };
        for (int mesh_info_idx = next_mesh(mesh_begin); mesh_info_idx < mesh_end;
             mesh_info_idx = next_mesh(mesh_info_idx + 1)) {
            const ArenaMeshBvhMeshInfo& mesh_info = mesh_bvh_meshes[mesh_info_idx];
#else
        for (int mesh_info_idx = mesh_begin; mesh_info_idx < mesh_end; ++mesh_info_idx) {
            if (mesh_info_idx < 32 && ((mesh_visit_mask >> mesh_info_idx) & 1u) == 0u) {
                continue;
            }
            const ArenaMeshBvhMeshInfo mesh_info = mesh_bvh_meshes[mesh_info_idx];
#endif
            if (!CarWorldMeshInfoUsable(mesh_info, mesh_bvh_quantization_count, mesh_bvh_node_count)) {
                continue;
            }
            if (skip_out_of_reach && !(reach_hinted ? ((mesh_reach_hint >> mesh_info_idx) & 1u) != 0u
                                                    : CarWorldMeshInReach(mesh_reach, mesh_info))) {
                if (CarWorldMeshRetainedManifold* const retained = find_retained_mesh_manifold(mesh_info_idx)) {
                    retained->count = 0;
                }
                continue;
            }
            if (stage_in != nullptr && broadphase_grid != nullptr && ((stage_in->mesh_mask >> mesh_info_idx) & 1u) == 0) {
                CarWorldMeshRetainedManifold* const retained = find_retained_mesh_manifold(mesh_info_idx);
                const bool exists = check_car_mesh_manifold_exists(mesh_info, mesh_info_idx);
                if (!exists && retained != nullptr) {
                    retained->count = 0;
                }
                if (!exists || retained == nullptr) {
                    if (exists) {
                        append_local_world_order_group(
                            100 + Max(mesh_info.mesh_index, 0), normal_row_count, 0, friction_row_count, 0);
                    }
                    continue;
                }
            }

            UninitArray<MeshContactCandidate, kMaxMeshCandidates> candidates;
            int candidate_count = 0;
#if !defined(ROCKETGPU_USE_CUDA)
            UninitArray<CarWorldMeshRetainedPoint, kMaxMeshCandidates> seed_point_slots;
            CarWorldMeshRetainedPoint* const seed_points = seed_point_slots;
#else
            CarWorldMeshRetainedPoint seed_points[kMaxMeshCandidates];
#endif
            bool mesh_seeded = false;
            if (retained_mesh_manifolds != nullptr && broadphase_grid != nullptr) {
                CarWorldMeshRetainedManifold* const retained = find_retained_mesh_manifold(mesh_info_idx);
                if (retained != nullptr && check_car_mesh_manifold_exists(mesh_info, mesh_info_idx)) {
                    for (int k = 0; k < kMaxMeshCandidates; ++k) {
                        candidates[k] = MeshContactCandidate{};
                    }
                    merge_retained_mesh_manifold(mesh_info_idx, mesh_info, true, candidates, candidate_count);
                    for (int k = 0; k < candidate_count; ++k) {
                        seed_points[k] = retained->points[k];
                        candidates[k].tri_idx = -2 - k;
                        candidates[k].add_distance_bt = candidates[k].distance_bt;
                    }
                    mesh_seeded = true;
                }
            }
            current_mesh_broadphase_hits = 0;
            gathered_count = 0;
            const ArenaMeshBvhQuantization& quantization = mesh_bvh_quantization[mesh_info.mesh_index];
            bool query_box_quantized = false;
            auto ensure_quantized_query_box = [&]() {
                if (!query_box_quantized && query_frame_used) {
                    QuantizeArenaMeshBvhQueryBox(quantization, box_center_bt, query_frame.extent_bt, mesh_info.cal_origin_y,
                                                 quantized_query_aabb_min, quantized_query_aabb_max);
                    query_box_quantized = true;
                }
            };

            int mesh_subtree_overlap_count = 0;
            const ArenaMeshBvhSubtreeInfo* mesh_subtrees = nullptr;
            int mesh_subtree_count = 0;
            if (mesh_bvh_subtrees != nullptr &&
                mesh_info.subtree_offset >= 0 &&
                mesh_info.subtree_count > 0 &&
                mesh_info.subtree_offset + mesh_info.subtree_count <= mesh_bvh_subtree_count) {
                mesh_subtrees = mesh_bvh_subtrees + mesh_info.subtree_offset;
                mesh_subtree_count = mesh_info.subtree_count;

                if (broadphase_grid == nullptr && stage_out == nullptr && mesh_subtree_count > 0) {
                    ensure_quantized_query_box();
                }
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

            const bool mesh_aabb_may_overlap =
                query_frame_used && CarWorldMeshAabbMayOverlap(mesh_info, aabb_min_bt, aabb_max_bt);
            const bool seeded_live = mesh_seeded && candidate_count > 0;
            const CarWorldMeshCandidateStageMesh* const staged_entry =
                stage_in != nullptr && ((stage_in->mesh_mask >> mesh_info_idx) & 1u) != 0
                    ? &stage_in->meshes[CarWorldMeshStageSlot(stage_in->mesh_mask, mesh_info_idx)]
                    : nullptr;
            bool used_pairs = false;
            if (stage_in != nullptr && seeded_live && (staged_entry == nullptr || staged_entry->pairs_replayable != 0)) {
                if (staged_entry != nullptr) {
                    for (int k = 0; k < staged_entry->pair_count; ++k) {
                        consider_triangle(staged_entry->pair_tris[k], candidates, candidate_count, &staged_entry->pair_hits[k]);
                    }
                    current_mesh_broadphase_hits = staged_entry->broadphase_hits;
                }
            } else if (stage_in != nullptr && !seeded_live) {
                if (staged_entry == nullptr) {
                } else {
                const CarWorldMeshCandidateStageMesh& staged =
                    stage_in->meshes[CarWorldMeshStageSlot(stage_in->mesh_mask, mesh_info_idx)];
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
                       stage_out->meshes[CarWorldMeshStageSlot(stage_out->mesh_mask, mesh_info_idx)].pair_count >= 0) {
                const CarWorldMeshCandidateStageMesh& gathered =
                    stage_out->meshes[CarWorldMeshStageSlot(stage_out->mesh_mask, mesh_info_idx)];
                for (int k = 0; k < gathered.pair_count; ++k) {
                    consider_triangle(triangle_pairs->tri_idx[gathered.pair_start + k], candidates, candidate_count,
                                      &triangle_pairs->hits[gathered.pair_start + k]);
                }
                current_mesh_broadphase_hits = gathered.broadphase_hits;
                used_pairs = true;
            } else if (mesh_aabb_may_overlap) {
                ensure_quantized_query_box();
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
            if constexpr (kAllowTrianglePairsGather) {
            if (stage_out != nullptr && triangle_pairs_pass == kTrianglePairsGather) {
                CarWorldMeshCandidateStageMesh& staged = stage_out->meshes[CarWorldMeshStageSlot(stage_out->mesh_mask, mesh_info_idx)];
                staged.broadphase_hits = current_mesh_broadphase_hits;
                staged.pair_count = -1;
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
            }
            if (stage_out != nullptr) {
                CarWorldMeshCandidateStageMesh& staged = stage_out->meshes[CarWorldMeshStageSlot(stage_out->mesh_mask, mesh_info_idx)];
                staged.candidate_count = candidate_count;
                for (int c = 0; c < candidate_count && c < kMaxMeshCandidates; ++c) {
                    staged.candidates[c] = candidates[c];
                }
                staged.broadphase_hits = current_mesh_broadphase_hits;
                staged.has_collected_normal = has_collected_normal ? 1 : 0;
                staged.last_collected_normal = last_collected_normal;
                staged.pairs_replayable = used_pairs ? 1 : 0;
                if (used_pairs) {
                    staged.pair_tris = triangle_pairs->tri_idx + staged.pair_start;
                    staged.pair_hits = triangle_pairs->hits + staged.pair_start;
                }
                continue;
            }

            const bool need_manifold_for_merge =
                retained_mesh_manifolds != nullptr &&
                (candidate_count > 0 || find_retained_mesh_manifold(mesh_info_idx) != nullptr);
            const bool need_manifold_for_order =
                broadphase_grid != nullptr && local_world_order_groups != nullptr &&
                local_world_order_group_count != nullptr && current_mesh_broadphase_hits == 0 && candidate_count == 0;
            const bool mesh_manifold_exists =
                (need_manifold_for_merge || need_manifold_for_order) &&
                check_car_mesh_manifold_exists(mesh_info, mesh_info_idx);
            merge_retained_mesh_manifold(mesh_info_idx, mesh_info, mesh_manifold_exists, candidates, candidate_count,
                                         mesh_seeded ? seed_points : nullptr);

            const int group_normal_start = normal_row_count;
            const int group_friction_start = friction_row_count;
            emit_candidates(candidates, candidate_count);
            const int group_normal_count = normal_row_count - group_normal_start;
            const int group_friction_count = friction_row_count - group_friction_start;

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
                const float origin_y = mutator.cal_arena_local_tris != nullptr
                    ? static_cast<float>(mutator.cal_arena_local_tris[tri_idx].cal_origin_sign) * kGoalMeshOriginYBt : 0.0f;
                QuantizeArenaMeshBvhQueryBox(quantization, box_center_bt, query_frame.extent_bt, origin_y,
                                             quantized_query_aabb_min, quantized_query_aabb_max);
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

template <bool kStagedHitsOnly = false, bool kAllowTrianglePairsGather = false, typename... Args>
ROCKETGPU_HD void AppendCarWorldMeshContactRows(Args&&... args) {
    AppendCarWorldMeshContactRowsBody<kStagedHitsOnly, kAllowTrianglePairsGather>(static_cast<Args&&>(args)...);
}
