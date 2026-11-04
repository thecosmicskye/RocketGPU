ROCKETGPU_HD constexpr int CarCarVelocitySolverIterations() {
    return 10;
}

ROCKETGPU_HD Vec3 CarCarBulletRelativePointUU(const Vec3& point_uu, const Vec3& origin_bt) {
    return FromBtLinearVec(ToBtLinearVec(point_uu) - origin_bt);
}

ROCKETGPU_HD float CarCarBodyImpulseDenominatorFromRel(
    const CarState& car,
    const CarConfig& cfg,
    float mass,
    const Vec3& rel_world,
    const Vec3& normal_world) {
    const Vec3 c0 = Cross(rel_world, normal_world);
    const Vec3 vec = Cross(ApplyInverseBodyInertiaWorld(car, cfg, mass, c0), rel_world);
    return (1.0f / mass) + Dot(normal_world, vec);
}

ROCKETGPU_HD void ClearCarCarPairCachedContact(CarCarPairCachedContact& cache) {
    if (cache.valid == 0) {
        return;
    }
    cache = CarCarPairCachedContact{};
}

ROCKETGPU_HD float GetCarCarCachedPenetrationUU(const CarCarPairCachedContact& cache) {
    float penetration_uu = 0.0f;
    const int count = Min(static_cast<int>(cache.contact_count), kMaxApproxCarCarContactPoints);
    for (int i = 0; i < count; ++i) {
        if (cache.contacts[i].valid == 0) {
            continue;
        }
        penetration_uu = Max(penetration_uu, cache.contacts[i].penetration_uu);
    }
    return penetration_uu;
}

ROCKETGPU_HD float CarCarBreakingThresholdBt(const CarConfig& cfg) {
    return CarContactBreakingThresholdBt(cfg);
}

ROCKETGPU_HD void RefreshCarCarRetainedManifold(
    CarCarRetainedManifold& manifold, const CarState& a, const CarState& b, float threshold_bt) {
    const int count = manifold.count;
    if (count <= 0) {
        return;
    }
    for (int i = 0; i < count; ++i) {
        CarCarRetainedPoint& point = manifold.points[i];
        point.pos_a_bt = CarBodyWorldPointBt(a, point.local_a_bt);
        point.pos_b_bt = CarBodyWorldPointBt(b, point.local_b_bt);
        point.distance_bt = Dot(point.pos_b_bt - point.pos_a_bt, point.normal);
    }
    const float threshold_sq = threshold_bt * threshold_bt;
    int kept = count;
    for (int i = count - 1; i >= 0; --i) {
        const CarCarRetainedPoint point = manifold.points[i];
        bool remove = point.distance_bt > threshold_bt;
        if (!remove) {
            const Vec3 projected = point.pos_b_bt - point.normal * point.distance_bt;
            const Vec3 diff = point.pos_a_bt - projected;
            remove = Dot(diff, diff) > threshold_sq;
        }
        if (remove) {
            manifold.points[i] = manifold.points[kept - 1];
            --kept;
        }
    }
    manifold.count = static_cast<uint8_t>(kept);
}

ROCKETGPU_HD float CarCarRetainedRes(const Vec3& local, const Vec3& p1, const Vec3& p2, const Vec3& p3) {
    const Vec3 c = Cross(local - p1, p2 - p3);
    return Dot(c, c);
}

ROCKETGPU_HD void AddCarCarRetainedPoint(
    CarCarRetainedManifold& manifold, const CarCarRetainedPoint& point, float threshold_bt) {
    const int count = manifold.count;
    float shortest = threshold_bt * threshold_bt;
    int nearest = -1;
    for (int i = 0; i < count; ++i) {
        const Vec3 d = manifold.points[i].local_b_bt - point.local_b_bt;
        const float dist_sq = Dot(d, d);
        if (dist_sq < shortest) {
            shortest = dist_sq;
            nearest = i;
        }
    }
    if (nearest >= 0) {
        manifold.points[nearest] = point;
        return;
    }
    if (count < 4) {
        manifold.points[count] = point;
        manifold.count = static_cast<uint8_t>(count + 1);
        return;
    }
    int max_pen_idx = 4;
    float max_pen = point.distance_bt;
    for (int i = 0; i < 4; ++i) {
        if (manifold.points[i].distance_bt < max_pen) {
            max_pen_idx = i;
            max_pen = manifold.points[i].distance_bt;
        }
    }
    const Vec3& l = point.local_b_bt;
    const Vec3& p0 = manifold.points[0].local_b_bt;
    const Vec3& p1 = manifold.points[1].local_b_bt;
    const Vec3& p2 = manifold.points[2].local_b_bt;
    const Vec3& p3 = manifold.points[3].local_b_bt;
    const float res0 = max_pen_idx == 0 ? 0.0f : CarCarRetainedRes(l, p1, p3, p2);
    const float res1 = max_pen_idx == 1 ? 0.0f : CarCarRetainedRes(l, p0, p3, p2);
    const float res2 = max_pen_idx == 2 ? 0.0f : CarCarRetainedRes(l, p0, p3, p1);
    const float res3 = max_pen_idx == 3 ? 0.0f : CarCarRetainedRes(l, p0, p2, p1);
    int biggest = 0;
    float max_area = res0;
    if (res1 > max_area) {
        biggest = 1;
        max_area = res1;
    }
    if (res2 > max_area) {
        biggest = 2;
        max_area = res2;
    }
    if (res3 > max_area) {
        biggest = 3;
    }
    manifold.points[biggest] = point;
}

ROCKETGPU_HD bool CarCarCentersBeyondBoundingSpheres(const Vec3& pos_a, const Vec3& pos_b, float ra, float rb) {
    const float min_dist = ra + rb;
    const float min_dist_sq = min_dist * min_dist;
    return !(LengthSq(pos_b - pos_a) < min_dist_sq);
}

ROCKETGPU_HD void CarCarCcmGateMasks(
    int count, const CarState* cars, const CarConfig* car_configs, const CarCarRetainedManifold* retained,
    uint32_t& reject_slots) {
    reject_slots = 0;
    if (count < 2) {
        return;
    }
#if defined(__CUDA_ARCH__)
    if (count == 2) {
        const CarState& c0 = cars[0];
        const CarState& c1 = cars[1];
        const uint32_t pair_live =
            static_cast<uint32_t>((c0.is_demoed == 0) & (c0.just_respawned == 0) & (c1.is_demoed == 0) & (c1.just_respawned == 0));
        const uint32_t empty = retained[0].count == 0;
        const uint32_t beyond = CarCarCentersBeyondBoundingSpheres(
            c1.pos, c0.pos, BoundingSphereRadius(car_configs[1]), BoundingSphereRadius(car_configs[0]));
        reject_slots = empty & (beyond | (pair_live ^ 1u));
        return;
    }
#endif
    Vec3 pos[8];
    float radius[8];
    uint32_t live = 0;
#if defined(__CUDA_ARCH__)
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int kk = k < count ? k : 0;
        const CarState& car = cars[kk];
        pos[k] = car.pos;
        radius[k] = k < count ? BoundingSphereRadius(car_configs[kk]) : 0.0f;
        live |= static_cast<uint32_t>((k < count) & (car.is_demoed == 0) & (car.just_respawned == 0)) << k;
    }
    uint32_t reject = 0;
#pragma unroll
    for (int hi = 1; hi < 8; ++hi) {
#pragma unroll
        for (int lo = 0; lo < hi; ++lo) {
            const int slot = CarCarRetainedSlot(lo, hi);
            const uint32_t pair_live = (live >> lo) & (live >> hi) & 1u;
            const uint32_t empty = retained[hi < count ? slot : 0].count == 0;
            const uint32_t beyond = CarCarCentersBeyondBoundingSpheres(pos[hi], pos[lo], radius[hi], radius[lo]);
            reject |= (empty & (beyond | (pair_live ^ 1u))) << slot;
        }
    }
#else
    const int n = count < 8 ? count : 8;
    for (int k = 0; k < n; ++k) {
        const CarState& car = cars[k];
        pos[k] = car.pos;
        radius[k] = BoundingSphereRadius(car_configs[k]);
        live |= static_cast<uint32_t>((car.is_demoed == 0) & (car.just_respawned == 0)) << k;
    }
    uint32_t reject = 0;
    for (int hi = 1; hi < n; ++hi) {
        for (int lo = 0; lo < hi; ++lo) {
            const int slot = CarCarRetainedSlot(lo, hi);
            const uint32_t pair_live = (live >> lo) & (live >> hi) & 1u;
            const uint32_t empty = retained[slot].count == 0;
            const uint32_t beyond = CarCarCentersBeyondBoundingSpheres(pos[hi], pos[lo], radius[hi], radius[lo]);
            reject |= (empty & (beyond | (pair_live ^ 1u))) << slot;
        }
    }
#endif
    reject_slots = reject;
}

struct CarCarEmptyFallbackState {};

ROCKETGPU_HD ROCKETGPU_NOINLINE bool PrepareSharedCarCarCollisionContacts(
    const CarState& a,
    const CarConfig& cfg_a,
    const CarState& b,
    const CarConfig& cfg_b,
    CarCarPairCachedContact* pair_cache,
    bool use_cached_contact,
    float min_dist_sq,
    CarCarRetainedManifold& retained_manifold,
    Vec3& normal,
    int& demo_contact_count,
    float* demo_local_x_a,
    float* demo_local_x_b) {
    const float ccm_threshold_bt = Min(CarCarBreakingThresholdBt(cfg_a), CarCarBreakingThresholdBt(cfg_b));
    RefreshCarCarRetainedManifold(retained_manifold, a, b, ccm_threshold_bt);
    const bool ccm_fresh_gate = LengthSq(b.pos - a.pos) < min_dist_sq;
    if (!ccm_fresh_gate && retained_manifold.count == 0) {
        if (pair_cache) {
            ClearCarCarPairCachedContact(*pair_cache);
        }
        return false;
    }
    UninitArray<ObbData, 2> ccm_boxes_bt;
    ccm_boxes_bt[0] = MakeCarObbBulletTransformBt(a, cfg_a);
    ccm_boxes_bt[1] = MakeCarObbBulletTransformBt(b, cfg_b);
    const ObbData& box_a = ccm_boxes_bt[0];
    const ObbData& box_b = ccm_boxes_bt[1];
    bool aabbs_overlap = true;
    for (int k = 0; k < 3; ++k) {
        float ext_a = 0.0f, ext_b = 0.0f;
        for (int i = 0; i < 3; ++i) {
            ext_a += fabsf((&box_a.axis[i].x)[k]) * (&box_a.half_extent.x)[i];
            ext_b += fabsf((&box_b.axis[i].x)[k]) * (&box_b.half_extent.x)[i];
        }
        const float ca = (&box_a.center.x)[k], cb = (&box_b.center.x)[k];
        aabbs_overlap = aabbs_overlap && !(ca - ext_a > cb + ext_b || ca + ext_a < cb - ext_b);
    }
    if (!aabbs_overlap) {
        retained_manifold.count = 0;
        if (pair_cache) {
            ClearCarCarPairCachedContact(*pair_cache);
        }
        return false;
    }

    constexpr float kFreshContactThresholdUU = 0.0f;
    float contact_overlap_uu = 0.0f;
    ObbContactFeature contact_feature;
    Vec3 current_normal(1.0f, 0.0f, 0.0f);
    ObbContactFeature current_feature;
    float current_overlap_bt = 0.0f;
    const bool have_current_contact =
        !use_cached_contact &&
        ccm_fresh_gate &&
        ComputeObbContactGameOrder(
            ccm_boxes_bt[0],
            ccm_boxes_bt[1],
            kFreshContactThresholdUU,
            current_normal,
            current_overlap_bt,
            &current_feature);
    const float current_overlap_uu = current_overlap_bt * UnitsBTToUU();
    bool have_contact = use_cached_contact;
    if (use_cached_contact) {
        normal = pair_cache->normal;
        contact_overlap_uu = GetCarCarCachedPenetrationUU(*pair_cache);
    } else if (have_current_contact) {
        normal = current_normal;
        contact_overlap_uu = current_overlap_uu;
        contact_feature = current_feature;
        have_contact = true;
    }
    const bool ccm_fresh = have_contact;
    if (!have_contact && retained_manifold.count > 0) {
        normal = retained_manifold.points[0].normal;
        contact_overlap_uu = 0.0f;
        have_contact = true;
    }
    if (!have_contact) {
        if (pair_cache) {
            ClearCarCarPairCachedContact(*pair_cache);
        }
        return false;
    }

    Vec3 contact_points_a[kMaxApproxCarCarContactPoints];
    Vec3 contact_points_b[kMaxApproxCarCarContactPoints];
    Vec3 contact_points_a_bt[kMaxApproxCarCarContactPoints];
    Vec3 contact_points_b_bt[kMaxApproxCarCarContactPoints];
    Vec3 contact_game_points_a_bt[kMaxApproxCarCarContactPoints];
    float contact_game_depths_bt[kMaxApproxCarCarContactPoints];
    bool have_contact_game_points = false;
    Vec3 contact_game_normal_bt(0.0f, 0.0f, 0.0f);

    int contact_point_count = 1;
    if (use_cached_contact) {
        contact_point_count = Min(static_cast<int>(pair_cache->contact_count), kMaxApproxCarCarContactPoints);
        if (contact_point_count <= 0) {
            contact_point_count = 1;
        }
        for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
            contact_points_a[contact_idx] = pair_cache->contacts[contact_idx].point_a;
            contact_points_b[contact_idx] = pair_cache->contacts[contact_idx].point_b;
            if (pair_cache->contacts[contact_idx].has_bt_points) {
                contact_points_a_bt[contact_idx] = pair_cache->contacts[contact_idx].point_a_bt;
                contact_points_b_bt[contact_idx] = pair_cache->contacts[contact_idx].point_b_bt;
            } else {
                contact_points_a_bt[contact_idx] = ToBtLinearVec(contact_points_a[contact_idx]);
                contact_points_b_bt[contact_idx] = ToBtLinearVec(contact_points_b[contact_idx]);
            }
        }
    } else if (!ccm_fresh) {
        contact_point_count = Min(static_cast<int>(retained_manifold.count), kMaxApproxCarCarContactPoints);
        for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
            contact_points_a[contact_idx] = FromBtLinearVec(retained_manifold.points[contact_idx].pos_a_bt);
            contact_points_b[contact_idx] = FromBtLinearVec(retained_manifold.points[contact_idx].pos_b_bt);
        }
    } else {
        contact_point_count = ComputeApproxCarCarContactAnchorPairsWithBtObb(
            a,
            cfg_a,
            ccm_boxes_bt[0],
            b,
            cfg_b,
            ccm_boxes_bt[1],
            normal,
            contact_feature,
            contact_points_a,
            contact_points_b,
            kMaxApproxCarCarContactPoints,
            contact_overlap_uu,
            contact_points_a_bt,
            contact_points_b_bt,
            true,
            contact_game_points_a_bt,
            contact_game_depths_bt,
            &have_contact_game_points,
            &contact_game_normal_bt);
    }

    demo_contact_count = 0;
    const bool ccm_touch = !ccm_fresh;
    if (!use_cached_contact && (have_current_contact || ccm_touch) && contact_point_count > 0) {
        demo_contact_count = contact_point_count;
        for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
            demo_local_x_a[contact_idx] = InverseRotateCarBasis(a, contact_points_a[contact_idx] - a.pos).x;
            demo_local_x_b[contact_idx] = InverseRotateCarBasis(b, contact_points_b[contact_idx] - b.pos).x;
        }
    }

    if (ccm_fresh && contact_point_count > 0) {
        for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
            const Vec3 point_on_a =
                have_contact_game_points ? contact_game_points_a_bt[contact_idx] : contact_points_a_bt[contact_idx];
            const float depth = have_contact_game_points
                                    ? contact_game_depths_bt[contact_idx]
                                    : Dot(contact_points_b_bt[contact_idx] - point_on_a, normal);
            if (depth > ccm_threshold_bt) {
                continue;
            }
            const Vec3 point_normal = have_contact_game_points ? contact_game_normal_bt : normal;
            CarCarRetainedPoint point;
            point.normal = point_normal;
            point.local_a_bt = CarBodyLocalPointBt(a, point_on_a);
            point.local_b_bt = CarBodyLocalPointBt(b, point_on_a + point_normal * depth);
            point.distance_bt = depth;
            AddCarCarRetainedPoint(retained_manifold, point, ccm_threshold_bt);
        }
        RefreshCarCarRetainedManifold(retained_manifold, a, b, ccm_threshold_bt);
    }
    return true;
}

ROCKETGPU_HD ROCKETGPU_TRIANGLE_CONTACT_NOINLINE bool EmitSharedCarCarContactRows(
    CarState& a,
    CarState& b,
    const CarConfig& cfg_a,
    const CarConfig& cfg_b,
    const MutatorConfig& mutator,
    const PreContactCar& base_a,
    const PreContactCar& base_b,
    const Vec3& solver_external_linear_delta_a,
    const Vec3& solver_external_linear_delta_b,
    const Vec3& external_angular_delta_a,
    const Vec3& external_angular_delta_b,
    float dt,
    int respawn_slot_a,
    int respawn_slot_b,
    int pair_idx,
    CarCarPairCachedContact* pair_cache,
    const CarCarRetainedManifold& retained_manifold,
    const Vec3& normal,
    EnvStridedArray<CarSharedSplitContactRow> shared_split_rows,
    int* shared_split_row_count,
    EnvStridedArray<CarSharedVelocityContactRow> shared_velocity_rows,
    int* shared_velocity_row_count,
    EnvStridedArray<CarSharedFrictionContactRow> shared_friction_rows,
    int* shared_friction_row_count) {
    const int contact_point_count = Min(static_cast<int>(retained_manifold.count), kMaxApproxCarCarContactPoints);
    if (contact_point_count == 0) {
        if (pair_cache) {
            ClearCarCarPairCachedContact(*pair_cache);
        }
        return false;
    }
    float max_penetration_uu = 0.0f;
    for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
        max_penetration_uu = Max(max_penetration_uu, -retained_manifold.points[contact_idx].distance_bt * UnitsBTToUU());
    }
    const float penetration = Max(0.0f, max_penetration_uu);
    float resolved_normal_impulse[kMaxApproxCarCarContactPoints] = {};
    float resolved_friction_impulse[kMaxApproxCarCarContactPoints] = {};
    Vec3 cc_inv_bt_a0, cc_inv_bt_a1, cc_inv_bt_a2;
    Vec3 cc_inv_bt_b0, cc_inv_bt_b1, cc_inv_bt_b2;
    BuildInvInertiaTensorWorldBt(a, cfg_a, mutator.car_mass, cc_inv_bt_a0, cc_inv_bt_a1, cc_inv_bt_a2);
    BuildInvInertiaTensorWorldBt(b, cfg_b, mutator.car_mass, cc_inv_bt_b0, cc_inv_bt_b1, cc_inv_bt_b2);
    const CarInvInertiaRowsUu cc_inv_uu_a = MakeCarInvInertiaRowsUu(a, cfg_a, mutator.car_mass);
    const CarInvInertiaRowsUu cc_inv_uu_b = MakeCarInvInertiaRowsUu(b, cfg_b, mutator.car_mass);
    const float denom_inv_mass_bt = 1.0f / mutator.car_mass;
    const Vec3 lin_a_bt = base_a.vel_bt;
    const Vec3 lin_b_bt = base_b.vel_bt;
    const Vec3 ext_lin_a_bt = a.ext_force_impulse_bt_valid
        ? a.ext_force_impulse_bt
        : ToBtLinearVec(solver_external_linear_delta_a);
    const Vec3 ext_lin_b_bt = b.ext_force_impulse_bt_valid
        ? b.ext_force_impulse_bt
        : ToBtLinearVec(solver_external_linear_delta_b);
    const Vec3 ext_ang_a_bt = a.ext_force_impulse_bt_valid ? a.ext_torque_impulse : external_angular_delta_a;
    const Vec3 ext_ang_b_bt = b.ext_force_impulse_bt_valid ? b.ext_torque_impulse : external_angular_delta_b;
    const Vec3 base_lin_a_bt = lin_a_bt + ext_lin_a_bt;
    const Vec3 base_lin_b_bt = lin_b_bt + ext_lin_b_bt;

    for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
        const CarCarRetainedPoint& retained_pt = retained_manifold.points[contact_idx];
        const Vec3 normal_pt = retained_pt.normal;
        const Vec3 contact_point_a_bt = retained_pt.pos_a_bt;
        const Vec3 contact_point_b_bt = retained_pt.pos_b_bt;
        const Vec3 rel_a_bt = contact_point_a_bt - a.pos_bt;
        const Vec3 rel_b_bt = contact_point_b_bt - b.pos_bt;
        const Vec3 rel_a = FromBtLinearVec(rel_a_bt);
        const Vec3 rel_b = FromBtLinearVec(rel_b_bt);
        CarCarPairCachedPoint* cached_point =
            pair_cache && contact_idx < kMaxApproxCarCarContactPoints
                ? &pair_cache->contacts[contact_idx]
                : nullptr;
        const float accumulated_normal_impulse =
            (cached_point && cached_point->valid != 0) ? cached_point->applied_normal_impulse : 0.0f;
        const float accumulated_friction_impulse =
            (cached_point && cached_point->valid != 0) ? cached_point->applied_friction_impulse : 0.0f;
        resolved_normal_impulse[contact_idx] = accumulated_normal_impulse;
        resolved_friction_impulse[contact_idx] = accumulated_friction_impulse;

        const Vec3 normal_a_bt = -normal_pt;
        const Vec3 normal_b_bt = normal_pt;
        const Vec3 rel_a_cross_normal_bt = Cross(rel_a_bt, normal_a_bt);
        const Vec3 denom_ang_a = Mul3x3Vec(cc_inv_bt_a0, cc_inv_bt_a1, cc_inv_bt_a2, rel_a_cross_normal_bt);
        const Vec3 denom_vec_a = Cross(denom_ang_a, rel_a_bt);
        const float denom_a = denom_inv_mass_bt + Dot(normal_a_bt, denom_vec_a);

        const Vec3 rel_b_cross_normal_bt = Cross(rel_b_bt, -normal_a_bt);
        const Vec3 denom_ang_b = Mul3x3Vec(cc_inv_bt_b0, cc_inv_bt_b1, cc_inv_bt_b2, rel_b_cross_normal_bt);
        const Vec3 denom_vec_b = Cross(-denom_ang_b, rel_b_bt);
        const float denom_b = denom_inv_mass_bt + Dot(normal_a_bt, denom_vec_b);
        const float denom = denom_a + denom_b;

        if (!(denom <= 1.0e-6f) && *shared_velocity_row_count < kMaxCarPairSharedVelocityContactRows) {
            const float vel1_dotn_bt =
                Dot(normal_a_bt, base_lin_a_bt) +
                Dot(rel_a_cross_normal_bt, base_a.ang_vel + ext_ang_a_bt);
            const float vel2_dotn_bt =
                Dot(normal_b_bt, base_lin_b_bt) +
                Dot(rel_b_cross_normal_bt, base_b.ang_vel + ext_ang_b_bt);
            const float base_vn_bt = vel1_dotn_bt + vel2_dotn_bt;
            const float jac_diag_ab_inv = 1.0f / denom;

            const Vec3 vel_a_no_ext_bt = lin_a_bt + Cross(base_a.ang_vel, rel_a_bt);
            const Vec3 vel_b_no_ext_bt = lin_b_bt + Cross(base_b.ang_vel, rel_b_bt);
            const float base_vn_no_ext_bt = Dot(normal_b_bt, vel_b_no_ext_bt - vel_a_no_ext_bt);
            const float restitution_velocity_bt = Max(CarCarCollisionRestitution() * -base_vn_no_ext_bt, 0.0f);
            const float rhs_bt = (restitution_velocity_bt - base_vn_bt) * jac_diag_ab_inv;
            const float rhs_uu = rhs_bt * UnitsBTToUU();

            const Vec3 shared_rel_a_cross_normal = Cross(rel_a, normal_a_bt);
            const Vec3 shared_rel_b_cross_normal = Cross(rel_b, normal_b_bt);
            const Vec3 shared_angular_component_a = ApplyInverseBodyInertiaWorld(cc_inv_uu_a, shared_rel_a_cross_normal);
            const Vec3 shared_angular_component_b = ApplyInverseBodyInertiaWorld(cc_inv_uu_b, shared_rel_b_cross_normal);
            CarSharedVelocityContactRow& shared_row = shared_velocity_rows[*shared_velocity_row_count];
            shared_row = CarSharedVelocityContactRow{};
            shared_row.valid = 1;
            shared_row.source_tag = 2000 + contact_idx;
            shared_row.body_a_local_idx = respawn_slot_a;
            shared_row.body_b_local_idx = respawn_slot_b;
            shared_row.pair_cache_idx = pair_idx;
            shared_row.pair_contact_idx = contact_idx;
            shared_row.normal_a = normal_a_bt;
            shared_row.rel_contact_a = rel_a;
            shared_row.rel_contact_b = rel_b;
            shared_row.rel_a_cross_normal = shared_rel_a_cross_normal;
            shared_row.rel_b_cross_normal = shared_rel_b_cross_normal;
            shared_row.angular_component_a = shared_angular_component_a;
            shared_row.angular_component_b = shared_angular_component_b;
            shared_row.inv_mass_a = denom_inv_mass_bt;
            shared_row.inv_mass_b = denom_inv_mass_bt;
            shared_row.jac_diag_ab_inv = jac_diag_ab_inv;
            shared_row.rhs = rhs_uu;
            shared_row.rhs_bt = rhs_bt;
            shared_row.has_rhs_bt = 1;
            shared_row.rel_a_cross_normal_bt = rel_a_cross_normal_bt;
            shared_row.rel_b_cross_normal_bt = rel_b_cross_normal_bt;
            shared_row.angular_component_a_bt = denom_ang_a;
            shared_row.angular_component_b_bt = denom_ang_b;
            shared_row.has_bt = 1;
            shared_row.applied_impulse = accumulated_normal_impulse;
            const int shared_normal_row_idx = *shared_velocity_row_count;
            *shared_velocity_row_count += 1;

            if (*shared_friction_row_count < kMaxCarPairSharedFrictionContactRows) {
                const Vec3 base_vel_a_bt =
                    base_lin_a_bt +
                    Cross(base_a.ang_vel + ext_ang_a_bt, rel_a_bt);
                const Vec3 base_vel_b_bt =
                    base_lin_b_bt +
                    Cross(base_b.ang_vel + ext_ang_b_bt, rel_b_bt);
                const Vec3 base_rel_vel_bt = base_vel_b_bt - base_vel_a_bt;
                Vec3 tangent = base_rel_vel_bt - normal_pt * Dot(base_rel_vel_bt, normal_pt);
                const float lat_rel_vel_bt = LengthSq(tangent);
                const float tangent_len = SqrtScalar(lat_rel_vel_bt);
                tangent = lat_rel_vel_bt > 1.1920928955078125e-07f
                    ? tangent * (1.0f / tangent_len)
                    : BulletPlaneSpace1First(normal_pt);

                const Vec3 tangent_a = tangent;
                const Vec3 tangent_b = -tangent;
                const Vec3 tangent_angular_component_a = ApplyInverseBodyInertiaWorld(cc_inv_uu_a, Cross(rel_a, tangent_a));
                const Vec3 tangent_angular_component_b = ApplyInverseBodyInertiaWorld(cc_inv_uu_b, Cross(rel_b, tangent_b));
                const Vec3 rel_a_cross_tangent_bt = Cross(rel_a_bt, tangent_a);
                const Vec3 rel_b_cross_tangent_bt = Cross(rel_b_bt, tangent_b);
                const Vec3 denom_angular_component_a_bt = Mul3x3Vec(cc_inv_bt_a0, cc_inv_bt_a1, cc_inv_bt_a2, rel_a_cross_tangent_bt);
                const Vec3 denom_angular_component_b_bt = Mul3x3Vec(cc_inv_bt_b0, cc_inv_bt_b1, cc_inv_bt_b2, rel_b_cross_tangent_bt);
                const Vec3 denom_vec_a_bt = Cross(denom_angular_component_a_bt, rel_a_bt);
                const Vec3 denom_vec_b_bt = Cross(-denom_angular_component_b_bt, rel_b_bt);
                const float denom_dot_a_bt = Dot(tangent_a, denom_vec_a_bt);
                const float denom_dot_b_bt = Dot(tangent_a, denom_vec_b_bt);
                const float denom_t_a = denom_inv_mass_bt + denom_dot_a_bt;
                const float denom_t_b = denom_inv_mass_bt + denom_dot_b_bt;
                const float denom_t = denom_t_a + denom_t_b;
                if (!(denom_t <= 1.0e-6f)) {
                    CarSharedFrictionContactRow& friction_row = shared_friction_rows[*shared_friction_row_count];
                    friction_row = CarSharedFrictionContactRow{};
                    friction_row.valid = 1;
                    friction_row.normal_row_idx = shared_normal_row_idx;
                    friction_row.source_tag = 3000 + contact_idx;
                    friction_row.body_a_local_idx = respawn_slot_a;
                    friction_row.body_b_local_idx = respawn_slot_b;
                    friction_row.pair_cache_idx = pair_idx;
                    friction_row.pair_contact_idx = contact_idx;
                    friction_row.tangent_a = tangent_a;
                    friction_row.rel_contact_a = rel_a;
                    friction_row.rel_contact_b = rel_b;
                    friction_row.rel_a_cross_tangent = Cross(rel_a, tangent_a);
                    friction_row.rel_b_cross_tangent = Cross(rel_b, tangent_b);
                    friction_row.angular_component_a = tangent_angular_component_a;
                    friction_row.angular_component_b = tangent_angular_component_b;
                    friction_row.inv_mass_a = denom_inv_mass_bt;
                    friction_row.inv_mass_b = denom_inv_mass_bt;
                    friction_row.jac_diag_ab_inv = 1.0f / denom_t;
                    const float rhs_lin1_dott_bt = Dot(tangent_a, base_lin_a_bt);
                    const float rhs_ang1_dott_bt = Dot(rel_a_cross_tangent_bt, base_a.ang_vel);
                    const float rhs_lin2_dott_bt = Dot(tangent_b, base_lin_b_bt);
                    const float rhs_ang2_dott_bt = Dot(rel_b_cross_tangent_bt, base_b.ang_vel);
                    const float rhs_vel1_dott_bt = rhs_lin1_dott_bt + rhs_ang1_dott_bt;
                    const float rhs_vel2_dott_bt = rhs_lin2_dott_bt + rhs_ang2_dott_bt;

                    const float rhs_rel_t_bt = -(rhs_vel1_dott_bt + rhs_vel2_dott_bt);
                    const float friction_rhs_bt = rhs_rel_t_bt * friction_row.jac_diag_ab_inv;
                    friction_row.rhs_bt = friction_rhs_bt;
                    friction_row.has_rhs_bt = 1;

                    friction_row.rel_a_cross_tangent_bt = rel_a_cross_tangent_bt;
                    friction_row.rel_b_cross_tangent_bt = rel_b_cross_tangent_bt;
                    friction_row.angular_component_a_bt = denom_angular_component_a_bt;
                    friction_row.angular_component_b_bt = denom_angular_component_b_bt;
                    friction_row.has_bt = 1;
                    friction_row.rhs = friction_rhs_bt * UnitsBTToUU();
                    friction_row.friction = CarCarCollisionFriction();
                    friction_row.applied_impulse = accumulated_friction_impulse;
                    *shared_friction_row_count += 1;
                }
            }
        }

        if (penetration > 0.0f && dt > 0.0f) {
            const Vec3 contact_point_a = FromBtLinearVec(contact_point_a_bt);
            const Vec3 contact_point_b = FromBtLinearVec(contact_point_b_bt);
            const Vec3 split_rel_a = CarCarBulletRelativePointUU(contact_point_a, a.pos_bt);
            const Vec3 split_rel_b = CarCarBulletRelativePointUU(contact_point_b, b.pos_bt);
            const Vec3 point_delta = contact_point_b - contact_point_a;
            const float point_penetration = Max(0.0f, -Dot(point_delta, normal_pt));
            if (!(point_penetration <= 0.0f)) {
                const Vec3 rel_a_cross_normal = Cross(split_rel_a, normal_a_bt);
                const Vec3 rel_b_cross_normal = Cross(split_rel_b, normal_b_bt);
                const Vec3 angular_component_a =
                    ApplyInverseBodyInertiaWorld(cc_inv_uu_a, rel_a_cross_normal);
                const Vec3 angular_component_b =
                    ApplyInverseBodyInertiaWorld(cc_inv_uu_b, rel_b_cross_normal);

                float split_denom = 0.0f;
                split_denom += denom_inv_mass_bt + Dot(normal_a_bt, Cross(angular_component_a, split_rel_a));
                split_denom += denom_inv_mass_bt + Dot(normal_b_bt, Cross(angular_component_b, split_rel_b));
                if (!(split_denom <= 1.0e-8f)) {
                    const float jac_diag_ab_inv = 1.0f / split_denom;
                    const float rhs_penetration =
                        ((point_penetration * SplitImpulsePenetrationErp()) / dt) * jac_diag_ab_inv;
                    const float point_penetration_bt =
                        Max(0.0f, -Dot(contact_point_b_bt - contact_point_a_bt, normal_pt));
                    float denom_bt = 0.0f;
                    denom_bt += denom_inv_mass_bt + Dot(normal_a_bt, Cross(denom_ang_a, rel_a_bt));
                    denom_bt += denom_inv_mass_bt + Dot(normal_b_bt, Cross(denom_ang_b, rel_b_bt));
                    const float jac_diag_ab_inv_bt = denom_bt > 1.0e-8f ? 1.0f / denom_bt : 0.0f;
                    const float rhs_penetration_bt =
                        ((point_penetration_bt * SplitImpulsePenetrationErp()) / dt) * jac_diag_ab_inv_bt;
                    const int row_before = *shared_split_row_count;
                    AppendCarPairSharedSplitRow(
                        shared_split_rows,
                        *shared_split_row_count,
                        respawn_slot_a,
                        respawn_slot_b,
                        1000 + contact_idx,
                        normal_a_bt,
                        normal_b_bt,
                        split_rel_a,
                        split_rel_b,
                        angular_component_a,
                        angular_component_b,
                        denom_inv_mass_bt,
                        denom_inv_mass_bt,
                        jac_diag_ab_inv,
                        rhs_penetration);
                    if (*shared_split_row_count == row_before + 1 && jac_diag_ab_inv_bt > 0.0f &&
                        point_penetration_bt > 0.0f) {
                        CarSharedSplitContactRow& row_bt = shared_split_rows[row_before];
                        row_bt.rel_a_cross_normal_bt = rel_a_cross_normal_bt;
                        row_bt.angular_component_a_bt = denom_ang_a;
                        row_bt.rel_b_cross_normal_bt = rel_b_cross_normal_bt;
                        row_bt.angular_component_b_bt = denom_ang_b;
                        row_bt.jac_diag_ab_inv_bt = jac_diag_ab_inv_bt;
                        row_bt.rhs_penetration_bt = rhs_penetration_bt;
                        row_bt.applied_push_impulse_bt = 0.0f;
                        row_bt.has_bt = 1;
                    }
                }
            }
        }
    }
    {
        const Vec3 rocketgpu_dva = a.vel - a.vel;
        if (rocketgpu_dva.x != 0.0f || rocketgpu_dva.y != 0.0f || rocketgpu_dva.z != 0.0f) {
            SetCarLinearVelocityUu(a, a.vel + rocketgpu_dva);
        }
    }
    a.ang_vel += a.ang_vel - a.ang_vel;
    SetCarOriginBt(a, a.pos_bt + (a.pos_bt - a.pos_bt));
    {
        const Vec3 rocketgpu_dvb = b.vel - b.vel;
        if (rocketgpu_dvb.x != 0.0f || rocketgpu_dvb.y != 0.0f || rocketgpu_dvb.z != 0.0f) {
            SetCarLinearVelocityUu(b, b.vel + rocketgpu_dvb);
        }
    }
    b.ang_vel += b.ang_vel - b.ang_vel;
    SetCarOriginBt(b, b.pos_bt + (b.pos_bt - b.pos_bt));

    if (pair_cache) {
        pair_cache->valid = 1;
        pair_cache->normal = normal;
        pair_cache->contact_count = static_cast<uint8_t>(contact_point_count);
        for (int contact_idx = 0; contact_idx < kMaxApproxCarCarContactPoints; ++contact_idx) {
            CarCarPairCachedPoint& cached_point = pair_cache->contacts[contact_idx];
            if (contact_idx < contact_point_count) {
                const CarCarRetainedPoint& retained_pt = retained_manifold.points[contact_idx];
                cached_point.valid = 1;
                cached_point.point_a = FromBtLinearVec(retained_pt.pos_a_bt);
                cached_point.point_b = FromBtLinearVec(retained_pt.pos_b_bt);
                cached_point.point_a_bt = retained_pt.pos_a_bt;
                cached_point.point_b_bt = retained_pt.pos_b_bt;
                cached_point.has_bt_points = 1;
                cached_point.penetration_uu = penetration;
                cached_point.applied_normal_impulse = resolved_normal_impulse[contact_idx];
                cached_point.applied_friction_impulse = resolved_friction_impulse[contact_idx];
            } else {
                cached_point = CarCarPairCachedPoint{};
            }
        }
        return true;
    }
    return false;
}

struct CarCarFallbackState {
#if defined(__CUDA_ARCH__)
    CarState predicted_a;
    CarState predicted_b;
#else
    UninitArray<CarState, 2> predicted_storage;
#endif
    ObbData current_obb_a;
    ObbData current_obb_b;
    ObbData obb_a;
    ObbData obb_b;
    CarState solver_seed_a;
    CarState solver_seed_b;
    CarState solver_a;
    CarState solver_b;
    Vec3 resolved_tangent[kMaxApproxCarCarContactPoints];
    uint8_t resolved_tangent_valid[kMaxApproxCarCarContactPoints] = {};
    Vec3 contact_rel_a[kMaxApproxCarCarContactPoints];
    Vec3 contact_rel_b[kMaxApproxCarCarContactPoints];
    Vec3 contact_rel_a_bt[kMaxApproxCarCarContactPoints];
    Vec3 contact_rel_b_bt[kMaxApproxCarCarContactPoints];
    float normal_row_jac_diag_ab_inv[kMaxApproxCarCarContactPoints] = {};
    float normal_row_rhs[kMaxApproxCarCarContactPoints] = {};
    float normal_row_rhs_bt[kMaxApproxCarCarContactPoints] = {};
    uint8_t normal_row_valid[kMaxApproxCarCarContactPoints] = {};
};

template <bool kSharedCcm, bool kRequireScratch = false>
ROCKETGPU_HD ROCKETGPU_NOINLINE bool ResolveCarCarCollisionImpl(
    CarState& a,
    uint32_t id_a,
    Team team_a,
    CarState& b,
    uint32_t id_b,
    Team team_b,
    const CarConfig& cfg_a,
    const CarConfig& cfg_b,
    const MutatorConfig& mutator,
    const PreContactCar& base_a,
    const PreContactCar& base_b,
    const Vec3& external_linear_delta_a,
    const Vec3& external_linear_delta_b,
    const Vec3& external_angular_delta_a,
    const Vec3& external_angular_delta_b,
    float dt,
    uint8_t enable_demos,
    int respawn_slot_a,
    int respawn_slot_b,
    CarCarPairCacheView pair_contact_cache,
    EnvStridedArray<CarSharedSplitContactRow> shared_split_rows,
    int* shared_split_row_count,
    EnvStridedArray<CarSharedVelocityContactRow> shared_velocity_rows,
    int* shared_velocity_row_count,
    EnvStridedArray<CarSharedFrictionContactRow> shared_friction_rows,
    int* shared_friction_row_count,
    const Vec3* start_vel_a,
    const Vec3* start_vel_b,
    CarCarRetainedManifold* retained_manifold,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch) {
    if (a.is_demoed || b.is_demoed || a.just_respawned || b.just_respawned) {
        if (retained_manifold) {
            retained_manifold->count = 0;
        }
        return false;
    }

    int pair_idx = -1;
    if (respawn_slot_a >= 0 && respawn_slot_b >= 0) {
        pair_idx = pair_contact_cache.index(Max(respawn_slot_a, respawn_slot_b), Min(respawn_slot_a, respawn_slot_b));
    }
    CarCarPairCachedContact* pair_cache =
        pair_contact_cache && pair_idx >= 0 ? &pair_contact_cache.slots[pair_idx] : nullptr;
    const bool use_cached_contact =
        pair_cache && pair_cache->valid != 0 && pair_cache->contact_count > 0;
    const bool use_shared_velocity_rows =
        kSharedCcm ||
        (shared_velocity_rows && shared_velocity_row_count &&
         shared_friction_rows && shared_friction_row_count);

    const float ra = BoundingSphereRadius(cfg_a);
    const float rb = BoundingSphereRadius(cfg_b);
    const float min_dist = ra + rb;
    const float min_dist_sq = min_dist * min_dist;
    if constexpr (kSharedCcm) {
        Vec3 normal(1.0f, 0.0f, 0.0f);
        int demo_contact_count = 0;
        float demo_local_x_a[kMaxApproxCarCarContactPoints];
        float demo_local_x_b[kMaxApproxCarCarContactPoints];
        if (!PrepareSharedCarCarCollisionContacts(
                a, cfg_a, b, cfg_b, pair_cache, use_cached_contact, min_dist_sq,
                *retained_manifold, normal, demo_contact_count, demo_local_x_a, demo_local_x_b)) {
            return false;
        }
        if (demo_contact_count > 0) {
            const bool a_is_lower_id = id_a > id_b;
            CarState* hitter1 = a_is_lower_id ? &a : &b;
            CarState* victim1 = a_is_lower_id ? &b : &a;
            const uint32_t hitter1_id = a_is_lower_id ? id_a : id_b;
            const uint32_t victim1_id = a_is_lower_id ? id_b : id_a;
            const Team hitter1_team = a_is_lower_id ? team_a : team_b;
            const Team victim1_team = a_is_lower_id ? team_b : team_a;
            const CarConfig& hitter1_cfg = a_is_lower_id ? cfg_a : cfg_b;
            const Vec3 normal1 = a_is_lower_id ? normal : -normal;
            bool hitter1_demoed_here = false;
            bool victim1_demoed_here = false;
            for (int contact_idx = 0; contact_idx < demo_contact_count; ++contact_idx) {
                if (a.is_demoed && b.is_demoed) {
                    break;
                }
                const float hitter1_bumper_x = a_is_lower_id ? demo_local_x_a[contact_idx] : demo_local_x_b[contact_idx];
                const float hitter2_bumper_x = a_is_lower_id ? demo_local_x_b[contact_idx] : demo_local_x_a[contact_idx];
                if (!victim1->is_demoed && (!hitter1->is_demoed || hitter1_demoed_here)) {
                    (void)ResolveCarCarDirectionalContact<kRequireScratch>(
                        *hitter1, hitter1_id, hitter1_team, hitter1_cfg,
                        *victim1, victim1_id, victim1_team,
                        mutator, enable_demos, normal1, true, hitter1_bumper_x,
                        a_is_lower_id ? start_vel_a : start_vel_b,
                        a_is_lower_id ? &cfg_b : &cfg_a, hitter1->is_demoed != 0,
                        a_is_lower_id ? start_vel_b : start_vel_a, dt, gjk_epa_scratch);
                    victim1_demoed_here = victim1_demoed_here || victim1->is_demoed;
                }
                if (a.is_demoed && b.is_demoed) {
                    break;
                }
                if (!hitter1->is_demoed && (!victim1->is_demoed || victim1_demoed_here)) {
                    (void)ResolveCarCarDirectionalContact<kRequireScratch>(
                        *victim1, victim1_id, victim1_team,
                        a_is_lower_id ? cfg_b : cfg_a,
                        *hitter1, hitter1_id, hitter1_team,
                        mutator, enable_demos, -normal1, true, hitter2_bumper_x,
                        a_is_lower_id ? start_vel_b : start_vel_a,
                        &hitter1_cfg, victim1->is_demoed != 0,
                        a_is_lower_id ? start_vel_a : start_vel_b, dt, gjk_epa_scratch);
                    hitter1_demoed_here = hitter1_demoed_here || hitter1->is_demoed;
                }
            }
        }
        return EmitSharedCarCarContactRows(
            a,
            b,
            cfg_a,
            cfg_b,
            mutator,
            base_a,
            base_b,
            external_linear_delta_a,
            external_linear_delta_b,
            external_angular_delta_a,
            external_angular_delta_b,
            dt,
            respawn_slot_a,
            respawn_slot_b,
            pair_idx,
            pair_cache,
            *retained_manifold,
            normal,
            shared_split_rows,
            shared_split_row_count,
            shared_velocity_rows,
            shared_velocity_row_count,
            shared_friction_rows,
            shared_friction_row_count);
    }
    bool force_current_contact_seed = false;
    bool force_current_from_broadphase_gap = false;

    const bool cal_ccm = kSharedCcm || retained_manifold != nullptr;
    float ccm_threshold_bt = 0.0f;
    if (cal_ccm) {
        ccm_threshold_bt = Min(CarCarBreakingThresholdBt(cfg_a), CarCarBreakingThresholdBt(cfg_b));
        RefreshCarCarRetainedManifold(*retained_manifold, a, b, ccm_threshold_bt);
    }
    const bool ccm_fresh_gate = !cal_ccm || LengthSq(b.pos - a.pos) < min_dist_sq;
    if (cal_ccm && !ccm_fresh_gate && retained_manifold->count == 0) {
        if (pair_cache) {
            ClearCarCarPairCachedContact(*pair_cache);
        }
        return false;
    }
    UninitArray<ObbData, 2> ccm_boxes_bt;
    if (cal_ccm) {
        ccm_boxes_bt[0] = MakeCarObbBulletTransformBt(a, cfg_a);
        ccm_boxes_bt[1] = MakeCarObbBulletTransformBt(b, cfg_b);
        const ObbData& box_a = ccm_boxes_bt[0];
        const ObbData& box_b = ccm_boxes_bt[1];
        bool aabbs_overlap = true;
        for (int k = 0; k < 3; ++k) {
            float ext_a = 0.0f, ext_b = 0.0f;
            for (int i = 0; i < 3; ++i) {
                ext_a += fabsf((&box_a.axis[i].x)[k]) * (&box_a.half_extent.x)[i];
                ext_b += fabsf((&box_b.axis[i].x)[k]) * (&box_b.half_extent.x)[i];
            }
            const float ca = (&box_a.center.x)[k], cb = (&box_b.center.x)[k];
            aabbs_overlap = aabbs_overlap && !(ca - ext_a > cb + ext_b || ca + ext_a < cb - ext_b);
        }
        if (!aabbs_overlap) {
            retained_manifold->count = 0;
            if (pair_cache) {
                ClearCarCarPairCachedContact(*pair_cache);
            }
            return false;
        }
    }

    typename std::conditional<kSharedCcm, CarCarEmptyFallbackState, CarCarFallbackState>::type fb;
    const bool predicted_unused = use_cached_contact || cal_ccm;
    bool extreme_speed_pair = false;
    if constexpr (!kSharedCcm) {
#if defined(__CUDA_ARCH__)
        fb.predicted_a = predicted_unused
            ? a
            : BuildPredictedCarSolverState(a, external_linear_delta_a, external_angular_delta_a, dt, NativeSseMath(mutator));
        fb.predicted_b = predicted_unused
            ? b
            : BuildPredictedCarSolverState(b, external_linear_delta_b, external_angular_delta_b, dt, NativeSseMath(mutator));
        const CarState& predicted_a = fb.predicted_a;
        const CarState& predicted_b = fb.predicted_b;
#else
        if (!predicted_unused) {
            fb.predicted_storage[0] = BuildPredictedCarSolverState(a, external_linear_delta_a, external_angular_delta_a, dt, NativeSseMath(mutator));
            fb.predicted_storage[1] = BuildPredictedCarSolverState(b, external_linear_delta_b, external_angular_delta_b, dt, NativeSseMath(mutator));
        }
        const CarState& predicted_a = predicted_unused ? a : fb.predicted_storage[0];
        const CarState& predicted_b = predicted_unused ? b : fb.predicted_storage[1];
#endif
        const float speed_threshold = mutator.car_max_speed * 4.0f;
        const float speed_threshold_sq = speed_threshold * speed_threshold;
        extreme_speed_pair =
            LengthSq(predicted_a.vel) > speed_threshold_sq ||
            LengthSq(predicted_b.vel) > speed_threshold_sq;
        const Vec3 broad_delta = predicted_b.pos - predicted_a.pos;
        if (!cal_ccm && !use_cached_contact && LengthSq(broad_delta) >= min_dist_sq) {
            if (!extreme_speed_pair) {
                if (pair_cache) {
                    ClearCarCarPairCachedContact(*pair_cache);
                }
                return false;
            }
            const Vec3 current_broad_delta = b.pos - a.pos;
            if (LengthSq(current_broad_delta) >= min_dist_sq) {
                if (pair_cache) {
                    ClearCarCarPairCachedContact(*pair_cache);
                }
                return false;
            }
            force_current_contact_seed = true;
            force_current_from_broadphase_gap = true;
        }
    }
    const Vec3 solver_external_linear_delta_a = external_linear_delta_a;
    const Vec3 solver_external_linear_delta_b = external_linear_delta_b;

    Vec3 normal(1.0f, 0.0f, 0.0f);

    constexpr float kFreshContactThresholdUU = 0.0f;
    float contact_overlap_uu = 0.0f;
    ObbContactFeature contact_feature;
    Vec3 current_normal(1.0f, 0.0f, 0.0f);
    float current_overlap_uu = 0.0f;
    ObbContactFeature current_feature;

    float current_overlap_bt = 0.0f;
    const bool have_current_contact =
        !use_cached_contact &&
        ccm_fresh_gate &&
        ComputeObbContactGameOrder(
            cal_ccm ? ccm_boxes_bt[0] : MakeCarObbBulletTransformBt(a, cfg_a),
            cal_ccm ? ccm_boxes_bt[1] : MakeCarObbBulletTransformBt(b, cfg_b),
            kFreshContactThresholdUU,
            current_normal,
            current_overlap_bt,
            &current_feature);
    current_overlap_uu = current_overlap_bt * UnitsBTToUU();
    bool have_contact = use_cached_contact;
    if (use_cached_contact) {
        normal = pair_cache->normal;
        contact_overlap_uu = GetCarCarCachedPenetrationUU(*pair_cache);
    } else if (have_current_contact) {
        normal = current_normal;
        contact_overlap_uu = current_overlap_uu;
        contact_feature = current_feature;
        force_current_contact_seed = true;
        have_contact = true;
    } else if constexpr (!kSharedCcm) {
#if defined(__CUDA_ARCH__)
        const CarState& predicted_a = fb.predicted_a;
        const CarState& predicted_b = fb.predicted_b;
#else
        const CarState& predicted_a = predicted_unused ? a : fb.predicted_storage[0];
        const CarState& predicted_b = predicted_unused ? b : fb.predicted_storage[1];
#endif
        float predicted_overlap_bt = 0.0f;
        if (!cal_ccm && ComputeObbContactGameOrder(
                        MakeCarObbBulletTransformBt(force_current_contact_seed ? a : predicted_a, cfg_a),
                        MakeCarObbBulletTransformBt(force_current_contact_seed ? b : predicted_b, cfg_b),
                        kFreshContactThresholdUU,
                        normal, predicted_overlap_bt, &contact_feature)) {
            contact_overlap_uu = predicted_overlap_bt * UnitsBTToUU();
        } else if (extreme_speed_pair && have_current_contact) {
            normal = current_normal;
            contact_overlap_uu = current_overlap_uu;
            contact_feature = current_feature;
            force_current_contact_seed = true;
            have_contact = true;
        }
    }
    const bool ccm_fresh = have_contact;
    if (cal_ccm && !have_contact && retained_manifold->count > 0) {
        normal = retained_manifold->points[0].normal;
        contact_overlap_uu = 0.0f;
        force_current_contact_seed = true;
        have_contact = true;
    }
    if (!have_contact) {
        if (pair_cache) {
            ClearCarCarPairCachedContact(*pair_cache);
        }
        return false;
    }

    float bumper_local_x_a = -1.0e30f;
    float bumper_local_x_b = -1.0e30f;
    bool has_bumper_local_x_override = false;
    bool skip_velocity_impulse = false;

    if constexpr (!kSharedCcm) {
#if defined(__CUDA_ARCH__)
        const CarState& predicted_a = fb.predicted_a;
        const CarState& predicted_b = fb.predicted_b;
#else
        const CarState& predicted_a = predicted_unused ? a : fb.predicted_storage[0];
        const CarState& predicted_b = predicted_unused ? b : fb.predicted_storage[1];
#endif
        fb.current_obb_a = MakeCarObb(a, cfg_a);
        fb.current_obb_b = MakeCarObb(b, cfg_b);
        fb.obb_a = (force_current_contact_seed || predicted_unused) ? fb.current_obb_a : MakeCarObb(predicted_a, cfg_a);
        fb.obb_b = (force_current_contact_seed || predicted_unused) ? fb.current_obb_b : MakeCarObb(predicted_b, cfg_b);

        if (!use_cached_contact &&
            extreme_speed_pair &&
            contact_overlap_uu > 0.0f &&
            contact_feature.kind == OBB_CONTACT_FEATURE_EDGE_EDGE) {
            force_current_contact_seed = true;
        }

        skip_velocity_impulse =
            !force_current_contact_seed &&
            !use_cached_contact &&
            extreme_speed_pair;

        fb.solver_seed_a = (force_current_contact_seed || use_cached_contact) ? a : predicted_a;
        fb.solver_seed_b = (force_current_contact_seed || use_cached_contact) ? b : predicted_b;
        if (force_current_from_broadphase_gap) {
            fb.solver_seed_a.vel = predicted_a.vel;
            fb.solver_seed_a.ang_vel = predicted_a.ang_vel;
            fb.solver_seed_b.vel = predicted_b.vel;
            fb.solver_seed_b.ang_vel = predicted_b.ang_vel;
        }
        fb.solver_a = fb.solver_seed_a;
        fb.solver_b = fb.solver_seed_b;
    }

    Vec3 contact_points_a[kMaxApproxCarCarContactPoints];
    Vec3 contact_points_b[kMaxApproxCarCarContactPoints];
    Vec3 contact_points_a_bt[kMaxApproxCarCarContactPoints];
    Vec3 contact_points_b_bt[kMaxApproxCarCarContactPoints];

    Vec3 contact_game_points_a_bt[kMaxApproxCarCarContactPoints];
    float contact_game_depths_bt[kMaxApproxCarCarContactPoints];
    bool have_contact_game_points = false;
    Vec3 contact_game_normal_bt(0.0f, 0.0f, 0.0f);

    int contact_point_count = 1;
    if (use_cached_contact) {
        contact_point_count = Min(static_cast<int>(pair_cache->contact_count), kMaxApproxCarCarContactPoints);
        if (contact_point_count <= 0) {
            contact_point_count = 1;
        }
        for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
            contact_points_a[contact_idx] = pair_cache->contacts[contact_idx].point_a;
            contact_points_b[contact_idx] = pair_cache->contacts[contact_idx].point_b;
            if (pair_cache->contacts[contact_idx].has_bt_points) {
                contact_points_a_bt[contact_idx] = pair_cache->contacts[contact_idx].point_a_bt;
                contact_points_b_bt[contact_idx] = pair_cache->contacts[contact_idx].point_b_bt;
            } else {
                contact_points_a_bt[contact_idx] = ToBtLinearVec(contact_points_a[contact_idx]);
                contact_points_b_bt[contact_idx] = ToBtLinearVec(contact_points_b[contact_idx]);
            }
        }
    } else if (cal_ccm && !ccm_fresh) {
        contact_point_count = Min(static_cast<int>(retained_manifold->count), kMaxApproxCarCarContactPoints);
        for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
            contact_points_a[contact_idx] = FromBtLinearVec(retained_manifold->points[contact_idx].pos_a_bt);
            contact_points_b[contact_idx] = FromBtLinearVec(retained_manifold->points[contact_idx].pos_b_bt);
        }
    } else if constexpr (kSharedCcm) {
        contact_point_count = ComputeApproxCarCarContactAnchorPairsWithBtObb(
            a,
            cfg_a,
            ccm_boxes_bt[0],
            b,
            cfg_b,
            ccm_boxes_bt[1],
            normal,
            contact_feature,
            contact_points_a,
            contact_points_b,
            kMaxApproxCarCarContactPoints,
            contact_overlap_uu,
            contact_points_a_bt,
            contact_points_b_bt,
            cal_ccm,
            contact_game_points_a_bt,
            contact_game_depths_bt,
            &have_contact_game_points,
            &contact_game_normal_bt);
    } else {
        contact_point_count = ComputeApproxCarCarContactAnchorPairs(
            fb.solver_a,
            cfg_a,
            fb.obb_a,
            fb.solver_b,
            cfg_b,
            fb.obb_b,
            normal,
            contact_feature,
            contact_points_a,
            contact_points_b,
            kMaxApproxCarCarContactPoints,
            contact_overlap_uu,
            contact_points_a_bt,
            contact_points_b_bt,
            cal_ccm,
            contact_game_points_a_bt,
            contact_game_depths_bt,
            &have_contact_game_points,
            &contact_game_normal_bt);
        if (contact_point_count <= 0 && !cal_ccm) {
            contact_point_count = 1;
            ComputeApproxCarCarContactAnchors(
                fb.solver_a,
                cfg_a,
                fb.obb_a,
                fb.solver_b,
                cfg_b,
                fb.obb_b,
                normal,
                contact_feature,
                contact_points_a[0],
                contact_points_b[0]);
            contact_points_a_bt[0] = ToBtLinearVec(contact_points_a[0]);
            contact_points_b_bt[0] = ToBtLinearVec(contact_points_b[0]);
        }
    }
    if constexpr (!kSharedCcm) {
#if defined(__CUDA_ARCH__)
        const CarState& predicted_a = fb.predicted_a;
        const CarState& predicted_b = fb.predicted_b;
#else
        const CarState& predicted_a = predicted_unused ? a : fb.predicted_storage[0];
        const CarState& predicted_b = predicted_unused ? b : fb.predicted_storage[1];
#endif
        if (!use_cached_contact &&
            !force_current_contact_seed &&
            have_current_contact &&
            contact_overlap_uu > 0.0f &&
            current_overlap_uu > 0.0f &&
            contact_point_count > 1) {
            fb.obb_a = fb.current_obb_a;
            fb.obb_b = fb.current_obb_b;
            normal = current_normal;
            contact_overlap_uu = current_overlap_uu;
            contact_feature = current_feature;
            fb.solver_seed_a = a;
            fb.solver_seed_b = b;

            fb.solver_seed_a.vel = predicted_a.vel;
            fb.solver_seed_a.ang_vel = predicted_a.ang_vel;
            fb.solver_seed_b.vel = predicted_b.vel;
            fb.solver_seed_b.ang_vel = predicted_b.ang_vel;
            fb.solver_a = fb.solver_seed_a;
            fb.solver_b = fb.solver_seed_b;
            contact_point_count = ComputeApproxCarCarContactAnchorPairs(
                fb.solver_a,
                cfg_a,
                fb.obb_a,
                fb.solver_b,
                cfg_b,
                fb.obb_b,
                normal,
                contact_feature,
                contact_points_a,
                contact_points_b,
                kMaxApproxCarCarContactPoints,
                contact_overlap_uu,
                contact_points_a_bt,
                contact_points_b_bt);
            if (contact_point_count <= 0) {
                contact_point_count = 1;
                ComputeApproxCarCarContactAnchors(
                    fb.solver_a,
                    cfg_a,
                    fb.obb_a,
                    fb.solver_b,
                    cfg_b,
                    fb.obb_b,
                    normal,
                    contact_feature,
                    contact_points_a[0],
                    contact_points_b[0]);
                contact_points_a_bt[0] = ToBtLinearVec(contact_points_a[0]);
                contact_points_b_bt[0] = ToBtLinearVec(contact_points_b[0]);
            }
        }
    }
    const bool ccm_touch = cal_ccm && !ccm_fresh;
    if (!use_cached_contact && (have_current_contact || ccm_touch) && contact_point_count > 0) {
        bumper_local_x_a = -1.0e30f;
        bumper_local_x_b = -1.0e30f;
        for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
            const Vec3 local_a = InverseRotateCarBasis(a, contact_points_a[contact_idx] - a.pos);
            const Vec3 local_b = InverseRotateCarBasis(b, contact_points_b[contact_idx] - b.pos);
            bumper_local_x_a = Max(bumper_local_x_a, local_a.x);
            bumper_local_x_b = Max(bumper_local_x_b, local_b.x);
        }
        has_bumper_local_x_override = true;
    }

    if (!use_cached_contact && (have_current_contact || ccm_touch) && has_bumper_local_x_override) {
        const bool a_is_lower_id = id_a > id_b;
        CarState* hitter1 = a_is_lower_id ? &a : &b;
        CarState* victim1 = a_is_lower_id ? &b : &a;
        const uint32_t hitter1_id = a_is_lower_id ? id_a : id_b;
        const uint32_t victim1_id = a_is_lower_id ? id_b : id_a;
        const Team hitter1_team = a_is_lower_id ? team_a : team_b;
        const Team victim1_team = a_is_lower_id ? team_b : team_a;
        const CarConfig& hitter1_cfg = a_is_lower_id ? cfg_a : cfg_b;
        const Vec3 normal1 = a_is_lower_id ? normal : -normal;

        (void)bumper_local_x_a;
        (void)bumper_local_x_b;
        {
            bool hitter1_demoed_here = false;
            bool victim1_demoed_here = false;
            for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
                if (a.is_demoed && b.is_demoed) {
                    break;
                }
                const Vec3 point_local_a = InverseRotateCarBasis(a, contact_points_a[contact_idx] - a.pos);
                const Vec3 point_local_b = InverseRotateCarBasis(b, contact_points_b[contact_idx] - b.pos);
                const float hitter1_bumper_x = a_is_lower_id ? point_local_a.x : point_local_b.x;
                const float hitter2_bumper_x = a_is_lower_id ? point_local_b.x : point_local_a.x;
                if (!victim1->is_demoed && (!hitter1->is_demoed || hitter1_demoed_here)) {
                    (void)ResolveCarCarDirectionalContact<kRequireScratch>(
                        *hitter1, hitter1_id, hitter1_team, hitter1_cfg,
                        *victim1, victim1_id, victim1_team,
                        mutator, enable_demos, normal1, has_bumper_local_x_override, hitter1_bumper_x,
                        a_is_lower_id ? start_vel_a : start_vel_b,
                        a_is_lower_id ? &cfg_b : &cfg_a, hitter1->is_demoed != 0,
                        a_is_lower_id ? start_vel_b : start_vel_a, dt, gjk_epa_scratch);
                    victim1_demoed_here = victim1_demoed_here || victim1->is_demoed;
                }
                if (a.is_demoed && b.is_demoed) {
                    break;
                }
                if (!hitter1->is_demoed && (!victim1->is_demoed || victim1_demoed_here)) {
                    (void)ResolveCarCarDirectionalContact<kRequireScratch>(
                        *victim1, victim1_id, victim1_team,
                        a_is_lower_id ? cfg_b : cfg_a,
                        *hitter1, hitter1_id, hitter1_team,
                        mutator, enable_demos, -normal1, has_bumper_local_x_override, hitter2_bumper_x,
                        a_is_lower_id ? start_vel_b : start_vel_a,
                        &hitter1_cfg, victim1->is_demoed != 0,
                        a_is_lower_id ? start_vel_a : start_vel_b, dt, gjk_epa_scratch);
                    hitter1_demoed_here = hitter1_demoed_here || hitter1->is_demoed;
                }
            }
        }
    }
    Vec3 cc_point_normals[kMaxApproxCarCarContactPoints];
    for (int contact_idx = 0; contact_idx < kMaxApproxCarCarContactPoints; ++contact_idx) {
        cc_point_normals[contact_idx] = normal;
    }
    if (cal_ccm) {
        if (ccm_fresh && contact_point_count > 0) {
            for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
                const Vec3 point_on_a =
                    have_contact_game_points ? contact_game_points_a_bt[contact_idx] : contact_points_a_bt[contact_idx];
                const float depth = have_contact_game_points
                                        ? contact_game_depths_bt[contact_idx]
                                        : Dot(contact_points_b_bt[contact_idx] - point_on_a, normal);
                if (depth > ccm_threshold_bt) {
                    continue;
                }
                const Vec3 point_normal = have_contact_game_points ? contact_game_normal_bt : normal;
                CarCarRetainedPoint point;
                point.normal = point_normal;
                point.local_a_bt = CarBodyLocalPointBt(a, point_on_a);
                point.local_b_bt = CarBodyLocalPointBt(b, point_on_a + point_normal * depth);
                point.distance_bt = depth;
                AddCarCarRetainedPoint(*retained_manifold, point, ccm_threshold_bt);
            }
            RefreshCarCarRetainedManifold(*retained_manifold, a, b, ccm_threshold_bt);
        }
        contact_point_count = Min(static_cast<int>(retained_manifold->count), kMaxApproxCarCarContactPoints);
        float max_penetration_uu = 0.0f;
        for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
            const CarCarRetainedPoint& point = retained_manifold->points[contact_idx];
            contact_points_a_bt[contact_idx] = point.pos_a_bt;
            contact_points_b_bt[contact_idx] = point.pos_b_bt;
            contact_points_a[contact_idx] = FromBtLinearVec(point.pos_a_bt);
            contact_points_b[contact_idx] = FromBtLinearVec(point.pos_b_bt);
            cc_point_normals[contact_idx] = point.normal;
            max_penetration_uu = Max(max_penetration_uu, -point.distance_bt * UnitsBTToUU());
        }
        contact_overlap_uu = max_penetration_uu;
        if (contact_point_count == 0) {
            if (pair_cache) {
                ClearCarCarPairCachedContact(*pair_cache);
            }
            return false;
        }
    }
    const float penetration = Max(0.0f, contact_overlap_uu);
    if constexpr (!kSharedCcm) {
        if (skip_velocity_impulse && contact_point_count > 1 && contact_overlap_uu > 1.0f) {
            skip_velocity_impulse = false;
        }
        if (skip_velocity_impulse && contact_overlap_uu > 0.0f) {
            skip_velocity_impulse = false;
        }
        float resolved_normal_impulse[kMaxApproxCarCarContactPoints] = {};
        float resolved_friction_impulse[kMaxApproxCarCarContactPoints] = {};
        Vec3 cc_inv_bt_a0, cc_inv_bt_a1, cc_inv_bt_a2;
        Vec3 cc_inv_bt_b0, cc_inv_bt_b1, cc_inv_bt_b2;
        BuildInvInertiaTensorWorldBt(fb.solver_seed_a, cfg_a, mutator.car_mass, cc_inv_bt_a0, cc_inv_bt_a1, cc_inv_bt_a2);
        BuildInvInertiaTensorWorldBt(fb.solver_seed_b, cfg_b, mutator.car_mass, cc_inv_bt_b0, cc_inv_bt_b1, cc_inv_bt_b2);
        const CarInvInertiaRowsUu cc_inv_uu_a = MakeCarInvInertiaRowsUu(fb.solver_seed_a, cfg_a, mutator.car_mass);
        const CarInvInertiaRowsUu cc_inv_uu_b = MakeCarInvInertiaRowsUu(fb.solver_seed_b, cfg_b, mutator.car_mass);
        const float denom_inv_mass_bt = 1.0f / mutator.car_mass;
        const Vec3 lin_a_bt = base_a.vel_bt;
        const Vec3 lin_b_bt = base_b.vel_bt;
        const Vec3 ext_lin_a_bt = a.ext_force_impulse_bt_valid
            ? a.ext_force_impulse_bt
            : ToBtLinearVec(solver_external_linear_delta_a);
        const Vec3 ext_lin_b_bt = b.ext_force_impulse_bt_valid
            ? b.ext_force_impulse_bt
            : ToBtLinearVec(solver_external_linear_delta_b);
        const Vec3 ext_ang_a_bt = a.ext_force_impulse_bt_valid ? a.ext_torque_impulse : external_angular_delta_a;
        const Vec3 ext_ang_b_bt = b.ext_force_impulse_bt_valid ? b.ext_torque_impulse : external_angular_delta_b;
        const Vec3 base_lin_a_bt = lin_a_bt + ext_lin_a_bt;
        const Vec3 base_lin_b_bt = lin_b_bt + ext_lin_b_bt;
        for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
            const Vec3 normal_pt = cc_point_normals[contact_idx];
            const Vec3 rel_a_bt = contact_points_a_bt[contact_idx] - fb.solver_a.pos_bt;
            const Vec3 rel_b_bt = contact_points_b_bt[contact_idx] - fb.solver_b.pos_bt;
            const Vec3 rel_a = FromBtLinearVec(rel_a_bt);
            const Vec3 rel_b = FromBtLinearVec(rel_b_bt);
            fb.contact_rel_a[contact_idx] = rel_a;
            fb.contact_rel_b[contact_idx] = rel_b;
            fb.contact_rel_a_bt[contact_idx] = rel_a_bt;
            fb.contact_rel_b_bt[contact_idx] = rel_b_bt;
            CarCarPairCachedPoint* cached_point =
                pair_cache && contact_idx < kMaxApproxCarCarContactPoints
                    ? &pair_cache->contacts[contact_idx]
                    : nullptr;
            float accumulated_normal_impulse =
                (cached_point && cached_point->valid != 0) ? cached_point->applied_normal_impulse : 0.0f;
            if (!use_shared_velocity_rows && accumulated_normal_impulse > 0.0f) {
                const Vec3 warmstart_impulse = normal_pt * accumulated_normal_impulse;
                ApplyImpulseAtPoint(fb.solver_a, cfg_a, mutator.car_mass, -warmstart_impulse, rel_a);
                ApplyImpulseAtPoint(fb.solver_b, cfg_b, mutator.car_mass, warmstart_impulse, rel_b);
            }
            resolved_normal_impulse[contact_idx] = accumulated_normal_impulse;
            resolved_friction_impulse[contact_idx] =
                (cached_point && cached_point->valid != 0) ? cached_point->applied_friction_impulse : 0.0f;

            float denom_a, denom_b;
            {
                const Vec3 denom_normal_axis = -normal_pt;
                const Vec3 denom_torque_a = Cross(rel_a_bt, denom_normal_axis);
                const Vec3 denom_ang_a = Mul3x3Vec(cc_inv_bt_a0, cc_inv_bt_a1, cc_inv_bt_a2, denom_torque_a);
                const Vec3 denom_vec_a = Cross(denom_ang_a, rel_a_bt);
                denom_a = denom_inv_mass_bt + Dot(denom_normal_axis, denom_vec_a);

                const Vec3 denom_torque_b = Cross(rel_b_bt, -denom_normal_axis);
                const Vec3 denom_ang_b = Mul3x3Vec(cc_inv_bt_b0, cc_inv_bt_b1, cc_inv_bt_b2, denom_torque_b);
                const Vec3 denom_vec_b = Cross(-denom_ang_b, rel_b_bt);
                denom_b = denom_inv_mass_bt + Dot(denom_normal_axis, denom_vec_b);
            }
            const float denom = denom_a + denom_b;
            if (denom <= 1.0e-6f) {
                continue;
            }

            const Vec3 normal_a_bt = -normal_pt;
            const Vec3 normal_b_bt = normal_pt;
            const Vec3 rel_a_cross_normal_bt = Cross(rel_a_bt, normal_a_bt);
            const Vec3 rel_b_cross_normal_bt = Cross(rel_b_bt, normal_b_bt);
            const float vel1_dotn_bt =
                Dot(normal_a_bt, base_lin_a_bt) +
                Dot(rel_a_cross_normal_bt, base_a.ang_vel + ext_ang_a_bt);
            const float vel2_dotn_bt =
                Dot(normal_b_bt, base_lin_b_bt) +
                Dot(rel_b_cross_normal_bt, base_b.ang_vel + ext_ang_b_bt);
            const float base_vn_bt = vel1_dotn_bt + vel2_dotn_bt;
            const float jac_diag_ab_inv = 1.0f / denom;

            const Vec3 vel_a_no_ext_bt = lin_a_bt + Cross(base_a.ang_vel, rel_a_bt);
            const Vec3 vel_b_no_ext_bt = lin_b_bt + Cross(base_b.ang_vel, rel_b_bt);
            const float base_vn_no_ext_bt = Dot(normal_b_bt, vel_b_no_ext_bt - vel_a_no_ext_bt);
            const float restitution_velocity_bt = Max(CarCarCollisionRestitution() * -base_vn_no_ext_bt, 0.0f);
            const float rhs_bt = (restitution_velocity_bt - base_vn_bt) * jac_diag_ab_inv;

            fb.normal_row_jac_diag_ab_inv[contact_idx] = jac_diag_ab_inv;
            fb.normal_row_rhs[contact_idx] = rhs_bt * UnitsBTToUU();
            fb.normal_row_rhs_bt[contact_idx] = rhs_bt;
            fb.normal_row_valid[contact_idx] = 1;
        }

        if (use_shared_velocity_rows) {
            for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
                const Vec3 normal_pt = cc_point_normals[contact_idx];
                if (skip_velocity_impulse || fb.normal_row_valid[contact_idx] == 0) {
                    continue;
                }
                if (*shared_velocity_row_count >= kMaxCarPairSharedVelocityContactRows) {
                    break;
                }

                const Vec3 rel_a = fb.contact_rel_a[contact_idx];
                const Vec3 rel_b = fb.contact_rel_b[contact_idx];
                const Vec3 shared_normal_a = -normal_pt;
                const Vec3 shared_normal_b = normal_pt;
                const Vec3 shared_rel_a_cross_normal = Cross(rel_a, shared_normal_a);
                const Vec3 shared_rel_b_cross_normal = Cross(rel_b, shared_normal_b);
                const Vec3 shared_angular_component_a = ApplyInverseBodyInertiaWorld(cc_inv_uu_a, shared_rel_a_cross_normal);
                const Vec3 shared_angular_component_b = ApplyInverseBodyInertiaWorld(cc_inv_uu_b, shared_rel_b_cross_normal);
                CarSharedVelocityContactRow& shared_row = shared_velocity_rows[*shared_velocity_row_count];
                shared_row = CarSharedVelocityContactRow{};
                shared_row.valid = 1;
                shared_row.source_tag = 2000 + contact_idx;
                shared_row.body_a_local_idx = respawn_slot_a;
                shared_row.body_b_local_idx = respawn_slot_b;
                shared_row.pair_cache_idx = pair_idx;
                shared_row.pair_contact_idx = contact_idx;
                shared_row.normal_a = shared_normal_a;
                shared_row.rel_contact_a = rel_a;
                shared_row.rel_contact_b = rel_b;
                shared_row.rel_a_cross_normal = shared_rel_a_cross_normal;
                shared_row.rel_b_cross_normal = shared_rel_b_cross_normal;
                shared_row.angular_component_a = shared_angular_component_a;
                shared_row.angular_component_b = shared_angular_component_b;
                shared_row.inv_mass_a = 1.0f / mutator.car_mass;
                shared_row.inv_mass_b = 1.0f / mutator.car_mass;
                shared_row.jac_diag_ab_inv = fb.normal_row_jac_diag_ab_inv[contact_idx];
                shared_row.rhs = fb.normal_row_rhs[contact_idx];
                shared_row.rhs_bt = fb.normal_row_rhs_bt[contact_idx];
                shared_row.has_rhs_bt = 1;
                {
                    const Vec3 rel_a_cross_normal_bt_row = Cross(fb.contact_rel_a_bt[contact_idx], shared_normal_a);
                    const Vec3 rel_b_cross_normal_bt_row = Cross(fb.contact_rel_b_bt[contact_idx], shared_normal_b);
                    shared_row.rel_a_cross_normal_bt = rel_a_cross_normal_bt_row;
                    shared_row.rel_b_cross_normal_bt = rel_b_cross_normal_bt_row;
                    shared_row.angular_component_a_bt =
                        Mul3x3Vec(cc_inv_bt_a0, cc_inv_bt_a1, cc_inv_bt_a2, rel_a_cross_normal_bt_row);
                    shared_row.angular_component_b_bt =
                        Mul3x3Vec(cc_inv_bt_b0, cc_inv_bt_b1, cc_inv_bt_b2, rel_b_cross_normal_bt_row);
                    shared_row.has_bt = 1;
                }
                shared_row.applied_impulse = resolved_normal_impulse[contact_idx];
                const int shared_normal_row_idx = *shared_velocity_row_count;
                *shared_velocity_row_count += 1;

                if (*shared_friction_row_count >= kMaxCarPairSharedFrictionContactRows) {
                    continue;
                }

                const Vec3 rel_a_bt = fb.contact_rel_a_bt[contact_idx];
                const Vec3 rel_b_bt = fb.contact_rel_b_bt[contact_idx];
                const Vec3 base_vel_a_bt =
                    base_lin_a_bt +
                    Cross(base_a.ang_vel + ext_ang_a_bt, rel_a_bt);
                const Vec3 base_vel_b_bt =
                    base_lin_b_bt +
                    Cross(base_b.ang_vel + ext_ang_b_bt, rel_b_bt);
                const Vec3 base_rel_vel_bt = base_vel_b_bt - base_vel_a_bt;
                Vec3 tangent = base_rel_vel_bt - normal_pt * Dot(base_rel_vel_bt, normal_pt);
                const float lat_rel_vel_bt = LengthSq(tangent);
                const float tangent_len = SqrtScalar(lat_rel_vel_bt);
                tangent = lat_rel_vel_bt > 1.1920928955078125e-07f
                    ? tangent * (1.0f / tangent_len)
                    : BulletPlaneSpace1First(normal_pt);

                const Vec3 tangent_a = tangent;
                const Vec3 tangent_b = -tangent;
                const Vec3 tangent_angular_component_a = ApplyInverseBodyInertiaWorld(cc_inv_uu_a, Cross(rel_a, tangent_a));
                const Vec3 tangent_angular_component_b = ApplyInverseBodyInertiaWorld(cc_inv_uu_b, Cross(rel_b, tangent_b));
                const Vec3 rel_a_cross_tangent_bt = Cross(rel_a_bt, tangent_a);
                const Vec3 rel_b_cross_tangent_bt = Cross(rel_b_bt, tangent_b);
                const Vec3 denom_angular_component_a_bt = Mul3x3Vec(cc_inv_bt_a0, cc_inv_bt_a1, cc_inv_bt_a2, rel_a_cross_tangent_bt);
                const Vec3 denom_angular_component_b_bt = Mul3x3Vec(cc_inv_bt_b0, cc_inv_bt_b1, cc_inv_bt_b2, rel_b_cross_tangent_bt);
                const Vec3 denom_vec_a_bt = Cross(denom_angular_component_a_bt, rel_a_bt);
                const Vec3 denom_vec_b_bt = Cross(-denom_angular_component_b_bt, rel_b_bt);
                const float denom_dot_a_bt = Dot(tangent_a, denom_vec_a_bt);
                const float denom_dot_b_bt = Dot(tangent_a, denom_vec_b_bt);
                const float denom_t_a = (1.0f / mutator.car_mass) + denom_dot_a_bt;
                const float denom_t_b = (1.0f / mutator.car_mass) + denom_dot_b_bt;
                const float denom_t = denom_t_a + denom_t_b;
                if (denom_t <= 1.0e-6f) {
                    continue;
                }

                CarSharedFrictionContactRow& friction_row = shared_friction_rows[*shared_friction_row_count];
                friction_row = CarSharedFrictionContactRow{};
                friction_row.valid = 1;
                friction_row.normal_row_idx = shared_normal_row_idx;
                friction_row.source_tag = 3000 + contact_idx;
                friction_row.body_a_local_idx = respawn_slot_a;
                friction_row.body_b_local_idx = respawn_slot_b;
                friction_row.pair_cache_idx = pair_idx;
                friction_row.pair_contact_idx = contact_idx;
                friction_row.tangent_a = tangent_a;
                friction_row.rel_contact_a = rel_a;
                friction_row.rel_contact_b = rel_b;
                friction_row.rel_a_cross_tangent = Cross(rel_a, tangent_a);
                friction_row.rel_b_cross_tangent = Cross(rel_b, tangent_b);
                friction_row.angular_component_a = tangent_angular_component_a;
                friction_row.angular_component_b = tangent_angular_component_b;
                friction_row.inv_mass_a = 1.0f / mutator.car_mass;
                friction_row.inv_mass_b = 1.0f / mutator.car_mass;
                friction_row.jac_diag_ab_inv = 1.0f / denom_t;
                const float rhs_lin1_dott_bt = Dot(tangent_a, base_lin_a_bt);
                const float rhs_ang1_dott_bt = Dot(rel_a_cross_tangent_bt, base_a.ang_vel);
                const float rhs_lin2_dott_bt = Dot(tangent_b, base_lin_b_bt);
                const float rhs_ang2_dott_bt = Dot(rel_b_cross_tangent_bt, base_b.ang_vel);
                const float rhs_vel1_dott_bt = rhs_lin1_dott_bt + rhs_ang1_dott_bt;
                const float rhs_vel2_dott_bt = rhs_lin2_dott_bt + rhs_ang2_dott_bt;

                const float rhs_rel_t_bt = -(rhs_vel1_dott_bt + rhs_vel2_dott_bt);
                const float friction_rhs_bt = rhs_rel_t_bt * friction_row.jac_diag_ab_inv;
                friction_row.rhs_bt = friction_rhs_bt;
                friction_row.has_rhs_bt = 1;

                friction_row.rel_a_cross_tangent_bt = rel_a_cross_tangent_bt;
                friction_row.rel_b_cross_tangent_bt = rel_b_cross_tangent_bt;
                friction_row.angular_component_a_bt = denom_angular_component_a_bt;
                friction_row.angular_component_b_bt = denom_angular_component_b_bt;
                friction_row.has_bt = 1;
                friction_row.rhs = friction_rhs_bt * UnitsBTToUU();
                friction_row.friction = CarCarCollisionFriction();
                friction_row.applied_impulse = resolved_friction_impulse[contact_idx];
                *shared_friction_row_count += 1;
            }
        } else for (int iter = 0; iter < CarCarVelocitySolverIterations(); ++iter) {
            for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
                const Vec3 normal_pt = cc_point_normals[contact_idx];
                const Vec3 rel_a = fb.contact_rel_a[contact_idx];
                const Vec3 rel_b = fb.contact_rel_b[contact_idx];
                float accumulated_normal_impulse = resolved_normal_impulse[contact_idx];
                const Vec3 rel_vel = VelocityAtPoint(fb.solver_b, rel_b) - VelocityAtPoint(fb.solver_a, rel_a);
                const float vn = Dot(rel_vel, normal_pt);
                const bool solve_normal_row =
                    !skip_velocity_impulse &&
                    (vn < 0.0f || penetration > 0.0f || accumulated_normal_impulse > 0.0f);
                if (!solve_normal_row) {
                    continue;
                }
                if (fb.normal_row_valid[contact_idx] == 0) {
                    continue;
                }

                const Vec3 base_vel_a =
                    (base_a.vel + external_linear_delta_a) +
                    Cross(base_a.ang_vel + external_angular_delta_a, rel_a);
                const Vec3 base_vel_b =
                    (base_b.vel + external_linear_delta_b) +
                    Cross(base_b.ang_vel + external_angular_delta_b, rel_b);
                const Vec3 cur_vel_a = VelocityAtPoint(fb.solver_a, rel_a);
                const Vec3 cur_vel_b = VelocityAtPoint(fb.solver_b, rel_b);
                const float delta_vel_a = Dot(base_vel_a - cur_vel_a, normal_pt);
                const float delta_vel_b = Dot(cur_vel_b - base_vel_b, normal_pt);
                const float normal_impulse_delta =
                    fb.normal_row_rhs[contact_idx] -
                    (delta_vel_a + delta_vel_b) * fb.normal_row_jac_diag_ab_inv[contact_idx];
                const float next_accumulated = Max(0.0f, accumulated_normal_impulse + normal_impulse_delta);
                const float impulse_mag = next_accumulated - accumulated_normal_impulse;
                accumulated_normal_impulse = next_accumulated;

                const Vec3 impulse = normal_pt * impulse_mag;
                ApplyImpulseAtPoint(fb.solver_a, cfg_a, mutator.car_mass, -impulse, rel_a);
                ApplyImpulseAtPoint(fb.solver_b, cfg_b, mutator.car_mass, impulse, rel_b);
                resolved_normal_impulse[contact_idx] = accumulated_normal_impulse;
            }

            for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
                const Vec3 normal_pt = cc_point_normals[contact_idx];
                const Vec3 rel_a = fb.contact_rel_a[contact_idx];
                const Vec3 rel_b = fb.contact_rel_b[contact_idx];
                const float friction_normal_limit_impulse = resolved_normal_impulse[contact_idx];
                if (friction_normal_limit_impulse <= 0.0f) {
                    continue;
                }

                float accumulated_friction_impulse = resolved_friction_impulse[contact_idx];
                const Vec3 rel_vel_after_normal = VelocityAtPoint(fb.solver_b, rel_b) - VelocityAtPoint(fb.solver_a, rel_a);
                const float vn_after_normal = Dot(rel_vel_after_normal, normal_pt);
                const Vec3 tangent_vel = rel_vel_after_normal - normal_pt * vn_after_normal;
                const float tangent_len = Length(tangent_vel);
                if (tangent_len <= 1e-5f) {
                    continue;
                }

                Vec3 tangent = tangent_vel / tangent_len;

                const float denom_t_a =
                    CarCarBodyImpulseDenominatorFromRel(fb.solver_a, cfg_a, mutator.car_mass, rel_a, tangent);
                const float denom_t_b =
                    CarCarBodyImpulseDenominatorFromRel(fb.solver_b, cfg_b, mutator.car_mass, rel_b, tangent);
                const float inv_mass_sum_t = denom_t_a + denom_t_b;
                if (inv_mass_sum_t <= 1.0e-6f) {
                    continue;
                }

                const float max_friction_impulse = CarCarCollisionFriction() * friction_normal_limit_impulse;
                const float jt_ideal = tangent_len / inv_mass_sum_t;
                const float next_friction_accum =
                    Clamp(accumulated_friction_impulse + jt_ideal, -max_friction_impulse, max_friction_impulse);
                const float jt = next_friction_accum - accumulated_friction_impulse;
                accumulated_friction_impulse = next_friction_accum;
                const Vec3 friction_impulse = tangent * jt;
                ApplyImpulseAtPoint(fb.solver_a, cfg_a, mutator.car_mass, friction_impulse, rel_a);
                ApplyImpulseAtPoint(fb.solver_b, cfg_b, mutator.car_mass, -friction_impulse, rel_b);
                resolved_friction_impulse[contact_idx] = accumulated_friction_impulse;
                fb.resolved_tangent[contact_idx] = tangent;
                fb.resolved_tangent_valid[contact_idx] = 1;
            }
        }

        if (penetration > 0.0f && dt > 0.0f) {
            if (shared_split_rows && shared_split_row_count) {
                const float inv_mass_a = 1.0f / mutator.car_mass;
                const float inv_mass_b = 1.0f / mutator.car_mass;
                for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
                    const Vec3 normal_pt = cc_point_normals[contact_idx];
                    const Vec3 normal_a = -normal_pt;
                    const Vec3 normal_b = normal_pt;
                    const Vec3 contact_point_a = contact_points_a[contact_idx];
                    const Vec3 contact_point_b = contact_points_b[contact_idx];
                    const Vec3 rel_a = CarCarBulletRelativePointUU(contact_point_a, fb.solver_seed_a.pos_bt);
                    const Vec3 rel_b = CarCarBulletRelativePointUU(contact_point_b, fb.solver_seed_b.pos_bt);
                    const Vec3 point_delta = contact_point_b - contact_point_a;
                    const float point_penetration = Max(0.0f, -Dot(point_delta, normal_pt));
                    if (point_penetration <= 0.0f) {
                        continue;
                    }

                    const Vec3 rel_a_cross_normal = Cross(rel_a, normal_a);
                    const Vec3 rel_b_cross_normal = Cross(rel_b, normal_b);
                    const Vec3 angular_component_a =
                        ApplyInverseBodyInertiaWorld(cc_inv_uu_a, rel_a_cross_normal);
                    const Vec3 angular_component_b =
                        ApplyInverseBodyInertiaWorld(cc_inv_uu_b, rel_b_cross_normal);

                    float denom = 0.0f;
                    denom += inv_mass_a + Dot(normal_a, Cross(angular_component_a, rel_a));
                    denom += inv_mass_b + Dot(normal_b, Cross(angular_component_b, rel_b));
                    if (denom <= 1.0e-8f) {
                        continue;
                    }

                    const float jac_diag_ab_inv = 1.0f / denom;
                    const float rhs_penetration =
                        ((point_penetration * SplitImpulsePenetrationErp()) / dt) * jac_diag_ab_inv;

                    const Vec3 contact_point_a_bt = contact_points_a_bt[contact_idx];
                    const Vec3 contact_point_b_bt = contact_points_b_bt[contact_idx];
                    const Vec3 rel_a_bt = contact_point_a_bt - fb.solver_seed_a.pos_bt;
                    const Vec3 rel_b_bt = contact_point_b_bt - fb.solver_seed_b.pos_bt;
                    const float point_penetration_bt =
                        Max(0.0f, -Dot(contact_point_b_bt - contact_point_a_bt, normal_pt));
                    const Vec3 rel_a_cross_normal_bt = Cross(rel_a_bt, normal_a);
                    const Vec3 rel_b_cross_normal_bt = Cross(rel_b_bt, normal_b);
                    const Vec3 angular_component_a_bt =
                        Mul3x3Vec(cc_inv_bt_a0, cc_inv_bt_a1, cc_inv_bt_a2, rel_a_cross_normal_bt);
                    const Vec3 angular_component_b_bt =
                        Mul3x3Vec(cc_inv_bt_b0, cc_inv_bt_b1, cc_inv_bt_b2, rel_b_cross_normal_bt);
                    float denom_bt = 0.0f;
                    denom_bt += inv_mass_a + Dot(normal_a, Cross(angular_component_a_bt, rel_a_bt));
                    denom_bt += inv_mass_b + Dot(normal_b, Cross(angular_component_b_bt, rel_b_bt));
                    const float jac_diag_ab_inv_bt = denom_bt > 1.0e-8f ? 1.0f / denom_bt : 0.0f;
                    const float rhs_penetration_bt =
                        ((point_penetration_bt * SplitImpulsePenetrationErp()) / dt) * jac_diag_ab_inv_bt;
                    const int row_before = *shared_split_row_count;
                    AppendCarPairSharedSplitRow(
                        shared_split_rows,
                        *shared_split_row_count,
                        respawn_slot_a,
                        respawn_slot_b,
                        1000 + contact_idx,
                        normal_a,
                        normal_b,
                        rel_a,
                        rel_b,
                        angular_component_a,
                        angular_component_b,
                        inv_mass_a,
                        inv_mass_b,
                        jac_diag_ab_inv,
                        rhs_penetration);
                    if (*shared_split_row_count == row_before + 1 && jac_diag_ab_inv_bt > 0.0f &&
                        point_penetration_bt > 0.0f) {
                        CarSharedSplitContactRow& row_bt = shared_split_rows[row_before];
                        row_bt.rel_a_cross_normal_bt = rel_a_cross_normal_bt;
                        row_bt.angular_component_a_bt = angular_component_a_bt;
                        row_bt.rel_b_cross_normal_bt = rel_b_cross_normal_bt;
                        row_bt.angular_component_b_bt = angular_component_b_bt;
                        row_bt.jac_diag_ab_inv_bt = jac_diag_ab_inv_bt;
                        row_bt.rhs_penetration_bt = rhs_penetration_bt;
                        row_bt.applied_push_impulse_bt = 0.0f;
                        row_bt.has_bt = 1;
                    }
                }
            } else {
                SplitImpulseAccumulator split_a;
                SplitImpulseAccumulator split_b;
                float applied_push_impulse[kMaxApproxCarCarContactPoints] = {};
                const float inv_mass_a = 1.0f / mutator.car_mass;
                const float inv_mass_b = 1.0f / mutator.car_mass;

                for (int iter = 0; iter < BallSplitImpulseIterations(); ++iter) {
                    for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
                        const Vec3 normal_pt = cc_point_normals[contact_idx];
                        const Vec3 normal_a = -normal_pt;
                        const Vec3 normal_b = normal_pt;

                        const Vec3 contact_point_a = contact_points_a_bt[contact_idx];
                        const Vec3 contact_point_b = contact_points_b_bt[contact_idx];
                        const Vec3 rel_a = contact_point_a - fb.solver_seed_a.pos_bt;
                        const Vec3 rel_b = contact_point_b - fb.solver_seed_b.pos_bt;
                        const Vec3 point_delta = contact_point_b - contact_point_a;
                        const float point_penetration = Max(0.0f, -Dot(point_delta, normal_pt));
                        if (point_penetration <= 0.0f) {
                            continue;
                        }

                        const Vec3 rel_a_cross_normal = Cross(rel_a, normal_a);
                        const Vec3 rel_b_cross_normal = Cross(rel_b, normal_b);
                        const Vec3 angular_component_a =
                            Mul3x3Vec(cc_inv_bt_a0, cc_inv_bt_a1, cc_inv_bt_a2, rel_a_cross_normal);
                        const Vec3 angular_component_b =
                            Mul3x3Vec(cc_inv_bt_b0, cc_inv_bt_b1, cc_inv_bt_b2, rel_b_cross_normal);

                        float denom = 0.0f;
                        denom += inv_mass_a + Dot(normal_a, Cross(angular_component_a, rel_a));
                        denom += inv_mass_b + Dot(normal_b, Cross(angular_component_b, rel_b));
                        if (denom <= 1.0e-8f) {
                            continue;
                        }

                        const float jac_diag_ab_inv = 1.0f / denom;
                        const float rhs_penetration =
                            ((point_penetration * SplitImpulsePenetrationErp()) / dt) * jac_diag_ab_inv;

                        float delta_impulse = rhs_penetration;
                        delta_impulse -= (
                            BtSimdDot3LikeBullet(normal_b, split_b.push_velocity) +
                            BtSimdDot3LikeBullet(rel_b_cross_normal, split_b.turn_velocity)) * jac_diag_ab_inv;
                        delta_impulse -= (
                            BtSimdDot3LikeBullet(normal_a, split_a.push_velocity) +
                            BtSimdDot3LikeBullet(rel_a_cross_normal, split_a.turn_velocity)) * jac_diag_ab_inv;

                        const float sum = applied_push_impulse[contact_idx] + delta_impulse;
                        if (sum < 0.0f) {
                            delta_impulse = -applied_push_impulse[contact_idx];
                            applied_push_impulse[contact_idx] = 0.0f;
                        } else {
                            applied_push_impulse[contact_idx] = sum;
                        }

                        if (delta_impulse == 0.0f) {
                            continue;
                        }

                        split_a.push_velocity += (normal_a * inv_mass_a) * delta_impulse;
                        split_a.turn_velocity += angular_component_a * delta_impulse;
                        split_b.push_velocity += (normal_b * inv_mass_b) * delta_impulse;
                        split_b.turn_velocity += angular_component_b * delta_impulse;
                    }
                }

                IntegrateSplitImpulseTransformBt(a, split_a, dt, NativeSseMath(mutator));
                IntegrateSplitImpulseTransformBt(b, split_b, dt, NativeSseMath(mutator));
            }
        }

        {
            const Vec3 rocketgpu_dva = fb.solver_a.vel - fb.solver_seed_a.vel;
            if (rocketgpu_dva.x != 0.0f || rocketgpu_dva.y != 0.0f || rocketgpu_dva.z != 0.0f) {
                SetCarLinearVelocityUu(a, a.vel + rocketgpu_dva);
            }
        }
        a.ang_vel += fb.solver_a.ang_vel - fb.solver_seed_a.ang_vel;
        SetCarOriginBt(a, a.pos_bt + (fb.solver_a.pos_bt - fb.solver_seed_a.pos_bt));
        {
            const Vec3 rocketgpu_dvb = fb.solver_b.vel - fb.solver_seed_b.vel;
            if (rocketgpu_dvb.x != 0.0f || rocketgpu_dvb.y != 0.0f || rocketgpu_dvb.z != 0.0f) {
                SetCarLinearVelocityUu(b, b.vel + rocketgpu_dvb);
            }
        }
        b.ang_vel += fb.solver_b.ang_vel - fb.solver_seed_b.ang_vel;
        SetCarOriginBt(b, b.pos_bt + (fb.solver_b.pos_bt - fb.solver_seed_b.pos_bt));

        if (pair_cache) {
            pair_cache->valid = 1;
            pair_cache->normal = normal;
            pair_cache->contact_count = static_cast<uint8_t>(contact_point_count);
            for (int contact_idx = 0; contact_idx < kMaxApproxCarCarContactPoints; ++contact_idx) {
                CarCarPairCachedPoint& cached_point = pair_cache->contacts[contact_idx];
                if (contact_idx < contact_point_count) {
                    cached_point.valid = 1;
                    cached_point.point_a = contact_points_a[contact_idx];
                    cached_point.point_b = contact_points_b[contact_idx];
                    cached_point.point_a_bt = contact_points_a_bt[contact_idx];
                    cached_point.point_b_bt = contact_points_b_bt[contact_idx];
                    cached_point.has_bt_points = 1;
                    cached_point.penetration_uu = penetration;
                    cached_point.applied_normal_impulse = resolved_normal_impulse[contact_idx];
                    cached_point.applied_friction_impulse = resolved_friction_impulse[contact_idx];
                    if (fb.resolved_tangent_valid[contact_idx] != 0) {
                        cached_point.tangent = fb.resolved_tangent[contact_idx];
                        cached_point.tangent_valid = 1;
                    }
                } else {
                    cached_point = CarCarPairCachedPoint{};
                }
            }
            return true;
        }
        return false;
    }
}

ROCKETGPU_HD bool ResolveCarCarCollision(
    CarState& a,
    uint32_t id_a,
    Team team_a,
    CarState& b,
    uint32_t id_b,
    Team team_b,
    const CarConfig& cfg_a,
    const CarConfig& cfg_b,
    const MutatorConfig& mutator,
    const PreContactCar& base_a,
    const PreContactCar& base_b,
    const Vec3& external_linear_delta_a,
    const Vec3& external_linear_delta_b,
    const Vec3& external_angular_delta_a,
    const Vec3& external_angular_delta_b,
    float dt,
    uint8_t enable_demos,
    int respawn_slot_a,
    int respawn_slot_b,
    CarCarPairCacheView pair_contact_cache = CarCarPairCacheView{},
    EnvStridedArray<CarSharedSplitContactRow> shared_split_rows = nullptr,
    int* shared_split_row_count = nullptr,
    EnvStridedArray<CarSharedVelocityContactRow> shared_velocity_rows = nullptr,
    int* shared_velocity_row_count = nullptr,
    EnvStridedArray<CarSharedFrictionContactRow> shared_friction_rows = nullptr,
    int* shared_friction_row_count = nullptr,
    const Vec3* start_vel_a = nullptr,
    const Vec3* start_vel_b = nullptr,
    CarCarRetainedManifold* retained_manifold = nullptr,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch = nullptr) {
    const bool use_shared_velocity_rows =
        shared_velocity_rows && shared_velocity_row_count &&
        shared_friction_rows && shared_friction_row_count;
    if (retained_manifold && use_shared_velocity_rows && shared_split_rows && shared_split_row_count) {
        return ResolveCarCarCollisionImpl<true, false>(
            a, id_a, team_a, b, id_b, team_b, cfg_a, cfg_b, mutator, base_a, base_b,
            external_linear_delta_a, external_linear_delta_b, external_angular_delta_a, external_angular_delta_b,
            dt, enable_demos, respawn_slot_a, respawn_slot_b, pair_contact_cache,
            shared_split_rows, shared_split_row_count, shared_velocity_rows, shared_velocity_row_count,
            shared_friction_rows, shared_friction_row_count, start_vel_a, start_vel_b,
            retained_manifold, gjk_epa_scratch);
    }
    return ResolveCarCarCollisionImpl<false, false>(
        a, id_a, team_a, b, id_b, team_b, cfg_a, cfg_b, mutator, base_a, base_b,
        external_linear_delta_a, external_linear_delta_b, external_angular_delta_a, external_angular_delta_b,
        dt, enable_demos, respawn_slot_a, respawn_slot_b, pair_contact_cache,
        shared_split_rows, shared_split_row_count, shared_velocity_rows, shared_velocity_row_count,
        shared_friction_rows, shared_friction_row_count, start_vel_a, start_vel_b,
        retained_manifold, gjk_epa_scratch);
}

ROCKETGPU_HD void UpdateCarCarPairCacheFromSharedVelocityRows(
    const CarCarPairCacheView& pair_contact_cache,
    EnvStridedArray<const CarSharedVelocityContactRow> shared_velocity_rows,
    int shared_velocity_row_count,
    EnvStridedArray<const CarSharedFrictionContactRow> shared_friction_rows,
    int shared_friction_row_count) {
    if (!pair_contact_cache) {
        return;
    }

    for (int row_idx = 0; row_idx < shared_velocity_row_count; ++row_idx) {
        const CarSharedVelocityContactRow& row = shared_velocity_rows[row_idx];
        if (!row.valid ||
            row.pair_cache_idx < 0 || row.pair_cache_idx >= pair_contact_cache.size() ||
            row.pair_contact_idx < 0 || row.pair_contact_idx >= kMaxApproxCarCarContactPoints) {
            continue;
        }
        CarCarPairCachedContact& cache = pair_contact_cache.slots[row.pair_cache_idx];
        if (!cache.valid) {
            continue;
        }
        cache.contacts[row.pair_contact_idx].applied_normal_impulse = row.applied_impulse;
    }

    for (int row_idx = 0; row_idx < shared_friction_row_count; ++row_idx) {
        const CarSharedFrictionContactRow& row = shared_friction_rows[row_idx];
        if (!row.valid ||
            row.pair_cache_idx < 0 || row.pair_cache_idx >= pair_contact_cache.size() ||
            row.pair_contact_idx < 0 || row.pair_contact_idx >= kMaxApproxCarCarContactPoints) {
            continue;
        }
        CarCarPairCachedContact& cache = pair_contact_cache.slots[row.pair_cache_idx];
        if (!cache.valid) {
            continue;
        }
        cache.contacts[row.pair_contact_idx].applied_friction_impulse = row.applied_impulse;
        cache.contacts[row.pair_contact_idx].tangent = row.tangent_a;
        cache.contacts[row.pair_contact_idx].tangent_valid = 1;
    }
}

ROCKETGPU_HD bool BoostPadSweepEnterBt(
    const Vec3& pad_pos_uu, const BoostPadState& pad, const Vec3& from_bt, const Vec3& to_bt, float& enter) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    const float rad_bt = (pad.is_big ? 160.0f + 48.0f : 128.0f + 48.0f) * kUUToBT;
    const float half_z_bt = (64.0f + 32.0f) * kUUToBT;
    const float pad_z_bt = (pad_pos_uu.z - (pad.is_big ? 1.0f : 5.92f)) * kUUToBT;
    float s0 = 0.0f;
    float s1 = 1.0f;
    const float z0 = from_bt.z - pad_z_bt;
    const float dz = to_bt.z - from_bt.z;
    if (Abs(dz) < 1e-7f) {
        if (Abs(z0) >= half_z_bt) {
            return false;
        }
    } else {
        const float ta = (-half_z_bt - z0) / dz;
        const float tb = (half_z_bt - z0) / dz;
        s0 = Max(s0, Min(ta, tb));
        s1 = Min(s1, Max(ta, tb));
    }
    const float ax = from_bt.x - pad_pos_uu.x * kUUToBT;
    const float ay = from_bt.y - pad_pos_uu.y * kUUToBT;
    const float dx = to_bt.x - from_bt.x;
    const float dy = to_bt.y - from_bt.y;
    const float a = dx * dx + dy * dy;
    const float b = ax * dx + ay * dy;
    const float c = ax * ax + ay * ay - rad_bt * rad_bt;
    if (a < 1e-12f) {
        enter = s0;
        return c < 0.0f && s0 < s1;
    }
    const float disc = b * b - a * c;
    if (disc <= 0.0f) {
        return false;
    }
    const float root = sqrtf(disc);
    enter = Max(s0, (-b - root) / a);
    return enter < Min(s1, (-b + root) / a);
}

ROCKETGPU_HD bool BoostPadSweepCollideBt(
    const Vec3& pad_pos_uu, const BoostPadState& pad, const Vec3& from_bt, const Vec3& to_bt) {
    float enter = 0.0f;
    return BoostPadSweepEnterBt(pad_pos_uu, pad, from_bt, to_bt, enter);
}

ROCKETGPU_HD float FrameFrac(const SimConfig& cfg, int env_idx) {
    return cfg.env_frame_frac != nullptr ? cfg.env_frame_frac[env_idx] : cfg.mutator.boost_pad_frame_frac;
}

ROCKETGPU_HD float FrameStartDt(const SimConfig& cfg, int env_idx) {
    return cfg.env_frame_start_dt != nullptr ? cfg.env_frame_start_dt[env_idx] : cfg.mutator.frame_start_dt;
}

ROCKETGPU_HD uint16_t BoostPadFracQ(float frame_frac) {
    return static_cast<uint16_t>(Min(Max(frame_frac, 0.0f), 1.0f) * 65535.0f + 0.5f);
}

ROCKETGPU_HD void PreTickBoostPad(BoostPadState& pad, float dt) {
    if (!pad.is_active || pad.cooldown > 0.0f) {
        pad.cooldown -= dt;
        pad.is_active = 0;
    }
    if (pad.cur_locked_car_id != 0) {
        pad.cur_locked_car_id = 0;
    }
}

ROCKETGPU_HD bool BoostPadRespawnDue(const BoostPadState& pad, float dt, float frame_frac) {
    const float left = floorf(pad.cooldown / dt + 0.5f) + 1.0f;
    return left < 0.0f || (left == 0.0f && pad.pickup_frac_q < BoostPadFracQ(frame_frac));
}

ROCKETGPU_HD void PreTickBoostPadsForEnv(int env_idx, const SimConfig& cfg, BoostPadState* pads, float dt) {
    if (!cfg.enable_boost_pads) {
        return;
    }

    const int pad_base = env_idx * BoostPadCount(cfg);
    for (int p = 0; p < BoostPadCount(cfg); ++p) {
        PreTickBoostPad(pads[pad_base + p], dt);
    }
}

ROCKETGPU_HD constexpr float BoostPadGridExtentZUU() {
    return BoostPadCylinderHeight() + 250.0f;
}

ROCKETGPU_HD constexpr int BoostPadGridCellsX() { return 8; }
ROCKETGPU_HD constexpr int BoostPadGridCellsY() { return 10; }
ROCKETGPU_HD constexpr int BoostPadGridCellSizeX() { return 1024; }
ROCKETGPU_HD constexpr int BoostPadGridCellSizeY() { return 1024; }

ROCKETGPU_HD constexpr int BoostPadGridIndexX(float x) {
    return static_cast<int>(x / static_cast<float>(BoostPadGridCellSizeX())) +
           (BoostPadGridCellsX() / 2);
}

ROCKETGPU_HD constexpr int BoostPadGridIndexY(float y) {
    return static_cast<int>(y / static_cast<float>(BoostPadGridCellSizeY())) +
           (BoostPadGridCellsY() / 2);
}


ROCKETGPU_HD constexpr uint64_t SoccarBoostPadVisitMaskX(int car_cell_x) {
    const int lo = car_cell_x - 1 > 0 ? car_cell_x - 1 : 0;
    const int hi = car_cell_x + 1 < BoostPadGridCellsX() - 1 ? car_cell_x + 1 : BoostPadGridCellsX() - 1;
    if (lo > hi) {
        return 0u;
    }
    uint64_t mask = 0u;
    for (int p = 0; p < BoostPadCountSoccar(); ++p) {
        const int cx = BoostPadGridIndexX(BoostPadLocSoccar(p).x);
        if (cx >= lo && cx <= hi) {
            mask |= uint64_t{1} << p;
        }
    }
    return mask;
}

ROCKETGPU_HD constexpr uint64_t SoccarBoostPadVisitMaskY(int car_cell_y) {
    const int lo = car_cell_y - 1 > 0 ? car_cell_y - 1 : 0;
    const int hi = car_cell_y + 1 < BoostPadGridCellsY() - 1 ? car_cell_y + 1 : BoostPadGridCellsY() - 1;
    if (lo > hi) {
        return 0u;
    }
    uint64_t mask = 0u;
    for (int p = 0; p < BoostPadCountSoccar(); ++p) {
        const int cy = BoostPadGridIndexY(BoostPadLocSoccar(p).y);
        if (cy >= lo && cy <= hi) {
            mask |= uint64_t{1} << p;
        }
    }
    return mask;
}

struct SoccarBoostPadShortMasks {
    uint64_t x[64]{};
    uint64_t y[80]{};
    constexpr SoccarBoostPadShortMasks() {
        for (int ix = 0; ix < 64; ++ix) {
            const float x0 = -4096.0f + static_cast<float>(ix) * 128.0f;
            const float x1 = x0 + 128.0f;
            uint64_t m = 0u;
            for (int p = 0; p < BoostPadCountSoccar(); ++p) {
                const float px = BoostPadLocSoccar(p).x;
                if (px >= x0 - 274.0f && px <= x1 + 274.0f) {
                    m |= uint64_t{1} << p;
                }
            }
            x[ix] = m;
        }
        for (int iy = 0; iy < 80; ++iy) {
            const float y0 = -5120.0f + static_cast<float>(iy) * 128.0f;
            const float y1 = y0 + 128.0f;
            uint64_t m = 0u;
            for (int p = 0; p < BoostPadCountSoccar(); ++p) {
                const float py = BoostPadLocSoccar(p).y;
                if (py >= y0 - 274.0f && py <= y1 + 274.0f) {
                    m |= uint64_t{1} << p;
                }
            }
            y[iy] = m;
        }
    }
};

struct SoccarBoostPadGridMasks {
    uint64_t x[BoostPadGridCellsX() + 2]{
        SoccarBoostPadVisitMaskX(-1), SoccarBoostPadVisitMaskX(0), SoccarBoostPadVisitMaskX(1),
        SoccarBoostPadVisitMaskX(2),  SoccarBoostPadVisitMaskX(3), SoccarBoostPadVisitMaskX(4),
        SoccarBoostPadVisitMaskX(5),  SoccarBoostPadVisitMaskX(6), SoccarBoostPadVisitMaskX(7),
        SoccarBoostPadVisitMaskX(8),
    };
    uint64_t y[BoostPadGridCellsY() + 2]{
        SoccarBoostPadVisitMaskY(-1), SoccarBoostPadVisitMaskY(0),  SoccarBoostPadVisitMaskY(1),
        SoccarBoostPadVisitMaskY(2),  SoccarBoostPadVisitMaskY(3),  SoccarBoostPadVisitMaskY(4),
        SoccarBoostPadVisitMaskY(5),  SoccarBoostPadVisitMaskY(6),  SoccarBoostPadVisitMaskY(7),
        SoccarBoostPadVisitMaskY(8),  SoccarBoostPadVisitMaskY(9),  SoccarBoostPadVisitMaskY(10),
    };
};

#if defined(__CUDACC__)
static __constant__ const SoccarBoostPadGridMasks kDeviceSoccarBoostPadGridMasks{};
static __constant__ const SoccarBoostPadShortMasks kDeviceSoccarBoostPadShortMasks{};
#endif

ROCKETGPU_HD uint64_t SoccarBoostPadGridCandidateMask(int car_cell_x, int car_cell_y) {
    const unsigned int ix = static_cast<unsigned int>(car_cell_x + 1);
    const unsigned int iy = static_cast<unsigned int>(car_cell_y + 1);
    if (ix >= BoostPadGridCellsX() + 2 || iy >= BoostPadGridCellsY() + 2) {
        return 0u;
    }
#if defined(__CUDA_ARCH__)
    return kDeviceSoccarBoostPadGridMasks.x[ix] & kDeviceSoccarBoostPadGridMasks.y[iy];
#else
    static constexpr SoccarBoostPadGridMasks kGridMasks{};
    return kGridMasks.x[ix] & kGridMasks.y[iy];
#endif
}

ROCKETGPU_HD uint64_t SoccarBoostPadCandidateMask(float x, float y, bool short_sweep) {
    if (short_sweep && x >= -4096.0f && x < 4096.0f && y >= -5120.0f && y < 5120.0f) {
        const unsigned int ix = static_cast<unsigned int>((x + 4096.0f) * (1.0f / 128.0f));
        const unsigned int iy = static_cast<unsigned int>((y + 5120.0f) * (1.0f / 128.0f));
        if (ix < 64u && iy < 80u) {
#if defined(__CUDA_ARCH__)
            return kDeviceSoccarBoostPadShortMasks.x[ix] & kDeviceSoccarBoostPadShortMasks.y[iy];
#else
            static constexpr SoccarBoostPadShortMasks kShortMasks{};
            return kShortMasks.x[ix] & kShortMasks.y[iy];
#endif
        }
    }
    return SoccarBoostPadGridCandidateMask(BoostPadGridIndexX(x), BoostPadGridIndexY(y));
}

ROCKETGPU_HD bool BoostPadBoxOverlap(
    const Vec3& pad_pos_uu, bool is_big, const Vec3& car_uu, const Vec3 axes[3], const CarConfig& car_cfg) {
    const float pad_r = is_big ? 160.0f : 128.0f;
    const float h[3] = {pad_r, pad_r, 64.0f};
    const float e[3] = {car_cfg.hitbox_size.x * 0.5f - 1.25f, car_cfg.hitbox_size.y * 0.5f - 1.25f,
                        car_cfg.hitbox_size.z * 0.5f - 1.25f};
    const Vec3 centre = car_uu + axes[0] * car_cfg.hitbox_offset.x + axes[1] * car_cfg.hitbox_offset.y +
                        axes[2] * car_cfg.hitbox_offset.z;
    const float t[3] = {centre.x - pad_pos_uu.x, centre.y - pad_pos_uu.y,
                        centre.z - (pad_pos_uu.z - (is_big ? 1.0f : 5.917984f))};
    float r[3][3];
    float ar[3][3];
    for (int j = 0; j < 3; ++j) {
        const float c[3] = {axes[j].x, axes[j].y, axes[j].z};
        for (int i = 0; i < 3; ++i) {
            r[i][j] = c[i];
            ar[i][j] = Abs(c[i]) + 1e-6f;
        }
    }
    for (int i = 0; i < 3; ++i) {
        if (Abs(t[i]) > h[i] + e[0] * ar[i][0] + e[1] * ar[i][1] + e[2] * ar[i][2]) {
            return false;
        }
    }
    for (int j = 0; j < 3; ++j) {
        const float tb = t[0] * r[0][j] + t[1] * r[1][j] + t[2] * r[2][j];
        if (Abs(tb) > h[0] * ar[0][j] + h[1] * ar[1][j] + h[2] * ar[2][j] + e[j]) {
            return false;
        }
    }
    for (int i = 0; i < 3; ++i) {
        const int i1 = (i + 1) % 3;
        const int i2 = (i + 2) % 3;
        for (int j = 0; j < 3; ++j) {
            const int j1 = (j + 1) % 3;
            const int j2 = (j + 2) % 3;
            const float ra = h[i1] * ar[i2][j] + h[i2] * ar[i1][j];
            const float rb = e[j1] * ar[i][j2] + e[j2] * ar[i][j1];
            if (Abs(t[i2] * r[i1][j] - t[i1] * r[i2][j]) > ra + rb) {
                return false;
            }
        }
    }
    return true;
}

ROCKETGPU_HD Vec3 BoostPadRbPosUu(const Vec3& pos_bt) {
    const Vec3 p = pos_bt * 50.0f;
    return Vec3(QuantRoundPosV3(p.x, 100.0f), QuantRoundPosV3(p.y, 100.0f), QuantRoundPosV3(p.z, 100.0f));
}

ROCKETGPU_HD Vec3 BoostPadFramePos(const SimConfig& cfg, int env_idx, const CarState& car) {
    const Vec3 p0 = BoostPadRbPosUu(car.tick_start_pos_bt);
    const Vec3 p1 = BoostPadRbPosUu(car.pos_bt);
    return p0 + (p1 - p0) * FrameFrac(cfg, env_idx);
}

ROCKETGPU_HD void BoostPadFrameAxes(const SimConfig& cfg, int env_idx, const CarState& car, Vec3 axes[3]) {
    CarBasisColumns(car, axes[0], axes[1], axes[2]);
    const Vec3 ang_vel = ((car.reported_flags & CarState::kReportedValid) != 0 || car.is_demoed)
        ? car.ang_vel
        : NativeSseClampLength(car.ang_vel, cfg.mutator.car_max_ang_speed,
                               cfg.mutator.native_sphere_rsqrt_table, cfg.mutator.native_sphere_rsqrt_shift);
    const Vec3 spin_back = ang_vel * (-(1.0f - FrameFrac(cfg, env_idx)) * cfg.tick_dt);
    for (int a = 0; a < 3; ++a) {
        axes[a] = axes[a] + Cross(spin_back, axes[a]);
    }
}

ROCKETGPU_HD Vec3 BoostPadSweepFromBt(const CarState& car, const Vec3& car_bt) {
    if (car.pad_frame_pos_valid) {
        return car.pad_frame_pos_bt;
    }
    const Vec3 p0 = BoostPadRbPosUu(car.tick_start_pos_bt);
    const Vec3 p1 = BoostPadRbPosUu(car.pos_bt);
    return car_bt - (p1 - p0) * (1.0f / 50.0f);
}

ROCKETGPU_HD void BoostPadStoreFramePos(const SimConfig& cfg, int env_idx, CarState& car) {
    car.pad_frame_pos_bt = BoostPadFramePos(cfg, env_idx, car) * (1.0f / 50.0f);
    car.pad_frame_pos_valid = car.is_demoed ? 0 : 1;
}

#if !defined(ROCKETGPU_USE_CUDA) && !defined(__CUDACC__)
struct SoccarBoostPadPositions {
    Vec3 positions[BoostPadCountSoccar()]{};
    constexpr SoccarBoostPadPositions() {
        for (int p = 0; p < BoostPadCountSoccar(); ++p) positions[p] = BoostPadLocSoccar(p);
    }
};
#endif

ROCKETGPU_HD uint64_t BoostPadContactsForCar(
    int env_idx,
    const SimConfig& cfg,
    const CarState& car,
    const CarConfig& car_cfg,
    const BoostPadState* pads,
    uint64_t& overlap,
    Vec3* frame_pos_bt_out = nullptr) {
    static_assert(kMaxBoostPads <= 64, "pad mask");
    uint64_t hits = 0u;
    overlap = 0u;
    const float frame_frac = FrameFrac(cfg, env_idx);
    if (!cfg.enable_boost_pads || car.is_demoed || frame_frac < 0.0f) {
        return hits;
    }

    const int pad_base = env_idx * BoostPadCount(cfg);
    const Vec3 car_pos = BoostPadFramePos(cfg, env_idx, car);
    const Vec3 car_bt = car_pos * (1.0f / 50.0f);
    if (frame_pos_bt_out != nullptr) {
        *frame_pos_bt_out = car_bt;
    }
    const Vec3 prev_car_bt = BoostPadSweepFromBt(car, car_bt);
    const Vec3 move_bt = car_bt - prev_car_bt;
    const float move_len_sq = Dot(move_bt, move_bt);
    const bool long_sweep = move_len_sq > (240.0f / 50.0f) * (240.0f / 50.0f);
    const bool short_sweep = move_len_sq <= (32.0f / 50.0f) * (32.0f / 50.0f);
    const Vec3 lo_bt(Min(car_bt.x, prev_car_bt.x), Min(car_bt.y, prev_car_bt.y), Min(car_bt.z, prev_car_bt.z));
    const Vec3 hi_bt(Max(car_bt.x, prev_car_bt.x), Max(car_bt.y, prev_car_bt.y), Max(car_bt.z, prev_car_bt.z));
    constexpr float kUUToBT = 1.0f / 50.0f;
    if (cfg.use_custom_boost_pads == 0 && cfg.boost_pad_count <= 0) {
        if ((car_pos.z <= BoostPadGridExtentZUU() || long_sweep) &&
            lo_bt.z - 135.0f * kUUToBT <= 73.0f * kUUToBT &&
            hi_bt.z + 135.0f * kUUToBT >= 70.0f * kUUToBT) {
            uint64_t cand = long_sweep
                ? ((uint64_t{1} << BoostPadCountSoccar()) - 1u)
                : SoccarBoostPadCandidateMask(car_pos.x, car_pos.y, short_sweep);
            for (; cand != 0u; cand &= cand - 1u) {
                const int p = CountTrailingZeros64(cand);
#if !defined(ROCKETGPU_USE_CUDA) && !defined(__CUDACC__)
                static constexpr SoccarBoostPadPositions positions{};
                const Vec3 pad_pos = positions.positions[p];
#else
                const Vec3 pad_pos = BoostPadLocSoccar(p);
#endif
                const Vec3 pad_bt = pad_pos * kUUToBT;
                if (pad_bt.x < lo_bt.x - 240.0f * kUUToBT || pad_bt.x > hi_bt.x + 240.0f * kUUToBT ||
                    pad_bt.y < lo_bt.y - 240.0f * kUUToBT || pad_bt.y > hi_bt.y + 240.0f * kUUToBT ||
                    pad_bt.z < lo_bt.z - 135.0f * kUUToBT || pad_bt.z > hi_bt.z + 135.0f * kUUToBT) {
                    continue;
                }
                if (BoostPadSweepCollideBt(pad_pos, pads[pad_base + p], prev_car_bt, car_bt)) {
                    hits |= uint64_t{1} << p;
                }
            }
        }
    } else {
        for (int p = 0; p < BoostPadCount(cfg); ++p) {
            const Vec3 pad_pos = BoostPadLoc(cfg, p);
            const Vec3 pad_bt = pad_pos * kUUToBT;
            if (pad_bt.x < lo_bt.x - 240.0f * kUUToBT || pad_bt.x > hi_bt.x + 240.0f * kUUToBT ||
                pad_bt.y < lo_bt.y - 240.0f * kUUToBT || pad_bt.y > hi_bt.y + 240.0f * kUUToBT ||
                pad_bt.z < lo_bt.z - 135.0f * kUUToBT || pad_bt.z > hi_bt.z + 135.0f * kUUToBT) {
                continue;
            }
            const BoostPadState& pad = pads[pad_base + p];
            if (BoostPadSweepCollideBt(pad_pos, pad, prev_car_bt, car_bt)) {
                hits |= uint64_t{1} << p;
            }
        }
    }
    uint64_t check_overlap = hits;
    if (car.boost_pad_touching != 0) {
        check_overlap |= uint64_t{1} << (car.boost_pad_touching - 1);
    }
    if (check_overlap != 0u) {
        Vec3 axes[3];
        BoostPadFrameAxes(cfg, env_idx, car, axes);
        for (; check_overlap != 0u; check_overlap &= check_overlap - 1u) {
            const int p = CountTrailingZeros64(check_overlap);
            if (BoostPadBoxOverlap(BoostPadLoc(cfg, p), pads[pad_base + p].is_big != 0, car_pos, axes, car_cfg)) {
                overlap |= uint64_t{1} << p;
            }
        }
    }
    return hits;
}

ROCKETGPU_HD bool BoostPadEligible(const SimConfig& cfg, const CarState& car) {
    return !car.is_demoed && !car.driving_disabled && car.boost < cfg.mutator.boost_max;
}

ROCKETGPU_HD void BoostPadTouchStep(
    BoostPadState& pad, int p, CarState& car, uint32_t car_id, bool in_cylinder, bool overlapping, bool eligible) {
    if (!pad.is_active || pad.cur_locked_car_id != 0) {
        return;
    }
    const uint8_t touching = static_cast<uint8_t>(p + 1);
    if (car.boost_pad_touching == touching) {
        if (!overlapping) {
            car.boost_pad_touching = 0;
        }
    } else if (in_cylinder) {
        if (eligible) {
            car.boost_pad_touching = touching;
            pad.cur_locked_car_id = car_id;
        } else if (overlapping) {
            car.boost_pad_touching = touching;
        }
    }
}

ROCKETGPU_HD void BoostPadRespawnStep(
    int env_idx, int count, const SimConfig& cfg, CarState* cars, const CarConfig* car_configs, const EnvState& env,
    BoostPadState& pad, int p) {
    if (pad.is_active || !BoostPadRespawnDue(pad, cfg.tick_dt, FrameFrac(cfg, env_idx))) {
        return;
    }
    pad.cooldown = 0.0f;
    pad.is_active = 1;
    const int base = env_idx * cfg.max_cars_per_env;
    const Vec3 pad_pos = BoostPadLoc(cfg, p);
    for (int k = count - 1; k >= 0; --k) {
        const int i = CarVisitSlot(env, k, count);
        CarState& car = cars[base + i];
        if (car.is_demoed) {
            continue;
        }
        const Vec3 car_uu = BoostPadFramePos(cfg, env_idx, car);
        Vec3 axes[3];
        BoostPadFrameAxes(cfg, env_idx, car, axes);
        if (BoostPadBoxOverlap(pad_pos, pad.is_big != 0, car_uu, axes, car_configs[base + i])) {
            car.boost_pad_touching = static_cast<uint8_t>(p + 1);
        }
    }
}

ROCKETGPU_HD void BoostPadTouchListStep(
    BoostPadState& pad, int p, const CarState& car, uint32_t car_id, bool eligible) {
    if (pad.is_active && pad.cur_locked_car_id == 0 && car.boost_pad_touching == p + 1 && eligible) {
        pad.cur_locked_car_id = car_id;
    }
}

ROCKETGPU_HD void BoostPadGive(int env_idx, const SimConfig& cfg, CarState* cars, BoostPadState& pad, int p,
                               uint32_t car_id) {
    CarState& car = cars[car_id - 1u];
    const float boost_amount = pad.is_big ? 100.0f : 12.0f;
    const float event_boost_before = car.boost;
    car.boost = Min(cfg.mutator.boost_max, car.boost + boost_amount);
    pad.is_active = 0;
    pad.cooldown = (pad.is_big ? cfg.mutator.boost_pad_cooldown_big : cfg.mutator.boost_pad_cooldown_small) -
                   cfg.tick_dt;
    pad.pickup_frac_q = BoostPadFracQ(FrameFrac(cfg, env_idx));
    const int car_base = env_idx * cfg.max_cars_per_env;
    for (int i = 0; i < cfg.max_cars_per_env; ++i) {
        if (cars[car_base + i].boost_pad_touching == p + 1) {
            cars[car_base + i].boost_pad_touching = 0;
        }
    }
    if (cfg.mutator.physics_events.windows) {
        PhysicsEvent event;
        event.type = PhysicsEventType::BoostPickup;
        event.car_id = car_id;
        event.pad_index = p;
        event.is_big_pad = pad.is_big;
        event.boost_before = event_boost_before;
        event.boost_after = car.boost;
        cfg.mutator.physics_events.Emit(env_idx, event);
    }
}

ROCKETGPU_HD void BoostPadSweepTouchSteps(
    int env_idx, const SimConfig& cfg, CarState* cars, BoostPadState* pads, int car_slot, uint64_t hits,
    uint64_t overlap) {
    CarState& car = cars[car_slot];
    const uint32_t car_id = static_cast<uint32_t>(car_slot + 1);
    const Vec3 to_bt = BoostPadFramePos(cfg, env_idx, car) * (1.0f / 50.0f);
    const Vec3 from_bt = BoostPadSweepFromBt(car, to_bt);
    if (car.boost_pad_touching != 0 && ((hits >> (car.boost_pad_touching - 1)) & 1u) == 0u) {
        const int p = car.boost_pad_touching - 1;
        BoostPadTouchStep(pads[p], p, car, car_id, false, ((overlap >> p) & 1u) != 0u, false);
    }
    for (uint64_t left = hits; left != 0u;) {
        int p = -1;
        float best = 0.0f;
        for (uint64_t m = left; m != 0u; m &= m - 1u) {
            const int q = CountTrailingZeros64(m);
            float s = 0.0f;
            BoostPadSweepEnterBt(BoostPadLoc(cfg, q), pads[q], from_bt, to_bt, s);
            if (p < 0 || s < best) {
                p = q;
                best = s;
            }
        }
        if (p < 0) {
            break;
        }
        left &= ~(uint64_t{1} << p);
        BoostPadState& pad = pads[p];
        if (!pad.is_active || pad.cur_locked_car_id != 0) {
            continue;
        }
        if (car.boost_pad_touching == p + 1) {
            if (((overlap >> p) & 1u) == 0u) {
                car.boost_pad_touching = 0;
            }
            continue;
        }
        if (BoostPadEligible(cfg, car)) {
            car.boost_pad_touching = static_cast<uint8_t>(p + 1);
            pad.cur_locked_car_id = car_id;
            BoostPadGive(env_idx, cfg, cars, pad, p, car_id);
        } else if (((overlap >> p) & 1u) != 0u) {
            car.boost_pad_touching = static_cast<uint8_t>(p + 1);
        }
    }
}

ROCKETGPU_HD void BoostPadTouchPassForEnv(
    int env_idx, int count, const SimConfig& cfg, CarState* cars, const CarConfig* car_configs, const EnvState& env,
    BoostPadState* pads) {
    if (!cfg.enable_boost_pads || FrameFrac(cfg, env_idx) < 0.0f) {
        return;
    }
    const int base = env_idx * cfg.max_cars_per_env;
    const int pad_count = BoostPadCount(cfg);
    const int pad_base = env_idx * pad_count;
    for (int k = 0; k < count; ++k) {
        const int i = CarVisitSlot(env, k, count);
        CarState& car = cars[base + i];
        uint64_t overlap = 0u;
        Vec3 frame_pos_bt;
        const uint64_t hits = BoostPadContactsForCar(env_idx, cfg, car, car_configs[base + i], pads, overlap, &frame_pos_bt);
        if ((hits & (hits - 1u)) != 0u) {
            BoostPadSweepTouchSteps(env_idx, cfg, cars, pads + pad_base, base + i, hits, overlap);
            if (car.is_demoed) {
                BoostPadStoreFramePos(cfg, env_idx, car);
            } else {
                car.pad_frame_pos_bt = frame_pos_bt;
                car.pad_frame_pos_valid = 1;
            }
            continue;
        }
        const bool eligible = BoostPadEligible(cfg, car);
        uint64_t near = hits;
        if (car.boost_pad_touching != 0) {
            near |= uint64_t{1} << (car.boost_pad_touching - 1);
        }
        for (; near != 0u; near &= near - 1u) {
            const int p = CountTrailingZeros64(near);
            BoostPadTouchStep(pads[pad_base + p], p, car, static_cast<uint32_t>(base + i + 1), ((hits >> p) & 1u) != 0u,
                              ((overlap >> p) & 1u) != 0u, eligible);
        }
        if (car.is_demoed) {
            BoostPadStoreFramePos(cfg, env_idx, car);
        } else {
            car.pad_frame_pos_bt = frame_pos_bt;
            car.pad_frame_pos_valid = 1;
        }
    }
    for (int p = 0; p < pad_count; ++p) {
        if (!pads[pad_base + p].is_active) {
            BoostPadRespawnStep(env_idx, count, cfg, cars, car_configs, env, pads[pad_base + p], p);
        }
    }
    for (int k = count - 1; k >= 0; --k) {
        const int i = CarVisitSlot(env, k, count);
        const CarState& car = cars[base + i];
        if (car.boost_pad_touching != 0) {
            const int p = car.boost_pad_touching - 1;
            BoostPadTouchListStep(pads[pad_base + p], p, car, static_cast<uint32_t>(base + i + 1),
                                  BoostPadEligible(cfg, car));
        }
    }
}

ROCKETGPU_HD void PostTickBoostPad(int env_idx, const SimConfig& cfg, CarState* cars, BoostPadState* pads, int p) {
    {
        BoostPadState& pad = pads[env_idx * BoostPadCount(cfg) + p];
        const uint32_t locked_car_id = pad.cur_locked_car_id;
        if (locked_car_id != 0) {
            if (pad.is_active) {
                BoostPadGive(env_idx, cfg, cars, pad, p, locked_car_id);
            }
        }
        if (pad.prev_locked_car_id != locked_car_id) {
            pad.prev_locked_car_id = locked_car_id;
        }
    }
}

ROCKETGPU_HD void PostTickBoostPadsForEnv(int env_idx, const SimConfig& cfg, CarState* cars, BoostPadState* pads) {
    if (!cfg.enable_boost_pads) {
        return;
    }
    for (int p = 0; p < BoostPadCount(cfg); ++p) {
        PostTickBoostPad(env_idx, cfg, cars, pads, p);
    }
}

ROCKETGPU_HD void RecordArenaWorldContact(CarState& car, const ContactAccumulator& acc) {
    if (!acc.hit) {
        return;
    }
    car.world_contact.has_contact = 1;
    car.world_contact.contact_normal = acc.latest_normal;
}

ROCKETGPU_HD int BodyDampingClampCar(CarState& car, const MutatorConfig& mutator, bool linear, bool angular) {
    int clamped = 0;
    if (linear) {
        const Vec3 v = car.vel_bt;
        const float max_lin = mutator.car_max_speed * 0.02f;
        const float vsq = (v.x * v.x + v.y * v.y) + v.z * v.z;
        if (0.0f < max_lin && max_lin * max_lin < vsq) {
            const float s = 1.0f / sqrtf(vsq);
            car.vel_bt = Vec3((v.x * s) * max_lin, (v.y * s) * max_lin, (v.z * s) * max_lin);
            clamped |= 1;
        }
    }
    if (angular) {
        const Vec3 w = car.ang_vel;
        const float max_ang = mutator.car_max_ang_speed;
        const float lsq = (w.x * w.x + w.y * w.y) + w.z * w.z;
        if (0.0f < max_ang && max_ang * max_ang < lsq) {
            const float s = 1.0f / sqrtf(lsq);
            car.ang_vel = Vec3((w.x * s) * max_ang, (w.y * s) * max_ang, (w.z * s) * max_ang);
            clamped |= 2;
        }
    }
    return clamped;
}

ROCKETGPU_HD void BodyDampingClampBall(BallState& ball, const MutatorConfig& mutator) {
    const Vec3 v = ball.vel_bt;
    const float max_lin = mutator.ball_max_speed * 0.02f;
    const float vsq = (v.x * v.x + v.y * v.y) + v.z * v.z;
    if (0.0f < max_lin && max_lin * max_lin < vsq) {
        const float s = 1.0f / sqrtf(vsq);
        SetBallLinearVelocityBt(ball, Vec3((v.x * s) * max_lin, (v.y * s) * max_lin, (v.z * s) * max_lin));
    }
    const Vec3 w = ball.ang_vel;
    const float max_ang = mutator.ball_max_ang_speed;
    const float lsq = (w.x * w.x + w.y * w.y) + w.z * w.z;
    if (0.0f < max_ang && max_ang * max_ang < lsq) {
        const float s = 1.0f / sqrtf(lsq);
        ball.ang_vel = Vec3((w.x * s) * max_ang, (w.y * s) * max_ang, (w.z * s) * max_ang);
    }
}

ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE CarControls PreTickCarControls(const CarState& car, CarControls controls, float& raw_pitch) {
    ClampControls(controls);
    raw_pitch = controls.pitch;
    controls = QuantizeAxisControls(controls);
    if (car.driving_disabled) {
        const float throttle = controls.throttle;
        controls = CarControls{};
        controls.throttle = throttle;
    }
    return controls;
}

constexpr uint8_t kCalPreAirJumpStarted = 1, kCalPreFlipStarted = 2, kCalPreFlipVelChanged = 4,
    kCalPreAirJumpUsedBefore = 8, kCalPreSetV3Tick = 16, kCalPreFlippingAtStart = 32, kCalPreWasJumping = 64;

ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE void PreTickCarPrePhysics(
    CarState& car,
    CarControls controls,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    float frame_frac,
    float dt,
    WheelRuntimeState* ROCKETGPU_RESTRICT wheel_runtime_state,
    const CarRbSnapshot& tick_start_snapshot) {
    car.cal_pre_flags = 0;
    const uint8_t start_is_flipping = car.is_flipping;
    const uint8_t start_has_flipped = car.has_flipped;
    const uint8_t start_has_jumped = car.has_jumped;
    const uint8_t start_has_double_jumped = car.has_double_jumped;
    const uint8_t start_is_jumping = car.is_jumping;
    const uint8_t start_is_on_ground = car.is_on_ground;
    const float start_internal_flip_time = car.internal_flip_time;
    const float start_internal_jump_time = car.internal_jump_time;
    const bool start_last_jump = car.last_controls.jump;
    const uint64_t start_tick_count_since_update = car.tick_count_since_update;
    const int start_wheel_contacts =
        static_cast<int>(car.wheels_with_contact[0] != 0) + static_cast<int>(car.wheels_with_contact[1] != 0) +
        static_cast<int>(car.wheels_with_contact[2] != 0) + static_cast<int>(car.wheels_with_contact[3] != 0);
    car.demo_no_contact_response = car.is_demoed ? 1 : 0;
    if (car.is_demoed) {
        return;
    }
    if (car.driving_disabled) {
        controls.jump = false;
    }
    const bool cal_set_v3_tick = car.supersonic_set_v3 && start_tick_count_since_update == 0;
    if (car.supersonic_set_v3) {
        car.supersonic_set_v3 = 0;
    } else {
        UpdateSupersonicCalDemo(car, mutator, dt);
    }

    car.vel_touched_solve_stage = 0;
    const uint8_t flipping_at_start = start_is_flipping;
    bool flip_started = false;

    const bool jump_pressed = controls.jump && !start_last_jump;
    const bool need_pre_frame =
        cfg.three_wheels != 0 || car.is_auto_flipping != 0 ||
        (jump_pressed && car.world_contact.has_contact && car.world_contact.contact_normal.z > 0.707f);
    CarFrame car_frame;
    const CarFrame* frame = nullptr;
    if (need_pre_frame) {
        car_frame = MakeCarFrame(car, cfg, mutator.car_mass);
        frame = &car_frame;
    }
    const uint8_t cal_has_flipped_at_start = start_has_flipped;
    if (start_tick_count_since_update == 0 && wheel_runtime_state != nullptr && wheel_runtime_state->cal_restored) {
        car.flip_timer_live = 1;
    }
    const float cal_pitchlock_restart_time =
        (!cal_has_flipped_at_start && !start_is_flipping && (start_internal_flip_time > 0.0f || car.flip_timer_live))
        ? start_internal_flip_time : -1.0f;

    bool cal_air_jump_started = false;

    const bool cal_jump = wheel_runtime_state != nullptr;
    const bool cal_jump_g = cal_jump && start_wheel_contacts >= 3;
    if (cal_jump) {
        WheelRuntimeState& jw = *wheel_runtime_state;
        if (start_tick_count_since_update == 0 && !jw.cal_restored) {
            jw.cal_game_jumped = start_has_jumped ? 1 : 0;
            // landing clears the game's double-jumped flag, so a car set on the ground mid-dodge holds 0
            jw.cal_game_double_jumped = (start_has_double_jumped || start_has_flipped) && !cal_jump_g ? 1 : 0;
            jw.cal_cached_ground = cal_jump_g ? 1 : 0;
            jw.cal_jump_inactive_time = start_is_jumping ? 0.0f : start_internal_jump_time;
            jw.cal_frame_open = 0;
        }
        jw.cal_restored = 0;
        if (!jw.cal_frame_open) {
            if (cal_jump_g && !jw.cal_cached_ground) {
                jw.cal_game_jumped = 0;
                jw.cal_game_double_jumped = 0;
            }
            jw.cal_cached_ground = cal_jump_g ? 1 : 0;
        }
        if (jw.cal_game_jumped && cal_jump_g && !start_is_jumping && jw.cal_jump_inactive_time >= 0.125f) {
            jw.cal_game_jumped = 0;
            jw.cal_game_double_jumped = 0;
            jw.cal_cached_ground = 1;
        }
        jw.cal_frame_open = frame_frac < 0.0f ? 1 : 0;
    }
    const bool cal_jump_blocked = cal_jump && wheel_runtime_state->cal_game_jumped;
    const uint8_t was_jumping = start_is_jumping;
    float forward_speed_uu = SnapshotForwardSpeed(tick_start_snapshot);
    if (jump_pressed && start_is_on_ground && !start_is_jumping && !cal_jump_blocked) {
        StartJump(car, cfg, mutator, frame);
        cal_air_jump_started = true;
    }

    const bool cal_air_window_closed = cal_jump &&
        (wheel_runtime_state->cal_game_double_jumped ||
         (wheel_runtime_state->cal_game_jumped && wheel_runtime_state->cal_jump_inactive_time > mutator.doublejump_max_delay));
    const bool air_jump_pressed = jump_pressed && !(cal_jump && cal_jump_g) && !cal_air_window_closed;
    const bool cal_press_as_air = cal_jump && !cal_jump_g && jump_pressed && !cal_air_window_closed;
    const uint8_t air_jump_used_before = (car.has_double_jumped || car.has_flipped) ? 1 : 0;

    bool cal_flip_before_wheels_vel_changed = false;
    {
        const bool try_air_jump = air_jump_pressed && !cal_air_jump_started;
        float cal_dodge_raw_pitch = 2.0f;
        if (try_air_jump) {
            controls = PreTickCarControls(car, controls, cal_dodge_raw_pitch);
        }
        const float cal_flip_before_wheels_forward_speed = try_air_jump ? WheelForwardSpeed(car) : 0.0f;
        const Vec3 fbw_vel_bt = car.vel_bt;
        UpdateDoubleJumpOrFlipState(car, controls, cfg, mutator, try_air_jump,
                                    cal_flip_before_wheels_forward_speed, dt, cal_press_as_air && !cal_air_jump_started,
                                    cal_dodge_raw_pitch, &flip_started, frame);
        cal_flip_before_wheels_vel_changed =
            car.vel_bt.x != fbw_vel_bt.x || car.vel_bt.y != fbw_vel_bt.y || car.vel_bt.z != fbw_vel_bt.z;
    }

    ApplyAutoFlip(car, controls, cfg, mutator, dt, frame);
    car.cal_pre_flags = static_cast<uint8_t>(
        (cal_air_jump_started ? kCalPreAirJumpStarted : 0) | (flip_started ? kCalPreFlipStarted : 0) |
        (cal_flip_before_wheels_vel_changed ? kCalPreFlipVelChanged : 0) |
        (air_jump_used_before ? kCalPreAirJumpUsedBefore : 0) | (cal_set_v3_tick ? kCalPreSetV3Tick : 0) |
        (flipping_at_start ? kCalPreFlippingAtStart : 0) | (was_jumping ? kCalPreWasJumping : 0));
    car.cal_pre_forward_speed_uu = forward_speed_uu;
    car.cal_pre_pitchlock_restart_time = cal_pitchlock_restart_time;
}

template <bool kStagedIndependent = false>
ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE void PreTickCar(
    CarState& car,
    CarControls controls,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    float dt,
    WheelRuntimeState* ROCKETGPU_RESTRICT wheel_runtime_state = nullptr,
    const ArenaMeshTriangle* mesh_triangles = nullptr,
    int32_t mesh_triangle_count = 0,
    const CarState* cars_in_env = nullptr,
    const CarConfig* car_cfgs_in_env = nullptr,
    int32_t cars_in_env_count = 0,
    int32_t self_car_idx = -1,
    const BallState* ball = nullptr,
    float ball_radius = 0.0f,
    int32_t env_idx = -1,
    const ArenaMeshBvhQuantization* mesh_bvh_quantization = nullptr,
    int32_t mesh_bvh_quantization_count = 0,
    const ArenaMeshBvhMeshInfo* mesh_bvh_meshes = nullptr,
    int32_t mesh_bvh_mesh_count = 0,
    const ArenaMeshBvhNode* mesh_bvh_nodes = nullptr,
    int32_t mesh_bvh_node_count = 0,
    const ArenaMeshBvhSubtreeInfo* mesh_bvh_subtrees = nullptr,
    int32_t mesh_bvh_subtree_count = 0,
    const WheelRayStage* staged_rays = nullptr,
    CarState* cal_push_cars = nullptr,
    BallState* cal_push_ball = nullptr,
    uint32_t* cal_pushed_cars_mask = nullptr,
    bool staged_rays_velocity_stale = false,
    uint32_t wheel_ray_excluded_bodies = 0u,
    CarFrame* frame_out = nullptr,
    const ArenaMeshLeafGrid* leaf_grid = nullptr) {
    car.demo_no_contact_response = car.is_demoed ? 1 : 0;
    if (car.is_demoed) {
        return;
    }
    float cal_dodge_raw_pitch = 2.0f;
    controls = PreTickCarControls(car, controls, cal_dodge_raw_pitch);
    const uint8_t cal_pre = car.cal_pre_flags;
    const CarFrame car_frame = MakeCarFrame(car, cfg, mutator.car_mass);
    const CarFrame* const frame = &car_frame;
    if (frame_out != nullptr) {
        frame_out->inv_world_row0_bt = car_frame.inv_world_row0_bt;
        frame_out->inv_world_row1_bt = car_frame.inv_world_row1_bt;
        frame_out->inv_world_row2_bt = car_frame.inv_world_row2_bt;
    }
    const bool cal_set_v3_tick = (cal_pre & kCalPreSetV3Tick) != 0;
    if (wheel_runtime_state) {
        const auto rot = car.rot;
        const Vec3 basis_forward = car.basis_forward;
        const Vec3 basis_right = car.basis_right;
        const Vec3 basis_up = car.basis_up;
        const auto basis_authoritative = car.basis_authoritative;
        car_frame.queue_forces = true;
        wheel_runtime_state->pre_solve_rot = rot;
        wheel_runtime_state->pre_solve_basis_forward = car.tick_start_basis_forward;
        wheel_runtime_state->pre_solve_basis_right = car.tick_start_basis_right;
        wheel_runtime_state->pre_solve_basis_up = car.tick_start_basis_up;
        wheel_runtime_state->pre_solve_basis_authoritative = 1;
        (void)basis_forward;
        (void)basis_right;
        (void)basis_up;
        (void)basis_authoritative;
    }
    const uint8_t flipping_at_start = (cal_pre & kCalPreFlippingAtStart) ? 1 : 0;
    const bool flip_started = (cal_pre & kCalPreFlipStarted) != 0;
    const float cal_pitchlock_restart_time = car.cal_pre_pitchlock_restart_time;
    const bool cal_air_jump_started = (cal_pre & kCalPreAirJumpStarted) != 0;
    const bool cal_jump = wheel_runtime_state != nullptr;
    const uint8_t was_jumping = (cal_pre & kCalPreWasJumping) ? 1 : 0;
    float forward_speed_uu = car.cal_pre_forward_speed_uu;
    const uint8_t air_jump_used_before = (cal_pre & kCalPreAirJumpUsedBefore) ? 1 : 0;
    const bool cal_flip_before_wheels_vel_changed = (cal_pre & kCalPreFlipVelChanged) != 0;
    const int prev_wheel_contacts =
        static_cast<int>(car.wheels_with_contact[0] != 0) + static_cast<int>(car.wheels_with_contact[1] != 0) +
        static_cast<int>(car.wheels_with_contact[2] != 0) + static_cast<int>(car.wheels_with_contact[3] != 0);

    const bool autoroll_cal = wheel_runtime_state != nullptr;
    const bool autoroll_prev_contact = autoroll_cal && car.tick_count_since_update != 0;
    Vec3 autoroll_normal_sum =
        autoroll_prev_contact ? wheel_runtime_state->cal_prev_wheel_normal_sum : Vec3(0.0f, 0.0f, 0.0f);
    int autoroll_world_contacts = autoroll_prev_contact ? wheel_runtime_state->cal_prev_world_wheel_contacts : 0;

    UninitArray<WheelStepState, 4> wheels;
    const int num_wheels_in_contact = PreTickVehicleDynamics<kStagedIndependent>(
        car,
        controls,
        cfg,
        mutator,
        dt,
        wheels,
        forward_speed_uu,
        wheel_runtime_state,
        mesh_triangles,
        mesh_triangle_count,
        cars_in_env,
        car_cfgs_in_env,
        cars_in_env_count,
        self_car_idx,
        env_idx,
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
        staged_rays,

        cal_air_jump_started || staged_rays_velocity_stale || cal_flip_before_wheels_vel_changed,
        cal_push_cars,
        cal_push_ball,
        cal_pushed_cars_mask,
        frame,
        wheel_ray_excluded_bodies,
        leaf_grid);

    bool wheel_world_contact = false;
#pragma unroll
    for (int wheel = 0; wheel < 4; ++wheel)
        wheel_world_contact = wheel_world_contact ||
                              (wheels[wheel].is_in_contact != 0 && wheels[wheel].is_in_contact_with_world != 0);
    if (wheel_world_contact) {
        car.RecordWorldSurface(false);
    }
    if (car.HasFlipOrJump() && !car.flip_or_jump_last_tick)
        car.last_flip_or_jump_start = car.tick_count_since_update;

    const int air_gate_wheel_contacts =
        (car.tick_count_since_update != 0 || cal_set_v3_tick) ? prev_wheel_contacts :
        ((prev_wheel_contacts > num_wheels_in_contact) ? prev_wheel_contacts : num_wheels_in_contact);

    const int landing_prev_contacts = (car.tick_count_since_update == 0 && !cal_set_v3_tick)
        ? (Max(num_wheels_in_contact, prev_wheel_contacts))
        : prev_wheel_contacts;
    const uint8_t cal_flipping_before_air = car.is_flipping;
    const Vec3* air_damp_ang_vel = &car.cal_tick_start_ang_vel;
    const bool robin_air_blocked = cfg.three_wheels && car.world_contact.has_contact;
    if (cfg.three_wheels && !robin_air_blocked && !car.driving_disabled && landing_prev_contacts == 0 &&
        car.is_boosting) {
        ApplyRobinAirTorque(car, cfg, mutator.car_mass, dt, wheel_runtime_state, frame);
    }
    if (air_gate_wheel_contacts < 3) {
        Vec3 dodge_torque_bt(0.0f, 0.0f, 0.0f);
        ApplyFlipTorque(car, controls, cfg, mutator, dt, wheel_runtime_state, cal_dodge_raw_pitch, frame, &dodge_torque_bt);

        if (!car.driving_disabled && !robin_air_blocked) ApplyAirControl(car, controls, cfg, mutator, dt, landing_prev_contacts == 0, wheel_runtime_state, landing_prev_contacts == 0, cal_pitchlock_restart_time,
                        air_damp_ang_vel, frame);
        if (dodge_torque_bt.x != 0.0f || dodge_torque_bt.y != 0.0f || dodge_torque_bt.z != 0.0f) {
            QueueOfficialTorqueBt(wheel_runtime_state, dodge_torque_bt, frame);
        }
    } else {
        car.is_flipping = 0;

        if (landing_prev_contacts == 0 && !car.driving_disabled && !robin_air_blocked) {
            ApplyAirControl(car, controls, cfg, mutator, dt, true, wheel_runtime_state, true, cal_pitchlock_restart_time,
                            air_damp_ang_vel, frame);
        }
    }
    const bool cal_flip_ended = air_gate_wheel_contacts < 3 &&
        cal_flipping_before_air && !car.is_flipping;

    UpdateJumpState(car, controls, mutator, dt, wheel_runtime_state, frame);

    if (cal_jump) {
        WheelRuntimeState& jw = *wheel_runtime_state;

        if ((car.is_jumping && !was_jumping) || (!air_jump_used_before && (car.has_double_jumped || car.has_flipped))) {
            jw.cal_game_jumped = 1;
        }
        if (!air_jump_used_before && (car.has_double_jumped || car.has_flipped)) {
            jw.cal_game_double_jumped = 1;
        }
        jw.cal_jump_inactive_time = car.is_jumping ? 0.0f : jw.cal_jump_inactive_time + dt;
    }

    const bool jump_changed = cal_air_jump_started || (was_jumping && !car.is_jumping);
    const bool flip_changed = flip_started || car.is_flipping != flipping_at_start;
    if (jump_changed || !car.just_respawned) {
        const float jump_ticks = rintf(car.jump_time / dt) + 1.0f;
        car.jump_time = jump_changed ? dt : (jump_ticks <= 1024.0f ? car.jump_time + dt : jump_ticks * dt);
    }
    if (flip_changed || !car.just_respawned) {
        car.flip_time = flip_changed ? dt : car.flip_time + dt;
    }
    if (cal_air_jump_started || !car.is_jumping) {
        car.jump_hold_broken = 0;
    } else if (!controls.jump) {
        car.jump_hold_broken = 1;
    }

    if (autoroll_cal && !autoroll_prev_contact) {
        autoroll_normal_sum = wheel_runtime_state->cal_prev_wheel_normal_sum;
        autoroll_world_contacts = wheel_runtime_state->cal_prev_world_wheel_contacts;
    }
    int autoroll_contacts = autoroll_cal ? autoroll_world_contacts : num_wheels_in_contact;
    bool autoroll_gate = autoroll_cal
        ? (autoroll_contacts < 4 && (autoroll_contacts > 0 || car.world_contact.has_contact))
        : (((autoroll_contacts > 0) && (autoroll_contacts < 4)) || car.world_contact.has_contact);
    if (autoroll_cal && !autoroll_prev_contact &&
        prev_wheel_contacts > 0 && prev_wheel_contacts < 4) {
        const bool sticky = autoroll_world_contacts > 0;
        autoroll_gate = sticky || car.world_contact.has_contact;
        autoroll_contacts = sticky ? prev_wheel_contacts : 0;
    }
    if constexpr (!kStagedIndependent) {
        if (autoroll_cal && !autoroll_prev_contact && car.set_wheel_normals_mask) {
            autoroll_world_contacts = SetStateWorldWheelNormals(car, cfg, wheels, ball, ball_radius, cars_in_env,
                                                                car_cfgs_in_env, cars_in_env_count, self_car_idx,
                                                                autoroll_normal_sum, mesh_triangles, mesh_triangle_count,
                                                                mesh_bvh_quantization, mesh_bvh_quantization_count,
                                                                mesh_bvh_meshes, mesh_bvh_mesh_count, mesh_bvh_nodes,
                                                                mesh_bvh_node_count, mesh_bvh_subtrees,
                                                                mesh_bvh_subtree_count, mutator.cal_arena_local_tris);
            autoroll_contacts = autoroll_world_contacts;
            autoroll_gate = autoroll_contacts < 4 && (autoroll_contacts > 0 || car.world_contact.has_contact);
        }
    }
    if (controls.throttle && !(car.is_flipping) && !flipping_at_start && !cal_flip_ended && autoroll_gate &&
        !cfg.three_wheels) {
        ApplyAutoRollReference(car, cfg, mutator, autoroll_contacts, wheels, dt, wheel_runtime_state,
                               autoroll_cal ? &autoroll_normal_sum : nullptr, frame);
    }
    car.set_wheel_normals_mask = 0;

    car.world_contact.has_contact = 0;

    if (!car.just_respawned) {
        ApplyWheelSuspensionAndFriction(car, cfg, mutator, controls, dt, wheels, frame);
    }
    ApplyBoost(car, controls, mutator, dt, wheel_runtime_state, frame);
    if (wheel_runtime_state) {
        if (car_frame.has_queued_jump_force) {
            car_frame.queued_force += car_frame.queued_jump_force;
        }
        wheel_runtime_state->accumulated_force = car_frame.queued_force;
        wheel_runtime_state->accumulated_ang_accel = car_frame.queued_ang_accel;
        wheel_runtime_state->accumulated_torque_bt = car_frame.queued_torque_bt;
        wheel_runtime_state->solver_time_step = dt;
        if (env_idx >= 0 && self_car_idx >= 0) {
            if (mutator.physics_events.windows) {
                bool has_selected = false;
                Vec3 selected_normal(0.0f, 0.0f, 0.0f);
                int32_t selected_static_id = -1;
#pragma unroll
                for (int wheel = 0; wheel < 4; ++wheel) {
                    if (wheels[wheel].is_in_contact && wheels[wheel].is_in_contact_with_world &&
                        (!has_selected || wheels[wheel].contact_normal_ws.z > selected_normal.z)) {
                        has_selected = true;
                        selected_normal = wheels[wheel].contact_normal_ws;
                        selected_static_id = wheels[wheel].ground_static_id;
                    }
                }
                bool previous = false;
                if (has_selected && car.tick_count_since_update != 0) {
#pragma unroll
                    for (int wheel = 0; wheel < 4; ++wheel) {
                        const auto old = wheel_runtime_state->event_world_normals[wheel];
                        previous |= wheel_runtime_state->event_world_ids[wheel] == selected_static_id
                            && old.x == selected_normal.x && old.y == selected_normal.y && old.z == selected_normal.z;
                    }
                }
                if (has_selected && !previous) {
                    PhysicsEvent event;
                    event.type = PhysicsEventType::CarWorldContact;
                    event.car_id = env_idx * mutator.physics_events.cars_per_env + self_car_idx + 1;
                    event.position = car.pos;
                    event.car_velocity = car.vel;
                    event.car_on_ground = car.is_on_ground;
                    event.normal = selected_normal;
                    mutator.physics_events.Emit(env_idx, event);
                }
            }
#pragma unroll
            for (int wheel = 0; wheel < 4; ++wheel) {
                const bool world = wheels[wheel].is_in_contact && wheels[wheel].is_in_contact_with_world;
                wheel_runtime_state->event_world_ids[wheel] = world ? wheels[wheel].ground_static_id : -1;
                wheel_runtime_state->event_world_normals[wheel] = world ? wheels[wheel].contact_normal_ws : Vec3{};
            }
        }
    }
}

constexpr float kBallLinearSleepingThresholdBt = 0.8f;
constexpr float kBallAngularSleepingThreshold = 1.0f;
constexpr float kBulletDeactivationTime = 2.0f;

ROCKETGPU_HD uint8_t BallWakeStateAtTickStart(const BallState& ball, const MutatorConfig& mutator) {
    if (ball.sleeping != 2) {
        return ball.sleeping;
    }
    constexpr float kSoccarBallRadiusUu = 91.24999f;
    constexpr float kAwakeRestToleranceUu = 0.2f;
    const float rest_z = mutator.ball_radius + (BallRestZ() - kSoccarBallRadiusUu);
    return (mutator.ball_shape == 0 && LengthSq(ball.vel) == 0.0f && LengthSq(ball.ang_vel) == 0.0f &&
            ball.pos.z < rest_z - kAwakeRestToleranceUu) ? 1u : 0u;
}

ROCKETGPU_HD bool BallDeactivatesAtTickStart(const BallState& ball, uint8_t sleeping, const WorldSolverScratch* scratch) {
    if (sleeping != 0 || !(ball.deactivation_time > kBulletDeactivationTime)) {
        return false;
    }
    bool car_in_island = false;
    for (int i = 0; scratch != nullptr && i < scratch->bp.bp_pair_count; ++i) {
        car_in_island = car_in_island || BulletPairKeyKind(scratch->bp.bp_pairs[i]) == 1;
    }
    return !car_in_island;
}

ROCKETGPU_HD bool ResolveBallActivationAtTickStart(
    BallState& ball, const MutatorConfig& mutator, const WorldSolverScratch* scratch) {
    ball.sleeping = BallWakeStateAtTickStart(ball, mutator);
    if (BallDeactivatesAtTickStart(ball, ball.sleeping, scratch)) {
        ball.sleeping = 1;
        SetBallLinearVelocityBt(ball, Vec3(0.0f, 0.0f, 0.0f));
        ball.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
    }
    return ball.sleeping == 1;
}

ROCKETGPU_HD bool SettleBitsEqual(float a, float b) {
    return GlibcFloatBits(a) == GlibcFloatBits(b);
}

ROCKETGPU_HD bool SettleBitsEqual(const Vec3& a, const Vec3& b) {
    return SettleBitsEqual(a.x, b.x) & SettleBitsEqual(a.y, b.y) & SettleBitsEqual(a.z, b.z);
}

ROCKETGPU_HD bool BallSettleStateEqual(const BallState& a, const BallState& b) {
    static_assert(sizeof(BallState) == 288, "BallState changed: compare the new fields in BallSettleStateEqual");
    return SettleBitsEqual(a.pos, b.pos) &
           SettleBitsEqual(a.vel, b.vel) &
           SettleBitsEqual(a.ang_vel, b.ang_vel) &
           SettleBitsEqual(a.pos_bt, b.pos_bt) &
           SettleBitsEqual(a.vel_bt, b.vel_bt) &
           SettleBitsEqual(a.rot.w, b.rot.w) & SettleBitsEqual(a.rot.x, b.rot.x) &
           SettleBitsEqual(a.rot.y, b.rot.y) & SettleBitsEqual(a.rot.z, b.rot.z) &
           SettleBitsEqual(a.basis_forward, b.basis_forward) &
           SettleBitsEqual(a.basis_right, b.basis_right) &
           SettleBitsEqual(a.basis_up, b.basis_up) &
           a.basis_authoritative == b.basis_authoritative &
           SettleBitsEqual(a.inertia_basis_forward, b.inertia_basis_forward) &
           SettleBitsEqual(a.inertia_basis_right, b.inertia_basis_right) &
           SettleBitsEqual(a.inertia_basis_up, b.inertia_basis_up) &
           a.inertia_basis_valid == b.inertia_basis_valid &
           SettleBitsEqual(a.restored_inv_inertia_bt[0], b.restored_inv_inertia_bt[0]) &
           SettleBitsEqual(a.restored_inv_inertia_bt[1], b.restored_inv_inertia_bt[1]) &
           SettleBitsEqual(a.restored_inv_inertia_bt[2], b.restored_inv_inertia_bt[2]) &
           a.restored_inv_inertia_valid == b.restored_inv_inertia_valid &
           SettleBitsEqual(a.vel_impulse_cache, b.vel_impulse_cache) &
           SettleBitsEqual(a.vel_impulse_cache_bt, b.vel_impulse_cache_bt) &
           a.tick_count_when_extra_impulse_applied == b.tick_count_when_extra_impulse_applied &
           SettleBitsEqual(a.hs_info.y_target_dir, b.hs_info.y_target_dir) &
           SettleBitsEqual(a.hs_info.cur_target_speed, b.hs_info.cur_target_speed) &
           SettleBitsEqual(a.hs_info.time_since_hit, b.hs_info.time_since_hit) &
           a.ds_info.charge_level == b.ds_info.charge_level &
           SettleBitsEqual(a.ds_info.accumulated_hit_force, b.ds_info.accumulated_hit_force) &
           SettleBitsEqual(a.ds_info.y_target_dir, b.ds_info.y_target_dir) &
           a.ds_info.has_damaged == b.ds_info.has_damaged &
           a.ds_info.reserved0 == b.ds_info.reserved0 &
           a.ds_info.reserved1 == b.ds_info.reserved1 &
           a.ds_info.reserved2 == b.ds_info.reserved2 &
           a.ds_info.last_damage_tick == b.ds_info.last_damage_tick &
           a.sleeping == b.sleeping;
}

ROCKETGPU_HD bool BallWorldMeshManifoldsEmpty(const WorldSolverScratch& scratch) {
    for (int i = 0; i < kMaxBallWorldMeshPersistentManifolds; ++i) {
        if (((scratch.ball_mesh_maybe_retained >> i) & 1u) != 0u && scratch.ball_world_mesh_manifolds[i].count != 0) {
            return false;
        }
    }
    return true;
}

ROCKETGPU_HD bool BallWorldManifoldsEmpty(const WorldSolverScratch& scratch) {
    for (int i = 0; i < kBallWorldPlaneSlots; ++i) {
        if (scratch.ball_world_plane_manifolds[i].count != 0) {
            return false;
        }
    }
    return BallWorldMeshManifoldsEmpty(scratch);
}

static_assert(kMaxApproxCarCarContactPoints <= kMaxCarPairRowsPerPair, "pair row caps assume <= 4 rows per car pair");

template <typename T>
ROCKETGPU_HD void CopyCarWorldRow(T& dst, const T& src) {
    if (&dst == &src) {
        return;
    }
#if defined(__CUDA_ARCH__)
    memcpy(__builtin_assume_aligned(&dst, 16), __builtin_assume_aligned(&src, 16), sizeof(T));
#else
    memcpy(&dst, &src, sizeof(T));
#endif
}

ROCKETGPU_HD void MoveCarWorldRowsDown(const WorldSolverScratchRows& rows, bool move_split, int split_dst,
                                       int split_src, int split_n, bool move_velocity, int velocity_dst,
                                       int velocity_src, int velocity_n, bool move_friction, int friction_dst,
                                       int friction_src, int friction_n, int normal_shift) {
    const int n_split = move_split ? split_n : 0;
    const int n_velocity = move_velocity ? velocity_n : 0;
    const int n_friction = (move_friction || normal_shift != 0) ? friction_n : 0;
    const int n = n_split > n_velocity ? (n_split > n_friction ? n_split : n_friction)
                                       : (n_velocity > n_friction ? n_velocity : n_friction);
    for (int j = 0; j < n; ++j) {
        if (j < n_split) {
            CopyCarWorldRow(rows.world_split_rows[split_dst + j], rows.world_split_rows[split_src + j]);
        }
        if (j < n_velocity) {
            CopyCarWorldRow(rows.world_velocity_rows[velocity_dst + j], rows.world_velocity_rows[velocity_src + j]);
        }
        if (j < n_friction) {
            CarSharedFrictionContactRow& dst = rows.world_friction_rows[friction_dst + j];
            if (move_friction) {
                CopyCarWorldRow(dst, rows.world_friction_rows[friction_src + j]);
            }
            dst.normal_row_idx -= normal_shift;
        }
    }
}
ROCKETGPU_HD void PackCarWorldRows(
    int count,
    const CarWorldRowsOut* outs,
    const WorldSolverScratchRows& rows,
    int& split_row_count,
    int& velocity_row_count,
    int& friction_row_count,
    int& group_count,
    bool copy_groups = true) {
    static_assert(kMaxCarWorldSplitContactRows < 256 && kMaxCarWorldVelocityContactRows < 256 &&
                      kMaxCarWorldFrictionContactRows < 256 && kMaxCarWorldMeshesPerCar + 4 < 256,
                  "a car's row and group counts fit a byte");
    uint64_t split_counts = 0;
    uint64_t velocity_counts = 0;
    uint64_t friction_counts = 0;
    uint64_t group_counts = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (i < count) {
            split_counts |= static_cast<uint64_t>(outs[i].split_row_count & 0xff) << (8 * i);
            velocity_counts |= static_cast<uint64_t>(outs[i].velocity_row_count & 0xff) << (8 * i);
            friction_counts |= static_cast<uint64_t>(outs[i].friction_row_count & 0xff) << (8 * i);
            if (copy_groups) {
                group_counts |= static_cast<uint64_t>(outs[i].group_count & 0xff) << (8 * i);
            }
        }
    }
    for (int i = 0; i < count; ++i) {
        const CarWorldRowsOut& out = outs[i];
        const int shift = 8 * (i & 7);
        const int out_split_count = i < 8 ? static_cast<int>((split_counts >> shift) & 0xff) : out.split_row_count;
        const int out_velocity_count =
            i < 8 ? static_cast<int>((velocity_counts >> shift) & 0xff) : out.velocity_row_count;
        const int out_friction_count =
            i < 8 ? static_cast<int>((friction_counts >> shift) & 0xff) : out.friction_row_count;
        const int out_group_count =
            copy_groups ? (i < 8 ? static_cast<int>((group_counts >> shift) & 0xff) : out.group_count) : 0;
        const int split_from = i * kMaxCarWorldSplitContactRows;
        const int velocity_shift = i * kMaxCarWorldVelocityContactRows - velocity_row_count;
        const int friction_shift = i * kMaxCarWorldFrictionContactRows - friction_row_count;
        MoveCarWorldRowsDown(rows, split_from != split_row_count, split_row_count, split_from, out_split_count,
                             velocity_shift != 0, velocity_row_count, velocity_row_count + velocity_shift,
                             out_velocity_count, friction_shift != 0, friction_row_count,
                             friction_row_count + friction_shift, out_friction_count, velocity_shift);
        if (copy_groups) {
            SharedContactOrderGroup next_group;
            if (out_group_count > 0) {
                next_group = out.groups[0];
            }
            for (int g = 0; g < out_group_count && group_count < kMaxSharedContactOrderGroups; ++g) {
                SharedContactOrderGroup group = next_group;
                if (g + 1 < out_group_count) {
                    next_group = out.groups[g + 1];
                }
                if (group.normal_row_start >= 0) {
                    group.normal_row_start -= velocity_shift;
                }
                if (group.friction_row_start >= 0) {
                    group.friction_row_start -= friction_shift;
                }
                rows.shared_contact_order_groups[group_count++] = group;
            }
        }
        split_row_count += out_split_count;
        velocity_row_count += out_velocity_count;
        friction_row_count += out_friction_count;
    }
}

enum : int {
    kWorldStepWhole = 0, kWorldStepContacts = 1, kWorldStepRows = 2,
    kWorldStepContactsCarsApart = 3, kWorldStepRowsCarsApart = 4,
    kWorldStepContactsCarsApartPrefix = 5,
    kWorldStepContactsCarsApartInfo = 6
};
ROCKETGPU_HD constexpr bool WorldPartIsPrefix(int part) {
    return part == kWorldStepContactsCarsApartPrefix || part == kWorldStepContactsCarsApartInfo;
}
ROCKETGPU_HD constexpr bool WorldPartIsContacts(int part) {
    return part == kWorldStepContacts || part == kWorldStepContactsCarsApart;
}
ROCKETGPU_HD constexpr bool WorldPartIsRows(int part) {
    return part == kWorldStepRows || part == kWorldStepRowsCarsApart;
}

ROCKETGPU_HD void WorldStepPrestepCar(
    CarState& car, const CarConfig& car_config, const WheelRuntimeState* wheel_runtime_state,
    const MutatorConfig& mutator, float sub_dt, PreContactCar& pre_contact_car,
    Vec3& deferred_linear_delta, Vec3& deferred_angular_delta, const CarFrame* frame = nullptr) {
    const PreContactCar pre_contact = MakePreContactCar(car);
    const ::rocketgpu::Vec3 linear_delta =
        ComputeDeferredLinearVelocityDelta(car, wheel_runtime_state, mutator, sub_dt);
    const bool inactive = car.is_demoed || car.just_respawned;
    const ::rocketgpu::Vec3 ext_force_impulse_bt = inactive
        ? ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f)
        : ComputeOfficialExternalForceImpulseBt(wheel_runtime_state, mutator, sub_dt);
    ::rocketgpu::Vec3 ext_torque(0.0f, 0.0f, 0.0f);
    if (wheel_runtime_state && !inactive) {
        const ::rocketgpu::Vec3& t_bt = wheel_runtime_state->accumulated_torque_bt;
        if (t_bt.x != 0.0f || t_bt.y != 0.0f || t_bt.z != 0.0f) {
            ::rocketgpu::Vec3 inv_r0, inv_r1, inv_r2;
            if (frame != nullptr) {
                inv_r0 = frame->inv_world_row0_bt;
                inv_r1 = frame->inv_world_row1_bt;
                inv_r2 = frame->inv_world_row2_bt;
            } else {
                BuildPreSolveInvInertiaTensorWorldBt(
                    car, *wheel_runtime_state, car_config, mutator.car_mass, inv_r0, inv_r1, inv_r2);
            }
            ext_torque = VecMul3x3(t_bt, inv_r0, inv_r1, inv_r2) * sub_dt;
        }
    }
    const ::rocketgpu::Vec3 angular_delta = ComputeDeferredAngularVelocityDelta(
        car, car_config, mutator.car_mass, wheel_runtime_state, sub_dt);

    pre_contact_car = pre_contact;
    deferred_linear_delta = linear_delta;
    car.ext_force_impulse_bt = ext_force_impulse_bt;
    car.ext_force_impulse_bt_valid = 1;
    car.solve_delta_linear_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    car.solve_delta_angular = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    car.ext_torque_impulse = ext_torque;
    deferred_angular_delta = angular_delta;
}

ROCKETGPU_HD void WorldStepFinishCar(
    CarState& car, const CarConfig& car_config, WheelRuntimeState* wheel_runtime_state, const MutatorConfig& mutator,
    float sub_dt, const PreContactCar& pre_contact_car, const Vec3& deferred_linear_delta,
    const Vec3& deferred_angular_delta) {
    if (pre_contact_car.is_demoed) {
        return;
    }
    ApplyDeferredVelocityDeltas(car, car_config, deferred_linear_delta, deferred_angular_delta, wheel_runtime_state,
                                mutator, sub_dt);
    if (car.just_respawned) {
        return;
    }
    IntegrateCarTransformOnly(car, sub_dt, NativeSseMath(mutator));
}

template <typename MeshLanes = ArenaRayMeshOneLane>
ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE bool PredictiveContactRayForBody(
    int body, const BallState& ball, bool ball_sleeping, const CarState* cars, const PreContactCar* pre_contact_cars,
    const CarConfig* car_configs, float dt, bool native_sse_math,
    const ArenaMeshTriangle* mesh_triangles, int32_t mesh_triangle_count,
    const ArenaMeshBvhQuantization* mesh_bvh_quantization, int32_t mesh_bvh_quantization_count,
    const ArenaMeshBvhMeshInfo* mesh_bvh_meshes, int32_t mesh_bvh_mesh_count,
    const ArenaMeshBvhNode* mesh_bvh_nodes, int32_t mesh_bvh_node_count,
    const ArenaMeshBvhSubtreeInfo* mesh_bvh_subtrees, int32_t mesh_bvh_subtree_count,
    const ArenaMeshLeafGrid* leaf_grid, Vec3& normal_out, float& distance_bt_out,
    const MeshLanes& mesh_lanes = MeshLanes{},
    const ArenaMeshTriangle* local_triangles = nullptr) {
    constexpr float kCcdMotionThresholdBt = 0.02f;
    constexpr float kRayBackoffBt = 0.04f;
    const bool have_bvh_free_box =
        mesh_triangles != nullptr && mesh_triangle_count > 0 &&
        mesh_bvh_quantization != nullptr && mesh_bvh_quantization_count > 0 &&
        mesh_bvh_nodes != nullptr && mesh_bvh_node_count > 0 && mesh_bvh_meshes != nullptr && mesh_bvh_mesh_count > 0;
    Vec3 cur, to;
    if (body == 0) {
        if (ball_sleeping) {
            return false;
        }
        cur = ball.pos_bt;
        to = ball.pos_bt + ball.vel_bt * dt;
    } else {
        const CarState& car = cars[body - 1];
        if (pre_contact_cars[body - 1].is_demoed) {
            return false;
        }
        Vec3 ax(1.0f, 0.0f, 0.0f), ay(0.0f, 1.0f, 0.0f), az(0.0f, 0.0f, 1.0f);
        CarBasisColumns(car, ax, ay, az);
        const Vec3 child = car_configs[body - 1].hitbox_offset * (1.0f / 50.0f);
        cur = Vec3(car.pos_bt.x + (ax.x * child.x + ay.x * child.y + az.x * child.z),
                   car.pos_bt.y + (ax.y * child.x + ay.y * child.y + az.y * child.z),
                   car.pos_bt.z + (ax.z * child.x + ay.z * child.y + az.z * child.z));
        if (have_bvh_free_box && car.basis_authoritative) {
            const float margin = kRayBackoffBt + 1e-3f;
            const float fx = Min(mesh_bvh_meshes[0].free_half_x, 81.8f) - margin;
            const float fy = Min(mesh_bvh_meshes[0].free_half_y, 199.9f) - margin;
            const float ang_l1 = (Abs(car.ang_vel.x) + Abs(car.ang_vel.y) + Abs(car.ang_vel.z)) * dt;
            if (ang_l1 <= 0.5f && cur.x > -fx && cur.x < fx && cur.y > -fy && cur.y < fy &&
                cur.z > margin && cur.z < 40.75f) {
                const float child_l1 = Abs(child.x) + Abs(child.y) + Abs(child.z);
                const float rb = (ang_l1 + 1e-3f) * child_l1 + 1e-3f;
                const Vec3 to_approx = cur + car.vel_bt * dt;
                if (to_approx.x > -fx + rb && to_approx.x < fx - rb &&
                    to_approx.y > -fy + rb && to_approx.y < fy - rb &&
                    to_approx.z > rb && to_approx.z < 40.8f - rb) {
                    return false;
                }
            }
        }
        Quat rot = car.rot;
        Vec3 fwd = car.basis_forward, right = car.basis_right, up = car.basis_up;
        if (car.basis_authoritative && !native_sse_math) {
            GameIntegrateOrientationBasisColumns(fwd, right, up, rot, car.ang_vel, dt);
        } else if (car.basis_authoritative) {
            IntegrateOrientationBasisColumns(fwd, right, up, rot, car.ang_vel, dt, native_sse_math);
        } else {
            IntegrateOrientation(rot, car.ang_vel, dt);
            GetBasisVectors(rot, fwd, right, up);
        }
        const Vec3 pred_origin = car.pos_bt + car.vel_bt * dt;
        to = Vec3(pred_origin.x + (fwd.x * child.x + right.x * child.y + up.x * child.z),
                  pred_origin.y + (fwd.y * child.x + right.y * child.y + up.y * child.z),
                  pred_origin.z + (fwd.z * child.x + right.z * child.y + up.z * child.z));
    }
    if (have_bvh_free_box) {
        const float margin = kRayBackoffBt + 1e-3f;
        const float fx = Min(mesh_bvh_meshes[0].free_half_x, 81.8f) - margin;
        const float fy = Min(mesh_bvh_meshes[0].free_half_y, 199.9f) - margin;
        if (cur.x > -fx && cur.x < fx && to.x > -fx && to.x < fx &&
            cur.y > -fy && cur.y < fy && to.y > -fy && to.y < fy &&
            cur.z > margin && to.z > 0.0f && cur.z < 40.75f && to.z < 40.8f) {
            return false;
        }
    }
    const Vec3 m(to.x - cur.x, to.y - cur.y, to.z - cur.z);
    const float m2 = m.x * m.x + m.y * m.y + m.z * m.z;
    if (!(m2 > kCcdMotionThresholdBt * kCcdMotionThresholdBt)) {
        return false;
    }
    const float inv_len = 1.0f / SqrtScalar(m2);
    const Vec3 from(cur.x - inv_len * m.x * kRayBackoffBt, cur.y - inv_len * m.y * kRayBackoffBt,
                    cur.z - inv_len * m.z * kRayBackoffBt);
    const Vec3 delta(to.x - from.x, to.y - from.y, to.z - from.z);
    const ArenaRayHit hit = CastArenaRayDeltaBt(
        ArenaRayHit{}, from, delta, Length(delta) * 50.0f, mesh_triangles, mesh_triangle_count,
        mesh_bvh_quantization, mesh_bvh_quantization_count, mesh_bvh_meshes, mesh_bvh_mesh_count, mesh_bvh_nodes,
        mesh_bvh_node_count, mesh_bvh_subtrees, mesh_bvh_subtree_count, nullptr, 0, mesh_lanes, leaf_grid,
        local_triangles);
    if (!mesh_lanes.Leader() || !hit.hit || !(hit.ray_fraction < 1.0f)) {
        return false;
    }
    Vec3 n = hit.normal;
    if (hit.kind == ROCKETGPU_ARENA_RAY_HIT_ARENA_MESH && hit.tri_index >= 0 && hit.tri_index < mesh_triangle_count) {
        const ArenaMeshTriangle& tri = (local_triangles != nullptr ? local_triangles : mesh_triangles)[hit.tri_index];
        const float origin_y = local_triangles != nullptr ? static_cast<float>(tri.cal_origin_sign) * kGoalMeshOriginYBt : 0.0f;
        const Vec3 local_from(from.x, from.y - origin_y, from.z);
        n = Cross(tri.b - tri.a, tri.c - tri.a);
        const bool behind = Dot(n, local_from) - Dot(tri.a, n) <= 0.0f;
        const float inv_len = 1.0f / SqrtScalar((n.y * n.y + n.x * n.x) + n.z * n.z);
        n = Vec3(n.x * inv_len, n.y * inv_len, n.z * inv_len);
        if (behind) {
            n = -n;
        }
    }
    const float f = hit.ray_fraction;
    const Vec3 d(delta.x * f, delta.y * f, delta.z * f);
    const float d_len = SqrtScalar(d.x * d.x + d.y * d.y + d.z * d.z);
    const float s = d_len > kRayBackoffBt ? d_len - kRayBackoffBt : 0.0f;
    const float inv_d = 1.0f / d_len;
    const Vec3 d_short(inv_d * d.x * s, inv_d * d.y * s, inv_d * d.z * s);
    normal_out = n;
    distance_bt_out = ((delta.y - d_short.y) * n.y + (delta.x - d_short.x) * n.x) + (delta.z - d_short.z) * n.z;
    return true;
}

ROCKETGPU_HD ROCKETGPU_NOINLINE void ComputePredictiveContactHits(
    const BallState& ball, bool ball_sleeping, const CarState* cars, const PreContactCar* pre_contact_cars,
    const CarConfig* car_configs, int count, float dt, bool native_sse_math,
    const ArenaMeshTriangle* mesh_triangles, int32_t mesh_triangle_count,
    const ArenaMeshBvhQuantization* mesh_bvh_quantization, int32_t mesh_bvh_quantization_count,
    const ArenaMeshBvhMeshInfo* mesh_bvh_meshes, int32_t mesh_bvh_mesh_count,
    const ArenaMeshBvhNode* mesh_bvh_nodes, int32_t mesh_bvh_node_count,
    const ArenaMeshBvhSubtreeInfo* mesh_bvh_subtrees, int32_t mesh_bvh_subtree_count,
    const ArenaMeshLeafGrid* leaf_grid, WorldSolverScratch& scratch,
    const ArenaMeshTriangle* local_triangles = nullptr) {
    scratch.predictive_hit_mask = 0;
    const int body_count = Min(count + 1, 9);
    for (int body = 0; body < body_count; ++body) {
        Vec3 n;
        float distance_bt;
        if (PredictiveContactRayForBody(body, ball, ball_sleeping, cars, pre_contact_cars, car_configs, dt,
                                        native_sse_math, mesh_triangles, mesh_triangle_count, mesh_bvh_quantization,
                                        mesh_bvh_quantization_count, mesh_bvh_meshes, mesh_bvh_mesh_count,
                                        mesh_bvh_nodes, mesh_bvh_node_count, mesh_bvh_subtrees,
                                        mesh_bvh_subtree_count, leaf_grid, n, distance_bt,
                                        ArenaRayMeshOneLane{}, local_triangles)) {
            scratch.predictive_hit_mask = static_cast<uint16_t>(scratch.predictive_hit_mask | (1u << body));
            scratch.predictive_normal[body] = n;
            scratch.predictive_distance_bt[body] = distance_bt;
        }
    }
}

ROCKETGPU_HD ROCKETGPU_NOINLINE void AppendPredictiveContactRows(
    const BallState& ball, const CarState* cars, const MutatorConfig& mutator, const WorldSolverScratch& scratch,
    float dt, const Vec3* deferred_linear_deltas, const Vec3* deferred_angular_deltas,
    const Vec3& ball_external_linear_delta, const Vec3* ball_solver_linear_velocity_bt,
    EnvStridedArray<CarSharedSplitContactRow> world_split_rows, int& world_split_row_count,
    EnvStridedArray<CarSharedVelocityContactRow> world_velocity_rows, int& world_velocity_row_count,
    EnvStridedArray<CarSharedFrictionContactRow> world_friction_rows, int& world_friction_row_count,
    EnvStridedArray<BallSplitContactRow> ball_split_rows, int& ball_split_row_count,
    EnvStridedArray<BallVelocityContactRow> ball_normal_rows, int& ball_normal_row_count,
    EnvStridedArray<BallFrictionContactRow> ball_friction_rows, int& ball_friction_row_count) {
    const BulletBroadphaseHistory& bp = scratch.bp;
    const Vec3 zero(0.0f, 0.0f, 0.0f);
    for (int k = 0; k < bp.bp_predictive_count; ++k) {
        const BulletManifoldEntry key = UnpackBulletPairKey(bp.bp_predictive[k]);
        const int body = key.kind == 0 ? 0 : key.body_a + 1;
        if (((scratch.predictive_hit_mask >> body) & 1u) == 0) {
            continue;
        }
        const Vec3 n = scratch.predictive_normal[body];
        const float distance_bt = scratch.predictive_distance_bt[body];
        const float penetration_bt = -distance_bt;
        if (body == 0) {
            AppendBallStaticSplitRow(ball_split_rows, ball_split_row_count, penetration_bt * 50.0f, n, zero,
                                     mutator, dt, &zero, &penetration_bt, &zero,
                                     static_cast<int16_t>(kBulletPredictiveStaticId));
            const Vec3 rhs_ball_linear = ball.vel + ball_external_linear_delta;
            const Vec3 bt_linear = ball_solver_linear_velocity_bt ? *ball_solver_linear_velocity_bt
                                                                  : BallStoredLinearVelocityToBt(ball.vel);
            const Vec3 bt_tangent_basis_rel_vel = bt_linear + UuVecToBtDiv(ball_external_linear_delta);
            const int before = ball_normal_row_count;
            AppendBallVelocityContactRows(
                ball_normal_rows, ball_normal_row_count, ball_friction_rows, ball_friction_row_count, n,
                0.0f, mutator.ball_friction, 1.0f / mutator.ball_mass, zero, zero, -1, 0.0f, zero, zero, zero, zero,
                ball.vel, ball.ang_vel, zero, zero,
                ball_external_linear_delta, zero, zero, zero, rhs_ball_linear, rhs_ball_linear, &zero, nullptr,
                ball_solver_linear_velocity_bt, &zero, nullptr, &zero, nullptr, &bt_tangent_basis_rel_vel, nullptr,
                nullptr, nullptr, nullptr, true);
            if (ball_normal_row_count > before) {
                ball_normal_rows[before].predictive = 1;
            }
            continue;
        }
        const int i = body - 1;
        const CarState& car = cars[i];
        const float inv_mass = 1.0f / mutator.car_mass;
        const int source_tag = 100 + kBulletPredictiveStaticId;
        CarWorldSplitContactRow split_row;
        InitializeCarWorldSplitContactRow(split_row, penetration_bt * 50.0f, n, inv_mass, zero, zero, dt, &zero, &zero,
                                          &penetration_bt);
        split_row.source_tag = source_tag;
        AppendCarWorldToSharedSplitRows(world_split_rows, world_split_row_count, i, &split_row, 1);

        CarWorldVelocityTraceInputs trace{car.vel, car.ang_vel, deferred_linear_deltas[i], deferred_angular_deltas[i],
                                          car.pos};
        trace.rel_contact_bt = zero;
        trace.has_rel_contact_bt = 1;
        trace.linear_velocity_bt = car.vel_bt;
        trace.car_pos_bt = car.pos_bt;
        trace.external_force_impulse_bt = car.ext_force_impulse_bt;
        trace.has_bt_inputs = car.ext_force_impulse_bt_valid;
        trace.external_torque_impulse = car.ext_torque_impulse;
        trace.restitution_no_velocity_threshold = true;
        const Vec3 base_rel_vel = VelocityAtPoint(car, zero, deferred_linear_deltas[i], deferred_angular_deltas[i]);
        const Vec3 friction_base_rel_vel = VelocityAtPoint(car, zero, deferred_linear_deltas[i], zero);
        Vec3 tangent(0.0f, 0.0f, 0.0f);
        if (car.ext_force_impulse_bt_valid && car.basis_authoritative) {
            const Vec3 vel_bt = car.vel_bt + car.ext_force_impulse_bt;
            const Vec3 lateral_bt = vel_bt - n * Dot(n, vel_bt);
            const float lat_rel_vel = LengthSq(lateral_bt);
            tangent = lat_rel_vel > 1.1920928955078125e-7f ? lateral_bt * (1.0f / SqrtScalar(lat_rel_vel))
                                                            : BulletPlaneSpace1First(n);
        } else {
            tangent = base_rel_vel - n * Dot(base_rel_vel, n);
            const float tangent_len = Length(tangent);
            tangent = tangent_len > 1e-6f ? (tangent / tangent_len) : OrthonormalTangent(n);
        }
        CarWorldVelocityContactRow normal_row[1];
        CarWorldFrictionContactRow friction_row[1];
        int normal_count = 0, friction_count = 0;
        AppendCarWorldVelocityContactRows(normal_row, normal_count, friction_row, friction_count, n, 0.0f,
                                          mutator.arena_friction, inv_mass, zero, zero, zero,
                                          VelocityAtPoint(car, zero), base_rel_vel,
                                          base_rel_vel, friction_base_rel_vel, &trace, &zero, &zero, &tangent);
        normal_row[0].source_tag = source_tag;
        friction_row[0].source_tag = source_tag;
        AppendCarWorldToSharedVelocityRows(world_velocity_rows, world_velocity_row_count, world_friction_rows,
                                           world_friction_row_count, i, normal_row, normal_count, friction_row,
                                           friction_count);
    }
}

ROCKETGPU_HD bool WorldStepPrefixRuns(const BallState& ball, const MutatorConfig& mutator, uint8_t enable_car_car_collisions,
                                      const ArenaBroadphaseGrid& grid, const ArenaMeshBvhMeshInfo* mesh_bvh_meshes,
                                      int32_t mesh_bvh_mesh_count) {
    return enable_car_car_collisions && mutator.game_mode != static_cast<uint8_t>(Mode::Heatseeker) && grid.valid() &&
           mesh_bvh_meshes != nullptr && mesh_bvh_mesh_count > 0 &&
           (BallWakeStateAtTickStart(ball, mutator) != 0 || !(ball.deactivation_time > kBulletDeactivationTime));
}

ROCKETGPU_HD ROCKETGPU_NOINLINE void SortBulletManifoldListByIsland(
    int built,
    const uint32_t* packed,
    const int8_t* manifold_body_islands,
    EnvStridedArray<BulletManifoldEntry> bullet_manifolds) {
    int8_t order[kMaxBulletManifoldEntries];
    int8_t island_ids[kMaxBulletManifoldEntries];
    for (int i = 0; i < built; ++i) {
        order[i] = static_cast<int8_t>(i);
        const BulletManifoldEntry key = UnpackBulletPairKey(packed[i]);
        island_ids[i] = manifold_body_islands[key.kind <= 1 ? 0 : key.body_a + 1];
    }
    QuickSortSharedContactOrderIndices(order, island_ids, 0, built - 1);

    for (int i = 0; i < built; ++i) {
        if (order[i] != i) {
            BulletManifoldEntry entry = UnpackBulletPairKey(packed[order[i]]);
            entry.island_id = island_ids[order[i]];
            bullet_manifolds[i] = entry;
        }
    }
}

ROCKETGPU_HD ROCKETGPU_NOINLINE void RecordBallCarInfoExtraCars(
    int env_idx,
    int base,
    int count,
    const Team* teams,
    CarState* cars,
    WorldSolverScratch* solver_scratch,
    const CarConfig* car_configs,
    BallState& ball,
    const MutatorConfig& mutator,
    uint64_t current_tick,
    float sub_dt,
    const Vec3& cal_ball_start_vel) {
    uint32_t ball_car_info_mask = 0;
    for (int i = kMaxBallCarRetainedManifolds; i < count; ++i) {
        if (BallCarCollisionInfoMayOverlap(ball, cars[base + i], car_configs[base + i], mutator, sub_dt, env_idx)) {
            ball_car_info_mask |= 1u << i;
        }
    }
    for (int i = kMaxBallCarRetainedManifolds; i < count; ++i) {
        if (!(ball_car_info_mask & (1u << i))) {
            continue;
        }
        RecordBallCarCollisionInfoIfOverlapping(
            ball,
            cars[base + i],
            static_cast<uint32_t>(base + i + 1),
            cars[base + i],
            car_configs[base + i],
            mutator,
            current_tick,
            ball,
            &solver_scratch->ball_car_gjk_epa_scratch,
            sub_dt,
            env_idx,
            static_cast<int8_t>(teams[base + i]),
            (i < 8) ? &solver_scratch->cal_start_vel[i] : nullptr,
            (&cal_ball_start_vel));
    }
}

ROCKETGPU_HD void StepEnvWorldInfo(
    int env_idx,
    const SimConfig& cfg,
    const Team* teams,
    const int32_t* env_car_count,
    CarState* cars,
    WorldSolverScratch* solver_scratch,
    const CarConfig* car_configs,
    BallState* balls,
    const EnvState* env_states) {
    if (solver_scratch == nullptr || DebugWorldSubsteps() != 1) {
        return;
    }
    WorldSolverScratch& scratch = solver_scratch[env_idx];
    BallState& ball = balls[env_idx];
    const ArenaBroadphaseGrid grid = MakeArenaBroadphaseGrid(cfg);
    if (!WorldStepPrefixRuns(ball, cfg.mutator, cfg.enable_car_car_collisions, grid, cfg.arena_mesh_bvh_meshes,
                             cfg.arena_mesh_bvh_mesh_count)) {
        return;
    }
    const int base = env_idx * cfg.max_cars_per_env;
    const int count = env_car_count[env_idx];
    scratch.hit_ball_pos = ball.pos;
    for (int i = 0; i < count && i < kMaxBallCarRetainedManifolds; ++i) {
        scratch.hit_car_pos[i] = cars[base + i].pos;
        scratch.hit_car_rot[i] = cars[base + i].rot;
    }
    if (cfg.max_cars_per_env > kMaxBallCarRetainedManifolds && count > kMaxBallCarRetainedManifolds) {
        RecordBallCarInfoExtraCars(env_idx, base, count, teams, cars, &scratch, car_configs, ball, cfg.mutator,
                                   static_cast<uint64_t>(env_states[env_idx].tick), cfg.tick_dt,
                                   scratch.cal_ball_start_vel);
    }
    scratch.w3_info_done = 1;
}

ROCKETGPU_HD void StepEnvWorldPrefix(
    int env_idx,
    const SimConfig& cfg,
    const int32_t* env_car_count,
    const CarState* cars,
    WorldSolverScratch* solver_scratch,
    const WorldSolverScratchRowPool& row_pool,
    const CarConfig* car_configs,
    const BallState* balls) {
    if (solver_scratch == nullptr) {
        return;
    }
    WorldSolverScratch& scratch = solver_scratch[env_idx];
    const BallState& ball = balls[env_idx];
    const ArenaBroadphaseGrid grid = MakeArenaBroadphaseGrid(cfg);
    if (DebugWorldSubsteps() == 1 &&
        WorldStepPrefixRuns(ball, cfg.mutator, cfg.enable_car_car_collisions, grid, cfg.arena_mesh_bvh_meshes,
                            cfg.arena_mesh_bvh_mesh_count)) {
        const int base = env_idx * cfg.max_cars_per_env;
        const int count = env_car_count[env_idx];
        BuildBulletManifoldList<1>(ball, count, cars + base, scratch.pre_contact_cars, car_configs + base,
                                   cfg.mutator.ball_radius, cfg.tick_dt, grid, cfg.arena_mesh_bvh_meshes,
                                   cfg.arena_mesh_bvh_mesh_count, row_pool.view(env_idx).bullet_manifolds,
                                   kMaxBulletManifoldEntries, cfg.mutator.arena_collision != 0, &scratch,
                                   NativeSseMath(cfg.mutator), nullptr, scratch.w3_proxies_done != 0,
                                   cfg.mutator.game_mode == static_cast<uint8_t>(Mode::Soccar));
        scratch.w3_prefix_done = 1;
    }
    scratch.w3_proxies_done = 0;
}

#if defined(__CUDA_ARCH__) && defined(_MSC_VER)
#define ROCKETGPU_WORLD_STEP_INLINE __forceinline__
#elif defined(__CUDA_ARCH__)
#define ROCKETGPU_WORLD_STEP_INLINE __attribute__((always_inline))
#else
#define ROCKETGPU_WORLD_STEP_INLINE
#endif
ROCKETGPU_HD ROCKETGPU_WORLD_STEP_INLINE void WorldStepSoccar(
    int env_idx,
    int base,
    int count,
    Team* teams,
    CarState* cars,
    WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch,
    const WorldSolverScratchRows& rows,
    CarConfig* car_configs,
    BallState& ball,
    const MutatorConfig& mutator,
    uint8_t enable_car_car_collisions,
    uint8_t enable_demos,
    uint64_t current_tick,
    float dt,
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
    const ArenaBroadphaseGrid& broadphase_grid = ArenaBroadphaseGrid{},
    int world_part = kWorldStepWhole,
    uint8_t cars_apart_mask = 0,
    bool register_iterations = false) {
    const int solver_passes = WorldSolverPasses();
    const int world_substeps = DebugWorldSubsteps();
    const float sub_dt = dt / static_cast<float>(world_substeps);

    EnvStridedArray<CarSharedSplitContactRow> const pair_split_rows = rows.pair_split_rows;
    EnvStridedArray<CarSharedSplitContactRow> const world_split_rows = rows.world_split_rows;
    EnvStridedArray<CarSharedVelocityContactRow> const pair_velocity_rows = rows.pair_velocity_rows;
    EnvStridedArray<CarSharedFrictionContactRow> const pair_friction_rows = rows.pair_friction_rows;
    EnvStridedArray<CarSharedVelocityContactRow> const world_velocity_rows = rows.world_velocity_rows;
    EnvStridedArray<CarSharedFrictionContactRow> const world_friction_rows = rows.world_friction_rows;
    PreContactCar* const pre_contact_cars = solver_scratch->pre_contact_cars;
    Vec3* const deferred_linear_deltas = solver_scratch->deferred_linear_deltas;
    Vec3* const deferred_angular_deltas = solver_scratch->deferred_angular_deltas;
    EnvStridedArray<BallSplitContactRow> const split_rows = rows.ball_split_rows;
    EnvStridedArray<BallVelocityContactRow> const normal_rows = rows.ball_velocity_rows;
    EnvStridedArray<BallFrictionContactRow> const friction_rows = rows.ball_friction_rows;
    const CarCarPairCacheView pair_contact_cache = rows.pair_contact_cache;

    const auto should_stop_after_stage = [&](uint8_t stage) {
        return 0 != 0 && 0 <= stage;
    };
    const auto should_stop_after_world_car = [&](int local_car_index) {
        return false;
    };
    const auto record_ball_car_info = [&](const Vec3& cal_ball_start_vel) {
        solver_scratch->hit_ball_pos = ball.pos;
        for (int i = 0; i < count && i < kMaxBallCarRetainedManifolds; ++i) {
            solver_scratch->hit_car_pos[i] = cars[base + i].pos;
            solver_scratch->hit_car_rot[i] = cars[base + i].rot;
        }
        if (count > kMaxBallCarRetainedManifolds) {
            RecordBallCarInfoExtraCars(env_idx, base, count, teams, cars, solver_scratch, car_configs, ball, mutator,
                                       current_tick, sub_dt, cal_ball_start_vel);
        }
    };
    if (WorldPartIsPrefix(world_part)) {
        if (world_part == kWorldStepContactsCarsApartInfo) {
            if (world_substeps == 1 && solver_scratch != nullptr &&
                WorldStepPrefixRuns(ball, mutator, enable_car_car_collisions, broadphase_grid, mesh_bvh_meshes,
                                    mesh_bvh_mesh_count)) {
                record_ball_car_info(solver_scratch->cal_ball_start_vel);
                solver_scratch->w3_info_done = 1;
            }
            return;
        }
        if (world_substeps == 1 && solver_scratch != nullptr &&
            WorldStepPrefixRuns(ball, mutator, enable_car_car_collisions, broadphase_grid, mesh_bvh_meshes,
                                mesh_bvh_mesh_count)) {
            BuildBulletManifoldList<1>(ball, count, cars + base, pre_contact_cars, car_configs + base, mutator.ball_radius,
                                       sub_dt, broadphase_grid, mesh_bvh_meshes, mesh_bvh_mesh_count,
                                       rows.bullet_manifolds, kMaxBulletManifoldEntries, mutator.arena_collision != 0,
                                       solver_scratch, NativeSseMath(mutator), nullptr,
                                       solver_scratch->w3_proxies_done != 0,
                                       mutator.game_mode == static_cast<uint8_t>(Mode::Soccar));
            solver_scratch->w3_prefix_done = 1;
        }
        if (solver_scratch != nullptr) {
            solver_scratch->w3_proxies_done = 0;
        }
        return;
    }
    for (int substep = 0; substep < world_substeps; ++substep) {
        WorldStepCarry carry;
        if (WorldPartIsRows(world_part)) {
            carry = solver_scratch->world_step_carry;
        } else {
            carry.ball_vehicle_woke = (ball.sleeping & 4u) != 0;
            ball.sleeping &= 3u;
            carry.ball_sleeping = ResolveBallActivationAtTickStart(ball, mutator, solver_scratch);
            if (carry.ball_vehicle_woke) {
                ball.deactivation_time = 0.0f;
            }
        }
        int& pair_split_row_count = carry.pair_split_row_count;
        int world_split_row_count = 0;
        int& pair_velocity_row_count = carry.pair_velocity_row_count;
        int& pair_friction_row_count = carry.pair_friction_row_count;
        int world_velocity_row_count = 0;
        int world_friction_row_count = 0;
        int& split_row_count = carry.split_row_count;
        int& normal_row_count = carry.normal_row_count;
        int& friction_row_count = carry.friction_row_count;
        int& sorted_manifold_count = carry.sorted_manifold_count;
        Vec3& ball_solver_linear_velocity_bt = carry.ball_solver_linear_velocity_bt;
        const bool ball_sleeping = carry.ball_sleeping;
        bool& ball_awake_after_contact_rows = carry.ball_awake_after_contact_rows;
        bool& ball_settled_skip = carry.ball_settled_skip;
        bool& ball_settled_verify = carry.ball_settled_verify;
        const Vec3 sleeping_start_ball_external_linear_delta = ball_sleeping
            ? Vec3(0.0f, 0.0f, 0.0f)
            : Vec3(0.0f, 0.0f, mutator.gravity_z * sub_dt);
        const bool w3_prefix_done = world_part == kWorldStepContactsCarsApart && solver_scratch->w3_prefix_done != 0;
        const bool w3_info_done = world_part == kWorldStepContactsCarsApart && solver_scratch->w3_info_done != 0;
        const bool w3_proxies_done = world_part == kWorldStepContactsCarsApart && solver_scratch->w3_proxies_done != 0;
        if (!WorldPartIsRows(world_part)) {
            solver_scratch->w3_proxies_done = 0;
            solver_scratch->w3_prefix_done = 0;
            solver_scratch->w3_info_done = 0;
            solver_scratch->shared_contact_order_group_count = 0;
            if (solver_scratch->retained_reset_pending) {
                const bool car_world = world_part != kWorldStepContactsCarsApart;
                ClearRetainedContactManifolds(rows, car_world);
                for (int k = 0; car_world && k < 8; ++k) {
                    solver_scratch->car_world_reset[k] = 0;
                }
                solver_scratch->retained_reset_pending = 0;
            }

            const int pair_cache_cars = Min(Max(count, static_cast<int>(solver_scratch->pair_cache_cars)), pair_contact_cache.cars);
            solver_scratch->pair_cache_cars = static_cast<uint8_t>(Min(count, pair_contact_cache.cars));
            uint64_t valid_pairs = 0;
            for (uint32_t maybe = pair_contact_cache ? solver_scratch->pair_cache_maybe_valid : 0; maybe != 0;
                 maybe &= maybe - 1) {
                const int retained_slot = CountTrailingZeros64(maybe);
                int a = 1;
                while (CarCarRetainedSlot(0, a + 1) <= retained_slot) {
                    ++a;
                }
                const int b = retained_slot - CarCarRetainedSlot(0, a);
                if (a < pair_cache_cars && b < pair_cache_cars &&
                    pair_contact_cache.slots[a * pair_contact_cache.cars + b].valid != 0) {
                    valid_pairs |= 1ull << (a * 8 + b);
                }
            }
            solver_scratch->pair_cache_maybe_valid = 0;
            for (; valid_pairs != 0; valid_pairs &= valid_pairs - 1) {
                const int bit = static_cast<int>(CountTrailingZeros64(valid_pairs));
                CarCarPairCachedContact& slot = pair_contact_cache.slots[(bit / 8) * pair_contact_cache.cars + bit % 8];
                slot.valid = 0;
                slot.contact_count = 0;
                for (int k = 0; k < kMaxCarCarCachedPoints; ++k) {
                    slot.contacts[k].valid = 0;
                    slot.contacts[k].tangent_valid = 0;
                }
            }
            for (int i = 0; world_part != kWorldStepContactsCarsApart && i < count; ++i) {
                if ((cars_apart_mask >> i) & 1u) {
                    continue;
                }
                WorldStepPrestepCar(cars[base + i], car_configs[base + i],
                                    wheel_runtime_states ? &wheel_runtime_states[base + i] : nullptr, mutator, sub_dt,
                                    pre_contact_cars[i], deferred_linear_deltas[i], deferred_angular_deltas[i]);
            }

            ball_solver_linear_velocity_bt = ball.vel_bt;
            const Vec3 cal_ball_start_vel = solver_scratch != nullptr ? solver_scratch->cal_ball_start_vel : ball.vel;

            if (mutator.game_mode == static_cast<uint8_t>(Mode::Heatseeker)) {
                BodyDampingClampBall(ball, mutator);
                ball_solver_linear_velocity_bt = ball.vel_bt;
                if (!ball_sleeping) {
                    ball_solver_linear_velocity_bt = BulletDampedLinearVelocityFromBt(ball.vel_bt, mutator, sub_dt);
                    ApplyBulletLinearDampingBall(ball, mutator, sub_dt);
                }
                if (solver_scratch != nullptr) {
                    solver_scratch->ball_predict_vel_bt = ball.vel_bt;
                }
            }

            if (!w3_info_done) {
                record_ball_car_info(cal_ball_start_vel);
            }

            const bool predictive_staged = solver_scratch->predictive_staged != 0;
            solver_scratch->predictive_staged = 0;
            if (!(broadphase_grid.valid() && mesh_bvh_meshes != nullptr && mesh_bvh_mesh_count > 0 &&
                  mutator.arena_collision != 0)) {
                solver_scratch->predictive_hit_mask = 0;
            } else if (!predictive_staged) {
                ComputePredictiveContactHits(
                    ball, ball_sleeping && !carry.ball_vehicle_woke, cars + base, pre_contact_cars, car_configs + base,
                    count, sub_dt,
                    NativeSseMath(mutator), mesh_triangles, mesh_triangle_count, mesh_bvh_quantization,
                    mesh_bvh_quantization_count, mesh_bvh_meshes, mesh_bvh_mesh_count, mesh_bvh_nodes,
                    mesh_bvh_node_count, mesh_bvh_subtrees, mesh_bvh_subtree_count, &broadphase_grid.leaf_grid,
                    *solver_scratch, mutator.cal_arena_local_tris);
            }
            int prebuilt_manifold_count = -1;
            int8_t manifold_body_islands[9];
            if (enable_car_car_collisions) {
                const bool shared_rows_active =
                    pair_velocity_rows && pair_friction_rows;
                const int car_car_passes = shared_rows_active ? 1 : solver_passes;

                if (broadphase_grid.valid() && mesh_bvh_meshes != nullptr && mesh_bvh_mesh_count > 0) {
                    prebuilt_manifold_count = w3_prefix_done
                        ? BuildBulletManifoldList<2>(
                            ball,
                            count,
                            cars + base,
                            pre_contact_cars,
                            car_configs + base,
                            mutator.ball_radius,
                            sub_dt,
                            broadphase_grid,
                            mesh_bvh_meshes,
                            mesh_bvh_mesh_count,
                            rows.bullet_manifolds,
                            kMaxBulletManifoldEntries,
                            mutator.arena_collision != 0,
                            solver_scratch,
                            NativeSseMath(mutator),
                            manifold_body_islands,
                            false,
                            mutator.game_mode == static_cast<uint8_t>(Mode::Soccar),
                            ball_sleeping && !carry.ball_vehicle_woke)
                        : BuildBulletManifoldList<0>(
                            ball,
                            count,
                            cars + base,
                            pre_contact_cars,
                            car_configs + base,
                            mutator.ball_radius,
                            sub_dt,
                            broadphase_grid,
                            mesh_bvh_meshes,
                            mesh_bvh_mesh_count,
                            rows.bullet_manifolds,
                            kMaxBulletManifoldEntries,
                            mutator.arena_collision != 0,
                            solver_scratch,
                            NativeSseMath(mutator),
                            manifold_body_islands,
                            w3_proxies_done,
                            mutator.game_mode == static_cast<uint8_t>(Mode::Soccar),
                            ball_sleeping && !carry.ball_vehicle_woke);
                }
                uint32_t ccm_reject_slots = 0;
                CarCarCcmGateMasks(
                    count, cars + base, car_configs + base, rows.car_car_retained_manifolds, ccm_reject_slots);
                const int ccm_pair_cars = Min(count, 8);
                const uint32_t ccm_all_slots =
                    ccm_pair_cars >= 2 ? (1u << (ccm_pair_cars * (ccm_pair_cars - 1) / 2)) - 1u : 0u;
                const bool ccm_all_rejected = (ccm_reject_slots & ccm_all_slots) == ccm_all_slots;
                int8_t pair_order_i[28];
                int8_t pair_order_j[28];
                int pair_order_count = 0;
                uint32_t pair_listed_mask = 0u;
                for (int e = 0; !ccm_all_rejected && e < prebuilt_manifold_count; ++e) {
                    const BulletManifoldEntry& en = rows.bullet_manifolds[e];
                    if (en.kind != 3) {
                        continue;
                    }
                    const int pi = Min((int)en.body_a, (int)en.body_b);
                    const int pj = Max((int)en.body_a, (int)en.body_b);
                    if (pi < 0 || pj >= count || pj >= 8 || pi == pj) {
                        continue;
                    }
                    const uint32_t slot_bit = 1u << CarCarRetainedSlot(pi, pj);
                    if ((pair_listed_mask & slot_bit) != 0u) {
                        continue;
                    }
                    pair_listed_mask |= slot_bit;
                    if (car_car_passes == 1 && (ccm_reject_slots & slot_bit) != 0u) {
                        continue;
                    }
                    pair_order_i[pair_order_count] = static_cast<int8_t>(pi);
                    pair_order_j[pair_order_count] = static_cast<int8_t>(pj);
                    ++pair_order_count;
                }
                for (int pi = 0; !ccm_all_rejected && pi < count && pi < 8; ++pi) {
                    for (int pj = pi + 1; pj < count && pj < 8; ++pj) {
                        const uint32_t slot_bit = 1u << CarCarRetainedSlot(pi, pj);
                        if ((pair_listed_mask & slot_bit) != 0u) {
                            continue;
                        }
                        if (car_car_passes == 1 && (ccm_reject_slots & slot_bit) != 0u) {
                            continue;
                        }
                        if (pair_order_count < 28) {
                            pair_order_i[pair_order_count] = static_cast<int8_t>(pi);
                            pair_order_j[pair_order_count] = static_cast<int8_t>(pj);
                            ++pair_order_count;
                        }
                    }
                }
                uint32_t resolved_slots = 0;
                for (int pass = 0; pass < car_car_passes; ++pass) {
                    for (int po = 0; po < pair_order_count; ++po) {
                        const int i = pair_order_i[po];
                        const int j = pair_order_j[po];
                        const uint32_t slot_bit = 1u << CarCarRetainedSlot(i, j);
                        if ((ccm_reject_slots & slot_bit) != 0) {
                            if (pair_contact_cache && pair_contact_cache.index(j, i) >= 0 && (resolved_slots & slot_bit) != 0) {
                                ClearCarCarPairCachedContact(pair_contact_cache.slots[pair_contact_cache.index(j, i)]);
                            }
                            continue;
                        }
                        resolved_slots |= slot_bit;
                        {
                            const int pair_velocity_rows_before = pair_velocity_row_count;
                            const int pair_friction_rows_before = pair_friction_row_count;
                            const uint8_t* const body_handle = solver_scratch->bp.sweep.sap_body_handle;
                            const bool i_is_body0 = body_handle[i + 1] <= body_handle[j + 1];
                            const int a = i_is_body0 ? j : i;
                            const int b = i_is_body0 ? i : j;
                            const bool cache_written = ResolveCarCarCollisionImpl<true, true>(
                                cars[base + a], static_cast<uint32_t>(base + a + 1), teams[base + a],
                                cars[base + b], static_cast<uint32_t>(base + b + 1), teams[base + b],
                                car_configs[base + a], car_configs[base + b],
                                mutator,
                                pre_contact_cars[a],
                                pre_contact_cars[b],
                                deferred_linear_deltas[a],
                                deferred_linear_deltas[b],
                                deferred_angular_deltas[a],
                                deferred_angular_deltas[b],
                                sub_dt,
                                enable_demos,
                                a,
                                b,
                                pair_contact_cache,
                                pair_split_rows,
                                &pair_split_row_count,
                                pair_velocity_rows,
                                &pair_velocity_row_count,
                                pair_friction_rows,
                                &pair_friction_row_count,
                                (&solver_scratch->cal_start_vel[a]),
                                (&solver_scratch->cal_start_vel[b]),
                                (i < j && j < 8)
                                    ? &rows.car_car_retained_manifolds[CarCarRetainedSlot(i, j)]
                                    : nullptr,
                                &solver_scratch->ball_car_gjk_epa_scratch);
                            if (cache_written) {
                                solver_scratch->pair_cache_maybe_valid |= 1u << CarCarRetainedSlot(i, j);
                            }
                            if (pair_velocity_row_count > pair_velocity_rows_before) {
                                AppendSharedContactOrderGroup(
                                    rows.shared_contact_order_groups,
                                    solver_scratch->shared_contact_order_group_count,
                                    SharedContactOrderBucket::Pair,
                                    a,
                                    b,
                                    j * 8 + i,
                                    pair_velocity_rows_before,
                                    pair_velocity_row_count - pair_velocity_rows_before,
                                    pair_friction_rows_before,
                                    pair_friction_row_count - pair_friction_rows_before,
                                    j * 100000 + 2000 + i);
                            }
                        }
                        if (pass + 1 < car_car_passes) {
                            CarCarCcmGateMasks(
                                count, cars + base, car_configs + base, rows.car_car_retained_manifolds, ccm_reject_slots);
                        }
                    }
                }
            }
            if (should_stop_after_stage(1)) {
                return;
            }
            uint32_t ball_car_overlap_mask = 0;
            for (int i = 0; i < count && i < kMaxBallCarRetainedManifolds; ++i) {
                if (BallCarRetainedPairOverlaps(
                        ball, cars[base + i], car_configs[base + i], mutator, pre_contact_cars[i].is_demoed == 0)) {
                    ball_car_overlap_mask |= 1u << i;
                }
            }
            for (int i = 0; i < count; ++i) {
                if (i < kMaxBallCarRetainedManifolds) {
                    if (!(ball_car_overlap_mask & (1u << i))) {
                        if (rows.ball_car_retained_manifolds[i].count != 0) {
                            rows.ball_car_retained_manifolds[i].count = 0;
                        }
                        continue;
                    }
                    AppendBallCarContactRowsRetained(
                        ball,
                        cars[base + i],
                        car_configs[base + i],
                        mutator,
                        sleeping_start_ball_external_linear_delta,
                        deferred_linear_deltas[i],
                        deferred_angular_deltas[i],
                        normal_rows,
                        normal_row_count,
                        friction_rows,
                        friction_row_count,
                        split_rows,
                        split_row_count,
                        i,
                        sub_dt,
                        env_idx,
                        &solver_scratch->ball_car_gjk_epa_scratch,
                        pre_contact_cars[i].is_demoed == 0,
                        &ball_solver_linear_velocity_bt,
                        rows.ball_car_retained_manifolds[i]);
                    if (!pre_contact_cars[i].is_demoed && !cars[base + i].just_respawned) {
                        BallCarHitStart start;
                        start.car_pos = solver_scratch->hit_car_pos[i];
                        start.car_rot = solver_scratch->hit_car_rot[i];
                        start.ball_pos = solver_scratch->hit_ball_pos;
                        CommitBallCarHitFromManifold(
                            ball, cars[base + i], static_cast<uint32_t>(base + i + 1), cars[base + i], mutator,
                            current_tick, ball, rows.ball_car_retained_manifolds[i],
                            static_cast<int8_t>(teams[base + i]), &solver_scratch->cal_start_vel[i],
                            &solver_scratch->cal_ball_start_vel, &start);
                    }
                } else
                AppendBallCarContactRows(
                    ball,
                    cars[base + i],
                    car_configs[base + i],
                    mutator,
                    sleeping_start_ball_external_linear_delta,
                    deferred_linear_deltas[i],
                    deferred_angular_deltas[i],
                    normal_rows,
                    normal_row_count,
                    friction_rows,
                    friction_row_count,
                    split_rows,
                    split_row_count,
                    i,
                    sub_dt,
                    env_idx,
                    &solver_scratch->ball_car_gjk_epa_scratch,
                    pre_contact_cars[i].is_demoed == 0,
                    &ball_solver_linear_velocity_bt);
            }
            if (should_stop_after_stage(2)) {
                return;
            }

            if (broadphase_grid.valid() && mesh_bvh_meshes != nullptr && mesh_bvh_mesh_count > 0) {
                const int built = prebuilt_manifold_count >= 0 ? prebuilt_manifold_count : BuildBulletManifoldList(
                    ball,
                    count,
                    cars + base,
                    pre_contact_cars,
                    car_configs + base,
                    mutator.ball_radius,
                    sub_dt,
                    broadphase_grid,
                    mesh_bvh_meshes,
                    mesh_bvh_mesh_count,
                    rows.bullet_manifolds,
                    kMaxBulletManifoldEntries,
                    mutator.arena_collision != 0,
                    solver_scratch,
                    NativeSseMath(mutator),
                    manifold_body_islands,
                    false,
                    mutator.game_mode == static_cast<uint8_t>(Mode::Soccar),
                    ball_sleeping && !carry.ball_vehicle_woke);
                if (built > 1) {
                    SortBulletManifoldListByIsland(
                        built, solver_scratch->bp.bp_manifolds, manifold_body_islands, rows.bullet_manifolds);
                }
                sorted_manifold_count = built;
            }
            solver_scratch->bullet_manifold_count = sorted_manifold_count;

            ball_awake_after_contact_rows =
                !ball_sleeping || carry.ball_vehicle_woke || split_row_count > 0 || normal_row_count > 0 ||
                friction_row_count > 0;

            if (solver_scratch != nullptr) {
                int16_t settle_world_ids[kBallSettledMaxWorldEntries] = {};
                int settle_world_count = 0;
                bool settle_candidate =
                    !ball_sleeping && mutator.arena_collision != 0 &&
                    mutator.game_mode == static_cast<uint8_t>(Mode::Soccar) && !mutator.physics_events.HasNonGoalEvents() &&
                    split_row_count == 0 && normal_row_count == 0 && friction_row_count == 0 &&
                    ball.vel.y == 0.0f && ball.vel.z == 0.0f &&
                    ball.ang_vel.x == 0.0f && ball.ang_vel.y == 0.0f && ball.ang_vel.z == 0.0f;
                for (int e = 0; settle_candidate && e < sorted_manifold_count; ++e) {
                    const BulletManifoldEntry& en = rows.bullet_manifolds[e];
                    if (en.kind != 0) {
                        continue;
                    }
                    if (settle_world_count >= kBallSettledMaxWorldEntries) {
                        settle_candidate = false;
                        break;
                    }
                    settle_world_ids[settle_world_count++] = en.static_id;
                }
                settle_candidate = settle_candidate && settle_world_count > 0 &&
                    BallWorldMeshManifoldsEmpty(*solver_scratch);
                if (!settle_candidate) {
                    solver_scratch->ball_settled = 0;
                } else {
                    bool inputs_match =
                        solver_scratch->ball_settled != 0 &&
                        solver_scratch->ball_settled_mutator_epoch == mutator.ball_settle_epoch &&
                        SettleBitsEqual(solver_scratch->ball_settled_sub_dt, sub_dt) &&
                        SettleBitsEqual(solver_scratch->ball_settled_solver_vel_bt, ball_solver_linear_velocity_bt) &&
                        solver_scratch->ball_settled_world_entry_count == settle_world_count;
                    for (int k = 0; inputs_match && k < settle_world_count; ++k) {
                        inputs_match = solver_scratch->ball_settled_world_entries[k] == settle_world_ids[k];
                    }
                    inputs_match = inputs_match && BallSettleStateEqual(ball, solver_scratch->ball_settled_state);
                    if (inputs_match) {
                        ball_settled_skip = true;
                    } else {
                        solver_scratch->ball_settled = 0;
                        solver_scratch->ball_settled_mutator_epoch = mutator.ball_settle_epoch;
                        solver_scratch->ball_settled_sub_dt = sub_dt;
                        solver_scratch->ball_settled_solver_vel_bt = ball_solver_linear_velocity_bt;
                        solver_scratch->ball_settled_world_entry_count = static_cast<uint8_t>(settle_world_count);
                        for (int k = 0; k < kBallSettledMaxWorldEntries; ++k) {
                            solver_scratch->ball_settled_world_entries[k] = settle_world_ids[k];
                        }
                        solver_scratch->ball_settled_state = ball;
                        ball_settled_verify = true;
                    }
                }
            }

            const bool ball_world_dispatched = !ball_sleeping || carry.ball_vehicle_woke || solver_scratch == nullptr ||
                                               !BallWorldManifoldsEmpty(*solver_scratch);
            if (ball_awake_after_contact_rows && ball_world_dispatched && !ball_settled_skip && mutator.arena_collision) {
                const BallWorldStage* const ball_stage =
                    world_part == kWorldStepContactsCarsApart && solver_scratch != nullptr
                        ? ConsumableBallWorldStage(env_idx, ball, mutator, current_tick, mesh_triangles, mesh_bvh_meshes,
                                                   mesh_bvh_mesh_count)
                        : nullptr;
                AppendBallWorldSpecialContactRows(
                    ball,
                    mutator,
                    sleeping_start_ball_external_linear_delta,
                    solver_scratch,
                    rows.bullet_manifolds,
                    split_rows,
                    split_row_count,
                    sub_dt,
                    normal_rows,
                    normal_row_count,
                    friction_rows,
                    friction_row_count,
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
                    &ball_solver_linear_velocity_bt, env_idx, ball_stage);
                if (mutator.game_mode == static_cast<uint8_t>(Mode::Heatseeker) && solver_scratch) {
                    for (int hs_idx = 0; hs_idx < solver_scratch->hs_world_normal_count; ++hs_idx) {
                        if (HeatseekerOnWorldContact(ball, solver_scratch->hs_world_normals[hs_idx])) {
                            break;
                        }
                    }
                }
            }
            if (should_stop_after_stage(3)) {
                return;
            }
            if (WorldPartIsContacts(world_part)) {
                solver_scratch->world_step_carry = carry;
                return;
            }
        }
        const bool ball_settled_rows_ok = normal_row_count > 0;

        if (WorldPartIsRows(world_part)) {
            for (int i = 0; i < count && i < 8; ++i) {
                if (solver_scratch->car_world_rows_out[i].world_surface) {
                    cars[base + i].RecordWorldSurface(true);
                }
            }
            const bool copy_groups =
                (pair_velocity_row_count > 0 || pair_friction_row_count > 0) && sorted_manifold_count <= 0;
            PackCarWorldRows(count, solver_scratch->car_world_rows_out, rows, world_split_row_count,
                             world_velocity_row_count, world_friction_row_count,
                             solver_scratch->shared_contact_order_group_count, copy_groups);
        }
        for (int i = 0; !WorldPartIsRows(world_part) && i < count; ++i) {
            const int car_idx = base + i;
            if (pre_contact_cars[i].is_demoed || cars[car_idx].just_respawned) {
                if (i < kMaxCarWorldMeshPersistentCars) {
                    for (int plane_idx = 0; plane_idx < kCarWorldPlaneRetainedManifoldsPerCar; ++plane_idx) {
                        rows.car_world_plane_manifolds[i * kCarWorldPlaneRetainedManifoldsPerCar + plane_idx].count = 0;
                    }
                    for (int k = 0; k < kCarWorldMeshRetainedManifoldsPerCar; ++k) {
                        rows.car_world_mesh_manifolds[i * kCarWorldMeshRetainedManifoldsPerCar + k].count = 0;
                    }
                }
                continue;
            }
            ContactAccumulator acc = SolveCarWorldStaticContacts(
                cars[car_idx],
                car_configs[car_idx],
                mutator,
                deferred_linear_deltas[i],
                deferred_angular_deltas[i],
                sub_dt,
                env_idx,
                i,
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
                world_split_rows,
                &world_split_row_count,
                world_velocity_rows,
                &world_velocity_row_count,
                world_friction_rows,
                &world_friction_row_count,
                &solver_scratch->car_triangle_gjk_epa_scratch,
                rows.shared_contact_order_groups,
                &solver_scratch->shared_contact_order_group_count,
                &broadphase_grid,
                CarWorldMeshCandidateStageFor(env_idx, i),
                (i < kMaxCarWorldMeshPersistentCars)
                    ? &rows.car_world_plane_manifolds[i * kCarWorldPlaneRetainedManifoldsPerCar]
                    : nullptr,
                (i < kMaxCarWorldMeshPersistentCars)
                    ? &rows.car_world_mesh_manifolds[i * kCarWorldMeshRetainedManifoldsPerCar]
                    : nullptr,
                rows.car_world_local_rows ? &rows.car_world_local_rows[i] : nullptr,
                0u,
                false,
                nullptr,
                (i < 8 && solver_scratch->car_predict_valid[i]) ? &solver_scratch->car_predict_vel[i] : nullptr,
                (i < 8 && solver_scratch->car_predict_valid[i]) ? &solver_scratch->car_predict_ang_vel[i] : nullptr);
            RecordArenaWorldContact(cars[car_idx], acc);
            if (should_stop_after_world_car(i)) {
                return;
            }
        }
        if (should_stop_after_stage(5)) {
            return;
        }

        if (solver_scratch->predictive_hit_mask != 0) {
            AppendPredictiveContactRows(
                ball, cars + base, mutator, *solver_scratch, sub_dt, deferred_linear_deltas, deferred_angular_deltas,
                sleeping_start_ball_external_linear_delta, &ball_solver_linear_velocity_bt, world_split_rows,
                world_split_row_count, world_velocity_rows, world_velocity_row_count, world_friction_rows,
                world_friction_row_count, split_rows, split_row_count, normal_rows, normal_row_count, friction_rows,
                friction_row_count);
            ApplyPredictiveCarWorldContactNormal(cars + base, count, solver_scratch->predictive_hit_mask,
                                                 world_velocity_rows, world_velocity_row_count);
        }
        SolveUnifiedVelocityContactRows(
            ball,
            count,
            cars + base,
            car_configs + base,
            mutator,
            normal_rows,
            normal_row_count,
            friction_rows,
            friction_row_count,
            world_velocity_rows,
            world_velocity_row_count,
            pair_velocity_rows,
            pair_velocity_row_count,
            world_friction_rows,
            world_friction_row_count,
            pair_friction_rows,
            pair_friction_row_count,
            rows.shared_contact_order_groups,
            solver_scratch->shared_contact_order_group_count,
            &ball_solver_linear_velocity_bt,
            rows.bullet_manifolds,
            sorted_manifold_count,
            register_iterations);
        if (should_stop_after_stage(4)) {
            return;
        }
        if (should_stop_after_stage(6)) {
            return;
        }

        if (pair_velocity_row_count > 0 || pair_friction_row_count > 0) {
            UpdateCarCarPairCacheFromSharedVelocityRows(
                pair_contact_cache,
                pair_velocity_rows,
                pair_velocity_row_count,
                pair_friction_rows,
                pair_friction_row_count);
        }

        SolveUnifiedSplitContactRows(
            ball,
            count,
            cars + base,
            split_rows,
            split_row_count,
            world_split_rows,
            world_split_row_count,
            pair_split_rows,
            pair_split_row_count,
            sub_dt,
            env_idx,
            rows.bullet_manifolds,
            sorted_manifold_count,
            mutator.ball_shape == 0 && !mutator.ball_no_rot,
            NativeSseMath(mutator));
        if (should_stop_after_stage(7)) {
            return;
        }

        if (!ball_sleeping && !ball_settled_skip) {
            SetBallLinearVelocityBt(
                ball,
                ball.vel_bt + UuVecToBtDiv(sleeping_start_ball_external_linear_delta));
        }

        if (ball_settled_skip) {
            const uint64_t ball_ticks = ball.tick_count_since_update;
            const float ball_deactivation_time = ball.deactivation_time;
            const uint16_t ball_constrained_2d = ball.constrained_2d;
            ball = solver_scratch->ball_settled_state;
            ball.tick_count_since_update = ball_ticks;
            ball.deactivation_time = ball_deactivation_time;
            ball.constrained_2d = ball_constrained_2d;
        } else if (ball_awake_after_contact_rows) {
            IntegrateBallTransformOnly(ball, mutator, sub_dt);
        }
        if (ball_awake_after_contact_rows) {
            ball.sleeping = 0;
            const bool below = LengthSq(ball.vel_bt) < kBallLinearSleepingThresholdBt * kBallLinearSleepingThresholdBt &&
                               LengthSq(ball.ang_vel) < kBallAngularSleepingThreshold * kBallAngularSleepingThreshold;
            ball.deactivation_time = below && !ball_sleeping ? ball.deactivation_time + sub_dt : 0.0f;
        }
        if (ball_settled_verify) {
            solver_scratch->ball_settled =
                (ball_settled_rows_ok && BallWorldMeshManifoldsEmpty(*solver_scratch) &&
                 BallSettleStateEqual(ball, solver_scratch->ball_settled_state)) ? 1 : 0;
        }

        for (int i = 0; world_part != kWorldStepRowsCarsApart && i < count; ++i) {
            WorldStepFinishCar(cars[base + i], car_configs[base + i],
                               wheel_runtime_states ? &wheel_runtime_states[base + i] : nullptr, mutator, sub_dt,
                               pre_contact_cars[i], deferred_linear_deltas[i], deferred_angular_deltas[i]);
        }
        if (should_stop_after_stage(9)) {
            return;
        }
    }
}

struct InternalControlsAccessor {
    CarControls* controls = nullptr;
    int base = 0;

    ROCKETGPU_HD CarControls get(int local_car_idx, int tick) const {
        (void)tick;
        return controls[base + local_car_idx];
    }
};

struct StreamControlsAccessor {
    const CarControls* controls_by_tick = nullptr;
    int base = 0;
    int stride = 0;

    ROCKETGPU_HD CarControls get(int local_car_idx, int tick) const {
        return controls_by_tick[static_cast<size_t>(tick) * static_cast<size_t>(stride) + static_cast<size_t>(base + local_car_idx)];
    }
};

struct NoopEnvCommandAccessor {
    ROCKETGPU_HD bool has_commands() const { return false; }
    ROCKETGPU_HD EnvCommand get(int tick) const {
        (void)tick;
        return EnvCommand{};
    }
};

struct StreamEnvCommandAccessor {
    const EnvCommand* commands_by_tick = nullptr;
    int base = 0;
    int stride = 0;

    ROCKETGPU_HD bool has_commands() const { return commands_by_tick != nullptr; }
    ROCKETGPU_HD EnvCommand get(int tick) const {
        if (!commands_by_tick) {
            return EnvCommand{};
        }
        return commands_by_tick[tick * stride + base];
    }
};

ROCKETGPU_HD void ApplyEnvCommandBodies(
    WorldSolverScratch& scratch, const EnvCommand& command, int car_count, const CarState* env_cars) {
    if (command.reset_to_kickoff && command.reset_scoreboard) {
        scratch.bp = BulletBroadphaseHistory{};
        return;
    }
    if (!command.reset_to_kickoff && !command.new_round_bodies) {
        return;
    }
    BulletSweepNewRound(scratch.bp, car_count, env_cars);
    for (int manifold_idx = 0; manifold_idx < kMaxBallWorldMeshPersistentManifolds; ++manifold_idx) {
        ResetBallMeshPersistentManifold(scratch.ball_world_mesh_manifolds[manifold_idx]);
    }
    for (int manifold_idx = 0; manifold_idx < kBallWorldPlaneSlots; ++manifold_idx) {
        ResetBallMeshPersistentManifold(scratch.ball_world_plane_manifolds[manifold_idx]);
    }
    scratch.retained_reset_pending = 1;
    for (int k = 0; k < 8; ++k) {
        scratch.car_world_reset[k] = 1;
    }
}

ROCKETGPU_HD void ApplyEnvCommand(
    int env_idx,
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    CarControls* controls,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    const EnvCommand& command,
    RespawnRngState* respawn_rng = nullptr,
    const int* kickoff_order_override = nullptr,
    const CarConfig* car_configs = nullptr) {
    if (!command.reset_to_kickoff) {
        return;
    }

    ResetEnvToKickoff(
        env_idx,
        cfg,
        teams,
        env_car_count,
        cars,
        controls,
        balls,
        pads,
        command.kickoff_seed,
        command.reset_scoreboard == 0,
        env_states,
        command.kickoff_seed_mode,
        respawn_rng,
        kickoff_order_override,
        command.keep_boost_pads != 0,
        car_configs,
        command.kickoff_countdown_ticks != 0);
    for (int i = 0; i < env_car_count[env_idx]; ++i) {
        const int slot = CarVisitSlot(env_states[env_idx], i, env_car_count[env_idx]);
        cars[env_idx * cfg.max_cars_per_env + slot].driving_disabled = command.kickoff_countdown_ticks;
    }
    balls[env_idx].constrained_2d = command.kickoff_countdown_ticks;

    EnvState& env = env_states[env_idx];
    if (command.reset_scoreboard) {
        env.blue_score = 0;
        env.orange_score = 0;
        env.last_scored_team = -1;
    }
    if (command.reset_tick_count) {
        env.tick = 0;
    }
    env.goal_scored_this_tick = 0;
    env.countdown_phase = command.kickoff_countdown_ticks == kKickoffCountdownGameTimers ? 1 : 0;
    env.countdown_fires = 0;
    env.countdown_count = 0.0f;
}

template <bool kStagedIndependent = false>
ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE void RocketSim_Car_PreTickUpdate(
    CarState& car,
    CarControls controls,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    float dt,
    WheelRuntimeState* ROCKETGPU_RESTRICT wheel_runtime_state,
    const ArenaMeshTriangle* mesh_triangles = nullptr,
    int32_t mesh_triangle_count = 0,
    const CarState* cars_in_env = nullptr,
    const CarConfig* car_cfgs_in_env = nullptr,
    int32_t cars_in_env_count = 0,
    int32_t self_car_idx = -1,
    const BallState* ball = nullptr,
    float ball_radius = 0.0f,
    int32_t env_idx = -1,
    const ArenaMeshBvhQuantization* mesh_bvh_quantization = nullptr,
    int32_t mesh_bvh_quantization_count = 0,
    const ArenaMeshBvhMeshInfo* mesh_bvh_meshes = nullptr,
    int32_t mesh_bvh_mesh_count = 0,
    const ArenaMeshBvhNode* mesh_bvh_nodes = nullptr,
    int32_t mesh_bvh_node_count = 0,
    const ArenaMeshBvhSubtreeInfo* mesh_bvh_subtrees = nullptr,
    int32_t mesh_bvh_subtree_count = 0,
    const WheelRayStage* staged_rays = nullptr,
    CarState* cal_push_cars = nullptr,
    BallState* cal_push_ball = nullptr,
    uint32_t* cal_pushed_cars_mask = nullptr,
    bool staged_rays_velocity_stale = false,
    uint32_t wheel_ray_excluded_bodies = 0u,
    CarFrame* frame_out = nullptr,
    const ArenaMeshLeafGrid* leaf_grid = nullptr) {
    PreTickCar<kStagedIndependent>(
        car,
        controls,
        cfg,
        mutator,
        dt,
        wheel_runtime_state,
        mesh_triangles,
        mesh_triangle_count,
        cars_in_env,
        car_cfgs_in_env,
        cars_in_env_count,
        self_car_idx,
        ball,
        ball_radius,
        env_idx,
        mesh_bvh_quantization,
        mesh_bvh_quantization_count,
        mesh_bvh_meshes,
        mesh_bvh_mesh_count,
        mesh_bvh_nodes,
        mesh_bvh_node_count,
        mesh_bvh_subtrees,
        mesh_bvh_subtree_count,
        staged_rays,
        cal_push_cars,
        cal_push_ball,
        cal_pushed_cars_mask,
        staged_rays_velocity_stale,
        wheel_ray_excluded_bodies,
        frame_out,
        leaf_grid);
}

ROCKETGPU_HD void RocketSim_Car_PostTickUpdate(
    CarState& car,
    CarControls controls,
    float dt) {
    PostTickCar(car, controls, dt);
}

ROCKETGPU_HD void RocketSim_Car_FinishPhysicsTick(CarState& car) {
    FinishCarTick(car);
}

ROCKETGPU_HD void ReportRLBotCarFields(CarState& car, const WheelRuntimeState* jw, const MutatorConfig& mutator, float dt);

ROCKETGPU_HD void AdvanceRLBotCarFields(CarState& car, const WheelRuntimeState* jw, const MutatorConfig& mutator, float dt) {
    if (!(LengthSq(car.vel_bt) < 1.0e38f && LengthSq(car.ang_vel) < 1.0e38f)) {
        ReportRLBotCarFields(car, jw, mutator, dt);
        return;
    }
    car.reported_flags = 0;
    if (car.is_demoed && car.tick_count_since_update == 0) {
        return;
    }
    const bool dj_started = car.has_double_jumped && !car.internal_has_double_jumped;
    car.internal_has_double_jumped = car.has_double_jumped;
    car.double_jump_ticks = !car.has_double_jumped ? 255 :
        (dj_started ? 0 : (car.double_jump_ticks < 255 ? car.double_jump_ticks + 1 : 255));
    const bool on_ground_state = car.is_on_ground && !car.is_jumping && !car.is_flipping;
    if (car.is_jumping) {
        car.game_has_jumped = 1;
    } else if (on_ground_state && (!car.has_jumped ||
               car.internal_jump_time > mutator.jump_min_time + mutator.jump_reset_time_pad)) {
        car.game_has_jumped = 0;
    }
}

ROCKETGPU_HD void ReportRLBotCarFields(CarState& car, const WheelRuntimeState* jw, const MutatorConfig& mutator, float dt) {
    if (car.is_demoed && car.tick_count_since_update == 0) {
        car.reported_flags = 0;
        return;
    }
    const bool dj_started = car.has_double_jumped && !car.internal_has_double_jumped;
    car.internal_vel = car.vel;
    car.internal_ang_vel = car.ang_vel;
    car.internal_has_jumped = car.has_jumped;
    car.internal_has_double_jumped = car.has_double_jumped;
    car.internal_has_flipped = car.has_flipped;
    car.internal_is_supersonic = car.is_supersonic;

    car.double_jump_ticks = !car.has_double_jumped ? 255 :
        (dj_started ? 0 : (car.double_jump_ticks < 255 ? car.double_jump_ticks + 1 : 255));
    const bool on_ground_state = car.is_on_ground && !car.is_jumping && !car.is_flipping;
    if (car.is_jumping) {
        car.game_has_jumped = 1;
    } else if (on_ground_state && (!car.has_jumped ||
               car.internal_jump_time > mutator.jump_min_time + mutator.jump_reset_time_pad)) {
        car.game_has_jumped = 0;
    }

    const float max_speed_bt = mutator.car_max_speed * UnitsUUToBT();
    if (!car.is_demoed && LengthSq(car.vel_bt) > max_speed_bt * max_speed_bt) {
        const Vec3 vel_bt_normalized = mutator.native_sphere_rsqrt_table
            ? NativeSseSphereNormalizer{mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift}(car.vel_bt)
            : car.vel_bt * RsqrtScalar(LengthSq(car.vel_bt));
        car.vel = (vel_bt_normalized * max_speed_bt) * 50.0f;
    }
    if (!car.is_demoed) {
        car.ang_vel = NativeSseClampLength(car.ang_vel, mutator.car_max_ang_speed,
            mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift);
    }
    car.has_jumped = car.game_has_jumped;
    car.has_double_jumped = car.internal_has_double_jumped && !on_ground_state;
    car.has_flipped = car.internal_has_flipped && !on_ground_state;
    car.is_supersonic = LengthSq(car.vel) >= mutator.supersonic_start_speed * mutator.supersonic_start_speed;

    car.air_state = car.is_jumping ? 1 : (car.is_flipping ? 3 : (car.is_on_ground ? 0 :
        (car.double_jump_ticks < 13 ? 2 : 4)));
    const float jump_inactive = jw ? jw->cal_jump_inactive_time : car.air_time_since_jump;
    const bool dodge_window = !car.is_on_ground && !car.is_jumping && car.has_jumped && !car.internal_has_double_jumped &&
        !car.internal_has_flipped && jump_inactive < mutator.doublejump_max_delay - 0.5f * dt;
    car.dodge_timeout = dodge_window ? mutator.doublejump_max_delay - jump_inactive : -1.0f;
    car.dodge_elapsed = (car.internal_has_flipped && !on_ground_state) ? car.internal_flip_time : 0.0f;
    car.dodge_dir_x = car.flip_rel_torque.y;
    car.dodge_dir_y = -car.flip_rel_torque.x;

    car.reported_vel = car.vel;
    car.reported_ang_vel = car.ang_vel;
    car.reported_flags = CarState::kReportedValid |
        (car.has_jumped ? CarState::kReportedHasJumped : 0) |
        (car.has_double_jumped ? CarState::kReportedHasDoubleJumped : 0) |
        (car.has_flipped ? CarState::kReportedHasFlipped : 0) |
        (car.is_supersonic ? CarState::kReportedIsSupersonic : 0);
}

ROCKETGPU_HD void RestoreStepCarFields(CarState& car) {
    if (car.tick_count_since_update != 0) {
        car.RestoreWorkingFields();
        return;
    }
    if (car.supersonic_time > 0.0f) {
        car.is_supersonic = 1;
    }
    car.internal_has_double_jumped = car.has_double_jumped;
    car.game_has_jumped = car.has_jumped;
    car.double_jump_ticks = 255;
}

ROCKETGPU_HD void RocketSim_Ball_FinishPhysicsTick(BallState& ball) {
    FinishBallTick(ball);
}

ROCKETGPU_HD void RocketSim_BoostPad_PreTickUpdate(int env_idx, const SimConfig& cfg, BoostPadState* pads, float dt) {
    PreTickBoostPadsForEnv(env_idx, cfg, pads, dt);
}

ROCKETGPU_HD void RocketSim_BoostPad_PostTickUpdate(int env_idx, const SimConfig& cfg, CarState* cars, BoostPadState* pads) {
    PostTickBoostPadsForEnv(env_idx, cfg, cars, pads);
}

ROCKETGPU_HD ROCKETGPU_WORLD_STEP_INLINE void RocketSim_Arena_WorldStepSoccar(
    int env_idx,
    int base,
    int count,
    Team* teams,
    CarState* cars,
    WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch,
    const WorldSolverScratchRows& rows,
    CarConfig* car_configs,
    BallState& ball,
    const MutatorConfig& mutator,
    uint8_t enable_car_car_collisions,
    uint8_t enable_demos,
    uint64_t current_tick,
    float dt,
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
    const ArenaBroadphaseGrid& broadphase_grid = ArenaBroadphaseGrid{},
    int world_part = kWorldStepWhole,
    uint8_t cars_apart_mask = 0,
    bool register_iterations = false) {
    WorldStepSoccar(
        env_idx,
        base,
        count,
        teams,
        cars,
        wheel_runtime_states,
        solver_scratch,
        rows,
        car_configs,
        ball,
        mutator,
        enable_car_car_collisions,
        enable_demos,
        current_tick,
        dt,
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
        broadphase_grid,
        world_part,
        cars_apart_mask,
        register_iterations);
}

template <typename MeshLanes = ArenaRayMeshOneLane>
ROCKETGPU_HD void StageWheelRaysForEnv(
    int lane,
    int env_idx,
    const SimConfig& cfg,
    const CarState* cars,
    const CarConfig* car_configs,
    const WheelRuntimeState* wheel_runtime_states,
    const BallState* balls,
    const EnvState& env,
    WorldSolverScratch& scratch,
    int base,
    int count,
    const MeshLanes& mesh_lanes = MeshLanes{},
    const ArenaMeshBvhMeshInfo* mesh_bvh_meshes_override = nullptr) {
    constexpr int kStageCars = kWheelRayStageSlots / 4;
    if (lane < 0 || lane >= kWheelRayStageSlots) {
        return;
    }
    WheelRayStage& stage = scratch.wheel_ray_stage[lane];
    uint8_t& stage_ordered = scratch.wheel_ray_ordered[lane];
    const int car_slot = lane / 4;
    const int wheel_idx = lane % 4;
    if (car_slot >= count || car_slot >= cfg.max_cars_per_env || count > kStageCars) {
        if (mesh_lanes.Leader()) {
            stage.valid = 0;
            stage_ordered = car_slot < count ? 0xFFu : 0u;
        }
        return;
    }
    const CarState& car = cars[base + car_slot];
    if (car.is_demoed) {
        if (mesh_lanes.Leader()) {
            stage.is_in_contact = 0;
            stage.valid = 2;
            stage_ordered = 0u;
        }
        return;
    }
    if (car.set_wheel_normals_mask != 0) {
        if (mesh_lanes.Leader()) {
            stage.valid = 0;
            stage_ordered = 0xFFu;
        }
        return;
    }
    const CarConfig& car_cfg = car_configs[base + car_slot];
    WheelStepState wheel;
    uint32_t excluded_bodies =
        WheelRayExcludedBodies(car, car_cfg, &scratch.bp.sweep, car_slot, count, &wheel, wheel_idx);
    const uint32_t other_car_cands =
        count < 31 ? (((1u << count) - 1u) & ~(1u << car_slot) & ~(excluded_bodies >> 1)) : 1u;
    int32_t no_response = 0;
    if (other_car_cands != 0u) {
        if (count < 31) {
            for (uint32_t m = other_car_cands; m != 0u; m &= m - 1u) {
                const int j = CountTrailingZeros32(m);
                if (cars[base + j].is_demoed) {
                    excluded_bodies |= 1u << (j + 1);
                } else if (cars[base + j].demo_no_contact_response != 0) {
                    no_response |= 1 << j;
                }
            }
        } else {
            for (int j = 0; j < count; ++j) {
                no_response |= (cars[base + j].demo_no_contact_response != 0 ? 1 : 0) << j;
            }
        }
        if (no_response != 0 || count >= 31) {
            for (int k = 0; k < count; ++k) {
                const int j = CarVisitSlot(env, k, count);
                if (j == car_slot) {
                    break;
                }
                if (j >= 0 && j < count) {
                    no_response = cars[base + j].is_demoed ? (no_response | (1 << j)) : (no_response & ~(1 << j));
                }
            }
        }
    }
    const ArenaBroadphaseGrid grid = MakeArenaBroadphaseGrid(cfg);
    RayCastWheelAgainstArena(
        car,
        car_cfg,
        wheel,
        cfg.arena_mesh_tris,
        cfg.arena_mesh_tri_count,
        &cars[base],
        &car_configs[base],
        count,
        car_slot,
        &balls[env_idx],
        cfg.mutator.ball_radius,
        cfg.arena_mesh_bvh_quantization,
        cfg.arena_mesh_bvh_quantization_count,
        mesh_bvh_meshes_override != nullptr ? mesh_bvh_meshes_override : cfg.arena_mesh_bvh_meshes,
        cfg.arena_mesh_bvh_mesh_count,
        cfg.arena_mesh_bvh_nodes,
        cfg.arena_mesh_bvh_node_count,
        cfg.arena_mesh_bvh_subtrees,
        cfg.arena_mesh_bvh_subtree_count,
        cfg.mutator.arena_collision != 0, cfg.mutator.native_sphere_rsqrt_table, cfg.mutator.native_sphere_rsqrt_shift,
        no_response, CalMinWheelHitDistSqBt(car_cfg), mesh_lanes, cfg.mutator.cal_arena_local_tris,
        excluded_bodies, &grid.leaf_grid);
    if (mesh_lanes.Leader()) {
        if (wheel.is_in_contact) {
            wheel.extra_pushback =
                wheel_runtime_states ? wheel_runtime_states[base + car_slot].extra_pushback[wheel_idx] : 0.0f;
            StoreWheelRayStage(wheel, stage);
            stage.valid = 1;
        } else {
            stage.is_in_contact = 0;
            stage.valid = 2;
        }
        uint8_t ord = 0u;
        if (wheel.is_in_contact && !wheel.is_in_contact_with_world) {
            if (wheel.ground_kind == ROCKETGPU_ARENA_RAY_HIT_BALL) {
                ord = static_cast<uint8_t>(1u << car_slot);
            } else if (wheel.ground_kind == ROCKETGPU_ARENA_RAY_HIT_CAR &&
                       wheel.ground_car_idx >= 0 && wheel.ground_car_idx < count) {
                ord = static_cast<uint8_t>((1u << car_slot) | (1u << wheel.ground_car_idx));
            } else {
                ord = 0xFFu;
            }
        }
        stage_ordered = ord;
    }
}

enum : int { kStepEnvPhaseFull = 0, kStepEnvPhasePost = 2, kStepEnvPhasePostVehiclesDone = 3 };

template <bool kStagedIndependent = false>
ROCKETGPU_HD ROCKETGPU_ALWAYS_INLINE void StepEnvVehicleUpdateCarInline(
    int env_idx,
    int i,
    int count,
    const SimConfig& cfg,
    CarState* cars,
    WheelRuntimeState* wheel_runtime_states,
    CarConfig* car_configs,
    BallState* balls,
    const CarControls& controls,
    const WheelRayStage* staged_rays_for_env,
    uint32_t& pushed_cars_mask,
    const BulletSweepHistory* sweep = nullptr,
    CarFrame* frame_out = nullptr) {
    const int base = env_idx * cfg.max_cars_per_env;
    const ArenaMeshLeafGrid leaf_grid{
        cfg.arena_mesh_leaf_grid_infos,
        cfg.arena_mesh_leaf_cell_start,
        cfg.arena_mesh_leaf_entries};
    RocketSim_Car_PreTickUpdate<kStagedIndependent>(
        cars[base + i],
        controls,
        car_configs[base + i],
        cfg.mutator,
        cfg.tick_dt,
        wheel_runtime_states ? &wheel_runtime_states[base + i] : nullptr,
        cfg.arena_mesh_tris,
        cfg.arena_mesh_tri_count,
        &cars[base],
        &car_configs[base],
        count,
        i,
        &balls[env_idx],
        cfg.mutator.ball_radius,
        env_idx,
        cfg.arena_mesh_bvh_quantization,
        cfg.arena_mesh_bvh_quantization_count,
        cfg.arena_mesh_bvh_meshes,
        cfg.arena_mesh_bvh_mesh_count,
        cfg.arena_mesh_bvh_nodes,
        cfg.arena_mesh_bvh_node_count,
        cfg.arena_mesh_bvh_subtrees,
        cfg.arena_mesh_bvh_subtree_count,
        staged_rays_for_env != nullptr ? staged_rays_for_env + 4 * i : nullptr,
        &cars[base],
        &balls[env_idx],
        &pushed_cars_mask,
        i < 32 && ((pushed_cars_mask >> i) & 1u),
        (kStagedIndependent || cars[base + i].is_demoed ||
         (staged_rays_for_env != nullptr && staged_rays_for_env[4 * i].valid != 0))
            ? 0u
            : WheelRayExcludedBodies(cars[base + i], car_configs[base + i], sweep, i, count),
        frame_out,
        kStagedIndependent ? nullptr : &leaf_grid);
}

ROCKETGPU_HD void StepEnvVehicleUpdateCar(
    int env_idx,
    int i,
    int count,
    const SimConfig& cfg,
    CarState* cars,
    WheelRuntimeState* wheel_runtime_states,
    CarConfig* car_configs,
    BallState* balls,
    const CarControls& controls,
    const WheelRayStage* staged_rays_for_env,
    uint32_t& pushed_cars_mask,
    const BulletSweepHistory* sweep = nullptr) {
    StepEnvVehicleUpdateCarInline(env_idx, i, count, cfg, cars, wheel_runtime_states, car_configs, balls, controls,
                                  staged_rays_for_env, pushed_cars_mask, sweep);
}

template <bool kMeshesOutOfReach = false, int kPart = 0>
ROCKETGPU_HD ROCKETGPU_CAR_WORLD_ALWAYS_INLINE void BuildCarWorldRowsForCar(
    int env_idx,
    int i,
    const SimConfig& cfg,
    CarState* cars,
    const CarConfig* car_configs,
    WorldSolverScratch& scratch,
    const WorldSolverScratchRows& rows,
    bullet_port::btGjkEpaScratch* gjk_scratch,
    uint32_t mesh_reach_hint = 0u,
    CarWorldLocalRows* plane_rows = nullptr,
    CarWorldRowsPart* part = nullptr) {
    const int car_idx = env_idx * cfg.max_cars_per_env + i;
    CarWorldRowsOut& out = scratch.car_world_rows_out[i];
    if (kPart != 2) {
        out.group_count = 0;
        out.world_surface = 0;
    }
    if (i < kMaxCarWorldMeshPersistentCars && scratch.car_world_reset[i]) {
        for (int plane_idx = 0; kPart != 1 && plane_idx < kCarWorldPlaneRetainedManifoldsPerCar; ++plane_idx) {
            rows.car_world_plane_manifolds[i * kCarWorldPlaneRetainedManifoldsPerCar + plane_idx].count = 0;
        }
        for (int k = 0; kPart != 2 && k < kCarWorldMeshRetainedManifoldsPerCar; ++k) {
            rows.car_world_mesh_manifolds[i * kCarWorldMeshRetainedManifoldsPerCar + k].count = 0;
        }
        if (kPart == 0) {
            scratch.car_world_reset[i] = 0;
        }
    }
    int split_end = i * kMaxCarWorldSplitContactRows;
    int velocity_end = i * kMaxCarWorldVelocityContactRows;
    int friction_end = i * kMaxCarWorldFrictionContactRows;
    if (kPart == 0 && (scratch.pre_contact_cars[i].is_demoed || cars[car_idx].just_respawned)) {
        if (i < kMaxCarWorldMeshPersistentCars) {
            for (int plane_idx = 0; plane_idx < kCarWorldPlaneRetainedManifoldsPerCar; ++plane_idx) {
                rows.car_world_plane_manifolds[i * kCarWorldPlaneRetainedManifoldsPerCar + plane_idx].count = 0;
            }
            for (int k = 0; k < kCarWorldMeshRetainedManifoldsPerCar; ++k) {
                rows.car_world_mesh_manifolds[i * kCarWorldMeshRetainedManifoldsPerCar + k].count = 0;
            }
        }
    } else {
        const ArenaBroadphaseGrid grid = MakeArenaBroadphaseGrid(cfg);
        const ContactAccumulator acc = SolveCarWorldStaticContacts<kMeshesOutOfReach, kPart>(
            cars[car_idx], car_configs[car_idx], cfg.mutator,
            scratch.deferred_linear_deltas[i], scratch.deferred_angular_deltas[i],
            cfg.tick_dt / static_cast<float>(DebugWorldSubsteps()), env_idx, i,
            cfg.arena_mesh_tris, cfg.arena_mesh_tri_count,
            cfg.arena_mesh_bvh_quantization, cfg.arena_mesh_bvh_quantization_count,
            cfg.arena_mesh_bvh_meshes, cfg.arena_mesh_bvh_mesh_count,
            cfg.arena_mesh_bvh_nodes, cfg.arena_mesh_bvh_node_count,
            cfg.arena_mesh_bvh_subtrees, cfg.arena_mesh_bvh_subtree_count,
            rows.world_split_rows, &split_end,
            rows.world_velocity_rows, &velocity_end,
            rows.world_friction_rows, &friction_end,
            gjk_scratch,
            EnvStridedArray<SharedContactOrderGroup>(out.groups), &out.group_count,
            &grid,
            CarWorldMeshCandidateStageFor(env_idx, i),
            (i < kMaxCarWorldMeshPersistentCars)
                ? &rows.car_world_plane_manifolds[i * kCarWorldPlaneRetainedManifoldsPerCar]
                : nullptr,
            (i < kMaxCarWorldMeshPersistentCars)
                ? &rows.car_world_mesh_manifolds[i * kCarWorldMeshRetainedManifoldsPerCar]
                : nullptr,
            kPart == 2 ? plane_rows : rows.car_world_local_rows ? &rows.car_world_local_rows[i] : nullptr,
            mesh_reach_hint,
            true,
            part,
            (i < 8 && scratch.car_predict_valid[i]) ? &scratch.car_predict_vel[i] : nullptr,
            (i < 8 && scratch.car_predict_valid[i]) ? &scratch.car_predict_ang_vel[i] : nullptr);
        if (kPart != 0) {
            return;
        }
        RecordArenaWorldContact(cars[car_idx], acc);
        out.world_surface = acc.surface;
    }
    out.split_row_count = split_end - i * kMaxCarWorldSplitContactRows;
    out.velocity_row_count = velocity_end - i * kMaxCarWorldVelocityContactRows;
    out.friction_row_count = friction_end - i * kMaxCarWorldFrictionContactRows;
}

ROCKETGPU_HD ROCKETGPU_CAR_WORLD_ALWAYS_INLINE void FinishCarWorldRowsForCar(
    int env_idx,
    int i,
    const SimConfig& cfg,
    CarState* cars,
    WorldSolverScratch& scratch,
    const WorldSolverScratchRows& rows,
    const CarWorldRowsPart& mesh_part,
    CarWorldLocalRows& plane_rows,
    const CarWorldRowsPart& plane_part) {
    const int car_idx = env_idx * cfg.max_cars_per_env + i;
    CarWorldRowsOut& out = scratch.car_world_rows_out[i];
    if (scratch.car_world_reset[i]) {
        scratch.car_world_reset[i] = 0;
    }
    int split_end = i * kMaxCarWorldSplitContactRows;
    int velocity_end = i * kMaxCarWorldVelocityContactRows;
    int friction_end = i * kMaxCarWorldFrictionContactRows;
    const ContactAccumulator acc = FinishCarWorldSplitRows(
        i, rows.car_world_local_rows[i], mesh_part, plane_rows, plane_part,
        rows.world_split_rows, &split_end,
        rows.world_velocity_rows, &velocity_end,
        rows.world_friction_rows, &friction_end,
        EnvStridedArray<SharedContactOrderGroup>(out.groups), &out.group_count);
    RecordArenaWorldContact(cars[car_idx], acc);
    out.world_surface = acc.surface;
    out.split_row_count = split_end - i * kMaxCarWorldSplitContactRows;
    out.velocity_row_count = velocity_end - i * kMaxCarWorldVelocityContactRows;
    out.friction_row_count = friction_end - i * kMaxCarWorldFrictionContactRows;
}

ROCKETGPU_HD bool VehicleUpdatesIndependent(const SimConfig& cfg, const WheelRayStage* staged_rays_for_env, int count) {
    if (staged_rays_for_env == nullptr || cfg.mutator.physics_events.HasNonGoalEvents() || count > kWheelRayStageSlots / 4) {
        return false;
    }
    for (int lane = 0; lane < 4 * count; ++lane) {
        const WheelRayStage& ray = staged_rays_for_env[lane];
        if (ray.valid == 0 || (ray.valid == 1 && ray.is_in_contact && !ray.is_in_contact_with_world)) {
            return false;
        }
    }
    return true;
}

ROCKETGPU_HD void CountdownUpdateTimers(EnvState& env, CarState* env_cars, int count, BallState& ball, float frame_dt) {
    if (env.countdown_phase == 0) {
        return;
    }
    env.countdown_count += frame_dt;
    if (env.countdown_count > 1.0f) {
        if (env.countdown_phase == 1) {
            env.countdown_phase = 2;
            env.countdown_count = 0.0f;
        } else {
            env.countdown_count -= 1.0f;
            if (++env.countdown_fires == 3) {
                env.countdown_phase = 0;
                env.countdown_fires = 0;
                for (int i = 0; i < count; ++i) {
                    env_cars[i].driving_disabled = 0;
                }
                ball.constrained_2d = 0;
            }
        }
    }
}

ROCKETGPU_HD void StepEnvPreTickEnvState(
    int env_idx,
    const SimConfig& cfg,
    int32_t* env_car_count,
    CarState* cars,
    BallState* balls,
    EnvState* env_states) {
    EnvState& env = env_states[env_idx];
    const float frame_start_dt = FrameStartDt(cfg, env_idx);
    if (cfg.mutator.countdown_on_frame_start && frame_start_dt >= 0.0f) {
        CountdownUpdateTimers(env, cars + env_idx * cfg.max_cars_per_env, env_car_count[env_idx], balls[env_idx],
                              frame_start_dt);
    }
    UpdateGameCarVisitOrder(env, cars + env_idx * cfg.max_cars_per_env, env_car_count[env_idx]);
    if (cfg.mutator.physics_events.windows) {
        cfg.mutator.physics_events.windows[env_idx].tick = static_cast<uint64_t>(env.tick);
    }
    env.goal_scored_this_tick = 0;
}

ROCKETGPU_HD void StepEnvPreTickEnv(
    int env_idx,
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    WorldSolverScratch* solver_scratch,
    CarControls* controls_for_reset,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    const EnvCommand& command,
    const int* kickoff_order_override,
    const CarConfig* car_configs = nullptr) {
    if (solver_scratch) {
        ApplyEnvCommandBodies(
            solver_scratch[env_idx], command, env_car_count[env_idx], cars + env_idx * cfg.max_cars_per_env);
    }
    ApplyEnvCommand(
        env_idx,
        cfg,
        teams,
        env_car_count,
        cars,
        controls_for_reset,
        balls,
        env_states,
        pads,
        command,
        nullptr,
        kickoff_order_override,
        car_configs);
    StepEnvPreTickEnvState(env_idx, cfg, env_car_count, cars, balls, env_states);
    if (solver_scratch && command.reset_to_kickoff) {
        for (int manifold_idx = 0; manifold_idx < kMaxBallWorldMeshPersistentManifolds; ++manifold_idx) {
            ResetBallMeshPersistentManifold(solver_scratch[env_idx].ball_world_mesh_manifolds[manifold_idx]);
        }
        for (int manifold_idx = 0; manifold_idx < kBallWorldPlaneSlots; ++manifold_idx) {
            ResetBallMeshPersistentManifold(solver_scratch[env_idx].ball_world_plane_manifolds[manifold_idx]);
        }
        solver_scratch[env_idx].retained_reset_pending = 1;
        for (int k = 0; k < 8; ++k) {
            solver_scratch[env_idx].car_world_reset[k] = 1;
        }
    }
}

ROCKETGPU_HD void StepEnvPreTickCar(
    int env_idx,
    int i,
    const SimConfig& cfg,
    Team* teams,
    CarState* cars,
    WorldSolverScratch* solver_scratch,
    const EnvState& env,
    const uint8_t* respawn_slot_overrides,
    uint64_t seed,
    CarControls controls,
    const CarConfig& car_cfg,
    WheelRuntimeState* wheel_runtime_state) {
    const int base = env_idx * cfg.max_cars_per_env;
    CarState& car = cars[base + i];
    car.ball_contact_this_tick = 0;
    RestoreStepCarFields(car);
    if (car.tick_count_since_update == 0) {
        CalInternalTimersFromSetState(car, cfg.mutator, cfg.tick_dt);
    }
    if (car.just_respawned) {
        car.just_respawned = 0;
    }
    bool respawned = false;
    if (car.is_demoed) {
        car.demo_respawn_timer = Max(0.0f, car.demo_respawn_timer - cfg.tick_dt);
        if (car.demo_respawn_timer <= 0.0f) {
            const uint32_t car_id = static_cast<uint32_t>(base + i + 1);
            int respawn_slot = -1;
            if (car.respawn_slot_override_plus1 != 0) {
                respawn_slot = car.respawn_slot_override_plus1 - 1;
                car.respawn_slot_override_plus1 = 0;
            } else if (respawn_slot_overrides) {
                const uint8_t override_slot = respawn_slot_overrides[base + i];
                if (override_slot != 0xFFu) {
                    respawn_slot = static_cast<int>(override_slot);
                }
            }
            if (respawn_slot < 0) {
                respawn_slot = PickRespawnSlot(seed, env_idx, static_cast<uint64_t>(env.tick), car_id);
            }
            if (respawn_slot >= kKickoffSpawnOverridePlus1 - 1) {
                KickoffSpawnCar(car, teams[base + i], respawn_slot - (kKickoffSpawnOverridePlus1 - 1), cfg.mutator,
                                car_cfg.spawn_z);
            } else {
                RespawnCar(car, teams[base + i], respawn_slot, cfg.mutator, car_cfg.spawn_z);
            }
            respawned = true;
        }
    }
    const Quat spawn_rot = car.rot;
    car.tick_start_pos_bt = car.pos_bt;
    CalTickStartLimitAndQuantizeCar(car, cfg.mutator);
    if (respawned) {
        SpawnInertiaBasis(spawn_rot, car.tick_start_basis_forward, car.tick_start_basis_right, car.tick_start_basis_up);
    }
    if (solver_scratch != nullptr && i < 8) {
        solver_scratch[env_idx].cal_start_vel[i] = car.vel;
    }
    const CarRbSnapshot tick_start_snapshot = TickStartCarRbSnapshot(car);
    CalFlipZDampPrepass(car, cfg.mutator, cfg.tick_dt);
    car.cal_tick_start_ang_vel = car.ang_vel;
    const Vec3 vel_before_components = car.vel_bt;
    PreTickCarPrePhysics(car, controls, car_cfg, cfg.mutator, FrameFrac(cfg, env_idx), cfg.tick_dt, wheel_runtime_state, tick_start_snapshot);
    if (!car.is_demoed) {
        const bool impulsed = car.vel_bt.x != vel_before_components.x || car.vel_bt.y != vel_before_components.y ||
            car.vel_bt.z != vel_before_components.z;
        if ((BodyDampingClampCar(car, cfg.mutator, true, true) & 1) && impulsed) {
            SetCarLinearVelocityBt(car, car.vel_bt);
        }
    }
    if (solver_scratch != nullptr && i < 8) {
        solver_scratch[env_idx].car_predict_vel[i] = car.vel;
        solver_scratch[env_idx].car_predict_ang_vel[i] = car.ang_vel;
        solver_scratch[env_idx].car_predict_valid[i] = 1;
    }
}

ROCKETGPU_HD void StepEnvPreTickBall(int env_idx, const SimConfig& cfg, WorldSolverScratch* solver_scratch, BallState* balls) {
    CalTickStartLimitAndQuantizeBall(balls[env_idx], cfg.mutator);
    if (solver_scratch != nullptr) {
        solver_scratch[env_idx].cal_ball_start_vel = balls[env_idx].vel;
    }
    if (cfg.mutator.game_mode != static_cast<uint8_t>(Mode::Heatseeker)) {
        BodyDampingClampBall(balls[env_idx], cfg.mutator);
        ApplyBulletLinearDampingBall(balls[env_idx], cfg.mutator, cfg.tick_dt);
        if (solver_scratch != nullptr) {
            solver_scratch[env_idx].ball_predict_vel_bt = balls[env_idx].vel_bt;
        }
    }
}

ROCKETGPU_HD void StepEnvPreTickBallAndPads(
    int env_idx,
    const SimConfig& cfg,
    int count,
    WorldSolverScratch* solver_scratch,
    BallState* balls,
    BoostPadState* pads) {
    StepEnvPreTickBall(env_idx, cfg, solver_scratch, balls);
    if (cfg.enable_boost_pads && count > 0) {
        RocketSim_BoostPad_PreTickUpdate(env_idx, cfg, pads, cfg.tick_dt);
    }
}

ROCKETGPU_HD void StepEnvPostCar(
    int env_idx, int i, const SimConfig& cfg, CarState* cars, WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch, const CarConfig* car_configs, const CarControls& controls, bool finish_world,
    bool report_fields = true) {
    const int car_idx = env_idx * cfg.max_cars_per_env + i;
    CarState& car = cars[car_idx];
    if (solver_scratch) {
        const WorldSolverScratch& scratch = solver_scratch[env_idx];
        if (finish_world) {
            WorldStepFinishCar(car, car_configs[car_idx], wheel_runtime_states ? &wheel_runtime_states[car_idx] : nullptr,
                               cfg.mutator, cfg.tick_dt / static_cast<float>(DebugWorldSubsteps()),
                               scratch.pre_contact_cars[i], scratch.deferred_linear_deltas[i],
                               scratch.deferred_angular_deltas[i]);
        }
        if (!scratch.pre_contact_cars[i].is_demoed && !car.just_respawned && car.is_demoed) {
            car.rot = scratch.pre_contact_cars[i].rot;
        }
    }
    RocketSim_Car_PostTickUpdate(car, controls, cfg.tick_dt);
    RocketSim_Car_FinishPhysicsTick(car);
    if (report_fields) {
        ReportRLBotCarFields(car, wheel_runtime_states ? &wheel_runtime_states[car_idx] : nullptr, cfg.mutator, cfg.tick_dt);
    } else {
        AdvanceRLBotCarFields(car, wheel_runtime_states ? &wheel_runtime_states[car_idx] : nullptr, cfg.mutator, cfg.tick_dt);
    }
}

ROCKETGPU_HD void ApplyBallExtraHits(int env_idx, int count, const SimConfig& cfg, CarState* cars, BallState& ball) {
    const int base = env_idx * cfg.max_cars_per_env;
    const float ball_inv_mass = 1.0f / cfg.mutator.ball_mass;
    const float ball_mass_bt = 1.0f / ball_inv_mass;
    Vec3 v = ball.vel_bt;
    bool any = false;
    for (int i = 0; i < count; ++i) {
        BallHitInfo& hit = cars[base + i].ball_hit_info;
        if (!hit.extra_pending) {
            continue;
        }
        const Vec3 a = hit.extra_hit_vel;
        v = Vec3(v.x + ((a.x * 0.02f) * ball_mass_bt) * ball_inv_mass, v.y + ((a.y * 0.02f) * ball_mass_bt) * ball_inv_mass,
                 v.z + ((a.z * 0.02f) * ball_mass_bt) * ball_inv_mass);
        hit.extra_pending = 0;
        any = true;
    }
    if (any) {
        SetBallLinearVelocityBt(ball, v);
    }
}

ROCKETGPU_HD void StepEnvTail(
    int env_idx, int count, const SimConfig& cfg, CarState* cars, const CarConfig* car_configs, BallState& ball,
    EnvState& env, BoostPadState* pads, bool pads_post, bool car_pad_checks = true) {
    if (car_pad_checks) {
        BoostPadTouchPassForEnv(env_idx, count, cfg, cars, car_configs, env, pads);
    }

    if (pads_post && cfg.enable_boost_pads && count > 0) {
        RocketSim_BoostPad_PostTickUpdate(env_idx, cfg, cars, pads);
    }

    for (int k = 0; k < count; ++k) {
        const int i = CarVisitSlot(env, k, count);
        CarState& car = cars[env_idx * cfg.max_cars_per_env + i];
        if (!car.is_demoed && car.driving_disabled) {
            --car.driving_disabled;
        }
    }

    ApplyBallExtraHits(env_idx, count, cfg, cars, ball);
    RocketSim_Ball_FinishPhysicsTick(ball);

    if (cfg.enable_scoring) {
        const float score_threshold = cfg.mutator.goal_base_threshold_y + cfg.mutator.ball_radius;
        if (ball.pos.y >= score_threshold) {
            env.last_scored_team = static_cast<int32_t>(Team::Blue);
            env.goal_scored_this_tick = 1;
            env.blue_score += 1;
        } else if (ball.pos.y <= -score_threshold) {
            env.last_scored_team = static_cast<int32_t>(Team::Orange);
            env.goal_scored_this_tick = 1;
            env.orange_score += 1;
        }
    }

    if (env.goal_scored_this_tick && cfg.mutator.physics_events.windows) {
        PhysicsEvent event;
        event.type = PhysicsEventType::Goal;
        event.team = env.last_scored_team;
        event.ball_position = ball.pos;
        event.ball_velocity = ball.vel;
        cfg.mutator.physics_events.Emit(env_idx, event);
    }

    env.frame_ticks += 1;
    const float frame_frac = FrameFrac(cfg, env_idx);
    if (frame_frac >= 0.0f) {
        const float frame_dt = (static_cast<float>(env.frame_ticks) + frame_frac - env.frame_frac_prev) * cfg.tick_dt;
        env.frame_ticks = 0;
        env.frame_frac_prev = frame_frac;
        if (!cfg.mutator.countdown_on_frame_start) {
            CountdownUpdateTimers(env, cars + env_idx * cfg.max_cars_per_env, count, ball, frame_dt);
        }
    }
    env.tick += 1;
}

template <typename ControlsAccessor, typename EnvCommandAccessor>
ROCKETGPU_HD ROCKETGPU_WORLD_STEP_INLINE void StepEnvGeneric(
    int env_idx,
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch,
    const WorldSolverScratchRowPool& row_pool,
    CarControls* controls_for_reset,
    const ControlsAccessor& controls_accessor,
    const EnvCommandAccessor& env_command_accessor,
    CarConfig* car_configs,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    const uint8_t* respawn_slot_overrides,
    uint64_t seed,
    int ticks,
    const int* kickoff_order_override = nullptr,

    int phase = kStepEnvPhaseFull,
    int tick_offset = 0,
    int world_part = kWorldStepWhole,
    uint8_t cars_apart_mask = 0,
    uint8_t cars_updated_mask = 0,
    bool register_iterations = false) {
    const int base = env_idx * cfg.max_cars_per_env;
    const WheelRayStage* const staged_rays_for_env =
        ((phase == kStepEnvPhasePost || phase == kStepEnvPhasePostVehiclesDone) && solver_scratch != nullptr)
            ? solver_scratch[env_idx].wheel_ray_stage : nullptr;

    const WorldSolverScratchRows rows = row_pool.view(env_idx);
    for (int t = 0; t < ticks; ++t) {
        const int tick_idx = tick_offset + t;
        if (phase == kStepEnvPhaseFull) {
            if (env_command_accessor.has_commands()) {
                const EnvCommand command = env_command_accessor.get(tick_idx);
                StepEnvPreTickEnv(env_idx, cfg, teams, env_car_count, cars, solver_scratch, controls_for_reset, balls,
                                  env_states, pads, command, kickoff_order_override, car_configs);
            } else {
                StepEnvPreTickEnvState(env_idx, cfg, env_car_count, cars, balls, env_states);
            }
            const int pre_count = env_car_count[env_idx];
            for (int k = 0; k < pre_count; ++k) {
                const int slot = CarVisitSlot(env_states[env_idx], k, pre_count);
                StepEnvPreTickCar(env_idx, slot, cfg, teams, cars, solver_scratch, env_states[env_idx],
                                  respawn_slot_overrides, seed, controls_accessor.get(slot, tick_idx),
                                  car_configs[base + slot],
                                  wheel_runtime_states ? &wheel_runtime_states[base + slot] : nullptr);
            }
            StepEnvPreTickBallAndPads(env_idx, cfg, pre_count, solver_scratch, balls, pads);
        }
        const int count = env_car_count[env_idx];
        EnvState& env = env_states[env_idx];
        const uint64_t current_tick = static_cast<uint64_t>(env.tick);

        uint32_t cal_pushed_cars_mask = 0u;
        const bool vehicle_updates_done = WorldPartIsRows(world_part) || world_part == kWorldStepContactsCarsApart ||
            WorldPartIsPrefix(world_part) ||
            (phase == kStepEnvPhasePostVehiclesDone && VehicleUpdatesIndependent(cfg, staged_rays_for_env, count));
        for (int k = 0; !vehicle_updates_done && k < count; ++k) {
            const int i = CarVisitSlot(env_states[env_idx], k, count);
            if ((cars_updated_mask >> i) & 1u) {
                continue;
            }
            StepEnvVehicleUpdateCar(env_idx, i, count, cfg, cars, wheel_runtime_states, car_configs, balls,
                                    controls_accessor.get(i, tick_idx), staged_rays_for_env, cal_pushed_cars_mask,
                                    solver_scratch != nullptr ? &solver_scratch[env_idx].bp.sweep : nullptr);
        }

        BallState& ball = balls[env_idx];
        if (!WorldPartIsRows(world_part) && !WorldPartIsPrefix(world_part) &&
            cfg.mutator.game_mode == static_cast<uint8_t>(Mode::Heatseeker)) {
            HeatseekerBallPreTick(ball, cfg.tick_dt, [](BallState& b, const Vec3& v_bt) { SetBallLinearVelocityBt(b, v_bt); });
        }
        RocketSim_Arena_WorldStepSoccar(
            env_idx,
            base,
            count,
            teams,
            cars,
            wheel_runtime_states,
            solver_scratch ? solver_scratch + env_idx : nullptr,
            rows,
            car_configs,
            ball,
            cfg.mutator,
            cfg.enable_car_car_collisions,
            cfg.enable_demos,
            current_tick,
            cfg.tick_dt,
            cfg.arena_mesh_tris,
            cfg.arena_mesh_tri_count,
            cfg.arena_mesh_bvh_quantization,
            cfg.arena_mesh_bvh_quantization_count,
            cfg.arena_mesh_bvh_meshes,
            cfg.arena_mesh_bvh_mesh_count,
            cfg.arena_mesh_bvh_nodes,
            cfg.arena_mesh_bvh_node_count,
            cfg.arena_mesh_bvh_subtrees,
            cfg.arena_mesh_bvh_subtree_count,
            MakeArenaBroadphaseGrid(cfg),
            world_part,
            cars_apart_mask,
            register_iterations);
        if (WorldPartIsContacts(world_part) || world_part == kWorldStepRowsCarsApart || WorldPartIsPrefix(world_part)) {
            continue;
        }

        const bool report_fields = (t + 1 == ticks) || env_command_accessor.has_commands();
        for (int k = 0; k < count; ++k) {
            const int i = CarVisitSlot(env_states[env_idx], k, count);
            StepEnvPostCar(env_idx, i, cfg, cars, wheel_runtime_states, solver_scratch, car_configs,
                           controls_accessor.get(i, tick_idx), false, report_fields);
        }
        StepEnvTail(env_idx, count, cfg, cars, car_configs, ball, env, pads, true);
    }
}

ROCKETGPU_HD ROCKETGPU_NOINLINE void StepEnv(
    int env_idx,
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch,
    const WorldSolverScratchRowPool& row_pool,
    CarControls* controls,
    CarConfig* car_configs,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    const uint8_t* respawn_slot_overrides,
    uint64_t seed,
    int ticks,
    int phase = kStepEnvPhaseFull,
    int tick_offset = 0,
    int world_part = kWorldStepWhole) {
    InternalControlsAccessor accessor;
    accessor.controls = controls;
    accessor.base = env_idx * cfg.max_cars_per_env;
    NoopEnvCommandAccessor no_env_commands;
    StepEnvGeneric(env_idx, cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls, accessor, no_env_commands, car_configs, balls, env_states, pads, respawn_slot_overrides, seed, ticks, nullptr, phase, tick_offset, world_part);
}

ROCKETGPU_HD void StepEnvWithControlStream(
    int env_idx,
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch,
    const WorldSolverScratchRowPool& row_pool,
    CarControls* controls_for_reset,
    const CarControls* controls_by_tick,
    int control_stride,
    CarConfig* car_configs,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    const uint8_t* respawn_slot_overrides,
    uint64_t seed,
    int phase = kStepEnvPhaseFull,
    int ticks = 1) {
    StreamControlsAccessor accessor;
    accessor.controls_by_tick = controls_by_tick;
    accessor.base = env_idx * cfg.max_cars_per_env;
    accessor.stride = control_stride;
    NoopEnvCommandAccessor no_env_commands;
    StepEnvGeneric(env_idx, cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls_for_reset, accessor, no_env_commands, car_configs, balls, env_states, pads, respawn_slot_overrides, seed, ticks, nullptr, phase, 0);
}

ROCKETGPU_HD void StepEnvWithCommandStreams(
    int env_idx,
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch,
    const WorldSolverScratchRowPool& row_pool,
    CarControls* controls_for_reset,
    const CarControls* controls_by_tick,
    int control_stride,
    const EnvCommand* env_commands_by_tick,
    int env_command_stride,
    CarConfig* car_configs,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    const uint8_t* respawn_slot_overrides,
    uint64_t seed,
    const int* kickoff_order_override = nullptr,
    int phase = kStepEnvPhaseFull) {
    StreamControlsAccessor controls_accessor;
    controls_accessor.controls_by_tick = controls_by_tick;
    controls_accessor.base = env_idx * cfg.max_cars_per_env;
    controls_accessor.stride = control_stride;

    StreamEnvCommandAccessor env_accessor;
    env_accessor.commands_by_tick = env_commands_by_tick;
    env_accessor.base = env_idx;
    env_accessor.stride = env_command_stride;

    StepEnvGeneric(env_idx, cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls_for_reset, controls_accessor, env_accessor, car_configs, balls, env_states, pads, respawn_slot_overrides, seed, 1, kickoff_order_override, phase, 0);
}
