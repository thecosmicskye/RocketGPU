

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
    // Every read is behind the slot's or a point's valid flag, and an invalid slot has no flagged points.
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

// cal_car_car_manifold helpers (v3 persistent_manifold.rs, car pair orientation in CarCarRetainedPoint).
// v3 CollisionShape::get_contact_breaking_threshold of the car compound: (bounding sphere radius of the
// compound's local AABB + |its center|) * CONTACT_BREAKING_THRESHOLD; BoxShape::new keeps implicit_dim =
// half - margin, margin = min(0.1 * min half extent, CONVEX_DISTANCE_MARGIN).
ROCKETGPU_HD float CarCarBreakingThresholdBt(const CarConfig& cfg) {
    const Vec3 half = (cfg.hitbox_size * 0.02f) * 0.5f;
    const float margin = Min(0.1f * Min(Min(half.x, half.y), half.z), 0.04f);
    const Vec3 margin_v(margin, margin, margin);
    const Vec3 half_with_margin = (half - margin_v) + margin_v;
    const Vec3 center = cfg.hitbox_offset * 0.02f;
    const Vec3 aabb_min = center - half_with_margin;
    const Vec3 aabb_max = center + half_with_margin;
    const Vec3 sphere_center = (aabb_min + aabb_max) * 0.5f;
    const float radius = Length(aabb_max - aabb_min) * 0.5f;
    return (radius + Length(sphere_center)) * 0.02f;
}

// v3 PersistentManifold::refresh_contact_points.
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

// v3 PersistentManifold::add_contact_without_callback (get_cache_entry, replace, add_manifold_point,
// sort_cached_points on v3 local_point_a = local_b_bt).
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

ROCKETGPU_HD void ResolveCarCarCollision(
    int env_idx,
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
    CarCarPairCachedContact* pair_contact_cache = nullptr,
    EnvStridedArray<CarSharedSplitContactRow> shared_split_rows = nullptr,
    int* shared_split_row_count = nullptr,
    EnvStridedArray<CarSharedVelocityContactRow> shared_velocity_rows = nullptr,
    int* shared_velocity_row_count = nullptr,
    EnvStridedArray<CarSharedFrictionContactRow> shared_friction_rows = nullptr,
    int* shared_friction_row_count = nullptr,
    const Vec3* start_vel_a = nullptr,
    const Vec3* start_vel_b = nullptr,
    CarCarRetainedManifold* retained_manifold = nullptr) {
    if (a.is_demoed || b.is_demoed || a.just_respawned || b.just_respawned) {
        if (retained_manifold) {
            retained_manifold->count = 0;
        }
        return;
    }

    int pair_idx = -1;
    if (respawn_slot_a >= 0 && respawn_slot_b >= 0) {
        pair_idx = respawn_slot_a * 8 + respawn_slot_b;
    }
    CarCarPairCachedContact* pair_cache =
        pair_contact_cache && pair_idx >= 0 ? &pair_contact_cache[pair_idx] : nullptr;
    const bool use_cached_contact =
        pair_cache && pair_cache->valid != 0 && pair_cache->contact_count > 0;
    const bool use_shared_velocity_rows =
        shared_velocity_rows && shared_velocity_row_count &&
        shared_friction_rows && shared_friction_row_count;

    const float ra = BoundingSphereRadius(cfg_a);
    const float rb = BoundingSphereRadius(cfg_b);
    const float min_dist = ra + rb;
    const float min_dist_sq = min_dist * min_dist;
    bool force_current_contact_seed = false;
    bool force_current_from_broadphase_gap = false;

    // cal_car_car_manifold: refresh the pair's retained v3 manifold on the tick-start transforms; v3 runs the
    // OBB-OBB detector on those transforms only (no predicted contact), behind a pair AABB test.
    const bool cal_ccm = retained_manifold != nullptr;
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
        return;
    }

    // Predicted states only after the retained-manifold gate, which most pairs fail.
    const CarState predicted_a = use_cached_contact
        ? a
        : BuildPredictedCarSolverState(a, external_linear_delta_a, external_angular_delta_a, dt, NativeSseMath(mutator));
    const CarState predicted_b = use_cached_contact
        ? b
        : BuildPredictedCarSolverState(b, external_linear_delta_b, external_angular_delta_b, dt, NativeSseMath(mutator));
    const Vec3 solver_external_linear_delta_a = external_linear_delta_a;
    const Vec3 solver_external_linear_delta_b = external_linear_delta_b;
    const float speed_threshold = mutator.car_max_speed * 4.0f;
    const float speed_threshold_sq = speed_threshold * speed_threshold;
    const bool extreme_speed_pair =
        LengthSq(predicted_a.vel) > speed_threshold_sq ||
        LengthSq(predicted_b.vel) > speed_threshold_sq;
    const Vec3 broad_delta = predicted_b.pos - predicted_a.pos;
    if (!cal_ccm && !use_cached_contact && LengthSq(broad_delta) >= min_dist_sq) {
        if (!extreme_speed_pair) {
            if (pair_cache) {
                ClearCarCarPairCachedContact(*pair_cache);
            }
            return;
        }
        const Vec3 current_broad_delta = b.pos - a.pos;
        if (LengthSq(current_broad_delta) >= min_dist_sq) {
            if (pair_cache) {
                ClearCarCarPairCachedContact(*pair_cache);
            }
            return;
        }
        force_current_contact_seed = true;
        force_current_from_broadphase_gap = true;
    }

    ObbData obb_a = MakeCarObb(force_current_contact_seed ? a : predicted_a, cfg_a);
    ObbData obb_b = MakeCarObb(force_current_contact_seed ? b : predicted_b, cfg_b);
    const ObbData current_obb_a = MakeCarObb(a, cfg_a);
    const ObbData current_obb_b = MakeCarObb(b, cfg_b);
    Vec3 normal(1.0f, 0.0f, 0.0f);

    constexpr float kFreshContactThresholdUU = 0.0f;
    float contact_overlap_uu = 0.0f;
    ObbContactFeature contact_feature;
    Vec3 current_normal(1.0f, 0.0f, 0.0f);
    float current_overlap_uu = 0.0f;
    ObbContactFeature current_feature;

    const ObbData current_obb_a_bt = MakeCarObbBulletTransformBt(a, cfg_a);
    const ObbData current_obb_b_bt = MakeCarObbBulletTransformBt(b, cfg_b);
    float current_overlap_bt = 0.0f;
    const bool have_current_contact =
        !use_cached_contact &&
        ccm_fresh_gate &&
        ComputeObbContact(
            current_obb_a_bt,
            current_obb_b_bt,
            kFreshContactThresholdUU,
            current_normal,
            current_overlap_bt,
            &current_feature,
            env_idx,
            respawn_slot_a,
            respawn_slot_b,
                         0);
    current_overlap_uu = current_overlap_bt * UnitsBTToUU();
    float predicted_overlap_bt = 0.0f;
    bool have_contact = use_cached_contact;
    if (use_cached_contact) {
        normal = pair_cache->normal;
        contact_overlap_uu = GetCarCarCachedPenetrationUU(*pair_cache);
    } else if (have_current_contact) {
        obb_a = current_obb_a;
        obb_b = current_obb_b;
        normal = current_normal;
        contact_overlap_uu = current_overlap_uu;
        contact_feature = current_feature;
        force_current_contact_seed = true;
        have_contact = true;
    } else if (!cal_ccm && ComputeObbContact(
                   MakeCarObbBulletTransformBt(force_current_contact_seed ? a : predicted_a, cfg_a),
                   MakeCarObbBulletTransformBt(force_current_contact_seed ? b : predicted_b, cfg_b),
                   kFreshContactThresholdUU,
                   normal, predicted_overlap_bt, &contact_feature, env_idx,
                   respawn_slot_a, respawn_slot_b,
                                1)) {
        contact_overlap_uu = predicted_overlap_bt * UnitsBTToUU();
    } else {
        if (extreme_speed_pair && have_current_contact) {
            obb_a = current_obb_a;
            obb_b = current_obb_b;
            normal = current_normal;
            contact_overlap_uu = current_overlap_uu;
            contact_feature = current_feature;
            force_current_contact_seed = true;
            have_contact = true;
        }
    }
    const bool ccm_fresh = have_contact;
    if (cal_ccm && !have_contact && retained_manifold->count > 0) {
        // Retained points only (v3 (Some(cached), None)): rows on the tick-start state, no car-hit callback.
        normal = retained_manifold->points[0].normal;
        contact_overlap_uu = 0.0f;
        force_current_contact_seed = true;
        have_contact = true;
    }
    if (!have_contact) {
        if (pair_cache) {
            ClearCarCarPairCachedContact(*pair_cache);
        }
        return;
    }

    float bumper_local_x_a = -1.0e30f;
    float bumper_local_x_b = -1.0e30f;
    bool has_bumper_local_x_override = false;

    if (!use_cached_contact &&
        extreme_speed_pair &&
        contact_overlap_uu > 0.0f &&
        contact_feature.kind == OBB_CONTACT_FEATURE_EDGE_EDGE) {
        force_current_contact_seed = true;
    }

    bool skip_velocity_impulse =
        !force_current_contact_seed &&
        !use_cached_contact &&
        extreme_speed_pair;

    CarState solver_seed_a = (force_current_contact_seed || use_cached_contact) ? a : predicted_a;
    CarState solver_seed_b = (force_current_contact_seed || use_cached_contact) ? b : predicted_b;
    if (force_current_from_broadphase_gap) {
        solver_seed_a.vel = predicted_a.vel;
        solver_seed_a.ang_vel = predicted_a.ang_vel;
        solver_seed_b.vel = predicted_b.vel;
        solver_seed_b.ang_vel = predicted_b.ang_vel;
    }
    CarState solver_a = solver_seed_a;
    CarState solver_b = solver_seed_b;

    Vec3 contact_points_a[kMaxApproxCarCarContactPoints];
    Vec3 contact_points_b[kMaxApproxCarCarContactPoints];
    Vec3 contact_points_a_bt[kMaxApproxCarCarContactPoints];
    Vec3 contact_points_b_bt[kMaxApproxCarCarContactPoints];

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
        contact_point_count = 0;
    } else {
        contact_point_count = ComputeApproxCarCarContactAnchorPairs(
            solver_a,
            cfg_a,
            obb_a,
            solver_b,
            cfg_b,
            obb_b,
            normal,
            contact_feature,
            contact_points_a,
            contact_points_b,
            kMaxApproxCarCarContactPoints,
            contact_overlap_uu,
            contact_points_a_bt,
            contact_points_b_bt,
            env_idx,
            respawn_slot_a,
            respawn_slot_b);
        if (contact_point_count <= 0) {
            contact_point_count = 1;
            ComputeApproxCarCarContactAnchors(
                solver_a,
                cfg_a,
                obb_a,
                solver_b,
                cfg_b,
                obb_b,
                normal,
                contact_feature,
                contact_points_a[0],
                contact_points_b[0]);
            contact_points_a_bt[0] = ToBtLinearVec(contact_points_a[0]);
            contact_points_b_bt[0] = ToBtLinearVec(contact_points_b[0]);
        }
    }
    if (!use_cached_contact &&
        !force_current_contact_seed &&
        have_current_contact &&
        contact_overlap_uu > 0.0f &&
        current_overlap_uu > 0.0f &&
        contact_point_count > 1) {
        obb_a = current_obb_a;
        obb_b = current_obb_b;
        normal = current_normal;
        contact_overlap_uu = current_overlap_uu;
        contact_feature = current_feature;
        solver_seed_a = a;
        solver_seed_b = b;

        solver_seed_a.vel = predicted_a.vel;
        solver_seed_a.ang_vel = predicted_a.ang_vel;
        solver_seed_b.vel = predicted_b.vel;
        solver_seed_b.ang_vel = predicted_b.ang_vel;
        solver_a = solver_seed_a;
        solver_b = solver_seed_b;
        contact_point_count = ComputeApproxCarCarContactAnchorPairs(
            solver_a,
            cfg_a,
            obb_a,
            solver_b,
            cfg_b,
            obb_b,
            normal,
            contact_feature,
            contact_points_a,
            contact_points_b,
            kMaxApproxCarCarContactPoints,
            contact_overlap_uu,
            contact_points_a_bt,
            contact_points_b_bt,
            env_idx,
            respawn_slot_a,
            respawn_slot_b);
        if (contact_point_count <= 0) {
            contact_point_count = 1;
            ComputeApproxCarCarContactAnchors(
                solver_a,
                cfg_a,
                obb_a,
                solver_b,
                cfg_b,
                obb_b,
                normal,
                contact_feature,
                contact_points_a[0],
                contact_points_b[0]);
            contact_points_a_bt[0] = ToBtLinearVec(contact_points_a[0]);
            contact_points_b_bt[0] = ToBtLinearVec(contact_points_b[0]);
        }
    }
    if (!use_cached_contact && have_current_contact && contact_point_count > 0) {
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

    if (!use_cached_contact && have_current_contact && has_bumper_local_x_override) {
        const bool a_is_lower_id = id_a > id_b;
        CarState* hitter1 = a_is_lower_id ? &a : &b;
        CarState* victim1 = a_is_lower_id ? &b : &a;
        const CarState* hitter1_predicted = a_is_lower_id ? &predicted_a : &predicted_b;
        const CarState* victim1_predicted = a_is_lower_id ? &predicted_b : &predicted_a;
        const uint32_t hitter1_id = a_is_lower_id ? id_a : id_b;
        const uint32_t victim1_id = a_is_lower_id ? id_b : id_a;
        const Team hitter1_team = a_is_lower_id ? team_a : team_b;
        const Team victim1_team = a_is_lower_id ? team_b : team_a;
        const ObbData& hitter1_obb = a_is_lower_id ? obb_a : obb_b;
        const ObbData& victim1_obb = a_is_lower_id ? obb_b : obb_a;
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
                    (void)ResolveCarCarDirectionalContact(
                        *hitter1, *hitter1_predicted, hitter1_id, hitter1_team, hitter1_obb, hitter1_cfg,
                        *victim1, *victim1_predicted, victim1_id, victim1_team, victim1_obb,
                        mutator, enable_demos, normal1, has_bumper_local_x_override, hitter1_bumper_x,
                        a_is_lower_id ? start_vel_a : start_vel_b,
                        a_is_lower_id ? &cfg_b : &cfg_a, hitter1->is_demoed != 0,
                        a_is_lower_id ? start_vel_b : start_vel_a, dt);
                    victim1_demoed_here = victim1_demoed_here || victim1->is_demoed;
                }
                if (a.is_demoed && b.is_demoed) {
                    break;
                }
                if (!hitter1->is_demoed && (!victim1->is_demoed || victim1_demoed_here)) {
                    (void)ResolveCarCarDirectionalContact(
                        *victim1, *victim1_predicted, victim1_id, victim1_team, victim1_obb,
                        a_is_lower_id ? cfg_b : cfg_a,
                        *hitter1, *hitter1_predicted, hitter1_id, hitter1_team, hitter1_obb,
                        mutator, enable_demos, -normal1, has_bumper_local_x_override, hitter2_bumper_x,
                        a_is_lower_id ? start_vel_b : start_vel_a,
                        &hitter1_cfg, victim1->is_demoed != 0,
                        a_is_lower_id ? start_vel_a : start_vel_b, dt);
                    hitter1_demoed_here = hitter1_demoed_here || hitter1->is_demoed;
                }
            }
        }
    }
    // Per-point contact normals: the pair normal unless cal_car_car_manifold supplies retained points.
    Vec3 cc_point_normals[kMaxApproxCarCarContactPoints];
    for (int contact_idx = 0; contact_idx < kMaxApproxCarCarContactPoints; ++contact_idx) {
        cc_point_normals[contact_idx] = normal;
    }
    if (cal_ccm) {
        // v3 near_callback_with_bounds: the fresh BoxBoxDetector manifold (add_contact_point per point,
        // refresh_contact_points) merges into the refreshed retained manifold (merge_contact_points); the
        // solver gets every retained point, positive distances included.
        if (ccm_fresh && contact_point_count > 0) {
            CarCarRetainedManifold fresh_manifold;
            for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
                const Vec3 point_on_a = contact_points_a_bt[contact_idx];
                const float depth = Dot(contact_points_b_bt[contact_idx] - point_on_a, normal);
                if (depth > ccm_threshold_bt) {
                    continue;
                }
                CarCarRetainedPoint point;
                point.normal = normal;
                point.local_a_bt = CarBodyLocalPointBt(a, point_on_a);
                point.local_b_bt = CarBodyLocalPointBt(b, point_on_a + normal * depth);
                point.distance_bt = depth;
                AddCarCarRetainedPoint(fresh_manifold, point, ccm_threshold_bt);
            }
            RefreshCarCarRetainedManifold(fresh_manifold, a, b, ccm_threshold_bt);
            if (retained_manifold->count == 0) {
                *retained_manifold = fresh_manifold;
            } else {
                for (int contact_idx = 0; contact_idx < fresh_manifold.count; ++contact_idx) {
                    AddCarCarRetainedPoint(*retained_manifold, fresh_manifold.points[contact_idx], ccm_threshold_bt);
                }
            }
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
            return;
        }
    }
    if (skip_velocity_impulse && contact_point_count > 1 && contact_overlap_uu > 1.0f) {
        skip_velocity_impulse = false;
    }
    if (skip_velocity_impulse && contact_overlap_uu > 0.0f) {
        skip_velocity_impulse = false;
    }
    const float penetration = Max(0.0f, contact_overlap_uu);
    float resolved_normal_impulse[kMaxApproxCarCarContactPoints] = {};
    float resolved_friction_impulse[kMaxApproxCarCarContactPoints] = {};
    Vec3 resolved_tangent[kMaxApproxCarCarContactPoints];
    uint8_t resolved_tangent_valid[kMaxApproxCarCarContactPoints] = {};
    Vec3 cached_tangent[kMaxApproxCarCarContactPoints];
    uint8_t cached_tangent_valid[kMaxApproxCarCarContactPoints] = {};
    Vec3 contact_rel_a[kMaxApproxCarCarContactPoints];
    Vec3 contact_rel_b[kMaxApproxCarCarContactPoints];
    Vec3 contact_rel_a_bt[kMaxApproxCarCarContactPoints];
    Vec3 contact_rel_b_bt[kMaxApproxCarCarContactPoints];
    float normal_row_jac_diag_ab_inv[kMaxApproxCarCarContactPoints] = {};
    float normal_row_base_vn_bt[kMaxApproxCarCarContactPoints] = {};
#if !defined(__CUDA_ARCH__)
    float normal_row_restitution_velocity_bt[kMaxApproxCarCarContactPoints] = {};
#endif
    float normal_row_rhs[kMaxApproxCarCarContactPoints] = {};
    float normal_row_rhs_bt[kMaxApproxCarCarContactPoints] = {};
    uint8_t normal_row_valid[kMaxApproxCarCarContactPoints] = {};

    for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
        const Vec3 normal = cc_point_normals[contact_idx];
        const Vec3 rel_a_bt = contact_points_a_bt[contact_idx] - solver_a.pos_bt;
        const Vec3 rel_b_bt = contact_points_b_bt[contact_idx] - solver_b.pos_bt;
        const Vec3 rel_a = FromBtLinearVec(rel_a_bt);
        const Vec3 rel_b = FromBtLinearVec(rel_b_bt);
        contact_rel_a[contact_idx] = rel_a;
        contact_rel_b[contact_idx] = rel_b;
        contact_rel_a_bt[contact_idx] = rel_a_bt;
        contact_rel_b_bt[contact_idx] = rel_b_bt;
        CarCarPairCachedPoint* cached_point =
            pair_cache && contact_idx < kMaxApproxCarCarContactPoints
                ? &pair_cache->contacts[contact_idx]
                : nullptr;
        if (cached_point && cached_point->tangent_valid != 0) {
            cached_tangent[contact_idx] = cached_point->tangent;
            cached_tangent_valid[contact_idx] = 1;
        }
        float accumulated_normal_impulse =
            (cached_point && cached_point->valid != 0) ? cached_point->applied_normal_impulse : 0.0f;
        if (!use_shared_velocity_rows && accumulated_normal_impulse > 0.0f) {
            const Vec3 warmstart_impulse = normal * accumulated_normal_impulse;
            ApplyImpulseAtPoint(solver_a, cfg_a, mutator.car_mass, -warmstart_impulse, rel_a);
            ApplyImpulseAtPoint(solver_b, cfg_b, mutator.car_mass, warmstart_impulse, rel_b);
        }
        resolved_normal_impulse[contact_idx] = accumulated_normal_impulse;
        resolved_friction_impulse[contact_idx] =
            (cached_point && cached_point->valid != 0) ? cached_point->applied_friction_impulse : 0.0f;

        CarState solver_a_bt = solver_a;
        CarState solver_b_bt = solver_b;
        solver_a_bt.pos = solver_a.pos_bt;
        solver_b_bt.pos = solver_b.pos_bt;

        const float denom_inv_mass_bt = 1.0f / mutator.car_mass;
        float denom_a, denom_b;
        {
            const Vec3 denom_normal_axis = -normal;
            const Vec3 denom_torque_a = Cross(rel_a_bt, denom_normal_axis);
            const Vec3 denom_ang_a = ApplyInverseBodyInertiaWorldBt(solver_a_bt, cfg_a, mutator.car_mass, denom_torque_a);
            const Vec3 denom_vec_a = Cross(denom_ang_a, rel_a_bt);
            denom_a = denom_inv_mass_bt + Dot(denom_normal_axis, denom_vec_a);

            const Vec3 denom_torque_b = Cross(rel_b_bt, -denom_normal_axis);
            const Vec3 denom_ang_b = ApplyInverseBodyInertiaWorldBt(solver_b_bt, cfg_b, mutator.car_mass, denom_torque_b);
            const Vec3 denom_vec_b = Cross(-denom_ang_b, rel_b_bt);
            denom_b = denom_inv_mass_bt + Dot(denom_normal_axis, denom_vec_b);
        }
        const float denom = denom_a + denom_b;
        if (denom <= 1.0e-6f) {
            continue;
        }

        const Vec3 normal_a_bt = -normal;
        const Vec3 normal_b_bt = normal;
        const Vec3 rel_a_cross_normal_bt = Cross(rel_a_bt, normal_a_bt);
        const Vec3 rel_b_cross_normal_bt = Cross(rel_b_bt, normal_b_bt);

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
        const float vel1_dotn_bt =
            Dot(normal_a_bt, lin_a_bt + ext_lin_a_bt) +
            Dot(rel_a_cross_normal_bt, base_a.ang_vel + ext_ang_a_bt);
        const float vel2_dotn_bt =
            Dot(normal_b_bt, lin_b_bt + ext_lin_b_bt) +
            Dot(rel_b_cross_normal_bt, base_b.ang_vel + ext_ang_b_bt);
        const float base_vn_bt = vel1_dotn_bt + vel2_dotn_bt;
        const float jac_diag_ab_inv = 1.0f / denom;

        const Vec3 vel_a_no_ext_bt = lin_a_bt + Cross(base_a.ang_vel, rel_a_bt);
        const Vec3 vel_b_no_ext_bt = lin_b_bt + Cross(base_b.ang_vel, rel_b_bt);
        const float base_vn_no_ext_bt = Dot(normal_b_bt, vel_b_no_ext_bt - vel_a_no_ext_bt);
        const float restitution_velocity_bt =
            BulletRestitutionCurveBt(base_vn_no_ext_bt, CarCarCollisionRestitution());
        const float rhs_bt = (restitution_velocity_bt - base_vn_bt) * jac_diag_ab_inv;

        normal_row_jac_diag_ab_inv[contact_idx] = jac_diag_ab_inv;
        normal_row_base_vn_bt[contact_idx] = base_vn_bt;
#if !defined(__CUDA_ARCH__)
        normal_row_restitution_velocity_bt[contact_idx] = restitution_velocity_bt;
#endif
        normal_row_rhs[contact_idx] = rhs_bt * UnitsBTToUU();
        normal_row_rhs_bt[contact_idx] = rhs_bt;
        normal_row_valid[contact_idx] = 1;
    }

    if (use_shared_velocity_rows) {
        for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
            const Vec3 normal = cc_point_normals[contact_idx];
            if (skip_velocity_impulse || normal_row_valid[contact_idx] == 0) {
                continue;
            }
            if (*shared_velocity_row_count >= kMaxCarPairSharedVelocityContactRows) {
                break;
            }

            const Vec3 rel_a = contact_rel_a[contact_idx];
            const Vec3 rel_b = contact_rel_b[contact_idx];
            const Vec3 shared_normal_a = -normal;
            const Vec3 shared_normal_b = normal;
            const Vec3 shared_rel_a_cross_normal = Cross(rel_a, shared_normal_a);
            const Vec3 shared_rel_b_cross_normal = Cross(rel_b, shared_normal_b);
            const Vec3 shared_angular_component_a = ApplyInverseBodyInertiaWorld(
                solver_seed_a,
                cfg_a,
                mutator.car_mass,
                shared_rel_a_cross_normal);
            const Vec3 shared_angular_component_b = ApplyInverseBodyInertiaWorld(
                solver_seed_b,
                cfg_b,
                mutator.car_mass,
                shared_rel_b_cross_normal);
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
            shared_row.jac_diag_ab_inv = normal_row_jac_diag_ab_inv[contact_idx];
            shared_row.rhs = normal_row_rhs[contact_idx];
            shared_row.rhs_bt = normal_row_rhs_bt[contact_idx];
            shared_row.has_rhs_bt = 1;

            {
                const Vec3 rel_a_cross_normal_bt_row = Cross(contact_rel_a_bt[contact_idx], shared_normal_a);
                const Vec3 rel_b_cross_normal_bt_row = Cross(contact_rel_b_bt[contact_idx], shared_normal_b);
                shared_row.rel_a_cross_normal_bt = rel_a_cross_normal_bt_row;
                shared_row.rel_b_cross_normal_bt = rel_b_cross_normal_bt_row;
                shared_row.angular_component_a_bt =
                    ApplyInverseBodyInertiaWorldBt(solver_a, cfg_a, mutator.car_mass, rel_a_cross_normal_bt_row);
                shared_row.angular_component_b_bt =
                    ApplyInverseBodyInertiaWorldBt(solver_b, cfg_b, mutator.car_mass, rel_b_cross_normal_bt_row);
                shared_row.has_bt = 1;
            }
            shared_row.applied_impulse = resolved_normal_impulse[contact_idx];
            const int shared_normal_row_idx = *shared_velocity_row_count;
            *shared_velocity_row_count += 1;

            if (*shared_friction_row_count >= kMaxCarPairSharedFrictionContactRows) {
                continue;
            }

            const Vec3 rel_a_bt = contact_rel_a_bt[contact_idx];
            const Vec3 rel_b_bt = contact_rel_b_bt[contact_idx];

            const Vec3 base_lin_a_bt = base_a.vel_bt + (a.ext_force_impulse_bt_valid
                ? a.ext_force_impulse_bt
                : ToBtLinearVec(solver_external_linear_delta_a));
            const Vec3 base_lin_b_bt = base_b.vel_bt + (b.ext_force_impulse_bt_valid
                ? b.ext_force_impulse_bt
                : ToBtLinearVec(solver_external_linear_delta_b));
            const Vec3 fric_ext_ang_a_bt = a.ext_force_impulse_bt_valid ? a.ext_torque_impulse : external_angular_delta_a;
            const Vec3 fric_ext_ang_b_bt = b.ext_force_impulse_bt_valid ? b.ext_torque_impulse : external_angular_delta_b;
            const Vec3 base_vel_a_bt =
                base_lin_a_bt +
                Cross(base_a.ang_vel + fric_ext_ang_a_bt, rel_a_bt);
            const Vec3 base_vel_b_bt =
                base_lin_b_bt +
                Cross(base_b.ang_vel + fric_ext_ang_b_bt, rel_b_bt);
            const Vec3 base_rel_vel_bt = base_vel_b_bt - base_vel_a_bt;
            Vec3 tangent = base_rel_vel_bt - normal * Dot(base_rel_vel_bt, normal);
            const float lat_rel_vel_bt = LengthSq(tangent);
            const float tangent_len = SqrtScalar(lat_rel_vel_bt);
            tangent = lat_rel_vel_bt > 1.1920928955078125e-07f
                ? tangent * (1.0f / tangent_len)
                : BulletPlaneSpace1First(normal);

            const Vec3 tangent_a = tangent;
            const Vec3 tangent_b = -tangent;
            const Vec3 tangent_angular_component_a = ApplyInverseBodyInertiaWorld(
                solver_seed_a,
                cfg_a,
                mutator.car_mass,
                Cross(rel_a, tangent_a));
            const Vec3 tangent_angular_component_b = ApplyInverseBodyInertiaWorld(
                solver_seed_b,
                cfg_b,
                mutator.car_mass,
                Cross(rel_b, tangent_b));
            const Vec3 rel_a_cross_tangent_bt = Cross(rel_a_bt, tangent_a);
            const Vec3 rel_b_cross_tangent_bt = Cross(rel_b_bt, tangent_b);
            const Vec3 denom_angular_component_a_bt = ApplyInverseBodyInertiaWorldBt(
                solver_seed_a,
                cfg_a,
                mutator.car_mass,
                rel_a_cross_tangent_bt);
            const Vec3 denom_angular_component_b_bt = ApplyInverseBodyInertiaWorldBt(
                solver_seed_b,
                cfg_b,
                mutator.car_mass,
                rel_b_cross_tangent_bt);
            const Vec3 denom_vec_a_bt = Cross(denom_angular_component_a_bt, rel_a_bt);
            const Vec3 denom_vec_b_bt = Cross(-denom_angular_component_b_bt, rel_b_bt);
            const float denom_dot_a_bt = Dot(tangent_a, denom_vec_a_bt);
            const float denom_dot_b_bt = Dot(tangent_a, denom_vec_b_bt);
            const float denom_t_a =
                (1.0f / mutator.car_mass) +
                denom_dot_a_bt;
            const float denom_t_b =
                (1.0f / mutator.car_mass) +
                denom_dot_b_bt;
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
            const Vec3 rhs_lin_a_bt = base_lin_a_bt;
            const Vec3 rhs_lin_b_bt = base_lin_b_bt;

            const float rhs_lin1_dott_bt = Dot(tangent_a, rhs_lin_a_bt);
            const float rhs_ang1_dott_bt = Dot(rel_a_cross_tangent_bt, base_a.ang_vel);
            const float rhs_lin2_dott_bt = Dot(tangent_b, rhs_lin_b_bt);
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
            const Vec3 normal = cc_point_normals[contact_idx];
            const Vec3 rel_a = contact_rel_a[contact_idx];
            const Vec3 rel_b = contact_rel_b[contact_idx];
            float accumulated_normal_impulse = resolved_normal_impulse[contact_idx];
            const Vec3 rel_vel = VelocityAtPoint(solver_b, rel_b) - VelocityAtPoint(solver_a, rel_a);
            const float vn = Dot(rel_vel, normal);
            const bool solve_normal_row =
                !skip_velocity_impulse &&
                (vn < 0.0f || penetration > 0.0f || accumulated_normal_impulse > 0.0f);
            if (!solve_normal_row) {
                continue;
            }
            if (normal_row_valid[contact_idx] == 0) {
                continue;
            }

            const Vec3 base_vel_a =
                (base_a.vel + external_linear_delta_a) +
                Cross(base_a.ang_vel + external_angular_delta_a, rel_a);
            const Vec3 base_vel_b =
                (base_b.vel + external_linear_delta_b) +
                Cross(base_b.ang_vel + external_angular_delta_b, rel_b);
            const Vec3 cur_vel_a = VelocityAtPoint(solver_a, rel_a);
            const Vec3 cur_vel_b = VelocityAtPoint(solver_b, rel_b);
            const float delta_vel_a = Dot(base_vel_a - cur_vel_a, normal);
            const float delta_vel_b = Dot(cur_vel_b - base_vel_b, normal);
            const float normal_impulse_delta =
                normal_row_rhs[contact_idx] -
                (delta_vel_a + delta_vel_b) * normal_row_jac_diag_ab_inv[contact_idx];
            const float next_accumulated = Max(0.0f, accumulated_normal_impulse + normal_impulse_delta);
            const float impulse_mag = next_accumulated - accumulated_normal_impulse;
            accumulated_normal_impulse = next_accumulated;

            const Vec3 impulse = normal * impulse_mag;
            ApplyImpulseAtPoint(solver_a, cfg_a, mutator.car_mass, -impulse, rel_a);
            ApplyImpulseAtPoint(solver_b, cfg_b, mutator.car_mass, impulse, rel_b);
            resolved_normal_impulse[contact_idx] = accumulated_normal_impulse;
        }

        for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
            const Vec3 normal = cc_point_normals[contact_idx];
            const Vec3 rel_a = contact_rel_a[contact_idx];
            const Vec3 rel_b = contact_rel_b[contact_idx];
            const float friction_normal_limit_impulse = resolved_normal_impulse[contact_idx];
            if (friction_normal_limit_impulse <= 0.0f) {
                continue;
            }

            float accumulated_friction_impulse = resolved_friction_impulse[contact_idx];
            const Vec3 rel_vel_after_normal = VelocityAtPoint(solver_b, rel_b) - VelocityAtPoint(solver_a, rel_a);
            const float vn_after_normal = Dot(rel_vel_after_normal, normal);
            const Vec3 tangent_vel = rel_vel_after_normal - normal * vn_after_normal;
            const float tangent_len = Length(tangent_vel);
            if (tangent_len <= 1e-5f) {
                continue;
            }

            Vec3 tangent = tangent_vel / tangent_len;

            const float denom_t_a =
                CarCarBodyImpulseDenominatorFromRel(solver_a, cfg_a, mutator.car_mass, rel_a, tangent);
            const float denom_t_b =
                CarCarBodyImpulseDenominatorFromRel(solver_b, cfg_b, mutator.car_mass, rel_b, tangent);
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
            ApplyImpulseAtPoint(solver_a, cfg_a, mutator.car_mass, friction_impulse, rel_a);
            ApplyImpulseAtPoint(solver_b, cfg_b, mutator.car_mass, -friction_impulse, rel_b);
            resolved_friction_impulse[contact_idx] = accumulated_friction_impulse;
            resolved_tangent[contact_idx] = tangent;
            resolved_tangent_valid[contact_idx] = 1;
        }
    }

    if (penetration > 0.0f && dt > 0.0f) {
        if (shared_split_rows && shared_split_row_count) {
            const float inv_mass_a = 1.0f / mutator.car_mass;
            const float inv_mass_b = 1.0f / mutator.car_mass;
            for (int contact_idx = 0; contact_idx < contact_point_count; ++contact_idx) {
                const Vec3 normal = cc_point_normals[contact_idx];
                const Vec3 normal_a = -normal;
                const Vec3 normal_b = normal;
                const Vec3 contact_point_a = contact_points_a[contact_idx];
                const Vec3 contact_point_b = contact_points_b[contact_idx];
                const Vec3 rel_a = CarCarBulletRelativePointUU(contact_point_a, solver_seed_a.pos_bt);
                const Vec3 rel_b = CarCarBulletRelativePointUU(contact_point_b, solver_seed_b.pos_bt);
                const Vec3 point_delta = contact_point_b - contact_point_a;
                const float point_penetration = Max(0.0f, -Dot(point_delta, normal));
                if (point_penetration <= 0.0f) {
                    continue;
                }

                const Vec3 rel_a_cross_normal = Cross(rel_a, normal_a);
                const Vec3 rel_b_cross_normal = Cross(rel_b, normal_b);
                const Vec3 angular_component_a =
                    ApplyInverseBodyInertiaWorld(solver_seed_a, cfg_a, mutator.car_mass, rel_a_cross_normal);
                const Vec3 angular_component_b =
                    ApplyInverseBodyInertiaWorld(solver_seed_b, cfg_b, mutator.car_mass, rel_b_cross_normal);

                float denom = 0.0f;
                denom += inv_mass_a + Dot(normal_a, Cross(angular_component_a, rel_a));
                denom += inv_mass_b + Dot(normal_b, Cross(angular_component_b, rel_b));
                if (denom <= 1.0e-8f) {
                    continue;
                }

                const float jac_diag_ab_inv = 1.0f / denom;
                const float rhs_penetration =
                    point_penetration * SplitImpulsePenetrationErp() * (1.0f / dt) * jac_diag_ab_inv;

                const Vec3 contact_point_a_bt = contact_points_a_bt[contact_idx];
                const Vec3 contact_point_b_bt = contact_points_b_bt[contact_idx];
                const Vec3 rel_a_bt = contact_point_a_bt - solver_seed_a.pos_bt;
                const Vec3 rel_b_bt = contact_point_b_bt - solver_seed_b.pos_bt;
                const float point_penetration_bt =
                    Max(0.0f, -Dot(contact_point_b_bt - contact_point_a_bt, normal));
                const Vec3 rel_a_cross_normal_bt = Cross(rel_a_bt, normal_a);
                const Vec3 rel_b_cross_normal_bt = Cross(rel_b_bt, normal_b);
                const Vec3 angular_component_a_bt =
                    ApplyInverseBodyInertiaWorldBt(solver_seed_a, cfg_a, mutator.car_mass, rel_a_cross_normal_bt);
                const Vec3 angular_component_b_bt =
                    ApplyInverseBodyInertiaWorldBt(solver_seed_b, cfg_b, mutator.car_mass, rel_b_cross_normal_bt);
                float denom_bt = 0.0f;
                denom_bt += inv_mass_a + Dot(normal_a, Cross(angular_component_a_bt, rel_a_bt));
                denom_bt += inv_mass_b + Dot(normal_b, Cross(angular_component_b_bt, rel_b_bt));
                const float jac_diag_ab_inv_bt = denom_bt > 1.0e-8f ? 1.0f / denom_bt : 0.0f;
                const float rhs_penetration_bt =
                    point_penetration_bt * SplitImpulsePenetrationErp() * (1.0f / dt) * jac_diag_ab_inv_bt;
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
                const Vec3 normal = cc_point_normals[contact_idx];
                const Vec3 normal_a = -normal;
                const Vec3 normal_b = normal;

                const Vec3 contact_point_a = contact_points_a_bt[contact_idx];
                const Vec3 contact_point_b = contact_points_b_bt[contact_idx];
                const Vec3 rel_a = contact_point_a - solver_seed_a.pos_bt;
                const Vec3 rel_b = contact_point_b - solver_seed_b.pos_bt;
                const Vec3 point_delta = contact_point_b - contact_point_a;
                const float point_penetration = Max(0.0f, -Dot(point_delta, normal));
                if (point_penetration <= 0.0f) {
                    continue;
                }

                const Vec3 rel_a_cross_normal = Cross(rel_a, normal_a);
                const Vec3 rel_b_cross_normal = Cross(rel_b, normal_b);
                const Vec3 angular_component_a =
                    ApplyInverseBodyInertiaWorldBt(solver_seed_a, cfg_a, mutator.car_mass, rel_a_cross_normal);
                const Vec3 angular_component_b =
                    ApplyInverseBodyInertiaWorldBt(solver_seed_b, cfg_b, mutator.car_mass, rel_b_cross_normal);

                float denom = 0.0f;
                denom += inv_mass_a + Dot(normal_a, Cross(angular_component_a, rel_a));
                denom += inv_mass_b + Dot(normal_b, Cross(angular_component_b, rel_b));
                if (denom <= 1.0e-8f) {
                    continue;
                }

                const float jac_diag_ab_inv = 1.0f / denom;
                const float rhs_penetration =
                    point_penetration * SplitImpulsePenetrationErp() * (1.0f / dt) * jac_diag_ab_inv;

                float delta_impulse = rhs_penetration;
                delta_impulse -= (
                    BtSimdDot3LikeBullet(normal_a, split_a.push_velocity) +
                    BtSimdDot3LikeBullet(rel_a_cross_normal, split_a.turn_velocity)) * jac_diag_ab_inv;
                delta_impulse -= (
                    BtSimdDot3LikeBullet(normal_b, split_b.push_velocity) +
                    BtSimdDot3LikeBullet(rel_b_cross_normal, split_b.turn_velocity)) * jac_diag_ab_inv;

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
        const Vec3 rocketgpu_dva = solver_a.vel - solver_seed_a.vel;
        if (rocketgpu_dva.x != 0.0f || rocketgpu_dva.y != 0.0f || rocketgpu_dva.z != 0.0f) {
            SetCarLinearVelocityUu(a, a.vel + rocketgpu_dva);
        }
    }
    a.ang_vel += solver_a.ang_vel - solver_seed_a.ang_vel;
    SetCarOriginBt(a, a.pos_bt + (solver_a.pos_bt - solver_seed_a.pos_bt));
    {
        const Vec3 rocketgpu_dvb = solver_b.vel - solver_seed_b.vel;
        if (rocketgpu_dvb.x != 0.0f || rocketgpu_dvb.y != 0.0f || rocketgpu_dvb.z != 0.0f) {
            SetCarLinearVelocityUu(b, b.vel + rocketgpu_dvb);
        }
    }
    b.ang_vel += solver_b.ang_vel - solver_seed_b.ang_vel;
    SetCarOriginBt(b, b.pos_bt + (solver_b.pos_bt - solver_seed_b.pos_bt));

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
                if (resolved_tangent_valid[contact_idx] != 0) {
                    cached_point.tangent = resolved_tangent[contact_idx];
                    cached_point.tangent_valid = 1;
                }
            } else {
                cached_point = CarCarPairCachedPoint{};
            }
        }
    }
}

ROCKETGPU_HD void UpdateCarCarPairCacheFromSharedVelocityRows(
    CarCarPairCachedContact* pair_contact_cache,
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
            row.pair_cache_idx < 0 || row.pair_cache_idx >= kMaxCarCarPairSlots ||
            row.pair_contact_idx < 0 || row.pair_contact_idx >= kMaxApproxCarCarContactPoints) {
            continue;
        }
        CarCarPairCachedContact& cache = pair_contact_cache[row.pair_cache_idx];
        if (!cache.valid) {
            continue;
        }
        cache.contacts[row.pair_contact_idx].applied_normal_impulse = row.applied_impulse;
    }

    for (int row_idx = 0; row_idx < shared_friction_row_count; ++row_idx) {
        const CarSharedFrictionContactRow& row = shared_friction_rows[row_idx];
        if (!row.valid ||
            row.pair_cache_idx < 0 || row.pair_cache_idx >= kMaxCarCarPairSlots ||
            row.pair_contact_idx < 0 || row.pair_contact_idx >= kMaxApproxCarCarContactPoints) {
            continue;
        }
        CarCarPairCachedContact& cache = pair_contact_cache[row.pair_cache_idx];
        if (!cache.valid) {
            continue;
        }
        cache.contacts[row.pair_contact_idx].applied_friction_impulse = row.applied_impulse;
        cache.contacts[row.pair_contact_idx].tangent = row.tangent_a;
        cache.contacts[row.pair_contact_idx].tangent_valid = 1;
    }
}

ROCKETGPU_HD bool BoostPadCheckCollideBt(const Vec3& pad_pos_uu, const BoostPadState& pad, const Vec3& car_bt) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    const Vec3 pad_bt(pad_pos_uu.x * kUUToBT, pad_pos_uu.y * kUUToBT, pad_pos_uu.z * kUUToBT);
    const float rad_bt = (pad.is_big ? 160.0f + 48.0f : 128.0f + 48.0f) * kUUToBT;
    const float pad_z_bt = (pad_pos_uu.z - (pad.is_big ? 1.0f : 5.92f)) * kUUToBT;
    const float dx = car_bt.x - pad_bt.x;
    const float dy = car_bt.y - pad_bt.y;
    return (dx * dx + dy * dy) < (rad_bt * rad_bt) && Abs(car_bt.z - pad_z_bt) < ((64.0f + 32.0f) * kUUToBT);
}

ROCKETGPU_HD void PreTickBoostPadsForEnv(int env_idx, const SimConfig& cfg, BoostPadState* pads, float dt) {
    if (!cfg.enable_boost_pads) {
        return;
    }

    const int pad_base = env_idx * BoostPadCount(cfg);
    for (int p = 0; p < BoostPadCount(cfg); ++p) {
        BoostPadState& pad = pads[pad_base + p];
        // RL counts an inactive pad's timer down every tick, past zero, and brings the pad back on the tick after it
        // reads -1/120: 481 / 1201 ticks after the pickup, in time for a car on the pad to take it again that tick.
        if (!pad.is_active || pad.cooldown > 0.0f) {
            if (pad.cooldown < -0.5f * dt) {
                pad.cooldown = 0.0f;
                pad.is_active = 1;
            } else {
                pad.cooldown -= dt;
                pad.is_active = 0;
            }
        }
        pad.cur_locked_car_id = 0;
    }
}

ROCKETGPU_HD constexpr float BoostPadGridExtentZUU() {
    return BoostPadCylinderHeight() + 250.0f;
}

ROCKETGPU_HD constexpr int BoostPadGridCellsX() { return 8; }
ROCKETGPU_HD constexpr int BoostPadGridCellsY() { return 10; }
ROCKETGPU_HD constexpr int BoostPadGridCellSizeX() { return 1024; }
ROCKETGPU_HD constexpr int BoostPadGridCellSizeY() { return 1024; }

ROCKETGPU_HD int BoostPadGridIndexX(float x) {
    return static_cast<int>(x / static_cast<float>(BoostPadGridCellSizeX())) +
           (BoostPadGridCellsX() / 2);
}

ROCKETGPU_HD int BoostPadGridIndexY(float y) {
    return static_cast<int>(y / static_cast<float>(BoostPadGridCellSizeY())) +
           (BoostPadGridCellsY() / 2);
}

ROCKETGPU_HD void CheckBoostPadCollisionForCar(
    int env_idx,
    const SimConfig& cfg,
    const CarState& car,
    uint32_t car_id,
    BoostPadState* pads) {
    if (!cfg.enable_boost_pads || car.is_demoed || car.boost >= cfg.mutator.boost_max) {
        return;
    }

    const int pad_base = env_idx * BoostPadCount(cfg);
    // The game's pickups land on this tick or the next at random, about evenly when the car enters the pad's
    // cylinder halfway through the tick (RLBot captures). Test the car's position halfway through the tick.
    const Vec3 car_bt = car.pos_bt - car.vel_bt * (0.5f * cfg.tick_dt);
    const Vec3 car_pos = car_bt * 50.0f;
    const int car_cell_x = BoostPadGridIndexX(car_pos.x);
    const int car_cell_y = BoostPadGridIndexY(car_pos.y);
    const bool car_in_boost_grid_z = car_pos.z <= BoostPadGridExtentZUU();
    const int visit_min_x = Max(car_cell_x - 1, 0);
    const int visit_max_x = Min(car_cell_x + 1, BoostPadGridCellsX() - 1);
    const int visit_min_y = Max(car_cell_y - 1, 0);
    const int visit_max_y = Min(car_cell_y + 1, BoostPadGridCellsY() - 1);
    constexpr float kUUToBT = 1.0f / 50.0f;
    for (int p = 0; p < BoostPadCount(cfg); ++p) {
        const Vec3 pad_pos = BoostPadLoc(cfg, p);
        // Loose bounds on BoostPadCheckCollideBt for either pad size, so most pads skip loading their state.
        const float near_dx = car_bt.x - pad_pos.x * kUUToBT;
        const float near_dy = car_bt.y - pad_pos.y * kUUToBT;
        if (near_dx * near_dx + near_dy * near_dy > (216.0f * kUUToBT) * (216.0f * kUUToBT) ||
            Abs(car_bt.z - pad_pos.z * kUUToBT) > (110.0f * kUUToBT)) {
            continue;
        }
        BoostPadState& pad = pads[pad_base + p];
        const int pad_cell_x = BoostPadGridIndexX(pad_pos.x);
        const int pad_cell_y = BoostPadGridIndexY(pad_pos.y);

        const bool official_grid_visit =
            cfg.use_custom_boost_pads != 0 ||
            (car_in_boost_grid_z &&
             pad_cell_x >= visit_min_x && pad_cell_x <= visit_max_x &&
             pad_cell_y >= visit_min_y && pad_cell_y <= visit_max_y);
        if (!official_grid_visit) {
            continue;
        }
        const bool colliding = BoostPadCheckCollideBt(pad_pos, pad, car_bt);
        if (colliding) {
            pad.cur_locked_car_id = car_id;
        }
    }
}

ROCKETGPU_HD void PostTickBoostPadsForEnv(int env_idx, const SimConfig& cfg, CarState* cars, BoostPadState* pads) {
    if (!cfg.enable_boost_pads) {
        return;
    }

    const int pad_base = env_idx * BoostPadCount(cfg);
    for (int p = 0; p < BoostPadCount(cfg); ++p) {
        BoostPadState& pad = pads[pad_base + p];
        const uint32_t locked_car_id = pad.cur_locked_car_id;
        if (locked_car_id != 0) {
            if (pad.is_active) {
                CarState& car = cars[locked_car_id - 1u];
                const float boost_amount = pad.is_big ? 100.0f : 12.0f;
                const float event_boost_before = car.boost;
                car.boost = Min(cfg.mutator.boost_max, car.boost + boost_amount);
                pad.is_active = 0;
                // RLBot packets show the pickup tick's timer already counted once (4 - 1/120).
                pad.cooldown = (pad.is_big ? cfg.mutator.boost_pad_cooldown_big : cfg.mutator.boost_pad_cooldown_small) -
                               cfg.tick_dt;
                if (cfg.mutator.physics_events.windows) {
                    PhysicsEvent event;
                    event.type = PhysicsEventType::BoostPickup;
                    event.car_id = locked_car_id;
                    event.pad_index = p;
                    event.is_big_pad = pad.is_big;
                    event.boost_before = event_boost_before;
                    event.boost_after = car.boost;
                    cfg.mutator.physics_events.Emit(env_idx, event);
                }
            }
        }
        pad.prev_locked_car_id = locked_car_id;
    }
}

ROCKETGPU_HD void RecordArenaWorldContact(CarState& car, const ContactAccumulator& acc) {
    if (!acc.hit) {
        return;
    }
    car.world_contact.has_contact = 1;
    car.world_contact.contact_normal = acc.latest_normal;
}

ROCKETGPU_HD void PreTickCar(
    CarState& car,
    CarControls controls,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    float dt,
    WheelRuntimeState* wheel_runtime_state = nullptr,
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
    bool staged_rays_velocity_stale = false) {
    ClampControls(controls);
    // The dodge direction and dodge pitch cancel quantize the negated raw pitch (v3).
    const float cal_dodge_raw_pitch = controls.pitch;
    controls = QuantizeAxisControls(controls);

    car.demo_no_contact_response = car.is_demoed ? 1 : 0;
    if (car.is_demoed) {
        return;
    }
    if (car.supersonic_set_v3) {
        car.supersonic_set_v3 = 0;
    } else {
        UpdateSupersonicCalDemo(car, mutator, dt);
    }

    if (wheel_runtime_state) {
        wheel_runtime_state->accumulated_force = Vec3(0.0f, 0.0f, 0.0f);
        wheel_runtime_state->accumulated_ang_accel = Vec3(0.0f, 0.0f, 0.0f);
        wheel_runtime_state->accumulated_torque_bt = Vec3(0.0f, 0.0f, 0.0f);
        wheel_runtime_state->pre_solve_rot = car.rot;
        wheel_runtime_state->pre_solve_basis_forward = car.basis_forward;
        wheel_runtime_state->pre_solve_basis_right = car.basis_right;
        wheel_runtime_state->pre_solve_basis_up = car.basis_up;
        wheel_runtime_state->pre_solve_basis_authoritative = car.basis_authoritative;
    }
    car.vel_touched_solve_stage = 0;
    const uint8_t flipping_at_start = car.is_flipping;
    bool flip_started = false;

    const bool jump_pressed = controls.jump && !car.last_controls.jump;
    const uint8_t cal_has_flipped_at_start = car.has_flipped;
    // v3 update_air_torque (base.rs:507-514) locks pitch while prev_flip_time < PITCHLOCK_EXTRA_TIME, where flip_time is
    // the restart timer once no flip is active. Tick-start value, as v3's prev_flip_time.
    // flip_time == 0 with no flip active means "unknown" (a set state without the timer, e.g. RocketSim-convention 0): no lock.
    const float cal_pitchlock_restart_time =
        (!cal_has_flipped_at_start && !car.is_flipping && car.internal_flip_time > 0.0f) ? car.internal_flip_time : -1.0f;

    bool cal_air_jump_started = false;

    const bool cal_jump = wheel_runtime_state != nullptr;
    const bool cal_jump_g = cal_jump &&
        (static_cast<int>(car.wheels_with_contact[0] != 0) + static_cast<int>(car.wheels_with_contact[1] != 0) +
         static_cast<int>(car.wheels_with_contact[2] != 0) + static_cast<int>(car.wheels_with_contact[3] != 0)) >= 3;
    if (cal_jump) {
        WheelRuntimeState& jw = *wheel_runtime_state;
        if (car.tick_count_since_update == 0) {
            jw.cal_game_jumped = car.has_jumped ? 1 : 0;
            jw.cal_game_double_jumped = (car.has_double_jumped || car.has_flipped) ? 1 : 0;
            jw.cal_cached_ground = cal_jump_g ? 1 : 0;
            jw.cal_jump_inactive_time = car.is_jumping ? 0.0f : car.internal_jump_time;
        }
        if (cal_jump_g && !jw.cal_cached_ground) {
            jw.cal_game_jumped = 0;
            jw.cal_game_double_jumped = 0;
        }
        jw.cal_cached_ground = cal_jump_g ? 1 : 0;
        if (jw.cal_game_jumped && cal_jump_g && !car.is_jumping && jw.cal_jump_inactive_time >= 0.125f) {
            jw.cal_game_jumped = 0;
            jw.cal_game_double_jumped = 0;
        }
    }
    const bool cal_jump_blocked = cal_jump && wheel_runtime_state->cal_game_jumped;
    const uint8_t was_jumping = car.is_jumping;
    if (jump_pressed && car.is_on_ground && !car.is_jumping && !cal_jump_blocked) {
        StartJump(car, mutator);
        cal_air_jump_started = true;
    }

    const bool cal_air_window_closed = cal_jump &&
        (wheel_runtime_state->cal_game_double_jumped ||
         (wheel_runtime_state->cal_game_jumped && wheel_runtime_state->cal_jump_inactive_time > mutator.doublejump_max_delay));
    const bool air_jump_pressed = jump_pressed && !(cal_jump && cal_jump_g) && !cal_air_window_closed;
    const bool cal_press_as_air = cal_jump && !cal_jump_g && jump_pressed && !cal_air_window_closed;
    const uint8_t air_jump_used_before = (car.has_double_jumped || car.has_flipped) ? 1 : 0;

    const int prev_wheel_contacts =
        static_cast<int>(car.wheels_with_contact[0] != 0) + static_cast<int>(car.wheels_with_contact[1] != 0) +
        static_cast<int>(car.wheels_with_contact[2] != 0) + static_cast<int>(car.wheels_with_contact[3] != 0);

    const bool autoroll_cal = wheel_runtime_state != nullptr;
    const bool autoroll_prev_contact = autoroll_cal && car.tick_count_since_update != 0;
    Vec3 autoroll_normal_sum =
        autoroll_prev_contact ? wheel_runtime_state->cal_prev_wheel_normal_sum : Vec3(0.0f, 0.0f, 0.0f);
    int autoroll_world_contacts = autoroll_prev_contact ? wheel_runtime_state->cal_prev_world_wheel_contacts : 0;

    // v3 Car::pre_tick_update runs update_jump / update_double_jump_or_flip (dodge and double-jump impulses, using the
    // tick-start forward speed and the previous tick's is_on_ground) before bullet_vehicle.update, whose raycast,
    // suspension and friction then read the post-impulse chassis velocity. Run the flip/double-jump update here, as the
    // jump start above already is.
    bool cal_flip_before_wheels_done = false;
    bool cal_flip_before_wheels_vel_changed = false;
    float cal_flip_before_wheels_forward_speed = 0.0f;
    {
        cal_flip_before_wheels_forward_speed = WheelForwardSpeed(car);
        const Vec3 fbw_vel_bt = car.vel_bt;
        UpdateDoubleJumpOrFlipState(car, controls, cfg, mutator, air_jump_pressed && !cal_air_jump_started,
                                    cal_flip_before_wheels_forward_speed, dt, cal_press_as_air && !cal_air_jump_started,
                                    cal_dodge_raw_pitch, &flip_started);
        cal_flip_before_wheels_done = true;
        cal_flip_before_wheels_vel_changed =
            car.vel_bt.x != fbw_vel_bt.x || car.vel_bt.y != fbw_vel_bt.y || car.vel_bt.z != fbw_vel_bt.z;
    }

    UninitArray<WheelStepState, 4> wheels;
    float forward_speed_uu = 0.0f;
    const int num_wheels_in_contact = PreTickVehicleDynamics(
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
        cal_flip_before_wheels_done ? &cal_flip_before_wheels_forward_speed : nullptr);

    for (int wheel = 0; wheel < 4; ++wheel)
        if (wheels[wheel].is_in_contact && wheels[wheel].is_in_contact_with_world) {
            car.RecordWorldSurface(false);
            break;
        }
    if (car.HasFlipOrJump() && !car.flip_or_jump_last_tick)
        car.last_flip_or_jump_start = car.tick_count_since_update;

    if (!cal_flip_before_wheels_done) {
        UpdateDoubleJumpOrFlipState(car, controls, cfg, mutator, air_jump_pressed && !cal_air_jump_started, forward_speed_uu, dt,
                                    cal_press_as_air && !cal_air_jump_started, cal_dodge_raw_pitch, &flip_started);
    }
    // v3 Car::pre_tick_update gates the dodge torque (and the is_flipping reset) on num_wheels_in_contact read before
    // bullet_vehicle.update, i.e. the previous tick's wheel contacts and the matching cached is_on_ground; this tick's
    // raycasts only take effect next tick. The tick after a state set keeps the current-raycast rule (a set state
    // carries no reliable previous wheel contacts).
    // The dodge-press tick keeps the current-raycast rule: user-suite capture c3 car1 t96761 (press with last step < 3
    // contacts, this tick's raycast grounded) regresses when the prev-contact gate also covers the press tick.
    const bool cal_flip_started_this_tick = car.has_flipped && !cal_has_flipped_at_start;
    const int air_gate_wheel_contacts =
        (car.tick_count_since_update != 0 && !cal_flip_started_this_tick) ? prev_wheel_contacts :
        ((prev_wheel_contacts > num_wheels_in_contact) ? prev_wheel_contacts : num_wheels_in_contact);

    // v3 gates update_air_torque on the cached contacts a set state carries
    // (CarState::set_contact_flags, base.rs:964/997), also on the first tick after it. Only a set state that says
    // grounded suppresses air control here (max with this tick's raycasts, like air_gate_wheel_contacts above):
    // RocketSim-convention set states mark landings airborne from a coarse height test.
    const int landing_prev_contacts = (car.tick_count_since_update == 0)
        ? (Max(num_wheels_in_contact, prev_wheel_contacts))
        : prev_wheel_contacts;
    // v3 update_double_jump_or_flip returns flip_ended when the flip timer runs out this tick, and pre_tick_update
    // skips auto-roll on that tick (the car still counts as flipping for auto-roll on its last flip tick).
    const uint8_t cal_flipping_before_air = car.is_flipping;
    if (air_gate_wheel_contacts < 3) {
           ApplyFlipTorque(car, controls, cfg, mutator, dt, wheel_runtime_state, cal_dodge_raw_pitch);

        ApplyAirControl(car, controls, cfg, mutator, dt, landing_prev_contacts == 0, wheel_runtime_state, landing_prev_contacts == 0, cal_pitchlock_restart_time);
    } else {
        car.is_flipping = 0;

        if (landing_prev_contacts == 0) {
            ApplyAirControl(car, controls, cfg, mutator, dt, true, wheel_runtime_state, true, cal_pitchlock_restart_time);
        }
    }
    const bool cal_flip_ended = air_gate_wheel_contacts < 3 &&
        cal_flipping_before_air && !car.is_flipping;

    UpdateJumpState(car, controls, mutator, jump_pressed, dt, wheel_runtime_state, cal_jump_blocked);

    ApplyAutoFlip(car, controls, mutator, dt);
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

    // Report v3's jump_ticks and flip_time (car/base.rs update_jump, update_double_jump_or_flip, pre_tick_update): each
    // reads one tick on the tick its activity changes (jump start or end; flip start, end or landing mid-flip) and
    // otherwise advances one tick, on the ground too. v3 skips the car update on the tick a demoed car respawns.
    // jump_time adds dt like the internal jump clocks for its first 1024 ticks (a set state then restores them bit for
    // bit), then holds ticks * dt so it stays exact.
    const bool jump_changed = cal_air_jump_started || (was_jumping && !car.is_jumping);
    const bool flip_changed = flip_started || car.is_flipping != flipping_at_start;
    if (jump_changed || !car.just_respawned) {
        const float jump_ticks = rintf(car.jump_time / dt) + 1.0f;
        car.jump_time = jump_changed ? dt : (jump_ticks <= 1024.0f ? car.jump_time + dt : jump_ticks * dt);
    }
    if (flip_changed || !car.just_respawned) {
        car.flip_time = flip_changed ? dt : car.flip_time + dt;
    }
    if (cal_air_jump_started) {
        car.jump_hold_broken = 0;
    } else if (was_jumping && !controls.jump) {
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
    // On the first tick after a set state, v3 Car::pre_tick_update gates auto-roll on
    // the cached wheel contacts the set state carries (base.rs:966/1011, 0 < n < 4 or a world-contact normal), and
    // update_auto_roll then needs the sticky gate refreshed from the set pose (base.rs:797/801) for the wheel-normal
    // up direction. RocketGPU used this tick's raycast count for both. Only a set state with partial wheel contacts
    // (1-3 wheels) is taken as per-wheel data: RocketSim-convention set states mark all four wheels from one coarse
    // ground flag, and for those this tick's raycast stays the better estimate (user-suite landing fixtures).
    if (autoroll_cal && !autoroll_prev_contact &&
        prev_wheel_contacts > 0 && prev_wheel_contacts < 4) {
        const bool sticky = autoroll_world_contacts > 0;
        autoroll_gate = sticky || car.world_contact.has_contact;
        autoroll_contacts = sticky ? prev_wheel_contacts : 0;
    }
    if (controls.throttle && !(car.is_flipping) && !cal_flip_ended && autoroll_gate) {
        ApplyAutoRollReference(car, cfg, mutator, autoroll_contacts, wheels, dt, wheel_runtime_state,
                               autoroll_cal ? &autoroll_normal_sum : nullptr);
    }

    car.world_contact.has_contact = 0;

    if (!car.just_respawned) {
        ApplyWheelSuspensionAndFriction(car, cfg, mutator, controls, dt, wheels);
    }
    ApplyBoost(car, controls, mutator, dt, wheel_runtime_state);
    if (wheel_runtime_state) {
        wheel_runtime_state->solver_time_step = dt;
        if (env_idx >= 0 && self_car_idx >= 0) {
            int selected = -1;
            for (int wheel = 0; wheel < 4; ++wheel)
                if (wheels[wheel].is_in_contact && wheels[wheel].is_in_contact_with_world
                    && (selected < 0 || wheels[wheel].contact_normal_ws.z > wheels[selected].contact_normal_ws.z))
                    selected = wheel;
            bool previous = false;
            if (selected >= 0 && car.tick_count_since_update != 0) {
                const auto normal = wheels[selected].contact_normal_ws;
                for (int wheel = 0; wheel < 4; ++wheel) {
                    const auto old = wheel_runtime_state->event_world_normals[wheel];
                    previous |= wheel_runtime_state->event_world_ids[wheel] == wheels[selected].ground_static_id
                        && old.x == normal.x && old.y == normal.y && old.z == normal.z;
                }
            }
            if (selected >= 0 && !previous && mutator.physics_events.windows) {
                PhysicsEvent event;
                event.type = PhysicsEventType::CarWorldContact;
                event.car_id = env_idx * mutator.physics_events.cars_per_env + self_car_idx + 1;
                event.position = car.pos;
                event.car_velocity = car.vel;
                event.car_on_ground = car.is_on_ground;
                event.normal = wheels[selected].contact_normal_ws;
                mutator.physics_events.Emit(env_idx, event);
            }
            for (int wheel = 0; wheel < 4; ++wheel) {
                const bool world = wheels[wheel].is_in_contact && wheels[wheel].is_in_contact_with_world;
                wheel_runtime_state->event_world_ids[wheel] = world ? wheels[wheel].ground_static_id : -1;
                wheel_runtime_state->event_world_normals[wheel] = world ? wheels[wheel].contact_normal_ws : Vec3{};
            }
        }
    }
}

ROCKETGPU_HD bool ShouldSleepBallAtTickStart(const BallState& ball, const MutatorConfig& mutator);

ROCKETGPU_HD bool ShouldSleepBallAtTickStart(const BallState& ball) {
    return LengthSq(ball.vel) == 0.0f && LengthSq(ball.ang_vel) == 0.0f;
}

// v3 never sleeps a ball because its velocity is exactly zero: a state set activates the ball and resets its
// deactivation timer (Arena::set_ball_state), so a resting or hanging ball still gets gravity and contacts.
ROCKETGPU_HD bool ShouldSleepBallAtTickStart(const BallState& ball, const MutatorConfig& mutator) {
    return false;
}

// Settled-ball fast path (cal_ball_wake_on_set). With no ball-car rows, the rest of a world substep acts on the
// ball only through its static-world stage S: ball-world contact rows (AppendBallWorldSpecialContactRows), their
// velocity/split solve, gravity and integration. S reads the ball, the retained ball-mesh manifolds, the solver's
// damped ball velocity, the ball's world (kind 0) entries in the sorted manifold list, sub_dt and the mutator,
// and nothing that belongs to a car; car rows never read or write the ball. When S returns its input ball
// bit-identical (a ball resting on a plane), a later substep whose S inputs are bit-identical would compute the
// same identity, so it skips S and restores the verified ball. Car results do not change: their rows, order and
// solver paths are the same with or without the ball's world rows (both solvers stay on the unified order
// because the ball's world entries remain in the manifold list).
ROCKETGPU_HD bool SettleBitsEqual(float a, float b) {
    return GlibcFloatBits(a) == GlibcFloatBits(b);
}

ROCKETGPU_HD bool SettleBitsEqual(const Vec3& a, const Vec3& b) {
    return SettleBitsEqual(a.x, b.x) && SettleBitsEqual(a.y, b.y) && SettleBitsEqual(a.z, b.z);
}

// Every BallState field except tick_count_since_update, which only FinishBallTick advances (no step stage reads
// it), so a settled ball keeps settling while its tick counter runs.
ROCKETGPU_HD bool BallSettleStateEqual(const BallState& a, const BallState& b) {
    static_assert(sizeof(BallState) == 200, "BallState changed: compare the new fields in BallSettleStateEqual");
    return SettleBitsEqual(a.pos, b.pos) &&
           SettleBitsEqual(a.vel, b.vel) &&
           SettleBitsEqual(a.ang_vel, b.ang_vel) &&
           SettleBitsEqual(a.pos_bt, b.pos_bt) &&
           SettleBitsEqual(a.vel_bt, b.vel_bt) &&
           SettleBitsEqual(a.rot.w, b.rot.w) && SettleBitsEqual(a.rot.x, b.rot.x) &&
           SettleBitsEqual(a.rot.y, b.rot.y) && SettleBitsEqual(a.rot.z, b.rot.z) &&
           SettleBitsEqual(a.basis_forward, b.basis_forward) &&
           SettleBitsEqual(a.basis_right, b.basis_right) &&
           SettleBitsEqual(a.basis_up, b.basis_up) &&
           a.basis_authoritative == b.basis_authoritative &&
           SettleBitsEqual(a.vel_impulse_cache, b.vel_impulse_cache) &&
           SettleBitsEqual(a.vel_impulse_cache_bt, b.vel_impulse_cache_bt) &&
           a.tick_count_when_extra_impulse_applied == b.tick_count_when_extra_impulse_applied &&
           SettleBitsEqual(a.hs_info.y_target_dir, b.hs_info.y_target_dir) &&
           SettleBitsEqual(a.hs_info.cur_target_speed, b.hs_info.cur_target_speed) &&
           SettleBitsEqual(a.hs_info.time_since_hit, b.hs_info.time_since_hit) &&
           a.ds_info.charge_level == b.ds_info.charge_level &&
           SettleBitsEqual(a.ds_info.accumulated_hit_force, b.ds_info.accumulated_hit_force) &&
           SettleBitsEqual(a.ds_info.y_target_dir, b.ds_info.y_target_dir) &&
           a.ds_info.has_damaged == b.ds_info.has_damaged &&
           a.ds_info.reserved0 == b.ds_info.reserved0 &&
           a.ds_info.reserved1 == b.ds_info.reserved1 &&
           a.ds_info.reserved2 == b.ds_info.reserved2 &&
           a.ds_info.last_damage_tick == b.ds_info.last_damage_tick;
}

ROCKETGPU_HD bool BallWorldMeshManifoldsEmpty(const WorldSolverScratch& scratch) {
    for (int i = 0; i < kMaxBallWorldMeshPersistentManifolds; ++i) {
        if (scratch.ball_world_mesh_manifolds[i].count != 0) {
            return false;
        }
    }
    return true;
}

// Moves each car's world rows from its slot (CarWorldRowsOut) up behind the previous car's, and appends its order
// groups: the shared rows and groups SolveCarWorldStaticContacts appends car by car in the WorldStepSoccar loop.
ROCKETGPU_HD void PackCarWorldRows(
    int count,
    const CarWorldRowsOut* outs,
    const WorldSolverScratchRows& rows,
    int& split_row_count,
    int& velocity_row_count,
    int& friction_row_count,
    int& group_count) {
    for (int i = 0; i < count; ++i) {
        const CarWorldRowsOut& out = outs[i];
        const int split_from = i * kMaxCarWorldSplitContactRows;
        const int velocity_shift = i * kMaxCarWorldVelocityContactRows - velocity_row_count;
        const int friction_shift = i * kMaxCarWorldFrictionContactRows - friction_row_count;
        for (int k = 0; split_from != split_row_count && k < out.split_row_count; ++k) {
            rows.world_split_rows[split_row_count + k] = rows.world_split_rows[split_from + k];
        }
        for (int k = 0; velocity_shift != 0 && k < out.velocity_row_count; ++k) {
            rows.world_velocity_rows[velocity_row_count + k] = rows.world_velocity_rows[velocity_row_count + velocity_shift + k];
        }
        for (int k = 0; (friction_shift != 0 || velocity_shift != 0) && k < out.friction_row_count; ++k) {
            CarSharedFrictionContactRow& row = rows.world_friction_rows[friction_row_count + k];
            if (friction_shift != 0) {
                row = rows.world_friction_rows[friction_row_count + friction_shift + k];
            }
            row.normal_row_idx -= velocity_shift;
        }
        for (int g = 0; g < out.group_count && group_count < kMaxSharedContactOrderGroups; ++g) {
            SharedContactOrderGroup group = out.groups[g];
            if (group.normal_row_start >= 0) {
                group.normal_row_start -= velocity_shift;
            }
            if (group.friction_row_start >= 0) {
                group.friction_row_start -= friction_shift;
            }
            rows.shared_contact_order_groups[group_count++] = group;
        }
        split_row_count += out.split_row_count;
        velocity_row_count += out.velocity_row_count;
        friction_row_count += out.friction_row_count;
    }
}

// WorldStepSoccar parts. Contacts runs up to the car-world rows and saves its locals (WorldStepCarry); a thread per
// car then builds those rows (BuildCarWorldRowsForCar), and Rows packs them in car order and runs the rest.
enum : int { kWorldStepWhole = 0, kWorldStepContacts = 1, kWorldStepRows = 2 };

ROCKETGPU_HD void WorldStepSoccar(
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
    int world_part = kWorldStepWhole) {
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
    CarCarPairCachedContact* const pair_contact_cache = solver_scratch->pair_contact_cache;

    const auto should_stop_after_stage = [&](uint8_t stage) {
        return 0 != 0 && 0 <= stage;
    };
    const auto should_stop_after_world_car = [&](int local_car_index) {
        return false;
    };
    for (int substep = 0; substep < world_substeps; ++substep) {
        WorldStepCarry carry;
        if (world_part == kWorldStepRows) {
            carry = solver_scratch->world_step_carry;
        } else {
            carry.ball_sleeping = ShouldSleepBallAtTickStart(ball, mutator);
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
        if (world_part != kWorldStepRows) {
            solver_scratch->shared_contact_order_group_count = 0;

            for (int pair_idx = 0; pair_idx < kMaxCarCarPairSlots; ++pair_idx) {
                CarCarPairCachedContact& slot = pair_contact_cache[pair_idx];
                // Points are only written with slot.valid set (or cleared whole), so an invalid slot is already clear.
                if (slot.valid == 0) {
                    continue;
                }
                slot.valid = 0;
                slot.contact_count = 0;
                for (int k = 0; k < kMaxCarCarCachedPoints; ++k) {
                    slot.contacts[k].valid = 0;
                    slot.contacts[k].tangent_valid = 0;
                }
            }
            for (int i = 0; i < count; ++i) {
                CalPreStepClampCarSpeed(cars[base + i], mutator);

                pre_contact_cars[i] = MakePreContactCar(cars[base + i]);
                const WheelRuntimeState* wheel_runtime_state = wheel_runtime_states ? &wheel_runtime_states[base + i] : nullptr;
                deferred_linear_deltas[i] = ComputeDeferredLinearVelocityDelta(cars[base + i], wheel_runtime_state, mutator, sub_dt);

                cars[base + i].ext_force_impulse_bt =
                    (cars[base + i].is_demoed || cars[base + i].just_respawned)
                        ? ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f)
                        : ComputeOfficialExternalForceImpulseBt(wheel_runtime_state, mutator, sub_dt);
                cars[base + i].ext_force_impulse_bt_valid = 1;

                cars[base + i].solve_delta_linear_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
                cars[base + i].solve_delta_angular = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
                {
                    ::rocketgpu::Vec3 ext_torque(0.0f, 0.0f, 0.0f);
                    if (wheel_runtime_state && !(cars[base + i].is_demoed || cars[base + i].just_respawned)) {
                        ::rocketgpu::Vec3 inv_r0, inv_r1, inv_r2;
                        BuildPreSolveInvInertiaTensorWorldBt(
                            *wheel_runtime_state, car_configs[base + i], mutator.car_mass, inv_r0, inv_r1, inv_r2);
                        ext_torque = VecMul3x3(wheel_runtime_state->accumulated_torque_bt, inv_r0, inv_r1, inv_r2) * sub_dt;
                    }
                    cars[base + i].ext_torque_impulse = ext_torque;
                }
                deferred_angular_deltas[i] = ComputeDeferredAngularVelocityDelta(
                    cars[base + i], car_configs[base + i], mutator.car_mass, wheel_runtime_state, sub_dt);
            }

            ball_solver_linear_velocity_bt = ball.vel_bt;
            // v3 Ball::on_hit reads the ball's state.phys velocity, synced at tick start (captured in the tick-start block,
            // before any car pre-tick wheel push or heatseeker pre-tick) and before the Bullet step damps it.
            const Vec3 cal_ball_start_vel = solver_scratch != nullptr ? solver_scratch->cal_ball_start_vel : ball.vel;

            if (!ball_sleeping) {
                ball_solver_linear_velocity_bt = BulletDampedLinearVelocityFromBt(ball.vel_bt, mutator, sub_dt);
                ApplyBulletLinearDampingBall(ball, mutator, sub_dt);
            }

            for (int i = 0; i < count; ++i) {
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
                    i,
                    static_cast<int8_t>(teams[base + i]),
                    (i < 8) ? &solver_scratch->cal_start_vel[i] : nullptr,
                    (&cal_ball_start_vel));
            }

            int prebuilt_manifold_count = -1;
            if (enable_car_car_collisions) {
                const bool shared_rows_active =
                    pair_velocity_rows && pair_friction_rows;
                const int car_car_passes = shared_rows_active ? 1 : solver_passes;

                if (broadphase_grid.valid() && mesh_bvh_meshes != nullptr && mesh_bvh_mesh_count > 0) {
                    prebuilt_manifold_count = BuildBulletManifoldList(
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
                        NativeSseMath(mutator));
                }
                int8_t pair_order_i[28];
                int8_t pair_order_j[28];
                int pair_order_count = 0;
                uint8_t pair_listed[8][8] = {};
                for (int e = 0; e < prebuilt_manifold_count; ++e) {
                    const BulletManifoldEntry& en = rows.bullet_manifolds[e];
                    if (en.kind != 3) {
                        continue;
                    }
                    const int pi = Min((int)en.body_a, (int)en.body_b);
                    const int pj = Max((int)en.body_a, (int)en.body_b);
                    if (pi < 0 || pj >= count || pj >= 8 || pi == pj || pair_listed[pi][pj]) {
                        continue;
                    }
                    pair_listed[pi][pj] = 1;
                    pair_order_i[pair_order_count] = static_cast<int8_t>(pi);
                    pair_order_j[pair_order_count] = static_cast<int8_t>(pj);
                    ++pair_order_count;
                }
                for (int pi = 0; pi < count && pi < 8; ++pi) {
                    for (int pj = pi + 1; pj < count && pj < 8; ++pj) {
                        if (!pair_listed[pi][pj] && pair_order_count < 28) {
                            pair_order_i[pair_order_count] = static_cast<int8_t>(pi);
                            pair_order_j[pair_order_count] = static_cast<int8_t>(pj);
                            ++pair_order_count;
                        }
                    }
                }
                for (int pass = 0; pass < car_car_passes; ++pass) {
                    for (int po = 0; po < pair_order_count; ++po) {
                        const int i = pair_order_i[po];
                        const int j = pair_order_j[po];
                        {
                            const int pair_velocity_rows_before = pair_velocity_row_count;
                            const int pair_friction_rows_before = pair_friction_row_count;
                            ResolveCarCarCollision(
                                env_idx,
                                cars[base + j], static_cast<uint32_t>(base + j + 1), teams[base + j],
                                cars[base + i], static_cast<uint32_t>(base + i + 1), teams[base + i],
                                car_configs[base + j], car_configs[base + i],
                                mutator,
                                pre_contact_cars[j],
                                pre_contact_cars[i],
                                deferred_linear_deltas[j],
                                deferred_linear_deltas[i],
                                deferred_angular_deltas[j],
                                deferred_angular_deltas[i],
                                sub_dt,
                                enable_demos,
                                j,
                                i,
                                pair_contact_cache,
                                pair_split_rows,
                                &pair_split_row_count,
                                pair_velocity_rows,
                                &pair_velocity_row_count,
                                pair_friction_rows,
                                &pair_friction_row_count,
                                (&solver_scratch->cal_start_vel[j]),
                                (&solver_scratch->cal_start_vel[i]),
                                (i < j && j < 8)
                                    ? &solver_scratch->car_car_retained_manifolds[CarCarRetainedSlot(i, j)]
                                    : nullptr);
                            if (pair_velocity_row_count > pair_velocity_rows_before) {
                                AppendSharedContactOrderGroup(
                                    rows.shared_contact_order_groups,
                                    solver_scratch->shared_contact_order_group_count,
                                    SharedContactOrderBucket::Pair,
                                    j,
                                    i,
                                    j * 8 + i,
                                    pair_velocity_rows_before,
                                    pair_velocity_row_count - pair_velocity_rows_before,
                                    pair_friction_rows_before,
                                    pair_friction_row_count - pair_friction_rows_before,
                                    j * 100000 + 2000 + i);
                            }
                        }
                    }
                }
            }
            if (should_stop_after_stage(1)) {
                return;
            }
            for (int i = 0; i < count; ++i) {
                if (i < kMaxBallCarRetainedManifolds) {
                    AppendBallCarContactRowsRetained(
                        ball,
                        cars[base + i],
                        static_cast<uint32_t>(base + i + 1),
                        car_configs[base + i],
                        mutator,
                        current_tick,
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
                        static_cast<int8_t>(teams[base + i]),
                        solver_scratch->ball_car_retained_manifolds[i]);
                } else
                AppendBallCarContactRows(
                    ball,
                    cars[base + i],
                    static_cast<uint32_t>(base + i + 1),
                    car_configs[base + i],
                    mutator,
                    current_tick,
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
                    static_cast<int8_t>(teams[base + i]));
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
                    NativeSseMath(mutator));
                if (built > 1) {
                    int order[kMaxBulletManifoldEntries];
                    int island_ids[kMaxBulletManifoldEntries];
                    for (int i = 0; i < built; ++i) {
                        order[i] = i;
                        island_ids[i] = rows.bullet_manifolds[i].island_id;
                    }
                    QuickSortSharedContactOrderIndices(order, island_ids, 0, built - 1);

                    UninitArray<BulletManifoldEntry, kMaxBulletManifoldEntries> sorted;
                    for (int i = 0; i < built; ++i) {
                        sorted[i] = rows.bullet_manifolds[order[i]];
                    }
                    for (int i = 0; i < built; ++i) {
                        rows.bullet_manifolds[i] = sorted[i];
                    }
                }
                sorted_manifold_count = built;
            }
            solver_scratch->bullet_manifold_count = sorted_manifold_count;

            ball_awake_after_contact_rows =
                !ball_sleeping || split_row_count > 0 || normal_row_count > 0 || friction_row_count > 0;

            // Settled-ball fast path (see BallSettleStateEqual). Only a still ball with no ball-car rows, empty retained
            // mesh manifolds and 1..kBallSettledMaxWorldEntries world entries is a candidate; anything else clears it.
            if (solver_scratch != nullptr) {
                int16_t settle_world_ids[kBallSettledMaxWorldEntries] = {};
                int settle_world_count = 0;
                bool settle_candidate =
                    !ball_sleeping && mutator.arena_collision != 0 &&
                    mutator.game_mode == static_cast<uint8_t>(Mode::Soccar) && !mutator.physics_events.windows &&
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

            if (!ball_sleeping && !ball_settled_skip && mutator.arena_collision) {
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
                    &ball_solver_linear_velocity_bt, env_idx);
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
            if (world_part == kWorldStepContacts) {
                solver_scratch->world_step_carry = carry;
                return;
            }
        }
        // A verified fixed point needs a ball world velocity row, so the velocity solver writes the ball back
        // whether or not cars add rows. Without ball split rows the split solver leaves the ball untouched
        // (zero push and turn), so its ball split rows may be absent.
        const bool ball_settled_rows_ok = normal_row_count > 0;

        if (world_part == kWorldStepRows) {
            PackCarWorldRows(count, solver_scratch->car_world_rows_out, rows, world_split_row_count,
                             world_velocity_row_count, world_friction_row_count,
                             solver_scratch->shared_contact_order_group_count);
        }
        for (int i = 0; world_part != kWorldStepRows && i < count; ++i) {
            const int car_idx = base + i;
            if (pre_contact_cars[i].is_demoed || cars[car_idx].just_respawned) {
                if (i < kMaxCarWorldMeshPersistentCars) {
                    for (int plane_idx = 0; plane_idx < kCarWorldPlaneRetainedManifoldsPerCar; ++plane_idx) {
                        solver_scratch->car_world_plane_manifolds[i * kCarWorldPlaneRetainedManifoldsPerCar + plane_idx].count = 0;
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
                    ? &solver_scratch->car_world_plane_manifolds[i * kCarWorldPlaneRetainedManifoldsPerCar]
                    : nullptr);
            RecordArenaWorldContact(cars[car_idx], acc);
            if (should_stop_after_world_car(i)) {
                return;
            }
        }
        if (should_stop_after_stage(5)) {
            return;
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
            env_idx,
            current_tick,
            substep,
            &ball_solver_linear_velocity_bt,
            rows.bullet_manifolds,
            sorted_manifold_count);
        if (should_stop_after_stage(4)) {
            return;
        }
        if (should_stop_after_stage(6)) {
            return;
        }

        UpdateCarCarPairCacheFromSharedVelocityRows(
            pair_contact_cache,
            pair_velocity_rows,
            pair_velocity_row_count,
            pair_friction_rows,
            pair_friction_row_count);

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
            current_tick,
            substep,
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

        for (int i = 0; i < count; ++i) {
            if (pre_contact_cars[i].is_demoed) {
                continue;
            }
            ApplyDeferredVelocityDeltas(
                cars[base + i], car_configs[base + i], deferred_linear_deltas[i], deferred_angular_deltas[i],
                wheel_runtime_states ? &wheel_runtime_states[base + i] : nullptr, mutator, sub_dt);
        }
        if (should_stop_after_stage(8)) {
            return;
        }

        if (ball_settled_skip) {
            // The skipped stage returns this ball unchanged; the car solve may have rewritten it in between.
            const uint64_t ball_ticks = ball.tick_count_since_update;
            ball = solver_scratch->ball_settled_state;
            ball.tick_count_since_update = ball_ticks;
        } else if (ball_awake_after_contact_rows) {
            IntegrateBallTransformOnly(ball, mutator, sub_dt);
        }
        if (ball_settled_verify) {
            solver_scratch->ball_settled =
                (ball_settled_rows_ok && BallWorldMeshManifoldsEmpty(*solver_scratch) &&
                 BallSettleStateEqual(ball, solver_scratch->ball_settled_state)) ? 1 : 0;
        }

        for (int i = 0; i < count; ++i) {
            if (pre_contact_cars[i].is_demoed || cars[base + i].just_respawned) {
                continue;
            }
            IntegrateCarTransformOnly(cars[base + i], sub_dt, mutator.game_mode == static_cast<uint8_t>(Mode::Soccar), NativeSseMath(mutator));
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
        return controls_by_tick[tick * stride + base + local_car_idx];
    }
};

struct NoopEnvCommandAccessor {
    ROCKETGPU_HD EnvCommand get(int tick) const {
        (void)tick;
        return EnvCommand{};
    }
};

struct StreamEnvCommandAccessor {
    const EnvCommand* commands_by_tick = nullptr;
    int base = 0;
    int stride = 0;

    ROCKETGPU_HD EnvCommand get(int tick) const {
        if (!commands_by_tick) {
            return EnvCommand{};
        }
        return commands_by_tick[tick * stride + base];
    }
};

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
    const int* kickoff_order_override = nullptr) {
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
        kickoff_order_override);

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
}

ROCKETGPU_HD void RocketSim_Car_PreTickUpdate(
    CarState& car,
    CarControls controls,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    float dt,
    WheelRuntimeState* wheel_runtime_state,
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
    bool staged_rays_velocity_stale = false) {
    PreTickCar(
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
        staged_rays_velocity_stale);
}

ROCKETGPU_HD void RocketSim_Car_PostTickUpdate(
    CarState& car,
    CarControls controls,
    const MutatorConfig& mutator,
    float dt) {
    PostTickCar(car, controls, mutator, dt);
}

ROCKETGPU_HD void RocketSim_Car_FinishPhysicsTick(CarState& car, const MutatorConfig& mutator) {
    FinishCarTick(car, mutator);
}

// End of tick: keep the step's values in the internal_* fields and write RLBot's values into the reported ones
// (rocketgpu.h CarState). StepEnvPreTickCar puts the step's values back.
ROCKETGPU_HD void ReportRLBotCarFields(CarState& car, const WheelRuntimeState* jw, const MutatorConfig& mutator, float dt) {
    if (car.is_demoed && car.tick_count_since_update == 0) {
        // Demoed before its first tick after a set: the next tick start still reads the fields as a set state.
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
    // The game's has_jumped: set by a jump, cleared on a tick in the OnGround air state once the jump started more than
    // JUMP_MIN_TIME + JUMP_RESET_TIME_PAD ago. RocketGPU checks its own flag before the jump ends in the tick (so a car
    // that releases the jump on the ground and leaves it next tick keeps it), and also clears it while dodging on the
    // ground.
    const bool on_ground_state = car.is_on_ground && !car.is_jumping && !car.is_flipping;
    if (car.is_jumping) {
        car.game_has_jumped = 1;
    } else if (on_ground_state && (!car.has_jumped ||
               car.internal_jump_time > mutator.jump_min_time + mutator.jump_reset_time_pad)) {
        car.game_has_jumped = 0;
    }

    // Velocity caps, as the next tick start applies them (CalTickStartLimitAndQuantizeCar, before quantizing).
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
    // The game's jump inactive time (WheelRuntimeState::cal_jump_inactive_time) counts the release tick as one tick.
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

// Tick start: the step's values back into the reported fields. After a state set (tick_count_since_update == 0) the
// written fields are the step's values; the game's supersonic flag still holds while supersonic_time runs.
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

ROCKETGPU_HD void RocketSim_Ball_FinishPhysicsTick(BallState& ball, const MutatorConfig& mutator) {
    FinishBallTick(ball, mutator);
}

ROCKETGPU_HD void RocketSim_BoostPad_PreTickUpdate(int env_idx, const SimConfig& cfg, BoostPadState* pads, float dt) {
    PreTickBoostPadsForEnv(env_idx, cfg, pads, dt);
}

ROCKETGPU_HD void RocketSim_BoostPad_CheckCollide(
    int env_idx,
    const SimConfig& cfg,
    const CarState& car,
    uint32_t car_id,
    BoostPadState* pads) {
    CheckBoostPadCollisionForCar(env_idx, cfg, car, car_id, pads);
}

ROCKETGPU_HD void RocketSim_BoostPad_PostTickUpdate(int env_idx, const SimConfig& cfg, CarState* cars, BoostPadState* pads) {
    PostTickBoostPadsForEnv(env_idx, cfg, cars, pads);
}

ROCKETGPU_HD void RocketSim_Arena_WorldStepSoccar(
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
    int world_part = kWorldStepWhole) {
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
        world_part);
}

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
    int count) {
    constexpr int kStageCars = kWheelRayStageSlots / 4;
    if (lane < 0 || lane >= kWheelRayStageSlots) {
        return;
    }
    WheelRayStage& stage = scratch.wheel_ray_stage[lane];
    stage.valid = 0;
    const int car_slot = lane / 4;
    const int wheel_idx = lane % 4;
    if (car_slot >= count || car_slot >= cfg.max_cars_per_env || count > kStageCars) {
        return;
    }
    const CarState& car = cars[base + car_slot];
    if (car.is_demoed) {
        return;
    }
    uint8_t no_response[kStageCars];
    for (int j = 0; j < count; ++j) {
        no_response[j] = cars[base + j].demo_no_contact_response;
    }
    for (int k = 0; k < count; ++k) {
        const int j = CarVisitSlot(env, k, count);
        if (j == car_slot) {
            break;
        }
        if (j >= 0 && j < count) {
            no_response[j] = cars[base + j].is_demoed ? 1 : 0;
        }
    }
    const CarConfig& car_cfg = car_configs[base + car_slot];
    WheelStepState wheel = MakeWheelStepState(car, car_cfg, wheel_idx);
    wheel.extra_pushback = wheel_runtime_states ? wheel_runtime_states[base + car_slot].extra_pushback[wheel_idx] : 0.0f;
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
        env_idx,
        wheel_idx,
        &balls[env_idx],
        cfg.mutator.ball_radius,
        cfg.arena_mesh_bvh_quantization,
        cfg.arena_mesh_bvh_quantization_count,
        cfg.arena_mesh_bvh_meshes,
        cfg.arena_mesh_bvh_mesh_count,
        cfg.arena_mesh_bvh_nodes,
        cfg.arena_mesh_bvh_node_count,
        cfg.arena_mesh_bvh_subtrees,
        cfg.arena_mesh_bvh_subtree_count,
        cfg.mutator.arena_collision != 0, cfg.mutator.native_sphere_rsqrt_table, cfg.mutator.native_sphere_rsqrt_shift,
        no_response, CalMinWheelHitDistSqBt(car_cfg));
    StoreWheelRayStage(wheel, stage);
    stage.valid = 1;
}

// Full runs a whole tick; Post skips the tick start (StepEnvPreTick*) and reads the staged wheel rays;
// PostVehiclesDone is Post after VehicleUpdateKernel ran the independent vehicle updates.
enum : int { kStepEnvPhaseFull = 0, kStepEnvPhasePost = 2, kStepEnvPhasePostVehiclesDone = 3 };

// Vehicle update of car slot i at the start of the physics tick (RocketSim_Car_PreTickUpdate). A wheel resting on
// another car or the ball pushes that body, which the cars visited after it then see.
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
    uint32_t& pushed_cars_mask) {
    const int base = env_idx * cfg.max_cars_per_env;
    RocketSim_Car_PreTickUpdate(
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
        i < 32 && ((pushed_cars_mask >> i) & 1u));
}

// Whether the cars' vehicle updates can run in any order: every wheel ray is staged (no live raycast against the
// other cars) and none rests on a car or the ball (the only bodies an update pushes), and no physics events are
// recorded (they are emitted in visit order). Each update then reads and writes only its own car.
// Car slot i's turn in the WorldStepSoccar car-world loop, between kWorldStepContacts and kWorldStepRows: its rows go
// to the env's shared world rows at the car's slot and its order groups to its CarWorldRowsOut (see PackCarWorldRows).
ROCKETGPU_HD void BuildCarWorldRowsForCar(
    int env_idx,
    int i,
    const SimConfig& cfg,
    CarState* cars,
    const CarConfig* car_configs,
    WorldSolverScratch& scratch,
    const WorldSolverScratchRows& rows,
    bullet_port::btGjkEpaScratch* gjk_scratch) {
    const int car_idx = env_idx * cfg.max_cars_per_env + i;
    CarWorldRowsOut& out = scratch.car_world_rows_out[i];
    out.group_count = 0;
    int split_end = i * kMaxCarWorldSplitContactRows;
    int velocity_end = i * kMaxCarWorldVelocityContactRows;
    int friction_end = i * kMaxCarWorldFrictionContactRows;
    if (scratch.pre_contact_cars[i].is_demoed || cars[car_idx].just_respawned) {
        if (i < kMaxCarWorldMeshPersistentCars) {
            for (int plane_idx = 0; plane_idx < kCarWorldPlaneRetainedManifoldsPerCar; ++plane_idx) {
                scratch.car_world_plane_manifolds[i * kCarWorldPlaneRetainedManifoldsPerCar + plane_idx].count = 0;
            }
        }
    } else {
        const ArenaBroadphaseGrid grid = MakeArenaBroadphaseGrid(cfg);
        const ContactAccumulator acc = SolveCarWorldStaticContacts(
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
                ? &scratch.car_world_plane_manifolds[i * kCarWorldPlaneRetainedManifoldsPerCar]
                : nullptr);
        RecordArenaWorldContact(cars[car_idx], acc);
    }
    out.split_row_count = split_end - i * kMaxCarWorldSplitContactRows;
    out.velocity_row_count = velocity_end - i * kMaxCarWorldVelocityContactRows;
    out.friction_row_count = friction_end - i * kMaxCarWorldFrictionContactRows;
}

ROCKETGPU_HD bool VehicleUpdatesIndependent(const SimConfig& cfg, const WheelRayStage* staged_rays_for_env, int count) {
    if (staged_rays_for_env == nullptr || cfg.mutator.physics_events.windows != nullptr || count > kWheelRayStageSlots / 4) {
        return false;
    }
    for (int lane = 0; lane < 4 * count; ++lane) {
        const WheelRayStage& ray = staged_rays_for_env[lane];
        if (ray.valid != 1 || (ray.is_in_contact && !ray.is_in_contact_with_world)) {
            return false;
        }
    }
    return true;
}

// Tick start, split so the CUDA pre-tick runs a thread per car: StepEnvPreTickEnv first, then StepEnvPreTickCar for
// each car and StepEnvPreTickBallAndPads in any order (each touches only its own car, or the ball and pads).
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
    const int* kickoff_order_override) {
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
        kickoff_order_override);
    EnvState& env = env_states[env_idx];
    if (cfg.mutator.physics_events.windows) {
        cfg.mutator.physics_events.windows[env_idx].tick = static_cast<uint64_t>(env.tick);
    }
    env.goal_scored_this_tick = 0;
    if (solver_scratch && command.reset_to_kickoff) {
        for (int manifold_idx = 0; manifold_idx < kMaxBallWorldMeshPersistentManifolds; ++manifold_idx) {
            ResetBallMeshPersistentManifold(solver_scratch[env_idx].ball_world_mesh_manifolds[manifold_idx]);
        }
        for (int manifold_idx = 0; manifold_idx < 4; ++manifold_idx) {
            ResetBallMeshPersistentManifold(solver_scratch[env_idx].ball_world_plane_manifolds[manifold_idx]);
        }
        for (int manifold_idx = 0; manifold_idx < kMaxCarWorldPlaneRetainedManifolds; ++manifold_idx) {
            solver_scratch[env_idx].car_world_plane_manifolds[manifold_idx].count = 0;
        }
        for (int manifold_idx = 0; manifold_idx < kMaxCarCarRetainedManifolds; ++manifold_idx) {
            solver_scratch[env_idx].car_car_retained_manifolds[manifold_idx].count = 0;
        }
        for (int manifold_idx = 0; manifold_idx < kMaxBallCarRetainedManifolds; ++manifold_idx) {
            solver_scratch[env_idx].ball_car_retained_manifolds[manifold_idx].count = 0;
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
    uint64_t seed) {
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
    if (car.is_demoed) {
        car.demo_respawn_timer = Max(0.0f, car.demo_respawn_timer - cfg.tick_dt);
        if (car.demo_respawn_timer <= 0.0f) {
            const uint32_t car_id = static_cast<uint32_t>(base + i + 1);
            int respawn_slot = -1;
            if (respawn_slot_overrides) {
                const uint8_t override_slot = respawn_slot_overrides[base + i];
                if (override_slot != 0xFFu) {
                    respawn_slot = static_cast<int>(override_slot);
                }
            }
            if (respawn_slot < 0) {
                respawn_slot = PickRespawnSlot(seed, env_idx, static_cast<uint64_t>(env.tick), car_id);
            }
            RespawnCar(car, teams[base + i], respawn_slot, cfg.mutator);
        }
    }
    CalTickStartLimitAndQuantizeCar(car, cfg.mutator);
    // v3 contact callbacks read state.phys.vel, synced after the tick-start limit/quantize and before
    // the z-damp prepass touches the rigid body; capture the hit/bump start velocity here.
    if (solver_scratch != nullptr && i < 8) {
        solver_scratch[env_idx].cal_start_vel[i] = car.vel;
    }
    CalFlipZDampPrepass(car, cfg.mutator, cfg.tick_dt);
}

ROCKETGPU_HD void StepEnvPreTickBallAndPads(
    int env_idx,
    const SimConfig& cfg,
    int count,
    WorldSolverScratch* solver_scratch,
    BallState* balls,
    BoostPadState* pads) {
    CalTickStartLimitAndQuantizeBall(balls[env_idx], cfg.mutator);
    if (solver_scratch != nullptr) {
        // v3 Ball::on_hit reads state.phys.vel, synced here (after the tick-start limit/quantize); the car
        // pre-tick wheel pushes onto the ball and the heatseeker pre-tick only change the rigid body.
        solver_scratch[env_idx].cal_ball_start_vel = balls[env_idx].vel;
    }
    if (cfg.enable_boost_pads && count > 0) {
        RocketSim_BoostPad_PreTickUpdate(env_idx, cfg, pads, cfg.tick_dt);
    }
}

template <typename ControlsAccessor, typename EnvCommandAccessor>
ROCKETGPU_HD void StepEnvGeneric(
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
    int world_part = kWorldStepWhole) {
    const int base = env_idx * cfg.max_cars_per_env;
    const WheelRayStage* const staged_rays_for_env =
        ((phase == kStepEnvPhasePost || phase == kStepEnvPhasePostVehiclesDone) && solver_scratch != nullptr)
            ? solver_scratch[env_idx].wheel_ray_stage : nullptr;

    const WorldSolverScratchRows rows = row_pool.view(env_idx);
    for (int t = 0; t < ticks; ++t) {
        const int tick_idx = tick_offset + t;
        if (phase == kStepEnvPhaseFull) {
            const EnvCommand command = env_command_accessor.get(tick_idx);
            StepEnvPreTickEnv(env_idx, cfg, teams, env_car_count, cars, solver_scratch, controls_for_reset, balls,
                              env_states, pads, command, kickoff_order_override);
            const int pre_count = env_car_count[env_idx];
            for (int k = 0; k < pre_count; ++k) {
                StepEnvPreTickCar(env_idx, CarVisitSlot(env_states[env_idx], k, pre_count), cfg, teams, cars,
                                  solver_scratch, env_states[env_idx], respawn_slot_overrides, seed);
            }
            StepEnvPreTickBallAndPads(env_idx, cfg, pre_count, solver_scratch, balls, pads);
        }
        const int count = env_car_count[env_idx];
        EnvState& env = env_states[env_idx];
        const uint64_t current_tick = static_cast<uint64_t>(env.tick);

        uint32_t cal_pushed_cars_mask = 0u;
        const bool vehicle_updates_done = world_part == kWorldStepRows ||
            (phase == kStepEnvPhasePostVehiclesDone && VehicleUpdatesIndependent(cfg, staged_rays_for_env, count));
        for (int k = 0; !vehicle_updates_done && k < count; ++k) {
            const int i = CarVisitSlot(env_states[env_idx], k, count);
            StepEnvVehicleUpdateCar(env_idx, i, count, cfg, cars, wheel_runtime_states, car_configs, balls,
                                    controls_accessor.get(i, tick_idx), staged_rays_for_env, cal_pushed_cars_mask);
        }

        BallState& ball = balls[env_idx];
        if (world_part != kWorldStepRows && cfg.mutator.game_mode == static_cast<uint8_t>(Mode::Heatseeker)) {
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
            world_part);
        if (world_part == kWorldStepContacts) {
            continue;
        }

        if (solver_scratch) {
            const PreContactCar* const pre_contact_cars =
                solver_scratch[env_idx].pre_contact_cars;
            for (int i = 0; i < count; ++i) {
                if (pre_contact_cars[i].is_demoed || cars[base + i].just_respawned) {
                    continue;
                }
                if (cars[base + i].is_demoed) {
                    cars[base + i].rot = pre_contact_cars[i].rot;
                }
            }
        }

        for (int k = 0; k < count; ++k) {
            const int i = CarVisitSlot(env_states[env_idx], k, count);
            RocketSim_Car_PostTickUpdate(cars[base + i], controls_accessor.get(i, tick_idx), cfg.mutator, cfg.tick_dt);
            RocketSim_Car_FinishPhysicsTick(cars[base + i], cfg.mutator);
            if (cfg.enable_boost_pads) {
                RocketSim_BoostPad_CheckCollide(env_idx, cfg, cars[base + i], static_cast<uint32_t>(base + i + 1), pads);
            }
        }

        for (int i = 0; i < count; ++i)
            cars[base + i].ball_contact_last_tick = cars[base + i].ball_contact_this_tick;
        if (cfg.enable_boost_pads && count > 0) {
            RocketSim_BoostPad_PostTickUpdate(env_idx, cfg, cars, pads);
        }

        RocketSim_Ball_FinishPhysicsTick(ball, cfg.mutator);

        if (cfg.enable_scoring) {
            const float score_threshold = cfg.mutator.goal_base_threshold_y + cfg.mutator.ball_radius;
            if (ball.pos.y > score_threshold) {
                env.last_scored_team = static_cast<int32_t>(Team::Blue);
                env.goal_scored_this_tick = 1;
                env.blue_score += 1;
            } else if (ball.pos.y < -score_threshold) {
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
        for (int i = 0; i < count; ++i) {
            ReportRLBotCarFields(cars[base + i], wheel_runtime_states ? &wheel_runtime_states[base + i] : nullptr,
                                 cfg.mutator, cfg.tick_dt);
        }
        env.tick += 1;
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
    int phase = kStepEnvPhaseFull) {
    StreamControlsAccessor accessor;
    accessor.controls_by_tick = controls_by_tick;
    accessor.base = env_idx * cfg.max_cars_per_env;
    accessor.stride = control_stride;
    NoopEnvCommandAccessor no_env_commands;
    StepEnvGeneric(env_idx, cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls_for_reset, accessor, no_env_commands, car_configs, balls, env_states, pads, respawn_slot_overrides, seed, 1, nullptr, phase, 0);
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
