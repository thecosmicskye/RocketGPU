

ROCKETGPU_HD void DetectCarWorldFloorRampContact(
    const CarState& car,
    const CarConfig& cfg,
    const Vec3& center_pos,
    float R,
    float ramp_axis_center,
    float car_axis_coord,
    float sign_axis,
    float support_radius,
    ContactAccumulator& acc) {
    const float h = (car_axis_coord - sign_axis * ramp_axis_center) * sign_axis;

    if (h < -support_radius || h > R + support_radius) {
        return;
    }

    if (center_pos.z > R + support_radius) {
        return;
    }

    const float circle_axis_world = sign_axis * ramp_axis_center;
    const float circle_z = R;

    const float h_clamped = Clamp(h, 0.0f, R);
    const float s0 = SqrtScalar(R * R - h_clamped * h_clamped);
    const float nh0 = h_clamped / R;
    const float nz0 = s0 / R;

    Vec3 initial_normal;
    initial_normal = Vec3(-sign_axis * nh0, 0.0f, nz0);

    const Vec3 support_local = SupportPointOnCarHitboxLocal(cfg, InverseRotateCarBasis(car, -initial_normal));
    const Vec3 support_world = car.pos + RotateCarBasis(car, support_local);

    const float sp_axis_world = support_world.x;
    const float sp_da = (sp_axis_world - circle_axis_world) * sign_axis;
    const float sp_dz = support_world.z - circle_z;

    const float dist_2d = SqrtScalar(sp_da * sp_da + sp_dz * sp_dz);

    float closest_da, closest_dz;
    if (dist_2d < 1e-6f) {
        closest_da = h_clamped;
        closest_dz = -s0;
    } else {
        float proj_da = R * sp_da / dist_2d;
        float proj_dz = R * sp_dz / dist_2d;

        if (proj_da < 0.0f) {
            return;
        } else if (proj_dz > 0.0f) {
            return;
        } else {
            closest_da = proj_da;
            closest_dz = proj_dz;
        }
    }

    const float final_nh = closest_da / R;
    const float final_nz = -closest_dz / R;

    Vec3 final_normal;
    final_normal = Vec3(-sign_axis * final_nh, 0.0f, final_nz);

    const float surface_axis_world = circle_axis_world + sign_axis * closest_da;
    const float surface_z = circle_z + closest_dz;

    const float boundary_base = surface_axis_world * (-sign_axis * final_nh) + surface_z * final_nz;

    const float shape_margin = BallCarShapeMargin(cfg);
    const float boundary = boundary_base + shape_margin;

    const float margin = Dot(support_world, final_normal) - boundary;
    if (margin >= 0.0f) {
        return;
    }

    const float min_arc_da = R * 0.20f;
    const float max_arc_da = R * 0.80f;
    if (closest_da < min_arc_da || closest_da > max_arc_da) {
        return;
    }

    acc.add(final_normal, -margin);
}

ROCKETGPU_HD bool DebugCarWorldSolverTraceEnabledFor(int env_idx, int car_idx) {
    (void)env_idx;
    (void)car_idx;
    return false;
}

// The game's car world-contact normal (read by auto-roll) is built
// after the step: per colliding world object, the unit sum of its contact point normals,
// then the unit sum over those objects. v3 keeps the last contact-added normal instead. RLPR records
// the averaged normal (cw_turtle_autoroll_slope frame 17568: slope points (-0.955,0,0.297) and (-0.637,0,0.771),
// recorded (-0.830,0,0.557)).
ROCKETGPU_HD void ApplyGameCarWorldContactNormal(
    ContactAccumulator& acc,
    const CarWorldVelocityContactRow* normal_rows,
    int normal_row_count) {
    if (!acc.hit || !normal_rows || normal_row_count <= 0) {
        return;
    }
    constexpr int kMaxObjects = 8;
    int tags[kMaxObjects];
    Vec3 sums[kMaxObjects];
    int object_count = 0;
    for (int row_idx = 0; row_idx < normal_row_count; ++row_idx) {
        if (!normal_rows[row_idx].valid) {
            continue;
        }
        const int tag = normal_rows[row_idx].source_tag;
        int slot = 0;
        while (slot < object_count && tags[slot] != tag) {
            ++slot;
        }
        if (slot == object_count) {
            if (object_count == kMaxObjects) {
                continue;
            }
            tags[slot] = tag;
            sums[slot] = Vec3(0.0f, 0.0f, 0.0f);
            ++object_count;
        }
        sums[slot] += normal_rows[row_idx].normal;
    }
    Vec3 total(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < object_count; ++i) {
        const float len_sq = Dot(sums[i], sums[i]);
        if (len_sq >= 1e-8f) {
            total += sums[i] * (1.0f / SqrtScalar(len_sq));
        }
    }
    const float total_len_sq = Dot(total, total);
    if (total_len_sq >= 1e-8f) {
        acc.latest_normal = total * (1.0f / SqrtScalar(total_len_sq));
    }
}


// kMeshesOutOfReach: the caller found no usable arena mesh in reach (CarWorldUsableMeshesOutOfReach), so the mesh
// query reduces to releasing the retained mesh manifolds.
template <bool kMeshesOutOfReach = false>
ROCKETGPU_HD ContactAccumulator SolveCarWorldStaticContacts(
    CarState& car,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    const Vec3& external_linear_delta,
    const Vec3& external_angular_delta,
    float dt,
    int env_idx = -1,
    int car_idx = -1,
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
    EnvStridedArray<CarSharedSplitContactRow> shared_split_rows = nullptr,
    int* shared_split_row_count = nullptr,
    EnvStridedArray<CarSharedVelocityContactRow> shared_velocity_rows = nullptr,
    int* shared_velocity_row_count = nullptr,
    EnvStridedArray<CarSharedFrictionContactRow> shared_friction_rows = nullptr,
    int* shared_friction_row_count = nullptr,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch = nullptr,
    EnvStridedArray<SharedContactOrderGroup> shared_contact_order_groups = nullptr,
    int* shared_contact_order_group_count = nullptr,
    const ArenaBroadphaseGrid* broadphase_grid = nullptr,
    const CarWorldMeshCandidateStage* candidate_stage = nullptr,
    CarWorldPlaneRetainedManifold* retained_plane_manifolds = nullptr,
    CarWorldMeshRetainedManifold* retained_mesh_manifolds = nullptr,
    CarWorldLocalRows* local_rows = nullptr) {
    ContactAccumulator acc;
    acc.history_car = &car;

    // car_world_early_out: a car out of reach of every arena mesh and clear of the four planes gets no rows, so skip
    // the query and leave only what it would: an empty World group for each plane inside its broadphase window, and
    // the retained plane manifolds emptied (their refresh culls every point). Retained mesh points and physics events
    // take the full query. Host, and the GPU's out-of-reach build (kMeshesOutOfReach), where nearly every car skips:
    // elsewhere on the GPU the per-car test costs more than the queries it skips (a warp saves nothing unless all its
    // cars skip), and the full query gives the same result.
#if defined(__CUDA_ARCH__)
    constexpr bool kEarlyOut = kMeshesOutOfReach;
#else
    constexpr bool kEarlyOut = true;
#endif
    if (kEarlyOut && mutator.arena_collision && !mutator.physics_events.HasNonGoalEvents() && mesh_triangles != nullptr &&
        mesh_triangle_count > 0) {
        bool has_retained_mesh_points = false;
        for (int k = 0; retained_mesh_manifolds != nullptr && k < kCarWorldMeshRetainedManifoldsPerCar; ++k) {
            has_retained_mesh_points |= retained_mesh_manifolds[k].count > 0;
        }
        const int plane_groups = has_retained_mesh_points ? -1 : CarWorldPlanesOutOfReach(car, cfg);
        if (plane_groups >= 0 && CarWorldMeshesOutOfReach(car, cfg, mesh_bvh_meshes, mesh_bvh_mesh_count, dt)) {
            if (retained_plane_manifolds != nullptr) {
                for (int plane_idx = 0; plane_idx < kCarWorldPlaneRetainedManifoldsPerCar; ++plane_idx) {
                    retained_plane_manifolds[plane_idx].count = 0;
                }
            }
            if (shared_velocity_rows && shared_velocity_row_count && shared_friction_rows && shared_friction_row_count &&
                shared_contact_order_groups && shared_contact_order_group_count) {
                int group_idx = 0;
                const auto append_plane_group = [&](int bit, int source_tag) {
                    if ((plane_groups & bit) == 0) {
                        return;
                    }
                    AppendSharedContactOrderGroup(
                        shared_contact_order_groups, *shared_contact_order_group_count, SharedContactOrderBucket::World,
                        car_idx, -1, source_tag, -1, 0, -1, 0, car_idx * 100000 + 1000 + group_idx);
                    ++group_idx;
                };
                // +X wall, -X wall, floor, ceiling (CarWorldPlaneSourceTag).
                append_plane_group(1, 2);
                append_plane_group(2, 1);
                append_plane_group(4, 6);
                append_plane_group(8, 5);
            }
            return acc;
        }
    }

    const CarWorldMeshCandidateStage* mesh_candidate_stage_in = nullptr;
    if (candidate_stage != nullptr && candidate_stage->valid != 0) {
        if (CarWorldMeshCandidateStageMatches(*candidate_stage, car, cfg)) {
            mesh_candidate_stage_in = candidate_stage;
        }
    }
    if (mutator.physics_events.windows && env_idx >= 0 && car_idx >= 0) {
        acc.event_stream = mutator.physics_events;
        acc.event.type = PhysicsEventType::CarWorldContact;
        acc.event.car_id = env_idx * mutator.physics_events.cars_per_env + car_idx + 1;
        acc.event.position = car.pos;
        acc.event.car_velocity = car.vel;
        acc.event.car_on_ground = car.is_on_ground;
    }

    if (!mutator.arena_collision) {
        return acc;
    }
#if defined(__CUDA_ARCH__)
    if (local_rows == nullptr) {
        __trap();
    }
    CarWorldLocalRows& local = *local_rows;
#else
    UninitArray<CarWorldLocalRows, 1> host_local_rows;
    CarWorldLocalRows& local = local_rows != nullptr ? *local_rows : host_local_rows.data()[0];
#endif
    CarWorldSplitContactRow* const split_rows = local.split_rows;
    int split_row_count = 0;
    CarWorldVelocityContactRow* const normal_rows = local.normal_rows;
    int normal_row_count = 0;
    CarWorldFrictionContactRow* const friction_rows = local.friction_rows;
    int friction_row_count = 0;
    SharedContactOrderGroup* const local_world_order_groups = local.order_groups;
    int local_world_order_group_count = 0;
    const bool use_mesh_world_rows =
        mesh_triangles != nullptr && mesh_triangle_count > 0;

    const Vec3 center = car.pos + RotateCarBasis(car, cfg.hitbox_offset);

    const float support_radius = BoundingSphereRadius(cfg);
    const bool in_goal_window_pos_y = InGoalWindow(center.x, center.z, support_radius) && center.y > 0.0f;
    const bool in_goal_window_neg_y = InGoalWindow(center.x, center.z, support_radius) && center.y < 0.0f;
    const bool past_pos_y_wall2 = center.y > ArenaExtentY();
    const bool past_neg_y_wall2 = center.y < -ArenaExtentY();

    if (!use_mesh_world_rows && !in_goal_window_pos_y && !past_pos_y_wall2) {
        AppendCarWorldPlaneContactRows(
            car,
            cfg,
            mutator,
            Vec3(0.0f, -1.0f, 0.0f),
            -ArenaExtentY(),
            external_linear_delta,
            external_angular_delta,
            acc,
            split_rows,
            split_row_count,
            normal_rows,
            normal_row_count,
            friction_rows,
            friction_row_count,
            dt);
    }
    if (!use_mesh_world_rows && !in_goal_window_neg_y && !past_neg_y_wall2) {
        AppendCarWorldPlaneContactRows(
            car,
            cfg,
            mutator,
            Vec3(0.0f, 1.0f, 0.0f),
            -ArenaExtentY(),
            external_linear_delta,
            external_angular_delta,
            acc,
            split_rows,
            split_row_count,
            normal_rows,
            normal_row_count,
            friction_rows,
            friction_row_count,
            dt);
    }

    if (!use_mesh_world_rows) {
        const float R = FloorRampRadius();

        const float abs_x = Abs(center.x);
        const float abs_y = Abs(center.y);
        const float ramp_x_center = ArenaExtentX() - R;
        if (abs_x > ramp_x_center - support_radius && center.z < R + support_radius) {
            if (abs_y <= CornerCenterY()) {
                const float sx = Sign(center.x);
                DetectCarWorldFloorRampContact(
                    car, cfg, center, R,
                    ramp_x_center, center.x, sx, support_radius,
                    acc);
            }
        }
    }

    {
        AppendCarWorldPlaneContactRows(
            car,
            cfg,
            mutator,
            Vec3(1.0f, 0.0f, 0.0f),
            -ArenaExtentX(),
            external_linear_delta,
            external_angular_delta,
            acc,
            split_rows,
            split_row_count,
            normal_rows,
            normal_row_count,
            friction_rows,
            friction_row_count,
            dt,
            true,
            local_world_order_groups,
            &local_world_order_group_count,
            car_idx,
            retained_plane_manifolds ? &retained_plane_manifolds[2] : nullptr);
        AppendCarWorldPlaneContactRows(
            car,
            cfg,
            mutator,
            Vec3(-1.0f, 0.0f, 0.0f),
            -ArenaExtentX(),
            external_linear_delta,
            external_angular_delta,
            acc,
            split_rows,
            split_row_count,
            normal_rows,
            normal_row_count,
            friction_rows,
            friction_row_count,
            dt,
            true,
            local_world_order_groups,
            &local_world_order_group_count,
            car_idx,
            retained_plane_manifolds ? &retained_plane_manifolds[3] : nullptr);
    }

    if (use_mesh_world_rows && kMeshesOutOfReach) {
        ReleaseCarWorldMeshManifoldsOutOfReach(mesh_bvh_meshes, mesh_bvh_mesh_count, mesh_bvh_quantization_count,
                                               mesh_bvh_node_count,
                                               broadphase_grid != nullptr ? retained_mesh_manifolds : nullptr);
    }
    if (use_mesh_world_rows && !kMeshesOutOfReach) {
        AppendCarWorldMeshContactRows(
            car,
            cfg,
            mutator,
            external_linear_delta,
            external_angular_delta,
            acc,
            split_rows,
            split_row_count,
            normal_rows,
            normal_row_count,
            friction_rows,
            friction_row_count,
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
            env_idx,
            car_idx,
            gjk_epa_scratch,
            local_world_order_groups,
            &local_world_order_group_count,
            broadphase_grid,
            dt,
            nullptr,
            mesh_candidate_stage_in,
            -1,
            nullptr,
            nullptr,
            kTrianglePairsNone,
            retained_mesh_manifolds);
    }

    AppendCarWorldPlaneContactRows(
        car,
        cfg,
        mutator,
        Vec3(0.0f, 0.0f, 1.0f),
        0.0f,
        external_linear_delta,
        external_angular_delta,
        acc,
        split_rows,
        split_row_count,
        normal_rows,
        normal_row_count,
        friction_rows,
        friction_row_count,
        dt,
        true,
        local_world_order_groups,
        &local_world_order_group_count,
        car_idx,
        retained_plane_manifolds ? &retained_plane_manifolds[0] : nullptr);
    AppendCarWorldPlaneContactRows(
        car,
        cfg,
        mutator,
        Vec3(0.0f, 0.0f, -1.0f),
        -ArenaHeight(),
        external_linear_delta,
        external_angular_delta,
        acc,
        split_rows,
        split_row_count,
        normal_rows,
        normal_row_count,
        friction_rows,
        friction_row_count,
        dt,
        true,
        local_world_order_groups,
        &local_world_order_group_count,
        car_idx,
        retained_plane_manifolds ? &retained_plane_manifolds[1] : nullptr);

    if (!use_mesh_world_rows && Abs(center.y) > ArenaExtentY()) {
        if (center.z < ApproximateGoalHeight() + support_radius) {
            AppendCarWorldPlaneContactRows(
                car,
                cfg,
                mutator,
                Vec3(-1.0f, 0.0f, 0.0f),
                -ApproximateGoalHalfWidth(),
                external_linear_delta,
                external_angular_delta,
                acc,
                split_rows,
                split_row_count,
                normal_rows,
                normal_row_count,
                friction_rows,
                friction_row_count,
                dt);
            AppendCarWorldPlaneContactRows(
                car,
                cfg,
                mutator,
                Vec3(1.0f, 0.0f, 0.0f),
                -ApproximateGoalHalfWidth(),
                external_linear_delta,
                external_angular_delta,
                acc,
                split_rows,
                split_row_count,
                normal_rows,
                normal_row_count,
                friction_rows,
                friction_row_count,
                dt);
            AppendCarWorldPlaneContactRows(
                car,
                cfg,
                mutator,
                Vec3(0.0f, 0.0f, -1.0f),
                -ApproximateGoalHeight(),
                external_linear_delta,
                external_angular_delta,
                acc,
                split_rows,
                split_row_count,
                normal_rows,
                normal_row_count,
                friction_rows,
                friction_row_count,
                dt);
            AppendCarWorldPlaneContactRows(
                car,
                cfg,
                mutator,
                Vec3(0.0f, center.y > 0.0f ? -1.0f : 1.0f, 0.0f),
                -(ArenaExtentY() + ApproximateGoalDepth()),
                external_linear_delta,
                external_angular_delta,
                acc,
                split_rows,
                split_row_count,
                normal_rows,
                normal_row_count,
                friction_rows,
                friction_row_count,
                dt);
        }
    }

    ApplyGameCarWorldContactNormal(acc, normal_rows, normal_row_count);

    if (shared_split_rows && shared_split_row_count && split_row_count > 0) {
        AppendCarWorldToSharedSplitRows(shared_split_rows, *shared_split_row_count, car_idx, split_rows, split_row_count);
    } else if (split_row_count > 0) {
        SolveCarWorldSplitContactRows(car, split_rows, split_row_count, dt, env_idx, car_idx, NativeSseMath(mutator));
    }

    if (shared_velocity_rows && shared_velocity_row_count && shared_friction_rows && shared_friction_row_count) {
        const int shared_normal_base = *shared_velocity_row_count;

        const int shared_friction_base = *shared_friction_row_count;
        AppendCarWorldToSharedVelocityRows(
            shared_velocity_rows,
            *shared_velocity_row_count,
            shared_friction_rows,
            *shared_friction_row_count,
            car_idx,
            normal_rows,
            normal_row_count,
            friction_rows,
            friction_row_count);
        if (shared_contact_order_groups && shared_contact_order_group_count) {
            for (int group_idx = 0; group_idx < local_world_order_group_count; ++group_idx) {
                const SharedContactOrderGroup& local_group = local_world_order_groups[group_idx];
                if (!local_group.valid) {
                    continue;
                }
                AppendSharedContactOrderGroup(
                    shared_contact_order_groups,
                    *shared_contact_order_group_count,
                    SharedContactOrderBucket::World,
                    car_idx,
                    -1,
                    local_group.source_tag,
                    local_group.normal_row_count > 0 ? shared_normal_base + local_group.normal_row_start : -1,
                    local_group.normal_row_count,
                    local_group.friction_row_count > 0 ? shared_friction_base + local_group.friction_row_start : -1,
                    local_group.friction_row_count,
                    car_idx * 100000 + 1000 + group_idx);
            }
        }
    } else if (normal_row_count > 0 || friction_row_count > 0) {
        SolveCarWorldVelocityContactRows(car, normal_rows, normal_row_count, friction_rows, friction_row_count);
    }

    if (!use_mesh_world_rows) {
        ResolveObbCornerCylinder(
            car,
            cfg,
            mutator.arena_restitution,
            mutator.arena_friction,
            acc,
            external_linear_delta);
    }

    return acc;
}

ROCKETGPU_HD int SortBallMeshCachedPoints(
    const BallMeshManifoldPoint cache[4],
    const BallMeshManifoldPoint& new_pt) {
    int maxPenetrationIndex = -1;
    float maxPen = new_pt.pen_uu;
    for (int i = 0; i < 4; ++i) {
        if (cache[i].pen_uu > maxPen) {
            maxPenetrationIndex = i;
            maxPen = cache[i].pen_uu;
        }
    }

    float res0 = 0.0f, res1 = 0.0f, res2 = 0.0f, res3 = 0.0f;
    if (maxPenetrationIndex != 0) {
        const Vec3 a0 = new_pt.local_point_a_bt - cache[1].local_point_a_bt;
        const Vec3 b0 = cache[3].local_point_a_bt - cache[2].local_point_a_bt;
        const Vec3 c0 = Cross(a0, b0);
        res0 = Dot(c0, c0);
    }
    if (maxPenetrationIndex != 1) {
        const Vec3 a1 = new_pt.local_point_a_bt - cache[0].local_point_a_bt;
        const Vec3 b1 = cache[3].local_point_a_bt - cache[2].local_point_a_bt;
        const Vec3 c1 = Cross(a1, b1);
        res1 = Dot(c1, c1);
    }
    if (maxPenetrationIndex != 2) {
        const Vec3 a2 = new_pt.local_point_a_bt - cache[0].local_point_a_bt;
        const Vec3 b2 = cache[3].local_point_a_bt - cache[1].local_point_a_bt;
        const Vec3 c2 = Cross(a2, b2);
        res2 = Dot(c2, c2);
    }
    if (maxPenetrationIndex != 3) {
        const Vec3 a3 = new_pt.local_point_a_bt - cache[0].local_point_a_bt;
        const Vec3 b3 = cache[2].local_point_a_bt - cache[1].local_point_a_bt;
        const Vec3 c3 = Cross(a3, b3);
        res3 = Dot(c3, c3);
    }

    int biggest = 0;
    float best = res0;
    if (res1 > best) { best = res1; biggest = 1; }
    if (res2 > best) { best = res2; biggest = 2; }
    if (res3 > best) { best = res3; biggest = 3; }
    return biggest;
}

ROCKETGPU_HD void InsertBallMeshManifoldPoint(
    BallMeshManifoldPoint cache[4],
    int& count,
    const BallMeshManifoldPoint& new_pt,
    float merge_threshold_bt = -1.0f) {
    if (merge_threshold_bt >= 0.0f) {
        float shortest_dist = merge_threshold_bt * merge_threshold_bt;
        int nearest = -1;
        for (int i = 0; i < count; ++i) {
            const Vec3 diff_a = cache[i].local_point_a_bt - new_pt.local_point_a_bt;
            const float dist_to_point = Dot(diff_a, diff_a);
            if (dist_to_point < shortest_dist) {
                shortest_dist = dist_to_point;
                nearest = i;
            }
        }
        if (nearest >= 0) {
            cache[nearest] = new_pt;
            return;
        }
    }
    int insert_index = count;
    if (insert_index == 4) {
        insert_index = SortBallMeshCachedPoints(cache, new_pt);
    } else {
        ++count;
    }
    if (insert_index < 0) {
        insert_index = 0;
    }
    cache[insert_index] = new_pt;
}

ROCKETGPU_HD void RemoveBallMeshManifoldPoint(
    BallMeshManifoldPoint cache[4],
    int& count,
    int remove_index,
    bool swap_remove = false) {
    if (remove_index < 0 || remove_index >= count) {
        return;
    }
    if (swap_remove) {
        // btPersistentManifold::removeContactPoint: the last point moves into the freed slot.
        cache[remove_index] = cache[count - 1];
        cache[count - 1] = BallMeshManifoldPoint{};
        --count;
        return;
    }
    for (int i = remove_index; i + 1 < count; ++i) {
        cache[i] = cache[i + 1];
    }
    if (count > 0) {
        cache[count - 1] = BallMeshManifoldPoint{};
        --count;
    }
}

ROCKETGPU_HD void RefreshBallMeshPersistentManifold(
    BallMeshManifoldPoint cache[4],
    int& count,
    const Vec3& ball_center_bt,
    const BallState& ball,
    float contact_breaking_threshold_bt,
    float cal_origin_y = 0.0f,
    bool swap_remove = false) {
    constexpr float kBTToUU = 50.0f;
    Vec3 ball_basis_x;
    Vec3 ball_basis_y;
    Vec3 ball_basis_z;

    GetBallBasisColumns(ball, ball_basis_x, ball_basis_y, ball_basis_z);
    for (int i = count - 1; i >= 0; --i) {
        BallMeshManifoldPoint& manifold_point = cache[i];

        const Vec3 rotated_local_a(
            ball_basis_x.x * manifold_point.local_point_a_bt.x +
                ball_basis_y.x * manifold_point.local_point_a_bt.y +
                ball_basis_z.x * manifold_point.local_point_a_bt.z,
            ball_basis_x.y * manifold_point.local_point_a_bt.x +
                ball_basis_y.y * manifold_point.local_point_a_bt.y +
                ball_basis_z.y * manifold_point.local_point_a_bt.z,
            ball_basis_x.z * manifold_point.local_point_a_bt.x +
                ball_basis_y.z * manifold_point.local_point_a_bt.y +
                ball_basis_z.z * manifold_point.local_point_a_bt.z);
        const Vec3 position_world_on_a = rotated_local_a + ball_center_bt;

        const Vec3 position_world_on_b = cal_origin_y != 0.0f
            ? Vec3(manifold_point.local_point_b_bt.x, manifold_point.local_point_b_bt.y + cal_origin_y, manifold_point.local_point_b_bt.z)
            : manifold_point.local_point_b_bt;
        const float distance_bt = Dot(
            position_world_on_a - position_world_on_b,
            manifold_point.normal);
        manifold_point.pen_uu = -distance_bt * kBTToUU;
        manifold_point.distance_bt = distance_bt;
        manifold_point.rel_pos1_bt = position_world_on_a - ball_center_bt;

        manifold_point.rel_pos1_length_bt = Length(position_world_on_a - ball_center_bt);

        if (distance_bt > contact_breaking_threshold_bt) {
            RemoveBallMeshManifoldPoint(cache, count, i, swap_remove);
            continue;
        }

        const Vec3 projected_point =
            position_world_on_a - manifold_point.normal * distance_bt;
        const Vec3 projected_difference = position_world_on_b - projected_point;
        const float distance2d = Dot(projected_difference, projected_difference);
        if (distance2d > contact_breaking_threshold_bt * contact_breaking_threshold_bt) {
            RemoveBallMeshManifoldPoint(cache, count, i, swap_remove);
        }
    }
}

ROCKETGPU_HD void AppendBallWorldSpecialContactRows(
    const BallState& ball,
    const MutatorConfig& mutator,
    const Vec3& external_linear_delta,
    WorldSolverScratch* solver_scratch,
    EnvStridedArray<const BulletManifoldEntry> bullet_manifolds,
    EnvStridedArray<BallSplitContactRow> split_rows,
    int& split_row_count,
    float split_dt,
    EnvStridedArray<BallVelocityContactRow> normal_rows,
    int& normal_row_count,
    EnvStridedArray<BallFrictionContactRow> friction_rows,
    int& friction_row_count,
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
    const Vec3* bt_ball_linear_velocity_override = nullptr,
    int event_env_idx = -1) {
    if (solver_scratch) {
        solver_scratch->hs_world_normal_count = 0;
    }

    BallWorldContactAggregate aggregate;
    struct BallWorldSpecialCp {
        Vec3 normal = Vec3(0.0f, 0.0f, 1.0f);
        float penetration_uu = 0.0f;

        float rel_pos1_length_bt = 0.0f;

        int16_t static_id = -1;

        uint8_t has_bt = 0;
        Vec3 rel_pos1_bt = Vec3(0.0f, 0.0f, 0.0f);
        float penetration_bt = 0.0f;
    };
    constexpr int kMaxBallWorldSpecialCps = 16;
    BallWorldSpecialCp special_cps[kMaxBallWorldSpecialCps];
    int special_cp_count = 0;
    const float collision_r = mutator.ball_radius;
    constexpr float kUUToBT_local = 1.0f / 50.0f;
    const float ball_radius_bt = collision_r * kUUToBT_local;
    const float arena_margin = BallArenaMeshCollisionMargin();

    const float plane_detection_r =
        collision_r * 0.02f + 50.0f * 0.08f * 0.02f;

    const auto emit_world_event = [&](const Vec3& normal) {
        if (event_env_idx >= 0 && mutator.physics_events.windows) {
            PhysicsEvent event;
            event.type = PhysicsEventType::BallWorldContact;
            event.position = ball.pos;
            event.normal = normal;
            event.ball_position = ball.pos;
            event.ball_velocity = ball.vel;
            mutator.physics_events.Emit(event_env_idx, event);
        }
    };
    const auto append_special_cp_with_len = [&](const Vec3& normal, float penetration_uu, float rel_pos1_length_bt, int16_t static_id, const Vec3* rel_pos1_bt = nullptr, const float* penetration_bt = nullptr, bool emit_event = true) {
        if (emit_event) emit_world_event(normal);
        if (solver_scratch && solver_scratch->hs_world_normal_count < WorldSolverScratch::kMaxHeatseekerWorldNormals) {
            solver_scratch->hs_world_normals[solver_scratch->hs_world_normal_count++] = normal;
        }
        if (special_cp_count < kMaxBallWorldSpecialCps) {
            special_cps[special_cp_count].normal = normal;
            special_cps[special_cp_count].penetration_uu = penetration_uu;
            special_cps[special_cp_count].rel_pos1_length_bt = rel_pos1_length_bt;
            special_cps[special_cp_count].static_id = static_id;
            if (rel_pos1_bt && penetration_bt) {
                special_cps[special_cp_count].has_bt = 1;
                special_cps[special_cp_count].rel_pos1_bt = *rel_pos1_bt;
                special_cps[special_cp_count].penetration_bt = *penetration_bt;
            }
            ++special_cp_count;
        } else {
            AccumulateBallWorldContact(aggregate, normal, penetration_uu, rel_pos1_length_bt);
        }
    };

    const auto official_plane_rel_len = [&](const Vec3& inward_normal, const Vec3& plane_pos_uu, Vec3* out_rel_bt = nullptr, float* out_distance_bt = nullptr, Vec3* out_local_a_bt = nullptr, Vec3* out_point_b_bt = nullptr) -> float {
        const Vec3 plane_pos_bt(
            plane_pos_uu.x * kUUToBT_local,
            plane_pos_uu.y * kUUToBT_local,
            plane_pos_uu.z * kUUToBT_local);
        Vec3 bx(1.0f, 0.0f, 0.0f);
        Vec3 by(0.0f, 1.0f, 0.0f);
        Vec3 bz(0.0f, 0.0f, 1.0f);
        GetBallBasisColumns(ball, bx, by, bz);
        const Vec3 neg_n = -inward_normal;
        const Vec3 dir_local(Dot(bx, neg_n), Dot(by, neg_n), Dot(bz, neg_n));
        const float dir_len_sq =
            dir_local.x * dir_local.x + dir_local.y * dir_local.y + dir_local.z * dir_local.z;

        const float inv_sqrt3 = 1.0f / sqrtf(3.0f);
        Vec3 dir_unit(-inv_sqrt3, -inv_sqrt3, -inv_sqrt3);
        if (dir_len_sq >= 1.4210854715202004e-14f) {
            const float inv_dir_len = 1.0f / sqrtf(dir_len_sq);
            dir_unit = Vec3(dir_local.x * inv_dir_len, dir_local.y * inv_dir_len, dir_local.z * inv_dir_len);
        }
        const Vec3 vtx_local = dir_unit * ball_radius_bt;
        const Vec3 trans_origin(
            ball.pos_bt.x - plane_pos_bt.x,
            ball.pos_bt.y - plane_pos_bt.y,
            ball.pos_bt.z - plane_pos_bt.z);
        const Vec3 vtx_in_plane(
            (bx.x * vtx_local.x + by.x * vtx_local.y + bz.x * vtx_local.z) + trans_origin.x,
            (bx.y * vtx_local.x + by.y * vtx_local.y + bz.y * vtx_local.z) + trans_origin.y,
            (bx.z * vtx_local.x + by.z * vtx_local.y + bz.z * vtx_local.z) + trans_origin.z);
        const float distance = Dot(inward_normal, vtx_in_plane);
        const Vec3 projected = vtx_in_plane - inward_normal * distance;
        const Vec3 world = projected + plane_pos_bt;
        const Vec3 point_a = world + inward_normal * distance;

        const Vec3 raw_rel(
            point_a.x - ball.pos_bt.x,
            point_a.y - ball.pos_bt.y,
            point_a.z - ball.pos_bt.z);
        const Vec3 local_a(Dot(raw_rel, bx), Dot(raw_rel, by), Dot(raw_rel, bz));
        const Vec3 world_a(
            (bx.x * local_a.x + by.x * local_a.y + bz.x * local_a.z) + ball.pos_bt.x,
            (bx.y * local_a.x + by.y * local_a.y + bz.y * local_a.z) + ball.pos_bt.y,
            (bx.z * local_a.x + by.z * local_a.y + bz.z * local_a.z) + ball.pos_bt.z);
        const Vec3 rel(
            world_a.x - ball.pos_bt.x,
            world_a.y - ball.pos_bt.y,
            world_a.z - ball.pos_bt.z);
        if (out_rel_bt) {
            *out_rel_bt = rel;
        }
        if (out_distance_bt) {
            *out_distance_bt = Dot(world_a - world, inward_normal);
        }
        if (out_local_a_bt) {
            *out_local_a_bt = local_a;
        }
        if (out_point_b_bt) {
            *out_point_b_bt = world;
        }
        return sqrtf(rel.x * rel.x + rel.y * rel.y + rel.z * rel.z);
    };
    const auto bullet_plane_rank = [](const Vec3& inward_normal) -> int16_t {
        if (inward_normal.z > 0.5f) return 100 + 0;
        if (inward_normal.z < -0.5f) return 100 + 1;
        if (inward_normal.x > 0.5f) return 100 + 2;
        if (inward_normal.x < -0.5f) return 100 + 3;
        return -1;
    };
    const float plane_breaking_threshold_bt = (ball_radius_bt + 0.08f) * 0.02f;
    const bool retain_plane_points = solver_scratch != nullptr;
    const auto append_plane_cp = [&](const Vec3& inward_normal, float boundary, const Vec3& plane_pos_uu, int plane_slot) {
        const float margin = Dot(ball.pos, inward_normal) - boundary;
        if (retain_plane_points) {
            // v3 collision_dispatcher near_callback_with_bounds: refresh the cached ball-plane manifold, then merge the
            // fresh convex_plane_collision_alg point (get_cache_entry on the ball-local point); every retained point
            // stays a special sample (persistent_manifold.rs refresh_contact_points / merge_contact_points).
            BallMeshPersistentManifold& m = solver_scratch->ball_world_plane_manifolds[plane_slot];
            int count = static_cast<int>(m.count);
            if (count > 0) {
                RefreshBallMeshPersistentManifold(m.points, count, ball.pos_bt, ball, plane_breaking_threshold_bt, 0.0f, true);
            }
            if (margin < collision_r + plane_detection_r) {
                Vec3 rel_bt(0.0f, 0.0f, 0.0f);
                float distance_bt = 0.0f;
                Vec3 local_a_bt(0.0f, 0.0f, 0.0f);
                Vec3 point_b_bt(0.0f, 0.0f, 0.0f);
                const float rel_len = official_plane_rel_len(inward_normal, plane_pos_uu, &rel_bt, &distance_bt, &local_a_bt, &point_b_bt);
                if (distance_bt < plane_breaking_threshold_bt) {
                    BallMeshManifoldPoint new_pt;
                    new_pt.normal = inward_normal;
                    new_pt.pen_uu = -distance_bt * 50.0f;
                    new_pt.local_point_a_bt = local_a_bt;
                    new_pt.local_point_b_bt = point_b_bt;
                    new_pt.triangle_index = -1;
                    new_pt.rel_pos1_length_bt = rel_len;
                    new_pt.rel_pos1_bt = rel_bt;
                    new_pt.distance_bt = distance_bt;
                    if (count == 0) {
                        m.points[0] = new_pt;
                        count = 1;
                    } else {
                        InsertBallMeshManifoldPoint(m.points, count, new_pt, plane_breaking_threshold_bt);
                    }
                }
            }
            m.count = static_cast<uint8_t>(count);
            for (int i = 0; i < count; ++i) {
                const BallMeshManifoldPoint& q = m.points[i];
                const float pen_bt = -q.distance_bt;
                append_special_cp_with_len(
                    inward_normal,
                    q.pen_uu,
                    q.rel_pos1_length_bt,
                    bullet_plane_rank(inward_normal),
                    &q.rel_pos1_bt,
                    &pen_bt);
            }
            return;
        }
        if (margin >= collision_r + plane_detection_r) {
            return;
        }
        Vec3 plane_rel_bt(0.0f, 0.0f, 0.0f);
        float plane_distance_bt = 0.0f;
        const float plane_rel_len = official_plane_rel_len(inward_normal, plane_pos_uu, &plane_rel_bt, &plane_distance_bt);
        const float plane_pen_bt = -plane_distance_bt;
        append_special_cp_with_len(
            inward_normal,
            collision_r - margin,
            plane_rel_len,
            bullet_plane_rank(inward_normal),
            &plane_rel_bt,
            &plane_pen_bt);
    };

    append_plane_cp(Vec3(0.0f, 0.0f, 1.0f), arena_margin, Vec3(0.0f, 0.0f, 0.0f), 0);
    append_plane_cp(Vec3(0.0f, 0.0f, -1.0f), -(ArenaHeight() - arena_margin), Vec3(0.0f, 0.0f, ArenaHeight()), 1);

    append_plane_cp(Vec3(-1.0f, 0.0f, 0.0f), -(ArenaExtentX() - arena_margin), Vec3(ArenaExtentX(), 0.0f, ArenaHeight() / 2.0f), 2);
    append_plane_cp(Vec3(1.0f, 0.0f, 0.0f), -(ArenaExtentX() - arena_margin), Vec3(-ArenaExtentX(), 0.0f, ArenaHeight() / 2.0f), 3);

    if (mesh_triangles != nullptr && mesh_triangle_count > 0 &&
        mesh_bvh_quantization != nullptr && mesh_bvh_meshes != nullptr &&
        mesh_bvh_nodes != nullptr && mesh_bvh_mesh_count > 0 &&
        mesh_bvh_node_count > 0) {
        constexpr float kUUToBT = 1.0f / 50.0f;
        const Vec3 ball_center_bt = ball.pos_bt;
        const float radius_bt = collision_r * kUUToBT;

        const float contact_breaking_threshold_bt = (radius_bt + 0.08f) * 0.02f;
        const float radiusWithThreshold = radius_bt + contact_breaking_threshold_bt;

        const Vec3 aabb_min_bt = ball_center_bt - Vec3(radiusWithThreshold, radiusWithThreshold, radiusWithThreshold);
        const Vec3 aabb_max_bt = ball_center_bt + Vec3(radiusWithThreshold, radiusWithThreshold, radiusWithThreshold);

        uint16_t quantized_query_aabb_min[3] = {0, 0, 0};
        uint16_t quantized_query_aabb_max[3] = {0, 0, 0};

        // A contact point lies on its triangle within radiusWithThreshold of the ball's center, so a mesh whose
        // AABB (the union of its triangles' AABBs) misses the query box by more than float rounding has no
        // triangle that adds a point: its walk is skipped.
        constexpr float kBallMeshWalkSkipMarginBt = 0.01f;
        const auto mesh_walk_needed = [&](const ArenaMeshBvhMeshInfo& info) {
            return !(info.aabb_max.x < aabb_min_bt.x - kBallMeshWalkSkipMarginBt ||
                     info.aabb_min.x > aabb_max_bt.x + kBallMeshWalkSkipMarginBt ||
                     info.aabb_max.y < aabb_min_bt.y - kBallMeshWalkSkipMarginBt ||
                     info.aabb_min.y > aabb_max_bt.y + kBallMeshWalkSkipMarginBt ||
                     info.aabb_max.z < aabb_min_bt.z - kBallMeshWalkSkipMarginBt ||
                     info.aabb_min.z > aabb_max_bt.z + kBallMeshWalkSkipMarginBt);
        };
        // A mesh with no walk and an empty retained manifold changes nothing (its refresh has no points and its
        // count stays 0). Those meshes are found first, in one pass whose loads overlap, and skipped below.
        uint32_t idle_meshes = 0u;
        if (mutator.cal_ball_world != 0 && solver_scratch != nullptr) {
#pragma unroll
            for (int mesh_info_idx = 0; mesh_info_idx < kMaxBallWorldMeshPersistentManifolds; ++mesh_info_idx) {
                if (mesh_info_idx < mesh_bvh_mesh_count &&
                    solver_scratch->ball_world_mesh_manifolds[mesh_info_idx].count == 0 &&
                    !mesh_walk_needed(mesh_bvh_meshes[mesh_info_idx])) {
                    idle_meshes |= 1u << mesh_info_idx;
                }
            }
        }

        for (int mesh_info_idx = 0; mesh_info_idx < mesh_bvh_mesh_count; ++mesh_info_idx) {
            if (mesh_info_idx < kMaxBallWorldMeshPersistentManifolds && ((idle_meshes >> mesh_info_idx) & 1u) != 0u) {
                continue;
            }
            const ArenaMeshBvhMeshInfo& mesh_info = mesh_bvh_meshes[mesh_info_idx];
            if (mesh_info.mesh_index >= mesh_bvh_quantization_count ||
                mesh_info.node_offset < 0 ||
                mesh_info.node_offset + mesh_info.node_count > mesh_bvh_node_count) {
                continue;
            }

            // The transient manifold is never retained, so it is always reset below before use.
            UninitArray<BallMeshPersistentManifold, 1> transient_storage;
            BallMeshPersistentManifold* const transient_manifold = transient_storage.data();
            BallMeshPersistentManifold* persistent_manifold = transient_manifold;
            if (solver_scratch != nullptr &&
                mesh_info_idx >= 0 &&
                mesh_info_idx < kMaxBallWorldMeshPersistentManifolds) {
                persistent_manifold = &solver_scratch->ball_world_mesh_manifolds[mesh_info_idx];
            }

            // cal_retained_ball_world_contacts: Bullet keeps the ball-mesh persistent manifold across ticks
            // (btConvexConcaveCollisionAlgorithm never clears it); new points merge into the retained cache and
            // refreshContactPoints drops the separated ones (v3 sphere_concave_collision_alg::process_collision_into).
            // Retention needs the cal_ball_world merge threshold below; without it (Snowday forces cal_ball_world=0)
            // every tick's points would pile up as duplicates of the retained ones.
            const bool retain_ball_mesh_points =
                mutator.cal_ball_world != 0 &&
                persistent_manifold != transient_manifold;
            if (!retain_ball_mesh_points) {
                ResetBallMeshPersistentManifold(*persistent_manifold);
            }
            int persistent_count = retain_ball_mesh_points ? static_cast<int>(persistent_manifold->count) : 0;

            // A skipped walk is an empty node range; the mesh's retained points are still refreshed below.
            const bool walk_mesh = mesh_walk_needed(mesh_info);
            if (walk_mesh) {
                const ArenaMeshBvhQuantization& quantization = mesh_bvh_quantization[mesh_info.mesh_index];
                QuantizeArenaMeshBvhPoint(quantization, aabb_min_bt, 0, quantized_query_aabb_min);
                QuantizeArenaMeshBvhPoint(quantization, aabb_max_bt, 1, quantized_query_aabb_max);
            }

            const float cal_origin_y = mesh_info.cal_origin_y;

            const ArenaMeshBvhSubtreeInfo* mesh_subtrees = nullptr;
            int32_t mesh_subtree_count = 0;
            if (mesh_bvh_subtrees != nullptr &&
                mesh_info.subtree_offset >= 0 &&
                mesh_info.subtree_count > 0 &&
                mesh_info.subtree_offset + mesh_info.subtree_count <= mesh_bvh_subtree_count) {
                mesh_subtrees = mesh_bvh_subtrees + mesh_info.subtree_offset;
                mesh_subtree_count = mesh_info.subtree_count;
            }
            WalkArenaMeshBvhStacklessQuantizedTreeCacheFriendly(
                mesh_bvh_nodes,
                mesh_subtrees,
                // cal_bvh_dfs_order: stock btQuantizedBvh walks TRAVERSAL_STACKLESS (whole-tree DFS); no subtree headers.
                // The ball query is large enough that the whole-tree walk beats scanning the node-ordered headers
                // (SelectArenaMeshBvhDfsOrderSubtrees, same leaves in the same order) on the GPU; car queries use those.
                true ? 0 : mesh_subtree_count,
                quantized_query_aabb_min,
                quantized_query_aabb_max,
                mesh_info.node_offset,
                walk_mesh ? mesh_info.node_count : 0,
                [&](int32_t local_tri_idx) {
                    if (local_tri_idx < 0 || local_tri_idx >= mesh_info.triangle_count) {
                        return;
                    }
                    const int32_t global_tri_idx = mesh_info.triangle_start + local_tri_idx;
                    if (global_tri_idx >= mesh_triangle_count) {
                        return;
                    }
                    const ArenaMeshTriangle& tri = mesh_triangles[global_tri_idx];

                    const ArenaMeshTriangle* cal_game_tri =
                        mutator.cal_arena_local_tris != nullptr ? &mutator.cal_arena_local_tris[global_tri_idx] : nullptr;
                    Vec3 contact_normal(0.0f, 0.0f, 1.0f);
                    float pen_uu = 0.0f;
                    Vec3 contact_on_tri_bt(0.0f, 0.0f, 0.0f);
                    float depth_bt = 0.0f;

                    const bool mesh_local_frame = cal_origin_y != 0.0f;
                    const ArenaMeshTriangle& detect_tri = mesh_local_frame ? *cal_game_tri : tri;
                    const Vec3 detect_center_bt = mesh_local_frame
                        ? Vec3(ball_center_bt.x, ball_center_bt.y - cal_origin_y, ball_center_bt.z)
                        : ball_center_bt;
                    const bool mesh_hit = DetectBallMeshTriangleContactEx(
                            detect_center_bt,
                            radius_bt,
                            contact_breaking_threshold_bt,
                            detect_tri.a, detect_tri.b, detect_tri.c,
                            contact_normal, pen_uu, contact_on_tri_bt, &depth_bt,
                            mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift,
                            mutator.cal_ball_world != 0);
                    if (mesh_hit && mesh_local_frame) {
                        contact_on_tri_bt = Vec3(contact_on_tri_bt.x, contact_on_tri_bt.y + cal_origin_y, contact_on_tri_bt.z);
                    }
                    if (mesh_hit) {
                        const Vec3 raw_normal = contact_normal;

                        emit_world_event(raw_normal);

                        const float distance_bt = depth_bt;

                        const Vec3 point_world_on_a_bt =
                            contact_on_tri_bt + raw_normal * distance_bt;
                        const Vec3 raw_rel_pos1_bt =
                            point_world_on_a_bt - ball_center_bt;
                        Vec3 ball_basis_x;
                        Vec3 ball_basis_y;
                        Vec3 ball_basis_z;
                        GetBallBasisColumns(ball, ball_basis_x, ball_basis_y, ball_basis_z);
                        const Vec3 local_point_a_bt(
                            Dot(raw_rel_pos1_bt, ball_basis_x),
                            Dot(raw_rel_pos1_bt, ball_basis_y),
                            Dot(raw_rel_pos1_bt, ball_basis_z));
                        const float rel_pos1_length_bt = Length(raw_rel_pos1_bt);

                        if (EnableArenaTriangleInternalEdgeAdjust()) {
                            MaybeAdjustArenaTriangleInternalEdgeContact(
                                tri,
                                point_world_on_a_bt,
                                distance_bt,
                                contact_on_tri_bt,
                                contact_normal, nullptr,
                                mutator.game_mode == static_cast<uint8_t>(Mode::Soccar),
                                mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift,
                                cal_game_tri);
                        }

                        BallMeshManifoldPoint new_pt;
                        new_pt.normal = contact_normal;
                        new_pt.pen_uu = pen_uu;
                        new_pt.local_point_a_bt = local_point_a_bt;

                        new_pt.local_point_b_bt = cal_origin_y != 0.0f
                            ? Vec3(contact_on_tri_bt.x, contact_on_tri_bt.y - cal_origin_y, contact_on_tri_bt.z)
                            : contact_on_tri_bt;
                        new_pt.triangle_index = global_tri_idx;
                        new_pt.rel_pos1_length_bt = rel_pos1_length_bt;
                        InsertBallMeshManifoldPoint(
                            persistent_manifold->points,
                            persistent_count,
                            new_pt,
                            mutator.cal_ball_world != 0 ? contact_breaking_threshold_bt : -1.0f);
                    }
                });

            RefreshBallMeshPersistentManifold(
                persistent_manifold->points,
                persistent_count,
                ball_center_bt,
                ball,
                contact_breaking_threshold_bt,
                cal_origin_y,
                retain_ball_mesh_points);
            persistent_manifold->count = static_cast<uint8_t>(persistent_count);

            for (int i = 0; i < persistent_count; ++i) {
                const float mesh_slot_pen_bt = -persistent_manifold->points[i].distance_bt;
                append_special_cp_with_len(
                    persistent_manifold->points[i].normal,
                    persistent_manifold->points[i].pen_uu,
                    persistent_manifold->points[i].rel_pos1_length_bt,
                    static_cast<int16_t>(mesh_info.mesh_index),
                    &persistent_manifold->points[i].rel_pos1_bt,
                    &mesh_slot_pen_bt, false);
            }
        }
    }

    {
        bool consumed[kMaxBallWorldSpecialCps];
        for (int i = 0; i < special_cp_count; ++i) {
            consumed[i] = false;
        }
        if (solver_scratch != nullptr && solver_scratch->bullet_manifold_count > 0) {
            for (int m = 0; m < solver_scratch->bullet_manifold_count; ++m) {
                const BulletManifoldEntry& entry = bullet_manifolds[m];
                if (entry.kind != 0) {
                    continue;
                }
                for (int i = 0; i < special_cp_count; ++i) {
                    if (!consumed[i] && special_cps[i].static_id == entry.static_id) {
                        consumed[i] = true;
                        AccumulateBallWorldContact(
                            aggregate,
                            special_cps[i].normal,
                            special_cps[i].penetration_uu,
                            special_cps[i].rel_pos1_length_bt);
                    }
                }
            }
        }
        for (int i = 0; i < special_cp_count; ++i) {
            if (!consumed[i]) {
                AccumulateBallWorldContact(
                    aggregate,
                    special_cps[i].normal,
                    special_cps[i].penetration_uu,
                    special_cps[i].rel_pos1_length_bt);
            }
        }
    }
    for (int probe_i = 0; probe_i < special_cp_count; ++probe_i) {
    }

    if (!aggregate.valid) {
        return;
    }

    for (int i = 0; i < special_cp_count; ++i) {
        const Vec3 rel_ball_contact = -special_cps[i].normal * mutator.ball_radius;

        const Vec3 ball_angular_component_bt = special_cps[i].has_bt
            ? ApplyInverseBallInertiaWorldBtMutator(ball, mutator,
                  Cross(special_cps[i].rel_pos1_bt, special_cps[i].normal))
            : Vec3(0.0f, 0.0f, 0.0f);
        AppendBallStaticSplitRow(
            split_rows,
            split_row_count,
            special_cps[i].penetration_uu,
            special_cps[i].normal,
            rel_ball_contact,
            mutator,
            split_dt,
            special_cps[i].has_bt ? &special_cps[i].rel_pos1_bt : nullptr,
            special_cps[i].has_bt ? &special_cps[i].penetration_bt : nullptr,
            special_cps[i].has_bt ? &ball_angular_component_bt : nullptr,
            special_cps[i].static_id);
    }

    const float inv_count = 1.0f / static_cast<float>(aggregate.count);
    Vec3 normal = aggregate.normal_sum * inv_count;
    if (mutator.cal_ball_world) {
        const float l2 = Dot(normal, normal);
        if (l2 >= SIMD_EPSILON * SIMD_EPSILON) {
            normal = normal * (1.0f / SqrtScalar(l2));
        } else {
            normal = Vec3(1.0f, 0.0f, 0.0f);
        }
    }

    const float synthetic_total_dist_bt = aggregate.rel_pos1_length_sum_bt;

    const float synthetic_distance_bt =
        synthetic_total_dist_bt / static_cast<float>(aggregate.count);
    constexpr float kBTToUU = 50.0f;
    const Vec3 rel_ball_contact_bt = normal * (-synthetic_distance_bt);
    const Vec3 rel_ball_contact = rel_ball_contact_bt * kBTToUU;

    {
    }

    const Vec3 normal_restitution_rel_vel = BallVelocityAtPoint(ball, rel_ball_contact);
    const Vec3 normal_base_rel_vel = BallVelocityAtPoint(ball, rel_ball_contact, external_linear_delta);
    const Vec3 tangent_basis_rel_vel = normal_base_rel_vel;
    const Vec3 friction_base_rel_vel = normal_base_rel_vel;

    const Vec3 restitution_ball_linear = ball.vel;
    const Vec3 rhs_ball_linear = ball.vel + external_linear_delta;
    const Vec3 ball_angular_world = ball.ang_vel;
    const float ball_inv_mass = 1.0f / mutator.ball_mass;
    const Vec3 ball_normal_component = ApplyInverseBallInertiaWorld(
        mutator.ball_mass,
        mutator.ball_radius,
        Cross(rel_ball_contact, normal));

    const Vec3 bt_ball_linear_for_tangent = bt_ball_linear_velocity_override
        ? *bt_ball_linear_velocity_override
        : BallStoredLinearVelocityToBt(ball.vel);
    const Vec3 bt_tangent_basis_rel_vel =
        (bt_ball_linear_for_tangent + UuVecToBtDiv(external_linear_delta)) +
        Cross(ball_angular_world, rel_ball_contact_bt);
    Vec3 tangent;
    {
        const float bt_rel_vel_n = Dot(normal, bt_tangent_basis_rel_vel);
        const Vec3 bt_lat = bt_tangent_basis_rel_vel - normal * bt_rel_vel_n;
        const float bt_lat_len2 = Dot(bt_lat, bt_lat);
        tangent = bt_lat_len2 > 1.1920928955078125e-7f
            ? bt_lat * (1.0f / sqrtf(bt_lat_len2))
            : OrthonormalTangent(normal);
    }
    const Vec3 ball_tangent_component = ApplyInverseBallInertiaWorld(
        mutator.ball_mass,
        mutator.ball_radius,
        Cross(rel_ball_contact, tangent));
    const Vec3 bt_ball_normal_component = ApplyInverseBallInertiaWorldBtMutator(ball, mutator,
        Cross(rel_ball_contact_bt, normal));
    const Vec3 bt_ball_tangent_component = ApplyInverseBallInertiaWorldBtMutator(ball, mutator,
        Cross(rel_ball_contact_bt, tangent));
    AppendBallVelocityContactRows(
        normal_rows,
        normal_row_count,
        friction_rows,
        friction_row_count,
        normal,
        mutator.ball_restitution,
        mutator.ball_friction,
        ball_inv_mass,
        rel_ball_contact,
        ball_normal_component,
        -1,
        0.0f,
        Vec3(0.0f, 0.0f, 0.0f),
        Vec3(0.0f, 0.0f, 0.0f),
        ball_tangent_component,
        Vec3(0.0f, 0.0f, 0.0f),
        restitution_ball_linear,
        ball_angular_world,
        Vec3(0.0f, 0.0f, 0.0f),
        Vec3(0.0f, 0.0f, 0.0f),
        rhs_ball_linear,
        ball_angular_world,
        Vec3(0.0f, 0.0f, 0.0f),
        Vec3(0.0f, 0.0f, 0.0f),
        external_linear_delta,
        Vec3(0.0f, 0.0f, 0.0f),
        Vec3(0.0f, 0.0f, 0.0f),
        Vec3(0.0f, 0.0f, 0.0f),
        tangent_basis_rel_vel,
        friction_base_rel_vel,
        &rel_ball_contact_bt,
        nullptr,
        bt_ball_linear_velocity_override,
        &bt_ball_normal_component,
        nullptr,
        &bt_ball_tangent_component,
        nullptr,
        &bt_tangent_basis_rel_vel,
        nullptr, nullptr, nullptr, nullptr,
        true);
    (void)normal_restitution_rel_vel;
    (void)normal_base_rel_vel;
}

ROCKETGPU_HD void ClampControls(CarControls& c) {
    c.throttle = Clamp(c.throttle, -1.0f, 1.0f);
    c.steer = Clamp(c.steer, -1.0f, 1.0f);
    c.pitch = Clamp(c.pitch, -1.0f, 1.0f);
    c.yaw = Clamp(c.yaw, -1.0f, 1.0f);
    c.roll = Clamp(c.roll, -1.0f, 1.0f);
    c.jump = c.jump ? 1 : 0;
    c.boost = c.boost ? 1 : 0;
    c.handbrake = c.handbrake ? 1 : 0;
}

ROCKETGPU_HD void ResetCarCommonState(CarState& car, const MutatorConfig& mutator, float spawn_boost) {
    SetCarLinearVelocityBt(car, Vec3(0.0f, 0.0f, 0.0f));
    car.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
    car.world_contact = {};
    car.flip_rel_torque = Vec3(0.0f, 0.0f, 0.0f);
    car.boost = spawn_boost;
    car.jump_time = 0.0f;
    car.flip_time = 0.0f;
    car.internal_jump_time = 0.0f;
    car.internal_flip_time = 0.0f;
    car.air_time = 0.0f;
    car.air_time_since_jump = 0.0f;
    car.boosting_time = 0.0f;
    car.time_since_boosted = 0.0f;
    car.supersonic_time = 0.0f;
    car.handbrake_val = 0.0f;
    car.auto_flip_timer = 0.0f;
    car.auto_flip_torque_scale = 0.0f;
    car.demo_respawn_timer = 0.0f;
    car.tick_count_since_update = 0;
    car.is_on_ground = 1;

    const uint8_t spawn_contact = 0;
    car.wheels_with_contact[0] = spawn_contact;
    car.wheels_with_contact[1] = spawn_contact;
    car.wheels_with_contact[2] = spawn_contact;
    car.wheels_with_contact[3] = spawn_contact;
    car.has_jumped = 0;
    car.has_double_jumped = 0;
    car.has_flipped = 0;
    car.is_flipping = 0;
    car.is_jumping = 0;
    car.is_boosting = 0;
    car.is_supersonic = 0;
    car.is_auto_flipping = 0;
    car.is_demoed = 0;
    car.just_respawned = 0;
    car.car_contact.other_car_id = 0;
    car.car_contact.cooldown_timer = 0.0f;
    car.vel_impulse_cache = Vec3(0.0f, 0.0f, 0.0f);
    car.ball_hit_info = BallHitInfo{};
    car.last_controls = CarControls{};
    (void)mutator;
}

ROCKETGPU_HD void PlaceCarAtKickoffSpawn(
    CarState& car,
    Team team,
    const SpawnPos& spawn,
    float z,
    const MutatorConfig& mutator,
    float boost_amount) {
    float x = spawn.x;
    float y = spawn.y;
    float yaw = spawn.yaw;

    float yaw_official = spawn.yaw;
    if (team == Team::Orange) {
        x = -x;
        y = -y;
        yaw += Pi();
        yaw_official = static_cast<float>(static_cast<double>(spawn.yaw) + 3.14159265358979323846);
    }
    SetCarOriginUu(car, Vec3(x, y, z));
    car.rot = FromYaw(WrapAngle(yaw));
    SeedCarBasisFromSetEulerYPR(car, yaw_official);
    ResetCarCommonState(car, mutator, boost_amount);
}

ROCKETGPU_HD void PlaceCarAtRespawnSpawn(
    CarState& car,
    Team team,
    const SpawnPos& spawn,
    float z,
    const MutatorConfig& mutator,
    float boost_amount) {
    const float x = spawn.x;
    const float y = (team == Team::Blue) ? spawn.y : -spawn.y;
    const float yaw = spawn.yaw + (team == Team::Blue ? 0.0f : Pi());

    const float yaw_official = (team == Team::Blue)
        ? spawn.yaw
        : static_cast<float>(static_cast<double>(spawn.yaw) + 3.14159265358979323846);
    SetCarOriginUu(car, Vec3(x, y, z));
    car.rot = FromYaw(WrapAngle(yaw));
    SeedCarBasisFromSetEulerYPR(car, yaw_official);
    ResetCarCommonState(car, mutator, boost_amount);
}

ROCKETGPU_HD bool BoostPadLayoutFromConfig(const SimConfig& cfg) {
    return cfg.use_custom_boost_pads != 0 || cfg.boost_pad_count > 0;
}
ROCKETGPU_HD int BoostPadCount(const SimConfig& cfg) {
    return BoostPadLayoutFromConfig(cfg) ? cfg.boost_pad_count : BoostPadCountSoccar();
}
ROCKETGPU_HD Vec3 BoostPadLoc(const SimConfig& cfg, int p) {
    return BoostPadLayoutFromConfig(cfg) ? cfg.boost_pads[p].pos : BoostPadLocSoccar(p);
}
ROCKETGPU_HD uint8_t BoostPadIsBig(const SimConfig& cfg, int p) {
    return BoostPadLayoutFromConfig(cfg) ? cfg.boost_pads[p].is_big : BoostPadIsBigSoccar(p);
}

ROCKETGPU_HD uint32_t AdvanceDeterministicRng(uint32_t state) {
    return state * 1664525u + 1013904223u;
}

ROCKETGPU_HD uint32_t MixSeed(uint64_t seed, int env_idx, int extra);

ROCKETGPU_HD void BuildKickoffOrder(uint64_t seed, int env_idx, int* order) {
    for (int i = 0; i < 5; ++i) {
        order[i] = i;
    }
    uint32_t state = MixSeed(seed, env_idx, 0x6b1d4f2a);
    for (int i = 5 - 1; i > 0; --i) {
        state = AdvanceDeterministicRng(state);
        const int j = static_cast<int>(state % static_cast<uint32_t>(i + 1));
        const int tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }
}

ROCKETGPU_HD void PlaceNthTeamCarAtKickoff(
    int base,
    int count,
    Team team,
    int nth,
    Team* teams,
    CarState* cars,
    const SpawnPos& spawn,
    const MutatorConfig& mutator,
    const EnvState& env_state) {
    int seen = 0;
    for (int k = 0; k < count; ++k) {
        const int i = CarVisitSlot(env_state, k, count);
        const int idx = base + i;
        if (teams[idx] != team) {
            continue;
        }
        if (seen == nth) {
            PlaceCarAtKickoffSpawn(cars[idx], team, spawn, CarSpawnRestZ(), mutator, mutator.boost_spawn_amount);
            return;
        }
        ++seen;
    }
}

ROCKETGPU_HD void ResetBall(BallState& ball) {
    constexpr float kUuToBt = 1.0f / 50.0f;
    SetBallOriginBt(ball, Vec3(0.0f, 0.0f, BallRestZ()) * kUuToBt);
    ball.rot = Quat(1.0f, 0.0f, 0.0f, 0.0f);
    ball.basis_forward = Vec3(1.0f, 0.0f, 0.0f);
    ball.basis_right = Vec3(0.0f, 1.0f, 0.0f);
    ball.basis_up = Vec3(0.0f, 0.0f, 1.0f);
    ball.basis_authoritative = 1;
    SetBallLinearVelocityBt(ball, Vec3(0.0f, 0.0f, 0.0f));
    ball.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
    ball.vel_impulse_cache = Vec3(0.0f, 0.0f, 0.0f);
    ball.vel_impulse_cache_bt = Vec3(0.0f, 0.0f, 0.0f);
    ball.tick_count_when_extra_impulse_applied = 0;
    ball.tick_count_since_update = 0;
    ball.hs_info = BallState::HeatseekerInfo{};
    ball.ds_info = BallState::DropshotInfo{};
    ball.sleeping = 0;
    ball.deactivation_time = 0.0f;
}

ROCKETGPU_HD void ResetBall(BallState& ball, uint8_t game_mode) {
    ResetBall(ball);
    if (game_mode == static_cast<uint8_t>(Mode::Soccar)) {
        // The game spawns the kickoff ball asleep at z 92.75 (every recorded kickoff: exactly still, 0.4 uu inside the
        // ball's 93.15 rest height, which a simulated ball can't hold), so it takes no gravity until a car touches it.
        constexpr float kUuToBt = 1.0f / 50.0f;
        SetBallOriginBt(ball, Vec3(0.0f, 0.0f, 92.75f) * kUuToBt);
        ball.sleeping = 1;
    }
    if (game_mode == static_cast<uint8_t>(Mode::Snowday)) {
        constexpr float kUuToBt = 1.0f / 50.0f;
        SetBallLinearVelocityBt(ball, Vec3(0.0f, 0.0f, 1.1920928955078125e-7f) * kUuToBt);
    }
}

ROCKETGPU_HD void ResetBoostPads(const SimConfig& cfg, BoostPadState* pads, int env_idx) {
    const int base = env_idx * BoostPadCount(cfg);
    for (int p = 0; p < BoostPadCount(cfg); ++p) {
        pads[base + p].cooldown = 0.0f;
        pads[base + p].is_active = 1;
        pads[base + p].is_big = BoostPadIsBig(cfg, p);
        pads[base + p].prev_locked_car_id = 0;
        pads[base + p].cur_locked_car_id = 0;
    }
}

ROCKETGPU_HD uint32_t MixSeed(uint64_t seed, int env_idx, int extra) {
    uint64_t x = seed;
    x ^= static_cast<uint64_t>(env_idx + 0x9e3779b9U) + (x << 6U) + (x >> 2U);
    x ^= static_cast<uint64_t>(extra + 0x85ebca6bU) + (x << 13U) + (x >> 7U);
    return static_cast<uint32_t>(x & 0xffffffffu);
}

ROCKETGPU_HD void ResetEnvToKickoff(
    int env_idx,
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    CarControls* controls,
    BallState* balls,
    BoostPadState* pads,
    uint64_t seed,
    bool preserve_score,
    EnvState* env_states,
    uint8_t kickoff_seed_mode = 0,
    RespawnRngState* respawn_rng = nullptr,
    const int* kickoff_order_override = nullptr) {
    constexpr float kCarSpawnExtraOffsetY = 250.0f;

    EnvState saved = env_states[env_idx];
    env_states[env_idx] = EnvState{};

    env_states[env_idx].car_visit_order_mode = saved.car_visit_order_mode;
    env_states[env_idx].car_visit_order_packed = saved.car_visit_order_packed;
    if (preserve_score) {
        env_states[env_idx].blue_score = saved.blue_score;
        env_states[env_idx].orange_score = saved.orange_score;
        env_states[env_idx].tick = saved.tick;
        env_states[env_idx].last_scored_team = saved.last_scored_team;
    }

    ResetBall(balls[env_idx], cfg.mutator.game_mode);
    if (cfg.mutator.game_mode == static_cast<uint8_t>(Mode::Heatseeker)) {
        const uint32_t draw = AdvanceDeterministicRng(MixSeed(seed ^ 0x9e3779b97f4a7c15ull, env_idx, 0x48534b4f));
        const float y_scale = (draw % 2u) ? 1.0f : -1.0f;
        BallState& hs_ball = balls[env_idx];
        hs_ball.pos = Vec3(-1000.0f, -2220.0f * y_scale, 92.75f);
        hs_ball.vel = Vec3(0.0f, -65.0f * y_scale, 650.0f);
        hs_ball.pos_bt = hs_ball.pos * (1.0f / 50.0f);
        hs_ball.vel_bt = hs_ball.vel * (1.0f / 50.0f);
        hs_ball.hs_info = {};
    }
    ResetBoostPads(cfg, pads, env_idx);

    const int base = env_idx * cfg.max_cars_per_env;
    const int count = env_car_count[env_idx];
    int blue_count = 0;
    int orange_count = 0;
    for (int i = 0; i < count; ++i) {
        const Team team = teams[base + i];
        if (team == Team::Blue) {
            ++blue_count;
        } else {
            ++orange_count;
        }
    }

    for (int i = 0; i < cfg.max_cars_per_env; ++i) {
        controls[base + i] = CarControls{};
    }

    int kickoff_order[5] = {0, 1, 2, 3, 4};
    if (kickoff_order_override != nullptr) {
        for (int i = 0; i < 5; ++i) kickoff_order[i] = kickoff_order_override[i];
    } else if (respawn_rng != nullptr) {
        RespawnRngState temp_engine;
        RespawnRngState* engine = nullptr;
        const uint32_t engine_seed = kickoff_seed_mode == 1
            ? static_cast<uint32_t>(seed)
            : MixSeed(seed, env_idx, 0x6b1d4f2a);
        if (kickoff_seed_mode == 2 || static_cast<int32_t>(engine_seed) == -1) {
            engine = &respawn_rng[env_idx];
        } else {
            SeedRespawnRng(temp_engine, respawn_rng[env_idx].kind, engine_seed);
            engine = &temp_engine;
        }
        ShuffleLikePlatformStd(kickoff_order, *engine);
    } else {
        BuildKickoffOrder(seed, env_idx, kickoff_order);
    }

    const int kickoff_position_amount = (blue_count > orange_count) ? blue_count : orange_count;
    int respawn_counts[4] = {0, 0, 0, 0};
    for (int slot = 0; slot < kickoff_position_amount; ++slot) {
        SpawnPos spawn{};
        if (slot < 5) {
            spawn = cfg.mutator.game_mode == static_cast<uint8_t>(Mode::Heatseeker)
                        ? HeatseekerKickoffSpawn(kickoff_order[slot])
                        : SoccarKickoffSpawn(kickoff_order[slot]);
        } else {
            const int respawn_idx = (slot - 5) % 4;
            spawn = SoccarRespawnSpawn(respawn_idx);
            spawn.y += kCarSpawnExtraOffsetY * static_cast<float>(respawn_counts[respawn_idx]);
            ++respawn_counts[respawn_idx];
        }

        if (slot < blue_count) {
            PlaceNthTeamCarAtKickoff(base, count, Team::Blue, slot, teams, cars, spawn, cfg.mutator, env_states[env_idx]);
        }
        if (slot < orange_count) {
            PlaceNthTeamCarAtKickoff(base, count, Team::Orange, slot, teams, cars, spawn, cfg.mutator, env_states[env_idx]);
        }
    }
}

ROCKETGPU_HD void DemolishCar(CarState& car, const MutatorConfig& mutator) {
    car.is_demoed = 1;
    car.demo_respawn_timer = mutator.demo_respawn_time;
}

ROCKETGPU_HD int PickRespawnSlot(uint64_t seed, int env_idx, uint64_t current_tick, uint32_t car_id) {
    uint32_t state = MixSeed(seed ^ (current_tick * 0x9e3779b97f4a7c15ull), env_idx, static_cast<int>(car_id * 0x45d9f3bu));
    state = AdvanceDeterministicRng(state);
    return static_cast<int>(state % 4u);
}

ROCKETGPU_HD void RespawnCar(CarState& car, Team team, int slot, const MutatorConfig& mutator) {
    const SpawnPos spawn = SoccarRespawnSpawn(slot % 4);
    PlaceCarAtRespawnSpawn(car, team, spawn, CarRespawnZ(), mutator, mutator.boost_spawn_amount);
    car.vel_impulse_cache = Vec3(0.0f, 0.0f, 0.0f);
    car.just_respawned = 1;
}

ROCKETGPU_HD void UpdateSupersonicCalDemo(CarState& car, const MutatorConfig& mutator, float dt) {
    const float speed_sq = LengthSq(car.vel);
    if (speed_sq >= (mutator.supersonic_start_speed * mutator.supersonic_start_speed)) {
        car.is_supersonic = 1;
        car.supersonic_time = 0.0f;
    } else if (car.is_supersonic) {
        car.supersonic_time += dt;
        if (car.supersonic_time >= mutator.supersonic_maintain_max_time ||
            speed_sq < (mutator.supersonic_maintain_min_speed * mutator.supersonic_maintain_min_speed)) {
            car.is_supersonic = 0;
            car.supersonic_time = 0.0f;
        }
    }
}

enum ArenaRayHitKind : uint8_t {
    ROCKETGPU_ARENA_RAY_HIT_UNKNOWN = 0,
    ROCKETGPU_ARENA_RAY_HIT_STATIC_PLANE = 1,
    ROCKETGPU_ARENA_RAY_HIT_CORNER_CYLINDER = 2,
    ROCKETGPU_ARENA_RAY_HIT_ARENA_MESH = 3,
    ROCKETGPU_ARENA_RAY_HIT_CAR = 4,
    ROCKETGPU_ARENA_RAY_HIT_BALL = 5,
};

struct ArenaRayHit {
    float t = 0.0f;
    Vec3 point = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 point_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 normal = Vec3(0.0f, 0.0f, 1.0f);
    float ray_fraction = 0.0f;
    uint8_t hit = 0;
    uint8_t kind = ROCKETGPU_ARENA_RAY_HIT_UNKNOWN;
    int32_t mesh_index = -1;
    int32_t tri_index = -1;

    float cal_min_fraction_sq = 0.0f;
};

ROCKETGPU_HD void MaybeUpdateBestHit(
    bool valid,
    float t,
    const Vec3& point,
    const Vec3& point_bt,
    const Vec3& normal,
    float max_len,
    ArenaRayHit& best,
    uint8_t kind = ROCKETGPU_ARENA_RAY_HIT_UNKNOWN,
    int32_t mesh_index = -1,
    int32_t tri_index = -1);
ROCKETGPU_HD void MaybeUpdateBestHitFraction(
    bool valid,
    float fraction,
    float max_len,
    const Vec3& point,
    const Vec3& point_bt,
    const Vec3& normal,
    ArenaRayHit& best,
    uint8_t kind = ROCKETGPU_ARENA_RAY_HIT_UNKNOWN,
    int32_t mesh_index = -1,
    int32_t tri_index = -1);

inline thread_local const int32_t* g_host_validation_respawn_order_cache = nullptr;
inline thread_local int32_t g_host_validation_respawn_order_env_count = 0;
inline thread_local int32_t g_host_validation_respawn_order_stride = 0;

inline bool TryGetHostValidationRespawnOrder(
    int env_idx,
    const int32_t*& order) {
    return false;
}

#if defined(ROCKETGPU_CAN_COMPILE_OFFICIAL_HOST_HELPERS)
struct ArenaMeshTriangleHost {
    Vec3 a = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 b = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 c = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 normal = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 min = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 max = Vec3(0.0f, 0.0f, 0.0f);
    float edge_v0v1_angle = 2.0f * Pi();
    float edge_v1v2_angle = 2.0f * Pi();
    float edge_v2v0_angle = 2.0f * Pi();
    uint8_t edge_flags = 0;
    int32_t mesh_index = -1;
    int32_t tri_index = -1;
    uint16_t quantized_aabb_min[3] = {0, 0, 0};
    uint16_t quantized_aabb_max[3] = {0, 0, 0};
};

ROCKETGPU_HD bool ComputeCarArenaTriangleContactHost(
    const CarState& car,
    const CarConfig& cfg,
    const ArenaMeshTriangleHost& tri,
    float contact_threshold_bt,
    Vec3& out_normal,
    float& out_distance_bt);

ROCKETGPU_HD const std::vector<ArenaMeshTriangleHost>& ArenaMeshTrianglesHost() {
    static std::once_flag once;
    static std::vector<ArenaMeshTriangleHost> triangles;
    std::call_once(once, []() {
        struct TriangleIndex {
            int32_t vertex_index[3];
        };
        struct MeshVertex {
            float x;
            float y;
            float z;
        };
        auto get_internal_edge_angle = [](const Vec3& ref_axis0, const Vec3& ref_axis1, const Vec3& swing_axis) {
            return SafeAtan2(Dot(swing_axis, ref_axis0), Dot(swing_axis, ref_axis1));
        };
        auto populate_triangle_internal_edge_info = [&](const std::vector<TriangleIndex>& tri_indices,
                                                        std::vector<ArenaMeshTriangleHost>& out_triangles,
                                                        int tri_a_idx,
                                                        int tri_b_idx) {
            (void)tri_indices;

            constexpr float kEqualVertexThresholdSq = 1.0e-4f * 1.0e-4f;
            const ArenaMeshTriangleHost& tri_a_ref = out_triangles[static_cast<size_t>(tri_a_idx)];
            const ArenaMeshTriangleHost& tri_b_ref = out_triangles[static_cast<size_t>(tri_b_idx)];
            const Vec3 verts_a[3] = {tri_a_ref.a, tri_a_ref.b, tri_a_ref.c};
            const Vec3 verts_b[3] = {tri_b_ref.a, tri_b_ref.b, tri_b_ref.c};
            int shared_verts_a[2] = {-1, -1};
            int shared_verts_b[2] = {-1, -1};
            int shared_count = 0;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    const Vec3 delta = verts_a[i] - verts_b[j];
                    if (LengthSq(delta) < kEqualVertexThresholdSq) {
                        shared_verts_a[shared_count] = i;
                        shared_verts_b[shared_count] = j;
                        ++shared_count;
                        if (shared_count >= 3) {
                            return;
                        }
                        break;
                    }
                }
                if (shared_count >= 3) {
                    return;
                }
            }
            if (shared_count != 2) {
                return;
            }

            if (shared_verts_a[0] == 0 && shared_verts_a[1] == 2) {
                std::swap(shared_verts_a[0], shared_verts_a[1]);
                std::swap(shared_verts_b[0], shared_verts_b[1]);
            }

            ArenaMeshTriangleHost& tri_a = out_triangles[static_cast<size_t>(tri_a_idx)];
            const ArenaMeshTriangleHost& tri_b = out_triangles[static_cast<size_t>(tri_b_idx)];
            const Vec3 tri_a_vertices[3] = {tri_a.a, tri_a.b, tri_a.c};
            const Vec3 tri_b_vertices[3] = {tri_b.a, tri_b.b, tri_b.c};

            const int sum_verts_a = shared_verts_a[0] + shared_verts_a[1];
            const int other_index_a = 3 - sum_verts_a;
            const int other_index_b = 3 - (shared_verts_b[0] + shared_verts_b[1]);

            Vec3 edge = tri_a_vertices[shared_verts_a[1]] - tri_a_vertices[shared_verts_a[0]];
            if (LengthSq(edge) <= 1.0e-12f) {
                return;
            }
            edge = Normalize(edge, Vec3(1.0f, 0.0f, 0.0f));

            const Vec3 normal_a = Normalize(tri_a.normal, Vec3(0.0f, 0.0f, 1.0f));
            const Vec3 tri_b0 = tri_b_vertices[shared_verts_b[1]];
            const Vec3 tri_b1 = tri_b_vertices[shared_verts_b[0]];
            const Vec3 tri_b2 = tri_b_vertices[other_index_b];
            Vec3 normal_b = Normalize(Cross(tri_b1 - tri_b0, tri_b2 - tri_b0), Normalize(tri_b.normal, Vec3(0.0f, 0.0f, 1.0f)));

            Vec3 edge_cross_a = Normalize(Cross(edge, normal_a), normal_a);
            if (Dot(edge_cross_a, tri_a_vertices[other_index_a] - tri_a_vertices[shared_verts_a[0]]) < 0.0f) {
                edge_cross_a = -edge_cross_a;
            }

            Vec3 edge_cross_b = Normalize(Cross(edge, normal_b), normal_b);
            if (Dot(edge_cross_b, tri_b_vertices[other_index_b] - tri_b_vertices[shared_verts_b[0]]) < 0.0f) {
                edge_cross_b = -edge_cross_b;
            }

            float corrected_angle = 0.0f;
            bool is_convex = false;
            const Vec3 calculated_edge = Cross(edge_cross_a, edge_cross_b);
            if (LengthSq(calculated_edge) >= 0.0001f) {
                const Vec3 calculated_edge_unit = Normalize(calculated_edge, edge);
                const Vec3 calculated_normal_a = Normalize(Cross(calculated_edge_unit, edge_cross_a), normal_a);
                const float angle2 = get_internal_edge_angle(calculated_normal_a, edge_cross_a, edge_cross_b);
                const float ang4 = Pi() - angle2;
                is_convex = Dot(normal_a, edge_cross_b) < 0.0f;
                corrected_angle = is_convex ? ang4 : -ang4;
            }

            auto set_edge_info = [&](float& edge_angle, uint8_t convex_flag, uint8_t swap_flag, const Vec3& edge_axis) {
                Vec3 computed_normal_b = Rotate(
                    AxisAngle(Normalize(edge_axis, Vec3(1.0f, 0.0f, 0.0f)), -corrected_angle),
                    normal_a);
                if (Dot(computed_normal_b, normal_b) < 0.0f) {
                    computed_normal_b = -computed_normal_b;
                    tri_a.edge_flags = static_cast<uint8_t>(tri_a.edge_flags | swap_flag);
                }
                edge_angle = -corrected_angle;
                if (is_convex) {
                    tri_a.edge_flags = static_cast<uint8_t>(tri_a.edge_flags | convex_flag);
                }
            };

            switch (sum_verts_a) {
                case 1:
                    set_edge_info(
                        tri_a.edge_v0v1_angle,
                        ROCKETGPU_TRI_INFO_V0V1_CONVEX,
                        ROCKETGPU_TRI_INFO_V0V1_SWAP_NORMALB,
                        tri_a_vertices[0] - tri_a_vertices[1]);
                    break;
                case 2:
                    set_edge_info(
                        tri_a.edge_v2v0_angle,
                        ROCKETGPU_TRI_INFO_V2V0_CONVEX,
                        ROCKETGPU_TRI_INFO_V2V0_SWAP_NORMALB,
                        tri_a_vertices[2] - tri_a_vertices[0]);
                    break;
                case 3:
                    set_edge_info(
                        tri_a.edge_v1v2_angle,
                        ROCKETGPU_TRI_INFO_V1V2_CONVEX,
                        ROCKETGPU_TRI_INFO_V1V2_SWAP_NORMALB,
                        tri_a_vertices[1] - tri_a_vertices[2]);
                    break;
                default:
                    break;
            }
        };

        std::filesystem::path root(OfficialCollisionMeshRoot());
        const std::filesystem::path soccar_dir = root / "soccar";
        if (!std::filesystem::exists(soccar_dir)) {
            throw std::runtime_error("official RocketSim collision meshes not found");
        }

        for (int idx = 0; idx < 16; ++idx) {
            const std::filesystem::path mesh_path = soccar_dir / ("mesh_" + std::to_string(idx) + ".cmf");
            std::ifstream input(mesh_path, std::ios::binary);
            if (!input) {
                throw std::runtime_error("failed to open collision mesh: " + mesh_path.string());
            }

            int32_t num_tris = 0;
            int32_t num_vertices = 0;
            input.read(reinterpret_cast<char*>(&num_tris), sizeof(num_tris));
            input.read(reinterpret_cast<char*>(&num_vertices), sizeof(num_vertices));
            if (!input || num_tris <= 0 || num_vertices <= 0) {
                throw std::runtime_error("invalid collision mesh header: " + mesh_path.string());
            }

            std::vector<TriangleIndex> tri_indices(static_cast<size_t>(num_tris));
            std::vector<MeshVertex> vertices(static_cast<size_t>(num_vertices));
            input.read(reinterpret_cast<char*>(tri_indices.data()), static_cast<std::streamsize>(tri_indices.size() * sizeof(TriangleIndex)));
            input.read(reinterpret_cast<char*>(vertices.data()), static_cast<std::streamsize>(vertices.size() * sizeof(MeshVertex)));
            if (!input) {
                throw std::runtime_error("failed to read collision mesh: " + mesh_path.string());
            }

            Vec3 mesh_min(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
            Vec3 mesh_max(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
            for (int32_t tri_i = 0; tri_i < num_tris; ++tri_i) {
                const TriangleIndex& tri_idx = tri_indices[static_cast<size_t>(tri_i)];
                if (tri_idx.vertex_index[0] < 0 || tri_idx.vertex_index[0] >= num_vertices ||
                    tri_idx.vertex_index[1] < 0 || tri_idx.vertex_index[1] >= num_vertices ||
                    tri_idx.vertex_index[2] < 0 || tri_idx.vertex_index[2] >= num_vertices) {
                    throw std::runtime_error("collision mesh triangle index out of range: " + mesh_path.string());
                }

                const MeshVertex& va = vertices[static_cast<size_t>(tri_idx.vertex_index[0])];
                const MeshVertex& vb = vertices[static_cast<size_t>(tri_idx.vertex_index[1])];
                const MeshVertex& vc = vertices[static_cast<size_t>(tri_idx.vertex_index[2])];

                ArenaMeshTriangleHost tri;
                tri.a = Vec3(va.x, va.y, va.z);
                tri.b = Vec3(vb.x, vb.y, vb.z);
                tri.c = Vec3(vc.x, vc.y, vc.z);
                tri.normal = Normalize(Cross(tri.b - tri.a, tri.c - tri.a), Vec3(0.0f, 0.0f, 1.0f));
                tri.min = Vec3(
                    Min(tri.a.x, Min(tri.b.x, tri.c.x)),
                    Min(tri.a.y, Min(tri.b.y, tri.c.y)),
                    Min(tri.a.z, Min(tri.b.z, tri.c.z)));
                tri.max = Vec3(
                    Max(tri.a.x, Max(tri.b.x, tri.c.x)),
                    Max(tri.a.y, Max(tri.b.y, tri.c.y)),
                    Max(tri.a.z, Max(tri.b.z, tri.c.z)));
                tri.mesh_index = idx;
                tri.tri_index = tri_i;
                mesh_min = MinComponents(mesh_min, tri.min);
                mesh_max = MaxComponents(mesh_max, tri.max);
                triangles.push_back(tri);
            }

            ArenaMeshBvhQuantization quantization;
            SetArenaMeshBvhQuantization(quantization, mesh_min, mesh_max);
            for (size_t tri_pos = triangles.size() - static_cast<size_t>(num_tris); tri_pos < triangles.size(); ++tri_pos) {
                ArenaMeshTriangleHost& tri = triangles[tri_pos];
                QuantizeArenaMeshBvhPoint(quantization, tri.min, 0, tri.quantized_aabb_min);
                QuantizeArenaMeshBvhPoint(quantization, tri.max, 1, tri.quantized_aabb_max);
            }

            const size_t mesh_base = triangles.size() - static_cast<size_t>(num_tris);
            for (int32_t tri_a = 0; tri_a < num_tris; ++tri_a) {
                const ArenaMeshTriangleHost& tri_a_ref = triangles[mesh_base + static_cast<size_t>(tri_a)];
                for (int32_t tri_b = 0; tri_b < num_tris; ++tri_b) {
                    if (tri_a == tri_b) continue;
                    const ArenaMeshTriangleHost& tri_b_ref = triangles[mesh_base + static_cast<size_t>(tri_b)];
                    if (tri_a_ref.max.x < tri_b_ref.min.x || tri_a_ref.min.x > tri_b_ref.max.x ||
                        tri_a_ref.max.y < tri_b_ref.min.y || tri_a_ref.min.y > tri_b_ref.max.y ||
                        tri_a_ref.max.z < tri_b_ref.min.z || tri_a_ref.min.z > tri_b_ref.max.z) {
                        continue;
                    }
                    populate_triangle_internal_edge_info(
                        tri_indices,
                        triangles,
                        static_cast<int>(mesh_base + static_cast<size_t>(tri_a)),
                        static_cast<int>(mesh_base + static_cast<size_t>(tri_b)));
                }
            }
        }
    });
    return triangles;
}

#endif
