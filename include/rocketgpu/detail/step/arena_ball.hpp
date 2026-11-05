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

ROCKETGPU_HD int CarWorldRowOrderKey(int source_tag) {
    SharedContactOrderGroup group;
    group.bucket = SharedContactOrderBucket::World;
    group.source_tag = source_tag;
    return BulletContactOrderInitialKey(group);
}

template <int kSegs = 1>
ROCKETGPU_HD void ApplyGameCarWorldContactNormal(
    ContactAccumulator& acc,
    const CarWorldVelocityContactRow* const (&seg_rows)[kSegs],
    const int (&seg_counts)[kSegs]) {
    if (!acc.hit) {
        return;
    }
    int min_key = 0x7fffffff;
    int max_key = -1;
    Vec3 single_key_sum(0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int seg = 0; seg < kSegs; ++seg) {
        const CarWorldVelocityContactRow* const normal_rows = seg_rows[seg];
        const int normal_row_count = normal_rows ? seg_counts[seg] : 0;
        for (int row_idx = 0; row_idx < normal_row_count; ++row_idx) {
            if (!normal_rows[row_idx].valid) {
                continue;
            }
            const int k = CarWorldRowOrderKey(normal_rows[row_idx].source_tag);
            if (k < min_key) {
                min_key = k;
            }
            if (k > max_key) {
                max_key = k;
            }
            single_key_sum += normal_rows[row_idx].normal;
        }
    }
    if (max_key < 0) {
        return;
    }
    Vec3 sum = single_key_sum;
    if (min_key != max_key) {
        sum = Vec3(0.0f, 0.0f, 0.0f);
        int key = min_key;
        for (;;) {
#pragma unroll
            for (int seg = 0; seg < kSegs; ++seg) {
                const CarWorldVelocityContactRow* const normal_rows = seg_rows[seg];
                const int normal_row_count = normal_rows ? seg_counts[seg] : 0;
                for (int row_idx = 0; row_idx < normal_row_count; ++row_idx) {
                    if (normal_rows[row_idx].valid && CarWorldRowOrderKey(normal_rows[row_idx].source_tag) == key) {
                        sum += normal_rows[row_idx].normal;
                    }
                }
            }
            const int prev_key = key;
            key = 0x7fffffff;
            bool found = false;
#pragma unroll
            for (int seg = 0; seg < kSegs; ++seg) {
                const CarWorldVelocityContactRow* const normal_rows = seg_rows[seg];
                const int normal_row_count = normal_rows ? seg_counts[seg] : 0;
                for (int row_idx = 0; row_idx < normal_row_count; ++row_idx) {
                    if (!normal_rows[row_idx].valid) {
                        continue;
                    }
                    const int k = CarWorldRowOrderKey(normal_rows[row_idx].source_tag);
                    if (k > prev_key && k < key) {
                        key = k;
                        found = true;
                    }
                }
            }
            if (!found) {
                break;
            }
        }
    }
    const float len_sq = Dot(sum, sum);
    if (len_sq >= 1e-8f) {
        const Vec3 n = sum * (1.0f / SqrtScalar(len_sq));
        acc.latest_normal = Vec3(0.0f + n.x, 0.0f + n.y, 0.0f + n.z);
    }
}
ROCKETGPU_HD void ApplyPredictiveCarWorldContactNormal(
    CarState* cars,
    int count,
    uint32_t predictive_hit_mask,
    EnvStridedArray<CarSharedVelocityContactRow> rows,
    int row_count) {
    for (int i = 0; i < count && i < 8; ++i) {
        CarState& car = cars[i];
        if (((predictive_hit_mask >> (i + 1)) & 1u) == 0 || !car.world_contact.has_contact) {
            continue;
        }
        Vec3 sum(0.0f, 0.0f, 0.0f);
        bool any = false;
        int prev_key = -1;
        for (;;) {
            int key = 0x7fffffff;
            for (int r = 0; r < row_count; ++r) {
                const CarSharedVelocityContactRow& row = rows[r];
                if (row.valid && row.body_a_local_idx == i) {
                    const int k = CarWorldRowOrderKey(row.source_tag);
                    if (k > prev_key && k < key) {
                        key = k;
                    }
                }
            }
            if (key == 0x7fffffff) {
                break;
            }
            for (int r = 0; r < row_count; ++r) {
                const CarSharedVelocityContactRow& row = rows[r];
                if (row.valid && row.body_a_local_idx == i && CarWorldRowOrderKey(row.source_tag) == key) {
                    sum += row.normal_a;
                    any = true;
                }
            }
            prev_key = key;
        }
        const float len_sq = Dot(sum, sum);
        if (any && len_sq >= 1e-8f) {
            const Vec3 n = sum * (1.0f / SqrtScalar(len_sq));
            car.world_contact.contact_normal = Vec3(0.0f + n.x, 0.0f + n.y, 0.0f + n.z);
        }
    }
}
ROCKETGPU_HD void ApplyGameCarWorldContactNormal(
    ContactAccumulator& acc,
    const CarWorldVelocityContactRow* normal_rows,
    int normal_row_count) {
    if (!normal_rows || normal_row_count <= 0) {
        return;
    }
    const CarWorldVelocityContactRow* const seg_rows[1] = {normal_rows};
    const int seg_counts[1] = {normal_row_count};
    ApplyGameCarWorldContactNormal<1>(acc, seg_rows, seg_counts);
}

template <bool kMeshesOutOfReach = false, int kPart = 0>
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
    CarWorldLocalRows* local_rows = nullptr,
    uint32_t mesh_reach_hint = 0u,
    bool defer_world_surface = false,
    CarWorldRowsPart* part_out = nullptr,
    const Vec3* predict_vel = nullptr,
    const Vec3* predict_ang_vel = nullptr) {
    ContactAccumulator acc;
    acc.history_car = defer_world_surface ? nullptr : &car;

#if defined(__CUDA_ARCH__)
    constexpr bool kEarlyOut = kMeshesOutOfReach;
    constexpr bool kUsePlaneOutOfReach = (kPart != 1);
#else
    constexpr bool kEarlyOut = kPart == 0;
    constexpr bool kUsePlaneOutOfReach = (kPart != 1);
#endif
    float plane_d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float plane_threshold_bt = 0.0f;
    float plane_window_bt = 0.0f;
    if (kUsePlaneOutOfReach && mutator.arena_collision) {
        CarWorldPlaneSupportDistances(car, cfg, plane_d, plane_threshold_bt, plane_window_bt);
    }
    if (kEarlyOut && mutator.arena_collision && !mutator.physics_events.HasNonGoalEvents() && mesh_triangles != nullptr &&
        mesh_triangle_count > 0 && mesh_bvh_quantization != nullptr && mesh_bvh_meshes != nullptr &&
        mesh_bvh_nodes != nullptr && mesh_bvh_mesh_count > 0 && mesh_bvh_node_count > 0) {
        float early_plane_d[4];
        float early_plane_threshold_bt = plane_threshold_bt;
        float early_plane_window_bt = plane_window_bt;
        if (!kUsePlaneOutOfReach) {
            CarWorldPlaneSupportDistances(car, cfg, early_plane_d, early_plane_threshold_bt, early_plane_window_bt);
        } else {
            early_plane_d[0] = plane_d[0];
            early_plane_d[1] = plane_d[1];
            early_plane_d[2] = plane_d[2];
            early_plane_d[3] = plane_d[3];
        }
        bool has_retained_mesh_points = false;
        for (int k = 0; retained_mesh_manifolds != nullptr && k < kCarWorldMeshRetainedManifoldsPerCar; ++k) {
            has_retained_mesh_points |= retained_mesh_manifolds[k].count > 0;
        }
        int plane_groups = has_retained_mesh_points ? -1 : 0;
        const auto classify_plane = [&](float d, int bit) {
            if (plane_groups < 0) {
                return;
            }
            if (d < early_plane_threshold_bt + kCarWorldPlaneReachSlackBt ||
                Abs(d - early_plane_window_bt) <= kCarWorldPlaneReachSlackBt) {
                plane_groups = -1;
            } else if (d < early_plane_window_bt) {
                plane_groups |= bit;
            }
        };
        classify_plane(early_plane_d[0], 1);
        classify_plane(early_plane_d[1], 2);
        classify_plane(early_plane_d[2], 4);
        classify_plane(early_plane_d[3], 8);
        if (plane_groups >= 0 && (plane_groups & 4) != 0 && mutator.game_mode == static_cast<uint8_t>(Mode::Soccar)) {
            const bool gf_held = retained_plane_manifolds != nullptr &&
                                 (retained_plane_manifolds[4].count > 0 || retained_plane_manifolds[5].count > 0);
            const float gf_reach_bt = BoundingSphereRadius(cfg) * (1.0f / 50.0f) + 0.1f;
            const Vec3 gf_center_bt = (car.pos + RotateCarBasis(car, cfg.hitbox_offset)) * (1.0f / 50.0f);
            if (gf_held || (!(Abs(gf_center_bt.x) > kGoalFloorHalfXBt + gf_reach_bt) &&
                            (!(Abs(gf_center_bt.y - GoalFloorOriginBt(0).y) > kGoalFloorHalfYBt + gf_reach_bt) ||
                             !(Abs(gf_center_bt.y - GoalFloorOriginBt(1).y) > kGoalFloorHalfYBt + gf_reach_bt)))) {
                plane_groups = -1;
            }
        }
        if (plane_groups >= 0 &&
            (kMeshesOutOfReach || CarWorldMeshesOutOfReach(
                car, cfg, mesh_bvh_meshes, mesh_bvh_mesh_count, dt,
                broadphase_grid != nullptr && mesh_reach_hint == 0u ? &mesh_reach_hint : nullptr))) {
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
                append_plane_group(1, 2);
                append_plane_group(2, 1);
                append_plane_group(4, 6);
                append_plane_group(8, 5);
            }
            return acc;
        }
    }

    const CarWorldMeshCandidateStage* mesh_candidate_stage_in = nullptr;
    if (!kMeshesOutOfReach && kPart != 2 && candidate_stage != nullptr && candidate_stage->valid != 0) {
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
        (kPart != 0) || kMeshesOutOfReach || (mesh_triangles != nullptr && mesh_triangle_count > 0);

    const Vec3 center = car.pos + RotateCarBasis(car, cfg.hitbox_offset);
    const auto plane_out_of_reach = [&](int k, const Vec3& inward_normal, float boundary,
                                        CarWorldPlaneRetainedManifold* retained) {
        if (!kUsePlaneOutOfReach) {
            return false;
        }
        const float d = plane_d[k];
        if (d < plane_threshold_bt + kCarWorldPlaneReachSlackBt ||
            Abs(d - plane_window_bt) <= kCarWorldPlaneReachSlackBt) {
            return false;
        }
        if (retained != nullptr) {
            retained->count = 0;
        }
        if (d < plane_window_bt && car_idx >= 0 && local_world_order_groups) {
            AppendSharedContactOrderGroup(
                local_world_order_groups, local_world_order_group_count, SharedContactOrderBucket::World, car_idx, -1,
                CarWorldPlaneSourceTag(inward_normal, boundary), normal_row_count, 0, friction_row_count, 0,
                local_world_order_group_count);
        }
        return true;
    };
    const float precomputed_plane_threshold_bt = kUsePlaneOutOfReach ? plane_threshold_bt : -1.0f;

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

    if (kPart != 1) {
        if (!plane_out_of_reach(0, Vec3(1.0f, 0.0f, 0.0f), -ArenaExtentX(),
                                retained_plane_manifolds ? &retained_plane_manifolds[2] : nullptr)) {
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
                retained_plane_manifolds ? &retained_plane_manifolds[2] : nullptr,
                false,
                -1,
                precomputed_plane_threshold_bt);
        }
        if (!plane_out_of_reach(1, Vec3(-1.0f, 0.0f, 0.0f), -ArenaExtentX(),
                                retained_plane_manifolds ? &retained_plane_manifolds[3] : nullptr)) {
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
                retained_plane_manifolds ? &retained_plane_manifolds[3] : nullptr,
                false,
                -1,
                precomputed_plane_threshold_bt);
        }
    }
    if (kPart == 2) {
        part_out->split[0] = split_row_count;
        part_out->normal[0] = normal_row_count;
        part_out->friction[0] = friction_row_count;
        part_out->groups[0] = local_world_order_group_count;
        part_out->hit[0] = acc.hit;
        part_out->surface[0] = acc.surface;
        part_out->latest_normal[0] = acc.latest_normal;
        acc.hit = 0;
        acc.surface = 0;
        acc.latest_normal = Vec3(0.0f, 0.0f, 1.0f);
    }

    if (use_mesh_world_rows && kMeshesOutOfReach) {
        ReleaseCarWorldMeshManifoldsOutOfReach(mesh_bvh_meshes, mesh_bvh_mesh_count, mesh_bvh_quantization_count,
                                               mesh_bvh_node_count,
                                               broadphase_grid != nullptr ? retained_mesh_manifolds : nullptr);
    }
    if (use_mesh_world_rows && !kMeshesOutOfReach && kPart != 2) {
        AppendCarWorldMeshContactRowsBody<false>(
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
            broadphase_grid != nullptr ? &broadphase_grid->leaf_grid : nullptr,
            nullptr,
            kTrianglePairsNone,
            retained_mesh_manifolds,
            mesh_reach_hint,
            predict_vel,
            predict_ang_vel);
    }

    if (kPart != 1) {
        if (!plane_out_of_reach(2, Vec3(0.0f, 0.0f, 1.0f), 0.0f,
                                retained_plane_manifolds ? &retained_plane_manifolds[0] : nullptr)) {
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
                retained_plane_manifolds ? &retained_plane_manifolds[0] : nullptr,
                kPart == 2,
                -1,
                precomputed_plane_threshold_bt);
        }
        for (int gf = 0; mutator.game_mode == static_cast<uint8_t>(Mode::Soccar) && gf < 2; ++gf) {
            CarWorldPlaneRetainedManifold* const gf_retained =
                retained_plane_manifolds ? &retained_plane_manifolds[4 + gf] : nullptr;
            const bool gf_points = gf_retained != nullptr && gf_retained->count > 0;
            const float reach_bt = support_radius * (1.0f / 50.0f) + 0.1f;
            const Vec3 center_bt = center * (1.0f / 50.0f);
            if (!gf_points && (Abs(center_bt.x) > kGoalFloorHalfXBt + reach_bt ||
                               Abs(center_bt.y - GoalFloorOriginBt(gf).y) > kGoalFloorHalfYBt + reach_bt)) {
                continue;
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
                gf_retained,
                false,
                gf,
                precomputed_plane_threshold_bt);
        }
        if (!plane_out_of_reach(3, Vec3(0.0f, 0.0f, -1.0f), -ArenaHeight(),
                                retained_plane_manifolds ? &retained_plane_manifolds[1] : nullptr)) {
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
                retained_plane_manifolds ? &retained_plane_manifolds[1] : nullptr,
                kPart == 2,
                -1,
                precomputed_plane_threshold_bt);
        }
    }

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

    if (kPart != 0) {
        const int at = kPart == 2 ? 1 : 0;
        part_out->split[at] = split_row_count;
        part_out->normal[at] = normal_row_count;
        part_out->friction[at] = friction_row_count;
        part_out->groups[at] = local_world_order_group_count;
        part_out->hit[at] = acc.hit;
        part_out->surface[at] = acc.surface;
        part_out->latest_normal[at] = acc.latest_normal;
        return acc;
    }

    ApplyGameCarWorldContactNormal(acc, normal_rows, normal_row_count);

    if (shared_split_rows && shared_split_row_count && split_row_count > 0) {
        AppendCarWorldToSharedSplitRows(shared_split_rows, *shared_split_row_count, car_idx, split_rows, split_row_count);
    } else if (split_row_count > 0) {
        SolveCarWorldSplitContactRows(car, split_rows, split_row_count, dt, NativeSseMath(mutator));
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

ROCKETGPU_HD ContactAccumulator FinishCarWorldSplitRows(
    int car_idx,
    CarWorldLocalRows& mesh,
    const CarWorldRowsPart& mp,
    CarWorldLocalRows& planes,
    const CarWorldRowsPart& pp,
    EnvStridedArray<CarSharedSplitContactRow> shared_split_rows,
    int* shared_split_row_count,
    EnvStridedArray<CarSharedVelocityContactRow> shared_velocity_rows,
    int* shared_velocity_row_count,
    EnvStridedArray<CarSharedFrictionContactRow> shared_friction_rows,
    int* shared_friction_row_count,
    EnvStridedArray<SharedContactOrderGroup> shared_contact_order_groups,
    int* shared_contact_order_group_count) {
    ContactAccumulator acc;
    acc.history_car = nullptr;
    acc.hit = pp.hit[0] | mp.hit[0] | pp.hit[1];
    acc.surface = pp.surface[0] | mp.surface[0] | pp.surface[1];
    acc.latest_normal = pp.hit[1] ? pp.latest_normal[1] : (mp.hit[0] ? mp.latest_normal[0] : pp.latest_normal[0]);
    const int xs = pp.split[0];
    const int xn = pp.normal[0];
    const int xf = pp.friction[0];
    const int ms = mp.split[0];
    const int mn = mp.normal[0];
    const int mf = mp.friction[0];
    {
        const CarWorldVelocityContactRow* const seg_rows[3] = {
            planes.normal_rows, mesh.normal_rows, planes.normal_rows + xn};
        const int seg_counts[3] = {xn, mn, pp.normal[1] - xn};
        ApplyGameCarWorldContactNormal<3>(acc, seg_rows, seg_counts);
    }
    if (xs + ms + (pp.split[1] - xs) > 0) {
        AppendCarWorldToSharedSplitRows(shared_split_rows, *shared_split_row_count, car_idx, planes.split_rows, xs);
        AppendCarWorldToSharedSplitRows(shared_split_rows, *shared_split_row_count, car_idx, mesh.split_rows, ms);
        AppendCarWorldToSharedSplitRows(
            shared_split_rows, *shared_split_row_count, car_idx, planes.split_rows + xs, pp.split[1] - xs);
    }
    const int shared_normal_base = *shared_velocity_row_count;
    const int shared_friction_base = *shared_friction_row_count;
    AppendCarWorldToSharedVelocityRows(
        shared_velocity_rows, *shared_velocity_row_count, shared_friction_rows, *shared_friction_row_count, car_idx,
        planes.normal_rows, xn, planes.friction_rows, xf, shared_normal_base, 0);
    AppendCarWorldToSharedVelocityRows(
        shared_velocity_rows, *shared_velocity_row_count, shared_friction_rows, *shared_friction_row_count, car_idx,
        mesh.normal_rows, mn, mesh.friction_rows, mf, shared_normal_base, xn);
    AppendCarWorldToSharedVelocityRows(
        shared_velocity_rows, *shared_velocity_row_count, shared_friction_rows, *shared_friction_row_count, car_idx,
        planes.normal_rows + xn, pp.normal[1] - xn, planes.friction_rows + xf, pp.friction[1] - xf,
        shared_normal_base, mn);
    int group_idx = 0;
    int previous_tag = 0;
    const auto append = [&](const SharedContactOrderGroup& g, int normal_offset, int friction_offset) {
        if (g.valid == 2 && group_idx > 0 && previous_tag == 101) {
            return;
        }
        AppendSharedContactOrderGroup(
            shared_contact_order_groups,
            *shared_contact_order_group_count,
            SharedContactOrderBucket::World,
            car_idx,
            -1,
            g.source_tag,
            g.normal_row_count > 0 ? shared_normal_base + normal_offset + g.normal_row_start : -1,
            g.normal_row_count,
            g.friction_row_count > 0 ? shared_friction_base + friction_offset + g.friction_row_start : -1,
            g.friction_row_count,
            car_idx * 100000 + 1000 + group_idx);
        previous_tag = g.source_tag;
        ++group_idx;
    };
    for (int k = 0; k < pp.groups[0]; ++k) {
        append(planes.order_groups[k], 0, 0);
    }
    for (int k = 0; k < mp.groups[0]; ++k) {
        append(mesh.order_groups[k], xn, xf);
    }
    for (int k = pp.groups[0]; k < pp.groups[1]; ++k) {
        append(planes.order_groups[k], mn, mf);
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
    bool swap_remove = false,
    const GamePlaneXfBt* plane_xf = nullptr) {
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

        const Vec3 position_world_on_b = plane_xf != nullptr
            ? GamePlanePointWorldBt(*plane_xf, manifold_point.local_point_b_bt - plane_xf->o)
            : cal_origin_y != 0.0f
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

ROCKETGPU_HD float BallPlaneRelLen(const BallState& ball, float ball_radius_bt, const Vec3& inward_normal,
                                   const Vec3& plane_pos_uu, Vec3* out_rel_bt, float* out_distance_bt,
                                   Vec3* out_local_a_bt, Vec3* out_point_b_bt,
                                   const GamePlaneXfBt* xf = nullptr, Vec3* out_in_plane_bt = nullptr,
                                   float* out_depth_bt = nullptr) {
    constexpr float kUUToBT_local = 1.0f / 50.0f;
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
    float distance = 0.0f;
    Vec3 world(0.0f, 0.0f, 0.0f);
    if (xf != nullptr) {
        const Vec3 in_plane = GamePlaneVertexInPlaneBt(*xf, bx, by, bz, ball.pos_bt, vtx_local);
        distance = in_plane.z;
        world = GamePlaneProjectedWorldBt(*xf, in_plane);
        if (out_in_plane_bt) {
            const Vec3 trans_origin(
                ball.pos_bt.x - plane_pos_bt.x,
                ball.pos_bt.y - plane_pos_bt.y,
                ball.pos_bt.z - plane_pos_bt.z);
            *out_in_plane_bt = Vec3(
                (bx.x * vtx_local.x + by.x * vtx_local.y + bz.x * vtx_local.z) + trans_origin.x,
                (bx.y * vtx_local.x + by.y * vtx_local.y + bz.y * vtx_local.z) + trans_origin.y,
                (bx.z * vtx_local.x + by.z * vtx_local.y + bz.z * vtx_local.z) + trans_origin.z);
        }
    } else {
        const Vec3 trans_origin(
            ball.pos_bt.x - plane_pos_bt.x,
            ball.pos_bt.y - plane_pos_bt.y,
            ball.pos_bt.z - plane_pos_bt.z);
        const Vec3 vtx_in_plane(
            (bx.x * vtx_local.x + by.x * vtx_local.y + bz.x * vtx_local.z) + trans_origin.x,
            (bx.y * vtx_local.x + by.y * vtx_local.y + bz.y * vtx_local.z) + trans_origin.y,
            (bx.z * vtx_local.x + by.z * vtx_local.y + bz.z * vtx_local.z) + trans_origin.z);
        distance = Dot(inward_normal, vtx_in_plane);
        world = (vtx_in_plane - inward_normal * distance) + plane_pos_bt;
        if (out_in_plane_bt) {
            *out_in_plane_bt = vtx_in_plane;
        }
    }
    if (out_depth_bt) {
        *out_depth_bt = distance;
    }
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
        const Vec3 world_b = xf != nullptr ? GamePlanePointWorldBt(*xf, world - xf->o) : world;
        *out_distance_bt = Dot(world_a - world_b, inward_normal);
    }
    if (out_local_a_bt) {
        *out_local_a_bt = local_a;
    }
    if (out_point_b_bt) {
        *out_point_b_bt = world;
    }
    return sqrtf(rel.x * rel.x + rel.y * rel.y + rel.z * rel.z);
}

ROCKETGPU_HD ROCKETGPU_NOINLINE bool BallMeshTriangleHit(const MutatorConfig& mutator, const BallState& ball, const Vec3& ball_center_bt,
                                      float radius_bt, float contact_breaking_threshold_bt, float cal_origin_y,
                                      const ArenaMeshTriangle& tri, int32_t global_tri_idx, BallMeshManifoldPoint& new_pt,
                                      Vec3& raw_normal_out) {
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

        raw_normal_out = raw_normal;

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
                contact_normal,
                mutator.native_sphere_rsqrt_table, mutator.native_sphere_rsqrt_shift,
                cal_game_tri);
        }

        new_pt = BallMeshManifoldPoint{};
        new_pt.normal = contact_normal;
        new_pt.pen_uu = pen_uu;
        new_pt.local_point_a_bt = local_point_a_bt;

        new_pt.local_point_b_bt = cal_origin_y != 0.0f
            ? Vec3(contact_on_tri_bt.x, contact_on_tri_bt.y - cal_origin_y, contact_on_tri_bt.z)
            : contact_on_tri_bt;
        new_pt.triangle_index = global_tri_idx;
        new_pt.rel_pos1_length_bt = rel_pos1_length_bt;
        return true;
    }
    return false;
}

ROCKETGPU_HD int16_t BallWorldPlaneStaticId(int slot) {
    constexpr int16_t kIds[kBallWorldPlaneSlots] = {100, 101, 103, 102, 104, 105};
    return kIds[slot];
}

ROCKETGPU_HD int BallWorldPlaneSlotCount(const MutatorConfig& mutator) {
    return mutator.game_mode == static_cast<uint8_t>(Mode::Soccar) ? kBallWorldPlaneSlots : 4;
}

ROCKETGPU_HD bool BallNearGoalFloor(const BallState& ball, int slot) {
    const float dy = ball.pos_bt.y - GoalFloorOriginBt(slot - 4).y;
    return Abs(ball.pos_bt.x) <= kGoalFloorHalfXBt + 0.1f && Abs(dy) <= kGoalFloorHalfYBt + 0.1f;
}

ROCKETGPU_HD void BallWorldPlane(int slot, float arena_margin, Vec3& inward_normal, float& boundary, Vec3& plane_pos_uu) {
    inward_normal = Vec3(0.0f, 0.0f, 1.0f);
    boundary = arena_margin;
    plane_pos_uu = Vec3(0.0f, 0.0f, 0.0f);
    if (slot >= 4) {
        plane_pos_uu = Vec3(0.0f, slot == 4 ? -5584.0f : 5584.0f, 0.0f);
    } else if (slot == 1) {
        inward_normal = Vec3(0.0f, 0.0f, -1.0f);
        boundary = -(ArenaHeight() - arena_margin);
        plane_pos_uu = Vec3(0.0f, 0.0f, ArenaHeight());
    } else if (slot == 2) {
        inward_normal = Vec3(-1.0f, 0.0f, 0.0f);
        boundary = -(ArenaExtentX() - arena_margin);
        plane_pos_uu = Vec3(ArenaExtentX(), 0.0f, 992.0f);
    } else if (slot == 3) {
        inward_normal = Vec3(1.0f, 0.0f, 0.0f);
        boundary = -(ArenaExtentX() - arena_margin);
        plane_pos_uu = Vec3(-ArenaExtentX(), 0.0f, 992.0f);
    }
}

constexpr int kBallWorldStageHits = 12;
constexpr int kBallWorldStageMaxMeshes = 32;
struct BallWorldStageKey {
    uint64_t tick = 0;
    Vec3 pos;
    Vec3 pos_bt;
    Quat rot;
    Vec3 basis_forward;
    Vec3 basis_right;
    Vec3 basis_up;
    uint32_t basis_authoritative = 0;
    float radius = 0.0f;
    int32_t mesh_count = 0;
    uint32_t meshes_lo = 0;
    uint32_t tris_lo = 0;
    uint32_t local_tris_lo = 0;
    uint32_t flags = 0;
    int32_t rsqrt_shift = 0;
    uint32_t pad = 0;
};
static_assert(sizeof(BallWorldStageKey) == 8 + 4 * 28, "BallWorldStageKey has no padding");
struct BallWorldStagePlane {
    Vec3 rel_bt;
    float distance_bt = 0.0f;
    Vec3 local_a_bt;
    Vec3 point_b_bt;
    float rel_len = 0.0f;
    Vec3 in_plane_bt;
    float depth_bt = 0.0f;
};
struct BallWorldStageHit {
    BallMeshManifoldPoint pt;
    int32_t mesh_info_idx = 0;
};
enum BallWorldStageState : uint32_t {
    kBallWorldStageNone = 0,
    kBallWorldStageReady = 1,
    kBallWorldStageOverflow = 2,
    kBallWorldStageEvents = 3,
};
struct BallWorldStage {
    BallWorldStageKey key;
    uint32_t state = kBallWorldStageNone;
    uint32_t plane_mask = 0;
    uint32_t walk_mask = 0;
    int32_t hit_count = 0;
    BallWorldStagePlane planes[kBallWorldPlaneSlots];
    BallWorldStageHit hits[kBallWorldStageHits];
};

ROCKETGPU_HD BallWorldStageKey MakeBallWorldStageKey(const BallState& ball, const MutatorConfig& mutator, uint64_t tick,
                                                     const ArenaMeshTriangle* mesh_triangles,
                                                     const ArenaMeshBvhMeshInfo* mesh_bvh_meshes,
                                                     int32_t mesh_bvh_mesh_count) {
    BallWorldStageKey k;
    k.tick = tick;
    k.pos = ball.pos;
    k.pos_bt = ball.pos_bt;
    k.rot = ball.rot;
    k.basis_forward = ball.basis_forward;
    k.basis_right = ball.basis_right;
    k.basis_up = ball.basis_up;
    k.basis_authoritative = ball.basis_authoritative;
    k.radius = mutator.ball_radius;
    k.mesh_count = mesh_bvh_mesh_count;
    k.meshes_lo = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(mesh_bvh_meshes));
    k.tris_lo = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(mesh_triangles));
    k.local_tris_lo = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(mutator.cal_arena_local_tris));
    k.flags = static_cast<uint32_t>(mutator.cal_ball_world) | static_cast<uint32_t>(mutator.game_mode) << 8 |
              static_cast<uint32_t>(mutator.arena_collision) << 16 |
              (mutator.native_sphere_rsqrt_table != nullptr ? 1u << 24 : 0u) |
              (EnableArenaTriangleInternalEdgeAdjust() ? 1u << 25 : 0u);
    k.rsqrt_shift = mutator.native_sphere_rsqrt_shift;
    return k;
}

ROCKETGPU_HD bool BallWorldStageKeyEqual(const BallWorldStageKey& a, const BallWorldStageKey& b) {
    const unsigned char* wa = reinterpret_cast<const unsigned char*>(&a);
    const unsigned char* wb = reinterpret_cast<const unsigned char*>(&b);
#if defined(__CUDA_ARCH__)
    wa = static_cast<const unsigned char*>(__builtin_assume_aligned(wa, 4));
    wb = static_cast<const unsigned char*>(__builtin_assume_aligned(wb, 4));
#endif
    uint32_t diff = 0u;
#pragma unroll
    for (int i = 0; i < static_cast<int>(sizeof(BallWorldStageKey) / 4); ++i) {
        uint32_t va, vb;
        memcpy(&va, wa + i * 4, sizeof(va));
        memcpy(&vb, wb + i * 4, sizeof(vb));
        diff |= va ^ vb;
    }
    return diff == 0u;
}

ROCKETGPU_HD int BallStageLowBit(uint32_t m) {
#if defined(__CUDA_ARCH__)
    return __ffs(static_cast<int>(m)) - 1;
#elif defined(_MSC_VER)
    unsigned long i;
    _BitScanForward(&i, m);
    return static_cast<int>(i);
#else
    return __builtin_ctz(m);
#endif
}

ROCKETGPU_HD void StageBallWorldContacts(
    const BallState& ball,
    const MutatorConfig& mutator,
    uint64_t tick,
    const ArenaMeshTriangle* mesh_triangles,
    int32_t mesh_triangle_count,
    const ArenaMeshBvhQuantization* mesh_bvh_quantization,
    int32_t mesh_bvh_quantization_count,
    const ArenaMeshBvhMeshInfo* mesh_bvh_meshes,
    int32_t mesh_bvh_mesh_count,
    const ArenaMeshBvhNode* mesh_bvh_nodes,
    int32_t mesh_bvh_node_count,
    BallWorldStage& out,
    const ArenaMeshBvhMeshInfo* mesh_infos = nullptr) {
    if (mesh_infos == nullptr) {
        mesh_infos = mesh_bvh_meshes;
    }
    out.key = MakeBallWorldStageKey(ball, mutator, tick, mesh_triangles, mesh_bvh_meshes, mesh_bvh_mesh_count);
    out.hit_count = 0;
    if (mutator.physics_events.windows != nullptr) {
        out.state = kBallWorldStageEvents;
        return;
    }
    const bool meshes_valid = mesh_triangles != nullptr && mesh_triangle_count > 0 &&
                              mesh_bvh_quantization != nullptr && mesh_bvh_meshes != nullptr &&
                              mesh_bvh_nodes != nullptr && mesh_bvh_mesh_count > 0 && mesh_bvh_node_count > 0;
    if (meshes_valid && mesh_bvh_mesh_count > kBallWorldStageMaxMeshes) {
        out.plane_mask = 0u;
        out.walk_mask = 0u;
        out.state = kBallWorldStageOverflow;
        return;
    }
    const float collision_r = mutator.ball_radius;
    constexpr float kUUToBT_local = 1.0f / 50.0f;
    const float ball_radius_bt = collision_r * kUUToBT_local;
    const float arena_margin = BallArenaMeshCollisionMargin();
    const float plane_detection_r = collision_r * 0.02f + 50.0f * 0.08f * 0.02f;
    constexpr float kUUToBT = 1.0f / 50.0f;
    const Vec3 ball_center_bt = ball.pos_bt;
    const float radius_bt = collision_r * kUUToBT;
    const float contact_breaking_threshold_bt = (radius_bt + 0.08f) * 0.02f;
    const float radiusWithThreshold = radius_bt + contact_breaking_threshold_bt;
    const Vec3 aabb_min_bt = ball_center_bt - Vec3(radiusWithThreshold, radiusWithThreshold, radiusWithThreshold);
    const Vec3 aabb_max_bt = ball_center_bt + Vec3(radiusWithThreshold, radiusWithThreshold, radiusWithThreshold);
    constexpr float kBallMeshWalkSkipMarginBt = 0.01f;

    uint32_t plane_mask = 0u;
    const int plane_slot_count = BallWorldPlaneSlotCount(mutator);
    for (int plane_slot = 0; plane_slot < plane_slot_count; ++plane_slot) {
        Vec3 inward_normal;
        float boundary;
        Vec3 plane_pos_uu;
        BallWorldPlane(plane_slot, arena_margin, inward_normal, boundary, plane_pos_uu);
        const float margin = Dot(ball.pos, inward_normal) - boundary;
        if (margin < collision_r + plane_detection_r && (plane_slot < 4 || BallNearGoalFloor(ball, plane_slot))) {
            plane_mask |= 1u << plane_slot;
        }
    }
    uint32_t walk_mask = 0u;
    if (meshes_valid) {
        const float fx = mesh_infos[0].free_half_x;
        const float fy = mesh_infos[0].free_half_y;
        const bool in_free_interior =
            aabb_max_bt.x + kBallMeshWalkSkipMarginBt < fx &&
            aabb_min_bt.x - kBallMeshWalkSkipMarginBt > -fx &&
            aabb_max_bt.y + kBallMeshWalkSkipMarginBt < fy &&
            aabb_min_bt.y - kBallMeshWalkSkipMarginBt > -fy;
        if (!in_free_interior) {
            for (int mesh_info_idx = 0; mesh_info_idx < mesh_bvh_mesh_count; ++mesh_info_idx) {
                const ArenaMeshBvhMeshInfo& mesh_info = mesh_infos[mesh_info_idx];
                const bool walk_mesh = !(mesh_info.aabb_max.x < aabb_min_bt.x - kBallMeshWalkSkipMarginBt ||
                                         mesh_info.aabb_min.x > aabb_max_bt.x + kBallMeshWalkSkipMarginBt ||
                                         mesh_info.aabb_max.y < aabb_min_bt.y - kBallMeshWalkSkipMarginBt ||
                                         mesh_info.aabb_min.y > aabb_max_bt.y + kBallMeshWalkSkipMarginBt ||
                                         mesh_info.aabb_max.z < aabb_min_bt.z - kBallMeshWalkSkipMarginBt ||
                                         mesh_info.aabb_min.z > aabb_max_bt.z + kBallMeshWalkSkipMarginBt);
                walk_mask |= walk_mesh ? 1u << mesh_info_idx : 0u;
            }
        }
    }
    out.plane_mask = plane_mask;
    out.walk_mask = walk_mask;
    if ((plane_mask | walk_mask) == 0u) {
        out.state = kBallWorldStageReady;
        return;
    }

    for (uint32_t m = plane_mask; m != 0u; m &= m - 1u) {
        const int plane_slot = BallStageLowBit(m);
        Vec3 inward_normal;
        float boundary;
        Vec3 plane_pos_uu;
        BallWorldPlane(plane_slot, arena_margin, inward_normal, boundary, plane_pos_uu);
        const bool soccar = mutator.game_mode == static_cast<uint8_t>(Mode::Soccar);
        const GamePlaneXfBt xf = GameSoccarPlaneXfBt(inward_normal, plane_pos_uu * kUUToBT_local);
        BallWorldStagePlane& sp = out.planes[plane_slot];
        sp.rel_len = BallPlaneRelLen(ball, ball_radius_bt, soccar ? GamePlaneNormalBt(xf) : inward_normal, plane_pos_uu,
                                     &sp.rel_bt, &sp.distance_bt, &sp.local_a_bt, &sp.point_b_bt,
                                     soccar ? &xf : nullptr, &sp.in_plane_bt, &sp.depth_bt);
    }
    uint32_t state = kBallWorldStageReady;
    int hit_count = 0;
    for (uint32_t m = walk_mask; m != 0u; m &= m - 1u) {
        const int mesh_info_idx = BallStageLowBit(m);
        const ArenaMeshBvhMeshInfo& mesh_info = mesh_infos[mesh_info_idx];
        if (mesh_info.mesh_index >= mesh_bvh_quantization_count ||
            mesh_info.node_offset < 0 ||
            mesh_info.node_offset + mesh_info.node_count > mesh_bvh_node_count) {
            continue;
        }
        uint16_t quantized_query_aabb_min[3] = {0, 0, 0};
        uint16_t quantized_query_aabb_max[3] = {0, 0, 0};
        const ArenaMeshBvhQuantization& quantization = mesh_bvh_quantization[mesh_info.mesh_index];
        QuantizeArenaMeshBvhQueryBox(quantization, ball_center_bt, Vec3(radiusWithThreshold, radiusWithThreshold, radiusWithThreshold),
                                     mesh_info.cal_origin_y, quantized_query_aabb_min, quantized_query_aabb_max);
        const float cal_origin_y = mesh_info.cal_origin_y;
        const int32_t triangle_count = mesh_info.triangle_count;
        const int32_t triangle_start = mesh_info.triangle_start;
        WalkArenaMeshBvhStacklessQuantizedTreeCacheFriendly(
            mesh_bvh_nodes,
            nullptr,
            0,
            quantized_query_aabb_min,
            quantized_query_aabb_max,
            mesh_info.node_offset,
            mesh_info.node_count,
            [&](int32_t local_tri_idx) {
                if (local_tri_idx < 0 || local_tri_idx >= triangle_count) {
                    return;
                }
                const int32_t global_tri_idx = triangle_start + local_tri_idx;
                if (global_tri_idx >= mesh_triangle_count) {
                    return;
                }
                BallMeshManifoldPoint new_pt;
                Vec3 raw_normal;
                if (BallMeshTriangleHit(mutator, ball, ball_center_bt, radius_bt, contact_breaking_threshold_bt,
                                        cal_origin_y, mesh_triangles[global_tri_idx], global_tri_idx, new_pt,
                                        raw_normal)) {
                    if (hit_count < kBallWorldStageHits) {
                        out.hits[hit_count].pt = new_pt;
                        out.hits[hit_count].mesh_info_idx = mesh_info_idx;
                    } else {
                        state = kBallWorldStageOverflow;
                    }
                    ++hit_count;
                }
            });
    }
    out.hit_count = hit_count < kBallWorldStageHits ? hit_count : kBallWorldStageHits;
    out.state = state;
}

#if defined(__CUDACC__)
extern __device__ BallWorldStage* g_rocketgpu_ball_world_stage;
#endif

ROCKETGPU_HD const BallWorldStage* ConsumableBallWorldStage(
    int env_idx, const BallState& ball, const MutatorConfig& mutator, uint64_t tick,
    const ArenaMeshTriangle* mesh_triangles, const ArenaMeshBvhMeshInfo* mesh_bvh_meshes, int32_t mesh_bvh_mesh_count) {
#if defined(__CUDA_ARCH__)
    const BallWorldStage* base = g_rocketgpu_ball_world_stage;
    if (base == nullptr || env_idx < 0) {
        return nullptr;
    }
    const BallWorldStage* stage = base + env_idx;
    if (stage->state != kBallWorldStageReady ||
        !BallWorldStageKeyEqual(stage->key, MakeBallWorldStageKey(ball, mutator, tick, mesh_triangles, mesh_bvh_meshes,
                                                                  mesh_bvh_mesh_count))) {
        return nullptr;
    }
    return stage;
#else
    (void)env_idx; (void)ball; (void)mutator; (void)tick; (void)mesh_triangles; (void)mesh_bvh_meshes;
    (void)mesh_bvh_mesh_count;
    return nullptr;
#endif
}

// A ball set from public state has no retained plane points, while the game's ball holds the points its previous
// tick's detection made (a resting, sleeping ball keeps its floor point through the tick a car wakes it). Prime each
// empty plane slot with the point the step's detection makes at the pose one tick back.
ROCKETGPU_HD void PrimeImportedBallWorldPlanes(BallState ball, const MutatorConfig& mutator, float dt,
                                               WorldSolverScratch& scratch) {
    if (!mutator.arena_collision) {
        return;
    }
    ball.pos_bt = ball.pos_bt - ball.vel_bt * dt;
    ball.pos = ball.pos_bt * 50.0f;
    const float collision_r = mutator.ball_radius;
    const float ball_radius_bt = collision_r * (1.0f / 50.0f);
    const float arena_margin = BallArenaMeshCollisionMargin();
    const float plane_detection_r = collision_r * 0.02f + 50.0f * 0.08f * 0.02f;
    const float plane_breaking_threshold_bt = (ball_radius_bt + 0.08f) * 0.02f;
    const bool soccar = mutator.game_mode == static_cast<uint8_t>(Mode::Soccar);
    const int plane_slot_count = BallWorldPlaneSlotCount(mutator);
    for (int plane_slot = 0; plane_slot < plane_slot_count; ++plane_slot) {
        BallMeshPersistentManifold& m = scratch.ball_world_plane_manifolds[plane_slot];
        if (m.count != 0) {
            continue;
        }
        Vec3 axis_normal;
        float boundary;
        Vec3 plane_pos_uu;
        BallWorldPlane(plane_slot, arena_margin, axis_normal, boundary, plane_pos_uu);
        const float margin = Dot(ball.pos, axis_normal) - boundary;
        if (!(margin < collision_r + plane_detection_r && (plane_slot < 4 || BallNearGoalFloor(ball, plane_slot)))) {
            continue;
        }
        const GamePlaneXfBt plane_xf = GameSoccarPlaneXfBt(axis_normal, plane_pos_uu * (1.0f / 50.0f));
        const GamePlaneXfBt* const xf = soccar ? &plane_xf : nullptr;
        const Vec3 inward_normal = soccar ? GamePlaneNormalBt(plane_xf) : axis_normal;
        Vec3 rel_bt(0.0f, 0.0f, 0.0f);
        float distance_bt = 0.0f;
        Vec3 local_a_bt(0.0f, 0.0f, 0.0f);
        Vec3 point_b_bt(0.0f, 0.0f, 0.0f);
        Vec3 in_plane_bt(0.0f, 0.0f, 0.0f);
        float depth_bt = 0.0f;
        const float rel_len = BallPlaneRelLen(ball, ball_radius_bt, inward_normal, plane_pos_uu, &rel_bt, &distance_bt,
                                              &local_a_bt, &point_b_bt, xf, &in_plane_bt, &depth_bt);
        const bool within_extent =
            !soccar || (plane_slot >= 4 ? GoalFloorExtentContainsBt(in_plane_bt, plane_breaking_threshold_bt)
                                        : SoccarPlaneExtentContainsInPlaneBt(inward_normal, in_plane_bt,
                                                                             plane_breaking_threshold_bt));
        if (!(depth_bt < plane_breaking_threshold_bt && within_extent)) {
            continue;
        }
        BallMeshManifoldPoint& p = m.points[0];
        p = BallMeshManifoldPoint{};
        p.normal = inward_normal;
        p.pen_uu = -depth_bt * 50.0f;
        p.local_point_a_bt = local_a_bt;
        p.local_point_b_bt = point_b_bt;
        p.triangle_index = -1;
        p.rel_pos1_length_bt = rel_len;
        p.rel_pos1_bt = rel_bt;
        p.distance_bt = distance_bt;
        m.count = 1;
    }
}

ROCKETGPU_HD ROCKETGPU_NOINLINE void AppendBallWorldSpecialContactRows(
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
    int event_env_idx = -1,
    const BallWorldStage* stage = nullptr) {
    int hs_world_normal_count = 0;

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
    UninitArray<BallWorldSpecialCp, kMaxBallWorldSpecialCps> special_cp_slots;
    BallWorldSpecialCp* const special_cps = special_cp_slots;
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
        if (solver_scratch && hs_world_normal_count < WorldSolverScratch::kMaxHeatseekerWorldNormals) {
            solver_scratch->hs_world_normals[hs_world_normal_count++] = normal;
        }
        if (special_cp_count < kMaxBallWorldSpecialCps) {
            special_cps[special_cp_count].normal = normal;
            special_cps[special_cp_count].penetration_uu = penetration_uu;
            special_cps[special_cp_count].rel_pos1_length_bt = rel_pos1_length_bt;
            special_cps[special_cp_count].static_id = static_id;
            special_cps[special_cp_count].has_bt = rel_pos1_bt && penetration_bt ? 1 : 0;
            if (rel_pos1_bt && penetration_bt) {
                special_cps[special_cp_count].rel_pos1_bt = *rel_pos1_bt;
                special_cps[special_cp_count].penetration_bt = *penetration_bt;
            }
            ++special_cp_count;
        } else {
            AccumulateBallWorldContact(aggregate, normal, penetration_uu, rel_pos1_length_bt);
        }
    };

    const auto official_plane_rel_len = [&](const Vec3& inward_normal, const Vec3& plane_pos_uu, Vec3* out_rel_bt = nullptr, float* out_distance_bt = nullptr, Vec3* out_local_a_bt = nullptr, Vec3* out_point_b_bt = nullptr, const GamePlaneXfBt* xf = nullptr, Vec3* out_in_plane_bt = nullptr, float* out_depth_bt = nullptr) -> float {
        return BallPlaneRelLen(ball, ball_radius_bt, inward_normal, plane_pos_uu, out_rel_bt, out_distance_bt,
                               out_local_a_bt, out_point_b_bt, xf, out_in_plane_bt, out_depth_bt);
    };
    const float plane_breaking_threshold_bt = (ball_radius_bt + 0.08f) * 0.02f;
    const bool retain_plane_points = solver_scratch != nullptr;
    const int plane_slot_count = BallWorldPlaneSlotCount(mutator);
    uint64_t plane_manifold_counts = 0;
    if (retain_plane_points) {
#pragma unroll
        for (int k = 0; k < kBallWorldPlaneSlots; ++k) {
            plane_manifold_counts |= static_cast<uint64_t>(solver_scratch->ball_world_plane_manifolds[k].count) << (8 * k);
        }
    }
    const auto append_plane_cp = [&](const Vec3& axis_normal, float boundary, const Vec3& plane_pos_uu, int plane_slot) {
        const float margin = Dot(ball.pos, axis_normal) - boundary;
        const bool in_reach =
            margin < collision_r + plane_detection_r && (plane_slot < 4 || BallNearGoalFloor(ball, plane_slot));
        int count = retain_plane_points ? static_cast<int>((plane_manifold_counts >> (8 * plane_slot)) & 0xFFu) : 0;
        if (!in_reach && count == 0) {
            return;
        }
        const bool soccar = mutator.game_mode == static_cast<uint8_t>(Mode::Soccar);
        const GamePlaneXfBt plane_xf = GameSoccarPlaneXfBt(axis_normal, plane_pos_uu * (1.0f / 50.0f));
        const GamePlaneXfBt* const xf = soccar ? &plane_xf : nullptr;
        const Vec3 inward_normal = soccar ? GamePlaneNormalBt(plane_xf) : axis_normal;
        if (retain_plane_points) {
            BallMeshPersistentManifold& m = solver_scratch->ball_world_plane_manifolds[plane_slot];
            if (in_reach) {
                Vec3 rel_bt(0.0f, 0.0f, 0.0f);
                float distance_bt = 0.0f;
                Vec3 local_a_bt(0.0f, 0.0f, 0.0f);
                Vec3 point_b_bt(0.0f, 0.0f, 0.0f);
                Vec3 in_plane_bt(0.0f, 0.0f, 0.0f);
                float rel_len;
                float fresh_depth_bt = 0.0f;
                if (stage != nullptr && ((stage->plane_mask >> plane_slot) & 1u) != 0u) {
                    const BallWorldStagePlane& sp = stage->planes[plane_slot];
                    rel_bt = sp.rel_bt;
                    distance_bt = sp.distance_bt;
                    local_a_bt = sp.local_a_bt;
                    point_b_bt = sp.point_b_bt;
                    in_plane_bt = sp.in_plane_bt;
                    fresh_depth_bt = sp.depth_bt;
                    rel_len = sp.rel_len;
                } else {
                    rel_len = official_plane_rel_len(inward_normal, plane_pos_uu, &rel_bt, &distance_bt, &local_a_bt,
                                                     &point_b_bt, xf, &in_plane_bt, &fresh_depth_bt);
                }
                const bool within_extent =
                    !soccar || (plane_slot >= 4 ? GoalFloorExtentContainsBt(in_plane_bt, plane_breaking_threshold_bt)
                                                : SoccarPlaneExtentContainsInPlaneBt(inward_normal, in_plane_bt,
                                                                                     plane_breaking_threshold_bt));
                if (fresh_depth_bt < plane_breaking_threshold_bt && within_extent) {
                    BallMeshManifoldPoint new_pt;
                    new_pt.normal = inward_normal;
                    new_pt.pen_uu = -fresh_depth_bt * 50.0f;
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
            if (count > 0) {
                RefreshBallMeshPersistentManifold(m.points, count, ball.pos_bt, ball, plane_breaking_threshold_bt, 0.0f, true,
                                                  xf);
            }
            m.count = static_cast<uint8_t>(count);
            for (int i = 0; i < count; ++i) {
                const BallMeshManifoldPoint& q = m.points[i];
                const float pen_bt = -q.distance_bt;
                append_special_cp_with_len(
                    inward_normal,
                    q.pen_uu,
                    q.rel_pos1_length_bt,
                    BallWorldPlaneStaticId(plane_slot),
                    &q.rel_pos1_bt,
                    &pen_bt);
            }
            return;
        }
        if (margin >= collision_r + plane_detection_r || (plane_slot >= 4 && !BallNearGoalFloor(ball, plane_slot))) {
            return;
        }
        Vec3 plane_rel_bt(0.0f, 0.0f, 0.0f);
        float plane_distance_bt = 0.0f;
        const float plane_rel_len =
            official_plane_rel_len(inward_normal, plane_pos_uu, &plane_rel_bt, &plane_distance_bt, nullptr, nullptr, xf);
        const float plane_pen_bt = -plane_distance_bt;
        append_special_cp_with_len(
            inward_normal,
            collision_r - margin,
            plane_rel_len,
            BallWorldPlaneStaticId(plane_slot),
            &plane_rel_bt,
            &plane_pen_bt);
    };

#pragma unroll 1
    for (int plane_slot = 0; plane_slot < plane_slot_count; ++plane_slot) {
        Vec3 inward_normal;
        float boundary;
        Vec3 plane_pos_uu;
        BallWorldPlane(plane_slot, arena_margin, inward_normal, boundary, plane_pos_uu);
        append_plane_cp(inward_normal, boundary, plane_pos_uu, plane_slot);
    }

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

        constexpr float kBallMeshWalkSkipMarginBt = 0.01f;
        const auto mesh_walk_needed = [&](const ArenaMeshBvhMeshInfo& info) {
            return !(info.aabb_max.x < aabb_min_bt.x - kBallMeshWalkSkipMarginBt ||
                     info.aabb_min.x > aabb_max_bt.x + kBallMeshWalkSkipMarginBt ||
                     info.aabb_max.y < aabb_min_bt.y - kBallMeshWalkSkipMarginBt ||
                     info.aabb_min.y > aabb_max_bt.y + kBallMeshWalkSkipMarginBt ||
                     info.aabb_max.z < aabb_min_bt.z - kBallMeshWalkSkipMarginBt ||
                     info.aabb_min.z > aabb_max_bt.z + kBallMeshWalkSkipMarginBt);
        };
        const float fx = mesh_bvh_meshes[0].free_half_x;
        const float fy = mesh_bvh_meshes[0].free_half_y;
        const bool ball_in_free_interior =
            stage != nullptr
                ? stage->walk_mask == 0u
                : (aabb_max_bt.x + kBallMeshWalkSkipMarginBt < fx &&
                   aabb_min_bt.x - kBallMeshWalkSkipMarginBt > -fx &&
                   aabb_max_bt.y + kBallMeshWalkSkipMarginBt < fy &&
                   aabb_min_bt.y - kBallMeshWalkSkipMarginBt > -fy);
        uint32_t idle_meshes = 0u;
        uint64_t retained_counts = 0u;
        const bool track_retained = mutator.cal_ball_world != 0 && solver_scratch != nullptr;
        const uint32_t maybe_retained = track_retained ? solver_scratch->ball_mesh_maybe_retained : 0u;
        uint32_t next_maybe_retained = maybe_retained;
        if (ball_in_free_interior && track_retained && maybe_retained == 0u &&
            mesh_bvh_mesh_count <= kMaxBallWorldMeshPersistentManifolds) {
            idle_meshes = ~0u;
        } else if (track_retained) {
#pragma unroll
            for (int mesh_info_idx = 0; mesh_info_idx < kMaxBallWorldMeshPersistentManifolds; ++mesh_info_idx) {
                const uint64_t retained = ((maybe_retained >> mesh_info_idx) & 1u) != 0u
                    ? solver_scratch->ball_world_mesh_manifolds[mesh_info_idx].count
                    : 0u;
                retained_counts |= retained << (3 * mesh_info_idx);
                if (mesh_info_idx < mesh_bvh_mesh_count && retained == 0 &&
                    (ball_in_free_interior ||
                     !(stage != nullptr ? ((stage->walk_mask >> mesh_info_idx) & 1u) != 0u
                                        : mesh_walk_needed(mesh_bvh_meshes[mesh_info_idx])))) {
                    idle_meshes |= 1u << mesh_info_idx;
                }
            }
        }

        int stage_hit = 0;
        const int active_mesh_count =
            (idle_meshes == ~0u && mesh_bvh_mesh_count <= kMaxBallWorldMeshPersistentManifolds) ? 0 : mesh_bvh_mesh_count;
        for (int mesh_info_idx = 0; mesh_info_idx < active_mesh_count; ++mesh_info_idx) {
            if (mesh_info_idx < kMaxBallWorldMeshPersistentManifolds && ((idle_meshes >> mesh_info_idx) & 1u) != 0u) {
                next_maybe_retained &= ~(1u << mesh_info_idx);
                continue;
            }
            const ArenaMeshBvhMeshInfo& mesh_info = mesh_bvh_meshes[mesh_info_idx];
            if (mesh_info.mesh_index >= mesh_bvh_quantization_count ||
                mesh_info.node_offset < 0 ||
                mesh_info.node_offset + mesh_info.node_count > mesh_bvh_node_count) {
                continue;
            }

            UninitArray<BallMeshPersistentManifold, 1> transient_storage;
            BallMeshPersistentManifold* const transient_manifold = transient_storage.data();
            BallMeshPersistentManifold* persistent_manifold = transient_manifold;
            if (solver_scratch != nullptr &&
                mesh_info_idx >= 0 &&
                mesh_info_idx < kMaxBallWorldMeshPersistentManifolds) {
                persistent_manifold = &solver_scratch->ball_world_mesh_manifolds[mesh_info_idx];
            }

            const bool retain_ball_mesh_points =
                mutator.cal_ball_world != 0 &&
                persistent_manifold != transient_manifold;
            if (!retain_ball_mesh_points) {
                ResetBallMeshPersistentManifold(*persistent_manifold);
            }
            int persistent_count =
                retain_ball_mesh_points ? static_cast<int>((retained_counts >> (3 * mesh_info_idx)) & 7u) : 0;

            const bool walk_mesh = stage == nullptr && mesh_walk_needed(mesh_info);
            if (walk_mesh) {
                const ArenaMeshBvhQuantization& quantization = mesh_bvh_quantization[mesh_info.mesh_index];
                QuantizeArenaMeshBvhQueryBox(quantization, ball_center_bt,
                                             Vec3(radiusWithThreshold, radiusWithThreshold, radiusWithThreshold),
                                             mesh_info.cal_origin_y, quantized_query_aabb_min, quantized_query_aabb_max);
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
            if (walk_mesh) {
                auto walk_ball_mesh_bvh = [&]() ROCKETGPU_NOINLINE {
                    WalkArenaMeshBvhStacklessQuantizedTreeCacheFriendly(
                        mesh_bvh_nodes,
                        mesh_subtrees,
                        true ? 0 : mesh_subtree_count,
                        quantized_query_aabb_min,
                        quantized_query_aabb_max,
                        mesh_info.node_offset,
                        mesh_info.node_count,
                        [&](int32_t local_tri_idx) {
                            if (local_tri_idx < 0 || local_tri_idx >= mesh_info.triangle_count) {
                                return;
                            }
                            const int32_t global_tri_idx = mesh_info.triangle_start + local_tri_idx;
                            if (global_tri_idx >= mesh_triangle_count) {
                                return;
                            }
                            const ArenaMeshTriangle& tri = mesh_triangles[global_tri_idx];

                            BallMeshManifoldPoint new_pt;
                            Vec3 raw_normal;
                            if (BallMeshTriangleHit(mutator, ball, ball_center_bt, radius_bt, contact_breaking_threshold_bt,
                                                    cal_origin_y, tri, global_tri_idx, new_pt, raw_normal)) {
                                emit_world_event(raw_normal);
                                InsertBallMeshManifoldPoint(
                                    persistent_manifold->points,
                                    persistent_count,
                                    new_pt,
                                    mutator.cal_ball_world != 0 ? contact_breaking_threshold_bt : -1.0f);
                            }
                        });
                };
                walk_ball_mesh_bvh();
            }
            if (stage != nullptr) {
                while (stage_hit < stage->hit_count && stage->hits[stage_hit].mesh_info_idx < mesh_info_idx) {
                    ++stage_hit;
                }
                while (stage_hit < stage->hit_count && stage->hits[stage_hit].mesh_info_idx == mesh_info_idx) {
                    InsertBallMeshManifoldPoint(
                        persistent_manifold->points,
                        persistent_count,
                        stage->hits[stage_hit].pt,
                        mutator.cal_ball_world != 0 ? contact_breaking_threshold_bt : -1.0f);
                    ++stage_hit;
                }
            }

            RefreshBallMeshPersistentManifold(
                persistent_manifold->points,
                persistent_count,
                ball_center_bt,
                ball,
                contact_breaking_threshold_bt,
                cal_origin_y,
                retain_ball_mesh_points);
            persistent_manifold->count = static_cast<uint8_t>(persistent_count);
            if (track_retained && mesh_info_idx < kMaxBallWorldMeshPersistentManifolds) {
                next_maybe_retained = persistent_count > 0 ? (next_maybe_retained | (1u << mesh_info_idx))
                                                           : (next_maybe_retained & ~(1u << mesh_info_idx));
            }

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
        if (track_retained) {
            solver_scratch->ball_mesh_maybe_retained = static_cast<uint16_t>(next_maybe_retained);
        }
    }

    {
        uint32_t consumed_mask = 0u;
        if (solver_scratch != nullptr && solver_scratch->bullet_manifold_count > 0) {
            for (int m = 0; m < solver_scratch->bullet_manifold_count; ++m) {
                const BulletManifoldEntry& entry = bullet_manifolds[m];
                if (entry.kind != 0) {
                    continue;
                }
                for (int i = 0; i < special_cp_count; ++i) {
                    if (((consumed_mask >> i) & 1u) == 0u && special_cps[i].static_id == entry.static_id) {
                        consumed_mask |= 1u << i;
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
            if (((consumed_mask >> i) & 1u) == 0u) {
                AccumulateBallWorldContact(
                    aggregate,
                    special_cps[i].normal,
                    special_cps[i].penetration_uu,
                    special_cps[i].rel_pos1_length_bt);
            }
        }
    }
    if (solver_scratch) {
        solver_scratch->hs_world_normal_count = hs_world_normal_count;
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
    float synthetic_distance_bt = aggregate.rel_pos1_length_sum_bt / static_cast<float>(aggregate.count);
    float synthetic_friction = mutator.ball_friction;
    float synthetic_restitution = ball.constrained_2d ? 0.0f : mutator.ball_restitution;
    if (mutator.cal_ball_world) {
        normal = aggregate.normal_sum;
        synthetic_distance_bt = aggregate.rel_pos1_length_sum_bt;
        if (aggregate.count > 1) {
            float friction_sum = synthetic_friction;
            float restitution_sum = synthetic_restitution;
            for (int i = 1; i < aggregate.count; ++i) {
                friction_sum += synthetic_friction;
                restitution_sum += synthetic_restitution;
            }
            synthetic_friction = inv_count * friction_sum;
            synthetic_restitution = inv_count * restitution_sum;
            const float l2 = Dot(normal, normal);
            normal = l2 >= SIMD_EPSILON * SIMD_EPSILON ? normal * (1.0f / SqrtScalar(l2)) : Vec3(1.0f, 0.0f, 0.0f);
            synthetic_distance_bt = inv_count * synthetic_distance_bt;
        }
    }
    constexpr float kBTToUU = 50.0f;
    const Vec3 rel_ball_contact_bt = normal * (-synthetic_distance_bt);
    const Vec3 rel_ball_contact = rel_ball_contact_bt * kBTToUU;

    {
    }

    const Vec3 tangent_basis_rel_vel(0.0f, 0.0f, 0.0f);
    const Vec3 friction_base_rel_vel(0.0f, 0.0f, 0.0f);

    const Vec3 restitution_ball_linear = ball.vel;
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
        synthetic_restitution,
        synthetic_friction,
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
    car.restored_inv_inertia_valid = 0;
    SetCarLinearVelocityBt(car, Vec3(0.0f, 0.0f, 0.0f));
    car.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
    car.world_contact = {};
    car.flip_rel_torque = Vec3(0.0f, 0.0f, 0.0f);
    car.boost = spawn_boost;
    car.jump_time = 0.0f;
    car.flip_time = 0.0f;
    car.internal_jump_time = 0.0f;
    car.internal_flip_time = 0.0f;
    car.flip_timer_live = 1;
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
    car.air_state = 0;
    car.dodge_timeout = -1.0f;
    car.dodge_elapsed = 0.0f;
    car.dodge_dir_x = 0.0f;
    car.dodge_dir_y = 0.0f;

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
    car.boost_pad_touching = 0;
    car.driving_disabled = 0;
    car.respawn_slot_override_plus1 = 0;
    car.pad_frame_pos_valid = 0;
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

    if (team == Team::Orange) {
        x = -x;
        y = -y;
        yaw += Pi();
    }
    SetCarOriginUu(car, Vec3(x, y, z));
    car.rot = FromYaw(WrapAngle(yaw));
    SeedCarBasisFromSpawnRotator(car, SpawnYawRotatorUnits(spawn.yaw), team == Team::Orange);
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

    SetCarOriginUu(car, Vec3(x, y, z));
    car.rot = FromYaw(WrapAngle(yaw));
    SeedCarBasisFromSpawnRotator(car, SpawnYawRotatorUnits(spawn.yaw), team == Team::Orange);
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

ROCKETGPU_HD int PlaceNthTeamCarAtKickoff(
    int base,
    int count,
    Team team,
    int nth,
    Team* teams,
    CarState* cars,
    const SpawnPos& spawn,
    const MutatorConfig& mutator,
    const EnvState& env_state,
    const CarConfig* car_configs) {
    int seen = 0;
    for (int k = 0; k < count; ++k) {
        const int i = CarVisitSlot(env_state, k, count);
        const int idx = base + i;
        if (teams[idx] != team) {
            continue;
        }
        if (seen == nth) {
            const float z = car_configs != nullptr ? car_configs[idx].spawn_z : CarSpawnRestZ();
            PlaceCarAtKickoffSpawn(cars[idx], team, spawn, z, mutator, mutator.boost_spawn_amount);
            return idx;
        }
        ++seen;
    }
    return -1;
}

// A word loop: a plain struct assignment of these large states inlines into hundreds of registers on the GPU.
template <typename T>
ROCKETGPU_HD void CopyStateWords(T& dst, const T& src) {
    static_assert(sizeof(T) % 4 == 0, "word copy");
    uint32_t* d = reinterpret_cast<uint32_t*>(&dst);
    const uint32_t* s = reinterpret_cast<const uint32_t*>(&src);
#pragma unroll 1
    for (size_t i = 0; i < sizeof(T) / 4; ++i) d[i] = s[i];
}

// mutator.kickoff_settled: car idx as the game's countdown leaves it at this spawn (SimConfig.kickoff_settled_rows).
ROCKETGPU_HD void SettleKickoffCar(const SimConfig& cfg, int idx, int spawn, CarState* cars) {
    if (cfg.kickoff_settled_rows == nullptr || spawn >= cfg.kickoff_settled_spawns || cfg.kickoff_settled_entry[idx] < 0) {
        return;
    }
    const KickoffSettledRow& row = cfg.kickoff_settled_rows[static_cast<size_t>(cfg.kickoff_settled_entry[idx]) * cfg.kickoff_settled_spawns + spawn];
    // The car's own history (bumps, touches, contact and flip stamps) stays as the reset left it.
    CarState& car = cars[idx];
    const auto bump_event_tick = car.bump_event_tick;
    const auto bump_event_other_id = car.bump_event_other_id;
    const auto bump_event_is_demo = car.bump_event_is_demo;
    const auto bump_event_was_bumper = car.bump_event_was_bumper;
    const auto previous_ball_contact_start = car.previous_ball_contact_start;
    const auto last_ball_contact_start = car.last_ball_contact_start;
    const auto last_world_surface_contact = car.last_world_surface_contact;
    const auto first_world_after_previous_ball = car.first_world_after_previous_ball;
    const auto first_world_after_last_ball = car.first_world_after_last_ball;
    const auto last_flip_or_jump_start = car.last_flip_or_jump_start;
    const auto last_directional_flip_start = car.last_directional_flip_start;
    const auto last_ball_started_with_world = car.last_ball_started_with_world;
    const auto flip_or_jump_last_tick = car.flip_or_jump_last_tick;
    const auto latest_touch_tick = car.latest_touch_tick;
    const auto latest_touch_location = car.latest_touch_location;
    const auto latest_touch_normal = car.latest_touch_normal;
    CopyStateWords(car, row.car);
    car.bump_event_tick = bump_event_tick;
    car.bump_event_other_id = bump_event_other_id;
    car.bump_event_is_demo = bump_event_is_demo;
    car.bump_event_was_bumper = bump_event_was_bumper;
    car.previous_ball_contact_start = previous_ball_contact_start;
    car.last_ball_contact_start = last_ball_contact_start;
    car.last_world_surface_contact = last_world_surface_contact;
    car.first_world_after_previous_ball = first_world_after_previous_ball;
    car.first_world_after_last_ball = first_world_after_last_ball;
    car.last_flip_or_jump_start = last_flip_or_jump_start;
    car.last_directional_flip_start = last_directional_flip_start;
    car.last_ball_started_with_world = last_ball_started_with_world;
    car.flip_or_jump_last_tick = flip_or_jump_last_tick;
    car.latest_touch_tick = latest_touch_tick;
    car.latest_touch_location = latest_touch_location;
    car.latest_touch_normal = latest_touch_normal;
    CopyStateWords(cfg.kickoff_settled_wheels[idx], row.wheels);
}

// After a kickoff placement without the countdown: every car and the ball as the countdown leaves them. A team's
// nth car (visit order, as PlaceNthTeamCarAtKickoff counts) took spawn kickoff_order[n] (n < 5), else respawn n.
// order_bits: kickoff_order, 3 bits per slot.
ROCKETGPU_HD ROCKETGPU_NOINLINE void SettleKickoffEnv(
    const SimConfig& cfg, int env_idx, int base, int count, const Team* teams, CarState* cars, BallState* balls,
    const EnvState& env_state, uint32_t order_bits) {
    int seen[2] = {0, 0};
    for (int k = 0; k < count; ++k) {
        const int idx = base + CarVisitSlot(env_state, k, count);
        const int t = teams[idx] == Team::Blue ? 0 : 1;
        const int nth = seen[t]++;
        SettleKickoffCar(cfg, idx, nth < 5 ? static_cast<int>((order_bits >> (3 * nth)) & 7u) : nth, cars);
    }
    if (cfg.mutator.game_mode != static_cast<uint8_t>(Mode::Heatseeker) && cfg.kickoff_settled_ball != nullptr) {
        CopyStateWords(balls[env_idx], *cfg.kickoff_settled_ball);
    }
}

ROCKETGPU_HD void PendKickoffSpawn(CarState& car, int spot, float tick_dt) {
    car.is_demoed = 1;
    car.demo_respawn_timer = 1.5f * tick_dt;
    car.respawn_slot_override_plus1 = static_cast<uint8_t>(kKickoffSpawnOverridePlus1 + spot);
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
    ball.constrained_2d = 0;
}

ROCKETGPU_HD void ResetBall(BallState& ball, uint8_t game_mode, bool game_spawn = false) {
    ResetBall(ball);
    if (game_mode == static_cast<uint8_t>(Mode::Soccar) && game_spawn) {
        SetBallOriginBt(ball, Vec3(0.0f, 0.0f, 2.0097436904907227f));
    } else if (game_mode == static_cast<uint8_t>(Mode::Soccar)) {
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
        pads[base + p].pickup_frac_q = 32768;
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
    const int* kickoff_order_override = nullptr,
    bool keep_boost_pads = false,
    const CarConfig* car_configs = nullptr,
    bool game_spawn = false) {
    constexpr float kCarSpawnExtraOffsetY = 250.0f;

    EnvState saved = env_states[env_idx];
    env_states[env_idx] = EnvState{};

    env_states[env_idx].car_visit_order_mode = saved.car_visit_order_mode;
    env_states[env_idx].car_visit_order_packed = saved.car_visit_order_packed;
    if (saved.car_visit_order_mode == 3) {
        env_states[env_idx].car_visit_order_packed = 0x76543210u;
        env_states[env_idx].car_visit_present_mask = 0xFFu;
    }
    env_states[env_idx].frame_ticks = saved.frame_ticks;
    env_states[env_idx].frame_frac_prev = saved.frame_frac_prev;
    if (preserve_score) {
        env_states[env_idx].blue_score = saved.blue_score;
        env_states[env_idx].orange_score = saved.orange_score;
        env_states[env_idx].tick = saved.tick;
        env_states[env_idx].last_scored_team = saved.last_scored_team;
    }

    ResetBall(balls[env_idx], cfg.mutator.game_mode, game_spawn);
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
    if (!keep_boost_pads) {
        ResetBoostPads(cfg, pads, env_idx);
    }

    const int base = env_idx * cfg.max_cars_per_env;
    const int count = env_car_count[env_idx];
    int blue_count = 0;
    int orange_count = 0;
    for (int i = 0; i < count; ++i) {
        const Team team = teams[base + CarVisitSlot(env_states[env_idx], i, count)];
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
    } else if (kickoff_seed_mode == 3) {   // slot order
    } else if (respawn_rng != nullptr) {
        const uint32_t engine_seed = kickoff_seed_mode == 1
            ? static_cast<uint32_t>(seed)
            : MixSeed(seed, env_idx, 0x6b1d4f2a);
        if (kickoff_seed_mode == 2 || static_cast<int32_t>(engine_seed) == -1) {
            ShuffleLikePlatformStd(kickoff_order, respawn_rng[env_idx]);
        } else {
            TempRespawnRngState temp_engine;
            SeedTempRespawnRng(temp_engine, respawn_rng[env_idx].kind, engine_seed);
            ShuffleLikePlatformStd(kickoff_order, temp_engine);
        }
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

        for (int t = 0; t < 2; ++t) {
            const Team team = t == 0 ? Team::Blue : Team::Orange;
            if (slot >= (t == 0 ? blue_count : orange_count)) continue;
            const int idx = PlaceNthTeamCarAtKickoff(base, count, team, slot, teams, cars, spawn, cfg.mutator,
                                                     env_states[env_idx], car_configs);
            if (game_spawn && idx >= 0 && slot < 5) PendKickoffSpawn(cars[idx], kickoff_order[slot], cfg.tick_dt);
        }
    }
    if (!game_spawn && cfg.kickoff_settled_rows != nullptr) {
        uint32_t order_bits = 0;
        for (int i = 0; i < 5; ++i) order_bits |= static_cast<uint32_t>(kickoff_order[i]) << (3 * i);
        SettleKickoffEnv(cfg, env_idx, base, count, teams, cars, balls, env_states[env_idx], order_bits);
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

ROCKETGPU_HD void RespawnCar(CarState& car, Team team, int slot, const MutatorConfig& mutator, float spawn_z) {
    const SpawnPos spawn = SoccarRespawnSpawn(slot % 4);
    PlaceCarAtRespawnSpawn(car, team, spawn, spawn_z, mutator, mutator.boost_spawn_amount);
    car.vel_impulse_cache = Vec3(0.0f, 0.0f, 0.0f);
}

ROCKETGPU_HD void KickoffSpawnCar(CarState& car, Team team, int spot, const MutatorConfig& mutator, float spawn_z) {
    const SpawnPos spawn = mutator.game_mode == static_cast<uint8_t>(Mode::Heatseeker) ? HeatseekerKickoffSpawn(spot)
                                                                                         : SoccarKickoffSpawn(spot);
    const uint16_t driving_disabled = car.driving_disabled;
    PlaceCarAtKickoffSpawn(car, team, spawn, spawn_z, mutator, mutator.boost_spawn_amount);
    car.driving_disabled = driving_disabled;
    car.vel_impulse_cache = Vec3(0.0f, 0.0f, 0.0f);
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
