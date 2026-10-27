

ROCKETGPU_HD Vec3 FixArenaTriangleContactNormalDirectionBullet(
    const bullet_port::btConvexShape& shape_a,
    const bullet_port::btTransform& local_trans_a,
    const bullet_port::btConvexShape& shape_b,
    const bullet_port::btTransform& local_trans_b,
    const Vec3& normal_in_b) {
    bullet_port::btVector3 aabb_min(0.0f, 0.0f, 0.0f);
    bullet_port::btVector3 aabb_max(0.0f, 0.0f, 0.0f);
    shape_a.getAabb(local_trans_a, aabb_min, aabb_max);
    const bullet_port::btVector3 pos_a = (aabb_max + aabb_min) * 0.5f;

    shape_b.getAabb(local_trans_b, aabb_min, aabb_max);
    const bullet_port::btVector3 pos_b = (aabb_max + aabb_min) * 0.5f;

    bullet_port::btVector3 normal_bt(normal_in_b.x, normal_in_b.y, normal_in_b.z);
    const bullet_port::btVector3 diff = pos_a - pos_b;
    if (diff.dot(normal_bt) < 0.0f) {
        normal_bt *= -1.0f;
    }
    return Vec3(normal_bt.x(), normal_bt.y(), normal_bt.z());
}

ROCKETGPU_HD Vec3 ArenaTriangleFaceNormal(const ArenaMeshTriangle& tri) {
    const Vec3 e1 = tri.b - tri.a;
    const Vec3 e2 = tri.c - tri.a;
    const Vec3 n(e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x);
    const float inv_len = 1.0f / SqrtScalar(n.x * n.x + n.y * n.y + n.z * n.z);
    return Vec3(n.x * inv_len, n.y * inv_len, n.z * inv_len);
}

ROCKETGPU_HD bullet_port::btVector3 ToBulletPortVec3(const Vec3& v) {
    return bullet_port::btVector3(v.x, v.y, v.z);
}

ROCKETGPU_HD Vec3 FromBulletPortVec3(const bullet_port::btVector3& v) {
    return Vec3(v.x(), v.y(), v.z());
}

struct BallCarContactGeometryDebug {
    uint8_t valid = 0;
    uint8_t used_gjk = 0;
    uint16_t reserved = 0;
    int32_t gjk_update_region = -1;
    int32_t gjk_pre_reduce_num_vertices = 0;
    int32_t gjk_triangle_region = -1;
    Vec3 gjk_cached_axis_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_normal_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_fixed_normal_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_point_on_car_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_point_on_ball_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_cached_p1_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_cached_p2_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_triangle_closest_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_simplex_w0_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_simplex_w1_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_simplex_w2_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_simplex_p0_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_simplex_p1_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_simplex_p2_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_simplex_q0_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_simplex_q1_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_simplex_q2_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_local_trans_a_origin_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_local_trans_a_basis0 = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_local_trans_a_basis1 = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_local_trans_a_basis2 = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_support_axis_a0 = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_support_axis_a1 = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_support_axis_a2 = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_support_p_in_a0 = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_support_p_in_a1 = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 gjk_support_p_in_a2 = Vec3(0.0f, 0.0f, 0.0f);
    float gjk_len_sqr = 0.0f;
    float gjk_inv_len = 0.0f;
    float gjk_distance_bt = 0.0f;
    float gjk_triangle_param = 0.0f;
    float gjk_barycentric0 = 0.0f;
    float gjk_barycentric1 = 0.0f;
    float gjk_barycentric2 = 0.0f;
    float gjk_barycentric3 = 0.0f;
};

ROCKETGPU_HD bool GetCarBasisForBulletTransform(const CarState& car, Vec3& fwd, Vec3& rgt, Vec3& upv) {
    if (car.basis_authoritative) {
        CarBasisColumns(car, fwd, rgt, upv);
        return false;
    }
    CarBasisColumns(car, fwd, rgt, upv);
    return false;
}

ROCKETGPU_HD bullet_port::btConvexShape MakeBulletPortBoxShape(const Vec3& half_extents_with_margin, float margin) {
    bullet_port::btConvexShape shape;
    shape.m_kind = bullet_port::BT_SHAPE_BOX;
    const float min_dimension = Min(half_extents_with_margin.x, Min(half_extents_with_margin.y, half_extents_with_margin.z));
    const float safe_margin = Min(margin, 0.1f * min_dimension);

    const float implicit_margin = safe_margin;
    shape.m_implicitShapeDimensions = ToBulletPortVec3(
        half_extents_with_margin - Vec3(implicit_margin, implicit_margin, implicit_margin));
    shape.m_margin = safe_margin;
    return shape;
}

ROCKETGPU_HD bullet_port::btConvexShape MakeBulletPortTriangleShape(const Vec3& a, const Vec3& b, const Vec3& c, float margin) {
    bullet_port::btConvexShape shape;
    shape.m_kind = bullet_port::BT_SHAPE_TRIANGLE;
    shape.m_vertices1[0] = ToBulletPortVec3(a);
    shape.m_vertices1[1] = ToBulletPortVec3(b);
    shape.m_vertices1[2] = ToBulletPortVec3(c);
    shape.m_margin = margin;
    return shape;
}

ROCKETGPU_HD bullet_port::btTransform MakeBulletPortBoxWorldTransform(const CarState& car, const Vec3& box_center_bt) {
    bullet_port::btTransform transform;
    Vec3 fwd;
    Vec3 rgt;
    Vec3 upv;
    const bool used_host_validation_basis = GetCarBasisForBulletTransform(car, fwd, rgt, upv);
    transform.setBasis(bullet_port::btMatrix3x3(
        bullet_port::btVector3(fwd.x, rgt.x, upv.x),
        bullet_port::btVector3(fwd.y, rgt.y, upv.y),
        bullet_port::btVector3(fwd.z, rgt.z, upv.z)));
    transform.setOrigin(ToBulletPortVec3(box_center_bt));
    return transform;
}

ROCKETGPU_HD bullet_port::btTransform MakeBulletPortBallWorldTransform(const BallState& ball, const Vec3& ball_center_bt) {
    bullet_port::btTransform transform;
    Vec3 fwd;
    Vec3 rgt;
    Vec3 upv;
    GetBallBasisColumns(ball, fwd, rgt, upv);
    transform.setBasis(bullet_port::btMatrix3x3(
        bullet_port::btVector3(fwd.x, rgt.x, upv.x),
        bullet_port::btVector3(fwd.y, rgt.y, upv.y),
        bullet_port::btVector3(fwd.z, rgt.z, upv.z)));
    transform.setOrigin(ToBulletPortVec3(ball_center_bt));
    return transform;
}

#if defined(__CUDACC__) && defined(_MSC_VER)
#define ROCKETGPU_TRIANGLE_CONTACT_NOINLINE __noinline__
#elif defined(__CUDACC__)
#define ROCKETGPU_TRIANGLE_CONTACT_NOINLINE __noinline__ __attribute__((noinline))
#elif defined(__GNUC__) || defined(__clang__)
#define ROCKETGPU_TRIANGLE_CONTACT_NOINLINE __attribute__((noinline))
#else
#define ROCKETGPU_TRIANGLE_CONTACT_NOINLINE
#endif

ROCKETGPU_HD bool StopCalcPenDepthDebug(uint8_t stop_phase,
    int guess_index,
    int reject_stage,
    bool penetration_valid,
    bool distance_valid,
    float distance_bt,
    const bullet_port::btVector3& guess_vector) {
    if (0 != stop_phase) {
        return false;
    }
    return true;
}

ROCKETGPU_HD ROCKETGPU_TRIANGLE_CONTACT_NOINLINE bool BulletPortCalcPenDepth(
    const bullet_port::btConvexShape& a,
    const bullet_port::btTransform& transform_a,
    const bullet_port::btConvexShape& b,
    const bullet_port::btTransform& transform_b,
    bullet_port::btVector3& io_v,
    bullet_port::btVector3& out_witness_on_a,
    bullet_port::btVector3& out_witness_on_b,
    bullet_port::btGjkEpaScratch* scratch) {
    const bullet_port::btVector3 delta_origin =
        transform_b.getOrigin() - transform_a.getOrigin();
    bullet_port::btVector3 guess_vectors[] = {
        delta_origin.safeNormalize(),
        (-delta_origin).safeNormalize(),
        bullet_port::btVector3(0.0f, 0.0f, 1.0f),
        bullet_port::btVector3(0.0f, 1.0f, 0.0f),
        bullet_port::btVector3(1.0f, 0.0f, 0.0f),
        bullet_port::btVector3(1.0f, 1.0f, 0.0f),
        bullet_port::btVector3(1.0f, 1.0f, 1.0f),
        bullet_port::btVector3(0.0f, 1.0f, 1.0f),
        bullet_port::btVector3(1.0f, 0.0f, 1.0f),
    };

    const int num_vectors = static_cast<int>(sizeof(guess_vectors) / sizeof(guess_vectors[0]));
    for (int i = 0; i < num_vectors; ++i) {
        bullet_port::btGjkEpaSolver2::sResults results;
        if (i == 0 && StopCalcPenDepthDebug(220,
                i,
                10,
                false,
                false,
                0.0f,
                guess_vectors[i])) {
            return false;
        }
        if (i == 0 && StopCalcPenDepthDebug(234,
                i,
                14,
                false,
                false,
                0.0f,
                guess_vectors[i])) {
            return false;
        }
        bullet_port::btGjkEpaDebugStop penetration_debug;

#if defined(__CUDA_ARCH__)
        if (scratch == nullptr) {
            return false;
        }
        const bool penetrated = bullet_port::btGjkEpaSolver2::PenetrationWithScratch(
            &a, transform_a, &b, transform_b, guess_vectors[i], results, *scratch, true, &penetration_debug);
#else
        const bool penetrated = scratch
            ? bullet_port::btGjkEpaSolver2::PenetrationWithScratch(&a, transform_a, &b, transform_b, guess_vectors[i], results, *scratch, true, &penetration_debug)
            : bullet_port::btGjkEpaSolver2::Penetration(&a, transform_a, &b, transform_b, guess_vectors[i], results, true);
#endif
        if (penetration_debug.hit_phase != 0) {
            return false;
        }
        if (i == 0 && StopCalcPenDepthDebug(221,
                i,
                11,
                penetrated,
                false,
                results.distance,
                guess_vectors[i])) {
            return false;
        }
        if (penetrated) {
            out_witness_on_a = results.witnesses[0];
            out_witness_on_b = results.witnesses[1];
            io_v = results.normal;
            return true;
        }
        if (i == 0 && StopCalcPenDepthDebug(230,
                i,
                12,
                false,
                false,
                results.distance,
                guess_vectors[i])) {
            return false;
        }
#if defined(__CUDA_ARCH__)
        const bool separated = bullet_port::btGjkEpaSolver2::DistanceWithScratch(
            &a, transform_a, &b, transform_b, guess_vectors[i], results, *scratch);
#else
        const bool separated = scratch
            ? bullet_port::btGjkEpaSolver2::DistanceWithScratch(&a, transform_a, &b, transform_b, guess_vectors[i], results, *scratch)
            : bullet_port::btGjkEpaSolver2::Distance(&a, transform_a, &b, transform_b, guess_vectors[i], results);
#endif
        if (i == 0 && StopCalcPenDepthDebug(231,
                i,
                13,
                false,
                separated,
                results.distance,
                guess_vectors[i])) {
            return false;
        }
        if (separated) {
            out_witness_on_a = results.witnesses[0];
            out_witness_on_b = results.witnesses[1];
            io_v = results.normal;
            return false;
        }
    }

    out_witness_on_a = bullet_port::btVector3(0.0f, 0.0f, 0.0f);
    out_witness_on_b = bullet_port::btVector3(0.0f, 0.0f, 0.0f);
    io_v = bullet_port::btVector3(0.0f, 0.0f, 0.0f);
    return false;
}

ROCKETGPU_HD Vec3 CarBodyLocalPointBt(const CarState& car, const Vec3& point_world_bt) {
    bullet_port::btTransform body_world;
    Vec3 fwd;
    Vec3 rgt;
    Vec3 upv;
    GetCarBasisForBulletTransform(car, fwd, rgt, upv);
    body_world.setBasis(bullet_port::btMatrix3x3(
        bullet_port::btVector3(fwd.x, rgt.x, upv.x),
        bullet_port::btVector3(fwd.y, rgt.y, upv.y),
        bullet_port::btVector3(fwd.z, rgt.z, upv.z)));
    body_world.setOrigin(ToBulletPortVec3(car.pos_bt));
    const bullet_port::btVector3 local = body_world.invXform(ToBulletPortVec3(point_world_bt));
    return FromBulletPortVec3(local);
}

ROCKETGPU_HD Vec3 CarBodyWorldPointBt(const CarState& car, const Vec3& point_local_bt) {
    bullet_port::btTransform body_world;
    Vec3 fwd;
    Vec3 rgt;
    Vec3 upv;
    GetCarBasisForBulletTransform(car, fwd, rgt, upv);
    body_world.setBasis(bullet_port::btMatrix3x3(
        bullet_port::btVector3(fwd.x, rgt.x, upv.x),
        bullet_port::btVector3(fwd.y, rgt.y, upv.y),
        bullet_port::btVector3(fwd.z, rgt.z, upv.z)));
    body_world.setOrigin(ToBulletPortVec3(car.pos_bt));
    const bullet_port::btVector3 world =
        body_world(ToBulletPortVec3(point_local_bt));
    return FromBulletPortVec3(world);
}

ROCKETGPU_HD Vec3 CarBodyManifoldWorldPointBt(const CarState& car, const Vec3& point_world_bt) {
    return CarBodyWorldPointBt(car, CarBodyLocalPointBt(car, point_world_bt));
}

ROCKETGPU_HD bool ComputeCarArenaTriangleContact(
    const CarState& car,
    const CarConfig& cfg,
    const ArenaMeshTriangle& tri,
    float contact_threshold_bt,
    Vec3& out_normal,
    float& out_distance_bt,
    Vec3* out_point_on_a_bt,
    Vec3* out_point_on_b_bt,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch,
    Vec3* io_cached_axis_bt,
    TriangleContactDebugSummary* debug_summary,
    uint8_t debug_triangle_phase,
    float tri_origin_y_bt) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kRelError2 = 1.0e-6f;
    constexpr int kGjkMaxIter = 1000;
    constexpr float kBulletBoxMarginBt = 0.04f;

    const float requested_margin_a_bt = kBulletBoxMarginBt;
    const Vec3 raw_half_extent_bt = (cfg.hitbox_size * 0.5f) * kUUToBT;

    const float requested_margin_b_bt = 0.0f;
    const float maximum_distance_squared = 100.0f;
    const bullet_port::btConvexShape box_shape =
        MakeBulletPortBoxShape(raw_half_extent_bt, requested_margin_a_bt);
    const bullet_port::btConvexShape tri_shape =
        MakeBulletPortTriangleShape(tri.a, tri.b, tri.c, requested_margin_b_bt);
    const float margin_a_bt = box_shape.getMarginNonVirtual();
    const float margin_b_bt = tri_shape.getMarginNonVirtual();
    const bullet_port::btTransform body_transform = MakeBulletPortBoxWorldTransform(car, car.pos_bt);
    const bullet_port::btVector3 child_origin_bt = ToBulletPortVec3(cfg.hitbox_offset * kUUToBT);
    bullet_port::btTransform box_transform = body_transform;
    box_transform.setOrigin(box_transform * child_origin_bt);
    const float margin_bt = margin_a_bt + margin_b_bt;

    bullet_port::btTransform tri_transform;
    if (tri_origin_y_bt != 0.0f) {
        tri_transform.setOrigin(bullet_port::btVector3(0.0f, tri_origin_y_bt, 0.0f));
    }
    bullet_port::btTransform local_trans_a = box_transform;
    bullet_port::btTransform local_trans_b = tri_transform;
    const bullet_port::btVector3 position_offset =
        (local_trans_a.getOrigin() + local_trans_b.getOrigin()) * 0.5f;
    local_trans_a.getOrigin() -= position_offset;
    local_trans_b.getOrigin() -= position_offset;
    const Vec3 position_offset_bt = FromBulletPortVec3(position_offset);

    GjkSimplexSolver simplex;
    simplex.reset();

    Vec3 cached_axis = io_cached_axis_bt != nullptr ? *io_cached_axis_bt : Vec3(0.0f, 1.0f, 0.0f);
    Vec3 last_nonzero_axis = cached_axis;
    Vec3 debug_gjk_axis_bt = cached_axis;
    float debug_gjk_len_sqr = 0.0f;
    float debug_gjk_squared_distance = 0.0f;

    bool is_valid = false;
    bool check_simplex = false;
    int degenerate_simplex = 0;
    int last_used_method = -1;
    int cur_iter = 0;
    float squared_distance = 1.0e30f;
    float distance_bt = 0.0f;
    bool used_penetration_depth_solver = false;
    bool penetration_valid = false;
    Vec3 normal_bt(0.0f, 0.0f, 0.0f);
    Vec3 point_on_a_bt(0.0f, 0.0f, 0.0f);
    Vec3 point_on_b_bt(0.0f, 0.0f, 0.0f);

    while (true) {
        const bullet_port::btVector3 cached_separating_axis = ToBulletPortVec3(cached_axis);
        const bullet_port::btVector3 separating_axis_in_a =
            (-cached_separating_axis) * box_transform.getBasis();
        const bullet_port::btVector3 separating_axis_in_b =
            cached_separating_axis * tri_transform.getBasis();
        const bullet_port::btVector3 p_in_a =
            box_shape.localGetSupportVertexWithoutMarginNonVirtual(separating_axis_in_a);
        const bullet_port::btVector3 tri_support_dots =
            separating_axis_in_b.dot3(tri_shape.m_vertices1[0], tri_shape.m_vertices1[1], tri_shape.m_vertices1[2]);
        const bullet_port::btVector3 q_in_b =
            tri_shape.localGetSupportVertexWithoutMarginNonVirtual(separating_axis_in_b);
        const bullet_port::btVector3 p_world_bt = local_trans_a(p_in_a);
        const bullet_port::btVector3 q_world_bt = local_trans_b(q_in_b);
        const Vec3 p_world = FromBulletPortVec3(p_world_bt);
        const Vec3 q_world = FromBulletPortVec3(q_world_bt);
        const Vec3 w = p_world - q_world;
        const float delta = Dot(cached_axis, w);
        const bool in_simplex = simplex.in_simplex(w);
        const float f0 = squared_distance - delta;
        const float f1 = squared_distance * kRelError2;

        if ((delta > 0.0f) && (delta * delta > squared_distance * maximum_distance_squared)) {
            degenerate_simplex = 10;
            check_simplex = true;
            break;
        }

        if (in_simplex) {
            degenerate_simplex = 1;
            check_simplex = true;
            break;
        }

        if (f0 <= f1) {
            degenerate_simplex = f0 <= 0.0f ? 2 : 11;
            check_simplex = true;
            break;
        }

        simplex.add_vertex(w, p_world, q_world);
        Vec3 new_cached_axis(0.0f, 0.0f, 0.0f);
        const bool closest_success = simplex.closest(new_cached_axis);
        if (!closest_success) {
            degenerate_simplex = 3;
            check_simplex = true;
            break;
        }

        if (LengthSq(new_cached_axis) >= kRelError2) {
            last_nonzero_axis = Normalize(new_cached_axis, last_nonzero_axis);
        }

        if (LengthSq(new_cached_axis) < kRelError2) {
            cached_axis = new_cached_axis;
            degenerate_simplex = 6;
            check_simplex = true;
            break;
        }

        const float previous_squared_distance = squared_distance;
        squared_distance = LengthSq(new_cached_axis);
        if (previous_squared_distance - squared_distance <= 1.1920929e-7f * previous_squared_distance) {
            check_simplex = true;
            degenerate_simplex = 12;
            break;
        }

        cached_axis = new_cached_axis;

        if (cur_iter++ > kGjkMaxIter) {
            break;
        }

        if (simplex.full_simplex()) {
            degenerate_simplex = 13;
            break;
        }
    }

    Vec3 gjk_normal_hint = last_nonzero_axis;
    if (check_simplex) {
        simplex.compute_points(point_on_a_bt, point_on_b_bt);
        if (LengthSq(cached_axis) > kRelError2) {
            gjk_normal_hint = Normalize(cached_axis, gjk_normal_hint);
        }

        const float len_sqr = LengthSq(cached_axis);
        debug_gjk_axis_bt = cached_axis;
        debug_gjk_len_sqr = len_sqr;
        debug_gjk_squared_distance = squared_distance;
        if (len_sqr < 0.0001f) {
            degenerate_simplex = 5;
        }
        if (len_sqr > 1.1920929e-7f * 1.1920929e-7f) {
            const float rlen = 1.0f / SqrtScalar(len_sqr);
            normal_bt = cached_axis * rlen;

            const float s = SqrtScalar(squared_distance);
            if (s > 0.0f) {
                point_on_a_bt -= cached_axis * (margin_a_bt / s);
                point_on_b_bt += cached_axis * (margin_b_bt / s);
                distance_bt = (1.0f / rlen) - margin_bt;
                is_valid = true;
                last_used_method = 1;
#if defined(__CUDA_ARCH__)

                {
                    const bool catch_degenerate_penetration_case_cuda =
                        degenerate_simplex && ((distance_bt + margin_bt) < 0.01f);
                    if (catch_degenerate_penetration_case_cuda) {
                        // Fall through to the shared catch_degenerate +
                    } else {
                        point_on_a_bt += position_offset_bt;
                        point_on_b_bt += position_offset_bt;
                        normal_bt = FixArenaTriangleContactNormalDirectionBullet(
                            box_shape, local_trans_a, tri_shape, local_trans_b, normal_bt);
                        out_normal = normal_bt;
                        out_distance_bt = distance_bt;
                        if (out_point_on_a_bt != nullptr) {
                            *out_point_on_a_bt = point_on_a_bt;
                        }
                        if (out_point_on_b_bt != nullptr) {
                            *out_point_on_b_bt = point_on_b_bt;
                        }
                        if (io_cached_axis_bt != nullptr) {
                            *io_cached_axis_bt = normal_bt;
                        }
                        return true;
                    }
                }
#endif
            } else {
                last_used_method = 2;
            }
        } else {
            last_used_method = 2;
        }
    }

    const bool catch_degenerate_penetration_case =
        degenerate_simplex && ((distance_bt + margin_bt) < 0.01f);

#if defined(__CUDA_ARCH__)
    const volatile int32_t method_before_penetration = last_used_method;
    const bool needs_penetration =
        !((method_before_penetration == 1) ||
          (method_before_penetration == 3) ||
          (method_before_penetration == 5) ||
          (method_before_penetration == 6) ||
          (method_before_penetration == 8)) ||
        catch_degenerate_penetration_case;
#else
    const bool needs_penetration = !is_valid || catch_degenerate_penetration_case;
#endif

    if (needs_penetration) {
        bullet_port::btVector3 cached_separating_axis(0.0f, 0.0f, 0.0f);
        bullet_port::btVector3 tmp_point_on_a_world(0.0f, 0.0f, 0.0f);
        bullet_port::btVector3 tmp_point_on_b_world(0.0f, 0.0f, 0.0f);

        const bool is_valid2 = BulletPortCalcPenDepth(
            box_shape,
            local_trans_a,
            tri_shape,
            local_trans_b,
            cached_separating_axis,
            tmp_point_on_a_world,
            tmp_point_on_b_world,
            gjk_epa_scratch);
        penetration_valid = is_valid2;

        if (is_valid2) {
            bullet_port::btVector3 tmp_normal_in_b = tmp_point_on_b_world - tmp_point_on_a_world;
            float len_sqr = tmp_normal_in_b.length2();
            if (len_sqr <= (1.1920929e-7f * 1.1920929e-7f)) {
                tmp_normal_in_b = cached_separating_axis;
                len_sqr = cached_separating_axis.length2();
            }

            if (len_sqr > (1.1920929e-7f * 1.1920929e-7f)) {
                tmp_normal_in_b /= SqrtScalar(len_sqr);
                const float distance2 = -Length(FromBulletPortVec3(tmp_point_on_a_world - tmp_point_on_b_world));
                if (!is_valid || (distance2 < distance_bt)) {
                    distance_bt = distance2;
                    point_on_a_bt = FromBulletPortVec3(tmp_point_on_a_world);
                    point_on_b_bt = FromBulletPortVec3(tmp_point_on_b_world);
                    normal_bt = FromBulletPortVec3(tmp_normal_in_b);
                    is_valid = true;
                    last_used_method = 3;
                    used_penetration_depth_solver = true;
                } else {
                    last_used_method = 8;
                }
            } else {
                last_used_method = 9;
            }
        } else {
            if (cached_separating_axis.length2() > 0.0f) {
                const float distance2 = Length(FromBulletPortVec3(tmp_point_on_a_world - tmp_point_on_b_world)) - margin_bt;
                if (!is_valid || (distance2 < distance_bt)) {
                    bullet_port::btVector3 point_on_a_world = tmp_point_on_a_world - cached_separating_axis * margin_a_bt;
                    bullet_port::btVector3 point_on_b_world = tmp_point_on_b_world + cached_separating_axis * margin_b_bt;
                    point_on_a_bt = FromBulletPortVec3(point_on_a_world);
                    point_on_b_bt = FromBulletPortVec3(point_on_b_world);
                    normal_bt = FromBulletPortVec3(cached_separating_axis);
                    normal_bt = Normalize(normal_bt, normal_bt);
                    distance_bt = distance2;
                    is_valid = true;
                    last_used_method = 6;
                } else {
                    last_used_method = 5;
                }
            }
        }
    }

    (void)last_used_method;

    const bool has_valid_contact = is_valid;

    if (!has_valid_contact) {
        if (io_cached_axis_bt != nullptr) {
            *io_cached_axis_bt = cached_axis;
        }
        return false;
    }

    const float filter_distance_bt = distance_bt;
    if (!((filter_distance_bt < 0.0f) ||
          (filter_distance_bt * filter_distance_bt < maximum_distance_squared))) {
        if (io_cached_axis_bt != nullptr) {
            *io_cached_axis_bt = cached_axis;
        }
        return false;
    }

    point_on_a_bt += position_offset_bt;
    point_on_b_bt += position_offset_bt;

    normal_bt = FixArenaTriangleContactNormalDirectionBullet(
        box_shape, local_trans_a, tri_shape, local_trans_b, normal_bt);

    out_normal = normal_bt;
    if (out_point_on_a_bt) {
        *out_point_on_a_bt = point_on_a_bt;
    }
    if (out_point_on_b_bt) {
        *out_point_on_b_bt = point_on_b_bt;
    }
    out_distance_bt = distance_bt;
    if (io_cached_axis_bt != nullptr) {
        *io_cached_axis_bt = is_valid ? normal_bt : cached_axis;
    }
    return true;
}

#if defined(ROCKETGPU_CAN_COMPILE_OFFICIAL_HOST_HELPERS)

ROCKETGPU_HD bool ComputeCarArenaTriangleContactHost(
    const CarState& car,
    const CarConfig& cfg,
    const ArenaMeshTriangleHost& tri,
    float contact_threshold_bt,
    Vec3& out_normal,
    float& out_distance_bt) {
    ArenaMeshTriangle common_tri;
    common_tri.a = tri.a;
    common_tri.b = tri.b;
    common_tri.c = tri.c;
    common_tri.normal = tri.normal;
    common_tri.min = tri.min;
    common_tri.max = tri.max;
    common_tri.mesh_index = static_cast<uint8_t>(tri.mesh_index);
    common_tri.quantized_aabb_min[0] = tri.quantized_aabb_min[0];
    common_tri.quantized_aabb_min[1] = tri.quantized_aabb_min[1];
    common_tri.quantized_aabb_min[2] = tri.quantized_aabb_min[2];
    common_tri.quantized_aabb_max[0] = tri.quantized_aabb_max[0];
    common_tri.quantized_aabb_max[1] = tri.quantized_aabb_max[1];
    common_tri.quantized_aabb_max[2] = tri.quantized_aabb_max[2];
    if (!ComputeCarArenaTriangleContact(car, cfg, common_tri, contact_threshold_bt, out_normal, out_distance_bt)) {
        return false;
    }
    return out_distance_bt <= contact_threshold_bt;
}
#endif

ROCKETGPU_HD float BallCarContactBreakingThresholdBT(const CarConfig& cfg, float ball_radius);

ROCKETGPU_HD bool ComputeBallCarContactGeometryGjk(
    const BallState& ball,
    const CarState& car,
    const CarConfig& cfg,
    float ball_radius,
    Vec3& out_normal,
    float& out_penetration,
    Vec3& out_rel_ball_contact,
    Vec3& out_rel_car_contact,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch = nullptr,
    BallCarContactGeometryDebug* debug = nullptr,
    Vec3* out_bt_rel_ball_contact_for_solver = nullptr,
    Vec3* out_bt_rel_car_contact_for_solver = nullptr) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kBTToUU = 50.0f;
    constexpr float kRelError2 = 1.0e-6f;
    constexpr int kGjkMaxIter = 1000;
    constexpr float kBulletBoxMarginBt = 0.04f;

    const Vec3 raw_half_extent_bt = (cfg.hitbox_size * 0.5f) * kUUToBT;
    const bullet_port::btConvexShape box_shape =
        MakeBulletPortBoxShape(raw_half_extent_bt, kBulletBoxMarginBt);
    const bullet_port::btSphereShape sphere_shape(ball_radius * kUUToBT);
    const float margin_a_bt = box_shape.getMarginNonVirtual();
    const float margin_b_bt = sphere_shape.getMarginNonVirtual();

    const float contact_breaking_threshold_bt = BallCarContactBreakingThresholdBT(cfg, ball_radius);
    float maximum_distance_squared = margin_a_bt + margin_b_bt + contact_breaking_threshold_bt;
    maximum_distance_squared *= maximum_distance_squared;
    const bullet_port::btTransform car_transform =
        MakeBulletPortBoxWorldTransform(car, car.pos_bt);
    bullet_port::btTransform box_transform = car_transform;

    box_transform.setOrigin(car_transform(ToBulletPortVec3(cfg.hitbox_offset * kUUToBT)));
    const bullet_port::btTransform sphere_transform =
        MakeBulletPortBallWorldTransform(ball, ball.pos_bt);
    bullet_port::btTransform local_trans_a = box_transform;
    bullet_port::btTransform local_trans_b = sphere_transform;
    const bullet_port::btVector3 position_offset =
        (local_trans_a.getOrigin() + local_trans_b.getOrigin()) * 0.5f;
    local_trans_a.getOrigin() -= position_offset;
    local_trans_b.getOrigin() -= position_offset;
    const Vec3 position_offset_bt = FromBulletPortVec3(position_offset);

    GjkSimplexSolver simplex;
    simplex.reset();

    Vec3 cached_axis(0.0f, 1.0f, 0.0f);
    Vec3 debug_support_axis_a[4];
    Vec3 debug_support_p_in_a[4];
    for (int i = 0; i < 4; ++i) {
        debug_support_axis_a[i] = Vec3(0.0f, 0.0f, 0.0f);
        debug_support_p_in_a[i] = Vec3(0.0f, 0.0f, 0.0f);
    }
    float squared_distance = 1.0e30f;
    bool check_simplex = false;
    bool is_valid = false;
    int degenerate_simplex = 0;
    int last_used_method = -1;
    int cur_iter = 0;
    float distance_bt = 0.0f;
    Vec3 normal_bt(0.0f, 0.0f, 0.0f);

    while (true) {
        const bullet_port::btVector3 cached_separating_axis = ToBulletPortVec3(cached_axis);
        const bullet_port::btVector3 separating_axis_in_a =
            (-cached_separating_axis) * box_transform.getBasis();
        const bullet_port::btVector3 separating_axis_in_b =
            cached_separating_axis * sphere_transform.getBasis();
        const bullet_port::btVector3 p_in_a =
            box_shape.localGetSupportVertexWithoutMarginNonVirtual(separating_axis_in_a);
        const bullet_port::btVector3 q_in_b =
            sphere_shape.localGetSupportVertexWithoutMarginNonVirtual(separating_axis_in_b);
        const Vec3 p_world = FromBulletPortVec3(local_trans_a(p_in_a));
        const Vec3 q_world = FromBulletPortVec3(local_trans_b(q_in_b));
        const Vec3 w = p_world - q_world;
        const float delta = Dot(cached_axis, w);

        if ((delta > 0.0f) && (delta * delta > squared_distance * maximum_distance_squared)) {
            degenerate_simplex = 10;
            check_simplex = true;
            break;
        }

        if (simplex.in_simplex(w)) {
            degenerate_simplex = 1;
            check_simplex = true;
            break;
        }

        const float f0 = squared_distance - delta;
        const float f1 = squared_distance * kRelError2;
        if (f0 <= f1) {
            degenerate_simplex = f0 <= 0.0f ? 2 : 11;
            check_simplex = true;
            break;
        }

        const int simplex_slot = simplex.num_vertices;
        if (simplex_slot >= 0 && simplex_slot < 4) {
            debug_support_axis_a[simplex_slot] = FromBulletPortVec3(separating_axis_in_a);
            debug_support_p_in_a[simplex_slot] = FromBulletPortVec3(p_in_a);
        }
        simplex.add_vertex(w, p_world, q_world);
        Vec3 new_cached_axis(0.0f, 0.0f, 0.0f);
        if (!simplex.closest(new_cached_axis)) {
            degenerate_simplex = 3;
            check_simplex = true;
            break;
        }

        if (LengthSq(new_cached_axis) < kRelError2) {
            cached_axis = new_cached_axis;
            degenerate_simplex = 6;
            check_simplex = true;
            break;
        }

        const float previous_squared_distance = squared_distance;
        squared_distance = LengthSq(new_cached_axis);
        if (previous_squared_distance - squared_distance <= 1.1920929e-7f * previous_squared_distance) {
            degenerate_simplex = 12;
            check_simplex = true;
            break;
        }

        cached_axis = new_cached_axis;
        if (cur_iter++ > kGjkMaxIter) {
            break;
        }
        if (simplex.full_simplex()) {
            degenerate_simplex = 13;
            break;
        }
    }

    if (!check_simplex) {
        return false;
    }

    Vec3 point_on_a_bt(0.0f, 0.0f, 0.0f);
    Vec3 point_on_b_bt(0.0f, 0.0f, 0.0f);
    simplex.compute_points(point_on_a_bt, point_on_b_bt);

    const float len_sqr = LengthSq(cached_axis);
    if (len_sqr < 0.0001f) {
        degenerate_simplex = 5;
    }

    float inv_len = 0.0f;
    const float s = SqrtScalar(squared_distance);
    if (len_sqr > 1.1920929e-7f * 1.1920929e-7f) {
        inv_len = 1.0f / SqrtScalar(len_sqr);
        normal_bt = cached_axis * inv_len;
        if (s > 0.0f) {
            point_on_a_bt -= cached_axis * (margin_a_bt / s);
            point_on_b_bt += cached_axis * (margin_b_bt / s);
            distance_bt = (1.0f / inv_len) - (margin_a_bt + margin_b_bt);
            is_valid = true;
            last_used_method = 1;
        } else {
            last_used_method = 2;
        }
    } else {
        last_used_method = 2;
    }

    const bool catch_degenerate_penetration_case =
        degenerate_simplex && ((distance_bt + margin_a_bt + margin_b_bt) < 0.01f);
    if (!is_valid || catch_degenerate_penetration_case) {
        bullet_port::btVector3 cached_separating_axis(0.0f, 0.0f, 0.0f);
        bullet_port::btVector3 tmp_point_on_a_world(0.0f, 0.0f, 0.0f);
        bullet_port::btVector3 tmp_point_on_b_world(0.0f, 0.0f, 0.0f);

        const bool is_valid2 = BulletPortCalcPenDepth(
            box_shape,
            local_trans_a,
            sphere_shape,
            local_trans_b,
            cached_separating_axis,
            tmp_point_on_a_world,
            tmp_point_on_b_world,
            gjk_epa_scratch);
        if (is_valid2) {
            bullet_port::btVector3 tmp_normal_in_b = tmp_point_on_b_world - tmp_point_on_a_world;
            float len_sqr_pen = tmp_normal_in_b.length2();
            if (len_sqr_pen <= 1.1920929e-7f * 1.1920929e-7f) {
                tmp_normal_in_b = cached_separating_axis;
                len_sqr_pen = cached_separating_axis.length2();
            }
            if (len_sqr_pen > 1.1920929e-7f * 1.1920929e-7f) {
                tmp_normal_in_b /= SqrtScalar(len_sqr_pen);
                const float distance2 = -Length(FromBulletPortVec3(tmp_point_on_a_world - tmp_point_on_b_world));
                if (!is_valid || distance2 < distance_bt) {
                    distance_bt = distance2;
                    point_on_a_bt = FromBulletPortVec3(tmp_point_on_a_world);
                    point_on_b_bt = FromBulletPortVec3(tmp_point_on_b_world);
                    normal_bt = FromBulletPortVec3(tmp_normal_in_b);
                    is_valid = true;
                    last_used_method = 3;
                } else {
                    last_used_method = 8;
                }
            } else {
                last_used_method = 9;
            }
        } else if (cached_separating_axis.length2() > 0.0f) {
            const float distance2 = Length(FromBulletPortVec3(tmp_point_on_a_world - tmp_point_on_b_world)) -
                (margin_a_bt + margin_b_bt);
            if (!is_valid || distance2 < distance_bt) {
                const bullet_port::btVector3 point_on_a_world =
                    tmp_point_on_a_world - cached_separating_axis * margin_a_bt;
                const bullet_port::btVector3 point_on_b_world =
                    tmp_point_on_b_world + cached_separating_axis * margin_b_bt;
                point_on_a_bt = FromBulletPortVec3(point_on_a_world);
                point_on_b_bt = FromBulletPortVec3(point_on_b_world);
                normal_bt = Normalize(FromBulletPortVec3(cached_separating_axis), normal_bt);
                distance_bt = distance2;
                is_valid = true;
                last_used_method = 6;
            } else {
                last_used_method = 5;
            }
        }
    }

    if (!is_valid) {
        return false;
    }

    if (!((distance_bt < 0.0f) || (distance_bt * distance_bt < maximum_distance_squared))) {
        return false;
    }
    if (distance_bt > contact_breaking_threshold_bt) {
        return false;
    }

    const Vec3 point_on_car_bt = point_on_a_bt + position_offset_bt;
    const Vec3 point_on_ball_bt = point_on_b_bt + position_offset_bt;
    const Vec3 bullet_normal_bt = FixArenaTriangleContactNormalDirectionBullet(
        box_shape, local_trans_a, sphere_shape, local_trans_b, normal_bt);
    const Vec3 manifold_point_on_car_bt =
        point_on_ball_bt + bullet_normal_bt * distance_bt;
    if (debug) {
        debug->valid = 1;
        debug->used_gjk = 1;
        debug->gjk_update_region = simplex.debug_update_region;
        debug->gjk_pre_reduce_num_vertices = simplex.debug_pre_reduce_num_vertices;
        debug->gjk_triangle_region = simplex.debug_triangle_region;
        debug->gjk_cached_axis_bt = cached_axis;
        debug->gjk_normal_bt = normal_bt;
        debug->gjk_fixed_normal_bt = bullet_normal_bt;
        debug->gjk_point_on_car_bt = point_on_car_bt;
        debug->gjk_point_on_ball_bt = point_on_ball_bt;
        debug->gjk_cached_p1_bt = simplex.debug_cached_p1;
        debug->gjk_cached_p2_bt = simplex.debug_cached_p2;
        debug->gjk_triangle_closest_bt = simplex.debug_triangle_closest_point;
        debug->gjk_simplex_w0_bt = simplex.debug_simplex_w[0];
        debug->gjk_simplex_w1_bt = simplex.debug_simplex_w[1];
        debug->gjk_simplex_w2_bt = simplex.debug_simplex_w[2];
        debug->gjk_simplex_p0_bt = simplex.debug_simplex_p[0];
        debug->gjk_simplex_p1_bt = simplex.debug_simplex_p[1];
        debug->gjk_simplex_p2_bt = simplex.debug_simplex_p[2];
        debug->gjk_simplex_q0_bt = simplex.debug_simplex_q[0];
        debug->gjk_simplex_q1_bt = simplex.debug_simplex_q[1];
        debug->gjk_simplex_q2_bt = simplex.debug_simplex_q[2];
        debug->gjk_local_trans_a_origin_bt = FromBulletPortVec3(local_trans_a.getOrigin());
        debug->gjk_local_trans_a_basis0 = FromBulletPortVec3(local_trans_a.getBasis()[0]);
        debug->gjk_local_trans_a_basis1 = FromBulletPortVec3(local_trans_a.getBasis()[1]);
        debug->gjk_local_trans_a_basis2 = FromBulletPortVec3(local_trans_a.getBasis()[2]);
        debug->gjk_support_axis_a0 = debug_support_axis_a[0];
        debug->gjk_support_axis_a1 = debug_support_axis_a[1];
        debug->gjk_support_axis_a2 = debug_support_axis_a[2];
        debug->gjk_support_p_in_a0 = debug_support_p_in_a[0];
        debug->gjk_support_p_in_a1 = debug_support_p_in_a[1];
        debug->gjk_support_p_in_a2 = debug_support_p_in_a[2];
        debug->gjk_len_sqr = len_sqr;
        debug->gjk_inv_len = inv_len;
        debug->gjk_distance_bt = distance_bt;
        debug->gjk_triangle_param = simplex.debug_triangle_param;
        debug->gjk_barycentric0 = simplex.debug_barycentric_coords[0];
        debug->gjk_barycentric1 = simplex.debug_barycentric_coords[1];
        debug->gjk_barycentric2 = simplex.debug_barycentric_coords[2];
        debug->gjk_barycentric3 = simplex.debug_barycentric_coords[3];
    }

    out_normal = -bullet_normal_bt;
    out_penetration = -distance_bt * kBTToUU;
    const Vec3 car_origin_bt = FromBulletPortVec3(car_transform.getOrigin());
    const Vec3 ball_origin_bt = FromBulletPortVec3(sphere_transform.getOrigin());
    const bullet_port::btVector3 point_on_car_local_bt =
        (ToBulletPortVec3(manifold_point_on_car_bt) - car_transform.getOrigin()) * car_transform.getBasis();
    const bullet_port::btVector3 point_on_ball_local_bt =
        (ToBulletPortVec3(point_on_ball_bt) - sphere_transform.getOrigin()) * sphere_transform.getBasis();
    const Vec3 refreshed_point_on_car_bt = FromBulletPortVec3(car_transform(point_on_car_local_bt));
    const Vec3 refreshed_point_on_ball_bt = FromBulletPortVec3(sphere_transform(point_on_ball_local_bt));
    const Vec3 bt_rel_ball_contact_for_solver = refreshed_point_on_ball_bt - ball_origin_bt;
    out_rel_ball_contact = bt_rel_ball_contact_for_solver * kBTToUU;
    out_rel_car_contact = (refreshed_point_on_car_bt - car_origin_bt) * kBTToUU;
    if (out_bt_rel_car_contact_for_solver) {
        *out_bt_rel_car_contact_for_solver = refreshed_point_on_car_bt - car_origin_bt;
    }
    if (out_bt_rel_ball_contact_for_solver) {
        *out_bt_rel_ball_contact_for_solver = bt_rel_ball_contact_for_solver;
    }
    return true;
}

ROCKETGPU_HD bool TraceBallCarGjkShadowEnabled() {
    return false;
}

ROCKETGPU_HD bool TraceBallCarSphereBoxShadowEnabled() {
    return false;
}

ROCKETGPU_HD bool DebugBallCarSolverTraceEnabledFor(int env_idx, int car_idx) {
    (void)env_idx;
    (void)car_idx;
    return false;
}

ROCKETGPU_HD void TraceBallCarContactGeometryGjkShadow(
    const BallState& ball,
    const CarState& car,
    const CarConfig& cfg,
    float ball_radius) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kRelError2 = 1.0e-6f;
    constexpr int kGjkMaxIter = 1000;
    constexpr float kBulletBoxMarginBt = 0.04f;
    constexpr float kMaximumDistanceSquared = 100.0f;

    const Vec3 raw_half_extent_bt = (cfg.hitbox_size * 0.5f) * kUUToBT;
    const bullet_port::btConvexShape box_shape =
        MakeBulletPortBoxShape(raw_half_extent_bt, kBulletBoxMarginBt);
    const bullet_port::btSphereShape sphere_shape(ball_radius * kUUToBT);
    const float margin_a_bt = box_shape.getMarginNonVirtual();
    const float margin_b_bt = sphere_shape.getMarginNonVirtual();
    const float margin_bt = margin_a_bt + margin_b_bt;
    const bullet_port::btTransform car_transform =
        MakeBulletPortBoxWorldTransform(car, car.pos_bt);
    bullet_port::btTransform box_transform = car_transform;
    box_transform.setOrigin(car_transform(ToBulletPortVec3(cfg.hitbox_offset * kUUToBT)));
    const bullet_port::btTransform sphere_transform =
        MakeBulletPortBallWorldTransform(ball, ball.pos_bt);

    bullet_port::btTransform local_trans_a = box_transform;
    bullet_port::btTransform local_trans_b = sphere_transform;
    const bullet_port::btVector3 position_offset =
        (local_trans_a.getOrigin() + local_trans_b.getOrigin()) * 0.5f;
    local_trans_a.getOrigin() -= position_offset;
    local_trans_b.getOrigin() -= position_offset;

    GjkSimplexSolver simplex;
    simplex.reset();

    Vec3 cached_axis(0.0f, 1.0f, 0.0f);
    float squared_distance = 1.0e30f;
    bool check_simplex = false;
    bool is_valid = false;
    int degenerate_simplex = 0;
    int last_used_method = -1;
    int cur_iter = 0;
    Vec3 point_on_a_bt(0.0f, 0.0f, 0.0f);
    Vec3 point_on_b_bt(0.0f, 0.0f, 0.0f);

    while (true) {
        const bullet_port::btVector3 cached_separating_axis = ToBulletPortVec3(cached_axis);
        const bullet_port::btVector3 separating_axis_in_a =
            (-cached_separating_axis) * box_transform.getBasis();
        const bullet_port::btVector3 separating_axis_in_b =
            cached_separating_axis * sphere_transform.getBasis();

        const bullet_port::btVector3 p_in_a =
            box_shape.localGetSupportVertexWithoutMarginNonVirtual(separating_axis_in_a);
        const bullet_port::btVector3 q_in_b =
            sphere_shape.localGetSupportVertexWithoutMarginNonVirtual(separating_axis_in_b);

        const bullet_port::btVector3 p_world_bt = local_trans_a(p_in_a);
        const bullet_port::btVector3 q_world_bt = local_trans_b(q_in_b);
        const Vec3 p_world = FromBulletPortVec3(p_world_bt);
        const Vec3 q_world = FromBulletPortVec3(q_world_bt);
        const Vec3 w = p_world - q_world;
        const float delta = Dot(cached_axis, w);

        if ((delta > 0.0f) && (delta * delta > squared_distance * kMaximumDistanceSquared)) {
            degenerate_simplex = 10;
            check_simplex = true;
            break;
        }

        if (simplex.in_simplex(w)) {
            degenerate_simplex = 1;
            check_simplex = true;
            break;
        }

        const float f0 = squared_distance - delta;
        const float f1 = squared_distance * kRelError2;
        if (f0 <= f1) {
            degenerate_simplex = f0 <= 0.0f ? 2 : 11;
            check_simplex = true;
            break;
        }

        simplex.add_vertex(w, p_world, q_world);
        Vec3 new_cached_axis(0.0f, 0.0f, 0.0f);
        if (!simplex.closest(new_cached_axis)) {
            degenerate_simplex = 3;
            check_simplex = true;
            break;
        }

        if (LengthSq(new_cached_axis) < kRelError2) {
            cached_axis = new_cached_axis;
            degenerate_simplex = 6;
            check_simplex = true;
            break;
        }

        const float previous_squared_distance = squared_distance;
        squared_distance = LengthSq(new_cached_axis);
        if (previous_squared_distance - squared_distance <= 1.1920929e-7f * previous_squared_distance) {
            degenerate_simplex = 12;
            check_simplex = true;
            break;
        }

        cached_axis = new_cached_axis;
        if (cur_iter++ > kGjkMaxIter) {
            break;
        }
        if (simplex.full_simplex()) {
            degenerate_simplex = 13;
            break;
        }
    }

    if (!check_simplex) {
        return;
    }

    simplex.compute_points(point_on_a_bt, point_on_b_bt);

    const float len_sqr = LengthSq(cached_axis);
    if (len_sqr < 0.0001f) {
        degenerate_simplex = 5;
    }

    float distance_bt = 0.0f;
    Vec3 normal_bt(0.0f, 0.0f, 0.0f);
    if (len_sqr > 1.1920929e-7f * 1.1920929e-7f) {
        const float rlen = 1.0f / SqrtScalar(len_sqr);
        normal_bt = cached_axis * rlen;

        const float s = SqrtScalar(squared_distance);
        if (s > 0.0f) {
            point_on_a_bt -= cached_axis * (margin_a_bt / s);
            point_on_b_bt += cached_axis * (margin_b_bt / s);
            distance_bt = (1.0f / rlen) - margin_bt;
            is_valid = true;
            last_used_method = 1;
        } else {
            last_used_method = 2;
        }
    } else {
        last_used_method = 2;
    }

    const float distance_plus_margin = distance_bt + margin_bt;
    const bool catch_degenerate_penetration_case =
        degenerate_simplex && (distance_plus_margin < 0.01f);
    const bool needs_penetration = !is_valid || catch_degenerate_penetration_case;
    if (needs_penetration) {
        bullet_port::btVector3 cached_separating_axis(0.0f, 0.0f, 0.0f);
        bullet_port::btVector3 tmp_point_on_a_world(0.0f, 0.0f, 0.0f);
        bullet_port::btVector3 tmp_point_on_b_world(0.0f, 0.0f, 0.0f);
        bullet_port::btGjkEpaScratch scratch = {};
        const bool is_valid2 = BulletPortCalcPenDepth(
            box_shape,
            local_trans_a,
            sphere_shape,
            local_trans_b,
            cached_separating_axis,
            tmp_point_on_a_world,
            tmp_point_on_b_world,
            &scratch);
        if (is_valid2) {
            bullet_port::btVector3 tmp_normal_in_b = tmp_point_on_b_world - tmp_point_on_a_world;
            float len_sqr_pen = tmp_normal_in_b.length2();
            if (len_sqr_pen <= 1.1920929e-7f * 1.1920929e-7f) {
                tmp_normal_in_b = cached_separating_axis;
                len_sqr_pen = cached_separating_axis.length2();
            }
            if (len_sqr_pen > 1.1920929e-7f * 1.1920929e-7f) {
                tmp_normal_in_b /= SqrtScalar(len_sqr_pen);
            }
        }
    }
    if (!is_valid) {
        return;
    }

    normal_bt = FixArenaTriangleContactNormalDirectionBullet(
        box_shape, local_trans_a, sphere_shape, local_trans_b, normal_bt);
}

ROCKETGPU_HD bool UseBallCarGjk() {
    return true;
}

ROCKETGPU_HD float BallCarContactBreakingThresholdBT(
    const CarConfig& cfg,
    float ball_radius) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    const float car_bounding_sphere_bt = Length(BulletBoxHalfExtentWithMargin(cfg) * kUUToBT);
    const float car_center_offset_bt = Length(cfg.hitbox_offset) * kUUToBT;
    const float car_angular_motion_disc_bt = car_bounding_sphere_bt + car_center_offset_bt;
    const float sphere_angular_motion_disc_bt = ball_radius * kUUToBT + 0.08f;
    return Min(car_angular_motion_disc_bt, sphere_angular_motion_disc_bt) * 0.02f;
}

struct BallCarRsBroadphaseProbe {
    uint8_t aabb_overlap = 0;
    uint8_t dynamic_neighbor = 0;
    int32_t car_i = -1;
    int32_t car_j = -1;
    int32_t car_k = -1;
    int32_t ball_i = -1;
    int32_t ball_j = -1;
    int32_t ball_k = -1;
    Vec3 car_aabb_min_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 car_aabb_max_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 ball_aabb_min_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 ball_aabb_max_bt = Vec3(0.0f, 0.0f, 0.0f);
};

ROCKETGPU_HD int ClampRsBroadphaseCellIndex(int value, int limit) {
    if (value < 0) {
        return 0;
    }
    if (value >= limit) {
        return limit - 1;
    }
    return value;
}

ROCKETGPU_HD void ComputeRsBroadphaseCellIndices(
    const Vec3& aabb_min_bt,
    int& i,
    int& j,
    int& k) {
    constexpr Vec3 kMinPosBt(-112.0f, -120.0f, 0.0f);
    constexpr float kCellSizeBt = 7.4f;
    constexpr int kCellsX = 31;
    constexpr int kCellsY = 33;
    constexpr int kCellsZ = 6;

    i = ClampRsBroadphaseCellIndex(static_cast<int>((aabb_min_bt.x - kMinPosBt.x) / kCellSizeBt), kCellsX);
    j = ClampRsBroadphaseCellIndex(static_cast<int>((aabb_min_bt.y - kMinPosBt.y) / kCellSizeBt), kCellsY);
    k = ClampRsBroadphaseCellIndex(static_cast<int>((aabb_min_bt.z - kMinPosBt.z) / kCellSizeBt), kCellsZ);
}

ROCKETGPU_HD void ExpandAabbWithPoint(Vec3& out_min, Vec3& out_max, const Vec3& in_min, const Vec3& in_max) {
    out_min.x = Min(out_min.x, in_min.x);
    out_min.y = Min(out_min.y, in_min.y);
    out_min.z = Min(out_min.z, in_min.z);
    out_max.x = Max(out_max.x, in_max.x);
    out_max.y = Max(out_max.y, in_max.y);
    out_max.z = Max(out_max.z, in_max.z);
}

ROCKETGPU_HD void ComputeBulletCarCompoundAabbBt(
    const CarState& car,
    const CarConfig& cfg,
    Vec3& out_min,
    Vec3& out_max) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kContactBreakingThresholdBt = 0.02f;

    Vec3 fwd;
    Vec3 rgt;
    Vec3 upv;
    GetCarBasisForBulletTransform(car, fwd, rgt, upv);

    const Vec3 local_center_bt = cfg.hitbox_offset * kUUToBT;
    const Vec3 local_half_extents_bt = (cfg.hitbox_size * 0.5f) * kUUToBT;
    const Vec3 center_bt =
        car.pos_bt +
        fwd * local_center_bt.x +
        rgt * local_center_bt.y +
        upv * local_center_bt.z;
    Vec3 extent_bt(
        Abs(fwd.x) * local_half_extents_bt.x + Abs(rgt.x) * local_half_extents_bt.y + Abs(upv.x) * local_half_extents_bt.z,
        Abs(fwd.y) * local_half_extents_bt.x + Abs(rgt.y) * local_half_extents_bt.y + Abs(upv.y) * local_half_extents_bt.z,
        Abs(fwd.z) * local_half_extents_bt.x + Abs(rgt.z) * local_half_extents_bt.y + Abs(upv.z) * local_half_extents_bt.z);
    extent_bt += Vec3(kContactBreakingThresholdBt, kContactBreakingThresholdBt, kContactBreakingThresholdBt);
    out_min = center_bt - extent_bt;
    out_max = center_bt + extent_bt;
}

ROCKETGPU_HD void ComputeBulletBallSphereAabbBt(
    const BallState& ball,
    float ball_radius,
    Vec3& out_min,
    Vec3& out_max) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kSphereAabbExtraMarginBt = 0.08f;
    constexpr float kContactBreakingThresholdBt = 0.02f;

    const Vec3 center_bt = ball.pos_bt;
    const float extent_bt = ball_radius * kUUToBT + kSphereAabbExtraMarginBt + kContactBreakingThresholdBt;
    const Vec3 extent(extent_bt, extent_bt, extent_bt);
    out_min = center_bt - extent;
    out_max = center_bt + extent;
}

ROCKETGPU_HD BallCarRsBroadphaseProbe ComputeBulletRsBroadphaseBallCarProbe(
    const BallState& ball,
    const CarState& car,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    float dt) {
    BallCarRsBroadphaseProbe probe;
    ComputeBulletCarCompoundAabbBt(car, cfg, probe.car_aabb_min_bt, probe.car_aabb_max_bt);
    ComputeBulletBallSphereAabbBt(ball, mutator.ball_radius, probe.ball_aabb_min_bt, probe.ball_aabb_max_bt);

    if (dt > 0.0f) {
        CarState predicted_car = car;
        BallState predicted_ball = ball;
        IntegrateCarTransformOnly(predicted_car, dt, false, NativeSseMath(mutator));
        IntegrateBallTransformOnly(predicted_ball, mutator, dt);

        Vec3 predicted_min;
        Vec3 predicted_max;
        ComputeBulletCarCompoundAabbBt(predicted_car, cfg, predicted_min, predicted_max);
        ExpandAabbWithPoint(probe.car_aabb_min_bt, probe.car_aabb_max_bt, predicted_min, predicted_max);
        ComputeBulletBallSphereAabbBt(predicted_ball, mutator.ball_radius, predicted_min, predicted_max);
        ExpandAabbWithPoint(probe.ball_aabb_min_bt, probe.ball_aabb_max_bt, predicted_min, predicted_max);
    }

    probe.aabb_overlap =
        !(probe.car_aabb_min_bt.x > probe.ball_aabb_max_bt.x || probe.car_aabb_max_bt.x < probe.ball_aabb_min_bt.x ||
          probe.car_aabb_min_bt.z > probe.ball_aabb_max_bt.z || probe.car_aabb_max_bt.z < probe.ball_aabb_min_bt.z ||
          probe.car_aabb_min_bt.y > probe.ball_aabb_max_bt.y || probe.car_aabb_max_bt.y < probe.ball_aabb_min_bt.y);

    ComputeRsBroadphaseCellIndices(probe.car_aabb_min_bt, probe.car_i, probe.car_j, probe.car_k);
    ComputeRsBroadphaseCellIndices(probe.ball_aabb_min_bt, probe.ball_i, probe.ball_j, probe.ball_k);
    const int di = probe.car_i > probe.ball_i ? probe.car_i - probe.ball_i : probe.ball_i - probe.car_i;
    const int dj = probe.car_j > probe.ball_j ? probe.car_j - probe.ball_j : probe.ball_j - probe.car_j;
    const int dk = probe.car_k > probe.ball_k ? probe.car_k - probe.ball_k : probe.ball_k - probe.car_k;
    probe.dynamic_neighbor = di <= 1 && dj <= 1 && dk <= 1;
    return probe;
}

ROCKETGPU_HD bool BulletRsBroadphaseAllowsBallCarPair(
    const BallState& ball,
    const CarState& car,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    float dt,
    int env_idx = -1,
    int car_idx = -1,
    int phase = 0) {
    if (dt <= 0.0f || env_idx < 0) {
        return true;
    }

    const BallCarRsBroadphaseProbe probe =
        ComputeBulletRsBroadphaseBallCarProbe(ball, car, cfg, mutator, dt);
    const bool allowed = probe.aabb_overlap && probe.dynamic_neighbor;
#if !defined(__CUDA_ARCH__)
#else
    (void)env_idx;
    (void)car_idx;
    (void)phase;
#endif
    return allowed;
}

ROCKETGPU_HD bool ComputeBallCarContactGeometrySphereBox(
    const BallState& ball,
    const CarState& car,
    const CarConfig& cfg,
    float ball_radius,
    float max_contact_distance_bt,
    Vec3& out_normal,
    float& out_penetration,
    Vec3& out_rel_ball_contact,
    Vec3& out_rel_car_contact) {
    constexpr float kUUToBT_BallCar = 1.0f / 50.0f;
    constexpr float kBTToUU_BallCar = 50.0f;
    Vec3 car_forward, car_right, car_up;
    CarBasisColumns(car, car_forward, car_right, car_up);
    const Vec3 ball_pos_bt = ball.pos_bt;
    const Vec3 car_pos_bt = car.pos_bt;
    const Vec3 hitbox_offset_bt = cfg.hitbox_offset * kUUToBT_BallCar;
    const Vec3 box_origin_bt = car_pos_bt + RotateViaBasis(car_forward, car_right, car_up, hitbox_offset_bt);
    const Vec3 sphere_rel_delta_bt = ball_pos_bt - box_origin_bt;
    const Vec3 sphere_rel_pos_bt(
        Dot(car_forward, sphere_rel_delta_bt),
        Dot(car_right, sphere_rel_delta_bt),
        Dot(car_up, sphere_rel_delta_bt));

    const float ball_radius_bt = ball_radius * kUUToBT_BallCar;
    const float box_margin_bt = BallCarShapeMargin(cfg) * kUUToBT_BallCar;
    const Vec3 raw_half_extent_bt = (cfg.hitbox_size * 0.5f) * kUUToBT_BallCar;
    const Vec3 implicit_inset_bt = BallCarHitboxInsetBt(cfg);
    const Vec3 box_half_extent_bt(
        raw_half_extent_bt.x - implicit_inset_bt.x,
        raw_half_extent_bt.y - implicit_inset_bt.y,
        raw_half_extent_bt.z - implicit_inset_bt.z);

    Vec3 closest_local_bt = ClampVec(sphere_rel_pos_bt, -box_half_extent_bt, box_half_extent_bt);
    Vec3 normal_local_bt = sphere_rel_pos_bt - closest_local_bt;
    const float intersection_dist_bt = ball_radius_bt + box_margin_bt;
    const float contact_dist_bt = intersection_dist_bt + max_contact_distance_bt;
    const float dist_sq_bt = LengthSq(normal_local_bt);
    if (dist_sq_bt > (contact_dist_bt * contact_dist_bt)) {
        return false;
    }

    float distance_bt = 0.0f;
    if (dist_sq_bt <= 1.1920929e-7f) {
        float face_dist = box_half_extent_bt.x - sphere_rel_pos_bt.x;
        float min_dist = face_dist;
        closest_local_bt = sphere_rel_pos_bt;
        closest_local_bt.x = box_half_extent_bt.x;
        normal_local_bt = Vec3(1.0f, 0.0f, 0.0f);

        face_dist = box_half_extent_bt.x + sphere_rel_pos_bt.x;
        if (face_dist < min_dist) {
            min_dist = face_dist;
            closest_local_bt = sphere_rel_pos_bt;
            closest_local_bt.x = -box_half_extent_bt.x;
            normal_local_bt = Vec3(-1.0f, 0.0f, 0.0f);
        }

        face_dist = box_half_extent_bt.y - sphere_rel_pos_bt.y;
        if (face_dist < min_dist) {
            min_dist = face_dist;
            closest_local_bt = sphere_rel_pos_bt;
            closest_local_bt.y = box_half_extent_bt.y;
            normal_local_bt = Vec3(0.0f, 1.0f, 0.0f);
        }

        face_dist = box_half_extent_bt.y + sphere_rel_pos_bt.y;
        if (face_dist < min_dist) {
            min_dist = face_dist;
            closest_local_bt = sphere_rel_pos_bt;
            closest_local_bt.y = -box_half_extent_bt.y;
            normal_local_bt = Vec3(0.0f, -1.0f, 0.0f);
        }

        face_dist = box_half_extent_bt.z - sphere_rel_pos_bt.z;
        if (face_dist < min_dist) {
            min_dist = face_dist;
            closest_local_bt = sphere_rel_pos_bt;
            closest_local_bt.z = box_half_extent_bt.z;
            normal_local_bt = Vec3(0.0f, 0.0f, 1.0f);
        }

        face_dist = box_half_extent_bt.z + sphere_rel_pos_bt.z;
        if (face_dist < min_dist) {
            min_dist = face_dist;
            closest_local_bt = sphere_rel_pos_bt;
            closest_local_bt.z = -box_half_extent_bt.z;
            normal_local_bt = Vec3(0.0f, 0.0f, -1.0f);
        }

        distance_bt = -min_dist;
    } else {
        distance_bt = SqrtScalar(dist_sq_bt);

        normal_local_bt *= (1.0f / distance_bt);
    }

    const Vec3 point_on_box_local_bt = closest_local_bt + normal_local_bt * box_margin_bt;
    const float penetration_depth_bt = distance_bt - intersection_dist_bt;
    out_penetration = -penetration_depth_bt * kBTToUU_BallCar;

    const Vec3 point_on_box_world_bt = box_origin_bt + RotateViaBasis(car_forward, car_right, car_up, point_on_box_local_bt);
    out_normal = RotateViaBasis(car_forward, car_right, car_up, normal_local_bt);
    out_rel_ball_contact = -out_normal * ball_radius;
    out_rel_car_contact = point_on_box_world_bt * kBTToUU_BallCar - car.pos;
    return true;
}

ROCKETGPU_HD bool ComputeBallCarContactGeometry(
    const BallState& ball,
    const CarState& car,
    const CarConfig& cfg,
    float ball_radius,
    Vec3& out_normal,
    float& out_penetration,
    Vec3& out_rel_ball_contact,
    Vec3& out_rel_car_contact,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch = nullptr,
    BallCarContactGeometryDebug* debug = nullptr,
    Vec3* out_bt_rel_ball_contact_for_solver = nullptr,
    Vec3* out_bt_rel_car_contact_for_solver = nullptr) {
    if (TraceBallCarGjkShadowEnabled()) {
        TraceBallCarContactGeometryGjkShadow(ball, car, cfg, ball_radius);
    }
    if (TraceBallCarSphereBoxShadowEnabled()) {
        Vec3 sphere_box_normal(0.0f, 0.0f, 0.0f);
        float sphere_box_penetration = 0.0f;
        Vec3 sphere_box_rel_ball_contact(0.0f, 0.0f, 0.0f);
        Vec3 sphere_box_rel_car_contact(0.0f, 0.0f, 0.0f);
        (void)ComputeBallCarContactGeometrySphereBox(
            ball,
            car,
            cfg,
            ball_radius,
            0.0f,
            sphere_box_normal,
            sphere_box_penetration,
            sphere_box_rel_ball_contact,
            sphere_box_rel_car_contact);
    }

    if (UseBallCarGjk() && ComputeBallCarContactGeometryGjk(
            ball,
            car,
            cfg,
            ball_radius,
            out_normal,
            out_penetration,
            out_rel_ball_contact,
            out_rel_car_contact,
            gjk_epa_scratch,
            debug,
            out_bt_rel_ball_contact_for_solver,
            out_bt_rel_car_contact_for_solver)) {
        return true;
    }

    return ComputeBallCarContactGeometrySphereBox(
        ball,
        car,
        cfg,
        ball_radius,
        0.0f,
        out_normal,
        out_penetration,
        out_rel_ball_contact,
        out_rel_car_contact);
}

ROCKETGPU_HD void RecordBallCarCollisionInfoIfOverlapping(
    const BallState& ball,
    const CarState& car,
    uint32_t car_id,
    CarState& car_mut,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    uint64_t env_tick,
    BallState& ball_mut,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch = nullptr,
    float broadphase_dt = -1.0f,
    int env_idx = -1,
    int other_local_idx = -1,
    int8_t car_team = -1,
    const Vec3* cal_car_start_vel = nullptr,
    const Vec3* cal_ball_start_vel = nullptr) {
    if (car.is_demoed || car.just_respawned) {
        return;
    }
    if (!BulletRsBroadphaseAllowsBallCarPair(
            ball,
            car,
            cfg,
            mutator,
            broadphase_dt,
            env_idx,
            other_local_idx,
            0)) {
        return;
    }

    Vec3 normal(1.0f, 0.0f, 0.0f);
    float penetration = 0.0f;
    Vec3 rel_ball_contact(0.0f, 0.0f, 0.0f);
    Vec3 rel_car_contact(0.0f, 0.0f, 0.0f);
    if (!ComputeBallCarContactGeometry(ball, car, cfg, mutator.ball_radius, normal, penetration, rel_ball_contact, rel_car_contact, gjk_epa_scratch) &&
        !ComputeBallCarContactGeometrySphereBox(
            ball,
            car,
            cfg,
            mutator.ball_radius,
            BallCarContactBreakingThresholdBT(cfg, mutator.ball_radius),
            normal,
            penetration,
            rel_ball_contact,
            rel_car_contact)) {
        return;
    }

    const bool contact_started = !car_mut.ball_contact_last_tick && !car_mut.ball_contact_this_tick;
    if (contact_started) car_mut.RecordBallContactStart();
    car_mut.ball_contact_this_tick = 1;
    if (contact_started && mutator.physics_events.windows) {
        PhysicsEvent event;
        event.type = PhysicsEventType::BallTouchStart;
        event.car_id = car_id;
        event.team = car_team;
        event.car_on_ground = car.is_on_ground;
        event.position = car.pos;
        event.car_velocity = car.vel;
        event.ball_position = ball.pos;
        event.ball_velocity = ball.vel;
        mutator.physics_events.EmitForCar(car_id, event);
    }
    const uint8_t had_prior_ball_hit = car_mut.ball_hit_info.is_valid;
    const uint64_t prev_extra_impulse_tick = car_mut.ball_hit_info.tick_count_when_extra_impulse_applied;
    car_mut.ball_hit_info.is_valid = 1;
    car_mut.ball_hit_info.tick_count_when_hit = env_tick;
    car_mut.ball_hit_info.hit_car_id = car_id;
    car_mut.ball_hit_info.relative_pos_on_ball = InverseRotate(ball.rot, rel_ball_contact);
    car_mut.ball_hit_info.ball_pos = ball.pos;
    car_mut.ball_hit_info.extra_hit_vel = Vec3(0.0f, 0.0f, 0.0f);
    if (mutator.game_mode == static_cast<uint8_t>(Mode::Heatseeker) && car_team >= 0) {
        HeatseekerOnBallHit(ball_mut, static_cast<int>(car_team));
    }

    const bool can_apply_extra = (!had_prior_ball_hit) ||
        (prev_extra_impulse_tick > env_tick) ||
        ((prev_extra_impulse_tick + 1ull) < env_tick);

    if (can_apply_extra) {
        const Vec3 rel_pos_for_extra = ball.pos - car.pos;

        const Vec3 rel_vel_for_extra =
            (cal_ball_start_vel ? *cal_ball_start_vel : ball.vel) - (cal_car_start_vel ? *cal_car_start_vel : car.vel);
        const float rel_speed = Min(Length(rel_vel_for_extra), mutator.ball_car_extra_impulse_max_delta_vel);
        if (rel_speed > 0.0f) {
            Vec3 ei_fwd(1.0f, 0.0f, 0.0f);
            Vec3 ei_rgt(0.0f, 1.0f, 0.0f);
            Vec3 ei_up(0.0f, 0.0f, 1.0f);
            CarBasisColumns(car, ei_fwd, ei_rgt, ei_up);
            (void)ei_rgt; (void)ei_up;
            const Vec3 car_forward = ei_fwd;

            Vec3 hit_dir = RocketSimVecNormalized(
                Vec3(
                    rel_pos_for_extra.x,
                    rel_pos_for_extra.y,
                    rel_pos_for_extra.z * mutator.ball_car_extra_impulse_z_scale));
            const Vec3 forward_dir_adjustment =
                (car_forward * Dot(hit_dir, car_forward)) *
                (1.0f - mutator.ball_car_extra_impulse_forward_scale);
            hit_dir = RocketSimVecNormalized(hit_dir - forward_dir_adjustment);
            const Vec3 added_vel =
                ((hit_dir * rel_speed) * EvalBallCarExtraImpulseFactor(rel_speed)) *
                mutator.ball_hit_extra_force_scale;
            ball_mut.vel_impulse_cache += added_vel;
            ball_mut.vel_impulse_cache_bt += added_vel * (1.0f / 50.0f);
            ball_mut.tick_count_when_extra_impulse_applied = env_tick;
            car_mut.ball_hit_info.tick_count_when_extra_impulse_applied = env_tick;
            car_mut.ball_hit_info.extra_hit_vel = added_vel;
        }
    }
}

// cal_ball_car_manifold: one retained manifold point handed to AppendBallCarContactRows in place of the
// narrowphase (normal car toward ball; bt offsets from the body origins; penetration_uu = -distance * 50).
struct BallCarContactOverride {
    Vec3 normal = Vec3(0.0f, 0.0f, 1.0f);
    float penetration_uu = 0.0f;
    Vec3 bt_rel_ball = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 bt_rel_car = Vec3(0.0f, 0.0f, 0.0f);
};

ROCKETGPU_HD void AppendBallCarContactRows(
    BallState& ball,
    CarState& car,
    uint32_t car_id,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    uint64_t env_tick,
    const Vec3& ball_external_linear_delta,
    const Vec3& external_linear_delta,
    const Vec3& external_angular_delta,
    EnvStridedArray<BallVelocityContactRow> normal_rows,
    int& normal_row_count,
    EnvStridedArray<BallFrictionContactRow> friction_rows,
    int& friction_row_count,
    EnvStridedArray<BallSplitContactRow> split_rows,
    int& split_row_count,
    int32_t other_local_idx,
    float split_dt,
    int env_idx = -1,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch = nullptr,
    bool allow_midtick_demo_contact = false,
    const Vec3* bt_ball_linear_velocity_override = nullptr,
    int8_t car_team = -1,
    const BallCarContactOverride* contact_override = nullptr) {
    if ((!allow_midtick_demo_contact && car.is_demoed) || car.just_respawned) {
        return;
    }

    if (!contact_override && !BulletRsBroadphaseAllowsBallCarPair(
            ball,
            car,
            cfg,
            mutator,
            split_dt,
            env_idx,
            other_local_idx,
            1)) {
         return;
    }

    Vec3 normal(1.0f, 0.0f, 0.0f);
    float penetration = 0.0f;
    Vec3 rel_ball_contact(0.0f, 0.0f, 0.0f);
    Vec3 rel_car_contact(0.0f, 0.0f, 0.0f);
    BallCarContactGeometryDebug contact_debug = {};
    Vec3 bt_rel_ball_contact_for_solver_exact(0.0f, 0.0f, 0.0f);
    Vec3 bt_rel_car_contact_for_solver_exact(0.0f, 0.0f, 0.0f);
    if (contact_override) {
        normal = contact_override->normal;
        penetration = contact_override->penetration_uu;
        rel_ball_contact = contact_override->bt_rel_ball * 50.0f;
        rel_car_contact = contact_override->bt_rel_car * 50.0f;
    }
    const bool has_contact = contact_override != nullptr ||
        ComputeBallCarContactGeometry(
            ball,
            car,
            cfg,
            mutator.ball_radius,
            normal,
            penetration,
            rel_ball_contact,
            rel_car_contact,
            gjk_epa_scratch,
            &contact_debug,
            &bt_rel_ball_contact_for_solver_exact,
            &bt_rel_car_contact_for_solver_exact) ||
        ComputeBallCarContactGeometrySphereBox(
            ball,
            car,
            cfg,
            mutator.ball_radius,
            BallCarContactBreakingThresholdBT(cfg, mutator.ball_radius),
            normal,
            penetration,
            rel_ball_contact,
            rel_car_contact);

    if (!has_contact) {
        return;
    }

    constexpr float kBtPerUuBallCar = 1.0f / 50.0f;
    const Vec3 bt_rel_ball_contact_for_solver = contact_override ? contact_override->bt_rel_ball :
        (contact_debug.valid && contact_debug.used_gjk)
            ? bt_rel_ball_contact_for_solver_exact
            : UuVecToBtDiv(rel_ball_contact);

    const Vec3 bc_car_pos_bt = car.pos_bt;
    const Vec3 bc_direct_contact_car_world_bt =
        bc_car_pos_bt + rel_car_contact * kBtPerUuBallCar;
    const Vec3 bc_car_local_point_bt =
        CarBodyLocalPointBt(car, bc_direct_contact_car_world_bt);
    const Vec3 bt_rel_car_contact_for_solver = contact_override ? contact_override->bt_rel_car :
        (contact_debug.valid && contact_debug.used_gjk)
            ? bt_rel_car_contact_for_solver_exact
            : (CarBodyWorldPointBt(car, bc_car_local_point_bt) - bc_car_pos_bt);

    const float bt_split_penetration_for_solver = Dot(
        (bc_car_pos_bt + bt_rel_car_contact_for_solver) - (ball.pos_bt + bt_rel_ball_contact_for_solver),
        normal);

#if !defined(__CUDA_ARCH__)
    constexpr float kBtPerUuTrace = kBtPerUuBallCar;
    const Vec3 ball_pos_bt = ball.pos_bt;
    const Vec3 car_pos_bt = car.pos_bt;
    const bullet_port::btTransform ball_transform =
        MakeBulletPortBallWorldTransform(ball, ball_pos_bt);
    const bullet_port::btVector3 direct_rel_car_contact_bt =
        ToBulletPortVec3(rel_car_contact * kBtPerUuTrace);
    const bullet_port::btVector3 direct_contact_point_car_world_bt =
        ToBulletPortVec3(car_pos_bt) + direct_rel_car_contact_bt;
    const Vec3 car_local_point_bt =
        CarBodyLocalPointBt(car, FromBulletPortVec3(direct_contact_point_car_world_bt));
    const Vec3 refreshed_rel_car_contact_bt =
        CarBodyWorldPointBt(car, car_local_point_bt) - car_pos_bt;
#endif

    AppendBallCarSplitRow(
        split_rows,
        split_row_count,
        penetration,
        normal,
        rel_ball_contact,
        rel_car_contact,
        other_local_idx,
        car,
        cfg,
        mutator,
        split_dt,
        &bt_rel_ball_contact_for_solver,
        &bt_rel_car_contact_for_solver,
        &bt_split_penetration_for_solver,
        &ball);

    const Vec3 restitution_ball_linear = ball.vel;
    const Vec3 restitution_car_linear = car.vel;
    const Vec3 ball_angular = ball.ang_vel;
    const Vec3 car_angular = car.ang_vel;
    const Vec3 rhs_ball_linear = ball.vel + ball_external_linear_delta;
    const Vec3 rhs_car_linear = car.vel + external_linear_delta;
    const Vec3 rhs_car_angular = car.ang_vel + external_angular_delta;
    constexpr float kBtPerUuTangent = 1.0f / 50.0f;
    const Vec3 bt_ball_linear_for_tangent = bt_ball_linear_velocity_override
        ? *bt_ball_linear_velocity_override
        : ball.vel_bt;
    const Vec3 bt_rhs_ball_linear_for_tangent =
        bt_ball_linear_for_tangent + UuVecToBtDiv(ball_external_linear_delta);

    const Vec3 bt_rhs_car_linear_for_tangent =
        car.vel_bt + (car.ext_force_impulse_bt_valid
                          ? car.ext_force_impulse_bt
                          : UuVecToBtDiv(external_linear_delta));
    const Vec3 rhs_car_angular_bt_for_tangent =
        car.ang_vel + (car.ext_force_impulse_bt_valid ? car.ext_torque_impulse : external_angular_delta);
    const Vec3 bt_rel_car_contact_for_tangent = bt_rel_car_contact_for_solver;
    const Vec3 tangent_basis_rel_vel =
        (bt_rhs_ball_linear_for_tangent + Cross(ball_angular, bt_rel_ball_contact_for_solver)) -
        (bt_rhs_car_linear_for_tangent + Cross(rhs_car_angular_bt_for_tangent, bt_rel_car_contact_for_tangent));
    const Vec3 friction_base_rel_vel = tangent_basis_rel_vel;
    const Vec3 ball_normal_component = ApplyInverseBallInertiaWorld(
        mutator.ball_mass,
        mutator.ball_radius,
        Cross(rel_ball_contact, normal));
    const Vec3 bt_ball_normal_component = ApplyInverseBallInertiaWorldBtMutator(ball, mutator,
        Cross(bt_rel_ball_contact_for_solver, normal));
    const Vec3 bt_car_normal_component = ApplyInverseBodyInertiaWorldBt(
        car,
        cfg,
        mutator.car_mass,
        Cross(bt_rel_car_contact_for_tangent, -normal));
    const Vec3 car_normal_component = bt_car_normal_component * kBtPerUuTangent;

    const Vec3& bt_tangent_basis_rel_vel = tangent_basis_rel_vel;
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
    const Vec3 bt_ball_tangent_component = ApplyInverseBallInertiaWorldBtMutator(ball, mutator,
        Cross(bt_rel_ball_contact_for_solver, tangent));
    const Vec3 bt_car_tangent_component = ApplyInverseBodyInertiaWorldBt(
        car,
        cfg,
        mutator.car_mass,
        Cross(bt_rel_car_contact_for_tangent, -tangent));
    const Vec3 car_tangent_component = bt_car_tangent_component * kBtPerUuTangent;
    AppendBallVelocityContactRows(
        normal_rows,
        normal_row_count,
        friction_rows,
        friction_row_count,
        normal,
        CarBallCollisionRestitution(),
        CarBallCollisionFriction(),
        1.0f / mutator.ball_mass,
        rel_ball_contact,
        ball_normal_component,
        other_local_idx,
        1.0f / mutator.car_mass,
        rel_car_contact,
        car_normal_component,
        ball_tangent_component,
        car_tangent_component,
        restitution_ball_linear,
        ball_angular,
        restitution_car_linear,
        car_angular,
        rhs_ball_linear,
        ball_angular,
        rhs_car_linear,
        rhs_car_angular,
        ball_external_linear_delta,
        Vec3(0.0f, 0.0f, 0.0f),
        external_linear_delta,
        external_angular_delta,
        tangent_basis_rel_vel,
        friction_base_rel_vel,
        &bt_rel_ball_contact_for_solver,

        &bt_rel_car_contact_for_solver,
        bt_ball_linear_velocity_override,
        &bt_ball_normal_component,
        &bt_car_normal_component,
        &bt_ball_tangent_component,
        &bt_car_tangent_component,
        &tangent_basis_rel_vel,
        &car.vel_bt,
        car.ext_force_impulse_bt_valid ? &car.ext_force_impulse_bt : nullptr,

        &car.ang_vel,

        car.ext_force_impulse_bt_valid ? &car.ext_torque_impulse : nullptr);
    if (contact_debug.valid && normal_row_count > 0) {
    }
}

// cal_ball_car_manifold helpers (v3 persistent_manifold.rs on the ball-car pair; body0 = ball, body1 = car).
ROCKETGPU_HD Vec3 BallCarRetainedBallWorldPointBt(const BallState& ball, const Vec3& local_bt) {
    Vec3 bx, by, bz;
    GetBallBasisColumns(ball, bx, by, bz);
    const Vec3 rotated(
        bx.x * local_bt.x + by.x * local_bt.y + bz.x * local_bt.z,
        bx.y * local_bt.x + by.y * local_bt.y + bz.y * local_bt.z,
        bx.z * local_bt.x + by.z * local_bt.y + bz.z * local_bt.z);
    return rotated + ball.pos_bt;
}

ROCKETGPU_HD Vec3 BallCarRetainedBallLocalPointBt(const BallState& ball, const Vec3& world_bt) {
    Vec3 bx, by, bz;
    GetBallBasisColumns(ball, bx, by, bz);
    const Vec3 rel = world_bt - ball.pos_bt;
    return Vec3(Dot(rel, bx), Dot(rel, by), Dot(rel, bz));
}

// v3 PersistentManifold::refresh_contact_points.
ROCKETGPU_HD void RefreshBallCarRetainedManifold(
    BallCarRetainedManifold& manifold, const BallState& ball, const CarState& car, float threshold_bt) {
    const int count = manifold.count;
    for (int i = 0; i < count; ++i) {
        BallCarRetainedPoint& point = manifold.points[i];
        point.pos_a_bt = BallCarRetainedBallWorldPointBt(ball, point.local_a_bt);
        point.pos_b_bt = CarBodyWorldPointBt(car, point.local_b_bt);
        point.distance_bt = Dot(point.pos_a_bt - point.pos_b_bt, point.normal);
    }
    const float threshold_sq = threshold_bt * threshold_bt;
    int kept = count;
    for (int i = count - 1; i >= 0; --i) {
        const BallCarRetainedPoint point = manifold.points[i];
        bool remove = point.distance_bt > threshold_bt;
        if (!remove) {
            const Vec3 projected = point.pos_a_bt - point.normal * point.distance_bt;
            const Vec3 diff = point.pos_b_bt - projected;
            remove = Dot(diff, diff) > threshold_sq;
        }
        if (remove) {
            manifold.points[i] = manifold.points[kept - 1];
            --kept;
        }
    }
    manifold.count = static_cast<uint8_t>(kept);
}

ROCKETGPU_HD float BallCarRetainedRes(const Vec3& local, const Vec3& p1, const Vec3& p2, const Vec3& p3) {
    const Vec3 c = Cross(local - p1, p2 - p3);
    return Dot(c, c);
}

// v3 PersistentManifold::add_contact_without_callback (get_cache_entry on local_point_a, replace,
// add_manifold_point, sort_cached_points keeping the deepest point).
ROCKETGPU_HD void AddBallCarRetainedPoint(
    BallCarRetainedManifold& manifold, const BallCarRetainedPoint& point, float threshold_bt) {
    const int count = manifold.count;
    float shortest = threshold_bt * threshold_bt;
    int nearest = -1;
    for (int i = 0; i < count; ++i) {
        const Vec3 d = manifold.points[i].local_a_bt - point.local_a_bt;
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
    const Vec3& l = point.local_a_bt;
    const Vec3& p0 = manifold.points[0].local_a_bt;
    const Vec3& p1 = manifold.points[1].local_a_bt;
    const Vec3& p2 = manifold.points[2].local_a_bt;
    const Vec3& p3 = manifold.points[3].local_a_bt;
    const float res0 = max_pen_idx == 0 ? 0.0f : BallCarRetainedRes(l, p1, p3, p2);
    const float res1 = max_pen_idx == 1 ? 0.0f : BallCarRetainedRes(l, p0, p3, p2);
    const float res2 = max_pen_idx == 2 ? 0.0f : BallCarRetainedRes(l, p0, p3, p1);
    const float res3 = max_pen_idx == 3 ? 0.0f : BallCarRetainedRes(l, p0, p2, p1);
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

// cal_ball_car_manifold: v3 near_callback_with_bounds for the ball-car pair. The car box child AABB must overlap
// the ball AABB (radius + SPHERE_RADIUS_MARGIN) or the manifold is released; the fresh sphere-OBB point (same
// narrowphase as AppendBallCarContactRows) merges into the refreshed retained manifold, and every retained point,
// positive distances included, becomes a ball-car solver row.
ROCKETGPU_HD void AppendBallCarContactRowsRetained(
    BallState& ball,
    CarState& car,
    uint32_t car_id,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    uint64_t env_tick,
    const Vec3& ball_external_linear_delta,
    const Vec3& external_linear_delta,
    const Vec3& external_angular_delta,
    EnvStridedArray<BallVelocityContactRow> normal_rows,
    int& normal_row_count,
    EnvStridedArray<BallFrictionContactRow> friction_rows,
    int& friction_row_count,
    EnvStridedArray<BallSplitContactRow> split_rows,
    int& split_row_count,
    int32_t other_local_idx,
    float split_dt,
    int env_idx,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch,
    bool allow_midtick_demo_contact,
    const Vec3* bt_ball_linear_velocity_override,
    int8_t car_team,
    BallCarRetainedManifold& retained) {
    if ((!allow_midtick_demo_contact && car.is_demoed) || car.just_respawned) {
        retained.count = 0;
        return;
    }
    constexpr float kUUToBT = 1.0f / 50.0f;
    {
        Vec3 ax(1.0f, 0.0f, 0.0f), ay(0.0f, 1.0f, 0.0f), az(0.0f, 0.0f, 1.0f);
        CarBasisColumns(car, ax, ay, az);
        const Vec3 off = cfg.hitbox_offset * kUUToBT;
        const Vec3 center = car.pos_bt + ax * off.x + ay * off.y + az * off.z;
        const Vec3 half = BulletBoxHalfExtentWithMarginBt(cfg);
        const Vec3 ext(
            Abs(ax.x) * half.x + Abs(ay.x) * half.y + Abs(az.x) * half.z,
            Abs(ax.y) * half.x + Abs(ay.y) * half.y + Abs(az.y) * half.z,
            Abs(ax.z) * half.x + Abs(ay.z) * half.y + Abs(az.z) * half.z);
        const float r = mutator.ball_radius * kUUToBT + 0.08f;
        const Vec3 bmin = ball.pos_bt - Vec3(r, r, r);
        const Vec3 bmax = ball.pos_bt + Vec3(r, r, r);
        const Vec3 cmin = center - ext;
        const Vec3 cmax = center + ext;
        const bool overlap =
            !(cmin.x > bmax.x || cmax.x < bmin.x || cmin.y > bmax.y || cmax.y < bmin.y ||
              cmin.z > bmax.z || cmax.z < bmin.z);
        if (!overlap) {
            retained.count = 0;
            return;
        }
    }
    const float threshold_bt = BallCarContactBreakingThresholdBT(cfg, mutator.ball_radius);

    Vec3 normal(1.0f, 0.0f, 0.0f);
    float penetration = 0.0f;
    Vec3 rel_ball_contact(0.0f, 0.0f, 0.0f);
    Vec3 rel_car_contact(0.0f, 0.0f, 0.0f);
    BallCarContactGeometryDebug contact_debug = {};
    Vec3 bt_rel_ball_exact(0.0f, 0.0f, 0.0f);
    Vec3 bt_rel_car_exact(0.0f, 0.0f, 0.0f);
    const bool has_fresh =
        ComputeBallCarContactGeometry(
            ball, car, cfg, mutator.ball_radius, normal, penetration, rel_ball_contact, rel_car_contact,
            gjk_epa_scratch, &contact_debug, &bt_rel_ball_exact, &bt_rel_car_exact) ||
        ComputeBallCarContactGeometrySphereBox(
            ball, car, cfg, mutator.ball_radius, BallCarContactBreakingThresholdBT(cfg, mutator.ball_radius),
            normal, penetration, rel_ball_contact, rel_car_contact);

    RefreshBallCarRetainedManifold(retained, ball, car, threshold_bt);
    if (has_fresh) {
        const bool exact = contact_debug.valid && contact_debug.used_gjk;
        const Vec3 bt_rel_ball = exact ? bt_rel_ball_exact : UuVecToBtDiv(rel_ball_contact);
        const Vec3 bt_rel_car = exact
            ? bt_rel_car_exact
            : (CarBodyWorldPointBt(car, CarBodyLocalPointBt(car, car.pos_bt + rel_car_contact * kUUToBT)) - car.pos_bt);
        const Vec3 pos_a = ball.pos_bt + bt_rel_ball;
        const Vec3 pos_b = car.pos_bt + bt_rel_car;
        const float depth = Dot(pos_a - pos_b, normal);
        if (depth <= threshold_bt) {
            BallCarRetainedManifold fresh;
            BallCarRetainedPoint& point = fresh.points[0];
            point.normal = normal;
            point.local_a_bt = BallCarRetainedBallLocalPointBt(ball, pos_a);
            point.local_b_bt = CarBodyLocalPointBt(car, pos_b);
            point.distance_bt = depth;
            fresh.count = 1;
            RefreshBallCarRetainedManifold(fresh, ball, car, threshold_bt);
            for (int i = 0; i < fresh.count; ++i) {
                AddBallCarRetainedPoint(retained, fresh.points[i], threshold_bt);
            }
        }
    }
    for (int i = 0; i < retained.count; ++i) {
        const BallCarRetainedPoint& point = retained.points[i];
        BallCarContactOverride ov;
        ov.normal = point.normal;
        ov.penetration_uu = -point.distance_bt * 50.0f;
        ov.bt_rel_ball = point.pos_a_bt - ball.pos_bt;
        ov.bt_rel_car = point.pos_b_bt - car.pos_bt;
        AppendBallCarContactRows(
            ball, car, car_id, cfg, mutator, env_tick, ball_external_linear_delta, external_linear_delta,
            external_angular_delta, normal_rows, normal_row_count, friction_rows, friction_row_count, split_rows,
            split_row_count, other_local_idx, split_dt, env_idx, gjk_epa_scratch, allow_midtick_demo_contact,
            bt_ball_linear_velocity_override, car_team, &ov);
    }
}

struct ObbData {
    Vec3 center;
    Vec3 axis[3];
    Vec3 half_extent;
};

enum ObbContactFeatureKind : uint8_t {
    OBB_CONTACT_FEATURE_FACE_A = 0,
    OBB_CONTACT_FEATURE_FACE_B = 1,
    OBB_CONTACT_FEATURE_EDGE_EDGE = 2,
};

struct ObbContactFeature {
    uint8_t kind = OBB_CONTACT_FEATURE_EDGE_EDGE;
    uint8_t axis_a = 0;
    uint8_t axis_b = 0;
};

struct FaceClipPoint {
    float u = 0.0f;
    float v = 0.0f;
    float w = 0.0f;
};

constexpr int kMaxApproxCarCarContactPoints = 4;
constexpr float kLargeFaceClipFloat = 1.0e30f;

ROCKETGPU_HD float ObbHalfExtentComponent(const ObbData& obb, int idx) {
    return (&obb.half_extent.x)[idx];
}

ROCKETGPU_HD ObbData MakeCarObbBulletTransformBt(const CarState& car, const CarConfig& cfg) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    ObbData obb;
    CarBasisColumns(car, obb.axis[0], obb.axis[1], obb.axis[2]);
    const Vec3 offset_bt = cfg.hitbox_offset * kUUToBT;
    obb.center = car.pos_bt + RotateViaBasis(obb.axis[0], obb.axis[1], obb.axis[2], offset_bt);

    obb.half_extent = BulletBoxHalfExtentWithMarginBt(cfg);
    return obb;
}

ROCKETGPU_HD int IntersectRectQuad2(const float half_extents[2], const float quad[8], float out[16]) {
    int in_count = 4;
    int out_count = 0;
    float buffer[16];
    const float* in_poly = quad;
    float* out_poly = out;
    for (int axis = 0; axis <= 1; ++axis) {
        for (int sign = -1; sign <= 1; sign += 2) {
            const float* in_ptr = in_poly;
            float* out_ptr = out_poly;
            out_count = 0;
            for (int i = in_count; i > 0; --i) {
                const bool inside_cur = sign * in_ptr[axis] < half_extents[axis];
                if (inside_cur) {
                    out_ptr[0] = in_ptr[0];
                    out_ptr[1] = in_ptr[1];
                    out_ptr += 2;
                    ++out_count;
                    if (out_count & 8) {
                        in_poly = out_poly;
                        goto done_face_clip;
                    }
                }
                const float* next_ptr = (i > 1) ? (in_ptr + 2) : in_poly;
                const bool inside_next = sign * next_ptr[axis] < half_extents[axis];
                if (inside_cur != inside_next) {
                    out_ptr[1 - axis] =
                        in_ptr[1 - axis] +
                        (next_ptr[1 - axis] - in_ptr[1 - axis]) /
                            (next_ptr[axis] - in_ptr[axis]) *
                            (sign * half_extents[axis] - in_ptr[axis]);
                    out_ptr[axis] = sign * half_extents[axis];
                    out_ptr += 2;
                    ++out_count;
                    if (out_count & 8) {
                        in_poly = out_poly;
                        goto done_face_clip;
                    }
                }
                in_ptr += 2;
            }
            in_poly = out_poly;
            out_poly = (in_poly == out) ? buffer : out;
            in_count = out_count;
        }
    }
done_face_clip:
    if (in_poly != out) {
        memcpy(out, in_poly, static_cast<size_t>(out_count) * 2 * sizeof(float));
    }
    return out_count;
}

ROCKETGPU_HD void CullFacePoints2(int count, float points[16], int max_points, int deepest_idx, int out_indices[8]) {
    float cx = 0.0f;
    float cy = 0.0f;
    if (count == 1) {
        cx = points[0];
        cy = points[1];
    } else if (count == 2) {
        cx = 0.5f * (points[0] + points[2]);
        cy = 0.5f * (points[1] + points[3]);
    } else {
        float area_sum = 0.0f;
        cx = 0.0f;
        cy = 0.0f;
        float q = 0.0f;
        for (int i = 0; i < count - 1; ++i) {
            q = points[i * 2] * points[i * 2 + 3] - points[i * 2 + 2] * points[i * 2 + 1];
            area_sum += q;
            cx += q * (points[i * 2] + points[i * 2 + 2]);
            cy += q * (points[i * 2 + 1] + points[i * 2 + 3]);
        }
        q = points[count * 2 - 2] * points[1] - points[0] * points[count * 2 - 1];
        const float denom = area_sum + q;
        const float scale = Abs(denom) > 1.0e-8f ? (1.0f / (3.0f * denom)) : kLargeFaceClipFloat;
        cx = scale * (cx + q * (points[count * 2 - 2] + points[0]));
        cy = scale * (cy + q * (points[count * 2 - 1] + points[1]));
    }

    float angles[8];
    for (int i = 0; i < count; ++i) {
        angles[i] = SafeAtan2(points[i * 2 + 1] - cy, points[i * 2] - cx);
    }

    int available[8];
    for (int i = 0; i < count; ++i) {
        available[i] = 1;
    }
    available[deepest_idx] = 0;
    out_indices[0] = deepest_idx;
    for (int slot = 1; slot < max_points; ++slot) {
        float target_angle = static_cast<float>(slot) * (2.0f * 3.14159265f / static_cast<float>(max_points)) +
            angles[deepest_idx];
        if (target_angle > 3.14159265f) {
            target_angle -= 2.0f * 3.14159265f;
        }
        float best_diff = 1.0e9f;
        int best_idx = deepest_idx;
        for (int i = 0; i < count; ++i) {
            if (!available[i]) {
                continue;
            }
            float diff = Abs(angles[i] - target_angle);
            if (diff > 3.14159265f) {
                diff = 2.0f * 3.14159265f - diff;
            }
            if (diff < best_diff) {
                best_diff = diff;
                best_idx = i;
            }
        }
        available[best_idx] = 0;
        out_indices[slot] = best_idx;
    }
}

ROCKETGPU_HD int ComputeFaceClippedCarCarContactAnchorPairs(
    const ObbData& ref_obb,
    int ref_axis_idx,
    const ObbData& inc_obb,
    const Vec3& ref_to_inc_normal,
    Vec3* out_ref_points,
    Vec3* out_inc_points,
    int max_points,
    int debug_env_idx = -1,
    int debug_car_idx_a = -1,
    int debug_car_idx_b = -1,
    int debug_ref_is_a = -1) {
    if (!out_ref_points || !out_inc_points || max_points <= 0) {
        return 0;
    }

    const float ref_axis_sign = Dot(ref_obb.axis[ref_axis_idx], ref_to_inc_normal) >= 0.0f ? 1.0f : -1.0f;
    const Vec3 ref_normal = ref_obb.axis[ref_axis_idx] * ref_axis_sign;

    float nr[3] = {
        Dot(inc_obb.axis[0], ref_normal),
        Dot(inc_obb.axis[1], ref_normal),
        Dot(inc_obb.axis[2], ref_normal),
    };
    float anr[3] = {Abs(nr[0]), Abs(nr[1]), Abs(nr[2])};

    int lanr = 0;
    int a1 = 1;
    int a2 = 2;
    if (anr[1] > anr[0]) {
        if (anr[1] > anr[2]) {
            a1 = 0;
            lanr = 1;
            a2 = 2;
        } else {
            a1 = 0;
            a2 = 1;
            lanr = 2;
        }
    } else {
        if (anr[0] > anr[2]) {
            lanr = 0;
            a1 = 1;
            a2 = 2;
        } else {
            a1 = 0;
            a2 = 1;
            lanr = 2;
        }
    }

    const Vec3 center =
        nr[lanr] < 0.0f
            ? (inc_obb.center - ref_obb.center + inc_obb.axis[lanr] * ObbHalfExtentComponent(inc_obb, lanr))
            : (inc_obb.center - ref_obb.center - inc_obb.axis[lanr] * ObbHalfExtentComponent(inc_obb, lanr));

    const int code1 = ref_axis_idx == 0 ? 1 : 0;
    const int code2 = ref_axis_idx == 2 ? 1 : 2;
    const float c1 = Dot(center, ref_obb.axis[code1]);
    const float c2 = Dot(center, ref_obb.axis[code2]);
    const float m11 = Dot(ref_obb.axis[code1], inc_obb.axis[a1]);
    const float m12 = Dot(ref_obb.axis[code1], inc_obb.axis[a2]);
    const float m21 = Dot(ref_obb.axis[code2], inc_obb.axis[a1]);
    const float m22 = Dot(ref_obb.axis[code2], inc_obb.axis[a2]);

    const float k1 = m11 * ObbHalfExtentComponent(inc_obb, a1);
    const float k2 = m21 * ObbHalfExtentComponent(inc_obb, a1);
    const float k3 = m12 * ObbHalfExtentComponent(inc_obb, a2);
    const float k4 = m22 * ObbHalfExtentComponent(inc_obb, a2);
    float quad[8] = {
        c1 - k1 - k3, c2 - k2 - k4,
        c1 - k1 + k3, c2 - k2 + k4,
        c1 + k1 + k3, c2 + k2 + k4,
        c1 + k1 - k3, c2 + k2 - k4,
    };
    float rect[2] = {
        ObbHalfExtentComponent(ref_obb, code1),
        ObbHalfExtentComponent(ref_obb, code2),
    };
    float clipped[16];
    int poly_count = IntersectRectQuad2(rect, quad, clipped);
    if (poly_count < 1) {
        return 0;
    }

    const float det = m11 * m22 - m12 * m21;
    if (Abs(det) <= 1.0e-8f) {
        return 0;
    }
    const float det_inv = 1.0f / det;
    const float im11 = m11 * det_inv;
    const float im12 = m12 * det_inv;
    const float im21 = m21 * det_inv;
    const float im22 = m22 * det_inv;

    float point_coords[24];
    float depths[8];
    int contact_count = 0;
    for (int i = 0; i < poly_count; ++i) {
        const float ret_u = clipped[i * 2 + 0];
        const float ret_v = clipped[i * 2 + 1];
        const float inc_k1 = im22 * (ret_u - c1) - im12 * (ret_v - c2);
        const float inc_k2 = -im21 * (ret_u - c1) + im11 * (ret_v - c2);
        const Vec3 incident_local =
            center + inc_obb.axis[a1] * inc_k1 + inc_obb.axis[a2] * inc_k2;
        point_coords[contact_count * 3 + 0] = incident_local.x;
        point_coords[contact_count * 3 + 1] = incident_local.y;
        point_coords[contact_count * 3 + 2] = incident_local.z;
        const float depth = ObbHalfExtentComponent(ref_obb, ref_axis_idx) - Dot(ref_normal, incident_local);
        depths[contact_count] = depth;
        if (depth >= 0.0f) {
            clipped[contact_count * 2 + 0] = ret_u;
            clipped[contact_count * 2 + 1] = ret_v;
            ++contact_count;
        }
    }
    if (contact_count < 1) {
        return 0;
    }

    int output_count = Min(contact_count, Min(max_points, kMaxApproxCarCarContactPoints));
    if (output_count < 1) {
        output_count = 1;
    }

    int selected_indices[8];
    if (contact_count <= output_count) {
        for (int i = 0; i < contact_count; ++i) {
            selected_indices[i] = i;
        }
        output_count = contact_count;
    } else {
        int deepest_idx = 0;
        float deepest_depth = depths[0];
        for (int i = 1; i < contact_count; ++i) {
            if (depths[i] > deepest_depth) {
                deepest_depth = depths[i];
                deepest_idx = i;
            }
        }
        CullFacePoints2(contact_count, clipped, output_count, deepest_idx, selected_indices);
    }

    for (int i = 0; i < output_count; ++i) {
        const int idx = selected_indices[i];
        const Vec3 incident_point =
            ref_obb.center +
            Vec3(
                point_coords[idx * 3 + 0],
                point_coords[idx * 3 + 1],
                point_coords[idx * 3 + 2]);

        const Vec3 ref_point = incident_point + ref_normal * depths[idx];
        out_ref_points[i] = ref_point;
        out_inc_points[i] = debug_ref_is_a ? incident_point : (ref_point + ref_normal * (-depths[idx]));
    }

    return output_count;
}

ROCKETGPU_HD void ComputeLineClosestApproachParams(
    const Vec3& pa,
    const Vec3& ua,
    const Vec3& pb,
    const Vec3& ub,
    float& out_alpha,
    float& out_beta) {
    const Vec3 p = pb - pa;
    const float uaub = Dot(ua, ub);
    const float q1 = Dot(ua, p);
    const float q2 = -Dot(ub, p);
    float d = 1.0f - uaub * uaub;
    if (d <= 1.0e-4f) {
        out_alpha = 0.0f;
        out_beta = 0.0f;
        return;
    }
    d = 1.0f / d;
    out_alpha = (q1 + uaub * q2) * d;
    out_beta = (uaub * q1 + q2) * d;
}

ROCKETGPU_HD bool ComputeEdgeEdgeCarCarContactAnchorsBulletStyle(
    const ObbData& a,
    int axis_a_idx,
    const ObbData& b,
    int axis_b_idx,
    const Vec3& normal,
    float penetration_bt,
    Vec3& out_point_a,
    Vec3& out_point_b) {
    if (axis_a_idx < 0 || axis_a_idx > 2 || axis_b_idx < 0 || axis_b_idx > 2) {
        return false;
    }

    Vec3 pa = a.center;
    for (int j = 0; j < 3; ++j) {
        const float sign = Dot(normal, a.axis[j]) > 0.0f ? 1.0f : -1.0f;
        pa += a.axis[j] * (sign * ObbHalfExtentComponent(a, j));
    }

    Vec3 pb = b.center;
    for (int j = 0; j < 3; ++j) {
        const float sign = Dot(normal, b.axis[j]) > 0.0f ? -1.0f : 1.0f;
        pb += b.axis[j] * (sign * ObbHalfExtentComponent(b, j));
    }

    const Vec3 ua = a.axis[axis_a_idx];
    const Vec3 ub = b.axis[axis_b_idx];
    float alpha = 0.0f;
    float beta = 0.0f;
    ComputeLineClosestApproachParams(pa, ua, pb, ub, alpha, beta);

    out_point_b = pb + ub * beta;
    if (penetration_bt > 0.0f) {
        out_point_a = out_point_b + normal * penetration_bt;
    } else {
        out_point_a = pa + ua * alpha;
    }
    return true;
}

ROCKETGPU_HD ObbData MakeCarObb(const CarState& car, const CarConfig& cfg) {
    ObbData obb;
    obb.center = car.pos + RotateCarBasis(car, cfg.hitbox_offset);
    CarBasisColumns(car, obb.axis[0], obb.axis[1], obb.axis[2]);

    obb.half_extent = BulletBoxHalfExtentWithMargin(cfg);
    return obb;
}

ROCKETGPU_HD Vec3 SupportPointOnCarHitboxLocal(const CarConfig& cfg, const Vec3& dir_local) {
    const Vec3 half_extent = BulletBoxHalfExtentWithMargin(cfg);
    return cfg.hitbox_offset + Vec3(
        dir_local.x >= 0.0f ? half_extent.x : -half_extent.x,
        dir_local.y >= 0.0f ? half_extent.y : -half_extent.y,
        dir_local.z >= 0.0f ? half_extent.z : -half_extent.z);
}

ROCKETGPU_HD bool ComputeObbContact(
    const ObbData& a,
    const ObbData& b,
    float contact_threshold_uu,
    Vec3& out_normal,
    float& out_signed_overlap,
    ObbContactFeature* out_feature = nullptr,
    int debug_env_idx = -1,
    int debug_slot_a = -1,
    int debug_slot_b = -1,
    int debug_call_tag = -1) {
    constexpr float kEps = 1e-5f;

    float R[3][3];
    float AbsR[3][3];
    float AbsRFace[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R[i][j] = Dot(a.axis[i], b.axis[j]);
            AbsRFace[i][j] = Abs(R[i][j]);
            AbsR[i][j] = AbsRFace[i][j] + kEps;
        }
    }

    const Vec3 t_world = b.center - a.center;
    float t[3] = {
        Dot(t_world, a.axis[0]),
        Dot(t_world, a.axis[1]),
        Dot(t_world, a.axis[2])
    };

    constexpr float kEdgeFudgeFactor = 1.05f;
    constexpr float kEdgeSepEps = 1.0e-6f;
    bool have_best_axis = false;
    float best_s2 = -1.0e30f;
    Vec3 best_axis(1.0f, 0.0f, 0.0f);
    bool best_invert = false;
    ObbContactFeature best_feature;
    auto consider_face_axis = [&](const Vec3& axis, float expr1, float expr2, uint8_t kind, uint8_t axis_a, uint8_t axis_b) {
        const float s2 = Abs(expr1) - expr2;
        if (s2 > contact_threshold_uu) {
            return false;
        }
        if (!have_best_axis || s2 > best_s2) {
            have_best_axis = true;
            best_s2 = s2;
            best_axis = axis;
            best_invert = expr1 < 0.0f;
            best_feature.kind = kind;
            best_feature.axis_a = axis_a;
            best_feature.axis_b = axis_b;
        }
        return true;
    };

    auto consider_edge_axis = [&](const Vec3& n_local, float expr1, float expr2, uint8_t axis_a, uint8_t axis_b) {
        const float s2_raw = Abs(expr1) - expr2;
        if (s2_raw > kEdgeSepEps) {
            return false;
        }
        const float axis_len = SqrtScalar(n_local.x * n_local.x + n_local.y * n_local.y + n_local.z * n_local.z);
        if (axis_len <= kEdgeSepEps) {
            return true;
        }
        const float s2 = s2_raw / axis_len;
        if (!have_best_axis || s2 * kEdgeFudgeFactor > best_s2) {
            have_best_axis = true;
            best_s2 = s2;
            const Vec3 normal_c(n_local.x / axis_len, n_local.y / axis_len, n_local.z / axis_len);
            best_axis = Vec3(
                a.axis[0].x * normal_c.x + a.axis[1].x * normal_c.y + a.axis[2].x * normal_c.z,
                a.axis[0].y * normal_c.x + a.axis[1].y * normal_c.y + a.axis[2].y * normal_c.z,
                a.axis[0].z * normal_c.x + a.axis[1].z * normal_c.y + a.axis[2].z * normal_c.z);
            best_invert = expr1 < 0.0f;
            best_feature.kind = OBB_CONTACT_FEATURE_EDGE_EDGE;
            best_feature.axis_a = axis_a;
            best_feature.axis_b = axis_b;
        }
        return true;
    };

    for (int i = 0; i < 3; ++i) {
        const float expr1 = t[i];
        const float expr2 =
            (((&a.half_extent.x)[i] + b.half_extent.x * AbsRFace[i][0]) + b.half_extent.y * AbsRFace[i][1]) +
            b.half_extent.z * AbsRFace[i][2];
        if (!consider_face_axis(a.axis[i], expr1, expr2, OBB_CONTACT_FEATURE_FACE_A, static_cast<uint8_t>(i), 0)) {
            return false;
        }
    }

    for (int j = 0; j < 3; ++j) {
        const float expr1 = Dot(t_world, b.axis[j]);
        const float expr2 =
            ((a.half_extent.x * AbsRFace[0][j] + a.half_extent.y * AbsRFace[1][j]) + a.half_extent.z * AbsRFace[2][j]) +
            (&b.half_extent.x)[j];
        if (!consider_face_axis(b.axis[j], expr1, expr2, OBB_CONTACT_FEATURE_FACE_B, 0, static_cast<uint8_t>(j))) {
            return false;
        }
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const int i1 = (i + 1) % 3;
            const int i2 = (i + 2) % 3;

            Vec3 n_local(0.0f, 0.0f, 0.0f);
            if (i == 0) {
                n_local = Vec3(0.0f, -R[2][j], R[1][j]);
            } else if (i == 1) {
                n_local = Vec3(R[2][j], 0.0f, -R[0][j]);
            } else {
                n_local = Vec3(-R[1][j], R[0][j], 0.0f);
            }

            const int a_lo = (i == 0) ? 1 : 0;
            const int a_hi = (i == 2) ? 1 : 2;
            const int b_lo = (j == 0) ? 1 : 0;
            const int b_hi = (j == 2) ? 1 : 2;
            const float expr1 = t[i2] * R[i1][j] - t[i1] * R[i2][j];
            const float expr2 =
                (((&a.half_extent.x)[a_lo] * AbsR[a_hi][j] + (&a.half_extent.x)[a_hi] * AbsR[a_lo][j]) +
                 (&b.half_extent.x)[b_lo] * AbsR[i][b_hi]) +
                (&b.half_extent.x)[b_hi] * AbsR[i][b_lo];
            const float axis_len = SqrtScalar(n_local.x * n_local.x + n_local.y * n_local.y + n_local.z * n_local.z);
            if (!consider_edge_axis(
                    n_local,
                    expr1,
                    expr2,
                    static_cast<uint8_t>(i),
                    static_cast<uint8_t>(j))) {
                return false;
            }
        }
    }

    if (!have_best_axis) {
        return false;
    }
    if (best_invert) {
        best_axis = -best_axis;
    }

    out_normal = best_axis;
    out_signed_overlap = -best_s2;
    if (out_feature) {
        *out_feature = best_feature;
    }
    return true;
}

ROCKETGPU_HD bool InForwardCone(
    const CarState& a, const CarConfig& a_cfg, const CarState& v, const CarConfig& v_cfg, bool reverse, float max_yaw_deg);

ROCKETGPU_HD void AdvancePose(CarState& s, const Vec3& vel, float t) {
    Vec3 f(1.0f, 0.0f, 0.0f), r(0.0f, 1.0f, 0.0f), u(0.0f, 0.0f, 1.0f);
    CarBasisColumns(s, f, r, u);
    s.pos = s.pos + vel * t;
    const float w = Length(s.ang_vel);
    const float th = w * t;
    if (w > 0.0f && th != 0.0f) {
        const Vec3 k = s.ang_vel / w;
        const float c = cosf(th), sn = sinf(th);
        auto rot = [&](const Vec3& x) { return x * c + Cross(k, x) * sn + k * (Dot(k, x) * (1.0f - c)); };
        f = rot(f); r = rot(r); u = rot(u);
    }
    s.basis_forward = f; s.basis_right = r; s.basis_up = u;
    s.basis_authoritative = 1;
}

ROCKETGPU_HD float BoxSeparation(const CarState& a, const CarConfig& a_cfg, const CarState& b, const CarConfig& b_cfg) {
    Vec3 ra[3], rb[3];
    CarBasisColumns(a, ra[0], ra[1], ra[2]);
    CarBasisColumns(b, rb[0], rb[1], rb[2]);
    const Vec3 ha = a_cfg.hitbox_size * 0.5f, hb = b_cfg.hitbox_size * 0.5f;
    const float h_a[3] = {ha.x, ha.y, ha.z}, h_b[3] = {hb.x, hb.y, hb.z};
    const Vec3 ca = a.pos + ra[0] * a_cfg.hitbox_offset.x + ra[1] * a_cfg.hitbox_offset.y + ra[2] * a_cfg.hitbox_offset.z;
    const Vec3 cb = b.pos + rb[0] * b_cfg.hitbox_offset.x + rb[1] * b_cfg.hitbox_offset.y + rb[2] * b_cfg.hitbox_offset.z;
    const Vec3 d = cb - ca;
    float best = -1e30f;
    auto test = [&](const Vec3& L) {
        float sep = fabsf(Dot(d, L));
        for (int i = 0; i < 3; ++i) {
            sep -= fabsf(Dot(ra[i], L)) * h_a[i] + fabsf(Dot(rb[i], L)) * h_b[i];
        }
        best = best > sep ? best : sep;
    };
    for (int i = 0; i < 3; ++i) {
        test(ra[i]);
        test(rb[i]);
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const Vec3 L = Cross(ra[i], rb[j]);
            const float n = Length(L);
            if (n > 1e-6f) {
                test(L / n);
            }
        }
    }
    return best;
}

ROCKETGPU_HD void CarCarTimeOfImpact(CarState& a, const CarConfig& a_cfg, const Vec3& a_vel,
                                   CarState& v, const CarConfig& v_cfg, const Vec3& v_vel, float tick_dt) {
    constexpr float kPenetration = 0.039f * 50.0f;
    auto sep = [&](float lambda) {
        const float t = (2.0f * lambda - 1.0f) * tick_dt;
        CarState ai = a, vi = v;
        AdvancePose(ai, a_vel, t);
        AdvancePose(vi, v_vel, t);
        return BoxSeparation(ai, a_cfg, vi, v_cfg) + kPenetration;
    };
    float lambda;
    if (sep(0.0f) <= 0.0f) {
        lambda = 0.0f;
    } else if (sep(1.0f) > 0.0f) {
        return;
    } else {
        float lo = 0.0f, hi = 1.0f;
        for (int i = 0; i < 24; ++i) {
            const float m = (lo + hi) * 0.5f;
            if (sep(m) > 0.0f) {
                lo = m;
            } else {
                hi = m;
            }
        }
        lambda = hi;
    }
    const float t = (2.0f * lambda - 1.0f) * tick_dt;
    AdvancePose(a, a_vel, t);
    AdvancePose(v, v_vel, t);
}

ROCKETGPU_HD bool ShouldDemolish(
    const CarState& a, const CarConfig& a_cfg, const Vec3& a_start_vel,
    const CarState& v, const CarConfig& v_cfg, const MutatorConfig& mutator,
    const Vec3* v_start_vel = nullptr, float tick_dt = 0.0f) {
    if (!a.is_supersonic) {
        return false;
    }
    Vec3 fwd(1.0f, 0.0f, 0.0f), right(0.0f, 1.0f, 0.0f), up(0.0f, 0.0f, 1.0f);
    CarBasisColumns(a, fwd, right, up);
    const float forward_speed = Dot(a_start_vel, fwd);
    const bool reverse = forward_speed < 0.0f;
    if (fabsf(forward_speed) < mutator.supersonic_maintain_min_speed) {
        return false;
    }
    if (v_start_vel != nullptr) {
        CarState ai = a, vi = v;
        CarCarTimeOfImpact(ai, a_cfg, a_start_vel, vi, v_cfg, *v_start_vel, tick_dt);
        return InForwardCone(ai, a_cfg, vi, v_cfg, reverse, 45.5729942f);
    }
    return InForwardCone(a, a_cfg, v, v_cfg, reverse, 45.5729942f);
}

ROCKETGPU_HD bool InForwardCone(
    const CarState& a, const CarConfig& a_cfg, const CarState& v, const CarConfig& v_cfg, bool reverse, float max_yaw_deg) {
    Vec3 fwd(1.0f, 0.0f, 0.0f), right(0.0f, 1.0f, 0.0f), up(0.0f, 0.0f, 1.0f);
    CarBasisColumns(a, fwd, right, up);
    const Vec3 pa = a.pos + InverseRotateCarBasis(a, a_cfg.hitbox_offset);
    const Vec3 pv = v.pos + InverseRotateCarBasis(v, v_cfg.hitbox_offset);
    Vec3 d = pv - pa;
    const float d2 = LengthSq(d);
    if (d2 != 1.0f) {
        d = (d2 < 1e-8f) ? Vec3(0.0f, 0.0f, 0.0f) : d * (1.0f / sqrtf(d2));
    }
    const Vec3 f = reverse ? fwd * -1.0f : fwd;
    auto angle_deg = [&](const Vec3& axis) {
        float k = Dot(d, axis);
        k = (k >= 0.0f) ? (k / 1.01f) : (k * 1.01f);
        Vec3 p = d - axis * k;
        const float p2 = LengthSq(p);
        if (p2 != 1.0f) {
            p = (p2 < 1e-8f) ? Vec3(0.0f, 0.0f, 0.0f) : p * (1.0f / sqrtf(p2));
        }
        float c = Dot(f, p);
        c = c < -1.0f ? -1.0f : (c > 1.0f ? 1.0f : c);
        return acosf(c) * 57.29578f;
    };
    if (angle_deg(right) > 36.8698959f) {
        return false;
    }
    return angle_deg(up) <= max_yaw_deg;
}

ROCKETGPU_HD bool BumpCone(
    CarState& hitter, uint32_t hitter_id, Team hitter_team, const CarConfig& hitter_cfg, const Vec3& hitter_start_vel,
    CarState& victim, uint32_t victim_id, Team victim_team, const CarConfig& victim_cfg,
    const MutatorConfig& mutator, uint8_t enable_demos, bool cal_demo, float tick_dt) {
    const bool bumper = InForwardCone(hitter, hitter_cfg, victim, victim_cfg, false, 70.0f);
    bool is_demo = false;
    if (!cal_demo && bumper && enable_demos) {
        is_demo = hitter.is_supersonic != 0;
        if (is_demo && !mutator.enable_team_demos) {
            is_demo = hitter_team != victim_team;
        }
    }
    if (is_demo) {
        DemolishCar(victim, mutator);
    } else if (bumper) {
        const float speed = SqrtScalar(LengthSq(hitter_start_vel));
        const Vec3 vel_dir = hitter_start_vel / speed;
        auto curve = [speed](float y0, float y1, float y2) {
            if (speed <= 0.0f) {
                return y0;
            }
            if (speed < 1400.0f) {
                return y0 + (y1 - y0) * (speed / 1400.0f);
            }
            if (speed < 2200.0f) {
                return y1 + (y2 - y1) * ((speed - 1400.0f) / 800.0f);
            }
            return y2;
        };
        Vec3 impulse;
        if (victim.is_on_ground != 0) {
            Vec3 v_fwd(1.0f, 0.0f, 0.0f), v_rgt(0.0f, 1.0f, 0.0f), v_up(0.0f, 0.0f, 1.0f);
            CarBasisColumns(victim, v_fwd, v_rgt, v_up);
            impulse = vel_dir * curve(150.0f, 200000.0f, 275000.0f) + v_up * curve(50.0f, 50000.0f, 75000.0f);
        } else {
            impulse = vel_dir * curve(150.0f, 250000.0f, 350000.0f);
        }

        const Vec3 impulse_bt = impulse * (mutator.bump_force_scale * ((1.0f / 50.0f) / 180.0f));
        victim.vel_impulse_cache_bt += impulse_bt;
        victim.vel_impulse_cache += impulse_bt * 50.0f;
    }
    hitter.car_contact.other_car_id = victim_id;
    hitter.car_contact.cooldown_timer = mutator.bump_cooldown_time + tick_dt * 0.5f;
    if (is_demo || bumper) {
        hitter.bump_event_tick = hitter.tick_count_since_update;
        hitter.bump_event_other_id = victim_id;
        hitter.bump_event_is_demo = is_demo ? 1 : 0;
        hitter.bump_event_was_bumper = 1;
        victim.bump_event_tick = victim.tick_count_since_update;
        victim.bump_event_other_id = hitter_id;
        victim.bump_event_is_demo = is_demo ? 1 : 0;
        victim.bump_event_was_bumper = 0;
        if (mutator.physics_events.windows) {
            PhysicsEvent event;
            event.type = PhysicsEventType::CarBump;
            event.car_id = hitter_id;
            event.other_car_id = victim_id;
            event.team = static_cast<int32_t>(hitter_team);
            event.is_demo = is_demo;
            event.car_on_ground = hitter.is_on_ground;
            event.position = hitter.pos;
            event.car_velocity = hitter.vel;
            mutator.physics_events.EmitForCar(hitter_id, event);
        }
    }
    return is_demo;
}

ROCKETGPU_HD bool ResolveCarCarDirectionalContact(
    CarState& hitter,
    const CarState& hitter_probe,
    uint32_t hitter_id,
    Team hitter_team,
    const ObbData& hitter_obb,
    const CarConfig& hitter_cfg,
    CarState& victim,
    const CarState& victim_probe,
    uint32_t victim_id,
    Team victim_team,
    const ObbData& victim_obb,
    const MutatorConfig& mutator,
    uint8_t enable_demos,
    const Vec3& fallback_dir,
    bool has_bumper_local_x_override,
    float bumper_local_x_override,
    const Vec3* hitter_start_vel = nullptr,
    const CarConfig* victim_cfg = nullptr,
    bool demo_only = false,
    const Vec3* victim_start_vel = nullptr,
    float tick_dt = 0.0f) {
    (void)hitter_id;
    (void)hitter_obb;
    (void)victim_obb;

    const bool cal_demo = true;
    if ((hitter.is_demoed && !(cal_demo && demo_only)) || victim.is_demoed || hitter.just_respawned || victim.just_respawned) {
        return false;
    }
    const bool in_cooldown = (hitter.car_contact.other_car_id == victim_id) && (hitter.car_contact.cooldown_timer > 0.0f);

    if (hitter_start_vel != nullptr && victim_start_vel != nullptr && victim_cfg != nullptr) {
        const Vec3 vel = *hitter_start_vel;
        const float speed = SqrtScalar(LengthSq(vel));
        if (!(speed > 0.0f)) {
            return false;
        }
        const Vec3 hit_dir = RocketSimVecNormalized(victim.pos - hitter.pos);
        const float speed_towards = Dot(hit_dir, vel);
        if (speed_towards <= 0.0f) {
            return false;
        }
        const Vec3 vel_dir = vel / speed;
        if (Dot(*victim_start_vel, vel_dir) >= speed_towards) {
            return false;
        }
        if (cal_demo) {
            bool cal_is_demo = enable_demos &&
                ShouldDemolish(hitter, hitter_cfg, vel, victim, *victim_cfg, mutator, victim_start_vel, tick_dt);
            if (cal_is_demo && !mutator.enable_team_demos) {
                cal_is_demo = hitter_team != victim_team;
            }
            if (cal_is_demo) {
                DemolishCar(victim, mutator);
                hitter.bump_event_tick = hitter.tick_count_since_update;
                hitter.bump_event_other_id = victim_id;
                hitter.bump_event_is_demo = 1;
                hitter.bump_event_was_bumper = 1;
                victim.bump_event_tick = victim.tick_count_since_update;
                victim.bump_event_other_id = hitter_id;
                victim.bump_event_is_demo = 1;
                victim.bump_event_was_bumper = 0;
                if (mutator.physics_events.windows) {
                    PhysicsEvent event;
                    event.type = PhysicsEventType::CarBump;
                    event.car_id = hitter_id;
                    event.other_car_id = victim_id;
                    event.team = static_cast<int32_t>(hitter_team);
                    event.is_demo = true;
                    event.car_on_ground = hitter.is_on_ground;
                    event.position = hitter.pos;
                    event.car_velocity = hitter.vel;
                    mutator.physics_events.EmitForCar(hitter_id, event);
                }
                return true;
            }
            if (demo_only) {
                return false;
            }
        }
        if (in_cooldown) {
            return false;
        }
        return BumpCone(hitter, hitter_id, hitter_team, hitter_cfg, vel, victim, victim_id, victim_team,
                                *victim_cfg, mutator, enable_demos, cal_demo, tick_dt);
    }
    if (in_cooldown && !cal_demo) {
        return false;
    }

    (void)hitter_probe;
    (void)victim_probe;
    const Vec3 delta_pos = victim.pos - hitter.pos;
    if (Dot(hitter.vel, delta_pos) <= 0.0f) {
        return false;
    }

    const Vec3 vel_dir = RocketSimVecNormalized(hitter.vel);
    const Vec3 dir_to_other = RocketSimVecNormalized(delta_pos);
    const float speed_towards_other = Dot(hitter.vel, dir_to_other);
    const float other_car_away_speed = Dot(victim.vel, vel_dir);
    if (speed_towards_other <= other_car_away_speed) {
        return false;
    }

    if (cal_demo) {
        bool cal_is_demo = enable_demos && victim_cfg != nullptr && hitter_start_vel != nullptr &&
            ShouldDemolish(hitter, hitter_cfg, *hitter_start_vel, victim, *victim_cfg, mutator, victim_start_vel, tick_dt);
        if (cal_is_demo && !mutator.enable_team_demos) {
            cal_is_demo = hitter_team != victim_team;
        }
        if (cal_is_demo) {
            DemolishCar(victim, mutator);
            hitter.bump_event_tick = hitter.tick_count_since_update;
            hitter.bump_event_other_id = victim_id;
            hitter.bump_event_is_demo = 1;
            hitter.bump_event_was_bumper = 1;
            victim.bump_event_tick = victim.tick_count_since_update;
            victim.bump_event_other_id = hitter_id;
            victim.bump_event_is_demo = 1;
            victim.bump_event_was_bumper = 0;
            if (mutator.physics_events.windows) {
                PhysicsEvent event;
                event.type = PhysicsEventType::CarBump;
                event.car_id = hitter_id;
                event.other_car_id = victim_id;
                event.team = static_cast<int32_t>(hitter_team);
                event.is_demo = true;
                event.car_on_ground = hitter.is_on_ground;
                event.position = hitter.pos;
                event.car_velocity = hitter.vel;
                mutator.physics_events.EmitForCar(hitter_id, event);
            }
            return true;
        }
        if (demo_only || in_cooldown) {
            return false;
        }
    }

    float bumper_local_x = bumper_local_x_override;
    if (!has_bumper_local_x_override) {
        const Vec3 contact_dir_world = Normalize(dir_to_other, fallback_dir);
        const Vec3 local_point = SupportPointOnCarHitboxLocal(hitter_cfg, InverseRotateCarBasis(hitter, contact_dir_world));
        bumper_local_x = local_point.x;
    }
    const bool hit_with_bumper = bumper_local_x > mutator.bump_min_forward_dist;
    if (!hit_with_bumper) {
        return false;
    }

    bool is_demo = false;
    if (enable_demos && !cal_demo) {
        is_demo = hitter.is_supersonic;
    }
    if (is_demo && !mutator.enable_team_demos) {
        is_demo = hitter_team != victim_team;
    }

    if (is_demo) {
        DemolishCar(victim, mutator);
    } else {
        const bool ground_hit = victim.is_on_ground != 0;

        float bump_speed = SqrtScalar(LengthSq(hitter.vel));

        const bool cal_car_contact = hitter_start_vel != nullptr;
        Vec3 bump_dir = vel_dir;
        if (cal_car_contact) {
            const float start_speed = SqrtScalar(LengthSq(*hitter_start_vel));
            if (start_speed > 0.0f) {
                bump_speed = start_speed;
                bump_dir = *hitter_start_vel / start_speed;
            }
        }
        const float base_scale = cal_car_contact
            ? (ground_hit ? EvalGameBumpVelGround(bump_speed) : EvalGameBumpVelAir(bump_speed))
            : (ground_hit ? EvalBumpVelGround(bump_speed) : EvalBumpVelAir(bump_speed));

        Vec3 vbump_fwd(1.0f, 0.0f, 0.0f);
        Vec3 vbump_rgt(0.0f, 1.0f, 0.0f);
        Vec3 vbump_up(0.0f, 0.0f, 1.0f);
        CarBasisColumns(victim, vbump_fwd, vbump_rgt, vbump_up);
        (void)vbump_fwd; (void)vbump_rgt;
        const Vec3 hit_up_dir = ground_hit ? vbump_up : Vec3(0.0f, 0.0f, 1.0f);

        const Vec3 bump_impulse = bump_dir * base_scale +
            (hit_up_dir * (cal_car_contact ? EvalGameBumpUpwardVel(bump_speed) : EvalBumpUpwardVel(bump_speed))) *
                mutator.bump_force_scale;
        victim.vel_impulse_cache += bump_impulse;

        victim.vel_impulse_cache_bt += bump_impulse * (1.0f / 50.0f);
    }

    hitter.car_contact.other_car_id = victim_id;
    hitter.car_contact.cooldown_timer = mutator.bump_cooldown_time;

    hitter.bump_event_tick = hitter.tick_count_since_update;
    hitter.bump_event_other_id = victim_id;
    hitter.bump_event_is_demo = is_demo ? 1 : 0;
    hitter.bump_event_was_bumper = 1;
    victim.bump_event_tick = victim.tick_count_since_update;
    victim.bump_event_other_id = hitter_id;
    victim.bump_event_is_demo = is_demo ? 1 : 0;
    victim.bump_event_was_bumper = 0;
    if (mutator.physics_events.windows) {
        PhysicsEvent event;
        event.type = PhysicsEventType::CarBump;
        event.car_id = hitter_id;
        event.other_car_id = victim_id;
        event.team = static_cast<int32_t>(hitter_team);
        event.is_demo = is_demo;
        event.car_on_ground = hitter.is_on_ground;
        event.position = hitter.pos;
        event.car_velocity = hitter.vel;
        mutator.physics_events.EmitForCar(hitter_id, event);
    }
    return is_demo;
}

ROCKETGPU_HD int ComputeApproxCarCarContactAnchorPairs(
    const CarState& a,
    const CarConfig& cfg_a,
    const ObbData& obb_a,
    const CarState& b,
    const CarConfig& cfg_b,
    const ObbData& obb_b,
    const Vec3& normal,
    const ObbContactFeature& feature,
    Vec3* out_points_a,
    Vec3* out_points_b,
    int max_points,
    float contact_overlap_uu = 0.0f,
    Vec3* out_points_a_bt = nullptr,
    Vec3* out_points_b_bt = nullptr,
    int debug_env_idx = -1,
    int debug_car_idx_a = -1,
    int debug_car_idx_b = -1) {
    if (!out_points_a || !out_points_b || max_points <= 0) {
        return 0;
    }

    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kBTToUU = 50.0f;
    (void)obb_a;
    (void)obb_b;
    const ObbData obb_a_bt = MakeCarObbBulletTransformBt(a, cfg_a);
    const ObbData obb_b_bt = MakeCarObbBulletTransformBt(b, cfg_b);

    if (feature.kind == OBB_CONTACT_FEATURE_FACE_A) {
        Vec3 ref_points_bt[kMaxApproxCarCarContactPoints];
        Vec3 inc_points_bt[kMaxApproxCarCarContactPoints];
        const int pair_count_bt = ComputeFaceClippedCarCarContactAnchorPairs(
            obb_a_bt,
            feature.axis_a,
            obb_b_bt,
            normal,
            ref_points_bt,
            inc_points_bt,
            max_points,
            debug_env_idx,
            debug_car_idx_a,
            debug_car_idx_b,
            1);
        if (pair_count_bt > 0) {
            for (int i = 0; i < pair_count_bt; ++i) {
                const Vec3 point_a_bt = CarBodyManifoldWorldPointBt(a, ref_points_bt[i]);
                const Vec3 point_b_bt = CarBodyManifoldWorldPointBt(b, inc_points_bt[i]);
                if (out_points_a_bt) {
                    out_points_a_bt[i] = point_a_bt;
                }
                if (out_points_b_bt) {
                    out_points_b_bt[i] = point_b_bt;
                }
                out_points_a[i] = point_a_bt * kBTToUU;
                out_points_b[i] = point_b_bt * kBTToUU;
            }
            return pair_count_bt;
        }
    } else if (feature.kind == OBB_CONTACT_FEATURE_FACE_B) {
        Vec3 ref_points_bt[kMaxApproxCarCarContactPoints];
        Vec3 inc_points_bt[kMaxApproxCarCarContactPoints];
        const int pair_count_bt = ComputeFaceClippedCarCarContactAnchorPairs(
            obb_b_bt,
            feature.axis_b,
            obb_a_bt,
            -normal,
            ref_points_bt,
            inc_points_bt,
            max_points,
            debug_env_idx,
            debug_car_idx_a,
            debug_car_idx_b,
            0);
        if (pair_count_bt > 0) {
            for (int i = 0; i < pair_count_bt; ++i) {
                const Vec3 point_a_bt = CarBodyManifoldWorldPointBt(a, inc_points_bt[i]);
                const Vec3 point_b_bt = CarBodyManifoldWorldPointBt(b, ref_points_bt[i]);
                if (out_points_a_bt) {
                    out_points_a_bt[i] = point_a_bt;
                }
                if (out_points_b_bt) {
                    out_points_b_bt[i] = point_b_bt;
                }
                out_points_a[i] = point_a_bt * kBTToUU;
                out_points_b[i] = point_b_bt * kBTToUU;
            }
            return pair_count_bt;
        }
    }

    if (feature.kind == OBB_CONTACT_FEATURE_EDGE_EDGE) {
        Vec3 point_a_bt;
        Vec3 point_b_bt;
        if (ComputeEdgeEdgeCarCarContactAnchorsBulletStyle(
                obb_a_bt,
                static_cast<int>(feature.axis_a),
                obb_b_bt,
                static_cast<int>(feature.axis_b),
                normal,
                contact_overlap_uu * kUUToBT,
                point_a_bt,
                point_b_bt)) {
            float overlap_bt = contact_overlap_uu * kUUToBT;
            {
                Vec3 sat_normal_bt(0.0f, 0.0f, 0.0f);
                float sat_overlap_bt = 0.0f;
                ObbContactFeature sat_feature_bt;
                if (ComputeObbContact(obb_a_bt, obb_b_bt, 0.0f, sat_normal_bt, sat_overlap_bt, &sat_feature_bt) &&
                    sat_feature_bt.kind == OBB_CONTACT_FEATURE_EDGE_EDGE &&
                    sat_feature_bt.axis_a == feature.axis_a && sat_feature_bt.axis_b == feature.axis_b) {
                    overlap_bt = sat_overlap_bt;
                }
            }
            const Vec3 raw_point_b_bt = point_b_bt;
            const Vec3 raw_point_a_bt = overlap_bt > 0.0f ? (raw_point_b_bt + normal * overlap_bt) : point_a_bt;
            point_b_bt = CarBodyManifoldWorldPointBt(b, raw_point_b_bt);
            point_a_bt = CarBodyManifoldWorldPointBt(a, raw_point_a_bt);
            if (out_points_a_bt) {
                out_points_a_bt[0] = point_a_bt;
            }
            if (out_points_b_bt) {
                out_points_b_bt[0] = point_b_bt;
            }
            out_points_a[0] = point_a_bt * kBTToUU;
            out_points_b[0] = point_b_bt * kBTToUU;
            return 1;
        }
    }

    out_points_a[0] =
        a.pos + RotateCarBasis(a, SupportPointOnCarHitboxLocal(cfg_a, InverseRotateCarBasis(a, normal)));
    out_points_b[0] =
        b.pos + RotateCarBasis(b, SupportPointOnCarHitboxLocal(cfg_b, InverseRotateCarBasis(b, -normal)));
    if (out_points_a_bt) {
        out_points_a_bt[0] = out_points_a[0] * kUUToBT;
    }
    if (out_points_b_bt) {
        out_points_b_bt[0] = out_points_b[0] * kUUToBT;
    }
    return 1;
}

ROCKETGPU_HD void ComputeApproxCarCarContactAnchors(
    const CarState& a,
    const CarConfig& cfg_a,
    const ObbData& obb_a,
    const CarState& b,
    const CarConfig& cfg_b,
    const ObbData& obb_b,
    const Vec3& normal,
    const ObbContactFeature& feature,
    Vec3& out_point_a,
    Vec3& out_point_b) {
    Vec3 points_a[kMaxApproxCarCarContactPoints];
    Vec3 points_b[kMaxApproxCarCarContactPoints];
    const int pair_count = ComputeApproxCarCarContactAnchorPairs(
        a,
        cfg_a,
        obb_a,
        b,
        cfg_b,
        obb_b,
        normal,
        feature,
        points_a,
        points_b,
        1,
        0.0f);
    if (pair_count > 0) {
        out_point_a = points_a[0];
        out_point_b = points_b[0];
    }
}

ROCKETGPU_HD CarState BuildPredictedCarSolverState(
    const CarState& car,
    const Vec3& external_linear_delta,
    const Vec3& external_angular_delta,
    float dt,
    bool native_sse_math = false) {
    CarState predicted = car;
    IntegrateCarTransformOnly(predicted, dt, false, native_sse_math);
    predicted.vel += external_linear_delta;
    predicted.ang_vel += external_angular_delta;
    return predicted;
}
