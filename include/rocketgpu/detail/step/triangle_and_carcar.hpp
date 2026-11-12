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

ROCKETGPU_HD bullet_port::btTransform MakeBulletPortBoxWorldTransform(CarCRefT car, const Vec3& box_center_bt) {
    bullet_port::btTransform transform;
    Vec3 fwd;
    Vec3 rgt;
    Vec3 upv;
    CarBasisColumns(car, fwd, rgt, upv);
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
    };

    const int num_vectors = static_cast<int>(sizeof(guess_vectors) / sizeof(guess_vectors[0]));
    for (int i = 0; i < num_vectors; ++i) {
        bullet_port::btGjkEpaSolver2::sResults results;

#if defined(__CUDA_ARCH__)
        if (scratch == nullptr) {
            return false;
        }
        const bool penetrated = bullet_port::btGjkEpaSolver2::PenetrationWithScratch(
            &a, transform_a, &b, transform_b, guess_vectors[i], results, *scratch, true);
#else
        const bool penetrated = scratch
            ? bullet_port::btGjkEpaSolver2::PenetrationWithScratch(&a, transform_a, &b, transform_b, guess_vectors[i], results, *scratch, true)
            : bullet_port::btGjkEpaSolver2::Penetration(&a, transform_a, &b, transform_b, guess_vectors[i], results, true);
#endif
        if (penetrated) {
            out_witness_on_a = results.witnesses[0];
            out_witness_on_b = results.witnesses[1];
            io_v = results.normal;
            return true;
        }
#if defined(__CUDA_ARCH__)
        const bool separated = bullet_port::btGjkEpaSolver2::DistanceWithScratch(
            &a, transform_a, &b, transform_b, guess_vectors[i], results, *scratch);
#else
        const bool separated = scratch
            ? bullet_port::btGjkEpaSolver2::DistanceWithScratch(&a, transform_a, &b, transform_b, guess_vectors[i], results, *scratch)
            : bullet_port::btGjkEpaSolver2::Distance(&a, transform_a, &b, transform_b, guess_vectors[i], results);
#endif
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

ROCKETGPU_HD ROCKETGPU_TRIANGLE_CONTACT_NOINLINE bool BulletPortCalcPenDepthLocalScratch(
    const bullet_port::btConvexShape& a,
    const bullet_port::btTransform& transform_a,
    const bullet_port::btConvexShape& b,
    const bullet_port::btTransform& transform_b,
    bullet_port::btVector3& io_v,
    bullet_port::btVector3& out_witness_on_a,
    bullet_port::btVector3& out_witness_on_b) {
    bullet_port::btGjkEpaScratch scratch{};
    return BulletPortCalcPenDepth(
        a, transform_a, b, transform_b, io_v, out_witness_on_a, out_witness_on_b, &scratch);
}

ROCKETGPU_HD Vec3 CarBodyLocalPointBt(CarCRefT car, const Vec3& point_world_bt) {
    bullet_port::btTransform body_world;
    Vec3 fwd;
    Vec3 rgt;
    Vec3 upv;
    CarBasisColumns(car, fwd, rgt, upv);
    body_world.setBasis(bullet_port::btMatrix3x3(
        bullet_port::btVector3(fwd.x, rgt.x, upv.x),
        bullet_port::btVector3(fwd.y, rgt.y, upv.y),
        bullet_port::btVector3(fwd.z, rgt.z, upv.z)));
    body_world.setOrigin(ToBulletPortVec3(car.pos_bt));
    const bullet_port::btVector3 local = body_world.invXform(ToBulletPortVec3(point_world_bt));
    return FromBulletPortVec3(local);
}

ROCKETGPU_HD Vec3 CarBodyWorldPointBt(CarCRefT car, const Vec3& point_local_bt) {
    bullet_port::btTransform body_world;
    Vec3 fwd;
    Vec3 rgt;
    Vec3 upv;
    CarBasisColumns(car, fwd, rgt, upv);
    body_world.setBasis(bullet_port::btMatrix3x3(
        bullet_port::btVector3(fwd.x, rgt.x, upv.x),
        bullet_port::btVector3(fwd.y, rgt.y, upv.y),
        bullet_port::btVector3(fwd.z, rgt.z, upv.z)));
    body_world.setOrigin(ToBulletPortVec3(car.pos_bt));
    const bullet_port::btVector3 world =
        body_world(ToBulletPortVec3(point_local_bt));
    return FromBulletPortVec3(world);
}

ROCKETGPU_HD Vec3 CarBodyManifoldWorldPointBt(CarCRefT car, const Vec3& point_world_bt) {
    return CarBodyWorldPointBt(car, CarBodyLocalPointBt(car, point_world_bt));
}

ROCKETGPU_HD ROCKETGPU_TRIANGLE_CONTACT_NOINLINE bool RunCarArenaTriangleGjkSimplex(
    const bullet_port::btConvexShape& box_shape,
    const bullet_port::btConvexShape& tri_shape,
    const bullet_port::btTransform& local_trans_a,
    const bullet_port::btTransform& local_trans_b,
    Vec3& cached_axis,
    Vec3& last_nonzero_axis,
    int& degenerate_simplex,
    float& squared_distance,
    Vec3& point_on_a_bt,
    Vec3& point_on_b_bt) {
    constexpr float kRelError2 = 1.0e-6f;
    constexpr int kGjkMaxIter = 1000;
    constexpr float kMaxDistSq = 100.0f;
    GjkSimplexSolver simplex;
    simplex.reset();
    bool check_simplex = false;
    int cur_iter = 0;
    while (true) {
        const bullet_port::btVector3 cached_separating_axis = ToBulletPortVec3(cached_axis);
        const bullet_port::btVector3 separating_axis_in_a =
            (-cached_separating_axis) * local_trans_a.getBasis();
        const bullet_port::btVector3 separating_axis_in_b =
            cached_separating_axis * local_trans_b.getBasis();
        const bullet_port::btVector3 p_in_a =
            box_shape.localGetSupportVertexWithoutMarginNonVirtual(separating_axis_in_a);
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

        if ((delta > 0.0f) && (delta * delta > squared_distance * kMaxDistSq)) {
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
    if (check_simplex) {
        simplex.compute_points(point_on_a_bt, point_on_b_bt);
    }
    return check_simplex;
}

ROCKETGPU_HD bool ComputeCarArenaTriangleContact(
    CarCRefT car,
    const CarConfig& cfg,
    const ArenaMeshTriangle& tri,
    Vec3& out_normal,
    float& out_distance_bt,
    Vec3* out_point_on_a_bt,
    Vec3* out_point_on_b_bt,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch,
    Vec3* io_cached_axis_bt,
    float tri_origin_y_bt) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kRelError2 = 1.0e-6f;
    constexpr float kBulletBoxMarginBt = 0.04f;

    const float requested_margin_a_bt = kBulletBoxMarginBt;
    const Vec3 raw_half_extent_bt = (BulletHitboxSize(cfg) * 0.5f) * kUUToBT;

    const float requested_margin_b_bt = 0.0f;
    const float maximum_distance_squared = 100.0f;
    const bullet_port::btConvexShape box_shape =
        MakeBulletPortBoxShape(raw_half_extent_bt, requested_margin_a_bt);
    const bullet_port::btConvexShape tri_shape =
        MakeBulletPortTriangleShape(tri.a, tri.b, tri.c, requested_margin_b_bt);
    const float margin_a_bt = box_shape.getMarginNonVirtual();
    const float margin_b_bt = tri_shape.getMarginNonVirtual();
    const bullet_port::btVector3 child_origin_bt = ToBulletPortVec3(cfg.hitbox_offset * kUUToBT);
    bullet_port::btTransform local_trans_a = MakeBulletPortBoxWorldTransform(car, car.pos_bt);
    local_trans_a.setOrigin(local_trans_a * child_origin_bt);
    const float margin_bt = margin_a_bt + margin_b_bt;

    bullet_port::btTransform local_trans_b;
    if (tri_origin_y_bt != 0.0f) {
        local_trans_b.setOrigin(bullet_port::btVector3(0.0f, tri_origin_y_bt, 0.0f));
    }
    const bullet_port::btVector3 position_offset =
        (local_trans_a.getOrigin() + local_trans_b.getOrigin()) * 0.5f;
    local_trans_a.getOrigin() -= position_offset;
    local_trans_b.getOrigin() -= position_offset;
    const Vec3 position_offset_bt = FromBulletPortVec3(position_offset);

    Vec3 cached_axis = io_cached_axis_bt != nullptr ? *io_cached_axis_bt : Vec3(0.0f, 1.0f, 0.0f);
    Vec3 last_nonzero_axis = cached_axis;

    bool is_valid = false;
    int degenerate_simplex = 0;
    int last_used_method = -1;
    float squared_distance = 1.0e30f;
    float distance_bt = 0.0f;
    bool used_penetration_depth_solver = false;
    bool penetration_valid = false;
    Vec3 normal_bt(0.0f, 0.0f, 0.0f);
    Vec3 point_on_a_bt(0.0f, 0.0f, 0.0f);
    Vec3 point_on_b_bt(0.0f, 0.0f, 0.0f);

    const bool check_simplex = RunCarArenaTriangleGjkSimplex(
        box_shape, tri_shape, local_trans_a, local_trans_b,
        cached_axis, last_nonzero_axis, degenerate_simplex, squared_distance, point_on_a_bt, point_on_b_bt);

    Vec3 gjk_normal_hint = last_nonzero_axis;
    if (check_simplex) {
        if (LengthSq(cached_axis) > kRelError2) {
            gjk_normal_hint = Normalize(cached_axis, gjk_normal_hint);
        }

        const float len_sqr = LengthSq(cached_axis);
        if (len_sqr <= 0.0001f) {
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
                        degenerate_simplex && ((distance_bt + margin_bt) <= 0.01f);
                    if (catch_degenerate_penetration_case_cuda) {
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
        degenerate_simplex && ((distance_bt + margin_bt) <= 0.01f);

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

#endif

ROCKETGPU_HD float BallCarContactBreakingThresholdBT(const CarConfig& cfg, float ball_radius);

ROCKETGPU_HD ROCKETGPU_TRIANGLE_CONTACT_NOINLINE bool RunBallCarGjkSimplex(
    const bullet_port::btConvexShape& box_shape,
    const bullet_port::btSphereShape& sphere_shape,
    const bullet_port::btTransform& local_trans_a,
    const bullet_port::btTransform& local_trans_b,
    float maximum_distance_squared,
    Vec3& cached_axis,
    int& degenerate_simplex,
    float& squared_distance,
    Vec3& point_on_a_bt,
    Vec3& point_on_b_bt) {
    constexpr float kRelError2 = 1.0e-6f;
    constexpr int kGjkMaxIter = 1000;
    GjkSimplexSolver simplex;
    simplex.reset();
    bool check_simplex = false;
    int cur_iter = 0;
    while (true) {
        const bullet_port::btVector3 cached_separating_axis = ToBulletPortVec3(cached_axis);
        const bullet_port::btVector3 separating_axis_in_a =
            (-cached_separating_axis) * local_trans_a.getBasis();
        const bullet_port::btVector3 separating_axis_in_b =
            cached_separating_axis * local_trans_b.getBasis();
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
    simplex.compute_points(point_on_a_bt, point_on_b_bt);
    return true;
}

ROCKETGPU_HD ROCKETGPU_TRIANGLE_CONTACT_NOINLINE bool ComputeBallCarContactGeometryGjk(
    const BallState& ball,
    CarCArgT car_arg,
    const CarConfig& cfg,
    float ball_radius,
    Vec3& out_normal,
    float& out_penetration,
    Vec3& out_rel_ball_contact,
    Vec3& out_rel_car_contact,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch = nullptr,
    bool* out_exact = nullptr,
    Vec3* out_bt_rel_ball_contact_for_solver = nullptr,
    Vec3* out_bt_rel_car_contact_for_solver = nullptr,
    Vec3* out_ball_local_bt = nullptr,
    Vec3* out_car_local_bt = nullptr,
    float precomputed_threshold_bt = -1.0f) {
    CarCRefT car = car_arg;
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kBTToUU = 50.0f;
    constexpr float kBulletBoxMarginBt = 0.04f;

    const Vec3 raw_half_extent_bt = (BulletHitboxSize(cfg) * 0.5f) * kUUToBT;
    const bullet_port::btConvexShape box_shape =
        MakeBulletPortBoxShape(raw_half_extent_bt, kBulletBoxMarginBt);
    const bullet_port::btSphereShape sphere_shape(ball_radius * kUUToBT);
    const float margin_a_bt = box_shape.getMarginNonVirtual();
    const float margin_b_bt = sphere_shape.getMarginNonVirtual();

    const float contact_breaking_threshold_bt =
        precomputed_threshold_bt >= 0.0f ? precomputed_threshold_bt
                                         : BallCarContactBreakingThresholdBT(cfg, ball_radius);
    float maximum_distance_squared = margin_a_bt + margin_b_bt + contact_breaking_threshold_bt;
    maximum_distance_squared *= maximum_distance_squared;
    bullet_port::btTransform local_trans_a =
        MakeBulletPortBoxWorldTransform(car, car.pos_bt);
    local_trans_a.setOrigin(local_trans_a(ToBulletPortVec3(cfg.hitbox_offset * kUUToBT)));
    bullet_port::btTransform local_trans_b =
        MakeBulletPortBallWorldTransform(ball, ball.pos_bt);
    const bullet_port::btVector3 position_offset =
        (local_trans_a.getOrigin() + local_trans_b.getOrigin()) * 0.5f;
    local_trans_a.getOrigin() -= position_offset;
    local_trans_b.getOrigin() -= position_offset;
    const Vec3 position_offset_bt = FromBulletPortVec3(position_offset);

    Vec3 cached_axis(0.0f, 1.0f, 0.0f);
    float squared_distance = 1.0e30f;
    bool is_valid = false;
    int degenerate_simplex = 0;
    int last_used_method = -1;
    float distance_bt = 0.0f;
    Vec3 normal_bt(0.0f, 0.0f, 0.0f);
    Vec3 point_on_a_bt(0.0f, 0.0f, 0.0f);
    Vec3 point_on_b_bt(0.0f, 0.0f, 0.0f);

    if (!RunBallCarGjkSimplex(
            box_shape, sphere_shape, local_trans_a, local_trans_b,
            maximum_distance_squared, cached_axis, degenerate_simplex, squared_distance,
            point_on_a_bt, point_on_b_bt)) {
        return false;
    }

    const float len_sqr = LengthSq(cached_axis);
    if (len_sqr <= 0.0001f) {
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
        degenerate_simplex && ((distance_bt + (margin_a_bt + margin_b_bt)) <= 0.01f);
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

    const Vec3 point_on_ball_bt = point_on_b_bt + position_offset_bt;
    const Vec3 bullet_normal_bt = FixArenaTriangleContactNormalDirectionBullet(
        box_shape, local_trans_a, sphere_shape, local_trans_b, normal_bt);
    const Vec3 manifold_point_on_car_bt =
        point_on_ball_bt + bullet_normal_bt * distance_bt;
    if (out_exact) {
        *out_exact = true;
    }

    out_normal = -bullet_normal_bt;
    out_penetration = -distance_bt * kBTToUU;
    const Vec3 car_origin_bt = car.pos_bt;
    const Vec3 ball_origin_bt = ball.pos_bt;
    const bullet_port::btVector3 car_origin_vec = ToBulletPortVec3(car_origin_bt);
    const bullet_port::btVector3 ball_origin_vec = ToBulletPortVec3(ball_origin_bt);
    const bullet_port::btMatrix3x3& car_basis = local_trans_a.getBasis();
    const bullet_port::btMatrix3x3& ball_basis = local_trans_b.getBasis();
    const bullet_port::btVector3 point_on_car_local_bt =
        (ToBulletPortVec3(manifold_point_on_car_bt) - car_origin_vec) * car_basis;
    const bullet_port::btVector3 point_on_ball_local_bt =
        (ToBulletPortVec3(point_on_ball_bt) - ball_origin_vec) * ball_basis;
    const Vec3 refreshed_point_on_car_bt =
        FromBulletPortVec3(point_on_car_local_bt.dot3(car_basis[0], car_basis[1], car_basis[2]) + car_origin_vec);
    const Vec3 refreshed_point_on_ball_bt =
        FromBulletPortVec3(point_on_ball_local_bt.dot3(ball_basis[0], ball_basis[1], ball_basis[2]) + ball_origin_vec);
    const Vec3 bt_rel_ball_contact_for_solver = refreshed_point_on_ball_bt - ball_origin_bt;
    out_rel_ball_contact = bt_rel_ball_contact_for_solver * kBTToUU;
    out_rel_car_contact = (refreshed_point_on_car_bt - car_origin_bt) * kBTToUU;
    if (out_bt_rel_car_contact_for_solver) {
        *out_bt_rel_car_contact_for_solver = refreshed_point_on_car_bt - car_origin_bt;
    }
    if (out_bt_rel_ball_contact_for_solver) {
        *out_bt_rel_ball_contact_for_solver = bt_rel_ball_contact_for_solver;
    }
    if (out_ball_local_bt) {
        *out_ball_local_bt = FromBulletPortVec3(point_on_ball_local_bt);
    }
    if (out_car_local_bt) {
        *out_car_local_bt = FromBulletPortVec3(point_on_car_local_bt);
    }
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
    CarCRefT car,
    const CarConfig& cfg,
    Vec3& out_min,
    Vec3& out_max) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kContactBreakingThresholdBt = 0.02f;

    Vec3 fwd;
    Vec3 rgt;
    Vec3 upv;
    CarBasisColumns(car, fwd, rgt, upv);

    const Vec3 local_center_bt = cfg.hitbox_offset * kUUToBT;
    const Vec3 local_half_extents_bt = (BulletHitboxSize(cfg) * 0.5f) * kUUToBT;
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
    CarCRefT car,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    float dt) {
    BallCarRsBroadphaseProbe probe;
    ComputeBulletCarCompoundAabbBt(car, cfg, probe.car_aabb_min_bt, probe.car_aabb_max_bt);
    ComputeBulletBallSphereAabbBt(ball, mutator.ball_radius, probe.ball_aabb_min_bt, probe.ball_aabb_max_bt);

    if (dt > 0.0f) {
        CarState predicted_car = car;
        BallState predicted_ball = ball;
        IntegrateCarTransformOnly(predicted_car, dt, NativeSseMath(mutator));
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
    CarCRefT car,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    float dt,
    int env_idx = -1) {
    if (dt <= 0.0f || env_idx < 0) {
        return true;
    }
    const float reach_bt = (BoundingSphereRadius(cfg) + mutator.ball_radius) * (1.0f / 50.0f) +
                           (Length(car.vel_bt) + Length(ball.vel_bt)) * dt + 2.0f;
    const Vec3 gap_bt = ball.pos_bt - car.pos_bt;
    if (Abs(gap_bt.x) > reach_bt || Abs(gap_bt.y) > reach_bt || Abs(gap_bt.z) > reach_bt) {
        return false;
    }

    const BallCarRsBroadphaseProbe probe =
        ComputeBulletRsBroadphaseBallCarProbe(ball, car, cfg, mutator, dt);
    const bool allowed = probe.aabb_overlap && probe.dynamic_neighbor;
#if !defined(__CUDA_ARCH__)
#else
    (void)env_idx;
#endif
    return allowed;
}

ROCKETGPU_HD bool ComputeBallCarContactGeometrySphereBox(
    const BallState& ball,
    CarCRefT car,
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
    const Vec3 raw_half_extent_bt = (BulletHitboxSize(cfg) * 0.5f) * kUUToBT_BallCar;
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
    CarCRefT car,
    const CarConfig& cfg,
    float ball_radius,
    Vec3& out_normal,
    float& out_penetration,
    Vec3& out_rel_ball_contact,
    Vec3& out_rel_car_contact,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch = nullptr,
    bool* out_exact = nullptr,
    Vec3* out_bt_rel_ball_contact_for_solver = nullptr,
    Vec3* out_bt_rel_car_contact_for_solver = nullptr,
    Vec3* out_ball_local_bt = nullptr,
    Vec3* out_car_local_bt = nullptr,
    float precomputed_threshold_bt = -1.0f) {
    if (ComputeBallCarContactGeometryGjk(
            ball,
            car,
            cfg,
            ball_radius,
            out_normal,
            out_penetration,
            out_rel_ball_contact,
            out_rel_car_contact,
            gjk_epa_scratch,
            out_exact,
            out_bt_rel_ball_contact_for_solver,
            out_bt_rel_car_contact_for_solver,
            out_ball_local_bt,
            out_car_local_bt,
            precomputed_threshold_bt)) {
        return true;
    }

    return ComputeBallCarContactGeometrySphereBox(
        ball,
        car,
        cfg,
        ball_radius,
        Max(0.0f, precomputed_threshold_bt),
        out_normal,
        out_penetration,
        out_rel_ball_contact,
        out_rel_car_contact);
}

ROCKETGPU_HD bool BallCarCollisionInfoMayOverlap(
    const BallState& ball, CarCRefT car, const CarConfig& cfg, const MutatorConfig& mutator, float dt,
    int env_idx) {
    const uint8_t is_demoed = car.is_demoed;
    const uint8_t just_respawned = car.just_respawned;
    const float bounding_radius = BoundingSphereRadius(cfg);
    const float ball_radius = mutator.ball_radius;
    const Vec3 car_vel_bt = car.vel_bt;
    const Vec3 ball_vel_bt = ball.vel_bt;
    const Vec3 ball_pos_bt = ball.pos_bt;
    const Vec3 car_pos_bt = car.pos_bt;
    if (is_demoed || just_respawned) {
        return false;
    }
    if (dt <= 0.0f || env_idx < 0) {
        return true;
    }
    const float reach_bt = (bounding_radius + ball_radius) * (1.0f / 50.0f) +
                           (Length(car_vel_bt) + Length(ball_vel_bt)) * dt + 2.0f;
    const Vec3 gap_bt = ball_pos_bt - car_pos_bt;
    return !(Abs(gap_bt.x) > reach_bt || Abs(gap_bt.y) > reach_bt || Abs(gap_bt.z) > reach_bt);
}

struct BallCarHitStart {
    Vec3 car_pos = Vec3(0.0f, 0.0f, 0.0f);
    Quat car_rot = Quat(1.0f, 0.0f, 0.0f, 0.0f);
    Vec3 ball_pos = Vec3(0.0f, 0.0f, 0.0f);
};

ROCKETGPU_HD void CommitBallCarHit(
    const BallState& ball,
    CarCRefT car,
    uint32_t car_id,
    CarRefT car_mut,
    const MutatorConfig& mutator,
    uint64_t env_tick,
    BallState& ball_mut,
    const Vec3& normal,
    const Vec3& rel_ball_contact,
    const Vec3& rel_car_contact,
    int8_t car_team,
    const Vec3* cal_car_start_vel,
    const Vec3* cal_ball_start_vel,
    const BallCarHitStart* start = nullptr) {
    const Vec3 car_pos = start ? start->car_pos : car.pos;
    const Quat car_rot = start ? start->car_rot : car.rot;
    const Vec3 ball_pos = start ? start->ball_pos : ball.pos;
    const bool contact_started = !car_mut.ball_contact_last_tick && !car_mut.ball_contact_this_tick;
    if (contact_started) {
        car_mut.RecordBallContactStart();
        car_mut.latest_touch_tick = env_tick;
        car_mut.latest_touch_location = car_pos + rel_car_contact;
        car_mut.latest_touch_normal = normal;
    }
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
    car_mut.ball_hit_info.ball_pos = ball_pos;
    car_mut.ball_hit_info.extra_hit_vel = Vec3(0.0f, 0.0f, 0.0f);
    if (mutator.game_mode == static_cast<uint8_t>(Mode::Heatseeker) && car_team >= 0) {
        HeatseekerOnBallHit(ball_mut, static_cast<int>(car_team));
    }

    const bool can_apply_extra = (!had_prior_ball_hit) ||
        (car_mut.ball_hit_info.extra_tick_is_previous ? prev_extra_impulse_tick != env_tick
            : (prev_extra_impulse_tick > env_tick) || ((prev_extra_impulse_tick + 1ull) < env_tick));

    if (can_apply_extra) {
        const Vec3 rel_pos_for_extra = ball_pos - car_pos;

        const Vec3 rel_vel_for_extra =
            (cal_ball_start_vel ? *cal_ball_start_vel : ball.vel) - (cal_car_start_vel ? *cal_car_start_vel : car.vel);
        const float rel_speed = Min(Length(rel_vel_for_extra), mutator.ball_car_extra_impulse_max_delta_vel);
        if (rel_speed > 0.0f) {
            const Vec3 car_forward = Ue3RotatorForwardAxis(Ue3QuatToRotator(car_rot));

            Vec3 hit_dir = Ue3SafeNormal(rel_pos_for_extra);
            hit_dir.z *= mutator.ball_car_extra_impulse_z_scale;
            hit_dir = Ue3SafeNormal(hit_dir);
            const Vec3 forward_dir_adjustment =
                (car_forward * Dot(hit_dir, car_forward)) *
                (1.0f - mutator.ball_car_extra_impulse_forward_scale);
            hit_dir = Ue3SafeNormal(hit_dir - forward_dir_adjustment);
            const Vec3 added_vel =
                ((hit_dir * rel_speed) * EvalBallCarExtraImpulseFactor(rel_speed)) *
                mutator.ball_hit_extra_force_scale;
            car_mut.ball_hit_info.extra_pending = 1;
            ball_mut.tick_count_when_extra_impulse_applied = env_tick;
            car_mut.ball_hit_info.tick_count_when_extra_impulse_applied = env_tick;
            car_mut.ball_hit_info.extra_tick_is_previous = 0;
            car_mut.ball_hit_info.extra_hit_vel = added_vel;
        }
    }
}

ROCKETGPU_HD void RecordBallCarCollisionInfoIfOverlapping(
    const BallState& ball,
    CarCRefT car,
    uint32_t car_id,
    CarRefT car_mut,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
    uint64_t env_tick,
    BallState& ball_mut,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch = nullptr,
    float broadphase_dt = -1.0f,
    int env_idx = -1,
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
            env_idx)) {
        return;
    }

    Vec3 normal(1.0f, 0.0f, 0.0f);
    float penetration = 0.0f;
    Vec3 rel_ball_contact(0.0f, 0.0f, 0.0f);
    Vec3 rel_car_contact(0.0f, 0.0f, 0.0f);
    const float threshold_bt = BallCarContactBreakingThresholdBT(cfg, mutator.ball_radius);
    if (!ComputeBallCarContactGeometry(
            ball, car, cfg, mutator.ball_radius, normal, penetration, rel_ball_contact, rel_car_contact,
            gjk_epa_scratch, nullptr, nullptr, nullptr, nullptr, nullptr, threshold_bt)) {
        return;
    }

    CommitBallCarHit(ball, car, car_id, car_mut, mutator, env_tick, ball_mut, normal, rel_ball_contact, rel_car_contact,
                     car_team, cal_car_start_vel, cal_ball_start_vel);
}

ROCKETGPU_HD ROCKETGPU_TRIANGLE_CONTACT_NOINLINE void CommitBallCarHitFromManifold(
    const BallState& ball,
    CarCArgT car_arg,
    uint32_t car_id,
    CarArgT car_mut_arg,
    const MutatorConfig& mutator,
    uint64_t env_tick,
    BallState& ball_mut,
    const BallCarRetainedManifold& manifold,
    int8_t car_team,
    const Vec3* cal_car_start_vel,
    const Vec3* cal_ball_start_vel,
    const BallCarHitStart* start) {
    CarCRefT car = car_arg;
    CarRefT car_mut = car_mut_arg;
    if (manifold.count == 0) {
        return;
    }
    Vec3 pos_a(0.0f, 0.0f, 0.0f), pos_b(0.0f, 0.0f, 0.0f), normal(0.0f, 0.0f, 0.0f);
    for (int k = 0; k < manifold.count; ++k) {
        pos_a = pos_a + manifold.points[k].pos_a_bt;
        pos_b = pos_b + manifold.points[k].pos_b_bt;
        normal = normal + manifold.points[k].normal;
    }
    const float inv = 1.0f / static_cast<float>(manifold.count);
    CommitBallCarHit(ball, car, car_id, car_mut, mutator, env_tick, ball_mut, Ue3SafeNormal(normal * inv),
                     (pos_a * inv - ball.pos_bt) * 50.0f, (pos_b * inv - car.pos_bt) * 50.0f, car_team,
                     cal_car_start_vel, cal_ball_start_vel, start);
}

struct BallCarContactOverride {
    Vec3 normal = Vec3(0.0f, 0.0f, 1.0f);
    float penetration_uu = 0.0f;
    Vec3 bt_rel_ball = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 bt_rel_car = Vec3(0.0f, 0.0f, 0.0f);
    float distance_bt = 0.0f;
    bool has_distance_bt = false;
};

ROCKETGPU_HD ROCKETGPU_TRIANGLE_CONTACT_NOINLINE void AppendBallCarContactRows(
    BallState& ball,
    CarArgT car_arg,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
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
    const BallCarContactOverride* contact_override = nullptr) {
    CarRefT car = car_arg;
    if ((!allow_midtick_demo_contact && car.is_demoed) || car.just_respawned) {
        return;
    }

    if (!contact_override && !BulletRsBroadphaseAllowsBallCarPair(
            ball,
            car,
            cfg,
            mutator,
            split_dt,
            env_idx)) {
         return;
    }

    Vec3 normal(1.0f, 0.0f, 0.0f);
    float penetration = 0.0f;
    Vec3 rel_ball_contact(0.0f, 0.0f, 0.0f);
    Vec3 rel_car_contact(0.0f, 0.0f, 0.0f);
    bool exact_contact = false;
    Vec3 bt_rel_ball_contact_for_solver_exact(0.0f, 0.0f, 0.0f);
    Vec3 bt_rel_car_contact_for_solver_exact(0.0f, 0.0f, 0.0f);
    if (contact_override) {
        normal = contact_override->normal;
        penetration = contact_override->penetration_uu;
        rel_ball_contact = contact_override->bt_rel_ball * 50.0f;
        rel_car_contact = contact_override->bt_rel_car * 50.0f;
    } else {
        const float threshold_bt = BallCarContactBreakingThresholdBT(cfg, mutator.ball_radius);
        const bool has_contact =
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
                &exact_contact,
                &bt_rel_ball_contact_for_solver_exact,
                &bt_rel_car_contact_for_solver_exact,
                nullptr,
                nullptr,
                threshold_bt);
        if (!has_contact) {
            return;
        }
    }

    constexpr float kBtPerUuBallCar = 1.0f / 50.0f;
    const Vec3 bt_rel_ball_contact_for_solver = contact_override ? contact_override->bt_rel_ball :
        exact_contact
            ? bt_rel_ball_contact_for_solver_exact
            : UuVecToBtDiv(rel_ball_contact);

    const Vec3 bc_car_pos_bt = car.pos_bt;
    const Vec3 bt_rel_car_contact_for_solver = contact_override ? contact_override->bt_rel_car :
        exact_contact
            ? bt_rel_car_contact_for_solver_exact
            : (CarBodyWorldPointBt(
                   car, CarBodyLocalPointBt(car, bc_car_pos_bt + rel_car_contact * kBtPerUuBallCar)) -
               bc_car_pos_bt);

    const float bt_split_penetration_for_solver = (contact_override && contact_override->has_distance_bt)
        ? -contact_override->distance_bt
        : Dot((bc_car_pos_bt + bt_rel_car_contact_for_solver) - (ball.pos_bt + bt_rel_ball_contact_for_solver), normal);

    Vec3 car_inv_bt_rows[3];
    BuildInvInertiaTensorWorldBt(
        car, cfg, mutator.car_mass, car_inv_bt_rows[0], car_inv_bt_rows[1], car_inv_bt_rows[2]);

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
        &ball,
        car_inv_bt_rows);

    const Vec3 restitution_ball_linear = ball.vel;
    const Vec3 restitution_car_linear = car.vel;
    const Vec3 ball_angular = ball.ang_vel;
    const Vec3 car_angular = car.ang_vel;
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
    const Vec3 bt_car_normal_component = Mul3x3Vec(
        car_inv_bt_rows[0],
        car_inv_bt_rows[1],
        car_inv_bt_rows[2],
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
    const Vec3 bt_car_tangent_component = Mul3x3Vec(
        car_inv_bt_rows[0],
        car_inv_bt_rows[1],
        car_inv_bt_rows[2],
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
}

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

ROCKETGPU_HD void RefreshBallCarRetainedManifold(
    BallCarRetainedManifold& manifold, const BallState& ball, CarCRefT car, float threshold_bt) {
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

ROCKETGPU_HD void AddBallCarRetainedPoint(
    BallCarRetainedManifold& manifold, const BallCarRetainedPoint& point, float threshold_bt) {
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

ROCKETGPU_HD bool BallCarRetainedPairOverlaps(
    const BallState& ball, CarCRefT car, const CarConfig& cfg, const MutatorConfig& mutator,
    bool allow_midtick_demo_contact) {
    const uint8_t is_demoed = car.is_demoed;
    const uint8_t just_respawned = car.just_respawned;
    const uint8_t basis_authoritative = car.basis_authoritative;
    const Vec3 basis_forward = car.basis_forward;
    const Vec3 basis_right = car.basis_right;
    const Vec3 basis_up = car.basis_up;
    const Quat rot = car.rot;
    const Vec3 car_pos_bt = car.pos_bt;
    const Vec3 hitbox_offset = cfg.hitbox_offset;
    const Vec3 half = BulletBoxHalfExtentWithMarginBt(cfg);
    const float ball_radius = mutator.ball_radius;
    const Vec3 ball_pos_bt = ball.pos_bt;
    if ((!allow_midtick_demo_contact && is_demoed) || just_respawned) {
        return false;
    }
    constexpr float kUUToBT = 1.0f / 50.0f;
    Vec3 ax(1.0f, 0.0f, 0.0f), ay(0.0f, 1.0f, 0.0f), az(0.0f, 0.0f, 1.0f);
    if (basis_authoritative) {
        ax = basis_forward;
        ay = basis_right;
        az = basis_up;
    } else {
        GetBasisVectors(rot, ax, ay, az);
    }
    const Vec3 off = hitbox_offset * kUUToBT;
    const Vec3 center = car_pos_bt + ax * off.x + ay * off.y + az * off.z;
    const Vec3 ext(
        Abs(ax.x) * half.x + Abs(ay.x) * half.y + Abs(az.x) * half.z,
        Abs(ax.y) * half.x + Abs(ay.y) * half.y + Abs(az.y) * half.z,
        Abs(ax.z) * half.x + Abs(ay.z) * half.y + Abs(az.z) * half.z);
    const float r = ball_radius * kUUToBT + 0.08f;
    const Vec3 bmin = ball_pos_bt - Vec3(r, r, r);
    const Vec3 bmax = ball_pos_bt + Vec3(r, r, r);
    const Vec3 cmin = center - ext;
    const Vec3 cmax = center + ext;
    return !(cmin.x > bmax.x || cmax.x < bmin.x || cmin.y > bmax.y || cmax.y < bmin.y ||
             cmin.z > bmax.z || cmax.z < bmin.z);
}

ROCKETGPU_HD ROCKETGPU_TRIANGLE_CONTACT_NOINLINE void AppendBallCarContactRowsRetained(
    BallState& ball,
    CarArgT car_arg,
    const CarConfig& cfg,
    const MutatorConfig& mutator,
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
    BallCarRetainedManifold& retained) {
    CarRefT car = car_arg;
    constexpr float kUUToBT = 1.0f / 50.0f;
    const float threshold_bt = BallCarContactBreakingThresholdBT(cfg, mutator.ball_radius);

    Vec3 normal(1.0f, 0.0f, 0.0f);
    float penetration = 0.0f;
    Vec3 rel_ball_contact(0.0f, 0.0f, 0.0f);
    Vec3 rel_car_contact(0.0f, 0.0f, 0.0f);
    bool exact_contact = false;
    Vec3 bt_rel_ball_exact(0.0f, 0.0f, 0.0f);
    Vec3 bt_rel_car_exact(0.0f, 0.0f, 0.0f);
    Vec3 ball_local_exact(0.0f, 0.0f, 0.0f);
    Vec3 car_local_exact(0.0f, 0.0f, 0.0f);
    const bool has_fresh =
        ComputeBallCarContactGeometry(
            ball, car, cfg, mutator.ball_radius, normal, penetration, rel_ball_contact, rel_car_contact,
            gjk_epa_scratch, &exact_contact, &bt_rel_ball_exact, &bt_rel_car_exact, &ball_local_exact,
            &car_local_exact, threshold_bt);

    RefreshBallCarRetainedManifold(retained, ball, car, threshold_bt);
    if (has_fresh) {
        const bool exact = exact_contact;
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
            point.local_a_bt = exact ? ball_local_exact : BallCarRetainedBallLocalPointBt(ball, pos_a);
            point.local_b_bt = exact ? car_local_exact : CarBodyLocalPointBt(car, pos_b);
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
        ov.distance_bt = point.distance_bt;
        ov.has_distance_bt = true;
        AppendBallCarContactRows(
            ball, car, cfg, mutator, ball_external_linear_delta, external_linear_delta,
            external_angular_delta, normal_rows, normal_row_count, friction_rows, friction_row_count, split_rows,
            split_row_count, other_local_idx, split_dt, env_idx, gjk_epa_scratch, allow_midtick_demo_contact,
            bt_ball_linear_velocity_override, &ov);
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

constexpr int kMaxApproxCarCarContactPoints = 4;
constexpr float kLargeFaceClipFloat = 1.0e18f;

ROCKETGPU_HD float ObbHalfExtentComponent(const ObbData& obb, int idx) {
    return (&obb.half_extent.x)[idx];
}

ROCKETGPU_HD ObbData MakeCarObbBulletTransformBt(CarCRefT car, const CarConfig& cfg) {
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
        const float scale = Abs(denom) > 1.1920929e-7f ? (1.0f / (3.0f * denom)) : kLargeFaceClipFloat;
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
    int ref_is_a = -1,
    Vec3* out_game_points = nullptr,
    float* out_game_depths = nullptr) {
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
        out_inc_points[i] = ref_is_a ? incident_point : (ref_point + ref_normal * (-depths[idx]));
        if (out_game_points) {
            out_game_points[i] = ref_is_a ? ref_point : incident_point;
        }
        if (out_game_depths) {
            out_game_depths[i] = -depths[idx];
        }
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
    Vec3& out_normal,
    Vec3& out_point_a,
    Vec3& out_point_b,
    float* out_game_s = nullptr) {
    if (axis_a_idx < 0 || axis_a_idx > 2 || axis_b_idx < 0 || axis_b_idx > 2) {
        return false;
    }
    const ObbData& box1 = b;
    const ObbData& box2 = a;
    const int i = axis_b_idx;
    const int j = axis_a_idx;
    const Vec3 p = box2.center - box1.center;
    const float pp[3] = {Dot(box1.axis[0], p), Dot(box1.axis[1], p), Dot(box1.axis[2], p)};
    float r[3];
    for (int k = 0; k < 3; ++k) {
        r[k] = Dot(box1.axis[k], box2.axis[j]);
    }
    float expr1;
    Vec3 n_c;
    if (i == 0) {
        expr1 = pp[2] * r[1] - pp[1] * r[2];
        n_c = Vec3(0.0f, -r[2], r[1]);
    } else if (i == 1) {
        expr1 = pp[0] * r[2] - pp[2] * r[0];
        n_c = Vec3(r[2], 0.0f, -r[0]);
    } else {
        expr1 = pp[1] * r[0] - pp[0] * r[1];
        n_c = Vec3(-r[1], r[0], 0.0f);
    }
    const float l = SqrtScalar(n_c.x * n_c.x + n_c.y * n_c.y + n_c.z * n_c.z);
    if (!(l > 1.1920929e-07f)) {
        return false;
    }
    if (out_game_s) {
        constexpr float kFudge2 = 1.0e-5f;
        const int lo = i == 0 ? 1 : 0;
        const int hi = i == 2 ? 1 : 2;
        const int k1 = j == 0 ? 1 : 0;
        const int k2 = j == 2 ? 1 : 2;
        const float q_hi_j = Abs(r[hi]) + kFudge2;
        const float q_lo_j = Abs(r[lo]) + kFudge2;
        const float q_i_k2 = Abs(Dot(box1.axis[i], box2.axis[k2])) + kFudge2;
        const float q_i_k1 = Abs(Dot(box1.axis[i], box2.axis[k1])) + kFudge2;
        const float expr2 = ((ObbHalfExtentComponent(box1, lo) * q_hi_j + ObbHalfExtentComponent(box1, hi) * q_lo_j) +
                             ObbHalfExtentComponent(box2, k1) * q_i_k2) +
                            ObbHalfExtentComponent(box2, k2) * q_i_k1;
        *out_game_s = (Abs(expr1) - expr2) / l;
    }
    n_c = Vec3(n_c.x / l, n_c.y / l, n_c.z / l);
    Vec3 normal(
        box1.axis[0].x * n_c.x + box1.axis[1].x * n_c.y + box1.axis[2].x * n_c.z,
        box1.axis[0].y * n_c.x + box1.axis[1].y * n_c.y + box1.axis[2].y * n_c.z,
        box1.axis[0].z * n_c.x + box1.axis[1].z * n_c.y + box1.axis[2].z * n_c.z);
    if (expr1 < 0.0f) {
        normal = Vec3(-normal.x, -normal.y, -normal.z);
    }

    Vec3 pa = box1.center;
    for (int k = 0; k < 3; ++k) {
        const float sign = Dot(normal, box1.axis[k]) > 0.0f ? 1.0f : -1.0f;
        pa += box1.axis[k] * (sign * ObbHalfExtentComponent(box1, k));
    }
    Vec3 pb = box2.center;
    for (int k = 0; k < 3; ++k) {
        const float sign = Dot(normal, box2.axis[k]) > 0.0f ? -1.0f : 1.0f;
        pb += box2.axis[k] * (sign * ObbHalfExtentComponent(box2, k));
    }
    const Vec3 ua = box1.axis[i];
    const Vec3 ub = box2.axis[j];
    float alpha = 0.0f;
    float beta = 0.0f;
    ComputeLineClosestApproachParams(pa, ua, pb, ub, alpha, beta);
    out_normal = normal;
    out_point_a = pb + ub * beta;
    out_point_b = pa + ua * alpha;
    return true;
}

ROCKETGPU_HD ObbData MakeCarObb(CarCRefT car, const CarConfig& cfg) {
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
    ObbContactFeature* out_feature = nullptr) {
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
    constexpr float kEdgeSepEps = 1.1920929e-7f;
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

ROCKETGPU_HD bool ComputeObbContactGameOrder(
    const ObbData& a,
    const ObbData& b,
    float contact_threshold_uu,
    Vec3& out_normal,
    float& out_signed_overlap,
    ObbContactFeature* out_feature = nullptr) {
    Vec3 normal_ba;
    ObbContactFeature feature_ba;
    if (!ComputeObbContact(b, a, contact_threshold_uu, normal_ba, out_signed_overlap, &feature_ba)) {
        return false;
    }
    out_normal = -normal_ba;
    if (out_feature) {
        ObbContactFeature f = feature_ba;
        if (feature_ba.kind == OBB_CONTACT_FEATURE_FACE_A) {
            f.kind = OBB_CONTACT_FEATURE_FACE_B;
        } else if (feature_ba.kind == OBB_CONTACT_FEATURE_FACE_B) {
            f.kind = OBB_CONTACT_FEATURE_FACE_A;
        }
        f.axis_a = feature_ba.axis_b;
        f.axis_b = feature_ba.axis_a;
        *out_feature = f;
    }
    return true;
}

ROCKETGPU_HD bool InForwardConePose(
    const Vec3& a_pos, const Quat& a_rot, const CarConfig& a_cfg,
    const Vec3& v_pos, const Quat& v_rot, const CarConfig& v_cfg,
    bool reverse, float max_yaw_deg);

ROCKETGPU_HD bool InForwardCone(
    CarCRefT a, const CarConfig& a_cfg, CarCRefT v, const CarConfig& v_cfg, bool reverse, float max_yaw_deg);

struct CarToiTransform {
    Vec3 fwd, right, up, origin;
};

ROCKETGPU_HD bullet_port::btTransform CarToiBt(const CarToiTransform& t) {
    bullet_port::btTransform out;
    out.setBasis(bullet_port::btMatrix3x3(
        bullet_port::btVector3(t.fwd.x, t.right.x, t.up.x),
        bullet_port::btVector3(t.fwd.y, t.right.y, t.up.y),
        bullet_port::btVector3(t.fwd.z, t.right.z, t.up.z)));
    out.setOrigin(ToBulletPortVec3(t.origin));
    return out;
}

ROCKETGPU_HD CarToiTransform CarToiIntegrate(const CarToiTransform& t, const Vec3& lin, const Vec3& ang, float dt) {
    CarToiTransform out = t;
    out.origin = Vec3(dt * lin.x + t.origin.x, dt * lin.y + t.origin.y, dt * lin.z + t.origin.z);
    Quat q;
    GameIntegrateOrientationBasisColumns(out.fwd, out.right, out.up, q, ang, dt);
    return out;
}

ROCKETGPU_HD Vec3 CarToiAngularVelocity(const CarToiTransform& t0, const CarToiTransform& t1) {
    const float m00 = t0.fwd.x, m01 = t0.right.x, m02 = t0.up.x;
    const float m10 = t0.fwd.y, m11 = t0.right.y, m12 = t0.up.y;
    const float m20 = t0.fwd.z, m21 = t0.right.z, m22 = t0.up.z;
    const float co0 = m11 * m22 - m12 * m21, co1 = m12 * m20 - m10 * m22, co2 = m10 * m21 - m11 * m20;
    const float s = 1.0f / ((m00 * co0 + m01 * co1) + m02 * co2);
    const float i00 = co0 * s, i01 = (m02 * m21 - m01 * m22) * s, i02 = (m01 * m12 - m02 * m11) * s;
    const float i10 = co1 * s, i11 = (m00 * m22 - m02 * m20) * s, i12 = (m02 * m10 - m00 * m12) * s;
    const float i20 = co2 * s, i21 = (m01 * m20 - m00 * m21) * s, i22 = (m00 * m11 - m01 * m10) * s;
    auto row = [&](float b0, float b1, float b2) {
        return Vec3((i00 * b0 + i10 * b1) + i20 * b2, (i01 * b0 + i11 * b1) + i21 * b2, (i02 * b0 + i12 * b1) + i22 * b2);
    };
    const Vec3 d0 = row(t1.fwd.x, t1.right.x, t1.up.x);
    const Vec3 d1 = row(t1.fwd.y, t1.right.y, t1.up.y);
    const Vec3 d2 = row(t1.fwd.z, t1.right.z, t1.up.z);
    const Quat q = GameQuatNormalizeScalar(
        GameGetRotationScalar(Vec3(d0.x, d1.x, d2.x), Vec3(d0.y, d1.y, d2.y), Vec3(d0.z, d1.z, d2.z)), -1.0f);
    float w = q.w;
    w = -1.0f <= w ? w : -1.0f;
    w = w <= 1.0f ? w : 1.0f;
    const float half = RocketGpuUcrtAcosf(w);
    const float angle = half + half;
    const float len2 = (q.x * q.x + q.y * q.y) + q.z * q.z;
    Vec3 axis(1.0f, 0.0f, 0.0f);
    if (len2 >= 1.42108547e-14f) {
        const float inv = 1.0f / sqrtf(len2);
        axis = Vec3(inv * q.x, q.y * inv, inv * q.z);
    }
    return Vec3(axis.x * angle, axis.y * angle, axis.z * angle);
}

ROCKETGPU_HD float CarToiAngularMotionDisc(const bullet_port::btConvexShape& box) {
    const float m = box.getMarginNonVirtual();
    const float ex = box.m_implicitShapeDimensions.x() + m;
    const float ey = box.m_implicitShapeDimensions.y() + m;
    const float ez = box.m_implicitShapeDimensions.z() + m;
    const float dx = ex - -ex, dy = ey - -ey, dz = ez - -ez;
    return sqrtf((dx * dx + dy * dy) + dz * dz) * 0.5f;
}

ROCKETGPU_HD Quat GameUeQuatSlerp(const Quat& q1, const Quat& q2, float t) {
    const float raw = ((q1.x * q2.x + q1.y * q2.y) + q1.z * q2.z) + q1.w * q2.w;
    const float cosom = raw >= 0.0f ? raw : -raw;
    float scale0, scale1;
    if (cosom < 0.9999f) {
        const float c = -1.0f > cosom ? -1.0f : (cosom < 1.0f ? cosom : 1.0f);
        const float omega = RocketGpuUcrtAcosf(c);
        const float inv_sin = 1.0f / RocketGpuUcrtSinf(omega);
        scale0 = RocketGpuUcrtSinf((1.0f - t) * omega) * inv_sin;
        scale1 = RocketGpuUcrtSinf(omega * t) * inv_sin;
    } else {
        scale0 = 1.0f - t;
        scale1 = t;
    }
    if (!(raw >= 0.0f)) {
        scale1 = -scale1;
    }
    return Quat(scale1 * q2.w + q1.w * scale0, scale1 * q2.x + q1.x * scale0, scale1 * q2.y + q1.y * scale0,
                scale1 * q2.z + q1.z * scale0);
}

ROCKETGPU_HD ROCKETGPU_TRIANGLE_CONTACT_NOINLINE bool RunCarToiGjkSimplex(
    const bullet_port::btConvexShape& shape_a,
    const bullet_port::btTransform& local_trans_a,
    const bullet_port::btConvexShape& shape_b,
    const bullet_port::btTransform& local_trans_b,
    Vec3& cached_axis,
    float& squared_distance,
    int& degenerate_simplex,
    Vec3& point_on_a,
    Vec3& point_on_b) {
    constexpr float kRelError2 = 1.0e-6f;
    constexpr int kGjkMaxIter = 1000;
    constexpr float kMaximumDistanceSquared = 1.0e18f;
    GjkSimplexSolver simplex;
    simplex.reset();
    bool check_simplex = false;
    int cur_iter = 0;

    while (true) {
        const bullet_port::btVector3 axis_bt = ToBulletPortVec3(cached_axis);
        const bullet_port::btVector3 p_in_a =
            shape_a.localGetSupportVertexWithoutMarginNonVirtual((-axis_bt) * local_trans_a.getBasis());
        const bullet_port::btVector3 q_in_b =
            shape_b.localGetSupportVertexWithoutMarginNonVirtual(axis_bt * local_trans_b.getBasis());
        const Vec3 p_world = FromBulletPortVec3(local_trans_a(p_in_a));
        const Vec3 q_world = FromBulletPortVec3(local_trans_b(q_in_b));
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
        return false;
    }
    simplex.compute_points(point_on_a, point_on_b);
    return true;
}

template <bool kRequireScratch = false>
ROCKETGPU_HD ROCKETGPU_TRIANGLE_CONTACT_NOINLINE bool CarToiClosestPoints(
    const bullet_port::btConvexShape& shape_a, const CarToiTransform& ca,
    const bullet_port::btConvexShape& shape_b, const CarToiTransform& cb,
    float& out_distance, Vec3& out_normal, Vec3& out_point,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch = nullptr) {
    constexpr float kMaximumDistanceSquared = 1.0e18f;
    const float margin_a = shape_a.getMarginNonVirtual();
    const float margin_b = shape_b.getMarginNonVirtual();

    bullet_port::btTransform local_trans_a = CarToiBt(ca);
    bullet_port::btTransform local_trans_b = CarToiBt(cb);
    const bullet_port::btVector3 position_offset = (local_trans_a.getOrigin() + local_trans_b.getOrigin()) * 0.5f;
    local_trans_a.getOrigin() -= position_offset;
    local_trans_b.getOrigin() -= position_offset;

    Vec3 cached_axis(0.0f, 1.0f, 0.0f);
    float squared_distance = 1.0e30f;
    bool is_valid = false;
    int degenerate_simplex = 0;
    float distance = 0.0f;
    Vec3 normal(0.0f, 0.0f, 0.0f);
    Vec3 point_on_a(0.0f, 0.0f, 0.0f);
    Vec3 point_on_b(0.0f, 0.0f, 0.0f);

    if (!RunCarToiGjkSimplex(
            shape_a, local_trans_a, shape_b, local_trans_b,
            cached_axis, squared_distance, degenerate_simplex, point_on_a, point_on_b)) {
        return false;
    }
    const float len_sqr = LengthSq(cached_axis);
    if (len_sqr <= 0.0001f) {
        degenerate_simplex = 5;
    }
    const float s = SqrtScalar(squared_distance);
    if (len_sqr > 1.1920929e-7f * 1.1920929e-7f) {
        const float inv_len = 1.0f / SqrtScalar(len_sqr);
        normal = cached_axis * inv_len;
        if (s > 0.0f) {
            point_on_a -= cached_axis * (margin_a / s);
            point_on_b += cached_axis * (margin_b / s);
            distance = (1.0f / inv_len) - (margin_a + margin_b);
            is_valid = true;
        }
    }

    if (!is_valid || (degenerate_simplex && ((distance + (margin_a + margin_b)) <= 0.01f))) {
        bullet_port::btVector3 sep_axis(0.0f, 0.0f, 0.0f);
        bullet_port::btVector3 tmp_a(0.0f, 0.0f, 0.0f);
        bullet_port::btVector3 tmp_b(0.0f, 0.0f, 0.0f);
        bool pen_ok = false;
        if constexpr (kRequireScratch) {
            pen_ok = BulletPortCalcPenDepth(shape_a, local_trans_a, shape_b, local_trans_b, sep_axis, tmp_a, tmp_b, gjk_epa_scratch);
        } else {
            pen_ok = gjk_epa_scratch != nullptr
                ? BulletPortCalcPenDepth(shape_a, local_trans_a, shape_b, local_trans_b, sep_axis, tmp_a, tmp_b, gjk_epa_scratch)
                : BulletPortCalcPenDepthLocalScratch(shape_a, local_trans_a, shape_b, local_trans_b, sep_axis, tmp_a, tmp_b);
        }
        if (pen_ok) {
            bullet_port::btVector3 tmp_normal = tmp_b - tmp_a;
            float len_sqr_pen = tmp_normal.length2();
            if (len_sqr_pen <= 1.1920929e-7f * 1.1920929e-7f) {
                tmp_normal = sep_axis;
                len_sqr_pen = sep_axis.length2();
            }
            if (len_sqr_pen > 1.1920929e-7f * 1.1920929e-7f) {
                tmp_normal /= SqrtScalar(len_sqr_pen);
                const float distance2 = -Length(FromBulletPortVec3(tmp_a - tmp_b));
                if (!is_valid || distance2 < distance) {
                    distance = distance2;
                    point_on_a = FromBulletPortVec3(tmp_a);
                    point_on_b = FromBulletPortVec3(tmp_b);
                    normal = FromBulletPortVec3(tmp_normal);
                    is_valid = true;
                }
            }
        } else if (sep_axis.length2() > 0.0f) {
            const float distance2 = Length(FromBulletPortVec3(tmp_a - tmp_b)) - (margin_a + margin_b);
            if (!is_valid || distance2 < distance) {
                point_on_a = FromBulletPortVec3(tmp_a - sep_axis * margin_a);
                point_on_b = FromBulletPortVec3(tmp_b + sep_axis * margin_b);
                normal = Normalize(FromBulletPortVec3(sep_axis), normal);
                distance = distance2;
                is_valid = true;
            }
        }
    }
    if (!is_valid || !((distance < 0.0f) || (distance * distance < kMaximumDistanceSquared))) {
        return false;
    }
    out_normal = FixArenaTriangleContactNormalDirectionBullet(shape_a, local_trans_a, shape_b, local_trans_b, normal);
    out_point = point_on_b + FromBulletPortVec3(position_offset);
    out_distance = distance;
    return distance < 1.0e18f;
}

template <bool kRequireScratch = false>
ROCKETGPU_HD void CarCarTimeOfImpact(CarCRefT a, const CarConfig& a_cfg, const Vec3& a_vel,
                                   CarCRefT v, const CarConfig& v_cfg, const Vec3& v_vel, float tick_dt,
                                   Vec3& pos_a, Quat& rot_a, Vec3& pos_v, Quat& rot_v,
                                   bullet_port::btGjkEpaScratch* gjk_epa_scratch = nullptr) {
    constexpr float kAllowedPenetration = 0.04f;
    constexpr float kEpsilon = 1.1920929e-7f;
    constexpr float kUUToBT = 1.0f / 50.0f;
    auto sweep = [&](CarCRefT s, const Vec3& vel, CarToiTransform& from, CarToiTransform& to) {
        CarToiTransform t0;
        GameSetRotationScalarColumns(s.rot, t0.fwd, t0.right, t0.up);
        t0.origin = Vec3(s.pos.x * kUUToBT, s.pos.y * kUUToBT, s.pos.z * kUUToBT);
        const Vec3 lin(vel.x * kUUToBT, vel.y * kUUToBT, vel.z * kUUToBT);
        from = CarToiIntegrate(t0, Vec3(-lin.x, -lin.y, -lin.z), Vec3(-s.ang_vel.x, -s.ang_vel.y, -s.ang_vel.z), tick_dt);
        to = CarToiIntegrate(t0, lin, s.ang_vel, tick_dt);
    };
    auto child = [&](const CarToiTransform& t, const CarConfig& cfg) {
        CarToiTransform c = t;
        c.origin = FromBulletPortVec3(CarToiBt(t)(ToBulletPortVec3(cfg.hitbox_offset * kUUToBT)));
        return c;
    };
    const bullet_port::btConvexShape box_a = MakeBulletPortBoxShape((BulletHitboxSize(a_cfg) * 0.5f) * kUUToBT, 0.04f);
    const bullet_port::btConvexShape box_b = MakeBulletPortBoxShape((BulletHitboxSize(v_cfg) * 0.5f) * kUUToBT, 0.04f);
    CarToiTransform ca0, cb0;
    Vec3 lin_a, lin_b, ang_a, ang_b;
    {
        CarToiTransform from_a, to_a, from_b, to_b;
        sweep(a, a_vel, from_a, to_a);
        sweep(v, v_vel, from_b, to_b);
        ca0 = child(from_a, a_cfg);
        const CarToiTransform ca1 = child(to_a, a_cfg);
        cb0 = child(from_b, v_cfg);
        const CarToiTransform cb1 = child(to_b, v_cfg);
        lin_a = Vec3(ca1.origin.x - ca0.origin.x, ca1.origin.y - ca0.origin.y, ca1.origin.z - ca0.origin.z);
        lin_b = Vec3(cb1.origin.x - cb0.origin.x, cb1.origin.y - cb0.origin.y, cb1.origin.z - cb0.origin.z);
        ang_a = CarToiAngularVelocity(ca0, ca1);
        ang_b = CarToiAngularVelocity(cb0, cb1);
    }
    const float max_ang = sqrtf((ang_a.x * ang_a.x + ang_a.y * ang_a.y) + ang_a.z * ang_a.z) * CarToiAngularMotionDisc(box_a) +
                          sqrtf((ang_b.x * ang_b.x + ang_b.y * ang_b.y) + ang_b.z * ang_b.z) * CarToiAngularMotionDisc(box_b);
    const Vec3 rel(lin_b.x - lin_a.x, lin_b.y - lin_a.y, lin_b.z - lin_a.z);
    if (sqrtf((rel.x * rel.x + rel.y * rel.y) + rel.z * rel.z) + max_ang == 0.0f) {
        return;
    }
    float d = 0.0f;
    Vec3 n(0.0f, 0.0f, 0.0f), c(0.0f, 0.0f, 0.0f);
    if (!CarToiClosestPoints<kRequireScratch>(box_a, ca0, box_b, cb0, d, n, c, gjk_epa_scratch)) {
        return;
    }
    float dist = d + kAllowedPenetration;
    if (((rel.x * n.x + rel.y * n.y) + rel.z * n.z) + max_ang <= kEpsilon) {
        return;
    }
    float lambda = 0.0f, fraction = 0.0f;
    int iter = 0;
    while (dist > 0.001f) {
        const float proj = ((rel.x * n.x + rel.y * n.y) + rel.z * n.z) + max_ang;
        if (proj <= kEpsilon) {
            return;
        }
        fraction = lambda + dist / proj;
        if (fraction > 1.0f || fraction < 0.0f || fraction <= lambda) {
            return;
        }
        const CarToiTransform ia = CarToiIntegrate(ca0, lin_a, ang_a, fraction);
        const CarToiTransform ib = CarToiIntegrate(cb0, lin_b, ang_b, fraction);
        if (!CarToiClosestPoints<kRequireScratch>(box_a, ia, box_b, ib, d, n, c, gjk_epa_scratch)) {
            return;
        }
        dist = d + kAllowedPenetration;
        if (++iter > 64) {
            return;
        }
        lambda = fraction;
    }
    if (!(fraction < 1.0f)) {
        return;
    }

    auto pose = [&](Vec3& pos, Quat& rot, CarCRefT s, const Vec3& vel) {
        CarToiTransform from, to;
        sweep(s, vel, from, to);
        pos = Vec3((to.origin.x * 50.0f - from.origin.x * 50.0f) * fraction + from.origin.x * 50.0f,
                   (to.origin.y * 50.0f - from.origin.y * 50.0f) * fraction + from.origin.y * 50.0f,
                   (to.origin.z * 50.0f - from.origin.z * 50.0f) * fraction + from.origin.z * 50.0f);
        rot = GameQuatNormalizeScalar(
            GameUeQuatSlerp(GameGetRotationScalar(from.fwd, from.right, from.up),
                            GameGetRotationScalar(to.fwd, to.right, to.up), fraction),
            1e-8f);
    };
    pose(pos_a, rot_a, a, a_vel);
    pose(pos_v, rot_v, v, v_vel);
}

template <bool kRequireScratch = false>
ROCKETGPU_HD ROCKETGPU_TRIANGLE_CONTACT_NOINLINE bool ShouldDemolish(
    CarCArgT a_arg, const CarConfig& a_cfg, const Vec3& a_start_vel,
    CarCArgT v_arg, const CarConfig& v_cfg, const MutatorConfig& mutator,
    const Vec3* v_start_vel = nullptr, float tick_dt = 0.0f,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch = nullptr) {
    CarCRefT a = a_arg;
    CarCRefT v = v_arg;
    if (!a.is_supersonic) {
        return false;
    }
    const Vec3 fwd = Ue3RotatorForwardAxis(Ue3QuatToRotator(a.rot));
    const float forward_speed = Dot(a_start_vel, fwd);
    const bool reverse = forward_speed < 0.0f;
    if (fabsf(forward_speed) < mutator.supersonic_maintain_min_speed) {
        return false;
    }
    if (v_start_vel != nullptr) {
        Vec3 pos_a = a.pos, pos_v = v.pos;
        Quat rot_a = a.rot, rot_v = v.rot;
        CarCarTimeOfImpact<kRequireScratch>(a, a_cfg, a_start_vel, v, v_cfg, *v_start_vel, tick_dt, pos_a, rot_a, pos_v, rot_v, gjk_epa_scratch);
        return InForwardConePose(pos_a, rot_a, a_cfg, pos_v, rot_v, v_cfg, reverse, 45.5729942f);
    }
    return InForwardConePose(a.pos, a.rot, a_cfg, v.pos, v.rot, v_cfg, reverse, 45.5729942f);
}

ROCKETGPU_HD bool InForwardConePose(
    const Vec3& a_pos, const Quat& a_rot, const CarConfig& a_cfg,
    const Vec3& v_pos, const Quat& v_rot, const CarConfig& v_cfg,
    bool reverse, float max_yaw_deg) {
    Vec3 fwd, right, up, v_x, v_y, v_z;
    Ue3RotatorAxes(Ue3QuatToRotator(a_rot), fwd, right, up);
    Ue3RotatorAxes(Ue3QuatToRotator(v_rot), v_x, v_y, v_z);
    const Vec3 a_off = a_cfg.hitbox_offset, v_off = v_cfg.hitbox_offset;
    const Vec3 pa = a_pos + Vec3(Dot(fwd, a_off), Dot(right, a_off), Dot(up, a_off));
    const Vec3 pv = v_pos + Vec3(Dot(v_x, v_off), Dot(v_y, v_off), Dot(v_z, v_off));
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
        return RocketGpuUcrtAcosf(c) * 57.29578f;
    };
    if (angle_deg(right) > 36.8698959f) {
        return false;
    }
    return angle_deg(up) <= max_yaw_deg;
}

ROCKETGPU_HD bool InForwardCone(
    CarCRefT a, const CarConfig& a_cfg, CarCRefT v, const CarConfig& v_cfg, bool reverse, float max_yaw_deg) {
    return InForwardConePose(a.pos, a.rot, a_cfg, v.pos, v.rot, v_cfg, reverse, max_yaw_deg);
}

ROCKETGPU_HD void RecordCarHit(CarRefT hitter, uint32_t hitter_id, uint32_t victim_id, bool is_demo) {
    const uint32_t k = ((victim_id - hitter_id) & 7u) - 1u;
    if (is_demo) {
        hitter.bump_event_demos[k] = static_cast<uint8_t>(hitter.bump_event_demos[k] + 1u);
    } else {
        hitter.bump_event_bumps[k] = static_cast<uint8_t>(hitter.bump_event_bumps[k] + 1u);
    }
}

ROCKETGPU_HD bool BumpCone(
    CarRefT hitter, uint32_t hitter_id, Team hitter_team, const CarConfig& hitter_cfg, const Vec3& hitter_start_vel,
    CarRefT victim, uint32_t victim_id, Team victim_team, const CarConfig& victim_cfg,
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
        const Vec3 vel_dir = hitter_start_vel * (1.0f / speed);
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
            const Vec3 up_dir = Ue3RotatorUpAxis(Ue3QuatToRotator(victim.rot));
            impulse = vel_dir * curve(150.0f, 200000.0f, 275000.0f) + up_dir * curve(50.0f, 50000.0f, 75000.0f);
        } else {
            impulse = vel_dir * curve(150.0f, 250000.0f, 350000.0f);
        }

        const Vec3 impulse_bt = ((impulse * mutator.bump_force_scale) * (1.0f / 50.0f)) * (1.0f / mutator.car_mass);
        victim.vel_impulse_cache_bt += impulse_bt;
        victim.vel_impulse_cache += impulse_bt * 50.0f;
        if (victim.pending_bump_count < CarState::kMaxPendingBumps) {
            int slot = victim.pending_bump_count++;
            for (; slot > 0 && victim.pending_bump_hitter[slot - 1] > hitter_id; --slot) {
                victim.pending_bump_bt[slot] = victim.pending_bump_bt[slot - 1];
                victim.pending_bump_hitter[slot] = victim.pending_bump_hitter[slot - 1];
            }
            victim.pending_bump_bt[slot] = impulse_bt;
            victim.pending_bump_hitter[slot] = hitter_id;
        }
    }
    hitter.car_contact.other_car_id = victim_id;
    hitter.car_contact.cooldown_timer = mutator.bump_cooldown_time + tick_dt * 0.5f;
    if (is_demo || bumper) {
        RecordCarHit(hitter, hitter_id, victim_id, is_demo);
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

template <bool kRequireScratch = false>
ROCKETGPU_HD ROCKETGPU_TRIANGLE_CONTACT_NOINLINE bool ResolveCarCarDirectionalContact(
    CarArgT hitter_arg,
    uint32_t hitter_id,
    Team hitter_team,
    const CarConfig& hitter_cfg,
    CarArgT victim_arg,
    uint32_t victim_id,
    Team victim_team,
    const MutatorConfig& mutator,
    uint8_t enable_demos,
    const Vec3& fallback_dir,
    bool has_bumper_local_x_override,
    float bumper_local_x_override,
    const Vec3* hitter_start_vel = nullptr,
    const CarConfig* victim_cfg = nullptr,
    bool demo_only = false,
    const Vec3* victim_start_vel = nullptr,
    float tick_dt = 0.0f,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch = nullptr) {
    CarRefT hitter = hitter_arg;
    CarRefT victim = victim_arg;
    (void)hitter_id;

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
        const Vec3 vel_dir = vel * (1.0f / speed);
        if (Dot(*victim_start_vel, vel_dir) >= speed_towards) {
            return false;
        }
        if (cal_demo) {
            bool cal_is_demo = enable_demos &&
                ShouldDemolish<kRequireScratch>(hitter, hitter_cfg, vel, victim, *victim_cfg, mutator, victim_start_vel, tick_dt, gjk_epa_scratch);
            if (cal_is_demo && !mutator.enable_team_demos) {
                cal_is_demo = hitter_team != victim_team;
            }
            if (cal_is_demo) {
                DemolishCar(victim, mutator);
                RecordCarHit(hitter, hitter_id, victim_id, true);
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
            ShouldDemolish<kRequireScratch>(hitter, hitter_cfg, *hitter_start_vel, victim, *victim_cfg, mutator, victim_start_vel, tick_dt, gjk_epa_scratch);
        if (cal_is_demo && !mutator.enable_team_demos) {
            cal_is_demo = hitter_team != victim_team;
        }
        if (cal_is_demo) {
            DemolishCar(victim, mutator);
            RecordCarHit(hitter, hitter_id, victim_id, true);
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
        if (in_cooldown) {
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

    RecordCarHit(hitter, hitter_id, victim_id, is_demo);
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

ROCKETGPU_HD ROCKETGPU_TRIANGLE_CONTACT_NOINLINE int ComputeApproxCarCarContactAnchorPairsWithBtObb(
    CarCArgT a_arg,
    const CarConfig& cfg_a,
    const ObbData& obb_a_bt,
    CarCArgT b_arg,
    const CarConfig& cfg_b,
    const ObbData& obb_b_bt,
    const Vec3& normal,
    const ObbContactFeature& feature,
    Vec3* out_points_a,
    Vec3* out_points_b,
    int max_points,
    float contact_overlap_uu = 0.0f,
    Vec3* out_points_a_bt = nullptr,
    Vec3* out_points_b_bt = nullptr,
    bool game_face_contacts = false,
    Vec3* out_game_points_a_bt = nullptr,
    float* out_game_depths_bt = nullptr,
    bool* out_game_points_valid = nullptr,
    Vec3* out_game_normal_bt = nullptr) {
    CarCRefT a = a_arg;
    CarCRefT b = b_arg;
    if (out_game_points_valid) {
        *out_game_points_valid = false;
    }
    if (!out_points_a || !out_points_b || max_points <= 0) {
        return 0;
    }

    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kBTToUU = 50.0f;

    if (feature.kind == OBB_CONTACT_FEATURE_FACE_A || feature.kind == OBB_CONTACT_FEATURE_FACE_B) {
        const bool is_face_a = (feature.kind == OBB_CONTACT_FEATURE_FACE_A);
        Vec3 ref_points_bt[kMaxApproxCarCarContactPoints];
        Vec3 inc_points_bt[kMaxApproxCarCarContactPoints];
        const int pair_count_bt = ComputeFaceClippedCarCarContactAnchorPairs(
            is_face_a ? obb_a_bt : obb_b_bt,
            is_face_a ? feature.axis_a : feature.axis_b,
            is_face_a ? obb_b_bt : obb_a_bt,
            is_face_a ? normal : -normal,
            ref_points_bt,
            inc_points_bt,
            max_points,
            is_face_a ? 1 : 0,
            out_game_points_a_bt,
            out_game_depths_bt);
        if (pair_count_bt > 0) {
            if (out_game_points_valid) {
                *out_game_points_valid = out_game_points_a_bt && out_game_depths_bt && out_game_normal_bt;
            }
            if (out_game_normal_bt) {
                *out_game_normal_bt = normal;
            }
            for (int i = 0; i < pair_count_bt; ++i) {
                const Vec3 point_a_bt = CarBodyManifoldWorldPointBt(a, is_face_a ? ref_points_bt[i] : inc_points_bt[i]);
                const Vec3 point_b_bt = CarBodyManifoldWorldPointBt(b, is_face_a ? inc_points_bt[i] : ref_points_bt[i]);
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
    if (game_face_contacts && feature.kind != OBB_CONTACT_FEATURE_EDGE_EDGE) {
        return 0;
    }

    if (feature.kind == OBB_CONTACT_FEATURE_EDGE_EDGE) {
        Vec3 point_a_bt;
        Vec3 point_b_bt;
        Vec3 game_normal_bt;
        float game_s_bt = 0.0f;
        if (ComputeEdgeEdgeCarCarContactAnchorsBulletStyle(
                obb_a_bt,
                static_cast<int>(feature.axis_a),
                obb_b_bt,
                static_cast<int>(feature.axis_b),
                game_normal_bt,
                point_a_bt,
                point_b_bt,
                &game_s_bt)) {
            float overlap_bt = contact_overlap_uu * kUUToBT;
            {
                Vec3 sat_normal_bt(0.0f, 0.0f, 0.0f);
                float sat_overlap_bt = 0.0f;
                ObbContactFeature sat_feature_bt;
                if (ComputeObbContactGameOrder(obb_a_bt, obb_b_bt, 0.0f, sat_normal_bt, sat_overlap_bt, &sat_feature_bt) &&
                    sat_feature_bt.kind == OBB_CONTACT_FEATURE_EDGE_EDGE &&
                    sat_feature_bt.axis_a == feature.axis_a && sat_feature_bt.axis_b == feature.axis_b) {
                    overlap_bt = sat_overlap_bt;
                }
            }
            const Vec3 raw_point_a_bt = point_a_bt;
            if (out_game_points_a_bt && out_game_depths_bt && out_game_normal_bt) {
                out_game_points_a_bt[0] = raw_point_a_bt;
                out_game_depths_bt[0] = game_s_bt;
                *out_game_normal_bt = -game_normal_bt;
                if (out_game_points_valid) {
                    *out_game_points_valid = true;
                }
            }
            const Vec3 raw_point_b_bt = overlap_bt > 0.0f ? (raw_point_a_bt + game_normal_bt * overlap_bt) : point_b_bt;
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

ROCKETGPU_HD int ComputeApproxCarCarContactAnchorPairs(
    CarCRefT a,
    const CarConfig& cfg_a,
    const ObbData& obb_a,
    CarCRefT b,
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
    bool game_face_contacts = false,
    Vec3* out_game_points_a_bt = nullptr,
    float* out_game_depths_bt = nullptr,
    bool* out_game_points_valid = nullptr,
    Vec3* out_game_normal_bt = nullptr) {
    (void)obb_a;
    (void)obb_b;
    return ComputeApproxCarCarContactAnchorPairsWithBtObb(
        a,
        cfg_a,
        MakeCarObbBulletTransformBt(a, cfg_a),
        b,
        cfg_b,
        MakeCarObbBulletTransformBt(b, cfg_b),
        normal,
        feature,
        out_points_a,
        out_points_b,
        max_points,
        contact_overlap_uu,
        out_points_a_bt,
        out_points_b_bt,
        game_face_contacts,
        out_game_points_a_bt,
        out_game_depths_bt,
        out_game_points_valid,
        out_game_normal_bt);
}

ROCKETGPU_HD void ComputeApproxCarCarContactAnchors(
    CarCRefT a,
    const CarConfig& cfg_a,
    const ObbData& obb_a,
    CarCRefT b,
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
    CarCRefT car,
    const Vec3& external_linear_delta,
    const Vec3& external_angular_delta,
    float dt,
    bool native_sse_math = false) {
    CarState predicted = car;
    IntegrateCarTransformOnly(predicted, dt, native_sse_math);
    predicted.vel += external_linear_delta;
    predicted.ang_vel += external_angular_delta;
    return predicted;
}
