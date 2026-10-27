

using ::rocketgpu::BallState;
using ::rocketgpu::CarConfig;
using ::rocketgpu::CarState;
using ::rocketgpu::MutatorConfig;
using ::rocketgpu::Quat;
using ::rocketgpu::Vec3;
using namespace ::rocketgpu::detail;
using ::rocketgpu::detail::operator+;
using ::rocketgpu::detail::operator-;
using ::rocketgpu::detail::operator*;
using ::rocketgpu::detail::operator/;
using ::rocketgpu::detail::operator+=;
using ::rocketgpu::detail::operator-=;
using ::rocketgpu::detail::operator*=;
using ::rocketgpu::detail::operator/=;

ROCKETGPU_HD float BtSimdDot3LikeBullet(const ::rocketgpu::Vec3& a, const ::rocketgpu::Vec3& b) {
    return a.x * b.x + (a.y * b.y + a.z * b.z);
}

struct ContactAccumulator {
    CarState* history_car = nullptr;
    PhysicsEventStream event_stream = {};
    PhysicsEvent event = {};
    uint8_t hit = 0;
    ::rocketgpu::Vec3 normal_sum = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    float max_penetration = 0.0f;
    ::rocketgpu::Vec3 primary_normal = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    ::rocketgpu::Vec3 latest_normal = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);

    ROCKETGPU_HD void add(const ::rocketgpu::Vec3& n, float penetration) {
        if (history_car) history_car->RecordWorldSurface(true);
        if (event_stream.windows) {
            event.normal = n;
            event_stream.EmitForCar(event.car_id, event);
        }
        hit = 1;
        normal_sum += n * penetration;
        latest_normal = n;
        if (penetration > max_penetration) {
            max_penetration = penetration;
            primary_normal = n;
        }
    }

    ROCKETGPU_HD ::rocketgpu::Vec3 normal() const {
        return hit ? Normalize(normal_sum, primary_normal) : ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    }
};

struct CarWorldSpecialContactAccumulator {
    uint8_t hit = 0;
    int32_t count = 0;
    ::rocketgpu::Vec3 normal_sum = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 fallback_normal = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    float distance_sum = 0.0f;

    ROCKETGPU_HD void add(const ::rocketgpu::Vec3& n, float distance) {
        hit = 1;
        count += 1;
        normal_sum += n;
        fallback_normal = n;
        distance_sum += distance;
    }

    ROCKETGPU_HD ::rocketgpu::Vec3 normal() const {
        return hit ? Normalize(normal_sum, fallback_normal) : ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    }

    ROCKETGPU_HD float average_distance() const {
        return count > 0 ? (distance_sum / static_cast<float>(count)) : 0.0f;
    }
};

::rocketgpu::Vec3 ApplyInverseBallInertiaWorld(float mass, float radius, const ::rocketgpu::Vec3& torque_world);
::rocketgpu::Vec3 ApplyInverseBallInertiaWorldBt(const ::rocketgpu::Quat& ball_rot, float mass, float radius_uu, const ::rocketgpu::Vec3& torque_bt);
::rocketgpu::Vec3 ApplyInverseBallInertiaWorldBtColumns(const ::rocketgpu::Vec3& forward_col, const ::rocketgpu::Vec3& right_col, const ::rocketgpu::Vec3& up_col, float mass, float radius_uu, const ::rocketgpu::Vec3& torque_bt);
::rocketgpu::Vec3 ApplyInverseBallInertiaWorldBtBall(const ::rocketgpu::BallState& ball, float mass, float radius_uu, const ::rocketgpu::Vec3& torque_bt);
::rocketgpu::Vec3 ApplyInverseBallInertiaWorldBtMutator(const ::rocketgpu::BallState& ball, const ::rocketgpu::MutatorConfig& mutator, const ::rocketgpu::Vec3& torque_bt);

ROCKETGPU_HD void GetBallBasisColumns(const ::rocketgpu::BallState& ball, ::rocketgpu::Vec3& forward, ::rocketgpu::Vec3& right, ::rocketgpu::Vec3& up) {
    if (ball.basis_authoritative) {
        forward = ball.basis_forward;
        right = ball.basis_right;
        up = ball.basis_up;
        return;
    }
    GetOfficialUploadBasisVectors(ball.rot, forward, right, up);
}
::rocketgpu::Vec3 BallVelocityAtPoint(const ::rocketgpu::BallState& ball, const ::rocketgpu::Vec3& rel_point_world);
::rocketgpu::Vec3 BallVelocityAtPoint(const ::rocketgpu::BallState& ball, const ::rocketgpu::Vec3& rel_point_world, const ::rocketgpu::Vec3& external_linear_delta);
void ApplyImpulseAtPoint(::rocketgpu::BallState& ball, float mass, float radius, const ::rocketgpu::Vec3& impulse_world, const ::rocketgpu::Vec3& rel_point_world);
::rocketgpu::Vec3 SupportPointOnCarHitboxLocal(const ::rocketgpu::CarConfig& cfg, const ::rocketgpu::Vec3& dir_local);
::rocketgpu::Vec3 BulletBoxHalfExtentWithMarginBt(const ::rocketgpu::CarConfig& cfg);
	::rocketgpu::Vec3 VelocityAtPoint(const ::rocketgpu::CarState& car, const ::rocketgpu::Vec3& rel_point_world);
	::rocketgpu::Vec3 VelocityAtPoint(const ::rocketgpu::CarState& car, const ::rocketgpu::Vec3& rel_point_world, const ::rocketgpu::Vec3& external_linear_delta, const ::rocketgpu::Vec3& external_angular_delta);
	::rocketgpu::Vec3 ApplyInverseBodyInertiaWorld(const ::rocketgpu::CarState& car, const ::rocketgpu::CarConfig& cfg, float mass, const ::rocketgpu::Vec3& torque_world);
	::rocketgpu::Vec3 ApplyInverseBodyInertiaWorldBt(const ::rocketgpu::CarState& car, const ::rocketgpu::CarConfig& cfg, float mass, const ::rocketgpu::Vec3& torque_bt);
	::rocketgpu::Vec3 ApplyCarWorldContactAngularComponent(const ::rocketgpu::CarState& car, const ::rocketgpu::CarConfig& cfg, float mass, const ::rocketgpu::Vec3& rel_contact, const ::rocketgpu::Vec3& normal);
	::rocketgpu::Vec3 ApplyCarWorldContactAngularComponentBt(const ::rocketgpu::CarState& car, const ::rocketgpu::CarConfig& cfg, float mass, const ::rocketgpu::Vec3& rel_contact_bt, const ::rocketgpu::Vec3& normal);
	::rocketgpu::Vec3 CarWorldContactAngularComponentBtNative(const ::rocketgpu::CarState& car, const ::rocketgpu::CarConfig& cfg, float mass, const ::rocketgpu::Vec3& rel_contact_bt, const ::rocketgpu::Vec3& normal);
	void ApplyImpulseAtPoint(::rocketgpu::CarState& car, const ::rocketgpu::CarConfig& cfg, float mass, const ::rocketgpu::Vec3& impulse_world, const ::rocketgpu::Vec3& rel_point_world);
::rocketgpu::Vec3 CarBodyLocalPointBt(const ::rocketgpu::CarState& car, const ::rocketgpu::Vec3& point_world_bt);
::rocketgpu::Vec3 CarBodyWorldPointBt(const ::rocketgpu::CarState& car, const ::rocketgpu::Vec3& point_local_bt);
inline void PopulateCarWorldVelocityTraceInertia(
    struct CarWorldVelocityTraceInputs& trace,
    const ::rocketgpu::CarState& car,
    const ::rocketgpu::CarConfig& cfg,
    float mass,
    const ::rocketgpu::Vec3& rel_contact,
    const ::rocketgpu::Vec3& normal);
struct TriangleContactDebugSummary {
    uint8_t hit = 0;
    uint8_t is_valid = 0;
    uint8_t penetration_valid = 0;
    uint8_t reject_stage = 0;
    uint8_t comparison_flags = 0;
    uint8_t distance_negative = 0;
    uint8_t distance_near = 0;
    uint8_t reserved0 = 0;
    int32_t last_used_method = -1;
    int32_t degenerate_simplex = 0;
    uint32_t distance_bits = 0;
    uint32_t maximum_distance_squared_bits = 0;
    float distance_bt = 0.0f;
    float contact_threshold_bt = 0.0f;
    float detail0 = 0.0f;
    float detail1 = 0.0f;
    ::rocketgpu::Vec3 detail_vec0 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 detail_vec1 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    int32_t gjk_iter = -1;
    int32_t gjk_num_vertices = 0;
    uint8_t gjk_in_simplex = 0;
    uint8_t gjk_closest_success = 0;
    uint16_t gjk_reserved0 = 0;
    float gjk_delta = 0.0f;
    float gjk_squared_distance = 0.0f;
    float gjk_f0 = 0.0f;
    float gjk_f1 = 0.0f;
    ::rocketgpu::Vec3 gjk_cached_axis = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 gjk_new_cached_axis = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 gjk_w = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 gjk_p_world = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 gjk_q_world = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 gjk_last_w = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 pre_offset_point_on_a_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 pre_offset_point_on_b_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 pre_fix_normal_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    ::rocketgpu::Vec3 final_point_on_a_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 final_point_on_b_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 final_normal_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    ::rocketgpu::Vec3 gjk_init_wtrs0_row0 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 gjk_init_wtrs0_row1 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 gjk_init_wtrs0_row2 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 gjk_init_wtrs0_origin = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 gjk_init_wtrs1_row0 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 gjk_init_wtrs1_row1 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 gjk_init_wtrs1_row2 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 gjk_init_wtrs1_origin = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    int32_t epa_first_iteration = -1;
    int32_t epa_last_iteration = -1;
    int32_t epa_first_support1_max_axis = -1;
    int32_t epa_last_support1_max_axis = -1;
    float epa_first_best_d = 0.0f;
    float epa_first_wdist = 0.0f;
    float epa_last_best_d = 0.0f;
    float epa_last_wdist = 0.0f;
    ::rocketgpu::Vec3 epa_first_best_n = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_first_best_w0 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_first_best_w1 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_first_best_w2 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_first_support_vertex_d = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_first_support0 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_first_support1_dir_local = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_first_support1_toshape1_row0 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_first_support1_toshape1_row1 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_first_support1_toshape1_row2 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_first_support1_toshape0_row0 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_first_support1_toshape0_row1 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_first_support1_toshape0_row2 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_first_support1_toshape0_origin = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_first_support1_dots = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_first_support1_local = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_first_support1 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_first_support_w = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_last_best_n = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_last_best_w0 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_last_best_w1 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_last_best_w2 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_last_support_vertex_d = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_last_support0 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_last_support1_dir_local = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_last_support1_toshape1_row0 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_last_support1_toshape1_row1 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_last_support1_toshape1_row2 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_last_support1_toshape0_row0 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_last_support1_toshape0_row1 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_last_support1_toshape0_row2 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_last_support1_toshape0_origin = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_last_support1_dots = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_last_support1_local = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_last_support1 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 epa_last_support_w = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 gjk_simplex_w[4] = {
        ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f),
        ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f),
        ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f),
        ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f)};
    ::rocketgpu::Vec3 gjk_simplex_p[4] = {
        ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f),
        ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f),
        ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f),
        ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f)};
    ::rocketgpu::Vec3 gjk_simplex_q[4] = {
        ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f),
        ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f),
        ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f),
        ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f)};
};
bool ComputeCarArenaTriangleContact(
    const ::rocketgpu::CarState& car,
    const ::rocketgpu::CarConfig& cfg,
    const ArenaMeshTriangle& tri,
    float contact_threshold_bt,
    ::rocketgpu::Vec3& out_normal,
    float& out_distance_bt,
    ::rocketgpu::Vec3* out_point_on_a_bt = nullptr,
    ::rocketgpu::Vec3* out_point_on_b_bt = nullptr,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch = nullptr,
    ::rocketgpu::Vec3* io_cached_axis_bt = nullptr,
    TriangleContactDebugSummary* debug_summary = nullptr,
    uint8_t debug_triangle_phase = 0,
    float tri_origin_y_bt = 0.0f);
bool DebugCarWorldSolverTraceEnabledFor(int env_idx, int car_idx);
bool DebugBallCarSolverTraceEnabledFor(int env_idx, int car_idx);
bool ShouldSleepBallAtTickStart(const ::rocketgpu::BallState& ball);
struct SplitImpulseAccumulator;
void IntegrateSplitImpulseTransform(::rocketgpu::CarState& car, const SplitImpulseAccumulator& split, float dt, bool native_sse_math = false);
void IntegrateSplitImpulseTransformBt(::rocketgpu::CarState& car, const SplitImpulseAccumulator& split_bt, float dt, bool native_sse_math = false);

struct SplitImpulseAccumulator {
    ::rocketgpu::Vec3 push_velocity = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 turn_velocity = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);

    uint8_t is_bt = 0;
};

ROCKETGPU_HD bool InGoalWindow(float x, float z, float radius) {
    return Abs(x) < (ApproximateGoalHalfWidth() - radius) && z < (ApproximateGoalHeight() - radius);
}

ROCKETGPU_HD bool InMainCornerZone(float x, float y) {
    return Abs(x) > CornerCenterX() && Abs(y) > CornerCenterY() && Abs(y) <= ArenaExtentY();
}

ROCKETGPU_HD float BallArenaMeshCollisionMargin() {
    return 0.0f;
}

ROCKETGPU_HD float BallStaticSplitPenetrationScale() {
    return 1.0f;
}

ROCKETGPU_HD float BallCarSplitPenetrationScale() {
    return 1.0f;
}

ROCKETGPU_HD float BallCarShapeMargin(const ::rocketgpu::CarConfig& cfg) {
    const ::rocketgpu::Vec3 raw_half_extent = cfg.hitbox_size * 0.5f;
    const float min_half_extent = Min(raw_half_extent.x, Min(raw_half_extent.y, raw_half_extent.z));
    return Min(2.0f, min_half_extent * 0.1f);
}

ROCKETGPU_HD ::rocketgpu::Vec3 BallCarHitboxInset(const ::rocketgpu::CarConfig& cfg) {
    {
        const float exact_margin = BallCarShapeMargin(cfg);
        return ::rocketgpu::Vec3(exact_margin, exact_margin, exact_margin);
    }
    constexpr float margin = 2.0f;
    return ::rocketgpu::Vec3(margin, margin, margin);
}

ROCKETGPU_HD ::rocketgpu::Vec3 BallCarHitboxInsetBt(const ::rocketgpu::CarConfig& cfg) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    {
        const ::rocketgpu::Vec3 half_bt = (cfg.hitbox_size * kUUToBT) / 2.0f;
        const float min_half_bt = Min(half_bt.x, Min(half_bt.y, half_bt.z));
        const float safe_bt = 0.1f * min_half_bt;
        const float margin_bt = safe_bt < 0.04f ? safe_bt : 0.04f;
        return ::rocketgpu::Vec3(margin_bt, margin_bt, margin_bt);
    }
    return BallCarHitboxInset(cfg) * kUUToBT;
}

constexpr int kMaxBallSplitCars = 32;
constexpr int kMaxBallSplitContactRows = 64;
constexpr int kMaxBallVelocityContactRows = 64;
constexpr int kMaxBallFrictionContactRows = 64;
constexpr int kMaxCarCarPairSlots = 8 * 8;
constexpr int kMaxCarCarCachedPoints = 4;
constexpr int kMaxCarWorldSplitContactRows = 16;
constexpr int kMaxCarPairSplitContactRows = 128;
constexpr int kMaxCarSharedSplitContactRows = kMaxCarWorldSplitContactRows * 8 + kMaxCarPairSplitContactRows;
constexpr int kMaxCarWorldVelocityContactRows = 16;
constexpr int kMaxCarWorldFrictionContactRows = 16;
constexpr int kMaxCarWorldSharedVelocityContactRows = kMaxCarWorldVelocityContactRows * 8;
constexpr int kMaxCarWorldSharedFrictionContactRows = kMaxCarWorldFrictionContactRows * 8;
constexpr int kMaxCarPairSharedVelocityContactRows = 128;
constexpr int kMaxCarPairSharedFrictionContactRows = 128;
constexpr int kDebugBallVelocityMaxSamples = 16;

struct BallSplitContactRow {
    uint8_t valid = 0;
    int32_t other_local_idx = -1;
    ::rocketgpu::Vec3 normal = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    ::rocketgpu::Vec3 rel_ball_contact = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_other_contact = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_ball_cross_normal = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_other_cross_normal = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 ball_angular_component = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 other_angular_component = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    float ball_inv_mass = 0.0f;
    float other_inv_mass = 0.0f;
    float jac_diag_ab_inv = 0.0f;
    float rhs_penetration = 0.0f;
    float applied_push_impulse = 0.0f;

    ::rocketgpu::Vec3 rel_ball_cross_normal_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_other_cross_normal_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 ball_angular_component_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 other_angular_component_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    float jac_diag_ab_inv_bt = 0.0f;
    float rhs_penetration_bt = 0.0f;
    float applied_push_impulse_bt = 0.0f;
    uint8_t has_bt = 0;

    int16_t static_id = -1;
};

struct BallVelocityAccumulator {
    ::rocketgpu::Vec3 delta_linear = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 delta_angular = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
};

struct BallVelocityContactRow {
    uint8_t valid = 0;
    int32_t other_local_idx = -1;
    ::rocketgpu::Vec3 normal = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    ::rocketgpu::Vec3 rel_ball_contact = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_other_contact = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_ball_cross_normal = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_other_cross_normal = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 ball_angular_component = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 other_angular_component = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    float ball_inv_mass = 0.0f;
    float other_inv_mass = 0.0f;
    float jac_diag_ab_inv = 0.0f;
    float rhs = 0.0f;
    float applied_impulse = 0.0f;

    uint8_t has_bt = 0;
    float rhs_bt = 0.0f;
    ::rocketgpu::Vec3 rel_ball_cross_normal_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_other_cross_normal_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 ball_angular_component_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 other_angular_component_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
};

struct BallFrictionContactRow {
    uint8_t valid = 0;

    uint8_t has_bt = 0;
    float rhs_bt = 0.0f;
    ::rocketgpu::Vec3 rel_ball_cross_tangent_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_other_cross_tangent_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 ball_angular_component_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 other_angular_component_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    int32_t other_local_idx = -1;
    int32_t normal_row_idx = -1;
    ::rocketgpu::Vec3 tangent = ::rocketgpu::Vec3(1.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_ball_contact = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_other_contact = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_ball_cross_tangent = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_other_cross_tangent = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 ball_angular_component = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 other_angular_component = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    float ball_inv_mass = 0.0f;
    float other_inv_mass = 0.0f;
    float jac_diag_ab_inv = 0.0f;
    float rhs = 0.0f;
    float friction = 0.0f;
    float applied_impulse = 0.0f;
};

struct CarCarPairCachedPoint {
    uint8_t valid = 0;
    uint8_t tangent_valid = 0;
    ::rocketgpu::Vec3 point_a = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 point_b = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);

    ::rocketgpu::Vec3 point_a_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 point_b_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    uint8_t has_bt_points = 0;
    ::rocketgpu::Vec3 tangent = ::rocketgpu::Vec3(1.0f, 0.0f, 0.0f);
    float penetration_uu = 0.0f;
    float applied_normal_impulse = 0.0f;
    float applied_friction_impulse = 0.0f;
};

struct CarCarPairCachedContact {
    uint8_t valid = 0;
    uint8_t contact_count = 0;
    ::rocketgpu::Vec3 normal = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    CarCarPairCachedPoint contacts[kMaxCarCarCachedPoints];
};

struct CarWorldSplitContactRow {
    uint8_t valid = 0;
    int32_t source_tag = 0;
    ::rocketgpu::Vec3 normal = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    ::rocketgpu::Vec3 rel_contact = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_cross_normal = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 angular_component = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    float inv_mass = 0.0f;
    float jac_diag_ab_inv = 0.0f;
    float rhs_penetration = 0.0f;
    float applied_push_impulse = 0.0f;

    ::rocketgpu::Vec3 rel_contact_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_cross_normal_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 angular_component_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    float jac_diag_ab_inv_bt = 0.0f;
    float rhs_penetration_bt = 0.0f;
    float applied_push_impulse_bt = 0.0f;
};

struct CarSharedSplitContactRow {
    uint8_t valid = 0;
    int32_t source_tag = 0;
    int32_t body_a_local_idx = -1;
    int32_t body_b_local_idx = -1;
    ::rocketgpu::Vec3 normal_a = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    ::rocketgpu::Vec3 normal_b = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_contact_a = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_contact_b = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_a_cross_normal = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_b_cross_normal = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 angular_component_a = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 angular_component_b = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    float inv_mass_a = 0.0f;
    float inv_mass_b = 0.0f;
    float jac_diag_ab_inv = 0.0f;
    float rhs_penetration = 0.0f;
    float applied_push_impulse = 0.0f;

    ::rocketgpu::Vec3 rel_a_cross_normal_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 angular_component_a_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_b_cross_normal_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 angular_component_b_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    float jac_diag_ab_inv_bt = 0.0f;
    float rhs_penetration_bt = 0.0f;
    float applied_push_impulse_bt = 0.0f;
    uint8_t has_bt = 0;
};

struct CarWorldVelocityContactRow {
    uint8_t valid = 0;
    int32_t source_tag = 0;
    ::rocketgpu::Vec3 normal = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    ::rocketgpu::Vec3 rel_contact = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_contact_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_cross_normal = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 angular_component = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    float inv_mass = 0.0f;
    float jac_diag_ab_inv = 0.0f;
    float rhs = 0.0f;
    float rhs_bt = 0.0f;
    uint8_t has_rhs_bt = 0;
    float applied_impulse = 0.0f;

    uint8_t has_bt = 0;
    ::rocketgpu::Vec3 rel_cross_normal_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 angular_component_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
};

struct CarWorldVelocityTraceInputs {
    ::rocketgpu::Vec3 linear_velocity = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 angular_velocity = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 external_linear_delta = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 external_angular_delta = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);

    ::rocketgpu::Vec3 car_pos = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_contact_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    uint8_t has_rel_contact_bt = 0;

    ::rocketgpu::Vec3 linear_velocity_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 external_force_impulse_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    uint8_t has_bt_inputs = 0;

    ::rocketgpu::Vec3 car_pos_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 external_torque_impulse = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 torque_axis0_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 bodyA_basis_row0 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 bodyA_basis_row1 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 bodyA_basis_row2 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 bodyA_inv_inertia_local = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 bodyA_inv_inertia_world_row0 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 bodyA_inv_inertia_world_row1 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 bodyA_inv_inertia_world_row2 = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 angular_component_a_pre_factor = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 bodyA_angular_factor = ::rocketgpu::Vec3(1.0f, 1.0f, 1.0f);
    // cal_world_restitution_curve: v3 restitution_curve has no rest-velocity threshold.
    uint8_t restitution_no_velocity_threshold = 0;
};

struct CarWorldFrictionContactRow {
    uint8_t valid = 0;
    int32_t normal_row_idx = -1;
    int32_t source_tag = 0;
    ::rocketgpu::Vec3 tangent = ::rocketgpu::Vec3(1.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_contact = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_cross_tangent = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 angular_component = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    float inv_mass = 0.0f;
    float jac_diag_ab_inv = 0.0f;
    float rhs = 0.0f;
    float rhs_bt = 0.0f;
    uint8_t has_rhs_bt = 0;
    float friction = 0.0f;
    float applied_impulse = 0.0f;

    uint8_t has_bt = 0;
    ::rocketgpu::Vec3 rel_cross_tangent_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 angular_component_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
};

struct CarSharedVelocityContactRow {
    uint8_t valid = 0;
    int32_t source_tag = 0;
    int32_t body_a_local_idx = -1;
    int32_t body_b_local_idx = -1;
    int32_t pair_cache_idx = -1;
    int32_t pair_contact_idx = -1;
    ::rocketgpu::Vec3 normal_a = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    ::rocketgpu::Vec3 rel_contact_a = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_contact_b = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_a_cross_normal = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_b_cross_normal = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 angular_component_a = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 angular_component_b = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    float inv_mass_a = 0.0f;
    float inv_mass_b = 0.0f;
    float jac_diag_ab_inv = 0.0f;
    float rhs = 0.0f;
    float rhs_bt = 0.0f;
    uint8_t has_rhs_bt = 0;
    float applied_impulse = 0.0f;

    uint8_t has_bt = 0;
    ::rocketgpu::Vec3 rel_a_cross_normal_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 angular_component_a_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);

    ::rocketgpu::Vec3 rel_b_cross_normal_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 angular_component_b_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
};

struct CarSharedFrictionContactRow {
    uint8_t valid = 0;
    int32_t normal_row_idx = -1;
    int32_t source_tag = 0;
    int32_t body_a_local_idx = -1;
    int32_t body_b_local_idx = -1;
    int32_t pair_cache_idx = -1;
    int32_t pair_contact_idx = -1;
    ::rocketgpu::Vec3 tangent_a = ::rocketgpu::Vec3(1.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_contact_a = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_contact_b = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_a_cross_tangent = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 rel_b_cross_tangent = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 angular_component_a = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 angular_component_b = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    float inv_mass_a = 0.0f;
    float inv_mass_b = 0.0f;
    float jac_diag_ab_inv = 0.0f;
    float rhs = 0.0f;
    float rhs_bt = 0.0f;
    uint8_t has_rhs_bt = 0;
    float friction = 0.0f;
    float applied_impulse = 0.0f;

    uint8_t has_bt = 0;
    ::rocketgpu::Vec3 rel_a_cross_tangent_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 angular_component_a_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);

    ::rocketgpu::Vec3 rel_b_cross_tangent_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 angular_component_b_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
};

enum class SharedContactOrderBucket : uint8_t {
    Pair = 0,
    World = 1,
};

struct SharedContactOrderGroup {
    uint8_t valid = 0;
    SharedContactOrderBucket bucket = SharedContactOrderBucket::World;
    int32_t body_a_local_idx = -1;
    int32_t body_b_local_idx = -1;
    int32_t source_tag = 0;
    int32_t normal_row_start = -1;
    int32_t normal_row_count = 0;
    int32_t friction_row_start = -1;
    int32_t friction_row_count = 0;
    int32_t initial_order = 0;
};

constexpr int kMaxSharedContactOrderGroups =
    kMaxCarCarPairSlots + 8 * (16 + 4);

struct BallWorldContactAggregate {
    uint8_t valid = 0;
    int32_t count = 0;
    ::rocketgpu::Vec3 normal_sum = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 fallback_normal = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    float penetration_sum = 0.0f;

    float rel_pos1_length_sum_bt = 0.0f;
};

struct BallMeshManifoldPoint {
    ::rocketgpu::Vec3 normal = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    float pen_uu = 0.0f;
    ::rocketgpu::Vec3 local_point_a_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 local_point_b_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    int32_t triangle_index = -1;

    float rel_pos1_length_bt = 0.0f;

    ::rocketgpu::Vec3 rel_pos1_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    float distance_bt = 0.0f;
};

struct BallMeshPersistentManifold {
    uint8_t count = 0;
    BallMeshManifoldPoint points[4] = {};
};

// Retained car-world plane manifold (cal_retained_world_contacts), one per (car, static plane). Every point's
// normal is that plane's inward normal, so a point stores only its anchors and its refreshed distance. Slots at or
// past count are never read, so emptying a manifold only clears count.
struct CarWorldPlaneRetainedPoint {
    ::rocketgpu::Vec3 local_point_a_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    // Relative to the plane body origin (BulletStaticPlaneOriginBt).
    ::rocketgpu::Vec3 point_on_b_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    float distance_bt = 0.0f;
};

struct CarWorldPlaneRetainedManifold {
    uint8_t count = 0;
    CarWorldPlaneRetainedPoint points[4] = {};
};

// cal_car_car_manifold: v3 keeps each car pair's OBB-OBB persistent manifold across ticks
// (collision_dispatcher.rs near_callback_with_bounds, persistent_manifold.rs). RocketGPU's pair
// orientation is car A = higher local index = v3 body1, car B = lower = v3 body0, so v3's
// local_point_a is local_b_bt here and v3 distance_1 = (pos_b_bt - pos_a_bt).normal.
struct CarCarRetainedPoint {
    ::rocketgpu::Vec3 local_a_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 local_b_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 pos_a_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 pos_b_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    // v3 normal_world_on_b: fixed when the point is created.
    ::rocketgpu::Vec3 normal = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    float distance_bt = 0.0f;
};

struct CarCarRetainedManifold {
    uint8_t count = 0;
    CarCarRetainedPoint points[4] = {};
};

constexpr int kMaxCarCarRetainedManifolds = 28;

// Pair (lo, hi), lo < hi < 8.
ROCKETGPU_HD constexpr int CarCarRetainedSlot(int lo, int hi) {
    return hi * (hi - 1) / 2 + lo;
}

// cal_ball_car_manifold: v3 keeps each ball-car sphere-OBB persistent manifold across ticks while the car's box
// child AABB overlaps the ball AABB (collision_dispatcher.rs near_callback_with_bounds compound gate,
// sphere_obb_collision_alg.rs). v3 body0 = ball, body1 = car: local_a_bt is ball-local, local_b_bt car-local,
// normal is v3 normal_world_on_b (car toward ball), distance_bt = (pos_a_bt - pos_b_bt).normal.
struct BallCarRetainedPoint {
    ::rocketgpu::Vec3 local_a_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 local_b_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 pos_a_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 pos_b_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 normal = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    float distance_bt = 0.0f;
};

struct BallCarRetainedManifold {
    uint8_t count = 0;
    BallCarRetainedPoint points[4] = {};
};

constexpr int kMaxBallCarRetainedManifolds = 8;

ROCKETGPU_HD void ResetBallMeshPersistentManifold(BallMeshPersistentManifold& manifold) {
    manifold.count = 0;
    for (int manifold_idx = 0; manifold_idx < 4; ++manifold_idx) {
        manifold.points[manifold_idx] = BallMeshManifoldPoint{};
    }
}

constexpr int kMaxBallWorldMeshPersistentManifolds = 16;
constexpr int kBallSettledMaxWorldEntries = 4;
constexpr int kMaxCarWorldMeshesPerCar = kMaxBallWorldMeshPersistentManifolds;
constexpr int kMaxCarWorldMeshPersistentCars = 8;
// Retained car-world plane manifolds per car: floor, ceiling, +X wall, -X wall.
constexpr int kCarWorldPlaneRetainedManifoldsPerCar = 4;
constexpr int kMaxCarWorldPlaneRetainedManifolds =
    kCarWorldPlaneRetainedManifoldsPerCar * kMaxCarWorldMeshPersistentCars;

struct BulletManifoldEntry {
    int8_t kind = 0;
    int8_t body_a = -1;
    int8_t body_b = -1;
    int16_t static_id = -1;
    int32_t island_id = 0;
};

struct BulletBroadphaseProxy {
    ::rocketgpu::Vec3 aabb_min = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 aabb_max = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 child_aabb_min = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 child_aabb_max = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    int32_t cell_i = 0;
    int32_t cell_j = 0;
    int32_t cell_k = 0;
    uint8_t valid = 0;
    uint8_t is_ball = 0;
};

constexpr int kMaxBulletManifoldEntries = 96;

template <typename T>
struct EnvStridedArray {
    T* base = nullptr;
    int stride = 1;

    EnvStridedArray() = default;
    ROCKETGPU_HD EnvStridedArray(T* p) : base(p), stride(1) {}
    ROCKETGPU_HD EnvStridedArray(T* p, int s) : base(p), stride(s) {}
    ROCKETGPU_HD EnvStridedArray(std::nullptr_t) : base(nullptr), stride(1) {}
    template <typename U, typename = typename std::enable_if<std::is_convertible<U*, T*>::value>::type>
    ROCKETGPU_HD EnvStridedArray(const EnvStridedArray<U>& other) : base(other.base), stride(other.stride) {}

    ROCKETGPU_HD T& operator[](int i) const {
        return base[static_cast<std::ptrdiff_t>(i) * static_cast<std::ptrdiff_t>(stride)];
    }
    ROCKETGPU_HD explicit operator bool() const { return base != nullptr; }
    ROCKETGPU_HD bool operator==(std::nullptr_t) const { return base == nullptr; }
    ROCKETGPU_HD bool operator!=(std::nullptr_t) const { return base != nullptr; }
};

struct ContactSortScratch;

struct WorldSolverScratchRows {
    ContactSortScratch* contact_sort_scratch = nullptr;
    EnvStridedArray<BallSplitContactRow> ball_split_rows;
    EnvStridedArray<BallVelocityContactRow> ball_velocity_rows;
    EnvStridedArray<BallFrictionContactRow> ball_friction_rows;
    EnvStridedArray<CarSharedSplitContactRow> pair_split_rows;
    EnvStridedArray<CarSharedSplitContactRow> world_split_rows;
    EnvStridedArray<CarSharedVelocityContactRow> pair_velocity_rows;
    EnvStridedArray<CarSharedFrictionContactRow> pair_friction_rows;
    EnvStridedArray<CarSharedVelocityContactRow> world_velocity_rows;
    EnvStridedArray<CarSharedFrictionContactRow> world_friction_rows;
    EnvStridedArray<SharedContactOrderGroup> shared_contact_order_groups;
    EnvStridedArray<BulletManifoldEntry> bullet_manifolds;
};

struct WorldSolverScratchRowPool {
    ContactSortScratch* contact_sort_scratch = nullptr;
    size_t contact_sort_stride = 0;
    BallSplitContactRow* ball_split_rows = nullptr;
    BallVelocityContactRow* ball_velocity_rows = nullptr;
    BallFrictionContactRow* ball_friction_rows = nullptr;
    CarSharedSplitContactRow* pair_split_rows = nullptr;
    CarSharedSplitContactRow* world_split_rows = nullptr;
    CarSharedVelocityContactRow* pair_velocity_rows = nullptr;
    CarSharedFrictionContactRow* pair_friction_rows = nullptr;
    CarSharedVelocityContactRow* world_velocity_rows = nullptr;
    CarSharedFrictionContactRow* world_friction_rows = nullptr;
    SharedContactOrderGroup* shared_contact_order_groups = nullptr;
    BulletManifoldEntry* bullet_manifolds = nullptr;
    int env_count = 0;
    uint8_t interleaved = 0;

    template <typename T>
    ROCKETGPU_HD EnvStridedArray<T> view_one(T* base, int max_rows, int env_idx) const {
        if (base == nullptr) {
            return EnvStridedArray<T>(nullptr);
        }
        if (interleaved) {
            return EnvStridedArray<T>(base + env_idx, env_count);
        }
        return EnvStridedArray<T>(base + static_cast<std::ptrdiff_t>(env_idx) * max_rows, 1);
    }

    ROCKETGPU_HD WorldSolverScratchRows view(int env_idx) const {
        WorldSolverScratchRows rows;
        rows.contact_sort_scratch = contact_sort_scratch ? reinterpret_cast<ContactSortScratch*>(
            reinterpret_cast<unsigned char*>(contact_sort_scratch) + env_idx * contact_sort_stride) : nullptr;
        rows.ball_split_rows = view_one(ball_split_rows, kMaxBallSplitContactRows, env_idx);
        rows.ball_velocity_rows = view_one(ball_velocity_rows, kMaxBallVelocityContactRows, env_idx);
        rows.ball_friction_rows = view_one(ball_friction_rows, kMaxBallFrictionContactRows, env_idx);
        rows.pair_split_rows = view_one(pair_split_rows, kMaxCarPairSplitContactRows, env_idx);
        rows.world_split_rows = view_one(world_split_rows, kMaxCarSharedSplitContactRows, env_idx);
        rows.pair_velocity_rows = view_one(pair_velocity_rows, kMaxCarPairSharedVelocityContactRows, env_idx);
        rows.pair_friction_rows = view_one(pair_friction_rows, kMaxCarPairSharedFrictionContactRows, env_idx);
        rows.world_velocity_rows = view_one(world_velocity_rows, kMaxCarWorldSharedVelocityContactRows, env_idx);
        rows.world_friction_rows = view_one(world_friction_rows, kMaxCarWorldSharedFrictionContactRows, env_idx);
        rows.shared_contact_order_groups = view_one(shared_contact_order_groups, kMaxSharedContactOrderGroups, env_idx);
        rows.bullet_manifolds = view_one(bullet_manifolds, kMaxBulletManifoldEntries, env_idx);
        return rows;
    }
};

ROCKETGPU_HD size_t WorldSolverScratchRowBytesPerEnv() {
    return sizeof(BallSplitContactRow) * kMaxBallSplitContactRows +
           sizeof(BallVelocityContactRow) * kMaxBallVelocityContactRows +
           sizeof(BallFrictionContactRow) * kMaxBallFrictionContactRows +
           sizeof(CarSharedSplitContactRow) * kMaxCarPairSplitContactRows +
           sizeof(CarSharedSplitContactRow) * kMaxCarSharedSplitContactRows +
           sizeof(CarSharedVelocityContactRow) * kMaxCarPairSharedVelocityContactRows +
           sizeof(CarSharedFrictionContactRow) * kMaxCarPairSharedFrictionContactRows +
           sizeof(CarSharedVelocityContactRow) * kMaxCarWorldSharedVelocityContactRows +
           sizeof(CarSharedFrictionContactRow) * kMaxCarWorldSharedFrictionContactRows +
           sizeof(SharedContactOrderGroup) * kMaxSharedContactOrderGroups +
           sizeof(BulletManifoldEntry) * kMaxBulletManifoldEntries;
}

ROCKETGPU_HD void ZeroWorldSolverScratchRows(const WorldSolverScratchRows& rows) {
    if (!rows.ball_split_rows) {
        return;
    }
    for (int i = 0; i < kMaxBallSplitContactRows; ++i) rows.ball_split_rows[i] = BallSplitContactRow{};
    for (int i = 0; i < kMaxBallVelocityContactRows; ++i) rows.ball_velocity_rows[i] = BallVelocityContactRow{};
    for (int i = 0; i < kMaxBallFrictionContactRows; ++i) rows.ball_friction_rows[i] = BallFrictionContactRow{};
    for (int i = 0; i < kMaxCarPairSplitContactRows; ++i) rows.pair_split_rows[i] = CarSharedSplitContactRow{};
    for (int i = 0; i < kMaxCarSharedSplitContactRows; ++i) rows.world_split_rows[i] = CarSharedSplitContactRow{};
    for (int i = 0; i < kMaxCarPairSharedVelocityContactRows; ++i) rows.pair_velocity_rows[i] = CarSharedVelocityContactRow{};
    for (int i = 0; i < kMaxCarPairSharedFrictionContactRows; ++i) rows.pair_friction_rows[i] = CarSharedFrictionContactRow{};
    for (int i = 0; i < kMaxCarWorldSharedVelocityContactRows; ++i) rows.world_velocity_rows[i] = CarSharedVelocityContactRow{};
    for (int i = 0; i < kMaxCarWorldSharedFrictionContactRows; ++i) rows.world_friction_rows[i] = CarSharedFrictionContactRow{};
    for (int i = 0; i < kMaxSharedContactOrderGroups; ++i) rows.shared_contact_order_groups[i] = SharedContactOrderGroup{};
    for (int i = 0; i < kMaxBulletManifoldEntries; ++i) rows.bullet_manifolds[i] = BulletManifoldEntry{};
}

#if !defined(__CUDA_ARCH__)

struct WorldSolverScratchRowPoolHost {
    std::vector<BallSplitContactRow> ball_split_rows;
    std::vector<BallVelocityContactRow> ball_velocity_rows;
    std::vector<BallFrictionContactRow> ball_friction_rows;
    std::vector<CarSharedSplitContactRow> pair_split_rows;
    std::vector<CarSharedSplitContactRow> world_split_rows;
    std::vector<CarSharedVelocityContactRow> pair_velocity_rows;
    std::vector<CarSharedFrictionContactRow> pair_friction_rows;
    std::vector<CarSharedVelocityContactRow> world_velocity_rows;
    std::vector<CarSharedFrictionContactRow> world_friction_rows;
    std::vector<SharedContactOrderGroup> shared_contact_order_groups;
    std::vector<BulletManifoldEntry> bullet_manifolds;
    WorldSolverScratchRowPool pool;

    void reset(int env_count, bool interleaved) {
        const size_t n = static_cast<size_t>(env_count < 0 ? 0 : env_count);
        ball_split_rows.assign(n * kMaxBallSplitContactRows, BallSplitContactRow{});
        ball_velocity_rows.assign(n * kMaxBallVelocityContactRows, BallVelocityContactRow{});
        ball_friction_rows.assign(n * kMaxBallFrictionContactRows, BallFrictionContactRow{});
        pair_split_rows.assign(n * kMaxCarPairSplitContactRows, CarSharedSplitContactRow{});
        world_split_rows.assign(n * kMaxCarSharedSplitContactRows, CarSharedSplitContactRow{});
        pair_velocity_rows.assign(n * kMaxCarPairSharedVelocityContactRows, CarSharedVelocityContactRow{});
        pair_friction_rows.assign(n * kMaxCarPairSharedFrictionContactRows, CarSharedFrictionContactRow{});
        world_velocity_rows.assign(n * kMaxCarWorldSharedVelocityContactRows, CarSharedVelocityContactRow{});
        world_friction_rows.assign(n * kMaxCarWorldSharedFrictionContactRows, CarSharedFrictionContactRow{});
        shared_contact_order_groups.assign(n * kMaxSharedContactOrderGroups, SharedContactOrderGroup{});
        bullet_manifolds.assign(n * kMaxBulletManifoldEntries, BulletManifoldEntry{});
        pool = WorldSolverScratchRowPool{};
        pool.ball_split_rows = ball_split_rows.data();
        pool.ball_velocity_rows = ball_velocity_rows.data();
        pool.ball_friction_rows = ball_friction_rows.data();
        pool.pair_split_rows = pair_split_rows.data();
        pool.world_split_rows = world_split_rows.data();
        pool.pair_velocity_rows = pair_velocity_rows.data();
        pool.pair_friction_rows = pair_friction_rows.data();
        pool.world_velocity_rows = world_velocity_rows.data();
        pool.world_friction_rows = world_friction_rows.data();
        pool.shared_contact_order_groups = shared_contact_order_groups.data();
        pool.bullet_manifolds = bullet_manifolds.data();
        pool.env_count = env_count;
        pool.interleaved = interleaved ? 1 : 0;
    }

    WorldSolverScratchRows view(int env_idx) const { return pool.view(env_idx); }
};
#endif

struct WheelRayStage {
    ::rocketgpu::Vec3 contact_point_ws;
    ::rocketgpu::Vec3 contact_point_bt;
    ::rocketgpu::Vec3 contact_normal_ws;
    ::rocketgpu::Vec3 vel_at_contact;
    float suspension_length;
    float wheel_trace_length;
    float unclamped_wheel_trace_length;
    float wheel_trace_len_bt;
    float suspension_len_bt;
    float suspension_relative_velocity_bt;
    float debug_hit_fraction;
    float suspension_relative_velocity;
    float clipped_inv_contact_dot_suspension;
    float extra_pushback;
    int32_t ground_car_idx;
    int32_t ground_static_id;
    uint8_t ground_kind;
    uint8_t is_in_contact;
    uint8_t is_in_contact_with_world;

    uint8_t valid;
};
constexpr int kWheelRayStageSlots = 32;

// The car fields the contact pass reads from its snapshot of the car before contacts.
struct PreContactCar {
    ::rocketgpu::Quat rot = ::rocketgpu::Quat(1.0f, 0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 vel = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 vel_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 ang_vel = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    uint8_t is_demoed = 0;
};

ROCKETGPU_HD PreContactCar MakePreContactCar(const ::rocketgpu::CarState& car) {
    PreContactCar out;
    out.rot = car.rot;
    out.vel = car.vel;
    out.vel_bt = car.vel_bt;
    out.ang_vel = car.ang_vel;
    out.is_demoed = car.is_demoed;
    return out;
}

struct BallCarReorderRowSpan {
    int car_local_idx = -1;
    int normal_start = 0;
    int normal_count = 0;
    int friction_start = 0;
    int friction_count = 0;
    int split_start = 0;
    int split_count = 0;
};

// WorldStepSoccar locals that cross from its contact stages (kWorldStepContacts) to the car-world rows and the
// solve (kWorldStepRows).
struct WorldStepCarry {
    int pair_split_row_count = 0;
    int pair_velocity_row_count = 0;
    int pair_friction_row_count = 0;
    int split_row_count = 0;
    int normal_row_count = 0;
    int friction_row_count = 0;
    BallCarReorderRowSpan ball_car_row_spans[8] = {};
    int ball_car_row_span_count = 0;
    int sorted_manifold_count = 0;
    ::rocketgpu::Vec3 ball_solver_linear_velocity_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    bool ball_sleeping = false;
    bool ball_awake_after_contact_rows = false;
    bool ball_settled_skip = false;
    bool ball_settled_verify = false;
};

// One car's car-world rows built apart (BuildCarWorldRowsForCar): the rows sit in the env's shared world rows at the
// car's slot, kMaxCarWorld*ContactRows apart, until PackCarWorldRows moves them up in car order.
struct CarWorldRowsOut {
    int split_row_count = 0;
    int velocity_row_count = 0;
    int friction_row_count = 0;
    int group_count = 0;
    SharedContactOrderGroup groups[kMaxCarWorldMeshesPerCar + 4] = {};
};

struct WorldSolverScratch {
    static constexpr int kMaxHeatseekerWorldNormals = 16;
    ::rocketgpu::Vec3 hs_world_normals[kMaxHeatseekerWorldNormals];
    int hs_world_normal_count = 0;

    bullet_port::btGjkEpaScratch car_triangle_gjk_epa_scratch = {};
    bullet_port::btGjkEpaScratch ball_car_gjk_epa_scratch = {};
    CarCarPairCachedContact pair_contact_cache[kMaxCarCarPairSlots] = {};
    PreContactCar pre_contact_cars[8] = {};
    ::rocketgpu::Vec3 deferred_linear_deltas[8] = {};
    ::rocketgpu::Vec3 deferred_angular_deltas[8] = {};

    ::rocketgpu::Vec3 cal_start_vel[8] = {};
    // cal_ball_hit_start_vel: ball velocity right after the tick-start limit/quantize (v3 state.phys.vel).
    ::rocketgpu::Vec3 cal_ball_start_vel = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    BallMeshPersistentManifold ball_world_mesh_manifolds[kMaxBallWorldMeshPersistentManifolds] = {};
    // v3 keeps the ball-static-plane manifold across ticks (dispatcher refresh + merge);
    // slots: floor, ceiling, +X-facing wall, -X-facing wall (AppendBallWorldSpecialContactRows plane order).
    BallMeshPersistentManifold ball_world_plane_manifolds[4] = {};
    // Settled-ball fast path (cal_ball_wake_on_set, WorldStepSoccar): ball_settled_state is the ball at the
    // start of its static-world stage (after the ball-car rows) on a tick where that stage returned it
    // bit-identical. The other fields are the rest of that stage's inputs.
    uint8_t ball_settled = 0;
    uint8_t ball_settled_world_entry_count = 0;
    int16_t ball_settled_world_entries[kBallSettledMaxWorldEntries] = {};
    uint32_t ball_settled_mutator_epoch = 0;
    float ball_settled_sub_dt = 0.0f;
    ::rocketgpu::Vec3 ball_settled_solver_vel_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::BallState ball_settled_state = {};

    int32_t bp_cell_i[9] = {};
    int32_t bp_cell_j[9] = {};
    int32_t bp_cell_k[9] = {};
    uint32_t bp_insert_seq[9] = {};
    uint32_t bp_seq_counter = 0;
    uint8_t bp_initialized = 0;
    CarWorldPlaneRetainedManifold car_world_plane_manifolds[kMaxCarWorldPlaneRetainedManifolds] = {};
    CarCarRetainedManifold car_car_retained_manifolds[kMaxCarCarRetainedManifolds] = {};
    BallCarRetainedManifold ball_car_retained_manifolds[kMaxBallCarRetainedManifolds] = {};

    int32_t bullet_manifold_count = 0;
    int32_t shared_contact_order_group_count = 0;

    WheelRayStage wheel_ray_stage[kWheelRayStageSlots] = {};
    WorldStepCarry world_step_carry = {};
    CarWorldRowsOut car_world_rows_out[8] = {};
};

ROCKETGPU_HD float BallSplitTurnErp() {
    return 0.1f;
}

ROCKETGPU_HD float SplitImpulsePenetrationErp() {
    return 0.8f;
}

ROCKETGPU_HD float BallSplitPenetrationErp() {
    return 0.8f;
}

ROCKETGPU_HD int BallSplitImpulseIterations() {
    return 10;
}

ROCKETGPU_HD int BallVelocityContactIterations() {
    return 10;
}

ROCKETGPU_HD float BulletRestitutionCurve(float rel_vel, float restitution) {
    const float rest = Abs(rel_vel) < 10.0f ? 0.0f : (restitution * -rel_vel);
    return rest > 0.0f ? rest : 0.0f;
}

ROCKETGPU_HD float BulletRestitutionCurveBt(float rel_vel_bt, float restitution) {
    const float rest = Abs(rel_vel_bt) < 0.2f ? 0.0f : (restitution * -rel_vel_bt);
    return rest > 0.0f ? rest : 0.0f;
}

ROCKETGPU_HD ::rocketgpu::Vec3 OrthonormalTangent(const ::rocketgpu::Vec3& normal) {
    const ::rocketgpu::Vec3 axis = Abs(normal.z) < 0.70710678f ? ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f) : ::rocketgpu::Vec3(1.0f, 0.0f, 0.0f);
    return Normalize(Cross(axis, normal), ::rocketgpu::Vec3(1.0f, 0.0f, 0.0f));
}

ROCKETGPU_HD ::rocketgpu::Vec3 UuVecToBtDiv(const ::rocketgpu::Vec3& value_uu) {
    constexpr float kUuPerBt = 50.0f;
    return ::rocketgpu::Vec3(
        value_uu.x / kUuPerBt,
        value_uu.y / kUuPerBt,
        value_uu.z / kUuPerBt);
}

ROCKETGPU_HD ::rocketgpu::Vec3 BallStoredLinearVelocityToBt(const ::rocketgpu::Vec3& value_uu) {
    constexpr float kUuPerBt = 50.0f;
    return ::rocketgpu::Vec3(
        value_uu.x / kUuPerBt,
        value_uu.y / kUuPerBt,
        value_uu.z / kUuPerBt);
}

ROCKETGPU_HD void SetBallOriginBt(::rocketgpu::BallState& ball, const ::rocketgpu::Vec3& pos_bt) {
    constexpr float kUuPerBt = 50.0f;
    ball.pos_bt = pos_bt;
    ball.pos = pos_bt * kUuPerBt;
}

ROCKETGPU_HD void SetBallLinearVelocityBt(::rocketgpu::BallState& ball, const ::rocketgpu::Vec3& vel_bt) {
    constexpr float kUuPerBt = 50.0f;
    ball.vel_bt = vel_bt;
    ball.vel = vel_bt * kUuPerBt;
}

ROCKETGPU_HD void SetBallOriginUu(::rocketgpu::BallState& ball, const ::rocketgpu::Vec3& pos_uu) {
    constexpr float kUuToBt = 1.0f / 50.0f;
    SetBallOriginBt(ball, pos_uu * kUuToBt);
}

ROCKETGPU_HD void SetBallLinearVelocityUu(::rocketgpu::BallState& ball, const ::rocketgpu::Vec3& vel_uu) {
    constexpr float kUuToBt = 1.0f / 50.0f;
    SetBallLinearVelocityBt(ball, vel_uu * kUuToBt);
}

ROCKETGPU_HD void SetCarOriginBt(::rocketgpu::CarState& car, const ::rocketgpu::Vec3& pos_bt) {
    constexpr float kUuPerBt = 50.0f;
    car.pos_bt = pos_bt;
    car.pos = pos_bt * kUuPerBt;
}

ROCKETGPU_HD void SetCarOriginUu(::rocketgpu::CarState& car, const ::rocketgpu::Vec3& pos_uu) {
    constexpr float kUuToBt = 1.0f / 50.0f;
    SetCarOriginBt(car, pos_uu * kUuToBt);
}

ROCKETGPU_HD void SetCarLinearVelocityBt(::rocketgpu::CarState& car, const ::rocketgpu::Vec3& vel_bt) {
    constexpr float kUuPerBt = 50.0f;
    car.vel_bt = vel_bt;
    car.vel = vel_bt * kUuPerBt;
}

ROCKETGPU_HD void SetCarLinearVelocityUu(::rocketgpu::CarState& car, const ::rocketgpu::Vec3& vel_uu) {
    constexpr float kUuToBt = 1.0f / 50.0f;
    car.vel = vel_uu;
    car.vel_bt = vel_uu * kUuToBt;
    car.vel_touched_solve_stage = 1;
}

ROCKETGPU_HD void IntegrateBallOriginBt(
    ::rocketgpu::BallState& ball,
    const ::rocketgpu::Vec3& vel_bt,
    float dt) {
    if (vel_bt.x == 0.0f && vel_bt.y == 0.0f && vel_bt.z == 0.0f) {
        return;
    }

    SetBallOriginBt(ball, ball.pos_bt + vel_bt * dt);
}

ROCKETGPU_HD int CarWorldPlaneSourceTag(const ::rocketgpu::Vec3& inward_normal, float boundary) {
    if (inward_normal.x < -0.5f) return 1;
    if (inward_normal.x > 0.5f) return 2;
    if (inward_normal.y < -0.5f) return boundary < -ArenaExtentY() ? 10 : 3;
    if (inward_normal.y > 0.5f) return boundary < -ArenaExtentY() ? 11 : 4;
    if (inward_normal.z < -0.5f) return boundary < -ArenaHeight() ? 9 : 5;
    if (inward_normal.z > 0.5f) return 6;
    return 0;
}

ROCKETGPU_HD ::rocketgpu::Vec3 BallVelocityAtPoint(
    const ::rocketgpu::BallState& ball,
    const ::rocketgpu::Vec3& rel_point_world,
    const BallVelocityAccumulator& acc) {
    return ball.vel + acc.delta_linear + Cross(ball.ang_vel + acc.delta_angular, rel_point_world);
}

ROCKETGPU_HD ::rocketgpu::Vec3 BallVelocityAtPoint(
    const ::rocketgpu::BallState& ball,
    const ::rocketgpu::Vec3& rel_point_world,
    const ::rocketgpu::Vec3& external_linear_delta) {
    return ball.vel + external_linear_delta + Cross(ball.ang_vel, rel_point_world);
}

ROCKETGPU_HD ::rocketgpu::Vec3 VelocityAtPoint(
    const ::rocketgpu::CarState& car,
    const ::rocketgpu::Vec3& rel_point_world,
    const ::rocketgpu::Vec3& external_linear_delta,
    const ::rocketgpu::Vec3& external_angular_delta,
    const BallVelocityAccumulator& acc) {
    return car.vel + external_linear_delta + acc.delta_linear +
        Cross(car.ang_vel + external_angular_delta + acc.delta_angular, rel_point_world);
}

ROCKETGPU_HD void ApplyBallVelocityAccumulatorImpulse(
    BallVelocityAccumulator& acc,
    const ::rocketgpu::Vec3& linear_component,
    const ::rocketgpu::Vec3& angular_component,
    float impulse,
    bool use_fma = false) {
    if (use_fma) {
        acc.delta_linear = Vec3(
            fmaf(linear_component.x, impulse, acc.delta_linear.x),
            fmaf(linear_component.y, impulse, acc.delta_linear.y),
            fmaf(linear_component.z, impulse, acc.delta_linear.z));
        acc.delta_angular = Vec3(
            fmaf(angular_component.x, impulse, acc.delta_angular.x),
            fmaf(angular_component.y, impulse, acc.delta_angular.y),
            fmaf(angular_component.z, impulse, acc.delta_angular.z));
        return;
    }
    acc.delta_linear += linear_component * impulse;
    acc.delta_angular += angular_component * impulse;
}

ROCKETGPU_HD ::rocketgpu::Vec3 ClosestPointOnTriangleBT(const ::rocketgpu::Vec3& p, const ::rocketgpu::Vec3& a, const ::rocketgpu::Vec3& b, const ::rocketgpu::Vec3& c) {
    const ::rocketgpu::Vec3 ab = b - a;
    const ::rocketgpu::Vec3 ac = c - a;
    const ::rocketgpu::Vec3 ap = p - a;

    const float d1 = Dot(ab, ap);
    const float d2 = Dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    const ::rocketgpu::Vec3 bp = p - b;
    const float d3 = Dot(ab, bp);
    const float d4 = Dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    const ::rocketgpu::Vec3 cp = p - c;
    const float d5 = Dot(ab, cp);
    const float d6 = Dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        return a + ab * v;
    }

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float v = d2 / (d2 - d6);
        return a + ac * v;
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const float v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * v;
    }

    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    return a + ab * v + ac * w;
}

ROCKETGPU_HD bool PointInTriangleBT(const ::rocketgpu::Vec3& p, const ::rocketgpu::Vec3& a, const ::rocketgpu::Vec3& b, const ::rocketgpu::Vec3& c, const ::rocketgpu::Vec3& normal) {
    const ::rocketgpu::Vec3 u = b - a;
    const ::rocketgpu::Vec3 v = c - a;
    const ::rocketgpu::Vec3 n = Cross(u, v);
    const float nLenSq = Dot(n, n);

    const ::rocketgpu::Vec3 w = p - a;

    const float gamma = Dot(Cross(u, w), n) / nLenSq;
    const float beta  = Dot(Cross(w, v), n) / nLenSq;
    const float alpha = 1.0f - gamma - beta;

    return (0.0f <= alpha) && (alpha <= 1.0f) &&
           (0.0f <= beta)  && (beta  <= 1.0f) &&
           (0.0f <= gamma) && (gamma <= 1.0f);
}

ROCKETGPU_HD bool DetectBallMeshTriangleContactEx(
    const ::rocketgpu::Vec3& sphere_center_bt,
    float radius_bt,
    float contact_breaking_threshold_bt,
    const ::rocketgpu::Vec3& tri_a_bt,
    const ::rocketgpu::Vec3& tri_b_bt,
    const ::rocketgpu::Vec3& tri_c_bt,
    ::rocketgpu::Vec3& out_normal,
    float& out_penetration_uu,
    ::rocketgpu::Vec3& out_contact_on_tri_bt,
    float* out_depth_bt = nullptr,
    const uint32_t* native_table = nullptr, int native_shift = 0,
    bool reject_back_face = false) {
    const float radiusWithThreshold = radius_bt + contact_breaking_threshold_bt;

    ::rocketgpu::Vec3 tri_normal = Cross(tri_b_bt - tri_a_bt, tri_c_bt - tri_a_bt);
    const float l2 = Dot(tri_normal, tri_normal);
    if (l2 < SIMD_EPSILON * SIMD_EPSILON) {
        return false;
    }
    tri_normal = tri_normal * (1.0f / SqrtScalar(l2));

    const ::rocketgpu::Vec3 p1ToCentre = sphere_center_bt - tri_a_bt;
    float distanceFromPlane = Dot(p1ToCentre, tri_normal);

    if (distanceFromPlane < 0.0f) {
        if (reject_back_face) {
            return false;
        }
        distanceFromPlane *= -1.0f;
        tri_normal = tri_normal * -1.0f;
    }

    if (distanceFromPlane >= radiusWithThreshold) {
        return false;
    }

    bool hasContact = false;
    ::rocketgpu::Vec3 contactPoint(0.0f, 0.0f, 0.0f);

    if (PointInTriangleBT(sphere_center_bt, tri_a_bt, tri_b_bt, tri_c_bt, tri_normal)) {
        hasContact = true;
        contactPoint = sphere_center_bt - tri_normal * distanceFromPlane;
    } else {
        const float contactCapsuleRadiusSqr = radiusWithThreshold * radiusWithThreshold;
        ::rocketgpu::Vec3 nearestOnEdge = ClosestPointOnTriangleBT(sphere_center_bt, tri_a_bt, tri_b_bt, tri_c_bt);
        const ::rocketgpu::Vec3 diff = nearestOnEdge - sphere_center_bt;
        const float distanceSqr = Dot(diff, diff);
        if (distanceSqr < contactCapsuleRadiusSqr) {
            hasContact = true;
            contactPoint = nearestOnEdge;
        }
    }

    if (!hasContact) {
        return false;
    }

    const ::rocketgpu::Vec3 contactToCentre = sphere_center_bt - contactPoint;
    const float distanceSqr = Dot(contactToCentre, contactToCentre);

    if (distanceSqr >= radiusWithThreshold * radiusWithThreshold) {
        return false;
    }

    constexpr float kBTToUU = 50.0f;

    if (distanceSqr > SIMD_EPSILON) {
        const float distance = SqrtScalar(distanceSqr);

        out_normal = native_table
            ? NativeSseSphereNormalizer{native_table, native_shift}(contactToCentre)
            : contactToCentre * (1.0f / distance);
        out_penetration_uu = (radius_bt - distance) * kBTToUU;

        if (out_depth_bt) {
            *out_depth_bt = -(radius_bt - distance);
        }
    } else {
        out_normal = tri_normal;
        out_penetration_uu = radius_bt * kBTToUU;
        if (out_depth_bt) {
            *out_depth_bt = -radius_bt;
        }
    }

    out_contact_on_tri_bt = contactPoint;

    return true;
}

ROCKETGPU_HD void AccumulateBallWorldContact(
    BallWorldContactAggregate& aggregate,
    const ::rocketgpu::Vec3& normal,
    float penetration,
    float rel_pos1_length_bt) {
    aggregate.valid = 1;
    aggregate.count += 1;
    aggregate.normal_sum += normal;
    aggregate.fallback_normal = normal;
    aggregate.penetration_sum += penetration;

    aggregate.rel_pos1_length_sum_bt += rel_pos1_length_bt;
}

ROCKETGPU_HD void InitializeBallVelocityContactRow(
    BallVelocityContactRow& row,
    const ::rocketgpu::Vec3& normal,
    float restitution,
    float ball_inv_mass,
    const ::rocketgpu::Vec3& rel_ball_contact,
    const ::rocketgpu::Vec3& ball_angular_component,
    int32_t other_local_idx,
    float other_inv_mass,
    const ::rocketgpu::Vec3& rel_other_contact,
    const ::rocketgpu::Vec3& other_angular_component,
    const ::rocketgpu::Vec3& restitution_ball_linear,
    const ::rocketgpu::Vec3& restitution_ball_angular,
    const ::rocketgpu::Vec3& restitution_other_linear,
    const ::rocketgpu::Vec3& restitution_other_angular,
    const ::rocketgpu::Vec3& rhs_ball_linear,
    const ::rocketgpu::Vec3& rhs_ball_angular,
    const ::rocketgpu::Vec3& rhs_other_linear,
    const ::rocketgpu::Vec3& rhs_other_angular,
    const ::rocketgpu::Vec3& external_ball_linear_delta = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f),
    const ::rocketgpu::Vec3& external_ball_angular_delta = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f),
    const ::rocketgpu::Vec3& external_other_linear_delta = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f),
    const ::rocketgpu::Vec3& external_other_angular_delta = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f),
    const ::rocketgpu::Vec3* bt_rel_ball_contact_override = nullptr,
    const ::rocketgpu::Vec3* bt_rel_other_contact_override = nullptr,
    const ::rocketgpu::Vec3* bt_ball_linear_velocity_override = nullptr,
    const ::rocketgpu::Vec3* bt_ball_angular_component_override = nullptr,
    const ::rocketgpu::Vec3* bt_other_angular_component_override = nullptr,
    const ::rocketgpu::Vec3* bt_other_linear_velocity_override = nullptr,
    const ::rocketgpu::Vec3* bt_other_external_impulse_override = nullptr,
    const ::rocketgpu::Vec3* bt_other_external_torque_override = nullptr,
    bool restitution_no_threshold = false) {
    row = BallVelocityContactRow{};

    float denom = 0.0f;
    if (ball_inv_mass > 0.0f) {
        denom += ball_inv_mass + Dot(normal, Cross(ball_angular_component, rel_ball_contact));
    }
    if (other_inv_mass > 0.0f) {
        denom += other_inv_mass + Dot(normal, Cross(-other_angular_component, rel_other_contact));
    }
    if (denom <= 1e-8f) {
        return;
    }

    const ::rocketgpu::Vec3 rel_ball_cross_normal = Cross(rel_ball_contact, normal);
    const ::rocketgpu::Vec3 rel_other_cross_normal = Cross(rel_other_contact, normal);
    constexpr float kUuPerBt = 50.0f;
    const ::rocketgpu::Vec3 bt_rel_ball_contact =
        bt_rel_ball_contact_override ? *bt_rel_ball_contact_override : UuVecToBtDiv(rel_ball_contact);
    const ::rocketgpu::Vec3 bt_rel_other_contact =
        bt_rel_other_contact_override ? *bt_rel_other_contact_override : UuVecToBtDiv(rel_other_contact);
    const ::rocketgpu::Vec3 bt_rel_ball_cross_normal = Cross(bt_rel_ball_contact, normal);
    const ::rocketgpu::Vec3 bt_rel_other_cross_normal = Cross(bt_rel_other_contact, normal);
    const ::rocketgpu::Vec3 bt_linear_velocity_b =
        bt_ball_linear_velocity_override ? *bt_ball_linear_velocity_override : BallStoredLinearVelocityToBt(restitution_ball_linear);
    const ::rocketgpu::Vec3 bt_angular_velocity_b = restitution_ball_angular;
    const ::rocketgpu::Vec3 bt_external_force_impulse_b = UuVecToBtDiv(external_ball_linear_delta);
    const ::rocketgpu::Vec3 bt_external_torque_impulse_b = external_ball_angular_delta;

    const ::rocketgpu::Vec3 bt_linear_velocity_a = bt_other_linear_velocity_override
        ? *bt_other_linear_velocity_override
        : UuVecToBtDiv(restitution_other_linear);
    const ::rocketgpu::Vec3 bt_angular_velocity_a = restitution_other_angular;
    const ::rocketgpu::Vec3 bt_external_force_impulse_a = bt_other_external_impulse_override
        ? *bt_other_external_impulse_override
        : UuVecToBtDiv(external_other_linear_delta);
    const ::rocketgpu::Vec3 bt_external_torque_impulse_a = bt_other_external_torque_override
        ? *bt_other_external_torque_override
        : external_other_angular_delta;
    const ::rocketgpu::Vec3 bt_rhs_ball_linear = bt_linear_velocity_b + bt_external_force_impulse_b;
    const ::rocketgpu::Vec3 bt_rhs_other_linear = bt_linear_velocity_a + bt_external_force_impulse_a;
    const float bt_rhs_ball_vel_n =
        Dot(bt_rhs_ball_linear, normal) + Dot(bt_rel_ball_cross_normal, bt_angular_velocity_b + bt_external_torque_impulse_b);
    const float bt_rhs_other_vel_n =
        Dot(bt_rhs_other_linear, normal) + Dot(bt_rel_other_cross_normal, bt_angular_velocity_a + bt_external_torque_impulse_a);
    const float bt_rhs_rel_vel_n = bt_rhs_ball_vel_n - bt_rhs_other_vel_n;
    const ::rocketgpu::Vec3 bt_restitution_ball_point_vel =
        bt_linear_velocity_b + Cross(bt_angular_velocity_b, bt_rel_ball_contact);
    const ::rocketgpu::Vec3 bt_restitution_other_point_vel =
        bt_linear_velocity_a + Cross(bt_angular_velocity_a, bt_rel_other_contact);
    const ::rocketgpu::Vec3 bt_restitution_rel_vel =
        bt_restitution_ball_point_vel - bt_restitution_other_point_vel;
    const float bt_restitution_rel_vel_n = Dot(bt_restitution_rel_vel, normal);
    row.valid = 1;
    row.other_local_idx = other_local_idx;
    row.normal = normal;
    row.rel_ball_contact = rel_ball_contact;
    row.rel_other_contact = rel_other_contact;
    row.rel_ball_cross_normal = rel_ball_cross_normal;
    row.rel_other_cross_normal = -rel_other_cross_normal;
    row.ball_angular_component = ball_angular_component;
    row.other_angular_component = other_angular_component;
    row.ball_inv_mass = ball_inv_mass;
    row.other_inv_mass = other_inv_mass;
    row.jac_diag_ab_inv = 1.0f / denom;
    if (bt_ball_angular_component_override) {
        float bt_denom = 0.0f;
        if (ball_inv_mass > 0.0f) {
            bt_denom += ball_inv_mass +
                Dot(normal, Cross(*bt_ball_angular_component_override, bt_rel_ball_contact));
        }
        if (other_inv_mass > 0.0f) {
            const ::rocketgpu::Vec3 bt_other_comp = bt_other_angular_component_override
                ? *bt_other_angular_component_override
                : other_angular_component * kUuPerBt;
            bt_denom += other_inv_mass +
                Dot(normal, Cross(-bt_other_comp, bt_rel_other_contact));
        }
        if (bt_denom > 1e-8f) {
            row.jac_diag_ab_inv = 1.0f / bt_denom;
        }
    }
    // v3 solver_constraint.rs restitution_curve has no 0.2 bt/s threshold.
    const float bt_restitution_curve_runtime = restitution_no_threshold
        ? Max(restitution * -bt_restitution_rel_vel_n, 0.0f)
        : BulletRestitutionCurveBt(bt_restitution_rel_vel_n, restitution);
    const float bt_velocity_error_runtime = bt_restitution_curve_runtime - bt_rhs_rel_vel_n;
    row.rhs = bt_velocity_error_runtime * row.jac_diag_ab_inv * kUuPerBt;
    row.has_bt = 1;
    row.rhs_bt = bt_velocity_error_runtime * row.jac_diag_ab_inv;
    row.rel_ball_cross_normal_bt = bt_rel_ball_cross_normal;
    row.rel_other_cross_normal_bt = -bt_rel_other_cross_normal;
    row.ball_angular_component_bt = bt_ball_angular_component_override
        ? *bt_ball_angular_component_override
        : ball_angular_component * kUuPerBt;
    row.other_angular_component_bt = bt_other_angular_component_override
        ? *bt_other_angular_component_override
        : other_angular_component * kUuPerBt;
}

ROCKETGPU_HD void InitializeBallFrictionContactRow(
    BallFrictionContactRow& row,
    const BallVelocityContactRow& normal_row,
    int32_t normal_row_idx,
    float friction,
    const ::rocketgpu::Vec3& tangent,
    const ::rocketgpu::Vec3& ball_tangent_angular_component,
    const ::rocketgpu::Vec3& other_tangent_angular_component,
    const ::rocketgpu::Vec3& base_rel_vel,
    const ::rocketgpu::Vec3& friction_ball_linear,
    const ::rocketgpu::Vec3& friction_ball_angular,
    const ::rocketgpu::Vec3& friction_other_linear,
    const ::rocketgpu::Vec3& friction_other_angular,
    const ::rocketgpu::Vec3& external_ball_linear_delta,
    const ::rocketgpu::Vec3& external_other_linear_delta,
    const ::rocketgpu::Vec3* bt_rel_ball_contact_override = nullptr,
    const ::rocketgpu::Vec3* bt_rel_other_contact_override = nullptr,
    const ::rocketgpu::Vec3* bt_ball_linear_velocity_override = nullptr,
    const ::rocketgpu::Vec3* bt_ball_angular_component_override = nullptr,
    const ::rocketgpu::Vec3* bt_other_angular_component_override = nullptr,
    const ::rocketgpu::Vec3* bt_other_linear_velocity_override = nullptr,
    const ::rocketgpu::Vec3* bt_other_external_impulse_override = nullptr,
    const ::rocketgpu::Vec3* bt_other_angular_velocity_override = nullptr) {
    row = BallFrictionContactRow{};
    if (!normal_row.valid || friction <= 0.0f) {
        return;
    }

    const Vec3 normal_axis = -tangent;
    float denom0 = 0.0f;
    if (normal_row.other_inv_mass > 0.0f) {
        denom0 = normal_row.other_inv_mass +
            Dot(normal_axis, Cross(other_tangent_angular_component, normal_row.rel_other_contact));
    }
    float denom1 = 0.0f;
    if (normal_row.ball_inv_mass > 0.0f) {
        denom1 = normal_row.ball_inv_mass +
            Dot(normal_axis, Cross(-ball_tangent_angular_component, normal_row.rel_ball_contact));
    }
    const float denom = denom0 + denom1;
    if (denom <= 1e-8f) {
        return;
    }

    row.valid = 1;
    row.other_local_idx = normal_row.other_local_idx;
    row.normal_row_idx = normal_row_idx;
    row.tangent = tangent;
    row.rel_ball_contact = normal_row.rel_ball_contact;
    row.rel_other_contact = normal_row.rel_other_contact;
    row.rel_ball_cross_tangent = Cross(normal_row.rel_ball_contact, tangent);
    row.rel_other_cross_tangent = -Cross(normal_row.rel_other_contact, tangent);
    row.ball_angular_component = ball_tangent_angular_component;
    row.other_angular_component = other_tangent_angular_component;
    row.ball_inv_mass = normal_row.ball_inv_mass;
    row.other_inv_mass = normal_row.other_inv_mass;
    row.jac_diag_ab_inv = 1.0f / denom;
    constexpr float kUuPerBtFriction = 50.0f;
    const ::rocketgpu::Vec3 bt_rel_ball_contact =
        bt_rel_ball_contact_override ? *bt_rel_ball_contact_override : UuVecToBtDiv(normal_row.rel_ball_contact);
    const ::rocketgpu::Vec3 bt_rel_other_contact =
        bt_rel_other_contact_override ? *bt_rel_other_contact_override : UuVecToBtDiv(normal_row.rel_other_contact);
    const ::rocketgpu::Vec3 bt_ball_linear =
        bt_ball_linear_velocity_override ? *bt_ball_linear_velocity_override : BallStoredLinearVelocityToBt(friction_ball_linear);
    const ::rocketgpu::Vec3 bt_ball_external = UuVecToBtDiv(external_ball_linear_delta);

    const ::rocketgpu::Vec3 bt_other_linear = bt_other_linear_velocity_override
        ? *bt_other_linear_velocity_override
        : UuVecToBtDiv(friction_other_linear);
    const ::rocketgpu::Vec3 bt_other_external = bt_other_external_impulse_override
        ? *bt_other_external_impulse_override
        : UuVecToBtDiv(external_other_linear_delta);
    const ::rocketgpu::Vec3 bt_other_angular_for_rhs = bt_other_angular_velocity_override
        ? *bt_other_angular_velocity_override
        : friction_other_angular;
    const bool trace_other_body_first = row.other_local_idx >= 0;
    const ::rocketgpu::Vec3 bt_contact_normal1 = trace_other_body_first ? -tangent : tangent;
    const ::rocketgpu::Vec3 bt_contact_normal2 = trace_other_body_first ? tangent : ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    const ::rocketgpu::Vec3 bt_relpos1_cross_normal =
        trace_other_body_first ? Cross(bt_rel_other_contact, bt_contact_normal1)
                               : Cross(bt_rel_ball_contact, bt_contact_normal1);
    const ::rocketgpu::Vec3 bt_relpos2_cross_normal =
        trace_other_body_first ? Cross(bt_rel_ball_contact, bt_contact_normal2)
                               : ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    const float bt_vel1_dotn = trace_other_body_first
        ? Dot(bt_contact_normal1, bt_other_linear + bt_other_external) +
              Dot(bt_relpos1_cross_normal, bt_other_angular_for_rhs)
        : Dot(bt_contact_normal1, bt_ball_linear + bt_ball_external) +
              Dot(bt_relpos1_cross_normal, friction_ball_angular);
    const float bt_vel2_dotn = trace_other_body_first
        ? Dot(bt_contact_normal2, bt_ball_linear + bt_ball_external) +
              Dot(bt_relpos2_cross_normal, friction_ball_angular)
        : 0.0f;
    const float bt_rel_vel = bt_vel1_dotn + bt_vel2_dotn;
    const float bt_velocity_error = -bt_rel_vel;
    if (bt_ball_angular_component_override) {
        float bt_denom0 = 0.0f;
        if (normal_row.other_inv_mass > 0.0f) {
            const ::rocketgpu::Vec3 bt_other_comp = bt_other_angular_component_override
                ? *bt_other_angular_component_override
                : other_tangent_angular_component * kUuPerBtFriction;
            bt_denom0 = normal_row.other_inv_mass +
                Dot(normal_axis, Cross(bt_other_comp, bt_rel_other_contact));
        }
        float bt_denom1 = 0.0f;
        if (normal_row.ball_inv_mass > 0.0f) {
            bt_denom1 = normal_row.ball_inv_mass +
                Dot(normal_axis, Cross(-*bt_ball_angular_component_override, bt_rel_ball_contact));
        }
        const float bt_denom = bt_denom0 + bt_denom1;
        if (bt_denom > 1e-8f) {
            row.jac_diag_ab_inv = 1.0f / bt_denom;
        }
    }
    row.rhs = (bt_velocity_error * row.jac_diag_ab_inv) * kUuPerBtFriction;
    row.has_bt = 1;
    row.rhs_bt = bt_velocity_error * row.jac_diag_ab_inv;
    row.rel_ball_cross_tangent_bt = Cross(bt_rel_ball_contact, tangent);
    row.rel_other_cross_tangent_bt = -Cross(bt_rel_other_contact, tangent);
    row.ball_angular_component_bt = bt_ball_angular_component_override
        ? *bt_ball_angular_component_override
        : ball_tangent_angular_component * kUuPerBtFriction;
    row.other_angular_component_bt = bt_other_angular_component_override
        ? *bt_other_angular_component_override
        : other_tangent_angular_component * kUuPerBtFriction;
    row.friction = friction;
    (void)base_rel_vel;
}

ROCKETGPU_HD void AppendBallVelocityContactRows(
    EnvStridedArray<BallVelocityContactRow> normal_rows,
    int& normal_row_count,
    EnvStridedArray<BallFrictionContactRow> friction_rows,
    int& friction_row_count,
    const ::rocketgpu::Vec3& normal,
    float restitution,
    float friction,
    float ball_inv_mass,
    const ::rocketgpu::Vec3& rel_ball_contact,
    const ::rocketgpu::Vec3& ball_angular_component,
    int32_t other_local_idx,
    float other_inv_mass,
    const ::rocketgpu::Vec3& rel_other_contact,
    const ::rocketgpu::Vec3& other_angular_component,
    const ::rocketgpu::Vec3& ball_tangent_angular_component,
    const ::rocketgpu::Vec3& other_tangent_angular_component,
    const ::rocketgpu::Vec3& restitution_ball_linear,
    const ::rocketgpu::Vec3& restitution_ball_angular,
    const ::rocketgpu::Vec3& restitution_other_linear,
    const ::rocketgpu::Vec3& restitution_other_angular,
    const ::rocketgpu::Vec3& rhs_ball_linear,
    const ::rocketgpu::Vec3& rhs_ball_angular,
    const ::rocketgpu::Vec3& rhs_other_linear,
    const ::rocketgpu::Vec3& rhs_other_angular,
    const ::rocketgpu::Vec3& external_ball_linear_delta,
    const ::rocketgpu::Vec3& external_ball_angular_delta,
    const ::rocketgpu::Vec3& external_other_linear_delta,
    const ::rocketgpu::Vec3& external_other_angular_delta,
    const ::rocketgpu::Vec3& tangent_basis_rel_vel,
    const ::rocketgpu::Vec3& friction_base_rel_vel,
    const ::rocketgpu::Vec3* bt_rel_ball_contact_override = nullptr,
    const ::rocketgpu::Vec3* bt_rel_other_contact_override = nullptr,
    const ::rocketgpu::Vec3* bt_ball_linear_velocity_override = nullptr,
    const ::rocketgpu::Vec3* bt_ball_normal_angular_override = nullptr,
    const ::rocketgpu::Vec3* bt_other_normal_angular_override = nullptr,
    const ::rocketgpu::Vec3* bt_ball_tangent_angular_override = nullptr,
    const ::rocketgpu::Vec3* bt_other_tangent_angular_override = nullptr,
    const ::rocketgpu::Vec3* bt_tangent_basis_rel_vel_override = nullptr,
    const ::rocketgpu::Vec3* bt_other_linear_velocity_override = nullptr,
    const ::rocketgpu::Vec3* bt_other_external_impulse_override = nullptr,
    const ::rocketgpu::Vec3* bt_other_angular_velocity_override = nullptr,
    const ::rocketgpu::Vec3* bt_other_external_torque_override = nullptr,
    bool restitution_no_threshold = false) {
    if (!normal_rows || normal_row_count >= kMaxBallVelocityContactRows) {
        return;
    }

    BallVelocityContactRow& normal_row = normal_rows[normal_row_count];
    InitializeBallVelocityContactRow(
        normal_row,
        normal,
        restitution,
        ball_inv_mass,
        rel_ball_contact,
        ball_angular_component,
        other_local_idx,
        other_inv_mass,
        rel_other_contact,
        other_angular_component,
        restitution_ball_linear,
        restitution_ball_angular,
        restitution_other_linear,
        restitution_other_angular,
        rhs_ball_linear,
        rhs_ball_angular,
        rhs_other_linear,
        rhs_other_angular,
        external_ball_linear_delta,
        external_ball_angular_delta,
        external_other_linear_delta,
        external_other_angular_delta,
        bt_rel_ball_contact_override,
        bt_rel_other_contact_override,
        bt_ball_linear_velocity_override,
        bt_ball_normal_angular_override,
        bt_other_normal_angular_override,
        bt_other_linear_velocity_override,
        bt_other_external_impulse_override,
        bt_other_external_torque_override,
        restitution_no_threshold);
    if (!normal_row.valid) {
        return;
    }

    const int normal_row_idx = normal_row_count;
    ++normal_row_count;

    if (!friction_rows || friction_row_count >= kMaxBallFrictionContactRows) {
        return;
    }

    BallFrictionContactRow& friction_row = friction_rows[friction_row_count];
    ::rocketgpu::Vec3 tangent;
    if (bt_tangent_basis_rel_vel_override) {
        const ::rocketgpu::Vec3 bt_basis = *bt_tangent_basis_rel_vel_override;
        const float bt_rel_vel_n = Dot(normal, bt_basis);
        const ::rocketgpu::Vec3 bt_lat = bt_basis - normal * bt_rel_vel_n;
        const float bt_lat_len2 = Dot(bt_lat, bt_lat);
        if (bt_lat_len2 > 1.1920928955078125e-7f) {
            tangent = bt_lat * (1.0f / sqrtf(bt_lat_len2));
        } else {
            tangent = OrthonormalTangent(normal);
        }
    } else {
        const float tangent_basis_rel_vel_n = Dot(tangent_basis_rel_vel, normal);
        tangent = tangent_basis_rel_vel - normal * tangent_basis_rel_vel_n;
        const float tangent_len = Length(tangent);
        tangent = tangent_len > 1e-6f ? (tangent / tangent_len) : OrthonormalTangent(normal);
    }
    InitializeBallFrictionContactRow(
        friction_row,
        normal_row,
        normal_row_idx,
        friction,
        tangent,
        ball_tangent_angular_component,
        other_tangent_angular_component,
        friction_base_rel_vel,
        restitution_ball_linear,
        restitution_ball_angular,
        restitution_other_linear,
        restitution_other_angular,
        external_ball_linear_delta,
        external_other_linear_delta,
        bt_rel_ball_contact_override,
        bt_rel_other_contact_override,
        bt_ball_linear_velocity_override,
        bt_ball_tangent_angular_override,
        bt_other_tangent_angular_override,
        bt_other_linear_velocity_override,
        bt_other_external_impulse_override,
        bt_other_angular_velocity_override);
    if (friction_row.valid) {
        ++friction_row_count;
    }
}

ROCKETGPU_HD void InitializeCarWorldVelocityContactRow(
    CarWorldVelocityContactRow& row,
    const ::rocketgpu::Vec3& normal,
    float restitution,
    float inv_mass,
    const ::rocketgpu::Vec3& rel_contact,
    const ::rocketgpu::Vec3& angular_component,
    const ::rocketgpu::Vec3& restitution_rel_vel,
    const ::rocketgpu::Vec3& rhs_rel_vel,
    const CarWorldVelocityTraceInputs* trace_inputs = nullptr,
    const ::rocketgpu::Vec3* angular_component_bt_override = nullptr) {
    row = CarWorldVelocityContactRow{};
    constexpr float kCarWorldBtPerUu = 1.0f / 50.0f;
    constexpr float kCarWorldUuPerBt = 50.0f;
    const ::rocketgpu::Vec3 rel_contact_bt =
        (trace_inputs != nullptr && trace_inputs->has_rel_contact_bt)
            ? trace_inputs->rel_contact_bt
            : rel_contact * kCarWorldBtPerUu;
    const float denom = inv_mass + Dot(normal, Cross(angular_component, rel_contact));
    if (denom <= 1e-8f) {
        return;
    }

    const float restitution_rel_vel_n = Dot(restitution_rel_vel, normal);
    const float rhs_rel_vel_n = Dot(rhs_rel_vel, normal);
    row.valid = 1;
    row.normal = normal;
    row.rel_contact = rel_contact;
    row.rel_contact_bt = rel_contact_bt;
    row.rel_cross_normal_bt = Cross(rel_contact_bt, normal);
    row.rel_cross_normal = row.rel_cross_normal_bt * kCarWorldUuPerBt;
    row.angular_component = angular_component;
    row.angular_component_bt = angular_component_bt_override
        ? *angular_component_bt_override
        : angular_component * kCarWorldUuPerBt;
    row.has_bt = 1;
    row.inv_mass = inv_mass;
    row.jac_diag_ab_inv = 1.0f / denom;
    if (angular_component_bt_override) {
        const float bt_denom = inv_mass +
            Dot(normal, Cross(row.angular_component_bt, rel_contact_bt));
        if (bt_denom > 1e-8f) {
            row.jac_diag_ab_inv = 1.0f / bt_denom;
        }
    }
    float restitution_rel_vel_n_bt = restitution_rel_vel_n * kCarWorldBtPerUu;
    float rhs_rel_vel_n_bt = rhs_rel_vel_n * kCarWorldBtPerUu;
    float row_rhs_bt = (BulletRestitutionCurve(restitution_rel_vel_n, restitution) - rhs_rel_vel_n) *
        row.jac_diag_ab_inv * kCarWorldBtPerUu;
    if (trace_inputs != nullptr) {
        const ::rocketgpu::Vec3 rel_cross_normal_bt = row.rel_cross_normal_bt;
        const ::rocketgpu::Vec3 linear_velocity_bt = (trace_inputs->has_bt_inputs ? trace_inputs->linear_velocity_bt : trace_inputs->linear_velocity * kCarWorldBtPerUu);
        const ::rocketgpu::Vec3 external_force_impulse_bt = (trace_inputs->has_bt_inputs ? trace_inputs->external_force_impulse_bt : trace_inputs->external_linear_delta * kCarWorldBtPerUu);
        const bullet_port::btVector3 bt_normal(normal.x, normal.y, normal.z);
        const bullet_port::btVector3 bt_linear_velocity(linear_velocity_bt.x, linear_velocity_bt.y, linear_velocity_bt.z);
        const bullet_port::btVector3 bt_angular_velocity(
            trace_inputs->angular_velocity.x,
            trace_inputs->angular_velocity.y,
            trace_inputs->angular_velocity.z);
        const bullet_port::btVector3 bt_rel_contact(rel_contact_bt.x, rel_contact_bt.y, rel_contact_bt.z);
        restitution_rel_vel_n_bt =
            bt_normal.dot(bt_linear_velocity + bt_angular_velocity.cross(bt_rel_contact));
        rhs_rel_vel_n_bt =
            Dot(normal, linear_velocity_bt + external_force_impulse_bt) +
            Dot(rel_cross_normal_bt, trace_inputs->angular_velocity +
                (trace_inputs->has_bt_inputs ? trace_inputs->external_torque_impulse : trace_inputs->external_angular_delta));
        const float restitution_bt = trace_inputs->restitution_no_velocity_threshold
            ? Max(restitution * -restitution_rel_vel_n_bt, 0.0f)
            : BulletRestitutionCurveBt(restitution_rel_vel_n_bt, restitution);
        row_rhs_bt = (restitution_bt - rhs_rel_vel_n_bt) * row.jac_diag_ab_inv;
    }
    row.rhs = row_rhs_bt * kCarWorldUuPerBt;
    row.rhs_bt = row_rhs_bt;
    row.has_rhs_bt = 1;
}

ROCKETGPU_HD void InitializeCarWorldFrictionContactRow(
    CarWorldFrictionContactRow& row,
    const CarWorldVelocityContactRow& normal_row,
    int32_t normal_row_idx,
    float friction,
    const ::rocketgpu::Vec3& tangent,
    const ::rocketgpu::Vec3& angular_component,
    const ::rocketgpu::Vec3& base_rel_vel,
    const ::rocketgpu::Vec3* angular_component_bt_override = nullptr,
    const CarWorldVelocityTraceInputs* trace_inputs = nullptr) {
    row = CarWorldFrictionContactRow{};
    if (!normal_row.valid || friction <= 0.0f) {
        return;
    }

    const float denom = normal_row.inv_mass + Dot(tangent, Cross(angular_component, normal_row.rel_contact));
    if (denom <= 1e-8f) {
        return;
    }

    row.valid = 1;
    row.normal_row_idx = normal_row_idx;
    row.tangent = tangent;
    row.rel_contact = normal_row.rel_contact;
    row.rel_cross_tangent_bt = Cross(normal_row.rel_contact_bt, tangent);
    row.rel_cross_tangent = row.rel_cross_tangent_bt * 50.0f;
    row.angular_component = angular_component;
    row.angular_component_bt = angular_component_bt_override
        ? *angular_component_bt_override
        : angular_component * 50.0f;
    row.has_bt = 1;
    row.inv_mass = normal_row.inv_mass;
    row.jac_diag_ab_inv = 1.0f / denom;
    if (angular_component_bt_override) {
        const float bt_denom = normal_row.inv_mass +
            Dot(tangent, Cross(row.angular_component_bt, normal_row.rel_contact_bt));
        if (bt_denom > 1e-8f) {
            row.jac_diag_ab_inv = 1.0f / bt_denom;
        }
    }
    row.rhs = (-Dot(base_rel_vel, tangent)) * row.jac_diag_ab_inv;
    if (angular_component_bt_override && trace_inputs && trace_inputs->has_bt_inputs) {
        const float vel1_dotn_bt =
            Dot(tangent, trace_inputs->linear_velocity_bt + trace_inputs->external_force_impulse_bt) +
            Dot(row.rel_cross_tangent_bt, trace_inputs->angular_velocity);
        const float rel_vel_bt = vel1_dotn_bt + 0.0f;
        row.rhs_bt = (0.0f - rel_vel_bt) * row.jac_diag_ab_inv;
        row.has_rhs_bt = 1;
    }
    row.friction = friction;
}

ROCKETGPU_HD void AppendCarWorldVelocityContactRows(
    CarWorldVelocityContactRow* normal_rows,
    int& normal_row_count,
    CarWorldFrictionContactRow* friction_rows,
    int& friction_row_count,
    const ::rocketgpu::Vec3& normal,
    float restitution,
    float friction,
    float inv_mass,
    const ::rocketgpu::Vec3& rel_contact,
    const ::rocketgpu::Vec3& angular_component,
    const ::rocketgpu::Vec3& tangent_angular_component,
    const ::rocketgpu::Vec3& normal_restitution_rel_vel,
    const ::rocketgpu::Vec3& normal_base_rel_vel,
    const ::rocketgpu::Vec3& tangent_basis_rel_vel,
    const ::rocketgpu::Vec3& friction_base_rel_vel,
    const CarWorldVelocityTraceInputs* trace_inputs = nullptr,
    const ::rocketgpu::Vec3* angular_component_bt_override = nullptr,
    const ::rocketgpu::Vec3* tangent_angular_component_bt_override = nullptr,
    const ::rocketgpu::Vec3* tangent_override = nullptr) {
    if (!normal_rows || normal_row_count >= kMaxCarWorldVelocityContactRows) {
        return;
    }

    CarWorldVelocityContactRow& normal_row = normal_rows[normal_row_count];
    InitializeCarWorldVelocityContactRow(
        normal_row,
        normal,
        restitution,
        inv_mass,
        rel_contact,
        angular_component,
        normal_restitution_rel_vel,
        normal_base_rel_vel,
        trace_inputs,
        angular_component_bt_override);
    if (!normal_row.valid) {
        return;
    }

    const int normal_row_idx = normal_row_count;
    ++normal_row_count;

    if (!friction_rows || friction_row_count >= kMaxCarWorldFrictionContactRows) {
        return;
    }

    CarWorldFrictionContactRow& friction_row = friction_rows[friction_row_count];
    ::rocketgpu::Vec3 tangent(0.0f, 0.0f, 0.0f);
    if (tangent_override) {
        tangent = *tangent_override;
    } else {
        const float tangent_vel_n = Dot(tangent_basis_rel_vel, normal);
        tangent = tangent_basis_rel_vel - normal * tangent_vel_n;
        const float tangent_len = Length(tangent);
        tangent = tangent_len > 1e-6f ? (tangent / tangent_len) : OrthonormalTangent(normal);
    }
    InitializeCarWorldFrictionContactRow(
        friction_row,
        normal_row,
        normal_row_idx,
        friction,
        tangent,
        tangent_angular_component,
        friction_base_rel_vel,
        tangent_angular_component_bt_override,
        trace_inputs);
    if (friction_row.valid) {
        ++friction_row_count;
    }
}

ROCKETGPU_HD float SolverRowDot(const Vec3& a, const Vec3& b, bool use_fma) {
    return use_fma ? (a.x * b.x + a.y * b.y) + (a.z * b.z + 0.0f)
                   : BtSimdDot3LikeBullet(a, b);
}

ROCKETGPU_HD float SolverRowSubtract(float rhs, float velocity, float jac, bool use_fma) {
    return use_fma ? fmaf(-velocity, jac, rhs) : rhs - velocity * jac;
}

ROCKETGPU_HD void SolveBallVelocityNormalRowWithAccumulators(
    int count,
    BallVelocityContactRow& row,
    int row_idx,
    BallVelocityAccumulator& ball_acc,
    BallVelocityAccumulator* car_accs,
    int env_idx = -1,
    bool use_fma = false) {
    if (!row.valid) {
        return;
    }

    const bool trace_dynamic_row = row.other_local_idx >= 0 && row.other_local_idx < count;
    float delta_impulse = row.rhs;
    float trace_delta_vel1_dotn = 0.0f;
    float trace_delta_vel2_dotn = 0.0f;
    if (trace_dynamic_row && car_accs) {
        const BallVelocityAccumulator& other_acc = car_accs[row.other_local_idx];
        const float delta_vel1_dotn =
            SolverRowDot(-row.normal, other_acc.delta_linear, use_fma) +
            SolverRowDot(row.rel_other_cross_normal, other_acc.delta_angular, use_fma);
        const float delta_vel2_dotn =
            SolverRowDot(row.normal, ball_acc.delta_linear, use_fma) +
            SolverRowDot(row.rel_ball_cross_normal, ball_acc.delta_angular, use_fma);
        delta_impulse = SolverRowSubtract(delta_impulse, delta_vel1_dotn, row.jac_diag_ab_inv, use_fma);
        delta_impulse = SolverRowSubtract(delta_impulse, delta_vel2_dotn, row.jac_diag_ab_inv, use_fma);
        trace_delta_vel1_dotn = delta_vel1_dotn;
        trace_delta_vel2_dotn = delta_vel2_dotn;
    } else {
        const float delta_vel =
            SolverRowDot(row.normal, ball_acc.delta_linear, use_fma) +
            SolverRowDot(row.rel_ball_cross_normal, ball_acc.delta_angular, use_fma);
        delta_impulse = SolverRowSubtract(delta_impulse, delta_vel, row.jac_diag_ab_inv, use_fma);
    }
    (void)trace_delta_vel1_dotn;
    (void)trace_delta_vel2_dotn;

    const float sum = row.applied_impulse + delta_impulse;
    if (sum < 0.0f) {
        delta_impulse = -row.applied_impulse;
        row.applied_impulse = 0.0f;
    } else {
        row.applied_impulse = sum;
    }

    if (delta_impulse == 0.0f) {
#if !defined(__CUDA_ARCH__)
        if (trace_dynamic_row && car_accs) {
        } else if (!trace_dynamic_row) {
        }
#endif
        return;
    }

    ApplyBallVelocityAccumulatorImpulse(
        ball_acc,
        row.normal * row.ball_inv_mass,
        row.ball_angular_component,
        delta_impulse, use_fma);

    if (trace_dynamic_row && car_accs) {
        ApplyBallVelocityAccumulatorImpulse(
            car_accs[row.other_local_idx],
            (-row.normal) * row.other_inv_mass,
            row.other_angular_component,
            delta_impulse, use_fma);
    }

#if !defined(__CUDA_ARCH__)
        if (trace_dynamic_row && car_accs) {
        } else if (!trace_dynamic_row) {
        }
#endif
}

ROCKETGPU_HD void SolveBallVelocityFrictionRowWithAccumulators(
    int count,
    EnvStridedArray<BallVelocityContactRow> normal_rows,
    int normal_row_count,
    BallFrictionContactRow& row,
    int row_idx,
    BallVelocityAccumulator& ball_acc,
    BallVelocityAccumulator* car_accs,
    int env_idx = -1,
    bool use_fma = false) {
    if (!row.valid || !normal_rows || row.normal_row_idx < 0 || row.normal_row_idx >= normal_row_count) {
        return;
    }

    const float total_impulse = normal_rows[row.normal_row_idx].applied_impulse;
    if (total_impulse <= 0.0f) {
        return;
    }

    const bool trace_dynamic_row = row.other_local_idx >= 0 && row.other_local_idx < count;
    const float max_impulse = row.friction * total_impulse;
    float delta_impulse = row.rhs;
    float trace_delta_vel1_dotn = 0.0f;
    float trace_delta_vel2_dotn = 0.0f;
    if (trace_dynamic_row && car_accs) {
        const BallVelocityAccumulator& other_acc = car_accs[row.other_local_idx];
        const float delta_vel1_dotn =
            SolverRowDot(-row.tangent, other_acc.delta_linear, use_fma) +
            SolverRowDot(row.rel_other_cross_tangent, other_acc.delta_angular, use_fma);
        const float delta_vel2_dotn =
            SolverRowDot(row.tangent, ball_acc.delta_linear, use_fma) +
            SolverRowDot(row.rel_ball_cross_tangent, ball_acc.delta_angular, use_fma);
        delta_impulse = SolverRowSubtract(delta_impulse, delta_vel1_dotn, row.jac_diag_ab_inv, use_fma);
        delta_impulse = SolverRowSubtract(delta_impulse, delta_vel2_dotn, row.jac_diag_ab_inv, use_fma);
        trace_delta_vel1_dotn = delta_vel1_dotn;
        trace_delta_vel2_dotn = delta_vel2_dotn;
    } else {
        const float delta_vel =
            SolverRowDot(row.tangent, ball_acc.delta_linear, use_fma) +
            SolverRowDot(row.rel_ball_cross_tangent, ball_acc.delta_angular, use_fma);
        delta_impulse = SolverRowSubtract(delta_impulse, delta_vel, row.jac_diag_ab_inv, use_fma);
    }
    (void)trace_delta_vel1_dotn;
    (void)trace_delta_vel2_dotn;

    const float sum = row.applied_impulse + delta_impulse;
    if (sum < -max_impulse) {
        delta_impulse = -max_impulse - row.applied_impulse;
        row.applied_impulse = -max_impulse;
    } else if (sum > max_impulse) {
        delta_impulse = max_impulse - row.applied_impulse;
        row.applied_impulse = max_impulse;
    } else {
        row.applied_impulse = sum;
    }

    if (delta_impulse == 0.0f) {
#if !defined(__CUDA_ARCH__)
        if (trace_dynamic_row && car_accs) {
        } else if (!trace_dynamic_row) {
        }
#endif
        return;
    }

    ApplyBallVelocityAccumulatorImpulse(
        ball_acc,
        row.tangent * row.ball_inv_mass,
        row.ball_angular_component,
        delta_impulse, use_fma);

    if (trace_dynamic_row && car_accs) {
        ApplyBallVelocityAccumulatorImpulse(
            car_accs[row.other_local_idx],
            (-row.tangent) * row.other_inv_mass,
            row.other_angular_component,
            delta_impulse, use_fma);
    }

#if !defined(__CUDA_ARCH__)
        if (trace_dynamic_row && car_accs) {
        } else if (!trace_dynamic_row) {
        }
#endif
}

ROCKETGPU_HD void InitializeCarWorldSplitContactRow(
    CarWorldSplitContactRow& row,
    float penetration,
    const ::rocketgpu::Vec3& normal,
    float inv_mass,
    const ::rocketgpu::Vec3& rel_contact,
    const ::rocketgpu::Vec3& angular_component,
    float dt,
    const ::rocketgpu::Vec3* rel_contact_bt_override = nullptr,
    const ::rocketgpu::Vec3* angular_component_bt_override = nullptr,
    const float* penetration_bt_override = nullptr) {
    row = CarWorldSplitContactRow{};
    if (penetration <= 0.0f || dt <= 0.0f) {
        return;
    }

    const float denom = inv_mass + Dot(normal, Cross(angular_component, rel_contact));
    if (denom <= 1e-8f) {
        return;
    }

    row.valid = 1;
    row.normal = normal;
    row.rel_contact = rel_contact;
    row.rel_cross_normal = Cross(rel_contact, normal);
    row.angular_component = angular_component;
    row.inv_mass = inv_mass;
    row.jac_diag_ab_inv = 1.0f / denom;

    row.rhs_penetration = penetration * BallSplitPenetrationErp() * (1.0f / dt) * row.jac_diag_ab_inv;

    constexpr float kSplitUuToBt = 1.0f / 50.0f;
    row.rel_contact_bt = rel_contact_bt_override
        ? *rel_contact_bt_override
        : rel_contact * kSplitUuToBt;
    row.angular_component_bt = angular_component_bt_override
        ? *angular_component_bt_override
        : angular_component * 50.0f;
    row.rel_cross_normal_bt = Cross(row.rel_contact_bt, normal);
    const float denom_bt = inv_mass + Dot(normal, Cross(row.angular_component_bt, row.rel_contact_bt));
    row.jac_diag_ab_inv_bt = denom_bt > 1e-8f ? (1.0f / denom_bt) : row.jac_diag_ab_inv;

    const float penetration_bt = penetration_bt_override
        ? *penetration_bt_override
        : penetration * kSplitUuToBt;
    row.rhs_penetration_bt =
        penetration_bt * BallSplitPenetrationErp() * (1.0f / dt) * row.jac_diag_ab_inv_bt;
}

ROCKETGPU_HD void SolveCarWorldSplitContactRows(
    ::rocketgpu::CarState& car,
    CarWorldSplitContactRow* rows,
    int row_count,
    float dt,
    int env_idx = -1,
    int car_idx = -1,
    bool native_sse_math = false) {
    if (!rows || row_count <= 0) {
        return;
    }

    SplitImpulseAccumulator split;
    const bool debug_trace =
#if defined(__CUDA_ARCH__)
        false;
#else
        false;
#endif
    for (int iter = 0; iter < BallSplitImpulseIterations(); ++iter) {
        for (int row_idx = 0; row_idx < row_count; ++row_idx) {
            CarWorldSplitContactRow& row = rows[row_idx];
            if (!row.valid) {
                continue;
            }

            float delta_impulse = row.rhs_penetration_bt;
            const float delta_vel_linear = BtSimdDot3LikeBullet(row.normal, split.push_velocity);
            const float delta_vel_angular = BtSimdDot3LikeBullet(row.rel_cross_normal_bt, split.turn_velocity);
            delta_impulse -= (delta_vel_linear + delta_vel_angular) * row.jac_diag_ab_inv_bt;

            const float sum = row.applied_push_impulse_bt + delta_impulse;
            if (sum < 0.0f) {
                delta_impulse = -row.applied_push_impulse_bt;
                row.applied_push_impulse_bt = 0.0f;
            } else {
                row.applied_push_impulse_bt = sum;
            }

            if (delta_impulse == 0.0f) {
                continue;
            }

            split.push_velocity += (row.normal * row.inv_mass) * delta_impulse;
            split.turn_velocity += row.angular_component_bt * delta_impulse;
        }
    }

    IntegrateSplitImpulseTransformBt(car, split, dt, native_sse_math);
}

ROCKETGPU_HD void AppendCarWorldToSharedSplitRows(
    EnvStridedArray<CarSharedSplitContactRow> shared_rows,
    int& shared_row_count,
    int body_local_idx,
    const CarWorldSplitContactRow* rows,
    int row_count) {
    if (!shared_rows || !rows || row_count <= 0 || body_local_idx < 0) {
        return;
    }

    for (int row_idx = 0; row_idx < row_count; ++row_idx) {
        const CarWorldSplitContactRow& src = rows[row_idx];
        if (!src.valid || shared_row_count >= kMaxCarSharedSplitContactRows) {
            continue;
        }
        CarSharedSplitContactRow& dst = shared_rows[shared_row_count++];
        dst = CarSharedSplitContactRow{};
        dst.valid = 1;
        dst.source_tag = src.source_tag;
        dst.body_a_local_idx = body_local_idx;
        dst.body_b_local_idx = -1;
        dst.normal_a = src.normal;
        dst.rel_contact_a = src.rel_contact;
        dst.rel_a_cross_normal = src.rel_cross_normal;
        dst.angular_component_a = src.angular_component;
        dst.inv_mass_a = src.inv_mass;
        dst.jac_diag_ab_inv = src.jac_diag_ab_inv;
        dst.rhs_penetration = src.rhs_penetration;
        dst.applied_push_impulse = src.applied_push_impulse;
        dst.rel_a_cross_normal_bt = src.rel_cross_normal_bt;
        dst.angular_component_a_bt = src.angular_component_bt;
        dst.jac_diag_ab_inv_bt = src.jac_diag_ab_inv_bt;
        dst.rhs_penetration_bt = src.rhs_penetration_bt;
        dst.applied_push_impulse_bt = src.applied_push_impulse_bt;
        dst.has_bt = src.jac_diag_ab_inv_bt != 0.0f ? 1 : 0;
    }
}

ROCKETGPU_HD void AppendCarWorldToSharedVelocityRows(
    EnvStridedArray<CarSharedVelocityContactRow> shared_normal_rows,
    int& shared_normal_row_count,
    EnvStridedArray<CarSharedFrictionContactRow> shared_friction_rows,
    int& shared_friction_row_count,
    int body_local_idx,
    const CarWorldVelocityContactRow* normal_rows,
    int normal_row_count,
    const CarWorldFrictionContactRow* friction_rows,
    int friction_row_count) {
    if (body_local_idx < 0) {
        return;
    }

    const int normal_row_base = shared_normal_row_count;
    if (shared_normal_rows && normal_rows) {
        for (int row_idx = 0; row_idx < normal_row_count; ++row_idx) {
            const CarWorldVelocityContactRow& src = normal_rows[row_idx];
            if (!src.valid || shared_normal_row_count >= kMaxCarWorldSharedVelocityContactRows) {
                continue;
            }
            CarSharedVelocityContactRow& dst = shared_normal_rows[shared_normal_row_count++];
            dst = CarSharedVelocityContactRow{};
            dst.valid = 1;
            dst.source_tag = src.source_tag;
            dst.body_a_local_idx = body_local_idx;
            dst.body_b_local_idx = -1;
            dst.normal_a = src.normal;
            dst.rel_contact_a = src.rel_contact;
            dst.rel_a_cross_normal = src.rel_cross_normal;
            dst.angular_component_a = src.angular_component;
            dst.has_bt = src.has_bt;
            dst.rel_a_cross_normal_bt = src.rel_cross_normal_bt;
            dst.angular_component_a_bt = src.angular_component_bt;
            dst.inv_mass_a = src.inv_mass;
            dst.jac_diag_ab_inv = src.jac_diag_ab_inv;
            dst.rhs = src.rhs;
            dst.rhs_bt = src.rhs_bt;
            dst.has_rhs_bt = src.has_rhs_bt;
            dst.applied_impulse = src.applied_impulse;
        }
    }

    if (shared_friction_rows && friction_rows) {
        for (int row_idx = 0; row_idx < friction_row_count; ++row_idx) {
            const CarWorldFrictionContactRow& src = friction_rows[row_idx];
            if (!src.valid || shared_friction_row_count >= kMaxCarWorldSharedFrictionContactRows) {
                continue;
            }
            CarSharedFrictionContactRow& dst = shared_friction_rows[shared_friction_row_count++];
            dst = CarSharedFrictionContactRow{};
            dst.valid = 1;
            dst.normal_row_idx = normal_row_base + src.normal_row_idx;
            dst.source_tag = src.source_tag;
            dst.body_a_local_idx = body_local_idx;
            dst.body_b_local_idx = -1;
            dst.tangent_a = src.tangent;
            dst.rel_contact_a = src.rel_contact;
            dst.rel_a_cross_tangent = src.rel_cross_tangent;
            dst.angular_component_a = src.angular_component;
            dst.has_bt = src.has_bt;
            dst.rel_a_cross_tangent_bt = src.rel_cross_tangent_bt;
            dst.angular_component_a_bt = src.angular_component_bt;
            dst.inv_mass_a = src.inv_mass;
            dst.jac_diag_ab_inv = src.jac_diag_ab_inv;
            dst.rhs = src.rhs;

            dst.rhs_bt = src.rhs_bt;
            dst.has_rhs_bt = src.has_rhs_bt;
            dst.friction = src.friction;
            dst.applied_impulse = src.applied_impulse;
        }
    }
}

ROCKETGPU_HD void AppendSharedContactOrderGroup(
    EnvStridedArray<SharedContactOrderGroup> groups,
    int& group_count,
    SharedContactOrderBucket bucket,
    int body_a_local_idx,
    int body_b_local_idx,
    int source_tag,
    int normal_row_start,
    int normal_row_count,
    int friction_row_start,
    int friction_row_count,
    int initial_order) {
    if (!groups || group_count >= kMaxSharedContactOrderGroups || body_a_local_idx < 0) {
        return;
    }
    SharedContactOrderGroup& group = groups[group_count++];
    group = SharedContactOrderGroup{};
    group.valid = 1;
    group.bucket = bucket;
    group.body_a_local_idx = body_a_local_idx;
    group.body_b_local_idx = body_b_local_idx;
    group.source_tag = source_tag;
    group.normal_row_start = normal_row_start;
    group.normal_row_count = Max(0, normal_row_count);
    group.friction_row_start = friction_row_start;
    group.friction_row_count = Max(0, friction_row_count);
    group.initial_order = initial_order;
}

struct CarWorldSharedRowGroup {
    int start = 0;
    int count = 0;
    int island_id = 0;
    int source_tag = 0;
    int body_a_local_idx = -1;
    int body_b_local_idx = -1;
};

struct ContactSortScratch {
    CarWorldSharedRowGroup row_groups[kMaxSharedContactOrderGroups];
    int order[kMaxSharedContactOrderGroups];
    int island_ids[kMaxSharedContactOrderGroups];
    int desired_source_tags[kMaxSharedContactOrderGroups];
    int desired_body_a[kMaxSharedContactOrderGroups];
    int desired_body_b[kMaxSharedContactOrderGroups];
    uint8_t row_group_used[kMaxSharedContactOrderGroups];
};
template<bool External> struct ContactSortStorage {
    ContactSortScratch local;
    ROCKETGPU_HD ContactSortScratch& get(ContactSortScratch*) { return local; }
};
template<> struct ContactSortStorage<true> {
    ROCKETGPU_HD ContactSortScratch& get(ContactSortScratch* scratch) { return *scratch; }
};
#if defined(__CUDA_ARCH__)
constexpr bool kUseExternalContactSortScratch = true;
#else
constexpr bool kUseExternalContactSortScratch = false;
#endif

ROCKETGPU_HD void ReverseCarSharedVelocityRows(EnvStridedArray<CarSharedVelocityContactRow> rows, int begin, int end) {
    --end;
    while (begin < end) {
        const CarSharedVelocityContactRow tmp = rows[begin];
        rows[begin] = rows[end];
        rows[end] = tmp;
        ++begin;
        --end;
    }
}

ROCKETGPU_HD void ReverseCarSharedFrictionRows(EnvStridedArray<CarSharedFrictionContactRow> rows, int begin, int end) {
    --end;
    while (begin < end) {
        const CarSharedFrictionContactRow tmp = rows[begin];
        rows[begin] = rows[end];
        rows[end] = tmp;
        ++begin;
        --end;
    }
}

ROCKETGPU_HD void ReverseCarSharedSplitRows(EnvStridedArray<CarSharedSplitContactRow> rows, int begin, int end) {
    --end;
    while (begin < end) {
        const CarSharedSplitContactRow tmp = rows[begin];
        rows[begin] = rows[end];
        rows[end] = tmp;
        ++begin;
        --end;
    }
}

ROCKETGPU_HD int CarWorldIslandIdForRow(const CarSharedVelocityContactRow& row) {
    const int body_idx = row.body_a_local_idx >= 0 ? row.body_a_local_idx : row.body_b_local_idx;
    return body_idx >= 0 ? body_idx + 1 : -1;
}

ROCKETGPU_HD int CarWorldIslandIdForRow(const CarSharedSplitContactRow& row) {
    const int body_idx = row.body_a_local_idx >= 0 ? row.body_a_local_idx : row.body_b_local_idx;
    return body_idx >= 0 ? body_idx + 1 : -1;
}

ROCKETGPU_HD int FindSharedContactOrderParent(int* parent, int x);
ROCKETGPU_HD void UnionSharedContactOrderParents(int* parent, int a, int b);
ROCKETGPU_HD void ResolveSharedContactOrderCarIslandIds(int* parent, const int* component_max, int* car_island_id);
ROCKETGPU_HD int SharedContactOrderIslandIdForBody(const int* car_island_id, int body_a_local_idx);
ROCKETGPU_HD void QuickSortSharedContactOrderIndices(int* order, const int* island_ids, int lo, int hi);

ROCKETGPU_HD int CountSharedContactOrderGroupsForIsland(
    EnvStridedArray<const SharedContactOrderGroup> groups,
    int group_count,
    int car_count,
    int target_island_id) {
    if (!groups || group_count <= 0 || car_count <= 0 || target_island_id < 0) {
        return -1;
    }

    int parent[8];
    int component_max[8];
    for (int i = 0; i < 8; ++i) {
        parent[i] = i;
        component_max[i] = i;
    }
    for (int group_idx = 0; group_idx < group_count && group_idx < kMaxSharedContactOrderGroups; ++group_idx) {
        const SharedContactOrderGroup& group = groups[group_idx];
        if (!group.valid || group.bucket != SharedContactOrderBucket::Pair || group.normal_row_count <= 0) {
            continue;
        }
        UnionSharedContactOrderParents(parent, group.body_a_local_idx, group.body_b_local_idx);
    }
    for (int i = 0; i < car_count && i < 8; ++i) {
        const int root = FindSharedContactOrderParent(parent, i);
        if (root >= 0 && root < 8) {
            component_max[root] = Max(component_max[root], i);
        }
    }

    int island_group_count = 0;
    for (int group_idx = 0; group_idx < group_count && group_idx < kMaxSharedContactOrderGroups; ++group_idx) {
        const SharedContactOrderGroup& group = groups[group_idx];
        if (!group.valid ||
            group.normal_row_count <= 0 ||
            group.body_a_local_idx < 0 ||
            group.body_a_local_idx >= car_count ||
            group.body_a_local_idx >= 8) {
            continue;
        }
        const int root = FindSharedContactOrderParent(parent, group.body_a_local_idx);
        const int island_id = (root >= 0 && root < 8 ? component_max[root] : group.body_a_local_idx) + 1;
        if (island_id == target_island_id) {
            ++island_group_count;
        }
    }
    return island_group_count;
}

ROCKETGPU_HD bool IsBtStaticPlaneSourceTag(int source_tag) {
    return source_tag == 1 || source_tag == 2 || source_tag == 5 || source_tag == 6;
}

ROCKETGPU_HD bool IsBtXWallPlaneSourceTag(int source_tag) {
    return source_tag == 1 || source_tag == 2;
}

ROCKETGPU_HD bool IsBtFloorCeilingPlaneSourceTag(int source_tag) {
    return source_tag == 5 || source_tag == 6;
}

ROCKETGPU_HD int BulletContactOrderInitialKey(const SharedContactOrderGroup& group) {
    if (group.bucket == SharedContactOrderBucket::World) {
        const int src = group.source_tag;
        if (src >= 100) {
            return src - 100;
        }
        if (IsBtStaticPlaneSourceTag(src)) {
            int plane_rank = 0;
            if (src == 6)      plane_rank = 0;
            else if (src == 5) plane_rank = 1;
            else if (src == 2) plane_rank = 2;
            else if (src == 1) plane_rank = 3;
            return 1000 + plane_rank;
        }

        return group.initial_order;
    }

    return group.initial_order;
}

ROCKETGPU_HD bool SameCarWorldManifoldGroup(
    const CarSharedVelocityContactRow& a,
    const CarSharedVelocityContactRow& b) {
    return a.body_a_local_idx == b.body_a_local_idx &&
           a.body_b_local_idx == b.body_b_local_idx &&
           a.source_tag == b.source_tag;
}

ROCKETGPU_HD bool SameCarWorldManifoldGroup(
    const CarSharedSplitContactRow& a,
    const CarSharedSplitContactRow& b) {
    return a.body_a_local_idx == b.body_a_local_idx &&
           a.body_b_local_idx == b.body_b_local_idx &&
           a.source_tag == b.source_tag;
}

ROCKETGPU_HD bool SameCarWorldGroupOwner(
    const CarWorldSharedRowGroup& a,
    const CarWorldSharedRowGroup& b) {
    return a.island_id == b.island_id &&
           a.body_a_local_idx == b.body_a_local_idx &&
           a.body_b_local_idx == b.body_b_local_idx;
}

ROCKETGPU_HD void SwapAdjacentCarWorldSharedGroups(
    EnvStridedArray<CarSharedVelocityContactRow> normal_rows,
    EnvStridedArray<CarSharedFrictionContactRow> friction_rows,
    int friction_row_count,
    const CarWorldSharedRowGroup& first_group,
    const CarWorldSharedRowGroup& second_group) {
    const int first_start = first_group.start;
    const int first_end = first_group.start + first_group.count;
    const int second_start = second_group.start;
    const int second_end = second_group.start + second_group.count;
    if (first_end != second_start) {
        return;
    }

    ReverseCarSharedVelocityRows(normal_rows, first_start, first_end);
    ReverseCarSharedVelocityRows(normal_rows, second_start, second_end);
    ReverseCarSharedVelocityRows(normal_rows, first_start, second_end);

    if (!friction_rows || friction_row_count <= 0) {
        return;
    }

    int friction_first_start = -1;
    int friction_second_start = -1;
    int friction_end = -1;
    for (int i = 0; i < friction_row_count; ++i) {
        const int old_idx = friction_rows[i].normal_row_idx;
        const bool in_first = old_idx >= first_start && old_idx < first_end;
        const bool in_second = old_idx >= second_start && old_idx < second_end;
        if (in_first || in_second) {
            if (friction_first_start < 0) {
                friction_first_start = i;
            }
            if (in_second && friction_second_start < 0) {
                friction_second_start = i;
            }
            friction_end = i + 1;
        }
    }
    if (friction_first_start >= 0 && friction_second_start > friction_first_start && friction_end > friction_second_start) {
        ReverseCarSharedFrictionRows(friction_rows, friction_first_start, friction_second_start);
        ReverseCarSharedFrictionRows(friction_rows, friction_second_start, friction_end);
        ReverseCarSharedFrictionRows(friction_rows, friction_first_start, friction_end);
    }

    for (int i = 0; i < friction_row_count; ++i) {
        const int old_idx = friction_rows[i].normal_row_idx;
        if (old_idx >= first_start && old_idx < first_end) {
            friction_rows[i].normal_row_idx = old_idx + second_group.count;
        } else if (old_idx >= second_start && old_idx < second_end) {
            friction_rows[i].normal_row_idx = old_idx - first_group.count;
        }
    }
}

ROCKETGPU_HD void SwapAdjacentCarWorldSharedSplitGroups(
    EnvStridedArray<CarSharedSplitContactRow> rows,
    const CarWorldSharedRowGroup& first_group,
    const CarWorldSharedRowGroup& second_group) {
    const int first_start = first_group.start;
    const int first_end = first_group.start + first_group.count;
    const int second_start = second_group.start;
    const int second_end = second_group.start + second_group.count;
    if (!rows || first_end != second_start) {
        return;
    }

    ReverseCarSharedSplitRows(rows, first_start, first_end);
    ReverseCarSharedSplitRows(rows, second_start, second_end);
    ReverseCarSharedSplitRows(rows, first_start, second_end);
}

ROCKETGPU_HD void ReorderCarWorldSharedSplitRowsForBulletIslandSort(
    EnvStridedArray<CarSharedSplitContactRow> split_rows,
    int split_row_count,
    EnvStridedArray<const SharedContactOrderGroup> contact_order_groups = nullptr,
    int contact_order_group_count = 0,
    int car_count = 0) {
    if (!split_rows || split_row_count <= 1) {
        return;
    }

    int group_count = 0;
    int max_island_id = -1;
    int max_island_group_count = 0;
    CarWorldSharedRowGroup row_groups[kMaxSharedContactOrderGroups];
    CarWorldSharedRowGroup second_last_group;
    CarWorldSharedRowGroup last_group;
    CarWorldSharedRowGroup triple_mesh_group;
    CarWorldSharedRowGroup triple_plane_group;
    bool has_triple_mesh_plane_swap = false;
    for (int row_idx = 0; row_idx < split_row_count;) {
        const int start = row_idx;
        const CarSharedSplitContactRow& first = split_rows[start];
        ++row_idx;
        while (row_idx < split_row_count && SameCarWorldManifoldGroup(first, split_rows[row_idx])) {
            ++row_idx;
        }
        CarWorldSharedRowGroup group;
        group.start = start;
        group.count = row_idx - start;
        group.island_id = CarWorldIslandIdForRow(first);
        group.source_tag = first.source_tag;
        group.body_a_local_idx = first.body_a_local_idx;
        group.body_b_local_idx = first.body_b_local_idx;
        if (group_count < kMaxSharedContactOrderGroups) {
            row_groups[group_count] = group;
        }
        if (group.island_id > max_island_id) {
            max_island_id = group.island_id;
            max_island_group_count = 1;
        } else if (group.island_id == max_island_id) {
            ++max_island_group_count;
        }
        if (!has_triple_mesh_plane_swap &&
            group_count >= 2 &&
            SameCarWorldGroupOwner(second_last_group, last_group) &&
            SameCarWorldGroupOwner(last_group, group) &&
            IsBtXWallPlaneSourceTag(second_last_group.source_tag) &&
            last_group.source_tag >= 100 &&
            IsBtFloorCeilingPlaneSourceTag(group.source_tag)) {
            triple_mesh_group = last_group;
            triple_plane_group = group;
            has_triple_mesh_plane_swap = true;
        }
        second_last_group = last_group;
        last_group = group;
        ++group_count;
    }

    if (group_count <= 1) {
        return;
    }

    bool has_pair_order_group = false;
    bool applied_full_bullet_order = false;
    if (contact_order_groups && contact_order_group_count > 0 && car_count > 0 &&
        group_count <= kMaxSharedContactOrderGroups) {
        int parent[8];
        int component_max[8];
        for (int i = 0; i < 8; ++i) {
            parent[i] = i;
            component_max[i] = i;
        }
        for (int group_idx = 0; group_idx < contact_order_group_count && group_idx < kMaxSharedContactOrderGroups; ++group_idx) {
            const SharedContactOrderGroup& group = contact_order_groups[group_idx];
            if (!group.valid || group.bucket != SharedContactOrderBucket::Pair || group.normal_row_count <= 0) {
                continue;
            }
            has_pair_order_group = true;
            UnionSharedContactOrderParents(parent, group.body_a_local_idx, group.body_b_local_idx);
        }
        for (int i = 0; i < car_count && i < 8; ++i) {
            const int root = FindSharedContactOrderParent(parent, i);
            if (root >= 0 && root < 8) {
                component_max[root] = Max(component_max[root], i);
            }
        }

        int car_island_id[8];
        ResolveSharedContactOrderCarIslandIds(parent, component_max, car_island_id);

        int order[kMaxSharedContactOrderGroups];
        int island_ids[kMaxSharedContactOrderGroups];
        int active_count = 0;
        for (int group_idx = 0; group_idx < contact_order_group_count && group_idx < kMaxSharedContactOrderGroups; ++group_idx) {
            const SharedContactOrderGroup& group = contact_order_groups[group_idx];
            if (!group.valid ||
                group.normal_row_count <= 0 ||
                group.body_a_local_idx < 0 ||
                group.body_a_local_idx >= car_count) {
                continue;
            }
            int insert = active_count;
            const int group_order_key = BulletContactOrderInitialKey(group);
            while (insert > 0 &&
                   BulletContactOrderInitialKey(contact_order_groups[order[insert - 1]]) > group_order_key) {
                order[insert] = order[insert - 1];
                --insert;
            }
            order[insert] = group_idx;
            ++active_count;
        }
        for (int order_idx = 0; order_idx < active_count; ++order_idx) {
            const SharedContactOrderGroup& group = contact_order_groups[order[order_idx]];
            island_ids[order[order_idx]] = SharedContactOrderIslandIdForBody(car_island_id, group.body_a_local_idx);
        }
        if (active_count > 1) {
            QuickSortSharedContactOrderIndices(order, island_ids, 0, active_count - 1);
        }

        int desired_source_tags[kMaxSharedContactOrderGroups];
        int desired_body_a[kMaxSharedContactOrderGroups];
        int desired_body_b[kMaxSharedContactOrderGroups];
        uint8_t row_group_used[kMaxSharedContactOrderGroups] = {};
        int desired_count = 0;
        bool matched_all_world_groups = true;
        for (int order_idx = 0; order_idx < active_count; ++order_idx) {
            const SharedContactOrderGroup& group = contact_order_groups[order[order_idx]];
            if (!group.valid || group.bucket != SharedContactOrderBucket::World || group.normal_row_count <= 0) {
                continue;
            }
            int match_idx = -1;
            for (int row_group_idx = 0; row_group_idx < group_count; ++row_group_idx) {
                const CarWorldSharedRowGroup& row_group = row_groups[row_group_idx];
                if (!row_group_used[row_group_idx] &&
                    row_group.body_a_local_idx == group.body_a_local_idx &&
                    row_group.body_b_local_idx == group.body_b_local_idx &&
                    row_group.source_tag == group.source_tag) {
                    match_idx = row_group_idx;
                    break;
                }
            }
            if (match_idx < 0) {
                matched_all_world_groups = false;
                break;
            }
            row_group_used[match_idx] = 1;
            desired_source_tags[desired_count] = group.source_tag;
            desired_body_a[desired_count] = group.body_a_local_idx;
            desired_body_b[desired_count] = group.body_b_local_idx;
            ++desired_count;
        }

        if (matched_all_world_groups && desired_count == group_count) {
            for (int target_idx = 0; target_idx < desired_count; ++target_idx) {
                int match_idx = -1;
                for (int row_group_idx = target_idx; row_group_idx < group_count; ++row_group_idx) {
                    const CarWorldSharedRowGroup& row_group = row_groups[row_group_idx];
                    if (row_group.body_a_local_idx == desired_body_a[target_idx] &&
                        row_group.body_b_local_idx == desired_body_b[target_idx] &&
                        row_group.source_tag == desired_source_tags[target_idx]) {
                        match_idx = row_group_idx;
                        break;
                    }
                }
                while (match_idx > target_idx) {
                    CarWorldSharedRowGroup first = row_groups[match_idx - 1];
                    CarWorldSharedRowGroup second = row_groups[match_idx];
                    SwapAdjacentCarWorldSharedSplitGroups(split_rows, first, second);
                    second.start = first.start;
                    first.start = second.start + second.count;
                    row_groups[match_idx - 1] = second;
                    row_groups[match_idx] = first;
                    --match_idx;
                }
            }
            applied_full_bullet_order = true;
        }
    }
    if (applied_full_bullet_order) {
        return;
    }

    if (has_triple_mesh_plane_swap) {
        SwapAdjacentCarWorldSharedSplitGroups(split_rows, triple_mesh_group, triple_plane_group);
        return;
    }

    const bool has_two_group_max_suffix =
        second_last_group.island_id == max_island_id &&
        last_group.island_id == max_island_id;
    const int full_island_group_count = CountSharedContactOrderGroupsForIsland(
        contact_order_groups,
        contact_order_group_count,
        car_count,
        max_island_id);
    const int bullet_island_group_count =
        full_island_group_count >= 0 ? full_island_group_count : max_island_group_count;
    if (!has_two_group_max_suffix || bullet_island_group_count != 2) {
        return;
    }
    const bool has_static_mesh_suffix =
        (second_last_group.source_tag >= 100 && IsBtStaticPlaneSourceTag(last_group.source_tag)) ||
        (IsBtStaticPlaneSourceTag(second_last_group.source_tag) && last_group.source_tag >= 100);
    if (!has_static_mesh_suffix) {
        return;
    }
    if (!has_pair_order_group &&
        IsBtXWallPlaneSourceTag(second_last_group.source_tag) &&
        last_group.source_tag >= 100) {
        return;
    }
    SwapAdjacentCarWorldSharedSplitGroups(split_rows, second_last_group, last_group);
}

template<bool ExternalScratch = false>
ROCKETGPU_HD void ReorderCarWorldSharedRowsForBulletIslandSort(
    EnvStridedArray<CarSharedVelocityContactRow> normal_rows,
    int normal_row_count,
    EnvStridedArray<CarSharedFrictionContactRow> friction_rows,
    int friction_row_count,
    EnvStridedArray<const SharedContactOrderGroup> contact_order_groups = nullptr,
    int contact_order_group_count = 0,
    int car_count = 0, ContactSortScratch* external_scratch = nullptr) {
    if (!normal_rows || normal_row_count <= 1) {
        return;
    }

    int group_count = 0;
    int max_island_id = -1;
    int max_island_group_count = 0;
    ContactSortStorage<ExternalScratch> storage;
    auto& scratch = storage.get(external_scratch);
    auto& row_groups = scratch.row_groups;
    if constexpr (ExternalScratch) {
        for (int i=0;i<kMaxSharedContactOrderGroups;++i) row_groups[i]=CarWorldSharedRowGroup{};
    }
    CarWorldSharedRowGroup second_last_group;
    CarWorldSharedRowGroup last_group;
    CarWorldSharedRowGroup triple_mesh_group;
    CarWorldSharedRowGroup triple_plane_group;
    bool has_triple_mesh_plane_swap = false;
    for (int row_idx = 0; row_idx < normal_row_count;) {
        const int start = row_idx;
        const CarSharedVelocityContactRow& first = normal_rows[start];
        ++row_idx;
        while (row_idx < normal_row_count && SameCarWorldManifoldGroup(first, normal_rows[row_idx])) {
            ++row_idx;
        }
        CarWorldSharedRowGroup group;
        group.start = start;
        group.count = row_idx - start;
        group.island_id = CarWorldIslandIdForRow(first);
        group.source_tag = first.source_tag;
        group.body_a_local_idx = first.body_a_local_idx;
        group.body_b_local_idx = first.body_b_local_idx;
        if (group_count < kMaxSharedContactOrderGroups) {
            row_groups[group_count] = group;
        }
        if (group.island_id > max_island_id) {
            max_island_id = group.island_id;
            max_island_group_count = 1;
        } else if (group.island_id == max_island_id) {
            ++max_island_group_count;
        }
        if (!has_triple_mesh_plane_swap &&
            group_count >= 2 &&
            SameCarWorldGroupOwner(second_last_group, last_group) &&
            SameCarWorldGroupOwner(last_group, group) &&
            IsBtXWallPlaneSourceTag(second_last_group.source_tag) &&
            last_group.source_tag >= 100 &&
            IsBtFloorCeilingPlaneSourceTag(group.source_tag)) {
            triple_mesh_group = last_group;
            triple_plane_group = group;
            has_triple_mesh_plane_swap = true;
        }
        second_last_group = last_group;
        last_group = group;
        ++group_count;
    }

    if (group_count <= 1) {
        return;
    }

    bool has_pair_order_group = false;
    bool applied_full_bullet_order = false;
    if (contact_order_groups && contact_order_group_count > 0 && car_count > 0 &&
        group_count <= kMaxSharedContactOrderGroups) {
        int parent[8];
        int component_max[8];
        for (int i = 0; i < 8; ++i) {
            parent[i] = i;
            component_max[i] = i;
        }
        for (int group_idx = 0; group_idx < contact_order_group_count && group_idx < kMaxSharedContactOrderGroups; ++group_idx) {
            const SharedContactOrderGroup& group = contact_order_groups[group_idx];
            if (!group.valid || group.bucket != SharedContactOrderBucket::Pair || group.normal_row_count <= 0) {
                continue;
            }
            has_pair_order_group = true;
            UnionSharedContactOrderParents(parent, group.body_a_local_idx, group.body_b_local_idx);
        }
        for (int i = 0; i < car_count && i < 8; ++i) {
            const int root = FindSharedContactOrderParent(parent, i);
            if (root >= 0 && root < 8) {
                component_max[root] = Max(component_max[root], i);
            }
        }

        int car_island_id[8];
        ResolveSharedContactOrderCarIslandIds(parent, component_max, car_island_id);

        auto& order = scratch.order;
        auto& island_ids = scratch.island_ids;
        int active_count = 0;
        for (int group_idx = 0; group_idx < contact_order_group_count && group_idx < kMaxSharedContactOrderGroups; ++group_idx) {
            const SharedContactOrderGroup& group = contact_order_groups[group_idx];
            if (!group.valid ||
                group.normal_row_count <= 0 ||
                group.body_a_local_idx < 0 ||
                group.body_a_local_idx >= car_count) {
                continue;
            }
            int insert = active_count;
            const int group_order_key = BulletContactOrderInitialKey(group);
            while (insert > 0 &&
                   BulletContactOrderInitialKey(contact_order_groups[order[insert - 1]]) > group_order_key) {
                order[insert] = order[insert - 1];
                --insert;
            }
            order[insert] = group_idx;
            ++active_count;
        }
        for (int order_idx = 0; order_idx < active_count; ++order_idx) {
            const SharedContactOrderGroup& group = contact_order_groups[order[order_idx]];
            island_ids[order[order_idx]] = SharedContactOrderIslandIdForBody(car_island_id, group.body_a_local_idx);
        }
        if (active_count > 1) {
            QuickSortSharedContactOrderIndices(order, island_ids, 0, active_count - 1);
        }

        auto& desired_source_tags = scratch.desired_source_tags;
        auto& desired_body_a = scratch.desired_body_a;
        auto& desired_body_b = scratch.desired_body_b;
        auto& row_group_used = scratch.row_group_used;
        for (int i=0;i<kMaxSharedContactOrderGroups;++i) row_group_used[i]=0;
        int desired_count = 0;
        bool matched_all_world_groups = true;
        for (int order_idx = 0; order_idx < active_count; ++order_idx) {
            const SharedContactOrderGroup& group = contact_order_groups[order[order_idx]];
            if (!group.valid || group.bucket != SharedContactOrderBucket::World || group.normal_row_count <= 0) {
                continue;
            }
            int match_idx = -1;
            for (int row_group_idx = 0; row_group_idx < group_count; ++row_group_idx) {
                const CarWorldSharedRowGroup& row_group = row_groups[row_group_idx];
                if (!row_group_used[row_group_idx] &&
                    row_group.body_a_local_idx == group.body_a_local_idx &&
                    row_group.body_b_local_idx == group.body_b_local_idx &&
                    row_group.source_tag == group.source_tag) {
                    match_idx = row_group_idx;
                    break;
                }
            }
            if (match_idx < 0) {
                matched_all_world_groups = false;
                break;
            }
            row_group_used[match_idx] = 1;
            desired_source_tags[desired_count] = group.source_tag;
            desired_body_a[desired_count] = group.body_a_local_idx;
            desired_body_b[desired_count] = group.body_b_local_idx;
            ++desired_count;
        }

        if (matched_all_world_groups && desired_count == group_count) {
            for (int target_idx = 0; target_idx < desired_count; ++target_idx) {
                int match_idx = -1;
                for (int row_group_idx = target_idx; row_group_idx < group_count; ++row_group_idx) {
                    const CarWorldSharedRowGroup& row_group = row_groups[row_group_idx];
                    if (row_group.body_a_local_idx == desired_body_a[target_idx] &&
                        row_group.body_b_local_idx == desired_body_b[target_idx] &&
                        row_group.source_tag == desired_source_tags[target_idx]) {
                        match_idx = row_group_idx;
                        break;
                    }
                }
                while (match_idx > target_idx) {
                    CarWorldSharedRowGroup first = row_groups[match_idx - 1];
                    CarWorldSharedRowGroup second = row_groups[match_idx];
                    SwapAdjacentCarWorldSharedGroups(
                        normal_rows,
                        friction_rows,
                        friction_row_count,
                        first,
                        second);
                    second.start = first.start;
                    first.start = second.start + second.count;
                    row_groups[match_idx - 1] = second;
                    row_groups[match_idx] = first;
                    --match_idx;
                }
            }
            applied_full_bullet_order = true;
        }
    }
    if (applied_full_bullet_order) {
        return;
    }

    if (has_triple_mesh_plane_swap) {
        SwapAdjacentCarWorldSharedGroups(
            normal_rows,
            friction_rows,
            friction_row_count,
            triple_mesh_group,
            triple_plane_group);
        return;
    }

    const bool has_two_group_max_suffix =
        second_last_group.island_id == max_island_id &&
        last_group.island_id == max_island_id;
    const int full_island_group_count = CountSharedContactOrderGroupsForIsland(
        contact_order_groups,
        contact_order_group_count,
        car_count,
        max_island_id);
    const int bullet_island_group_count =
        full_island_group_count >= 0 ? full_island_group_count : max_island_group_count;
    if (!has_two_group_max_suffix || bullet_island_group_count != 2) {
        return;
    }
    const bool has_static_mesh_suffix =
        (second_last_group.source_tag >= 100 && IsBtStaticPlaneSourceTag(last_group.source_tag)) ||
        (IsBtStaticPlaneSourceTag(second_last_group.source_tag) && last_group.source_tag >= 100);
    if (!has_static_mesh_suffix) {
        return;
    }
    if (!has_pair_order_group &&
        IsBtXWallPlaneSourceTag(second_last_group.source_tag) &&
        last_group.source_tag >= 100) {
        return;
    }
    SwapAdjacentCarWorldSharedGroups(
        normal_rows,
        friction_rows,
        friction_row_count,
        second_last_group,
        last_group);
}

ROCKETGPU_HD void ReorderBallCarRowsForBulletQuickSortIslandFlip(
    EnvStridedArray<BallVelocityContactRow> normal_rows,
    int                     ,
    EnvStridedArray<BallFrictionContactRow> friction_rows,
    int friction_row_count,
    EnvStridedArray<BallSplitContactRow> split_rows,
    int                    ,
    const BallCarReorderRowSpan* row_spans,
    int row_span_count,
    EnvStridedArray<const SharedContactOrderGroup> contact_order_groups,
    int contact_order_group_count) {
    if (!normal_rows || !row_spans || row_span_count != 2) {
        return;
    }

    const BallCarReorderRowSpan& span_a = row_spans[0];
    const BallCarReorderRowSpan& span_b = row_spans[1];
    if (span_a.normal_count <= 0 || span_b.normal_count <= 0) {
        return;
    }

    if (span_a.normal_count != 1 || span_b.normal_count != 1 ||
        span_a.friction_count != 1 || span_b.friction_count != 1 ||
        span_a.split_count != 1 || span_b.split_count != 1) {
        return;
    }

    if (span_a.normal_start + span_a.normal_count != span_b.normal_start ||
        span_a.friction_start + span_a.friction_count != span_b.friction_start ||
        span_a.split_start + span_a.split_count != span_b.split_start) {
        return;
    }

    if (span_a.car_local_idx < 0 || span_a.car_local_idx >= 8 ||
        span_b.car_local_idx < 0 || span_b.car_local_idx >= 8) {
        return;
    }

    if (contact_order_groups && contact_order_group_count > 0) {
        for (int gi = 0; gi < contact_order_group_count && gi < kMaxSharedContactOrderGroups; ++gi) {
            const SharedContactOrderGroup& g = contact_order_groups[gi];
            if (!g.valid) {
                continue;
            }
            const bool a_touches = (g.body_a_local_idx == span_a.car_local_idx ||
                                    g.body_b_local_idx == span_a.car_local_idx);
            const bool b_touches = (g.body_a_local_idx == span_b.car_local_idx ||
                                    g.body_b_local_idx == span_b.car_local_idx);
            if (a_touches || b_touches) {
                return;
            }
        }
    }

    {
        const BallVelocityContactRow tmp = normal_rows[span_a.normal_start];
        normal_rows[span_a.normal_start] = normal_rows[span_b.normal_start];
        normal_rows[span_b.normal_start] = tmp;
    }
    if (friction_rows && friction_row_count > 0 &&
        span_a.friction_start < friction_row_count &&
        span_b.friction_start < friction_row_count) {
        BallFrictionContactRow tmp = friction_rows[span_a.friction_start];
        friction_rows[span_a.friction_start] = friction_rows[span_b.friction_start];
        friction_rows[span_b.friction_start] = tmp;

        friction_rows[span_a.friction_start].normal_row_idx = span_a.normal_start;
        friction_rows[span_b.friction_start].normal_row_idx = span_b.normal_start;
    }
    if (split_rows) {
        const BallSplitContactRow tmp = split_rows[span_a.split_start];
        split_rows[span_a.split_start] = split_rows[span_b.split_start];
        split_rows[span_b.split_start] = tmp;
    }
}

ROCKETGPU_HD void ApplySharedCarVelocityImpulse(
    BallVelocityAccumulator* car_accs,
    int body_local_idx,
    const ::rocketgpu::Vec3& linear_component,
    const ::rocketgpu::Vec3& angular_component,
    float impulse,
    bool use_fma = false) {
    if (!car_accs || body_local_idx < 0 || impulse == 0.0f) {
        return;
    }
    ApplyBallVelocityAccumulatorImpulse(
        car_accs[body_local_idx],
        linear_component,
        angular_component,
        impulse, use_fma);
}

ROCKETGPU_HD void WarmstartSharedCarVelocityRows(
    int count,
    const ::rocketgpu::CarState* cars,
    const ::rocketgpu::CarConfig* car_configs,
    float car_mass,
    BallVelocityAccumulator* car_accs,
    EnvStridedArray<CarSharedVelocityContactRow> rows,
    int row_count) {
    if (!cars || !car_configs || !car_accs || !rows || row_count <= 0) {
        return;
    }

    for (int row_idx = 0; row_idx < row_count; ++row_idx) {
        CarSharedVelocityContactRow& row = rows[row_idx];
        if (!row.valid || row.applied_impulse == 0.0f) {
            continue;
        }
        if (row.body_a_local_idx >= 0 && row.body_a_local_idx < count) {
            ApplySharedCarVelocityImpulse(
                car_accs,
                row.body_a_local_idx,
                row.normal_a * row.inv_mass_a,
                row.angular_component_a,
                row.applied_impulse);
        }
        if (row.body_b_local_idx >= 0 && row.body_b_local_idx < count) {
            const ::rocketgpu::Vec3 normal_b = -row.normal_a;
            ApplySharedCarVelocityImpulse(
                car_accs,
                row.body_b_local_idx,
                normal_b * row.inv_mass_b,
                row.angular_component_b,
                row.applied_impulse);
        }
    }
}

#if defined(__CUDA_ARCH__)
#define ROCKETGPU_SHARED_SOLVER_BACKEND_TAG " backend=device\n"
#else
#define ROCKETGPU_SHARED_SOLVER_BACKEND_TAG " backend=host\n"
#endif

ROCKETGPU_HD void SolveSharedCarVelocityNormalRow(
    int count,
    ::rocketgpu::CarState* cars,
    const ::rocketgpu::CarConfig* car_configs,
    float car_mass,
    CarSharedVelocityContactRow& row,
    BallVelocityAccumulator* car_accs,
    int iter,
    int row_idx,
    const char* bucket,
    int env_idx,
    bool use_fma = false) {
    if (!row.valid || !cars || !car_configs || !car_accs) {
        return;
    }
    if (row.body_a_local_idx < 0 || row.body_a_local_idx >= count) {
        return;
    }

    constexpr float kBtPerUu = 1.0f / 50.0f;
    constexpr float kUuPerBt = 50.0f;
    const ::rocketgpu::Vec3 normal_b = (row.body_b_local_idx >= 0 && row.body_b_local_idx < count) ? -row.normal_a : ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);

    const BallVelocityAccumulator& acc_a = car_accs[row.body_a_local_idx];
    const float delta_impulse_vel_a_bt =
        SolverRowDot(row.normal_a, acc_a.delta_linear, use_fma) +
        SolverRowDot(row.rel_a_cross_normal, acc_a.delta_angular, use_fma);
    float delta_impulse_vel_b_bt = 0.0f;

    if (row.body_b_local_idx >= 0 && row.body_b_local_idx < count) {
        const BallVelocityAccumulator& acc_b = car_accs[row.body_b_local_idx];
        delta_impulse_vel_b_bt =
            SolverRowDot(normal_b, acc_b.delta_linear, use_fma) +
            SolverRowDot(row.rel_b_cross_normal, acc_b.delta_angular, use_fma);
    }

    float delta_impulse = 0.0f;
    if (row.has_rhs_bt) {
        float delta_impulse_bt = row.rhs_bt;
        delta_impulse_bt = SolverRowSubtract(delta_impulse_bt, delta_impulse_vel_a_bt, row.jac_diag_ab_inv, use_fma);
        delta_impulse_bt = SolverRowSubtract(delta_impulse_bt, delta_impulse_vel_b_bt, row.jac_diag_ab_inv, use_fma);

        const float sum_bt = row.applied_impulse + delta_impulse_bt;
        if (sum_bt < 0.0f) {
            delta_impulse = -row.applied_impulse;
            row.applied_impulse = 0.0f;
        } else {
            delta_impulse = delta_impulse_bt;
            row.applied_impulse = sum_bt;
        }
    } else {
        const float delta_impulse_vel_a_uu = delta_impulse_vel_a_bt * kUuPerBt;
        const float delta_impulse_vel_b_uu = delta_impulse_vel_b_bt * kUuPerBt;
        const float applied_uu = row.applied_impulse * kUuPerBt;
        delta_impulse = row.rhs;
        delta_impulse = SolverRowSubtract(delta_impulse, delta_impulse_vel_a_uu, row.jac_diag_ab_inv, use_fma);
        delta_impulse = SolverRowSubtract(delta_impulse, delta_impulse_vel_b_uu, row.jac_diag_ab_inv, use_fma);

        const float sum = applied_uu + delta_impulse;
        if (sum < 0.0f) {
            delta_impulse = -applied_uu;
            row.applied_impulse = 0.0f;
        } else {
            row.applied_impulse = sum * kBtPerUu;
        }
        delta_impulse *= kBtPerUu;
    }

    if (delta_impulse == 0.0f) {
        return;
    }

    ApplySharedCarVelocityImpulse(
        car_accs,
        row.body_a_local_idx,
        row.normal_a * row.inv_mass_a,
        row.angular_component_a,
        delta_impulse, use_fma);
    if (row.body_b_local_idx >= 0 && row.body_b_local_idx < count) {
        ApplySharedCarVelocityImpulse(
            car_accs,
            row.body_b_local_idx,
            normal_b * row.inv_mass_b,
            row.angular_component_b,
            delta_impulse, use_fma);
    }
}

ROCKETGPU_HD void SolveSharedCarVelocityFrictionRow(
    int count,
    const ::rocketgpu::CarState* cars,
    const ::rocketgpu::CarConfig* car_configs,
    float car_mass,
    EnvStridedArray<const CarSharedVelocityContactRow> normal_rows,
    int normal_row_count,
    CarSharedFrictionContactRow& row,
    BallVelocityAccumulator* car_accs,
    int iter,
    int row_idx,
    const char* bucket,
    int env_idx,
    bool use_fma = false) {
    if (!row.valid || !cars || !car_configs || !car_accs || !normal_rows) {
        return;
    }
    if (row.normal_row_idx < 0 || row.normal_row_idx >= normal_row_count) {
        return;
    }
    if (row.body_a_local_idx < 0 || row.body_a_local_idx >= count) {
        return;
    }

    const ::rocketgpu::Vec3 tangent_b = (row.body_b_local_idx >= 0 && row.body_b_local_idx < count) ? -row.tangent_a : ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);

    const float total_impulse = normal_rows[row.normal_row_idx].applied_impulse;
    if (total_impulse <= 0.0f) {
        return;
    }

    constexpr float kBtPerUu = 1.0f / 50.0f;
    const float max_impulse = row.friction * total_impulse;
    float delta_impulse = row.has_rhs_bt ? row.rhs_bt : row.rhs * kBtPerUu;
    const BallVelocityAccumulator& acc_a = car_accs[row.body_a_local_idx];
    const float delta_impulse_vel_a_bt =
        SolverRowDot(row.tangent_a, acc_a.delta_linear, use_fma) +
        SolverRowDot(row.rel_a_cross_tangent, acc_a.delta_angular, use_fma);
    delta_impulse = SolverRowSubtract(delta_impulse, delta_impulse_vel_a_bt, row.jac_diag_ab_inv, use_fma);

    float delta_impulse_vel_b_bt = 0.0f;
    if (row.body_b_local_idx >= 0 && row.body_b_local_idx < count) {
        const BallVelocityAccumulator& acc_b = car_accs[row.body_b_local_idx];
        delta_impulse_vel_b_bt =
            SolverRowDot(tangent_b, acc_b.delta_linear, use_fma) +
            SolverRowDot(row.rel_b_cross_tangent, acc_b.delta_angular, use_fma);
        delta_impulse = SolverRowSubtract(delta_impulse, delta_impulse_vel_b_bt, row.jac_diag_ab_inv, use_fma);
    }

    const float sum = row.applied_impulse + delta_impulse;
    if (sum < -max_impulse) {
        delta_impulse = -max_impulse - row.applied_impulse;
        row.applied_impulse = -max_impulse;
    } else if (sum > max_impulse) {
        delta_impulse = max_impulse - row.applied_impulse;
        row.applied_impulse = max_impulse;
    } else {
        row.applied_impulse = sum;
    }

    if (delta_impulse == 0.0f) {
        return;
    }

    ApplySharedCarVelocityImpulse(
        car_accs,
        row.body_a_local_idx,
        row.tangent_a * row.inv_mass_a,
        row.angular_component_a,
        delta_impulse, use_fma);
    if (row.body_b_local_idx >= 0 && row.body_b_local_idx < count) {
        ApplySharedCarVelocityImpulse(
            car_accs,
            row.body_b_local_idx,
            tangent_b * row.inv_mass_b,
            row.angular_component_b,
            delta_impulse, use_fma);
    }
}

ROCKETGPU_HD void AppendCarPairSharedSplitRow(
    EnvStridedArray<CarSharedSplitContactRow> shared_rows,
    int& shared_row_count,
    int body_a_local_idx,
    int body_b_local_idx,
    int source_tag,
    const ::rocketgpu::Vec3& normal_a,
    const ::rocketgpu::Vec3& normal_b,
    const ::rocketgpu::Vec3& rel_contact_a,
    const ::rocketgpu::Vec3& rel_contact_b,
    const ::rocketgpu::Vec3& angular_component_a,
    const ::rocketgpu::Vec3& angular_component_b,
    float inv_mass_a,
    float inv_mass_b,
    float jac_diag_ab_inv,
    float rhs_penetration) {
    if (!shared_rows ||
        shared_row_count >= kMaxCarSharedSplitContactRows ||
        body_a_local_idx < 0 ||
        body_b_local_idx < 0 ||
        rhs_penetration <= 0.0f ||
        jac_diag_ab_inv <= 0.0f) {
        return;
    }

    CarSharedSplitContactRow& row = shared_rows[shared_row_count++];
    row = CarSharedSplitContactRow{};
    row.valid = 1;
    row.source_tag = source_tag;
    row.body_a_local_idx = body_a_local_idx;
    row.body_b_local_idx = body_b_local_idx;
    row.normal_a = normal_a;
    row.normal_b = normal_b;
    row.rel_contact_a = rel_contact_a;
    row.rel_contact_b = rel_contact_b;
    row.rel_a_cross_normal = Cross(rel_contact_a, normal_a);
    row.rel_b_cross_normal = Cross(rel_contact_b, normal_b);
    row.angular_component_a = angular_component_a;
    row.angular_component_b = angular_component_b;
    row.inv_mass_a = inv_mass_a;
    row.inv_mass_b = inv_mass_b;
    row.jac_diag_ab_inv = jac_diag_ab_inv;
    row.rhs_penetration = rhs_penetration;
}

ROCKETGPU_HD void MarkBtSplitAccumulators(
    int count,
    SplitImpulseAccumulator* car_splits,
    EnvStridedArray<const CarSharedSplitContactRow> world_rows,
    int world_row_count,
    EnvStridedArray<const CarSharedSplitContactRow> pair_rows,
    int pair_row_count,
    EnvStridedArray<const BallSplitContactRow> ball_rows,
    int ball_row_count,
    SplitImpulseAccumulator* ball_split = nullptr) {
    if (count < 0 || (count > 0 && !car_splits)) {
        return;
    }
    constexpr int kMarkCars = 8;
    bool touched_world[kMarkCars];
    bool blocked[kMarkCars];
    bool touched_ball = false;
    bool ball_blocked = false;
    for (int i = 0; i < kMarkCars; ++i) {
        touched_world[i] = false;
        blocked[i] = false;
    }
    for (int i = 0; i < count; ++i) {
        car_splits[i].is_bt = 0;
    }
    for (int r = 0; world_rows && r < world_row_count; ++r) {
        const CarSharedSplitContactRow& row = world_rows[r];
        if (!row.valid) {
            continue;
        }
        const int a = row.body_a_local_idx;
        if (a >= 0 && a < count && a < kMarkCars) {
            touched_world[a] = true;
            if (!row.has_bt || row.body_b_local_idx >= 0) {
                blocked[a] = true;
            }
        }
    }
    for (int r = 0; pair_rows && r < pair_row_count; ++r) {
        const CarSharedSplitContactRow& row = pair_rows[r];
        if (!row.valid) {
            continue;
        }
        const int a = row.body_a_local_idx;
        const int b = row.body_b_local_idx;
        if (a >= 0 && a < kMarkCars) {
            touched_world[a] = true;
            if (!row.has_bt) blocked[a] = true;
        }
        if (b >= 0 && b < kMarkCars) {
            touched_world[b] = true;
            if (!row.has_bt) blocked[b] = true;
        }
    }

    for (int r = 0; ball_rows && r < ball_row_count; ++r) {
        const BallSplitContactRow& row = ball_rows[r];
        if (!row.valid) {
            continue;
        }
        touched_ball = true;
        if (!row.has_bt) {
            ball_blocked = true;
        }
        if (row.other_local_idx >= 0 && row.other_local_idx < kMarkCars) {
            touched_world[row.other_local_idx] = true;
            if (!row.has_bt) {
                blocked[row.other_local_idx] = true;
            }
        }
    }

    for (int pass = 0; pass < kMarkCars; ++pass) {
        bool changed = false;
        for (int r = 0; pair_rows && r < pair_row_count; ++r) {
            const CarSharedSplitContactRow& row = pair_rows[r];
            if (!row.valid) {
                continue;
            }
            const int a = row.body_a_local_idx;
            const int b = row.body_b_local_idx;
            if (a < 0 || a >= kMarkCars || b < 0 || b >= kMarkCars) {
                continue;
            }
            if (blocked[a] != blocked[b]) {
                blocked[a] = true;
                blocked[b] = true;
                changed = true;
            }
        }
        for (int r = 0; ball_rows && r < ball_row_count; ++r) {
            const BallSplitContactRow& row = ball_rows[r];
            if (!row.valid || row.other_local_idx < 0 || row.other_local_idx >= kMarkCars) {
                continue;
            }
            if (blocked[row.other_local_idx] != ball_blocked) {
                blocked[row.other_local_idx] = true;
                ball_blocked = true;
                changed = true;
            }
        }
        if (!changed) {
            break;
        }
    }
    if (ball_split) {
        ball_split->is_bt = (touched_ball && !ball_blocked) ? 1 : 0;
    }
    for (int i = 0; i < count && i < kMarkCars; ++i) {
        car_splits[i].is_bt = (touched_world[i] && !blocked[i]) ? 1 : 0;
    }
}

#undef ROCKETGPU_SHARED_SOLVER_BACKEND_TAG

ROCKETGPU_HD void SolveSharedCarSplitRow(
    int count,
    ::rocketgpu::CarState* cars,
    CarSharedSplitContactRow& row,
    int iter,
    int row_idx,
    const char* bucket,
    SplitImpulseAccumulator* car_splits,
    int env_idx) {
    if (!row.valid || !cars || !car_splits) {
        return;
    }
    if (row.body_a_local_idx < 0 || row.body_a_local_idx >= count) {
        return;
    }

    SplitImpulseAccumulator& split_a = car_splits[row.body_a_local_idx];
    SplitImpulseAccumulator* split_b_bt = nullptr;
    if (row.body_b_local_idx >= 0 && row.body_b_local_idx < count) {
        split_b_bt = &car_splits[row.body_b_local_idx];
    }
    if (row.has_bt && split_a.is_bt && (row.body_b_local_idx < 0 || (split_b_bt && split_b_bt->is_bt))) {
        if (row.rhs_penetration_bt == 0.0f) {
            return;
        }

        float delta_impulse_bt = row.rhs_penetration_bt;
        const float delta_vel_linear_bt = BtSimdDot3LikeBullet(row.normal_a, split_a.push_velocity);
        const float delta_vel_angular_bt = BtSimdDot3LikeBullet(row.rel_a_cross_normal_bt, split_a.turn_velocity);
        delta_impulse_bt -= (delta_vel_linear_bt + delta_vel_angular_bt) * row.jac_diag_ab_inv_bt;
        if (split_b_bt) {
            const float delta_vel_linear_b_bt = BtSimdDot3LikeBullet(row.normal_b, split_b_bt->push_velocity);
            const float delta_vel_angular_b_bt = BtSimdDot3LikeBullet(row.rel_b_cross_normal_bt, split_b_bt->turn_velocity);
            delta_impulse_bt -= (delta_vel_linear_b_bt + delta_vel_angular_b_bt) * row.jac_diag_ab_inv_bt;
        }
        const float sum_bt = row.applied_push_impulse_bt + delta_impulse_bt;
        if (sum_bt < 0.0f) {
            delta_impulse_bt = 0.0f - row.applied_push_impulse_bt;
            row.applied_push_impulse_bt = 0.0f;
        } else {
            row.applied_push_impulse_bt = sum_bt;
        }
        if (delta_impulse_bt != 0.0f) {
            split_a.push_velocity += (row.normal_a * row.inv_mass_a) * delta_impulse_bt;
            split_a.turn_velocity += row.angular_component_a_bt * delta_impulse_bt;
            if (split_b_bt) {
                split_b_bt->push_velocity += (row.normal_b * row.inv_mass_b) * delta_impulse_bt;
                split_b_bt->turn_velocity += row.angular_component_b_bt * delta_impulse_bt;
            }
        }
        (void)iter; (void)row_idx; (void)bucket; (void)env_idx; (void)false; (void)-1;
        return;
    }
    const float delta_vel_a =
        BtSimdDot3LikeBullet(row.normal_a, split_a.push_velocity) +
        BtSimdDot3LikeBullet(row.rel_a_cross_normal, split_a.turn_velocity);

    float delta_vel_b = 0.0f;
    SplitImpulseAccumulator* split_b = nullptr;
    if (row.body_b_local_idx >= 0 && row.body_b_local_idx < count) {
        split_b = &car_splits[row.body_b_local_idx];
        delta_vel_b =
            BtSimdDot3LikeBullet(row.normal_b, split_b->push_velocity) +
            BtSimdDot3LikeBullet(row.rel_b_cross_normal, split_b->turn_velocity);
    }

    float delta_impulse = row.rhs_penetration;
    delta_impulse -= (delta_vel_a + delta_vel_b) * row.jac_diag_ab_inv;
    const float sum = row.applied_push_impulse + delta_impulse;
    if (sum < 0.0f) {
        delta_impulse = -row.applied_push_impulse;
        row.applied_push_impulse = 0.0f;
    } else {
        row.applied_push_impulse = sum;
    }
    if (delta_impulse != 0.0f) {
        split_a.push_velocity += row.normal_a * (row.inv_mass_a * delta_impulse);
        split_a.turn_velocity += row.angular_component_a * delta_impulse;
        if (split_b) {
            split_b->push_velocity += row.normal_b * (row.inv_mass_b * delta_impulse);
            split_b->turn_velocity += row.angular_component_b * delta_impulse;
        }
    }
}

ROCKETGPU_HD void SolveUnifiedBallSplitRow(
    int count,
    BallSplitContactRow& row,
    SplitImpulseAccumulator& ball_split,
    SplitImpulseAccumulator* car_splits) {
    if (!row.valid || !car_splits) {
        return;
    }

    SplitImpulseAccumulator* other_split_bt = nullptr;
    if (row.other_local_idx >= 0 && row.other_local_idx < count) {
        other_split_bt = &car_splits[row.other_local_idx];
    }
    if (row.has_bt && ball_split.is_bt && (!other_split_bt || other_split_bt->is_bt)) {
        float delta_impulse_bt = row.rhs_penetration_bt;
        const float ball_term_bt = (
            BtSimdDot3LikeBullet(row.normal, ball_split.push_velocity) +
            BtSimdDot3LikeBullet(row.rel_ball_cross_normal_bt, ball_split.turn_velocity)) * row.jac_diag_ab_inv_bt;
        if (other_split_bt) {
            delta_impulse_bt -= (
                BtSimdDot3LikeBullet(-row.normal, other_split_bt->push_velocity) +
                BtSimdDot3LikeBullet(row.rel_other_cross_normal_bt, other_split_bt->turn_velocity)) * row.jac_diag_ab_inv_bt;
        }
        delta_impulse_bt -= ball_term_bt;
        const float sum_bt = row.applied_push_impulse_bt + delta_impulse_bt;
        if (sum_bt < 0.0f) {
            delta_impulse_bt = -row.applied_push_impulse_bt;
            row.applied_push_impulse_bt = 0.0f;
        } else {
            row.applied_push_impulse_bt = sum_bt;
        }
        if (delta_impulse_bt == 0.0f) {
            return;
        }

        ball_split.push_velocity += (row.normal * row.ball_inv_mass) * delta_impulse_bt;
        ball_split.turn_velocity += row.ball_angular_component_bt * delta_impulse_bt;
        if (other_split_bt) {
            other_split_bt->push_velocity += ((-row.normal) * row.other_inv_mass) * delta_impulse_bt;
            other_split_bt->turn_velocity += row.other_angular_component_bt * delta_impulse_bt;
        }
        return;
    }

    float delta_impulse = row.rhs_penetration;
    const float ball_term = (
        BtSimdDot3LikeBullet(row.normal, ball_split.push_velocity) +
        BtSimdDot3LikeBullet(row.rel_ball_cross_normal, ball_split.turn_velocity)) * row.jac_diag_ab_inv;

    SplitImpulseAccumulator* other_split = nullptr;
    if (row.other_local_idx >= 0 && row.other_local_idx < count) {
        other_split = &car_splits[row.other_local_idx];
        delta_impulse -= (
            BtSimdDot3LikeBullet(-row.normal, other_split->push_velocity) +
            BtSimdDot3LikeBullet(row.rel_other_cross_normal, other_split->turn_velocity)) * row.jac_diag_ab_inv;
    }
    delta_impulse -= ball_term;

    const float sum = row.applied_push_impulse + delta_impulse;
    if (sum < 0.0f) {
        delta_impulse = -row.applied_push_impulse;
        row.applied_push_impulse = 0.0f;
    } else {
        row.applied_push_impulse = sum;
    }

    if (delta_impulse == 0.0f) {
        return;
    }

    ball_split.push_velocity += row.normal * (row.ball_inv_mass * delta_impulse);
    ball_split.turn_velocity += row.ball_angular_component * delta_impulse;
    if (other_split) {
        other_split->push_velocity += (-row.normal) * (row.other_inv_mass * delta_impulse);
        other_split->turn_velocity += row.other_angular_component * delta_impulse;
    }
}

ROCKETGPU_HD ArenaBroadphaseGrid MakeArenaBroadphaseGrid(const ::rocketgpu::SimConfig& cfg) {
    ArenaBroadphaseGrid grid;
    grid.min_bt = cfg.arena_broadphase_min_bt;
    grid.max_bt = cfg.arena_broadphase_max_bt;
    grid.cell_size_bt = cfg.arena_broadphase_cell_size_bt;
    grid.cells_x = cfg.arena_broadphase_cells_x;
    grid.cells_y = cfg.arena_broadphase_cells_y;
    grid.cells_z = cfg.arena_broadphase_cells_z;
    grid.cell_mesh_mask = cfg.arena_broadphase_cell_mask;
    grid.cell_count = cfg.arena_broadphase_cell_count;
    return grid;
}

ROCKETGPU_HD bool BulletStaticPlaneProxyAabb(int plane_rank, ::rocketgpu::Vec3& out_min, ::rocketgpu::Vec3& out_max) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kLarge = 1.0e18f;
    constexpr float kSlabBt = 0.2f;
    constexpr float kContactThresholdBt = 0.02f;
    const float arena_extent_x_bt = ArenaExtentX() * kUUToBT;
    const float arena_height_bt = ArenaHeight() * kUUToBT;
    out_min = ::rocketgpu::Vec3(-kLarge, -kLarge, -kLarge);
    out_max = ::rocketgpu::Vec3(kLarge, kLarge, kLarge);
    switch (plane_rank) {
        case 0:
            out_max.z = kSlabBt + kContactThresholdBt;
            return true;
        case 1:
            out_min.z = arena_height_bt - kSlabBt - kContactThresholdBt;
            return true;
        case 2:
            out_max.x = -arena_extent_x_bt + kSlabBt + kContactThresholdBt;
            return true;
        case 3:
            out_min.x = arena_extent_x_bt - kSlabBt - kContactThresholdBt;
            return true;
        default:
            return false;
    }
}

ROCKETGPU_HD bool BulletStaticPlaneShapeAabb(int plane_rank, ::rocketgpu::Vec3& out_min, ::rocketgpu::Vec3& out_max) {
    if (!BulletStaticPlaneProxyAabb(plane_rank, out_min, out_max)) {
        return false;
    }
    constexpr float kContactThresholdBt = 0.02f;
    switch (plane_rank) {
        case 0: out_max.z -= kContactThresholdBt; break;
        case 1: out_min.z += kContactThresholdBt; break;
        case 2: out_max.x -= kContactThresholdBt; break;
        case 3: out_min.x += kContactThresholdBt; break;
        default: break;
    }
    return true;
}

ROCKETGPU_HD bool BulletAabbOverlap(
    const ::rocketgpu::Vec3& a_min,
    const ::rocketgpu::Vec3& a_max,
    const ::rocketgpu::Vec3& b_min,
    const ::rocketgpu::Vec3& b_max) {
    return !(a_max.x < b_min.x || a_min.x > b_max.x ||
             a_max.y < b_min.y || a_min.y > b_max.y ||
             a_max.z < b_min.z || a_min.z > b_max.z);
}

ROCKETGPU_HD bool BulletPlaneRegisteredInCell(
    const ArenaBroadphaseGrid& grid,
    int plane_rank,
    int cell_i,
    int cell_j,
    int cell_k) {
    ::rocketgpu::Vec3 plane_min(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 plane_max(0.0f, 0.0f, 0.0f);
    if (!BulletStaticPlaneProxyAabb(plane_rank, plane_min, plane_max)) {
        return false;
    }
    const ::rocketgpu::Vec3 clamped_max(
        Min(plane_max.x, grid.max_bt.x),
        Min(plane_max.y, grid.max_bt.y),
        Min(plane_max.z, grid.max_bt.z));
    int i_min = 0, j_min = 0, k_min = 0, i_max = 0, j_max = 0, k_max = 0;
    grid.cell_indices(plane_min, i_min, j_min, k_min);
    grid.cell_indices(clamped_max, i_max, j_max, k_max);
    return cell_i >= i_min - 1 && cell_i <= i_max + 1 &&
           cell_j >= j_min - 1 && cell_j <= j_max + 1 &&
           cell_k >= k_min - 1 && cell_k <= k_max + 1;
}

ROCKETGPU_HD BulletBroadphaseProxy ComputeBulletCarProxyAabb(
    const ::rocketgpu::CarState& car,
    const ::rocketgpu::CarConfig& cfg,
    float dt,
    const ArenaBroadphaseGrid& grid,
    bool native_sse_math = false) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kContactThresholdBt = 0.02f;

    constexpr float kCompoundMarginBt = 0.0f;

    BulletBroadphaseProxy proxy;
    const ::rocketgpu::Vec3 half_extent_bt = BulletBoxHalfExtentWithMarginBt(cfg);
    const ::rocketgpu::Vec3 compound_half_bt(
        half_extent_bt.x + kCompoundMarginBt,
        half_extent_bt.y + kCompoundMarginBt,
        half_extent_bt.z + kCompoundMarginBt);
    const ::rocketgpu::Vec3 child_origin_bt = cfg.hitbox_offset * kUUToBT;

    ::rocketgpu::Quat predicted_rot = car.rot;
    ::rocketgpu::Vec3 pred_basis_fwd = car.basis_forward;
    ::rocketgpu::Vec3 pred_basis_right = car.basis_right;
    ::rocketgpu::Vec3 pred_basis_up = car.basis_up;
    if (car.basis_authoritative) {
        IntegrateOrientationBasisColumns(
            pred_basis_fwd, pred_basis_right, pred_basis_up, predicted_rot, car.ang_vel, dt, false, native_sse_math);
    } else {
        IntegrateOrientation(predicted_rot, car.ang_vel, dt);
    }
    const ::rocketgpu::Vec3 origin_bt = car.pos_bt;
    const ::rocketgpu::Vec3 predicted_origin_bt = origin_bt + car.vel * (kUUToBT * dt);

    ::rocketgpu::Vec3 axis_x(1.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 axis_y(0.0f, 1.0f, 0.0f);
    ::rocketgpu::Vec3 axis_z(0.0f, 0.0f, 1.0f);
    CarBasisColumns(car, axis_x, axis_y, axis_z);
    ::rocketgpu::Vec3 pred_axis_x(1.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 pred_axis_y(0.0f, 1.0f, 0.0f);
    ::rocketgpu::Vec3 pred_axis_z(0.0f, 0.0f, 1.0f);
    if (car.basis_authoritative) {
        pred_axis_x = pred_basis_fwd;
        pred_axis_y = pred_basis_right;
        pred_axis_z = pred_basis_up;
    } else {
        GetBasisVectors(predicted_rot, pred_axis_x, pred_axis_y, pred_axis_z);
    }

    const ::rocketgpu::Vec3 center_now(
        origin_bt.x + axis_x.x * child_origin_bt.x + axis_y.x * child_origin_bt.y + axis_z.x * child_origin_bt.z,
        origin_bt.y + axis_x.y * child_origin_bt.x + axis_y.y * child_origin_bt.y + axis_z.y * child_origin_bt.z,
        origin_bt.z + axis_x.z * child_origin_bt.x + axis_y.z * child_origin_bt.y + axis_z.z * child_origin_bt.z);
    const ::rocketgpu::Vec3 center_pred(
        predicted_origin_bt.x + pred_axis_x.x * child_origin_bt.x + pred_axis_y.x * child_origin_bt.y + pred_axis_z.x * child_origin_bt.z,
        predicted_origin_bt.y + pred_axis_x.y * child_origin_bt.x + pred_axis_y.y * child_origin_bt.y + pred_axis_z.y * child_origin_bt.z,
        predicted_origin_bt.z + pred_axis_x.z * child_origin_bt.x + pred_axis_y.z * child_origin_bt.y + pred_axis_z.z * child_origin_bt.z);
    const ::rocketgpu::Vec3 extent_now(
        Abs(axis_x.x) * compound_half_bt.x + Abs(axis_y.x) * compound_half_bt.y + Abs(axis_z.x) * compound_half_bt.z,
        Abs(axis_x.y) * compound_half_bt.x + Abs(axis_y.y) * compound_half_bt.y + Abs(axis_z.y) * compound_half_bt.z,
        Abs(axis_x.z) * compound_half_bt.x + Abs(axis_y.z) * compound_half_bt.y + Abs(axis_z.z) * compound_half_bt.z);
    const ::rocketgpu::Vec3 extent_pred(
        Abs(pred_axis_x.x) * compound_half_bt.x + Abs(pred_axis_y.x) * compound_half_bt.y + Abs(pred_axis_z.x) * compound_half_bt.z,
        Abs(pred_axis_x.y) * compound_half_bt.x + Abs(pred_axis_y.y) * compound_half_bt.y + Abs(pred_axis_z.y) * compound_half_bt.z,
        Abs(pred_axis_x.z) * compound_half_bt.x + Abs(pred_axis_y.z) * compound_half_bt.y + Abs(pred_axis_z.z) * compound_half_bt.z);
    proxy.aabb_min = ::rocketgpu::Vec3(
        Min(center_now.x - extent_now.x, center_pred.x - extent_pred.x) - kContactThresholdBt,
        Min(center_now.y - extent_now.y, center_pred.y - extent_pred.y) - kContactThresholdBt,
        Min(center_now.z - extent_now.z, center_pred.z - extent_pred.z) - kContactThresholdBt);
    proxy.aabb_max = ::rocketgpu::Vec3(
        Max(center_now.x + extent_now.x, center_pred.x + extent_pred.x) + kContactThresholdBt,
        Max(center_now.y + extent_now.y, center_pred.y + extent_pred.y) + kContactThresholdBt,
        Max(center_now.z + extent_now.z, center_pred.z + extent_pred.z) + kContactThresholdBt);
    const ::rocketgpu::Vec3 child_extent_now(
        Abs(axis_x.x) * half_extent_bt.x + Abs(axis_y.x) * half_extent_bt.y + Abs(axis_z.x) * half_extent_bt.z,
        Abs(axis_x.y) * half_extent_bt.x + Abs(axis_y.y) * half_extent_bt.y + Abs(axis_z.y) * half_extent_bt.z,
        Abs(axis_x.z) * half_extent_bt.x + Abs(axis_y.z) * half_extent_bt.y + Abs(axis_z.z) * half_extent_bt.z);
    proxy.child_aabb_min = ::rocketgpu::Vec3(
        center_now.x - child_extent_now.x,
        center_now.y - child_extent_now.y,
        center_now.z - child_extent_now.z);
    proxy.child_aabb_max = ::rocketgpu::Vec3(
        center_now.x + child_extent_now.x,
        center_now.y + child_extent_now.y,
        center_now.z + child_extent_now.z);
    if (grid.valid()) {
        grid.cell_indices(proxy.aabb_min, proxy.cell_i, proxy.cell_j, proxy.cell_k);
    }
    proxy.valid = 1;
    proxy.is_ball = 0;
    return proxy;
}

ROCKETGPU_HD bool BulletCarMeshManifoldExists(
    const BulletBroadphaseProxy& proxy,
    const ArenaBroadphaseGrid& grid,
    const ArenaMeshBvhMeshInfo& info) {
    constexpr float kContactThresholdBt = 0.02f;
    if (!grid.valid() || !proxy.valid || info.mesh_index < 0 || info.triangle_count <= 0) {
        return false;
    }
    const int cell_index = proxy.cell_i * grid.cells_y * grid.cells_z +
                           proxy.cell_j * grid.cells_z + proxy.cell_k;
    if (!grid.mesh_registered_in_cell(cell_index, info.mesh_index)) {
        return false;
    }
    const ::rocketgpu::Vec3 mesh_proxy_min(
        info.aabb_min.x - kContactThresholdBt,
        info.aabb_min.y - kContactThresholdBt,
        info.aabb_min.z - kContactThresholdBt);
    const ::rocketgpu::Vec3 mesh_proxy_max(
        info.aabb_max.x + kContactThresholdBt,
        info.aabb_max.y + kContactThresholdBt,
        info.aabb_max.z + kContactThresholdBt);
    if (!BulletAabbOverlap(proxy.aabb_min, proxy.aabb_max, mesh_proxy_min, mesh_proxy_max)) {
        return false;
    }
    return BulletAabbOverlap(proxy.child_aabb_min, proxy.child_aabb_max, info.aabb_min, info.aabb_max);
}

ROCKETGPU_HD int BuildBulletManifoldList(
    const ::rocketgpu::BallState& ball,
    int car_count,
    const ::rocketgpu::CarState* cars,
    const PreContactCar* pre_contact_cars,
    const ::rocketgpu::CarConfig* car_configs,
    float ball_radius,
    float dt,
    const ArenaBroadphaseGrid& grid,
    const ArenaMeshBvhMeshInfo* mesh_infos,
    int mesh_info_count,
    EnvStridedArray<BulletManifoldEntry> out_entries,
    int max_entries,
    bool arena_static = true,
    WorldSolverScratch* bp_history = nullptr,
    bool native_sse_math = false) {
    if (!out_entries || max_entries <= 0 || !grid.valid() || !mesh_infos || mesh_info_count <= 0) {
        return 0;
    }

    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kContactThresholdBt = 0.02f;
    constexpr float kSphereAabbExtraBt = 0.08f;
    constexpr float kCompoundMarginBt = 0.0f;

    // true (cal_ball_wake_on_set): the step simulates an exactly-still ball, so its world manifolds
    // must stay in the list that drives solver ordering (same predicate as ShouldSleepBallAtTickStart).
    constexpr int kMaxBodies = 9;

    BulletBroadphaseProxy proxies[kMaxBodies];

    {
        BulletBroadphaseProxy& proxy = proxies[0];

        const ::rocketgpu::Vec3 pos_bt = ball.pos_bt;
        const ::rocketgpu::Vec3 predicted_bt = pos_bt + ball.vel_bt * dt;
        const float extent = ball_radius * kUUToBT + kSphereAabbExtraBt;
        proxy.aabb_min = ::rocketgpu::Vec3(
            Min(pos_bt.x, predicted_bt.x) - extent - kContactThresholdBt,
            Min(pos_bt.y, predicted_bt.y) - extent - kContactThresholdBt,
            Min(pos_bt.z, predicted_bt.z) - extent - kContactThresholdBt);

        proxy.child_aabb_min = ::rocketgpu::Vec3(pos_bt.x - extent, pos_bt.y - extent, pos_bt.z - extent);
        proxy.child_aabb_max = ::rocketgpu::Vec3(pos_bt.x + extent, pos_bt.y + extent, pos_bt.z + extent);
        proxy.aabb_max = ::rocketgpu::Vec3(
            Max(pos_bt.x, predicted_bt.x) + extent + kContactThresholdBt,
            Max(pos_bt.y, predicted_bt.y) + extent + kContactThresholdBt,
            Max(pos_bt.z, predicted_bt.z) + extent + kContactThresholdBt);
        proxy.child_aabb_min = ::rocketgpu::Vec3(pos_bt.x - extent, pos_bt.y - extent, pos_bt.z - extent);
        proxy.child_aabb_max = ::rocketgpu::Vec3(pos_bt.x + extent, pos_bt.y + extent, pos_bt.z + extent);
        grid.cell_indices(proxy.aabb_min, proxy.cell_i, proxy.cell_j, proxy.cell_k);
        proxy.valid = 1;
        proxy.is_ball = 1;
    }

    for (int car_idx = 0; car_idx < car_count && car_idx < kMaxBodies - 1; ++car_idx) {
        BulletBroadphaseProxy& proxy = proxies[car_idx + 1];

        const ::rocketgpu::CarState& car = cars[car_idx];
        const bool demoed_before_contacts =
            pre_contact_cars != nullptr ? pre_contact_cars[car_idx].is_demoed != 0 : car.is_demoed != 0;
        if (demoed_before_contacts) {
            proxy.valid = 0;
            continue;
        }
        const ::rocketgpu::CarConfig& cfg = car_configs[car_idx];
        const ::rocketgpu::Vec3 half_extent_bt = BulletBoxHalfExtentWithMarginBt(cfg);
        const ::rocketgpu::Vec3 compound_half_bt(
            half_extent_bt.x + kCompoundMarginBt,
            half_extent_bt.y + kCompoundMarginBt,
            half_extent_bt.z + kCompoundMarginBt);
        const ::rocketgpu::Vec3 child_origin_bt = cfg.hitbox_offset * kUUToBT;

        ::rocketgpu::Quat predicted_rot = car.rot;
        ::rocketgpu::Vec3 pred_basis_fwd = car.basis_forward;
        ::rocketgpu::Vec3 pred_basis_right = car.basis_right;
        ::rocketgpu::Vec3 pred_basis_up = car.basis_up;
        if (car.basis_authoritative) {
            IntegrateOrientationBasisColumns(
                pred_basis_fwd, pred_basis_right, pred_basis_up, predicted_rot, car.ang_vel, dt, false, native_sse_math);
        } else {
            IntegrateOrientation(predicted_rot, car.ang_vel, dt);
        }
        const ::rocketgpu::Vec3 origin_bt = car.pos_bt;
        const ::rocketgpu::Vec3 predicted_origin_bt = origin_bt + car.vel * (kUUToBT * dt);

        ::rocketgpu::Vec3 axis_x(1.0f, 0.0f, 0.0f);
        ::rocketgpu::Vec3 axis_y(0.0f, 1.0f, 0.0f);
        ::rocketgpu::Vec3 axis_z(0.0f, 0.0f, 1.0f);
        CarBasisColumns(car, axis_x, axis_y, axis_z);
        ::rocketgpu::Vec3 pred_axis_x(1.0f, 0.0f, 0.0f);
        ::rocketgpu::Vec3 pred_axis_y(0.0f, 1.0f, 0.0f);
        ::rocketgpu::Vec3 pred_axis_z(0.0f, 0.0f, 1.0f);
        if (car.basis_authoritative) {
            pred_axis_x = pred_basis_fwd;
            pred_axis_y = pred_basis_right;
            pred_axis_z = pred_basis_up;
        } else {
            GetBasisVectors(predicted_rot, pred_axis_x, pred_axis_y, pred_axis_z);
        }

        const auto box_center = [](const ::rocketgpu::Vec3& origin,
                                   const ::rocketgpu::Vec3& ax,
                                   const ::rocketgpu::Vec3& ay,
                                   const ::rocketgpu::Vec3& az,
                                   const ::rocketgpu::Vec3& local) {
            return ::rocketgpu::Vec3(
                origin.x + ax.x * local.x + ay.x * local.y + az.x * local.z,
                origin.y + ax.y * local.x + ay.y * local.y + az.y * local.z,
                origin.z + ax.z * local.x + ay.z * local.y + az.z * local.z);
        };
        const auto box_extent = [](const ::rocketgpu::Vec3& ax,
                                   const ::rocketgpu::Vec3& ay,
                                   const ::rocketgpu::Vec3& az,
                                   const ::rocketgpu::Vec3& half) {
            return ::rocketgpu::Vec3(
                Abs(ax.x) * half.x + Abs(ay.x) * half.y + Abs(az.x) * half.z,
                Abs(ax.y) * half.x + Abs(ay.y) * half.y + Abs(az.y) * half.z,
                Abs(ax.z) * half.x + Abs(ay.z) * half.y + Abs(az.z) * half.z);
        };

        const ::rocketgpu::Vec3 center_now = box_center(origin_bt, axis_x, axis_y, axis_z, child_origin_bt);
        const ::rocketgpu::Vec3 center_pred =
            box_center(predicted_origin_bt, pred_axis_x, pred_axis_y, pred_axis_z, child_origin_bt);
        const ::rocketgpu::Vec3 extent_now = box_extent(axis_x, axis_y, axis_z, compound_half_bt);
        const ::rocketgpu::Vec3 extent_pred = box_extent(pred_axis_x, pred_axis_y, pred_axis_z, compound_half_bt);
        proxy.aabb_min = ::rocketgpu::Vec3(
            Min(center_now.x - extent_now.x, center_pred.x - extent_pred.x) - kContactThresholdBt,
            Min(center_now.y - extent_now.y, center_pred.y - extent_pred.y) - kContactThresholdBt,
            Min(center_now.z - extent_now.z, center_pred.z - extent_pred.z) - kContactThresholdBt);
        proxy.aabb_max = ::rocketgpu::Vec3(
            Max(center_now.x + extent_now.x, center_pred.x + extent_pred.x) + kContactThresholdBt,
            Max(center_now.y + extent_now.y, center_pred.y + extent_pred.y) + kContactThresholdBt,
            Max(center_now.z + extent_now.z, center_pred.z + extent_pred.z) + kContactThresholdBt);

        const ::rocketgpu::Vec3 child_extent_now = box_extent(axis_x, axis_y, axis_z, half_extent_bt);
        proxy.child_aabb_min = ::rocketgpu::Vec3(
            center_now.x - child_extent_now.x,
            center_now.y - child_extent_now.y,
            center_now.z - child_extent_now.z);
        proxy.child_aabb_max = ::rocketgpu::Vec3(
            center_now.x + child_extent_now.x,
            center_now.y + child_extent_now.y,
            center_now.z + child_extent_now.z);
        grid.cell_indices(proxy.aabb_min, proxy.cell_i, proxy.cell_j, proxy.cell_k);
        proxy.valid = 1;
        proxy.is_ball = 0;
    }

    const int body_count = Min(car_count + 1, kMaxBodies);
    uint8_t pair_seen[kMaxBodies][kMaxBodies] = {};
    uint8_t broadphase_pair_union[kMaxBodies][kMaxBodies] = {};
    int entry_count = 0;

    const auto push_entry = [&](int8_t kind, int8_t body_a, int8_t body_b, int16_t static_id) {
        if (entry_count >= max_entries || entry_count >= kMaxBulletManifoldEntries) {
            return;
        }
        BulletManifoldEntry& entry = out_entries[entry_count];
        entry.kind = kind;
        entry.body_a = body_a;
        entry.body_b = body_b;
        entry.static_id = static_id;
        entry.island_id = 0;
        ++entry_count;
    };

    uint32_t insert_seq[kMaxBodies] = {};
    if (bp_history != nullptr) {
        if (!bp_history->bp_initialized) {
            bp_history->bp_seq_counter = 0;
            for (int body = 0; body < body_count; ++body) {
                bp_history->bp_cell_i[body] = proxies[body].cell_i;
                bp_history->bp_cell_j[body] = proxies[body].cell_j;
                bp_history->bp_cell_k[body] = proxies[body].cell_k;
                bp_history->bp_insert_seq[body] = ++bp_history->bp_seq_counter;
            }
            bp_history->bp_initialized = 1;
        } else {
            for (int body = 0; body < body_count; ++body) {
                const BulletBroadphaseProxy& proxy = proxies[body];
                if (!proxy.valid) {
                    continue;
                }
                if (proxy.cell_i != bp_history->bp_cell_i[body] ||
                    proxy.cell_j != bp_history->bp_cell_j[body] ||
                    proxy.cell_k != bp_history->bp_cell_k[body]) {
                    bp_history->bp_cell_i[body] = proxy.cell_i;
                    bp_history->bp_cell_j[body] = proxy.cell_j;
                    bp_history->bp_cell_k[body] = proxy.cell_k;
                    bp_history->bp_insert_seq[body] = ++bp_history->bp_seq_counter;
                }
            }
        }
        for (int body = 0; body < body_count; ++body) {
            insert_seq[body] = bp_history->bp_insert_seq[body];
        }
    } else {
        for (int body = 0; body < body_count; ++body) {
            insert_seq[body] = static_cast<uint32_t>(body + 1);
        }
    }

    constexpr int kMaxUnionPairs = 64;
    int8_t union_pair_a[kMaxUnionPairs];
    int8_t union_pair_b[kMaxUnionPairs];
    int union_pair_count = 0;
    int partner_order[kMaxBodies];
    for (int i = 0; i < body_count; ++i) {
        partner_order[i] = i;
    }
    for (int i = 1; i < body_count; ++i) {
        const int v = partner_order[i];
        int j = i - 1;
        while (j >= 0 &&
               (insert_seq[partner_order[j]] > insert_seq[v] ||
                (insert_seq[partner_order[j]] == insert_seq[v] && partner_order[j] > v))) {
            partner_order[j + 1] = partner_order[j];
            --j;
        }
        partner_order[j + 1] = v;
    }

    for (int body = 0; body < body_count; ++body) {
        const BulletBroadphaseProxy& proxy = proxies[body];
        if (!proxy.valid) {
            continue;
        }
        const int cell_index = proxy.cell_i * grid.cells_y * grid.cells_z +
                               proxy.cell_j * grid.cells_z + proxy.cell_k;

        for (int mesh_slot = 0; arena_static && mesh_slot < mesh_info_count; ++mesh_slot) {
            const ArenaMeshBvhMeshInfo& info = mesh_infos[mesh_slot];
            if (info.mesh_index < 0 || info.triangle_count <= 0) {
                continue;
            }
            if (!grid.mesh_registered_in_cell(cell_index, info.mesh_index)) {
                continue;
            }
            const ::rocketgpu::Vec3 mesh_proxy_min(
                info.aabb_min.x - kContactThresholdBt,
                info.aabb_min.y - kContactThresholdBt,
                info.aabb_min.z - kContactThresholdBt);
            const ::rocketgpu::Vec3 mesh_proxy_max(
                info.aabb_max.x + kContactThresholdBt,
                info.aabb_max.y + kContactThresholdBt,
                info.aabb_max.z + kContactThresholdBt);
            if (!BulletAabbOverlap(proxy.aabb_min, proxy.aabb_max, mesh_proxy_min, mesh_proxy_max)) {
                continue;
            }
            if (!proxy.is_ball &&
                !BulletAabbOverlap(proxy.child_aabb_min, proxy.child_aabb_max, info.aabb_min, info.aabb_max)) {
                continue;
            }
            push_entry(
                proxy.is_ball ? 0 : 2,
                proxy.is_ball ? static_cast<int8_t>(-1) : static_cast<int8_t>(body - 1),
                -1,
                static_cast<int16_t>(info.mesh_index));
        }
        for (int plane_rank = 0; arena_static && plane_rank < 4; ++plane_rank) {
            if (!BulletPlaneRegisteredInCell(grid, plane_rank, proxy.cell_i, proxy.cell_j, proxy.cell_k)) {
                continue;
            }
            ::rocketgpu::Vec3 plane_proxy_min(0.0f, 0.0f, 0.0f);
            ::rocketgpu::Vec3 plane_proxy_max(0.0f, 0.0f, 0.0f);
            BulletStaticPlaneProxyAabb(plane_rank, plane_proxy_min, plane_proxy_max);
            if (!BulletAabbOverlap(proxy.aabb_min, proxy.aabb_max, plane_proxy_min, plane_proxy_max)) {
                continue;
            }
            if (!proxy.is_ball) {
                ::rocketgpu::Vec3 plane_shape_min(0.0f, 0.0f, 0.0f);
                ::rocketgpu::Vec3 plane_shape_max(0.0f, 0.0f, 0.0f);
                BulletStaticPlaneShapeAabb(plane_rank, plane_shape_min, plane_shape_max);
                if (!BulletAabbOverlap(proxy.child_aabb_min, proxy.child_aabb_max, plane_shape_min, plane_shape_max)) {
                    continue;
                }
            }
            push_entry(
                proxy.is_ball ? 0 : 2,
                proxy.is_ball ? static_cast<int8_t>(-1) : static_cast<int8_t>(body - 1),
                -1,
                static_cast<int16_t>(100 + plane_rank));
        }

        for (int order_i = 0; order_i < body_count; ++order_i) {
            const int other = partner_order[order_i];
            if (other == body) {
                continue;
            }
            const BulletBroadphaseProxy& other_proxy = proxies[other];
            if (!other_proxy.valid || pair_seen[body][other]) {
                continue;
            }
            if (Abs(other_proxy.cell_i - proxy.cell_i) > 1 ||
                Abs(other_proxy.cell_j - proxy.cell_j) > 1 ||
                Abs(other_proxy.cell_k - proxy.cell_k) > 1) {
                continue;
            }
            if (!BulletAabbOverlap(proxy.aabb_min, proxy.aabb_max, other_proxy.aabb_min, other_proxy.aabb_max)) {
                continue;
            }
            if (!BulletAabbOverlap(
                    proxy.child_aabb_min, proxy.child_aabb_max,
                    other_proxy.child_aabb_min, other_proxy.child_aabb_max)) {
                broadphase_pair_union[body][other] = 1;
                broadphase_pair_union[other][body] = 1;
                pair_seen[body][other] = 1;
                pair_seen[other][body] = 1;
                if (union_pair_count < kMaxUnionPairs) {
                    union_pair_a[union_pair_count] = static_cast<int8_t>(body);
                    union_pair_b[union_pair_count] = static_cast<int8_t>(other);
                    ++union_pair_count;
                }
                continue;
            }
            if (union_pair_count < kMaxUnionPairs) {
                union_pair_a[union_pair_count] = static_cast<int8_t>(body);
                union_pair_b[union_pair_count] = static_cast<int8_t>(other);
                ++union_pair_count;
            }
            if (proxy.is_ball || other_proxy.is_ball) {
                const int car_body = proxy.is_ball ? other : body;
                push_entry(1, static_cast<int8_t>(car_body - 1), -1, -1);
            } else {
                push_entry(
                    3,
                    static_cast<int8_t>(Min(body, other) - 1),
                    static_cast<int8_t>(Max(body, other) - 1),
                    -1);
            }
            pair_seen[body][other] = 1;
            pair_seen[other][body] = 1;
        }
    }

    int parent[kMaxBodies];
    for (int i = 0; i < kMaxBodies; ++i) {
        parent[i] = i;
    }
    const auto find_root = [&](int idx) {
        while (idx >= 0 && idx < kMaxBodies && parent[idx] != idx) {
            idx = parent[idx];
        }
        return idx;
    };
    for (int p = 0; p < union_pair_count; ++p) {
        const int a = union_pair_a[p];
        const int b = union_pair_b[p];
        if (a < 0 || a >= kMaxBodies || b < 0 || b >= kMaxBodies) {
            continue;
        }
        const int ra = find_root(a);
        const int rb = find_root(b);
        if (ra == rb) {
            continue;
        }
        parent[ra] = rb;
    }
    const auto island_for_body = [&](int body) {
        return find_root(body);
    };
    for (int i = 0; i < entry_count; ++i) {
        BulletManifoldEntry& entry = out_entries[i];
        const int body = entry.kind == 0 ? 0 : (entry.kind == 1 ? 0 : entry.body_a + 1);
        entry.island_id = island_for_body(body);
    }
    return entry_count;
}

ROCKETGPU_HD void SolveUnifiedSplitContactRows(
    ::rocketgpu::BallState& ball,
    int count,
    ::rocketgpu::CarState* cars,
    EnvStridedArray<BallSplitContactRow> ball_rows,
    int ball_row_count,
    EnvStridedArray<CarSharedSplitContactRow> world_rows,
    int world_row_count,
    EnvStridedArray<CarSharedSplitContactRow> pair_rows,
    int pair_row_count,
    float dt,
    int env_idx = -1,
    uint64_t current_tick = 0,
    int substep = 0,
    EnvStridedArray<const BulletManifoldEntry> sorted_manifolds = nullptr,
    int sorted_manifold_count = 0,
    bool rotate_soccar_ball = false,
    bool native_sse_math = false) {
    if (count < 0 || count > kMaxBallSplitCars || (count > 0 && !cars)) {
        return;
    }
    if ((!ball_rows || ball_row_count <= 0) &&
        (!world_rows || world_row_count <= 0) &&
        (!pair_rows || pair_row_count <= 0)) {
        return;
    }
    const int active_ball_rows = ball_rows ? ball_row_count : 0;
    const int active_world_rows = world_rows ? world_row_count : 0;
    const int active_pair_rows = pair_rows ? pair_row_count : 0;

    SplitImpulseAccumulator ball_split;
    SplitImpulseAccumulator car_splits[kMaxBallSplitCars];
    for (int i = 0; i < count; ++i) {
        car_splits[i] = SplitImpulseAccumulator{};
    }

    MarkBtSplitAccumulators(count, car_splits, world_rows, world_row_count, pair_rows, pair_row_count, ball_rows, ball_row_count, &ball_split);

    constexpr int kMaxUnifiedSplitRows =
        kMaxCarWorldSplitContactRows + kMaxCarPairSplitContactRows + kMaxBallSplitContactRows;
    uint8_t unified_split_bucket[kMaxUnifiedSplitRows];
    int16_t unified_split_row[kMaxUnifiedSplitRows];
    int unified_split_count = 0;
    bool unified_split_valid = false;
    if (sorted_manifolds != nullptr && sorted_manifold_count > 0 &&
        (active_world_rows + active_pair_rows + active_ball_rows) <= kMaxUnifiedSplitRows) {
        EnvStridedArray<const BulletManifoldEntry> entries = sorted_manifolds;
        const int entry_count = sorted_manifold_count;
        {
            uint8_t world_used[kMaxCarWorldSplitContactRows] = {};
            uint8_t pair_used[kMaxCarPairSplitContactRows] = {};
            uint8_t ball_used[kMaxBallSplitContactRows] = {};
            const auto push_split = [&](int bucket, int row_idx) {
                if (unified_split_count >= kMaxUnifiedSplitRows) {
                    return;
                }
                unified_split_bucket[unified_split_count] = static_cast<uint8_t>(bucket);
                unified_split_row[unified_split_count] = static_cast<int16_t>(row_idx);
                ++unified_split_count;
            };
            for (int order_idx = 0; order_idx < entry_count; ++order_idx) {
                const BulletManifoldEntry& entry = entries[order_idx];
                if (entry.kind == 2) {
                    int expected_tag = -1;
                    if (entry.static_id >= 100) {
                        const int plane_rank = entry.static_id - 100;
                        expected_tag = plane_rank == 0 ? 6 : (plane_rank == 1 ? 5 : (plane_rank == 2 ? 2 : 1));
                    } else if (entry.static_id >= 0) {
                        expected_tag = 100 + entry.static_id;
                    }
                    for (int row_idx = 0; row_idx < active_world_rows; ++row_idx) {
                        const CarSharedSplitContactRow& row = world_rows[row_idx];
                        if (world_used[row_idx] || !row.valid) {
                            continue;
                        }
                        if (row.body_a_local_idx != entry.body_a || row.source_tag != expected_tag) {
                            continue;
                        }
                        world_used[row_idx] = 1;
                        push_split(0, row_idx);
                    }
                } else if (entry.kind == 3) {
                    for (int row_idx = 0; row_idx < active_pair_rows; ++row_idx) {
                        const CarSharedSplitContactRow& row = pair_rows[row_idx];
                        if (pair_used[row_idx] || !row.valid) {
                            continue;
                        }
                        const int lo = Min(row.body_a_local_idx, row.body_b_local_idx);
                        const int hi = Max(row.body_a_local_idx, row.body_b_local_idx);
                        if (lo != entry.body_a || hi != entry.body_b) {
                            continue;
                        }
                        pair_used[row_idx] = 1;
                        push_split(1, row_idx);
                    }
                } else if (entry.kind == 1) {
                    for (int row_idx = 0; row_idx < active_ball_rows; ++row_idx) {
                        const BallSplitContactRow& row = ball_rows[row_idx];
                        if (ball_used[row_idx] || !row.valid || row.other_local_idx != entry.body_a) {
                            continue;
                        }
                        ball_used[row_idx] = 1;
                        push_split(2, row_idx);
                    }
                } else if (entry.kind == 0) {
                    for (int row_idx = 0; row_idx < active_ball_rows; ++row_idx) {
                        const BallSplitContactRow& row = ball_rows[row_idx];
                        if (ball_used[row_idx] || !row.valid || row.other_local_idx >= 0 ||
                            row.static_id < 0 || row.static_id != entry.static_id) {
                            continue;
                        }
                        ball_used[row_idx] = 1;
                        push_split(2, row_idx);
                    }
                }
            }
            for (int row_idx = 0; row_idx < active_world_rows; ++row_idx) {
                if (!world_used[row_idx]) {
                    push_split(0, row_idx);
                }
            }
            for (int row_idx = 0; row_idx < active_pair_rows; ++row_idx) {
                if (!pair_used[row_idx]) {
                    push_split(1, row_idx);
                }
            }
            for (int row_idx = 0; row_idx < active_ball_rows; ++row_idx) {
                if (!ball_used[row_idx]) {
                    push_split(2, row_idx);
                }
            }
            unified_split_valid =
                unified_split_count == active_world_rows + active_pair_rows + active_ball_rows;
        }
    }

    for (int iter = 0; iter < BallSplitImpulseIterations(); ++iter) {
        if (unified_split_valid) {
            for (int order_idx = 0; order_idx < unified_split_count; ++order_idx) {
                const int row_idx = unified_split_row[order_idx];
                switch (unified_split_bucket[order_idx]) {
                    case 0:
                        SolveSharedCarSplitRow(
                            count, cars, world_rows[row_idx], iter, row_idx, "world",
                            car_splits, env_idx);
                        break;
                    case 1:
                        SolveSharedCarSplitRow(
                            count, cars, pair_rows[row_idx], iter, row_idx, "pair",
                            car_splits, env_idx);
                        break;
                    default:
                        SolveUnifiedBallSplitRow(count, ball_rows[row_idx], ball_split, car_splits);
                        break;
                }
            }
            continue;
        }
        for (int row_idx = 0; row_idx < active_world_rows; ++row_idx) {
            SolveSharedCarSplitRow(
                count,
                cars,
                world_rows[row_idx],
                iter,
                row_idx,
                "world",
                car_splits,
                env_idx);
        }
        for (int row_idx = 0; row_idx < active_pair_rows; ++row_idx) {
            SolveSharedCarSplitRow(
                count,
                cars,
                pair_rows[row_idx],
                iter,
                row_idx,
                "pair",
                car_splits,
                env_idx);
        }
        for (int row_idx = 0; row_idx < active_ball_rows; ++row_idx) {
            SolveUnifiedBallSplitRow(count, ball_rows[row_idx], ball_split, car_splits);
        }
    }

    IntegrateBallOriginBt(
        ball,
        ball_split.is_bt ? ball_split.push_velocity : UuVecToBtDiv(ball_split.push_velocity),
        dt);
    if (rotate_soccar_ball &&
        (ball_split.push_velocity.x != 0.0f || ball_split.push_velocity.y != 0.0f || ball_split.push_velocity.z != 0.0f ||
         ball_split.turn_velocity.x != 0.0f || ball_split.turn_velocity.y != 0.0f || ball_split.turn_velocity.z != 0.0f)) {
        if (!ball.basis_authoritative) {
            GetOfficialUploadBasisVectors(ball.rot, ball.basis_forward, ball.basis_right, ball.basis_up);
        }
        IntegrateOrientationBasisColumns(ball.basis_forward, ball.basis_right, ball.basis_up,
            ball.rot, ball_split.turn_velocity * BallSplitTurnErp(), dt, false, native_sse_math);
        ball.basis_authoritative = 1;
    }
    for (int i = 0; i < count; ++i) {
        if (car_splits[i].is_bt) {
            IntegrateSplitImpulseTransformBt(cars[i], car_splits[i], dt, native_sse_math);
        } else {
            IntegrateSplitImpulseTransform(cars[i], car_splits[i], dt, native_sse_math);
        }
    }
}

ROCKETGPU_HD int FindSharedContactOrderParent(int* parent, int idx) {
    for (int step = 0; step < 8 && idx >= 0 && idx < 8 && parent[idx] != idx; ++step) {
        idx = parent[idx];
    }
    return idx;
}

ROCKETGPU_HD void ResolveSharedContactOrderCarIslandIds(int* parent, const int* component_max, int* car_island_id) {
    for (int i = 0; i < 8; ++i) {
        const int root = FindSharedContactOrderParent(parent, i);
        car_island_id[i] = (root >= 0 && root < 8 ? component_max[root] : i) + 1;
    }
}

ROCKETGPU_HD int SharedContactOrderIslandIdForBody(const int* car_island_id, int body_a_local_idx) {
    if (body_a_local_idx >= 0 && body_a_local_idx < 8) {
        return car_island_id[body_a_local_idx];
    }
    return body_a_local_idx + 1;
}

ROCKETGPU_HD void UnionSharedContactOrderParents(int* parent, int a, int b) {
    if (a < 0 || a >= 8 || b < 0 || b >= 8) {
        return;
    }
    const int root_a = FindSharedContactOrderParent(parent, a);
    const int root_b = FindSharedContactOrderParent(parent, b);
    if (root_a == root_b) {
        return;
    }
    if (root_a < root_b) {
        parent[root_a] = root_b;
    } else {
        parent[root_b] = root_a;
    }
}

ROCKETGPU_HD void QuickSortSharedContactOrderIndices(
    int* order,
    const int* island_ids,
    int lo,
    int hi) {
    int range_lo[kMaxSharedContactOrderGroups];
    int range_hi[kMaxSharedContactOrderGroups];
    int depth = 0;
    range_lo[depth] = lo;
    range_hi[depth] = hi;
    ++depth;
    while (depth > 0) {
        --depth;
        int i = range_lo[depth];
        int j = range_hi[depth];
        const int cur_lo = i;
        const int cur_hi = j;
        const int pivot = island_ids[order[(cur_lo + cur_hi) / 2]];
        do {
            while (island_ids[order[i]] < pivot) {
                ++i;
            }
            while (pivot < island_ids[order[j]]) {
                --j;
            }
            if (i <= j) {
                const int tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
                ++i;
                --j;
            }
        } while (i <= j);

        if (i < cur_hi && depth < kMaxSharedContactOrderGroups) {
            range_lo[depth] = i;
            range_hi[depth] = cur_hi;
            ++depth;
        }
        if (cur_lo < j && depth < kMaxSharedContactOrderGroups) {
            range_lo[depth] = cur_lo;
            range_hi[depth] = j;
            ++depth;
        }
    }
}

ROCKETGPU_HD int FirstSharedContactOrderBucket(
    EnvStridedArray<const SharedContactOrderGroup> groups,
    int group_count,
    int car_count) {
    if (!groups || group_count <= 0) {
        return -1;
    }

    int parent[8];
    int component_max[8];
    for (int i = 0; i < 8; ++i) {
        parent[i] = i;
        component_max[i] = i;
    }
    for (int group_idx = 0; group_idx < group_count && group_idx < kMaxSharedContactOrderGroups; ++group_idx) {
        const SharedContactOrderGroup& group = groups[group_idx];
        if (!group.valid || group.bucket != SharedContactOrderBucket::Pair || group.normal_row_count <= 0) {
            continue;
        }
        UnionSharedContactOrderParents(parent, group.body_a_local_idx, group.body_b_local_idx);
    }
    for (int i = 0; i < car_count && i < 8; ++i) {
        const int root = FindSharedContactOrderParent(parent, i);
        if (root >= 0 && root < 8) {
            component_max[root] = Max(component_max[root], i);
        }
    }

    int car_island_id[8];
    ResolveSharedContactOrderCarIslandIds(parent, component_max, car_island_id);

    int order[kMaxSharedContactOrderGroups];
    int island_ids[kMaxSharedContactOrderGroups];
    int active_count = 0;
    for (int group_idx = 0; group_idx < group_count && group_idx < kMaxSharedContactOrderGroups; ++group_idx) {
        const SharedContactOrderGroup& group = groups[group_idx];
        if (!group.valid || group.body_a_local_idx < 0 || group.body_a_local_idx >= car_count) {
            continue;
        }

        const int group_order_key = BulletContactOrderInitialKey(group);
        int insert = active_count;
        while (insert > 0 &&
               BulletContactOrderInitialKey(groups[order[insert - 1]]) > group_order_key) {
            order[insert] = order[insert - 1];
            --insert;
        }
        order[insert] = group_idx;
        ++active_count;
    }
    if (active_count <= 0) {
        return -1;
    }
    for (int order_idx = 0; order_idx < active_count; ++order_idx) {
        const SharedContactOrderGroup& group = groups[order[order_idx]];
        island_ids[order[order_idx]] = SharedContactOrderIslandIdForBody(car_island_id, group.body_a_local_idx);
    }
    if (active_count > 1) {
        QuickSortSharedContactOrderIndices(order, island_ids, 0, active_count - 1);
    }
    for (int order_idx = 0; order_idx < active_count; ++order_idx) {
        const SharedContactOrderGroup& group = groups[order[order_idx]];
        if (group.normal_row_count <= 0) {
            continue;
        }
        return group.bucket == SharedContactOrderBucket::Pair ? 0 : 1;
    }
    return -1;
}

ROCKETGPU_HD void SolveUnifiedVelocityContactRows(
    ::rocketgpu::BallState& ball,
    int count,
    ::rocketgpu::CarState* cars,
    const ::rocketgpu::CarConfig* car_configs,
    const ::rocketgpu::MutatorConfig& mutator,
    EnvStridedArray<BallVelocityContactRow> ball_normal_rows,
    int ball_normal_row_count,
    EnvStridedArray<BallFrictionContactRow> ball_friction_rows,
    int ball_friction_row_count,
    EnvStridedArray<CarSharedVelocityContactRow> world_normal_rows,
    int world_normal_row_count,
    EnvStridedArray<CarSharedVelocityContactRow> pair_normal_rows,
    int pair_normal_row_count,
    EnvStridedArray<CarSharedFrictionContactRow> world_friction_rows,
    int world_friction_row_count,
    EnvStridedArray<CarSharedFrictionContactRow> pair_friction_rows,
    int pair_friction_row_count,
    EnvStridedArray<const SharedContactOrderGroup> contact_order_groups = nullptr,
    int contact_order_group_count = 0,
    int env_idx = -1,
    uint64_t current_tick = 0,
    int substep = 0,
    const ::rocketgpu::Vec3* bt_ball_linear_velocity_override = nullptr,
    EnvStridedArray<const BulletManifoldEntry> sorted_manifolds = nullptr,
    int sorted_manifold_count = 0) {
    if (count < 0 || count > 8 || (count > 0 && (!cars || !car_configs))) {
        return;
    }
    if ((!ball_normal_rows || ball_normal_row_count <= 0) &&
        (!ball_friction_rows || ball_friction_row_count <= 0) &&
        (!world_normal_rows || world_normal_row_count <= 0) &&
        (!pair_normal_rows || pair_normal_row_count <= 0) &&
        (!world_friction_rows || world_friction_row_count <= 0) &&
        (!pair_friction_rows || pair_friction_row_count <= 0)) {
        return;
    }

    BallVelocityAccumulator ball_acc;
    BallVelocityAccumulator car_accs[8];
    for (int i = 0; i < count; ++i) {
        car_accs[i] = BallVelocityAccumulator{};
    }

    constexpr float kUnifiedRowConversionBtPerUu = 1.0f / 50.0f;
    constexpr float kUnifiedRowConversionUuPerBt = 50.0f;

    for (int i = 0; i < pair_normal_row_count; ++i) {
        CarSharedVelocityContactRow& row = pair_normal_rows[i];
        if (!row.valid) continue;
        row.applied_impulse *= kUnifiedRowConversionBtPerUu;
        if (row.has_bt) {
            row.rel_a_cross_normal = row.rel_a_cross_normal_bt;
            row.angular_component_a = row.angular_component_a_bt;
            if (row.body_b_local_idx >= 0) {
                row.rel_b_cross_normal = row.rel_b_cross_normal_bt;
                row.angular_component_b = row.angular_component_b_bt;
            } else {
                row.rel_b_cross_normal *= kUnifiedRowConversionBtPerUu;
                row.angular_component_b *= kUnifiedRowConversionUuPerBt;
            }
        } else {
            row.rel_a_cross_normal *= kUnifiedRowConversionBtPerUu;
            row.angular_component_a *= kUnifiedRowConversionUuPerBt;
            row.rel_b_cross_normal *= kUnifiedRowConversionBtPerUu;
            row.angular_component_b *= kUnifiedRowConversionUuPerBt;
        }
    }
    for (int i = 0; i < world_normal_row_count; ++i) {
        CarSharedVelocityContactRow& row = world_normal_rows[i];
        if (!row.valid) continue;
        row.applied_impulse *= kUnifiedRowConversionBtPerUu;
        if (row.has_bt) {
            row.rel_a_cross_normal = row.rel_a_cross_normal_bt;
            row.angular_component_a = row.angular_component_a_bt;
            if (row.body_b_local_idx >= 0) {
                row.rel_b_cross_normal = row.rel_b_cross_normal_bt;
                row.angular_component_b = row.angular_component_b_bt;
            } else {
                row.rel_b_cross_normal *= kUnifiedRowConversionBtPerUu;
                row.angular_component_b *= kUnifiedRowConversionUuPerBt;
            }
        } else {
            row.rel_a_cross_normal *= kUnifiedRowConversionBtPerUu;
            row.angular_component_a *= kUnifiedRowConversionUuPerBt;
            row.rel_b_cross_normal *= kUnifiedRowConversionBtPerUu;
            row.angular_component_b *= kUnifiedRowConversionUuPerBt;
        }
    }
    for (int i = 0; i < pair_friction_row_count; ++i) {
        CarSharedFrictionContactRow& row = pair_friction_rows[i];
        if (!row.valid) continue;
        row.applied_impulse *= kUnifiedRowConversionBtPerUu;
        if (row.has_bt) {
            row.rel_a_cross_tangent = row.rel_a_cross_tangent_bt;
            row.angular_component_a = row.angular_component_a_bt;
            if (row.body_b_local_idx >= 0) {
                row.rel_b_cross_tangent = row.rel_b_cross_tangent_bt;
                row.angular_component_b = row.angular_component_b_bt;
            } else {
                row.rel_b_cross_tangent *= kUnifiedRowConversionBtPerUu;
                row.angular_component_b *= kUnifiedRowConversionUuPerBt;
            }
        } else {
            row.rel_a_cross_tangent *= kUnifiedRowConversionBtPerUu;
            row.angular_component_a *= kUnifiedRowConversionUuPerBt;
            row.rel_b_cross_tangent *= kUnifiedRowConversionBtPerUu;
            row.angular_component_b *= kUnifiedRowConversionUuPerBt;
        }
    }
    for (int i = 0; i < world_friction_row_count; ++i) {
        CarSharedFrictionContactRow& row = world_friction_rows[i];
        if (!row.valid) continue;
        row.applied_impulse *= kUnifiedRowConversionBtPerUu;
        if (row.has_bt) {
            row.rel_a_cross_tangent = row.rel_a_cross_tangent_bt;
            row.angular_component_a = row.angular_component_a_bt;
            if (row.body_b_local_idx >= 0) {
                row.rel_b_cross_tangent = row.rel_b_cross_tangent_bt;
                row.angular_component_b = row.angular_component_b_bt;
            } else {
                row.rel_b_cross_tangent *= kUnifiedRowConversionBtPerUu;
                row.angular_component_b *= kUnifiedRowConversionUuPerBt;
            }
        } else {
            row.rel_a_cross_tangent *= kUnifiedRowConversionBtPerUu;
            row.angular_component_a *= kUnifiedRowConversionUuPerBt;
            row.rel_b_cross_tangent *= kUnifiedRowConversionBtPerUu;
            row.angular_component_b *= kUnifiedRowConversionUuPerBt;
        }
    }
    for (int i = 0; i < ball_normal_row_count; ++i) {
        BallVelocityContactRow& row = ball_normal_rows[i];
        if (!row.valid) continue;
        row.applied_impulse *= kUnifiedRowConversionBtPerUu;
        if (row.has_bt) {
            row.rhs = row.rhs_bt;
            row.rel_ball_cross_normal = row.rel_ball_cross_normal_bt;
            row.rel_other_cross_normal = row.rel_other_cross_normal_bt;
            row.ball_angular_component = row.ball_angular_component_bt;
            row.other_angular_component = row.other_angular_component_bt;
        } else {
            row.rhs *= kUnifiedRowConversionBtPerUu;
            row.rel_ball_cross_normal *= kUnifiedRowConversionBtPerUu;
            row.rel_other_cross_normal *= kUnifiedRowConversionBtPerUu;
            row.ball_angular_component *= kUnifiedRowConversionUuPerBt;
            row.other_angular_component *= kUnifiedRowConversionUuPerBt;
        }
    }
    for (int i = 0; i < ball_friction_row_count; ++i) {
        BallFrictionContactRow& row = ball_friction_rows[i];
        if (!row.valid) continue;
        row.applied_impulse *= kUnifiedRowConversionBtPerUu;
        if (row.has_bt) {
            row.rhs = row.rhs_bt;
            row.rel_ball_cross_tangent = row.rel_ball_cross_tangent_bt;
            row.rel_other_cross_tangent = row.rel_other_cross_tangent_bt;
            row.ball_angular_component = row.ball_angular_component_bt;
            row.other_angular_component = row.other_angular_component_bt;
        } else {
            row.rhs *= kUnifiedRowConversionBtPerUu;
            row.rel_ball_cross_tangent *= kUnifiedRowConversionBtPerUu;
            row.rel_other_cross_tangent *= kUnifiedRowConversionBtPerUu;
            row.ball_angular_component *= kUnifiedRowConversionUuPerBt;
            row.other_angular_component *= kUnifiedRowConversionUuPerBt;
        }
    }

    WarmstartSharedCarVelocityRows(count, cars, car_configs, mutator.car_mass, car_accs, pair_normal_rows, pair_normal_row_count);
    WarmstartSharedCarVelocityRows(count, cars, car_configs, mutator.car_mass, car_accs, world_normal_rows, world_normal_row_count);

    const int first_shared_contact_bucket = FirstSharedContactOrderBucket(
        contact_order_groups,
        contact_order_group_count,
        count);
    const bool solve_pair_normals_before_world = first_shared_contact_bucket == 0;

    const auto solve_world_normal_row = [&](int iter, int row_idx) {
            CarSharedVelocityContactRow& row = world_normal_rows[row_idx];
            SolveSharedCarVelocityNormalRow(
                count,
                cars,
                car_configs,
                mutator.car_mass,
                row,
                car_accs,
                iter,
                row_idx,
                "world",
                env_idx, mutator.solver_use_fma != 0);
    };
    const auto solve_pair_normal_row = [&](int iter, int row_idx) {
            CarSharedVelocityContactRow& row = pair_normal_rows[row_idx];
            SolveSharedCarVelocityNormalRow(
                count,
                cars,
                car_configs,
                mutator.car_mass,
                row,
                car_accs,
                iter,
                row_idx,
                "pair",
                env_idx, mutator.solver_use_fma != 0);
    };

    const auto solve_shared_friction_row = [&](
        int iter,
        int row_idx,
        EnvStridedArray<CarSharedFrictionContactRow> friction_rows,
        EnvStridedArray<CarSharedVelocityContactRow> normal_rows,
        int normal_row_count,
        int debug_bucket,
        const char* debug_bucket_name) {
            CarSharedFrictionContactRow& row = friction_rows[row_idx];
            SolveSharedCarVelocityFrictionRow(
                count,
                cars,
                car_configs,
                mutator.car_mass,
                normal_rows,
                normal_row_count,
                row,
                car_accs,
                iter,
                row_idx,
                debug_bucket_name,
                env_idx, mutator.solver_use_fma != 0);
    };

    int ball_normal_solve_order[kMaxBallVelocityContactRows];
    for (int i = 0; i < ball_normal_row_count; ++i) {
        ball_normal_solve_order[i] = i;
    }
    if (sorted_manifolds != nullptr && sorted_manifold_count > 0 && ball_normal_row_count > 1) {
        int out_count = 0;
        for (int order_idx = 0; order_idx < sorted_manifold_count; ++order_idx) {
            const BulletManifoldEntry& entry = sorted_manifolds[order_idx];
            if (entry.kind != 1) {
                continue;
            }
            for (int row_idx = 0; row_idx < ball_normal_row_count; ++row_idx) {
                const BallVelocityContactRow& row = ball_normal_rows[row_idx];
                if (!row.valid || row.other_local_idx != entry.body_a) {
                    continue;
                }
                bool already = false;
                for (int i = 0; i < out_count; ++i) {
                    if (ball_normal_solve_order[i] == row_idx) {
                        already = true;
                        break;
                    }
                }
                if (!already && out_count < ball_normal_row_count) {
                    ball_normal_solve_order[out_count++] = row_idx;
                }
            }
        }
        for (int row_idx = 0; row_idx < ball_normal_row_count && out_count < ball_normal_row_count; ++row_idx) {
            bool present = false;
            for (int i = 0; i < out_count; ++i) {
                if (ball_normal_solve_order[i] == row_idx) {
                    present = true;
                    break;
                }
            }
            if (!present) {
                ball_normal_solve_order[out_count++] = row_idx;
            }
        }
    }
    int ball_friction_solve_order[kMaxBallFrictionContactRows];
    {
        int friction_order_count = 0;
        for (int order_idx = 0; order_idx < ball_normal_row_count; ++order_idx) {
            const int normal_row_idx = ball_normal_solve_order[order_idx];
            for (int friction_idx = 0; friction_idx < ball_friction_row_count; ++friction_idx) {
                if (ball_friction_rows[friction_idx].normal_row_idx == normal_row_idx &&
                    friction_order_count < ball_friction_row_count) {
                    ball_friction_solve_order[friction_order_count++] = friction_idx;
                }
            }
        }
        for (int friction_idx = 0;
             friction_idx < ball_friction_row_count && friction_order_count < ball_friction_row_count;
             ++friction_idx) {
            bool present = false;
            for (int i = 0; i < friction_order_count; ++i) {
                if (ball_friction_solve_order[i] == friction_idx) {
                    present = true;
                    break;
                }
            }
            if (!present) {
                ball_friction_solve_order[friction_order_count++] = friction_idx;
            }
        }
    }

        const auto solve_ball_normal_row = [&](int iter, int row_idx) {
            BallVelocityContactRow& row = ball_normal_rows[row_idx];
            SolveBallVelocityNormalRowWithAccumulators(
                count,
                row,
                row_idx,
                ball_acc,
                car_accs,
                env_idx, mutator.solver_use_fma != 0);
        };

    constexpr int kUnifiedRowBucketWorld = 0;
    constexpr int kUnifiedRowBucketPair = 1;
    constexpr int kUnifiedRowBucketBall = 2;
    constexpr int kMaxUnifiedRows =
        kMaxCarWorldSharedVelocityContactRows + kMaxCarPairSharedVelocityContactRows + kMaxBallVelocityContactRows;
    uint8_t unified_normal_bucket[kMaxUnifiedRows];
    int16_t unified_normal_row[kMaxUnifiedRows];
    int unified_normal_count = 0;
    uint8_t unified_friction_bucket[kMaxUnifiedRows];
    int16_t unified_friction_row[kMaxUnifiedRows];
    int unified_friction_count = 0;
    bool unified_order_valid = false;
    if (sorted_manifolds != nullptr && sorted_manifold_count > 0 &&
        (world_normal_row_count + pair_normal_row_count + ball_normal_row_count) <= kMaxUnifiedRows) {
        EnvStridedArray<const BulletManifoldEntry> entries = sorted_manifolds;
        const int entry_count = sorted_manifold_count;
        {
            uint8_t world_used[kMaxCarWorldSharedVelocityContactRows] = {};
            uint8_t pair_used[kMaxCarPairSharedVelocityContactRows] = {};
            uint8_t ball_used[kMaxBallVelocityContactRows] = {};
            const auto push_row = [&](int bucket, int row_idx) {
                if (unified_normal_count >= kMaxUnifiedRows) {
                    return;
                }
                unified_normal_bucket[unified_normal_count] = static_cast<uint8_t>(bucket);
                unified_normal_row[unified_normal_count] = static_cast<int16_t>(row_idx);
                ++unified_normal_count;
            };
            for (int order_idx = 0; order_idx < entry_count; ++order_idx) {
                const BulletManifoldEntry& entry = entries[order_idx];
                if (entry.kind == 2) {
                    int expected_tag = -1;
                    if (entry.static_id >= 100) {
                        const int plane_rank = entry.static_id - 100;
                        expected_tag = plane_rank == 0 ? 6 : (plane_rank == 1 ? 5 : (plane_rank == 2 ? 2 : 1));
                    } else if (entry.static_id >= 0) {
                        expected_tag = 100 + entry.static_id;
                    }
                    for (int row_idx = 0; row_idx < world_normal_row_count; ++row_idx) {
                        const CarSharedVelocityContactRow& row = world_normal_rows[row_idx];
                        if (world_used[row_idx] || !row.valid) {
                            continue;
                        }
                        if (row.body_a_local_idx != entry.body_a || row.source_tag != expected_tag) {
                            continue;
                        }
                        world_used[row_idx] = 1;
                        push_row(kUnifiedRowBucketWorld, row_idx);
                    }
                } else if (entry.kind == 3) {
                    for (int row_idx = 0; row_idx < pair_normal_row_count; ++row_idx) {
                        const CarSharedVelocityContactRow& row = pair_normal_rows[row_idx];
                        if (pair_used[row_idx] || !row.valid) {
                            continue;
                        }
                        const int lo = Min(row.body_a_local_idx, row.body_b_local_idx);
                        const int hi = Max(row.body_a_local_idx, row.body_b_local_idx);
                        if (lo != entry.body_a || hi != entry.body_b) {
                            continue;
                        }
                        pair_used[row_idx] = 1;
                        push_row(kUnifiedRowBucketPair, row_idx);
                    }
                } else if (entry.kind == 1) {
                    for (int row_idx = 0; row_idx < ball_normal_row_count; ++row_idx) {
                        const BallVelocityContactRow& row = ball_normal_rows[row_idx];
                        if (ball_used[row_idx] || !row.valid || row.other_local_idx != entry.body_a) {
                            continue;
                        }
                        ball_used[row_idx] = 1;
                        push_row(kUnifiedRowBucketBall, row_idx);
                    }
                }
            }

            for (int row_idx = 0; row_idx < world_normal_row_count; ++row_idx) {
                if (!world_used[row_idx]) {
                    push_row(kUnifiedRowBucketWorld, row_idx);
                }
            }
            for (int row_idx = 0; row_idx < pair_normal_row_count; ++row_idx) {
                if (!pair_used[row_idx]) {
                    push_row(kUnifiedRowBucketPair, row_idx);
                }
            }
            for (int row_idx = 0; row_idx < ball_normal_row_count; ++row_idx) {
                if (!ball_used[row_idx]) {
                    push_row(kUnifiedRowBucketBall, row_idx);
                }
            }
            unified_order_valid =
                unified_normal_count == world_normal_row_count + pair_normal_row_count + ball_normal_row_count;
            if (unified_order_valid) {
                uint8_t world_friction_used[kMaxCarWorldSharedFrictionContactRows] = {};
                uint8_t pair_friction_used[kMaxCarPairSharedFrictionContactRows] = {};
                uint8_t ball_friction_used[kMaxBallFrictionContactRows] = {};
                const auto push_friction = [&](int bucket, int row_idx) {
                    if (unified_friction_count >= kMaxUnifiedRows) {
                        return;
                    }
                    unified_friction_bucket[unified_friction_count] = static_cast<uint8_t>(bucket);
                    unified_friction_row[unified_friction_count] = static_cast<int16_t>(row_idx);
                    ++unified_friction_count;
                };
                for (int order_idx = 0; order_idx < unified_normal_count; ++order_idx) {
                    const int normal_row_idx = unified_normal_row[order_idx];
                    switch (unified_normal_bucket[order_idx]) {
                        case kUnifiedRowBucketWorld:
                            for (int row_idx = 0; row_idx < world_friction_row_count; ++row_idx) {
                                if (!world_friction_used[row_idx] &&
                                    world_friction_rows[row_idx].normal_row_idx == normal_row_idx) {
                                    world_friction_used[row_idx] = 1;
                                    push_friction(kUnifiedRowBucketWorld, row_idx);
                                }
                            }
                            break;
                        case kUnifiedRowBucketPair:
                            for (int row_idx = 0; row_idx < pair_friction_row_count; ++row_idx) {
                                if (!pair_friction_used[row_idx] &&
                                    pair_friction_rows[row_idx].normal_row_idx == normal_row_idx) {
                                    pair_friction_used[row_idx] = 1;
                                    push_friction(kUnifiedRowBucketPair, row_idx);
                                }
                            }
                            break;
                        default:
                            for (int row_idx = 0; row_idx < ball_friction_row_count; ++row_idx) {
                                if (!ball_friction_used[row_idx] &&
                                    ball_friction_rows[row_idx].normal_row_idx == normal_row_idx) {
                                    ball_friction_used[row_idx] = 1;
                                    push_friction(kUnifiedRowBucketBall, row_idx);
                                }
                            }
                            break;
                    }
                }
                for (int row_idx = 0; row_idx < world_friction_row_count; ++row_idx) {
                    if (!world_friction_used[row_idx]) {
                        push_friction(kUnifiedRowBucketWorld, row_idx);
                    }
                }
                for (int row_idx = 0; row_idx < pair_friction_row_count; ++row_idx) {
                    if (!pair_friction_used[row_idx]) {
                        push_friction(kUnifiedRowBucketPair, row_idx);
                    }
                }
                for (int row_idx = 0; row_idx < ball_friction_row_count; ++row_idx) {
                    if (!ball_friction_used[row_idx]) {
                        push_friction(kUnifiedRowBucketBall, row_idx);
                    }
                }
                unified_order_valid =
                    unified_friction_count == world_friction_row_count + pair_friction_row_count + ball_friction_row_count;
            }
        }
    }

        const auto solve_world_friction_row = [&](int iter, int row_idx) {
            CarSharedFrictionContactRow& row = world_friction_rows[row_idx];
            SolveSharedCarVelocityFrictionRow(
                count,
                cars,
                car_configs,
                mutator.car_mass,
                world_normal_rows,
                world_normal_row_count,
                row,
                car_accs,
                iter,
                row_idx,
                "world",
                env_idx, mutator.solver_use_fma != 0);
        };

        const auto solve_ball_friction_row = [&](int iter, int row_idx) {
            BallFrictionContactRow& row = ball_friction_rows[row_idx];
            SolveBallVelocityFrictionRowWithAccumulators(
                count,
                ball_normal_rows,
                ball_normal_row_count,
                row,
                row_idx,
                ball_acc,
                car_accs,
                env_idx, mutator.solver_use_fma != 0);
        };

        const auto solve_pair_friction_row = [&](int iter, int row_idx) {
            CarSharedFrictionContactRow& row = pair_friction_rows[row_idx];
            SolveSharedCarVelocityFrictionRow(
                count,
                cars,
                car_configs,
                mutator.car_mass,
                pair_normal_rows,
                pair_normal_row_count,
                row,
                car_accs,
                iter,
                row_idx,
                "pair",
                env_idx, mutator.solver_use_fma != 0);
        };

    const int iterations = BallVelocityContactIterations();
    for (int iter = 0; iter < iterations; ++iter) {
        if (unified_order_valid) {
            for (int order_idx = 0; order_idx < unified_normal_count; ++order_idx) {
                const int row_idx = unified_normal_row[order_idx];
                switch (unified_normal_bucket[order_idx]) {
                    case kUnifiedRowBucketWorld: solve_world_normal_row(iter, row_idx); break;
                    case kUnifiedRowBucketPair: solve_pair_normal_row(iter, row_idx); break;
                    default: solve_ball_normal_row(iter, row_idx); break;
                }
            }
        } else {
            if (solve_pair_normals_before_world) {
                for (int row_idx = 0; row_idx < pair_normal_row_count; ++row_idx) {
                    solve_pair_normal_row(iter, row_idx);
                }
                for (int row_idx = 0; row_idx < world_normal_row_count; ++row_idx) {
                    solve_world_normal_row(iter, row_idx);
                }
            } else {
                for (int row_idx = 0; row_idx < world_normal_row_count; ++row_idx) {
                    solve_world_normal_row(iter, row_idx);
                }
                for (int row_idx = 0; row_idx < pair_normal_row_count; ++row_idx) {
                    solve_pair_normal_row(iter, row_idx);
                }
            }
            for (int order_idx = 0; order_idx < ball_normal_row_count; ++order_idx) {
                solve_ball_normal_row(iter, ball_normal_solve_order[order_idx]);
            }
        }
        if (unified_order_valid) {
            for (int order_idx = 0; order_idx < unified_friction_count; ++order_idx) {
                const int row_idx = unified_friction_row[order_idx];
                switch (unified_friction_bucket[order_idx]) {
                    case kUnifiedRowBucketWorld: solve_world_friction_row(iter, row_idx); break;
                    case kUnifiedRowBucketPair: solve_pair_friction_row(iter, row_idx); break;
                    default: solve_ball_friction_row(iter, row_idx); break;
                }
            }
        } else {
            if (solve_pair_normals_before_world) {
                for (int row_idx = 0; row_idx < pair_friction_row_count; ++row_idx) {
                    solve_shared_friction_row(iter, row_idx, pair_friction_rows, pair_normal_rows, pair_normal_row_count, 0, "pair");
                }
            }
            for (int row_idx = 0; row_idx < world_friction_row_count; ++row_idx) {
                solve_world_friction_row(iter, row_idx);
            }
            if (!solve_pair_normals_before_world) {
                for (int row_idx = 0; row_idx < pair_friction_row_count; ++row_idx) {
                    solve_pair_friction_row(iter, row_idx);
                }
            }
            for (int order_idx = 0; order_idx < ball_friction_row_count; ++order_idx) {
                solve_ball_friction_row(iter, ball_friction_solve_order[order_idx]);
            }
        }
    }

    for (int i = 0; i < pair_normal_row_count; ++i) {
        CarSharedVelocityContactRow& row = pair_normal_rows[i];
        if (!row.valid) continue;
        row.applied_impulse *= kUnifiedRowConversionUuPerBt;
        row.rel_a_cross_normal *= kUnifiedRowConversionUuPerBt;
        row.rel_b_cross_normal *= kUnifiedRowConversionUuPerBt;
        row.angular_component_a *= kUnifiedRowConversionBtPerUu;
        row.angular_component_b *= kUnifiedRowConversionBtPerUu;
    }
    for (int i = 0; i < world_normal_row_count; ++i) {
        CarSharedVelocityContactRow& row = world_normal_rows[i];
        if (!row.valid) continue;
        row.applied_impulse *= kUnifiedRowConversionUuPerBt;
        row.rel_a_cross_normal *= kUnifiedRowConversionUuPerBt;
        row.rel_b_cross_normal *= kUnifiedRowConversionUuPerBt;
        row.angular_component_a *= kUnifiedRowConversionBtPerUu;
        row.angular_component_b *= kUnifiedRowConversionBtPerUu;
    }
    for (int i = 0; i < pair_friction_row_count; ++i) {
        CarSharedFrictionContactRow& row = pair_friction_rows[i];
        if (!row.valid) continue;
        row.applied_impulse *= kUnifiedRowConversionUuPerBt;
        row.rel_a_cross_tangent *= kUnifiedRowConversionUuPerBt;
        row.rel_b_cross_tangent *= kUnifiedRowConversionUuPerBt;
        row.angular_component_a *= kUnifiedRowConversionBtPerUu;
        row.angular_component_b *= kUnifiedRowConversionBtPerUu;
    }
    for (int i = 0; i < world_friction_row_count; ++i) {
        CarSharedFrictionContactRow& row = world_friction_rows[i];
        if (!row.valid) continue;
        row.applied_impulse *= kUnifiedRowConversionUuPerBt;
        row.rel_a_cross_tangent *= kUnifiedRowConversionUuPerBt;
        row.rel_b_cross_tangent *= kUnifiedRowConversionUuPerBt;
        row.angular_component_a *= kUnifiedRowConversionBtPerUu;
        row.angular_component_b *= kUnifiedRowConversionBtPerUu;
    }
    for (int i = 0; i < ball_normal_row_count; ++i) {
        BallVelocityContactRow& row = ball_normal_rows[i];
        if (!row.valid) continue;
        row.applied_impulse *= kUnifiedRowConversionUuPerBt;
        row.rhs *= kUnifiedRowConversionUuPerBt;
        row.rel_ball_cross_normal *= kUnifiedRowConversionUuPerBt;
        row.rel_other_cross_normal *= kUnifiedRowConversionUuPerBt;
        row.ball_angular_component *= kUnifiedRowConversionBtPerUu;
        row.other_angular_component *= kUnifiedRowConversionBtPerUu;
    }
    for (int i = 0; i < ball_friction_row_count; ++i) {
        BallFrictionContactRow& row = ball_friction_rows[i];
        if (!row.valid) continue;
        row.applied_impulse *= kUnifiedRowConversionUuPerBt;
        row.rhs *= kUnifiedRowConversionUuPerBt;
        row.rel_ball_cross_tangent *= kUnifiedRowConversionUuPerBt;
        row.rel_other_cross_tangent *= kUnifiedRowConversionUuPerBt;
        row.ball_angular_component *= kUnifiedRowConversionBtPerUu;
        row.other_angular_component *= kUnifiedRowConversionBtPerUu;
    }

    const ::rocketgpu::Vec3 ball_base_linear_bt = bt_ball_linear_velocity_override
        ? *bt_ball_linear_velocity_override
        : ball.vel_bt;
    const ::rocketgpu::Vec3 ball_post_linear_bt = ball_base_linear_bt + ball_acc.delta_linear;
    SetBallLinearVelocityBt(ball, ball_post_linear_bt);
    ball.ang_vel += ball_acc.delta_angular;

    for (int i = 0; i < count; ++i) {
        SetCarLinearVelocityBt(cars[i], cars[i].vel_bt + car_accs[i].delta_linear);
        cars[i].ang_vel += car_accs[i].delta_angular;
    }
}

ROCKETGPU_HD void SolveCarWorldVelocityContactRows(
    ::rocketgpu::CarState& car,
    CarWorldVelocityContactRow* normal_rows,
    int normal_row_count,
    CarWorldFrictionContactRow* friction_rows,
    int friction_row_count) {
    if ((!normal_rows || normal_row_count <= 0) && (!friction_rows || friction_row_count <= 0)) {
        return;
    }

    BallVelocityAccumulator acc;
    const int iterations = BallVelocityContactIterations();

    bool all_bt = true;
    for (int row_idx = 0; row_idx < normal_row_count && all_bt; ++row_idx) {
        const CarWorldVelocityContactRow& row = normal_rows[row_idx];
        if (row.valid && !(row.has_rhs_bt && row.has_bt)) all_bt = false;
    }
    for (int row_idx = 0; row_idx < friction_row_count && all_bt; ++row_idx) {
        const CarWorldFrictionContactRow& row = friction_rows[row_idx];
        if (row.valid && !(row.has_rhs_bt && row.has_bt)) all_bt = false;
    }
    if (all_bt) {
        for (int iter = 0; iter < iterations; ++iter) {
            for (int row_idx = 0; row_idx < normal_row_count; ++row_idx) {
                CarWorldVelocityContactRow& row = normal_rows[row_idx];
                if (!row.valid) {
                    continue;
                }
                float delta_impulse = row.rhs_bt;
                const float delta_vel1_dotn = (
                    BtSimdDot3LikeBullet(row.normal, acc.delta_linear) +
                    BtSimdDot3LikeBullet(row.rel_cross_normal_bt, acc.delta_angular));
                delta_impulse -= delta_vel1_dotn * row.jac_diag_ab_inv;
                const float sum = row.applied_impulse + delta_impulse;
                if (sum < 0.0f) {
                    delta_impulse = 0.0f - row.applied_impulse;
                    row.applied_impulse = 0.0f;
                } else {
                    row.applied_impulse = sum;
                }
                ApplyBallVelocityAccumulatorImpulse(
                    acc,
                    row.normal * row.inv_mass,
                    row.angular_component_bt,
                    delta_impulse);
            }
            for (int row_idx = 0; row_idx < friction_row_count; ++row_idx) {
                CarWorldFrictionContactRow& row = friction_rows[row_idx];
                if (!row.valid || row.normal_row_idx < 0 || row.normal_row_idx >= normal_row_count) {
                    continue;
                }
                const float total_impulse = normal_rows[row.normal_row_idx].applied_impulse;
                if (total_impulse <= 0.0f) {
                    continue;
                }
                const float max_impulse = row.friction * total_impulse;
                float delta_impulse = row.rhs_bt;
                const float delta_vel1_dotn = (
                    BtSimdDot3LikeBullet(row.tangent, acc.delta_linear) +
                    BtSimdDot3LikeBullet(row.rel_cross_tangent_bt, acc.delta_angular));
                delta_impulse -= delta_vel1_dotn * row.jac_diag_ab_inv;
                const float sum = row.applied_impulse + delta_impulse;
                if (sum < -max_impulse) {
                    delta_impulse = -max_impulse - row.applied_impulse;
                    row.applied_impulse = -max_impulse;
                } else if (sum > max_impulse) {
                    delta_impulse = max_impulse - row.applied_impulse;
                    row.applied_impulse = max_impulse;
                } else {
                    row.applied_impulse = sum;
                }
                ApplyBallVelocityAccumulatorImpulse(
                    acc,
                    row.tangent * row.inv_mass,
                    row.angular_component_bt,
                    delta_impulse);
            }
        }
        car.solve_delta_linear_bt = acc.delta_linear;
        car.solve_delta_angular = acc.delta_angular;
        SetCarLinearVelocityBt(car, car.vel_bt + acc.delta_linear);
        car.ang_vel += acc.delta_angular;
        return;
    }
    for (int iter = 0; iter < iterations; ++iter) {
        for (int row_idx = 0; row_idx < normal_row_count; ++row_idx) {
            CarWorldVelocityContactRow& row = normal_rows[row_idx];
            if (!row.valid) {
                continue;
            }

            float delta_impulse = row.rhs;
            const float delta_impulse_vel_uu = (
                BtSimdDot3LikeBullet(row.normal, acc.delta_linear) +
                BtSimdDot3LikeBullet(row.rel_cross_normal, acc.delta_angular));
            delta_impulse -= delta_impulse_vel_uu * row.jac_diag_ab_inv;
            if (row.has_rhs_bt) {
                constexpr float kBtPerUu = 1.0f / 50.0f;
                constexpr float kUuPerBt = 50.0f;
                delta_impulse = (row.rhs_bt - delta_impulse_vel_uu * kBtPerUu * row.jac_diag_ab_inv) * kUuPerBt;
            }

            const float sum = row.applied_impulse + delta_impulse;
            if (sum < 0.0f) {
                delta_impulse = -row.applied_impulse;
                row.applied_impulse = 0.0f;
            } else {
                row.applied_impulse = sum;
            }

            if (delta_impulse == 0.0f) {
                continue;
            }

            ApplyBallVelocityAccumulatorImpulse(
                acc,
                row.normal * row.inv_mass,
                row.angular_component,
                delta_impulse);
        }

        for (int row_idx = 0; row_idx < friction_row_count; ++row_idx) {
            CarWorldFrictionContactRow& row = friction_rows[row_idx];
            if (!row.valid || row.normal_row_idx < 0 || row.normal_row_idx >= normal_row_count) {
                continue;
            }

            const float total_impulse = normal_rows[row.normal_row_idx].applied_impulse;
            if (total_impulse <= 0.0f) {
                continue;
            }

            const float max_impulse = row.friction * total_impulse;
            float delta_impulse = row.rhs;
            delta_impulse -= (
                BtSimdDot3LikeBullet(row.tangent, acc.delta_linear) +
                BtSimdDot3LikeBullet(row.rel_cross_tangent, acc.delta_angular)) * row.jac_diag_ab_inv;

            const float sum = row.applied_impulse + delta_impulse;
            if (sum < -max_impulse) {
                delta_impulse = -max_impulse - row.applied_impulse;
                row.applied_impulse = -max_impulse;
            } else if (sum > max_impulse) {
                delta_impulse = max_impulse - row.applied_impulse;
                row.applied_impulse = max_impulse;
            } else {
                row.applied_impulse = sum;
            }

            if (delta_impulse == 0.0f) {
                continue;
            }

            ApplyBallVelocityAccumulatorImpulse(
                acc,
                row.tangent * row.inv_mass,
                row.angular_component,
                delta_impulse);
        }
    }

    SetCarLinearVelocityUu(car, car.vel + acc.delta_linear);
    car.ang_vel += acc.delta_angular;
}

ROCKETGPU_HD int WorldSolverPasses() {
    return 10;
}

ROCKETGPU_HD void IntegrateSplitImpulseTransform(
    ::rocketgpu::CarState& car,
    const SplitImpulseAccumulator& split,
    float dt,
    bool native_sse_math) {
    SetCarOriginBt(car, car.pos_bt + UuVecToBtDiv(split.push_velocity) * dt);
    if (LengthSq(split.turn_velocity) > 0.0f) {
        IntegrateCarOrientation(car, split.turn_velocity * BallSplitTurnErp(), dt, "split_uu", false, native_sse_math);
    }
}

ROCKETGPU_HD void IntegrateSplitImpulseTransformBt(
    ::rocketgpu::CarState& car,
    const SplitImpulseAccumulator& split_bt,
    float dt,
    bool native_sse_math) {
    const bool has_push = split_bt.push_velocity.x != 0.0f ||
                          split_bt.push_velocity.y != 0.0f ||
                          split_bt.push_velocity.z != 0.0f;
    const bool has_turn = split_bt.turn_velocity.x != 0.0f ||
                          split_bt.turn_velocity.y != 0.0f ||
                          split_bt.turn_velocity.z != 0.0f;
    if (!has_push && !has_turn) {
        return;
    }
    SetCarOriginBt(car, car.pos_bt + split_bt.push_velocity * dt);
    IntegrateCarOrientation(car, split_bt.turn_velocity * BallSplitTurnErp(), dt, "split_bt", false, native_sse_math);
}

ROCKETGPU_HD void InitializeBallSplitContactRow(
    BallSplitContactRow& row,
    float penetration,
    const ::rocketgpu::Vec3& normal,
    float ball_inv_mass,
    const ::rocketgpu::Vec3& rel_ball_contact,
    const ::rocketgpu::Vec3& ball_angular_component,
    int32_t other_local_idx,
    float other_inv_mass,
    const ::rocketgpu::Vec3& rel_other_contact,
    const ::rocketgpu::Vec3& other_angular_component,
    float dt,
    const ::rocketgpu::Vec3* rel_ball_contact_bt_override = nullptr,
    const ::rocketgpu::Vec3* rel_other_contact_bt_override = nullptr,
    const float* penetration_bt_override = nullptr,
    const ::rocketgpu::Vec3* ball_angular_component_bt_override = nullptr,
    const ::rocketgpu::Vec3* other_angular_component_bt_override = nullptr) {
    row = BallSplitContactRow{};
    if (penetration <= 0.0f || dt <= 0.0f) {
        return;
    }

    float denom = 0.0f;
    if (ball_inv_mass > 0.0f) {
        denom += ball_inv_mass + Dot(normal, Cross(ball_angular_component, rel_ball_contact));
    }
    if (other_inv_mass > 0.0f) {
        denom += other_inv_mass + Dot(normal, Cross(-other_angular_component, rel_other_contact));
    }
    if (denom <= 1e-8f) {
        return;
    }

    row.valid = 1;
    row.other_local_idx = other_local_idx;
    row.normal = normal;
    row.rel_ball_contact = rel_ball_contact;
    row.rel_other_contact = rel_other_contact;
    row.rel_ball_cross_normal = Cross(rel_ball_contact, normal);
    row.rel_other_cross_normal = -Cross(rel_other_contact, normal);
    row.ball_angular_component = ball_angular_component;
    row.other_angular_component = other_angular_component;
    row.ball_inv_mass = ball_inv_mass;
    row.other_inv_mass = other_inv_mass;
    row.jac_diag_ab_inv = 1.0f / denom;
    row.rhs_penetration = penetration * SplitImpulsePenetrationErp() * (1.0f / dt) * row.jac_diag_ab_inv;

    constexpr float kSplitUuToBt = 1.0f / 50.0f;
    const ::rocketgpu::Vec3 rel_ball_bt = rel_ball_contact_bt_override
        ? *rel_ball_contact_bt_override
        : rel_ball_contact * kSplitUuToBt;
    const ::rocketgpu::Vec3 rel_other_bt = rel_other_contact_bt_override
        ? *rel_other_contact_bt_override
        : rel_other_contact * kSplitUuToBt;
    const ::rocketgpu::Vec3 ball_ang_bt = ball_angular_component_bt_override
        ? *ball_angular_component_bt_override
        : ball_angular_component * 50.0f;
    const ::rocketgpu::Vec3 other_ang_bt = other_angular_component_bt_override
        ? *other_angular_component_bt_override
        : other_angular_component * 50.0f;
    row.rel_ball_cross_normal_bt = Cross(rel_ball_bt, normal);
    row.rel_other_cross_normal_bt = -Cross(rel_other_bt, normal);
    row.ball_angular_component_bt = ball_ang_bt;
    row.other_angular_component_bt = other_ang_bt;
    float denom_bt = 0.0f;
    if (ball_inv_mass > 0.0f) {
        denom_bt += ball_inv_mass + Dot(normal, Cross(ball_ang_bt, rel_ball_bt));
    }
    if (other_inv_mass > 0.0f) {
        denom_bt += other_inv_mass + Dot(normal, Cross(-other_ang_bt, rel_other_bt));
    }
    row.jac_diag_ab_inv_bt = denom_bt > 1e-8f ? (1.0f / denom_bt) : row.jac_diag_ab_inv;
    const float penetration_bt = (penetration_bt_override && *penetration_bt_override > 0.0f)
        ? *penetration_bt_override
        : penetration * kSplitUuToBt;
    row.rhs_penetration_bt =
        penetration_bt * SplitImpulsePenetrationErp() * (1.0f / dt) * row.jac_diag_ab_inv_bt;
    row.has_bt = 1;
}

ROCKETGPU_HD void AppendBallStaticSplitRow(
    EnvStridedArray<BallSplitContactRow> rows,
    int& row_count,
    float penetration,
    const ::rocketgpu::Vec3& normal,
    const ::rocketgpu::Vec3& rel_ball_contact,
    const ::rocketgpu::MutatorConfig& mutator,
    float dt,
    const ::rocketgpu::Vec3* rel_ball_contact_bt = nullptr,
    const float* penetration_bt = nullptr,
    const ::rocketgpu::Vec3* ball_angular_component_bt = nullptr,
    int16_t static_id = -1) {
    if (!rows || row_count >= kMaxBallSplitContactRows) {
        return;
    }

    BallSplitContactRow& row = rows[row_count];
    InitializeBallSplitContactRow(
        row,
        penetration * BallStaticSplitPenetrationScale(),
        normal,
        1.0f / mutator.ball_mass,
        rel_ball_contact,
        ApplyInverseBallInertiaWorld(mutator.ball_mass, mutator.ball_radius, Cross(rel_ball_contact, normal)),
        -1,
        0.0f,
        ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f),
        ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f),
        dt,
        rel_ball_contact_bt,
        nullptr,
        penetration_bt,
        ball_angular_component_bt);
    if (row.valid) {
        row.static_id = static_id;
        ++row_count;
    }
}

ROCKETGPU_HD void AppendBallCarSplitRow(
    EnvStridedArray<BallSplitContactRow> rows,
    int& row_count,
    float penetration,
    const ::rocketgpu::Vec3& normal,
    const ::rocketgpu::Vec3& rel_ball_contact,
    const ::rocketgpu::Vec3& rel_car_contact,
    int32_t other_local_idx,
    const ::rocketgpu::CarState& car,
    const ::rocketgpu::CarConfig& cfg,
    const ::rocketgpu::MutatorConfig& mutator,
    float dt,
    const ::rocketgpu::Vec3* rel_ball_contact_bt = nullptr,
    const ::rocketgpu::Vec3* rel_car_contact_bt = nullptr,
    const float* penetration_bt = nullptr,
    const ::rocketgpu::BallState* ball_for_bt = nullptr) {
    if (!rows || row_count >= kMaxBallSplitContactRows) {
        return;
    }

    BallSplitContactRow& row = rows[row_count];

    ::rocketgpu::Vec3 car_angular_component_bt(0.0f, 0.0f, 0.0f);
    if (rel_car_contact_bt) {
        car_angular_component_bt =
            ApplyInverseBodyInertiaWorldBt(car, cfg, mutator.car_mass, -Cross(*rel_car_contact_bt, normal));
    }

    ::rocketgpu::Vec3 ball_angular_component_bt(0.0f, 0.0f, 0.0f);
    if (rel_ball_contact_bt && ball_for_bt) {
        ball_angular_component_bt = ApplyInverseBallInertiaWorldBtMutator(*ball_for_bt, mutator, Cross(*rel_ball_contact_bt, normal));
    }
    InitializeBallSplitContactRow(
        row,
        penetration * BallCarSplitPenetrationScale(),
        normal,
        1.0f / mutator.ball_mass,
        rel_ball_contact,
        ApplyInverseBallInertiaWorld(mutator.ball_mass, mutator.ball_radius, Cross(rel_ball_contact, normal)),
        other_local_idx,
        1.0f / mutator.car_mass,
        rel_car_contact,
        ApplyInverseBodyInertiaWorld(car, cfg, mutator.car_mass, -Cross(rel_car_contact, normal)),
        dt,
        rel_ball_contact_bt,
        rel_car_contact_bt,
        penetration_bt,
        (rel_ball_contact_bt && ball_for_bt) ? &ball_angular_component_bt : nullptr,
        rel_car_contact_bt ? &car_angular_component_bt : nullptr);
    if (row.valid) {
        ++row_count;
    }
}

ROCKETGPU_HD int DebugWorldSubsteps() {
    return 1;
}

ROCKETGPU_HD float CarWorldPenetrationSolveGround() {
    return 0.13333334f;
}

ROCKETGPU_HD float CarWorldPenetrationSolveAir() {
    return 0.0015f;
}

ROCKETGPU_HD float CarContactBreakingThresholdUU(const ::rocketgpu::CarConfig& cfg) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kBTToUU = 50.0f;
    const float margin = BallCarShapeMargin(cfg);
    const ::rocketgpu::Vec3 half_ext_with_margin = (cfg.hitbox_size * 0.5f) - BallCarHitboxInset(cfg) + ::rocketgpu::Vec3(margin, margin, margin);
    const float car_bounding_sphere_bt = Length(half_ext_with_margin * kUUToBT);
    const float car_center_offset_bt = Length(cfg.hitbox_offset) * kUUToBT;
    const float car_angular_motion_disc = car_bounding_sphere_bt + car_center_offset_bt;
    return car_angular_motion_disc * 0.02f * kBTToUU;
}

ROCKETGPU_HD void ResolveObbCornerCylinder(
    ::rocketgpu::CarState& car,
    const ::rocketgpu::CarConfig& cfg,
    float restitution,
    float friction,
    ContactAccumulator& acc,
    const ::rocketgpu::Vec3& external_linear_delta = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f)) {
    ::rocketgpu::Vec3& vel = car.vel;
    const ::rocketgpu::Quat& rot = car.rot;
    const ::rocketgpu::Vec3 center = car.pos + Rotate(rot, cfg.hitbox_offset);
    if (!InMainCornerZone(center.x, center.y)) {
        return;
    }

    const float radius = BoundingSphereRadius(cfg);
    const float sx = Sign(center.x);
    const float sy = Sign(center.y);
    const ::rocketgpu::Vec3 cyl_center(sx * CornerCenterX(), sy * CornerCenterY(), 0.0f);
    ::rocketgpu::Vec3 radial(center.x - cyl_center.x, center.y - cyl_center.y, 0.0f);
    const float dist = Length(radial);
    const float allowed = CornerRadius() - radius;
    if (dist <= allowed) {
        return;
    }

    const ::rocketgpu::Vec3 inward_normal = Normalize(-radial, ::rocketgpu::Vec3(-sx, -sy, 0.0f));
    const float penetration = dist - allowed;
    SetCarOriginBt(car, car.pos_bt + UuVecToBtDiv(inward_normal * penetration));
    const ::rocketgpu::Vec3 effective_vel = vel + external_linear_delta;
    const float vn = Dot(effective_vel, inward_normal);
    if (vn < 0.0f) {
        vel -= inward_normal * ((1.0f + restitution) * vn);
    }
    const ::rocketgpu::Vec3 tangent = effective_vel - inward_normal * Dot(effective_vel, inward_normal);
    vel -= tangent * Min(1.0f, friction * 0.5f);
    acc.add(inward_normal, penetration);
}
