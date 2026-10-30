

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
// Ball contact rows per env tick (rows past these are dropped). The Octane recordings (1v1 and 3v3, ~430k env-ticks)
// never use more than 9; 32 holds four full 4-point manifolds of world contact plus four touching cars.
constexpr int kMaxBallSplitContactRows = 32;
constexpr int kMaxBallVelocityContactRows = 32;
constexpr int kMaxBallFrictionContactRows = 32;
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

// Retained car-world mesh manifold (cal_retained_world_contacts), one per (car, arena mesh) pair holding points.
// Each point keeps the normal it was created with (v3 normal_world_on_b, internal-edge adjusted), and local_point_b_bt
// is in the mesh body frame (world minus the mesh's cal_origin_y). A slot with count 0 is free.
struct CarWorldMeshRetainedPoint {
    ::rocketgpu::Vec3 local_point_a_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 local_point_b_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 normal = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    float distance_bt = 0.0f;
};

struct CarWorldMeshRetainedManifold {
    uint8_t count = 0;
    uint8_t mesh = 0;  // ArenaMeshBvhMeshInfo index
    CarWorldMeshRetainedPoint points[4] = {};
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
// Arena meshes a car can hold retained points with at once (at most 2 on all four recording sets); a further mesh
// keeps only its fresh points.
constexpr int kCarWorldMeshRetainedManifoldsPerCar = 2;
constexpr int kMaxCarWorldMeshRetainedManifolds =
    kCarWorldMeshRetainedManifoldsPerCar * kMaxCarWorldMeshPersistentCars;

// SolveCarWorldStaticContacts' per-call rows and order groups, before they go to the shared rows (or are solved
// in place). On the GPU they live in the row pool, one per car slot, instead of on every thread's stack.
struct CarWorldLocalRows {
    UninitArray<CarWorldSplitContactRow, kMaxCarWorldSplitContactRows> split_rows;
    UninitArray<CarWorldVelocityContactRow, kMaxCarWorldVelocityContactRows> normal_rows;
    UninitArray<CarWorldFrictionContactRow, kMaxCarWorldFrictionContactRows> friction_rows;
    UninitArray<SharedContactOrderGroup, kMaxCarWorldMeshesPerCar + 4> order_groups;
};

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
    // The child shape in world space (bt): centre, axes and half extents with margin (a ball's are its getAabb extent).
    ::rocketgpu::Vec3 child_center = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 child_axis[3] = {::rocketgpu::Vec3(1.0f, 0.0f, 0.0f), ::rocketgpu::Vec3(0.0f, 1.0f, 0.0f),
                                       ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f)};
    ::rocketgpu::Vec3 child_half = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    int32_t cell_i = 0;
    int32_t cell_j = 0;
    int32_t cell_k = 0;
    uint8_t valid = 0;
    uint8_t is_ball = 0;
};

constexpr int kMaxBulletManifoldEntries = 96;
// btAxisSweep3 handles: arena meshes, the 4 planes, the ball and up to 8 cars (plus slack); edges per axis include
// the two sentinels.
constexpr int kAxisSweepMaxHandles = 40;
constexpr int kAxisSweepMaxEdges = 2 * kAxisSweepMaxHandles + 2;

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

// Per-tick car-car contact cache of one env: pair (a, b) of local car slots at [a * cars + b].
struct CarCarPairCacheView {
    EnvStridedArray<CarCarPairCachedContact> slots;
    int cars = 0;
    ROCKETGPU_HD explicit operator bool() const { return static_cast<bool>(slots); }
    ROCKETGPU_HD int size() const { return cars * cars; }
    ROCKETGPU_HD int index(int a, int b) const { return a < cars && b < cars ? a * cars + b : -1; }
};

struct WorldSolverScratchRows {
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
    CarCarPairCacheView pair_contact_cache;
    // Contact manifolds kept across ticks, contiguous per env: car slot i's planes at [i * 4], car pair (lo, hi) at
    // [CarCarRetainedSlot(lo, hi)], car slot i's ball manifold at [i]. Cleared on kickoff reset through
    // WorldSolverScratch::retained_reset_pending.
    CarWorldPlaneRetainedManifold* car_world_plane_manifolds = nullptr;
    CarWorldMeshRetainedManifold* car_world_mesh_manifolds = nullptr;  // car slot i's at [i * 2]
    CarCarRetainedManifold* car_car_retained_manifolds = nullptr;
    BallCarRetainedManifold* ball_car_retained_manifolds = nullptr;
    // SolveCarWorldStaticContacts scratch, contiguous per env: car slot i's at [i].
    CarWorldLocalRows* car_world_local_rows = nullptr;
    int car_world_plane_count = 0;
    int car_world_mesh_count = 0;
    int car_car_retained_count = 0;
    int ball_car_retained_count = 0;
};

// Rows each per-env pool holds. World pools fill at most kMaxCarWorld*ContactRows (16) rows per car (each car's rows
// come from local arrays of 16, appended car after car or placed at slot*16), and pair pools at most
// kMaxApproxCarCarContactPoints (4) rows per resolved car pair (one ResolveCarCarCollision per pair per tick). An env
// holds at most max_cars_per_env cars, so these caps are never reached; past kMax the existing kMax guards apply.
struct WorldSolverScratchRowCaps {
    int ball_split = kMaxBallSplitContactRows;
    int ball_velocity = kMaxBallVelocityContactRows;
    int ball_friction = kMaxBallFrictionContactRows;
    int pair_split = kMaxCarPairSplitContactRows;
    int world_split = kMaxCarSharedSplitContactRows;
    int pair_velocity = kMaxCarPairSharedVelocityContactRows;
    int pair_friction = kMaxCarPairSharedFrictionContactRows;
    int world_velocity = kMaxCarWorldSharedVelocityContactRows;
    int world_friction = kMaxCarWorldSharedFrictionContactRows;
    int order_groups = kMaxSharedContactOrderGroups;
    int manifolds = kMaxBulletManifoldEntries;
    int pair_cache_cars = 8;  // pair cache slots: pair_cache_cars^2
    int car_world_plane = kMaxCarWorldPlaneRetainedManifolds;
    int car_world_mesh = kMaxCarWorldMeshRetainedManifolds;
    int car_car_retained = kMaxCarCarRetainedManifolds;
    int ball_car_retained = kMaxBallCarRetainedManifolds;
    int car_world_local = kMaxCarWorldMeshPersistentCars;
};

constexpr int kMaxCarPairRowsPerPair = 4;  // kMaxApproxCarCarContactPoints; checked in env_step.hpp

ROCKETGPU_HD int CapRows(int max_rows, long long needed) {
    return needed < max_rows ? static_cast<int>(needed) : max_rows;
}

ROCKETGPU_HD WorldSolverScratchRowCaps MakeWorldSolverScratchRowCaps(int max_cars_per_env) {
    WorldSolverScratchRowCaps caps;
    const long long cars = max_cars_per_env < 0 ? 0 : max_cars_per_env;
    const long long pairs = cars * (cars - 1) / 2;
    caps.world_split = CapRows(kMaxCarSharedSplitContactRows, cars * kMaxCarWorldSplitContactRows);
    caps.world_velocity = CapRows(kMaxCarWorldSharedVelocityContactRows, cars * kMaxCarWorldVelocityContactRows);
    caps.world_friction = CapRows(kMaxCarWorldSharedFrictionContactRows, cars * kMaxCarWorldFrictionContactRows);
    caps.pair_split = CapRows(kMaxCarPairSplitContactRows, pairs * kMaxCarPairRowsPerPair);
    caps.pair_velocity = CapRows(kMaxCarPairSharedVelocityContactRows, pairs * kMaxCarPairRowsPerPair);
    caps.pair_friction = CapRows(kMaxCarPairSharedFrictionContactRows, pairs * kMaxCarPairRowsPerPair);
    caps.pair_cache_cars = static_cast<int>(cars < 8 ? cars : 8);
    caps.car_world_plane = CapRows(kMaxCarWorldPlaneRetainedManifolds, cars * kCarWorldPlaneRetainedManifoldsPerCar);
    caps.car_world_mesh = CapRows(kMaxCarWorldMeshRetainedManifolds, cars * kCarWorldMeshRetainedManifoldsPerCar);
    caps.car_car_retained = CapRows(kMaxCarCarRetainedManifolds, pairs);
    caps.ball_car_retained = CapRows(kMaxBallCarRetainedManifolds, cars);
    caps.car_world_local = CapRows(kMaxCarWorldMeshPersistentCars, cars);
    // Order groups: each car's world groups come from a local array of kMaxCarWorldMeshesPerCar + 4, and car pairs
    // add one group per resolved pair; cars^2 covers the pairs with room to spare.
    caps.order_groups = CapRows(kMaxSharedContactOrderGroups, cars * (kMaxCarWorldMeshesPerCar + 4) + cars * cars);
    return caps;
}

struct WorldSolverScratchRowPool {
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
    CarCarPairCachedContact* pair_contact_cache = nullptr;
    CarWorldPlaneRetainedManifold* car_world_plane_manifolds = nullptr;
    CarWorldMeshRetainedManifold* car_world_mesh_manifolds = nullptr;
    CarCarRetainedManifold* car_car_retained_manifolds = nullptr;
    BallCarRetainedManifold* ball_car_retained_manifolds = nullptr;
    CarWorldLocalRows* car_world_local_rows = nullptr;
    int env_count = 0;
    uint8_t interleaved = 0;
    WorldSolverScratchRowCaps caps;

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
        rows.ball_split_rows = view_one(ball_split_rows, caps.ball_split, env_idx);
        rows.ball_velocity_rows = view_one(ball_velocity_rows, caps.ball_velocity, env_idx);
        rows.ball_friction_rows = view_one(ball_friction_rows, caps.ball_friction, env_idx);
        rows.pair_split_rows = view_one(pair_split_rows, caps.pair_split, env_idx);
        rows.world_split_rows = view_one(world_split_rows, caps.world_split, env_idx);
        rows.pair_velocity_rows = view_one(pair_velocity_rows, caps.pair_velocity, env_idx);
        rows.pair_friction_rows = view_one(pair_friction_rows, caps.pair_friction, env_idx);
        rows.world_velocity_rows = view_one(world_velocity_rows, caps.world_velocity, env_idx);
        rows.world_friction_rows = view_one(world_friction_rows, caps.world_friction, env_idx);
        rows.shared_contact_order_groups = view_one(shared_contact_order_groups, caps.order_groups, env_idx);
        rows.bullet_manifolds = view_one(bullet_manifolds, caps.manifolds, env_idx);
        rows.pair_contact_cache.slots =
            view_one(pair_contact_cache, caps.pair_cache_cars * caps.pair_cache_cars, env_idx);
        rows.pair_contact_cache.cars = caps.pair_cache_cars;
        const auto contiguous = [env_idx](auto* base, int per_env) {
            return base == nullptr ? base : base + static_cast<std::ptrdiff_t>(env_idx) * per_env;
        };
        rows.car_world_plane_manifolds = contiguous(car_world_plane_manifolds, caps.car_world_plane);
        rows.car_world_mesh_manifolds = contiguous(car_world_mesh_manifolds, caps.car_world_mesh);
        rows.car_car_retained_manifolds = contiguous(car_car_retained_manifolds, caps.car_car_retained);
        rows.ball_car_retained_manifolds = contiguous(ball_car_retained_manifolds, caps.ball_car_retained);
        rows.car_world_local_rows = contiguous(car_world_local_rows, caps.car_world_local);
        rows.car_world_plane_count = caps.car_world_plane;
        rows.car_world_mesh_count = caps.car_world_mesh;
        rows.car_car_retained_count = caps.car_car_retained;
        rows.ball_car_retained_count = caps.ball_car_retained;
        return rows;
    }
};

ROCKETGPU_HD size_t WorldSolverScratchRowBytesPerEnv(const WorldSolverScratchRowCaps& caps = WorldSolverScratchRowCaps{}) {
    return sizeof(BallSplitContactRow) * caps.ball_split +
           sizeof(BallVelocityContactRow) * caps.ball_velocity +
           sizeof(BallFrictionContactRow) * caps.ball_friction +
           sizeof(CarSharedSplitContactRow) * caps.pair_split +
           sizeof(CarSharedSplitContactRow) * caps.world_split +
           sizeof(CarSharedVelocityContactRow) * caps.pair_velocity +
           sizeof(CarSharedFrictionContactRow) * caps.pair_friction +
           sizeof(CarSharedVelocityContactRow) * caps.world_velocity +
           sizeof(CarSharedFrictionContactRow) * caps.world_friction +
           sizeof(SharedContactOrderGroup) * caps.order_groups +
           sizeof(BulletManifoldEntry) * caps.manifolds +
           sizeof(CarCarPairCachedContact) * caps.pair_cache_cars * caps.pair_cache_cars +
           sizeof(CarWorldPlaneRetainedManifold) * caps.car_world_plane +
           sizeof(CarWorldMeshRetainedManifold) * caps.car_world_mesh +
           sizeof(CarCarRetainedManifold) * caps.car_car_retained +
           sizeof(BallCarRetainedManifold) * caps.ball_car_retained +
           sizeof(CarWorldLocalRows) * caps.car_world_local;
}

// Kickoff reset of the retained manifolds (what WorldSolverScratch::retained_reset_pending asks for).
ROCKETGPU_HD void ClearRetainedContactManifolds(const WorldSolverScratchRows& rows) {
    for (int i = 0; rows.car_world_plane_manifolds && i < rows.car_world_plane_count; ++i) rows.car_world_plane_manifolds[i].count = 0;
    for (int i = 0; rows.car_world_mesh_manifolds && i < rows.car_world_mesh_count; ++i) rows.car_world_mesh_manifolds[i].count = 0;
    for (int i = 0; rows.car_car_retained_manifolds && i < rows.car_car_retained_count; ++i) rows.car_car_retained_manifolds[i].count = 0;
    for (int i = 0; rows.ball_car_retained_manifolds && i < rows.ball_car_retained_count; ++i) rows.ball_car_retained_manifolds[i].count = 0;
}

ROCKETGPU_HD void ZeroWorldSolverScratchRows(const WorldSolverScratchRows& rows, const WorldSolverScratchRowCaps& caps) {
    if (!rows.ball_split_rows) {
        return;
    }
    for (int i = 0; i < caps.ball_split; ++i) rows.ball_split_rows[i] = BallSplitContactRow{};
    for (int i = 0; i < caps.ball_velocity; ++i) rows.ball_velocity_rows[i] = BallVelocityContactRow{};
    for (int i = 0; i < caps.ball_friction; ++i) rows.ball_friction_rows[i] = BallFrictionContactRow{};
    for (int i = 0; i < caps.pair_split; ++i) rows.pair_split_rows[i] = CarSharedSplitContactRow{};
    for (int i = 0; i < caps.world_split; ++i) rows.world_split_rows[i] = CarSharedSplitContactRow{};
    for (int i = 0; i < caps.pair_velocity; ++i) rows.pair_velocity_rows[i] = CarSharedVelocityContactRow{};
    for (int i = 0; i < caps.pair_friction; ++i) rows.pair_friction_rows[i] = CarSharedFrictionContactRow{};
    for (int i = 0; i < caps.world_velocity; ++i) rows.world_velocity_rows[i] = CarSharedVelocityContactRow{};
    for (int i = 0; i < caps.world_friction; ++i) rows.world_friction_rows[i] = CarSharedFrictionContactRow{};
    for (int i = 0; i < caps.order_groups; ++i) rows.shared_contact_order_groups[i] = SharedContactOrderGroup{};
    for (int i = 0; i < caps.manifolds; ++i) rows.bullet_manifolds[i] = BulletManifoldEntry{};
    if (rows.pair_contact_cache) {
        for (int i = 0; i < rows.pair_contact_cache.size(); ++i) rows.pair_contact_cache.slots[i] = CarCarPairCachedContact{};
    }
    for (int i = 0; rows.car_world_plane_manifolds && i < caps.car_world_plane; ++i) rows.car_world_plane_manifolds[i] = CarWorldPlaneRetainedManifold{};
    for (int i = 0; rows.car_world_mesh_manifolds && i < caps.car_world_mesh; ++i) rows.car_world_mesh_manifolds[i] = CarWorldMeshRetainedManifold{};
    for (int i = 0; rows.car_car_retained_manifolds && i < caps.car_car_retained; ++i) rows.car_car_retained_manifolds[i] = CarCarRetainedManifold{};
    for (int i = 0; rows.ball_car_retained_manifolds && i < caps.ball_car_retained; ++i) rows.ball_car_retained_manifolds[i] = BallCarRetainedManifold{};
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
    std::vector<CarCarPairCachedContact> pair_contact_cache;
    std::vector<CarWorldPlaneRetainedManifold> car_world_plane_manifolds;
    std::vector<CarWorldMeshRetainedManifold> car_world_mesh_manifolds;
    std::vector<CarCarRetainedManifold> car_car_retained_manifolds;
    std::vector<BallCarRetainedManifold> ball_car_retained_manifolds;
    std::vector<CarWorldLocalRows> car_world_local_rows;
    WorldSolverScratchRowPool pool;

    void reset(int env_count, bool interleaved, const WorldSolverScratchRowCaps& caps = WorldSolverScratchRowCaps{}) {
        const size_t n = static_cast<size_t>(env_count < 0 ? 0 : env_count);
        ball_split_rows.assign(n * caps.ball_split, BallSplitContactRow{});
        ball_velocity_rows.assign(n * caps.ball_velocity, BallVelocityContactRow{});
        ball_friction_rows.assign(n * caps.ball_friction, BallFrictionContactRow{});
        pair_split_rows.assign(n * caps.pair_split, CarSharedSplitContactRow{});
        world_split_rows.assign(n * caps.world_split, CarSharedSplitContactRow{});
        pair_velocity_rows.assign(n * caps.pair_velocity, CarSharedVelocityContactRow{});
        pair_friction_rows.assign(n * caps.pair_friction, CarSharedFrictionContactRow{});
        world_velocity_rows.assign(n * caps.world_velocity, CarSharedVelocityContactRow{});
        world_friction_rows.assign(n * caps.world_friction, CarSharedFrictionContactRow{});
        shared_contact_order_groups.assign(n * caps.order_groups, SharedContactOrderGroup{});
        bullet_manifolds.assign(n * caps.manifolds, BulletManifoldEntry{});
        pair_contact_cache.assign(n * caps.pair_cache_cars * caps.pair_cache_cars, CarCarPairCachedContact{});
        car_world_plane_manifolds.assign(n * caps.car_world_plane, CarWorldPlaneRetainedManifold{});
        car_world_mesh_manifolds.assign(n * caps.car_world_mesh, CarWorldMeshRetainedManifold{});
        car_car_retained_manifolds.assign(n * caps.car_car_retained, CarCarRetainedManifold{});
        ball_car_retained_manifolds.assign(n * caps.ball_car_retained, BallCarRetainedManifold{});
        car_world_local_rows.assign(n * caps.car_world_local, CarWorldLocalRows{});
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
        pool.pair_contact_cache = pair_contact_cache.data();
        pool.car_world_plane_manifolds = car_world_plane_manifolds.data();
        pool.car_world_mesh_manifolds = car_world_mesh_manifolds.data();
        pool.car_car_retained_manifolds = car_car_retained_manifolds.data();
        pool.ball_car_retained_manifolds = ball_car_retained_manifolds.data();
        pool.car_world_local_rows = car_world_local_rows.data();
        pool.env_count = env_count;
        pool.interleaved = interleaved ? 1 : 0;
        pool.caps = caps;
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

// WorldStepSoccar locals that cross from its contact stages (kWorldStepContacts) to the car-world rows and the
// solve (kWorldStepRows).
struct WorldStepCarry {
    int pair_split_row_count = 0;
    int pair_velocity_row_count = 0;
    int pair_friction_row_count = 0;
    int split_row_count = 0;
    int normal_row_count = 0;
    int friction_row_count = 0;
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

    // Bullet's bookkeeping carried across ticks (BuildBulletManifoldList): the overlapping pair cache and the
    // dispatcher's manifold array, as pair keys (island_id unused).
    uint8_t bp_pair_count = 0;
    uint8_t bp_manifold_count = 0;
    BulletManifoldEntry bp_pairs[kMaxBulletManifoldEntries] = {};
    BulletManifoldEntry bp_manifolds[kMaxBulletManifoldEntries] = {};
    // RL's btAxisSweep3 (BuildBulletManifoldList): live handle count (0 = not built), static handle count, the sorted
    // edges per axis (quantized position, owning handle; handle 0 is the sentinel) and each handle's edge indices.
    uint8_t sap_handle_count = 0;
    uint8_t sap_static_count = 0;
    uint16_t sap_edge_pos[3][kAxisSweepMaxEdges] = {};
    uint8_t sap_edge_handle[3][kAxisSweepMaxEdges] = {};
    uint8_t sap_min_edge[kAxisSweepMaxHandles + 1][3] = {};
    uint8_t sap_max_edge[kAxisSweepMaxHandles + 1][3] = {};
    int16_t sap_static_id[kAxisSweepMaxHandles + 1] = {};
    // Set by a kickoff reset: the next world step clears the retained manifolds in the row pool first.
    uint8_t retained_reset_pending = 0;

    int32_t bullet_manifold_count = 0;
    int32_t shared_contact_order_group_count = 0;

    WheelRayStage wheel_ray_stage[kWheelRayStageSlots] = {};
    WorldStepCarry world_step_carry = {};
    CarWorldRowsOut car_world_rows_out[8] = {};
    // Cars of the tick that last wrote pair_contact_cache (its valid slots are among theirs).
    uint8_t pair_cache_cars = 8;
    // Staged CUDA step: whether this tick's vehicle updates ran a thread per car (VehicleUpdatesIndependent).
    uint8_t vehicle_updates_independent = 0;
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

ROCKETGPU_HD bool IsBtStaticPlaneSourceTag(int source_tag) {
    return source_tag == 1 || source_tag == 2 || source_tag == 5 || source_tag == 6;
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
    grid.leaf_grid.infos = cfg.arena_mesh_leaf_grid_infos;
    grid.leaf_grid.cell_start = cfg.arena_mesh_leaf_cell_start;
    grid.leaf_grid.entries = cfg.arena_mesh_leaf_entries;
    return grid;
}

ROCKETGPU_HD bool BulletStaticPlaneProxyAabb(int plane_rank, ::rocketgpu::Vec3& out_min, ::rocketgpu::Vec3& out_max) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kLarge = 1.0e18f;
    // RL's broadphase AABB for its plane shape (used by addCollisionObject; arena statics are never
    // re-AABB'd: they are created with DISABLE_SIMULATION and RLPhysScene clears forceUpdateAllAabbs) is the plane's
    // rectangle swept 1e18 behind it, so nothing in front of the surface. The in-plane extents are treated as infinite.
    constexpr float kSlabBt = 0.0f;
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
    if (car.basis_authoritative && !native_sse_math) {
        GameIntegrateOrientationBasisColumns(pred_basis_fwd, pred_basis_right, pred_basis_up, predicted_rot, car.ang_vel, dt);
    } else if (car.basis_authoritative) {
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

    // A sleeping ball keeps its world manifolds in the list that drives solver ordering: Bullet keeps a sleeping
    // body's pairs and manifolds (see ResolveBallActivationAtTickStart).
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
        proxy.child_center = pos_bt;
        proxy.child_half = ::rocketgpu::Vec3(extent, extent, extent);
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
        if (car.basis_authoritative && !native_sse_math) {
            GameIntegrateOrientationBasisColumns(pred_basis_fwd, pred_basis_right, pred_basis_up, predicted_rot, car.ang_vel, dt);
        } else if (car.basis_authoritative) {
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
        proxy.child_center = center_now;
        proxy.child_axis[0] = axis_x;
        proxy.child_axis[1] = axis_y;
        proxy.child_axis[2] = axis_z;
        proxy.child_half = half_extent_bt;
        grid.cell_indices(proxy.aabb_min, proxy.cell_i, proxy.cell_j, proxy.cell_k);
        proxy.valid = 1;
        proxy.is_ball = 0;
    }

    const int body_count = Min(car_count + 1, kMaxBodies);
    if (bp_history == nullptr) {
        return 0;
    }
    WorldSolverScratch& h = *bp_history;

    // Pair keys use BulletManifoldEntry: kind 0 ball-static, 1 ball-car, 2 car-static, 3 car-car (body_a < body_b).
    const auto mesh_info_for = [&](int mesh_index) -> const ArenaMeshBvhMeshInfo* {
        for (int slot = 0; slot < mesh_info_count; ++slot) {
            if (mesh_infos[slot].mesh_index == mesh_index && mesh_infos[slot].triangle_count > 0) {
                return &mesh_infos[slot];
            }
        }
        return nullptr;
    };
    const auto body_proxy = [&](int body) -> const BulletBroadphaseProxy* {
        return body >= 0 && body < body_count && proxies[body].valid ? &proxies[body] : nullptr;
    };
    // Whether the narrowphase holds a manifold for a cached pair. The near callback does no AABB test of its own; a car's
    // compound child algorithm exists while the child AABB overlaps the other
    // shape's getAabb. A plane's getAabb is +-1e18, so a car-plane manifold lives as long as the pair.
    const auto overlap = [&](const BulletManifoldEntry& key) {
        const BulletBroadphaseProxy* pa = body_proxy(key.kind <= 1 ? 0 : key.body_a + 1);
        if (pa == nullptr) {
            return false;
        }
        if (key.kind == 1 || key.kind == 3) {
            const BulletBroadphaseProxy* pb = body_proxy(key.kind == 1 ? key.body_a + 1 : key.body_b + 1);
            return pb != nullptr &&
                   BulletAabbOverlap(pa->child_aabb_min, pa->child_aabb_max, pb->child_aabb_min, pb->child_aabb_max);
        }
        if (!arena_static) {
            return false;
        }
        if (key.static_id >= 100) {
            return true;
        }
        const ArenaMeshBvhMeshInfo* info = mesh_info_for(key.static_id);
        return info != nullptr &&
               BulletAabbOverlap(pa->child_aabb_min, pa->child_aabb_max, info->aabb_min, info->aabb_max);
    };
    // A car's compound child algorithm is made only for a child the compound's dbvt query returns, in the car's local
    // frame: the other shape's getAabb there (a ball: btCompoundCollisionAlgorithm's collideTV), or the other car's
    // child volume transformed into this car's frame as an AABB (car-car: MycollideTT, body0 = the lower handle).
    // The child world AABB test (overlap) then gates it; once
    // made it is dropped only when the child world AABBs separate (Wisp 11439 k42: the car-car manifold is made after
    // both ball-car ones, which sets the solver order of the three).
    const auto local_overlap = [&](const BulletManifoldEntry& key) {
        const BulletBroadphaseProxy* pa = body_proxy(key.body_a + 1);
        const BulletBroadphaseProxy* pb = body_proxy(key.kind == 1 ? 0 : key.body_b + 1);
        if (pa == nullptr || pb == nullptr) {
            return false;
        }
        const ::rocketgpu::Vec3 d = pb->child_center - pa->child_center;
        const float ha[3] = {pa->child_half.x, pa->child_half.y, pa->child_half.z};
        const float hb[3] = {pb->child_half.x, pb->child_half.y, pb->child_half.z};
        for (int k = 0; k < 3; ++k) {
            const ::rocketgpu::Vec3& ak = pa->child_axis[k];
            float e = hb[k];
            if (!pb->is_ball) {
                e = Abs(Dot(ak, pb->child_axis[0])) * hb[0] + Abs(Dot(ak, pb->child_axis[1])) * hb[1] +
                    Abs(Dot(ak, pb->child_axis[2])) * hb[2];
            }
            if (Abs(Dot(d, ak)) > ha[k] + e) {
                return false;
            }
        }
        return true;
    };
    const auto find = [](const BulletManifoldEntry* list, int n, const BulletManifoldEntry& key) {
        for (int i = 0; i < n; ++i) {
            if (list[i].kind == key.kind && list[i].body_a == key.body_a && list[i].body_b == key.body_b &&
                list[i].static_id == key.static_id) {
                return i;
            }
        }
        return -1;
    };
    const auto swap_remove = [](BulletManifoldEntry* list, uint8_t& n, int i) {
        if (i >= 0) {
            list[i] = list[--n];
        }
    };

    // RL's broadphase is btAxisSweep3<unsigned short>: RLPhysScene::RLPhysScene calls its ctor
    // with world AABB +-10485.74 bt, 0x4000 handles and a hashed pair
    // cache (v3 axis_sweep.rs). Handles are the arena meshes and 4 planes (raw shape AABBs), then the ball and
    // the cars, in creation order. Each body's setAabb moves its quantized edges; a min edge passing down over a
    // max edge (or a max edge passing up over a min edge) appends a pair when the other two axes overlap, and the
    // reverse moves remove it (swap-with-last, releasing its manifold the same way).
    constexpr float kSapHalfExtentBt = 10485.740234375f;
    const float sap_quantize = 65535.0f / (kSapHalfExtentBt + kSapHalfExtentBt);
    const auto quantize_point = [&](const ::rocketgpu::Vec3& p, uint16_t* out, int is_max) {
        const float v[3] = {p.x, p.y, p.z};
        for (int axis = 0; axis < 3; ++axis) {
            float scaled = (v[axis] + kSapHalfExtentBt) * sap_quantize;
            scaled = scaled >= 0.0f ? (scaled < 65535.0f ? scaled : 65535.0f) : 0.0f;
            out[axis] = static_cast<uint16_t>((static_cast<int>(scaled) & 0xfffe) | is_max);
        }
    };
    // Pair handles to a pair key: handles 1..sap_static_count are statics, then body 0 (ball), body 1.. (cars).
    const auto sap_key = [&](int ha, int hb, BulletManifoldEntry& key) {
        if (ha > hb) {
            const int t = ha;
            ha = hb;
            hb = t;
        }
        // Statics are added first, so a static hb means a static-static pair: needsBroadphaseCollision drops it
        // (static objects' filter mask excludes StaticFilter). sap_static_count is still 0 while they are added.
        if (h.sap_static_id[hb] >= 0) {
            return false;
        }
        const int statics = h.sap_static_count;
        const int body_b = hb - statics - 1;
        if (body_b < 0 || body_b >= body_count) {
            return false;
        }
        key = BulletManifoldEntry();
        if (ha <= statics) {
            key.kind = body_b == 0 ? 0 : 2;
            key.body_a = static_cast<int8_t>(body_b - 1);
            key.static_id = h.sap_static_id[ha];
        } else {
            const int body_a = ha - statics - 1;
            key.kind = body_a == 0 ? 1 : 3;
            key.body_a = static_cast<int8_t>(body_a == 0 ? body_b - 1 : body_a - 1);
            key.body_b = static_cast<int8_t>(body_a == 0 ? -1 : body_b - 1);
        }
        return true;
    };
    const auto sap_add_pair = [&](int ha, int hb) {
        BulletManifoldEntry key;
        if (sap_key(ha, hb, key) && find(h.bp_pairs, h.bp_pair_count, key) < 0 &&
            h.bp_pair_count < kMaxBulletManifoldEntries) {
            h.bp_pairs[h.bp_pair_count++] = key;
        }
    };
    const auto sap_remove_pair = [&](int ha, int hb) {
        BulletManifoldEntry key;
        if (!sap_key(ha, hb, key)) {
            return;
        }
        const int at = find(h.bp_pairs, h.bp_pair_count, key);
        if (at >= 0) {
            swap_remove(h.bp_pairs, h.bp_pair_count, at);
            swap_remove(h.bp_manifolds, h.bp_manifold_count, find(h.bp_manifolds, h.bp_manifold_count, key));
        }
    };
    // testOverlap2D: compares sorted edge indices on the two other axes; touching counts as overlap.
    const auto overlap_2d = [&](int a, int b, int axis) {
        const int axis1 = (1 << axis) & 3;
        const int axis2 = (1 << axis1) & 3;
        return !(h.sap_max_edge[a][axis1] < h.sap_min_edge[b][axis1] ||
                 h.sap_max_edge[b][axis1] < h.sap_min_edge[a][axis1] ||
                 h.sap_max_edge[a][axis2] < h.sap_min_edge[b][axis2] ||
                 h.sap_max_edge[b][axis2] < h.sap_min_edge[a][axis2]);
    };
    const auto swap_edges = [&](int axis, int i, int j) {
        const uint16_t pos = h.sap_edge_pos[axis][i];
        const uint8_t owner = h.sap_edge_handle[axis][i];
        h.sap_edge_pos[axis][i] = h.sap_edge_pos[axis][j];
        h.sap_edge_handle[axis][i] = h.sap_edge_handle[axis][j];
        h.sap_edge_pos[axis][j] = pos;
        h.sap_edge_handle[axis][j] = owner;
    };
    // sortMinDown / sortMaxDown walk down, sortMinUp / sortMaxUp walk up to the max sentinel (handle 0).
    const auto sort_down = [&](int axis, int index, bool is_max_edge, bool update) {
        const int moving = h.sap_edge_handle[axis][index];
        while (index > 0 && h.sap_edge_pos[axis][index] < h.sap_edge_pos[axis][index - 1]) {
            const int prev = h.sap_edge_handle[axis][index - 1];
            const bool prev_is_max = (h.sap_edge_pos[axis][index - 1] & 1) != 0;
            if (update && prev_is_max != is_max_edge && overlap_2d(moving, prev, axis)) {
                if (is_max_edge) {
                    sap_remove_pair(moving, prev);
                } else {
                    sap_add_pair(moving, prev);
                }
            }
            ++(prev_is_max ? h.sap_max_edge[prev][axis] : h.sap_min_edge[prev][axis]);
            --(is_max_edge ? h.sap_max_edge[moving][axis] : h.sap_min_edge[moving][axis]);
            swap_edges(axis, index, index - 1);
            --index;
        }
    };
    const auto sort_up = [&](int axis, int index, bool is_max_edge, bool update) {
        const int moving = h.sap_edge_handle[axis][index];
        const int len = 2 * h.sap_handle_count + 2;
        while (index + 1 < len && h.sap_edge_handle[axis][index + 1] != 0 &&
               h.sap_edge_pos[axis][index] >= h.sap_edge_pos[axis][index + 1]) {
            const int next = h.sap_edge_handle[axis][index + 1];
            const bool next_is_max = (h.sap_edge_pos[axis][index + 1] & 1) != 0;
            if (update && next_is_max != is_max_edge && overlap_2d(moving, next, axis)) {
                if (is_max_edge) {
                    sap_add_pair(moving, next);
                } else {
                    sap_remove_pair(moving, next);
                }
            }
            --(next_is_max ? h.sap_max_edge[next][axis] : h.sap_min_edge[next][axis]);
            ++(is_max_edge ? h.sap_max_edge[moving][axis] : h.sap_min_edge[moving][axis]);
            swap_edges(axis, index, index + 1);
            ++index;
        }
    };
    // addHandle: append the edges below the max sentinel, sort axes 0 and 1 down silently, axis 2 with pairs.
    const auto add_handle = [&](const ::rocketgpu::Vec3& aabb_min, const ::rocketgpu::Vec3& aabb_max, int16_t id) {
        uint16_t qmin[3], qmax[3];
        quantize_point(aabb_min, qmin, 0);
        quantize_point(aabb_max, qmax, 1);
        const int handle = ++h.sap_handle_count;
        const int limit = 2 * handle;
        h.sap_static_id[handle] = id;
        for (int axis = 0; axis < 3; ++axis) {
            h.sap_max_edge[0][axis] = static_cast<uint8_t>(h.sap_max_edge[0][axis] + 2);
            h.sap_edge_pos[axis][limit + 1] = h.sap_edge_pos[axis][limit - 1];
            h.sap_edge_handle[axis][limit + 1] = h.sap_edge_handle[axis][limit - 1];
            h.sap_edge_pos[axis][limit - 1] = qmin[axis];
            h.sap_edge_handle[axis][limit - 1] = static_cast<uint8_t>(handle);
            h.sap_edge_pos[axis][limit] = qmax[axis];
            h.sap_edge_handle[axis][limit] = static_cast<uint8_t>(handle);
            h.sap_min_edge[handle][axis] = static_cast<uint8_t>(limit - 1);
            h.sap_max_edge[handle][axis] = static_cast<uint8_t>(limit);
        }
        for (int axis = 0; axis < 3; ++axis) {
            sort_down(axis, h.sap_min_edge[handle][axis], false, axis == 2);
            sort_down(axis, h.sap_max_edge[handle][axis], true, axis == 2);
        }
    };
    if (h.sap_handle_count == 0) {
        for (int axis = 0; axis < 3; ++axis) {
            h.sap_edge_pos[axis][0] = 0;
            h.sap_edge_handle[axis][0] = 0;
            h.sap_edge_pos[axis][1] = 0xffff;
            h.sap_edge_handle[axis][1] = 0;
            h.sap_min_edge[0][axis] = 0;
            h.sap_max_edge[0][axis] = 1;
        }
        for (int slot = 0; arena_static && slot < mesh_info_count; ++slot) {
            const ArenaMeshBvhMeshInfo& info = mesh_infos[slot];
            if (info.mesh_index >= 0 && info.triangle_count > 0 &&
                h.sap_handle_count + 4 + kMaxBodies < kAxisSweepMaxHandles) {
                add_handle(info.aabb_min, info.aabb_max, static_cast<int16_t>(info.mesh_index));
            }
        }
        for (int plane_rank = 0; arena_static && plane_rank < 4; ++plane_rank) {
            ::rocketgpu::Vec3 shape_min(0.0f, 0.0f, 0.0f), shape_max(0.0f, 0.0f, 0.0f);
            BulletStaticPlaneShapeAabb(plane_rank, shape_min, shape_max);
            add_handle(shape_min, shape_max, static_cast<int16_t>(100 + plane_rank));
        }
        h.sap_static_count = h.sap_handle_count;
    }
    // updateAabbs: the ball, then the cars (a demoed car keeps its edges); a body seen first is added.
    for (int body = 0; body < body_count; ++body) {
        const BulletBroadphaseProxy& proxy = proxies[body];
        const int handle = h.sap_static_count + 1 + body;
        if (handle > h.sap_handle_count) {
            if (handle > kAxisSweepMaxHandles) {
                break;
            }
            const ::rocketgpu::Vec3 at = body > 0 ? cars[body - 1].pos_bt : ball.pos_bt;
            add_handle(proxy.valid ? proxy.aabb_min : at, proxy.valid ? proxy.aabb_max : at, -1);
            continue;
        }
        if (!proxy.valid) {
            continue;
        }
        uint16_t qmin[3], qmax[3];
        quantize_point(proxy.aabb_min, qmin, 0);
        quantize_point(proxy.aabb_max, qmax, 1);
        for (int axis = 0; axis < 3; ++axis) {
            const int emin = h.sap_min_edge[handle][axis];
            const int emax = h.sap_max_edge[handle][axis];
            const int dmin = static_cast<int>(qmin[axis]) - static_cast<int>(h.sap_edge_pos[axis][emin]);
            const int dmax = static_cast<int>(qmax[axis]) - static_cast<int>(h.sap_edge_pos[axis][emax]);
            if (dmin == 0 && dmax == 0) {
                continue;
            }
            h.sap_edge_pos[axis][emin] = qmin[axis];
            h.sap_edge_pos[axis][emax] = qmax[axis];
            if (dmin < 0) {
                sort_down(axis, emin, false, true);
            }
            if (dmax > 0) {
                sort_up(axis, emax, true, true);
            }
            if (dmin > 0) {
                sort_up(axis, emin, false, true);
            }
            if (dmax < 0) {
                sort_down(axis, emax, true, true);
            }
        }
    }

    // dispatchAllCollisionPairs in pair cache order: a manifold is appended when its algorithm first makes it and
    // swap-removed when a compound child algorithm is dropped.
    for (int i = 0; i < h.bp_pair_count; ++i) {
        const BulletManifoldEntry& key = h.bp_pairs[i];
        const int at = find(h.bp_manifolds, h.bp_manifold_count, key);
        // A ball-static algorithm makes its manifold when the pair is first dispatched (v3 collision_dispatcher.rs).
        const bool want = key.kind == 0 ? body_proxy(0) != nullptr : overlap(key);
        if (want && at < 0 && (key.kind == 1 || key.kind == 3) && !local_overlap(key)) {
            continue;
        }
        if (want && at < 0) {
            h.bp_manifolds[h.bp_manifold_count++] = key;
        } else if (!want) {
            swap_remove(h.bp_manifolds, h.bp_manifold_count, at);
        }
    }

    // Islands: btSimulationIslandManager::findUnions over the pair cache, proxy0 (lower unique id) uniting into proxy1.
    int parent[kMaxBodies];
    for (int i = 0; i < kMaxBodies; ++i) {
        parent[i] = i;
    }
    const auto find_root = [&](int idx) {
        while (parent[idx] != idx) {
            idx = parent[idx];
        }
        return idx;
    };
    for (int i = 0; i < h.bp_pair_count; ++i) {
        const BulletManifoldEntry& key = h.bp_pairs[i];
        if (key.kind == 1 || key.kind == 3) {
            const int ra = find_root(key.kind == 1 ? 0 : key.body_a + 1);
            const int rb = find_root(key.kind == 1 ? key.body_a + 1 : key.body_b + 1);
            if (ra != rb) {
                parent[ra] = rb;
            }
        }
    }
    const int entry_count = Min(static_cast<int>(h.bp_manifold_count), Min(max_entries, kMaxBulletManifoldEntries));
    for (int i = 0; i < entry_count; ++i) {
        BulletManifoldEntry entry = h.bp_manifolds[i];
        entry.island_id = find_root(entry.kind <= 1 ? 0 : entry.body_a + 1);
        out_entries[i] = entry;
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
        if (native_sse_math) {
            IntegrateOrientationBasisColumns(ball.basis_forward, ball.basis_right, ball.basis_up,
                ball.rot, ball_split.turn_velocity * BallSplitTurnErp(), dt, false, true);
        } else {
            GameIntegrateOrientationBasisColumns(ball.basis_forward, ball.basis_right, ball.basis_up,
                ball.rot, ball_split.turn_velocity * BallSplitTurnErp(), dt);
        }
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

// Partitions of disjoint ranges are independent, so taking the smaller side first gives the same order while
// bounding the pending ranges by log2(n) + 1.
template <typename Index, typename Key>
ROCKETGPU_HD void QuickSortSharedContactOrderIndices(
    Index* order,
    const Key* island_ids,
    int lo,
    int hi) {
    constexpr int kMaxPendingRanges = 32;
    int range_lo[kMaxPendingRanges];
    int range_hi[kMaxPendingRanges];
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
                const Index tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
                ++i;
                --j;
            }
        } while (i <= j);

        const bool right_first = cur_hi - i < j - cur_lo;  // push the larger side first, pop the smaller next
        for (int side = 0; side < 2; ++side) {
            const bool right = (side == 0) != right_first;
            const int range_start = right ? i : cur_lo;
            const int range_end = right ? cur_hi : j;
            if (range_start < range_end && depth < kMaxPendingRanges) {
                range_lo[depth] = range_start;
                range_hi[depth] = range_end;
                ++depth;
            }
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

    int16_t order[kMaxSharedContactOrderGroups];
    int16_t island_ids[kMaxSharedContactOrderGroups];  // car island ids (1..8) or body index + 1
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
        order[insert] = static_cast<int16_t>(group_idx);
        ++active_count;
    }
    if (active_count <= 0) {
        return -1;
    }
    for (int order_idx = 0; order_idx < active_count; ++order_idx) {
        const SharedContactOrderGroup& group = groups[order[order_idx]];
        island_ids[order[order_idx]] =
            static_cast<int16_t>(SharedContactOrderIslandIdForBody(car_island_id, group.body_a_local_idx));
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
