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
    return (a.x * b.x + a.y * b.y) + a.z * b.z;
}

struct ContactAccumulator {
    CarPtrT history_car = nullptr;
    PhysicsEventStream event_stream = {};
    uint8_t surface = 0;
#if defined(ROCKETGPU_USE_CUDA)
    PhysicsEvent event = {};
#else
    union {
        PhysicsEvent event;
    };
    ROCKETGPU_HD ContactAccumulator() {}
#endif
    uint8_t hit = 0;
    ::rocketgpu::Vec3 normal_sum = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    float max_penetration = 0.0f;
    ::rocketgpu::Vec3 primary_normal = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    ::rocketgpu::Vec3 latest_normal = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);

    ROCKETGPU_HD void add(const ::rocketgpu::Vec3& n, float penetration) {
        if (history_car) history_car->RecordWorldSurface(true);
        surface = 1;
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

::rocketgpu::Vec3 ApplyInverseBallInertiaWorld(float mass, float radius, const ::rocketgpu::Vec3& torque_world);
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
ROCKETGPU_HD void GetBallInertiaBasisColumns(const ::rocketgpu::BallState& ball, ::rocketgpu::Vec3& forward, ::rocketgpu::Vec3& right, ::rocketgpu::Vec3& up) {
    if (ball.inertia_basis_valid) {
        forward = ball.inertia_basis_forward;
        right = ball.inertia_basis_right;
        up = ball.inertia_basis_up;
        return;
    }
    GetBallBasisColumns(ball, forward, right, up);
}
void ApplyImpulseAtPoint(::rocketgpu::BallState& ball, float mass, float radius, const ::rocketgpu::Vec3& impulse_world, const ::rocketgpu::Vec3& rel_point_world);
::rocketgpu::Vec3 SupportPointOnCarHitboxLocal(const ::rocketgpu::CarConfig& cfg, const ::rocketgpu::Vec3& dir_local);
::rocketgpu::Vec3 BulletBoxHalfExtentWithMarginBt(const ::rocketgpu::CarConfig& cfg);
	::rocketgpu::Vec3 VelocityAtPoint(CarCRefT car, const ::rocketgpu::Vec3& rel_point_world);
	::rocketgpu::Vec3 VelocityAtPoint(CarCRefT car, const ::rocketgpu::Vec3& rel_point_world, const ::rocketgpu::Vec3& external_linear_delta, const ::rocketgpu::Vec3& external_angular_delta);
	::rocketgpu::Vec3 ApplyInverseBodyInertiaWorld(CarCRefT car, const ::rocketgpu::CarConfig& cfg, float mass, const ::rocketgpu::Vec3& torque_world);
	::rocketgpu::Vec3 ApplyInverseBodyInertiaWorldBt(CarCRefT car, const ::rocketgpu::CarConfig& cfg, float mass, const ::rocketgpu::Vec3& torque_bt);
	::rocketgpu::Vec3 ApplyCarWorldContactAngularComponent(CarCRefT car, const ::rocketgpu::CarConfig& cfg, float mass, const ::rocketgpu::Vec3& rel_contact, const ::rocketgpu::Vec3& normal);
	::rocketgpu::Vec3 ApplyCarWorldContactAngularComponentBt(CarCRefT car, const ::rocketgpu::CarConfig& cfg, float mass, const ::rocketgpu::Vec3& rel_contact_bt, const ::rocketgpu::Vec3& normal);
	::rocketgpu::Vec3 CarWorldContactAngularComponentBtNative(CarCRefT car, const ::rocketgpu::CarConfig& cfg, float mass, const ::rocketgpu::Vec3& rel_contact_bt, const ::rocketgpu::Vec3& normal);
	void ApplyImpulseAtPoint(CarRefT car, const ::rocketgpu::CarConfig& cfg, float mass, const ::rocketgpu::Vec3& impulse_world, const ::rocketgpu::Vec3& rel_point_world);
::rocketgpu::Vec3 CarBodyLocalPointBt(CarCRefT car, const ::rocketgpu::Vec3& point_world_bt);
::rocketgpu::Vec3 CarBodyWorldPointBt(CarCRefT car, const ::rocketgpu::Vec3& point_local_bt);
bool ComputeCarArenaTriangleContact(
    CarCRefT car,
    const ::rocketgpu::CarConfig& cfg,
    const ArenaMeshTriangle& tri,
    ::rocketgpu::Vec3& out_normal,
    float& out_distance_bt,
    ::rocketgpu::Vec3* out_point_on_a_bt = nullptr,
    ::rocketgpu::Vec3* out_point_on_b_bt = nullptr,
    bullet_port::btGjkEpaScratch* gjk_epa_scratch = nullptr,
    ::rocketgpu::Vec3* io_cached_axis_bt = nullptr,
    float tri_origin_y_bt = 0.0f);
struct SplitImpulseAccumulator;
void IntegrateSplitImpulseTransformBt(CarRefT car, const SplitImpulseAccumulator& split_bt, float dt, bool native_sse_math = false);

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
    const ::rocketgpu::Vec3 raw_half_extent = BulletHitboxSize(cfg) * 0.5f;
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
        const ::rocketgpu::Vec3 half_bt = (BulletHitboxSize(cfg) * kUUToBT) / 2.0f;
        const float min_half_bt = Min(half_bt.x, Min(half_bt.y, half_bt.z));
        const float safe_bt = 0.1f * min_half_bt;
        const float margin_bt = safe_bt < 0.04f ? safe_bt : 0.04f;
        return ::rocketgpu::Vec3(margin_bt, margin_bt, margin_bt);
    }
    return BallCarHitboxInset(cfg) * kUUToBT;
}

#if defined(__CUDA_ARCH__)
constexpr int kMaxBallSplitCars = 8;
#else
constexpr int kMaxBallSplitCars = 32;
#endif
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

struct RowUsedBits {
    uint64_t lo = 0;
    uint64_t hi = 0;
    ROCKETGPU_HD bool Test(int i) const { return (((i < 64 ? lo : hi) >> (i & 63)) & 1ull) != 0; }
    ROCKETGPU_HD void Set(int i) {
        const uint64_t bit = 1ull << (i & 63);
        if (i < 64) {
            lo |= bit;
        } else {
            hi |= bit;
        }
    }
};
static_assert(kMaxCarWorldSharedVelocityContactRows <= 128 && kMaxCarPairSharedVelocityContactRows <= 128 &&
                  kMaxCarWorldSharedFrictionContactRows <= 128 && kMaxCarPairSharedFrictionContactRows <= 128 &&
                  kMaxCarPairSplitContactRows <= 128 && kMaxBallVelocityContactRows <= 128 &&
                  kMaxBallFrictionContactRows <= 128 && kMaxBallSplitContactRows <= 128,
              "RowUsedBits covers every row list");

struct BallVelocityContactRow {
    uint8_t valid = 0;
    uint8_t predictive = 0;
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

struct alignas(16) CarSharedSplitContactRow {
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

struct alignas(16) CarSharedVelocityContactRow {
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

struct alignas(16) CarSharedFrictionContactRow {
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

struct CarWorldPlaneRetainedPoint {
    ::rocketgpu::Vec3 local_point_a_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 point_on_b_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    float distance_bt = 0.0f;
};

struct CarWorldPlaneRetainedManifold {
    uint8_t count = 0;
    CarWorldPlaneRetainedPoint points[4] = {};
};

struct CarWorldMeshRetainedPoint {
    ::rocketgpu::Vec3 local_point_a_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 local_point_b_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 normal = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    float distance_bt = 0.0f;
};

struct CarWorldMeshRetainedManifold {
    uint8_t count = 0;
    uint8_t mesh = 0;
    CarWorldMeshRetainedPoint points[4] = {};
};

struct CarCarRetainedPoint {
    ::rocketgpu::Vec3 local_a_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 local_b_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 pos_a_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 pos_b_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 normal = ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f);
    float distance_bt = 0.0f;
};

struct CarCarRetainedManifold {
    uint8_t count = 0;
    CarCarRetainedPoint points[4] = {};
};

constexpr int kMaxCarCarRetainedManifolds = 28;

ROCKETGPU_HD constexpr int CarCarRetainedSlot(int lo, int hi) {
    return hi * (hi - 1) / 2 + lo;
}

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
constexpr float kGoalFloorOriginYBt = 111.68f;
constexpr float kGoalFloorHalfXBt = 17.694719314575195f;
constexpr float kGoalFloorHalfYBt = 8.847359657287598f;
constexpr int kGoalFloorSourceTag = 12;

ROCKETGPU_HD ::rocketgpu::Vec3 GoalFloorOriginBt(int goal_floor) {
    return ::rocketgpu::Vec3(0.0f, goal_floor == 0 ? -kGoalFloorOriginYBt : kGoalFloorOriginYBt, 0.0f);
}

ROCKETGPU_HD bool GoalFloorExtentContainsBt(const ::rocketgpu::Vec3& vtx_in_plane_bt, float threshold_bt) {
    const float lim_x = threshold_bt + kGoalFloorHalfXBt;
    const float lim_y = threshold_bt + kGoalFloorHalfYBt;
    return !(vtx_in_plane_bt.x < -lim_x || vtx_in_plane_bt.y < -lim_y || lim_x < vtx_in_plane_bt.x ||
             lim_y < vtx_in_plane_bt.y);
}

constexpr int kCarWorldPlaneRetainedManifoldsPerCar = 6;
constexpr int kBallWorldPlaneSlots = 6;
constexpr int kMaxCarWorldPlaneRetainedManifolds =
    kCarWorldPlaneRetainedManifoldsPerCar * kMaxCarWorldMeshPersistentCars;
constexpr int kCarWorldMeshRetainedManifoldsPerCar = 2;
constexpr int kMaxCarWorldMeshRetainedManifolds =
    kCarWorldMeshRetainedManifoldsPerCar * kMaxCarWorldMeshPersistentCars;

struct CarWorldLocalRows {
    UninitArray<CarWorldSplitContactRow, kMaxCarWorldSplitContactRows> split_rows;
    UninitArray<CarWorldVelocityContactRow, kMaxCarWorldVelocityContactRows> normal_rows;
    UninitArray<CarWorldFrictionContactRow, kMaxCarWorldFrictionContactRows> friction_rows;
    UninitArray<SharedContactOrderGroup, kMaxCarWorldMeshesPerCar + 4> order_groups;
};

struct CarWorldRowsPart {
    int split[2];
    int normal[2];
    int friction[2];
    int groups[2];
    uint8_t hit[2];
    uint8_t surface[2];
    ::rocketgpu::Vec3 latest_normal[2];
};

struct BulletManifoldEntry {
    int8_t kind = 0;
    int8_t body_a = -1;
    int8_t body_b = -1;
    int16_t static_id = -1;
    int32_t island_id = 0;
};

struct CarListManifolds {
    uint32_t meshes = 0;
    uint32_t planes = 0;
};

struct BulletBroadphaseProxy {
    ::rocketgpu::Vec3 aabb_min = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 aabb_max = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 child_aabb_min = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 child_aabb_max = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 child_center = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 child_axis[3] = {::rocketgpu::Vec3(1.0f, 0.0f, 0.0f), ::rocketgpu::Vec3(0.0f, 1.0f, 0.0f),
                                       ::rocketgpu::Vec3(0.0f, 0.0f, 1.0f)};
    ::rocketgpu::Vec3 child_half = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
#if !defined(ROCKETGPU_USE_CUDA)
    ::rocketgpu::Vec3 child_local_min = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 child_local_max = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 origin = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
#endif
    int32_t cell_i = 0;
    int32_t cell_j = 0;
    int32_t cell_k = 0;
    uint8_t valid = 0;
    uint8_t is_ball = 0;
};

constexpr int kMaxBulletManifoldEntries = 96;
constexpr int kAxisSweepMaxHandles = 40;
constexpr int kAxisSweepMaxEdges = 2 * kAxisSweepMaxHandles + 2;

ROCKETGPU_HD constexpr uint32_t PackBulletPairKey(int kind, int body_a, int body_b, int static_id) {
    return (static_cast<uint32_t>(kind) & 3u) | ((static_cast<uint32_t>(body_a + 1) & 15u) << 2) |
           ((static_cast<uint32_t>(body_b + 1) & 15u) << 6) | ((static_cast<uint32_t>(static_id) & 0xffffu) << 16);
}
ROCKETGPU_HD constexpr int BulletPairKeyKind(uint32_t key) {
    return static_cast<int>(key & 3u);
}
ROCKETGPU_HD BulletManifoldEntry UnpackBulletPairKey(uint32_t key) {
    BulletManifoldEntry entry;
    entry.kind = static_cast<int8_t>(key & 3u);
    entry.body_a = static_cast<int8_t>(static_cast<int>((key >> 2) & 15u) - 1);
    entry.body_b = static_cast<int8_t>(static_cast<int>((key >> 6) & 15u) - 1);
    entry.static_id = static_cast<int16_t>(key >> 16);
    return entry;
}

constexpr int kBulletSweepMaxBodies = 9;
constexpr int kBulletPredictiveStaticId = 99;

struct alignas(16) BulletSweepHistory {
    uint8_t sap_handle_count = 0;
    uint8_t sap_static_count = 0;
    uint8_t sap_next_fresh = 0;
    uint8_t sap_free_count = 0;
    uint8_t sap_free[kBulletSweepMaxBodies] = {};
    uint8_t sap_body_handle[kBulletSweepMaxBodies] = {};
    uint8_t sap_obj_count = 0;
    uint8_t sap_obj[kBulletSweepMaxBodies] = {};
    uint16_t sap_edge_pos[3][kAxisSweepMaxEdges] = {};
    uint8_t sap_edge_handle[3][kAxisSweepMaxEdges] = {};
    uint8_t sap_min_edge[kAxisSweepMaxHandles + 1][3] = {};
    uint8_t sap_max_edge[kAxisSweepMaxHandles + 1][3] = {};
    int16_t sap_static_id[kAxisSweepMaxHandles + 1] = {};
    float sap_body_aabb[kBulletSweepMaxBodies][6] = {};
    uint16_t sap_body_aabb_valid = 0;
};

ROCKETGPU_HD void BulletSweepStoreBodyAabb(BulletSweepHistory& h, int body, const ::rocketgpu::Vec3& mn,
                                          const ::rocketgpu::Vec3& mx) {
    float* a = h.sap_body_aabb[body];
    a[0] = mn.x; a[1] = mn.y; a[2] = mn.z;
    a[3] = mx.x; a[4] = mx.y; a[5] = mx.z;
    h.sap_body_aabb_valid = static_cast<uint16_t>(h.sap_body_aabb_valid | (1u << body));
}

struct alignas(16) BulletBroadphaseHistory {
    BulletSweepHistory sweep;
    uint8_t bp_pair_count = 0;
    uint8_t bp_manifold_count = 0;
    uint32_t bp_pairs[kMaxBulletManifoldEntries] = {};
    uint32_t bp_manifolds[kMaxBulletManifoldEntries] = {};
    uint8_t bp_predictive_count = 0;
    uint32_t bp_predictive[kBulletSweepMaxBodies] = {};
};

ROCKETGPU_HD bool BulletPairKeyHasBody(uint32_t key, int body) {
    const int kind = static_cast<int>(key & 3u);
    const int body_a = static_cast<int>((key >> 2) & 15u);
    const int body_b = static_cast<int>((key >> 6) & 15u);
    if (kind == 0) {
        return body == 0;
    }
    if (kind == 1) {
        return body == 0 || body == body_a;
    }
    return body == body_a || (kind == 3 && body == body_b);
}

ROCKETGPU_HD void BulletSweepRemoveBody(BulletBroadphaseHistory& bp, int body) {
    BulletSweepHistory& h = bp.sweep;
    const int handle = h.sap_body_handle[body];
    if (handle == 0) {
        return;
    }
    for (int i = 0; i < bp.bp_pair_count; ++i) {
        const uint32_t key = bp.bp_pairs[i];
        if (!BulletPairKeyHasBody(key, body)) {
            continue;
        }
        for (int m = 0; m < bp.bp_manifold_count; ++m) {
            if (bp.bp_manifolds[m] == key) {
                bp.bp_manifolds[m] = bp.bp_manifolds[--bp.bp_manifold_count];
                break;
            }
        }
    }
    for (int i = 0; i < bp.bp_pair_count;) {
        if (BulletPairKeyHasBody(bp.bp_pairs[i], body)) {
            bp.bp_pairs[i] = bp.bp_pairs[--bp.bp_pair_count];
        } else {
            ++i;
        }
    }
    const int limit = 2 * h.sap_handle_count;
    for (int axis = 0; axis < 3; ++axis) {
        h.sap_max_edge[0][axis] = static_cast<uint8_t>(h.sap_max_edge[0][axis] - 2);
    }
    for (int axis = 0; axis < 3; ++axis) {
        for (int is_max = 1; is_max >= 0; --is_max) {
            int index = is_max ? h.sap_max_edge[handle][axis] : h.sap_min_edge[handle][axis];
            h.sap_edge_pos[axis][index] = 0xffff;
            while (h.sap_edge_handle[axis][index + 1] != 0 &&
                   h.sap_edge_pos[axis][index] >= h.sap_edge_pos[axis][index + 1]) {
                const int next = h.sap_edge_handle[axis][index + 1];
                const bool next_is_max = (h.sap_edge_pos[axis][index + 1] & 1) != 0;
                --(next_is_max ? h.sap_max_edge[next][axis] : h.sap_min_edge[next][axis]);
                ++(is_max ? h.sap_max_edge[handle][axis] : h.sap_min_edge[handle][axis]);
                const uint16_t pos = h.sap_edge_pos[axis][index];
                h.sap_edge_pos[axis][index] = h.sap_edge_pos[axis][index + 1];
                h.sap_edge_handle[axis][index] = h.sap_edge_handle[axis][index + 1];
                h.sap_edge_pos[axis][index + 1] = pos;
                h.sap_edge_handle[axis][index + 1] = static_cast<uint8_t>(handle);
                ++index;
            }
        }
        h.sap_edge_handle[axis][limit - 1] = 0;
        h.sap_edge_pos[axis][limit - 1] = 0xffff;
    }
    h.sap_free[h.sap_free_count++] = static_cast<uint8_t>(handle);
    h.sap_handle_count = static_cast<uint8_t>(h.sap_handle_count - 1);
    h.sap_body_handle[body] = 0;
    h.sap_body_aabb_valid = static_cast<uint16_t>(h.sap_body_aabb_valid & ~(1u << body));
    for (int i = 0; i < h.sap_obj_count; ++i) {
        if (h.sap_obj[i] == body) {
            h.sap_obj[i] = h.sap_obj[--h.sap_obj_count];
            break;
        }
    }
}

ROCKETGPU_HD void BulletSweepRemoveRoundBodies(BulletBroadphaseHistory& bp, int car_count) {
    BulletSweepRemoveBody(bp, 0);
    for (int slot = 0; slot < car_count; ++slot) {
        BulletSweepRemoveBody(bp, slot + 1);
    }
}

ROCKETGPU_HD void BulletSweepNewRound(BulletBroadphaseHistory& bp, int car_count, CarCSpanT cars) {
    if (bp.sweep.sap_handle_count == 0) {
        return;
    }
    const int n = car_count < kBulletSweepMaxBodies - 1 ? car_count : kBulletSweepMaxBodies - 1;
    for (int slot = 0; slot < n; ++slot) {
        if (cars[slot].is_demoed) {
            BulletSweepRemoveBody(bp, slot + 1);
        }
    }
    BulletSweepRemoveRoundBodies(bp, n);
}

template <typename T>
struct EnvStridedArray {
    T* base = nullptr;
#if defined(ROCKETGPU_USE_CUDA)
    int stride = 1;
#else
    static constexpr int stride = 1;
#endif

    EnvStridedArray() = default;
#if defined(ROCKETGPU_USE_CUDA)
    ROCKETGPU_HD EnvStridedArray(T* p) : base(p), stride(1) {}
    ROCKETGPU_HD EnvStridedArray(T* p, int s) : base(p), stride(s) {}
    ROCKETGPU_HD EnvStridedArray(std::nullptr_t) : base(nullptr), stride(1) {}
    template <typename U, typename = typename std::enable_if<std::is_convertible<U*, T*>::value>::type>
    ROCKETGPU_HD EnvStridedArray(const EnvStridedArray<U>& other) : base(other.base), stride(other.stride) {}
#else
    EnvStridedArray(T* p) : base(p) {}
    EnvStridedArray(std::nullptr_t) : base(nullptr) {}
    template <typename U, typename = typename std::enable_if<std::is_convertible<U*, T*>::value>::type>
    EnvStridedArray(const EnvStridedArray<U>& other) : base(other.base) {}
#endif

    ROCKETGPU_HD T& operator[](int i) const {
        return base[static_cast<std::ptrdiff_t>(i) * static_cast<std::ptrdiff_t>(stride)];
    }
    ROCKETGPU_HD explicit operator bool() const { return base != nullptr; }
    ROCKETGPU_HD bool operator==(std::nullptr_t) const { return base == nullptr; }
    ROCKETGPU_HD bool operator!=(std::nullptr_t) const { return base != nullptr; }
};

struct CarCarPairCacheView {
    EnvStridedArray<CarCarPairCachedContact> slots;
    int cars = 0;
    ROCKETGPU_HD explicit operator bool() const { return static_cast<bool>(slots); }
    ROCKETGPU_HD int size() const { return cars * cars; }
    ROCKETGPU_HD int index(int a, int b) const { return a < cars && b < cars ? a * cars + b : -1; }
};

struct WheelRayStage {
    ::rocketgpu::Vec3 contact_point_bt;
    ::rocketgpu::Vec3 contact_normal_ws;
    float wheel_trace_len_bt;
    float suspension_len_bt;
    float suspension_relative_velocity_bt;
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

struct PreContactCar {
    ::rocketgpu::Quat rot = ::rocketgpu::Quat(1.0f, 0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 vel = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 vel_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 ang_vel = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    uint8_t is_demoed = 0;
};

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
    bool ball_vehicle_woke = false;
    bool ball_awake_after_contact_rows = false;
    bool ball_settled_skip = false;
    bool ball_settled_verify = false;
#if defined(ROCKETGPU_USE_CUDA)
    uint16_t predictive_hit_mask = 0;
#endif
};

#if defined(ROCKETGPU_USE_CUDA)
ROCKETGPU_HD uint32_t CarryFloatBits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}
ROCKETGPU_HD float CarryBitsFloat(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}
ROCKETGPU_HD void StoreWorldStepCarry(uint4* words, int stride, const WorldStepCarry& c) {
    words[0] = make_uint4(static_cast<uint32_t>(c.pair_split_row_count), static_cast<uint32_t>(c.pair_velocity_row_count),
                          static_cast<uint32_t>(c.pair_friction_row_count), static_cast<uint32_t>(c.split_row_count));
    const uint32_t flags = (c.ball_sleeping ? 1u : 0u) | (c.ball_vehicle_woke ? 2u : 0u) |
                           (c.ball_awake_after_contact_rows ? 4u : 0u) | (c.ball_settled_skip ? 8u : 0u) |
                           (c.ball_settled_verify ? 16u : 0u);
    words[stride] = make_uint4(static_cast<uint32_t>(c.normal_row_count), static_cast<uint32_t>(c.friction_row_count),
                               static_cast<uint32_t>(c.sorted_manifold_count), flags);
    const ::rocketgpu::Vec3& v = c.ball_solver_linear_velocity_bt;
    words[2 * stride] = make_uint4(CarryFloatBits(v.x), CarryFloatBits(v.y), CarryFloatBits(v.z), c.predictive_hit_mask);
}
ROCKETGPU_HD WorldStepCarry LoadWorldStepCarry(const uint4* words, int stride) {
    const uint4 a = words[0], b = words[stride], v = words[2 * stride];
    WorldStepCarry c;
    c.pair_split_row_count = static_cast<int>(a.x);
    c.pair_velocity_row_count = static_cast<int>(a.y);
    c.pair_friction_row_count = static_cast<int>(a.z);
    c.split_row_count = static_cast<int>(a.w);
    c.normal_row_count = static_cast<int>(b.x);
    c.friction_row_count = static_cast<int>(b.y);
    c.sorted_manifold_count = static_cast<int>(b.z);
    c.ball_sleeping = (b.w & 1u) != 0;
    c.ball_vehicle_woke = (b.w & 2u) != 0;
    c.ball_awake_after_contact_rows = (b.w & 4u) != 0;
    c.ball_settled_skip = (b.w & 8u) != 0;
    c.ball_settled_verify = (b.w & 16u) != 0;
    c.ball_solver_linear_velocity_bt = ::rocketgpu::Vec3(CarryBitsFloat(v.x), CarryBitsFloat(v.y), CarryBitsFloat(v.z));
    c.predictive_hit_mask = static_cast<uint16_t>(v.w);
    return c;
}
#endif

struct CarWorldRowsOut {
    int split_row_count = 0;
    int velocity_row_count = 0;
    int friction_row_count = 0;
    int group_count = 0;
    uint8_t world_surface = 0;
    SharedContactOrderGroup groups[kMaxCarWorldMeshesPerCar + 4] = {};
};

#if defined(ROCKETGPU_USE_CUDA)
ROCKETGPU_HD uint64_t PackRowsOutCounts(int split, int velocity, int friction, int groups, uint8_t world_surface) {
    return static_cast<uint64_t>(split & 0xff) | static_cast<uint64_t>(velocity & 0xff) << 8 |
           static_cast<uint64_t>(friction & 0xff) << 16 | static_cast<uint64_t>(groups & 0xff) << 24 |
           static_cast<uint64_t>(world_surface) << 32;
}
#endif

struct TickCars {
#if defined(ROCKETGPU_USE_CUDA)
    static constexpr long long kPreContact = 0;
    static constexpr long long kV = static_cast<long long>(sizeof(::rocketgpu::Vec3));
    static constexpr long long kVec3 = kPreContact + static_cast<long long>(sizeof(PreContactCar));
    static constexpr long long kHitRot = kVec3 + 6 * kV;
    static constexpr long long kWheelRays = kHitRot + static_cast<long long>(sizeof(::rocketgpu::Quat));
    static constexpr long long kPredictValid = kWheelRays + 4 * static_cast<long long>(sizeof(WheelRayStage));
    static constexpr long long kRowsOut = kPredictValid + 1;
    static constexpr long long kBytesPerCar = kRowsOut + static_cast<long long>(sizeof(uint64_t));
    static constexpr long long kPlaneAlign = 32;
    static_assert(alignof(PreContactCar) <= kPlaneAlign && alignof(WheelRayStage) <= kPlaneAlign, "plane alignment");

    unsigned char* base = nullptr;
    long long plane_cars = 0;
    long long car_base = 0;

    template <typename T>
    ROCKETGPU_HD T* plane(long long offset, long long per_car = 1) const {
        return reinterpret_cast<T*>(base + plane_cars * offset) + per_car * car_base;
    }
    ROCKETGPU_HD PreContactCar* pre_contact_cars() const { return plane<PreContactCar>(kPreContact); }
    ROCKETGPU_HD ::rocketgpu::Vec3* deferred_linear_deltas() const { return plane<::rocketgpu::Vec3>(kVec3); }
    ROCKETGPU_HD ::rocketgpu::Vec3* deferred_angular_deltas() const { return plane<::rocketgpu::Vec3>(kVec3 + 1 * kV); }
    ROCKETGPU_HD ::rocketgpu::Vec3* cal_start_vel() const { return plane<::rocketgpu::Vec3>(kVec3 + 2 * kV); }
    ROCKETGPU_HD ::rocketgpu::Vec3* hit_car_pos() const { return plane<::rocketgpu::Vec3>(kVec3 + 3 * kV); }
    ROCKETGPU_HD ::rocketgpu::Vec3* car_predict_vel_bt() const { return plane<::rocketgpu::Vec3>(kVec3 + 4 * kV); }
    ROCKETGPU_HD ::rocketgpu::Vec3* car_predict_ang_vel() const { return plane<::rocketgpu::Vec3>(kVec3 + 5 * kV); }
    ROCKETGPU_HD ::rocketgpu::Quat* hit_car_rot() const { return plane<::rocketgpu::Quat>(kHitRot); }
    ROCKETGPU_HD WheelRayStage* wheel_ray_stage() const { return plane<WheelRayStage>(kWheelRays, 4); }
    ROCKETGPU_HD uint8_t* car_predict_valid() const { return plane<uint8_t>(kPredictValid); }
    ROCKETGPU_HD uint64_t* rows_out_counts() const { return plane<uint64_t>(kRowsOut); }

    ROCKETGPU_HD TickCars at(std::ptrdiff_t first_car) const {
        TickCars t = *this;
        if (base != nullptr) t.car_base += first_car;
        return t;
    }
#else
    PreContactCar* pre_contact_cars_ = nullptr;
    ::rocketgpu::Vec3* deferred_linear_deltas_ = nullptr;
    ::rocketgpu::Vec3* deferred_angular_deltas_ = nullptr;
    ::rocketgpu::Vec3* cal_start_vel_ = nullptr;
    ::rocketgpu::Vec3* hit_car_pos_ = nullptr;
    ::rocketgpu::Quat* hit_car_rot_ = nullptr;
    ::rocketgpu::Vec3* car_predict_vel_bt_ = nullptr;
    ::rocketgpu::Vec3* car_predict_ang_vel_ = nullptr;
    uint8_t* car_predict_valid_ = nullptr;
    WheelRayStage* wheel_ray_stage_ = nullptr;

    PreContactCar* pre_contact_cars() const { return pre_contact_cars_; }
    ::rocketgpu::Vec3* deferred_linear_deltas() const { return deferred_linear_deltas_; }
    ::rocketgpu::Vec3* deferred_angular_deltas() const { return deferred_angular_deltas_; }
    ::rocketgpu::Vec3* cal_start_vel() const { return cal_start_vel_; }
    ::rocketgpu::Vec3* hit_car_pos() const { return hit_car_pos_; }
    ::rocketgpu::Quat* hit_car_rot() const { return hit_car_rot_; }
    ::rocketgpu::Vec3* car_predict_vel_bt() const { return car_predict_vel_bt_; }
    ::rocketgpu::Vec3* car_predict_ang_vel() const { return car_predict_ang_vel_; }
    uint8_t* car_predict_valid() const { return car_predict_valid_; }
    WheelRayStage* wheel_ray_stage() const { return wheel_ray_stage_; }

    TickCars at(std::ptrdiff_t car_base) const {
        TickCars t;
        if (pre_contact_cars_ == nullptr) return t;
        t.pre_contact_cars_ = pre_contact_cars_ + car_base;
        t.deferred_linear_deltas_ = deferred_linear_deltas_ + car_base;
        t.deferred_angular_deltas_ = deferred_angular_deltas_ + car_base;
        t.cal_start_vel_ = cal_start_vel_ + car_base;
        t.hit_car_pos_ = hit_car_pos_ + car_base;
        t.hit_car_rot_ = hit_car_rot_ + car_base;
        t.car_predict_vel_bt_ = car_predict_vel_bt_ + car_base;
        t.car_predict_ang_vel_ = car_predict_ang_vel_ + car_base;
        t.car_predict_valid_ = car_predict_valid_ + car_base;
        t.wheel_ray_stage_ = wheel_ray_stage_ + 4 * car_base;
        return t;
    }
#endif
};

struct WorldTickScratch {
    static constexpr int kMaxHeatseekerWorldNormals = 16;
    ::rocketgpu::Vec3 hs_world_normals[kMaxHeatseekerWorldNormals];
    int hs_world_normal_count = 0;

    bullet_port::btGjkEpaScratch car_triangle_gjk_epa_scratch = {};
    bullet_port::btGjkEpaScratch ball_car_gjk_epa_scratch = {};

    ::rocketgpu::Vec3 cal_ball_start_vel = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 hit_ball_pos = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 ball_predict_vel_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    uint16_t predictive_hit_mask = 0;
    uint8_t predictive_staged = 0;
    ::rocketgpu::Vec3 predictive_normal[9] = {};
    float predictive_distance_bt[9] = {};

    uint8_t w3_prefix_done = 0;
    uint8_t w3_proxies_done = 0;
    uint8_t w3_info_done = 0;
    uint8_t w3_removed_count = 0;
    uint32_t w3_removed[kMaxBulletManifoldEntries] = {};
    BulletBroadphaseProxy w3_proxies[9];
    uint8_t vehicle_apart_mask = 0;

    int32_t bullet_manifold_count = 0;
    int32_t shared_contact_order_group_count = 0;

    alignas(16) uint8_t wheel_ray_ordered[kWheelRayStageSlots] = {};
    WorldStepCarry world_step_carry = {};
    CarWorldRowsOut car_world_rows_out[8] = {};
    uint8_t vehicle_updates_independent = 0;
};

struct WorldSolverScratchRows {
#if defined(ROCKETGPU_USE_CUDA)
    BallSplitContactRow* ball_split_rows_ = nullptr;
    BallVelocityContactRow* ball_velocity_rows_ = nullptr;
    BallFrictionContactRow* ball_friction_rows_ = nullptr;
    CarSharedSplitContactRow* pair_split_rows_ = nullptr;
    CarSharedSplitContactRow* world_split_rows_ = nullptr;
    CarSharedVelocityContactRow* pair_velocity_rows_ = nullptr;
    CarSharedFrictionContactRow* pair_friction_rows_ = nullptr;
    CarSharedVelocityContactRow* world_velocity_rows_ = nullptr;
    CarSharedFrictionContactRow* world_friction_rows_ = nullptr;
    SharedContactOrderGroup* shared_contact_order_groups_ = nullptr;
    BulletManifoldEntry* bullet_manifolds_ = nullptr;
    int row_stride = 1;
    ROCKETGPU_HD EnvStridedArray<BallSplitContactRow> ball_split_rows() const { return EnvStridedArray<BallSplitContactRow>(ball_split_rows_, row_stride); }
    ROCKETGPU_HD EnvStridedArray<BallVelocityContactRow> ball_velocity_rows() const { return EnvStridedArray<BallVelocityContactRow>(ball_velocity_rows_, row_stride); }
    ROCKETGPU_HD EnvStridedArray<BallFrictionContactRow> ball_friction_rows() const { return EnvStridedArray<BallFrictionContactRow>(ball_friction_rows_, row_stride); }
    ROCKETGPU_HD EnvStridedArray<CarSharedSplitContactRow> pair_split_rows() const { return EnvStridedArray<CarSharedSplitContactRow>(pair_split_rows_, row_stride); }
    ROCKETGPU_HD EnvStridedArray<CarSharedSplitContactRow> world_split_rows() const { return EnvStridedArray<CarSharedSplitContactRow>(world_split_rows_, row_stride); }
    ROCKETGPU_HD EnvStridedArray<CarSharedVelocityContactRow> pair_velocity_rows() const { return EnvStridedArray<CarSharedVelocityContactRow>(pair_velocity_rows_, row_stride); }
    ROCKETGPU_HD EnvStridedArray<CarSharedFrictionContactRow> pair_friction_rows() const { return EnvStridedArray<CarSharedFrictionContactRow>(pair_friction_rows_, row_stride); }
    ROCKETGPU_HD EnvStridedArray<CarSharedVelocityContactRow> world_velocity_rows() const { return EnvStridedArray<CarSharedVelocityContactRow>(world_velocity_rows_, row_stride); }
    ROCKETGPU_HD EnvStridedArray<CarSharedFrictionContactRow> world_friction_rows() const { return EnvStridedArray<CarSharedFrictionContactRow>(world_friction_rows_, row_stride); }
    ROCKETGPU_HD EnvStridedArray<SharedContactOrderGroup> shared_contact_order_groups() const { return EnvStridedArray<SharedContactOrderGroup>(shared_contact_order_groups_, row_stride); }
    ROCKETGPU_HD EnvStridedArray<BulletManifoldEntry> bullet_manifolds() const { return EnvStridedArray<BulletManifoldEntry>(bullet_manifolds_, row_stride); }
#else
    EnvStridedArray<BallSplitContactRow> ball_split_rows_;
    EnvStridedArray<BallVelocityContactRow> ball_velocity_rows_;
    EnvStridedArray<BallFrictionContactRow> ball_friction_rows_;
    EnvStridedArray<CarSharedSplitContactRow> pair_split_rows_;
    EnvStridedArray<CarSharedSplitContactRow> world_split_rows_;
    EnvStridedArray<CarSharedVelocityContactRow> pair_velocity_rows_;
    EnvStridedArray<CarSharedFrictionContactRow> pair_friction_rows_;
    EnvStridedArray<CarSharedVelocityContactRow> world_velocity_rows_;
    EnvStridedArray<CarSharedFrictionContactRow> world_friction_rows_;
    EnvStridedArray<SharedContactOrderGroup> shared_contact_order_groups_;
    EnvStridedArray<BulletManifoldEntry> bullet_manifolds_;
    EnvStridedArray<BallSplitContactRow> ball_split_rows() const { return ball_split_rows_; }
    EnvStridedArray<BallVelocityContactRow> ball_velocity_rows() const { return ball_velocity_rows_; }
    EnvStridedArray<BallFrictionContactRow> ball_friction_rows() const { return ball_friction_rows_; }
    EnvStridedArray<CarSharedSplitContactRow> pair_split_rows() const { return pair_split_rows_; }
    EnvStridedArray<CarSharedSplitContactRow> world_split_rows() const { return world_split_rows_; }
    EnvStridedArray<CarSharedVelocityContactRow> pair_velocity_rows() const { return pair_velocity_rows_; }
    EnvStridedArray<CarSharedFrictionContactRow> pair_friction_rows() const { return pair_friction_rows_; }
    EnvStridedArray<CarSharedVelocityContactRow> world_velocity_rows() const { return world_velocity_rows_; }
    EnvStridedArray<CarSharedFrictionContactRow> world_friction_rows() const { return world_friction_rows_; }
    EnvStridedArray<SharedContactOrderGroup> shared_contact_order_groups() const { return shared_contact_order_groups_; }
    EnvStridedArray<BulletManifoldEntry> bullet_manifolds() const { return bullet_manifolds_; }
#endif
    CarCarPairCacheView pair_contact_cache;
    CarWorldPlaneRetainedManifold* car_world_plane_manifolds = nullptr;
    CarWorldMeshRetainedManifold* car_world_mesh_manifolds = nullptr;
    CarCarRetainedManifold* car_car_retained_manifolds = nullptr;
    BallCarRetainedManifold* ball_car_retained_manifolds = nullptr;
    CarWorldLocalRows* car_world_local_rows = nullptr;
    WorldTickScratch* tick = nullptr;
    TickCars tick_cars;
#if defined(ROCKETGPU_USE_CUDA)
    uint4* carry_words = nullptr;
    int carry_stride = 0;
#endif
    int car_world_plane_count = 0;
    int car_world_mesh_count = 0;
    int car_car_retained_count = 0;
    int ball_car_retained_count = 0;
};

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
    int pair_cache_cars = 8;
    int car_world_plane = kMaxCarWorldPlaneRetainedManifolds;
    int car_world_mesh = kMaxCarWorldMeshRetainedManifolds;
    int car_car_retained = kMaxCarCarRetainedManifolds;
    int ball_car_retained = kMaxBallCarRetainedManifolds;
    int car_world_local = kMaxCarWorldMeshPersistentCars;
};

constexpr int kMaxCarPairRowsPerPair = 4;

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
    caps.order_groups = CapRows(kMaxSharedContactOrderGroups, cars * (kMaxCarWorldMeshesPerCar + 4) + cars * cars);
    return caps;
}

enum TickRowArray : int {
    kTickBallSplit, kTickBallVelocity, kTickBallFriction, kTickPairSplit, kTickWorldSplit, kTickPairVelocity,
    kTickPairFriction, kTickWorldVelocity, kTickWorldFriction, kTickOrderGroups, kTickManifolds, kTickPairCache,
    kTickCarWorldLocal, kTickRowArrays
};
constexpr int kRowGroups = 9;

ROCKETGPU_HD int TickRowsPerEnv(const WorldSolverScratchRowCaps& c, int a) {
    switch (a) {
        case kTickBallSplit: return c.ball_split;
        case kTickBallVelocity: return c.ball_velocity;
        case kTickBallFriction: return c.ball_friction;
        case kTickPairSplit: return c.pair_split;
        case kTickWorldSplit: return c.world_split;
        case kTickPairVelocity: return c.pair_velocity;
        case kTickPairFriction: return c.pair_friction;
        case kTickWorldVelocity: return c.world_velocity;
        case kTickWorldFriction: return c.world_friction;
        case kTickOrderGroups: return c.order_groups;
        case kTickManifolds: return c.manifolds;
        case kTickPairCache: return c.pair_cache_cars * c.pair_cache_cars;
        default: return c.car_world_local;
    }
}

struct WorldSolverRowGroup {
    int32_t n = 0;
    WorldSolverScratchRowCaps caps;
    int64_t first[kTickRowArrays] = {};
#if defined(ROCKETGPU_USE_CUDA)
    void* base[kTickRowArrays] = {};
    int32_t step_bytes[kTickRowArrays] = {};
#endif
};

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
    WorldTickScratch* tick_scratch = nullptr;
    TickCars car_planes;
#if defined(ROCKETGPU_USE_CUDA)
    uint4* carry_words = nullptr;
#endif
    int cars_per_env = 0;
#if !defined(ROCKETGPU_USE_CUDA)
    std::ptrdiff_t car_tick_env_bytes = 0;
#endif
    const uint32_t* env_row_slot = nullptr;
#if defined(ROCKETGPU_USE_CUDA)
    WorldSolverRowGroup row_group_table[kRowGroups];
    uint8_t has_row_groups = 0;
    ROCKETGPU_HD const WorldSolverRowGroup* groups() const { return has_row_groups ? row_group_table : nullptr; }
#else
    const WorldSolverRowGroup* row_groups = nullptr;
#endif
    int env_count = 0;
    uint8_t interleaved = 0;
    WorldSolverScratchRowCaps caps;
#if defined(ROCKETGPU_USE_CUDA)
    uint64_t* pad_work = nullptr;
#endif

#if defined(ROCKETGPU_USE_CUDA)
    void* tick_array_at(int a, int64_t first) const {
        const auto at = [first](auto* p) -> void* { return p == nullptr ? nullptr : p + first; };
        switch (a) {
            case kTickBallSplit: return at(ball_split_rows);
            case kTickBallVelocity: return at(ball_velocity_rows);
            case kTickBallFriction: return at(ball_friction_rows);
            case kTickPairSplit: return at(pair_split_rows);
            case kTickWorldSplit: return at(world_split_rows);
            case kTickPairVelocity: return at(pair_velocity_rows);
            case kTickPairFriction: return at(pair_friction_rows);
            case kTickWorldVelocity: return at(world_velocity_rows);
            case kTickWorldFriction: return at(world_friction_rows);
            case kTickOrderGroups: return at(shared_contact_order_groups);
            case kTickManifolds: return at(bullet_manifolds);
            case kTickPairCache: return at(pair_contact_cache);
            default: return at(car_world_local_rows);
        }
    }
#endif

    template <typename T>
    ROCKETGPU_HD EnvStridedArray<T> view_one(T* base, int max_rows, int env_idx, int envs = -1) const {
        if (base == nullptr) {
            return EnvStridedArray<T>(nullptr);
        }
#if defined(ROCKETGPU_USE_CUDA)
        if (interleaved) {
            return EnvStridedArray<T>(base + env_idx, envs < 0 ? env_count : envs);
        }
        return EnvStridedArray<T>(base + static_cast<std::ptrdiff_t>(env_idx) * max_rows, 1);
#else
        return EnvStridedArray<T>(base + static_cast<std::ptrdiff_t>(env_idx) * max_rows);
#endif
    }

    ROCKETGPU_HD WorldTickScratch* tick_at(int env_idx) const {
        return tick_scratch == nullptr ? nullptr : tick_scratch + env_idx;
    }

    ROCKETGPU_HD TickCars tick_cars_at(int env_idx) const {
#if defined(ROCKETGPU_USE_CUDA)
        return car_planes.at(static_cast<std::ptrdiff_t>(env_idx) * cars_per_env);
#else
        TickCars t;
        if (car_planes.pre_contact_cars_ == nullptr) return t;
        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(env_idx) * car_tick_env_bytes;
        const auto at = [off](auto* q) { return reinterpret_cast<decltype(q)>(reinterpret_cast<unsigned char*>(q) + off); };
        t.pre_contact_cars_ = at(car_planes.pre_contact_cars_);
        t.deferred_linear_deltas_ = at(car_planes.deferred_linear_deltas_);
        t.deferred_angular_deltas_ = at(car_planes.deferred_angular_deltas_);
        t.cal_start_vel_ = at(car_planes.cal_start_vel_);
        t.hit_car_pos_ = at(car_planes.hit_car_pos_);
        t.hit_car_rot_ = at(car_planes.hit_car_rot_);
        t.car_predict_vel_bt_ = at(car_planes.car_predict_vel_bt_);
        t.car_predict_ang_vel_ = at(car_planes.car_predict_ang_vel_);
        t.car_predict_valid_ = at(car_planes.car_predict_valid_);
        t.wheel_ray_stage_ = at(car_planes.wheel_ray_stage_);
        return t;
#endif
    }

    ROCKETGPU_HD WorldSolverScratchRowCaps env_caps(int env_idx) const {
#if defined(ROCKETGPU_USE_CUDA)
        const WorldSolverRowGroup* const row_groups = groups();
#endif
        WorldSolverScratchRowCaps c = caps;
        if (row_groups != nullptr) {
            const WorldSolverScratchRowCaps& t = row_groups[env_row_slot[env_idx] >> 24].caps;
            c.ball_split = t.ball_split;
            c.ball_velocity = t.ball_velocity;
            c.ball_friction = t.ball_friction;
            c.pair_split = t.pair_split;
            c.world_split = t.world_split;
            c.pair_velocity = t.pair_velocity;
            c.pair_friction = t.pair_friction;
            c.world_velocity = t.world_velocity;
            c.world_friction = t.world_friction;
            c.order_groups = t.order_groups;
            c.manifolds = t.manifolds;
            c.pair_cache_cars = t.pair_cache_cars;
            c.car_world_local = t.car_world_local;
        }
        return c;
    }

    ROCKETGPU_HD WorldSolverScratchRows view(int env_idx) const {
#if defined(ROCKETGPU_USE_CUDA)
        const WorldSolverRowGroup* const row_groups = groups();
#endif
        WorldSolverScratchRows rows;
        rows.tick = tick_at(env_idx);
        rows.tick_cars = tick_cars_at(env_idx);
#if defined(ROCKETGPU_USE_CUDA)
        rows.carry_words = carry_words == nullptr ? nullptr : carry_words + env_idx;
        rows.carry_stride = env_count;
#if defined(__CUDA_ARCH__)
        (void)row_groups;
        {
#else
        if (row_groups != nullptr) {
#endif
            const uint32_t slot = env_row_slot[env_idx];
            const int k = static_cast<int>(slot >> 24);
            const int i = static_cast<int>(slot & 0xFFFFFFu);
            const WorldSolverRowGroup& g = row_group_table[k];
            const auto at = [&](auto* base, int a) {
                using T = typename std::remove_pointer<decltype(base)>::type;
                return reinterpret_cast<T*>(static_cast<char*>(g.base[a]) + static_cast<std::ptrdiff_t>(i) * g.step_bytes[a]);
            };
            rows.row_stride = interleaved ? g.n : 1;
            rows.ball_split_rows_ = at(ball_split_rows, kTickBallSplit);
            rows.ball_velocity_rows_ = at(ball_velocity_rows, kTickBallVelocity);
            rows.ball_friction_rows_ = at(ball_friction_rows, kTickBallFriction);
            rows.pair_split_rows_ = at(pair_split_rows, kTickPairSplit);
            rows.world_split_rows_ = at(world_split_rows, kTickWorldSplit);
            rows.pair_velocity_rows_ = at(pair_velocity_rows, kTickPairVelocity);
            rows.pair_friction_rows_ = at(pair_friction_rows, kTickPairFriction);
            rows.world_velocity_rows_ = at(world_velocity_rows, kTickWorldVelocity);
            rows.world_friction_rows_ = at(world_friction_rows, kTickWorldFriction);
            rows.shared_contact_order_groups_ = at(shared_contact_order_groups, kTickOrderGroups);
            rows.bullet_manifolds_ = at(bullet_manifolds, kTickManifolds);
            rows.pair_contact_cache.slots =
                EnvStridedArray<CarCarPairCachedContact>(at(pair_contact_cache, kTickPairCache), rows.row_stride);
            rows.pair_contact_cache.cars = g.caps.pair_cache_cars;
            rows.car_world_local_rows = at(car_world_local_rows, kTickCarWorldLocal);
            const auto contiguous = [env_idx](auto* base, int per_env) {
                return base == nullptr ? base : base + static_cast<std::ptrdiff_t>(env_idx) * per_env;
            };
            rows.car_world_plane_manifolds = contiguous(car_world_plane_manifolds, caps.car_world_plane);
            rows.car_world_mesh_manifolds = contiguous(car_world_mesh_manifolds, caps.car_world_mesh);
            rows.car_car_retained_manifolds = contiguous(car_car_retained_manifolds, caps.car_car_retained);
            rows.ball_car_retained_manifolds = contiguous(ball_car_retained_manifolds, caps.ball_car_retained);
            rows.car_world_plane_count = caps.car_world_plane;
            rows.car_world_mesh_count = caps.car_world_mesh;
            rows.car_car_retained_count = caps.car_car_retained;
            rows.ball_car_retained_count = caps.ball_car_retained;
            return rows;
        }
#endif
        int i = env_idx;
        int envs = env_count;
#if defined(ROCKETGPU_USE_CUDA)
        void* const* first = nullptr;
#else
        const int64_t* first = nullptr;
#endif
        const WorldSolverScratchRowCaps* tc = &caps;
        if (row_groups != nullptr) {
            const uint32_t slot = env_row_slot[env_idx];
            const WorldSolverRowGroup& g = row_groups[slot >> 24];
            i = static_cast<int>(slot & 0xFFFFFFu);
            envs = g.n;
#if defined(ROCKETGPU_USE_CUDA)
            first = g.base;
#else
            first = g.first;
#endif
            tc = &g.caps;
        }
#if defined(ROCKETGPU_USE_CUDA)
        const auto tick_view = [&](auto* base, int a, int rows_per_env) {
            return view_one(first == nullptr ? base : static_cast<decltype(base)>(first[a]), rows_per_env, i, envs);
        };
        const auto tick_one = [&](auto* base, int a, int rows_per_env) { return tick_view(base, a, rows_per_env).base; };
        rows.row_stride = interleaved ? envs : 1;
#else
        const auto tick_one = [&](auto* base, int a, int rows_per_env) {
            return view_one(base == nullptr || first == nullptr ? base : base + first[a], rows_per_env, i, envs);
        };
#endif
        rows.ball_split_rows_ = tick_one(ball_split_rows, kTickBallSplit, tc->ball_split);
        rows.ball_velocity_rows_ = tick_one(ball_velocity_rows, kTickBallVelocity, tc->ball_velocity);
        rows.ball_friction_rows_ = tick_one(ball_friction_rows, kTickBallFriction, tc->ball_friction);
        rows.pair_split_rows_ = tick_one(pair_split_rows, kTickPairSplit, tc->pair_split);
        rows.world_split_rows_ = tick_one(world_split_rows, kTickWorldSplit, tc->world_split);
        rows.pair_velocity_rows_ = tick_one(pair_velocity_rows, kTickPairVelocity, tc->pair_velocity);
        rows.pair_friction_rows_ = tick_one(pair_friction_rows, kTickPairFriction, tc->pair_friction);
        rows.world_velocity_rows_ = tick_one(world_velocity_rows, kTickWorldVelocity, tc->world_velocity);
        rows.world_friction_rows_ = tick_one(world_friction_rows, kTickWorldFriction, tc->world_friction);
        rows.shared_contact_order_groups_ = tick_one(shared_contact_order_groups, kTickOrderGroups, tc->order_groups);
        rows.bullet_manifolds_ = tick_one(bullet_manifolds, kTickManifolds, tc->manifolds);
#if defined(ROCKETGPU_USE_CUDA)
        rows.pair_contact_cache.slots =
            tick_view(pair_contact_cache, kTickPairCache, tc->pair_cache_cars * tc->pair_cache_cars);
#else
        rows.pair_contact_cache.slots =
            tick_one(pair_contact_cache, kTickPairCache, tc->pair_cache_cars * tc->pair_cache_cars);
#endif
        rows.pair_contact_cache.cars = tc->pair_cache_cars;
#if defined(ROCKETGPU_USE_CUDA)
        CarWorldLocalRows* const local =
            first == nullptr ? car_world_local_rows : static_cast<CarWorldLocalRows*>(first[kTickCarWorldLocal]);
#else
        CarWorldLocalRows* const local = car_world_local_rows == nullptr || first == nullptr
                                             ? car_world_local_rows : car_world_local_rows + first[kTickCarWorldLocal];
#endif
        rows.car_world_local_rows = local == nullptr ? nullptr : local + static_cast<std::ptrdiff_t>(i) * tc->car_world_local;
        const auto contiguous = [env_idx](auto* base, int per_env) {
            return base == nullptr ? base : base + static_cast<std::ptrdiff_t>(env_idx) * per_env;
        };
        rows.car_world_plane_manifolds = contiguous(car_world_plane_manifolds, caps.car_world_plane);
        rows.car_world_mesh_manifolds = contiguous(car_world_mesh_manifolds, caps.car_world_mesh);
        rows.car_car_retained_manifolds = contiguous(car_car_retained_manifolds, caps.car_car_retained);
        rows.ball_car_retained_manifolds = contiguous(ball_car_retained_manifolds, caps.ball_car_retained);
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

ROCKETGPU_HD void ClearRetainedContactManifolds(const WorldSolverScratchRows& rows, bool car_world = true) {
    for (int i = 0; car_world && rows.car_world_plane_manifolds && i < rows.car_world_plane_count; ++i) rows.car_world_plane_manifolds[i].count = 0;
    for (int i = 0; car_world && rows.car_world_mesh_manifolds && i < rows.car_world_mesh_count; ++i) rows.car_world_mesh_manifolds[i].count = 0;
    for (int i = 0; rows.car_car_retained_manifolds && i < rows.car_car_retained_count; ++i) rows.car_car_retained_manifolds[i].count = 0;
    for (int i = 0; rows.ball_car_retained_manifolds && i < rows.ball_car_retained_count; ++i) rows.ball_car_retained_manifolds[i].count = 0;
}

ROCKETGPU_HD void ZeroTickRows(const WorldSolverScratchRows& rows, const WorldSolverScratchRowCaps& caps, int first = 0,
                               int stride = 1) {
    for (int i = first; i < caps.ball_split; i += stride) rows.ball_split_rows()[i] = BallSplitContactRow{};
    for (int i = first; i < caps.ball_velocity; i += stride) rows.ball_velocity_rows()[i] = BallVelocityContactRow{};
    for (int i = first; i < caps.ball_friction; i += stride) rows.ball_friction_rows()[i] = BallFrictionContactRow{};
    for (int i = first; i < caps.pair_split; i += stride) rows.pair_split_rows()[i] = CarSharedSplitContactRow{};
    for (int i = first; i < caps.world_split; i += stride) rows.world_split_rows()[i] = CarSharedSplitContactRow{};
    for (int i = first; i < caps.pair_velocity; i += stride) rows.pair_velocity_rows()[i] = CarSharedVelocityContactRow{};
    for (int i = first; i < caps.pair_friction; i += stride) rows.pair_friction_rows()[i] = CarSharedFrictionContactRow{};
    for (int i = first; i < caps.world_velocity; i += stride) rows.world_velocity_rows()[i] = CarSharedVelocityContactRow{};
    for (int i = first; i < caps.world_friction; i += stride) rows.world_friction_rows()[i] = CarSharedFrictionContactRow{};
    for (int i = first; i < caps.order_groups; i += stride) rows.shared_contact_order_groups()[i] = SharedContactOrderGroup{};
    for (int i = first; i < caps.manifolds; i += stride) rows.bullet_manifolds()[i] = BulletManifoldEntry{};
    if (rows.pair_contact_cache) {
        for (int i = first; i < rows.pair_contact_cache.size(); i += stride) rows.pair_contact_cache.slots[i] = CarCarPairCachedContact{};
    }
}

ROCKETGPU_HD void ZeroWorldSolverScratchRows(const WorldSolverScratchRows& rows, const WorldSolverScratchRowCaps& caps) {
    if (!rows.ball_split_rows()) {
        return;
    }
    ZeroTickRows(rows, caps);
    for (int i = 0; rows.car_world_plane_manifolds && i < caps.car_world_plane; ++i) rows.car_world_plane_manifolds[i] = CarWorldPlaneRetainedManifold{};
    for (int i = 0; rows.car_world_mesh_manifolds && i < caps.car_world_mesh; ++i) rows.car_world_mesh_manifolds[i] = CarWorldMeshRetainedManifold{};
    for (int i = 0; rows.car_car_retained_manifolds && i < caps.car_car_retained; ++i) rows.car_car_retained_manifolds[i] = CarCarRetainedManifold{};
    for (int i = 0; rows.ball_car_retained_manifolds && i < caps.ball_car_retained; ++i) rows.ball_car_retained_manifolds[i] = BallCarRetainedManifold{};
}

struct WheelRuntimeState {
    int32_t event_world_ids[4] = {-1, -1, -1, -1};
    Vec3 event_world_normals[4] = {};
    float steer_angle[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float engine_force[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float brake[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float lat_friction[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float long_friction[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float extra_pushback[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float solver_time_step = 1.0f / 60.0f;

    Vec3 accumulated_force = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 accumulated_ang_accel = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 accumulated_torque_bt = Vec3(0.0f, 0.0f, 0.0f);

    ::rocketgpu::Quat pre_solve_rot = ::rocketgpu::Quat(1.0f, 0.0f, 0.0f, 0.0f);

    Vec3 pre_solve_basis_forward = Vec3(1.0f, 0.0f, 0.0f);
    Vec3 pre_solve_basis_right = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 pre_solve_basis_up = Vec3(0.0f, 0.0f, 1.0f);
    uint8_t pre_solve_basis_authoritative = 0;

    Vec3 cal_prev_sticky_dir = Vec3(0.0f, 0.0f, 0.0f);

    Vec3 cal_prev_wheel_normal_sum = Vec3(0.0f, 0.0f, 0.0f);
    uint8_t cal_prev_world_wheel_contacts = 0;

    uint8_t cal_game_jumped = 0;
    uint8_t cal_game_double_jumped = 0;
    uint8_t cal_cached_ground = 0;
    float cal_jump_inactive_time = 0.0f;
    uint8_t cal_frame_open = 0;
    uint8_t cal_restored = 0;
};

struct KickoffSettledRow {
    CarState car;
    WheelRuntimeState wheels;
};

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
#if defined(ROCKETGPU_USE_CUDA)
        pool.interleaved = interleaved ? 1 : 0;
#else
        (void)interleaved;
        pool.interleaved = 0;
#endif
        pool.caps = caps;
    }

    WorldSolverScratchRows view(int env_idx) const { return pool.view(env_idx); }
};
#endif

#if !defined(ROCKETGPU_USE_CUDA)
struct BoostPadBudget {
    ::rocketgpu::Vec3 pos_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    float budget_bt = 0.0f;
};

struct CarRbSnapMemo {
    ::rocketgpu::Vec3 fwd, right, up;
    ::rocketgpu::Quat rot;
    ::rocketgpu::Quat raw;
    uint8_t valid = 0;
};
#endif
struct CarShapeCacheEntry {
    ::rocketgpu::Vec3 half_extent_bt;
    float bounding_sphere_radius = 0.0f;
    ::rocketgpu::Vec3 inv_inertia_local_uu;
    float contact_breaking_threshold_bt = 0.0f;
    float mesh_reach_bt = 0.0f;
    ::rocketgpu::Vec3 inv_inertia_local_bt;
    float susp_front_scale = 0.0f;
    float susp_back_scale = 0.0f;
    float min_wheel_hit_dist_sq_bt = 0.0f;
    float front_bone_z = 0.0f;
    float back_bone_z = 0.0f;
    alignas(16) float wheel_offset_bt_x[4] = {};
    alignas(16) float wheel_offset_bt_y[4] = {};
    alignas(16) float wheel_offset_bt_z[4] = {};
    float wheel_rest_len[4] = {};
    float wheel_ray_reach_bt = 0.0f;
    alignas(16) float wheel_ray_len_bt[4] = {};
#if !defined(ROCKETGPU_USE_CUDA)
    float wheel_ray_len_uu[4] = {};
#endif
    bool wheel_ray_len_positive = false;
    alignas(16) float wheel_pushback_thresh_bt[4] = {};
};

struct WorldSolverScratch {
    BallMeshPersistentManifold ball_world_mesh_manifolds[kMaxBallWorldMeshPersistentManifolds] = {};
    BallMeshPersistentManifold ball_world_plane_manifolds[kBallWorldPlaneSlots] = {};
    uint8_t ball_settled = 0;
    uint8_t ball_settled_world_entry_count = 0;
    int16_t ball_settled_world_entries[kBallSettledMaxWorldEntries] = {};
    uint32_t ball_settled_mutator_epoch = 0;
    float ball_settled_sub_dt = 0.0f;
    ::rocketgpu::Vec3 ball_settled_solver_vel_bt = ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);
    ::rocketgpu::BallState ball_settled_state = {};

    BulletBroadphaseHistory bp;
    uint8_t retained_reset_pending = 0;
    uint8_t car_world_reset[8] = {};
    uint8_t pair_cache_cars = 8;
    uint16_t ball_mesh_maybe_retained = 0xFFFFu;
    uint32_t pair_cache_maybe_valid = ~0u;
#if !defined(ROCKETGPU_USE_CUDA)
    CarRbSnapMemo rb_snap[8] = {};
    CarShapeCacheEntry car_shape[8] = {};
    ::rocketgpu::CarConfig car_shape_cfg[8] = {};
    float car_shape_mass[8] = {};
    uint8_t car_shape_valid = 0;
    BoostPadBudget pad_budget[8] = {};
#endif
};

#if !defined(ROCKETGPU_USE_CUDA)
inline const ::rocketgpu::Quat* CarRbSnapRaw(const WorldSolverScratch* s, int slot, CarCRefT car) {
    static_assert(offsetof(::rocketgpu::CarState, basis_right) == offsetof(::rocketgpu::CarState, basis_forward) + 12 &&
                      offsetof(::rocketgpu::CarState, basis_up) == offsetof(::rocketgpu::CarState, basis_forward) + 24 &&
                      offsetof(CarRbSnapMemo, right) == offsetof(CarRbSnapMemo, fwd) + 12 &&
                      offsetof(CarRbSnapMemo, up) == offsetof(CarRbSnapMemo, fwd) + 24,
                  "contiguous basis blocks");
    if (s == nullptr || slot >= 8 || !s->rb_snap[slot].valid) {
        return nullptr;
    }
    const CarRbSnapMemo& m = s->rb_snap[slot];
    const char* a = reinterpret_cast<const char*>(&car) + offsetof(::rocketgpu::CarState, basis_forward);
    const char* b = reinterpret_cast<const char*>(&m) + offsetof(CarRbSnapMemo, fwd);
    uint32_t a_tail, b_tail;
    std::memcpy(&a_tail, a + 32, 4);
    std::memcpy(&b_tail, b + 32, 4);
    return std::memcmp(a, b, 32) == 0 && a_tail == b_tail ? &m.raw : nullptr;
}
#endif

#if !defined(ROCKETGPU_STEP_TYPES_ONLY)
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
    return ::rocketgpu::Vec3(DivideBy50(value_uu.x), DivideBy50(value_uu.y), DivideBy50(value_uu.z));
}

ROCKETGPU_HD ::rocketgpu::Vec3 BallStoredLinearVelocityToBt(const ::rocketgpu::Vec3& value_uu) {
    return ::rocketgpu::Vec3(DivideBy50(value_uu.x), DivideBy50(value_uu.y), DivideBy50(value_uu.z));
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

ROCKETGPU_HD void SetCarOriginBt(CarRefT car, const ::rocketgpu::Vec3& pos_bt) {
    constexpr float kUuPerBt = 50.0f;
    car.pos_bt = pos_bt;
    car.pos = pos_bt * kUuPerBt;
}

ROCKETGPU_HD void SetCarOriginUu(CarRefT car, const ::rocketgpu::Vec3& pos_uu) {
    constexpr float kUuToBt = 1.0f / 50.0f;
    SetCarOriginBt(car, pos_uu * kUuToBt);
}

ROCKETGPU_HD void SetCarLinearVelocityBt(CarRefT car, const ::rocketgpu::Vec3& vel_bt) {
    constexpr float kUuPerBt = 50.0f;
    car.vel_bt = vel_bt;
    car.vel = vel_bt * kUuPerBt;
}

ROCKETGPU_HD void SetCarLinearVelocityUu(CarRefT car, const ::rocketgpu::Vec3& vel_uu) {
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

ROCKETGPU_HD ::rocketgpu::Vec3 VelocityAtPoint(
    CarCRefT car,
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

ROCKETGPU_HD bool NearestTriangleEdgePointBT(const ::rocketgpu::Vec3& p, const ::rocketgpu::Vec3& a, const ::rocketgpu::Vec3& b,
                                             const ::rocketgpu::Vec3& c, float max_dist_sq, ::rocketgpu::Vec3& out_nearest) {
    const ::rocketgpu::Vec3 from[3] = {a, b, c};
    const ::rocketgpu::Vec3 to[3] = {b, c, a};
    bool found = false;
    for (int i = 0; i < 3; ++i) {
        ::rocketgpu::Vec3 diff = p - from[i];
        const ::rocketgpu::Vec3 v = to[i] - from[i];
        float t = Dot(v, diff);
        if (t > 0.0f) {
            const float dotVV = Dot(v, v);
            if (t < dotVV) {
                t /= dotVV;
                diff -= v * t;
            } else {
                t = 1.0f;
                diff -= v;
            }
        } else {
            t = 0.0f;
        }
        const float dist_sq = Dot(diff, diff);
        if (dist_sq < max_dist_sq) {
            max_dist_sq = dist_sq;
            out_nearest = from[i] + v * t;
            found = true;
        }
    }
    return found;
}

ROCKETGPU_HD bool PointInTriangleBT(const ::rocketgpu::Vec3& p, const ::rocketgpu::Vec3& a, const ::rocketgpu::Vec3& b, const ::rocketgpu::Vec3& c, const ::rocketgpu::Vec3& normal) {
    const float r1 = Dot(Cross(b - a, normal), p - a);
    const float r2 = Dot(Cross(c - b, normal), p - b);
    const float r3 = Dot(Cross(a - c, normal), p - c);
    return (r1 > 0.0f && r2 > 0.0f && r3 > 0.0f) || (r1 <= 0.0f && r2 <= 0.0f && r3 <= 0.0f);
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
        hasContact = NearestTriangleEdgePointBT(
            sphere_center_bt, tri_a_bt, tri_b_bt, tri_c_bt, contactCapsuleRadiusSqr, contactPoint);
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

    constexpr float kUuPerBt = 50.0f;
    const ::rocketgpu::Vec3 bt_rel_ball_contact =
        bt_rel_ball_contact_override ? *bt_rel_ball_contact_override : UuVecToBtDiv(rel_ball_contact);
    const ::rocketgpu::Vec3 bt_rel_other_contact =
        bt_rel_other_contact_override ? *bt_rel_other_contact_override : UuVecToBtDiv(rel_other_contact);

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
    float jac_diag_ab_inv = 1.0f / denom;
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
            jac_diag_ab_inv = 1.0f / bt_denom;
        }
    }

    const ::rocketgpu::Vec3 rel_ball_cross_normal = Cross(rel_ball_contact, normal);
    const ::rocketgpu::Vec3 rel_other_cross_normal = Cross(rel_other_contact, normal);
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
    row.jac_diag_ab_inv = jac_diag_ab_inv;
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

    constexpr float kUuPerBtFriction = 50.0f;
    const ::rocketgpu::Vec3 bt_rel_ball_contact =
        bt_rel_ball_contact_override ? *bt_rel_ball_contact_override : UuVecToBtDiv(normal_row.rel_ball_contact);
    const ::rocketgpu::Vec3 bt_rel_other_contact =
        bt_rel_other_contact_override ? *bt_rel_other_contact_override : UuVecToBtDiv(normal_row.rel_other_contact);

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
    float jac_diag_ab_inv = 1.0f / denom;
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
            jac_diag_ab_inv = 1.0f / bt_denom;
        }
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
    row.jac_diag_ab_inv = jac_diag_ab_inv;
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

    BallVelocityContactRow normal_row;
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
    normal_rows[normal_row_count] = normal_row;
    if (!normal_row.valid) {
        return;
    }

    const int normal_row_idx = normal_row_count;
    ++normal_row_count;

    if (!friction_rows || friction_row_count >= kMaxBallFrictionContactRows) {
        return;
    }

    BallFrictionContactRow friction_row;
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
    friction_rows[friction_row_count] = friction_row;
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
    const ::rocketgpu::Vec3 ang_bt = angular_component_bt_override
        ? *angular_component_bt_override
        : angular_component * kCarWorldUuPerBt;
    const float denom = inv_mass + Dot(normal, Cross(angular_component, rel_contact));
    if (denom <= 1e-8f) {
        return;
    }
    float jac_diag_ab_inv = 1.0f / denom;
    if (angular_component_bt_override) {
        const float bt_denom = inv_mass + Dot(normal, Cross(ang_bt, rel_contact_bt));
        if (bt_denom > 1e-8f) {
            jac_diag_ab_inv = 1.0f / bt_denom;
        }
    }

    row.valid = 1;
    row.normal = normal;
    row.rel_contact = rel_contact;
    row.rel_contact_bt = rel_contact_bt;
    row.rel_cross_normal_bt = Cross(rel_contact_bt, normal);
    row.rel_cross_normal = row.rel_cross_normal_bt * kCarWorldUuPerBt;
    row.angular_component = angular_component;
    row.angular_component_bt = ang_bt;
    row.has_bt = 1;
    row.inv_mass = inv_mass;
    row.jac_diag_ab_inv = jac_diag_ab_inv;
    float row_rhs_bt = 0.0f;
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
        const float restitution_rel_vel_n_bt =
            bt_normal.dot(bt_linear_velocity + bt_angular_velocity.cross(bt_rel_contact));
        const float rhs_rel_vel_n_bt =
            Dot(normal, linear_velocity_bt + external_force_impulse_bt) +
            Dot(rel_cross_normal_bt, trace_inputs->angular_velocity +
                (trace_inputs->has_bt_inputs ? trace_inputs->external_torque_impulse : trace_inputs->external_angular_delta));
        const float restitution_bt = trace_inputs->restitution_no_velocity_threshold
            ? Max(restitution * -restitution_rel_vel_n_bt, 0.0f)
            : BulletRestitutionCurveBt(restitution_rel_vel_n_bt, restitution);
        row_rhs_bt = (restitution_bt - rhs_rel_vel_n_bt) * jac_diag_ab_inv;
    } else {
        const float restitution_rel_vel_n = Dot(restitution_rel_vel, normal);
        const float rhs_rel_vel_n = Dot(rhs_rel_vel, normal);
        row_rhs_bt = (BulletRestitutionCurve(restitution_rel_vel_n, restitution) - rhs_rel_vel_n) *
            jac_diag_ab_inv * kCarWorldBtPerUu;
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

    const ::rocketgpu::Vec3 ang_bt = angular_component_bt_override
        ? *angular_component_bt_override
        : angular_component * 50.0f;
    const float denom = normal_row.inv_mass + Dot(tangent, Cross(angular_component, normal_row.rel_contact));
    if (denom <= 1e-8f) {
        return;
    }
    float jac_diag_ab_inv = 1.0f / denom;
    if (angular_component_bt_override) {
        const float bt_denom = normal_row.inv_mass +
            Dot(tangent, Cross(ang_bt, normal_row.rel_contact_bt));
        if (bt_denom > 1e-8f) {
            jac_diag_ab_inv = 1.0f / bt_denom;
        }
    }

    row.valid = 1;
    row.normal_row_idx = normal_row_idx;
    row.tangent = tangent;
    row.rel_contact = normal_row.rel_contact;
    row.rel_cross_tangent_bt = Cross(normal_row.rel_contact_bt, tangent);
    row.rel_cross_tangent = row.rel_cross_tangent_bt * 50.0f;
    row.angular_component = angular_component;
    row.angular_component_bt = ang_bt;
    row.has_bt = 1;
    row.inv_mass = normal_row.inv_mass;
    row.jac_diag_ab_inv = jac_diag_ab_inv;
    row.rhs = (-Dot(base_rel_vel, tangent)) * jac_diag_ab_inv;
    if (angular_component_bt_override && trace_inputs && trace_inputs->has_bt_inputs) {
        const float vel1_dotn_bt =
            Dot(tangent, trace_inputs->linear_velocity_bt + trace_inputs->external_force_impulse_bt) +
            Dot(row.rel_cross_tangent_bt, trace_inputs->angular_velocity);
        const float rel_vel_bt = vel1_dotn_bt + 0.0f;
        row.rhs_bt = (0.0f - rel_vel_bt) * jac_diag_ab_inv;
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

    CarWorldVelocityContactRow normal_row;
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
    normal_rows[normal_row_count] = normal_row;
    if (!normal_row.valid) {
        return;
    }

    const int normal_row_idx = normal_row_count;
    ++normal_row_count;

    if (!friction_rows || friction_row_count >= kMaxCarWorldFrictionContactRows) {
        return;
    }

    CarWorldFrictionContactRow friction_row;
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
    friction_rows[friction_row_count] = friction_row;
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

ROCKETGPU_HD bool SolveBallVelocityNormalRowWithAccumulators(
    int count,
    BallVelocityContactRow& row,
    BallVelocityAccumulator& ball_acc,
    BallVelocityAccumulator* car_accs,
    bool use_fma = false) {
#if defined(__CUDA_ARCH__)
    const BallVelocityContactRow r = row;
#else
    const BallVelocityContactRow& r = row;
#endif
    if (!r.valid) {
        return false;
    }

    const bool trace_dynamic_row = r.other_local_idx >= 0 && r.other_local_idx < count;
    float delta_impulse = r.rhs;
    float trace_delta_vel1_dotn = 0.0f;
    float trace_delta_vel2_dotn = 0.0f;
    if (trace_dynamic_row && car_accs) {
        const BallVelocityAccumulator& other_acc = car_accs[r.other_local_idx];
        const float delta_vel1_dotn =
            SolverRowDot(-r.normal, other_acc.delta_linear, use_fma) +
            SolverRowDot(r.rel_other_cross_normal, other_acc.delta_angular, use_fma);
        const float delta_vel2_dotn =
            SolverRowDot(r.normal, ball_acc.delta_linear, use_fma) +
            SolverRowDot(r.rel_ball_cross_normal, ball_acc.delta_angular, use_fma);
        delta_impulse = SolverRowSubtract(delta_impulse, delta_vel1_dotn, r.jac_diag_ab_inv, use_fma);
        delta_impulse = SolverRowSubtract(delta_impulse, delta_vel2_dotn, r.jac_diag_ab_inv, use_fma);
        trace_delta_vel1_dotn = delta_vel1_dotn;
        trace_delta_vel2_dotn = delta_vel2_dotn;
    } else {
        const float delta_vel =
            SolverRowDot(r.normal, ball_acc.delta_linear, use_fma) +
            SolverRowDot(r.rel_ball_cross_normal, ball_acc.delta_angular, use_fma);
        delta_impulse = SolverRowSubtract(delta_impulse, delta_vel, r.jac_diag_ab_inv, use_fma);
    }
    (void)trace_delta_vel1_dotn;
    (void)trace_delta_vel2_dotn;

    const float applied_before = r.applied_impulse;
    const float sum = applied_before + delta_impulse;
    if (sum < 0.0f) {
        delta_impulse = -applied_before;
        row.applied_impulse = 0.0f;
    } else {
        row.applied_impulse = sum;
    }

    if (delta_impulse == 0.0f) {
        return GlibcFloatBits(applied_before) != GlibcFloatBits(row.applied_impulse);
    }

    ApplyBallVelocityAccumulatorImpulse(
        ball_acc,
        r.normal * r.ball_inv_mass,
        r.ball_angular_component,
        delta_impulse, use_fma);

    if (trace_dynamic_row && car_accs) {
        ApplyBallVelocityAccumulatorImpulse(
            car_accs[r.other_local_idx],
            (-r.normal) * r.other_inv_mass,
            r.other_angular_component,
            delta_impulse, use_fma);
    }
    return true;
}

ROCKETGPU_HD bool SolveBallVelocityFrictionRowWithAccumulators(
    int count,
    EnvStridedArray<BallVelocityContactRow> normal_rows,
    int normal_row_count,
    BallFrictionContactRow& row,
    BallVelocityAccumulator& ball_acc,
    BallVelocityAccumulator* car_accs,
    bool use_fma = false) {
#if defined(__CUDA_ARCH__)
    const BallFrictionContactRow r = row;
#else
    const BallFrictionContactRow& r = row;
#endif
    if (!r.valid || !normal_rows || r.normal_row_idx < 0 || r.normal_row_idx >= normal_row_count) {
        return false;
    }

    const float total_impulse = normal_rows[r.normal_row_idx].applied_impulse;
    if (total_impulse <= 0.0f) {
        return false;
    }

    const bool trace_dynamic_row = r.other_local_idx >= 0 && r.other_local_idx < count;
    const float max_impulse = r.friction * total_impulse;
    float delta_impulse = r.rhs;
    float trace_delta_vel1_dotn = 0.0f;
    float trace_delta_vel2_dotn = 0.0f;
    if (trace_dynamic_row && car_accs) {
        const BallVelocityAccumulator& other_acc = car_accs[r.other_local_idx];
        const float delta_vel1_dotn =
            SolverRowDot(-r.tangent, other_acc.delta_linear, use_fma) +
            SolverRowDot(r.rel_other_cross_tangent, other_acc.delta_angular, use_fma);
        const float delta_vel2_dotn =
            SolverRowDot(r.tangent, ball_acc.delta_linear, use_fma) +
            SolverRowDot(r.rel_ball_cross_tangent, ball_acc.delta_angular, use_fma);
        delta_impulse = SolverRowSubtract(delta_impulse, delta_vel1_dotn, r.jac_diag_ab_inv, use_fma);
        delta_impulse = SolverRowSubtract(delta_impulse, delta_vel2_dotn, r.jac_diag_ab_inv, use_fma);
        trace_delta_vel1_dotn = delta_vel1_dotn;
        trace_delta_vel2_dotn = delta_vel2_dotn;
    } else {
        const float delta_vel =
            SolverRowDot(r.tangent, ball_acc.delta_linear, use_fma) +
            SolverRowDot(r.rel_ball_cross_tangent, ball_acc.delta_angular, use_fma);
        delta_impulse = SolverRowSubtract(delta_impulse, delta_vel, r.jac_diag_ab_inv, use_fma);
    }
    (void)trace_delta_vel1_dotn;
    (void)trace_delta_vel2_dotn;

    const float applied_before = r.applied_impulse;
    const float sum = applied_before + delta_impulse;
    if (sum < -max_impulse) {
        delta_impulse = -max_impulse - applied_before;
        row.applied_impulse = -max_impulse;
    } else if (sum > max_impulse) {
        delta_impulse = max_impulse - applied_before;
        row.applied_impulse = max_impulse;
    } else {
        row.applied_impulse = sum;
    }

    if (delta_impulse == 0.0f) {
        return GlibcFloatBits(applied_before) != GlibcFloatBits(row.applied_impulse);
    }

    ApplyBallVelocityAccumulatorImpulse(
        ball_acc,
        r.tangent * r.ball_inv_mass,
        r.ball_angular_component,
        delta_impulse, use_fma);

    if (trace_dynamic_row && car_accs) {
        ApplyBallVelocityAccumulatorImpulse(
            car_accs[r.other_local_idx],
            (-r.tangent) * r.other_inv_mass,
            r.other_angular_component,
            delta_impulse, use_fma);
    }
    return true;
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

    row.rhs_penetration = ((penetration * BallSplitPenetrationErp()) / dt) * row.jac_diag_ab_inv;

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
        ((penetration_bt * BallSplitPenetrationErp()) / dt) * row.jac_diag_ab_inv_bt;
}

ROCKETGPU_HD void SolveCarWorldSplitContactRows(
    CarRefT car,
    CarWorldSplitContactRow* rows,
    int row_count,
    float dt,
    bool native_sse_math = false) {
    if (!rows || row_count <= 0) {
        return;
    }

    SplitImpulseAccumulator split;
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
        const CarWorldSplitContactRow src = rows[row_idx];
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
    int friction_row_count,
    int normal_row_base_in = -1,
    int normal_row_idx_offset = 0) {
    if (body_local_idx < 0) {
        return;
    }

    const int normal_row_base = normal_row_base_in >= 0 ? normal_row_base_in : shared_normal_row_count;
    if (shared_normal_rows && normal_rows) {
        for (int row_idx = 0; row_idx < normal_row_count; ++row_idx) {
            const CarWorldVelocityContactRow src = normal_rows[row_idx];
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
            const CarWorldFrictionContactRow src = friction_rows[row_idx];
            if (!src.valid || shared_friction_row_count >= kMaxCarWorldSharedFrictionContactRows) {
                continue;
            }
            CarSharedFrictionContactRow& dst = shared_friction_rows[shared_friction_row_count++];
            dst = CarSharedFrictionContactRow{};
            dst.valid = 1;
            dst.normal_row_idx = normal_row_base + normal_row_idx_offset + src.normal_row_idx;
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
    return source_tag == 1 || source_tag == 2 || source_tag == 5 || source_tag == 6 ||
           source_tag == kGoalFloorSourceTag || source_tag == kGoalFloorSourceTag + 1;
}

ROCKETGPU_HD int BulletPlaneStaticSourceTag(int static_id) {
    switch (static_id - 100) {
        case 0: return 6;
        case 1: return 5;
        case 2: return 2;
        case 3: return 1;
        default: return kGoalFloorSourceTag + (static_id - 104);
    }
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
            else plane_rank = 4 + (src - kGoalFloorSourceTag);
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
    CarCSpanT cars,
    const ::rocketgpu::CarConfig* car_configs,
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

ROCKETGPU_HD bool SolveSharedCarVelocityNormalRow(
    int count,
    CarSpanT cars,
    const ::rocketgpu::CarConfig* car_configs,
    CarSharedVelocityContactRow& row,
    BallVelocityAccumulator* car_accs,
    bool use_fma = false) {
#if defined(__CUDA_ARCH__)
    const CarSharedVelocityContactRow r = row;
#else
    const CarSharedVelocityContactRow& r = row;
#endif
    if (!r.valid || !cars || !car_configs || !car_accs) {
        return false;
    }
    if (r.body_a_local_idx < 0 || r.body_a_local_idx >= count) {
        return false;
    }

    constexpr float kBtPerUu = 1.0f / 50.0f;
    constexpr float kUuPerBt = 50.0f;
    const ::rocketgpu::Vec3 normal_b = (r.body_b_local_idx >= 0 && r.body_b_local_idx < count) ? -r.normal_a : ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);

    const BallVelocityAccumulator& acc_a = car_accs[r.body_a_local_idx];
    const float delta_impulse_vel_a_bt =
        SolverRowDot(r.normal_a, acc_a.delta_linear, use_fma) +
        SolverRowDot(r.rel_a_cross_normal, acc_a.delta_angular, use_fma);
    float delta_impulse_vel_b_bt = 0.0f;

    if (r.body_b_local_idx >= 0 && r.body_b_local_idx < count) {
        const BallVelocityAccumulator& acc_b = car_accs[r.body_b_local_idx];
        delta_impulse_vel_b_bt =
            SolverRowDot(normal_b, acc_b.delta_linear, use_fma) +
            SolverRowDot(r.rel_b_cross_normal, acc_b.delta_angular, use_fma);
    }

    const float applied_before = r.applied_impulse;
    float delta_impulse = 0.0f;
    if (r.has_rhs_bt) {
        float delta_impulse_bt = r.rhs_bt;
        delta_impulse_bt = SolverRowSubtract(delta_impulse_bt, delta_impulse_vel_b_bt, r.jac_diag_ab_inv, use_fma);
        delta_impulse_bt = SolverRowSubtract(delta_impulse_bt, delta_impulse_vel_a_bt, r.jac_diag_ab_inv, use_fma);

        const float sum_bt = applied_before + delta_impulse_bt;
        if (sum_bt < 0.0f) {
            delta_impulse = -applied_before;
            row.applied_impulse = 0.0f;
        } else {
            delta_impulse = delta_impulse_bt;
            row.applied_impulse = sum_bt;
        }
    } else {
        const float delta_impulse_vel_a_uu = delta_impulse_vel_a_bt * kUuPerBt;
        const float delta_impulse_vel_b_uu = delta_impulse_vel_b_bt * kUuPerBt;
        const float applied_uu = applied_before * kUuPerBt;
        delta_impulse = r.rhs;
        delta_impulse = SolverRowSubtract(delta_impulse, delta_impulse_vel_b_uu, r.jac_diag_ab_inv, use_fma);
        delta_impulse = SolverRowSubtract(delta_impulse, delta_impulse_vel_a_uu, r.jac_diag_ab_inv, use_fma);

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
        return GlibcFloatBits(applied_before) != GlibcFloatBits(row.applied_impulse);
    }

    ApplySharedCarVelocityImpulse(
        car_accs,
        r.body_a_local_idx,
        r.normal_a * r.inv_mass_a,
        r.angular_component_a,
        delta_impulse, use_fma);
    if (r.body_b_local_idx >= 0 && r.body_b_local_idx < count) {
        ApplySharedCarVelocityImpulse(
            car_accs,
            r.body_b_local_idx,
            normal_b * r.inv_mass_b,
            r.angular_component_b,
            delta_impulse, use_fma);
    }
    return true;
}

ROCKETGPU_HD bool SolveSharedCarVelocityFrictionRow(
    int count,
    CarCSpanT cars,
    const ::rocketgpu::CarConfig* car_configs,
    EnvStridedArray<const CarSharedVelocityContactRow> normal_rows,
    int normal_row_count,
    CarSharedFrictionContactRow& row,
    BallVelocityAccumulator* car_accs,
    bool use_fma = false) {
#if defined(__CUDA_ARCH__)
    const CarSharedFrictionContactRow r = row;
#else
    const CarSharedFrictionContactRow& r = row;
#endif
    if (!r.valid || !cars || !car_configs || !car_accs || !normal_rows) {
        return false;
    }
    if (r.normal_row_idx < 0 || r.normal_row_idx >= normal_row_count) {
        return false;
    }
    if (r.body_a_local_idx < 0 || r.body_a_local_idx >= count) {
        return false;
    }

    const ::rocketgpu::Vec3 tangent_b = (r.body_b_local_idx >= 0 && r.body_b_local_idx < count) ? -r.tangent_a : ::rocketgpu::Vec3(0.0f, 0.0f, 0.0f);

    const float total_impulse = normal_rows[r.normal_row_idx].applied_impulse;
    if (total_impulse <= 0.0f) {
        return false;
    }

    constexpr float kBtPerUu = 1.0f / 50.0f;
    const float max_impulse = r.friction * total_impulse;
    float delta_impulse = r.has_rhs_bt ? r.rhs_bt : r.rhs * kBtPerUu;
    if (r.body_b_local_idx >= 0 && r.body_b_local_idx < count) {
        const BallVelocityAccumulator& acc_b = car_accs[r.body_b_local_idx];
        const float delta_impulse_vel_b_bt =
            SolverRowDot(tangent_b, acc_b.delta_linear, use_fma) +
            SolverRowDot(r.rel_b_cross_tangent, acc_b.delta_angular, use_fma);
        delta_impulse = SolverRowSubtract(delta_impulse, delta_impulse_vel_b_bt, r.jac_diag_ab_inv, use_fma);
    }
    const BallVelocityAccumulator& acc_a = car_accs[r.body_a_local_idx];
    const float delta_impulse_vel_a_bt =
        SolverRowDot(r.tangent_a, acc_a.delta_linear, use_fma) +
        SolverRowDot(r.rel_a_cross_tangent, acc_a.delta_angular, use_fma);
    delta_impulse = SolverRowSubtract(delta_impulse, delta_impulse_vel_a_bt, r.jac_diag_ab_inv, use_fma);

    const float applied_before = r.applied_impulse;
    const float sum = applied_before + delta_impulse;
    if (sum < -max_impulse) {
        delta_impulse = -max_impulse - applied_before;
        row.applied_impulse = -max_impulse;
    } else if (sum > max_impulse) {
        delta_impulse = max_impulse - applied_before;
        row.applied_impulse = max_impulse;
    } else {
        row.applied_impulse = sum;
    }

    if (delta_impulse == 0.0f) {
        return GlibcFloatBits(applied_before) != GlibcFloatBits(row.applied_impulse);
    }

    ApplySharedCarVelocityImpulse(
        car_accs,
        r.body_a_local_idx,
        r.tangent_a * r.inv_mass_a,
        r.angular_component_a,
        delta_impulse, use_fma);
    if (r.body_b_local_idx >= 0 && r.body_b_local_idx < count) {
        ApplySharedCarVelocityImpulse(
            car_accs,
            r.body_b_local_idx,
            tangent_b * r.inv_mass_b,
            r.angular_component_b,
            delta_impulse, use_fma);
    }
    return true;
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

    if ((pair_rows && pair_row_count > 0) || (ball_rows && ball_row_count > 0)) {
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
    }
    if (ball_split) {
        ball_split->is_bt = (touched_ball && !ball_blocked) ? 1 : 0;
    }
    for (int i = 0; i < count && i < kMarkCars; ++i) {
        car_splits[i].is_bt = (touched_world[i] && !blocked[i]) ? 1 : 0;
    }
}

#undef ROCKETGPU_SHARED_SOLVER_BACKEND_TAG

ROCKETGPU_HD bool SolveSharedCarSplitRow(
    int count,
    CarSpanT cars,
    CarSharedSplitContactRow& row,
    int iter,
    int row_idx,
    const char* bucket,
    SplitImpulseAccumulator* car_splits,
    int env_idx) {
    if (!row.valid || !cars || !car_splits) {
        return false;
    }
    if (row.body_a_local_idx < 0 || row.body_a_local_idx >= count) {
        return false;
    }

    SplitImpulseAccumulator& split_a = car_splits[row.body_a_local_idx];
    SplitImpulseAccumulator* split_b_bt = nullptr;
    if (row.body_b_local_idx >= 0 && row.body_b_local_idx < count) {
        split_b_bt = &car_splits[row.body_b_local_idx];
    }
    if (row.has_bt && split_a.is_bt && (row.body_b_local_idx < 0 || (split_b_bt && split_b_bt->is_bt))) {
        if (row.rhs_penetration_bt == 0.0f) {
            return false;
        }
        const float applied_before_bt = row.applied_push_impulse_bt;

        float delta_impulse_bt = row.rhs_penetration_bt;
        const float delta_vel_linear_bt = BtSimdDot3LikeBullet(row.normal_a, split_a.push_velocity);
        const float delta_vel_angular_bt = BtSimdDot3LikeBullet(row.rel_a_cross_normal_bt, split_a.turn_velocity);
        if (split_b_bt) {
            const float delta_vel_linear_b_bt = BtSimdDot3LikeBullet(row.normal_b, split_b_bt->push_velocity);
            const float delta_vel_angular_b_bt = BtSimdDot3LikeBullet(row.rel_b_cross_normal_bt, split_b_bt->turn_velocity);
            delta_impulse_bt -= (delta_vel_linear_b_bt + delta_vel_angular_b_bt) * row.jac_diag_ab_inv_bt;
        }
        delta_impulse_bt -= (delta_vel_linear_bt + delta_vel_angular_bt) * row.jac_diag_ab_inv_bt;
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
        return delta_impulse_bt != 0.0f || GlibcFloatBits(applied_before_bt) != GlibcFloatBits(row.applied_push_impulse_bt);
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

    const float applied_before = row.applied_push_impulse;
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
    return delta_impulse != 0.0f || GlibcFloatBits(applied_before) != GlibcFloatBits(row.applied_push_impulse);
}

ROCKETGPU_HD bool SolveUnifiedBallSplitRow(
    int count,
    BallSplitContactRow& row,
    SplitImpulseAccumulator& ball_split,
    SplitImpulseAccumulator* car_splits) {
    if (!row.valid || !car_splits) {
        return false;
    }

    SplitImpulseAccumulator* other_split_bt = nullptr;
    if (row.other_local_idx >= 0 && row.other_local_idx < count) {
        other_split_bt = &car_splits[row.other_local_idx];
    }
    if (row.has_bt && ball_split.is_bt && (!other_split_bt || other_split_bt->is_bt)) {
        const float applied_before_bt = row.applied_push_impulse_bt;
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
            return GlibcFloatBits(applied_before_bt) != GlibcFloatBits(row.applied_push_impulse_bt);
        }

        ball_split.push_velocity += (row.normal * row.ball_inv_mass) * delta_impulse_bt;
        ball_split.turn_velocity += row.ball_angular_component_bt * delta_impulse_bt;
        if (other_split_bt) {
            other_split_bt->push_velocity += ((-row.normal) * row.other_inv_mass) * delta_impulse_bt;
            other_split_bt->turn_velocity += row.other_angular_component_bt * delta_impulse_bt;
        }
        return true;
    }

    const float applied_before = row.applied_push_impulse;
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
        return GlibcFloatBits(applied_before) != GlibcFloatBits(row.applied_push_impulse);
    }

    ball_split.push_velocity += row.normal * (row.ball_inv_mass * delta_impulse);
    ball_split.turn_velocity += row.ball_angular_component * delta_impulse;
    if (other_split) {
        other_split->push_velocity += (-row.normal) * (row.other_inv_mass * delta_impulse);
        other_split->turn_velocity += row.other_angular_component * delta_impulse;
    }
    return true;
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
#if !defined(ROCKETGPU_USE_CUDA)
    grid.leaf_grid.host_tables = cfg.host_tables != nullptr && cfg.host_tables->owner == &cfg ? cfg.host_tables : nullptr;
#else
    grid.leaf_grid.ray_infos = cfg.arena_mesh_ray_grid_infos;
    grid.leaf_grid.ray_cell_start = cfg.arena_mesh_ray_cell_start;
    grid.leaf_grid.ray_entries = cfg.arena_mesh_ray_entries;
    grid.leaf_grid.clearance = cfg.arena_mesh_clearance;
#endif
    grid.sweep_statics = cfg.arena_sweep_statics;
    return grid;
}

ROCKETGPU_HD bool BulletStaticPlaneProxyAabb(int plane_rank, ::rocketgpu::Vec3& out_min, ::rocketgpu::Vec3& out_max) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kLarge = 1.0e18f;
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
    if (plane_rank == 4 || plane_rank == 5) {
        const float y = GoalFloorOriginBt(plane_rank - 4).y;
        out_min = ::rocketgpu::Vec3(-kGoalFloorHalfXBt, y - kGoalFloorHalfYBt, -1.0e18f);
        out_max = ::rocketgpu::Vec3(kGoalFloorHalfXBt, y + kGoalFloorHalfYBt, 0.0f);
        return true;
    }
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

ROCKETGPU_HD BulletBroadphaseProxy ComputeBulletCarProxyAabb(
    CarCRefT car,
    const ::rocketgpu::CarConfig& cfg,
    float dt,
    const ArenaBroadphaseGrid& grid,
    bool native_sse_math = false,
    const ::rocketgpu::Vec3* predict_vel_bt_in = nullptr,
    const ::rocketgpu::Vec3* predict_ang_vel_in = nullptr) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kContactThresholdBt = 0.02f;

    constexpr float kCompoundMarginBt = 0.0f;
    const ::rocketgpu::Vec3 predict_vel_bt = predict_vel_bt_in != nullptr ? *predict_vel_bt_in : car.vel_bt;
    const ::rocketgpu::Vec3 predict_ang_vel = predict_ang_vel_in != nullptr ? *predict_ang_vel_in : car.ang_vel;

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
        GameIntegrateOrientationBasisColumns(pred_basis_fwd, pred_basis_right, pred_basis_up, predicted_rot, predict_ang_vel, dt);
    } else if (car.basis_authoritative) {
        IntegrateOrientationBasisColumns(
            pred_basis_fwd, pred_basis_right, pred_basis_up, predicted_rot, predict_ang_vel, dt, native_sse_math);
    } else {
        IntegrateOrientation(predicted_rot, predict_ang_vel, dt);
    }
    const ::rocketgpu::Vec3 origin_bt = car.pos_bt;
    const ::rocketgpu::Vec3 predicted_origin_bt = origin_bt + predict_vel_bt * dt;

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
        (axis_x.x * child_origin_bt.x + axis_y.x * child_origin_bt.y) + axis_z.x * child_origin_bt.z + origin_bt.x,
        (axis_x.y * child_origin_bt.x + axis_y.y * child_origin_bt.y) + axis_z.y * child_origin_bt.z + origin_bt.y,
        (axis_x.z * child_origin_bt.x + axis_y.z * child_origin_bt.y) + axis_z.z * child_origin_bt.z + origin_bt.z);
    const ::rocketgpu::Vec3 extent_now(
        Abs(axis_x.x) * compound_half_bt.x + Abs(axis_y.x) * compound_half_bt.y + Abs(axis_z.x) * compound_half_bt.z,
        Abs(axis_x.y) * compound_half_bt.x + Abs(axis_y.y) * compound_half_bt.y + Abs(axis_z.y) * compound_half_bt.z,
        Abs(axis_x.z) * compound_half_bt.x + Abs(axis_y.z) * compound_half_bt.y + Abs(axis_z.z) * compound_half_bt.z);
    const ::rocketgpu::Vec3 lmin = child_origin_bt - compound_half_bt, lmax = child_origin_bt + compound_half_bt;
    const ::rocketgpu::Vec3 lc = (lmax + lmin) * 0.5f, lh = (lmax - lmin) * 0.5f;
    const auto compound_center = [&lc](const ::rocketgpu::Vec3& o, const ::rocketgpu::Vec3& ax,
                                       const ::rocketgpu::Vec3& ay, const ::rocketgpu::Vec3& az) {
        return ::rocketgpu::Vec3((ax.x * lc.x + ay.x * lc.y) + az.x * lc.z + o.x,
                                 (ax.y * lc.x + ay.y * lc.y) + az.y * lc.z + o.y,
                                 (ax.z * lc.x + ay.z * lc.y) + az.z * lc.z + o.z);
    };
    const auto compound_extent = [&lh](const ::rocketgpu::Vec3& ax, const ::rocketgpu::Vec3& ay,
                                       const ::rocketgpu::Vec3& az) {
        return ::rocketgpu::Vec3((Abs(ax.x) * lh.x + Abs(ay.x) * lh.y) + Abs(az.x) * lh.z,
                                 (Abs(ax.y) * lh.x + Abs(ay.y) * lh.y) + Abs(az.y) * lh.z,
                                 (Abs(ax.z) * lh.x + Abs(ay.z) * lh.y) + Abs(az.z) * lh.z);
    };
    const ::rocketgpu::Vec3 outer_now = compound_center(origin_bt, axis_x, axis_y, axis_z);
    const ::rocketgpu::Vec3 outer_pred = compound_center(predicted_origin_bt, pred_axis_x, pred_axis_y, pred_axis_z);
    const ::rocketgpu::Vec3 outer_ext_now = compound_extent(axis_x, axis_y, axis_z);
    const ::rocketgpu::Vec3 outer_ext_pred = compound_extent(pred_axis_x, pred_axis_y, pred_axis_z);
    proxy.aabb_min = ::rocketgpu::Vec3(
        Min(outer_now.x - outer_ext_now.x, outer_pred.x - outer_ext_pred.x) - kContactThresholdBt,
        Min(outer_now.y - outer_ext_now.y, outer_pred.y - outer_ext_pred.y) - kContactThresholdBt,
        Min(outer_now.z - outer_ext_now.z, outer_pred.z - outer_ext_pred.z) - kContactThresholdBt);
    proxy.aabb_max = ::rocketgpu::Vec3(
        Max(outer_now.x + outer_ext_now.x, outer_pred.x + outer_ext_pred.x) + kContactThresholdBt,
        Max(outer_now.y + outer_ext_now.y, outer_pred.y + outer_ext_pred.y) + kContactThresholdBt,
        Max(outer_now.z + outer_ext_now.z, outer_pred.z + outer_ext_pred.z) + kContactThresholdBt);
    proxy.child_aabb_min = ::rocketgpu::Vec3(
        center_now.x - extent_now.x,
        center_now.y - extent_now.y,
        center_now.z - extent_now.z);
    proxy.child_aabb_max = ::rocketgpu::Vec3(
        center_now.x + extent_now.x,
        center_now.y + extent_now.y,
        center_now.z + extent_now.z);
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

#if defined(ROCKETGPU_HOST_LANES)
ROCKETGPU_HOST_INLINE inline void BulletCarProxyFromBases(
    const ::rocketgpu::CarState& car,
    const ::rocketgpu::Vec3& child_origin_bt,
    const ::rocketgpu::Vec3& half_extent_bt,
    const ::rocketgpu::Vec3& predict_vel_bt,
    float dt,
    const ::rocketgpu::Vec3& axis_x,
    const ::rocketgpu::Vec3& axis_y,
    const ::rocketgpu::Vec3& axis_z,
    const ::rocketgpu::Vec3& pred_axis_x,
    const ::rocketgpu::Vec3& pred_axis_y,
    const ::rocketgpu::Vec3& pred_axis_z,
    BulletBroadphaseProxy& proxy) {
    constexpr float kContactThresholdBt = 0.02f;
    using namespace lanes;
    const ::rocketgpu::Vec3 origin_bt = car.pos_bt;
    const F4 origin = LoadPadded(car.pos_bt);
    const F4 pred_origin = origin + Load(predict_vel_bt) * Splat(dt);
    const F4 lx = Splat(child_origin_bt.x), ly = Splat(child_origin_bt.y), lz = Splat(child_origin_bt.z);
    const F4 hx = Splat(half_extent_bt.x), hy = Splat(half_extent_bt.y), hz = Splat(half_extent_bt.z);
    const F4 ax = Load(axis_x), ay = Load(axis_y), az = Load(axis_z);
    const F4 px = Load(pred_axis_x), py = Load(pred_axis_y), pz = Load(pred_axis_z);
    const F4 center_now = ((ax * lx + ay * ly) + az * lz) + origin;
    const F4 extent_now = (Abs(ax) * hx + Abs(ay) * hy) + Abs(az) * hz;
    const F4 child_min = center_now - extent_now;
    const F4 child_max = center_now + extent_now;
    const F4 thr = Splat(kContactThresholdBt);
    const ::rocketgpu::Vec3 lmin = child_origin_bt - half_extent_bt, lmax = child_origin_bt + half_extent_bt;
    const ::rocketgpu::Vec3 lc = (lmax + lmin) * 0.5f, lh = (lmax - lmin) * 0.5f;
    const F4 cx = Splat(lc.x), cy = Splat(lc.y), cz = Splat(lc.z);
    const F4 ex = Splat(lh.x), ey = Splat(lh.y), ez = Splat(lh.z);
    const F4 outer_now = ((ax * cx + ay * cy) + az * cz) + origin;
    const F4 outer_pred = ((px * cx + py * cy) + pz * cz) + pred_origin;
    const F4 outer_ext_now = (Abs(ax) * ex + Abs(ay) * ey) + Abs(az) * ez;
    const F4 outer_ext_pred = (Abs(px) * ex + Abs(py) * ey) + Abs(pz) * ez;
    static_assert(offsetof(BulletBroadphaseProxy, child_axis) == 5 * sizeof(::rocketgpu::Vec3) &&
                      offsetof(::rocketgpu::CarState, pos_bt) + 16 <= sizeof(::rocketgpu::CarState),
                  "padded proxy stores / car position load");
    StorePadded(proxy.aabb_min, Min(outer_now - outer_ext_now, outer_pred - outer_ext_pred) - thr);
    StorePadded(proxy.aabb_max, Max(outer_now + outer_ext_now, outer_pred + outer_ext_pred) + thr);
    StorePadded(proxy.child_aabb_min, child_min);
    StorePadded(proxy.child_aabb_max, child_max);
    StorePadded(proxy.child_center, center_now);
    proxy.child_axis[0] = axis_x;
    proxy.child_axis[1] = axis_y;
    proxy.child_axis[2] = axis_z;
    proxy.child_half = half_extent_bt;
    proxy.child_local_min = lmin;
    proxy.child_local_max = lmax;
    proxy.origin = origin_bt;
    proxy.valid = 1;
    proxy.is_ball = 0;
}

#endif
ROCKETGPU_HD ROCKETGPU_HOST_INLINE void ComputeBulletBodyProxy(
    int body,
    const ::rocketgpu::BallState& ball,
    CarCSpanT cars,
    const TickCars& tcars,
    const ::rocketgpu::CarConfig* car_configs,
    float ball_radius,
    float dt,
    const WorldSolverScratch* bp_history,
    const WorldTickScratch* tick,
    bool native_sse_math,
    BulletBroadphaseProxy& proxy,
    const CarShapeCacheEntry* car_shapes = nullptr) {
    const PreContactCar* const pre_contact_cars = tcars.pre_contact_cars();
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kContactThresholdBt = 0.02f;
    constexpr float kSphereAabbExtraBt = 0.08f;
    if (body == 0) {
        const ::rocketgpu::Vec3 pos_bt = ball.pos_bt;
        const ::rocketgpu::Vec3 predict_vel_bt = tick != nullptr ? tick->ball_predict_vel_bt : ball.vel_bt;
        const ::rocketgpu::Vec3 predicted_bt = pos_bt + predict_vel_bt * dt;
        const float extent = ball_radius * kUUToBT + kSphereAabbExtraBt;
        const float proxy_extent = (ball_radius * kUUToBT + 0.5f * kSphereAabbExtraBt) + 0.5f * kSphereAabbExtraBt;
        proxy.aabb_min = ::rocketgpu::Vec3(
            Min(pos_bt.x, predicted_bt.x) - proxy_extent - kContactThresholdBt,
            Min(pos_bt.y, predicted_bt.y) - proxy_extent - kContactThresholdBt,
            Min(pos_bt.z, predicted_bt.z) - proxy_extent - kContactThresholdBt);
        proxy.aabb_max = ::rocketgpu::Vec3(
            Max(pos_bt.x, predicted_bt.x) + proxy_extent + kContactThresholdBt,
            Max(pos_bt.y, predicted_bt.y) + proxy_extent + kContactThresholdBt,
            Max(pos_bt.z, predicted_bt.z) + proxy_extent + kContactThresholdBt);
        proxy.child_aabb_min = ::rocketgpu::Vec3(pos_bt.x - extent, pos_bt.y - extent, pos_bt.z - extent);
        proxy.child_aabb_max = ::rocketgpu::Vec3(pos_bt.x + extent, pos_bt.y + extent, pos_bt.z + extent);
        proxy.child_center = pos_bt;
        proxy.child_half = ::rocketgpu::Vec3(extent, extent, extent);
        proxy.valid = 1;
        proxy.is_ball = 1;
        return;
    }
    const int car_idx = body - 1;

    CarCRefT car = cars[car_idx];
    const bool demoed_before_contacts =
        pre_contact_cars != nullptr ? pre_contact_cars[car_idx].is_demoed != 0 : car.is_demoed != 0;
    if (demoed_before_contacts) {
        proxy.valid = 0;
        return;
    }
    const ::rocketgpu::CarConfig& cfg = car_configs[car_idx];
    const ::rocketgpu::Vec3 half_extent_bt =
        car_shapes != nullptr ? car_shapes[car_idx].half_extent_bt : BulletBoxHalfExtentWithMarginBt(cfg);
    const ::rocketgpu::Vec3 child_origin_bt = cfg.hitbox_offset * kUUToBT;

    const bool predict_snap = tick != nullptr && car_idx < 8 && tcars.car_predict_valid()[car_idx] != 0;
    const ::rocketgpu::Vec3 predict_vel_bt = predict_snap ? tcars.car_predict_vel_bt()[car_idx] : car.vel_bt;
    const ::rocketgpu::Vec3 predict_ang_vel = predict_snap ? tcars.car_predict_ang_vel()[car_idx] : car.ang_vel;
    ::rocketgpu::Vec3 axis_x(1.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 axis_y(0.0f, 1.0f, 0.0f);
    ::rocketgpu::Vec3 axis_z(0.0f, 0.0f, 1.0f);
    ::rocketgpu::Vec3 pred_axis_x(1.0f, 0.0f, 0.0f);
    ::rocketgpu::Vec3 pred_axis_y(0.0f, 1.0f, 0.0f);
    ::rocketgpu::Vec3 pred_axis_z(0.0f, 0.0f, 1.0f);
    if (car.basis_authoritative) {
        axis_x = car.basis_forward;
        axis_y = car.basis_right;
        axis_z = car.basis_up;
        pred_axis_x = axis_x;
        pred_axis_y = axis_y;
        pred_axis_z = axis_z;
        ::rocketgpu::Quat unused_rot{};
        if (!native_sse_math) {
            const ::rocketgpu::Quat* basis_rot = nullptr;
#if !defined(ROCKETGPU_USE_CUDA)
            basis_rot = CarRbSnapRaw(bp_history, car_idx, car);
#endif
            GameIntegrateOrientationBasisColumns(
                pred_axis_x, pred_axis_y, pred_axis_z, unused_rot, predict_ang_vel, dt, basis_rot);
        } else {
            IntegrateOrientationBasisColumns(
                pred_axis_x, pred_axis_y, pred_axis_z, unused_rot, predict_ang_vel, dt, native_sse_math);
        }
    } else {
        ::rocketgpu::Quat predicted_rot = car.rot;
        GetBasisVectors(predicted_rot, axis_x, axis_y, axis_z);
        IntegrateOrientation(predicted_rot, predict_ang_vel, dt);
        GetBasisVectors(predicted_rot, pred_axis_x, pred_axis_y, pred_axis_z);
    }
    const ::rocketgpu::Vec3 origin_bt = car.pos_bt;
#if defined(ROCKETGPU_HOST_LANES)
    BulletCarProxyFromBases(car, child_origin_bt, half_extent_bt, predict_vel_bt, dt, axis_x, axis_y, axis_z, pred_axis_x,
                            pred_axis_y, pred_axis_z, proxy);
    return;
#endif
    const ::rocketgpu::Vec3 predicted_origin_bt = origin_bt + predict_vel_bt * dt;

    const auto box_center = [](const ::rocketgpu::Vec3& origin,
                               const ::rocketgpu::Vec3& ax,
                               const ::rocketgpu::Vec3& ay,
                               const ::rocketgpu::Vec3& az,
                               const ::rocketgpu::Vec3& local) {
        return ::rocketgpu::Vec3(
            (ax.x * local.x + ay.x * local.y) + az.x * local.z + origin.x,
            (ax.y * local.x + ay.y * local.y) + az.y * local.z + origin.y,
            (ax.z * local.x + ay.z * local.y) + az.z * local.z + origin.z);
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
    const ::rocketgpu::Vec3 extent_now = box_extent(axis_x, axis_y, axis_z, half_extent_bt);
    proxy.child_aabb_min = ::rocketgpu::Vec3(
        center_now.x - extent_now.x,
        center_now.y - extent_now.y,
        center_now.z - extent_now.z);
    proxy.child_aabb_max = ::rocketgpu::Vec3(
        center_now.x + extent_now.x,
        center_now.y + extent_now.y,
        center_now.z + extent_now.z);
    const ::rocketgpu::Vec3 lmin = child_origin_bt - half_extent_bt, lmax = child_origin_bt + half_extent_bt;
    const ::rocketgpu::Vec3 lc = (lmax + lmin) * 0.5f, lh = (lmax - lmin) * 0.5f;
    const auto compound_center = [](const ::rocketgpu::Vec3& origin,
                                    const ::rocketgpu::Vec3& ax,
                                    const ::rocketgpu::Vec3& ay,
                                    const ::rocketgpu::Vec3& az,
                                    const ::rocketgpu::Vec3& local) {
        return ::rocketgpu::Vec3(
            (ax.x * local.x + ay.x * local.y) + az.x * local.z + origin.x,
            (ax.y * local.x + ay.y * local.y) + az.y * local.z + origin.y,
            (ax.z * local.x + ay.z * local.y) + az.z * local.z + origin.z);
    };
    const ::rocketgpu::Vec3 outer_now = compound_center(origin_bt, axis_x, axis_y, axis_z, lc);
    const ::rocketgpu::Vec3 outer_pred = compound_center(predicted_origin_bt, pred_axis_x, pred_axis_y, pred_axis_z, lc);
    const ::rocketgpu::Vec3 outer_ext_now = box_extent(axis_x, axis_y, axis_z, lh);
    const ::rocketgpu::Vec3 outer_ext_pred = box_extent(pred_axis_x, pred_axis_y, pred_axis_z, lh);
    proxy.aabb_min = ::rocketgpu::Vec3(
        Min(outer_now.x - outer_ext_now.x, outer_pred.x - outer_ext_pred.x) - kContactThresholdBt,
        Min(outer_now.y - outer_ext_now.y, outer_pred.y - outer_ext_pred.y) - kContactThresholdBt,
        Min(outer_now.z - outer_ext_now.z, outer_pred.z - outer_ext_pred.z) - kContactThresholdBt);
    proxy.aabb_max = ::rocketgpu::Vec3(
        Max(outer_now.x + outer_ext_now.x, outer_pred.x + outer_ext_pred.x) + kContactThresholdBt,
        Max(outer_now.y + outer_ext_now.y, outer_pred.y + outer_ext_pred.y) + kContactThresholdBt,
        Max(outer_now.z + outer_ext_now.z, outer_pred.z + outer_ext_pred.z) + kContactThresholdBt);
    proxy.child_center = center_now;
    proxy.child_axis[0] = axis_x;
    proxy.child_axis[1] = axis_y;
    proxy.child_axis[2] = axis_z;
    proxy.child_half = half_extent_bt;
#if !defined(ROCKETGPU_USE_CUDA)
    proxy.child_local_min = lmin;
    proxy.child_local_max = lmax;
    proxy.origin = origin_bt;
#endif
    proxy.valid = 1;
    proxy.is_ball = 0;
}

ROCKETGPU_HD void BulletSweepAddStatics(BulletSweepHistory& h, bool arena_static, bool goal_floors,
                                       const ArenaMeshBvhMeshInfo* mesh_infos, int mesh_info_count,
                                       bool planes = true) {
    h.sap_next_fresh = 1;
    h.sap_free_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
        h.sap_edge_pos[axis][0] = 0;
        h.sap_edge_handle[axis][0] = 0;
        h.sap_edge_pos[axis][1] = 0xffff;
        h.sap_edge_handle[axis][1] = 0;
        h.sap_min_edge[0][axis] = 0;
        h.sap_max_edge[0][axis] = 1;
    }
    const auto sort_down = [&](int axis, int index, bool is_max_edge) {
        const int moving = h.sap_edge_handle[axis][index];
        while (index > 0 && h.sap_edge_pos[axis][index] < h.sap_edge_pos[axis][index - 1]) {
            const int prev = h.sap_edge_handle[axis][index - 1];
            const bool prev_is_max = (h.sap_edge_pos[axis][index - 1] & 1) != 0;
            ++(prev_is_max ? h.sap_max_edge[prev][axis] : h.sap_min_edge[prev][axis]);
            --(is_max_edge ? h.sap_max_edge[moving][axis] : h.sap_min_edge[moving][axis]);
            const uint16_t pos = h.sap_edge_pos[axis][index];
            const uint8_t owner = h.sap_edge_handle[axis][index];
            h.sap_edge_pos[axis][index] = h.sap_edge_pos[axis][index - 1];
            h.sap_edge_handle[axis][index] = h.sap_edge_handle[axis][index - 1];
            h.sap_edge_pos[axis][index - 1] = pos;
            h.sap_edge_handle[axis][index - 1] = owner;
            --index;
        }
    };
    const auto add_handle = [&](const ::rocketgpu::Vec3& aabb_min, const ::rocketgpu::Vec3& aabb_max, int16_t id) {
        const float mn[3] = {aabb_min.x, aabb_min.y, aabb_min.z};
        const float mx[3] = {aabb_max.x, aabb_max.y, aabb_max.z};
        const int handle = h.sap_next_fresh++;
        const int limit = 2 * ++h.sap_handle_count;
        h.sap_static_id[handle] = id;
        for (int axis = 0; axis < 3; ++axis) {
            h.sap_max_edge[0][axis] = static_cast<uint8_t>(h.sap_max_edge[0][axis] + 2);
            h.sap_edge_pos[axis][limit + 1] = h.sap_edge_pos[axis][limit - 1];
            h.sap_edge_handle[axis][limit + 1] = h.sap_edge_handle[axis][limit - 1];
            h.sap_edge_pos[axis][limit - 1] = ArenaSweepQuantize(mn[axis], 0);
            h.sap_edge_handle[axis][limit - 1] = static_cast<uint8_t>(handle);
            h.sap_edge_pos[axis][limit] = ArenaSweepQuantize(mx[axis], 1);
            h.sap_edge_handle[axis][limit] = static_cast<uint8_t>(handle);
            h.sap_min_edge[handle][axis] = static_cast<uint8_t>(limit - 1);
            h.sap_max_edge[handle][axis] = static_cast<uint8_t>(limit);
        }
        for (int axis = 0; axis < 3; ++axis) {
            sort_down(axis, h.sap_min_edge[handle][axis], false);
            sort_down(axis, h.sap_max_edge[handle][axis], true);
        }
    };
    for (int k = goal_floors ? 0 : 2; arena_static && planes && k < 6; ++k) {
        const int plane_rank = k < 2 ? 4 + k : k - 2;
        ::rocketgpu::Vec3 shape_min(0.0f, 0.0f, 0.0f), shape_max(0.0f, 0.0f, 0.0f);
        BulletStaticPlaneShapeAabb(plane_rank, shape_min, shape_max);
        add_handle(shape_min, shape_max, static_cast<int16_t>(100 + plane_rank));
    }
    for (int slot = 0; arena_static && slot < mesh_info_count; ++slot) {
        const ArenaMeshBvhMeshInfo& info = mesh_infos[slot];
        if (info.mesh_index >= 0 && info.triangle_count > 0 &&
            h.sap_handle_count + kBulletSweepMaxBodies < kAxisSweepMaxHandles) {
            add_handle(info.aabb_min, info.aabb_max, static_cast<int16_t>(info.mesh_index));
        }
    }
    h.sap_static_count = h.sap_handle_count;
}

ROCKETGPU_HD void BulletSweepCopyStatics(BulletSweepHistory& h, const BulletSweepHistory& s, bool whole) {
    if (whole) {
        struct alignas(16) Words {
            uint32_t v[4];
        };
        constexpr size_t end = offsetof(BulletSweepHistory, sap_static_id) + sizeof(BulletSweepHistory::sap_static_id);
        constexpr int words = static_cast<int>(end / sizeof(Words));
        static_assert(words * sizeof(Words) >= offsetof(BulletSweepHistory, sap_static_id), "tail is static ids");
#if defined(__CUDA_ARCH__)
        const auto* src = static_cast<const unsigned char*>(__builtin_assume_aligned(&s, 16));
        auto* dst = static_cast<unsigned char*>(__builtin_assume_aligned(&h, 16));
#else
        const auto* src = reinterpret_cast<const unsigned char*>(&s);
        auto* dst = reinterpret_cast<unsigned char*>(&h);
#endif
        for (int i = 0; i < words; i += 4) {
            Words w[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (i + j < words) {
                    memcpy(&w[j], src + (i + j) * sizeof(Words), sizeof(Words));
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (i + j < words) {
                    memcpy(dst + (i + j) * sizeof(Words), &w[j], sizeof(Words));
                }
            }
        }
        for (int k = static_cast<int>((words * sizeof(Words) - offsetof(BulletSweepHistory, sap_static_id)) / 2);
             k <= kAxisSweepMaxHandles; ++k) {
            h.sap_static_id[k] = s.sap_static_id[k];
        }
        return;
    }
    const int n = s.sap_handle_count;
    h.sap_handle_count = s.sap_handle_count;
    h.sap_static_count = s.sap_static_count;
    h.sap_next_fresh = s.sap_next_fresh;
    h.sap_free_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
        for (int e = 0; e < 2 * n + 2; ++e) {
            h.sap_edge_pos[axis][e] = s.sap_edge_pos[axis][e];
            h.sap_edge_handle[axis][e] = s.sap_edge_handle[axis][e];
        }
    }
    for (int k = 0; k <= n; ++k) {
        for (int axis = 0; axis < 3; ++axis) {
            h.sap_min_edge[k][axis] = s.sap_min_edge[k][axis];
            h.sap_max_edge[k][axis] = s.sap_max_edge[k][axis];
        }
        if (k > 0) {
            h.sap_static_id[k] = s.sap_static_id[k];
        }
    }
}

template <int kPhase = 0>
ROCKETGPU_HD int BuildBulletManifoldList(
    const ::rocketgpu::BallState& ball,
    int car_count,
    CarCSpanT cars,
    const TickCars& tcars,
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
    WorldTickScratch* tick = nullptr,
    bool native_sse_math = false,
    int8_t* body_islands = nullptr,
    bool proxies_ready = false,
    bool goal_floors = false,
    bool ball_static_pairs_skipped = false,
    const CarShapeCacheEntry* car_shapes = nullptr) {
    if (!out_entries || max_entries <= 0 || !grid.valid() || !mesh_infos || mesh_info_count <= 0 ||
        bp_history == nullptr || tick == nullptr) {
        return 0;
    }

    constexpr int kMaxBodies = 9;
    BulletBroadphaseProxy* const proxies = tick->w3_proxies;

    if constexpr (kPhase != 2) {
        if (!proxies_ready) {
            ComputeBulletBodyProxy(0, ball, cars, tcars, car_configs, ball_radius, dt, bp_history, tick, native_sse_math,
                                   proxies[0]);
#if defined(ROCKETGPU_HOST_LANES)
            const int proxy_cars = car_count < kMaxBodies - 1 ? car_count : kMaxBodies - 1;
            const PreContactCar* const pre_contact_cars = tcars.pre_contact_cars();
            ::rocketgpu::Vec3 pred_basis[kMaxBodies - 1][3];
            uint32_t predicted = 0u;
            if (!native_sse_math) {
                GameIntegrateLane lane[kMaxBodies - 1];
                int n = 0;
                for (int c = 0; c < proxy_cars; ++c) {
                    const ::rocketgpu::CarState& car = cars[c];
                    const bool demoed =
                        pre_contact_cars != nullptr ? pre_contact_cars[c].is_demoed != 0 : car.is_demoed != 0;
                    if (demoed || !car.basis_authoritative) {
                        continue;
                    }
                    GameIntegrateLane& l = lane[n++];
                    l.w = tcars.car_predict_valid()[c] != 0 ? tcars.car_predict_ang_vel()[c] : car.ang_vel;
                    const ::rocketgpu::Quat* raw = CarRbSnapRaw(bp_history, c, car);
                    l.o = raw != nullptr ? *raw : GameGetRotationScalar(car.basis_forward, car.basis_right, car.basis_up);
                    l.fwd = &pred_basis[c][0];
                    l.right = &pred_basis[c][1];
                    l.up = &pred_basis[c][2];
                    l.q = nullptr;
                    predicted |= 1u << c;
                }
                for (int k0 = 0; k0 < n; k0 += 4) {
                    const int m = n - k0 < 4 ? n - k0 : 4;
                    uint32_t left = GameIntegrateOrientationBasisColumns4(m, lane + k0, dt);
                    for (; left != 0u; left &= left - 1u) {
                        GameIntegrateLane& l = lane[k0 + CountTrailingZeros32(left)];
                        const int c = static_cast<int>(l.fwd - &pred_basis[0][0]) / 3;
                        *l.fwd = cars[c].basis_forward;
                        *l.right = cars[c].basis_right;
                        *l.up = cars[c].basis_up;
                        ::rocketgpu::Quat unused_rot{};
                        GameIntegrateOrientationBasisColumns(*l.fwd, *l.right, *l.up, unused_rot, l.w, dt, &l.o);
                    }
                }
            }
#pragma unroll 1
            for (int c = 0; c < proxy_cars; ++c) {
                if (((predicted >> c) & 1u) == 0u) {
                    ComputeBulletBodyProxy(c + 1, ball, cars, tcars, car_configs, ball_radius, dt, bp_history, tick,
                                           native_sse_math, proxies[c + 1], car_shapes);
                    continue;
                }
                const ::rocketgpu::CarState& car = cars[c];
                const ::rocketgpu::CarConfig& cfg = car_configs[c];
                const ::rocketgpu::Vec3 half_extent_bt =
                    car_shapes != nullptr ? car_shapes[c].half_extent_bt : BulletBoxHalfExtentWithMarginBt(cfg);
                const ::rocketgpu::Vec3 child_origin_bt = cfg.hitbox_offset * (1.0f / 50.0f);
                const ::rocketgpu::Vec3 predict_vel_bt =
                    tcars.car_predict_valid()[c] != 0 ? tcars.car_predict_vel_bt()[c] : car.vel_bt;
                BulletCarProxyFromBases(car, child_origin_bt, half_extent_bt, predict_vel_bt, dt, car.basis_forward,
                                        car.basis_right, car.basis_up, pred_basis[c][0], pred_basis[c][1],
                                        pred_basis[c][2], proxies[c + 1]);
            }
#else
#pragma unroll 1
            for (int car_idx = 0; car_idx < car_count && car_idx < kMaxBodies - 1; ++car_idx) {
                ComputeBulletBodyProxy(car_idx + 1, ball, cars, tcars, car_configs, ball_radius, dt, bp_history,
                                       tick, native_sse_math, proxies[car_idx + 1], car_shapes);
            }
#endif
        }
    }

    const int body_count = Min(car_count + 1, kMaxBodies);
    BulletBroadphaseHistory& bp = bp_history->bp;
    BulletSweepHistory& h = bp.sweep;

    const auto mesh_info_for = [&](int mesh_index) -> const ArenaMeshBvhMeshInfo* {
        if (mesh_index >= 0 && mesh_index < mesh_info_count && mesh_infos[mesh_index].mesh_index == mesh_index) {
            return mesh_infos[mesh_index].triangle_count > 0 ? &mesh_infos[mesh_index] : nullptr;
        }
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
    const auto compound_tree_overlap = [&](const BulletManifoldEntry& key) {
        const BulletBroadphaseProxy* pa = body_proxy(key.body_a + 1);
        const ArenaMeshBvhMeshInfo* info = mesh_info_for(key.static_id);
        if (pa == nullptr || info == nullptr) {
            return true;
        }
        const ::rocketgpu::Vec3* r = pa->child_axis;
#if defined(ROCKETGPU_USE_CUDA)
        const ::rocketgpu::Vec3& origin = cars[key.body_a].pos_bt;
        const ::rocketgpu::Vec3 child_origin_bt = car_configs[key.body_a].hitbox_offset * (1.0f / 50.0f);
        const ::rocketgpu::Vec3 child_local_min = child_origin_bt - pa->child_half;
        const ::rocketgpu::Vec3 child_local_max = child_origin_bt + pa->child_half;
        const ::rocketgpu::Vec3 no(-origin.x, -origin.y, -origin.z);
#else
        const ::rocketgpu::Vec3 no(-pa->origin.x, -pa->origin.y, -pa->origin.z);
#endif
        const ::rocketgpu::Vec3 c = (info->aabb_max + info->aabb_min) * 0.5f;
        const ::rocketgpu::Vec3 hh = (info->aabb_max - info->aabb_min) * 0.5f;
        for (int k = 0; k < 3; ++k) {
            const float io = (r[k].x * no.x + r[k].y * no.y) + r[k].z * no.z;
            const float ctr = (r[k].x * c.x + r[k].y * c.y) + r[k].z * c.z + io;
            const float ext = (hh.x * Abs(r[k].x) + hh.y * Abs(r[k].y)) + hh.z * Abs(r[k].z);
#if defined(ROCKETGPU_USE_CUDA)
            const float lo = (&child_local_min.x)[k], hi = (&child_local_max.x)[k];
#else
            const float lo = (&pa->child_local_min.x)[k], hi = (&pa->child_local_max.x)[k];
#endif
            if (!(ctr - ext <= hi && ctr + ext >= lo)) {
                return false;
            }
        }
        return true;
    };
    const auto local_overlap = [&](const BulletManifoldEntry& key) {
        int a = key.body_a + 1;
        int b = key.kind == 1 ? 0 : key.body_b + 1;
        if (key.kind == 3 && h.sap_body_handle[b] < h.sap_body_handle[a]) {
            const int t = a;
            a = b;
            b = t;
        }
        const BulletBroadphaseProxy* pa = body_proxy(a);
        const BulletBroadphaseProxy* pb = body_proxy(b);
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
    const auto find = [](const uint32_t* list, int n, uint32_t key) {
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            const uint32_t a = list[i], b = list[i + 1], c = list[i + 2], d = list[i + 3];
            if (a == key || b == key || c == key || d == key) {
                return a == key ? i : b == key ? i + 1 : c == key ? i + 2 : i + 3;
            }
        }
        for (; i < n; ++i) {
            if (list[i] == key) {
                return i;
            }
        }
        return -1;
    };
    const auto swap_remove = [](uint32_t* list, uint8_t& n, int i) {
        if (i >= 0) {
            list[i] = list[--n];
        }
    };

    const auto quantize_point = [&](const ::rocketgpu::Vec3& p, uint16_t* out, int is_max) {
        const float v[3] = {p.x, p.y, p.z};
        for (int axis = 0; axis < 3; ++axis) {
            out[axis] = ArenaSweepQuantize(v[axis], is_max);
        }
    };
    const auto sap_key = [&](int ha, int hb, uint32_t& key) {
        if (ha > hb) {
            const int t = ha;
            ha = hb;
            hb = t;
        }
        if (h.sap_static_id[hb] >= 0) {
            return false;
        }
        int body_b = -1 - h.sap_static_id[hb];
        if (body_b < 0 || body_b >= body_count) {
            return false;
        }
        if (h.sap_static_id[ha] >= 0) {
            key = PackBulletPairKey(body_b == 0 ? 0 : 2, body_b - 1, -1, h.sap_static_id[ha]);
        } else {
            int body_a = -1 - h.sap_static_id[ha];
            if (body_a > body_b) {
                const int t = body_a;
                body_a = body_b;
                body_b = t;
            }
            key = PackBulletPairKey(body_a == 0 ? 1 : 3, body_a == 0 ? body_b - 1 : body_a - 1,
                                    body_a == 0 ? -1 : body_b - 1, -1);
        }
        return true;
    };
    const auto sap_add_pair = [&](int ha, int hb) {
        uint32_t key = 0;
        if (sap_key(ha, hb, key) && find(bp.bp_pairs, bp.bp_pair_count, key) < 0 &&
            bp.bp_pair_count < kMaxBulletManifoldEntries) {
            bp.bp_pairs[bp.bp_pair_count++] = key;
        }
    };
    const auto sap_remove_pair = [&](int ha, int hb) {
        uint32_t key = 0;
        if (!sap_key(ha, hb, key)) {
            return;
        }
        const int at = find(bp.bp_pairs, bp.bp_pair_count, key);
        if (at >= 0) {
            swap_remove(bp.bp_pairs, bp.bp_pair_count, at);
            if constexpr (kPhase != 1) {
                swap_remove(bp.bp_manifolds, bp.bp_manifold_count, find(bp.bp_manifolds, bp.bp_manifold_count, key));
            } else if (find(bp.bp_manifolds, bp.bp_manifold_count, key) >= 0 &&
                       find(tick->w3_removed, tick->w3_removed_count, key) < 0) {
                tick->w3_removed[tick->w3_removed_count++] = key;
            }
        }
    };
    const auto overlap_2d = [&](int a, int b, int axis) {
        const int axis1 = (1 << axis) & 3;
        const int axis2 = (1 << axis1) & 3;
        return !(h.sap_max_edge[a][axis1] < h.sap_min_edge[b][axis1] ||
                 h.sap_max_edge[b][axis1] < h.sap_min_edge[a][axis1] ||
                 h.sap_max_edge[a][axis2] < h.sap_min_edge[b][axis2] ||
                 h.sap_max_edge[b][axis2] < h.sap_min_edge[a][axis2]);
    };
    const auto sort_down = [&](int axis, int index, bool is_max_edge, bool update) {
        const uint8_t moving_owner = h.sap_edge_handle[axis][index];
        const int moving = moving_owner;
        const uint16_t moving_pos = h.sap_edge_pos[axis][index];
        const int start_index = index;
        while (index > 0 && moving_pos < h.sap_edge_pos[axis][index - 1]) {
            const uint8_t prev_owner = h.sap_edge_handle[axis][index - 1];
            const uint16_t prev_pos = h.sap_edge_pos[axis][index - 1];
            const int prev = prev_owner;
            const bool prev_is_max = (prev_pos & 1) != 0;
            if (update && prev_is_max != is_max_edge && overlap_2d(moving, prev, axis)) {
                if (is_max_edge) {
                    sap_remove_pair(moving, prev);
                } else {
                    sap_add_pair(moving, prev);
                }
            }
            ++(prev_is_max ? h.sap_max_edge[prev][axis] : h.sap_min_edge[prev][axis]);
            --(is_max_edge ? h.sap_max_edge[moving][axis] : h.sap_min_edge[moving][axis]);
            h.sap_edge_pos[axis][index] = prev_pos;
            h.sap_edge_handle[axis][index] = prev_owner;
            --index;
        }
        if (index != start_index) {
            h.sap_edge_pos[axis][index] = moving_pos;
            h.sap_edge_handle[axis][index] = moving_owner;
        }
    };
    const auto sort_up = [&](int axis, int index, bool is_max_edge, bool update) {
        const uint8_t moving_owner = h.sap_edge_handle[axis][index];
        const int moving = moving_owner;
        const uint16_t moving_pos = h.sap_edge_pos[axis][index];
        const int start_index = index;
        const int len = 2 * h.sap_handle_count + 2;
        while (index + 1 < len && h.sap_edge_handle[axis][index + 1] != 0 &&
               moving_pos >= h.sap_edge_pos[axis][index + 1]) {
            const uint8_t next_owner = h.sap_edge_handle[axis][index + 1];
            const uint16_t next_pos = h.sap_edge_pos[axis][index + 1];
            const int next = next_owner;
            const bool next_is_max = (next_pos & 1) != 0;
            if (update && next_is_max != is_max_edge && overlap_2d(moving, next, axis)) {
                if (is_max_edge) {
                    sap_add_pair(moving, next);
                } else {
                    sap_remove_pair(moving, next);
                }
            }
            --(next_is_max ? h.sap_max_edge[next][axis] : h.sap_min_edge[next][axis]);
            ++(is_max_edge ? h.sap_max_edge[moving][axis] : h.sap_min_edge[moving][axis]);
            h.sap_edge_pos[axis][index] = next_pos;
            h.sap_edge_handle[axis][index] = next_owner;
            ++index;
        }
        if (index != start_index) {
            h.sap_edge_pos[axis][index] = moving_pos;
            h.sap_edge_handle[axis][index] = moving_owner;
        }
    };
    const auto sort_down_far = [&](int axis, int index, int moving, uint16_t pos, bool is_max_edge, bool update) {
        constexpr int kChunk = 4;
        for (bool more = true; more;) {
            uint16_t p[kChunk];
            uint8_t o[kChunk];
            int n = 0;
#pragma unroll
            for (int k = 0; k < kChunk; ++k) {
                const int j = index - 1 - k;
                p[k] = j >= 0 ? h.sap_edge_pos[axis][j] : 0;
                o[k] = j >= 0 ? h.sap_edge_handle[axis][j] : 0;
                n += n == k && pos < p[k];
            }
            uint8_t c[kChunk];
            bool ov[kChunk];
#pragma unroll
            for (int k = 0; k < kChunk; ++k) {
                if (k < n) {
                    c[k] = (p[k] & 1) ? h.sap_max_edge[o[k]][axis] : h.sap_min_edge[o[k]][axis];
                    ov[k] = update && ((p[k] & 1) != 0) != is_max_edge && overlap_2d(moving, o[k], axis);
                }
            }
#pragma unroll
            for (int k = 0; k < kChunk; ++k) {
                if (k < n && ov[k]) {
                    if (is_max_edge) {
                        sap_remove_pair(moving, o[k]);
                    } else {
                        sap_add_pair(moving, o[k]);
                    }
                }
            }
#pragma unroll
            for (int k = 0; k < kChunk; ++k) {
                if (k < n) {
                    ((p[k] & 1) ? h.sap_max_edge[o[k]][axis] : h.sap_min_edge[o[k]][axis]) =
                        static_cast<uint8_t>(c[k] + 1);
                    h.sap_edge_pos[axis][index - k] = p[k];
                    h.sap_edge_handle[axis][index - k] = o[k];
                }
            }
            index -= n;
            more = n == kChunk;
        }
        h.sap_edge_pos[axis][index] = pos;
        h.sap_edge_handle[axis][index] = static_cast<uint8_t>(moving);
        (is_max_edge ? h.sap_max_edge[moving][axis] : h.sap_min_edge[moving][axis]) = static_cast<uint8_t>(index);
    };
    const auto add_handle = [&](const ::rocketgpu::Vec3& aabb_min, const ::rocketgpu::Vec3& aabb_max, int16_t id) {
        uint16_t qmin[3], qmax[3];
        quantize_point(aabb_min, qmin, 0);
        quantize_point(aabb_max, qmax, 1);
        const int handle = h.sap_free_count > 0 ? h.sap_free[--h.sap_free_count] : h.sap_next_fresh++;
        const int limit = 2 * ++h.sap_handle_count;
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
            sort_down_far(axis, limit - 1, handle, qmin[axis], false, axis == 2);
            sort_down_far(axis, limit, handle, qmax[axis], true, axis == 2);
        }
    };
    if constexpr (kPhase != 2) {
    const bool sweep_fresh = h.sap_handle_count == 0;
    if constexpr (kPhase == 1) {
        tick->w3_removed_count = 0;
    }
    const BulletSweepHistory* statics_whole = nullptr;
    if (sweep_fresh) {
        if (arena_static && grid.sweep_statics) {
            bool whole = h.sap_obj_count == 0;
            for (int body = 0; whole && body < kBulletSweepMaxBodies; ++body) {
                whole = h.sap_body_handle[body] == 0;
            }
            BulletSweepCopyStatics(h, grid.sweep_statics[goal_floors ? 1 : 0], whole);
            statics_whole = whole ? &grid.sweep_statics[goal_floors ? 1 : 0] : nullptr;
        } else {
            BulletSweepAddStatics(h, arena_static, goal_floors, mesh_infos, mesh_info_count);
        }
    }
    for (int body = 1; body < body_count; ++body) {
        if (!proxies[body].valid && h.sap_body_handle[body] != 0) {
            BulletSweepRemoveBody(bp, body);
        }
    }
    const auto add_body = [&](int body) {
        const BulletBroadphaseProxy& proxy = proxies[body];
        if (proxy.valid && h.sap_body_handle[body] == 0 && h.sap_handle_count < kAxisSweepMaxHandles - 1) {
            h.sap_body_handle[body] = h.sap_free_count > 0 ? h.sap_free[h.sap_free_count - 1] : h.sap_next_fresh;
            add_handle(proxy.child_aabb_min, proxy.child_aabb_max, static_cast<int16_t>(-1 - body));
            h.sap_obj[h.sap_obj_count++] = static_cast<uint8_t>(body);
        }
    };
    const auto add_bodies_merged = [&](const BulletSweepHistory& s) {
        uint8_t new_handle[kBulletSweepMaxBodies];
        uint8_t new_body[kBulletSweepMaxBodies];
        uint16_t q[kBulletSweepMaxBodies][2][3];
        const int old_count = 2 * h.sap_handle_count + 2;
        int n = 0;
        for (int body = 0; body < body_count; ++body) {
            const BulletBroadphaseProxy& proxy = proxies[body];
            if (!proxy.valid || h.sap_handle_count >= kAxisSweepMaxHandles - 1) {
                continue;
            }
            const int handle = h.sap_free_count > 0 ? h.sap_free[--h.sap_free_count] : h.sap_next_fresh++;
            ++h.sap_handle_count;
            h.sap_body_handle[body] = static_cast<uint8_t>(handle);
            h.sap_static_id[handle] = static_cast<int16_t>(-1 - body);
            h.sap_obj[h.sap_obj_count++] = static_cast<uint8_t>(body);
            quantize_point(proxy.child_aabb_min, q[n][0], 0);
            quantize_point(proxy.child_aabb_max, q[n][1], 1);
            new_handle[n] = static_cast<uint8_t>(handle);
            new_body[n++] = static_cast<uint8_t>(body);
        }
        const int new_count = 2 * n;
        for (int axis = 0; axis < 3; ++axis) {
            uint32_t key[2 * kBulletSweepMaxBodies];
            for (int e = 0; e < new_count; ++e) {
                const uint32_t v = (static_cast<uint32_t>(q[e >> 1][e & 1][axis]) << 5) | static_cast<uint32_t>(e);
                int j = e;
                for (; j > 0 && key[j - 1] < v; --j) {
                    key[j] = key[j - 1];
                }
                key[j] = v;
            }
            int old = old_count - 1;
            int dst = old_count + new_count - 1;
            h.sap_edge_pos[axis][dst] = s.sap_edge_pos[axis][old];
            h.sap_edge_handle[axis][dst] = s.sap_edge_handle[axis][old];
            h.sap_max_edge[s.sap_edge_handle[axis][old]][axis] = static_cast<uint8_t>(dst);
            --old;
            --dst;
            for (int e = 0; e < new_count; --dst) {
                uint16_t pos = s.sap_edge_pos[axis][old];
                uint8_t handle;
                if (pos > static_cast<uint16_t>(key[e] >> 5)) {
                    handle = s.sap_edge_handle[axis][old--];
                } else {
                    pos = static_cast<uint16_t>(key[e] >> 5);
                    handle = new_handle[(key[e++] & 31u) >> 1];
                }
                h.sap_edge_pos[axis][dst] = pos;
                h.sap_edge_handle[axis][dst] = handle;
                ((pos & 1) ? h.sap_max_edge[handle][axis] : h.sap_min_edge[handle][axis]) = static_cast<uint8_t>(dst);
            }
        }
        for (int k = 0; k < n; ++k) {
            const int moving = new_handle[k];
            const int min_id = -1 - new_body[k];
            const uint8_t m0 = h.sap_min_edge[moving][0], m1 = h.sap_min_edge[moving][1];
            const uint8_t x0 = h.sap_max_edge[moving][0], x1 = h.sap_max_edge[moving][1];
            for (int is_max = 0; is_max < 2; ++is_max) {
                const uint16_t pos = q[k][is_max][2];
                for (int i = old_count + new_count - 2, more = 1; more; i -= 4) {
                    constexpr int kChunk = 4;
                    uint16_t p[kChunk];
                    uint8_t o[kChunk];
#pragma unroll
                    for (int c = 0; c < kChunk; ++c) {
                        p[c] = i - c >= 0 ? h.sap_edge_pos[2][i - c] : 0;
                        o[c] = i - c >= 0 ? h.sap_edge_handle[2][i - c] : 0;
                    }
                    int16_t id[kChunk];
                    uint8_t b0[kChunk], b1[kChunk], y0[kChunk], y1[kChunk];
#pragma unroll
                    for (int c = 0; c < kChunk; ++c) {
                        id[c] = h.sap_static_id[o[c]];
                        b0[c] = h.sap_min_edge[o[c]][0];
                        b1[c] = h.sap_min_edge[o[c]][1];
                        y0[c] = h.sap_max_edge[o[c]][0];
                        y1[c] = h.sap_max_edge[o[c]][1];
                    }
#pragma unroll
                    for (int c = 0; c < kChunk; ++c) {
                        more = more && p[c] > pos;
                        if (more && (p[c] & 1) != is_max && id[c] > min_id &&
                            !(x0 < b0[c] || y0[c] < m0 || x1 < b1[c] || y1[c] < m1)) {
                            if (is_max) {
                                sap_remove_pair(moving, o[c]);
                            } else {
                                sap_add_pair(moving, o[c]);
                            }
                        }
                    }
                }
            }
        }
    };
    bool round_cached = false;
    if (sweep_fresh) {
        bool empty = bp.bp_pair_count == 0 && bp.bp_manifold_count == 0 && h.sap_obj_count == 0;
        for (int body = 0; empty && body < body_count; ++body) {
            empty = h.sap_body_handle[body] == 0;
        }
        round_cached = !empty;
    }
    if (statics_whole != nullptr && !round_cached) {
        add_bodies_merged(*statics_whole);
    } else {
        for (int body = 0; body < body_count; ++body) {
            add_body(body);
        }
    }
    if (round_cached) {
        BulletSweepRemoveRoundBodies(bp, body_count - 1);
        for (int body = 0; body < body_count; ++body) {
            add_body(body);
        }
    }
#if defined(ROCKETGPU_HOST_LANES)
    const auto update_aabbs = [&]() {
        using namespace lanes;
        const auto quant = [](F4 v, int32_t is_max) {
            constexpr float kQ = 65535.0f / (kArenaSweepHalfExtentBt + kArenaSweepHalfExtentBt);
            F4 sc = (v + Splat(kArenaSweepHalfExtentBt)) * Splat(kQ);
            sc = Select(sc >= Splat(0.0f), Select(sc < Splat(65535.0f), sc, Splat(65535.0f)), Splat(0.0f));
            return (__builtin_convertvector(sc, I4) & I4{0xfffe, 0xfffe, 0xfffe, 0xfffe}) | I4{is_max, is_max, is_max, is_max};
        };
        I4 qmin_of[kBulletSweepMaxBodies], qmax_of[kBulletSweepMaxBodies];
        uint8_t handle_of[kBulletSweepMaxBodies];
        uint32_t moved = 0u;
        for (int obj = 0; obj < h.sap_obj_count; ++obj) {
            const int body = h.sap_obj[obj];
            if (body >= body_count) {
                continue;
            }
            const BulletBroadphaseProxy& proxy = proxies[body];
            const int handle = h.sap_body_handle[body];
            if (!proxy.valid || handle == 0) {
                continue;
            }
            BulletSweepStoreBodyAabb(h, body, proxy.aabb_min, proxy.aabb_max);
            const I4 kmin = quant(LoadPadded(proxy.aabb_min), 0);
            const I4 kmax = quant(LoadPadded(proxy.aabb_max), 1);
            const uint8_t* mn = h.sap_min_edge[handle];
            const uint8_t* mx = h.sap_max_edge[handle];
            const I4 omin{h.sap_edge_pos[0][mn[0]], h.sap_edge_pos[1][mn[1]], h.sap_edge_pos[2][mn[2]], 0};
            const I4 omax{h.sap_edge_pos[0][mx[0]], h.sap_edge_pos[1][mx[1]], h.sap_edge_pos[2][mx[2]], 0};
            const I4 ne = (kmin != omin) | (kmax != omax);
            const uint32_t axes = static_cast<uint32_t>((ne[0] & 1) | (ne[1] & 2) | (ne[2] & 4));
            qmin_of[obj] = kmin;
            qmax_of[obj] = kmax;
            handle_of[obj] = static_cast<uint8_t>(handle);
            moved |= axes << (3 * obj);
        }
        for (; moved != 0u; moved &= moved - 1u) {
            const int bit = CountTrailingZeros32(moved);
            const int obj = bit / 3;
            const int axis = bit - 3 * obj;
            const int handle = handle_of[obj];
            {
                const uint16_t qmin_axis = static_cast<uint16_t>(qmin_of[obj][axis]);
                const uint16_t qmax_axis = static_cast<uint16_t>(qmax_of[obj][axis]);
                const int emin = h.sap_min_edge[handle][axis];
                const int emax = h.sap_max_edge[handle][axis];
                const int dmin = static_cast<int>(qmin_axis) - static_cast<int>(h.sap_edge_pos[axis][emin]);
                const int dmax = static_cast<int>(qmax_axis) - static_cast<int>(h.sap_edge_pos[axis][emax]);
                uint16_t* const pos = h.sap_edge_pos[axis];
                const uint8_t* const owner = h.sap_edge_handle[axis];
                const int last = 2 * h.sap_handle_count + 1;
                pos[emin] = qmin_axis;
                pos[emax] = qmax_axis;
                const auto passes_down = [&](int e, int q, int d) {
                    return ((q - static_cast<int>(pos[e - (e > 0)])) & d) < 0;
                };
                const auto passes_up = [&](int e, int q, int d) {
                    const int next = e < last ? e + 1 : last;
                    const int next_owner = e < last ? static_cast<int>(owner[next]) : 0;
                    return ((-d) & (static_cast<int>(pos[next]) - q - 1) & (-next_owner)) < 0;
                };
                if (passes_down(emin, qmin_axis, dmin)) {
                    sort_down(axis, emin, false, true);
                }
                if (passes_up(emax, qmax_axis, dmax)) {
                    sort_up(axis, emax, true, true);
                }
                if (passes_up(emin, qmin_axis, dmin)) {
                    sort_up(axis, emin, false, true);
                }
                if (passes_down(emax, qmax_axis, dmax)) {
                    sort_down(axis, emax, true, true);
                }
            }
        }
    };
#else
    const auto update_aabbs = [&]() {
        for (int obj = 0; obj < h.sap_obj_count; ++obj) {
            const int body = h.sap_obj[obj];
            if (body >= body_count) {
                continue;
            }
            const BulletBroadphaseProxy& proxy = proxies[body];
            const int handle = h.sap_body_handle[body];
            if (!proxy.valid || handle == 0) {
                continue;
            }
            BulletSweepStoreBodyAabb(h, body, proxy.aabb_min, proxy.aabb_max);
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
                if (dmin != 0) {
                    h.sap_edge_pos[axis][emin] = qmin[axis];
                }
                if (dmax != 0) {
                    h.sap_edge_pos[axis][emax] = qmax[axis];
                }
#if defined(ROCKETGPU_HOST_LANES)
                {
                    const uint16_t* const pos = h.sap_edge_pos[axis];
                    const uint16_t min_prev = pos[emin > 0 ? emin - 1 : 0];
                    const uint16_t min_next = pos[emin + 1 < kAxisSweepMaxEdges ? emin + 1 : emin];
                    const uint16_t max_prev = pos[emax > 0 ? emax - 1 : 0];
                    const uint16_t max_next = pos[emax + 1 < kAxisSweepMaxEdges ? emax + 1 : emax];
                    const bool min_moves = ((dmin < 0) & (qmin[axis] < min_prev)) | ((dmin > 0) & (qmin[axis] >= min_next));
                    const bool max_moves = ((dmax < 0) & (qmax[axis] < max_prev)) | ((dmax > 0) & (qmax[axis] >= max_next));
                    if (!(min_moves | max_moves)) {
                        continue;
                    }
                }
#endif
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
    };
#endif
    if constexpr (kPhase == 1) {
        update_aabbs();
        return 0;
    } else {
    for (int i = 0; i < bp.bp_predictive_count; ++i) {
        swap_remove(bp.bp_manifolds, bp.bp_manifold_count,
                    find(bp.bp_manifolds, bp.bp_manifold_count, bp.bp_predictive[i]));
    }
    bp.bp_predictive_count = 0;
    for (int obj = 0; obj < h.sap_obj_count && tick->predictive_hit_mask != 0; ++obj) {
        const int body = h.sap_obj[obj];
        if (body >= body_count || ((tick->predictive_hit_mask >> body) & 1u) == 0 ||
            bp.bp_manifold_count >= kMaxBulletManifoldEntries) {
            continue;
        }
        const uint32_t key = PackBulletPairKey(body == 0 ? 0 : 2, body - 1, -1, kBulletPredictiveStaticId);
        bp.bp_manifolds[bp.bp_manifold_count++] = key;
        bp.bp_predictive[bp.bp_predictive_count++] = key;
    }
    update_aabbs();
    }
    } else {
    for (int i = 0; i < bp.bp_predictive_count; ++i) {
        swap_remove(bp.bp_manifolds, bp.bp_manifold_count,
                    find(bp.bp_manifolds, bp.bp_manifold_count, bp.bp_predictive[i]));
    }
    bp.bp_predictive_count = 0;
    for (int obj = 0; obj < h.sap_obj_count && tick->predictive_hit_mask != 0; ++obj) {
        const int body = h.sap_obj[obj];
        if (body >= body_count || ((tick->predictive_hit_mask >> body) & 1u) == 0 ||
            bp.bp_manifold_count >= kMaxBulletManifoldEntries) {
            continue;
        }
        const uint32_t key = PackBulletPairKey(body == 0 ? 0 : 2, body - 1, -1, kBulletPredictiveStaticId);
        bp.bp_manifolds[bp.bp_manifold_count++] = key;
        bp.bp_predictive[bp.bp_predictive_count++] = key;
    }
    for (int i = 0; i < tick->w3_removed_count; ++i) {
        swap_remove(bp.bp_manifolds, bp.bp_manifold_count,
                    find(bp.bp_manifolds, bp.bp_manifold_count, tick->w3_removed[i]));
    }
    }

    if constexpr (kPhase != 1) {
    for (int i = 0; i < bp.bp_pair_count; ++i) {
        const uint32_t packed = bp.bp_pairs[i];
        const BulletManifoldEntry key = UnpackBulletPairKey(packed);
        const int at = find(bp.bp_manifolds, bp.bp_manifold_count, packed);
        const bool want = key.kind == 0 ? body_proxy(0) != nullptr : overlap(key);
        if (want && at < 0 && (key.kind == 1 || key.kind == 3) && !local_overlap(key)) {
            continue;
        }
        if (want && at < 0 && key.kind == 2 && key.static_id < 100 && !compound_tree_overlap(key)) {
            continue;
        }
        constexpr int kPlaneSlot[6] = {0, 1, 3, 2, 4, 5};
        if (want && at < 0 && key.kind == 0 && ball_static_pairs_skipped &&
            !(key.static_id >= 100 && key.static_id <= 105 &&
              bp_history->ball_world_plane_manifolds[kPlaneSlot[key.static_id - 100]].count > 0)) {
            continue;
        }
        if (want && at < 0) {
            bp.bp_manifolds[bp.bp_manifold_count++] = packed;
        } else if (!want) {
            swap_remove(bp.bp_manifolds, bp.bp_manifold_count, at);
        }
    }

#if !defined(ROCKETGPU_USE_CUDA)
    int8_t tag[kMaxBodies];
    for (int b = 0; b < kMaxBodies; ++b) {
        tag[b] = -1;
    }
    for (int obj = 0; obj < h.sap_obj_count; ++obj) {
        tag[h.sap_obj[obj]] = static_cast<int8_t>(obj);
    }
    int parent[kMaxBodies];
    bool joined = false;
    const auto find_root = [&](int idx) {
        while (parent[idx] != idx) {
            idx = parent[idx];
        }
        return idx;
    };
    for (int i = 0; i < bp.bp_pair_count; ++i) {
        const BulletManifoldEntry key = UnpackBulletPairKey(bp.bp_pairs[i]);
        if (key.kind == 1 || key.kind == 3) {
            int body0 = key.kind == 1 ? 0 : key.body_a + 1;
            int body1 = key.kind == 1 ? key.body_a + 1 : key.body_b + 1;
            if (h.sap_body_handle[body0] > h.sap_body_handle[body1]) {
                const int t = body0;
                body0 = body1;
                body1 = t;
            }
            if (tag[body0] < 0 || tag[body1] < 0) {
                continue;
            }
            if (!joined) {
                for (int k = 0; k < kMaxBodies; ++k) {
                    parent[k] = k;
                }
                joined = true;
            }
            const int ra = find_root(tag[body0]);
            const int rb = find_root(tag[body1]);
            if (ra != rb) {
                parent[ra] = rb;
            }
        }
    }
    int8_t root_tag[kMaxBodies];
    for (int b = 0; b < kMaxBodies; ++b) {
        const int r = joined && tag[b] >= 0 ? find_root(tag[b]) : tag[b];
        root_tag[b] = static_cast<int8_t>(r);
        if (body_islands != nullptr) {
            body_islands[b] = static_cast<int8_t>(r);
        }
    }
#else
    int parent[kMaxBodies];
    for (int i = 0; i < kMaxBodies; ++i) {
        parent[i] = i;
    }
    int tag[kMaxBodies];
    for (int b = 0; b < kMaxBodies; ++b) {
        tag[b] = -1;
    }
    for (int obj = 0; obj < h.sap_obj_count; ++obj) {
        tag[h.sap_obj[obj]] = obj;
    }
    const auto find_root = [&](int idx) {
        while (parent[idx] != idx) {
            idx = parent[idx];
        }
        return idx;
    };
    for (int i = 0; i < bp.bp_pair_count; ++i) {
        const BulletManifoldEntry key = UnpackBulletPairKey(bp.bp_pairs[i]);
        if (key.kind == 1 || key.kind == 3) {
            int body0 = key.kind == 1 ? 0 : key.body_a + 1;
            int body1 = key.kind == 1 ? key.body_a + 1 : key.body_b + 1;
            if (h.sap_body_handle[body0] > h.sap_body_handle[body1]) {
                const int t = body0;
                body0 = body1;
                body1 = t;
            }
            if (tag[body0] < 0 || tag[body1] < 0) {
                continue;
            }
            const int ra = find_root(tag[body0]);
            const int rb = find_root(tag[body1]);
            if (ra != rb) {
                parent[ra] = rb;
            }
        }
    }
    int8_t root_tag[kMaxBodies];
    for (int b = 0; b < kMaxBodies; ++b) {
        const int r = tag[b] >= 0 ? find_root(tag[b]) : -1;
        root_tag[b] = static_cast<int8_t>(r);
        if (body_islands != nullptr) {
            body_islands[b] = static_cast<int8_t>(r);
        }
    }
#endif
    const int entry_count = Min(static_cast<int>(bp.bp_manifold_count), Min(max_entries, kMaxBulletManifoldEntries));
    for (int i = 0; i < entry_count; ++i) {
        BulletManifoldEntry entry = UnpackBulletPairKey(bp.bp_manifolds[i]);
        entry.island_id = root_tag[entry.kind <= 1 ? 0 : entry.body_a + 1];
        out_entries[i] = entry;
    }
    return entry_count;
    }
}

ROCKETGPU_HD ROCKETGPU_NOINLINE void SolveUnifiedSplitContactRows(
    ::rocketgpu::BallState& ball,
    int count,
    CarSpanT cars,
    EnvStridedArray<BallSplitContactRow> ball_rows,
    int ball_row_count,
    EnvStridedArray<CarSharedSplitContactRow> world_rows,
    int world_row_count,
    EnvStridedArray<CarSharedSplitContactRow> pair_rows,
    int pair_row_count,
    float dt,
    int env_idx = -1,
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
    UninitArray<SplitImpulseAccumulator, kMaxBallSplitCars> car_splits_storage;
    SplitImpulseAccumulator* const car_splits = car_splits_storage.data();
    for (int i = 0; i < count; ++i) {
        car_splits[i] = SplitImpulseAccumulator{};
    }

    MarkBtSplitAccumulators(count, car_splits, world_rows, world_row_count, pair_rows, pair_row_count, ball_rows, ball_row_count, &ball_split);

    constexpr int kMaxUnifiedSplitRows =
        kMaxCarWorldSplitContactRows + kMaxCarPairSplitContactRows + kMaxBallSplitContactRows;
    uint8_t unified_split_bucket[kMaxUnifiedSplitRows];
    uint8_t unified_split_row[kMaxUnifiedSplitRows];
    int unified_split_count = 0;
    bool unified_split_valid = false;
    if (sorted_manifolds != nullptr && sorted_manifold_count > 0 &&
        (active_world_rows + active_pair_rows + active_ball_rows) > 1 &&
        (active_world_rows + active_pair_rows + active_ball_rows) <= kMaxUnifiedSplitRows) {
        EnvStridedArray<const BulletManifoldEntry> entries = sorted_manifolds;
        const int entry_count = sorted_manifold_count;
        {
            static_assert(8 * kMaxCarWorldSplitContactRows <= 128, "RowUsedBits covers 8 cars' world split rows");
            RowUsedBits world_used;
            RowUsedBits pair_used;
            RowUsedBits ball_used;
            uint16_t world_key[128];
            for (int row_idx = 0; row_idx < active_world_rows; ++row_idx) {
                const CarSharedSplitContactRow& row = world_rows[row_idx];
                if (!row.valid || row.body_a_local_idx < 0 || row.body_a_local_idx >= 8 ||
                    row.source_tag < 0 || row.source_tag > 0x0FFE) {
                    world_key[row_idx] = 0xFFFFu;
                } else {
                    world_key[row_idx] =
                        (static_cast<uint16_t>(row.body_a_local_idx) << 12) |
                        static_cast<uint16_t>(row.source_tag);
                }
            }
            const auto push_split = [&](int bucket, int row_idx) {
                if (unified_split_count >= kMaxUnifiedSplitRows) {
                    return;
                }
                unified_split_bucket[unified_split_count] = static_cast<uint8_t>(bucket);
                unified_split_row[unified_split_count] = static_cast<uint8_t>(row_idx);
                ++unified_split_count;
            };
            for (int order_idx = 0; order_idx < entry_count; ++order_idx) {
                const BulletManifoldEntry& entry = entries[order_idx];
                if (entry.kind == 2) {
                    int expected_tag = -1;
                    if (entry.static_id >= 100) {
                        expected_tag = BulletPlaneStaticSourceTag(entry.static_id);
                    } else if (entry.static_id >= 0) {
                        expected_tag = 100 + entry.static_id;
                    }
                    const uint16_t expected_key =
                        (entry.body_a >= 0 && entry.body_a < 8 && expected_tag >= 0 && expected_tag <= 0x0FFE)
                            ? ((static_cast<uint16_t>(entry.body_a) << 12) | static_cast<uint16_t>(expected_tag))
                            : 0xFFFEu;
                    for (int row_idx = 0; row_idx < active_world_rows; ++row_idx) {
                        if (world_used.Test(row_idx) || world_key[row_idx] != expected_key) {
                            continue;
                        }
                        world_used.Set(row_idx);
                        push_split(0, row_idx);
                    }
                } else if (entry.kind == 3) {
                    for (int row_idx = 0; row_idx < active_pair_rows; ++row_idx) {
                        const CarSharedSplitContactRow& row = pair_rows[row_idx];
                        if (pair_used.Test(row_idx) || !row.valid) {
                            continue;
                        }
                        const int lo = Min(row.body_a_local_idx, row.body_b_local_idx);
                        const int hi = Max(row.body_a_local_idx, row.body_b_local_idx);
                        if (lo != entry.body_a || hi != entry.body_b) {
                            continue;
                        }
                        pair_used.Set(row_idx);
                        push_split(1, row_idx);
                    }
                } else if (entry.kind == 1) {
                    for (int row_idx = 0; row_idx < active_ball_rows; ++row_idx) {
                        const BallSplitContactRow& row = ball_rows[row_idx];
                        if (ball_used.Test(row_idx) || !row.valid || row.other_local_idx != entry.body_a) {
                            continue;
                        }
                        ball_used.Set(row_idx);
                        push_split(2, row_idx);
                    }
                } else if (entry.kind == 0) {
                    for (int row_idx = 0; row_idx < active_ball_rows; ++row_idx) {
                        const BallSplitContactRow& row = ball_rows[row_idx];
                        if (ball_used.Test(row_idx) || !row.valid || row.other_local_idx >= 0 ||
                            row.static_id < 0 || row.static_id != entry.static_id) {
                            continue;
                        }
                        ball_used.Set(row_idx);
                        push_split(2, row_idx);
                    }
                }
            }
            for (int row_idx = 0; row_idx < active_world_rows; ++row_idx) {
                if (!world_used.Test(row_idx)) {
                    push_split(0, row_idx);
                }
            }
            for (int row_idx = 0; row_idx < active_pair_rows; ++row_idx) {
                if (!pair_used.Test(row_idx)) {
                    push_split(1, row_idx);
                }
            }
            for (int row_idx = 0; row_idx < active_ball_rows; ++row_idx) {
                if (!ball_used.Test(row_idx)) {
                    push_split(2, row_idx);
                }
            }
            unified_split_valid =
                unified_split_count == active_world_rows + active_pair_rows + active_ball_rows;
        }
    }

    const int split_order_count =
        unified_split_valid ? unified_split_count : active_world_rows + active_pair_rows + active_ball_rows;
    for (int iter = 0; iter < BallSplitImpulseIterations(); ++iter) {
        bool changed = false;
        for (int order_idx = 0; order_idx < split_order_count; ++order_idx) {
            int bucket;
            int row_idx;
            if (unified_split_valid) {
                bucket = unified_split_bucket[order_idx];
                row_idx = unified_split_row[order_idx];
            } else if (order_idx < active_world_rows) {
                bucket = 0;
                row_idx = order_idx;
            } else if (order_idx < active_world_rows + active_pair_rows) {
                bucket = 1;
                row_idx = order_idx - active_world_rows;
            } else {
                bucket = 2;
                row_idx = order_idx - active_world_rows - active_pair_rows;
            }
            if (bucket < 2) {
                changed = SolveSharedCarSplitRow(
                    count, cars, (bucket == 0 ? world_rows : pair_rows)[row_idx], iter, row_idx,
                    bucket == 0 ? "world" : "pair", car_splits, env_idx) || changed;
            } else {
                changed = SolveUnifiedBallSplitRow(count, ball_rows[row_idx], ball_split, car_splits) || changed;
            }
        }
        if (!changed) {
            break;
        }
    }

    IntegrateBallOriginBt(
        ball,
        ball_split.is_bt ? ball_split.push_velocity : UuVecToBtDiv(ball_split.push_velocity),
        dt);
    SplitImpulseAccumulator next_split{};
    if (count > 0) {
        next_split = car_splits[0];
    }
    for (int i = -1; i < count; ++i) {
        const bool is_ball = i < 0;
        const SplitImpulseAccumulator car_split = next_split;
        if (!is_ball && i + 1 < count) {
            next_split = car_splits[i + 1];
        }
        const SplitImpulseAccumulator& split = is_ball ? ball_split : car_split;
        const bool nonzero =
            split.push_velocity.x != 0.0f || split.push_velocity.y != 0.0f || split.push_velocity.z != 0.0f ||
            split.turn_velocity.x != 0.0f || split.turn_velocity.y != 0.0f || split.turn_velocity.z != 0.0f;
        bool turn;
        if (is_ball) {
            turn = rotate_soccar_ball && nonzero;
        } else if (split.is_bt) {
            turn = nonzero;
            if (nonzero) {
                SetCarOriginBt(cars[i], cars[i].pos_bt + split.push_velocity * dt);
            }
        } else {
            turn = LengthSq(split.turn_velocity) > 0.0f;
            SetCarOriginBt(cars[i], cars[i].pos_bt + UuVecToBtDiv(split.push_velocity) * dt);
        }
        if (!turn) {
            continue;
        }
        if (is_ball && !ball.basis_authoritative) {
            GetOfficialUploadBasisVectors(ball.rot, ball.basis_forward, ball.basis_right, ball.basis_up);
        }
        ::rocketgpu::Vec3 fwd = is_ball ? ball.basis_forward : cars[i].basis_forward;
        ::rocketgpu::Vec3 right = is_ball ? ball.basis_right : cars[i].basis_right;
        ::rocketgpu::Vec3 up = is_ball ? ball.basis_up : cars[i].basis_up;
        ::rocketgpu::Quat rot = is_ball ? ball.rot : cars[i].rot;
        const ::rocketgpu::Vec3 turn_velocity = split.turn_velocity * BallSplitTurnErp();
        if (!is_ball && !cars[i].basis_authoritative) {
            IntegrateOrientation(rot, turn_velocity, dt);
        } else if (native_sse_math) {
            IntegrateOrientationBasisColumns(fwd, right, up, rot, turn_velocity, dt, true);
        } else {
            GameIntegrateOrientationBasisColumns(fwd, right, up, rot, turn_velocity, dt);
        }
        if (is_ball) {
            ball.basis_forward = fwd;
            ball.basis_right = right;
            ball.basis_up = up;
            ball.rot = rot;
            ball.basis_authoritative = 1;
        } else {
            cars[i].basis_forward = fwd;
            cars[i].basis_right = right;
            cars[i].basis_up = up;
            cars[i].rot = rot;
        }
    }
}

template <typename Index, typename Key>
ROCKETGPU_HD void QuickSortSharedContactOrderIndices(
    Index* order,
    const Key* island_ids,
    int lo,
    int hi) {
    constexpr int kMaxPendingRanges = 32;
    Index range_lo[kMaxPendingRanges];
    Index range_hi[kMaxPendingRanges];
    int depth = 0;
    range_lo[depth] = static_cast<Index>(lo);
    range_hi[depth] = static_cast<Index>(hi);
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

        const bool right_first = cur_hi - i < j - cur_lo;
        for (int side = 0; side < 2; ++side) {
            const bool right = (side == 0) != right_first;
            const int range_start = right ? i : cur_lo;
            const int range_end = right ? cur_hi : j;
            if (range_start < range_end && depth < kMaxPendingRanges) {
                range_lo[depth] = static_cast<Index>(range_start);
                range_hi[depth] = static_cast<Index>(range_end);
                ++depth;
            }
        }
    }
}

constexpr float kRowUnitsBtPerUu = 1.0f / 50.0f;
constexpr float kRowUnitsUuPerBt = 50.0f;

struct SharedRowUnits {
    bool valid;
    bool has_bt;
    bool has_b;
    float applied;
    ::rocketgpu::Vec3 rel_a, ang_a, rel_b, ang_b;
    ::rocketgpu::Vec3 rel_a_bt, ang_a_bt, rel_b_bt, ang_b_bt;
};

template <typename T>
ROCKETGPU_HD SharedRowUnits LoadSharedRowUnits(const T& row, bool with_bt) {
    constexpr bool kFriction = std::is_same<T, CarSharedFrictionContactRow>::value;
    SharedRowUnits s{};
    s.valid = row.valid != 0;
    s.has_bt = row.has_bt != 0;
    s.has_b = row.body_b_local_idx >= 0;
    s.applied = row.applied_impulse;
    if constexpr (kFriction) {
        s.rel_a = row.rel_a_cross_tangent;
        s.rel_b = row.rel_b_cross_tangent;
    } else {
        s.rel_a = row.rel_a_cross_normal;
        s.rel_b = row.rel_b_cross_normal;
    }
    s.ang_a = row.angular_component_a;
    s.ang_b = row.angular_component_b;
    if (with_bt) {
        if constexpr (kFriction) {
            s.rel_a_bt = row.rel_a_cross_tangent_bt;
            s.rel_b_bt = row.rel_b_cross_tangent_bt;
        } else {
            s.rel_a_bt = row.rel_a_cross_normal_bt;
            s.rel_b_bt = row.rel_b_cross_normal_bt;
        }
        s.ang_a_bt = row.angular_component_a_bt;
        s.ang_b_bt = row.angular_component_b_bt;
    }
    return s;
}

template <typename T>
ROCKETGPU_HD void StoreSharedRowUnits(T& row, float applied, const ::rocketgpu::Vec3& rel_a,
                                      const ::rocketgpu::Vec3& ang_a, const ::rocketgpu::Vec3& rel_b,
                                      const ::rocketgpu::Vec3& ang_b) {
    row.applied_impulse = applied;
    if constexpr (std::is_same<T, CarSharedFrictionContactRow>::value) {
        row.rel_a_cross_tangent = rel_a;
        row.rel_b_cross_tangent = rel_b;
    } else {
        row.rel_a_cross_normal = rel_a;
        row.rel_b_cross_normal = rel_b;
    }
    row.angular_component_a = ang_a;
    row.angular_component_b = ang_b;
}

template <typename T>
ROCKETGPU_HD void StoreSharedRowUnitsIn(T& row, const SharedRowUnits& s) {
    if (!s.valid) {
        return;
    }
    const float applied = s.applied * kRowUnitsBtPerUu;
    if (s.has_bt) {
        StoreSharedRowUnits(row, applied, s.rel_a_bt, s.ang_a_bt,
                            s.has_b ? s.rel_b_bt : s.rel_b * kRowUnitsBtPerUu,
                            s.has_b ? s.ang_b_bt : s.ang_b * kRowUnitsUuPerBt);
    } else {
        StoreSharedRowUnits(row, applied, s.rel_a * kRowUnitsBtPerUu, s.ang_a * kRowUnitsUuPerBt,
                            s.rel_b * kRowUnitsBtPerUu, s.ang_b * kRowUnitsUuPerBt);
    }
}

template <typename T>
ROCKETGPU_HD void StoreSharedRowUnitsOut(T& row, const SharedRowUnits& s) {
    if (!s.valid) {
        return;
    }
    StoreSharedRowUnits(row, s.applied * kRowUnitsUuPerBt, s.rel_a * kRowUnitsUuPerBt, s.ang_a * kRowUnitsBtPerUu,
                        s.rel_b * kRowUnitsUuPerBt, s.ang_b * kRowUnitsBtPerUu);
}

struct BallRowUnits {
    bool valid;
    bool has_bt;
    float applied;
    float rhs;
    float rhs_bt;
    ::rocketgpu::Vec3 rel_ball, rel_other, ang_ball, ang_other;
    ::rocketgpu::Vec3 rel_ball_bt, rel_other_bt, ang_ball_bt, ang_other_bt;
};

template <typename T>
ROCKETGPU_HD BallRowUnits LoadBallRowUnits(const T& row, bool with_bt) {
    constexpr bool kFriction = std::is_same<T, BallFrictionContactRow>::value;
    BallRowUnits s{};
    s.valid = row.valid != 0;
    s.has_bt = row.has_bt != 0;
    s.applied = row.applied_impulse;
    s.rhs = row.rhs;
    if constexpr (kFriction) {
        s.rel_ball = row.rel_ball_cross_tangent;
        s.rel_other = row.rel_other_cross_tangent;
    } else {
        s.rel_ball = row.rel_ball_cross_normal;
        s.rel_other = row.rel_other_cross_normal;
    }
    s.ang_ball = row.ball_angular_component;
    s.ang_other = row.other_angular_component;
    if (with_bt) {
        s.rhs_bt = row.rhs_bt;
        if constexpr (kFriction) {
            s.rel_ball_bt = row.rel_ball_cross_tangent_bt;
            s.rel_other_bt = row.rel_other_cross_tangent_bt;
        } else {
            s.rel_ball_bt = row.rel_ball_cross_normal_bt;
            s.rel_other_bt = row.rel_other_cross_normal_bt;
        }
        s.ang_ball_bt = row.ball_angular_component_bt;
        s.ang_other_bt = row.other_angular_component_bt;
    }
    return s;
}

template <typename T>
ROCKETGPU_HD void StoreBallRowUnits(T& row, float applied, float rhs, const ::rocketgpu::Vec3& rel_ball,
                                    const ::rocketgpu::Vec3& rel_other, const ::rocketgpu::Vec3& ang_ball,
                                    const ::rocketgpu::Vec3& ang_other) {
    row.applied_impulse = applied;
    row.rhs = rhs;
    if constexpr (std::is_same<T, BallFrictionContactRow>::value) {
        row.rel_ball_cross_tangent = rel_ball;
        row.rel_other_cross_tangent = rel_other;
    } else {
        row.rel_ball_cross_normal = rel_ball;
        row.rel_other_cross_normal = rel_other;
    }
    row.ball_angular_component = ang_ball;
    row.other_angular_component = ang_other;
}

template <typename T>
ROCKETGPU_HD void StoreBallRowUnitsIn(T& row, const BallRowUnits& s) {
    if (!s.valid) {
        return;
    }
    const float applied = s.applied * kRowUnitsBtPerUu;
    if (s.has_bt) {
        StoreBallRowUnits(row, applied, s.rhs_bt, s.rel_ball_bt, s.rel_other_bt, s.ang_ball_bt, s.ang_other_bt);
    } else {
        StoreBallRowUnits(row, applied, s.rhs * kRowUnitsBtPerUu, s.rel_ball * kRowUnitsBtPerUu,
                          s.rel_other * kRowUnitsBtPerUu, s.ang_ball * kRowUnitsUuPerBt,
                          s.ang_other * kRowUnitsUuPerBt);
    }
}

template <typename T>
ROCKETGPU_HD void StoreBallRowUnitsOut(T& row, const BallRowUnits& s) {
    if (!s.valid) {
        return;
    }
    StoreBallRowUnits(row, s.applied * kRowUnitsUuPerBt, s.rhs * kRowUnitsUuPerBt, s.rel_ball * kRowUnitsUuPerBt,
                      s.rel_other * kRowUnitsUuPerBt, s.ang_ball * kRowUnitsBtPerUu,
                      s.ang_other * kRowUnitsBtPerUu);
}

template <bool kIn, typename T>
ROCKETGPU_HD void ConvertRowUnits(EnvStridedArray<T> rows, int n) {
    constexpr bool kBall =
        std::is_same<T, BallVelocityContactRow>::value || std::is_same<T, BallFrictionContactRow>::value;
    for (int i = 0; i < n; i += 2) {
        const bool two = i + 1 < n;
        if constexpr (kBall) {
            const BallRowUnits a = LoadBallRowUnits(rows[i], kIn);
            BallRowUnits b = a;
            if (two) {
                b = LoadBallRowUnits(rows[i + 1], kIn);
            }
            kIn ? StoreBallRowUnitsIn(rows[i], a) : StoreBallRowUnitsOut(rows[i], a);
            if (two) {
                kIn ? StoreBallRowUnitsIn(rows[i + 1], b) : StoreBallRowUnitsOut(rows[i + 1], b);
            }
        } else {
            const SharedRowUnits a = LoadSharedRowUnits(rows[i], kIn);
            SharedRowUnits b = a;
            if (two) {
                b = LoadSharedRowUnits(rows[i + 1], kIn);
            }
            kIn ? StoreSharedRowUnitsIn(rows[i], a) : StoreSharedRowUnitsOut(rows[i], a);
            if (two) {
                kIn ? StoreSharedRowUnitsIn(rows[i + 1], b) : StoreSharedRowUnitsOut(rows[i + 1], b);
            }
        }
    }
}

ROCKETGPU_HD ROCKETGPU_NOINLINE ROCKETGPU_HOST_NOINLINE void SolveUnifiedVelocityContactRowsGeneral(
    ::rocketgpu::BallState& ball,
    int count,
    CarSpanT cars,
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
    const ::rocketgpu::Vec3* bt_ball_linear_velocity_override,
    EnvStridedArray<const BulletManifoldEntry> sorted_manifolds,
    int sorted_manifold_count) {
    BallVelocityAccumulator ball_acc;
    UninitArray<BallVelocityAccumulator, 8> car_accs_storage;
    BallVelocityAccumulator* const car_accs = car_accs_storage.data();
    for (int i = 0; i < count; ++i) {
        car_accs[i] = BallVelocityAccumulator{};
    }

    ConvertRowUnits<true>(pair_normal_rows, pair_normal_row_count);
    ConvertRowUnits<true>(world_normal_rows, world_normal_row_count);
    ConvertRowUnits<true>(pair_friction_rows, pair_friction_row_count);
    ConvertRowUnits<true>(world_friction_rows, world_friction_row_count);
    ConvertRowUnits<true>(ball_normal_rows, ball_normal_row_count);
    ConvertRowUnits<true>(ball_friction_rows, ball_friction_row_count);

    if (pair_normal_row_count > 0) {
        WarmstartSharedCarVelocityRows(count, cars, car_configs, car_accs, pair_normal_rows, pair_normal_row_count);
    }

    constexpr int kUnifiedRowBucketWorld = 0;
    constexpr int kUnifiedRowBucketPair = 1;
    constexpr int kUnifiedRowBucketBall = 2;
    constexpr int kMaxUnifiedRows =
        kMaxCarWorldSharedVelocityContactRows + kMaxCarPairSharedVelocityContactRows + kMaxBallVelocityContactRows;
    uint8_t unified_normal_bucket[kMaxUnifiedRows];
    uint8_t unified_normal_row[kMaxUnifiedRows];
    int unified_normal_count = 0;
    alignas(2) uint8_t unified_friction_storage[2 * kMaxUnifiedRows];
    uint8_t* const unified_friction_bucket = unified_friction_storage;
    uint8_t* const unified_friction_row = unified_friction_storage + kMaxUnifiedRows;
    int8_t scratch_i8[kMaxCarWorldSharedVelocityContactRows];
    int unified_friction_count = 0;
    if (world_normal_row_count + pair_normal_row_count + ball_normal_row_count <= 1 &&
        world_friction_row_count + pair_friction_row_count + ball_friction_row_count <= 1) {
        if (world_normal_row_count == 1) {
            unified_normal_bucket[0] = static_cast<uint8_t>(kUnifiedRowBucketWorld);
            unified_normal_row[0] = 0;
            unified_normal_count = 1;
        } else if (pair_normal_row_count == 1) {
            unified_normal_bucket[0] = static_cast<uint8_t>(kUnifiedRowBucketPair);
            unified_normal_row[0] = 0;
            unified_normal_count = 1;
        } else if (ball_normal_row_count == 1) {
            unified_normal_bucket[0] = static_cast<uint8_t>(kUnifiedRowBucketBall);
            unified_normal_row[0] = 0;
            unified_normal_count = 1;
        }
        if (world_friction_row_count == 1) {
            unified_friction_bucket[0] = static_cast<uint8_t>(kUnifiedRowBucketWorld);
            unified_friction_row[0] = 0;
            unified_friction_count = 1;
        } else if (pair_friction_row_count == 1) {
            unified_friction_bucket[0] = static_cast<uint8_t>(kUnifiedRowBucketPair);
            unified_friction_row[0] = 0;
            unified_friction_count = 1;
        } else if (ball_friction_row_count == 1) {
            unified_friction_bucket[0] = static_cast<uint8_t>(kUnifiedRowBucketBall);
            unified_friction_row[0] = 0;
            unified_friction_count = 1;
        }
    } else {
        EnvStridedArray<const BulletManifoldEntry> entries = sorted_manifolds;
        const int entry_count = sorted_manifolds != nullptr ? sorted_manifold_count : 0;
        {
            {
                RowUsedBits world_used;
                RowUsedBits pair_used;
                RowUsedBits ball_used;
                int8_t* const world_body_a = scratch_i8;
                int16_t* const world_source_tag = reinterpret_cast<int16_t*>(unified_friction_storage);
                for (int row_idx = 0; row_idx < world_normal_row_count; ++row_idx) {
                    const CarSharedVelocityContactRow& row = world_normal_rows[row_idx];
                    if (!row.valid) {
                        world_body_a[row_idx] = -2;
                        world_source_tag[row_idx] = -2;
                    } else {
                        world_body_a[row_idx] = static_cast<int8_t>(row.body_a_local_idx);
                        world_source_tag[row_idx] = static_cast<int16_t>(row.source_tag);
                    }
                }
                const auto push_row = [&](int bucket, int row_idx) {
                    if (unified_normal_count >= kMaxUnifiedRows) {
                        return;
                    }
                    unified_normal_bucket[unified_normal_count] = static_cast<uint8_t>(bucket);
                    unified_normal_row[unified_normal_count] = static_cast<uint8_t>(row_idx);
                    ++unified_normal_count;
                };
                for (int order_idx = 0; order_idx < entry_count; ++order_idx) {
                    const BulletManifoldEntry& entry = entries[order_idx];
                    if (entry.kind == 2) {
                        int expected_tag = -1;
                        if (entry.static_id >= 100) {
                            expected_tag = BulletPlaneStaticSourceTag(entry.static_id);
                        } else if (entry.static_id >= 0) {
                            expected_tag = 100 + entry.static_id;
                        }
                        for (int row_idx = 0; row_idx < world_normal_row_count; ++row_idx) {
                            if (world_used.Test(row_idx) || world_body_a[row_idx] != entry.body_a ||
                                world_source_tag[row_idx] != expected_tag) {
                                continue;
                            }
                            world_used.Set(row_idx);
                            push_row(kUnifiedRowBucketWorld, row_idx);
                        }
                    } else if (entry.kind == 3) {
                        for (int row_idx = 0; row_idx < pair_normal_row_count; ++row_idx) {
                            const CarSharedVelocityContactRow& row = pair_normal_rows[row_idx];
                            if (pair_used.Test(row_idx) || !row.valid) {
                                continue;
                            }
                            const int lo = Min(row.body_a_local_idx, row.body_b_local_idx);
                            const int hi = Max(row.body_a_local_idx, row.body_b_local_idx);
                            if (lo != entry.body_a || hi != entry.body_b) {
                                continue;
                            }
                            pair_used.Set(row_idx);
                            push_row(kUnifiedRowBucketPair, row_idx);
                        }
                    } else if (entry.kind == 1) {
                        for (int row_idx = 0; row_idx < ball_normal_row_count; ++row_idx) {
                            const BallVelocityContactRow& row = ball_normal_rows[row_idx];
                            if (ball_used.Test(row_idx) || !row.valid || row.other_local_idx != entry.body_a) {
                                continue;
                            }
                            ball_used.Set(row_idx);
                            push_row(kUnifiedRowBucketBall, row_idx);
                        }
                    } else if (entry.kind == 0 && entry.static_id == kBulletPredictiveStaticId) {
                        for (int row_idx = 0; row_idx < ball_normal_row_count; ++row_idx) {
                            const BallVelocityContactRow& row = ball_normal_rows[row_idx];
                            if (ball_used.Test(row_idx) || !row.valid || !row.predictive) {
                                continue;
                            }
                            ball_used.Set(row_idx);
                            push_row(kUnifiedRowBucketBall, row_idx);
                        }
                    }
                }

                for (int row_idx = 0; row_idx < world_normal_row_count; ++row_idx) {
                    if (!world_used.Test(row_idx)) {
                        push_row(kUnifiedRowBucketWorld, row_idx);
                    }
                }
                for (int row_idx = 0; row_idx < pair_normal_row_count; ++row_idx) {
                    if (!pair_used.Test(row_idx)) {
                        push_row(kUnifiedRowBucketPair, row_idx);
                    }
                }
                for (int row_idx = 0; row_idx < ball_normal_row_count; ++row_idx) {
                    if (!ball_used.Test(row_idx)) {
                        push_row(kUnifiedRowBucketBall, row_idx);
                    }
                }
            }
            {
                RowUsedBits world_friction_used;
                RowUsedBits pair_friction_used;
                RowUsedBits ball_friction_used;
                int8_t* const world_fric_by_norm = scratch_i8;
                for (int i = 0; i < world_normal_row_count; ++i) {
                    world_fric_by_norm[i] = -1;
                }
                bool world_fric_unique = true;
                for (int row_idx = 0; row_idx < world_friction_row_count; ++row_idx) {
                    const int n = world_friction_rows[row_idx].normal_row_idx;
                    if (n >= 0 && n < world_normal_row_count) {
                        if (world_fric_by_norm[n] >= 0) {
                            world_fric_unique = false;
                        } else {
                            world_fric_by_norm[n] = static_cast<int8_t>(row_idx);
                        }
                    }
                }
                const auto push_friction = [&](int bucket, int row_idx) {
                    if (unified_friction_count >= kMaxUnifiedRows) {
                        return;
                    }
                    unified_friction_bucket[unified_friction_count] = static_cast<uint8_t>(bucket);
                    unified_friction_row[unified_friction_count] = static_cast<uint8_t>(row_idx);
                    ++unified_friction_count;
                };
                for (int order_idx = 0; order_idx < unified_normal_count; ++order_idx) {
                    const int normal_row_idx = unified_normal_row[order_idx];
                    switch (unified_normal_bucket[order_idx]) {
                        case kUnifiedRowBucketWorld:
                            if (world_fric_unique) {
                                if (normal_row_idx >= 0 && normal_row_idx < world_normal_row_count) {
                                    const int f_idx = world_fric_by_norm[normal_row_idx];
                                    if (f_idx >= 0 && !world_friction_used.Test(f_idx)) {
                                        world_friction_used.Set(f_idx);
                                        push_friction(kUnifiedRowBucketWorld, f_idx);
                                    }
                                }
                            } else {
                                for (int row_idx = 0; row_idx < world_friction_row_count; ++row_idx) {
                                    if (!world_friction_used.Test(row_idx) &&
                                        world_friction_rows[row_idx].normal_row_idx == normal_row_idx) {
                                        world_friction_used.Set(row_idx);
                                        push_friction(kUnifiedRowBucketWorld, row_idx);
                                    }
                                }
                            }
                            break;
                        case kUnifiedRowBucketPair:
                            for (int row_idx = 0; row_idx < pair_friction_row_count; ++row_idx) {
                                if (!pair_friction_used.Test(row_idx) &&
                                    pair_friction_rows[row_idx].normal_row_idx == normal_row_idx) {
                                    pair_friction_used.Set(row_idx);
                                    push_friction(kUnifiedRowBucketPair, row_idx);
                                }
                            }
                            break;
                        default:
                            for (int row_idx = 0; row_idx < ball_friction_row_count; ++row_idx) {
                                if (!ball_friction_used.Test(row_idx) &&
                                    ball_friction_rows[row_idx].normal_row_idx == normal_row_idx) {
                                    ball_friction_used.Set(row_idx);
                                    push_friction(kUnifiedRowBucketBall, row_idx);
                                }
                            }
                            break;
                    }
                }
                for (int row_idx = 0; row_idx < world_friction_row_count; ++row_idx) {
                    if (!world_friction_used.Test(row_idx)) {
                        push_friction(kUnifiedRowBucketWorld, row_idx);
                    }
                }
                for (int row_idx = 0; row_idx < pair_friction_row_count; ++row_idx) {
                    if (!pair_friction_used.Test(row_idx)) {
                        push_friction(kUnifiedRowBucketPair, row_idx);
                    }
                }
                for (int row_idx = 0; row_idx < ball_friction_row_count; ++row_idx) {
                    if (!ball_friction_used.Test(row_idx)) {
                        push_friction(kUnifiedRowBucketBall, row_idx);
                    }
                }
            }
        }
    }

    static_assert(kMaxCarWorldSharedFrictionContactRows + kMaxCarPairSharedFrictionContactRows +
                      kMaxBallFrictionContactRows <= kMaxUnifiedRows, "friction order fits the unified lists");
    const int iterations = BallVelocityContactIterations();

    for (int iter = 0; iter < iterations; ++iter) {
        bool changed = false;
        for (int order_idx = 0; order_idx < unified_normal_count; ++order_idx) {
            const int row_idx = unified_normal_row[order_idx];
            const int bucket = unified_normal_bucket[order_idx];
            if (bucket == kUnifiedRowBucketBall) {
                changed = SolveBallVelocityNormalRowWithAccumulators(
                    count,
                    ball_normal_rows[row_idx],
                    ball_acc,
                    car_accs,
                    mutator.solver_use_fma != 0) || changed;
            } else {
                const bool world = bucket == kUnifiedRowBucketWorld;
                changed = SolveSharedCarVelocityNormalRow(
                    count,
                    cars,
                    car_configs,
                    (world ? world_normal_rows : pair_normal_rows)[row_idx],
                    car_accs,
                    mutator.solver_use_fma != 0) || changed;
            }
        }
        for (int order_idx = 0; order_idx < unified_friction_count; ++order_idx) {
            const int row_idx = unified_friction_row[order_idx];
            const int bucket = unified_friction_bucket[order_idx];
            if (bucket == kUnifiedRowBucketBall) {
                changed = SolveBallVelocityFrictionRowWithAccumulators(
                    count,
                    ball_normal_rows,
                    ball_normal_row_count,
                    ball_friction_rows[row_idx],
                    ball_acc,
                    car_accs,
                    mutator.solver_use_fma != 0) || changed;
            } else {
                const bool world = bucket == kUnifiedRowBucketWorld;
                changed = SolveSharedCarVelocityFrictionRow(
                    count,
                    cars,
                    car_configs,
                    world ? world_normal_rows : pair_normal_rows,
                    world ? world_normal_row_count : pair_normal_row_count,
                    (world ? world_friction_rows : pair_friction_rows)[row_idx],
                    car_accs,
                    mutator.solver_use_fma != 0) || changed;
            }
        }
        if (!changed) {
            break;
        }
    }

    ConvertRowUnits<false>(pair_normal_rows, pair_normal_row_count);
    ConvertRowUnits<false>(world_normal_rows, world_normal_row_count);
    ConvertRowUnits<false>(pair_friction_rows, pair_friction_row_count);
    ConvertRowUnits<false>(world_friction_rows, world_friction_row_count);
    ConvertRowUnits<false>(ball_normal_rows, ball_normal_row_count);
    ConvertRowUnits<false>(ball_friction_rows, ball_friction_row_count);

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

ROCKETGPU_HD ROCKETGPU_NOINLINE bool TrySolveUnifiedVelocityContactRowsReg(
    ::rocketgpu::BallState& ball,
    int count,
    CarSpanT cars,
    const ::rocketgpu::MutatorConfig& mutator,
    EnvStridedArray<BallVelocityContactRow> ball_normal_rows,
    int ball_normal_row_count,
    EnvStridedArray<BallFrictionContactRow> ball_friction_rows,
    int ball_friction_row_count,
    EnvStridedArray<CarSharedVelocityContactRow> world_normal_rows,
    int world_normal_row_count,
    EnvStridedArray<CarSharedFrictionContactRow> world_friction_rows,
    int world_friction_row_count,
    const ::rocketgpu::Vec3* bt_ball_linear_velocity_override,
    EnvStridedArray<const BulletManifoldEntry> sorted_manifolds,
    int sorted_manifold_count) {
    constexpr float kUnifiedRowConversionBtPerUu = 1.0f / 50.0f;
    constexpr float kUnifiedRowConversionUuPerBt = 50.0f;
    constexpr int kRegRows = 4;
    {
        struct RegRow {
            ::rocketgpu::Vec3 dir;
            ::rocketgpu::Vec3 linear;
            ::rocketgpu::Vec3 rel_cross;
            ::rocketgpu::Vec3 angular;
            float rhs;
            float jac;
            float applied;
            float friction;
            int normal_slot;
            int body_slot;
            bool active;
            bool has_rhs_bt;
            bool is_ball;
        };
        RegRow normal[kRegRows];
        RegRow fric[kRegRows];
        int body0 = -1;
        int body1 = -1;
        bool fits = true;
        const auto take_car = [&](int body, int& slot_out) {
            if (body0 < 0 || body0 == body) {
                body0 = body;
                slot_out = 0;
                return true;
            }
            if (body1 < 0 || body1 == body) {
                body1 = body;
                slot_out = 1;
                return true;
            }
            slot_out = 0;
            return false;
        };
        int norm_order_bucket[kRegRows];
        int norm_order_idx[kRegRows];
        int fric_order_bucket[kRegRows];
        int fric_order_idx[kRegRows];
        int norm_count = 0;
        int fric_count = 0;
        if (sorted_manifolds != nullptr && sorted_manifold_count > 0 &&
            (world_normal_row_count + ball_normal_row_count > 1 ||
             world_friction_row_count + ball_friction_row_count > 1)) {
            uint32_t world_used = 0u;
            uint32_t ball_used = 0u;
            int8_t world_body_a[kRegRows];
            int16_t world_source_tag[kRegRows];
#pragma unroll
            for (int r = 0; r < kRegRows; ++r) {
                if (r < world_normal_row_count && world_normal_rows[r].valid) {
                    world_body_a[r] = static_cast<int8_t>(world_normal_rows[r].body_a_local_idx);
                    world_source_tag[r] = static_cast<int16_t>(world_normal_rows[r].source_tag);
                } else {
                    world_body_a[r] = -2;
                    world_source_tag[r] = -2;
                }
            }
            for (int order_idx = 0; order_idx < sorted_manifold_count; ++order_idx) {
                const BulletManifoldEntry& entry = sorted_manifolds[order_idx];
                if (entry.kind == 2) {
                    int expected_tag = -1;
                    if (entry.static_id >= 100) {
                        expected_tag = BulletPlaneStaticSourceTag(entry.static_id);
                    } else if (entry.static_id >= 0) {
                        expected_tag = 100 + entry.static_id;
                    }
#pragma unroll
                    for (int r = 0; r < kRegRows; ++r) {
                        if (r < world_normal_row_count && ((world_used >> r) & 1u) == 0u &&
                            world_body_a[r] == entry.body_a && world_source_tag[r] == expected_tag &&
                            norm_count < kRegRows) {
                            world_used |= 1u << r;
                            norm_order_bucket[norm_count] = 0;
                            norm_order_idx[norm_count++] = r;
                        }
                    }
                } else if (entry.kind == 1) {
                    fits = false;
                } else if (entry.kind == 0 && entry.static_id == kBulletPredictiveStaticId) {
#pragma unroll
                    for (int r = 0; r < kRegRows; ++r) {
                        if (r < ball_normal_row_count && ((ball_used >> r) & 1u) == 0u &&
                            ball_normal_rows[r].valid && ball_normal_rows[r].predictive && norm_count < kRegRows) {
                            ball_used |= 1u << r;
                            norm_order_bucket[norm_count] = 2;
                            norm_order_idx[norm_count++] = r;
                        }
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < kRegRows; ++r) {
                if (r < world_normal_row_count && ((world_used >> r) & 1u) == 0u && norm_count < kRegRows) {
                    world_used |= 1u << r;
                    norm_order_bucket[norm_count] = 0;
                    norm_order_idx[norm_count++] = r;
                }
            }
#pragma unroll
            for (int r = 0; r < kRegRows; ++r) {
                if (r < ball_normal_row_count && ((ball_used >> r) & 1u) == 0u && norm_count < kRegRows) {
                    ball_used |= 1u << r;
                    norm_order_bucket[norm_count] = 2;
                    norm_order_idx[norm_count++] = r;
                }
            }
            uint32_t world_f_used = 0u;
            uint32_t ball_f_used = 0u;
#pragma unroll
            for (int o = 0; o < kRegRows; ++o) {
                if (o >= norm_count) {
                    continue;
                }
                const int n_idx = norm_order_idx[o];
                if (norm_order_bucket[o] == 0) {
#pragma unroll
                    for (int r = 0; r < kRegRows; ++r) {
                        if (r < world_friction_row_count && ((world_f_used >> r) & 1u) == 0u &&
                            world_friction_rows[r].normal_row_idx == n_idx && fric_count < kRegRows) {
                            world_f_used |= 1u << r;
                            fric_order_bucket[fric_count] = 0;
                            fric_order_idx[fric_count++] = r;
                        }
                    }
                } else {
#pragma unroll
                    for (int r = 0; r < kRegRows; ++r) {
                        if (r < ball_friction_row_count && ((ball_f_used >> r) & 1u) == 0u &&
                            ball_friction_rows[r].normal_row_idx == n_idx && fric_count < kRegRows) {
                            ball_f_used |= 1u << r;
                            fric_order_bucket[fric_count] = 2;
                            fric_order_idx[fric_count++] = r;
                        }
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < kRegRows; ++r) {
                if (r < world_friction_row_count && ((world_f_used >> r) & 1u) == 0u && fric_count < kRegRows) {
                    world_f_used |= 1u << r;
                    fric_order_bucket[fric_count] = 0;
                    fric_order_idx[fric_count++] = r;
                }
            }
#pragma unroll
            for (int r = 0; r < kRegRows; ++r) {
                if (r < ball_friction_row_count && ((ball_f_used >> r) & 1u) == 0u && fric_count < kRegRows) {
                    ball_f_used |= 1u << r;
                    fric_order_bucket[fric_count] = 2;
                    fric_order_idx[fric_count++] = r;
                }
            }
            fits = fits && (norm_count == world_normal_row_count + ball_normal_row_count) &&
                   (fric_count == world_friction_row_count + ball_friction_row_count);
        } else {
            fits = fits && (ball_friction_row_count <= 1);
#pragma unroll
            for (int r = 0; r < kRegRows; ++r) {
                if (r < world_normal_row_count && norm_count < kRegRows) {
                    norm_order_bucket[norm_count] = 0;
                    norm_order_idx[norm_count++] = r;
                }
            }
#pragma unroll
            for (int r = 0; r < kRegRows; ++r) {
                if (r < ball_normal_row_count && norm_count < kRegRows) {
                    norm_order_bucket[norm_count] = 2;
                    norm_order_idx[norm_count++] = r;
                }
            }
#pragma unroll
            for (int r = 0; r < kRegRows; ++r) {
                if (r < world_friction_row_count && fric_count < kRegRows) {
                    fric_order_bucket[fric_count] = 0;
                    fric_order_idx[fric_count++] = r;
                }
            }
#pragma unroll
            for (int r = 0; r < kRegRows; ++r) {
                if (r < ball_friction_row_count && fric_count < kRegRows) {
                    fric_order_bucket[fric_count] = 2;
                    fric_order_idx[fric_count++] = r;
                }
            }
        }
#pragma unroll
        for (int o = 0; o < kRegRows; ++o) {
            RegRow& s = normal[o];
            s.active = false;
            s.applied = 0.0f;
            if (o >= norm_count) {
                continue;
            }
            if (norm_order_bucket[o] == 2) {
                const BallVelocityContactRow& row = ball_normal_rows[norm_order_idx[o]];
                s.applied = row.valid ? row.applied_impulse * kUnifiedRowConversionBtPerUu : row.applied_impulse;
                if (!row.valid) {
                    continue;
                }
                if (row.other_local_idx >= 0 && row.other_local_idx < count) {
                    fits = false;
                    continue;
                }
                s.active = true;
                s.is_ball = true;
                s.body_slot = 2;
                s.dir = row.normal;
                s.linear = row.normal * row.ball_inv_mass;
                s.rel_cross = row.has_bt
                    ? row.rel_ball_cross_normal_bt
                    : row.rel_ball_cross_normal * kUnifiedRowConversionBtPerUu;
                s.angular = row.has_bt
                    ? row.ball_angular_component_bt
                    : row.ball_angular_component * kUnifiedRowConversionUuPerBt;
                s.has_rhs_bt = true;
                s.rhs = row.has_bt ? row.rhs_bt : row.rhs * kUnifiedRowConversionBtPerUu;
                s.jac = row.jac_diag_ab_inv;
                continue;
            }
            const CarSharedVelocityContactRow& row = world_normal_rows[norm_order_idx[o]];
            s.applied = row.valid ? row.applied_impulse * kUnifiedRowConversionBtPerUu : row.applied_impulse;
            if (!row.valid || row.body_a_local_idx < 0 || row.body_a_local_idx >= count) {
                continue;
            }
            fits = fits && !(row.body_b_local_idx >= 0 && row.body_b_local_idx < count) &&
                   take_car(row.body_a_local_idx, s.body_slot);
            s.active = true;
            s.is_ball = false;
            s.dir = row.normal_a;
            s.linear = row.normal_a * row.inv_mass_a;
            s.rel_cross = row.has_bt
                ? row.rel_a_cross_normal_bt
                : row.rel_a_cross_normal * kUnifiedRowConversionBtPerUu;
            s.angular = row.has_bt
                ? row.angular_component_a_bt
                : row.angular_component_a * kUnifiedRowConversionUuPerBt;
            s.has_rhs_bt = row.has_rhs_bt != 0;
            s.rhs = s.has_rhs_bt ? row.rhs_bt : row.rhs;
            s.jac = row.jac_diag_ab_inv;
        }
#pragma unroll
        for (int o = 0; o < kRegRows; ++o) {
            RegRow& s = fric[o];
            s.active = false;
            if (o >= fric_count) {
                continue;
            }
            if (fric_order_bucket[o] == 2) {
                const BallFrictionContactRow& row = ball_friction_rows[fric_order_idx[o]];
                if (!row.valid || row.normal_row_idx < 0 || row.normal_row_idx >= ball_normal_row_count) {
                    continue;
                }
                if (row.other_local_idx >= 0 && row.other_local_idx < count) {
                    fits = false;
                    continue;
                }
                bool normal_found = false;
                s.normal_slot = 0;
#pragma unroll
                for (int k = 0; k < kRegRows; ++k) {
                    if (!normal_found && k < norm_count && norm_order_bucket[k] == 2 &&
                        norm_order_idx[k] == row.normal_row_idx) {
                        s.normal_slot = k;
                        normal_found = true;
                    }
                }
                fits = fits && normal_found;
                s.active = true;
                s.is_ball = true;
                s.body_slot = 2;
                s.dir = row.tangent;
                s.linear = row.tangent * row.ball_inv_mass;
                s.rel_cross = row.has_bt
                    ? row.rel_ball_cross_tangent_bt
                    : row.rel_ball_cross_tangent * kUnifiedRowConversionBtPerUu;
                s.angular = row.has_bt
                    ? row.ball_angular_component_bt
                    : row.ball_angular_component * kUnifiedRowConversionUuPerBt;
                s.rhs = row.has_bt ? row.rhs_bt : row.rhs * kUnifiedRowConversionBtPerUu;
                s.jac = row.jac_diag_ab_inv;
                s.applied = row.applied_impulse * kUnifiedRowConversionBtPerUu;
                s.friction = row.friction;
                continue;
            }
            const CarSharedFrictionContactRow& row = world_friction_rows[fric_order_idx[o]];
            if (!row.valid || row.normal_row_idx < 0 || row.normal_row_idx >= world_normal_row_count ||
                row.body_a_local_idx < 0 || row.body_a_local_idx >= count) {
                continue;
            }
            bool normal_found = false;
            s.normal_slot = 0;
#pragma unroll
            for (int k = 0; k < kRegRows; ++k) {
                if (!normal_found && k < norm_count && norm_order_bucket[k] == 0 &&
                    norm_order_idx[k] == row.normal_row_idx) {
                    s.normal_slot = k;
                    normal_found = true;
                }
            }
            fits = fits && normal_found && !(row.body_b_local_idx >= 0 && row.body_b_local_idx < count) &&
                   take_car(row.body_a_local_idx, s.body_slot);
            s.active = true;
            s.is_ball = false;
            s.dir = row.tangent_a;
            s.linear = row.tangent_a * row.inv_mass_a;
            s.rel_cross = row.has_bt
                ? row.rel_a_cross_tangent_bt
                : row.rel_a_cross_tangent * kUnifiedRowConversionBtPerUu;
            s.angular = row.has_bt
                ? row.angular_component_a_bt
                : row.angular_component_a * kUnifiedRowConversionUuPerBt;
            s.rhs = row.has_rhs_bt ? row.rhs_bt : row.rhs * kUnifiedRowConversionBtPerUu;
            s.jac = row.jac_diag_ab_inv;
            s.applied = row.applied_impulse * kUnifiedRowConversionBtPerUu;
            s.friction = row.friction;
        }
        if (fits) {
            const int iterations = BallVelocityContactIterations();
            const bool use_fma = mutator.solver_use_fma != 0;
            BallVelocityAccumulator acc0{};
            BallVelocityAccumulator acc1{};
            BallVelocityAccumulator ball_acc{};
            for (int iter = 0; iter < iterations; ++iter) {
                bool changed = false;
#pragma unroll
                for (int o = 0; o < kRegRows; ++o) {
                    RegRow& s = normal[o];
                    if (!s.active) {
                        continue;
                    }
                    const float applied_before = s.applied;
                    BallVelocityAccumulator acc = s.body_slot == 2 ? ball_acc : (s.body_slot == 1 ? acc1 : acc0);
                    const float delta_impulse_vel_a_bt =
                        SolverRowDot(s.dir, acc.delta_linear, use_fma) +
                        SolverRowDot(s.rel_cross, acc.delta_angular, use_fma);
                    float delta_impulse;
                    if (s.has_rhs_bt) {
                        float delta_impulse_bt = s.rhs;
                        delta_impulse_bt = SolverRowSubtract(delta_impulse_bt, delta_impulse_vel_a_bt, s.jac, use_fma);
                        if (!s.is_ball) {
                            delta_impulse_bt = SolverRowSubtract(delta_impulse_bt, 0.0f, s.jac, use_fma);
                        }
                        const float sum_bt = s.applied + delta_impulse_bt;
                        if (sum_bt < 0.0f) {
                            delta_impulse = -s.applied;
                            s.applied = 0.0f;
                        } else {
                            delta_impulse = delta_impulse_bt;
                            s.applied = sum_bt;
                        }
                    } else {
                        const float delta_impulse_vel_a_uu = delta_impulse_vel_a_bt * kUnifiedRowConversionUuPerBt;
                        const float applied_uu = s.applied * kUnifiedRowConversionUuPerBt;
                        delta_impulse = s.rhs;
                        delta_impulse = SolverRowSubtract(delta_impulse, delta_impulse_vel_a_uu, s.jac, use_fma);
                        if (!s.is_ball) {
                            delta_impulse = SolverRowSubtract(delta_impulse, 0.0f, s.jac, use_fma);
                        }
                        const float sum = applied_uu + delta_impulse;
                        if (sum < 0.0f) {
                            delta_impulse = -applied_uu;
                            s.applied = 0.0f;
                        } else {
                            s.applied = sum * kUnifiedRowConversionBtPerUu;
                        }
                        delta_impulse *= kUnifiedRowConversionBtPerUu;
                    }
                    changed = changed || GlibcFloatBits(applied_before) != GlibcFloatBits(s.applied);
                    if (delta_impulse == 0.0f) {
                        continue;
                    }
                    changed = true;
                    ApplyBallVelocityAccumulatorImpulse(acc, s.linear, s.angular, delta_impulse, use_fma);
                    if (s.body_slot == 2) {
                        ball_acc = acc;
                    } else if (s.body_slot == 1) {
                        acc1 = acc;
                    } else {
                        acc0 = acc;
                    }
                }
#pragma unroll
                for (int o = 0; o < kRegRows; ++o) {
                    RegRow& s = fric[o];
                    if (!s.active) {
                        continue;
                    }
                    float total_impulse = normal[0].applied;
#pragma unroll
                    for (int k = 1; k < kRegRows; ++k) {
                        if (s.normal_slot == k) {
                            total_impulse = normal[k].applied;
                        }
                    }
                    if (total_impulse <= 0.0f) {
                        continue;
                    }
                    const float max_impulse = s.friction * total_impulse;
                    float delta_impulse = s.rhs;
                    BallVelocityAccumulator acc = s.body_slot == 2 ? ball_acc : (s.body_slot == 1 ? acc1 : acc0);
                    const float delta_impulse_vel_a_bt =
                        SolverRowDot(s.dir, acc.delta_linear, use_fma) +
                        SolverRowDot(s.rel_cross, acc.delta_angular, use_fma);
                    delta_impulse = SolverRowSubtract(delta_impulse, delta_impulse_vel_a_bt, s.jac, use_fma);
                    const float applied_before = s.applied;
                    const float sum = s.applied + delta_impulse;
                    if (sum < -max_impulse) {
                        delta_impulse = -max_impulse - s.applied;
                        s.applied = -max_impulse;
                    } else if (sum > max_impulse) {
                        delta_impulse = max_impulse - s.applied;
                        s.applied = max_impulse;
                    } else {
                        s.applied = sum;
                    }
                    changed = changed || GlibcFloatBits(applied_before) != GlibcFloatBits(s.applied);
                    if (delta_impulse == 0.0f) {
                        continue;
                    }
                    changed = true;
                    ApplyBallVelocityAccumulatorImpulse(acc, s.linear, s.angular, delta_impulse, use_fma);
                    if (s.body_slot == 2) {
                        ball_acc = acc;
                    } else if (s.body_slot == 1) {
                        acc1 = acc;
                    } else {
                        acc0 = acc;
                    }
                }
                if (!changed) {
                    break;
                }
            }
            const ::rocketgpu::Vec3 ball_base_linear_bt = bt_ball_linear_velocity_override
                ? *bt_ball_linear_velocity_override
                : ball.vel_bt;
            SetBallLinearVelocityBt(ball, ball_base_linear_bt + ball_acc.delta_linear);
            ball.ang_vel += ball_acc.delta_angular;
            for (int i = 0; i < count; ++i) {
                const BallVelocityAccumulator acc =
                    i == body0 ? acc0 : (i == body1 ? acc1 : BallVelocityAccumulator{});
                SetCarLinearVelocityBt(cars[i], cars[i].vel_bt + acc.delta_linear);
                cars[i].ang_vel += acc.delta_angular;
            }
            return true;
        }
    }
    return false;
}

ROCKETGPU_HD void SolveUnifiedVelocityContactRows(
    ::rocketgpu::BallState& ball,
    int count,
    CarSpanT cars,
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
    const ::rocketgpu::Vec3* bt_ball_linear_velocity_override = nullptr,
    EnvStridedArray<const BulletManifoldEntry> sorted_manifolds = nullptr,
    int sorted_manifold_count = 0,
    bool register_iterations = false) {
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

    constexpr int kRegRows = 4;

    if (register_iterations && pair_normal_row_count <= 0 && pair_friction_row_count <= 0 &&
        world_normal_row_count + ball_normal_row_count <= kRegRows &&
        world_friction_row_count + ball_friction_row_count <= kRegRows &&
        TrySolveUnifiedVelocityContactRowsReg(
            ball, count, cars, mutator,
            ball_normal_rows, ball_normal_row_count,
            ball_friction_rows, ball_friction_row_count,
            world_normal_rows, world_normal_row_count,
            world_friction_rows, world_friction_row_count,
            bt_ball_linear_velocity_override,
            sorted_manifolds, sorted_manifold_count)) {
        return;
    }

    SolveUnifiedVelocityContactRowsGeneral(
        ball, count, cars, car_configs, mutator,
        ball_normal_rows, ball_normal_row_count,
        ball_friction_rows, ball_friction_row_count,
        world_normal_rows, world_normal_row_count,
        pair_normal_rows, pair_normal_row_count,
        world_friction_rows, world_friction_row_count,
        pair_friction_rows, pair_friction_row_count,
        bt_ball_linear_velocity_override,
        sorted_manifolds, sorted_manifold_count);
}

ROCKETGPU_HD void SolveCarWorldVelocityContactRows(
    CarRefT car,
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

ROCKETGPU_HD void IntegrateSplitImpulseTransformBt(
    CarRefT car,
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
    IntegrateCarOrientation(car, split_bt.turn_velocity * BallSplitTurnErp(), dt, native_sse_math);
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
    row.rhs_penetration = ((penetration * SplitImpulsePenetrationErp()) / dt) * row.jac_diag_ab_inv;

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
        ((penetration_bt * SplitImpulsePenetrationErp()) / dt) * row.jac_diag_ab_inv_bt;
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

    const float scaled_penetration = penetration * BallStaticSplitPenetrationScale();
    if (scaled_penetration <= 0.0f || dt <= 0.0f) {
        rows[row_count] = BallSplitContactRow{};
        return;
    }

    BallSplitContactRow row;
    InitializeBallSplitContactRow(
        row,
        scaled_penetration,
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
    }
    rows[row_count] = row;
    if (row.valid) {
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
    CarCRefT car,
    const ::rocketgpu::CarConfig& cfg,
    const ::rocketgpu::MutatorConfig& mutator,
    float dt,
    const ::rocketgpu::Vec3* rel_ball_contact_bt = nullptr,
    const ::rocketgpu::Vec3* rel_car_contact_bt = nullptr,
    const float* penetration_bt = nullptr,
    const ::rocketgpu::BallState* ball_for_bt = nullptr,
    const ::rocketgpu::Vec3* car_inv_bt_rows = nullptr) {
    if (!rows || row_count >= kMaxBallSplitContactRows) {
        return;
    }

    const float scaled_penetration = penetration * BallCarSplitPenetrationScale();
    if (scaled_penetration <= 0.0f || dt <= 0.0f) {
        rows[row_count] = BallSplitContactRow{};
        return;
    }

    BallSplitContactRow row;

    ::rocketgpu::Vec3 car_angular_component_bt(0.0f, 0.0f, 0.0f);
    if (rel_car_contact_bt) {
        const ::rocketgpu::Vec3 torque_bt = -Cross(*rel_car_contact_bt, normal);
        car_angular_component_bt = car_inv_bt_rows
            ? ::rocketgpu::Vec3(
                  Dot(car_inv_bt_rows[0], torque_bt),
                  Dot(car_inv_bt_rows[1], torque_bt),
                  Dot(car_inv_bt_rows[2], torque_bt))
            : ApplyInverseBodyInertiaWorldBt(car, cfg, mutator.car_mass, torque_bt);
    }

    ::rocketgpu::Vec3 ball_angular_component_bt(0.0f, 0.0f, 0.0f);
    if (rel_ball_contact_bt && ball_for_bt) {
        ball_angular_component_bt = ApplyInverseBallInertiaWorldBtMutator(*ball_for_bt, mutator, Cross(*rel_ball_contact_bt, normal));
    }
    InitializeBallSplitContactRow(
        row,
        scaled_penetration,
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
    rows[row_count] = row;
    if (row.valid) {
        ++row_count;
    }
}

ROCKETGPU_HD int DebugWorldSubsteps() {
    return 1;
}

ROCKETGPU_HD float CarContactBreakingThresholdBt(const ::rocketgpu::CarConfig& cfg) {
    const ::rocketgpu::Vec3 half = (BulletHitboxSize(cfg) * 0.02f) * 0.5f;
    const float margin = Min(0.1f * Min(Min(half.x, half.y), half.z), 0.04f);
    const ::rocketgpu::Vec3 margin_v(margin, margin, margin);
    const ::rocketgpu::Vec3 half_with_margin = (half - margin_v) + margin_v;
    const ::rocketgpu::Vec3 center = cfg.hitbox_offset * 0.02f;
    const ::rocketgpu::Vec3 aabb_min = center - half_with_margin;
    const ::rocketgpu::Vec3 aabb_max = center + half_with_margin;
    const ::rocketgpu::Vec3 sphere_center = (aabb_min + aabb_max) * 0.5f;
    const float radius = Length(aabb_max - aabb_min) * 0.5f;
    return (radius + Length(sphere_center)) * 0.02f;
}

ROCKETGPU_HD void ResolveObbCornerCylinder(
    CarRefT car,
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
#endif
