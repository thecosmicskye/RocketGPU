#pragma once

#include <cstddef>
#include <cstdint>

#include "rocketgpu/detail/common.hpp"

namespace rocketgpu {
namespace detail {
struct ArenaMeshTriangle;
struct ArenaMeshBvhQuantization;
struct ArenaMeshBvhNode;
struct ArenaMeshBvhSubtreeInfo;
struct ArenaMeshLeafGridInfo;
struct ArenaMeshLeafEntry;
struct ArenaMeshBvhMeshInfo;
struct BulletSweepHistory;
struct WheelRuntimeState;
struct KickoffSettledRow;
struct WorldSolverScratch;
struct WorldSolverScratchRowPoolHost;
struct RespawnRngState;
#if !defined(ROCKETGPU_USE_CUDA)
struct HostStepTables;
#endif
}

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    ROCKETGPU_HD Vec3() = default;
    ROCKETGPU_HD constexpr Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

enum class PhysicsEventType : uint8_t {
    BallTouchStart, BallWorldContact, CarWorldContact, BoostPickup, CarBump, Goal
};

struct PhysicsEvent {
    uint64_t tick = 0;
    PhysicsEventType type = PhysicsEventType::BallTouchStart;
    uint32_t car_id = 0;
    uint32_t other_car_id = 0;
    int32_t team = -1;
    uint32_t pad_index = 0;
    uint8_t car_on_ground = 0;
    uint8_t is_big_pad = 0;
    uint8_t is_demo = 0;
    Vec3 position = {};
    Vec3 normal = {};
    Vec3 car_velocity = {};
    Vec3 ball_position = {};
    Vec3 ball_velocity = {};
    float boost_before = 0;
    float boost_after = 0;
};

struct PhysicsEventWindow {
    uint64_t tick = 0;
    uint32_t count = 0;
    uint32_t overflow = 0;
};

struct PhysicsEventStream {
    PhysicsEvent* records = nullptr;
    PhysicsEventWindow* windows = nullptr;
    uint32_t capacity_per_env = 0;
    uint32_t env_count = 0;
    uint32_t cars_per_env = 0;
    bool goal_only = false;

    ROCKETGPU_HD bool HasNonGoalEvents() const { return windows && !goal_only; }

    ROCKETGPU_HD void Emit(uint32_t env, PhysicsEvent event) const {
        if (goal_only && event.type != PhysicsEventType::Goal) return;
        if (!windows || env >= env_count) return;
        auto& window = windows[env];
        event.tick = window.tick;
        if (window.count >= capacity_per_env) {
            window.overflow = 1;
            return;
        }
        records[static_cast<size_t>(env) * capacity_per_env + window.count++] = event;
    }
    ROCKETGPU_HD void EmitForCar(uint32_t id, PhysicsEvent event) const {
        if (id && cars_per_env) Emit((id - 1) / cars_per_env, event);
    }
};

struct Quat {
    float w = 1.0f;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    ROCKETGPU_HD Quat() = default;
    ROCKETGPU_HD constexpr Quat(float w_, float x_, float y_, float z_) : w(w_), x(x_), y(y_), z(z_) {}
};

enum class Team : uint8_t {
    Blue = 0,
    Orange = 1,
};

enum class Mode : uint8_t {
    Soccar = 0,
    Hoops = 1,
    Heatseeker = 2,
    Snowday = 3,
    Dropshot = 4,
    TheVoid = 5,
};

struct CarControls {
    float throttle = 0.0f;
    float steer = 0.0f;
    float pitch = 0.0f;
    float yaw = 0.0f;
    float roll = 0.0f;
    uint8_t jump = 0;
    uint8_t boost = 0;
    uint8_t handbrake = 0;
};

struct CarConfig {
    // Hitbox rule (every body): Psyonix's published hitbox (as the game reports it) plus the 2.5 uu the game's Bullet
    // box adds; offsets as published. RocketSim's numbers are these rounded to 6 digits, which leaves the Octane
    // front face 1.5e-4 uu short (a constant split-impulse ball push deficit on front hits).
    Vec3 hitbox_size = Vec3(118.00738f + 2.5f, 84.19941f + 2.5f, 36.159073f + 2.5f);
    // The published extent itself: hitbox_size - 2.5 rounds 1 ulp off it for some bodies (Dominus, Hybrid), and the
    // game's inertia is built from it. Used only while hitbox_size == hitbox_published + 2.5.
    Vec3 hitbox_published = Vec3(118.00738f, 84.19941f, 36.159073f);
    Vec3 hitbox_offset = Vec3(13.87566f, 0.0f, 20.754988f);
    // Wheels (the game's wheel setup): x, y from the handling preset's axle rest position; z (the suspension ray's
    // start) the hitbox centre's z; rest = hitbox z - axle rest z + 12 (the suspension travel).
    Vec3 front_wheels_offset = Vec3(51.25f, 25.90f, 20.754988f);
    Vec3 back_wheels_offset = Vec3(-33.75f, 29.50f, 20.754988f);
    float front_wheel_radius = 12.50f;
    float back_wheel_radius = 15.00f;
    float front_suspension_rest = 38.75499f;
    float back_suspension_rest = 37.054989f;
    float dodge_deadzone = 0.5f;
    // The Psyclops sim (body_tritip's Mutator_Robin): three-wheel steer and lateral friction curves, no ground sticky
    // force, auto-flip time 0.2 s, wheel ContactForceDistance 4 uu, the position-seeded jump / double-jump torques,
    // and air control that stops on world contact and adds a position-seeded torque while boosting.
    uint8_t three_wheels = 0;
    // Steer angle curve (degrees) at |forward speed| 0, 500, 1000, 1500, 1750, 3000 uu/s, and the powerslide one at
    // 0, 2500. The game's keys are 35, 20, 11, 6.25, 5, 2 and 25, 7.5 degrees; with Ackermann steering off it replaces
    // each key by the mean of the inner wheel angle a and the outer one atan(L / (T + L / tan a)), L = wheelbase
    // (front x - back x), T = front track (2 * front y). So each body has its own curve (values below: Octane).
    float steer_angle_curve_deg[6] = {30.570510864257812f, 18.294261932373047f, 10.429570198059082f,
                                      6.0558881759643555f, 4.874049186706543f, 1.9791786670684814f};
    float powerslide_steer_angle_curve_deg[2] = {22.478458404541016f, 7.224311351776123f};
    // The car's z (uu) when the game spawns it (demo respawn, kickoff re-creation): 36 plus the body's visual suspension
    // offset, rounded to 0.01. The offset is the max over the front and back axles of 2 * (body wheel radius - preset
    // wheel radius) + (preset wheel rest z - the body mesh's wheel bone z), plus HandlingPresetZOffset, plus
    // SuspensionTravelMin for bodies without legacy suspension offsets (these use the wheel mesh radius); it is 0 for a
    // body without a preset or a front wheel bone. Bodies sharing a hitbox differ (Fennec 38.61, Octane 36.02), so set it
    // per body. Defaults: each preset's stock body (Octane 36.02; see the presets).
    float spawn_z = 36.02f;
};

struct BallHitInfo {
    uint8_t is_valid = 0;
    uint8_t extra_pending = 0;
    uint8_t extra_tick_is_previous = 0;
    uint8_t reserved2 = 0;
    uint64_t tick_count_when_hit = 0;
    uint64_t tick_count_when_extra_impulse_applied = 0;
    uint32_t hit_car_id = 0;
    Vec3 relative_pos_on_ball = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 ball_pos = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 extra_hit_vel = Vec3(0.0f, 0.0f, 0.0f);
};

struct CarState {
    Vec3 pos = Vec3(0.0f, 0.0f, 17.0f);
    Quat rot = Quat(1.0f, 0.0f, 0.0f, 0.0f);
    // Fields RLBot exposes hold RLBot's value for the tick (see internal_* at the end for the step's own values):
    // vel and ang_vel after the 2300 uu/s and 5.5 rad/s caps.
    Vec3 vel = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 ang_vel = Vec3(0.0f, 0.0f, 0.0f);

    Vec3 flip_rel_torque = Vec3(0.0f, 0.0f, 0.0f);

    // Timers follow RocketSim v3 (car_state.rs) and the game. Seconds; a tick is tick_dt.
    // Ticks since the jump started (while is_jumping) or ended (after), times tick_dt: v3 jump_ticks / 120.
    // Advances every tick the car is not demoed, on the ground too. Exact multiple of tick_dt.
    float jump_time = 0.0f;
    // Seconds since flip activity last changed: flip start, flip end, or landing mid-flip (v3 flip_time).
    // Reads one tick on the tick it changes, then advances every tick the car is not demoed, on the ground too.
    float flip_time = 0.0f;
    float air_time = 0.0f;  // Total time in the air; 0 on the ground.
    float air_time_since_jump = 0.0f;  // Time in the air since the jump ended; 0 on the ground or without a jump.
    float boost = 100.0f;
    float time_since_boosted = 0.0f;  // Seconds since boost was last applied.
    float boosting_time = 0.0f;  // Seconds of the current boost streak; 0 when not boosting.
    float supersonic_time = 0.0f;  // v3 supersonic_grace_timer: time below start speed while still supersonic.
    float handbrake_val = 0.0f;  // Smoothed handbrake, 0..1.
    float auto_flip_timer = 0.0f;  // Seconds left in auto-flip recovery.
    float auto_flip_torque_scale = 0.0f;  // Roll direction sign of the auto-flip, +1 or -1.
    float demo_respawn_timer = 0.0f;
    // Ticks since the last state set. Set it to 0 whenever you write a state: the next step then rebuilds
    // RocketGPU's internal clocks (internal_* below) from jump_time, flip_time and the flags.
    uint64_t tick_count_since_update = 0;

    uint8_t ball_contact_last_tick = 0;
    uint8_t ball_contact_this_tick = 0;
    ROCKETGPU_HD bool HasFlipOrJump() const {
        return is_on_ground || (!has_flipped && !has_double_jumped && air_time_since_jump < 1.25f);
    }
    ROCKETGPU_HD void ResetTrainingHistory() {
        restored_inv_inertia_valid = 0;
        world_contact.wheel_ball_mask = 0;
        previous_ball_contact_start = last_ball_contact_start = last_world_surface_contact = ~0ULL;
        first_world_after_previous_ball = first_world_after_last_ball = ~0ULL;
        last_flip_or_jump_start = last_directional_flip_start = ~0ULL;
        ball_contact_last_tick = ball_contact_this_tick = last_ball_started_with_world = 0;
        flip_or_jump_last_tick = HasFlipOrJump();
    }
    ROCKETGPU_HD void RecordWorldSurface(bool body_contact) {
        last_world_surface_contact = tick_count_since_update;
        if (body_contact && last_ball_contact_start == tick_count_since_update)
            last_ball_started_with_world = 1;
        if (last_ball_contact_start != ~0ULL && first_world_after_last_ball == ~0ULL
            && tick_count_since_update > last_ball_contact_start)
            first_world_after_last_ball = tick_count_since_update;
    }
    ROCKETGPU_HD void RecordBallContactStart() {
        previous_ball_contact_start = last_ball_contact_start;
        first_world_after_previous_ball = first_world_after_last_ball;
        last_ball_contact_start = tick_count_since_update;
        first_world_after_last_ball = ~0ULL;
        last_ball_started_with_world = last_world_surface_contact == tick_count_since_update;
    }

    uint8_t is_on_ground = 1;
    uint8_t wheels_with_contact[4] = {1,1,1,1};
    // RLBot: has_jumped clears on a tick air_state reads OnGround more than 0.05 s after the jump started (it stays
    // clear if the car then leaves the ground); has_double_jumped and has_flipped clear on the tick air_state reads
    // OnGround.
    uint8_t has_jumped = 0;
    uint8_t has_double_jumped = 0;
    uint8_t has_flipped = 0;
    uint8_t is_flipping = 0;
    uint8_t is_jumping = 0;
    uint8_t is_boosting = 0;
    uint8_t is_supersonic = 0;  // RLBot: |vel| >= 2200 on this tick. The game's flag, held by supersonic_time, is internal.
    uint8_t is_auto_flipping = 0;
    uint8_t is_demoed = 0;
    uint8_t just_respawned = 0;

    uint8_t demo_no_contact_response = 0;
    // The respawn slot (0-3) plus one that this car's next demo respawn takes instead of the RNG's pick; 0: none.
    // 5-9: the spawn is a kickoff re-creation at kickoff spot 0-4 (kKickoffSpawnOverridePlus1). Consumed by that spawn.
    uint8_t respawn_slot_override_plus1 = 0;

    struct {
        uint8_t has_contact = 0;
        uint8_t wheel_ball_mask = 0;
        uint8_t reserved1 = 0;
        uint8_t reserved2 = 0;
        Vec3 contact_normal = Vec3(0.0f, 0.0f, 0.0f);
    } world_contact;

    struct {
        uint32_t other_car_id = 0;
        float cooldown_timer = 0.0f;
    } car_contact;

    Vec3 vel_impulse_cache = Vec3(0.0f, 0.0f, 0.0f);
    BallHitInfo ball_hit_info = {};
    CarControls last_controls = {};

    Vec3 pos_bt = Vec3(0.0f, 0.0f, 17.0f * (1.0f / 50.0f));
    Vec3 vel_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 tick_start_pos_bt = Vec3(0.0f, 0.0f, 17.0f * (1.0f / 50.0f));
    Vec3 pad_frame_pos_bt = Vec3(0.0f, 0.0f, 17.0f * (1.0f / 50.0f));

    uint8_t vel_touched_solve_stage = 1;
    uint8_t basis_authoritative = 0;
    uint8_t ext_force_impulse_bt_valid = 0;
    uint8_t pad_frame_pos_valid = 0;

    Vec3 basis_forward = Vec3(1.0f, 0.0f, 0.0f);
    Vec3 basis_right = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 basis_up = Vec3(0.0f, 0.0f, 1.0f);

    Vec3 ext_force_impulse_bt = Vec3(0.0f, 0.0f, 0.0f);

    Vec3 ext_torque_impulse = Vec3(0.0f, 0.0f, 0.0f);

    Vec3 solve_delta_linear_bt = Vec3(0.0f, 0.0f, 0.0f);

    Vec3 vel_impulse_cache_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 solve_delta_angular = Vec3(0.0f, 0.0f, 0.0f);

    uint64_t bump_event_tick = 0;
    uint32_t bump_event_other_id = 0;
    uint8_t bump_event_is_demo = 0;
    uint8_t bump_event_was_bumper = 0;
    // The pad (index + 1, 0 for none) on the car's touch list: begun by entering its pickup cylinder at a render
    // frame's end, ended when the car stops overlapping it or anyone picks the pad up.
    uint8_t boost_pad_touching = 0;
    uint8_t bump_event_reserved1 = 0;

    uint64_t previous_ball_contact_start = ~0ULL;
    uint64_t last_ball_contact_start = ~0ULL;
    uint64_t last_world_surface_contact = ~0ULL;
    uint64_t first_world_after_previous_ball = ~0ULL;
    uint64_t first_world_after_last_ball = ~0ULL;
    uint64_t last_flip_or_jump_start = ~0ULL;
    uint64_t last_directional_flip_start = ~0ULL;
    uint8_t last_ball_started_with_world = 0;
    uint8_t flip_or_jump_last_tick = 1;

    float internal_jump_time = 0.0f;
    float internal_flip_time = 0.0f;
    uint8_t flip_timer_live = 0;

    // RLBot v5 car fields, reported after every tick (not read by the step).
    uint8_t air_state = 0;  // RLBot AirState: 0 OnGround, 1 Jumping, 2 DoubleJumping (13 ticks), 3 Dodging, 4 InAir.
    // Ticks left with the game's Car_TA bDriving off (the round's countdown; 0 = driving), counted down after each
    // tick. Not driving: no engine, brake (not even the coast brake), handbrake, wheel steer, jump, boost or air
    // control; the throttle input still gates the sticky force, auto-roll and tyre friction.
    uint16_t driving_disabled = 0;
    // Seconds left to dodge after the jump: 1.25 minus the time since the jump ended (the release tick counts one tick).
    // -1 on the ground, while jumping, after a double jump or dodge, without has_jumped, or once it runs out.
    float dodge_timeout = -1.0f;
    float dodge_elapsed = 0.0f;  // Seconds since the dodge started, while has_flipped (not OnGround); else 0.
    float dodge_dir_x = 0.0f;  // Direction of the last dodge: (flip_rel_torque.y, -flip_rel_torque.x).
    float dodge_dir_y = 0.0f;
    // Latest ball touch: the tick a ball contact started, the car-side contact point, and the unit normal from the
    // car toward the ball centre. latest_touch_tick is ~0 until the car first touches the ball.
    uint64_t latest_touch_tick = ~0ULL;
    Vec3 latest_touch_location = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 latest_touch_normal = Vec3(0.0f, 0.0f, 0.0f);

    Vec3 internal_vel = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 internal_ang_vel = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 reported_vel = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 reported_ang_vel = Vec3(0.0f, 0.0f, 0.0f);
    uint8_t internal_has_jumped = 0;
    uint8_t internal_has_double_jumped = 0;
    uint8_t internal_has_flipped = 0;
    uint8_t internal_is_supersonic = 0;
    uint8_t reported_flags = 0;
    uint8_t game_has_jumped = 0;
    uint8_t double_jump_ticks = 255;
    // v3's jump_hold_broken (car/base.rs update_jump): the jump's hold was released; set only while jumping.
    uint8_t jump_hold_broken = 0;
    uint8_t supersonic_set_v3 = 0;
    Vec3 set_wheel_normals[4] = {};
    uint8_t set_wheel_normals_mask = 0;
    uint8_t cal_pre_flags = 0;
    float cal_pre_forward_speed_uu = 0.0f;
    float cal_pre_pitchlock_restart_time = -1.0f;
    Vec3 cal_tick_start_ang_vel = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 tick_start_basis_forward = Vec3(1.0f, 0.0f, 0.0f);
    Vec3 tick_start_basis_right = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 tick_start_basis_up = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 restored_inv_inertia_bt[3] = {};
    uint8_t restored_inv_inertia_valid = 0;
    static constexpr int kMaxPendingBumps = 7;
    Vec3 pending_bump_bt[kMaxPendingBumps] = {};
    uint32_t pending_bump_hitter[kMaxPendingBumps] = {};
    uint8_t pending_bump_count = 0;

    static constexpr uint8_t kReportedVel = 1, kReportedAngVel = 2, kReportedHasJumped = 4,
        kReportedHasDoubleJumped = 8, kReportedHasFlipped = 16, kReportedIsSupersonic = 32, kReportedValid = 128;

    ROCKETGPU_HD uint8_t RestoreWorkingFields() {
        if (!(reported_flags & kReportedValid)) {
            return 0;
        }
        uint8_t kept = kReportedValid;
        if (vel.x == reported_vel.x && vel.y == reported_vel.y && vel.z == reported_vel.z) {
            vel = internal_vel;
            kept |= kReportedVel;
        }
        if (ang_vel.x == reported_ang_vel.x && ang_vel.y == reported_ang_vel.y && ang_vel.z == reported_ang_vel.z) {
            ang_vel = internal_ang_vel;
            kept |= kReportedAngVel;
        }
        RestoreWorkingFlag(has_jumped, internal_has_jumped, kReportedHasJumped, kept);
        RestoreWorkingFlag(has_double_jumped, internal_has_double_jumped, kReportedHasDoubleJumped, kept);
        RestoreWorkingFlag(has_flipped, internal_has_flipped, kReportedHasFlipped, kept);
        RestoreWorkingFlag(is_supersonic, internal_is_supersonic, kReportedIsSupersonic, kept);
        return kept;
    }
    ROCKETGPU_HD void ReReportFields(uint8_t kept) {
        if (!(kept & kReportedValid)) {
            return;
        }
        internal_vel = vel;
        internal_ang_vel = ang_vel;
        internal_has_jumped = has_jumped;
        internal_has_double_jumped = has_double_jumped;
        internal_has_flipped = has_flipped;
        internal_is_supersonic = is_supersonic;
        if (kept & kReportedVel) vel = reported_vel; else reported_vel = vel;
        if (kept & kReportedAngVel) ang_vel = reported_ang_vel; else reported_ang_vel = ang_vel;
        uint8_t flags = kReportedValid;
        ReReportFlag(has_jumped, kReportedHasJumped, kept, flags);
        ReReportFlag(has_double_jumped, kReportedHasDoubleJumped, kept, flags);
        ReReportFlag(has_flipped, kReportedHasFlipped, kept, flags);
        ReReportFlag(is_supersonic, kReportedIsSupersonic, kept, flags);
        reported_flags = flags;
    }
    ROCKETGPU_HD void RestoreWorkingFlag(uint8_t& field, uint8_t internal, uint8_t bit, uint8_t& kept) const {
        if ((field != 0) == ((reported_flags & bit) != 0)) {
            field = internal;
            kept |= bit;
        }
    }
    ROCKETGPU_HD void ReReportFlag(uint8_t& field, uint8_t bit, uint8_t kept, uint8_t& flags) const {
        if (kept & bit) {
            field = (reported_flags & bit) ? 1 : 0;
        }
        if (field) {
            flags |= bit;
        }
    }
};

struct BallState {
    struct HeatseekerInfo {
        float y_target_dir = 0.0f;
        float cur_target_speed = 2900.0f;
        float time_since_hit = 0.0f;
    } hs_info = {};

    struct DropshotInfo {
        int32_t charge_level = 1;
        float accumulated_hit_force = 0.0f;
        float y_target_dir = 0.0f;
        uint8_t has_damaged = 0;
        uint8_t reserved0 = 0;
        uint8_t reserved1 = 0;
        uint8_t reserved2 = 0;
        uint64_t last_damage_tick = 0;
    } ds_info = {};

    Vec3 pos = Vec3(0.0f, 0.0f, 93.15f);
    Quat rot = Quat(1.0f, 0.0f, 0.0f, 0.0f);
    Vec3 vel = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 ang_vel = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 vel_impulse_cache = Vec3(0.0f, 0.0f, 0.0f);
    uint64_t tick_count_when_extra_impulse_applied = 0;
    uint64_t tick_count_since_update = 0;

    Vec3 pos_bt = Vec3(0.0f, 0.0f, 93.15f * (1.0f / 50.0f));
    Vec3 vel_bt = Vec3(0.0f, 0.0f, 0.0f);

    Vec3 basis_forward = Vec3(1.0f, 0.0f, 0.0f);
    Vec3 basis_right = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 basis_up = Vec3(0.0f, 0.0f, 1.0f);
    uint32_t basis_authoritative = 0;
    Vec3 inertia_basis_forward = Vec3(1.0f, 0.0f, 0.0f);
    Vec3 inertia_basis_right = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 inertia_basis_up = Vec3(0.0f, 0.0f, 1.0f);
    uint32_t inertia_basis_valid = 0;
    Vec3 restored_inv_inertia_bt[3] = {};
    uint32_t restored_inv_inertia_valid = 0;

    Vec3 vel_impulse_cache_bt = Vec3(0.0f, 0.0f, 0.0f);

    // Bullet activation: 0 awake, 1 asleep (ISLAND_SLEEPING: no gravity, frozen until a car touches it), 2 just
    // uploaded (the next step infers it, see ResolveBallActivationAtTickStart); bit 4 is set within a step when a wheel
    // pushes the ball (the world step clears it). deactivation_time is Bullet's
    // m_deactivationTime: seconds spent below the sleeping thresholds.
    uint32_t sleeping = 0;
    float deactivation_time = 0.0f;
    // Ticks left with the game's ball 2D constraint (RBActor_TA SetConstrained2D: on from the ball's spawn in the
    // round countdown, off at StartRound; 0 = free), counted down after each tick. Ball_TA's contact callback gives
    // every new contact point restitution 0 while it is on, so the spawn drop lands without a bounce.
    uint16_t constrained_2d = 0;
    uint16_t reserved_c2d = 0;
};

struct BoostPadState {
    float cooldown = 0.0f;
    uint8_t is_active = 1;
    uint8_t is_big = 0;
    uint16_t pickup_frac_q = 32768;
    uint32_t prev_locked_car_id = 0;
    uint32_t cur_locked_car_id = 0;
};

struct BoostPadConfig {
    Vec3 pos = Vec3(0.0f, 0.0f, 0.0f);
    uint8_t is_big = 0;
    uint8_t reserved0 = 0;
    uint8_t reserved1 = 0;
    uint8_t reserved2 = 0;
};

constexpr int kMaxBoostPads = 48;
// CarState.respawn_slot_override_plus1 = kKickoffSpawnOverridePlus1 + spot: the car's next spawn is a kickoff
// re-creation at kickoff spot `spot` (0-4).
constexpr int kKickoffSpawnOverridePlus1 = 5;

struct EnvState {
    uint32_t tick = 0;
    uint32_t blue_score = 0;
    uint32_t orange_score = 0;
    int32_t last_scored_team = -1;
    uint8_t goal_scored_this_tick = 0;

    // The order the per-car loops (vehicle updates first) visit the cars: 3 (default) car_visit_order_packed kept as
    // the game's vehicle order (UpdateGameCarVisitOrder); for comparisons, 0 slot order, 1 car_visit_order_packed as
    // set, 2 reverse slot order.
    uint8_t car_visit_order_mode = 3;
    uint8_t car_visit_present_mask = 0xFFu;
    // Resets so far (wraps at 256); picks the kickoff shuffle when the reset's kickoff_seed is 0.
    uint8_t kickoff_resets = 0;

    // Slot k visited k-th, 4 bits each. In modes 1 and 3, a step needs the env's present slots (< its car count)
    // listed first; stepping any other order is undefined.
    uint32_t car_visit_order_packed = 0x76543210u;

    uint8_t countdown_phase = 0;
    uint8_t countdown_fires = 0;
    uint16_t frame_ticks = 0;
    float countdown_count = 0.0f;
    float frame_frac_prev = 0.5f;
};

ROCKETGPU_HD int CarVisitSlot(const EnvState& env, int k, int count) {
    if (env.car_visit_order_mode == 1 || env.car_visit_order_mode == 3) {
        return static_cast<int>((env.car_visit_order_packed >> (4 * k)) & 0xFu);
    }
    if (env.car_visit_order_mode == 2) return count - 1 - k;
    return k;
}

ROCKETGPU_HD uint32_t PackCarVisitOrder(const int* slots, int count) {
    uint32_t packed = 0;
    for (int k = 0; k < count && k < 8; ++k) {
        packed |= (static_cast<uint32_t>(slots[k]) & 0xFu) << (4 * k);
    }
    return packed;
}

ROCKETGPU_HD void UpdateGameCarVisitOrder(EnvState& env, const CarState* env_cars, int count) {
    if (env.car_visit_order_mode != 3) return;
#if !defined(ROCKETGPU_USE_CUDA)
    bool change = false;
    for (int s = 0; s < count && s < 8; ++s) {
        change |= (((env.car_visit_present_mask >> s) & 1u) != 0u) == (env_cars[s].is_demoed != 0);
    }
    if (!change) return;
#endif
    int order[8];
    for (int k = 0; k < 8; ++k) order[k] = static_cast<int>((env.car_visit_order_packed >> (4 * k)) & 0xFu);
    uint32_t present = env.car_visit_present_mask;
    const auto is_present = [&](int slot) { return slot < count && ((present >> slot) & 1u) != 0u; };
    const auto last_present = [&]() {
        int q = -1;
        for (int k = 0; k < 8; ++k) if (is_present(order[k])) q = k;
        return q;
    };
    for (int s = 0; s < count && s < 8; ++s) {
        if (((present >> s) & 1u) || env_cars[s].is_demoed) continue;
        int p = 0;
        while (order[p] != s) ++p;
        const int dst = last_present() + 1;
        for (int k = p; k > dst; --k) order[k] = order[k - 1];
        order[dst < p ? dst : p] = s;
        present |= 1u << s;
    }
    for (int s = 0; s < count && s < 8; ++s) {
        if (!((present >> s) & 1u) || !env_cars[s].is_demoed) continue;
        int p = 0;
        while (order[p] != s) ++p;
        const int q = last_present();
        order[p] = order[q];
        order[q] = s;
        present &= ~(1u << s);
    }
    env.car_visit_order_packed = PackCarVisitOrder(order, 8);
    env.car_visit_present_mask = static_cast<uint8_t>(present);
}
// The game's bDriving stays off for 480 or 481 ticks (4 s) from the round reset (RLBot phase Countdown).
constexpr uint16_t kKickoffCountdownTicks = 480;
// Hold until the game's countdown timers end the countdown (480-482 ticks at about 120 fps;
// 482 with every tick its own frame of exactly 1/120 s, the default fraction).
constexpr uint16_t kKickoffCountdownGameTimers = 0xFFFF;

struct EnvCommand {
    uint8_t reset_to_kickoff = 0;
    uint8_t reset_scoreboard = 0;
    uint8_t reset_tick_count = 0;

    // 2: a new shuffle on every reset, whatever kickoff_seed is. 3: no shuffle, a team's nth car takes kickoff spot n.
    uint8_t kickoff_seed_mode = 0;
    // The bodies' side of a round reset alone (a reset_to_kickoff without reset_scoreboard implies it): every body
    // leaves the world and is re-added the way the game respawns them after a goal (BulletSweepNewRound).
    uint8_t new_round_bodies = 0;
    // With reset_to_kickoff: keep the pads and their timers, as the game does at a goal's kickoff (it never resets
    // them). 0 (the default) starts every pad active, a fresh episode.
    uint8_t keep_boost_pads = 0;
    // With reset_to_kickoff: the round countdown. 0 (the default) starts the kickoff at once, as RocketSim does: the
    // ball asleep at 92.75 and the cars spawned and driving on the reset. kKickoffCountdownGameTimers runs the game's
    // countdown timers on the render frames (EnvState.countdown_phase); a tick count holds that many ticks. With
    // either, the reset is the game's: the ball re-created high and falling under the 2D constraint, the cars one tick
    // later, held (bDriving off) until the countdown ends.
    uint16_t kickoff_countdown_ticks = 0;
    // Seeds the kickoff spot shuffle. 0 (the default): each env takes a new shuffle on every reset.
    uint64_t kickoff_seed = 0;
};

struct MutatorConfig {
    PhysicsEventStream physics_events = {};
    float gravity_z = -650.0f;
    // Game: sphere element 92 uu -> PhysX radius 92 * 0.02 + 0.025 skin -> Bullet r * 50 * 0.02 - 0.04 margin
    // = 1.8249998 bt (bits 0x3fe99998), the float just below 91.25 uu.
    float ball_radius = 91.24999f;
    // Game: the ball body's inverse mass is 0x3d088888 = 1 / 30.000002 (1 / 30 rounds to 0x3d088889).
    float ball_mass = 30.000002f;
    float car_mass = 180.0f;

    float ball_drag = 0.03f;

    float cached_ball_damping_drag = -1.0f;
    float cached_ball_damping_dt = -1.0f;
    float cached_ball_damping_scale = -1.0f;

    uint32_t solver_use_fma = 0;

    const uint32_t* native_sphere_rsqrt_table = nullptr;
    int32_t native_sphere_rsqrt_shift = 0;
    float ball_friction = 0.35f;
    float ball_restitution = 0.60f;
    float ball_max_speed = 6000.0f;
    float ball_max_ang_speed = 6.0f;

    float car_max_speed = 2300.0f;
    float car_max_ang_speed = 5.5f;
    float car_air_throttle_accel = 200.0f / 3.0f;

    float boost_max = 100.0f;
    // CarComponent_Boost_TA BoostConsumptionRate 0.333 and StartBoostAmount 0.333 of MaxBoostAmount 1 (x 100 here).
    float boost_used_per_second = 33.3f;
    float boost_min_time = 0.10f;
    float boost_accel_ground = 2975.0f / 3.0f;
    float boost_accel_air = 3175.0f / 3.0f;
    float boost_spawn_amount = 33.3f;
    float recharge_boost_per_second = 10.0f;
    float recharge_boost_delay = 0.25f;
    uint8_t recharge_boost_enabled = 0;

    float jump_accel = 4375.0f / 3.0f;
    float jump_immediate_speed = 875.0f / 3.0f;
    float jump_min_time = 0.0249f;
    float jump_reset_time_pad = 1.0f / 40.0f;
    float jump_max_time = 0.20f;
    float doublejump_max_delay = 1.25f;

    float flip_torque_time = 0.65f;
    float flip_torque_min_time = 0.41f;
    float flip_pitchlock_time = 1.0f;
    float flip_pitchlock_extra_time = 0.30f;
    float flip_initial_vel_scale = 500.0f;
    float flip_torque_x = 260.0f;
    float flip_torque_y = 224.0f;
    float flip_forward_impulse_max_speed_scale = 1.0f;
    float flip_side_impulse_max_speed_scale = 1.9f;
    float flip_backward_impulse_max_speed_scale = 2.5f;
    float flip_backward_impulse_scale_x = 16.0f / 15.0f;
    uint8_t unlimited_flips = 0;
    uint8_t unlimited_double_jumps = 0;

    uint8_t enable_team_demos = 0;

    float supersonic_start_speed = 2200.0f;
    float supersonic_maintain_min_speed = 2100.0f;
    float supersonic_maintain_max_time = 1.0f;

    float powerslide_rise_rate = 5.0f;
    float powerslide_fall_rate = 2.0f;

    float arena_friction = 0.30f;
    float arena_restitution = 0.30f;

    float sticky_force = 325.0f;

    float ball_car_extra_impulse_z_scale = 0.35f;
    float ball_car_extra_impulse_forward_scale = 0.65f;
    float ball_car_extra_impulse_max_delta_vel = 4600.0f;
    float ball_hit_extra_force_scale = 1.0f;
    float bump_force_scale = 1.0f;
    // GoalVolume_TA mouth plane at |y| = 5120 uu + Ball_TA.Radius (95.4871826171875 uu, StaticMesh BoxExtent.Z + 1)
    // = 5215.4873046875 uu (0x45a2fbe6) = 5124.2373046875 + 91.25 (ball_radius).
    float goal_base_threshold_y = 5124.2373046875f;

    float bump_cooldown_time = 0.25f;
    float bump_min_forward_dist = 64.5f;
    float demo_respawn_time = 3.0f;
    float boost_pad_cooldown_small = 4.0f;
    float boost_pad_cooldown_big = 10.0f;
    // Render-frame fraction of the pad pickup test. The game tests pads once per render frame, at the car position
    // interpolated between the last two ticks by the frame's fraction f (lerp(prev, cur, f)); f follows the frame
    // timing, so it is an input here. 0.5 (default) tests halfway through the tick; < 0 skips the test this tick
    // (a render frame that runs more ticks after this one). Above 120 fps, pass the last frame's f before the next tick.
    float boost_pad_frame_frac = 0.5f;
    // The round countdown's timers: false (default) adds each render frame's DeltaSeconds at its end, derived from
    // boost_pad_frame_frac as (ticks + f - f_prev) / 120; true adds frame_start_dt on the frame's first tick, before
    // its physics, as the game does (exact through frames whose ticks Bullet's 5 substep cap cut short).
    bool countdown_on_frame_start = false;
    // With countdown_on_frame_start: the DeltaSeconds of the render frame starting on this tick, or < 0 (none starts).
    float frame_start_dt = -1.0f;

    float car_air_control_pitch = 130.0f;
    float car_air_control_yaw = 95.0f;
    float car_air_control_roll = 400.0f;
    float car_air_damp_pitch = 30.0f;
    float car_air_damp_yaw = 20.0f;
    float car_air_damp_roll = 50.0f;

    uint8_t arena_collision = 1;

    uint8_t game_mode = 0;

    uint8_t ball_shape = 0;

    uint8_t ball_no_rot = 1;

    uint8_t cal_ball_world = 0;

    // A kickoff reset without the countdown (kickoff_countdown_ticks 0, reset_to_kickoff) puts the cars and the ball
    // where the game's countdown leaves them: settled on their wheels, the ball at rest. BatchSim records those
    // states from a countdown run for every car config, team and spawn spot in use, again whenever this config,
    // the car configs or the teams change. 0: the cars stay where they spawn, as in RocketSim.
    uint8_t kickoff_settled = 1;

    uint32_t ball_settle_epoch = 0;

    const detail::ArenaMeshTriangle* cal_arena_local_tris = nullptr;
    Vec3 ball_local_inertia_bt = Vec3(0.0f, 0.0f, 0.0f);

    float ball_contact_breaking_threshold_bt = 0.0f;
    Vec3 puck_points_bt[40] = {};
};

struct SimConfig {
    int32_t env_count = 1;
    int32_t max_cars_per_env = 6;
    float tick_dt = 1.0f / 120.0f;
    Mode mode = Mode::Soccar;
    uint8_t enable_boost_pads = 1;
    uint8_t enable_car_car_collisions = 1;
    uint8_t enable_demos = 1;
    uint8_t enable_scoring = 1;

    uint8_t no_ball_rot = 1;

    uint8_t native_sphere_normalization = 0;
    uint8_t mirror_official_respawn_rng = 0;

    uint64_t respawn_rng_seed = 0;

    const detail::ArenaMeshTriangle* arena_mesh_tris = nullptr;
    int32_t arena_mesh_tri_count = 0;
    const detail::ArenaMeshBvhQuantization* arena_mesh_bvh_quantization = nullptr;
    int32_t arena_mesh_bvh_quantization_count = 0;
    const detail::ArenaMeshBvhMeshInfo* arena_mesh_bvh_meshes = nullptr;
    int32_t arena_mesh_bvh_mesh_count = 0;
    const detail::ArenaMeshBvhNode* arena_mesh_bvh_nodes = nullptr;
    int32_t arena_mesh_bvh_node_count = 0;
    const detail::ArenaMeshBvhSubtreeInfo* arena_mesh_bvh_subtrees = nullptr;
    int32_t arena_mesh_bvh_subtree_count = 0;
    const detail::ArenaMeshLeafGridInfo* arena_mesh_leaf_grid_infos = nullptr;
    const int32_t* arena_mesh_leaf_cell_start = nullptr;
    const detail::ArenaMeshLeafEntry* arena_mesh_leaf_entries = nullptr;

    const uint16_t* arena_broadphase_cell_mask = nullptr;
    int32_t arena_broadphase_cell_count = 0;
    Vec3 arena_broadphase_min_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 arena_broadphase_max_bt = Vec3(0.0f, 0.0f, 0.0f);
    float arena_broadphase_cell_size_bt = 0.0f;
    int32_t arena_broadphase_cells_x = 0;
    int32_t arena_broadphase_cells_y = 0;
    int32_t arena_broadphase_cells_z = 0;
    const detail::BulletSweepHistory* arena_sweep_statics = nullptr;
    const int32_t* env_step_order = nullptr;
    int64_t env_car_total = 0;
    const float* env_frame_frac = nullptr;
    const float* env_frame_start_dt = nullptr;
    CarConfig default_car = {};
    MutatorConfig mutator = {};

    uint8_t use_custom_boost_pads = 0;
    int32_t boost_pad_count = 0;
    BoostPadConfig boost_pads[kMaxBoostPads] = {};
    const detail::KickoffSettledRow* kickoff_settled_rows = nullptr;
    const int64_t* kickoff_settled_entry = nullptr;
    const BallState* kickoff_settled_ball = nullptr;
    detail::WheelRuntimeState* kickoff_settled_wheels = nullptr;
    int32_t kickoff_settled_spawns = 0;
#if !defined(ROCKETGPU_USE_CUDA)
    const detail::HostStepTables* host_tables = nullptr;
#else
    const detail::ArenaMeshLeafGridInfo* arena_mesh_ray_grid_infos = nullptr;
    const int32_t* arena_mesh_ray_cell_start = nullptr;
    const detail::ArenaMeshLeafEntry* arena_mesh_ray_entries = nullptr;
#endif
};

struct DeviceView {
    CarState* car_states = nullptr;
    CarControls* controls = nullptr;
    CarConfig* car_configs = nullptr;
    Team* teams = nullptr;
    BallState* ball_states = nullptr;
    EnvState* env_states = nullptr;
    BoostPadState* boost_pad_states = nullptr;
    int32_t* env_car_count = nullptr;
};

class BatchSim {
public:
    explicit BatchSim(const SimConfig& config);
    ~BatchSim();

    BatchSim(BatchSim&& other) noexcept;
    BatchSim& operator=(BatchSim&& other) noexcept;

    BatchSim(const BatchSim&) = delete;
    BatchSim& operator=(const BatchSim&) = delete;

    const SimConfig& config() const noexcept;

    void set_mutator_config(const MutatorConfig& mutator);
    // Each env's render frame fraction for the next steps, in place of mutator.boost_pad_frame_frac (count ==
    // env_count); count 0 goes back to the mutator's for every env.
    void set_env_frame_frac(const float* host_values, size_t count, cudaStream_t stream = nullptr);
    // Each env's frame_start_dt for the next steps, in place of mutator.frame_start_dt (count == env_count; 0 clears).
    void set_env_frame_start_dt(const float* host_values, size_t count, cudaStream_t stream = nullptr);

    void set_respawn_rng_env_indices(const int32_t* source_env_indices, size_t count);

    void set_respawn_rng_seed(uint64_t seed);
    DeviceView device_view() const noexcept;

    void set_physics_event_stream(const PhysicsEventStream& stream);

    size_t car_capacity() const noexcept;
    size_t env_count() const noexcept;
    size_t boost_pad_capacity() const noexcept;

    // A lower count leaves mode 1/3 visit orders as they are: before the next step, upload an order that lists the
    // present slots first (stepping one that does not is undefined).
    void set_env_car_count(const int32_t* host_counts, size_t count, cudaStream_t stream = nullptr);
    void set_teams(const Team* host_teams, size_t count, cudaStream_t stream = nullptr);
    void set_car_configs(const CarConfig* host_configs, size_t count, cudaStream_t stream = nullptr);

    enum class StateUploadMode { PublicState, PreserveInternal };
    void upload_car_states(const CarState* host_states, size_t count, cudaStream_t stream = nullptr,
        StateUploadMode mode = StateUploadMode::PublicState);
    void upload_ball_states(const BallState* host_states, size_t count, cudaStream_t stream = nullptr,
        StateUploadMode mode = StateUploadMode::PublicState);
    void upload_controls(const CarControls* host_controls, size_t count, cudaStream_t stream = nullptr);
    void upload_env_states(const EnvState* host_states, size_t count, cudaStream_t stream = nullptr);
    void upload_boost_pad_states(const BoostPadState* host_states, size_t count, cudaStream_t stream = nullptr);

    void download_car_states(CarState* host_states, size_t count, cudaStream_t stream = nullptr) const;
    void download_wheel_runtime_states(detail::WheelRuntimeState* host_states, size_t count, cudaStream_t stream = nullptr) const;

    void upload_wheel_runtime_states(const detail::WheelRuntimeState* host_states, size_t count, cudaStream_t stream = nullptr);

    void copy_wheel_runtime_states_from(const BatchSim& source, cudaStream_t stream = nullptr);

    void copy_solver_scratch_from(const BatchSim& source, cudaStream_t stream = nullptr);

    void copy_state_from(const BatchSim& source, cudaStream_t stream = nullptr);
    void download_ball_states(BallState* host_states, size_t count, cudaStream_t stream = nullptr) const;
    void download_env_states(EnvState* host_states, size_t count, cudaStream_t stream = nullptr) const;
    void download_boost_pad_states(BoostPadState* host_states, size_t count, cudaStream_t stream = nullptr) const;

    void download_car_states_async(CarState* host_states, size_t count, cudaStream_t stream = nullptr) const;
    void download_ball_states_async(BallState* host_states, size_t count, cudaStream_t stream = nullptr) const;
    void download_env_states_async(EnvState* host_states, size_t count, cudaStream_t stream = nullptr) const;
    void download_boost_pad_states_async(BoostPadState* host_states, size_t count, cudaStream_t stream = nullptr) const;
    void download_solver_scratch(detail::WorldSolverScratch* host_states, size_t count, cudaStream_t stream = nullptr) const;

    void download_solver_scratch_range(detail::WorldSolverScratch* host_states, size_t first, size_t count, cudaStream_t stream = nullptr) const;

    void download_solver_scratch_rows(detail::WorldSolverScratchRowPoolHost& out, size_t first, size_t count, cudaStream_t stream = nullptr) const;
    void upload_solver_scratch_range_dev(const detail::WorldSolverScratch* host_states, size_t first, size_t count, cudaStream_t stream = nullptr);
    void upload_solver_retained_rows_dev(const detail::WorldSolverScratchRowPoolHost& in, size_t first, size_t count, cudaStream_t stream = nullptr);

    size_t respawn_rng_state_count() const noexcept;
    void download_respawn_rng_states(detail::RespawnRngState* host_states, size_t count, cudaStream_t stream = nullptr) const;

    void reset_to_kickoff(uint64_t seed = 0, cudaStream_t stream = nullptr);
    void apply_env_commands(const EnvCommand* host_commands, size_t count, cudaStream_t stream = nullptr);
    void apply_env_commands_device(const EnvCommand* device_commands, size_t count, cudaStream_t stream = nullptr);
    void step(int32_t ticks = 1, cudaStream_t stream = nullptr);

    void step_with_device_controls(const CarControls* device_controls_by_tick, int32_t ticks, int32_t control_stride, cudaStream_t stream = nullptr);

    void step_with_device_commands(
        const CarControls* device_controls_by_tick,
        int32_t control_stride,
        const EnvCommand* device_env_commands_by_tick,
        int32_t env_command_stride,
        int32_t ticks,
        cudaStream_t stream = nullptr);
    void synchronize(cudaStream_t stream = nullptr) const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

inline uint32_t RespawnRngSeedForEnv(uint64_t respawn_rng_seed, int env_idx) {
    const uint64_t mixed = (respawn_rng_seed + 0x9E3779B97F4A7C15ull) ^ (static_cast<uint64_t>(env_idx) * 0xD1B54A32D192ED03ull);
    return static_cast<uint32_t>(mixed ^ (mixed >> 32)) | 1u;
}

void CanonicalizeImportedCarState(CarState& state);
void CanonicalizeImportedBallState(BallState& state);

CarConfig OctaneConfig();

CarConfig DominusConfig();
CarConfig PlankConfig();
CarConfig BreakoutConfig();
CarConfig HybridConfig();
CarConfig MercConfig();
CarConfig PsyclopsConfig();
CarConfig BatmobileConfig();

size_t SolverScratchBytesPerEnv();
}
