#pragma once

#include <cstdint>

extern "C" {
struct RgControls {
    float throttle, steer, pitch, yaw, roll;
    uint8_t jump, boost, handbrake, pad;
};

struct RgCarState {
    uint64_t last_extra_hit_tick;
    float pos[3], rot[9], vel[3], ang_vel[3], flip_rel_torque[3], world_contact_normal[3], wheel_normals[12];
    RgControls controls, prev_controls;
    uint32_t jump_ticks, bump_cooldown_victim;
    float flip_time, air_time, air_time_since_jump, boost, handbrake_val, supersonic_grace_timer, time_since_boosted,
        boosting_time, auto_flip_timer, auto_flip_torque_scale, bump_cooldown_timer, demo_respawn_timer;
    uint8_t has_jumped, has_double_jumped, has_flipped, is_flipping, is_jumping, jump_hold_broken, is_supersonic,
        is_boosting, is_auto_flipping, is_demoed, is_on_ground, wheels[4], has_world_contact,
        has_last_extra_hit_tick, has_prev_controls, has_wheel_normals, pad[5];
};

struct RgBall {
    float pos[3], rot[9], vel[3], ang_vel[3];
    float deactivation_time;
    uint32_t sleeping;
};

struct RgCarConfig {
    float hitbox_size[3], hitbox_published[3], hitbox_offset[3], front_wheels_offset[3], back_wheels_offset[3];
    float front_wheel_radius, back_wheel_radius, front_suspension_rest, back_suspension_rest, dodge_deadzone;
    float steer_angle_curve_deg[6], powerslide_steer_angle_curve_deg[2];
    float spawn_z;
    uint32_t body, three_wheels;
};

struct RgMutators {
    float gravity_z, car_mass, ball_mass, ball_max_speed, ball_drag, jump_accel, jump_immediate_speed,
        boost_accel_ground, boost_accel_air, boost_used_per_second, demo_respawn_time, bump_cooldown_time, boost_max,
        boost_spawn_amount, boost_pad_cooldown_big, boost_pad_cooldown_small, ball_hit_extra_force_scale,
        bump_force_scale, ball_radius, recharge_boost_per_second, recharge_boost_delay, goal_base_threshold_y;
    uint32_t unlimited_flips, unlimited_double_jumps, recharge_boost_enabled, enable_team_demos;
};

struct RgPad {
    float pos[3];
    uint32_t is_big;
};

struct RgEvent {
    uint32_t type, car_id, other_car_id, pad_index, is_demo;
    float position[3], normal[3];
};

struct RgArena;

const char* rg_last_error();
int rg_arena_new(uint32_t mode, uint32_t no_ball_rot, const RgPad* custom_pads, int32_t custom_pad_count, RgArena** out);
int rg_arena_clone(const RgArena* arena, RgArena** out);
void rg_arena_free(RgArena* arena);
int rg_add_car(RgArena* arena, uint32_t team, const RgCarConfig* config, uint32_t* id_out);
int rg_remove_car(RgArena* arena, uint32_t id);
int rg_step(RgArena* arena, uint32_t* event_count_out);
int rg_events(const RgArena* arena, RgEvent* out, uint32_t capacity);
uint64_t rg_tick_count(const RgArena* arena);
int rg_is_ball_scored(const RgArena* arena, uint32_t* out);
int rg_reset_kickoff(RgArena* arena, int32_t seed, uint32_t match_next);
int rg_get_car(const RgArena* arena, uint32_t id, RgCarState* out);
int rg_set_car(RgArena* arena, uint32_t id, const RgCarState* state);
int rg_set_controls(RgArena* arena, uint32_t id, const RgControls* controls);
int rg_get_ball(const RgArena* arena, RgBall* out);
int rg_set_ball(RgArena* arena, const RgBall* ball, uint32_t activation);
uint32_t rg_pad_count(const RgArena* arena);
int rg_pad_config(const RgArena* arena, uint32_t index, RgPad* out);
int rg_get_pad_cooldowns(const RgArena* arena, float* out, uint32_t count);
int rg_set_pad_cooldown(RgArena* arena, uint32_t index, float cooldown);
int rg_get_mutators(const RgArena* arena, RgMutators* out);
int rg_set_mutators(RgArena* arena, const RgMutators* mutators);
}
