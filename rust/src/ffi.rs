#![allow(dead_code)]

use std::os::raw::{c_char, c_int};

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RgControls {
    pub throttle: f32,
    pub steer: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32,
    pub jump: u8,
    pub boost: u8,
    pub handbrake: u8,
    pub pad: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RgCarState {
    pub last_extra_hit_tick: u64,
    pub pos: [f32; 3],
    pub rot: [f32; 9],
    pub vel: [f32; 3],
    pub ang_vel: [f32; 3],
    pub flip_rel_torque: [f32; 3],
    pub world_contact_normal: [f32; 3],
    pub wheel_normals: [f32; 12],
    pub controls: RgControls,
    pub prev_controls: RgControls,
    pub jump_ticks: u32,
    pub bump_cooldown_victim: u32,
    pub flip_time: f32,
    pub air_time: f32,
    pub air_time_since_jump: f32,
    pub boost: f32,
    pub handbrake_val: f32,
    pub supersonic_grace_timer: f32,
    pub time_since_boosted: f32,
    pub boosting_time: f32,
    pub auto_flip_timer: f32,
    pub auto_flip_torque_scale: f32,
    pub bump_cooldown_timer: f32,
    pub demo_respawn_timer: f32,
    pub has_jumped: u8,
    pub has_double_jumped: u8,
    pub has_flipped: u8,
    pub is_flipping: u8,
    pub is_jumping: u8,
    pub jump_hold_broken: u8,
    pub is_supersonic: u8,
    pub is_boosting: u8,
    pub is_auto_flipping: u8,
    pub is_demoed: u8,
    pub is_on_ground: u8,
    pub wheels: [u8; 4],
    pub has_world_contact: u8,
    pub has_last_extra_hit_tick: u8,
    pub has_prev_controls: u8,
    pub has_wheel_normals: u8,
    pub pad: [u8; 5],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RgBall {
    pub pos: [f32; 3],
    pub rot: [f32; 9],
    pub vel: [f32; 3],
    pub ang_vel: [f32; 3],
    pub deactivation_time: f32,
    pub sleeping: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RgCarConfig {
    pub hitbox_size: [f32; 3],
    pub hitbox_published: [f32; 3],
    pub hitbox_offset: [f32; 3],
    pub front_wheels_offset: [f32; 3],
    pub back_wheels_offset: [f32; 3],
    pub front_wheel_radius: f32,
    pub back_wheel_radius: f32,
    pub front_suspension_rest: f32,
    pub back_suspension_rest: f32,
    pub dodge_deadzone: f32,
    pub steer_angle_curve_deg: [f32; 6],
    pub powerslide_steer_angle_curve_deg: [f32; 2],
    pub spawn_z: f32,
    pub body: u32,
    pub three_wheels: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RgMutators {
    pub gravity_z: f32,
    pub car_mass: f32,
    pub ball_mass: f32,
    pub ball_max_speed: f32,
    pub ball_drag: f32,
    pub jump_accel: f32,
    pub jump_immediate_speed: f32,
    pub boost_accel_ground: f32,
    pub boost_accel_air: f32,
    pub boost_used_per_second: f32,
    pub demo_respawn_time: f32,
    pub bump_cooldown_time: f32,
    pub boost_max: f32,
    pub boost_spawn_amount: f32,
    pub boost_pad_cooldown_big: f32,
    pub boost_pad_cooldown_small: f32,
    pub ball_hit_extra_force_scale: f32,
    pub bump_force_scale: f32,
    pub ball_radius: f32,
    pub recharge_boost_per_second: f32,
    pub recharge_boost_delay: f32,
    pub goal_base_threshold_y: f32,
    pub unlimited_flips: u32,
    pub unlimited_double_jumps: u32,
    pub recharge_boost_enabled: u32,
    pub enable_team_demos: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RgPad {
    pub pos: [f32; 3],
    pub is_big: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RgEvent {
    pub kind: u32,
    pub car_id: u32,
    pub other_car_id: u32,
    pub pad_index: u32,
    pub is_demo: u32,
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

const _: () = assert!(size_of::<RgControls>() == 24);
const _: () = assert!(size_of::<RgCarState>() == 280);
const _: () = assert!(size_of::<RgBall>() == 80);
const _: () = assert!(size_of::<RgCarConfig>() == 124);
const _: () = assert!(size_of::<RgMutators>() == 104);
const _: () = assert!(size_of::<RgPad>() == 16);
const _: () = assert!(size_of::<RgEvent>() == 44);

#[repr(C)]
pub struct RgArena {
    _private: [u8; 0],
}

extern "C" {
    pub fn rg_last_error() -> *const c_char;
    pub fn rg_arena_new(mode: u32, no_ball_rot: u32, custom_pads: *const RgPad, custom_pad_count: i32, out: *mut *mut RgArena) -> c_int;
    pub fn rg_arena_clone(arena: *const RgArena, out: *mut *mut RgArena) -> c_int;
    pub fn rg_arena_free(arena: *mut RgArena);
    pub fn rg_add_car(arena: *mut RgArena, team: u32, config: *const RgCarConfig, id_out: *mut u32) -> c_int;
    pub fn rg_remove_car(arena: *mut RgArena, id: u32) -> c_int;
    pub fn rg_step(arena: *mut RgArena, event_count_out: *mut u32) -> c_int;
    pub fn rg_events(arena: *const RgArena, out: *mut RgEvent, capacity: u32) -> c_int;
    pub fn rg_tick_count(arena: *const RgArena) -> u64;
    pub fn rg_is_ball_scored(arena: *const RgArena, out: *mut u32) -> c_int;
    pub fn rg_reset_kickoff(arena: *mut RgArena, seed: i32, match_next: u32) -> c_int;
    pub fn rg_get_car(arena: *const RgArena, id: u32, out: *mut RgCarState) -> c_int;
    pub fn rg_set_car(arena: *mut RgArena, id: u32, state: *const RgCarState) -> c_int;
    pub fn rg_set_controls(arena: *mut RgArena, id: u32, controls: *const RgControls) -> c_int;
    pub fn rg_get_ball(arena: *const RgArena, out: *mut RgBall) -> c_int;
    pub fn rg_set_ball(arena: *mut RgArena, ball: *const RgBall, activation: u32) -> c_int;
    pub fn rg_pad_count(arena: *const RgArena) -> u32;
    pub fn rg_pad_config(arena: *const RgArena, index: u32, out: *mut RgPad) -> c_int;
    pub fn rg_get_pad_cooldowns(arena: *const RgArena, out: *mut f32, count: u32) -> c_int;
    pub fn rg_set_pad_cooldown(arena: *mut RgArena, index: u32, cooldown: f32) -> c_int;
    pub fn rg_get_mutators(arena: *const RgArena, out: *mut RgMutators) -> c_int;
    pub fn rg_set_mutators(arena: *mut RgArena, mutators: *const RgMutators) -> c_int;
}
