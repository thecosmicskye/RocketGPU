//! RocketGPU's arena with RocketSim v3's Rust API (the `rocketsim` crate, 0.3.6): the same type, field and method
//! names, so code written for v3 ports by changing the crate name.
//!
//! ```no_run
//! use rocketgpu::{Arena, CarBodyConfig, CarControls, GameMode, Team};
//!
//! rocketgpu::init("/path/to/collision_meshes", true).unwrap();
//! let mut arena = Arena::new(GameMode::Soccar);
//! let car = arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
//! arena.reset_to_random_kickoff(Some(7));
//! arena.set_car_controls(car, CarControls { throttle: 1.0, boost: true, ..Default::default() });
//! for _ in 0..120 {
//!     arena.step_tick();
//! }
//! println!("{:?}", arena.get_car_state(car).phys.pos);
//! ```
//!
//! [`CarState`] holds RocketSim v3's values (RocketGPU's `Car::GetStateV3` / `SetStateV3`): velocities before the
//! speed caps, v3's jump and flip flags, its supersonic grace timer. The physics is RocketGPU's.
//!
//! Not provided: RocketSim v3's raycasts, dropshot tiles, heatseeker and dropshot ball info, `CarTuning`, the
//! `CarCore` builders, `respawn_car` / `demolish_car`, `step_batch`, and the hit point of a wheel or an event (events
//! carry the cars, the pad and the contact normal).

use std::cell::OnceCell;
use std::ffi::CStr;
use std::io;
use std::ops::{Deref, DerefMut};
use std::os::raw::c_int;
use std::path::Path;
use std::ptr::NonNull;

pub use glam;
pub use glam::{Mat3A, Vec3A};

mod ffi;
include!(env!("ROCKETGPU_RS_PRESETS"));

pub mod consts {
    pub const TICK_RATE: f32 = 120.0;
    pub const TICK_TIME: f32 = 1.0 / TICK_RATE;
}

const MESH_DIR_VAR: &str = "ROCKETGPU_COLLISION_MESH_DIR";

/// Points RocketGPU at RocketSim's `collision_meshes` folder: sets `ROCKETGPU_COLLISION_MESH_DIR`, which arenas
/// read when they are created. Call it before starting other threads. With the variable already set in the
/// environment, no call is needed.
pub fn init<P: AsRef<Path>>(collision_meshes_folder: P, _silent: bool) -> io::Result<()> {
    let dir = collision_meshes_folder.as_ref();
    if !dir.is_dir() {
        return Err(io::Error::new(io::ErrorKind::NotFound, format!("no folder {}", dir.display())));
    }
    std::env::set_var(MESH_DIR_VAR, dir);
    Ok(())
}

/// [`init`] with `./collision_meshes`.
pub fn init_from_default(silent: bool) -> io::Result<()> {
    init("collision_meshes", silent)
}

pub fn is_initialized() -> bool {
    std::env::var_os(MESH_DIR_VAR).is_some()
}

/// HOOPS and DROPSHOT are not implemented: creating an arena with them panics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum GameMode {
    #[default]
    Soccar,
    Hoops,
    Heatseeker,
    Snowday,
    Dropshot,
    TheVoid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Team {
    Blue,
    Orange,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CarControls {
    pub throttle: f32,
    pub steer: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32,
    pub jump: bool,
    pub boost: bool,
    pub handbrake: bool,
}

impl CarControls {
    pub const DEFAULT: Self =
        Self { throttle: 0.0, steer: 0.0, pitch: 0.0, yaw: 0.0, roll: 0.0, jump: false, boost: false, handbrake: false };

    fn to_ffi(self) -> ffi::RgControls {
        ffi::RgControls {
            throttle: self.throttle,
            steer: self.steer,
            pitch: self.pitch,
            yaw: self.yaw,
            roll: self.roll,
            jump: self.jump as u8,
            boost: self.boost as u8,
            handbrake: self.handbrake as u8,
            pad: 0,
        }
    }

    fn from_ffi(c: &ffi::RgControls) -> Self {
        Self {
            throttle: c.throttle,
            steer: c.steer,
            pitch: c.pitch,
            yaw: c.yaw,
            roll: c.roll,
            jump: c.jump != 0,
            boost: c.boost != 0,
            handbrake: c.handbrake != 0,
        }
    }
}

impl Default for CarControls {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Position (uu), rotation (columns: forward, right, up), velocity (uu/s) and angular velocity (rad/s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysState {
    pub pos: Vec3A,
    pub rot_mat: Mat3A,
    pub vel: Vec3A,
    pub ang_vel: Vec3A,
}

fn rot_to_ffi(m: &Mat3A) -> [f32; 9] {
    let (f, r, u) = (m.x_axis, m.y_axis, m.z_axis);
    [f.x, f.y, f.z, r.x, r.y, r.z, u.x, u.y, u.z]
}

fn rot_from_ffi(r: &[f32; 9]) -> Mat3A {
    Mat3A::from_cols(Vec3A::new(r[0], r[1], r[2]), Vec3A::new(r[3], r[4], r[5]), Vec3A::new(r[6], r[7], r[8]))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WheelPairConfig {
    pub wheel_radius: f32,
    pub suspension_rest_length: f32,
    pub connection_point_offset: Vec3A,
}

/// A car body. The presets are RocketGPU's (`OctaneConfig()` ... in `rocketgpu.h`). The last five fields are
/// RocketGPU's own: the steering curves in degrees, the game's hitbox size, its body id and the spawn height.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CarBodyConfig {
    pub hitbox_size: Vec3A,
    pub hitbox_pos_offset: Vec3A,
    pub front_wheels: WheelPairConfig,
    pub back_wheels: WheelPairConfig,
    pub three_wheels: bool,
    pub dodge_deadzone: f32,
    pub steer_angle_curve_deg: [f32; 6],
    pub powerslide_steer_angle_curve_deg: [f32; 2],
    pub hitbox_published: Vec3A,
    pub body: u16,
    pub spawn_z: f32,
}

impl CarBodyConfig {
    pub const OCTANE: Self = Self::from_words(&PRESET_WORDS[0]);
    pub const DOMINUS: Self = Self::from_words(&PRESET_WORDS[1]);
    pub const PLANK: Self = Self::from_words(&PRESET_WORDS[2]);
    pub const BREAKOUT: Self = Self::from_words(&PRESET_WORDS[3]);
    pub const HYBRID: Self = Self::from_words(&PRESET_WORDS[4]);
    pub const MERC: Self = Self::from_words(&PRESET_WORDS[5]);
    pub const PSYCLOPS: Self = Self::from_words(&PRESET_WORDS[6]);
    pub const BATMOBILE: Self = Self::from_words(&PRESET_WORDS[7]);

    const fn from_words(w: &[u32; 31]) -> Self {
        const fn f(w: &[u32; 31], i: usize) -> f32 {
            f32::from_bits(w[i])
        }
        const fn v(w: &[u32; 31], i: usize) -> Vec3A {
            Vec3A::new(f(w, i), f(w, i + 1), f(w, i + 2))
        }
        Self {
            hitbox_size: v(w, 0),
            hitbox_published: v(w, 3),
            hitbox_pos_offset: v(w, 6),
            front_wheels: WheelPairConfig {
                wheel_radius: f(w, 15),
                suspension_rest_length: f(w, 17),
                connection_point_offset: v(w, 9),
            },
            back_wheels: WheelPairConfig {
                wheel_radius: f(w, 16),
                suspension_rest_length: f(w, 18),
                connection_point_offset: v(w, 12),
            },
            dodge_deadzone: f(w, 19),
            steer_angle_curve_deg: [f(w, 20), f(w, 21), f(w, 22), f(w, 23), f(w, 24), f(w, 25)],
            powerslide_steer_angle_curve_deg: [f(w, 26), f(w, 27)],
            spawn_z: f(w, 28),
            body: w[29] as u16,
            three_wheels: w[30] != 0,
        }
    }

    fn to_ffi(self) -> ffi::RgCarConfig {
        ffi::RgCarConfig {
            hitbox_size: self.hitbox_size.to_array(),
            hitbox_published: self.hitbox_published.to_array(),
            hitbox_offset: self.hitbox_pos_offset.to_array(),
            front_wheels_offset: self.front_wheels.connection_point_offset.to_array(),
            back_wheels_offset: self.back_wheels.connection_point_offset.to_array(),
            front_wheel_radius: self.front_wheels.wheel_radius,
            back_wheel_radius: self.back_wheels.wheel_radius,
            front_suspension_rest: self.front_wheels.suspension_rest_length,
            back_suspension_rest: self.back_wheels.suspension_rest_length,
            dodge_deadzone: self.dodge_deadzone,
            steer_angle_curve_deg: self.steer_angle_curve_deg,
            powerslide_steer_angle_curve_deg: self.powerslide_steer_angle_curve_deg,
            spawn_z: self.spawn_z,
            body: self.body as u32,
            three_wheels: self.three_wheels as u32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CarInfo {
    pub idx: usize,
    pub team: Team,
    pub config: CarBodyConfig,
}

/// A wheel in contact. RocketGPU reports which wheels touch, not where.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaycastHitInfo {
    _private: (),
}

/// The part of a car's state a caller usually sets ([`Arena::set_car_state`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CarCore {
    pub phys: PhysState,
    pub controls: CarControls,
    pub has_jumped: bool,
    pub has_double_jumped: bool,
    pub has_flipped: bool,
    pub flip_rel_torque: Vec3A,
    pub jump_ticks: u32,
    pub flip_time: f32,
    pub is_flipping: bool,
    pub is_jumping: bool,
    pub jump_hold_broken: bool,
    pub air_time_since_jump: f32,
    pub boost: f32,
    pub is_auto_flipping: bool,
    pub auto_flip_timer: f32,
    pub auto_flip_torque_scale: f32,
    pub bump_cooldown_timer: f32,
    pub bump_cooldown_victim: Option<usize>,
    pub last_extra_hit_tick: Option<u64>,
    pub is_demoed: bool,
    pub demo_respawn_timer: f32,
}

impl CarCore {
    pub const DEFAULT: Self = Self {
        phys: PhysState { pos: Vec3A::new(0.0, 0.0, 17.0), rot_mat: Mat3A::IDENTITY, vel: Vec3A::ZERO, ang_vel: Vec3A::ZERO },
        controls: CarControls::DEFAULT,
        has_jumped: false,
        has_double_jumped: false,
        has_flipped: false,
        flip_rel_torque: Vec3A::ZERO,
        jump_ticks: 0,
        flip_time: 0.0,
        is_flipping: false,
        is_jumping: false,
        jump_hold_broken: false,
        air_time_since_jump: 0.0,
        boost: 100.0 / 3.0,
        is_auto_flipping: false,
        auto_flip_timer: 0.0,
        auto_flip_torque_scale: 0.0,
        bump_cooldown_timer: 0.0,
        bump_cooldown_victim: None,
        last_extra_hit_tick: None,
        is_demoed: false,
        demo_respawn_timer: 0.0,
    };
}

impl Default for CarCore {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CarCache {
    is_on_ground: bool,
    wheels_with_contact: [Option<RaycastHitInfo>; 4],
    world_contact_normal: Option<Vec3A>,
    prev_controls: CarControls,
    has_prev_controls: bool,
    has_last_extra_hit_tick: bool,
    wheel_normals: Option<[Vec3A; 4]>,
}

/// A car's state in RocketSim v3's conventions, field for field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CarState {
    pub phys: PhysState,
    /// The controls for the next step.
    pub controls: CarControls,
    pub has_jumped: bool,
    pub has_double_jumped: bool,
    pub has_flipped: bool,
    pub flip_rel_torque: Vec3A,
    pub jump_ticks: u32,
    pub flip_time: f32,
    pub is_flipping: bool,
    pub is_jumping: bool,
    pub jump_hold_broken: bool,
    pub air_time: f32,
    pub air_time_since_jump: f32,
    pub boost: f32,
    pub handbrake_val: f32,
    pub is_supersonic: bool,
    pub supersonic_grace_timer: f32,
    pub time_since_boosted: f32,
    pub boosting_time: f32,
    pub is_boosting: bool,
    pub is_auto_flipping: bool,
    pub auto_flip_timer: f32,
    pub auto_flip_torque_scale: f32,
    pub bump_cooldown_timer: f32,
    pub bump_cooldown_victim: Option<usize>,
    pub last_extra_hit_tick: Option<u64>,
    pub is_demoed: bool,
    pub demo_respawn_timer: f32,
    cache: CarCache,
}

impl CarState {
    pub const DEFAULT: Self = Self::with_cache(
        CarCore::DEFAULT,
        false,
        CarCache {
            is_on_ground: true,
            wheels_with_contact: [None; 4],
            world_contact_normal: None,
            prev_controls: CarControls::DEFAULT,
            has_prev_controls: true,
            has_last_extra_hit_tick: false,
            wheel_normals: None,
        },
    );

    const fn with_cache(core: CarCore, is_supersonic: bool, cache: CarCache) -> Self {
        Self {
            phys: core.phys,
            controls: core.controls,
            has_jumped: core.has_jumped,
            has_double_jumped: core.has_double_jumped,
            has_flipped: core.has_flipped,
            flip_rel_torque: core.flip_rel_torque,
            jump_ticks: core.jump_ticks,
            flip_time: core.flip_time,
            is_flipping: core.is_flipping,
            is_jumping: core.is_jumping,
            jump_hold_broken: core.jump_hold_broken,
            air_time: 0.0,
            air_time_since_jump: core.air_time_since_jump,
            boost: core.boost,
            handbrake_val: 0.0,
            is_supersonic,
            supersonic_grace_timer: 0.0,
            time_since_boosted: 0.0,
            boosting_time: 0.0,
            is_boosting: false,
            is_auto_flipping: core.is_auto_flipping,
            auto_flip_timer: core.auto_flip_timer,
            auto_flip_torque_scale: core.auto_flip_torque_scale,
            bump_cooldown_timer: core.bump_cooldown_timer,
            bump_cooldown_victim: core.bump_cooldown_victim,
            last_extra_hit_tick: core.last_extra_hit_tick,
            is_demoed: core.is_demoed,
            demo_respawn_timer: core.demo_respawn_timer,
            cache,
        }
    }

    pub const fn core(&self) -> CarCore {
        CarCore {
            phys: self.phys,
            controls: self.controls,
            has_jumped: self.has_jumped,
            has_double_jumped: self.has_double_jumped,
            has_flipped: self.has_flipped,
            flip_rel_torque: self.flip_rel_torque,
            jump_ticks: self.jump_ticks,
            flip_time: self.flip_time,
            is_flipping: self.is_flipping,
            is_jumping: self.is_jumping,
            jump_hold_broken: self.jump_hold_broken,
            air_time_since_jump: self.air_time_since_jump,
            boost: self.boost,
            is_auto_flipping: self.is_auto_flipping,
            auto_flip_timer: self.auto_flip_timer,
            auto_flip_torque_scale: self.auto_flip_torque_scale,
            bump_cooldown_timer: self.bump_cooldown_timer,
            bump_cooldown_victim: self.bump_cooldown_victim,
            last_extra_hit_tick: self.last_extra_hit_tick,
            is_demoed: self.is_demoed,
            demo_respawn_timer: self.demo_respawn_timer,
        }
    }

    pub const fn has_flip_or_jump(&self) -> bool {
        self.cache.is_on_ground
            || (!self.has_flipped && !self.has_double_jumped && self.air_time_since_jump + consts::TICK_TIME < 1.25)
    }

    pub const fn has_flip_reset(&self) -> bool {
        !self.cache.is_on_ground && self.has_flip_or_jump() && !self.has_jumped
    }

    pub const fn got_flip_reset(&self) -> bool {
        !self.cache.is_on_ground && !self.has_jumped
    }

    pub fn num_wheels_in_contact(&self) -> usize {
        self.cache.wheels_with_contact.iter().filter(|w| w.is_some()).count()
    }

    pub const fn jump_time(&self) -> f32 {
        self.jump_ticks as f32 * consts::TICK_TIME
    }

    pub const fn is_on_ground(&self) -> bool {
        self.cache.is_on_ground
    }

    pub const fn wheels_with_contact(&self) -> &[Option<RaycastHitInfo>; 4] {
        &self.cache.wheels_with_contact
    }

    pub const fn world_contact_normal(&self) -> Option<Vec3A> {
        self.cache.world_contact_normal
    }

    pub const fn prev_controls(&self) -> &CarControls {
        &self.cache.prev_controls
    }

    pub fn set_contact_flags(&mut self, is_on_ground: bool, wheel_contacts: [bool; 4]) {
        self.cache.is_on_ground = is_on_ground;
        self.cache.wheels_with_contact = wheel_contacts.map(|c| c.then_some(RaycastHitInfo { _private: () }));
        self.cache.wheel_normals = None;
    }

    pub const fn set_prev_controls(&mut self, controls: CarControls) {
        self.cache.prev_controls = controls;
        self.cache.has_prev_controls = true;
    }
}

impl Default for CarState {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl From<CarCore> for CarState {
    fn from(mut core: CarCore) -> Self {
        core.boost = core.boost.clamp(0.0, 100.0);
        core.has_jumped |= core.is_jumping;
        core.has_flipped |= core.is_flipping;
        let controls = core.controls;
        Self::with_cache(
            core,
            core.phys.vel.length_squared() >= 2200.0 * 2200.0,
            CarCache {
                is_on_ground: false,
                wheels_with_contact: [None; 4],
                world_contact_normal: None,
                prev_controls: controls,
                has_prev_controls: false,
                has_last_extra_hit_tick: false,
                wheel_normals: None,
            },
        )
    }
}

impl Deref for CarState {
    type Target = PhysState;
    fn deref(&self) -> &PhysState {
        &self.phys
    }
}

impl DerefMut for CarState {
    fn deref_mut(&mut self) -> &mut PhysState {
        &mut self.phys
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ActivationState {
    #[default]
    Active,
    Sleeping,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BallState {
    pub phys: PhysState,
    pub activation_state: ActivationState,
    pub deactivation_time: f32,
}

impl BallState {
    pub const DEFAULT: Self = Self {
        phys: PhysState { pos: Vec3A::new(0.0, 0.0, 93.15), rot_mat: Mat3A::IDENTITY, vel: Vec3A::ZERO, ang_vel: Vec3A::ZERO },
        activation_state: ActivationState::Active,
        deactivation_time: 0.0,
    };

    pub const fn wake(&mut self) {
        self.activation_state = ActivationState::Active;
        self.deactivation_time = 0.0;
    }
}

impl Default for BallState {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Deref for BallState {
    type Target = PhysState;
    fn deref(&self) -> &PhysState {
        &self.phys
    }
}

impl DerefMut for BallState {
    fn deref_mut(&mut self) -> &mut PhysState {
        &mut self.phys
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoostPadConfig {
    pub pos: Vec3A,
    pub is_big: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct BoostPadState {
    /// Seconds until the pad is back; 0 while it is active.
    pub cooldown: f32,
}

impl BoostPadState {
    pub const DEFAULT: Self = Self { cooldown: 0.0 };

    pub const fn is_active(&self) -> bool {
        self.cooldown <= 0.0
    }
}

/// The mutators RocketGPU shares with RocketSim v3. `gravity` must point along z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MutatorConfig {
    pub gravity: Vec3A,
    pub car_mass: f32,
    pub ball_mass: f32,
    pub ball_max_speed: f32,
    pub ball_drag: f32,
    pub jump_accel: f32,
    pub jump_immediate_force: f32,
    pub boost_accel_ground: f32,
    pub boost_accel_air: f32,
    pub boost_used_per_second: f32,
    pub respawn_delay: f32,
    pub bump_cooldown_time: f32,
    pub car_max_boost_amount: f32,
    pub car_spawn_boost_amount: f32,
    pub boost_pad_cooldown_big: f32,
    pub boost_pad_cooldown_small: f32,
    pub ball_hit_extra_force_scale: f32,
    pub bump_force_scale: f32,
    pub ball_radius: f32,
    pub unlimited_flips: bool,
    pub unlimited_double_jumps: bool,
    pub recharge_boost_enabled: bool,
    pub recharge_boost_per_second: f32,
    pub recharge_boost_delay: f32,
    pub enable_team_demos: bool,
    pub goal_base_threshold_y: f32,
}

impl MutatorConfig {
    fn from_ffi(m: &ffi::RgMutators) -> Self {
        Self {
            gravity: Vec3A::new(0.0, 0.0, m.gravity_z),
            car_mass: m.car_mass,
            ball_mass: m.ball_mass,
            ball_max_speed: m.ball_max_speed,
            ball_drag: m.ball_drag,
            jump_accel: m.jump_accel,
            jump_immediate_force: m.jump_immediate_speed,
            boost_accel_ground: m.boost_accel_ground,
            boost_accel_air: m.boost_accel_air,
            boost_used_per_second: m.boost_used_per_second,
            respawn_delay: m.demo_respawn_time,
            bump_cooldown_time: m.bump_cooldown_time,
            car_max_boost_amount: m.boost_max,
            car_spawn_boost_amount: m.boost_spawn_amount,
            boost_pad_cooldown_big: m.boost_pad_cooldown_big,
            boost_pad_cooldown_small: m.boost_pad_cooldown_small,
            ball_hit_extra_force_scale: m.ball_hit_extra_force_scale,
            bump_force_scale: m.bump_force_scale,
            ball_radius: m.ball_radius,
            unlimited_flips: m.unlimited_flips != 0,
            unlimited_double_jumps: m.unlimited_double_jumps != 0,
            recharge_boost_enabled: m.recharge_boost_enabled != 0,
            recharge_boost_per_second: m.recharge_boost_per_second,
            recharge_boost_delay: m.recharge_boost_delay,
            enable_team_demos: m.enable_team_demos != 0,
            goal_base_threshold_y: m.goal_base_threshold_y,
        }
    }

    fn to_ffi(self) -> ffi::RgMutators {
        assert!(self.gravity.x == 0.0 && self.gravity.y == 0.0, "rocketgpu: gravity must point along z");
        ffi::RgMutators {
            gravity_z: self.gravity.z,
            car_mass: self.car_mass,
            ball_mass: self.ball_mass,
            ball_max_speed: self.ball_max_speed,
            ball_drag: self.ball_drag,
            jump_accel: self.jump_accel,
            jump_immediate_speed: self.jump_immediate_force,
            boost_accel_ground: self.boost_accel_ground,
            boost_accel_air: self.boost_accel_air,
            boost_used_per_second: self.boost_used_per_second,
            demo_respawn_time: self.respawn_delay,
            bump_cooldown_time: self.bump_cooldown_time,
            boost_max: self.car_max_boost_amount,
            boost_spawn_amount: self.car_spawn_boost_amount,
            boost_pad_cooldown_big: self.boost_pad_cooldown_big,
            boost_pad_cooldown_small: self.boost_pad_cooldown_small,
            ball_hit_extra_force_scale: self.ball_hit_extra_force_scale,
            bump_force_scale: self.bump_force_scale,
            ball_radius: self.ball_radius,
            recharge_boost_per_second: self.recharge_boost_per_second,
            recharge_boost_delay: self.recharge_boost_delay,
            goal_base_threshold_y: self.goal_base_threshold_y,
            unlimited_flips: self.unlimited_flips as u32,
            unlimited_double_jumps: self.unlimited_double_jumps as u32,
            recharge_boost_enabled: self.recharge_boost_enabled as u32,
            enable_team_demos: self.enable_team_demos as u32,
        }
    }
}

/// `mutators`: `None` takes the game mode's own; [`Arena::get_config`] returns the ones in use.
#[derive(Clone, Debug, PartialEq)]
pub struct ArenaConfig {
    pub game_mode: GameMode,
    pub mutators: Option<MutatorConfig>,
    pub no_ball_rot: bool,
    pub custom_boost_pads: Option<Vec<BoostPadConfig>>,
}

impl ArenaConfig {
    pub const DEFAULT: Self = Self { game_mode: GameMode::Soccar, mutators: None, no_ball_rot: false, custom_boost_pads: None };

    pub fn new(game_mode: GameMode) -> Self {
        Self { game_mode, ..Self::DEFAULT }
    }

    pub fn with_mutators(mut self, mutators: MutatorConfig) -> Self {
        self.mutators = Some(mutators);
        self
    }

    pub fn with_no_ball_rot(mut self, no_ball_rot: bool) -> Self {
        self.no_ball_rot = no_ball_rot;
        self
    }

    pub fn with_custom_boost_pads(mut self, custom_boost_pads: Vec<BoostPadConfig>) -> Self {
        self.custom_boost_pads = Some(custom_boost_pads);
        self
    }
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BallHitWorldEvent {
    pub contact_normal: Vec3A,
}

/// A car starts touching the ball.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CarHitBallEvent {
    pub car_idx: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CarHitCarEvent {
    pub bumper_car_idx: usize,
    pub victim_car_idx: usize,
    pub is_demo: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CarHitWorldEvent {
    pub car_idx: usize,
    pub contact_normal: Vec3A,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CarPickupBoostEvent {
    pub car_idx: usize,
    pub boost_pad_idx: usize,
}

/// RocketGPU's physics events of one tick, under RocketSim v3's names.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArenaEvent {
    BallHitWorld(BallHitWorldEvent),
    CarHitBall(CarHitBallEvent),
    CarHitCar(CarHitCarEvent),
    CarHitWorld(CarHitWorldEvent),
    CarPickupBoost(CarPickupBoostEvent),
}

struct CarEntry {
    id: u32,
    info: CarInfo,
    controls: CarControls,
    state: OnceCell<CarState>,
}

fn check(rc: c_int) {
    if rc != 0 {
        let msg = unsafe { CStr::from_ptr(ffi::rg_last_error()) }.to_string_lossy().into_owned();
        panic!("rocketgpu: {msg}");
    }
}

/// One game: cars, ball and boost pads, stepped at 120 Hz. Cars are addressed by index, in the order they were
/// added; removing one moves the later ones down.
pub struct Arena {
    raw: NonNull<ffi::RgArena>,
    config: ArenaConfig,
    cars: Vec<CarEntry>,
    ball: OnceCell<BallState>,
    pad_configs: Vec<BoostPadConfig>,
    pad_cooldowns: OnceCell<Vec<f32>>,
    events: Vec<ArenaEvent>,
}

unsafe impl Send for Arena {}

impl Drop for Arena {
    fn drop(&mut self) {
        unsafe { ffi::rg_arena_free(self.raw.as_ptr()) }
    }
}

impl Clone for Arena {
    fn clone(&self) -> Self {
        let mut raw = std::ptr::null_mut();
        check(unsafe { ffi::rg_arena_clone(self.raw.as_ptr(), &mut raw) });
        let mut arena = Self {
            raw: NonNull::new(raw).expect("rocketgpu: null arena"),
            config: self.config.clone(),
            cars: self
                .cars
                .iter()
                .map(|c| CarEntry { id: c.id, info: c.info, controls: c.controls, state: OnceCell::new() })
                .collect(),
            ball: OnceCell::new(),
            pad_configs: self.pad_configs.clone(),
            pad_cooldowns: OnceCell::new(),
            events: self.events.clone(),
        };
        arena.config.mutators = Some(arena.pull_mutators());
        arena
    }
}

impl Arena {
    pub fn new(game_mode: GameMode) -> Self {
        Self::new_with_config(ArenaConfig::new(game_mode))
    }

    pub fn new_with_config(mut config: ArenaConfig) -> Self {
        let pads: Option<Vec<ffi::RgPad>> = config
            .custom_boost_pads
            .as_ref()
            .map(|v| v.iter().map(|p| ffi::RgPad { pos: p.pos.to_array(), is_big: p.is_big as u32 }).collect());
        let (ptr, count) = pads.as_ref().map_or((std::ptr::null(), -1), |v| (v.as_ptr(), v.len() as i32));
        let mut raw = std::ptr::null_mut();
        check(unsafe { ffi::rg_arena_new(config.game_mode as u32, config.no_ball_rot as u32, ptr, count, &mut raw) });
        let raw = NonNull::new(raw).expect("rocketgpu: null arena");
        let pad_configs = (0..unsafe { ffi::rg_pad_count(raw.as_ptr()) })
            .map(|i| {
                let mut p = ffi::RgPad::default();
                check(unsafe { ffi::rg_pad_config(raw.as_ptr(), i, &mut p) });
                BoostPadConfig { pos: Vec3A::from_array(p.pos), is_big: p.is_big != 0 }
            })
            .collect();
        let requested = config.mutators.take();
        let mut arena = Self {
            raw,
            config,
            cars: Vec::new(),
            ball: OnceCell::new(),
            pad_configs,
            pad_cooldowns: OnceCell::new(),
            events: Vec::new(),
        };
        match requested {
            Some(m) => arena.set_mutator_config(m),
            None => arena.config.mutators = Some(arena.pull_mutators()),
        }
        arena
    }

    pub const fn get_config(&self) -> &ArenaConfig {
        &self.config
    }

    pub const fn game_mode(&self) -> GameMode {
        self.config.game_mode
    }

    pub fn tick_count(&self) -> u64 {
        unsafe { ffi::rg_tick_count(self.raw.as_ptr()) }
    }

    pub fn is_ball_scored(&self) -> bool {
        let mut out = 0;
        check(unsafe { ffi::rg_is_ball_scored(self.raw.as_ptr(), &mut out) });
        out != 0
    }

    /// A kickoff on a new random spot layout. `None` draws the seed.
    pub fn reset_to_random_kickoff(&mut self, rng_seed: Option<u64>) {
        let seed = rng_seed.map_or(-1, |s| (s & 0x7fff_ffff) as i32);
        check(unsafe { ffi::rg_reset_kickoff(self.raw.as_ptr(), seed, 0) });
        self.invalidate();
    }

    /// RocketGPU's own: the match's next kickoff, as the game sets it up after a goal (`Arena::ResetToKickoff`).
    pub fn reset_to_kickoff(&mut self) {
        check(unsafe { ffi::rg_reset_kickoff(self.raw.as_ptr(), -1, 1) });
        self.invalidate();
    }

    pub fn add_car(&mut self, team: Team, config: CarBodyConfig) -> usize {
        let mut id = 0;
        check(unsafe { ffi::rg_add_car(self.raw.as_ptr(), team as u32, &config.to_ffi(), &mut id) });
        let idx = self.cars.len();
        self.cars.push(CarEntry {
            id,
            info: CarInfo { idx, team, config },
            controls: CarControls::DEFAULT,
            state: OnceCell::new(),
        });
        self.invalidate();
        idx
    }

    pub fn remove_car(&mut self, car_idx: usize) {
        check(unsafe { ffi::rg_remove_car(self.raw.as_ptr(), self.cars[car_idx].id) });
        self.cars.remove(car_idx);
        for (idx, car) in self.cars.iter_mut().enumerate() {
            car.info.idx = idx;
        }
        self.invalidate();
    }

    pub fn remove_cars(&mut self) {
        while !self.cars.is_empty() {
            self.remove_car(self.cars.len() - 1);
        }
    }

    pub fn num_cars(&self) -> usize {
        self.cars.len()
    }

    /// Steps one tick and returns its events.
    pub fn step_tick(&mut self) -> &[ArenaEvent] {
        let mut count = 0;
        check(unsafe { ffi::rg_step(self.raw.as_ptr(), &mut count) });
        let mut raw = vec![ffi::RgEvent::default(); count as usize];
        if count > 0 {
            check(unsafe { ffi::rg_events(self.raw.as_ptr(), raw.as_mut_ptr(), count) });
        }
        self.events.clear();
        for e in &raw {
            let car = self.idx_of(e.car_id);
            let normal = Vec3A::from_array(e.normal);
            let event = match (e.kind, car) {
                (0, Some(car_idx)) => ArenaEvent::CarHitBall(CarHitBallEvent { car_idx }),
                (1, _) => ArenaEvent::BallHitWorld(BallHitWorldEvent { contact_normal: normal }),
                (2, Some(car_idx)) => ArenaEvent::CarHitWorld(CarHitWorldEvent { car_idx, contact_normal: normal }),
                (3, Some(car_idx)) => {
                    ArenaEvent::CarPickupBoost(CarPickupBoostEvent { car_idx, boost_pad_idx: e.pad_index as usize })
                }
                (4, Some(bumper_car_idx)) => match self.idx_of(e.other_car_id) {
                    Some(victim_car_idx) => {
                        ArenaEvent::CarHitCar(CarHitCarEvent { bumper_car_idx, victim_car_idx, is_demo: e.is_demo != 0 })
                    }
                    None => continue,
                },
                _ => continue,
            };
            self.events.push(event);
        }
        self.invalidate();
        &self.events
    }

    pub fn get_last_step_events(&self) -> &[ArenaEvent] {
        &self.events
    }

    pub fn mutator_config(&self) -> &MutatorConfig {
        self.config.mutators.as_ref().expect("set at creation")
    }

    pub fn set_mutator_config(&mut self, mutators: MutatorConfig) {
        check(unsafe { ffi::rg_set_mutators(self.raw.as_ptr(), &mutators.to_ffi()) });
        self.config.mutators = Some(self.pull_mutators());
        self.invalidate();
    }

    pub fn get_ball_state(&self) -> &BallState {
        self.ball.get_or_init(|| {
            let mut b = ffi::RgBall::default();
            check(unsafe { ffi::rg_get_ball(self.raw.as_ptr(), &mut b) });
            BallState {
                phys: PhysState {
                    pos: Vec3A::from_array(b.pos),
                    rot_mat: rot_from_ffi(&b.rot),
                    vel: Vec3A::from_array(b.vel),
                    ang_vel: Vec3A::from_array(b.ang_vel),
                },
                activation_state: if b.sleeping != 0 { ActivationState::Sleeping } else { ActivationState::Active },
                deactivation_time: b.deactivation_time,
            }
        })
    }

    pub fn set_ball_state(&mut self, ball_state: BallState) {
        let b = ffi::RgBall {
            pos: ball_state.phys.pos.to_array(),
            rot: rot_to_ffi(&ball_state.phys.rot_mat),
            vel: ball_state.phys.vel.to_array(),
            ang_vel: ball_state.phys.ang_vel.to_array(),
            deactivation_time: ball_state.deactivation_time,
            sleeping: 0,
        };
        let activation = if ball_state.activation_state == ActivationState::Sleeping { 2 } else { 1 };
        check(unsafe { ffi::rg_set_ball(self.raw.as_ptr(), &b, activation) });
        self.ball = OnceCell::new();
    }

    pub fn edit_ball_state(&mut self, edit: impl FnOnce(&mut BallState)) {
        let mut state = *self.get_ball_state();
        edit(&mut state);
        self.set_ball_state(state);
    }

    /// Stops the ball and puts it to sleep: it stays where it is until a car touches it.
    pub fn sleep_ball(&mut self) {
        self.edit_ball_state(|b| {
            b.phys.vel = Vec3A::ZERO;
            b.phys.ang_vel = Vec3A::ZERO;
            b.activation_state = ActivationState::Sleeping;
        });
    }

    pub fn wake_ball(&mut self) {
        self.edit_ball_state(BallState::wake);
    }

    pub fn get_car_info(&self, car_idx: usize) -> &CarInfo {
        &self.cars[car_idx].info
    }

    pub fn get_car_state(&self, car_idx: usize) -> &CarState {
        let car = &self.cars[car_idx];
        car.state.get_or_init(|| {
            let mut s = ffi::RgCarState::default();
            check(unsafe { ffi::rg_get_car(self.raw.as_ptr(), car.id, &mut s) });
            self.state_from_ffi(&s)
        })
    }

    pub fn get_car_info_and_state(&self, car_idx: usize) -> (&CarInfo, &CarState) {
        (self.get_car_info(car_idx), self.get_car_state(car_idx))
    }

    pub fn get_car_controls(&self, car_idx: usize) -> &CarControls {
        &self.cars[car_idx].controls
    }

    pub fn set_car_controls(&mut self, car_idx: usize, controls: CarControls) {
        let car = &mut self.cars[car_idx];
        check(unsafe { ffi::rg_set_controls(self.raw.as_ptr(), car.id, &controls.to_ffi()) });
        car.controls = controls;
        if let Some(state) = car.state.get_mut() {
            state.controls = controls;
        }
    }

    pub fn set_car_state(&mut self, car_idx: usize, core: CarCore) {
        self.set_car_state_full(car_idx, CarState::from(core));
    }

    pub fn set_car_state_full(&mut self, car_idx: usize, state: CarState) {
        let s = self.state_to_ffi(&state);
        let car = &mut self.cars[car_idx];
        check(unsafe { ffi::rg_set_car(self.raw.as_ptr(), car.id, &s) });
        car.controls = state.controls;
        car.state = OnceCell::new();
    }

    pub fn edit_car_state_full(&mut self, car_idx: usize, edit: impl FnOnce(&mut CarState)) {
        let mut state = *self.get_car_state(car_idx);
        edit(&mut state);
        self.set_car_state_full(car_idx, state);
    }

    pub fn set_car_boost(&mut self, car_idx: usize, boost: f32) {
        self.edit_car_state_full(car_idx, |s| s.boost = boost);
    }

    /// Does nothing: RocketGPU keeps no separate sticky-force cache to refresh after a state set.
    pub fn refresh_car_sticky_gate(&mut self, _car_idx: usize) {}

    pub fn num_boost_pads(&self) -> usize {
        self.pad_configs.len()
    }

    pub fn get_boost_pad_config(&self, idx: usize) -> &BoostPadConfig {
        &self.pad_configs[idx]
    }

    pub fn get_all_boost_pad_configs(&self) -> Vec<BoostPadConfig> {
        self.pad_configs.clone()
    }

    pub fn get_boost_pad_state(&self, idx: usize) -> BoostPadState {
        BoostPadState { cooldown: self.cooldowns()[idx] }
    }

    pub fn get_all_boost_pad_states(&self) -> Vec<BoostPadState> {
        self.cooldowns().iter().map(|&cooldown| BoostPadState { cooldown }).collect()
    }

    pub fn set_boost_pad_state(&mut self, idx: usize, state: BoostPadState) {
        assert!(idx < self.pad_configs.len(), "rocketgpu: no boost pad {idx}");
        check(unsafe { ffi::rg_set_pad_cooldown(self.raw.as_ptr(), idx as u32, state.cooldown) });
        self.pad_cooldowns = OnceCell::new();
    }

    pub fn edit_boost_pad_state(&mut self, idx: usize, edit: impl FnOnce(&mut BoostPadState)) {
        let mut state = self.get_boost_pad_state(idx);
        edit(&mut state);
        self.set_boost_pad_state(idx, state);
    }

    fn cooldowns(&self) -> &Vec<f32> {
        self.pad_cooldowns.get_or_init(|| {
            let mut out = vec![0.0f32; self.pad_configs.len()];
            check(unsafe { ffi::rg_get_pad_cooldowns(self.raw.as_ptr(), out.as_mut_ptr(), out.len() as u32) });
            out
        })
    }

    fn pull_mutators(&self) -> MutatorConfig {
        let mut m = ffi::RgMutators::default();
        check(unsafe { ffi::rg_get_mutators(self.raw.as_ptr(), &mut m) });
        MutatorConfig::from_ffi(&m)
    }

    fn invalidate(&mut self) {
        for car in &mut self.cars {
            car.state = OnceCell::new();
        }
        self.ball = OnceCell::new();
        self.pad_cooldowns = OnceCell::new();
    }

    fn idx_of(&self, id: u32) -> Option<usize> {
        if id == 0 {
            return None;
        }
        self.cars.iter().position(|c| c.id == id)
    }

    fn state_from_ffi(&self, s: &ffi::RgCarState) -> CarState {
        let v = |a: [f32; 3]| Vec3A::from_array(a);
        CarState {
            phys: PhysState { pos: v(s.pos), rot_mat: rot_from_ffi(&s.rot), vel: v(s.vel), ang_vel: v(s.ang_vel) },
            controls: CarControls::from_ffi(&s.controls),
            has_jumped: s.has_jumped != 0,
            has_double_jumped: s.has_double_jumped != 0,
            has_flipped: s.has_flipped != 0,
            flip_rel_torque: v(s.flip_rel_torque),
            jump_ticks: s.jump_ticks,
            flip_time: s.flip_time,
            is_flipping: s.is_flipping != 0,
            is_jumping: s.is_jumping != 0,
            jump_hold_broken: s.jump_hold_broken != 0,
            air_time: s.air_time,
            air_time_since_jump: s.air_time_since_jump,
            boost: s.boost,
            handbrake_val: s.handbrake_val,
            is_supersonic: s.is_supersonic != 0,
            supersonic_grace_timer: s.supersonic_grace_timer,
            time_since_boosted: s.time_since_boosted,
            boosting_time: s.boosting_time,
            is_boosting: s.is_boosting != 0,
            is_auto_flipping: s.is_auto_flipping != 0,
            auto_flip_timer: s.auto_flip_timer,
            auto_flip_torque_scale: s.auto_flip_torque_scale,
            bump_cooldown_timer: s.bump_cooldown_timer,
            bump_cooldown_victim: self.idx_of(s.bump_cooldown_victim),
            last_extra_hit_tick: (s.last_extra_hit_tick != u64::MAX).then_some(s.last_extra_hit_tick),
            is_demoed: s.is_demoed != 0,
            demo_respawn_timer: s.demo_respawn_timer,
            cache: CarCache {
                is_on_ground: s.is_on_ground != 0,
                wheels_with_contact: s.wheels.map(|w| (w != 0).then_some(RaycastHitInfo { _private: () })),
                world_contact_normal: (s.has_world_contact != 0).then(|| v(s.world_contact_normal)),
                prev_controls: CarControls::from_ffi(&s.prev_controls),
                has_prev_controls: s.has_prev_controls != 0,
                has_last_extra_hit_tick: s.has_last_extra_hit_tick != 0,
                wheel_normals: (s.has_wheel_normals != 0).then(|| {
                    let n = &s.wheel_normals;
                    [0, 3, 6, 9].map(|i| Vec3A::new(n[i], n[i + 1], n[i + 2]))
                }),
            },
        }
    }

    fn state_to_ffi(&self, s: &CarState) -> ffi::RgCarState {
        let c = &s.cache;
        let mut wheel_normals = [0.0f32; 12];
        if let Some(normals) = &c.wheel_normals {
            for (w, n) in normals.iter().enumerate() {
                wheel_normals[3 * w..3 * w + 3].copy_from_slice(&n.to_array());
            }
        }
        ffi::RgCarState {
            last_extra_hit_tick: s.last_extra_hit_tick.unwrap_or(u64::MAX),
            pos: s.phys.pos.to_array(),
            rot: rot_to_ffi(&s.phys.rot_mat),
            vel: s.phys.vel.to_array(),
            ang_vel: s.phys.ang_vel.to_array(),
            flip_rel_torque: s.flip_rel_torque.to_array(),
            world_contact_normal: c.world_contact_normal.unwrap_or(Vec3A::ZERO).to_array(),
            wheel_normals,
            controls: s.controls.to_ffi(),
            prev_controls: c.prev_controls.to_ffi(),
            jump_ticks: s.jump_ticks,
            bump_cooldown_victim: s.bump_cooldown_victim.and_then(|i| self.cars.get(i)).map_or(0, |car| car.id),
            flip_time: s.flip_time,
            air_time: s.air_time,
            air_time_since_jump: s.air_time_since_jump,
            boost: s.boost,
            handbrake_val: s.handbrake_val,
            supersonic_grace_timer: s.supersonic_grace_timer,
            time_since_boosted: s.time_since_boosted,
            boosting_time: s.boosting_time,
            auto_flip_timer: s.auto_flip_timer,
            auto_flip_torque_scale: s.auto_flip_torque_scale,
            bump_cooldown_timer: s.bump_cooldown_timer,
            demo_respawn_timer: s.demo_respawn_timer,
            has_jumped: s.has_jumped as u8,
            has_double_jumped: s.has_double_jumped as u8,
            has_flipped: s.has_flipped as u8,
            is_flipping: s.is_flipping as u8,
            is_jumping: s.is_jumping as u8,
            jump_hold_broken: s.jump_hold_broken as u8,
            is_supersonic: s.is_supersonic as u8,
            is_boosting: s.is_boosting as u8,
            is_auto_flipping: s.is_auto_flipping as u8,
            is_demoed: s.is_demoed as u8,
            is_on_ground: c.is_on_ground as u8,
            wheels: c.wheels_with_contact.map(|w| w.is_some() as u8),
            has_world_contact: c.world_contact_normal.is_some() as u8,
            has_last_extra_hit_tick: (c.has_last_extra_hit_tick || s.last_extra_hit_tick.is_some()) as u8,
            has_prev_controls: c.has_prev_controls as u8,
            has_wheel_normals: c.wheel_normals.is_some() as u8,
            pad: [0; 5],
        }
    }
}
