#include <array>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "rocketgpu/rocketsim_compat.h"
#include "rocketgpu/detail/constants.hpp"
#include "rocketgpu/rocketsim_compat_events.h"

#include "rocketgpu/rocketgpu.h"

namespace py = pybind11;

namespace {
using CarWorldContact = decltype(std::declval<rocketgpu::CarState>().world_contact);
using CarContact = decltype(std::declval<rocketgpu::CarState>().car_contact);

cudaStream_t StreamFromInt(uintptr_t raw_stream) {
    return reinterpret_cast<cudaStream_t>(raw_stream);
}

template <typename T, size_t N>
std::array<T, N> ArrayFromCArray(const T (&input)[N]) {
    std::array<T, N> out{};
    for (size_t idx = 0; idx < N; ++idx) {
        out[idx] = input[idx];
    }
    return out;
}

template <typename T, size_t N>
void CopyStdArrayToCArray(const std::array<T, N>& input, T (&output)[N]) {
    for (size_t idx = 0; idx < N; ++idx) {
        output[idx] = input[idx];
    }
}

uintptr_t RocketGpuPtrToInt(const void* ptr) {
    return reinterpret_cast<uintptr_t>(ptr);
}
}

PYBIND11_MODULE(rocketgpu_py, m) {
    m.doc() = "Python bindings for RocketGPU";
    m.def("physics_event_layout", []() {
        py::dict d;
        d["record_size"] = sizeof(rocketgpu::PhysicsEvent);
        d["window_size"] = sizeof(rocketgpu::PhysicsEventWindow);
        d["type_offset"] = offsetof(rocketgpu::PhysicsEvent,type);
        d["velocity_offset"] = offsetof(rocketgpu::PhysicsEvent,ball_velocity);
        d["count_offset"] = offsetof(rocketgpu::PhysicsEventWindow,count);
        d["overflow_offset"] = offsetof(rocketgpu::PhysicsEventWindow,overflow);
        d["goal_type"] = static_cast<int>(rocketgpu::PhysicsEventType::Goal);
        return d;
    });

    py::class_<rocketgpu::Vec3>(m, "Vec3")
        .def(py::init<>())
        .def(py::init<float, float, float>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("x", &rocketgpu::Vec3::x)
        .def_readwrite("y", &rocketgpu::Vec3::y)
        .def_readwrite("z", &rocketgpu::Vec3::z);

    py::class_<rocketgpu::Quat>(m, "Quat")
        .def(py::init<>())
        .def(py::init<float, float, float, float>(), py::arg("w"), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("w", &rocketgpu::Quat::w)
        .def_readwrite("x", &rocketgpu::Quat::x)
        .def_readwrite("y", &rocketgpu::Quat::y)
        .def_readwrite("z", &rocketgpu::Quat::z);

    py::enum_<rocketgpu::Team>(m, "Team")
        .value("Blue", rocketgpu::Team::Blue)
        .value("Orange", rocketgpu::Team::Orange)
        .export_values();

    py::enum_<rocketgpu::Mode>(m, "Mode")
        .value("Soccar", rocketgpu::Mode::Soccar)
        .value("Heatseeker", rocketgpu::Mode::Heatseeker)
        .value("TheVoid", rocketgpu::Mode::TheVoid)
        .export_values();

    py::class_<rocketgpu::CarControls>(m, "CarControls")
        .def(py::init<>())
        .def_readwrite("throttle", &rocketgpu::CarControls::throttle)
        .def_readwrite("steer", &rocketgpu::CarControls::steer)
        .def_readwrite("pitch", &rocketgpu::CarControls::pitch)
        .def_readwrite("yaw", &rocketgpu::CarControls::yaw)
        .def_readwrite("roll", &rocketgpu::CarControls::roll)

        .def_property("jump",
            [](const rocketgpu::CarControls& c) { return c.jump != 0; },
            [](rocketgpu::CarControls& c, bool v) { c.jump = v ? 1 : 0; })

        .def_property("boost",
            [](const rocketgpu::CarControls& c) { return c.boost != 0; },
            [](rocketgpu::CarControls& c, bool v) { c.boost = v ? 1 : 0; })

        .def_property("handbrake",
            [](const rocketgpu::CarControls& c) { return c.handbrake != 0; },
            [](rocketgpu::CarControls& c, bool v) { c.handbrake = v ? 1 : 0; });

    py::class_<rocketgpu::CarConfig>(m, "CarConfig")
        .def(py::init<>())
        .def_readwrite("hitbox_size", &rocketgpu::CarConfig::hitbox_size)
        .def_readwrite("hitbox_offset", &rocketgpu::CarConfig::hitbox_offset)
        .def_readwrite("front_wheels_offset", &rocketgpu::CarConfig::front_wheels_offset)
        .def_readwrite("back_wheels_offset", &rocketgpu::CarConfig::back_wheels_offset)
        .def_readwrite("front_wheel_radius", &rocketgpu::CarConfig::front_wheel_radius)
        .def_readwrite("back_wheel_radius", &rocketgpu::CarConfig::back_wheel_radius)
        .def_readwrite("front_suspension_rest", &rocketgpu::CarConfig::front_suspension_rest)
        .def_readwrite("back_suspension_rest", &rocketgpu::CarConfig::back_suspension_rest)
        .def_readwrite("dodge_deadzone", &rocketgpu::CarConfig::dodge_deadzone)
        .def_readwrite("three_wheels", &rocketgpu::CarConfig::three_wheels);

    py::class_<rocketgpu::BallHitInfo>(m, "BallHitInfo")
        .def(py::init<>())
        .def_readwrite("is_valid", &rocketgpu::BallHitInfo::is_valid)
        .def_readwrite("tick_count_when_hit", &rocketgpu::BallHitInfo::tick_count_when_hit)
        .def_readwrite("tick_count_when_extra_impulse_applied", &rocketgpu::BallHitInfo::tick_count_when_extra_impulse_applied)
        .def_readwrite("hit_car_id", &rocketgpu::BallHitInfo::hit_car_id)
        .def_readwrite("relative_pos_on_ball", &rocketgpu::BallHitInfo::relative_pos_on_ball)
        .def_readwrite("ball_pos", &rocketgpu::BallHitInfo::ball_pos)
        .def_readwrite("extra_hit_vel", &rocketgpu::BallHitInfo::extra_hit_vel);

    py::class_<CarWorldContact>(m, "WorldContact")
        .def(py::init<>())
        .def_readwrite("has_contact", &CarWorldContact::has_contact)
        .def_readwrite("contact_normal", &CarWorldContact::contact_normal);

    py::class_<CarContact>(m, "CarContact")
        .def(py::init<>())
        .def_readwrite("other_car_id", &CarContact::other_car_id)
        .def_readwrite("cooldown_timer", &CarContact::cooldown_timer);

    py::class_<rocketgpu::CarState>(m, "CarState")
        .def(py::init<>())
        .def_readwrite("pos", &rocketgpu::CarState::pos)
        .def_readwrite("rot", &rocketgpu::CarState::rot)
        .def_readwrite("vel", &rocketgpu::CarState::vel)
        .def_readwrite("ang_vel", &rocketgpu::CarState::ang_vel)
        .def_readwrite("flip_rel_torque", &rocketgpu::CarState::flip_rel_torque)
        .def_readwrite("jump_time", &rocketgpu::CarState::jump_time)
        .def_readwrite("flip_time", &rocketgpu::CarState::flip_time)
        .def_readwrite("air_time", &rocketgpu::CarState::air_time)
        .def_readwrite("air_time_since_jump", &rocketgpu::CarState::air_time_since_jump)
        .def_readwrite("boost", &rocketgpu::CarState::boost)
        .def_readwrite("time_since_boosted", &rocketgpu::CarState::time_since_boosted)
        .def_readwrite("boosting_time", &rocketgpu::CarState::boosting_time)
        .def_readwrite("supersonic_time", &rocketgpu::CarState::supersonic_time)
        .def_readwrite("handbrake_val", &rocketgpu::CarState::handbrake_val)
        .def_readwrite("auto_flip_timer", &rocketgpu::CarState::auto_flip_timer)
        .def_readwrite("auto_flip_torque_scale", &rocketgpu::CarState::auto_flip_torque_scale)
        .def_readwrite("demo_respawn_timer", &rocketgpu::CarState::demo_respawn_timer)
        .def_readwrite("tick_count_since_update", &rocketgpu::CarState::tick_count_since_update)
        .def_readwrite("is_on_ground", &rocketgpu::CarState::is_on_ground)
        .def_readwrite("has_jumped", &rocketgpu::CarState::has_jumped)
        .def_readwrite("has_double_jumped", &rocketgpu::CarState::has_double_jumped)
        .def_readwrite("has_flipped", &rocketgpu::CarState::has_flipped)
        .def_readwrite("is_flipping", &rocketgpu::CarState::is_flipping)
        .def_readwrite("is_jumping", &rocketgpu::CarState::is_jumping)
        .def_readwrite("is_boosting", &rocketgpu::CarState::is_boosting)
        .def_readwrite("is_supersonic", &rocketgpu::CarState::is_supersonic)
        .def_readwrite("is_auto_flipping", &rocketgpu::CarState::is_auto_flipping)
        .def_readwrite("is_demoed", &rocketgpu::CarState::is_demoed)
        .def_readwrite("just_respawned", &rocketgpu::CarState::just_respawned)
        .def_readwrite("world_contact", &rocketgpu::CarState::world_contact)
        .def_readwrite("car_contact", &rocketgpu::CarState::car_contact)
        .def_readwrite("vel_impulse_cache", &rocketgpu::CarState::vel_impulse_cache)
        .def_readwrite("ball_hit_info", &rocketgpu::CarState::ball_hit_info)
        .def_readwrite("last_controls", &rocketgpu::CarState::last_controls)
        .def_readwrite("air_state", &rocketgpu::CarState::air_state)
        .def_readwrite("dodge_timeout", &rocketgpu::CarState::dodge_timeout)
        .def_readwrite("dodge_elapsed", &rocketgpu::CarState::dodge_elapsed)
        .def_readwrite("dodge_dir_x", &rocketgpu::CarState::dodge_dir_x)
        .def_readwrite("dodge_dir_y", &rocketgpu::CarState::dodge_dir_y)
        .def_readwrite("latest_touch_tick", &rocketgpu::CarState::latest_touch_tick)
        .def_readwrite("latest_touch_location", &rocketgpu::CarState::latest_touch_location)
        .def_readwrite("latest_touch_normal", &rocketgpu::CarState::latest_touch_normal)
        .def_property(
            "wheels_with_contact",
            [](const rocketgpu::CarState& state) {
                return ArrayFromCArray(state.wheels_with_contact);
            },
            [](rocketgpu::CarState& state, const std::array<uint8_t, 4>& wheels) {
                CopyStdArrayToCArray(wheels, state.wheels_with_contact);
            });

    py::class_<rocketgpu::BallState::HeatseekerInfo>(m, "HeatseekerInfo")
        .def(py::init<>())
        .def_readwrite("y_target_dir", &rocketgpu::BallState::HeatseekerInfo::y_target_dir)
        .def_readwrite("cur_target_speed", &rocketgpu::BallState::HeatseekerInfo::cur_target_speed)
        .def_readwrite("time_since_hit", &rocketgpu::BallState::HeatseekerInfo::time_since_hit);

    py::class_<rocketgpu::BallState::DropshotInfo>(m, "DropshotInfo")
        .def(py::init<>())
        .def_readwrite("charge_level", &rocketgpu::BallState::DropshotInfo::charge_level)
        .def_readwrite("accumulated_hit_force", &rocketgpu::BallState::DropshotInfo::accumulated_hit_force)
        .def_readwrite("y_target_dir", &rocketgpu::BallState::DropshotInfo::y_target_dir)
        .def_readwrite("has_damaged", &rocketgpu::BallState::DropshotInfo::has_damaged)
        .def_readwrite("last_damage_tick", &rocketgpu::BallState::DropshotInfo::last_damage_tick);

    py::class_<rocketgpu::BallState>(m, "BallState")
        .def(py::init<>())
        .def_readwrite("hs_info", &rocketgpu::BallState::hs_info)
        .def_readwrite("ds_info", &rocketgpu::BallState::ds_info)
        .def_readwrite("pos", &rocketgpu::BallState::pos)
        .def_readwrite("rot", &rocketgpu::BallState::rot)
        .def_readwrite("vel", &rocketgpu::BallState::vel)
        .def_readwrite("ang_vel", &rocketgpu::BallState::ang_vel)
        .def_readwrite("vel_impulse_cache", &rocketgpu::BallState::vel_impulse_cache)
        .def_readwrite("tick_count_when_extra_impulse_applied", &rocketgpu::BallState::tick_count_when_extra_impulse_applied)
        .def_readwrite("tick_count_since_update", &rocketgpu::BallState::tick_count_since_update);

    py::class_<rocketgpu::BoostPadState>(m, "BoostPadState")
        .def(py::init<>())
        .def_readwrite("cooldown", &rocketgpu::BoostPadState::cooldown)
        .def_readwrite("is_active", &rocketgpu::BoostPadState::is_active)
        .def_readwrite("is_big", &rocketgpu::BoostPadState::is_big)
        .def_readwrite("prev_locked_car_id", &rocketgpu::BoostPadState::prev_locked_car_id)
        .def_readwrite("cur_locked_car_id", &rocketgpu::BoostPadState::cur_locked_car_id);

    py::class_<rocketgpu::EnvState>(m, "EnvState")
        .def(py::init<>())
        .def_readwrite("tick", &rocketgpu::EnvState::tick)
        .def_readwrite("blue_score", &rocketgpu::EnvState::blue_score)
        .def_readwrite("orange_score", &rocketgpu::EnvState::orange_score)
        .def_readwrite("last_scored_team", &rocketgpu::EnvState::last_scored_team)
        .def_readwrite("goal_scored_this_tick", &rocketgpu::EnvState::goal_scored_this_tick)
        .def_readwrite("car_visit_order_mode", &rocketgpu::EnvState::car_visit_order_mode)
        .def_readwrite("car_visit_order_packed", &rocketgpu::EnvState::car_visit_order_packed);

    py::class_<rocketgpu::EnvCommand>(m, "EnvCommand")
        .def(py::init<>())
        .def_readwrite("reset_to_kickoff", &rocketgpu::EnvCommand::reset_to_kickoff)
        .def_readwrite("reset_scoreboard", &rocketgpu::EnvCommand::reset_scoreboard)
        .def_readwrite("reset_tick_count", &rocketgpu::EnvCommand::reset_tick_count)
        .def_readwrite("kickoff_seed", &rocketgpu::EnvCommand::kickoff_seed)
        .def_readwrite("kickoff_seed_mode", &rocketgpu::EnvCommand::kickoff_seed_mode);

    py::class_<rocketgpu::MutatorConfig>(m, "MutatorConfig")
        .def(py::init<>())
        .def_readwrite("gravity_z", &rocketgpu::MutatorConfig::gravity_z)
        .def_readwrite("ball_radius", &rocketgpu::MutatorConfig::ball_radius)
        .def_readwrite("ball_mass", &rocketgpu::MutatorConfig::ball_mass)
        .def_readwrite("car_mass", &rocketgpu::MutatorConfig::car_mass)
        .def_readwrite("ball_drag", &rocketgpu::MutatorConfig::ball_drag)
        .def_readwrite("ball_friction", &rocketgpu::MutatorConfig::ball_friction)
        .def_readwrite("ball_restitution", &rocketgpu::MutatorConfig::ball_restitution)
        .def_readwrite("ball_max_speed", &rocketgpu::MutatorConfig::ball_max_speed)
        .def_readwrite("ball_max_ang_speed", &rocketgpu::MutatorConfig::ball_max_ang_speed)
        .def_readwrite("car_max_speed", &rocketgpu::MutatorConfig::car_max_speed)
        .def_readwrite("car_max_ang_speed", &rocketgpu::MutatorConfig::car_max_ang_speed)
        .def_readwrite("car_air_throttle_accel", &rocketgpu::MutatorConfig::car_air_throttle_accel)
        .def_readwrite("boost_max", &rocketgpu::MutatorConfig::boost_max)
        .def_readwrite("boost_used_per_second", &rocketgpu::MutatorConfig::boost_used_per_second)
        .def_readwrite("boost_min_time", &rocketgpu::MutatorConfig::boost_min_time)
        .def_readwrite("boost_accel_ground", &rocketgpu::MutatorConfig::boost_accel_ground)
        .def_readwrite("boost_accel_air", &rocketgpu::MutatorConfig::boost_accel_air)
        .def_readwrite("boost_spawn_amount", &rocketgpu::MutatorConfig::boost_spawn_amount)
        .def_readwrite("recharge_boost_per_second", &rocketgpu::MutatorConfig::recharge_boost_per_second)
        .def_readwrite("recharge_boost_delay", &rocketgpu::MutatorConfig::recharge_boost_delay)
        .def_readwrite("recharge_boost_enabled", &rocketgpu::MutatorConfig::recharge_boost_enabled)
        .def_readwrite("jump_accel", &rocketgpu::MutatorConfig::jump_accel)
        .def_readwrite("jump_immediate_speed", &rocketgpu::MutatorConfig::jump_immediate_speed)
        .def_readwrite("jump_min_time", &rocketgpu::MutatorConfig::jump_min_time)
        .def_readwrite("jump_reset_time_pad", &rocketgpu::MutatorConfig::jump_reset_time_pad)
        .def_readwrite("jump_max_time", &rocketgpu::MutatorConfig::jump_max_time)
        .def_readwrite("doublejump_max_delay", &rocketgpu::MutatorConfig::doublejump_max_delay)
        .def_readwrite("flip_torque_time", &rocketgpu::MutatorConfig::flip_torque_time)
        .def_readwrite("flip_torque_min_time", &rocketgpu::MutatorConfig::flip_torque_min_time)
        .def_readwrite("flip_pitchlock_time", &rocketgpu::MutatorConfig::flip_pitchlock_time)
        .def_readwrite("flip_pitchlock_extra_time", &rocketgpu::MutatorConfig::flip_pitchlock_extra_time)
        .def_readwrite("flip_initial_vel_scale", &rocketgpu::MutatorConfig::flip_initial_vel_scale)
        .def_readwrite("flip_torque_x", &rocketgpu::MutatorConfig::flip_torque_x)
        .def_readwrite("flip_torque_y", &rocketgpu::MutatorConfig::flip_torque_y)
        .def_readwrite("flip_forward_impulse_max_speed_scale", &rocketgpu::MutatorConfig::flip_forward_impulse_max_speed_scale)
        .def_readwrite("flip_side_impulse_max_speed_scale", &rocketgpu::MutatorConfig::flip_side_impulse_max_speed_scale)
        .def_readwrite("flip_backward_impulse_max_speed_scale", &rocketgpu::MutatorConfig::flip_backward_impulse_max_speed_scale)
        .def_readwrite("flip_backward_impulse_scale_x", &rocketgpu::MutatorConfig::flip_backward_impulse_scale_x)
        .def_readwrite("unlimited_flips", &rocketgpu::MutatorConfig::unlimited_flips)
        .def_readwrite("unlimited_double_jumps", &rocketgpu::MutatorConfig::unlimited_double_jumps)
        .def_readwrite("enable_team_demos", &rocketgpu::MutatorConfig::enable_team_demos)
        .def_readwrite("supersonic_start_speed", &rocketgpu::MutatorConfig::supersonic_start_speed)
        .def_readwrite("supersonic_maintain_min_speed", &rocketgpu::MutatorConfig::supersonic_maintain_min_speed)
        .def_readwrite("supersonic_maintain_max_time", &rocketgpu::MutatorConfig::supersonic_maintain_max_time)
        .def_readwrite("powerslide_rise_rate", &rocketgpu::MutatorConfig::powerslide_rise_rate)
        .def_readwrite("powerslide_fall_rate", &rocketgpu::MutatorConfig::powerslide_fall_rate)
        .def_readwrite("arena_friction", &rocketgpu::MutatorConfig::arena_friction)
        .def_readwrite("arena_restitution", &rocketgpu::MutatorConfig::arena_restitution)
        .def_readwrite("sticky_force", &rocketgpu::MutatorConfig::sticky_force)
        .def_readwrite("ball_car_extra_impulse_z_scale", &rocketgpu::MutatorConfig::ball_car_extra_impulse_z_scale)
        .def_readwrite("ball_car_extra_impulse_forward_scale", &rocketgpu::MutatorConfig::ball_car_extra_impulse_forward_scale)
        .def_readwrite("ball_car_extra_impulse_max_delta_vel", &rocketgpu::MutatorConfig::ball_car_extra_impulse_max_delta_vel)
        .def_readwrite("ball_hit_extra_force_scale", &rocketgpu::MutatorConfig::ball_hit_extra_force_scale)
        .def_readwrite("bump_force_scale", &rocketgpu::MutatorConfig::bump_force_scale)
        .def_readwrite("goal_base_threshold_y", &rocketgpu::MutatorConfig::goal_base_threshold_y)
        .def_readwrite("bump_cooldown_time", &rocketgpu::MutatorConfig::bump_cooldown_time)
        .def_readwrite("bump_min_forward_dist", &rocketgpu::MutatorConfig::bump_min_forward_dist)
        .def_readwrite("demo_respawn_time", &rocketgpu::MutatorConfig::demo_respawn_time)
        .def_readwrite("boost_pad_cooldown_small", &rocketgpu::MutatorConfig::boost_pad_cooldown_small)
        .def_readwrite("boost_pad_cooldown_big", &rocketgpu::MutatorConfig::boost_pad_cooldown_big)
        .def_readwrite("car_air_control_pitch", &rocketgpu::MutatorConfig::car_air_control_pitch)
        .def_readwrite("car_air_control_yaw", &rocketgpu::MutatorConfig::car_air_control_yaw)
        .def_readwrite("car_air_control_roll", &rocketgpu::MutatorConfig::car_air_control_roll)
        .def_readwrite("car_air_damp_pitch", &rocketgpu::MutatorConfig::car_air_damp_pitch)
        .def_readwrite("car_air_damp_yaw", &rocketgpu::MutatorConfig::car_air_damp_yaw)
        .def_readwrite("car_air_damp_roll", &rocketgpu::MutatorConfig::car_air_damp_roll);

    py::class_<rocketgpu::BoostPadConfig>(m, "BoostPadConfig")
        .def(py::init<>())
        .def_readwrite("pos", &rocketgpu::BoostPadConfig::pos)
        .def_readwrite("is_big", &rocketgpu::BoostPadConfig::is_big);

    py::class_<rocketgpu::SimConfig>(m, "SimConfig")
        .def(py::init<>())
        .def_readwrite("env_count", &rocketgpu::SimConfig::env_count)
        .def_readwrite("max_cars_per_env", &rocketgpu::SimConfig::max_cars_per_env)
        .def_readwrite("tick_dt", &rocketgpu::SimConfig::tick_dt)
        .def_readwrite("mode", &rocketgpu::SimConfig::mode)
        .def_readwrite("enable_boost_pads", &rocketgpu::SimConfig::enable_boost_pads)
        .def_readwrite("enable_car_car_collisions", &rocketgpu::SimConfig::enable_car_car_collisions)
        .def_readwrite("enable_demos", &rocketgpu::SimConfig::enable_demos)
        .def_readwrite("enable_scoring", &rocketgpu::SimConfig::enable_scoring)
        .def_readwrite("no_ball_rot", &rocketgpu::SimConfig::no_ball_rot)
        .def_readwrite("native_sphere_normalization", &rocketgpu::SimConfig::native_sphere_normalization)
        .def_readwrite("default_car", &rocketgpu::SimConfig::default_car)
        .def_readwrite("mutator", &rocketgpu::SimConfig::mutator)
        .def_readwrite("use_custom_boost_pads", &rocketgpu::SimConfig::use_custom_boost_pads)
        .def_readwrite("boost_pad_count", &rocketgpu::SimConfig::boost_pad_count)

        .def_property(
            "boost_pads",
            [](const rocketgpu::SimConfig& c) {
                std::vector<rocketgpu::BoostPadConfig> out;
                for (int p = 0; p < c.boost_pad_count && p < rocketgpu::kMaxBoostPads; ++p) out.push_back(c.boost_pads[p]);
                return out;
            },
            [](rocketgpu::SimConfig& c, const std::vector<rocketgpu::BoostPadConfig>& pads) {
                if (pads.size() > static_cast<size_t>(rocketgpu::kMaxBoostPads)) throw std::invalid_argument("too many boost pads");
                c.use_custom_boost_pads = 1;
                c.boost_pad_count = static_cast<int32_t>(pads.size());
                for (size_t p = 0; p < pads.size(); ++p) c.boost_pads[p] = pads[p];
            });

#if defined(ROCKETGPU_USE_CUDA)
    m.attr("is_cuda_build") = true;
#else
    m.attr("is_cuda_build") = false;
#endif

    m.def("boost_pad_locations", []() {
        py::list out;
        for (int i = 0; i < rocketgpu::detail::BoostPadCountSoccar(); ++i) {
            out.append(py::make_tuple(rocketgpu::detail::BoostPadLocSoccar(i), rocketgpu::detail::BoostPadIsBigSoccar(i) != 0));
        }
        return out;
    });
    m.def("state_layouts", []() {
        py::dict out;
        auto vec3 = [](const char* name, size_t off, py::list& fields) {
            fields.append(py::make_tuple(name, "<f4", off, py::make_tuple(3)));
        };
        auto quat = [](const char* name, size_t off, py::list& fields) {
            fields.append(py::make_tuple(name, "<f4", off, py::make_tuple(4)));
        };
        auto scalar = [](const char* name, const char* typestr, size_t off, py::list& fields) {
            fields.append(py::make_tuple(name, typestr, off, py::make_tuple()));
        };
        auto u8arr = [](const char* name, size_t off, size_t n, py::list& fields) {
            fields.append(py::make_tuple(name, "|u1", off, py::make_tuple(n)));
        };
        auto finish = [&](const char* name, size_t itemsize, py::list& fields) {
            py::dict d;
            d["itemsize"] = itemsize;
            d["fields"] = fields;
            out[name] = d;
        };
        auto controls_fields = [&](const std::string& prefix, size_t base, py::list& f) {
            using S = rocketgpu::CarControls;
            scalar((prefix + "throttle").c_str(), "<f4", base + offsetof(S, throttle), f);
            scalar((prefix + "steer").c_str(), "<f4", base + offsetof(S, steer), f);
            scalar((prefix + "pitch").c_str(), "<f4", base + offsetof(S, pitch), f);
            scalar((prefix + "yaw").c_str(), "<f4", base + offsetof(S, yaw), f);
            scalar((prefix + "roll").c_str(), "<f4", base + offsetof(S, roll), f);
            scalar((prefix + "jump").c_str(), "|u1", base + offsetof(S, jump), f);
            scalar((prefix + "boost").c_str(), "|u1", base + offsetof(S, boost), f);
            scalar((prefix + "handbrake").c_str(), "|u1", base + offsetof(S, handbrake), f);
        };
        {
            py::list f;
            controls_fields("", 0, f);
            finish("controls", sizeof(rocketgpu::CarControls), f);
        }
        {
            using S = rocketgpu::CarConfig;
            py::list f;
            vec3("hitbox_size", offsetof(S, hitbox_size), f);
            vec3("hitbox_offset", offsetof(S, hitbox_offset), f);
            vec3("front_wheels_offset", offsetof(S, front_wheels_offset), f);
            vec3("back_wheels_offset", offsetof(S, back_wheels_offset), f);
            scalar("front_wheel_radius", "<f4", offsetof(S, front_wheel_radius), f);
            scalar("back_wheel_radius", "<f4", offsetof(S, back_wheel_radius), f);
            scalar("front_suspension_rest", "<f4", offsetof(S, front_suspension_rest), f);
            scalar("back_suspension_rest", "<f4", offsetof(S, back_suspension_rest), f);
            scalar("dodge_deadzone", "<f4", offsetof(S, dodge_deadzone), f);
            scalar("three_wheels", "|u1", offsetof(S, three_wheels), f);
            finish("car_configs", sizeof(S), f);
        }
        {
            py::list f;
            scalar("team", "|u1", 0, f);
            finish("teams", sizeof(rocketgpu::Team), f);
        }
        {
            py::list f;
            scalar("count", "<i4", 0, f);
            finish("env_car_count", sizeof(int32_t), f);
        }
        auto hit_fields = [&](const std::string& prefix, size_t base, py::list& f) {
            using S = rocketgpu::BallHitInfo;
            scalar((prefix + "is_valid").c_str(), "|u1", base + offsetof(S, is_valid), f);
            scalar((prefix + "tick_count_when_hit").c_str(), "<u8", base + offsetof(S, tick_count_when_hit), f);
            scalar((prefix + "tick_count_when_extra_impulse_applied").c_str(), "<u8", base + offsetof(S, tick_count_when_extra_impulse_applied), f);
            scalar((prefix + "hit_car_id").c_str(), "<u4", base + offsetof(S, hit_car_id), f);
            vec3((prefix + "relative_pos_on_ball").c_str(), base + offsetof(S, relative_pos_on_ball), f);
            vec3((prefix + "ball_pos").c_str(), base + offsetof(S, ball_pos), f);
            vec3((prefix + "extra_hit_vel").c_str(), base + offsetof(S, extra_hit_vel), f);
        };
        {
            using S = rocketgpu::CarState;
            py::list f;
            vec3("pos", offsetof(S, pos), f);
            quat("rot", offsetof(S, rot), f);
            vec3("vel", offsetof(S, vel), f);
            vec3("ang_vel", offsetof(S, ang_vel), f);
            vec3("flip_rel_torque", offsetof(S, flip_rel_torque), f);
            scalar("jump_time", "<f4", offsetof(S, jump_time), f);
            scalar("flip_time", "<f4", offsetof(S, flip_time), f);
            scalar("air_time", "<f4", offsetof(S, air_time), f);
            scalar("air_time_since_jump", "<f4", offsetof(S, air_time_since_jump), f);
            scalar("boost", "<f4", offsetof(S, boost), f);
            scalar("time_since_boosted", "<f4", offsetof(S, time_since_boosted), f);
            scalar("boosting_time", "<f4", offsetof(S, boosting_time), f);
            scalar("supersonic_time", "<f4", offsetof(S, supersonic_time), f);
            scalar("handbrake_val", "<f4", offsetof(S, handbrake_val), f);
            scalar("auto_flip_timer", "<f4", offsetof(S, auto_flip_timer), f);
            scalar("auto_flip_torque_scale", "<f4", offsetof(S, auto_flip_torque_scale), f);
            scalar("demo_respawn_timer", "<f4", offsetof(S, demo_respawn_timer), f);
            scalar("tick_count_since_update", "<u8", offsetof(S, tick_count_since_update), f);
            scalar("is_on_ground", "|u1", offsetof(S, is_on_ground), f);
            u8arr("wheels_with_contact", offsetof(S, wheels_with_contact), 4, f);
            scalar("has_jumped", "|u1", offsetof(S, has_jumped), f);
            scalar("has_double_jumped", "|u1", offsetof(S, has_double_jumped), f);
            scalar("has_flipped", "|u1", offsetof(S, has_flipped), f);
            scalar("is_flipping", "|u1", offsetof(S, is_flipping), f);
            scalar("is_jumping", "|u1", offsetof(S, is_jumping), f);
            scalar("is_boosting", "|u1", offsetof(S, is_boosting), f);
            scalar("is_supersonic", "|u1", offsetof(S, is_supersonic), f);
            scalar("is_auto_flipping", "|u1", offsetof(S, is_auto_flipping), f);
            scalar("is_demoed", "|u1", offsetof(S, is_demoed), f);
            scalar("just_respawned", "|u1", offsetof(S, just_respawned), f);
            scalar("demo_no_contact_response", "|u1", offsetof(S, demo_no_contact_response), f);
            scalar("world_contact.has_contact", "|u1", offsetof(S, world_contact) + offsetof(CarWorldContact, has_contact), f);
            scalar("world_contact.wheel_ball_mask", "|u1", offsetof(S, world_contact) + offsetof(CarWorldContact, wheel_ball_mask), f);
            vec3("world_contact.contact_normal", offsetof(S, world_contact) + offsetof(CarWorldContact, contact_normal), f);
            scalar("car_contact.other_car_id", "<u4", offsetof(S, car_contact) + offsetof(CarContact, other_car_id), f);
            scalar("car_contact.cooldown_timer", "<f4", offsetof(S, car_contact) + offsetof(CarContact, cooldown_timer), f);
            vec3("vel_impulse_cache", offsetof(S, vel_impulse_cache), f);
            hit_fields("ball_hit_info.", offsetof(S, ball_hit_info), f);
            controls_fields("last_controls.", offsetof(S, last_controls), f);
            vec3("pos_bt", offsetof(S, pos_bt), f);
            vec3("vel_bt", offsetof(S, vel_bt), f);
            vec3("basis_forward", offsetof(S, basis_forward), f);
            vec3("basis_right", offsetof(S, basis_right), f);
            vec3("basis_up", offsetof(S, basis_up), f);
            scalar("basis_authoritative", "|u1", offsetof(S, basis_authoritative), f);
            scalar("bump_event_tick", "<u8", offsetof(S, bump_event_tick), f);
            scalar("bump_event_other_id", "<u4", offsetof(S, bump_event_other_id), f);
            scalar("bump_event_is_demo", "|u1", offsetof(S, bump_event_is_demo), f);
            scalar("bump_event_was_bumper", "|u1", offsetof(S, bump_event_was_bumper), f);
            scalar("air_state", "|u1", offsetof(S, air_state), f);
            scalar("dodge_timeout", "<f4", offsetof(S, dodge_timeout), f);
            scalar("dodge_elapsed", "<f4", offsetof(S, dodge_elapsed), f);
            scalar("dodge_dir_x", "<f4", offsetof(S, dodge_dir_x), f);
            scalar("dodge_dir_y", "<f4", offsetof(S, dodge_dir_y), f);
            scalar("latest_touch_tick", "<u8", offsetof(S, latest_touch_tick), f);
            vec3("latest_touch_location", offsetof(S, latest_touch_location), f);
            vec3("latest_touch_normal", offsetof(S, latest_touch_normal), f);
            finish("car_states", sizeof(S), f);
        }
        {
            using S = rocketgpu::BallState;
            py::list f;
            scalar("hs_info.y_target_dir", "<f4", offsetof(S, hs_info) + offsetof(S::HeatseekerInfo, y_target_dir), f);
            scalar("hs_info.cur_target_speed", "<f4", offsetof(S, hs_info) + offsetof(S::HeatseekerInfo, cur_target_speed), f);
            scalar("hs_info.time_since_hit", "<f4", offsetof(S, hs_info) + offsetof(S::HeatseekerInfo, time_since_hit), f);
            scalar("ds_info.charge_level", "<i4", offsetof(S, ds_info) + offsetof(S::DropshotInfo, charge_level), f);
            scalar("ds_info.accumulated_hit_force", "<f4", offsetof(S, ds_info) + offsetof(S::DropshotInfo, accumulated_hit_force), f);
            scalar("ds_info.y_target_dir", "<f4", offsetof(S, ds_info) + offsetof(S::DropshotInfo, y_target_dir), f);
            scalar("ds_info.has_damaged", "|u1", offsetof(S, ds_info) + offsetof(S::DropshotInfo, has_damaged), f);
            scalar("ds_info.last_damage_tick", "<u8", offsetof(S, ds_info) + offsetof(S::DropshotInfo, last_damage_tick), f);
            vec3("pos", offsetof(S, pos), f);
            quat("rot", offsetof(S, rot), f);
            vec3("vel", offsetof(S, vel), f);
            vec3("ang_vel", offsetof(S, ang_vel), f);
            vec3("vel_impulse_cache", offsetof(S, vel_impulse_cache), f);
            scalar("tick_count_when_extra_impulse_applied", "<u8", offsetof(S, tick_count_when_extra_impulse_applied), f);
            scalar("tick_count_since_update", "<u8", offsetof(S, tick_count_since_update), f);
            vec3("pos_bt", offsetof(S, pos_bt), f);
            vec3("vel_bt", offsetof(S, vel_bt), f);
            vec3("basis_forward", offsetof(S, basis_forward), f);
            vec3("basis_right", offsetof(S, basis_right), f);
            vec3("basis_up", offsetof(S, basis_up), f);
            scalar("basis_authoritative", "<u4", offsetof(S, basis_authoritative), f);
            vec3("vel_impulse_cache_bt", offsetof(S, vel_impulse_cache_bt), f);
            finish("ball_states", sizeof(S), f);
        }
        {
            using S = rocketgpu::BoostPadState;
            py::list f;
            scalar("cooldown", "<f4", offsetof(S, cooldown), f);
            scalar("is_active", "|u1", offsetof(S, is_active), f);
            scalar("is_big", "|u1", offsetof(S, is_big), f);
            scalar("prev_locked_car_id", "<u4", offsetof(S, prev_locked_car_id), f);
            scalar("cur_locked_car_id", "<u4", offsetof(S, cur_locked_car_id), f);
            finish("boost_pad_states", sizeof(S), f);
        }
        {
            using S = rocketgpu::EnvState;
            py::list f;
            scalar("tick", "<u4", offsetof(S, tick), f);
            scalar("blue_score", "<u4", offsetof(S, blue_score), f);
            scalar("orange_score", "<u4", offsetof(S, orange_score), f);
            scalar("last_scored_team", "<i4", offsetof(S, last_scored_team), f);
            scalar("goal_scored_this_tick", "|u1", offsetof(S, goal_scored_this_tick), f);
            scalar("car_visit_order_mode", "|u1", offsetof(S, car_visit_order_mode), f);
            scalar("car_visit_order_packed", "<u4", offsetof(S, car_visit_order_packed), f);
            finish("env_states", sizeof(S), f);
        }
        return out;
    });

    py::class_<rocketgpu::DeviceView>(m, "DeviceView")
        .def_property_readonly("car_states", [](const rocketgpu::DeviceView& view) { return RocketGpuPtrToInt(view.car_states); })
        .def_property_readonly("controls", [](const rocketgpu::DeviceView& view) { return RocketGpuPtrToInt(view.controls); })
        .def_property_readonly("car_configs", [](const rocketgpu::DeviceView& view) { return RocketGpuPtrToInt(view.car_configs); })
        .def_property_readonly("teams", [](const rocketgpu::DeviceView& view) { return RocketGpuPtrToInt(view.teams); })
        .def_property_readonly("ball_states", [](const rocketgpu::DeviceView& view) { return RocketGpuPtrToInt(view.ball_states); })
        .def_property_readonly("env_states", [](const rocketgpu::DeviceView& view) { return RocketGpuPtrToInt(view.env_states); })
        .def_property_readonly("boost_pad_states", [](const rocketgpu::DeviceView& view) { return RocketGpuPtrToInt(view.boost_pad_states); })
        .def_property_readonly("env_car_count", [](const rocketgpu::DeviceView& view) { return RocketGpuPtrToInt(view.env_car_count); });

    py::class_<rocketgpu::BatchSim>(m, "BatchSim")
        .def(py::init<const rocketgpu::SimConfig&>(), py::arg("config"))
        .def_property_readonly("config", &rocketgpu::BatchSim::config, py::return_value_policy::reference_internal)
        .def("device_view", &rocketgpu::BatchSim::device_view)
        .def("set_physics_event_stream", [](rocketgpu::BatchSim& sim, uintptr_t records,
              uintptr_t windows, uint32_t capacity) {
            rocketgpu::PhysicsEventStream stream;
            stream.records = reinterpret_cast<rocketgpu::PhysicsEvent*>(records);
            stream.windows = reinterpret_cast<rocketgpu::PhysicsEventWindow*>(windows);
            stream.capacity_per_env = capacity;
            stream.env_count = static_cast<uint32_t>(sim.env_count());
            stream.cars_per_env = static_cast<uint32_t>(sim.car_capacity()/sim.env_count());
            sim.set_physics_event_stream(stream);
        })
        .def("device_arrays",
             [](rocketgpu::BatchSim& sim) {
                 const rocketgpu::DeviceView v = sim.device_view();
                 py::dict out;
                 auto add = [&](const char* name, const void* ptr, size_t count, size_t elem) {
                     py::dict cai;
                     cai["data"] = py::make_tuple(RocketGpuPtrToInt(ptr), false);
                     cai["shape"] = py::make_tuple(count);
                     cai["typestr"] = "|V" + std::to_string(elem);
                     cai["strides"] = py::none();
                     cai["version"] = 3;
                     py::dict entry;
                     entry["__cuda_array_interface__"] = cai;
                     entry["itemsize"] = elem;
                     entry["count"] = count;
                     out[name] = entry;
                 };
                 add("car_states", v.car_states, sim.car_capacity(), sizeof(rocketgpu::CarState));
                 add("controls", v.controls, sim.car_capacity(), sizeof(rocketgpu::CarControls));
                 add("car_configs", v.car_configs, sim.car_capacity(), sizeof(rocketgpu::CarConfig));
                 add("teams", v.teams, sim.car_capacity(), sizeof(rocketgpu::Team));
                 add("ball_states", v.ball_states, sim.env_count(), sizeof(rocketgpu::BallState));
                 add("env_states", v.env_states, sim.env_count(), sizeof(rocketgpu::EnvState));
                 add("boost_pad_states", v.boost_pad_states, sim.boost_pad_capacity(), sizeof(rocketgpu::BoostPadState));
                 add("env_car_count", v.env_car_count, sim.env_count(), sizeof(int32_t));
                 return out;
             },
             "Zero-copy CUDA array interface dicts (per state array) for CuPy/PyTorch/Numba")
        .def("car_capacity", &rocketgpu::BatchSim::car_capacity)
        .def("env_count", &rocketgpu::BatchSim::env_count)
        .def("boost_pad_capacity", &rocketgpu::BatchSim::boost_pad_capacity)
        .def(
            "set_env_car_count",
            [](rocketgpu::BatchSim& sim, const std::vector<int32_t>& counts, uintptr_t stream) {
                sim.set_env_car_count(counts.data(), counts.size(), StreamFromInt(stream));
            },
            py::arg("counts"),
            py::arg("stream") = 0)
        .def(
            "set_teams",
            [](rocketgpu::BatchSim& sim, const std::vector<rocketgpu::Team>& teams, uintptr_t stream) {
                sim.set_teams(teams.data(), teams.size(), StreamFromInt(stream));
            },
            py::arg("teams"),
            py::arg("stream") = 0)
        .def(
            "set_car_configs",
            [](rocketgpu::BatchSim& sim, const std::vector<rocketgpu::CarConfig>& configs, uintptr_t stream) {
                sim.set_car_configs(configs.data(), configs.size(), StreamFromInt(stream));
            },
            py::arg("configs"),
            py::arg("stream") = 0)
        .def(
            "upload_car_states",
            [](rocketgpu::BatchSim& sim, const std::vector<rocketgpu::CarState>& states, uintptr_t stream) {
                sim.upload_car_states(states.data(), states.size(), StreamFromInt(stream));
            },
            py::arg("states"),
            py::arg("stream") = 0)
        .def(
            "upload_ball_states",
            [](rocketgpu::BatchSim& sim, const std::vector<rocketgpu::BallState>& states, uintptr_t stream) {
                sim.upload_ball_states(states.data(), states.size(), StreamFromInt(stream));
            },
            py::arg("states"),
            py::arg("stream") = 0)
        .def(
            "upload_controls",
            [](rocketgpu::BatchSim& sim, const std::vector<rocketgpu::CarControls>& controls, uintptr_t stream) {
                sim.upload_controls(controls.data(), controls.size(), StreamFromInt(stream));
            },
            py::arg("controls"),
            py::arg("stream") = 0)
        .def(
            "upload_env_states",
            [](rocketgpu::BatchSim& sim, const std::vector<rocketgpu::EnvState>& states, uintptr_t stream) {
                sim.upload_env_states(states.data(), states.size(), StreamFromInt(stream));
            },
            py::arg("states"),
            py::arg("stream") = 0)
        .def(
            "upload_boost_pad_states",
            [](rocketgpu::BatchSim& sim, const std::vector<rocketgpu::BoostPadState>& states, uintptr_t stream) {
                sim.upload_boost_pad_states(states.data(), states.size(), StreamFromInt(stream));
            },
            py::arg("states"),
            py::arg("stream") = 0)
        .def(
            "download_car_states",
            [](const rocketgpu::BatchSim& sim, size_t count, uintptr_t stream) {
                std::vector<rocketgpu::CarState> states(count);
                sim.download_car_states(states.data(), count, StreamFromInt(stream));
                return states;
            },
            py::arg("count"),
            py::arg("stream") = 0)
        .def(
            "download_ball_states",
            [](const rocketgpu::BatchSim& sim, size_t count, uintptr_t stream) {
                std::vector<rocketgpu::BallState> states(count);
                sim.download_ball_states(states.data(), count, StreamFromInt(stream));
                return states;
            },
            py::arg("count"),
            py::arg("stream") = 0)
        .def(
            "download_env_states",
            [](const rocketgpu::BatchSim& sim, size_t count, uintptr_t stream) {
                std::vector<rocketgpu::EnvState> states(count);
                sim.download_env_states(states.data(), count, StreamFromInt(stream));
                return states;
            },
            py::arg("count"),
            py::arg("stream") = 0)
        .def(
            "download_boost_pad_states",
            [](const rocketgpu::BatchSim& sim, size_t count, uintptr_t stream) {
                std::vector<rocketgpu::BoostPadState> states(count);
                sim.download_boost_pad_states(states.data(), count, StreamFromInt(stream));
                return states;
            },
            py::arg("count"),
            py::arg("stream") = 0)
        .def(
            "reset_to_kickoff",
            [](rocketgpu::BatchSim& sim, uint64_t seed, uintptr_t stream) {
                sim.reset_to_kickoff(seed, StreamFromInt(stream));
            },
            py::arg("seed") = 0,
            py::arg("stream") = 0)
        .def(
            "apply_env_commands",
            [](rocketgpu::BatchSim& sim, const std::vector<rocketgpu::EnvCommand>& commands, uintptr_t stream) {
                sim.apply_env_commands(commands.data(), commands.size(), StreamFromInt(stream));
            },
            py::arg("commands"),
            py::arg("stream") = 0)
        .def(
            "apply_env_commands_device",
            [](rocketgpu::BatchSim& sim, uintptr_t device_commands, size_t count, uintptr_t stream) {
                sim.apply_env_commands_device(
                    reinterpret_cast<const rocketgpu::EnvCommand*>(device_commands),
                    count,
                    StreamFromInt(stream));
            },
            py::arg("device_commands"),
            py::arg("count"),
            py::arg("stream") = 0)
        .def(
            "step",
            [](rocketgpu::BatchSim& sim, int32_t ticks, uintptr_t stream) {
                sim.step(ticks, StreamFromInt(stream));
            },
            py::arg("ticks") = 1,
            py::arg("stream") = 0)
        .def(
            "step_with_device_controls",
            [](rocketgpu::BatchSim& sim, uintptr_t device_controls, int32_t ticks, int32_t control_stride, uintptr_t stream) {
                sim.step_with_device_controls(
                    reinterpret_cast<const rocketgpu::CarControls*>(device_controls),
                    ticks,
                    control_stride,
                    StreamFromInt(stream));
            },
            py::arg("device_controls"),
            py::arg("ticks"),
            py::arg("control_stride"),
            py::arg("stream") = 0)
        .def(
            "step_with_device_commands",
            [](rocketgpu::BatchSim& sim,
               uintptr_t device_controls,
               int32_t control_stride,
               uintptr_t device_commands,
               int32_t env_command_stride,
               int32_t ticks,
               uintptr_t stream) {
                sim.step_with_device_commands(
                    reinterpret_cast<const rocketgpu::CarControls*>(device_controls),
                    control_stride,
                    reinterpret_cast<const rocketgpu::EnvCommand*>(device_commands),
                    env_command_stride,
                    ticks,
                    StreamFromInt(stream));
            },
            py::arg("device_controls"),
            py::arg("control_stride"),
            py::arg("device_commands"),
            py::arg("env_command_stride"),
            py::arg("ticks"),
            py::arg("stream") = 0)
        .def(
            "synchronize",
            [](const rocketgpu::BatchSim& sim, uintptr_t stream) {
                sim.synchronize(StreamFromInt(stream));
            },
            py::arg("stream") = 0);

    m.def("OctaneConfig", &rocketgpu::OctaneConfig);

    {
        namespace rc = rocketgpu::compat;
        py::module_ c = m.def_submodule("compat", "RocketSim-shaped single-arena API over BatchSim");
        py::enum_<rc::GameMode>(c, "GameMode").value("SOCCAR", rc::GameMode::SOCCAR).export_values()
            .value("HOOPS", rc::GameMode::HOOPS)
            .value("HEATSEEKER", rc::GameMode::HEATSEEKER)
            .value("SNOWDAY", rc::GameMode::SNOWDAY)
            .value("DROPSHOT", rc::GameMode::DROPSHOT)
            .value("THE_VOID", rc::GameMode::THE_VOID);
        py::class_<rc::RotMat>(c, "RotMat")
            .def(py::init<>())
            .def_readwrite("forward", &rc::RotMat::forward)
            .def_readwrite("right", &rc::RotMat::right)
            .def_readwrite("up", &rc::RotMat::up)
            .def_static("get_identity", &rc::RotMat::GetIdentity);
        py::class_<rc::BallHitInfoCompat>(c, "BallHitInfo")
            .def(py::init<>())
            .def_readwrite("is_valid", &rc::BallHitInfoCompat::isValid)
            .def_readwrite("relative_pos_on_ball", &rc::BallHitInfoCompat::relativePosOnBall)
            .def_readwrite("ball_pos", &rc::BallHitInfoCompat::ballPos)
            .def_readwrite("extra_hit_vel", &rc::BallHitInfoCompat::extraHitVel)
            .def_readwrite("tick_count_when_hit", &rc::BallHitInfoCompat::tickCountWhenHit)
            .def_readwrite("tick_count_when_extra_impulse_applied", &rc::BallHitInfoCompat::tickCountWhenExtraImpulseApplied);

        py::class_<rc::BallStateCompat>(c, "BallState", py::dynamic_attr())
            .def_readwrite("ball_hit_info", &rc::BallStateCompat::ballHitInfo)
            .def_readwrite("last_hit_car_id", &rc::BallStateCompat::lastHitCarID)

            .def_readwrite("tick_count_since_update", &rc::BallStateCompat::tickCountSinceUpdate)
            .def(py::init<>())
            .def_readwrite("pos", &rc::BallStateCompat::pos)
            .def_readwrite("rot_mat", &rc::BallStateCompat::rotMat)
            .def_readwrite("vel", &rc::BallStateCompat::vel)
            .def_readwrite("ang_vel", &rc::BallStateCompat::angVel);
        py::class_<rc::CarStateCompat> cs(c, "CarState");
        cs.def_readwrite("ball_hit_info", &rc::CarStateCompat::ballHitInfo);

        cs.def_readwrite("tick_count_since_update", &rc::CarStateCompat::tickCountSinceUpdate);
        cs.def(py::init<>())
            .def_readwrite("pos", &rc::CarStateCompat::pos)
            .def_readwrite("rot_mat", &rc::CarStateCompat::rotMat)
            .def_readwrite("vel", &rc::CarStateCompat::vel)
            .def_readwrite("ang_vel", &rc::CarStateCompat::angVel)
            .def_readwrite("is_on_ground", &rc::CarStateCompat::isOnGround)
            .def_property("wheels_with_contact",
                [](const rc::CarStateCompat& st) { return std::vector<bool>(st.wheelsWithContact, st.wheelsWithContact + 4); },
                [](rc::CarStateCompat& st, const std::vector<bool>& v) { for (size_t i = 0; i < 4 && i < v.size(); ++i) st.wheelsWithContact[i] = v[i]; })
            .def_readwrite("has_jumped", &rc::CarStateCompat::hasJumped)
            .def_readwrite("has_double_jumped", &rc::CarStateCompat::hasDoubleJumped)
            .def_readwrite("has_flipped", &rc::CarStateCompat::hasFlipped)
            .def_readwrite("flip_rel_torque", &rc::CarStateCompat::flipRelTorque)
            .def_readwrite("jump_time", &rc::CarStateCompat::jumpTime)
            .def_readwrite("flip_time", &rc::CarStateCompat::flipTime)
            .def_readwrite("is_flipping", &rc::CarStateCompat::isFlipping)
            .def_readwrite("is_jumping", &rc::CarStateCompat::isJumping)
            .def_readwrite("air_time", &rc::CarStateCompat::airTime)
            .def_readwrite("air_time_since_jump", &rc::CarStateCompat::airTimeSinceJump)
            .def_readwrite("boost", &rc::CarStateCompat::boost)
            .def_readwrite("time_since_boosted", &rc::CarStateCompat::timeSinceBoosted)
            .def_readwrite("is_boosting", &rc::CarStateCompat::isBoosting)
            .def_readwrite("boosting_time", &rc::CarStateCompat::boostingTime)
            .def_readwrite("is_supersonic", &rc::CarStateCompat::isSupersonic)
            .def_readwrite("supersonic_time", &rc::CarStateCompat::supersonicTime)
            .def_readwrite("handbrake_val", &rc::CarStateCompat::handbrakeVal)
            .def_readwrite("is_auto_flipping", &rc::CarStateCompat::isAutoFlipping)
            .def_readwrite("auto_flip_timer", &rc::CarStateCompat::autoFlipTimer)
            .def_readwrite("auto_flip_torque_scale", &rc::CarStateCompat::autoFlipTorqueScale)
            .def_property("has_world_contact",
                [](const rc::CarStateCompat& st) { return st.worldContact.hasContact; },
                [](rc::CarStateCompat& st, bool v) { st.worldContact.hasContact = v; })
            .def_property("world_contact_normal",
                [](const rc::CarStateCompat& st) { return st.worldContact.contactNormal; },
                [](rc::CarStateCompat& st, const rocketgpu::Vec3& v) { st.worldContact.contactNormal = v; })
            .def_property("car_contact_id",
                [](const rc::CarStateCompat& st) { return st.carContact.otherCarID; },
                [](rc::CarStateCompat& st, uint32_t v) { st.carContact.otherCarID = v; })
            .def_property("car_contact_cooldown_timer",
                [](const rc::CarStateCompat& st) { return st.carContact.cooldownTimer; },
                [](rc::CarStateCompat& st, float v) { st.carContact.cooldownTimer = v; })
            .def_readwrite("is_demoed", &rc::CarStateCompat::isDemoed)
            .def_readwrite("demo_respawn_timer", &rc::CarStateCompat::demoRespawnTimer)
            .def_readwrite("last_controls", &rc::CarStateCompat::lastControls)
            .def_readwrite("air_state", &rc::CarStateCompat::airState)
            .def_readwrite("dodge_timeout", &rc::CarStateCompat::dodgeTimeout)
            .def_readwrite("dodge_elapsed", &rc::CarStateCompat::dodgeElapsed)
            .def_readwrite("dodge_dir_x", &rc::CarStateCompat::dodgeDirX)
            .def_readwrite("dodge_dir_y", &rc::CarStateCompat::dodgeDirY)
            .def_property("latest_touch_tick",
                [](const rc::CarStateCompat& st) { return st.latestTouch.tick; },
                [](rc::CarStateCompat& st, uint64_t v) { st.latestTouch.tick = v; })
            .def_property("latest_touch_location",
                [](const rc::CarStateCompat& st) { return st.latestTouch.location; },
                [](rc::CarStateCompat& st, const rocketgpu::Vec3& v) { st.latestTouch.location = v; })
            .def_property("latest_touch_normal",
                [](const rc::CarStateCompat& st) { return st.latestTouch.normal; },
                [](rc::CarStateCompat& st, const rocketgpu::Vec3& v) { st.latestTouch.normal = v; });
        // RocketSim v3's CarState (rocketsim crate), field for field; None where v3 has an empty Option.
        py::class_<rc::CarStateV3>(c, "CarStateV3")
            .def(py::init<>())
            .def_readwrite("pos", &rc::CarStateV3::pos)
            .def_readwrite("rot_mat", &rc::CarStateV3::rotMat)
            .def_readwrite("vel", &rc::CarStateV3::vel)
            .def_readwrite("ang_vel", &rc::CarStateV3::angVel)
            .def_readwrite("controls", &rc::CarStateV3::controls)
            .def_readwrite("has_jumped", &rc::CarStateV3::hasJumped)
            .def_readwrite("has_double_jumped", &rc::CarStateV3::hasDoubleJumped)
            .def_readwrite("has_flipped", &rc::CarStateV3::hasFlipped)
            .def_readwrite("flip_rel_torque", &rc::CarStateV3::flipRelTorque)
            .def_readwrite("jump_ticks", &rc::CarStateV3::jumpTicks)
            .def_readwrite("flip_time", &rc::CarStateV3::flipTime)
            .def_readwrite("is_flipping", &rc::CarStateV3::isFlipping)
            .def_readwrite("is_jumping", &rc::CarStateV3::isJumping)
            .def_readwrite("jump_hold_broken", &rc::CarStateV3::jumpHoldBroken)
            .def_readwrite("air_time", &rc::CarStateV3::airTime)
            .def_readwrite("air_time_since_jump", &rc::CarStateV3::airTimeSinceJump)
            .def_readwrite("boost", &rc::CarStateV3::boost)
            .def_readwrite("handbrake_val", &rc::CarStateV3::handbrakeVal)
            .def_readwrite("is_supersonic", &rc::CarStateV3::isSupersonic)
            .def_readwrite("supersonic_grace_timer", &rc::CarStateV3::supersonicGraceTimer)
            .def_readwrite("time_since_boosted", &rc::CarStateV3::timeSinceBoosted)
            .def_readwrite("boosting_time", &rc::CarStateV3::boostingTime)
            .def_readwrite("is_boosting", &rc::CarStateV3::isBoosting)
            .def_readwrite("is_auto_flipping", &rc::CarStateV3::isAutoFlipping)
            .def_readwrite("auto_flip_timer", &rc::CarStateV3::autoFlipTimer)
            .def_readwrite("auto_flip_torque_scale", &rc::CarStateV3::autoFlipTorqueScale)
            .def_readwrite("bump_cooldown_timer", &rc::CarStateV3::bumpCooldownTimer)
            .def_property("bump_cooldown_victim",
                [](const rc::CarStateV3& st) -> py::object {
                    return st.bumpCooldownVictim ? py::object(py::int_(st.bumpCooldownVictim)) : py::object(py::none());
                },
                [](rc::CarStateV3& st, py::object v) { st.bumpCooldownVictim = v.is_none() ? 0u : v.cast<uint32_t>(); })
            .def_property("last_extra_hit_tick",
                [](const rc::CarStateV3& st) -> py::object {
                    return st.lastExtraHitTick != ~0ULL ? py::object(py::int_(st.lastExtraHitTick)) : py::object(py::none());
                },
                [](rc::CarStateV3& st, py::object v) { st.lastExtraHitTick = v.is_none() ? ~0ULL : v.cast<uint64_t>(); })
            .def_readwrite("is_demoed", &rc::CarStateV3::isDemoed)
            .def_readwrite("demo_respawn_timer", &rc::CarStateV3::demoRespawnTimer)
            .def_readwrite("is_on_ground", &rc::CarStateV3::isOnGround)
            .def_property("wheels_with_contact",
                [](const rc::CarStateV3& st) { return std::vector<bool>(st.wheelsWithContact, st.wheelsWithContact + 4); },
                [](rc::CarStateV3& st, const std::vector<bool>& v) { for (size_t i = 0; i < 4 && i < v.size(); ++i) st.wheelsWithContact[i] = v[i]; })
            .def_property("world_contact_normal",
                [](const rc::CarStateV3& st) -> py::object {
                    return st.hasWorldContact ? py::cast(st.worldContactNormal) : py::object(py::none());
                },
                [](rc::CarStateV3& st, py::object v) {
                    st.hasWorldContact = !v.is_none();
                    st.worldContactNormal = v.is_none() ? rocketgpu::Vec3(0.0f, 0.0f, 0.0f) : v.cast<rocketgpu::Vec3>();
                })
            .def_readwrite("prev_controls", &rc::CarStateV3::prevControls);
        py::class_<rc::Car, std::unique_ptr<rc::Car, py::nodelete>>(c, "Car")
            .def_readonly("id", &rc::Car::id)
            .def_readonly("team", &rc::Car::team)
            .def("get_state", &rc::Car::GetState)
            .def("set_state", &rc::Car::SetState)
            .def("get_state_v3", &rc::Car::GetStateV3)
            .def("set_state_v3", &rc::Car::SetStateV3)
            .def("set_controls", &rc::Car::SetControls);
        py::class_<rc::Ball, std::unique_ptr<rc::Ball, py::nodelete>>(c, "Ball")
            .def("get_state", &rc::Ball::GetState)
            .def("set_state", &rc::Ball::SetState);

        py::class_<rc::ArenaConfig>(c, "ArenaConfig", py::dynamic_attr())
            .def(py::init<>())
            .def_readwrite("no_ball_rot", &rc::ArenaConfig::noBallRot)
            .def_readwrite("noBallRot", &rc::ArenaConfig::noBallRot)

            .def_readwrite("native_sse_math", &rc::ArenaConfig::nativeSseMath)
            .def_readwrite("nativeSseMath", &rc::ArenaConfig::nativeSseMath)

            .def_readwrite("use_custom_boost_pads", &rc::ArenaConfig::useCustomBoostPads)
            .def_readwrite("custom_boost_pads", &rc::ArenaConfig::customBoostPads);
        py::class_<rc::BoostPadConfig>(c, "BoostPadConfig")
            .def(py::init<>())
            .def_readwrite("pos", &rc::BoostPadConfig::pos)
            .def_readwrite("is_big", &rc::BoostPadConfig::isBig);

        py::class_<rc::BoostPad>(c, "BoostPad")
            .def_readonly("config", &rc::BoostPad::config)
            .def("get_state", &rc::BoostPad::GetState)
            .def("set_state", &rc::BoostPad::SetState, py::arg("state"))
            .def("get_pos", &rc::BoostPad::GetPos)
            .def_property_readonly("is_big", &rc::BoostPad::IsBig)
            .def_property_readonly("index", &rc::BoostPad::GetIndex)
            .def_property_readonly("pad_state_index", &rc::BoostPad::GetPadStateIndex);
        py::class_<rc::Arena>(c, "Arena")
            .def(py::init([](rc::GameMode mode, float tick_rate) { return rc::Arena::Create(mode, tick_rate); }),
                 py::arg("game_mode") = rc::GameMode::SOCCAR, py::arg("tick_rate") = 120.0f)
            .def(py::init([](rc::GameMode mode, const rc::ArenaConfig& config, float tick_rate) {
                return rc::Arena::Create(mode, config, tick_rate);
            }), py::arg("game_mode"), py::arg("config"), py::arg("tick_rate") = 120.0f)
            .def("get_arena_config", &rc::Arena::GetArenaConfig, py::return_value_policy::copy)
            .def("add_car", &rc::Arena::AddCar, py::arg("team"), py::arg("config") = rocketgpu::CarConfig{},
                 py::return_value_policy::reference_internal)
            .def("remove_car", [](rc::Arena& a, rc::Car* car) { return a.RemoveCar(car); })
            .def("remove_car_by_id", [](rc::Arena& a, uint32_t id) { return a.RemoveCar(id); })
            .def("get_car", &rc::Arena::GetCar, py::return_value_policy::reference_internal)
            .def("get_cars", [](rc::Arena& a) { return a.GetCars(); }, py::return_value_policy::reference_internal)
            .def_property_readonly("ball", [](rc::Arena& a) { return a.ball; }, py::return_value_policy::reference_internal)
            .def("step", &rc::Arena::Step, py::arg("ticks") = 1)
            .def("reset_to_random_kickoff", &rc::Arena::ResetToRandomKickoff, py::arg("seed") = -1)
            .def("set_respawn_rng_seed", &rc::Arena::SetRespawnRngSeed, py::arg("seed"))
            .def("set_car_visit_order_mode", &rc::Arena::SetCarVisitOrderMode, py::arg("mode"))
            .def("set_car_visit_order", &rc::Arena::SetCarVisitOrder, py::arg("compat_ids_in_visit_order"))
            .def("get_car_visit_order_mode", &rc::Arena::GetCarVisitOrderMode)
            .def("is_ball_scored", &rc::Arena::IsBallScored)
            .def("set_mutator_config", &rc::Arena::SetMutatorConfig, py::arg("config"),
                 "RocketSim Arena::SetMutatorConfig: replace the mutators between steps")
            .def("get_mutator_config", &rc::Arena::GetMutatorConfig, py::return_value_policy::copy)
            .def("serialize", [](const rc::Arena& a) { std::vector<uint8_t> out; a.Serialize(out); return py::bytes(reinterpret_cast<const char*>(out.data()), out.size()); },
                 "RocketSim Arena::Serialize byte format (loads in RocketSim's DeserializeNew)")
            .def_static("deserialize_new", [](py::bytes data) { const std::string s = data; return rc::Arena::DeserializeNew(std::vector<uint8_t>(s.begin(), s.end())); },
                        py::arg("data"), py::return_value_policy::take_ownership)
            .def("serialize_to_file", &rc::Arena::SerializeToFile, py::arg("path"), py::arg("write_version_check") = true)
            .def_static("deserialize_new_from_file", &rc::Arena::DeserializeNewFromFile, py::arg("path"), py::arg("version_check") = true,
                        py::return_value_policy::take_ownership)
            .def("get_last_car_id", &rc::Arena::GetLastCarID)
            .def("is_ball_probably_going_in",
                 [](const rc::Arena& a, float max_time, float extra_margin) {
                     rocketgpu::Team team = rocketgpu::Team::Blue;
                     const bool going_in = a.IsBallProbablyGoingIn(max_time, extra_margin, &team);
                     return py::make_tuple(going_in, team);
                 },
                 py::arg("max_time") = 2.0f, py::arg("extra_margin") = 0.0f,
                 "RocketSim Arena::IsBallProbablyGoingIn -> (bool, scored_on_team)")
            .def("clone", [](const rc::Arena& a, bool copy_callbacks) { return a.Clone(copy_callbacks); }, py::arg("copy_callbacks") = false)
            .def("get_boost_pad_states", &rc::Arena::GetBoostPadStates)
            .def("get_boost_pads", &rc::Arena::GetBoostPads, py::return_value_policy::reference_internal)
            .def("set_boost_pad_states", &rc::Arena::SetBoostPadStates, py::arg("states"))
            .def("get_tick_rate", &rc::Arena::GetTickRate)
            .def("get_tick_count", &rc::Arena::GetTickCount)
            .def("set_goal_score_callback",
                 [](rc::Arena& a, py::function fn) {
                     a.SetGoalScoreCallback([fn](rc::Arena* arena, rocketgpu::Team team, void*) {
                         py::gil_scoped_acquire gil;
                         fn(py::cast(arena, py::return_value_policy::reference), team);
                     }, nullptr);
                 })
            .def("set_car_bump_callback",
                 [](rc::Arena& a, py::function fn) {
                     a.SetCarBumpCallback([fn](rc::Arena* arena, rc::Car* bumper, rc::Car* victim, bool is_demo, void*) {
                         py::gil_scoped_acquire gil;
                         fn(py::cast(arena, py::return_value_policy::reference),
                            py::cast(bumper, py::return_value_policy::reference),
                            py::cast(victim, py::return_value_policy::reference), is_demo);
                     }, nullptr);
                 });

        py::class_<rc::GameEventTrackerConfig>(c, "GameEventTrackerConfig")
            .def(py::init<>())
            .def_readwrite("shot_min_speed", &rc::GameEventTrackerConfig::shotMinSpeed)
            .def_readwrite("shot_touch_min_delay", &rc::GameEventTrackerConfig::shotTouchMinDelay)
            .def_readwrite("pred_score_extra_margin", &rc::GameEventTrackerConfig::predScoreExtraMargin)
            .def_readwrite("shot_event_cooldown", &rc::GameEventTrackerConfig::shotEventCooldown)
            .def_readwrite("shot_min_score_time", &rc::GameEventTrackerConfig::shotMinScoreTime)
            .def_readwrite("goal_max_touch_time", &rc::GameEventTrackerConfig::goalMaxTouchTime)
            .def_readwrite("pass_max_touch_time", &rc::GameEventTrackerConfig::passMaxTouchTime);
        py::class_<rc::GameEventTracker>(c, "GameEventTracker",
                                         "Port of RocketSim's GameEventTracker: shots, saves, goals with shooter/passer attribution")
            .def(py::init<>())
            .def_readwrite("config", &rc::GameEventTracker::config)
            .def_readwrite("auto_state_set_detection", &rc::GameEventTracker::autoStateSetDetection)
            .def("set_shot_callback",
                 [](rc::GameEventTracker& t, std::function<void(rc::Arena*, rc::Car*, rc::Car*)> fn) {
                     t.SetShotCallback([fn](rc::Arena* a, rc::Car* s, rc::Car* p, void*) { fn(a, s, p); }, nullptr);
                 }, py::arg("callback"), "callback(arena, shooter, passer_or_None)")
            .def("set_goal_callback",
                 [](rc::GameEventTracker& t, std::function<void(rc::Arena*, rc::Car*, rc::Car*)> fn) {
                     t.SetGoalCallback([fn](rc::Arena* a, rc::Car* s, rc::Car* p, void*) { fn(a, s, p); }, nullptr);
                 }, py::arg("callback"), "callback(arena, scorer, passer_or_None)")
            .def("set_save_callback",
                 [](rc::GameEventTracker& t, std::function<void(rc::Arena*, rc::Car*)> fn) {
                     t.SetSaveCallback([fn](rc::Arena* a, rc::Car* s, void*) { fn(a, s); }, nullptr);
                 }, py::arg("callback"), "callback(arena, saver)")
            .def("update", &rc::GameEventTracker::Update, py::arg("arena"))
            .def("reset_persistent_info", &rc::GameEventTracker::ResetPersistentInfo);
        py::class_<rc::BallPredTracker>(c, "BallPredTracker",
                                        "Port of RocketSim's BallPredTracker: predicts the ball num_pred_ticks ahead in a car-less copy of the arena")
            .def(py::init<rc::Arena*, size_t>(), py::arg("arena"), py::arg("num_pred_ticks"), py::keep_alive<1, 2>())
            .def_readonly("pred_data", &rc::BallPredTracker::predData)
            .def_readonly("num_pred_ticks", &rc::BallPredTracker::numPredTicks)
            .def_readonly("last_update_tick_count", &rc::BallPredTracker::lastUpdateTickCount)
            .def("update_pred_from_arena", &rc::BallPredTracker::UpdatePredFromArena, py::arg("arena"))
            .def("update_pred_manual", &rc::BallPredTracker::UpdatePredManual, py::arg("cur_ball_state"), py::arg("ticks_since_last_update"))
            .def("force_update_all_pred", &rc::BallPredTracker::ForceUpdateAllPred, py::arg("initial_ball_state"))
            .def("get_ball_state_for_time", &rc::BallPredTracker::GetBallStateForTime, py::arg("pred_time"));
        c.def("ball_state_matches", &rc::BallStateMatches, py::arg("a"), py::arg("b"), py::arg("margin_pos") = 0.8f,
              py::arg("margin_vel") = 0.4f, py::arg("margin_ang_vel") = 0.02f, "RocketSim BallState::Matches");

        c.def("rand_int_like_rocketsim",
              [](int min, int max, int seed) {
                  if (seed == -1) throw std::invalid_argument("rand_int_like_rocketsim: seed -1 is RocketSim's shared engine");
                  if (max <= min) throw std::invalid_argument("rand_int_like_rocketsim: max must exceed min");
                  std::default_random_engine engine(static_cast<std::default_random_engine::result_type>(seed));
                  return min + static_cast<int>(engine() % static_cast<unsigned>(max - min));
              },
              py::arg("min"), py::arg("max"), py::arg("seed"),
              "RocketSim Math::RandInt(min, max, seed) on this platform's std::default_random_engine (seed != -1)");
    }

    m.def("DominusConfig", &rocketgpu::DominusConfig);

    m.def("PlankConfig", &rocketgpu::PlankConfig);

    m.def("BreakoutConfig", &rocketgpu::BreakoutConfig);

    m.def("HybridConfig", &rocketgpu::HybridConfig);

    m.def("MercConfig", &rocketgpu::MercConfig);

    m.def("PsyclopsConfig", &rocketgpu::PsyclopsConfig);
}
