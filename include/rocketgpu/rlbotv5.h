#pragma once

#include "rocketgpu/rocketgpu.h"

// RLBot v5 values, bit for bit as RLBot sends them with one physics tick per frame. CarState holds the game's own
// values; where RLBot sends something else, the functions here give it. These CarState fields hold RLBot's values as
// they are: air_state, dodge_dir_x / dodge_dir_y and the latest touch's location and normal.
// Pass states as read from the simulator (download_car_states / download_ball_states; compat GetNativeState) and the
// simulator's own MutatorConfig (BatchSim::config().mutator; compat GetMutatorConfig). tick_dt: SimConfig.tick_dt.
namespace rocketgpu {

// RLBot's Physics. RLBot sends the game's network state of a body: the body after the speed caps, with the location
// rounded to 0.01 uu, the velocity to 0.01 uu/s, the angular velocity to 0.0001 rad/s and the rotation to pitch, yaw
// and roll in radians on a 65536-step turn. The state's own pos, vel, ang_vel and rot are not rounded.
struct RlbotV5Physics {
    Vec3 location = Vec3(0.0f, 0.0f, 0.0f);
    float pitch = 0.0f;
    float yaw = 0.0f;
    float roll = 0.0f;
    Vec3 velocity = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 angular_velocity = Vec3(0.0f, 0.0f, 0.0f);
};

// PlayerInfo.physics.
RlbotV5Physics RlbotV5CarPhysics(const CarState& car, const MutatorConfig& mutator);

// BallInfo.physics.
RlbotV5Physics RlbotV5BallPhysics(const BallState& ball, const MutatorConfig& mutator);

// PlayerInfo.is_supersonic: the speed RLBot sends (physics.velocity: after the cap, rounded) is at least
// supersonic_start_speed on this tick. CarState's own is_supersonic is the game's flag, which holds through
// supersonic_time.
bool RlbotV5IsSupersonic(const CarState& car, const MutatorConfig& mutator);

// PlayerInfo.has_jumped, has_double_jumped and has_dodged. RLBot's has_jumped clears on a tick that ends OnGround and
// started more than 0.05 s after the jump did, and a double jump or dodge with no jump before it leaves it clear; its
// has_double_jumped and has_dodged clear on the tick air_state reads OnGround; has_dodged (and dodge_elapsed with it)
// stays through a landing that goes straight into a jump. CarState's own has_jumped, has_double_jumped and has_flipped
// are the game's flags.
ROCKETGPU_HD bool RlbotV5HasJumped(const CarState& car) {
    return (car.reported_flags & CarState::kReportedValid) ? car.game_has_jumped != 0 : car.has_jumped != 0;
}
ROCKETGPU_HD bool RlbotV5HasDoubleJumped(const CarState& car) {
    return (car.reported_flags & CarState::kReportedValid) ? car.internal_has_double_jumped && car.air_state != 0
                                                           : car.has_double_jumped != 0;
}
ROCKETGPU_HD bool RlbotV5HasDodged(const CarState& car) {
    return (car.reported_flags & CarState::kReportedValid) ? car.rlbotv5_dodged != 0 : car.has_flipped != 0;
}

namespace detail {
// vel and ang_vel after the game's speed caps, not rounded.
void CarVelAfterCaps(const CarState& car, const MutatorConfig& mutator, Vec3& vel, Vec3& ang_vel);
}

ROCKETGPU_HD double RlbotV5TickRate(float tick_dt) {
    return static_cast<double>(static_cast<long long>(1.0 / static_cast<double>(tick_dt) + 0.5));
}

// MatchInfo.seconds_elapsed on the env's own clock: EnvState.tick in seconds.
ROCKETGPU_HD float RlbotV5SecondsElapsed(const EnvState& env, float tick_dt = 1.0f / 120.0f) {
    return static_cast<float>(static_cast<double>(env.tick) / RlbotV5TickRate(tick_dt));
}

// Touch.game_seconds of the car's latest ball touch: RlbotV5SecondsElapsed of the state the touch first shows in. -1
// before the car's first touch.
ROCKETGPU_HD float RlbotV5TouchSeconds(const CarState& car, float tick_dt = 1.0f / 120.0f) {
    if (car.latest_touch_tick == ~0ULL) {
        return -1.0f;
    }
    return static_cast<float>(static_cast<double>(car.latest_touch_tick + 1) / RlbotV5TickRate(tick_dt));
}

// PlayerInfo.demolished_timeout: -1 while the car is not demolished, 0 on the tick it is demolished, then
// demo_respawn_time (MutatorConfig) minus the seconds since that tick, and -1 again on the last tick before the car is
// back: RLBot shows the new car one tick before it is simulated.
ROCKETGPU_HD float RlbotV5DemolishedTimeout(const CarState& car, float demo_respawn_time = 3.0f,
                                         float tick_dt = 1.0f / 120.0f) {
    const bool kickoff_spawn = car.respawn_slot_override_plus1 >= kKickoffSpawnOverridePlus1 &&
                               car.respawn_slot_override_plus1 < kKickoffSpawnOverridePlus1 + 5;
    if (!car.is_demoed || kickoff_spawn || !(car.demo_respawn_timer - tick_dt > 0.0f)) {
        return -1.0f;
    }
    const long long ticks =
        static_cast<long long>((demo_respawn_time + tick_dt - car.demo_respawn_timer) / tick_dt + 0.5f);
    if (ticks <= 0) {
        return 0.0f;
    }
    const float left = demo_respawn_time - static_cast<float>(static_cast<double>(ticks) / RlbotV5TickRate(tick_dt));
    return left > 0.0f ? left : 0.0f;
}

// PlayerInfo.dodge_timeout: CarState.dodge_timeout on RLBot's tick grid (doublejump_max_delay: MutatorConfig).
ROCKETGPU_HD float RlbotV5DodgeTimeout(const CarState& car, float doublejump_max_delay = 1.25f,
                                    float tick_dt = 1.0f / 120.0f) {
    if (car.dodge_timeout < 0.0f) {
        return -1.0f;
    }
    const long long ticks = static_cast<long long>((doublejump_max_delay - car.dodge_timeout) / tick_dt + 0.5f);
    return doublejump_max_delay - static_cast<float>(static_cast<double>(ticks) / RlbotV5TickRate(tick_dt));
}

// PlayerInfo.dodge_elapsed: CarState.dodge_elapsed on RLBot's tick grid.
ROCKETGPU_HD float RlbotV5DodgeElapsed(const CarState& car, float tick_dt = 1.0f / 120.0f) {
    const long long ticks = static_cast<long long>(car.dodge_elapsed / tick_dt + 0.5f);
    return static_cast<float>(static_cast<double>(ticks) / RlbotV5TickRate(tick_dt));
}

// PlayerInfo.boost. RLBot counts its own value from the game's whole-number boost, tick by tick: keep one RlbotV5Boost
// per car and call Update with the car's state after every tick (step(1)). A new RlbotV5Boost is a newly spawned car;
// start one when you reset or write the car. tick_dt: SimConfig.tick_dt.
struct RlbotV5Boost {
    float boost = 100.0f / 3.0f;  // PlayerInfo.boost as of the last Update
    int32_t whole = 34;            // the whole-number boost at the last Update

    ROCKETGPU_HD float Update(const CarState& car, float tick_dt = 1.0f / 120.0f) {
        if (car.is_demoed && car.respawn_slot_override_plus1 >= kKickoffSpawnOverridePlus1 &&
            car.respawn_slot_override_plus1 < kKickoffSpawnOverridePlus1 + 5) {
            *this = RlbotV5Boost{};
            return boost;
        }
        // A demolished car has no boost; on the last tick before it is back RLBot shows the new car's.
        int now = 0;
        if (!car.is_demoed || !(car.demo_respawn_timer - tick_dt > 0.0f)) {
            now = static_cast<int>(car.boost);
            now += static_cast<float>(now) < car.boost ? 1 : 0;
            // A pad that came back under the car gave its boost after RLBot read the car: it shows on the next tick.
            if (car.rlbotv5_boost_hold != 0 && !car.is_demoed) {
                now = car.rlbotv5_boost_hold - 1;
            }
        }
        if (now < whole) {
            boost = static_cast<float>(now);
        } else {
            if (now > whole) {
                boost = now >= 100 ? 100.0f : boost + static_cast<float>(now - whole);
            }
            if (car.is_boosting && !car.is_demoed) {
                const float counted = boost - 100.0f / 360.0f;
                const float least = now > 0 ? static_cast<float>(now - 1) : 0.0f;
                boost = counted > least ? counted : least;
            } else if (now >= 100) {
                boost = 100.0f;
            }
        }
        whole = now;
        return boost;
    }
};

}  // namespace rocketgpu
