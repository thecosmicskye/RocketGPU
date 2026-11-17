#include "shim.h"

#include <cstring>
#include <exception>
#include <map>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <rocketgpu/rocketsim_compat.h>
#if defined(ROCKETGPU_USE_CUDA)
#include <cuda_runtime.h>
#endif

namespace rc = rocketgpu::compat;
using rocketgpu::Vec3;

static_assert(sizeof(RgControls) == 24, "RgControls");
static_assert(sizeof(RgCarState) == 280, "RgCarState");
static_assert(sizeof(RgBall) == 80, "RgBall");
static_assert(sizeof(RgCarConfig) == 124, "RgCarConfig");
static_assert(sizeof(RgMutators) == 104, "RgMutators");
static_assert(sizeof(RgPad) == 16, "RgPad");
static_assert(sizeof(RgEvent) == 44, "RgEvent");

constexpr uint32_t kEventRecords = 256;

template <typename T>
T* EventAlloc(size_t n) {
    void* p = nullptr;
#if defined(ROCKETGPU_USE_CUDA)
    if (cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped) != cudaSuccess) {
        throw std::runtime_error("cudaHostAlloc: physics event buffer");
    }
#else
    p = ::operator new(n * sizeof(T));
#endif
    T* t = static_cast<T*>(p);
    for (size_t i = 0; i < n; ++i) new (t + i) T{};
    return t;
}

void EventFree(void* p) {
    if (!p) return;
#if defined(ROCKETGPU_USE_CUDA)
    cudaFreeHost(p);
#else
    ::operator delete(p);
#endif
}

template <typename T>
T* EventDevicePointer(T* p) {
#if defined(ROCKETGPU_USE_CUDA)
    void* d = nullptr;
    if (cudaHostGetDevicePointer(&d, p, 0) != cudaSuccess) {
        throw std::runtime_error("cudaHostGetDevicePointer: physics event buffer");
    }
    return static_cast<T*>(d);
#else
    return p;
#endif
}

struct RgArena {
    rc::Arena* arena = nullptr;
    uint32_t slot_ids[8] = {};
    std::vector<int> pad_state_index;
    rocketgpu::PhysicsEvent* records = nullptr;
    rocketgpu::PhysicsEventWindow* window = nullptr;
    std::vector<RgEvent> events;

    RgArena() = default;
    RgArena(const RgArena&) = delete;
    RgArena& operator=(const RgArena&) = delete;
    ~RgArena() {
        delete arena;
        EventFree(records);
        EventFree(window);
    }

    void Attach() {
        pad_state_index.clear();
        for (const rc::BoostPad* pad : arena->GetBoostPads()) pad_state_index.push_back(pad->GetPadStateIndex());
        if (!records) records = EventAlloc<rocketgpu::PhysicsEvent>(kEventRecords);
        if (!window) window = EventAlloc<rocketgpu::PhysicsEventWindow>(1);
        rocketgpu::PhysicsEventStream stream;
        stream.records = EventDevicePointer(records);
        stream.windows = EventDevicePointer(window);
        stream.capacity_per_env = kEventRecords;
        stream.env_count = 1;
        stream.cars_per_env = static_cast<uint32_t>(arena->sim().car_capacity());
        arena->sim().set_physics_event_stream(stream);
    }

    rc::Car* Car(uint32_t id) const {
        rc::Car* car = arena->GetCar(id);
        if (!car) throw std::runtime_error("no car with id " + std::to_string(id));
        return car;
    }

    uint32_t IdOfDevice(uint32_t device_id) const { return device_id >= 1 && device_id <= 8 ? slot_ids[device_id - 1] : 0; }
};

namespace {
thread_local std::string g_error;

template <typename F>
int Guard(F&& f) {
    try {
        f();
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
    } catch (...) {
        g_error = "unknown error";
    }
    return 1;
}

void Put(float* out, const Vec3& v) {
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
}
Vec3 Get(const float* in) { return Vec3(in[0], in[1], in[2]); }

void PutRot(float* out, const rc::RotMat& m) {
    Put(out, m.forward);
    Put(out + 3, m.right);
    Put(out + 6, m.up);
}
rc::RotMat GetRot(const float* in) {
    rc::RotMat m;
    m.forward = Get(in);
    m.right = Get(in + 3);
    m.up = Get(in + 6);
    return m;
}

RgControls ToC(const rocketgpu::CarControls& c) {
    return RgControls{c.throttle, c.steer, c.pitch, c.yaw, c.roll, c.jump, c.boost, c.handbrake, 0};
}
rocketgpu::CarControls FromC(const RgControls& c) {
    rocketgpu::CarControls out;
    out.throttle = c.throttle;
    out.steer = c.steer;
    out.pitch = c.pitch;
    out.yaw = c.yaw;
    out.roll = c.roll;
    out.jump = c.jump ? 1 : 0;
    out.boost = c.boost ? 1 : 0;
    out.handbrake = c.handbrake ? 1 : 0;
    return out;
}

template <typename T>
std::vector<uint8_t> Bytes(const T& v) {
    std::vector<uint8_t> out(sizeof(T));
    std::memcpy(out.data(), &v, sizeof(T));
    return out;
}
}

const char* rg_last_error() { return g_error.c_str(); }

int rg_arena_new(uint32_t mode, uint32_t no_ball_rot, const RgPad* custom_pads, int32_t custom_pad_count, RgArena** out) {
    return Guard([&] {
        rc::ArenaConfig config;
        config.noBallRot = no_ball_rot != 0;
        if (custom_pad_count >= 0) {
            config.useCustomBoostPads = true;
            for (int32_t i = 0; i < custom_pad_count; ++i) {
                config.customBoostPads.push_back(rc::BoostPadConfig{Get(custom_pads[i].pos), custom_pads[i].is_big != 0});
            }
        }
        auto a = std::make_unique<RgArena>();
        a->arena = rc::Arena::Create(static_cast<rc::GameMode>(mode), config);
        a->Attach();
        *out = a.release();
    });
}

int rg_arena_clone(const RgArena* arena, RgArena** out) {
    return Guard([&] {
        auto a = std::make_unique<RgArena>();
        a->arena = arena->arena->Clone(false);
        std::memcpy(a->slot_ids, arena->slot_ids, sizeof(a->slot_ids));
        a->Attach();
        *out = a.release();
    });
}

void rg_arena_free(RgArena* arena) { delete arena; }

int rg_add_car(RgArena* arena, uint32_t team, const RgCarConfig* config, uint32_t* id_out) {
    return Guard([&] {
        rocketgpu::CarConfig c;
        c.hitbox_size = Get(config->hitbox_size);
        c.hitbox_published = Get(config->hitbox_published);
        c.hitbox_offset = Get(config->hitbox_offset);
        c.front_wheels_offset = Get(config->front_wheels_offset);
        c.back_wheels_offset = Get(config->back_wheels_offset);
        c.front_wheel_radius = config->front_wheel_radius;
        c.back_wheel_radius = config->back_wheel_radius;
        c.front_suspension_rest = config->front_suspension_rest;
        c.back_suspension_rest = config->back_suspension_rest;
        c.dodge_deadzone = config->dodge_deadzone;
        std::memcpy(c.steer_angle_curve_deg, config->steer_angle_curve_deg, sizeof(c.steer_angle_curve_deg));
        std::memcpy(c.powerslide_steer_angle_curve_deg, config->powerslide_steer_angle_curve_deg,
                    sizeof(c.powerslide_steer_angle_curve_deg));
        c.spawn_z = config->spawn_z;
        c.body = static_cast<uint16_t>(config->body);
        c.three_wheels = config->three_wheels ? 1 : 0;
        int slot = 0;
        while (slot < 8 && arena->slot_ids[slot]) ++slot;
        const rc::Car* car = arena->arena->AddCar(team ? rocketgpu::Team::Orange : rocketgpu::Team::Blue, c);
        arena->slot_ids[slot] = car->id;
        *id_out = car->id;
    });
}

int rg_remove_car(RgArena* arena, uint32_t id) {
    return Guard([&] {
        if (!arena->arena->RemoveCar(id)) throw std::runtime_error("no car with id " + std::to_string(id));
        for (uint32_t& s : arena->slot_ids) {
            if (s == id) s = 0;
        }
    });
}

int rg_step(RgArena* arena, uint32_t* event_count_out) {
    return Guard([&] {
        arena->window->count = 0;
        arena->window->overflow = 0;
        arena->window->tick = arena->arena->GetTickCount();
        arena->arena->Step(1);
#if defined(ROCKETGPU_USE_CUDA)
        if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("cudaDeviceSynchronize after the step");
#endif
        arena->events.clear();
        for (uint32_t i = 0; i < arena->window->count; ++i) {
            const rocketgpu::PhysicsEvent& e = arena->records[i];
            RgEvent out{};
            out.type = static_cast<uint32_t>(e.type);
            out.car_id = arena->IdOfDevice(e.car_id);
            out.other_car_id = arena->IdOfDevice(e.other_car_id);
            out.is_demo = e.is_demo;
            for (size_t p = 0; p < arena->pad_state_index.size(); ++p) {
                if (arena->pad_state_index[p] == static_cast<int>(e.pad_index)) out.pad_index = static_cast<uint32_t>(p);
            }
            Put(out.position, e.position);
            Put(out.normal, e.normal);
            arena->events.push_back(out);
        }
        *event_count_out = static_cast<uint32_t>(arena->events.size());
    });
}

int rg_events(const RgArena* arena, RgEvent* out, uint32_t capacity) {
    return Guard([&] {
        if (capacity < arena->events.size()) throw std::runtime_error("event buffer too small");
        if (!arena->events.empty()) std::memcpy(out, arena->events.data(), arena->events.size() * sizeof(RgEvent));
    });
}

uint64_t rg_tick_count(const RgArena* arena) { return arena->arena->GetTickCount(); }

int rg_is_ball_scored(const RgArena* arena, uint32_t* out) {
    return Guard([&] { *out = arena->arena->IsBallScored() ? 1 : 0; });
}

int rg_reset_kickoff(RgArena* arena, int32_t seed, uint32_t match_next) {
    return Guard([&] {
        if (match_next) {
            arena->arena->ResetToKickoff();
        } else {
            arena->arena->ResetToRandomKickoff(seed);
        }
    });
}

int rg_get_car(const RgArena* arena, uint32_t id, RgCarState* out) {
    return Guard([&] {
        const rc::CarStateV3 s = arena->Car(id)->GetStateV3();
        RgCarState o{};
        o.last_extra_hit_tick = s.lastExtraHitTick;
        Put(o.pos, s.pos);
        PutRot(o.rot, s.rotMat);
        Put(o.vel, s.vel);
        Put(o.ang_vel, s.angVel);
        Put(o.flip_rel_torque, s.flipRelTorque);
        Put(o.world_contact_normal, s.worldContactNormal);
        for (int w = 0; w < 4; ++w) Put(o.wheel_normals + 3 * w, s.wheelContactNormals[w]);
        o.controls = ToC(s.controls);
        o.prev_controls = ToC(s.prevControls);
        o.jump_ticks = s.jumpTicks;
        o.bump_cooldown_victim = s.bumpCooldownVictim;
        o.flip_time = s.flipTime;
        o.air_time = s.airTime;
        o.air_time_since_jump = s.airTimeSinceJump;
        o.boost = s.boost;
        o.handbrake_val = s.handbrakeVal;
        o.supersonic_grace_timer = s.supersonicGraceTimer;
        o.time_since_boosted = s.timeSinceBoosted;
        o.boosting_time = s.boostingTime;
        o.auto_flip_timer = s.autoFlipTimer;
        o.auto_flip_torque_scale = s.autoFlipTorqueScale;
        o.bump_cooldown_timer = s.bumpCooldownTimer;
        o.demo_respawn_timer = s.demoRespawnTimer;
        o.has_jumped = s.hasJumped;
        o.has_double_jumped = s.hasDoubleJumped;
        o.has_flipped = s.hasFlipped;
        o.is_flipping = s.isFlipping;
        o.is_jumping = s.isJumping;
        o.jump_hold_broken = s.jumpHoldBroken;
        o.is_supersonic = s.isSupersonic;
        o.is_boosting = s.isBoosting;
        o.is_auto_flipping = s.isAutoFlipping;
        o.is_demoed = s.isDemoed;
        o.is_on_ground = s.isOnGround;
        for (int w = 0; w < 4; ++w) o.wheels[w] = s.wheelsWithContact[w];
        o.has_world_contact = s.hasWorldContact;
        o.has_last_extra_hit_tick = s.hasLastExtraHitTick;
        o.has_prev_controls = s.hasPrevControls;
        o.has_wheel_normals = s.hasWheelContactNormals;
        *out = o;
    });
}

int rg_set_car(RgArena* arena, uint32_t id, const RgCarState* state) {
    return Guard([&] {
        const RgCarState& i = *state;
        rc::CarStateV3 s;
        s.lastExtraHitTick = i.last_extra_hit_tick;
        s.pos = Get(i.pos);
        s.rotMat = GetRot(i.rot);
        s.vel = Get(i.vel);
        s.angVel = Get(i.ang_vel);
        s.flipRelTorque = Get(i.flip_rel_torque);
        s.worldContactNormal = Get(i.world_contact_normal);
        for (int w = 0; w < 4; ++w) s.wheelContactNormals[w] = Get(i.wheel_normals + 3 * w);
        s.controls = FromC(i.controls);
        s.prevControls = FromC(i.prev_controls);
        s.jumpTicks = i.jump_ticks;
        s.bumpCooldownVictim = i.bump_cooldown_victim;
        s.flipTime = i.flip_time;
        s.airTime = i.air_time;
        s.airTimeSinceJump = i.air_time_since_jump;
        s.boost = i.boost;
        s.handbrakeVal = i.handbrake_val;
        s.supersonicGraceTimer = i.supersonic_grace_timer;
        s.timeSinceBoosted = i.time_since_boosted;
        s.boostingTime = i.boosting_time;
        s.autoFlipTimer = i.auto_flip_timer;
        s.autoFlipTorqueScale = i.auto_flip_torque_scale;
        s.bumpCooldownTimer = i.bump_cooldown_timer;
        s.demoRespawnTimer = i.demo_respawn_timer;
        s.hasJumped = i.has_jumped != 0;
        s.hasDoubleJumped = i.has_double_jumped != 0;
        s.hasFlipped = i.has_flipped != 0;
        s.isFlipping = i.is_flipping != 0;
        s.isJumping = i.is_jumping != 0;
        s.jumpHoldBroken = i.jump_hold_broken != 0;
        s.isSupersonic = i.is_supersonic != 0;
        s.isBoosting = i.is_boosting != 0;
        s.isAutoFlipping = i.is_auto_flipping != 0;
        s.isDemoed = i.is_demoed != 0;
        s.isOnGround = i.is_on_ground != 0;
        for (int w = 0; w < 4; ++w) s.wheelsWithContact[w] = i.wheels[w] != 0;
        s.hasWorldContact = i.has_world_contact != 0;
        s.hasLastExtraHitTick = i.has_last_extra_hit_tick != 0;
        s.hasPrevControls = i.has_prev_controls != 0;
        s.hasWheelContactNormals = i.has_wheel_normals != 0;
        arena->Car(id)->SetStateV3(s);
    });
}

int rg_set_controls(RgArena* arena, uint32_t id, const RgControls* controls) {
    return Guard([&] { arena->Car(id)->SetControls(FromC(*controls)); });
}

int rg_get_ball(const RgArena* arena, RgBall* out) {
    return Guard([&] {
        const rc::BallStateCompat s = arena->arena->ball->GetState();
        const rocketgpu::BallState native = arena->arena->ball->GetNativeState();
        RgBall o{};
        Put(o.pos, s.pos);
        PutRot(o.rot, s.rotMat);
        Put(o.vel, s.vel);
        Put(o.ang_vel, s.angVel);
        o.deactivation_time = native.deactivation_time;
        o.sleeping = (native.sleeping & 3u) == 1u ? 1 : 0;
        *out = o;
    });
}

int rg_set_ball(RgArena* arena, const RgBall* ball, uint32_t activation) {
    return Guard([&] {
        rc::BallStateCompat s;
        s.pos = Get(ball->pos);
        s.rotMat = GetRot(ball->rot);
        s.vel = Get(ball->vel);
        s.angVel = Get(ball->ang_vel);
        arena->arena->ball->SetState(s);
        if (activation) {
            const uint32_t sleeping = activation == 2 ? 1 : 0;
            arena->arena->SetFullState({{"ball_states.sleeping", Bytes(sleeping)},
                                        {"ball_states.deactivation_time", Bytes(ball->deactivation_time)}});
        }
    });
}

uint32_t rg_pad_count(const RgArena* arena) { return static_cast<uint32_t>(arena->pad_state_index.size()); }

int rg_pad_config(const RgArena* arena, uint32_t index, RgPad* out) {
    return Guard([&] {
        const rc::BoostPad* pad = arena->arena->GetBoostPads().at(index);
        Put(out->pos, pad->config.pos);
        out->is_big = pad->config.isBig ? 1 : 0;
    });
}

int rg_get_pad_cooldowns(const RgArena* arena, float* out, uint32_t count) {
    return Guard([&] {
        const std::vector<rocketgpu::BoostPadState> states = arena->arena->GetBoostPadStates();
        for (uint32_t i = 0; i < count; ++i) out[i] = states.at(static_cast<size_t>(arena->pad_state_index.at(i))).cooldown;
    });
}

int rg_set_pad_cooldown(RgArena* arena, uint32_t index, float cooldown) {
    return Guard([&] {
        std::vector<rocketgpu::BoostPadState> states = arena->arena->GetBoostPadStates();
        rocketgpu::BoostPadState& s = states.at(static_cast<size_t>(arena->pad_state_index.at(index)));
        s.cooldown = cooldown;
        s.is_active = cooldown <= 0.0f ? 1 : 0;
        arena->arena->SetBoostPadStates(states);
    });
}

int rg_get_mutators(const RgArena* arena, RgMutators* out) {
    return Guard([&] {
        const rocketgpu::MutatorConfig& m = arena->arena->GetMutatorConfig();
        *out = RgMutators{m.gravity_z, m.car_mass, m.ball_mass, m.ball_max_speed, m.ball_drag, m.jump_accel,
                          m.jump_immediate_speed, m.boost_accel_ground, m.boost_accel_air, m.boost_used_per_second,
                          m.demo_respawn_time, m.bump_cooldown_time, m.boost_max, m.boost_spawn_amount,
                          m.boost_pad_cooldown_big, m.boost_pad_cooldown_small, m.ball_hit_extra_force_scale,
                          m.bump_force_scale, m.ball_radius, m.recharge_boost_per_second, m.recharge_boost_delay,
                          m.goal_base_threshold_y, m.unlimited_flips, m.unlimited_double_jumps,
                          m.recharge_boost_enabled, m.enable_team_demos};
    });
}

int rg_set_mutators(RgArena* arena, const RgMutators* mutators) {
    return Guard([&] {
        const RgMutators& i = *mutators;
        rocketgpu::MutatorConfig m = arena->arena->GetMutatorConfig();
        m.gravity_z = i.gravity_z;
        m.car_mass = i.car_mass;
        m.ball_mass = i.ball_mass;
        m.ball_max_speed = i.ball_max_speed;
        m.ball_drag = i.ball_drag;
        m.jump_accel = i.jump_accel;
        m.jump_immediate_speed = i.jump_immediate_speed;
        m.boost_accel_ground = i.boost_accel_ground;
        m.boost_accel_air = i.boost_accel_air;
        m.boost_used_per_second = i.boost_used_per_second;
        m.demo_respawn_time = i.demo_respawn_time;
        m.bump_cooldown_time = i.bump_cooldown_time;
        m.boost_max = i.boost_max;
        m.boost_spawn_amount = i.boost_spawn_amount;
        m.boost_pad_cooldown_big = i.boost_pad_cooldown_big;
        m.boost_pad_cooldown_small = i.boost_pad_cooldown_small;
        m.ball_hit_extra_force_scale = i.ball_hit_extra_force_scale;
        m.bump_force_scale = i.bump_force_scale;
        m.ball_radius = i.ball_radius;
        m.recharge_boost_per_second = i.recharge_boost_per_second;
        m.recharge_boost_delay = i.recharge_boost_delay;
        m.goal_base_threshold_y = i.goal_base_threshold_y;
        m.unlimited_flips = i.unlimited_flips ? 1 : 0;
        m.unlimited_double_jumps = i.unlimited_double_jumps ? 1 : 0;
        m.recharge_boost_enabled = i.recharge_boost_enabled ? 1 : 0;
        m.enable_team_demos = i.enable_team_demos ? 1 : 0;
        arena->arena->SetMutatorConfig(m);
    });
}
