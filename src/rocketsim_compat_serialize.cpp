

#include <rocketgpu/rocketsim_compat.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace rocketgpu {
namespace compat {
namespace {
constexpr char kRocketSimVersion[] = "2.2.1";

constexpr uint32_t RocketSimVersionId() {
    uint32_t result = 0;
    for (size_t i = 0; i < sizeof(kRocketSimVersion); ++i) {
        const int v = kRocketSimVersion[i] - '0' + 1;
        result = static_cast<uint32_t>(v > 0 ? v : 0) + result * 10u;
    }
    return result;
}

struct Out {
    std::vector<uint8_t>& d;
    void Bytes(const void* p, size_t n) { const uint8_t* b = static_cast<const uint8_t*>(p); d.insert(d.end(), b, b + n); }
    template <class T> void Raw(const T& v) { Bytes(&v, sizeof(T)); }
    void U8(uint8_t v) { Raw(v); }
    void Bool(bool v) { U8(v ? 1u : 0u); }
    void U16(uint16_t v) { Raw(v); }
    void U32(uint32_t v) { Raw(v); }
    void U64(uint64_t v) { Raw(v); }
    void I32(int32_t v) { Raw(v); }
    void F(float v) { Raw(v); }
    void Vec(const Vec3& v) { F(v.x); F(v.y); F(v.z); F(0.0f); }
    void Rot(const RotMat& m) { Vec(m.forward); Vec(m.right); Vec(m.up); }
    void Count(uint32_t n) { U32(n); }
};

struct In {
    const std::vector<uint8_t>& d;
    size_t pos = 0;
    void Bytes(void* p, size_t n) {
        if (pos + n > d.size()) throw std::runtime_error("rocketgpu::compat::Arena::DeserializeNew: truncated data");
        std::memcpy(p, d.data() + pos, n);
        pos += n;
    }
    template <class T> T Raw() { T v{}; Bytes(&v, sizeof(T)); return v; }
    uint8_t U8() { return Raw<uint8_t>(); }
    bool Bool() { return U8() != 0; }
    uint16_t U16() { return Raw<uint16_t>(); }
    uint32_t U32() { return Raw<uint32_t>(); }
    uint64_t U64() { return Raw<uint64_t>(); }
    int32_t I32() { return Raw<int32_t>(); }
    float F() { return Raw<float>(); }
    Vec3 Vec() { const float x = F(), y = F(), z = F(); (void)F(); return Vec3(x, y, z); }
    RotMat Rot() { RotMat m; m.forward = Vec(); m.right = Vec(); m.up = Vec(); return m; }
    void Count(uint32_t expected, const char* what) {
        const uint32_t n = U32();
        if (n != expected) {
            throw std::runtime_error(std::string("rocketgpu::compat::Arena::DeserializeNew: ") + what + " field count " +
                                     std::to_string(n) + " != " + std::to_string(expected));
        }
    }
};

void WriteControlsRaw(Out& o, const CarControls& c) {
    o.F(c.throttle); o.F(c.steer); o.F(c.pitch); o.F(c.yaw); o.F(c.roll);
    o.U8(c.jump); o.U8(c.boost); o.U8(c.handbrake); o.U8(0);
}
CarControls ReadControlsRaw(In& i) {
    CarControls c;
    c.throttle = i.F(); c.steer = i.F(); c.pitch = i.F(); c.yaw = i.F(); c.roll = i.F();
    c.jump = i.U8(); c.boost = i.U8(); c.handbrake = i.U8(); (void)i.U8();
    return c;
}

int RocketGpuPadIndexFromOfficial(int official) { return official < 6 ? 28 + official : official - 6; }
}

struct ArenaSerializer {
    static void Write(const Arena& a, std::vector<uint8_t>& out) {
        Out o{out};
        a.PullStates();
        const SimConfig& cfg = a.sim_->config();
        const MutatorConfig& m = cfg.mutator;

        o.Count(4);
        o.U8(static_cast<uint8_t>(a.mode_));
        o.F(1.0f / a.tick_rate_);
        o.U64(a.tick_count_);
        o.U32(a.next_car_id_ - 1);

        o.Count(5);
        o.Vec(Vec3(-5600.0f, -6000.0f, 0.0f));
        o.Vec(Vec3(5600.0f, 6000.0f, 2200.0f));
        o.F(370.0f);
        o.Bool(a.arena_config_.noBallRot);
        o.Bool(true);
        o.Bool(false);

        o.U32(static_cast<uint32_t>(a.cars.size()));
        for (const Car* car : a.cars) {
            o.U8(static_cast<uint8_t>(car->team));
            o.U32(car->id);
            const CarControls& c = a.controls_[static_cast<size_t>(car->slot_)];
            o.Count(8);
            o.F(c.throttle); o.F(c.steer); o.F(c.pitch); o.F(c.yaw); o.F(c.roll);
            o.Bool(c.boost != 0); o.Bool(c.jump != 0); o.Bool(c.handbrake != 0);
            const CarConfig& k = a.configs_[static_cast<size_t>(car->slot_)];
            o.Count(9);
            o.F(k.dodge_deadzone); o.Vec(k.hitbox_offset); o.Vec(k.hitbox_size);
            o.Vec(k.front_wheels_offset); o.F(k.front_suspension_rest); o.F(k.front_wheel_radius);
            o.Vec(k.back_wheels_offset); o.F(k.back_suspension_rest); o.F(k.back_wheel_radius);
            const CarStateCompat s = car->GetState();
            o.Bool(s.ballHitInfo.isValid);
            if (s.ballHitInfo.isValid) {
                o.Count(5);
                o.Vec(s.ballHitInfo.relativePosOnBall); o.Vec(s.ballHitInfo.ballPos); o.Vec(s.ballHitInfo.extraHitVel);
                o.U64(s.ballHitInfo.tickCountWhenHit); o.U64(s.ballHitInfo.tickCountWhenExtraImpulseApplied);
            }
            o.Count(30);
            o.Vec(s.pos); o.Rot(s.rotMat); o.Vec(s.vel); o.Vec(s.angVel);
            o.Bool(s.isOnGround); o.Bool(s.hasJumped); o.Bool(s.hasDoubleJumped); o.Bool(s.hasFlipped);
            o.Vec(s.flipRelTorque); o.F(s.jumpTime); o.Bool(s.isFlipping); o.F(s.flipTime); o.Bool(s.isJumping);
            o.F(s.airTimeSinceJump); o.F(s.boost); o.Bool(s.isBoosting); o.F(s.boostingTime); o.F(s.timeSinceBoosted);
            o.F(s.supersonicTime); o.F(s.handbrakeVal); o.Bool(s.isAutoFlipping); o.F(s.autoFlipTimer);
            o.F(s.autoFlipTorqueScale); o.Bool(s.isDemoed); o.F(s.demoRespawnTimer);
            WriteControlsRaw(o, s.lastControls);
            o.Bool(s.worldContact.hasContact); o.Vec(s.worldContact.contactNormal);
            o.U32(s.carContact.otherCarID); o.F(s.carContact.cooldownTimer);
        }

        const std::vector<BoostPadState> pads = a.GetBoostPadStates();
        if (!pads.empty() && a.mode_ != GameMode::THE_VOID) {
            o.U32(static_cast<uint32_t>(pads.size()));
            for (size_t official = 0; official < pads.size(); ++official) {
                const BoostPadState& p = pads[static_cast<size_t>(RocketGpuPadIndexFromOfficial(static_cast<int>(official)))];
                o.Count(3);
                o.Bool(p.is_active != 0); o.F(p.cooldown); o.U32(p.prev_locked_car_id);
            }
        }

        {
            const BallStateCompat b = a.ball->GetState();
            const BallState& raw = a.ball_states_[0];
            o.Count(12);
            o.Vec(b.pos); o.Rot(b.rotMat); o.Vec(b.vel); o.Vec(b.angVel);
            o.F(raw.hs_info.y_target_dir); o.F(raw.hs_info.cur_target_speed); o.F(raw.hs_info.time_since_hit);
            o.I32(raw.ds_info.charge_level); o.F(raw.ds_info.accumulated_hit_force); o.F(raw.ds_info.y_target_dir);
            o.Bool(raw.ds_info.has_damaged != 0); o.U64(raw.ds_info.last_damage_tick);
        }

        o.U16(30);
        o.Count(30);
        o.Vec(Vec3(0.0f, 0.0f, m.gravity_z));
        o.F(m.car_mass); o.F(m.arena_friction); o.F(m.arena_restitution); o.F(m.ball_mass);
        o.F(m.ball_max_speed); o.F(m.ball_drag); o.F(m.ball_friction); o.F(m.ball_restitution); o.F(m.jump_accel);
        o.F(m.jump_immediate_speed); o.F(m.boost_accel_ground); o.F(m.boost_accel_air); o.F(m.boost_used_per_second);
        o.F(m.demo_respawn_time); o.F(m.boost_spawn_amount); o.F(m.bump_cooldown_time);
        o.F(m.boost_pad_cooldown_big); o.F(m.boost_pad_cooldown_small);
        o.F(m.ball_hit_extra_force_scale); o.F(m.bump_force_scale); o.F(m.ball_radius);
        o.Bool(m.unlimited_flips != 0); o.Bool(m.unlimited_double_jumps != 0);
        o.Bool(m.recharge_boost_enabled != 0); o.F(m.recharge_boost_per_second); o.F(m.recharge_boost_delay);
        o.U8(cfg.enable_demos ? 0u : 2u);
        o.Bool(m.enable_team_demos != 0);
        o.F(m.goal_base_threshold_y);
    }

    static Arena* Read(const std::vector<uint8_t>& data) {
        In i{data};
        i.Count(4, "arena header");
        const GameMode mode = static_cast<GameMode>(i.U8());
        const float tick_time = i.F();
        const uint64_t tick_count = i.U64();
        const uint32_t last_car_id = i.U32();
        i.Count(5, "arena config");
        ArenaConfig arena_config;
        (void)i.Vec(); (void)i.Vec(); (void)i.F();
        arena_config.noBallRot = i.Bool();
        (void)i.Bool();
        if (i.Bool()) throw std::runtime_error("rocketgpu::compat::Arena::DeserializeNew: custom boost pads are not supported");

        Arena* a = Arena::Create(mode, arena_config, 1.0f / tick_time);
        a->tick_count_ = tick_count;
        const uint32_t car_count = i.U32();
        for (uint32_t c = 0; c < car_count; ++c) {
            const Team team = static_cast<Team>(i.U8());
            const uint32_t id = i.U32();
            i.Count(8, "car controls");
            CarControls ctl;
            ctl.throttle = i.F(); ctl.steer = i.F(); ctl.pitch = i.F(); ctl.yaw = i.F(); ctl.roll = i.F();
            ctl.boost = i.Bool() ? 1 : 0; ctl.jump = i.Bool() ? 1 : 0; ctl.handbrake = i.Bool() ? 1 : 0;
            i.Count(9, "car config");
            CarConfig k;
            k.dodge_deadzone = i.F(); k.hitbox_offset = i.Vec(); k.hitbox_size = i.Vec();
            k.front_wheels_offset = i.Vec(); k.front_suspension_rest = i.F(); k.front_wheel_radius = i.F();
            k.back_wheels_offset = i.Vec(); k.back_suspension_rest = i.F(); k.back_wheel_radius = i.F();
            CarStateCompat s;
            s.ballHitInfo.isValid = i.Bool();
            if (s.ballHitInfo.isValid) {
                i.Count(5, "ball hit info");
                s.ballHitInfo.relativePosOnBall = i.Vec(); s.ballHitInfo.ballPos = i.Vec(); s.ballHitInfo.extraHitVel = i.Vec();
                s.ballHitInfo.tickCountWhenHit = i.U64(); s.ballHitInfo.tickCountWhenExtraImpulseApplied = i.U64();
            }
            i.Count(30, "car state");
            s.pos = i.Vec(); s.rotMat = i.Rot(); s.vel = i.Vec(); s.angVel = i.Vec();
            s.isOnGround = i.Bool(); s.hasJumped = i.Bool(); s.hasDoubleJumped = i.Bool(); s.hasFlipped = i.Bool();
            s.flipRelTorque = i.Vec(); s.jumpTime = i.F(); s.isFlipping = i.Bool(); s.flipTime = i.F(); s.isJumping = i.Bool();
            s.airTimeSinceJump = i.F(); s.boost = i.F(); s.isBoosting = i.Bool(); s.boostingTime = i.F(); s.timeSinceBoosted = i.F();
            s.supersonicTime = i.F(); s.handbrakeVal = i.F(); s.isAutoFlipping = i.Bool(); s.autoFlipTimer = i.F();
            s.autoFlipTorqueScale = i.F(); s.isDemoed = i.Bool(); s.demoRespawnTimer = i.F();
            s.lastControls = ReadControlsRaw(i);
            s.worldContact.hasContact = i.Bool(); s.worldContact.contactNormal = i.Vec();
            s.carContact.otherCarID = i.U32(); s.carContact.cooldownTimer = i.F();
            Car* car = a->AddCar(team, k);
            car->id = id;
            a->slot_to_id_[static_cast<size_t>(car->slot_)] = id;

            s.boost = a->GetMutatorConfig().boost_spawn_amount;
            car->SetState(s);
            car->SetControls(ctl);
        }
        a->next_car_id_ = last_car_id + 1;
        std::vector<BoostPadState> pads = a->GetBoostPadStates();
        if (!pads.empty() && mode != GameMode::THE_VOID) {
            const uint32_t pad_count = i.U32();
            if (pad_count != pads.size()) {
                throw std::runtime_error("rocketgpu::compat::Arena::DeserializeNew: different boost pad amount written in file (" +
                                         std::to_string(pad_count) + "/" + std::to_string(pads.size()) + ")");
            }
            for (uint32_t official = 0; official < pad_count; ++official) {
                BoostPadState& p = pads[static_cast<size_t>(RocketGpuPadIndexFromOfficial(static_cast<int>(official)))];
                i.Count(3, "boost pad");
                p.is_active = i.Bool() ? 1 : 0; p.cooldown = i.F(); p.prev_locked_car_id = i.U32();
                p.cur_locked_car_id = 0;
            }
            a->SetBoostPadStates(pads);
        }
        {
            i.Count(12, "ball state");
            BallStateCompat b;
            b.pos = i.Vec(); b.rotMat = i.Rot(); b.vel = i.Vec(); b.angVel = i.Vec();
            a->ball->SetState(b);
            a->PullStates();
            BallState& raw = a->ball_states_[0];
            raw.hs_info.y_target_dir = i.F(); raw.hs_info.cur_target_speed = i.F(); raw.hs_info.time_since_hit = i.F();
            raw.ds_info.charge_level = i.I32(); raw.ds_info.accumulated_hit_force = i.F(); raw.ds_info.y_target_dir = i.F();
            raw.ds_info.has_damaged = i.Bool() ? 1 : 0; raw.ds_info.last_damage_tick = i.U64();
            a->PushStates();
        }
        {
            const uint16_t arg_count = i.U16();
            if (arg_count != 30) {
                throw std::runtime_error("rocketgpu::compat::Arena::DeserializeNew: mutator config is from a different version of RocketSim, fields don't match");
            }
            i.Count(30, "mutator config");
            MutatorConfig m = a->GetMutatorConfig();
            const Vec3 gravity = i.Vec();
            m.gravity_z = gravity.z;
            m.car_mass = i.F(); m.arena_friction = i.F(); m.arena_restitution = i.F(); m.ball_mass = i.F();
            m.ball_max_speed = i.F(); m.ball_drag = i.F(); m.ball_friction = i.F(); m.ball_restitution = i.F(); m.jump_accel = i.F();
            m.jump_immediate_speed = i.F(); m.boost_accel_ground = i.F(); m.boost_accel_air = i.F(); m.boost_used_per_second = i.F();
            m.demo_respawn_time = i.F(); m.boost_spawn_amount = i.F(); m.bump_cooldown_time = i.F();
            m.boost_pad_cooldown_big = i.F(); m.boost_pad_cooldown_small = i.F();
            m.ball_hit_extra_force_scale = i.F(); m.bump_force_scale = i.F(); m.ball_radius = i.F();
            m.unlimited_flips = i.Bool() ? 1 : 0; m.unlimited_double_jumps = i.Bool() ? 1 : 0;
            m.recharge_boost_enabled = i.Bool() ? 1 : 0; m.recharge_boost_per_second = i.F(); m.recharge_boost_delay = i.F();
            (void)i.U8();
            m.enable_team_demos = i.Bool() ? 1 : 0;
            m.goal_base_threshold_y = i.F();
            a->SetMutatorConfig(m);
        }
        return a;
    }
};

void Arena::Serialize(std::vector<uint8_t>& out) const { ArenaSerializer::Write(*this, out); }
Arena* Arena::DeserializeNew(const std::vector<uint8_t>& data) { return ArenaSerializer::Read(data); }

void Arena::SerializeToFile(const std::string& path, bool write_version_check) const {
    std::vector<uint8_t> bytes;
    if (write_version_check) {
        const uint32_t version = RocketSimVersionId();
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&version);
        bytes.insert(bytes.end(), b, b + sizeof(version));
    }
    Serialize(bytes);
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("rocketgpu::compat::Arena::SerializeToFile: cannot open " + path);
    std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
}

Arena* Arena::DeserializeNewFromFile(const std::string& path, bool version_check) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("rocketgpu::compat::Arena::DeserializeNewFromFile: cannot open " + path);
    std::vector<uint8_t> bytes;
    uint8_t buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) bytes.insert(bytes.end(), buf, buf + n);
    std::fclose(f);
    if (version_check) {
        if (bytes.size() < 4) throw std::runtime_error("rocketgpu::compat::Arena::DeserializeNewFromFile: file too short");
        uint32_t version = 0;
        std::memcpy(&version, bytes.data(), 4);
        if (version != RocketSimVersionId()) {
            throw std::runtime_error("rocketgpu::compat::Arena::DeserializeNewFromFile: file is invalid or from a different version of RocketSim");
        }
        bytes.erase(bytes.begin(), bytes.begin() + 4);
    }
    return DeserializeNew(bytes);
}
}
}
