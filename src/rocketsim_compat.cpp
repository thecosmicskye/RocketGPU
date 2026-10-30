#include <rocketgpu/rocketsim_compat.h>
#include <rocketgpu/detail/math.hpp>
#include <rocketgpu/detail/constants.hpp>

#include <algorithm>
#include <functional>
#include <cfloat>
#include <cmath>
#include <stdexcept>

namespace rocketgpu {
namespace compat {
RotMat RotMatFromQuat(const Quat& q) {
    const float n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    const float s = (n > 0.0f) ? (1.0f / n) : 1.0f;
    const float w = q.w * s, x = q.x * s, y = q.y * s, z = q.z * s;
    RotMat m;
    m.forward = Vec3(1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y));
    m.right = Vec3(2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x));
    m.up = Vec3(2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y));
    return m;
}

void SeedCarBasisFromRotMat(CarState& s, const RotMat& m) {
    s.basis_forward = m.forward;
    s.basis_right = m.right;
    s.basis_up = m.up;
    s.basis_authoritative = 1;
    s.rot = rocketgpu::detail::BulletQuatFromBasisColumns(s.basis_forward, s.basis_right, s.basis_up);
}

RotMat RotMatFromCarState(const CarState& s) {
    if (s.basis_authoritative) {
        RotMat out;
        out.forward = s.basis_forward;
        out.right = s.basis_right;
        out.up = s.basis_up;
        return out;
    }
    return RotMatFromQuat(s.rot);
}

RotMat PublicRotMatFromCarState(const CarState& s) {
    return s.is_demoed ? RotMatFromQuat(s.rot) : RotMatFromCarState(s);
}

Quat QuatFromRotMat(const RotMat& m) {
    const float m00 = m.forward.x, m10 = m.forward.y, m20 = m.forward.z;
    const float m01 = m.right.x, m11 = m.right.y, m21 = m.right.z;
    const float m02 = m.up.x, m12 = m.up.y, m22 = m.up.z;
    const float tr = m00 + m11 + m22;
    Quat q;
    if (tr > 0.0f) {
        const float s = std::sqrt(tr + 1.0f) * 2.0f;
        q.w = 0.25f * s; q.x = (m21 - m12) / s; q.y = (m02 - m20) / s; q.z = (m10 - m01) / s;
    } else if (m00 > m11 && m00 > m22) {
        const float s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
        q.w = (m21 - m12) / s; q.x = 0.25f * s; q.y = (m01 + m10) / s; q.z = (m02 + m20) / s;
    } else if (m11 > m22) {
        const float s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
        q.w = (m02 - m20) / s; q.x = (m01 + m10) / s; q.y = 0.25f * s; q.z = (m12 + m21) / s;
    } else {
        const float s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
        q.w = (m10 - m01) / s; q.x = (m02 + m20) / s; q.y = (m12 + m21) / s; q.z = 0.25f * s;
    }
    return q;
}

Arena* Arena::Create(GameMode mode, float tick_rate) {
    return Create(mode, ArenaConfig{}, tick_rate);
}

Arena* Arena::Create(GameMode mode, const ArenaConfig& arena_config, float tick_rate) {
    if (mode != GameMode::SOCCAR && mode != GameMode::THE_VOID && mode != GameMode::HEATSEEKER && mode != GameMode::SNOWDAY) {
        throw std::runtime_error("rocketgpu::compat::Arena: only SOCCAR, HEATSEEKER, SNOWDAY and THE_VOID are implemented (HOOPS/DROPSHOT are blocked on their arena meshes)");
    }
    Arena* a = new Arena();
    a->tick_rate_ = tick_rate;
    a->mode_ = mode;
    a->arena_config_ = arena_config;

    SimConfig cfg{};
    cfg.mode = mode == GameMode::THE_VOID ? Mode::TheVoid
        : (mode == GameMode::HEATSEEKER ? Mode::Heatseeker
        : (mode == GameMode::SNOWDAY ? Mode::Snowday : Mode::Soccar));
    cfg.env_count = 1;
    cfg.max_cars_per_env = 8;
    cfg.tick_dt = 1.0f / tick_rate;
    cfg.no_ball_rot = arena_config.noBallRot;
    cfg.native_sphere_normalization = arena_config.nativeSseMath ? 1 : 0;
    if (arena_config.useCustomBoostPads) {
        if (arena_config.customBoostPads.size() > static_cast<size_t>(kMaxBoostPads)) {
            throw std::runtime_error("rocketgpu::compat::Arena: too many custom boost pads (kMaxBoostPads)");
        }
        cfg.use_custom_boost_pads = 1;
        cfg.boost_pad_count = static_cast<int32_t>(arena_config.customBoostPads.size());
        for (int p = 0; p < cfg.boost_pad_count; ++p) {
            cfg.boost_pads[p].pos = arena_config.customBoostPads[static_cast<size_t>(p)].pos;
            cfg.boost_pads[p].is_big = arena_config.customBoostPads[static_cast<size_t>(p)].isBig ? 1 : 0;
        }
    }
    cfg.enable_demos = 1;
    cfg.enable_car_car_collisions = 1;

    cfg.mirror_official_respawn_rng = 1;

    cfg.respawn_rng_seed = 0;
    a->sim_ = std::make_unique<BatchSim>(cfg);

    a->car_states_.resize(a->sim_->car_capacity());
    a->ball_states_.resize(a->sim_->env_count());
    a->controls_.resize(a->sim_->car_capacity());
    a->configs_.assign(a->sim_->car_capacity(), CarConfig{});
    a->teams_.assign(a->sim_->car_capacity(), Team::Blue);
    a->slot_used_.assign(a->sim_->car_capacity(), 0);
    a->slot_to_id_.assign(a->sim_->car_capacity(), 0u);
    a->env_states_.resize(a->sim_->env_count());

    const int32_t zero = 0;
    a->sim_->set_env_car_count(&zero, 1);
    a->ball = new Ball(a);

    {
        const SimConfig& sim_cfg = a->sim_->config();
        const int pad_count = sim_cfg.boost_pad_count;
        std::vector<int> rocketgpu_order;
        if (sim_cfg.use_custom_boost_pads) {
            for (int i = 0; i < pad_count; ++i) rocketgpu_order.push_back(i);
        } else {
            for (int pass = 0; pass < 2; ++pass) {
                for (int i = 0; i < pad_count; ++i) {
                    const bool big = sim_cfg.boost_pads[i].is_big != 0;
                    if ((pass == 0) == big) {
                        rocketgpu_order.push_back(i);
                    }
                }
            }
        }
        for (int official_idx = 0; official_idx < static_cast<int>(rocketgpu_order.size()); ++official_idx) {
            const int rocketgpu_idx = rocketgpu_order[static_cast<size_t>(official_idx)];
            BoostPad* pad = new BoostPad(a, official_idx, rocketgpu_idx);
            pad->config.pos = sim_cfg.boost_pads[rocketgpu_idx].pos;
            pad->config.isBig = sim_cfg.boost_pads[rocketgpu_idx].is_big != 0;
            a->boost_pads_.push_back(pad);
        }
    }
    return a;
}

BoostPadState BoostPad::GetState() const {
    const std::vector<BoostPadState> states = arena_->GetBoostPadStates();
    if (rocketgpu_index_ < 0 || static_cast<size_t>(rocketgpu_index_) >= states.size()) {
        return BoostPadState{};
    }
    return states[static_cast<size_t>(rocketgpu_index_)];
}

void BoostPad::SetState(const BoostPadState& state) {
    std::vector<BoostPadState> states = arena_->GetBoostPadStates();
    if (rocketgpu_index_ < 0 || static_cast<size_t>(rocketgpu_index_) >= states.size()) {
        return;
    }
    states[static_cast<size_t>(rocketgpu_index_)] = state;
    arena_->SetBoostPadStates(states);
}

Arena::~Arena() {
    for (Car* c : cars) delete c;
    for (BoostPad* p : boost_pads_) delete p;
    delete ball;
}

Car* Arena::AddCar(Team team, const CarConfig& config) {
    if (static_cast<size_t>(car_count_) >= sim_->car_capacity()) {
        throw std::runtime_error("rocketgpu::compat::Arena: car capacity exceeded");
    }
    int slot = -1;
    for (size_t i = 0; i < slot_used_.size(); ++i) {
        if (!slot_used_[i]) {
            slot = static_cast<int>(i);
            break;
        }
    }
    if (slot < 0) {
        throw std::runtime_error("rocketgpu::compat::Arena: car capacity exceeded");
    }
    slot_used_[static_cast<size_t>(slot)] = 1;
    if (slot >= car_count_) {
        car_count_ = slot + 1;
    }
    PullStates();
    car_states_[static_cast<size_t>(slot)] = CarState{};
    for (int w = 0; w < 4; ++w) car_states_[static_cast<size_t>(slot)].wheels_with_contact[w] = 0;
    PushStates();
    teams_[slot] = team;
    configs_[slot] = config;

    sim_->set_teams(teams_.data(), teams_.size());
    sim_->set_car_configs(configs_.data(), configs_.size());
    const int32_t n = car_count_;
    sim_->set_env_car_count(&n, 1);

    Car* car = new Car(this, slot);
    car->id = next_car_id_++;
    slot_to_id_[static_cast<size_t>(slot)] = car->id;
    car->team = team;
    cars.push_back(car);
    dirty_ = true;
    return car;
}

void Arena::PullStates() const {
    if (!dirty_) return;
    sim_->download_car_states(car_states_.data(), car_states_.size());
    sim_->download_ball_states(ball_states_.data(), ball_states_.size());
    dirty_ = false;
}

void Arena::PushStates() {
    sim_->upload_car_states(car_states_.data(), car_states_.size());
    sim_->upload_ball_states(ball_states_.data(), ball_states_.size());

    dirty_ = true;
}

void Arena::SetGoalScoreCallback(GoalScoreEventFn callback_fn, void* user_info) {
    if (mode_ == GameMode::THE_VOID) {
        throw std::runtime_error("Cannot set a goal score callback when on THE_VOID gamemode");
    }
    goal_callback_ = std::move(callback_fn);
    goal_user_info_ = user_info;
}
void Arena::SetCarBumpCallback(CarBumpEventFn callback_fn, void* user_info) {
    bump_callback_ = std::move(callback_fn);
    bump_user_info_ = user_info;

    PullStates();
    bump_event_seen_tick_.assign(sim_->car_capacity(), 0u);
    for (size_t i = 0; i < car_states_.size(); ++i) {
        bump_event_seen_tick_[i] = car_states_[i].bump_event_tick;
    }
}
bool Arena::RemoveCar(Car* car) {
    return car ? RemoveCar(car->id) : false;
}
bool Arena::RemoveCar(uint32_t id) {
    auto it = std::find_if(cars.begin(), cars.end(), [id](const Car* c) { return c->id == id; });
    if (it == cars.end()) {
        return false;
    }
    Car* car = *it;
    const int slot = car->slot_;
    PullStates();

    car_states_[static_cast<size_t>(slot)] = CarState{};
    car_states_[static_cast<size_t>(slot)].is_demoed = 1;
    car_states_[static_cast<size_t>(slot)].demo_respawn_timer = 1.0e30f;
    car_states_[static_cast<size_t>(slot)].demo_no_contact_response = 1;
    slot_used_[static_cast<size_t>(slot)] = 0;
    slot_to_id_[static_cast<size_t>(slot)] = 0u;
    int highest = -1;
    for (size_t i = 0; i < slot_used_.size(); ++i) {
        if (slot_used_[i]) {
            highest = static_cast<int>(i);
        }
    }
    car_count_ = highest + 1;
    const int32_t n = car_count_;
    sim_->set_env_car_count(&n, 1);
    PushStates();
    cars.erase(it);
    delete car;
    dirty_ = true;
    return true;
}
Car* Arena::GetCar(uint32_t id) {
    for (Car* c : cars) {
        if (c->id == id) {
            return c;
        }
    }
    return nullptr;
}
Arena* Arena::Clone(bool copy_callbacks) const {
    PullStates();
    Arena* a = Create(mode_, arena_config_, tick_rate_);
    a->tick_count_ = tick_count_;
    a->car_count_ = car_count_;
    a->next_car_id_ = next_car_id_;
    a->slot_used_ = slot_used_;
    a->slot_to_id_ = slot_to_id_;
    a->teams_ = teams_;
    a->configs_ = configs_;
    a->controls_ = controls_;
    a->car_states_ = car_states_;
    a->ball_states_ = ball_states_;
    a->sim_->set_teams(a->teams_.data(), a->teams_.size());
    a->sim_->set_car_configs(a->configs_.data(), a->configs_.size());
    const int32_t n = car_count_;
    a->sim_->set_env_car_count(&n, 1);

    a->sim_->copy_state_from(*sim_);
    for (const Car* car : cars) {
        Car* nc = new Car(a, car->slot_);
        nc->id = car->id;
        nc->team = car->team;
        a->cars.push_back(nc);
    }
    if (copy_callbacks) {
        a->goal_callback_ = goal_callback_;
        a->goal_user_info_ = goal_user_info_;
        a->bump_callback_ = bump_callback_;
        a->bump_user_info_ = bump_user_info_;
        a->bump_event_seen_tick_ = bump_event_seen_tick_;
    }
    a->dirty_ = true;
    return a;
}
bool Arena::IsBallScored() const {
    PullStates();
    const BallState& b = ball_states_[0];
    const MutatorConfig& m = sim_->config().mutator;
    return std::fabs(b.pos.y) > (m.goal_base_threshold_y + m.ball_radius);
}
bool Arena::IsBallProbablyGoingIn(float max_time, float extra_margin, rocketgpu::Team* goal_team_out) const {
    if (mode_ != GameMode::SOCCAR) {
        throw std::runtime_error("Arena::IsBallProbablyGoingIn() is not supported for this game mode");
    }
    PullStates();
    const BallState& b = ball_states_[0];
    const MutatorConfig& m = sim_->config().mutator;

    constexpr float kBTToUU = 50.0f;
    const rocketgpu::Vec3 ball_pos(b.pos_bt.x * kBTToUU, b.pos_bt.y * kBTToUU, b.pos_bt.z * kBTToUU);
    const rocketgpu::Vec3 ball_vel(b.vel_bt.x * kBTToUU, b.vel_bt.y * kBTToUU, b.vel_bt.z * kBTToUU);
    if (std::fabs(ball_vel.y) < FLT_EPSILON) {
        return false;
    }
    const float score_dir_sgn = static_cast<float>((ball_vel.y > 0.0f) - (ball_vel.y < 0.0f));
    const float goal_y = m.goal_base_threshold_y * score_dir_sgn;
    const float dist_to_goal = std::fabs(ball_pos.y - goal_y);
    const float time_to_goal = dist_to_goal / std::fabs(ball_vel.y);
    if (time_to_goal > max_time) {
        return false;
    }

    const rocketgpu::Vec3 gravity(0.0f, 0.0f, m.gravity_z);
    const rocketgpu::Vec3 extrap(
        ball_pos.x + ball_vel.x * time_to_goal + (gravity.x * time_to_goal * time_to_goal) / 2.0f,
        ball_pos.y + ball_vel.y * time_to_goal + (gravity.y * time_to_goal * time_to_goal) / 2.0f,
        ball_pos.z + ball_vel.z * time_to_goal + (gravity.z * time_to_goal * time_to_goal) / 2.0f);
    constexpr float kApproxGoalHalfWidth = 892.755f;
    constexpr float kApproxGoalHeight = 642.775f;
    const float score_margin = m.ball_radius * 0.1f + extra_margin;
    if (extrap.z > kApproxGoalHeight + score_margin) {
        return false;
    }
    if (std::fabs(extrap.x) > kApproxGoalHalfWidth + score_margin) {
        return false;
    }
    if (goal_team_out) {
        *goal_team_out = score_dir_sgn < 0.0f ? rocketgpu::Team::Blue : rocketgpu::Team::Orange;
    }
    return true;
}
void Arena::SetMutatorConfig(const MutatorConfig& config) {
    sim_->set_mutator_config(config);
}
const MutatorConfig& Arena::GetMutatorConfig() const {
    return sim_->config().mutator;
}
void Arena::SetBoostPadStates(const std::vector<BoostPadState>& states) {
    pad_states_ = states;
    pad_states_.resize(sim_->boost_pad_capacity());
    sim_->upload_boost_pad_states(pad_states_.data(), pad_states_.size());
    dirty_ = true;
}
std::vector<BoostPadState> Arena::GetBoostPadStates() const {
    pad_states_.resize(sim_->boost_pad_capacity());
    sim_->download_boost_pad_states(pad_states_.data(), pad_states_.size());
    return pad_states_;
}
uint32_t Arena::CompatIdFromDeviceId(uint32_t device_id) const {
    if (device_id == 0 || device_id > slot_to_id_.size()) {
        return 0u;
    }
    return slot_to_id_[static_cast<size_t>(device_id - 1)];
}
void Arena::DrainBumpEvents() {
    if (!bump_callback_) {
        return;
    }
    PullStates();

    for (Car* bumper : cars) {
        const CarState& s = car_states_[static_cast<size_t>(bumper->slot_)];
        if (!s.bump_event_was_bumper || s.bump_event_tick == 0 ||
            s.bump_event_tick == bump_event_seen_tick_[static_cast<size_t>(bumper->slot_)]) {
            continue;
        }
        bump_event_seen_tick_[static_cast<size_t>(bumper->slot_)] = s.bump_event_tick;
        Car* victim = GetCar(CompatIdFromDeviceId(s.bump_event_other_id));
        if (victim) {
            bump_callback_(this, bumper, victim, s.bump_event_is_demo != 0, bump_user_info_);
        }
    }
}

void Arena::Step(int ticks_to_simulate) {
    sim_->upload_controls(controls_.data(), controls_.size());

    if ((goal_callback_ || bump_callback_) && ticks_to_simulate > 1) {
        for (int i = 0; i < ticks_to_simulate; ++i) {
            Step(1);
        }
        return;
    }
    sim_->step(ticks_to_simulate);
    tick_count_ += static_cast<uint64_t>(ticks_to_simulate);
    dirty_ = true;
    DrainBumpEvents();

    if (goal_callback_) {
        sim_->download_env_states(env_states_.data(), env_states_.size());
        const EnvState& env = env_states_[0];
        if (env.goal_scored_this_tick != 0) {
            const Team scoring = (env.last_scored_team == static_cast<int32_t>(Team::Orange))
                                     ? Team::Orange
                                     : Team::Blue;
            goal_callback_(this, scoring, goal_user_info_);
        }
    }
}

void Arena::ResetToRandomKickoff(int seed) {
    EnvCommand command;
    command.reset_to_kickoff = 1;
    command.reset_scoreboard = 1;
    command.reset_tick_count = 1;
    command.kickoff_seed_mode = seed == -1 ? 2 : 1;
    command.kickoff_seed = static_cast<uint64_t>(static_cast<uint32_t>(seed));
    sim_->apply_env_commands(&command, 1);
    dirty_ = true;
}

void Arena::SetRespawnRngSeed(uint64_t seed) {
    sim_->set_respawn_rng_seed(seed);
    dirty_ = true;
}

void Arena::SetCarVisitOrderMode(uint8_t mode) {
    if (mode != 0 && mode != 2) {
        throw std::invalid_argument("SetCarVisitOrderMode: mode must be 0 (slot order) or 2 (reverse slot order); use SetCarVisitOrder for an explicit order");
    }
    EnvState env;
    sim_->download_env_states(&env, 1);
    env.car_visit_order_mode = mode;
    sim_->upload_env_states(&env, 1);
    dirty_ = true;
}

void Arena::SetCarVisitOrder(const std::vector<uint32_t>& compat_ids_in_visit_order) {
    const int max_cars = sim_->config().max_cars_per_env;
    int slots[8];
    int count = 0;
    uint32_t seen = 0;
    for (uint32_t id : compat_ids_in_visit_order) {
        Car* car = GetCar(id);
        if (!car || count >= max_cars || count >= 8) {
            throw std::invalid_argument("SetCarVisitOrder: unknown car id or too many cars");
        }
        if (seen & (1u << car->slot_)) {
            throw std::invalid_argument("SetCarVisitOrder: car id listed twice");
        }
        seen |= 1u << car->slot_;
        slots[count++] = car->slot_;
    }

    for (int slot = 0; slot < max_cars && slot < 8 && count < 8; ++slot) {
        if (!(seen & (1u << slot))) slots[count++] = slot;
    }
    EnvState env;
    sim_->download_env_states(&env, 1);
    env.car_visit_order_mode = 1;
    env.car_visit_order_packed = PackCarVisitOrder(slots, count);
    sim_->upload_env_states(&env, 1);
    dirty_ = true;
}

uint8_t Arena::GetCarVisitOrderMode() const {
    EnvState env;
    sim_->download_env_states(&env, 1);
    return env.car_visit_order_mode;
}

namespace {
BallHitInfoCompat BallHitInfoFromState(const BallHitInfo& in) {
    BallHitInfoCompat out;
    out.isValid = in.is_valid != 0;
    out.relativePosOnBall = in.relative_pos_on_ball;
    out.ballPos = in.ball_pos;
    out.extraHitVel = in.extra_hit_vel;
    out.tickCountWhenHit = in.tick_count_when_hit;
    out.tickCountWhenExtraImpulseApplied = in.tick_count_when_extra_impulse_applied;
    return out;
}
}

CarStateCompat Car::GetState() const {
    arena_->PullStates();
    const CarState& s = arena_->car_states_[slot_];
    CarStateCompat out;
    out.pos = s.pos;
    out.rotMat = PublicRotMatFromCarState(s);
    out.vel = s.vel;
    out.angVel = s.ang_vel;
    out.isOnGround = s.is_on_ground != 0;
    for (int w = 0; w < 4; ++w) out.wheelsWithContact[w] = s.wheels_with_contact[w] != 0;
    out.hasJumped = s.has_jumped != 0;
    out.hasDoubleJumped = s.has_double_jumped != 0;
    out.hasFlipped = s.has_flipped != 0;
    out.flipRelTorque = s.flip_rel_torque;
    out.jumpTime = s.jump_time;
    out.flipTime = s.flip_time;
    out.isFlipping = s.is_flipping != 0;
    out.isJumping = s.is_jumping != 0;
    out.airTime = s.air_time;
    out.airTimeSinceJump = s.air_time_since_jump;
    out.boost = s.boost;
    out.timeSinceBoosted = s.time_since_boosted;
    out.isBoosting = s.is_boosting != 0;
    out.boostingTime = s.boosting_time;
    out.isSupersonic = s.is_supersonic != 0;
    out.supersonicTime = s.supersonic_time;
    out.handbrakeVal = s.handbrake_val;
    out.isAutoFlipping = s.is_auto_flipping != 0;
    out.autoFlipTimer = s.auto_flip_timer;
    out.autoFlipTorqueScale = s.auto_flip_torque_scale;
    out.worldContact.hasContact = s.world_contact.has_contact != 0;
    out.worldContact.contactNormal = s.world_contact.contact_normal;
    out.carContact.otherCarID = arena_->CompatIdFromDeviceId(s.car_contact.other_car_id);
    out.carContact.cooldownTimer = s.car_contact.cooldown_timer;
    out.isDemoed = s.is_demoed != 0;
    out.demoRespawnTimer = s.demo_respawn_timer;
    out.tickCountSinceUpdate = s.tick_count_since_update;
    out.ballHitInfo = BallHitInfoFromState(s.ball_hit_info);
    out.lastControls = s.last_controls;
    out.airState = s.air_state;
    out.dodgeTimeout = s.dodge_timeout;
    out.dodgeElapsed = s.dodge_elapsed;
    out.dodgeDirX = s.dodge_dir_x;
    out.dodgeDirY = s.dodge_dir_y;
    out.latestTouch.tick = s.latest_touch_tick;
    out.latestTouch.location = s.latest_touch_location;
    out.latestTouch.normal = s.latest_touch_normal;
    return out;
}

void Car::SetState(const CarStateCompat& state) {
    arena_->PullStates();
    CarState& s = arena_->car_states_[slot_];
    s.pos = state.pos;
    SeedCarBasisFromRotMat(s, state.rotMat);
    s.set_wheel_normals_mask = 0;
    // Velocities and the supersonic flag left as reported keep the step's values.
    const bool reported = (s.reported_flags & CarState::kReportedValid) != 0;
    const auto same = [](const Vec3& a, const Vec3& b) { return a.x == b.x && a.y == b.y && a.z == b.z; };
    s.vel = (reported && same(state.vel, s.reported_vel)) ? s.internal_vel : state.vel;
    s.ang_vel = (reported && same(state.angVel, s.reported_ang_vel)) ? s.internal_ang_vel : state.angVel;
    s.is_on_ground = state.isOnGround ? 1 : 0;
    for (int w = 0; w < 4; ++w) s.wheels_with_contact[w] = state.wheelsWithContact[w] ? 1 : 0;
    s.has_jumped = state.hasJumped ? 1 : 0;
    s.has_double_jumped = state.hasDoubleJumped ? 1 : 0;
    s.has_flipped = state.hasFlipped ? 1 : 0;
    s.flip_rel_torque = state.flipRelTorque;
    s.jump_time = state.jumpTime;
    s.flip_time = state.flipTime;
    s.is_flipping = state.isFlipping ? 1 : 0;
    s.is_jumping = state.isJumping ? 1 : 0;
    s.air_time = state.airTime;
    s.air_time_since_jump = state.airTimeSinceJump;
    s.boost = state.boost;
    s.time_since_boosted = state.timeSinceBoosted;
    s.is_boosting = state.isBoosting ? 1 : 0;
    s.boosting_time = state.boostingTime;
    s.is_supersonic = (reported && state.isSupersonic == ((s.reported_flags & CarState::kReportedIsSupersonic) != 0))
        ? s.internal_is_supersonic : (state.isSupersonic ? 1 : 0);
    s.supersonic_time = state.supersonicTime;
    s.handbrake_val = state.handbrakeVal;
    s.is_auto_flipping = state.isAutoFlipping ? 1 : 0;
    s.auto_flip_timer = state.autoFlipTimer;
    s.auto_flip_torque_scale = state.autoFlipTorqueScale;
    s.world_contact.has_contact = state.worldContact.hasContact ? 1 : 0;
    s.world_contact.contact_normal = state.worldContact.contactNormal;
    s.car_contact.other_car_id = state.carContact.otherCarID;
    s.car_contact.cooldown_timer = state.carContact.cooldownTimer;
    s.is_demoed = state.isDemoed ? 1 : 0;
    s.demo_respawn_timer = state.demoRespawnTimer;

    s.ball_hit_info.is_valid = state.ballHitInfo.isValid ? 1 : 0;
    s.ball_hit_info.relative_pos_on_ball = state.ballHitInfo.relativePosOnBall;
    s.ball_hit_info.ball_pos = state.ballHitInfo.ballPos;
    s.ball_hit_info.extra_hit_vel = state.ballHitInfo.extraHitVel;
    s.ball_hit_info.tick_count_when_hit = state.ballHitInfo.tickCountWhenHit;
    s.ball_hit_info.tick_count_when_extra_impulse_applied = state.ballHitInfo.tickCountWhenExtraImpulseApplied;
    s.latest_touch_tick = state.latestTouch.tick;
    s.latest_touch_location = state.latestTouch.location;
    s.latest_touch_normal = state.latestTouch.normal;
    s.tick_count_since_update = 0;
    s.last_controls = state.lastControls;
    arena_->PushStates();
}

CarStateV3 Car::GetStateV3() const {
    arena_->PullStates();
    CarState s = arena_->car_states_[slot_];
    if (s.tick_count_since_update != 0) {
        s.RestoreWorkingFields();  // the step's own velocities and flags, unless written since the step
    } else if (s.supersonic_time > 0.0f) {
        s.is_supersonic = 1;  // as the next step reads a set state
    }
    if (!s.supersonic_set_v3 && !s.is_demoed) {
        // v3 updates the supersonic flag from the velocity at the end of a tick (car/base.rs post_tick_update); the
        // step does it at the start of the next. Apply v3's update to this velocity.
        const MutatorConfig& m = arena_->GetMutatorConfig();
        const float speed_sq = s.vel.x * s.vel.x + s.vel.y * s.vel.y + s.vel.z * s.vel.z;
        const float start_sq = m.supersonic_start_speed * m.supersonic_start_speed;
        if (!s.is_supersonic) {
            s.is_supersonic = speed_sq >= start_sq ? 1 : 0;
            s.supersonic_time = 0.0f;
        } else if (speed_sq >= start_sq) {
            s.supersonic_time = 0.0f;
        } else if (speed_sq < m.supersonic_maintain_min_speed * m.supersonic_maintain_min_speed) {
            s.is_supersonic = 0;
            s.supersonic_time = 0.0f;
        } else {
            s.supersonic_time += arena_->GetTickTime();
            if (s.supersonic_time >= m.supersonic_maintain_max_time) {
                s.is_supersonic = 0;
                s.supersonic_time = 0.0f;
            }
        }
    }
    CarStateV3 out;
    out.pos = s.pos;
    out.rotMat = PublicRotMatFromCarState(s);
    out.vel = s.vel;
    out.angVel = s.ang_vel;
    out.controls = arena_->controls_[slot_];
    out.hasJumped = s.has_jumped != 0;
    out.hasDoubleJumped = s.has_double_jumped != 0;
    out.hasFlipped = s.has_flipped != 0;
    out.flipRelTorque = s.flip_rel_torque;
    out.jumpTicks = static_cast<uint32_t>(rintf(s.jump_time / arena_->GetTickTime()));
    out.flipTime = s.flip_time;
    out.isFlipping = s.is_flipping != 0;
    out.isJumping = s.is_jumping != 0;
    out.jumpHoldBroken = s.jump_hold_broken != 0;
    out.airTime = s.air_time;
    out.airTimeSinceJump = s.air_time_since_jump;
    out.boost = s.boost;
    out.handbrakeVal = s.handbrake_val;
    out.isSupersonic = s.is_supersonic != 0;
    out.supersonicGraceTimer = s.supersonic_time;
    out.timeSinceBoosted = s.time_since_boosted;
    out.boostingTime = s.boosting_time;
    out.isBoosting = s.is_boosting != 0;
    out.isAutoFlipping = s.is_auto_flipping != 0;
    out.autoFlipTimer = s.auto_flip_timer;
    out.autoFlipTorqueScale = s.auto_flip_torque_scale;
    out.bumpCooldownTimer = s.car_contact.cooldown_timer;
    out.bumpCooldownVictim = s.car_contact.other_car_id ? arena_->CompatIdFromDeviceId(s.car_contact.other_car_id) : 0;
    out.lastExtraHitTick = s.ball_hit_info.is_valid ? s.ball_hit_info.tick_count_when_extra_impulse_applied : ~0ULL;
    out.isDemoed = s.is_demoed != 0;
    out.demoRespawnTimer = s.demo_respawn_timer;
    out.isOnGround = s.is_on_ground != 0;
    // v3 numbers each wheel pair the other way round (v3 wheel w is RocketGPU wheel w ^ 1).
    for (int w = 0; w < 4; ++w) out.wheelsWithContact[w] = s.wheels_with_contact[w ^ 1] != 0;
    out.hasWorldContact = s.world_contact.has_contact != 0;
    out.worldContactNormal = s.world_contact.contact_normal;
    out.prevControls = s.last_controls;
    // Wheel contact normals a set state supplied stay pending until the next step reads them.
    out.hasWheelContactNormals = s.set_wheel_normals_mask != 0;
    for (int w = 0; w < 4; ++w) out.wheelContactNormals[w] = s.set_wheel_normals[w ^ 1];
    return out;
}

void Car::SetStateV3(const CarStateV3& state) {
    arena_->PullStates();
    CarState& s = arena_->car_states_[slot_];
    const float dt = arena_->GetTickTime();
    s.pos = state.pos;
    SeedCarBasisFromRotMat(s, state.rotMat);
    s.vel = state.vel;
    s.ang_vel = state.angVel;
    s.has_jumped = state.hasJumped ? 1 : 0;
    s.has_double_jumped = state.hasDoubleJumped ? 1 : 0;
    s.has_flipped = state.hasFlipped ? 1 : 0;
    s.flip_rel_torque = state.flipRelTorque;
    // jump_time sums dt per tick for its first 1024 ticks, like the step's jump clock (see CarState::jump_time).
    float jump_time = 0.0f;
    if (state.jumpTicks > 1024) {
        jump_time = static_cast<float>(state.jumpTicks) * dt;
    } else {
        for (uint32_t i = 0; i < state.jumpTicks; ++i) jump_time += dt;
    }
    s.jump_time = jump_time;
    s.flip_time = state.flipTime;
    s.is_flipping = state.isFlipping ? 1 : 0;
    s.is_jumping = state.isJumping ? 1 : 0;
    s.jump_hold_broken = state.jumpHoldBroken ? 1 : 0;
    s.air_time = state.airTime;
    s.air_time_since_jump = state.airTimeSinceJump;
    s.boost = state.boost;
    s.handbrake_val = state.handbrakeVal;
    s.is_supersonic = state.isSupersonic ? 1 : 0;
    s.supersonic_time = state.supersonicGraceTimer;
    s.supersonic_set_v3 = state.isDemoed ? 0 : 1;
    s.time_since_boosted = state.timeSinceBoosted;
    s.boosting_time = state.boostingTime;
    s.is_boosting = state.isBoosting ? 1 : 0;
    s.is_auto_flipping = state.isAutoFlipping ? 1 : 0;
    s.auto_flip_timer = state.autoFlipTimer;
    s.auto_flip_torque_scale = state.autoFlipTorqueScale;
    s.car_contact.cooldown_timer = state.bumpCooldownTimer;
    s.car_contact.other_car_id = state.bumpCooldownVictim;
    if (state.lastExtraHitTick == ~0ULL) {
        s.ball_hit_info.tick_count_when_extra_impulse_applied = ~0ULL;
    } else {
        s.ball_hit_info.is_valid = 1;
        s.ball_hit_info.tick_count_when_extra_impulse_applied = state.lastExtraHitTick;
    }
    s.is_demoed = state.isDemoed ? 1 : 0;
    s.demo_respawn_timer = state.demoRespawnTimer;
    s.is_on_ground = state.isOnGround ? 1 : 0;
    for (int w = 0; w < 4; ++w) s.wheels_with_contact[w ^ 1] = state.wheelsWithContact[w] ? 1 : 0;
    s.world_contact.has_contact = state.hasWorldContact ? 1 : 0;
    s.world_contact.contact_normal = state.hasWorldContact ? state.worldContactNormal : Vec3(0.0f, 0.0f, 0.0f);
    s.last_controls = state.prevControls;
    s.set_wheel_normals_mask = 0;
    if (state.hasWheelContactNormals) {
        s.set_wheel_normals_mask = 0x10;
        for (int w = 0; w < 4; ++w) {
            s.set_wheel_normals[w ^ 1] = state.wheelContactNormals[w];
            if (state.wheelsWithContact[w]) s.set_wheel_normals_mask |= static_cast<uint8_t>(1u << (w ^ 1));
        }
    }
    s.reported_flags = 0;  // every field above is the step's own value
    s.tick_count_since_update = 0;
    arena_->controls_[slot_] = state.controls;
    arena_->PushStates();
}

void Car::SetControls(const CarControls& controls) {
    arena_->controls_[slot_] = controls;
}

BallStateCompat Ball::GetState() const {
    arena_->PullStates();
    const BallState& s = arena_->ball_states_[0];
    BallStateCompat out;
    out.pos = s.pos;

    if (s.basis_authoritative) {
        out.rotMat.forward = s.basis_forward;
        out.rotMat.right = s.basis_right;
        out.rotMat.up = s.basis_up;
    } else {
        out.rotMat = RotMatFromQuat(s.rot);
    }
    out.vel = s.vel;
    out.angVel = s.ang_vel;
    out.tickCountSinceUpdate = s.tick_count_since_update;

    int last_slot = -1;
    for (int slot = 0; slot < arena_->car_count_; ++slot) {
        const BallHitInfo& hit = arena_->car_states_[slot].ball_hit_info;
        if (hit.is_valid && (last_slot < 0 || hit.tick_count_when_hit > arena_->car_states_[last_slot].ball_hit_info.tick_count_when_hit)) {
            last_slot = slot;
        }
    }
    if (last_slot >= 0) {
        out.ballHitInfo = BallHitInfoFromState(arena_->car_states_[last_slot].ball_hit_info);
        out.lastHitCarID = arena_->CompatIdFromDeviceId(arena_->car_states_[last_slot].ball_hit_info.hit_car_id);
    }
    return out;
}

void Ball::SetState(const BallStateCompat& state) {
    arena_->PullStates();
    BallState& s = arena_->ball_states_[0];
    s.pos = state.pos;

    s.basis_forward = state.rotMat.forward;
    s.basis_right = state.rotMat.right;
    s.basis_up = state.rotMat.up;
    s.basis_authoritative = 1;
    s.rot = rocketgpu::detail::BulletQuatFromBasisColumns(s.basis_forward, s.basis_right, s.basis_up);
    s.vel = state.vel;
    s.ang_vel = state.angVel;
    s.tick_count_since_update = 0;
    arena_->PushStates();
}
}
}

#include <rocketgpu/rocketsim_compat_events.h>

namespace rocketgpu {
namespace compat {
namespace {
bool GetShooterPasser(Arena* arena, Team team, Car*& shooter_out, bool find_passer, Car*& passer_out,
                      uint64_t max_shooter_ticks, uint64_t max_passer_ticks) {
    shooter_out = passer_out = nullptr;
    const uint64_t tick_count = arena->GetTickCount();
    uint64_t shooter_hit_tick = 0;
    for (Car* car : arena->cars) {
        if (car->team != team) continue;
        const BallHitInfoCompat hit = car->GetState().ballHitInfo;
        if (!hit.isValid) continue;
        if (hit.tickCountWhenHit + max_shooter_ticks >= tick_count) {
            if (!shooter_out || hit.tickCountWhenHit > shooter_hit_tick) {
                shooter_out = car;
                shooter_hit_tick = hit.tickCountWhenHit;
            }
        }
    }
    if (shooter_out && find_passer) {
        const uint64_t shoot_tick = shooter_hit_tick;
        uint64_t passer_hit_tick = 0;
        for (Car* car : arena->cars) {
            if (car->team != team) continue;
            const BallHitInfoCompat hit = car->GetState().ballHitInfo;
            if (!hit.isValid) continue;
            if (car == shooter_out) continue;
            if (hit.tickCountWhenHit + max_passer_ticks >= shoot_tick) {
                if (!passer_out || hit.tickCountWhenHit > passer_hit_tick) {
                    passer_out = car;
                    passer_hit_tick = hit.tickCountWhenHit;
                }
            }
        }
    }
    return shooter_out != nullptr;
}

Team OppositeTeam(Team t) { return t == Team::Blue ? Team::Orange : Team::Blue; }
Team TeamFromY(float y) { return y < 0.0f ? Team::Blue : Team::Orange; }
}

void GameEventTracker::Update(Arena* arena) {
    const bool scored = arena->IsBallScored();
    const float tickrate = arena->GetTickRate();
    const BallStateCompat ball = arena->ball->GetState();
    const uint64_t ball_update_count = ball.tickCountSinceUpdate;

    if (ball_update_count > last_ball_update_count_ || !autoStateSetDetection) {
        const uint64_t delta_ticks = ball_update_count - last_ball_update_count_;
        const float delta_time = static_cast<float>(delta_ticks) * arena->GetTickTime();

        if (scored && !ball_scored_last_) {
            Car* shooter;
            Car* passer;
            if (GetShooterPasser(arena, TeamFromY(-ball.pos.y), shooter, true, passer,
                                 static_cast<uint64_t>(config.goalMaxTouchTime * tickrate),
                                 static_cast<uint64_t>(config.passMaxTouchTime * tickrate))) {
                if (goal_callback_) goal_callback_(arena, shooter, passer, goal_user_info_);
            }
        } else {
            if (!ball_shot_) {
                if (shot_cooldown_ > 0.0f) {
                    shot_cooldown_ = std::max(shot_cooldown_ - delta_time, 0.0f);
                } else {
                    const float speed_sq = ball.vel.x * ball.vel.x + ball.vel.y * ball.vel.y + ball.vel.z * ball.vel.z;
                    if (speed_sq >= config.shotMinSpeed * config.shotMinSpeed) {
                        Team goal_team = Team::Blue;
                        if (arena->IsBallProbablyGoingIn(config.shotMinScoreTime, config.predScoreExtraMargin, &goal_team)) {
                            const Team shooter_team = OppositeTeam(goal_team);
                            const uint64_t shot_min_touch_delay_ticks = static_cast<uint64_t>(config.shotTouchMinDelay * tickrate);
                            Car* shooter;
                            Car* passer;
                            if (GetShooterPasser(arena, shooter_team, shooter, true, passer,
                                                 delta_ticks + shot_min_touch_delay_ticks,
                                                 static_cast<uint64_t>(config.passMaxTouchTime * tickrate))) {
                                const uint64_t ticks_since_hit = arena->GetTickCount() - shooter->GetState().ballHitInfo.tickCountWhenHit;
                                if (ticks_since_hit >= shot_min_touch_delay_ticks) {
                                    ball_shot_ = true;
                                    ball_shot_goal_team_ = goal_team;
                                    shot_cooldown_ = config.shotEventCooldown;
                                    if (shot_callback_) shot_callback_(arena, shooter, passer, shot_user_info_);
                                }
                            }
                        }
                    }
                }
            } else {
                const bool will_score = arena->IsBallProbablyGoingIn(config.shotMinScoreTime, config.predScoreExtraMargin);
                if (!will_score) {
                    Car* saver;
                    Car* unused;
                    if (GetShooterPasser(arena, ball_shot_goal_team_, saver, false, unused, delta_ticks, 0)) {
                        if (save_callback_) save_callback_(arena, saver, save_user_info_);
                    }
                    ball_shot_ = false;
                }
            }
        }
    } else if (ball_update_count == last_ball_update_count_) {
        return;
    } else {
        ResetPersistentInfo();
    }
    ball_scored_last_ = scored;
    last_ball_update_count_ = ball_update_count;
}

void GameEventTracker::ResetPersistentInfo() {
    ball_scored_last_ = false;
    ball_shot_ = false;
    shot_cooldown_ = 0.0f;
}
}
}

namespace rocketgpu {
namespace compat {
namespace {
float DistSq3(const Vec3& a, const Vec3& b) {
    const float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}
}

bool BallStateMatches(const BallStateCompat& a, const BallStateCompat& b, float margin_pos, float margin_vel, float margin_ang_vel) {
    return DistSq3(a.pos, b.pos) < (margin_pos * margin_pos) &&
           DistSq3(a.vel, b.vel) < (margin_vel * margin_vel) &&
           DistSq3(a.angVel, b.angVel) < (margin_ang_vel * margin_ang_vel);
}

BallPredTracker::BallPredTracker(Arena* arena, size_t num_pred_ticks) : numPredTicks(num_pred_ticks) {
    ballPredArena = Arena::Create(arena->mode_, arena->GetArenaConfig(), arena->GetTickRate());
    ballPredArena->SetMutatorConfig(arena->GetMutatorConfig());
    lastUpdateTickCount = 0;
    predData.reserve(numPredTicks);
    UpdatePredFromArena(arena);
}

BallPredTracker::~BallPredTracker() { delete ballPredArena; }

void BallPredTracker::UpdatePredFromArena(Arena* arena) {
    const BallStateCompat bs = arena->ball->GetState();
    const int ticks_since_last_update = static_cast<int>(arena->GetTickCount()) - lastUpdateTickCount;
    UpdatePredManual(bs, ticks_since_last_update);
}

void BallPredTracker::UpdatePredManual(const BallStateCompat& cur_ball_state, int ticks_since_last_update) {
    bool needs_full_repred;
    if (ticks_since_last_update >= 0 && static_cast<size_t>(ticks_since_last_update) < predData.size()) {
        if (BallStateMatches(predData[static_cast<size_t>(ticks_since_last_update)], cur_ball_state)) {
            needs_full_repred = false;
            if (ticks_since_last_update > 0) {
                predData.erase(predData.begin(), predData.begin() + ticks_since_last_update);
                ballPredArena->ball->SetState(predData.back());
                while (predData.size() < numPredTicks) {
                    ballPredArena->Step(1);
                    predData.push_back(ballPredArena->ball->GetState());
                }
            }
        } else {
            needs_full_repred = true;
        }
    } else {
        needs_full_repred = true;
    }
    if (needs_full_repred) {
        ForceUpdateAllPred(cur_ball_state);
    }
    lastUpdateTickCount += ticks_since_last_update;
}

void BallPredTracker::ForceUpdateAllPred(const BallStateCompat& initial_ball_state) {
    ballPredArena->ball->SetState(initial_ball_state);
    predData.resize(numPredTicks);
    predData[0] = initial_ball_state;
    for (size_t i = 1; i < numPredTicks; ++i) {
        ballPredArena->Step(1);
        predData[i] = ballPredArena->ball->GetState();
    }
}

BallStateCompat BallPredTracker::GetBallStateForTime(float pred_time) const {
    if (predData.empty()) {
        throw std::runtime_error("BallPredTracker::GetBallStateForTime(): Predicted ball data is empty, update prediction before calling");
    }

    const float raw = pred_time / ballPredArena->GetTickTime();
    const float hi = static_cast<float>(predData.size() - 1);
    const float clamped = raw < 0.0f ? 0.0f : (raw > hi ? hi : raw);
    return predData[static_cast<size_t>(static_cast<int>(clamped))];
}
}
}
