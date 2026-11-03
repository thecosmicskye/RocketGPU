

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <rocketgpu/rocketgpu.h>

namespace rocketgpu {
namespace compat {
enum class GameMode { SOCCAR = 0, HOOPS = 1, HEATSEEKER = 2, SNOWDAY = 3, DROPSHOT = 4, THE_VOID = 5 };

struct BoostPadConfig {
    Vec3 pos{};
    bool isBig = false;
};

struct ArenaConfig {
    bool noBallRot = true;
    bool nativeSseMath = false;
    bool useCustomBoostPads = false;
    std::vector<BoostPadConfig> customBoostPads;
};

struct RotMat {
    Vec3 forward{1.0f, 0.0f, 0.0f};
    Vec3 right{0.0f, 1.0f, 0.0f};
    Vec3 up{0.0f, 0.0f, 1.0f};

    static RotMat GetIdentity() { return RotMat{}; }
};

RotMat RotMatFromQuat(const Quat& q);
Quat QuatFromRotMat(const RotMat& m);

void SeedCarBasisFromRotMat(CarState& s, const RotMat& m);

RotMat RotMatFromCarState(const CarState& s);

RotMat PublicRotMatFromCarState(const CarState& s);

struct PhysState {
    Vec3 pos{};
    RotMat rotMat = RotMat::GetIdentity();
    Vec3 vel{};
    Vec3 angVel{};

    uint64_t tickCountSinceUpdate = 0;
};

struct BallHitInfoCompat {
    bool isValid = false;
    Vec3 relativePosOnBall{};
    Vec3 ballPos{};
    Vec3 extraHitVel{};
    uint64_t tickCountWhenHit = ~0ULL;
    uint64_t tickCountWhenExtraImpulseApplied = ~0ULL;
};

struct CarStateCompat : PhysState {
    bool isOnGround = true;
    bool wheelsWithContact[4] = {true, true, true, true};
    bool hasJumped = false;
    bool hasDoubleJumped = false;
    bool hasFlipped = false;
    Vec3 flipRelTorque{};
    // RocketSim v3 timer conventions, in both GetState and SetState (see CarState in rocketgpu.h):
    // jumpTime is v3 jump_ticks / 120 (ticks since the jump started, or ended), flipTime is seconds since flip
    // activity last changed. SetState rebuilds the step's internal clocks from them.
    float jumpTime = 0.0f;
    float flipTime = 0.0f;
    bool isFlipping = false;
    bool isJumping = false;
    float airTime = 0.0f;
    float airTimeSinceJump = 0.0f;
    float boost = 33.3f;
    float timeSinceBoosted = 0.0f;
    bool isBoosting = false;
    float boostingTime = 0.0f;
    bool isSupersonic = false;
    float supersonicTime = 0.0f;  // v3 supersonic_grace_timer
    float handbrakeVal = 0.0f;
    bool isAutoFlipping = false;
    float autoFlipTimer = 0.0f;
    float autoFlipTorqueScale = 0.0f;
    struct {
        bool hasContact = false;
        Vec3 contactNormal{0.0f, 0.0f, 0.0f};
    } worldContact;
    struct {
        uint32_t otherCarID = 0;
        float cooldownTimer = 0.0f;
    } carContact;
    bool isDemoed = false;
    float demoRespawnTimer = 0.0f;
    BallHitInfoCompat ballHitInfo{};
    CarControls lastControls{};
    // RLBot v5 fields (CarState in rocketgpu.h). GetState reports them; SetState ignores the derived ones (airState,
    // dodge*), which the next step recomputes. vel, angVel and isSupersonic left as GetState returned them keep the
    // step's own values (uncapped velocities, the game's supersonic flag).
    uint8_t airState = 0;
    float dodgeTimeout = -1.0f;
    float dodgeElapsed = 0.0f;
    float dodgeDirX = 0.0f;
    float dodgeDirY = 0.0f;
    struct {
        uint64_t tick = ~0ULL;
        Vec3 location{};
        Vec3 normal{};
    } latestTouch;
};

// A car's state in RocketSim v3's conventions, field for field (rocketsim crate, CarState in sim/car/car_state.rs):
// velocities before the speed caps, the step's own jump/flip flags, the grace-timer supersonic flag as of the end of
// the tick, wheels in v3's order. Car::GetStateV3 / SetStateV3 convert, so a v3 state goes in and comes out unchanged.
struct CarStateV3 {
    Vec3 pos{};
    RotMat rotMat = RotMat::GetIdentity();
    Vec3 vel{};
    Vec3 angVel{};
    CarControls controls{};  // controls for the next step (Car::SetControls)
    bool hasJumped = false;
    bool hasDoubleJumped = false;
    bool hasFlipped = false;
    Vec3 flipRelTorque{};
    uint32_t jumpTicks = 0;
    float flipTime = 0.0f;
    bool isFlipping = false;
    bool isJumping = false;
    bool jumpHoldBroken = false;  // tracked and stored, but RocketGPU's jump does not read it (a re-press resumes it)
    float airTime = 0.0f;
    float airTimeSinceJump = 0.0f;
    float boost = 33.3f;
    float handbrakeVal = 0.0f;
    bool isSupersonic = false;
    float supersonicGraceTimer = 0.0f;
    float timeSinceBoosted = 0.0f;
    float boostingTime = 0.0f;
    bool isBoosting = false;
    bool isAutoFlipping = false;
    float autoFlipTimer = 0.0f;
    float autoFlipTorqueScale = 0.0f;
    float bumpCooldownTimer = 0.0f;
    uint32_t bumpCooldownVictim = 0;  // victim's car id; 0 is None
    uint64_t lastExtraHitTick = ~0ULL;  // ~0 is None
    bool isDemoed = false;
    float demoRespawnTimer = 0.0f;
    bool isOnGround = true;
    bool wheelsWithContact[4] = {true, true, true, true};
    bool hasWorldContact = false;  // world_contact_normal is Some
    Vec3 worldContactNormal{};
    CarControls prevControls{};
    // Optional (SetStateV3 only, not v3 state): the game's contact normal per wheel for the wheels in wheelsWithContact,
    // i.e. the previous tick's wheel contacts, which the next step's sticky force and auto-roll read.
    bool hasWheelContactNormals = false;
    Vec3 wheelContactNormals[4]{};
};

struct BallStateCompat : PhysState {
    BallHitInfoCompat ballHitInfo{};
    uint32_t lastHitCarID = 0;
};

class Arena;

using GoalScoreEventFn = std::function<void(Arena* arena, Team scoring_team, void* user_info)>;
class Car;

using CarBumpEventFn = std::function<void(Arena* arena, Car* bumper, Car* victim, bool is_demo, void* user_info)>;

class Car {
public:
    uint32_t id = 0;
    Team team = Team::Blue;

    CarStateCompat GetState() const;
    void SetState(const CarStateCompat& state);
    CarStateV3 GetStateV3() const;
    void SetStateV3(const CarStateV3& state);
    void SetControls(const CarControls& controls);

private:
    friend class Arena;
    friend struct ArenaSerializer;
    Car(Arena* arena, int slot) : arena_(arena), slot_(slot) {}
    Arena* arena_ = nullptr;
    int slot_ = 0;
};

class Ball {
public:
    BallStateCompat GetState() const;
    void SetState(const BallStateCompat& state);

private:
    friend class Arena;
    explicit Ball(Arena* arena) : arena_(arena) {}
    Arena* arena_ = nullptr;
};

class BoostPad {
public:
    BoostPadConfig config;
    BoostPadState GetState() const;
    void SetState(const BoostPadState& state);
    Vec3 GetPos() const { return config.pos; }
    bool IsBig() const { return config.isBig; }

    int GetIndex() const { return official_index_; }

    int GetPadStateIndex() const { return rocketgpu_index_; }

private:
    friend class Arena;
    BoostPad(Arena* arena, int official_index, int rocketgpu_index)
        : arena_(arena), official_index_(official_index), rocketgpu_index_(rocketgpu_index) {}
    Arena* arena_ = nullptr;
    int official_index_ = 0;
    int rocketgpu_index_ = 0;
};

class Arena {
public:

    static Arena* Create(GameMode mode = GameMode::SOCCAR, float tick_rate = 120.0f);
    static Arena* Create(GameMode mode, const ArenaConfig& config, float tick_rate = 120.0f);
    const ArenaConfig& GetArenaConfig() const { return arena_config_; }
    ~Arena();

    Car* AddCar(Team team, const CarConfig& config = CarConfig{});
    void Step(int ticks_to_simulate = 1);
    void ResetToRandomKickoff(int seed = -1);
    // The bodies' side of a goal reset alone (EnvCommand::new_round_bodies), e.g. replaying a recorded goal.
    void StartNewRoundBodies();

    void SetRespawnRngSeed(uint64_t seed);
    // The respawn slot (0-3) the car's next demo respawn takes instead of the RNG's pick; -1 clears it.
    void SetRespawnSlotOverride(uint32_t car_id, int slot);
    // The car's next spawn (a demoed car's, when its demo_respawn_timer runs out) is a kickoff re-creation at kickoff
    // spot `spot` (0-4, the blue-side SoccarKickoffSpawn index; orange mirrored) at its CarConfig.spawn_z; -1 clears it.
    void SetKickoffSpawnOverride(uint32_t car_id, int spot);

    void SetCarVisitOrderMode(uint8_t mode);
    // game_order: keep the order as the game's vehicle order (mode 3) across demolitions, respawns and round resets.
    // List the cars in the world first; the unlisted slots follow.
    void SetCarVisitOrder(const std::vector<uint32_t>& compat_ids_in_visit_order, bool game_order = false);
    uint8_t GetCarVisitOrderMode() const;
    void SetGoalScoreCallback(GoalScoreEventFn callback_fn, void* user_info = nullptr);

    void SetCarBumpCallback(CarBumpEventFn callback_fn, void* user_info = nullptr);

    bool RemoveCar(uint32_t id);
    bool RemoveCar(Car* car);
    Car* GetCar(uint32_t id);
    const std::vector<Car*>& GetCars() const { return cars; }

    bool IsBallScored() const;

    bool IsBallProbablyGoingIn(float max_time = 2.0f, float extra_margin = 0.0f, rocketgpu::Team* goal_team_out = nullptr) const;

    Arena* Clone(bool copy_callbacks) const;

    std::vector<BoostPadState> GetBoostPadStates() const;

    const std::vector<BoostPad*>& GetBoostPads() const { return boost_pads_; }

    void SetMutatorConfig(const MutatorConfig& config);
    const MutatorConfig& GetMutatorConfig() const;
    float GetTickTime() const { return 1.0f / tick_rate_; }

    void SetBoostPadStates(const std::vector<BoostPadState>& states);

    void Serialize(std::vector<uint8_t>& out) const;
    static Arena* DeserializeNew(const std::vector<uint8_t>& data);
    void SerializeToFile(const std::string& path, bool write_version_check = true) const;
    static Arena* DeserializeNewFromFile(const std::string& path, bool version_check = true);

    uint32_t GetLastCarID() const { return next_car_id_ - 1; }

    Ball* ball = nullptr;
    std::vector<Car*> cars;

    float GetTickRate() const { return tick_rate_; }
    uint64_t GetTickCount() const { return tick_count_; }

    BatchSim& sim() { return *sim_; }

    // Dev only (full-state restore, src/rocketsim_compat_dev.cpp): env 0's state arrays as raw bytes. what:
    // "car_states", "ball_states", "env_states", "wheel_runtime", "solver_scratch", and the retained manifolds
    // "car_world_plane", "car_world_mesh", "car_car", "ball_car". DevSetRaw uploads without canonicalizing.
    std::vector<uint8_t> DevGetRaw(const std::string& what) const;
    void DevSetRaw(const std::string& what, const std::vector<uint8_t>& bytes);
    // Field layouts for those arrays: "Struct.field.path" -> (offset, size); "Struct" -> (0, sizeof).
    struct DevField {
        std::string name;
        size_t offset;
        size_t size;
    };
    static std::vector<DevField> DevLayouts();

private:
    friend class Car;
    friend class Ball;
    Arena() = default;
    void PullStates() const;
    void NormalizeCarVisitOrder();
    void PushCarState(int slot);
    void PushBallState();

    std::unique_ptr<BatchSim> sim_;
    float tick_rate_ = 120.0f;
    GameMode mode_ = GameMode::SOCCAR;
    ArenaConfig arena_config_;
    uint64_t tick_count_ = 0;
    int car_count_ = 0;

    mutable std::vector<CarState> car_states_;
    mutable std::vector<BallState> ball_states_;
    std::vector<CarControls> controls_;
    std::vector<CarConfig> configs_;
    std::vector<Team> teams_;
    mutable bool dirty_ = true;

    GoalScoreEventFn goal_callback_;
    void* goal_user_info_ = nullptr;
    std::vector<EnvState> env_states_;
    CarBumpEventFn bump_callback_;
    void* bump_user_info_ = nullptr;
    std::vector<uint64_t> bump_event_seen_tick_;
    void DrainBumpEvents();
    mutable std::vector<BoostPadState> pad_states_;
    std::vector<BoostPad*> boost_pads_;
    std::vector<uint8_t> slot_used_;
    std::vector<uint32_t> slot_to_id_;
    uint32_t next_car_id_ = 1;
    uint32_t CompatIdFromDeviceId(uint32_t device_id) const;
    uint32_t DeviceIdFromCompatId(uint32_t compat_id) const;
    friend struct ArenaSerializer;
    friend struct BallPredTracker;
    friend class DevBatch;
};
}
}
