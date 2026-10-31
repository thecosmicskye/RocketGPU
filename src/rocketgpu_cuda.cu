#include "rocketgpu/detail/cuda_api.hpp"

#if defined(ROCKETGPU_USE_CUDA)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "rocketgpu/detail/step.hpp"
#include "rocketgpu/detail/respawn_rng.hpp"

namespace rocketgpu {
namespace detail {
__device__ CarWorldMeshCandidateStage* g_rocketgpu_car_world_stage = nullptr;
__device__ int g_rocketgpu_car_world_stage_stride = 0;
namespace bullet_port {
__device__ EPA* g_rocketgpu_epa_pool = nullptr;
__device__ unsigned int* g_rocketgpu_epa_busy = nullptr;
__device__ unsigned int g_rocketgpu_epa_slots = 0;
}  // namespace bullet_port

// Thread -> env through BatchSim's car-count-grouped order (device traces address envs by thread, so keep identity).
ROCKETGPU_DEVICE void MapStepEnv(const SimConfig& cfg, int& env_idx) {
    if (cfg.env_step_order != nullptr) {
        env_idx = cfg.env_step_order[env_idx];
    }
}

ROCKETGPU_DEVICE void PrepareRespawnSlotsForEnv(
    int env, const SimConfig& cfg, const CarState* cars, const int32_t* counts,
    const EnvState* envs, RespawnRngState* rng, const int32_t* order,
    uint8_t* slots) {
    const int base = env * cfg.max_cars_per_env;
    const int count = counts[env];
    for (int i = 0; i < cfg.max_cars_per_env; ++i) slots[base + i] = 0xFFu;
    for (int order_idx = 0; order_idx < count; ++order_idx) {
        const int candidate = order ? order[base + order_idx] : order_idx;
        const int i = candidate >= 0 && candidate < count ? candidate : order_idx;
        const CarState& car = cars[base + i];
        if (!car.is_demoed || Max(0.0f, car.demo_respawn_timer - cfg.tick_dt) > 0.0f) continue;
        const uint8_t slot = static_cast<uint8_t>(NextRespawnRng(rng[env]) % 4);
        slots[base + i] = slot;
    }
}

ROCKETGPU_GLOBAL void PrepareRespawnSlotsKernel(
    const __grid_constant__ SimConfig cfg, const CarState* cars, const int32_t* counts,
    const EnvState* envs, RespawnRngState* rng, const int32_t* order,
    uint8_t* slots) {
    const int env = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (env >= cfg.env_count) return;
    PrepareRespawnSlotsForEnv(env, cfg, cars, counts, envs, rng, order,
                             slots);
}

void LaunchPrepareRespawnSlots(
    const SimConfig& cfg, const CarState* cars, const int32_t* counts,
    const EnvState* envs, RespawnRngState* rng, const int32_t* order,
    uint8_t* slots, cudaStream_t stream) {
    constexpr int threads = 128;
    PrepareRespawnSlotsKernel<<<(cfg.env_count + threads - 1) / threads, threads, 0, stream>>>(
        cfg, cars, counts, envs, rng, order, slots);
    CheckCuda(cudaGetLastError(), "PrepareRespawnSlotsKernel launch");
}

// The game reports a body's state in uu as its Bullet value times 50, and quantizes that product at tick start. An
// imported uu value is restored to the Bullet value whose product with 50 gives it back exactly (the nearest float to
// value / 50 that does), so the readback and the tick-start quantize see the recorded value bit for bit.
ROCKETGPU_HD float ImportedLinearScalarBt(float value) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kBTToUU = 50.0f;
    const float bt = value * kUUToBT;
    if (bt * kBTToUU == value) {
        return bt;
    }
    const float up = nextafterf(bt, INFINITY);
    const float down = nextafterf(bt, -INFINITY);
    if (up * kBTToUU == value) {
        return up;
    }
    if (down * kBTToUU == value) {
        return down;
    }
    return bt;
}

ROCKETGPU_HD Vec3 ImportedCarLinearVec3Bt(const Vec3& value) {
    return Vec3(
        ImportedLinearScalarBt(value.x),
        ImportedLinearScalarBt(value.y),
        ImportedLinearScalarBt(value.z));
}

ROCKETGPU_HD Vec3 ImportedBallLinearVec3Bt(const Vec3& value) {
    return ImportedCarLinearVec3Bt(value);
}

ROCKETGPU_GLOBAL void ResetToKickoffKernel(
    const __grid_constant__ SimConfig cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    CarControls* controls,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    uint64_t seed,
    RespawnRngState* respawn_rng,
    uint8_t kickoff_seed_mode) {
    const int env_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (env_idx >= cfg.env_count) {
        return;
    }
    ResetEnvToKickoff(env_idx, cfg, teams, env_car_count, cars, controls, balls, pads, seed, false, env_states,
                      kickoff_seed_mode, respawn_rng, nullptr);
}

ROCKETGPU_DEVICE void ApplyEnvCommandForEnv(
    int env_idx,
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    CarControls* controls,
    WorldSolverScratch* solver_scratch,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    const EnvCommand& command,
    RespawnRngState* respawn_rng) {
    ApplyEnvCommand(env_idx, cfg, teams, env_car_count, cars, controls, balls, env_states, pads, command, respawn_rng, nullptr);
    if (solver_scratch != nullptr && command.reset_to_kickoff) {
        for (int manifold_idx = 0; manifold_idx < kMaxBallWorldMeshPersistentManifolds; ++manifold_idx) {
            solver_scratch[env_idx].ball_world_mesh_manifolds[manifold_idx] = BallMeshPersistentManifold{};
        }
        for (int manifold_idx = 0; manifold_idx < 4; ++manifold_idx) {
            solver_scratch[env_idx].ball_world_plane_manifolds[manifold_idx] = BallMeshPersistentManifold{};
        }
        solver_scratch[env_idx].retained_reset_pending = 1;
    }
}

ROCKETGPU_GLOBAL void ApplyEnvCommandsKernel(
    const __grid_constant__ SimConfig cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    CarControls* controls,
    WorldSolverScratch* solver_scratch,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    const EnvCommand* env_commands,
    int32_t command_count,
    RespawnRngState* respawn_rng) {
    const int env_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (env_idx >= command_count) {
        return;
    }
    ApplyEnvCommandForEnv(env_idx, cfg, teams, env_car_count, cars, controls, solver_scratch, balls, env_states, pads,
                          env_commands[env_idx], respawn_rng);
}

ROCKETGPU_DEVICE void StageStreamControlsForEnv(
    int env_idx, const SimConfig& cfg, CarControls* controls,
    const CarControls* controls_by_tick, int32_t control_stride, int tick_idx) {
    const int base = env_idx * cfg.max_cars_per_env;
    const CarControls* src = controls_by_tick +
        static_cast<size_t>(tick_idx) * static_cast<size_t>(control_stride) + static_cast<size_t>(base);
    for (int i = 0; i < cfg.max_cars_per_env; ++i) {
        controls[base + i] = src[i];
    }
}

ROCKETGPU_GLOBAL void CanonicalizeImportedCarsKernel(CarState* cars, int32_t count) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= count) {
        return;
    }
    constexpr float kCarBTToUU = 50.0f;
    const uint8_t kept = cars[idx].tick_count_since_update != 0 ? cars[idx].RestoreWorkingFields() : 0;
    cars[idx].ResetTrainingHistory();
    cars[idx].pos_bt = ImportedCarLinearVec3Bt(cars[idx].pos);
    cars[idx].vel_bt = ImportedCarLinearVec3Bt(cars[idx].vel);
    cars[idx].pos = cars[idx].pos_bt * kCarBTToUU;
    cars[idx].vel = cars[idx].vel_bt * kCarBTToUU;
    CanonicalizeImportedCarOrientation(cars[idx]);
    cars[idx].ReReportFields(kept);
}

ROCKETGPU_GLOBAL void CanonicalizeImportedBallsKernel(BallState* balls, int32_t count) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= count) {
        return;
    }
    constexpr float kBTToUU = 50.0f;
    balls[idx].pos_bt = ImportedBallLinearVec3Bt(balls[idx].pos);
    balls[idx].vel_bt = ImportedBallLinearVec3Bt(balls[idx].vel);
    balls[idx].pos = balls[idx].pos_bt * kBTToUU;
    balls[idx].vel = balls[idx].vel_bt * kBTToUU;
    balls[idx].sleeping = 2;
    balls[idx].deactivation_time = 0.0f;
}

// Tick start: a thread per car slot, a block holding whole envs. Each env's first thread applies the env's
// command, stream controls and respawn slots; then every car, and the ball and pads, start the tick in parallel.
ROCKETGPU_GLOBAL void PreTickKernel(
    const __grid_constant__ SimConfig cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    WorldSolverScratch* solver_scratch,
    CarControls* controls,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    const CarControls* controls_by_tick,
    int32_t control_stride,
    const EnvCommand* env_commands_by_tick,
    int32_t env_command_stride,
    const uint8_t* respawn_slots,
    uint8_t* writable_respawn_slots,
    RespawnRngState* respawn_rng,
    const int32_t* respawn_order,
    uint64_t seed,
    int32_t tick_idx,
    bool pad_pre_tick = true) {
    const int slots = cfg.max_cars_per_env;
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int env_idx = tid / slots;
    const int slot = tid - env_idx * slots;
    const bool active = env_idx < cfg.env_count;
    if (active) {
        MapStepEnv(cfg, env_idx);
    }
    if (active && slot == 0) {
        if (env_commands_by_tick) {
            const EnvCommand command = env_commands_by_tick[
                static_cast<size_t>(tick_idx) * static_cast<size_t>(env_command_stride) + static_cast<size_t>(env_idx)];
            ApplyEnvCommandForEnv(env_idx, cfg, teams, env_car_count, cars, controls, solver_scratch,
                                  balls, env_states, pads, command, respawn_rng);
        }
        if (controls_by_tick) {
            StageStreamControlsForEnv(env_idx, cfg, controls, controls_by_tick, control_stride, tick_idx);
        }
        if (respawn_rng) {
            PrepareRespawnSlotsForEnv(env_idx, cfg, cars, env_car_count, env_states,
                respawn_rng, respawn_order, writable_respawn_slots);
        }
        StepEnvPreTickEnv(env_idx, cfg, teams, env_car_count, cars, solver_scratch, controls, balls, env_states, pads,
                          EnvCommand{}, nullptr);
    }
    __syncthreads();
    if (active) {
        const int count = env_car_count[env_idx];
        if (slot < count) {
            StepEnvPreTickCar(env_idx, slot, cfg, teams, cars, solver_scratch, env_states[env_idx], respawn_slots, seed);
        }
        if (slot == 0) {
            StepEnvPreTickBall(env_idx, cfg, solver_scratch, balls);
        }
    }
    // The block's pads, a thread per pad: the pads are independent, and consecutive threads touch consecutive pads.
    if (pad_pre_tick && cfg.enable_boost_pads) {
        const int pad_count = BoostPadCount(cfg);
        const int block_env0 = static_cast<int>(blockIdx.x * blockDim.x) / slots;
        const int block_envs = Min(static_cast<int>(blockDim.x) / slots, cfg.env_count - block_env0);
        for (int q = static_cast<int>(threadIdx.x); q < block_envs * pad_count; q += static_cast<int>(blockDim.x)) {
            const int local_env = q / pad_count;
            int pad_env = block_env0 + local_env;
            MapStepEnv(cfg, pad_env);
            if (env_car_count[pad_env] > 0) {
                PreTickBoostPad(pads[pad_env * pad_count + (q - local_env * pad_count)], cfg.tick_dt);
            }
        }
    }
}

// The rest of the tick, after PreTickKernel and the staged wheel rays, `lanes` threads per env (a power of two):
// with more than one, the cars' vehicle updates (when independent) and car-world rows run a lane per car, and the
// rest of the world step on the env's first lane, as the sequential step would.
// Not inlined, nor is StepEnv, so StepKernel's stack is the larger of their frames rather than the sum.
__device__ __noinline__ void BuildCarWorldRowsForLane(int env_idx, int lane, int lanes, int count, const SimConfig& cfg,
    CarState* cars, const CarConfig* car_configs, WorldSolverScratch* solver_scratch, WorldSolverScratchRowPool row_pool,
    bullet_port::btGjkEpaScratch* gjk_scratch) {
    for (int slot = lane; slot < count; slot += lanes) {
        BuildCarWorldRowsForCar(env_idx, slot, cfg, cars, car_configs, solver_scratch[env_idx], row_pool.view(env_idx),
                                gjk_scratch);
    }
}

// One-lane step on a thread-local copy of the env's cars, wheels, ball and env state. Local memory is interleaved
// across a warp's threads, so the step's many accesses to these coalesce; the AoS global copies are read and written
// once. StepEnv indexes them as [env_idx * stride + slot] and [env_idx], so it gets bases shifted to the locals.
// Used only when the envs outnumber the threads the GPU holds at once: there the step is throughput bound and the
// fewer global accesses pay (RTX 5090 1v1: +7% at 49k envs, even at 32k, -3% at 16k, where the copies only add latency).
constexpr int kLocalStepMaxSlots = 2;

__device__ __noinline__ void StepEnvLocalState(
    int env_idx, int count, const SimConfig& cfg, Team* teams, int32_t* env_car_count, CarState* cars,
    WheelRuntimeState* wheel_runtime_states, WorldSolverScratch* solver_scratch, const WorldSolverScratchRowPool& row_pool,
    CarControls* controls, CarConfig* car_configs, BallState* balls, EnvState* env_states, BoostPadState* pads,
    uint64_t seed) {
    const int stride = cfg.max_cars_per_env;
    const long long base = static_cast<long long>(env_idx) * stride;
    CarState local_cars[kLocalStepMaxSlots];
    WheelRuntimeState local_wheels[kLocalStepMaxSlots];
    BallState local_ball = balls[env_idx];
    EnvState local_env = env_states[env_idx];
    for (int k = 0; k < count; ++k) {
        local_cars[k] = cars[base + k];
        local_wheels[k] = wheel_runtime_states[base + k];
    }
    StepEnv(env_idx, cfg, teams, env_car_count, local_cars - base, local_wheels - base, solver_scratch, row_pool,
            controls, car_configs, &local_ball - env_idx, &local_env - env_idx, pads, nullptr, seed, 1,
            kStepEnvPhasePost, 0, kWorldStepWhole);
    for (int k = 0; k < count; ++k) {
        cars[base + k] = local_cars[k];
        wheel_runtime_states[base + k] = local_wheels[k];
    }
    balls[env_idx] = local_ball;
    env_states[env_idx] = local_env;
}

ROCKETGPU_GLOBAL void StepKernel(
    const __grid_constant__ SimConfig cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch,
    WorldSolverScratchRowPool row_pool,
    CarControls* controls,
    CarConfig* car_configs,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    uint64_t seed,
    int lanes,
    bullet_port::btGjkEpaScratch* gjk_pool,
    bool local_state) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int lane = tid & (lanes - 1);
    int env_idx = tid / lanes;
    const bool active = env_idx < cfg.env_count;
    if (active) {
        MapStepEnv(cfg, env_idx);
    }
    const int stride = cfg.max_cars_per_env;
    const int count = active ? env_car_count[env_idx] : 0;
    const bool apart = lanes > 1;
    // Physics events must come out in env step order, so their car-world rows stay in the world step.
    const bool rows_apart = apart && !cfg.mutator.physics_events.windows;
    if (apart && active && VehicleUpdatesIndependent(cfg, solver_scratch[env_idx].wheel_ray_stage, count)) {
        for (int slot = lane; slot < count; slot += lanes) {
            uint32_t pushed_cars_mask = 0u;
            StepEnvVehicleUpdateCar(env_idx, slot, count, cfg, cars, wheel_runtime_states, car_configs, balls,
                                    controls[env_idx * stride + slot], solver_scratch[env_idx].wheel_ray_stage,
                                    pushed_cars_mask);
        }
    }
    __syncwarp();
    if (active && local_state) {
        StepEnvLocalState(env_idx, count, cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch,
                          row_pool, controls, car_configs, balls, env_states, pads, seed);
        return;
    }
    if (active && lane == 0) {
        StepEnv(env_idx, cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls,
                car_configs, balls, env_states, pads, nullptr, seed, 1,
                apart ? kStepEnvPhasePostVehiclesDone : kStepEnvPhasePost, 0,
                rows_apart ? kWorldStepContacts : kWorldStepWhole);
    }
    if (!rows_apart) {
        return;
    }
    __syncwarp();
    if (active) {
        BuildCarWorldRowsForLane(env_idx, lane, lanes, count, cfg, cars, car_configs, solver_scratch, row_pool, gjk_pool + tid);
    }
    __syncwarp();
    if (active && lane == 0) {
        StepEnv(env_idx, cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls,
                car_configs, balls, env_states, pads, nullptr, seed, 1, kStepEnvPhasePostVehiclesDone, 0,
                kWorldStepRows);
    }
}

// Each env's VehicleUpdatesIndependent, a thread per env, after the staged wheel rays: StagedCarPreKernel and the
// contacts kernels read it, so the envs whose updates are not independent can step apart from StagedCarPreKernel.
ROCKETGPU_GLOBAL void StagedVehicleIndependenceKernel(
    const __grid_constant__ SimConfig cfg, const int32_t* env_car_count, WorldSolverScratch* solver_scratch) {
    int env_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (env_idx >= cfg.env_count) {
        return;
    }
    MapStepEnv(cfg, env_idx);
    solver_scratch[env_idx].vehicle_updates_independent =
        VehicleUpdatesIndependent(cfg, solver_scratch[env_idx].wheel_ray_stage, env_car_count[env_idx]) ? 1 : 0;
}

// The staged step: StepKernel's lanes as kernels, each over all envs. A thread per car slot runs the independent
// vehicle updates and later the car-world rows; a thread per env runs the world step's contact stages, then its
// solve and the tick's end. Each env sees the same calls in the same order as StepKernel with lanes > 1.
ROCKETGPU_GLOBAL void StagedCarPreKernel(
    const __grid_constant__ SimConfig cfg, int32_t* env_car_count, CarState* cars, WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch, CarControls* controls, CarConfig* car_configs, BallState* balls) {
    const int stride = cfg.max_cars_per_env;
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    int env_idx = static_cast<int>(tid / stride);
    const int slot = static_cast<int>(tid - static_cast<long long>(env_idx) * stride);
    if (env_idx >= cfg.env_count) {
        return;
    }
    MapStepEnv(cfg, env_idx);
    const int count = env_car_count[env_idx];
    const bool independent = solver_scratch[env_idx].vehicle_updates_independent != 0;
    if (slot >= count || !independent) {
        return;
    }
    uint32_t pushed_cars_mask = 0u;
    const int car_idx = env_idx * stride + slot;
    StepEnvVehicleUpdateCar(env_idx, slot, count, cfg, cars, wheel_runtime_states, car_configs, balls,
                            controls[car_idx], solver_scratch[env_idx].wheel_ray_stage, pushed_cars_mask);
    WorldSolverScratch& scratch = solver_scratch[env_idx];
    WorldStepPrestepCar(cars[car_idx], car_configs[car_idx], wheel_runtime_states ? &wheel_runtime_states[car_idx] : nullptr,
                        cfg.mutator, cfg.tick_dt / static_cast<float>(DebugWorldSubsteps()),
                        scratch.pre_contact_cars[slot], scratch.deferred_linear_deltas[slot],
                        scratch.deferred_angular_deltas[slot]);
}

// The tick's per-car end after the env's world step (StepEnvPostCar), a thread per car slot.
ROCKETGPU_GLOBAL void StagedCarPostKernel(
    const __grid_constant__ SimConfig cfg, int32_t* env_car_count, CarState* cars, WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch, CarControls* controls, CarConfig* car_configs) {
    const int stride = cfg.max_cars_per_env;
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    int env_idx = static_cast<int>(tid / stride);
    const int slot = static_cast<int>(tid - static_cast<long long>(env_idx) * stride);
    if (env_idx >= cfg.env_count) {
        return;
    }
    MapStepEnv(cfg, env_idx);
    if (slot >= env_car_count[env_idx]) {
        return;
    }
    StepEnvPostCar(env_idx, slot, cfg, cars, wheel_runtime_states, solver_scratch, car_configs,
                   controls[env_idx * stride + slot], true);
}

// The pads' tick start (PreTickBoostPad), a thread per pad, after PreTickKernel's env commands (a kickoff resets the
// pads). Only the env tail and the pads' ends read the pads after it, so it runs on a side branch.
ROCKETGPU_GLOBAL void StagedPadPreKernel(const __grid_constant__ SimConfig cfg, const int32_t* env_car_count, BoostPadState* pads) {
    const int pad_count = BoostPadCount(cfg);
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    int env_idx = static_cast<int>(tid / pad_count);
    if (env_idx >= cfg.env_count) {
        return;
    }
    const int p = static_cast<int>(tid - static_cast<long long>(env_idx) * pad_count);
    MapStepEnv(cfg, env_idx);
    if (env_car_count[env_idx] > 0) {
        PreTickBoostPad(pads[env_idx * pad_count + p], cfg.tick_dt);
    }
}

// The tick's env end (StepEnvTail), a thread per env; with pads_apart the pads' ends follow a thread per pad.
ROCKETGPU_GLOBAL void StagedEnvTailKernel(
    const __grid_constant__ SimConfig cfg, int32_t* env_car_count, CarState* cars, BallState* balls, EnvState* env_states, BoostPadState* pads,
    bool pads_apart) {
    int env_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (env_idx >= cfg.env_count) {
        return;
    }
    MapStepEnv(cfg, env_idx);
    StepEnvTail(env_idx, env_car_count[env_idx], cfg, cars, balls[env_idx], env_states[env_idx], pads, !pads_apart);
}

// Whether no car can be in two pads' pickup cylinders at once (the standard layout's pads are 1283+ uu apart), so a
// pad's pickup never meets another's on the same car and the pads' ends run in any order.
inline bool BoostPadsApart(const SimConfig& cfg) {
    const int n = BoostPadCount(cfg);
    for (int a = 0; a < n; ++a) {
        for (int b = a + 1; b < n; ++b) {
            const Vec3 pa = BoostPadLoc(cfg, a);
            const Vec3 pb = BoostPadLoc(cfg, b);
            const double reach = (BoostPadIsBig(cfg, a) ? 208.0 : 176.0) + (BoostPadIsBig(cfg, b) ? 208.0 : 176.0) + 8.0;
            const double dx = static_cast<double>(pa.x) - pb.x;
            const double dy = static_cast<double>(pa.y) - pb.y;
            if (dx * dx + dy * dy <= reach * reach) {
                return false;
            }
        }
    }
    return true;
}

// The pads' ends (PostTickBoostPad), a thread per pad, when BoostPadsApart.
ROCKETGPU_GLOBAL void StagedPadPostKernel(const __grid_constant__ SimConfig cfg, int32_t* env_car_count, CarState* cars, BoostPadState* pads) {
    const int pad_count = BoostPadCount(cfg);
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    int env_idx = static_cast<int>(tid / pad_count);
    if (env_idx >= cfg.env_count) {
        return;
    }
    const int p = static_cast<int>(tid - static_cast<long long>(env_idx) * pad_count);
    MapStepEnv(cfg, env_idx);
    if (env_car_count[env_idx] > 0) {
        PostTickBoostPad(env_idx, cfg, cars, pads, p);
    }
}

// A kernel per world part (a compile-time constant), so each holds only its part's code and registers.
template <int kWorldPart>
ROCKETGPU_GLOBAL void StagedWorldKernel(
    const __grid_constant__ SimConfig cfg, Team* teams, int32_t* env_car_count, CarState* cars, WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch, WorldSolverScratchRowPool row_pool, CarControls* controls,
    CarConfig* car_configs, BallState* balls, EnvState* env_states, BoostPadState* pads, uint64_t seed) {
    int env_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (env_idx >= cfg.env_count) {
        return;
    }
    MapStepEnv(cfg, env_idx);
    if (WorldPartIsContacts(kWorldPart)) {
        // Envs whose vehicle updates are independent ran them (and the prestep) a thread per car: CarsApart.
        // The rest run them here in visit order, then the prestep.
        const bool independent = solver_scratch[env_idx].vehicle_updates_independent != 0;
        if (independent != (kWorldPart == kWorldStepContactsCarsApart)) {
            return;
        }
    }
    InternalControlsAccessor accessor;
    accessor.controls = controls;
    accessor.base = env_idx * cfg.max_cars_per_env;
    // kWorldStepContacts runs only envs whose vehicle updates were not independent: kStepEnvPhasePost runs them
    // without testing that again.
    StepEnvGeneric(env_idx, cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls,
                   accessor, NoopEnvCommandAccessor{}, car_configs, balls, env_states, pads, nullptr, seed, 1, nullptr,
                   kWorldPart == kWorldStepContacts ? kStepEnvPhasePost : kStepEnvPhasePostVehiclesDone, 0, kWorldPart);
}

// Grid-stride over car slots: each thread holds one GJK/EPA scratch from the pool.
ROCKETGPU_GLOBAL void StagedCarWorldRowsKernel(
    const __grid_constant__ SimConfig cfg, int32_t* env_car_count, CarState* cars, const CarConfig* car_configs,
    WorldSolverScratch* solver_scratch, WorldSolverScratchRowPool row_pool, bullet_port::btGjkEpaScratch* gjk_pool) {
    const int stride = cfg.max_cars_per_env;
    const long long first = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long car_slots = static_cast<long long>(cfg.env_count) * stride;
    for (long long tid = first; tid < car_slots; tid += static_cast<long long>(gridDim.x) * blockDim.x) {
        int env_idx = static_cast<int>(tid / stride);
        const int slot = static_cast<int>(tid - static_cast<long long>(env_idx) * stride);
        MapStepEnv(cfg, env_idx);
        if (slot < env_car_count[env_idx]) {
            BuildCarWorldRowsForCar(env_idx, slot, cfg, cars, car_configs, solver_scratch[env_idx],
                                    row_pool.view(env_idx), gjk_pool + first);
        }
    }
}

// The car-world rows a thread per car, in two builds: cars with no usable arena mesh in reach (most of them) and
// demoed cars take the kMeshesOutOfReach build, which compiles the mesh query out, the rest the full one. The split
// kernel lists each kind so their warps run one build; every car writes only its own rows, so the list orders do not
// matter. row_car_count: the two list lengths, zeroed by the car-world staging of the tick.
ROCKETGPU_GLOBAL void StagedCarWorldRowsSplitKernel(
    const __grid_constant__ SimConfig cfg, const int32_t* env_car_count, const CarState* cars, const CarConfig* car_configs,
    const WorldSolverScratch* solver_scratch, int32_t* row_cars, int32_t* row_car_count) {
    const int stride = cfg.max_cars_per_env;
    const long long car_slots = static_cast<long long>(cfg.env_count) * stride;
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    int kind = -1;
    int32_t car_idx = 0;
    if (tid < car_slots) {
        int env_idx = static_cast<int>(tid / stride);
        const int slot = static_cast<int>(tid - static_cast<long long>(env_idx) * stride);
        MapStepEnv(cfg, env_idx);
        if (slot < env_car_count[env_idx]) {
            car_idx = env_idx * stride + slot;
            const bool out_of_reach =
                solver_scratch[env_idx].pre_contact_cars[slot].is_demoed || cars[car_idx].just_respawned ||
                CarWorldUsableMeshesOutOfReach(cars[car_idx], car_configs[car_idx], cfg.arena_mesh_bvh_quantization,
                                               cfg.arena_mesh_bvh_quantization_count, cfg.arena_mesh_bvh_meshes,
                                               cfg.arena_mesh_bvh_mesh_count, cfg.arena_mesh_bvh_nodes,
                                               cfg.arena_mesh_bvh_node_count,
                                               cfg.tick_dt / static_cast<float>(DebugWorldSubsteps()));
            kind = out_of_reach ? 0 : 1;
        }
    }
    const unsigned lane = threadIdx.x & 31u;
    for (int k = 0; k < 2; ++k) {
        const unsigned mask = __ballot_sync(0xffffffffu, kind == k);
        if (mask == 0) {
            continue;
        }
        const int leader = __ffs(static_cast<int>(mask)) - 1;
        int base = 0;
        if (static_cast<int>(lane) == leader) {
            base = atomicAdd(&row_car_count[k], __popc(mask));
        }
        base = __shfl_sync(0xffffffffu, base, leader);
        if (kind == k) {
            row_cars[k * car_slots + base + __popc(mask & ((1u << lane) - 1u))] = car_idx;
        }
    }
}

// Grid-stride over one list of StagedCarWorldRowsSplitKernel: each thread holds one GJK/EPA scratch from the pool.
template <bool kMeshesOutOfReach>
ROCKETGPU_GLOBAL void StagedCarWorldRowsListKernel(
    const __grid_constant__ SimConfig cfg, CarState* cars, const CarConfig* car_configs, WorldSolverScratch* solver_scratch,
    WorldSolverScratchRowPool row_pool, const int32_t* row_cars, const int32_t* row_car_count,
    bullet_port::btGjkEpaScratch* gjk_pool) {
    const int stride = cfg.max_cars_per_env;
    const long long car_slots = static_cast<long long>(cfg.env_count) * stride;
    const int list = kMeshesOutOfReach ? 0 : 1;
    const long long count = row_car_count[list];
    const int32_t* const list_cars = row_cars + list * car_slots;
    const long long first = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (long long k = first; k < count; k += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int32_t car_idx = list_cars[k];
        const int env_idx = car_idx / stride;
        const int slot = car_idx - env_idx * stride;
        BuildCarWorldRowsForCar<kMeshesOutOfReach>(env_idx, slot, cfg, cars, car_configs, solver_scratch[env_idx],
                                                   row_pool.view(env_idx), kMeshesOutOfReach ? nullptr : gjk_pool + first);
    }
}

ROCKETGPU_GLOBAL void StageWheelRaysKernel(
    const __grid_constant__ SimConfig cfg,
    const CarState* cars,
    const CarConfig* car_configs,
    const WheelRuntimeState* wheel_runtime_states,
    const BallState* balls,
    const EnvState* env_states,
    WorldSolverScratch* solver_scratch,
    const int32_t* env_car_count) {
    const int rays_per_env = 4 * cfg.max_cars_per_env;
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int env_idx = tid / rays_per_env;
    if (env_idx >= cfg.env_count || solver_scratch == nullptr) {
        return;
    }
    const int slot = tid - env_idx * rays_per_env;
    StageWheelRaysForEnv(slot, env_idx, cfg, cars, car_configs, wheel_runtime_states, balls,
                         env_states[env_idx], solver_scratch[env_idx],
                         env_idx * cfg.max_cars_per_env, env_car_count[env_idx]);
}

// Car-world mesh staging. The prefilter runs a thread per car slot: it writes the stage header, marks the meshes
// the car's query AABB may touch, and lists each (car, overlapping mesh) pair under its mesh. The staging kernel
// then runs a thread per listed pair, mesh-major so a warp walks one mesh's BVH, twice: the gather pass lists the
// pair's triangles that pass the plane test, CarWorldTriangleHitsKernel runs their GJK/EPA a thread per
// car-triangle, and the merge pass folds the hits into the pair's candidates in walk order. Every pair writes only
// its own entry, so the list orders do not affect the result.
ROCKETGPU_GLOBAL void ZeroInt32Kernel(int32_t* data, int count) {
    for (int i = static_cast<int>(threadIdx.x); i < count; i += static_cast<int>(blockDim.x)) {
        data[i] = 0;
    }
}

ROCKETGPU_GLOBAL void PrefilterCarWorldMeshCandidatesKernel(
    const __grid_constant__ SimConfig cfg,
    const CarState* cars,
    const CarConfig* car_configs,
    const int32_t* env_car_count,
    CarWorldMeshCandidateStage* stages,
    int32_t* mesh_work,
    int32_t* mesh_work_count,
    CarState* car_snapshots) {
    // Every car tests the same mesh infos: read them into shared memory once per block.
    __shared__ alignas(ArenaMeshBvhMeshInfo) unsigned char mesh_storage[kCarWorldMeshStageMeshes * sizeof(ArenaMeshBvhMeshInfo)];
    ArenaMeshBvhMeshInfo* const meshes = reinterpret_cast<ArenaMeshBvhMeshInfo*>(mesh_storage);
    if (cfg.arena_mesh_bvh_meshes != nullptr && static_cast<int>(threadIdx.x) < Min(cfg.arena_mesh_bvh_mesh_count, kCarWorldMeshStageMeshes)) {
        memcpy(&meshes[threadIdx.x], &cfg.arena_mesh_bvh_meshes[threadIdx.x], sizeof(ArenaMeshBvhMeshInfo));
    }
    __syncthreads();
    const int stride = cfg.max_cars_per_env;
    const long long car_slots = static_cast<long long>(cfg.env_count) * stride;
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= car_slots) {
        return;
    }
    const int car_idx = static_cast<int>(tid);
    const int env_idx = car_idx / stride;
    const int slot = car_idx - env_idx * stride;
    CarWorldMeshCandidateStage& stage = stages[car_idx];
    stage.valid = 0;
    if (slot >= env_car_count[env_idx]) {
        return;
    }
    const CarState& car = cars[car_idx];
    if (car.is_demoed || car.just_respawned || !cfg.mutator.arena_collision ||
        cfg.arena_mesh_tris == nullptr || cfg.arena_mesh_tri_count <= 0) {
        return;
    }
    const CarConfig& car_cfg = car_configs[car_idx];
    const bool have_bvh_traversal =
        cfg.arena_mesh_bvh_quantization != nullptr && cfg.arena_mesh_bvh_meshes != nullptr &&
        cfg.arena_mesh_bvh_nodes != nullptr && cfg.arena_mesh_bvh_mesh_count > 0 && cfg.arena_mesh_bvh_node_count > 0 &&
        cfg.arena_mesh_bvh_mesh_count <= kCarWorldMeshStageMeshes;
    if (!have_bvh_traversal) {
        return;
    }
    const CarWorldMeshQueryFrame frame = ComputeCarWorldMeshQueryFrame(car, car_cfg);
    uint32_t mesh_mask = 0;
    int staged_meshes = 0;
    for (int mesh = 0; mesh < cfg.arena_mesh_bvh_mesh_count; ++mesh) {
        const ArenaMeshBvhMeshInfo& mesh_info = meshes[mesh];
        if (CarWorldMeshInfoUsable(mesh_info, cfg.arena_mesh_bvh_quantization_count, cfg.arena_mesh_bvh_node_count) &&
            CarWorldMeshAabbMayOverlap(mesh_info, frame.aabb_min_bt, frame.aabb_max_bt)) {
            mesh_mask |= 1u << mesh;
            ++staged_meshes;
        }
    }
    // No mesh, or more meshes than stage entries: valid stays 0 and the step walks this car itself. The empty walk
    // is cheap, and it saves the fingerprint write here and its check in the step.
    if (staged_meshes == 0 || staged_meshes > kCarWorldMeshStageSlots) {
        return;
    }
    // The later staging passes read this copy of each listed car, so the contacts kernels may move the cars
    // meanwhile: the step checks each stage's fingerprint against the live car, and a moved car walks its meshes
    // itself. (car_snapshots is null when nothing moves the cars before the staging joins.)
    if (car_snapshots != nullptr) {
        car_snapshots[car_idx] = car;
    }
    FillCarWorldMeshCandidateFingerprint(car, car_cfg, stage.fingerprint);
    stage.mesh_count = cfg.arena_mesh_bvh_mesh_count;
    stage.mesh_mask = mesh_mask;
    stage.valid = 1;
    for (uint32_t rest = mesh_mask; rest != 0; rest &= rest - 1u) {
        const int mesh = __ffs(static_cast<int>(rest)) - 1;
        const int item = atomicAdd(&mesh_work_count[mesh], 1);
        mesh_work[static_cast<long long>(mesh) * car_slots + item] = car_idx;
    }
}

ROCKETGPU_GLOBAL void StageCarWorldMeshCandidatesKernel(
    const __grid_constant__ SimConfig cfg,
    const CarState* cars,
    const CarConfig* car_configs,
    CarWorldMeshCandidateStage* stages,
    const int32_t* mesh_work,
    const int32_t* mesh_work_count,
    CarWorldTrianglePairs pairs,
    int pairs_pass,
    bullet_port::btGjkEpaScratch* gjk_pool) {
    const int stride = cfg.max_cars_per_env;
    const long long car_slots = static_cast<long long>(cfg.env_count) * stride;
    const long long first = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    // GJK/EPA scratch from the pool, a slot per thread (GJK/EPA initialize it on use); a grid-stride loop over the
    // pairs, each of which writes only its own entry.
    bullet_port::btGjkEpaScratch* const gjk_scratch = gjk_pool + first;
    const ArenaMeshLeafGrid leaf_grid = MakeArenaBroadphaseGrid(cfg).leaf_grid;
    const int meshes = Min(cfg.arena_mesh_bvh_mesh_count, kCarWorldMeshStageMeshes);
    // Listed pairs in mesh-major order (a warp walks one mesh's BVH): pair t is item t - (pairs of earlier meshes).
    for (long long tid = first;; tid += static_cast<long long>(gridDim.x) * blockDim.x) {
        int mesh = 0;
        long long item = tid;
        for (; mesh < meshes; ++mesh) {
            const long long listed = mesh_work_count[mesh];
            if (item < listed) {
                break;
            }
            item -= listed;
        }
        if (mesh >= meshes) {
            break;
        }
        const int car_idx = mesh_work[static_cast<long long>(mesh) * car_slots + item];
        const int env_idx = car_idx / stride;
        const int slot = car_idx - env_idx * stride;
        if (pairs_pass == kTrianglePairsMerge) {
            // No gathered hits (more triangles than the gather holds, or the pair list was full): the step walks
            // this car itself, as it does a car over more meshes than the stage holds.
            CarWorldMeshCandidateStage& stage = stages[car_idx];
            if (stage.meshes[CarWorldMeshStageSlot(stage.mesh_mask, mesh)].pair_count < 0) {
                stage.valid = 0;
                continue;
            }
        }
        ContactAccumulator acc;
        int split_row_count = 0;
        int normal_row_count = 0;
        int friction_row_count = 0;
        AppendCarWorldMeshContactRows<true>(
            cars[car_idx], car_configs[car_idx], cfg.mutator, Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 0.0f), acc,
            nullptr, split_row_count, nullptr, normal_row_count, nullptr, friction_row_count, 0.0f,
            cfg.arena_mesh_tris, cfg.arena_mesh_tri_count,
            cfg.arena_mesh_bvh_quantization, cfg.arena_mesh_bvh_quantization_count,
            cfg.arena_mesh_bvh_meshes, cfg.arena_mesh_bvh_mesh_count,
            cfg.arena_mesh_bvh_nodes, cfg.arena_mesh_bvh_node_count,
            cfg.arena_mesh_bvh_subtrees, cfg.arena_mesh_bvh_subtree_count,
            env_idx, slot, gjk_scratch, nullptr, nullptr, nullptr, 0.0f,
            &stages[car_idx], nullptr, mesh, &leaf_grid, &pairs, pairs_pass);
    }
}

// The contact pass between gather and merge: one car-triangle GJK/EPA per thread.
ROCKETGPU_GLOBAL void CarWorldTriangleHitsKernel(
    const __grid_constant__ SimConfig cfg,
    const CarState* cars,
    const CarConfig* car_configs,
    CarWorldTrianglePairs pairs,
    bullet_port::btGjkEpaScratch* gjk_pool) {
    const int count = Min(*pairs.count, pairs.capacity);
    bullet_port::btGjkEpaScratch* const gjk_scratch = gjk_pool + (blockIdx.x * blockDim.x + threadIdx.x);
    for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); i < count; i += static_cast<int>(gridDim.x * blockDim.x)) {
        const int car_idx = pairs.car_idx[i];
        const CarConfig& car_cfg = car_configs[car_idx];
        pairs.hits[i] = ComputeCarWorldTriangleHit(
            cars[car_idx], car_cfg, cfg.mutator, cfg.arena_mesh_tris, pairs.tri_idx[i],
            CarContactBreakingThresholdUU(car_cfg) * (1.0f / 50.0f), gjk_scratch);
    }
}

// The step kernels' call chains are statically sized (no recursion or indirect calls), so each kernel's
// localSizeBytes is its whole stack: the limit is their maximum, with no headroom. The driver reserves this much
// for every thread the GPU can hold (170 SMs x 1,536 on an RTX 5090), so each KB here costs ~0.27 GB of VRAM.
// Sized for the kernels this step launches: the staged step never runs StepKernel (its largest frame), and the
// driver grows the reservation itself should a later launch need more.
inline void EnsureCudaStepStackLimit(bool step_kernel) {
    static bool configured[2] = {false, false};
    if (configured[step_kernel ? 1 : 0]) {
        return;
    }
    size_t required_stack_bytes = 0;
    const void* const step_kernels[] = {
        step_kernel ? reinterpret_cast<const void*>(&StepKernel) : reinterpret_cast<const void*>(&PreTickKernel),
        reinterpret_cast<const void*>(&PreTickKernel),
        reinterpret_cast<const void*>(&StageWheelRaysKernel),
        reinterpret_cast<const void*>(&PrefilterCarWorldMeshCandidatesKernel),
        reinterpret_cast<const void*>(&StageCarWorldMeshCandidatesKernel),
        reinterpret_cast<const void*>(&CarWorldTriangleHitsKernel),
        reinterpret_cast<const void*>(&StagedCarPreKernel),
        reinterpret_cast<const void*>(&StagedWorldKernel<kWorldStepContacts>),
        reinterpret_cast<const void*>(&StagedWorldKernel<kWorldStepContactsCarsApart>),
        reinterpret_cast<const void*>(&StagedWorldKernel<kWorldStepRowsCarsApart>),
        reinterpret_cast<const void*>(&StagedCarPostKernel),
        reinterpret_cast<const void*>(&StagedEnvTailKernel),
        reinterpret_cast<const void*>(&StagedPadPostKernel),
        reinterpret_cast<const void*>(&StagedPadPreKernel),
        reinterpret_cast<const void*>(&StagedCarWorldRowsKernel),
        reinterpret_cast<const void*>(&StagedCarWorldRowsSplitKernel),
        reinterpret_cast<const void*>(&StagedCarWorldRowsListKernel<true>),
        reinterpret_cast<const void*>(&StagedCarWorldRowsListKernel<false>),
    };
    for (const void* kernel : step_kernels) {
        cudaFuncAttributes attr{};
        if (cudaFuncGetAttributes(&attr, kernel) != cudaSuccess) {
            cudaGetLastError();
            continue;
        }
        required_stack_bytes = std::max(required_stack_bytes, static_cast<size_t>(attr.localSizeBytes));
    }
    size_t current_limit = 0;
    CheckCuda(cudaDeviceGetLimit(&current_limit, cudaLimitStackSize), "cudaDeviceGetLimit(cudaLimitStackSize)");
    if (required_stack_bytes > 0 && current_limit < required_stack_bytes) {
        CheckCuda(cudaDeviceSetLimit(cudaLimitStackSize, required_stack_bytes), "cudaDeviceSetLimit(cudaLimitStackSize)");
    }
    configured[step_kernel ? 1 : 0] = true;
}

inline int CudaSmCount() {
    static int sm_count = -1;
    if (sm_count < 0) {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
            sm_count <= 0) {
            cudaGetLastError();
            sm_count = 128;
        }
    }
    return sm_count;
}

// One thread per env; 64-thread blocks once there are 64+ envs per SM.
inline int GetCudaStepBlockSize(int env_count) {
    return env_count >= 64 * CudaSmCount() ? 64 : 32;
}

constexpr int kCudaRayKernelBlockSize = 128;

// Threads of StepKernel the GPU holds at once with this block size.
inline long long StepKernelResidentThreads(int block_size) {
    static int cached_block_size = 0;
    static long long cached = 0;
    if (block_size == cached_block_size) {
        return cached;
    }
    int blocks_per_sm = 0;
    CheckCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, StepKernel, block_size, 0),
              "cudaOccupancyMaxActiveBlocksPerMultiprocessor(StepKernel)");
    cached_block_size = block_size;
    cached = static_cast<long long>(blocks_per_sm) * block_size * CudaSmCount();
    return cached;
}

// GJK/EPA scratch (~27 KB a thread) for the car-world staging kernels and StepKernel's car-world lanes, in global
// memory rather than each thread's stack: the driver reserves stack for every thread the GPU could hold, so a
// per-thread copy cost ~7 GB; the pool holds one per launched thread, and those launches are capped to fit it.
// Bumped whenever a buffer or symbol the step's launches bake in changes, so cached step graphs stop matching.
inline uint64_t& StepResourceEpoch() {
    static uint64_t epoch = 0;
    return epoch;
}
struct GjkScratchPool {
    bullet_port::btGjkEpaScratch* data = nullptr;
    size_t capacity = 0;
};
inline GjkScratchPool& GetGjkScratchPool() {
    static GjkScratchPool pool;
    return pool;
}
// Threads the staging kernels launch at most: two 128-thread blocks per SM, their occupancy at ~255 registers.
inline size_t CarWorldStagingThreads() {
    return static_cast<size_t>(CudaSmCount()) * 2 * kCudaRayKernelBlockSize;
}
static bool StreamIsCapturing(cudaStream_t stream);
// Grows the pool to `threads` slots outside stream capture; returns the pool, or nullptr when it is too small.
inline bullet_port::btGjkEpaScratch* EnsureGjkScratchPool(size_t threads, cudaStream_t stream) {
    GjkScratchPool& pool = GetGjkScratchPool();
    if (pool.capacity < threads) {
        if (StreamIsCapturing(stream)) {
            return nullptr;
        }
        CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(GJK scratch pool grow)");
        if (pool.data != nullptr) {
            cudaFree(pool.data);
            pool.data = nullptr;
            pool.capacity = 0;
        }
        CheckCuda(cudaMalloc(&pool.data, threads * sizeof(bullet_port::btGjkEpaScratch)), "cudaMalloc(GJK scratch pool)");
        pool.capacity = threads;
        ++StepResourceEpoch();
    }
    return pool.data;
}
constexpr size_t kCarWorldTrianglePairsPerCar = 16;

struct CarWorldStageBuffers {
    CarWorldMeshCandidateStage* stages = nullptr;
    int32_t* mesh_work = nullptr;        // per mesh, the car slots whose query AABB may touch it
    int32_t* mesh_work_count = nullptr;  // kCarWorldMeshStageMeshes counters
    CarWorldTrianglePairs pairs{};       // count lives in mesh_work_count[kCarWorldMeshStageMeshes]
    size_t capacity = 0;
    CarWorldMeshCandidateStage* published = nullptr;
    CarState* car_snapshots = nullptr;   // the cars as the prefilter saw them, read by the later staging passes
    int32_t* row_cars = nullptr;         // StagedCarWorldRowsSplitKernel's lists, counts after the mesh work counts
    int published_stride = 0;
};
inline CarWorldStageBuffers& GetCarWorldStageBuffers() {
    static CarWorldStageBuffers buffers;
    return buffers;
}

inline void PublishCarWorldStage(CarWorldMeshCandidateStage* stages, int stride, cudaStream_t stream) {
    CarWorldStageBuffers& b = GetCarWorldStageBuffers();
    if (b.published == stages && b.published_stride == stride) {
        return;
    }
    b.published = stages;
    b.published_stride = stride;
    ++StepResourceEpoch();
    CheckCuda(cudaMemcpyToSymbolAsync(g_rocketgpu_car_world_stage, &stages, sizeof(stages), 0,
                                      cudaMemcpyHostToDevice, stream),
              "cudaMemcpyToSymbolAsync(g_rocketgpu_car_world_stage)");
    CheckCuda(cudaMemcpyToSymbolAsync(g_rocketgpu_car_world_stage_stride, &stride, sizeof(stride), 0,
                                      cudaMemcpyHostToDevice, stream),
              "cudaMemcpyToSymbolAsync(g_rocketgpu_car_world_stage_stride)");
}
inline bool EnsureCarWorldStageArmed(const SimConfig& cfg, cudaStream_t stream) {
    const size_t count = static_cast<size_t>(cfg.env_count) * static_cast<size_t>(cfg.max_cars_per_env);
    if (count == 0) {
        return false;
    }
    CarWorldStageBuffers& b = GetCarWorldStageBuffers();
    const bool capturing = StreamIsCapturing(stream);
    if (b.capacity < count) {
        if (capturing) {
            return false;
        }
        CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(car-world stage grow)");
        if (b.stages != nullptr) {
            cudaFree(b.stages);
            cudaFree(b.car_snapshots);
            cudaFree(b.row_cars);
            cudaFree(b.mesh_work);
            cudaFree(b.pairs.car_idx);
            cudaFree(b.pairs.tri_idx);
            cudaFree(b.pairs.hits);
            b.stages = nullptr;
            b.mesh_work = nullptr;
            b.pairs = CarWorldTrianglePairs{};
        }
        if (b.mesh_work_count == nullptr) {
            CheckCuda(cudaMalloc(&b.mesh_work_count, (kCarWorldMeshStageMeshes + 3) * sizeof(int32_t)), "cudaMalloc(car-world mesh work count)");
        }
        const size_t pair_capacity = count * kCarWorldTrianglePairsPerCar;
        CheckCuda(cudaMalloc(&b.pairs.car_idx, pair_capacity * sizeof(int32_t)), "cudaMalloc(car-world triangle pairs)");
        CheckCuda(cudaMalloc(&b.pairs.tri_idx, pair_capacity * sizeof(int32_t)), "cudaMalloc(car-world triangle pairs)");
        CheckCuda(cudaMalloc(&b.pairs.hits, pair_capacity * sizeof(CarWorldTriangleHit)), "cudaMalloc(car-world triangle hits)");
        b.pairs.count = b.mesh_work_count + kCarWorldMeshStageMeshes;
        b.pairs.capacity = static_cast<int32_t>(std::min<size_t>(pair_capacity, INT32_MAX));
        b.capacity = 0;
        b.published = nullptr;
        CheckCuda(cudaMalloc(&b.mesh_work, count * kCarWorldMeshStageMeshes * sizeof(int32_t)), "cudaMalloc(car-world mesh work)");
        CheckCuda(cudaMalloc(&b.stages, count * sizeof(CarWorldMeshCandidateStage)), "cudaMalloc(car-world stage)");
        CheckCuda(cudaMalloc(&b.car_snapshots, count * sizeof(CarState)), "cudaMalloc(car-world stage car snapshots)");
        CheckCuda(cudaMalloc(&b.row_cars, 2 * count * sizeof(int32_t)), "cudaMalloc(car-world rows car lists)");
        CheckCuda(cudaMemset(b.stages, 0, count * sizeof(CarWorldMeshCandidateStage)), "cudaMemset(car-world stage)");
        b.capacity = count;
        ++StepResourceEpoch();
    }
    if (b.published != b.stages || b.published_stride != cfg.max_cars_per_env) {
        if (capturing) {
            return false;
        }
        PublishCarWorldStage(b.stages, cfg.max_cars_per_env, stream);
    }
    return true;
}
// EPA slots leased by btEpaLease (bullet_gjk_epa2_port.hpp). A lease lasts one EPA run, so the slots only need to
// cover the threads in EPA at the same moment; a thread that finds them all busy waits for one to free.
constexpr unsigned int kEpaScratchSlots = 8192;
inline void EnsureEpaScratchPool() {
    static bool ready = false;
    if (ready) {
        return;
    }
    bullet_port::EPA* pool = nullptr;
    unsigned int* busy = nullptr;
    const unsigned int slots = kEpaScratchSlots;
    CheckCuda(cudaMalloc(&pool, slots * sizeof(bullet_port::EPA)), "cudaMalloc(EPA scratch pool)");
    CheckCuda(cudaMalloc(&busy, slots * sizeof(unsigned int)), "cudaMalloc(EPA scratch busy flags)");
    CheckCuda(cudaMemset(busy, 0, slots * sizeof(unsigned int)), "cudaMemset(EPA scratch busy flags)");
    CheckCuda(cudaMemcpyToSymbol(bullet_port::g_rocketgpu_epa_pool, &pool, sizeof(pool)), "cudaMemcpyToSymbol(EPA pool)");
    CheckCuda(cudaMemcpyToSymbol(bullet_port::g_rocketgpu_epa_busy, &busy, sizeof(busy)), "cudaMemcpyToSymbol(EPA busy)");
    CheckCuda(cudaMemcpyToSymbol(bullet_port::g_rocketgpu_epa_slots, &slots, sizeof(slots)), "cudaMemcpyToSymbol(EPA slots)");
    ready = true;
}

// The step's process-wide resources (scratch pools, staging buffers, side streams, graph cache) serve one step at a
// time. StepResourcesGuard holds a process-wide lock while a step is queued, and a step queued on another stream than
// the last one waits on the GPU for that one. The resources live on the first GPU a step ran on; stepping a simulator
// on another GPU is an error.
class StepResourcesGuard {
public:
    explicit StepResourcesGuard(cudaStream_t stream)
        : lock_(Mutex()), stream_(stream), ordered_(!StreamIsCapturing(stream)) {
        State& s = GetState();
        int device = 0;
        CheckCuda(cudaGetDevice(&device), "cudaGetDevice(step)");
        if (s.device < 0) {
            CheckCuda(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming), "cudaEventCreate(step done)");
            s.device = device;
        } else if (s.device != device) {
            throw std::runtime_error("RocketGPU: every simulator in a process must step on the same GPU");
        }
        if (ordered_ && s.has_last && s.last_stream != stream) {
            CheckCuda(cudaStreamWaitEvent(stream, s.done, 0), "cudaStreamWaitEvent(previous step)");
        }
    }
    ~StepResourcesGuard() {
        State& s = GetState();
        if (ordered_ && cudaEventRecord(s.done, stream_) == cudaSuccess) {
            s.last_stream = stream_;
            s.has_last = true;
        }
    }
    StepResourcesGuard(const StepResourcesGuard&) = delete;
    StepResourcesGuard& operator=(const StepResourcesGuard&) = delete;

private:
    struct State {
        int device = -1;
        cudaEvent_t done = nullptr;
        cudaStream_t last_stream = nullptr;
        bool has_last = false;
    };
    static std::recursive_mutex& Mutex() {
        static std::recursive_mutex mutex;
        return mutex;
    }
    static State& GetState() {
        static State state;
        return state;
    }
    std::lock_guard<std::recursive_mutex> lock_;
    cudaStream_t stream_;
    bool ordered_;
};

void PrepareCarWorldStage(int env_count, int max_cars_per_env, cudaStream_t stream) {
    const StepResourcesGuard guard(stream);
    EnsureEpaScratchPool();
    SimConfig cfg;
    cfg.env_count = env_count;
    cfg.max_cars_per_env = max_cars_per_env;
    (void)EnsureCarWorldStageArmed(cfg, stream);
}
inline void LaunchStageCarWorldMeshCandidates(
    const SimConfig& cfg,
    const CarState* cars,
    const CarConfig* car_configs,
    const int32_t* env_car_count,
    cudaStream_t stream,
    cudaEvent_t prefilter_done = nullptr) {
    // prefilter_done: recorded once the cars are copied (after the prefilter), when the caller passes one.
    const auto mark_prefilter_done = [&] {
        if (prefilter_done != nullptr) {
            CheckCuda(cudaEventRecord(prefilter_done, stream), "cudaEventRecord(car-world prefilter)");
        }
    };
    bullet_port::btGjkEpaScratch* const gjk_pool = EnsureGjkScratchPool(CarWorldStagingThreads(), stream);
    if (gjk_pool == nullptr || !EnsureCarWorldStageArmed(cfg, stream)) {
        if (!StreamIsCapturing(stream)) {
            PublishCarWorldStage(nullptr, 0, stream);
        }
        mark_prefilter_done();
        return;
    }
    CarWorldStageBuffers& b = GetCarWorldStageBuffers();
    const size_t car_slots = static_cast<size_t>(cfg.env_count) * static_cast<size_t>(cfg.max_cars_per_env);
    const int block_size = kCudaRayKernelBlockSize;
    // A kernel rather than cudaMemsetAsync: in a captured step the memset node held this branch back until
    // StagedCarPreKernel on the other branch had finished. It also empties the car-world rows lists of the tick.
    ZeroInt32Kernel<<<1, 32, 0, stream>>>(b.mesh_work_count, kCarWorldMeshStageMeshes + 3);
    CheckCuda(cudaGetLastError(), "ZeroInt32Kernel launch (car-world mesh work count)");
    PrefilterCarWorldMeshCandidatesKernel<<<static_cast<unsigned>((car_slots + block_size - 1) / block_size),
                                            static_cast<unsigned>(block_size), 0, stream>>>(
        cfg, cars, car_configs, env_car_count, b.stages, b.mesh_work, b.mesh_work_count,
        prefilter_done != nullptr ? b.car_snapshots : nullptr);
    CheckCuda(cudaGetLastError(), "PrefilterCarWorldMeshCandidatesKernel launch");
    mark_prefilter_done();
    if (prefilter_done != nullptr) {
        cars = b.car_snapshots;
    }
    const int meshes = std::max(0, std::min(cfg.arena_mesh_bvh_mesh_count, kCarWorldMeshStageMeshes));
    const size_t pair_count = car_slots * static_cast<size_t>(meshes);
    if (pair_count == 0) {
        return;
    }
    // Gather each pair's plane-test survivors, run their GJK/EPA a thread per car-triangle, then merge in walk order.
    CarWorldTrianglePairs pairs = b.pairs;
    pairs.car_stride = cfg.max_cars_per_env;
    const size_t stage_threads = std::min(pair_count, CarWorldStagingThreads());
    const unsigned stage_blocks = static_cast<unsigned>((stage_threads + block_size - 1) / block_size);
    StageCarWorldMeshCandidatesKernel<<<stage_blocks, static_cast<unsigned>(block_size), 0, stream>>>(
        cfg, cars, car_configs, b.stages, b.mesh_work, b.mesh_work_count, pairs, kTrianglePairsGather, gjk_pool);
    CheckCuda(cudaGetLastError(), "StageCarWorldMeshCandidatesKernel gather launch");
    const size_t hit_threads = std::min<size_t>(static_cast<size_t>(pairs.capacity), CarWorldStagingThreads());
    CarWorldTriangleHitsKernel<<<static_cast<unsigned>((hit_threads + block_size - 1) / block_size),
                                 static_cast<unsigned>(block_size), 0, stream>>>(cfg, cars, car_configs, pairs, gjk_pool);
    CheckCuda(cudaGetLastError(), "CarWorldTriangleHitsKernel launch");
    StageCarWorldMeshCandidatesKernel<<<stage_blocks, static_cast<unsigned>(block_size), 0, stream>>>(
        cfg, cars, car_configs, b.stages, b.mesh_work, b.mesh_work_count, pairs, kTrianglePairsMerge, gjk_pool);
    CheckCuda(cudaGetLastError(), "StageCarWorldMeshCandidatesKernel merge launch");
}
inline void LaunchStageWheelRays(
    const SimConfig& cfg,
    const CarState* cars,
    const CarConfig* car_configs,
    const WheelRuntimeState* wheel_runtime_states,
    const BallState* balls,
    const EnvState* env_states,
    WorldSolverScratch* solver_scratch,
    const int32_t* env_car_count,
    cudaStream_t stream,
    bool with_car_world_stage = true) {
    const long long rays = static_cast<long long>(cfg.env_count) * 4ll * static_cast<long long>(cfg.max_cars_per_env);
    if (rays <= 0) {
        return;
    }
    const int block_size = kCudaRayKernelBlockSize;
    const long long blocks = (rays + block_size - 1) / block_size;
    StageWheelRaysKernel<<<static_cast<unsigned>(blocks), static_cast<unsigned>(block_size), 0, stream>>>(
        cfg, cars, car_configs, wheel_runtime_states, balls, env_states, solver_scratch, env_car_count);
    CheckCuda(cudaGetLastError(), "StageWheelRaysKernel launch");
    if (with_car_world_stage) {
        LaunchStageCarWorldMeshCandidates(cfg, cars, car_configs, env_car_count, stream);
    }
}

constexpr int kStagedBlockSize = 64;
inline bool UseStagedConcurrency() {
    static const bool concurrent = [] {
        const char* v = std::getenv("ROCKETGPU_STAGED_CONCURRENT");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return concurrent;
}
// Side streams and fork/join events of the staged step's concurrent branches (created once, kept for the process,
// used under StepResourcesGuard).
struct StagedStreams {
    cudaStream_t stage = nullptr;
    cudaStream_t dependent = nullptr;
    cudaEvent_t pre_done = nullptr;
    cudaEvent_t prefilter_done = nullptr;
    cudaEvent_t stage_done = nullptr;
    cudaEvent_t independence_done = nullptr;
    cudaEvent_t dependent_done = nullptr;
    cudaEvent_t rows_split_done = nullptr;
    cudaEvent_t rows_light_done = nullptr;
};
inline StagedStreams& GetStagedStreams() {
    static StagedStreams streams = [] {
        StagedStreams s;
        CheckCuda(cudaStreamCreateWithFlags(&s.stage, cudaStreamNonBlocking), "cudaStreamCreate(staged stage)");
        CheckCuda(cudaStreamCreateWithFlags(&s.dependent, cudaStreamNonBlocking), "cudaStreamCreate(staged dependent)");
        for (cudaEvent_t* e : {&s.pre_done, &s.prefilter_done, &s.stage_done, &s.independence_done, &s.dependent_done,
                              &s.rows_split_done, &s.rows_light_done}) {
            CheckCuda(cudaEventCreateWithFlags(e, cudaEventDisableTiming), "cudaEventCreate(staged)");
        }
        return s;
    }();
    return streams;
}
inline bool UseCarWorldRowsConcurrent() {
    static const bool on = [] {
        const char* v = std::getenv("ROCKETGPU_CWR_CONCURRENT");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}
inline bool UseCarWorldRowsSplit() {
    static const bool split = [] {
        const char* v = std::getenv("ROCKETGPU_CWR_SPLIT");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return split;
}
// The staged step (kernels per phase, car-world staging and the contacts on concurrent branches) unless
// ROCKETGPU_STAGED_STEP says otherwise. The one-kernel step wins only with many envs of at most two cars (RTX 5090,
// M arena-ticks/s, staged vs one-kernel: 1v1 16.3 vs 15.8 at 65k envs, 17.1 vs 17.6 at 82k, 17.1 vs 17.5 at 98k,
// 17.1 vs 17.7 at 131k; 1v1 in six-car slots 14.5 vs 14.3 at 32k, 13.9 vs 13.6 at 65k; 3v3 6.8 vs 6.0 at 32k,
// 6.8 vs 5.3 at 49k, 6.8 vs 5.7 at 65k, 6.8 vs 5.7 at 98k; training mix of 1v1/2v2/3v3 in six-car slots 7.3 vs 4.9 at 8k, 9.9 vs 9.0 at 32k,
// 10.3 vs 9.9 at 65k).
constexpr int kStagedStepMaxTwoCarEnvs = 65536;
inline bool UseStagedStep(const SimConfig& cfg) {
    static const int forced = [] {
        const char* v = std::getenv("ROCKETGPU_STAGED_STEP");
        return v == nullptr ? -1 : (std::atoi(v) != 0 ? 1 : 0);
    }();
    return forced >= 0 ? forced != 0 : (cfg.max_cars_per_env > 2 || cfg.env_count <= kStagedStepMaxTwoCarEnvs);
}

// Each tick: PreTickKernel, the staged wheel rays and car-world candidates, then StepKernel. controls_by_tick and
// env_commands_by_tick are optional per-tick streams; respawn_slots are the slots the step respawns cars into.
inline void LaunchStepTicksDirect(
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch,
    const WorldSolverScratchRowPool& row_pool,
    CarControls* controls,
    const CarControls* controls_by_tick,
    int32_t control_stride,
    const EnvCommand* env_commands_by_tick,
    int32_t env_command_stride,
    CarConfig* car_configs,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    const uint8_t* respawn_slots,
    uint8_t* writable_respawn_slots,
    RespawnRngState* respawn_rng,
    const int32_t* respawn_order,
    uint64_t seed,
    int32_t ticks,
    cudaStream_t stream) {
    EnsureCudaStepStackLimit(!UseStagedStep(cfg) || cfg.mutator.physics_events.windows != nullptr);
    EnsureEpaScratchPool();
    constexpr int kPreTickEnvsPerBlock = 32;
    const int pre_block_size = kPreTickEnvsPerBlock * cfg.max_cars_per_env;
    const int pre_blocks = (cfg.env_count + kPreTickEnvsPerBlock - 1) / kPreTickEnvsPerBlock;
    // Lanes per env, up to one per car, while that stays within 2 warps per SM: beyond it the extra warps cost more
    // latency hiding than the parallel car phases save (RTX 5090: 8 lanes at 1,024 envs, 2 at 4,096, 1 from 8,192).
    int lanes = 1;
    while (lanes < 32 && lanes < cfg.max_cars_per_env &&
           static_cast<long long>(cfg.env_count) * lanes * 2 <= CudaSmCount() * 32ll * 2) {
        lanes *= 2;
    }
    const int block_size = GetCudaStepBlockSize(cfg.env_count * lanes);
    const int blocks = static_cast<int>((static_cast<long long>(cfg.env_count) * lanes + block_size - 1) / block_size);
    if (cfg.env_count <= 0 || pre_block_size <= 0) {
        return;
    }
    // StepKernel's car-world lanes (lanes > 1, few envs) take a GJK/EPA scratch per thread from the pool.
    bullet_port::btGjkEpaScratch* step_gjk_pool = nullptr;
    if (lanes > 1) {
        step_gjk_pool = EnsureGjkScratchPool(std::max(static_cast<size_t>(blocks) * block_size, CarWorldStagingThreads()), stream);
        if (step_gjk_pool == nullptr) {
            lanes = 1;  // capturing with too small a pool: the sequential car-world rows need no pool
        }
    }
    const bool local_state = lanes == 1 && cfg.max_cars_per_env <= kLocalStepMaxSlots &&
                             cfg.env_count > StepKernelResidentThreads(block_size);
    // Physics events come out in env step order, so they keep the one-kernel step.
    bullet_port::btGjkEpaScratch* const staged_gjk_pool =
        (UseStagedStep(cfg) && !cfg.mutator.physics_events.windows) ? EnsureGjkScratchPool(CarWorldStagingThreads(), stream) : nullptr;
    if (staged_gjk_pool != nullptr) {
        const long long car_slots = static_cast<long long>(cfg.env_count) * cfg.max_cars_per_env;
        const unsigned car_blocks = static_cast<unsigned>((car_slots + kStagedBlockSize - 1) / kStagedBlockSize);
        const unsigned env_blocks = static_cast<unsigned>((cfg.env_count + kStagedBlockSize - 1) / kStagedBlockSize);
        const bool pads_apart = cfg.enable_boost_pads && BoostPadsApart(cfg);
        const long long pad_slots = pads_apart ? static_cast<long long>(cfg.env_count) * BoostPadCount(cfg) : 0;
        const unsigned pad_post_blocks = static_cast<unsigned>((pad_slots + kStagedBlockSize - 1) / kStagedBlockSize);
        const long long pad_pre_slots = cfg.enable_boost_pads ? static_cast<long long>(cfg.env_count) * BoostPadCount(cfg) : 0;
        const unsigned pad_pre_blocks = static_cast<unsigned>((pad_pre_slots + kStagedBlockSize - 1) / kStagedBlockSize);
        const unsigned row_blocks = static_cast<unsigned>(
            (std::min<long long>(car_slots, static_cast<long long>(CarWorldStagingThreads())) + kStagedBlockSize - 1) / kStagedBlockSize);
        // Branches of the tick that touch disjoint data run on side streams (graph branches when captured):
        //  - the car-world mesh staging reads the cars' transforms, which nothing writes before the contacts
        //    kernels (car-car split pushes and demos move cars there). Its prefilter copies the cars and the later
        //    passes read the copies, so only the prefilter runs before the contacts; the rest joins before
        //    StagedCarWorldRowsKernel, whose fingerprint check drops the stage of a car the contacts moved;
        //  - the contacts of envs whose vehicle updates are not independent (StagedWorldKernel<kWorldStepContacts>)
        //    touch only those envs, so they run beside StagedCarPreKernel and the other envs' contacts.
        const bool concurrent = UseStagedConcurrency();
        // The car-world rows split needs the staging buffers, which the staging of each tick then arms as well.
        const bool split_rows = UseCarWorldRowsSplit() && EnsureCarWorldStageArmed(cfg, stream);
        const bool concurrent_rows = concurrent && UseCarWorldRowsConcurrent();
        StagedStreams& side = GetStagedStreams();
        const cudaStream_t stage_stream = concurrent ? side.stage : stream;
        const cudaStream_t dep_stream = concurrent ? side.dependent : stream;
        for (int32_t t = 0; t < ticks; ++t) {
            PreTickKernel<<<pre_blocks, pre_block_size, 0, stream>>>(
                cfg, teams, env_car_count, cars, solver_scratch, controls, balls, env_states, pads,
                controls_by_tick, control_stride, env_commands_by_tick, env_command_stride,
                respawn_slots, writable_respawn_slots, respawn_rng, respawn_order, seed, t, pad_pre_blocks == 0);
            CheckCuda(cudaGetLastError(), "PreTickKernel launch");
            if (concurrent) {
                CheckCuda(cudaEventRecord(side.pre_done, stream), "cudaEventRecord(pre tick)");
                CheckCuda(cudaStreamWaitEvent(stage_stream, side.pre_done, 0), "cudaStreamWaitEvent(stage)");
            }
            if (pad_pre_blocks > 0) {
                StagedPadPreKernel<<<pad_pre_blocks, kStagedBlockSize, 0, stage_stream>>>(cfg, env_car_count, pads);
                CheckCuda(cudaGetLastError(), "StagedPadPreKernel launch");
            }
            if (concurrent) {
                LaunchStageCarWorldMeshCandidates(cfg, cars, car_configs, env_car_count, stage_stream, side.prefilter_done);
                CheckCuda(cudaEventRecord(side.stage_done, stage_stream), "cudaEventRecord(stage)");
            }
            LaunchStageWheelRays(cfg, cars, car_configs, wheel_runtime_states, balls, env_states, solver_scratch,
                                 env_car_count, stream, !concurrent);
            StagedVehicleIndependenceKernel<<<env_blocks, kStagedBlockSize, 0, stream>>>(cfg, env_car_count, solver_scratch);
            CheckCuda(cudaGetLastError(), "StagedVehicleIndependenceKernel launch");
            if (concurrent) {
                CheckCuda(cudaEventRecord(side.independence_done, stream), "cudaEventRecord(independence)");
                CheckCuda(cudaStreamWaitEvent(dep_stream, side.independence_done, 0), "cudaStreamWaitEvent(dependent)");
                CheckCuda(cudaStreamWaitEvent(dep_stream, side.prefilter_done, 0), "cudaStreamWaitEvent(dependent prefilter)");
            }
            StagedCarPreKernel<<<car_blocks, kStagedBlockSize, 0, stream>>>(
                cfg, env_car_count, cars, wheel_runtime_states, solver_scratch, controls, car_configs, balls);
            CheckCuda(cudaGetLastError(), "StagedCarPreKernel launch");
            StagedWorldKernel<kWorldStepContacts><<<env_blocks, kStagedBlockSize, 0, dep_stream>>>(
                cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls, car_configs,
                balls, env_states, pads, seed);
            CheckCuda(cudaGetLastError(), "StagedWorldKernel contacts launch");
            if (concurrent) {
                CheckCuda(cudaEventRecord(side.dependent_done, dep_stream), "cudaEventRecord(dependent)");
                CheckCuda(cudaStreamWaitEvent(stream, side.prefilter_done, 0), "cudaStreamWaitEvent(prefilter)");
            }
            StagedWorldKernel<kWorldStepContactsCarsApart><<<env_blocks, kStagedBlockSize, 0, stream>>>(
                cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls, car_configs,
                balls, env_states, pads, seed);
            CheckCuda(cudaGetLastError(), "StagedWorldKernel contacts (cars apart) launch");
            if (concurrent) {
                CheckCuda(cudaStreamWaitEvent(stream, side.dependent_done, 0), "cudaStreamWaitEvent(join dependent)");
                CheckCuda(cudaStreamWaitEvent(stream, side.stage_done, 0), "cudaStreamWaitEvent(join stage)");
            }
            if (split_rows) {
                CarWorldStageBuffers& b = GetCarWorldStageBuffers();
                int32_t* const row_car_count = b.mesh_work_count + kCarWorldMeshStageMeshes + 1;
                StagedCarWorldRowsSplitKernel<<<car_blocks, kStagedBlockSize, 0, stream>>>(
                    cfg, env_car_count, cars, car_configs, solver_scratch, b.row_cars, row_car_count);
                CheckCuda(cudaGetLastError(), "StagedCarWorldRowsSplitKernel launch");
                // The two lists hold different cars, so with concurrency the out-of-reach list runs on a side branch.
                const cudaStream_t light_stream = concurrent_rows ? side.dependent : stream;
                if (concurrent_rows) {
                    CheckCuda(cudaEventRecord(side.rows_split_done, stream), "cudaEventRecord(rows split)");
                    CheckCuda(cudaStreamWaitEvent(light_stream, side.rows_split_done, 0), "cudaStreamWaitEvent(rows split)");
                }
                StagedCarWorldRowsListKernel<true><<<car_blocks, kStagedBlockSize, 0, light_stream>>>(
                    cfg, cars, car_configs, solver_scratch, row_pool, b.row_cars, row_car_count, staged_gjk_pool);
                CheckCuda(cudaGetLastError(), "StagedCarWorldRowsListKernel<true> launch");
                StagedCarWorldRowsListKernel<false><<<row_blocks, kStagedBlockSize, 0, stream>>>(
                    cfg, cars, car_configs, solver_scratch, row_pool, b.row_cars, row_car_count, staged_gjk_pool);
                CheckCuda(cudaGetLastError(), "StagedCarWorldRowsListKernel<false> launch");
                if (concurrent_rows) {
                    CheckCuda(cudaEventRecord(side.rows_light_done, light_stream), "cudaEventRecord(rows light)");
                    CheckCuda(cudaStreamWaitEvent(stream, side.rows_light_done, 0), "cudaStreamWaitEvent(rows light)");
                }
            } else {
                StagedCarWorldRowsKernel<<<row_blocks, kStagedBlockSize, 0, stream>>>(
                    cfg, env_car_count, cars, car_configs, solver_scratch, row_pool, staged_gjk_pool);
                CheckCuda(cudaGetLastError(), "StagedCarWorldRowsKernel launch");
            }
            StagedWorldKernel<kWorldStepRowsCarsApart><<<env_blocks, kStagedBlockSize, 0, stream>>>(
                cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls, car_configs,
                balls, env_states, pads, seed);
            CheckCuda(cudaGetLastError(), "StagedWorldKernel rows launch");
            StagedCarPostKernel<<<car_blocks, kStagedBlockSize, 0, stream>>>(
                cfg, env_car_count, cars, wheel_runtime_states, solver_scratch, controls, car_configs);
            CheckCuda(cudaGetLastError(), "StagedCarPostKernel launch");
            StagedEnvTailKernel<<<env_blocks, kStagedBlockSize, 0, stream>>>(
                cfg, env_car_count, cars, balls, env_states, pads, pads_apart);
            CheckCuda(cudaGetLastError(), "StagedEnvTailKernel launch");
            if (pad_post_blocks > 0) {
                StagedPadPostKernel<<<pad_post_blocks, kStagedBlockSize, 0, stream>>>(cfg, env_car_count, cars, pads);
                CheckCuda(cudaGetLastError(), "StagedPadPostKernel launch");
            }
        }
        return;
    }
    // The car-world mesh staging and the wheel rays read the cars and write apart, so with
    // ROCKETGPU_STAGED_CONCURRENT they run side by side and join before StepKernel.
    const bool concurrent = UseStagedConcurrency();
    for (int32_t t = 0; t < ticks; ++t) {
        PreTickKernel<<<pre_blocks, pre_block_size, 0, stream>>>(
            cfg, teams, env_car_count, cars, solver_scratch, controls, balls, env_states, pads,
            controls_by_tick, control_stride, env_commands_by_tick, env_command_stride,
            respawn_slots, writable_respawn_slots, respawn_rng, respawn_order, seed, t);
        CheckCuda(cudaGetLastError(), "PreTickKernel launch");
        if (concurrent) {
            StagedStreams& side = GetStagedStreams();
            CheckCuda(cudaEventRecord(side.pre_done, stream), "cudaEventRecord(pre tick)");
            CheckCuda(cudaStreamWaitEvent(side.stage, side.pre_done, 0), "cudaStreamWaitEvent(stage)");
            LaunchStageCarWorldMeshCandidates(cfg, cars, car_configs, env_car_count, side.stage);
            CheckCuda(cudaEventRecord(side.stage_done, side.stage), "cudaEventRecord(stage)");
            LaunchStageWheelRays(cfg, cars, car_configs, wheel_runtime_states, balls, env_states, solver_scratch,
                                 env_car_count, stream, false);
            CheckCuda(cudaStreamWaitEvent(stream, side.stage_done, 0), "cudaStreamWaitEvent(join stage)");
        } else {
            LaunchStageWheelRays(cfg, cars, car_configs, wheel_runtime_states, balls, env_states, solver_scratch,
                                 env_car_count, stream);
        }
        StepKernel<<<blocks, block_size, 0, stream>>>(
            cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls, car_configs,
            balls, env_states, pads, seed, lanes, step_gjk_pool, local_state);
        CheckCuda(cudaGetLastError(), "StepKernel launch");
    }
}

inline bool UseStepGraphs() {
    static const bool graphs = [] {
        const char* v = std::getenv("ROCKETGPU_STEP_GRAPHS");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return graphs;
}

// A step call's arguments, compared bytewise: a call that repeats the previous one with the same key runs as a CUDA
// graph of its launches. Its launches cost the host ~2 KB of kernel arguments each, more than a staged tick's kernels
// take to run at 1k envs.
struct StepGraphKey {
    SimConfig cfg;
    WorldSolverScratchRowPool row_pool;
    const void* ptrs[17];
    int32_t control_stride;
    int32_t env_command_stride;
    int32_t ticks;
    uint64_t seed;
    cudaStream_t stream;
    uint64_t resource_epoch;
};
struct StepGraphEntry {
    StepGraphKey key;
    cudaGraphExec_t exec = nullptr;
    uint64_t last_use = 0;
};

// Each tick of a repeated call launches the same kernels with the same arguments, so after a direct first call (which
// also grows the pools the step allocates on demand) the call is captured once and replayed.
inline void LaunchStepTicks(
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch,
    const WorldSolverScratchRowPool& row_pool,
    CarControls* controls,
    const CarControls* controls_by_tick,
    int32_t control_stride,
    const EnvCommand* env_commands_by_tick,
    int32_t env_command_stride,
    CarConfig* car_configs,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    const uint8_t* respawn_slots,
    uint8_t* writable_respawn_slots,
    RespawnRngState* respawn_rng,
    const int32_t* respawn_order,
    uint64_t seed,
    int32_t ticks,
    cudaStream_t stream) {
    const StepResourcesGuard guard(stream);
    const auto direct = [&](cudaStream_t s) {
        LaunchStepTicksDirect(cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls,
                              controls_by_tick, control_stride, env_commands_by_tick, env_command_stride, car_configs,
                              balls, env_states, pads, respawn_slots, writable_respawn_slots, respawn_rng,
                              respawn_order, seed, ticks, s);
    };
    if (!UseStepGraphs() || ticks <= 0 || StreamIsCapturing(stream)) {
        direct(stream);
        return;
    }
    StepGraphKey key;
    std::memset(&key, 0, sizeof(key));
    std::memcpy(&key.cfg, &cfg, sizeof(SimConfig));
    std::memcpy(&key.row_pool, &row_pool, sizeof(WorldSolverScratchRowPool));
    const void* const ptrs[17] = {teams, env_car_count, cars, wheel_runtime_states, solver_scratch, controls,
                                  controls_by_tick, env_commands_by_tick, car_configs, balls, env_states, pads,
                                  respawn_slots, writable_respawn_slots, respawn_rng, respawn_order, nullptr};
    std::memcpy(key.ptrs, ptrs, sizeof(ptrs));
    key.control_stride = control_stride;
    key.env_command_stride = env_command_stride;
    key.ticks = ticks;
    key.seed = seed;
    key.stream = stream;
    key.resource_epoch = StepResourceEpoch();

    static std::vector<StepGraphEntry> entries;
    static uint64_t clock = 0;
    ++clock;
    StepGraphEntry* entry = nullptr;
    for (StepGraphEntry& e : entries) {
        if (std::memcmp(&e.key, &key, sizeof(key)) == 0) {
            entry = &e;
            break;
        }
    }
    if (entry == nullptr) {
        direct(stream);
        key.resource_epoch = StepResourceEpoch();
        constexpr size_t kMaxEntries = 8;
        if (entries.size() >= kMaxEntries) {
            auto oldest = std::min_element(entries.begin(), entries.end(),
                [](const StepGraphEntry& a, const StepGraphEntry& b) { return a.last_use < b.last_use; });
            if (oldest->exec != nullptr) {
                cudaGraphExecDestroy(oldest->exec);
            }
            entries.erase(oldest);
        }
        entries.push_back(StepGraphEntry{key, nullptr, clock});
        return;
    }
    entry->last_use = clock;
    if (entry->exec == nullptr) {
        static cudaStream_t capture_stream = nullptr;
        if (capture_stream == nullptr) {
            CheckCuda(cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking), "cudaStreamCreate(step graph)");
        }
        CheckCuda(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal), "begin step graph capture");
        cudaGraph_t graph = nullptr;
        try {
            direct(capture_stream);
        } catch (...) {
            cudaStreamEndCapture(capture_stream, &graph);
            if (graph != nullptr) {
                cudaGraphDestroy(graph);
            }
            throw;
        }
        CheckCuda(cudaStreamEndCapture(capture_stream, &graph), "end step graph capture");
        const cudaError_t inst = cudaGraphInstantiate(&entry->exec, graph, 0);
        cudaGraphDestroy(graph);
        CheckCuda(inst, "cudaGraphInstantiate(step graph)");
    }
    CheckCuda(cudaGraphLaunch(entry->exec, stream), "cudaGraphLaunch(step graph)");
}

void LaunchResetToKickoff(
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    CarControls* controls,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    uint64_t seed,
    RespawnRngState* respawn_rng,
    uint8_t kickoff_seed_mode,
    cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    const int blocks = (cfg.env_count + kBlockSize - 1) / kBlockSize;
    ResetToKickoffKernel<<<blocks, kBlockSize, 0, stream>>>(
        cfg, teams, env_car_count, cars, controls, balls, env_states, pads, seed, respawn_rng, kickoff_seed_mode);
    CheckCuda(cudaGetLastError(), "ResetToKickoffKernel launch");
}

void LaunchApplyEnvCommands(
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    CarControls* controls,
    WorldSolverScratch* solver_scratch,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    const EnvCommand* env_commands,
    int32_t command_count,
    RespawnRngState* respawn_rng,
    cudaStream_t stream) {
    if (command_count <= 0) {
        return;
    }
    constexpr int kBlockSize = 256;
    const int blocks = (command_count + kBlockSize - 1) / kBlockSize;
    ApplyEnvCommandsKernel<<<blocks, kBlockSize, 0, stream>>>(
        cfg, teams, env_car_count, cars, controls, solver_scratch, balls, env_states, pads, env_commands, command_count, respawn_rng);
    CheckCuda(cudaGetLastError(), "ApplyEnvCommandsKernel launch");
}

void LaunchCanonicalizeImportedCars(
    CarState* cars,
    int32_t count,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    constexpr int kBlockSize = 256;
    const int blocks = (count + kBlockSize - 1) / kBlockSize;
    CanonicalizeImportedCarsKernel<<<blocks, kBlockSize, 0, stream>>>(cars, count);
    CheckCuda(cudaGetLastError(), "CanonicalizeImportedCarsKernel launch");
}

void LaunchCanonicalizeImportedBalls(
    BallState* balls,
    int32_t count,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    constexpr int kBlockSize = 256;
    const int blocks = (count + kBlockSize - 1) / kBlockSize;
    CanonicalizeImportedBallsKernel<<<blocks, kBlockSize, 0, stream>>>(balls, count);
    CheckCuda(cudaGetLastError(), "CanonicalizeImportedBallsKernel launch");
}

static bool StreamIsCapturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return status != cudaStreamCaptureStatusNone;
}

void LaunchStep(
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch,
    const WorldSolverScratchRowPool& row_pool,
    CarControls* controls,
    CarConfig* car_configs,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    const uint8_t* respawn_slot_overrides,
    uint64_t seed,
    int32_t ticks,
    cudaStream_t stream,
    RespawnRngState* respawn_rng,
    const int32_t* respawn_order,
    uint8_t* writable_respawn_slots) {
    LaunchStepTicks(cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls,
                    nullptr, 0, nullptr, 0, car_configs, balls, env_states, pads,
                    respawn_rng ? writable_respawn_slots : respawn_slot_overrides, writable_respawn_slots,
                    respawn_rng, respawn_order, seed, ticks, stream);
}

void LaunchStepWithControlStream(
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch,
    const WorldSolverScratchRowPool& row_pool,
    CarControls* controls,
    const CarControls* controls_by_tick,
    int32_t control_stride,
    CarConfig* car_configs,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    uint64_t seed,
    int32_t ticks,
    cudaStream_t stream,
    RespawnRngState* respawn_rng,
    const int32_t* respawn_order,
    uint8_t* respawn_slots) {
    LaunchStepTicks(cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls,
                    controls_by_tick, control_stride, nullptr, 0, car_configs, balls, env_states, pads,
                    respawn_slots, respawn_slots, respawn_rng, respawn_order, seed, ticks, stream);
}

void LaunchStepWithCommandStreams(
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch,
    const WorldSolverScratchRowPool& row_pool,
    CarControls* controls,
    const CarControls* controls_by_tick,
    int32_t control_stride,
    const EnvCommand* env_commands_by_tick,
    int32_t env_command_stride,
    CarConfig* car_configs,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    uint64_t seed,
    int32_t ticks,
    cudaStream_t stream,
    RespawnRngState* respawn_rng,
    const int32_t* respawn_order,
    uint8_t* respawn_slots) {
    LaunchStepTicks(cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls,
                    controls_by_tick, control_stride, env_commands_by_tick, env_command_stride, car_configs, balls,
                    env_states, pads, respawn_slots, respawn_slots, respawn_rng, respawn_order, seed, ticks, stream);
}
}
}

#endif
