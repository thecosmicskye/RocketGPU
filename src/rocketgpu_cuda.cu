#include "rocketgpu/detail/cuda_api.hpp"

#if defined(ROCKETGPU_USE_CUDA)

#include <algorithm>
#include <chrono>
#include <cstddef>
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
__device__ BallWorldStage* g_rocketgpu_ball_world_stage = nullptr;
namespace bullet_port {
__device__ EPA* g_rocketgpu_epa_pool = nullptr;
__device__ unsigned int* g_rocketgpu_epa_busy = nullptr;
__device__ unsigned int g_rocketgpu_epa_slots = 0;
}

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
        if (car.respawn_slot_override_plus1 >= kKickoffSpawnOverridePlus1) continue;
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
    uint8_t kickoff_seed_mode,
    const CarConfig* car_configs) {
    const int env_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (env_idx >= cfg.env_count) {
        return;
    }
    ResetEnvToKickoff(env_idx, cfg, teams, env_car_count, cars, controls, balls, pads, seed, false, env_states,
                      kickoff_seed_mode, respawn_rng, nullptr, false, car_configs);
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
    RespawnRngState* respawn_rng,
    const CarConfig* car_configs) {
    if (solver_scratch != nullptr) {
        ApplyEnvCommandBodies(
            solver_scratch[env_idx], command, env_car_count[env_idx], cars + env_idx * cfg.max_cars_per_env);
    }
    ApplyEnvCommand(env_idx, cfg, teams, env_car_count, cars, controls, balls, env_states, pads, command, respawn_rng, nullptr,
                    car_configs);
    if (solver_scratch != nullptr && command.reset_to_kickoff) {
        for (int manifold_idx = 0; manifold_idx < kMaxBallWorldMeshPersistentManifolds; ++manifold_idx) {
            solver_scratch[env_idx].ball_world_mesh_manifolds[manifold_idx] = BallMeshPersistentManifold{};
        }
        for (int manifold_idx = 0; manifold_idx < kBallWorldPlaneSlots; ++manifold_idx) {
            solver_scratch[env_idx].ball_world_plane_manifolds[manifold_idx] = BallMeshPersistentManifold{};
        }
        solver_scratch[env_idx].retained_reset_pending = 1;
        for (int k = 0; k < 8; ++k) {
            solver_scratch[env_idx].car_world_reset[k] = 1;
        }
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
    RespawnRngState* respawn_rng,
    const CarConfig* car_configs) {
    const int env_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (env_idx >= command_count) {
        return;
    }
    ApplyEnvCommandForEnv(env_idx, cfg, teams, env_car_count, cars, controls, solver_scratch, balls, env_states, pads,
                          env_commands[env_idx], respawn_rng, car_configs);
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

__device__ __forceinline__ void PreTickBlock(
    const SimConfig& cfg,
    int block_env_count,
    int block_env0,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    WorldSolverScratch* solver_scratch,
    CarControls* controls,
    CarConfig* car_configs,
    WheelRuntimeState* wheel_runtime_states,
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
    bool pad_pre_tick,
    int32_t* zero_ints,
    int zero_count) {
    if (zero_ints != nullptr && blockIdx.x == 0) {
        for (int i = static_cast<int>(threadIdx.x); i < zero_count; i += static_cast<int>(blockDim.x)) {
            zero_ints[i] = 0;
        }
    }
    const int slots = cfg.max_cars_per_env;
    const int car_threads = block_env_count * slots;
    const int t = static_cast<int>(threadIdx.x);
    const bool ball_thread = t >= car_threads;
    const int local_env = ball_thread ? t - car_threads : t / slots;
    const int slot = ball_thread ? 0 : t - local_env * slots;
    int env_idx = block_env0 + local_env;
    const bool active = local_env < block_env_count && env_idx < cfg.env_count;
    if (active) {
        MapStepEnv(cfg, env_idx);
    }
    if (active && ball_thread) {
        if (env_commands_by_tick) {
            const EnvCommand command = env_commands_by_tick[
                static_cast<size_t>(tick_idx) * static_cast<size_t>(env_command_stride) + static_cast<size_t>(env_idx)];
            ApplyEnvCommandForEnv(env_idx, cfg, teams, env_car_count, cars, controls, solver_scratch,
                                  balls, env_states, pads, command, respawn_rng, car_configs);
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
        if (ball_thread) {
            StepEnvPreTickBall(env_idx, cfg, solver_scratch, balls);
        } else if (slot < env_car_count[env_idx]) {
            const int car_idx = env_idx * slots + slot;
            StepEnvPreTickCar(env_idx, slot, cfg, teams, cars, solver_scratch, env_states[env_idx], respawn_slots, seed,
                              controls[car_idx], car_configs[car_idx],
                              wheel_runtime_states ? &wheel_runtime_states[car_idx] : nullptr);
        }
    }
    if (pad_pre_tick && cfg.enable_boost_pads) {
        const int pad_count = BoostPadCount(cfg);
        const int block_envs = Min(block_env_count, cfg.env_count - block_env0);
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
ROCKETGPU_GLOBAL void PreTickKernel(
    const __grid_constant__ SimConfig cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    WorldSolverScratch* solver_scratch,
    CarControls* controls,
    CarConfig* car_configs,
    WheelRuntimeState* wheel_runtime_states,
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
    bool pad_pre_tick = true,
    int32_t* zero_ints = nullptr,
    int zero_count = 0) {
    const int block_env_count = static_cast<int>(blockDim.x) / (cfg.max_cars_per_env + 1);
    PreTickBlock(cfg, block_env_count, static_cast<int>(blockIdx.x) * block_env_count, teams, env_car_count, cars,
                 solver_scratch, controls, car_configs, wheel_runtime_states, balls, env_states, pads, controls_by_tick,
                 control_stride,
                 env_commands_by_tick, env_command_stride, respawn_slots, writable_respawn_slots, respawn_rng,
                 respawn_order, seed, tick_idx, pad_pre_tick, zero_ints, zero_count);
}

__device__ __noinline__ void BuildCarWorldRowsForLane(int env_idx, int lane, int lanes, int count, const SimConfig& cfg,
    CarState* cars, const CarConfig* car_configs, WorldSolverScratch* solver_scratch, WorldSolverScratchRowPool row_pool,
    bullet_port::btGjkEpaScratch* gjk_scratch) {
    for (int slot = lane; slot < count; slot += lanes) {
        BuildCarWorldRowsForCar(env_idx, slot, cfg, cars, car_configs, solver_scratch[env_idx], row_pool.view(env_idx),
                                gjk_scratch);
    }
}

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
    const bool rows_apart = apart && !cfg.mutator.physics_events.HasNonGoalEvents();
    if (apart && active && VehicleUpdatesIndependent(cfg, solver_scratch[env_idx].wheel_ray_stage, count)) {
        for (int slot = lane; slot < count; slot += lanes) {
            uint32_t pushed_cars_mask = 0u;
            StepEnvVehicleUpdateCar(env_idx, slot, count, cfg, cars, wheel_runtime_states, car_configs, balls,
                                    controls[env_idx * stride + slot], solver_scratch[env_idx].wheel_ray_stage,
                                    pushed_cars_mask, &solver_scratch[env_idx].bp.sweep);
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

ROCKETGPU_GLOBAL void StagedBallWorldKernel(
    const __grid_constant__ SimConfig cfg, const BallState* balls, const EnvState* env_states, BallWorldStage* stages) {
    __shared__ alignas(ArenaMeshBvhMeshInfo) unsigned char mesh_storage[kBallWorldStageMaxMeshes * sizeof(ArenaMeshBvhMeshInfo)];
    ArenaMeshBvhMeshInfo* meshes = reinterpret_cast<ArenaMeshBvhMeshInfo*>(mesh_storage);
    const int mesh_count = cfg.arena_mesh_bvh_meshes != nullptr ? cfg.arena_mesh_bvh_mesh_count : 0;
    if (mesh_count <= kBallWorldStageMaxMeshes) {
        for (int m = static_cast<int>(threadIdx.x); m < mesh_count; m += static_cast<int>(blockDim.x)) {
            memcpy(&meshes[m], &cfg.arena_mesh_bvh_meshes[m], sizeof(ArenaMeshBvhMeshInfo));
        }
    } else {
        meshes = nullptr;
    }
    __syncthreads();
    int env_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (env_idx >= cfg.env_count) {
        return;
    }
    MapStepEnv(cfg, env_idx);
    StageBallWorldContacts(balls[env_idx], cfg.mutator, static_cast<uint64_t>(env_states[env_idx].tick),
                           cfg.arena_mesh_tris, cfg.arena_mesh_tri_count, cfg.arena_mesh_bvh_quantization,
                           cfg.arena_mesh_bvh_quantization_count, cfg.arena_mesh_bvh_meshes,
                           cfg.arena_mesh_bvh_mesh_count, cfg.arena_mesh_bvh_nodes, cfg.arena_mesh_bvh_node_count,
                           stages[env_idx], meshes);
}

static_assert(offsetof(WheelRayStage, valid) == offsetof(WheelRayStage, ground_kind) + 3 &&
              offsetof(WheelRayStage, ground_kind) % 4 == 0 && sizeof(WheelRayStage) % 4 == 0, "ray flags word");
__device__ __forceinline__ void LoadStagedRayFlags(const WorldSolverScratch& scratch, int count,
                                                   uint32_t (&flags)[kWheelRayStageSlots]) {
#pragma unroll
    for (int lane = 0; lane < kWheelRayStageSlots; ++lane) {
        flags[lane] = lane < 4 * count
            ? *reinterpret_cast<const uint32_t*>(&scratch.wheel_ray_stage[lane].ground_kind) : 0x01000000u;
    }
}
__device__ __forceinline__ bool StagedVehiclesIndependentFromFlags(
    const SimConfig& cfg, int count, const uint32_t (&flags)[kWheelRayStageSlots]) {
    if (cfg.mutator.physics_events.HasNonGoalEvents() || count > kWheelRayStageSlots / 4) {
        return false;
    }
    uint32_t fails = 0u;
#pragma unroll
    for (int lane = 0; lane < kWheelRayStageSlots; ++lane) {
        const uint32_t w = flags[lane];
        const uint32_t invalid = (w >> 24) != 1u ? 1u : 0u;
        const uint32_t in_contact = ((w >> 8) & 0xFFu) != 0u ? 1u : 0u;
        const uint32_t with_world = ((w >> 16) & 0xFFu) != 0u ? 1u : 0u;
        fails |= invalid | (in_contact & (with_world ^ 1u));
    }
    return fails == 0u;
}
__device__ __forceinline__ bool StagedVehiclesIndependent(const SimConfig& cfg, const WorldSolverScratch& scratch, int count) {
    if (cfg.mutator.physics_events.HasNonGoalEvents() || count > kWheelRayStageSlots / 4) {
        return false;
    }
    uint32_t flags[kWheelRayStageSlots];
    LoadStagedRayFlags(scratch, count, flags);
    return StagedVehiclesIndependentFromFlags(cfg, count, flags);
}
__device__ __forceinline__ uint8_t StagedVehicleApartMaskFromFlags(
    const SimConfig& cfg, const WorldSolverScratch& scratch, int count, const uint32_t (&flags)[kWheelRayStageSlots]) {
    if (cfg.mutator.physics_events.HasNonGoalEvents() || count > kWheelRayStageSlots / 4) {
        return 0u;
    }
    uint32_t ordered = 0u;
    bool all_ordered = false;
#pragma unroll
    for (int lane = 0; lane < kWheelRayStageSlots; ++lane) {
        const uint32_t w = flags[lane];
        if ((w >> 24) != 1u) {
            all_ordered = all_ordered || lane < 4 * count;
        } else if (((w >> 8) & 0xFFu) != 0u && ((w >> 16) & 0xFFu) == 0u) {
            const int target = scratch.wheel_ray_stage[lane].ground_car_idx;
            if ((w & 0xFFu) == ROCKETGPU_ARENA_RAY_HIT_BALL) {
                ordered |= 1u << (lane / 4);
            } else if ((w & 0xFFu) != ROCKETGPU_ARENA_RAY_HIT_CAR || target < 0 || target >= count) {
                all_ordered = true;
            } else {
                ordered |= (1u << (lane / 4)) | (1u << target);
            }
        }
    }
    return static_cast<uint8_t>(all_ordered ? 0u : (((1u << count) - 1u) & ~ordered));
}
__device__ __forceinline__ uint8_t StagedVehicleApartMask(const SimConfig& cfg, const WorldSolverScratch& scratch, int count) {
    if (cfg.mutator.physics_events.HasNonGoalEvents() || count > kWheelRayStageSlots / 4) {
        return 0u;
    }
    uint32_t flags[kWheelRayStageSlots];
    LoadStagedRayFlags(scratch, count, flags);
    return StagedVehicleApartMaskFromFlags(cfg, scratch, count, flags);
}
ROCKETGPU_GLOBAL void StagedVehicleIndependenceKernel(
    const __grid_constant__ SimConfig cfg, const int32_t* env_car_count, WorldSolverScratch* solver_scratch) {
    int env_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (env_idx >= cfg.env_count) {
        return;
    }
    MapStepEnv(cfg, env_idx);
    WorldSolverScratch& scratch = solver_scratch[env_idx];
    const int count = env_car_count[env_idx];
    if (cfg.mutator.physics_events.HasNonGoalEvents() || count > kWheelRayStageSlots / 4) {
        scratch.vehicle_updates_independent = 0;
        scratch.vehicle_apart_mask = 0u;
        return;
    }
    uint32_t flags[kWheelRayStageSlots];
    LoadStagedRayFlags(scratch, count, flags);
    const bool independent = StagedVehiclesIndependentFromFlags(cfg, count, flags);
    scratch.vehicle_updates_independent = independent ? 1 : 0;
    scratch.vehicle_apart_mask = independent
        ? static_cast<uint8_t>((1u << count) - 1u)
        : StagedVehicleApartMaskFromFlags(cfg, scratch, count, flags);
}
__device__ __forceinline__ bool EnvVehiclesIndependent(const SimConfig& cfg, const WorldSolverScratch& scratch, int count,
                                                       bool rays_inline) {
    return rays_inline ? StagedVehiclesIndependent(cfg, scratch, count) : scratch.vehicle_updates_independent != 0;
}

template <typename T>
__device__ __forceinline__ void PrefetchLinesL2(const T* p) {
    const char* c = reinterpret_cast<const char*>(p);
    for (int off = 0; off < static_cast<int>(sizeof(T)); off += 128) {
        asm volatile("prefetch.global.L2 [%0];" :: "l"(c + off));
    }
}
ROCKETGPU_GLOBAL void StagedCarPreKernel(
    const __grid_constant__ SimConfig cfg, int32_t* __restrict__ env_car_count, CarState* __restrict__ cars,
    WheelRuntimeState* __restrict__ wheel_runtime_states, WorldSolverScratch* __restrict__ solver_scratch,
    CarControls* __restrict__ controls, CarConfig* __restrict__ car_configs, BallState* __restrict__ balls, bool rays_inline) {
    const int stride = cfg.max_cars_per_env;
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    int env_idx = static_cast<int>(tid / stride);
    const int slot = static_cast<int>(tid - static_cast<long long>(env_idx) * stride);
    if (env_idx >= cfg.env_count) {
        return;
    }
    MapStepEnv(cfg, env_idx);
    const int car_idx = env_idx * stride + slot;
    const CarControls car_controls = controls[car_idx];
    const int count = env_car_count[env_idx];
    if (slot >= count || !EnvVehiclesIndependent(cfg, solver_scratch[env_idx], count, rays_inline)) {
        return;
    }
    uint32_t pushed_cars_mask = 0u;
    PrefetchLinesL2(&cars[car_idx]);
    if (wheel_runtime_states != nullptr) PrefetchLinesL2(&wheel_runtime_states[car_idx]);
    PrefetchLinesL2(&car_configs[car_idx]);
    for (int w = 0; w < 4; ++w) PrefetchLinesL2(&solver_scratch[env_idx].wheel_ray_stage[4 * slot + w]);
    StepEnvVehicleUpdateCarInline<true>(env_idx, slot, count, cfg, cars, wheel_runtime_states, car_configs, balls,
                                  car_controls, solver_scratch[env_idx].wheel_ray_stage, pushed_cars_mask,
                                  &solver_scratch[env_idx].bp.sweep);
    WorldSolverScratch& scratch = solver_scratch[env_idx];
    WorldStepPrestepCar(cars[car_idx], car_configs[car_idx], wheel_runtime_states ? &wheel_runtime_states[car_idx] : nullptr,
                        cfg.mutator, cfg.tick_dt,
                        scratch.pre_contact_cars[slot], scratch.deferred_linear_deltas[slot],
                        scratch.deferred_angular_deltas[slot]);
}

__device__ __forceinline__ void StagedCarPostOne(
    const SimConfig& cfg, long long tid, int32_t* env_car_count, CarState* cars, WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch, CarControls* controls, CarConfig* car_configs) {
    const int stride = cfg.max_cars_per_env;
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
ROCKETGPU_GLOBAL void StagedCarPostKernel(
    const __grid_constant__ SimConfig cfg, int32_t* env_car_count, CarState* cars, WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch, CarControls* controls, CarConfig* car_configs) {
    StagedCarPostOne(cfg, static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x, env_car_count, cars,
                     wheel_runtime_states, solver_scratch, controls, car_configs);
}

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

constexpr int kEnvTailLanes = 8;
__device__ __forceinline__ void StagedEnvTailOne(
    const SimConfig& cfg, int tid, int32_t* env_car_count, CarState* cars, const CarConfig* car_configs,
    BallState* balls, EnvState* env_states, BoostPadState* pads, bool pads_apart) {
    int env_idx = tid / kEnvTailLanes;
    const int lane = tid - env_idx * kEnvTailLanes;
    if (env_idx >= cfg.env_count) {
        return;
    }
    MapStepEnv(cfg, env_idx);
    const int count = env_car_count[env_idx];
    bool lanes_apart = cfg.enable_boost_pads && cfg.max_cars_per_env <= kEnvTailLanes && pads_apart;
    if (lanes_apart && FrameFrac(cfg, env_idx) >= 0.0f) {
        const unsigned group = ((1u << kEnvTailLanes) - 1u)
                               << (threadIdx.x & 31u & ~static_cast<unsigned>(kEnvTailLanes - 1));
        const int base = env_idx * cfg.max_cars_per_env;
        uint64_t hits = 0u;
        uint64_t overlap = 0u;
        uint64_t near = 0u;
        int touching = 0;
        int eligible = 0;
        if (lane < count) {
            const CarState& car = cars[base + lane];
            hits = BoostPadContactsForCar(env_idx, cfg, car, car_configs[base + lane], pads, overlap);
            touching = car.boost_pad_touching;
            near = hits;
            if (touching != 0) near |= uint64_t{1} << (touching - 1);
            eligible = BoostPadEligible(cfg, car) ? 1 : 0;
        }
        if (__any_sync(group, BoostPadHitCount(near) > 1)) {
            lanes_apart = false;
        } else {
            const EnvState& env = env_states[env_idx];
            const int pad_count = BoostPadCount(cfg);
            const int pad_base = env_idx * pad_count;
            uint64_t share = 0u;
            for (int p = lane; p < pad_count; p += kEnvTailLanes) {
                share |= uint64_t{1} << p;
            }
            for (int k = 0; k < count; ++k) {
                const int i = CarVisitSlot(env, k, count);
                const uint64_t car_hits = __shfl_sync(group, hits, i, kEnvTailLanes);
                const uint64_t car_overlap = __shfl_sync(group, overlap, i, kEnvTailLanes);
                const int car_eligible = __shfl_sync(group, eligible, i, kEnvTailLanes);
                const int car_touching = __shfl_sync(group, touching, i, kEnvTailLanes);
                CarState& car = cars[base + i];
                uint64_t mine = car_hits & share;
                if (car_touching != 0 && ((share >> (car_touching - 1)) & 1u)) {
                    mine |= uint64_t{1} << (car_touching - 1);
                }
                while (mine != 0u) {
                    const int p = __ffsll(static_cast<long long>(mine)) - 1;
                    mine &= mine - 1u;
                    BoostPadTouchStep(pads[pad_base + p], p, car, static_cast<uint32_t>(base + i + 1),
                                      ((car_hits >> p) & 1u) != 0u, ((car_overlap >> p) & 1u) != 0u, car_eligible != 0);
                }
            }
            __syncwarp(group);
            bool respawn_due = false;
            for (int p = lane; p < pad_count; p += kEnvTailLanes) {
                const BoostPadState& pad = pads[pad_base + p];
                respawn_due = respawn_due || (!pad.is_active && BoostPadRespawnDue(pad, cfg.tick_dt, FrameFrac(cfg, env_idx)));
            }
            if (__any_sync(group, respawn_due) && lane == 0) {
                for (int p = 0; p < pad_count; ++p) {
                    BoostPadRespawnStep(env_idx, count, cfg, cars, car_configs, env, pads[pad_base + p], p);
                }
            }
            __syncwarp(group);
            for (int k = count - 1; k >= 0; --k) {
                const int i = CarVisitSlot(env, k, count);
                const int car_eligible = __shfl_sync(group, eligible, i, kEnvTailLanes);
                const CarState& car = cars[base + i];
                const int touching = car.boost_pad_touching;
                if (touching != 0 && ((share >> (touching - 1)) & 1u)) {
                    BoostPadTouchListStep(pads[pad_base + touching - 1], touching - 1, car,
                                          static_cast<uint32_t>(base + i + 1), car_eligible != 0);
                }
            }
            __syncwarp(group);
            if (lane < count) {
                BoostPadStoreFramePos(cfg, env_idx, cars[base + lane]);
            }
        }
    }
    if (lane != 0) {
        return;
    }
    StepEnvTail(env_idx, count, cfg, cars, car_configs, balls[env_idx], env_states[env_idx], pads, true,
                !lanes_apart);
}
ROCKETGPU_GLOBAL void StagedEnvTailKernel(
    const __grid_constant__ SimConfig cfg, int32_t* env_car_count, CarState* cars, const CarConfig* car_configs,
    BallState* balls, EnvState* env_states, BoostPadState* pads, bool pads_apart) {
    StagedEnvTailOne(cfg, static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x), env_car_count, cars, car_configs,
                     balls, env_states, pads, pads_apart);
}

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

__device__ __forceinline__ void PostFusedBlock(
    const SimConfig& cfg, int32_t* env_car_count, CarState* cars, WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch, CarControls* controls, CarConfig* car_configs, BallState* balls,
    EnvState* env_states, BoostPadState* pads, bool pads_apart, int envs_per_block) {
    const long long e0 = static_cast<long long>(blockIdx.x) * envs_per_block;
    const int stride = cfg.max_cars_per_env;
    for (int t = threadIdx.x; t < envs_per_block * stride; t += blockDim.x) {
        StagedCarPostOne(cfg, e0 * stride + t, env_car_count, cars, wheel_runtime_states, solver_scratch, controls,
                         car_configs);
    }
    __syncthreads();
    for (int t = threadIdx.x; t < envs_per_block * kEnvTailLanes; t += blockDim.x) {
        StagedEnvTailOne(cfg, static_cast<int>(e0 * kEnvTailLanes + t), env_car_count, cars, car_configs, balls,
                         env_states, pads, pads_apart);
    }
}
ROCKETGPU_GLOBAL void StagedPostFusedKernel(
    const __grid_constant__ SimConfig cfg, int32_t* env_car_count, CarState* cars, WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch, CarControls* controls, CarConfig* car_configs, BallState* balls,
    EnvState* env_states, BoostPadState* pads, bool pads_apart, int envs_per_block) {
    PostFusedBlock(cfg, env_car_count, cars, wheel_runtime_states, solver_scratch, controls, car_configs, balls,
                   env_states, pads, pads_apart, envs_per_block);
}

ROCKETGPU_GLOBAL void StagedPostPreFusedKernel(
    const __grid_constant__ SimConfig cfg, Team* teams, int32_t* env_car_count, CarState* cars,
    WheelRuntimeState* wheel_runtime_states, WorldSolverScratch* solver_scratch, CarControls* controls,
    CarConfig* car_configs, BallState* balls, EnvState* env_states, BoostPadState* pads, bool pads_apart,
    int envs_per_block, const CarControls* controls_by_tick, int32_t control_stride,
    const EnvCommand* env_commands_by_tick, int32_t env_command_stride, const uint8_t* respawn_slots,
    uint8_t* writable_respawn_slots, RespawnRngState* respawn_rng, const int32_t* respawn_order, uint64_t seed,
    int32_t next_tick_idx, bool pad_pre_tick, int32_t* zero_ints, int zero_count) {
    PostFusedBlock(cfg, env_car_count, cars, wheel_runtime_states, solver_scratch, controls, car_configs, balls,
                   env_states, pads, pads_apart, envs_per_block);
    __syncthreads();
    PreTickBlock(cfg, envs_per_block, static_cast<int>(blockIdx.x) * envs_per_block, teams, env_car_count, cars,
                 solver_scratch, controls, car_configs, wheel_runtime_states, balls, env_states, pads, controls_by_tick,
                 control_stride,
                 env_commands_by_tick, env_command_stride, respawn_slots, writable_respawn_slots, respawn_rng,
                 respawn_order, seed, next_tick_idx, pad_pre_tick, zero_ints, zero_count);
}

constexpr int kWorld1Lanes = 4;
template <int kWorldPart, int kLanes = 1, bool kRegisterIterations = false, int kW1Split = 0>
ROCKETGPU_GLOBAL void StagedWorldKernel(
    const __grid_constant__ SimConfig cfg, Team* teams, int32_t* env_car_count, CarState* cars, WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch, WorldSolverScratchRowPool row_pool, CarControls* controls,
    CarConfig* car_configs, BallState* balls, EnvState* env_states, BoostPadState* pads, uint64_t seed,
    bool rays_inline = false) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int env_idx = tid / kLanes;
    const int lane = tid - env_idx * kLanes;
    if (env_idx >= cfg.env_count) {
        return;
    }
    MapStepEnv(cfg, env_idx);
    if (WorldPartIsContacts(kWorldPart) || WorldPartIsPrefix(kWorldPart)) {
        const bool independent = EnvVehiclesIndependent(cfg, solver_scratch[env_idx], env_car_count[env_idx], rays_inline);
        if (independent != ((kWorldPart == kWorldStepContactsCarsApart && kW1Split != 2) ||
                            WorldPartIsPrefix(kWorldPart))) {
            return;
        }
    }
    uint8_t apart_mask = 0;
    uint8_t updated_mask = 0;
    if constexpr (kLanes > 1) {
        const int stride = cfg.max_cars_per_env;
        const int count = env_car_count[env_idx];
        WorldSolverScratch& scratch = solver_scratch[env_idx];
        apart_mask = rays_inline ? StagedVehicleApartMask(cfg, scratch, count) : scratch.vehicle_apart_mask;
        updated_mask = static_cast<uint8_t>((1u << count) - 1u);
        const unsigned env_lanes = ((1u << kLanes) - 1u) << (threadIdx.x & 31u & ~static_cast<unsigned>(kLanes - 1));
        uint32_t mine = 0u;
        if (lane != 0) {
            int nth = 0;
            for (uint32_t m = apart_mask; m != 0u; m &= m - 1u, ++nth) {
                if (nth % (kLanes - 1) == lane - 1) {
                    mine |= m & (0u - m);
                }
            }
        }
        const int n_apart = __popc(static_cast<unsigned>(apart_mask));
        const int rounds = max(count - n_apart, (n_apart + kLanes - 2) / (kLanes - 1));
        uint32_t pushed_cars_mask = 0u;
        int k = 0;
        uint32_t left = mine;
        for (int r = 0; r < rounds; ++r) {
            int slot = -1;
            if (lane == 0) {
                while (k < count) {
                    const int i = CarVisitSlot(env_states[env_idx], k++, count);
                    if (!((apart_mask >> i) & 1u)) {
                        slot = i;
                        break;
                    }
                }
            } else if (left != 0u) {
                slot = __ffs(static_cast<int>(left)) - 1;
                left &= left - 1u;
            }
            __syncwarp(env_lanes);
            if (slot >= 0) {
                StepEnvVehicleUpdateCar(env_idx, slot, count, cfg, cars, wheel_runtime_states, car_configs, balls,
                                        controls[env_idx * stride + slot], scratch.wheel_ray_stage, pushed_cars_mask,
                                        &scratch.bp.sweep);
            }
        }
        for (uint32_t m = mine; m != 0u; m &= m - 1u) {
            const int slot = __ffs(static_cast<int>(m)) - 1;
            const int car_idx = env_idx * stride + slot;
            WorldStepPrestepCar(cars[car_idx], car_configs[car_idx],
                                wheel_runtime_states ? &wheel_runtime_states[car_idx] : nullptr, cfg.mutator,
                                cfg.tick_dt, scratch.pre_contact_cars[slot],
                                scratch.deferred_linear_deltas[slot], scratch.deferred_angular_deltas[slot]);
        }
        if constexpr (kW1Split == 1) {
            for (int i = 0; lane == 0 && i < count; ++i) {
                if ((apart_mask >> i) & 1u) {
                    continue;
                }
                const int car_idx = env_idx * stride + i;
                WorldStepPrestepCar(cars[car_idx], car_configs[car_idx],
                                    wheel_runtime_states ? &wheel_runtime_states[car_idx] : nullptr, cfg.mutator,
                                    cfg.tick_dt, scratch.pre_contact_cars[i],
                                    scratch.deferred_linear_deltas[i], scratch.deferred_angular_deltas[i]);
            }
            return;
        }
        __syncwarp(env_lanes);
        if (lane != 0) {
            return;
        }
    }
    InternalControlsAccessor accessor;
    accessor.controls = controls;
    accessor.base = env_idx * cfg.max_cars_per_env;
    StepEnvGeneric(env_idx, cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls,
                   accessor, NoopEnvCommandAccessor{}, car_configs, balls, env_states, pads, nullptr, seed, 1, nullptr,
                   kWorldPart == kWorldStepContacts ? kStepEnvPhasePost : kStepEnvPhasePostVehiclesDone, 0, kWorldPart,
                   apart_mask, updated_mask, kRegisterIterations);
}

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

constexpr int kRowsFullKeys = kCarWorldMeshStageMeshes;
ROCKETGPU_GLOBAL void StagedCarWorldRowsSplitKernel(
    const __grid_constant__ SimConfig cfg, const int32_t* env_car_count, const CarState* cars, const CarConfig* car_configs,
    const WorldSolverScratch* solver_scratch, int32_t* row_cars, int32_t* row_car_count, uint32_t* row_reach) {
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
            const CarState& car = cars[car_idx];
            const uint8_t demoed = solver_scratch[env_idx].pre_contact_cars[slot].is_demoed;
            const uint8_t respawned = car.just_respawned;
            if (demoed != 0 || respawned != 0) {
                kind = 0;
            } else {
                const CarConfig& car_cfg = car_configs[car_idx];
                bool in_reach = cfg.arena_mesh_bvh_quantization == nullptr || cfg.arena_mesh_bvh_meshes == nullptr ||
                                cfg.arena_mesh_bvh_nodes == nullptr || cfg.arena_mesh_bvh_mesh_count <= 0 ||
                                cfg.arena_mesh_bvh_node_count <= 0;
                int key = kRowsFullKeys - 1;
                uint32_t reach_mask = 0u;
                if (!in_reach) {
                    const CarWorldMeshReach reach = ComputeCarWorldMeshReach(
                        car, car_cfg, cfg.tick_dt);
                    const float fx = cfg.arena_mesh_bvh_meshes[0].free_half_x;
                    const float fy = cfg.arena_mesh_bvh_meshes[0].free_half_y;
                    if (!(reach.hi.x < fx && reach.lo.x > -fx && reach.hi.y < fy && reach.lo.y > -fy)) {
                        for (int m = 0; m < cfg.arena_mesh_bvh_mesh_count; ++m) {
                            const ArenaMeshBvhMeshInfo& info = cfg.arena_mesh_bvh_meshes[m];
                            const bool hit = CarWorldMeshInfoUsable(info, cfg.arena_mesh_bvh_quantization_count,
                                                                    cfg.arena_mesh_bvh_node_count) &
                                             CarWorldMeshInReach(reach, info);
                            key = hit && !in_reach && m < key ? m : key;
                            reach_mask |= hit && m < 32 ? 1u << m : 0u;
                            in_reach |= hit;
                        }
                    }
                }
                if (in_reach) {
                    kind = 3 + key;
                    row_reach[car_idx] = reach_mask;
                } else {
                    kind = CarWorldPlanesOutOfReach(car, car_cfg) < 0 ? 2 : 0;
                }
            }
        }
    }
    const unsigned lane = threadIdx.x & 31u;
    const unsigned mask = __match_any_sync(0xffffffffu, kind);
    const int leader = __ffs(static_cast<int>(mask)) - 1;
    int base = 0;
    if (kind >= 0 && static_cast<int>(lane) == leader) {
        base = atomicAdd(&row_car_count[kind], __popc(mask));
    }
    base = __shfl_sync(0xffffffffu, base, leader);
    if (kind >= 0) {
        const long long rank = base + __popc(mask & ((1u << lane) - 1u));
        row_cars[kind == 2 ? car_slots - 1 - rank : (kind == 0 ? rank : (kind - 2) * car_slots + rank)] = car_idx;
    }
}

template <bool kMeshesOutOfReach>
ROCKETGPU_GLOBAL void StagedCarWorldRowsListKernel(
    const __grid_constant__ SimConfig cfg, CarState* cars, const CarConfig* car_configs, WorldSolverScratch* solver_scratch,
    WorldSolverScratchRowPool row_pool, const int32_t* row_cars, const int32_t* row_car_count,
    bullet_port::btGjkEpaScratch* gjk_pool, const uint32_t* row_reach) {
    const int stride = cfg.max_cars_per_env;
    const long long car_slots = static_cast<long long>(cfg.env_count) * stride;
    int32_t key_count[kRowsFullKeys];
    long long full_count = 0;
#pragma unroll
    for (int key = 0; key < kRowsFullKeys; ++key) {
        key_count[key] = kMeshesOutOfReach ? 0 : row_car_count[3 + key];
        full_count += key_count[key];
    }
    const long long front = kMeshesOutOfReach ? row_car_count[0] : full_count;
    const long long count = front + (kMeshesOutOfReach ? row_car_count[2] : 0);
    if (count <= 0) {
        return;
    }
    const long long first = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (long long k = first; k < count; k += static_cast<long long>(gridDim.x) * blockDim.x) {
        long long at = k < front ? k : car_slots - 1 - (k - front);
        if (!kMeshesOutOfReach) {
            long long rest = k;
            int key = 0;
#pragma unroll
            for (int b = 0; b < kRowsFullKeys - 1; ++b) {
                const bool past = key == b && rest >= key_count[b];
                rest -= past ? key_count[b] : 0;
                key += past ? 1 : 0;
            }
            at = (1 + key) * car_slots + rest;
        }
        const int32_t car_idx = row_cars[at];
        const int env_idx = car_idx / stride;
        const int slot = car_idx - env_idx * stride;
        BuildCarWorldRowsForCar<kMeshesOutOfReach>(env_idx, slot, cfg, cars, car_configs, solver_scratch[env_idx],
                                                   row_pool.view(env_idx), kMeshesOutOfReach ? nullptr : gjk_pool + first,
                                                   kMeshesOutOfReach ? 0u : row_reach[car_idx]);
    }
}

ROCKETGPU_DEVICE bool CarWorldRowsSplitPlanes(
    const SimConfig& cfg, const CarState& car, const CarConfig& car_cfg, const WorldSolverScratch& scratch, int slot,
    uint32_t hint) {
    return hint != 0u && slot < kMaxCarWorldMeshPersistentCars && cfg.mutator.arena_collision &&
           !cfg.mutator.physics_events.windows && cfg.arena_mesh_tris != nullptr && cfg.arena_mesh_tri_count > 0 &&
           cfg.arena_mesh_bvh_mesh_count <= 32 && !scratch.pre_contact_cars[slot].is_demoed && !car.just_respawned &&
           4 * (__popc(hint) + CarWorldPlanesInReachCount(car, car_cfg)) <= kMaxCarWorldVelocityContactRows;
}

ROCKETGPU_GLOBAL void __launch_bounds__(64) StagedCarWorldRowsSplitPlanesKernel(
    const __grid_constant__ SimConfig cfg, CarState* cars, const CarConfig* car_configs, WorldSolverScratch* solver_scratch,
    WorldSolverScratchRowPool row_pool, const int32_t* row_cars, const int32_t* row_car_count,
    bullet_port::btGjkEpaScratch* gjk_pool, const uint32_t* row_reach, CarWorldLocalRows* plane_rows,
    CarWorldRowsPart* plane_parts, int bs) {
    const int stride = cfg.max_cars_per_env;
    const long long car_slots = static_cast<long long>(cfg.env_count) * stride;
    int32_t key_count[kRowsFullKeys];
    long long count = 0;
#pragma unroll
    for (int key = 0; key < kRowsFullKeys; ++key) {
        key_count[key] = row_car_count[3 + key];
        count += key_count[key];
    }
    if (count <= 0) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const bool planes_warp = threadIdx.x >= 32u;
    for (long long base = static_cast<long long>(blockIdx.x) * bs; base < count;
         base += static_cast<long long>(gridDim.x) * bs) {
        const long long k = base + lane;
        const bool active = lane < bs && k < count;
        int env_idx = 0;
        int slot = 0;
        int car_idx = 0;
        uint32_t hint = 0u;
        bool split = false;
        if (active) {
            long long rest = k;
            int key = 0;
#pragma unroll
            for (int b = 0; b < kRowsFullKeys - 1; ++b) {
                const bool past = key == b && rest >= key_count[b];
                rest -= past ? key_count[b] : 0;
                key += past ? 1 : 0;
            }
            car_idx = row_cars[(1 + key) * car_slots + rest];
            env_idx = car_idx / stride;
            slot = car_idx - env_idx * stride;
            hint = row_reach[car_idx];
            split = CarWorldRowsSplitPlanes(cfg, cars[car_idx], car_configs[car_idx], solver_scratch[env_idx], slot, hint);
        }
        CarWorldRowsPart mesh_part;
        if (active && !planes_warp) {
            bullet_port::btGjkEpaScratch* const gjk = gjk_pool + static_cast<long long>(blockIdx.x) * bs + lane;
            if (split) {
                BuildCarWorldRowsForCar<false, 1>(env_idx, slot, cfg, cars, car_configs, solver_scratch[env_idx],
                                                  row_pool.view(env_idx), gjk, hint, nullptr, &mesh_part);
            } else {
                BuildCarWorldRowsForCar<false>(env_idx, slot, cfg, cars, car_configs, solver_scratch[env_idx],
                                               row_pool.view(env_idx), gjk, hint);
            }
        } else if (active && split) {
            BuildCarWorldRowsForCar<false, 2>(env_idx, slot, cfg, cars, car_configs, solver_scratch[env_idx],
                                              row_pool.view(env_idx), nullptr, hint, plane_rows + car_idx,
                                              plane_parts + car_idx);
        }
        __syncthreads();
        if (active && !planes_warp && split) {
            FinishCarWorldRowsForCar(env_idx, slot, cfg, cars, solver_scratch[env_idx], row_pool.view(env_idx),
                                     mesh_part, plane_rows[car_idx], plane_parts[car_idx]);
        }
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
    int env_idx = tid / rays_per_env;
    if (env_idx >= cfg.env_count || solver_scratch == nullptr) {
        return;
    }
    const int slot = tid - env_idx * rays_per_env;
    MapStepEnv(cfg, env_idx);
    StageWheelRaysForEnv(slot, env_idx, cfg, cars, car_configs, wheel_runtime_states, balls,
                         env_states[env_idx], solver_scratch[env_idx],
                         env_idx * cfg.max_cars_per_env, env_car_count[env_idx]);
}

template <int kLanes>
struct WarpRayMeshLanes {
    static constexpr bool kSplitHeaders = kLanes >= 4;
    int first = 0;
    int step = kLanes;
    unsigned group_mask = 0u;
    __device__ bool Leader() const { return first == 0; }
    __device__ ArenaRayHit Combine(const ArenaRayHit& hit, int& order, float key) const {
        constexpr int kNoMesh = 0x7fffffff;
        const int lane_order = hit.hit ? order : kNoMesh;
        const unsigned same_order = __match_any_sync(group_mask, lane_order);
        const bool first_order_lane = __ffs(same_order) == (static_cast<int>(threadIdx.x) & 31) + 1;
        int meshes[kLanes];
        float fractions[kLanes];
#pragma unroll
        for (int l = 0; l < kLanes; ++l) {
            meshes[l] = __shfl_sync(group_mask, first_order_lane ? lane_order : kNoMesh, l, kLanes);
            fractions[l] = __shfl_sync(group_mask, key, l, kLanes);
        }
        int winner = -1;
        int winner_mesh = kNoMesh;
        float winner_fraction = 0.0f;
#pragma unroll
        for (int l = 0; l < kLanes; ++l) {
            if (meshes[l] >= 0 && meshes[l] < winner_mesh) {
                winner = l;
                winner_mesh = meshes[l];
                winner_fraction = fractions[l];
            }
        }
        if (winner < 0) {
            return hit;
        }
#pragma unroll
        for (int l = 0; l < kLanes; ++l) {
            if (meshes[l] >= 0 && meshes[l] != kNoMesh &&
                (fractions[l] < winner_fraction ||
                 (fractions[l] == winner_fraction && meshes[l] < winner_mesh))) {
                winner = l;
                winner_mesh = meshes[l];
                winner_fraction = fractions[l];
            }
        }
        order = winner_mesh;
        constexpr int kWords = static_cast<int>(sizeof(ArenaRayHit) / sizeof(uint32_t));
        static_assert(sizeof(ArenaRayHit) % sizeof(uint32_t) == 0, "ArenaRayHit is copied by 32-bit words");
        uint32_t words[kWords];
        memcpy(words, &hit, sizeof(ArenaRayHit));
        for (int w = 0; w < kWords; ++w) {
            words[w] = __shfl_sync(group_mask, words[w], winner, kLanes);
        }
        ArenaRayHit out;
        memcpy(&out, words, sizeof(ArenaRayHit));
        return out;
    }
};

template <int kLanes>
ROCKETGPU_GLOBAL void StagedPredictiveRaysKernel(
    const __grid_constant__ SimConfig cfg, const int32_t* __restrict__ env_car_count, const CarState* __restrict__ cars,
    const CarConfig* __restrict__ car_configs, const BallState* __restrict__ balls,
    WorldSolverScratch* __restrict__ solver_scratch, int body_shift, bool rays_inline) {
    __shared__ alignas(ArenaMeshBvhMeshInfo) unsigned char mesh_storage[kCarWorldMeshStageMeshes * sizeof(ArenaMeshBvhMeshInfo)];
    ArenaMeshBvhMeshInfo* const shared_meshes = reinterpret_cast<ArenaMeshBvhMeshInfo*>(mesh_storage);
    const bool use_shared_meshes = cfg.arena_mesh_bvh_meshes != nullptr && cfg.arena_mesh_bvh_mesh_count > 0 &&
                                   cfg.arena_mesh_bvh_mesh_count <= kCarWorldMeshStageMeshes;
    if (use_shared_meshes && static_cast<int>(threadIdx.x) < cfg.arena_mesh_bvh_mesh_count) {
        memcpy(&shared_meshes[threadIdx.x], &cfg.arena_mesh_bvh_meshes[threadIdx.x], sizeof(ArenaMeshBvhMeshInfo));
    }
    __syncthreads();
    const ArenaMeshBvhMeshInfo* const mesh_infos = use_shared_meshes ? shared_meshes : cfg.arena_mesh_bvh_meshes;
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long ray = tid / kLanes;
    const int lane = static_cast<int>(tid & (kLanes - 1));
    int env_idx = static_cast<int>(ray >> body_shift);
    const int body = static_cast<int>(ray & ((1 << body_shift) - 1));
    bool staged = false;
    bool hit = false;
    Vec3 n;
    float distance_bt = 0.0f;
    if (env_idx < cfg.env_count) {
        MapStepEnv(cfg, env_idx);
        const WorldSolverScratch& scratch = solver_scratch[env_idx];
        const int count = env_car_count[env_idx];
        staged = EnvVehiclesIndependent(cfg, scratch, count, rays_inline);
        if (staged && body < Min(count + 1, 9)) {
            const BallState& ball = balls[env_idx];
            const uint8_t wake = BallWakeStateAtTickStart(ball, cfg.mutator);
            const bool ball_sleeping = wake == 1 || BallDeactivatesAtTickStart(ball, wake, &scratch);
            const int base = env_idx * cfg.max_cars_per_env;
            const ArenaBroadphaseGrid grid = MakeArenaBroadphaseGrid(cfg);
            const auto cast = [&](const auto& mesh_lanes) {
                return PredictiveContactRayForBody(
                    body, ball, ball_sleeping, cars + base, scratch.pre_contact_cars, car_configs + base,
                    cfg.tick_dt, NativeSseMath(cfg.mutator),
                    cfg.arena_mesh_tris, cfg.arena_mesh_tri_count, cfg.arena_mesh_bvh_quantization,
                    cfg.arena_mesh_bvh_quantization_count, mesh_infos, cfg.arena_mesh_bvh_mesh_count,
                    cfg.arena_mesh_bvh_nodes, cfg.arena_mesh_bvh_node_count, cfg.arena_mesh_bvh_subtrees,
                    cfg.arena_mesh_bvh_subtree_count, &grid.leaf_grid, n, distance_bt, mesh_lanes,
                    cfg.mutator.cal_arena_local_tris);
            };
            if constexpr (kLanes == 1) {
                hit = cast(ArenaRayMeshOneLane{});
            } else {
                WarpRayMeshLanes<kLanes> mesh_lanes;
                mesh_lanes.first = lane;
                mesh_lanes.group_mask = ((1u << kLanes) - 1u) << ((threadIdx.x & 31u) & ~static_cast<unsigned>(kLanes - 1));
                hit = cast(mesh_lanes);
            }
        }
    }
    const unsigned hits = __ballot_sync(0xffffffffu, hit && lane == 0);
    if (!staged || lane != 0) {
        return;
    }
    WorldSolverScratch& scratch = solver_scratch[env_idx];
    if (hit) {
        scratch.predictive_normal[body] = n;
        scratch.predictive_distance_bt[body] = distance_bt;
    }
    if (body == 0) {
        const int first = static_cast<int>(threadIdx.x & 31u);
        uint32_t mask = 0u;
        for (int b = 0; b < (1 << body_shift); ++b) {
            mask |= ((hits >> (first + b * kLanes)) & 1u) << b;
        }
        scratch.predictive_hit_mask = static_cast<uint16_t>(mask);
        scratch.predictive_staged = 1;
    }
}

ROCKETGPU_GLOBAL void StagedW3ProxiesKernel(
    const __grid_constant__ SimConfig cfg, const int32_t* __restrict__ env_car_count, const CarState* __restrict__ cars,
    const CarConfig* __restrict__ car_configs, const BallState* __restrict__ balls,
    WorldSolverScratch* __restrict__ solver_scratch, int body_shift, bool rays_inline, bool dependent) {
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    int env_idx = static_cast<int>(tid >> body_shift);
    const int body = static_cast<int>(tid & ((1 << body_shift) - 1));
    if (env_idx >= cfg.env_count) {
        return;
    }
    MapStepEnv(cfg, env_idx);
    WorldSolverScratch& scratch = solver_scratch[env_idx];
    const int count = env_car_count[env_idx];
    const BallState& ball = balls[env_idx];
    if (body >= Min(count + 1, 9) || EnvVehiclesIndependent(cfg, scratch, count, rays_inline) == dependent) {
        return;
    }
    const ArenaBroadphaseGrid grid = MakeArenaBroadphaseGrid(cfg);
    if (dependent ? !(cfg.enable_car_car_collisions && cfg.mutator.game_mode != static_cast<uint8_t>(Mode::Heatseeker) &&
                      grid.valid() && cfg.arena_mesh_bvh_meshes != nullptr && cfg.arena_mesh_bvh_mesh_count > 0)
                  : !WorldStepPrefixRuns(ball, cfg.mutator, cfg.enable_car_car_collisions, grid,
                                         cfg.arena_mesh_bvh_meshes, cfg.arena_mesh_bvh_mesh_count)) {
        return;
    }
    const int base = env_idx * cfg.max_cars_per_env;
    ComputeBulletBodyProxy(body, ball, cars + base, scratch.pre_contact_cars, car_configs + base, cfg.mutator.ball_radius,
                           cfg.tick_dt, &scratch, NativeSseMath(cfg.mutator),
                           scratch.w3_proxies[body]);
    if (body == 0) {
        scratch.w3_proxies_done = 1;
    }
}

inline bool UseStagedPredictiveRays(const SimConfig& cfg) {
    return cfg.mutator.game_mode != static_cast<uint8_t>(Mode::Heatseeker) && cfg.mutator.arena_collision != 0 &&
           cfg.arena_mesh_bvh_meshes != nullptr && cfg.arena_mesh_bvh_mesh_count > 0 &&
           MakeArenaBroadphaseGrid(cfg).valid();
}

inline int StagedPredictiveRaysLanes(int body_shift) {
    return (2 << body_shift) <= 32 ? 2 : 1;
}
template <typename... Args>
void LaunchStagedPredictiveRays(int lanes, unsigned blocks, unsigned threads, cudaStream_t stream, const Args&... args) {
    if (lanes == 2) {
        StagedPredictiveRaysKernel<2><<<blocks, threads, 0, stream>>>(args...);
    } else {
        StagedPredictiveRaysKernel<1><<<blocks, threads, 0, stream>>>(args...);
    }
    CheckCuda(cudaGetLastError(), "StagedPredictiveRaysKernel launch");
}

template <int kLanes>
ROCKETGPU_GLOBAL void StageWheelRaysLanesKernel(
    const __grid_constant__ SimConfig cfg,
    const CarState* cars,
    const CarConfig* car_configs,
    const WheelRuntimeState* wheel_runtime_states,
    const BallState* balls,
    const EnvState* env_states,
    WorldSolverScratch* solver_scratch,
    const int32_t* env_car_count) {
    static_assert(kLanes >= 2 && kLanes <= 32 && (kLanes & (kLanes - 1)) == 0, "kLanes: a power of two up to 32");
    const int rays_per_env = 4 * cfg.max_cars_per_env;
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int ray = static_cast<int>(tid / kLanes);
    int env_idx = ray / rays_per_env;
    if (env_idx >= cfg.env_count || solver_scratch == nullptr) {
        return;
    }
    const int slot = ray - env_idx * rays_per_env;
    MapStepEnv(cfg, env_idx);
    WarpRayMeshLanes<kLanes> lanes;
    lanes.first = static_cast<int>(threadIdx.x) & (kLanes - 1);
    lanes.group_mask = ((1u << kLanes) - 1u) << ((threadIdx.x & 31u) & ~static_cast<unsigned>(kLanes - 1));
    if (kLanes == 32) {
        lanes.group_mask = 0xffffffffu;
    }
    StageWheelRaysForEnv(slot, env_idx, cfg, cars, car_configs, wheel_runtime_states, balls,
                         env_states[env_idx], solver_scratch[env_idx],
                         env_idx * cfg.max_cars_per_env, env_car_count[env_idx], lanes);
}

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
    __shared__ alignas(ArenaMeshBvhMeshInfo) unsigned char mesh_storage[kCarWorldMeshStageMeshes * sizeof(ArenaMeshBvhMeshInfo)];
    ArenaMeshBvhMeshInfo* const meshes = reinterpret_cast<ArenaMeshBvhMeshInfo*>(mesh_storage);
    if (cfg.arena_mesh_bvh_meshes != nullptr && static_cast<int>(threadIdx.x) < Min(cfg.arena_mesh_bvh_mesh_count, kCarWorldMeshStageMeshes)) {
        memcpy(&meshes[threadIdx.x], &cfg.arena_mesh_bvh_meshes[threadIdx.x], sizeof(ArenaMeshBvhMeshInfo));
    }
    __syncthreads();
    const int stride = cfg.max_cars_per_env;
    const long long car_slots = static_cast<long long>(cfg.env_count) * stride;
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t mesh_mask = 0;
    const int staged_car = [&]() -> int {
        if (tid >= car_slots) {
            return -1;
        }
        int env_idx = static_cast<int>(tid / stride);
        const int slot = static_cast<int>(tid - static_cast<long long>(env_idx) * stride);
        MapStepEnv(cfg, env_idx);
        const int car_idx = env_idx * stride + slot;
        stages[car_idx].valid = 0;
        if (slot >= env_car_count[env_idx]) {
            return -1;
        }
        const CarState& car = cars[car_idx];
        if (car.is_demoed || car.just_respawned || !cfg.mutator.arena_collision ||
            cfg.arena_mesh_tris == nullptr || cfg.arena_mesh_tri_count <= 0) {
            return -1;
        }
        const bool have_bvh_traversal =
            cfg.arena_mesh_bvh_quantization != nullptr && cfg.arena_mesh_bvh_meshes != nullptr &&
            cfg.arena_mesh_bvh_nodes != nullptr && cfg.arena_mesh_bvh_mesh_count > 0 && cfg.arena_mesh_bvh_node_count > 0 &&
            cfg.arena_mesh_bvh_mesh_count <= kCarWorldMeshStageMeshes;
        if (!have_bvh_traversal) {
            return -1;
        }
        const CarWorldMeshQueryFrame frame = ComputeCarWorldMeshQueryFrame(car, car_configs[car_idx]);
        const float fx = meshes[0].free_half_x;
        const float fy = meshes[0].free_half_y;
        if (frame.aabb_max_bt.x < fx && frame.aabb_min_bt.x > -fx &&
            frame.aabb_max_bt.y < fy && frame.aabb_min_bt.y > -fy) {
            return -1;
        }
        int staged_meshes = 0;
        for (int mesh = 0; mesh < cfg.arena_mesh_bvh_mesh_count; ++mesh) {
            const ArenaMeshBvhMeshInfo& mesh_info = meshes[mesh];
            if (CarWorldMeshInfoUsable(mesh_info, cfg.arena_mesh_bvh_quantization_count, cfg.arena_mesh_bvh_node_count) &&
                CarWorldMeshAabbMayOverlap(mesh_info, frame.aabb_min_bt, frame.aabb_max_bt)) {
                mesh_mask |= 1u << mesh;
                ++staged_meshes;
            }
        }
        return staged_meshes == 0 || staged_meshes > kCarWorldMeshStageSlots ? -1 : car_idx;
    }();
    if (car_snapshots != nullptr) {
        static_assert(sizeof(CarState) % sizeof(unsigned long long) == 0 &&
                          alignof(CarState) >= alignof(unsigned long long),
                      "the snapshot copies CarState in 8-byte words");
        constexpr int kCarWords = static_cast<int>(sizeof(CarState) / sizeof(unsigned long long));
        const unsigned warp_first = threadIdx.x & ~31u;
        const unsigned warp_lanes = blockDim.x - warp_first < 32u ? blockDim.x - warp_first : 32u;
        const unsigned warp_mask = warp_lanes == 32u ? 0xffffffffu : (1u << warp_lanes) - 1u;
        for (unsigned rest = __ballot_sync(warp_mask, staged_car >= 0); rest != 0; rest &= rest - 1u) {
            const int car_idx = __shfl_sync(warp_mask, staged_car, __ffs(static_cast<int>(rest)) - 1);
            const unsigned long long* const src = reinterpret_cast<const unsigned long long*>(&cars[car_idx]);
            unsigned long long* const dst = reinterpret_cast<unsigned long long*>(&car_snapshots[car_idx]);
            for (int w = static_cast<int>(threadIdx.x - warp_first); w < kCarWords; w += static_cast<int>(warp_lanes)) {
                dst[w] = src[w];
            }
        }
    }
    if (staged_car < 0) {
        return;
    }
    const int car_idx = staged_car;
    CarWorldMeshCandidateStage& stage = stages[car_idx];
    FillCarWorldMeshCandidateFingerprint(cars[car_idx], car_configs[car_idx], stage.fingerprint);
    stage.mesh_count = cfg.arena_mesh_bvh_mesh_count;
    stage.mesh_mask = mesh_mask;
    stage.valid = 1;
    for (uint32_t rest = mesh_mask; rest != 0; rest &= rest - 1u) {
        const int mesh = __ffs(static_cast<int>(rest)) - 1;
        const int item = atomicAdd(&mesh_work_count[mesh], 1);
        mesh_work[static_cast<long long>(mesh) * car_slots + item] = car_idx;
    }
}

__device__ __forceinline__ int StagePairsPerWarp(const int32_t* mesh_work_count, int meshes) {
    long long listed = 0;
    for (int mesh = 0; mesh < meshes; ++mesh) {
        listed += mesh_work_count[mesh];
    }
    if (listed <= 0) {
        return 0;
    }
    const long long lanes = blockDim.x < 32u ? blockDim.x : 32u;
    const long long warps = static_cast<long long>(gridDim.x) * blockDim.x / lanes;
    const long long per_warp = (listed + warps - 1) / warps;
    return static_cast<int>(per_warp < 1 ? 1 : per_warp > lanes ? lanes : per_warp);
}
__device__ __forceinline__ long long StagePairIndex(long long round, int per_warp) {
    const long long lanes = blockDim.x < 32u ? blockDim.x : 32u;
    const long long thread = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int lane = static_cast<int>(thread % lanes);
    if (lane >= per_warp) {
        return 1ll << 60;
    }
    const long long warps = static_cast<long long>(gridDim.x) * blockDim.x / lanes;
    return (thread / lanes + round * warps) * per_warp + lane;
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
    const int meshes = Min(cfg.arena_mesh_bvh_mesh_count, kCarWorldMeshStageMeshes);
    const int per_warp = StagePairsPerWarp(mesh_work_count, meshes);
    if (per_warp <= 0) {
        return;
    }
    const int stride = cfg.max_cars_per_env;
    const long long car_slots = static_cast<long long>(cfg.env_count) * stride;
    const long long first = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    bullet_port::btGjkEpaScratch* const gjk_scratch = gjk_pool + first;
    const ArenaMeshLeafGrid leaf_grid = MakeArenaBroadphaseGrid(cfg).leaf_grid;
    for (long long round = 0;; ++round) {
        int mesh = 0;
        long long item = StagePairIndex(round, per_warp);
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

ROCKETGPU_GLOBAL void StageCarWorldGatherKernel(
    const __grid_constant__ SimConfig cfg,
    const CarState* cars,
    const CarConfig* car_configs,
    CarWorldMeshCandidateStage* stages,
    const int32_t* mesh_work,
    const int32_t* mesh_work_count,
    CarWorldTrianglePairs pairs,
    bullet_port::btGjkEpaScratch* gjk_pool) {
    const int meshes = Min(cfg.arena_mesh_bvh_mesh_count, kCarWorldMeshStageMeshes);
    const int per_warp = StagePairsPerWarp(mesh_work_count, meshes);
    if (per_warp <= 0) {
        return;
    }
    const int stride = cfg.max_cars_per_env;
    const long long car_slots = static_cast<long long>(cfg.env_count) * stride;
    const long long first = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    bullet_port::btGjkEpaScratch* const gjk_scratch = gjk_pool + first;
    const ArenaMeshLeafGrid leaf_grid = MakeArenaBroadphaseGrid(cfg).leaf_grid;
    const unsigned warp_first = threadIdx.x & ~31u;
    const int warp_lanes = static_cast<int>(blockDim.x - warp_first < 32u ? blockDim.x - warp_first : 32u);
    const unsigned warp_mask = warp_lanes == 32 ? 0xffffffffu : (1u << warp_lanes) - 1u;
    const int lane = static_cast<int>(threadIdx.x - warp_first);
    constexpr int kGatherBufferWarps = 4;
    constexpr int kGatherBufferEntries = 256;
    static_assert(kCarWorldMeshGatherTriangles <= 128, "a reserved pair's rank fits in 7 bits");
    __shared__ uint32_t gather_buffer[kGatherBufferWarps][kGatherBufferEntries];
    const int warp_buffer_size =
        threadIdx.x < 32u * kGatherBufferWarps && cfg.arena_mesh_tri_count <= (1 << 20) ? kGatherBufferEntries : 0;
    const bool have_bvh_traversal =
        cfg.arena_mesh_tris != nullptr && cfg.arena_mesh_tri_count > 0 && cfg.arena_mesh_bvh_quantization != nullptr &&
        cfg.arena_mesh_bvh_meshes != nullptr && cfg.arena_mesh_bvh_nodes != nullptr && cfg.arena_mesh_bvh_mesh_count > 0 &&
        cfg.arena_mesh_bvh_node_count > 0 && cfg.arena_mesh_bvh_mesh_count <= kCarWorldMeshStageMeshes &&
        leaf_grid.infos != nullptr;
    for (long long round = 0;; ++round) {
        int mesh = 0;
        long long item = StagePairIndex(round, per_warp);
        for (; mesh < meshes; ++mesh) {
            const long long listed = mesh_work_count[mesh];
            if (item < listed) {
                break;
            }
            item -= listed;
        }
        const bool have = mesh < meshes;
        if (__ballot_sync(warp_mask, have) == 0u) {
            break;
        }
        int car_idx = 0;
        int env_idx = 0;
        int slot = 0;
        bool on_grid = false;
        CarWorldMeshQueryFrame frame;
        float threshold_bt = 0.0f;
        uint16_t qmin[3] = {0, 0, 0};
        uint16_t qmax[3] = {0, 0, 0};
        int entry_first = 0;
        int entries = 0;
        int tri_start = 0;
        int tri_count = 0;
        if (have) {
            car_idx = mesh_work[static_cast<long long>(mesh) * car_slots + item];
            env_idx = car_idx / stride;
            slot = car_idx - env_idx * stride;
            const ArenaMeshBvhMeshInfo mesh_info = have_bvh_traversal ? cfg.arena_mesh_bvh_meshes[mesh] : ArenaMeshBvhMeshInfo{};
            if (have_bvh_traversal && leaf_grid.infos[mesh].cell_offset >= 0 &&
                CarWorldMeshInfoUsable(mesh_info, cfg.arena_mesh_bvh_quantization_count, cfg.arena_mesh_bvh_node_count)) {
                const CarConfig& car_cfg = car_configs[car_idx];
                frame = ComputeCarWorldMeshQueryFrame(cars[car_idx], car_cfg);
                if (CarWorldMeshAabbMayOverlap(mesh_info, frame.aabb_min_bt, frame.aabb_max_bt)) {
                    const ArenaMeshBvhQuantization& quantization = cfg.arena_mesh_bvh_quantization[mesh_info.mesh_index];
                    QuantizeArenaMeshBvhQueryBox(quantization, frame.box_center_bt, frame.extent_bt, mesh_info.cal_origin_y,
                                                 qmin, qmax);
                    const ArenaMeshLeafGridInfo& info = leaf_grid.infos[mesh];
                    on_grid = true;
                    int cell[3];
                    for (int a = 0; a < 3; ++a) {
                        if (qmax[a] - qmin[a] > info.cell_width[a]) {
                            on_grid = false;
                            break;
                        }
                        cell[a] = qmin[a] / info.cell_width[a];
                        cell[a] = cell[a] < info.cells[a] ? cell[a] : info.cells[a] - 1;
                    }
                    if (on_grid) {
                        const int cell_idx = info.cell_offset + (cell[2] * info.cells[1] + cell[1]) * info.cells[0] + cell[0];
                        entry_first = leaf_grid.cell_start[cell_idx];
                        entries = leaf_grid.cell_start[cell_idx + 1] - entry_first;
                        threshold_bt = CarContactBreakingThresholdBt(car_cfg);
                        tri_start = mesh_info.triangle_start;
                        tri_count = mesh_info.triangle_count;
                    }
                }
            }
            if (!on_grid) {
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
                    &stages[car_idx], nullptr, mesh, &leaf_grid, &pairs, kTrianglePairsGather);
            }
        }
        const uint32_t q01 = static_cast<uint32_t>(qmin[0]) | (static_cast<uint32_t>(qmin[1]) << 16);
        const uint32_t q23 = static_cast<uint32_t>(qmin[2]) | (static_cast<uint32_t>(qmax[0]) << 16);
        const uint32_t q45 = static_cast<uint32_t>(qmax[1]) | (static_cast<uint32_t>(qmax[2]) << 16);
        const int car_global = env_idx * pairs.car_stride + slot;
        int broadphase_hits = 0;
        int gathered = 0;
        int pair_start = -1;
        uint32_t* const warp_buffer = gather_buffer[(threadIdx.x >> 5) % kGatherBufferWarps];
        int buffered = 0;
        const auto scan = [&](int scan_entries, bool write) {
            int prefix = scan_entries;
            for (int d = 1; d < 32; d <<= 1) {
                const int v = __shfl_up_sync(warp_mask, prefix, d);
                if (lane >= d) {
                    prefix += v;
                }
            }
            const int total = __shfl_sync(warp_mask, prefix, warp_lanes - 1);
            prefix -= scan_entries;
            constexpr int kRoundEntries = 4;
            for (int base = 0; base < total; base += kRoundEntries * warp_lanes) {
                int owner[kRoundEntries];
                uint4 raw[kRoundEntries];
#pragma unroll
                for (int k = 0; k < kRoundEntries; ++k) {
                    const int g = base + k * warp_lanes + lane;
                    int o = 0;
                    for (int step = 16; step > 0; step >>= 1) {
                        const int c = o + step;
                        const int p = __shfl_sync(warp_mask, prefix, c < warp_lanes ? c : warp_lanes - 1);
                        if (c < warp_lanes && p <= g) {
                            o = c;
                        }
                    }
                    owner[k] = o;
                    const int e = __shfl_sync(warp_mask, entry_first, o) + (g - __shfl_sync(warp_mask, prefix, o));
                    raw[k] = g < total ? __ldg(reinterpret_cast<const uint4*>(&leaf_grid.entries[e])) : make_uint4(0u, 0u, 0u, 0u);
                }
                int tri[kRoundEntries];
                ArenaMeshTriangle tris[kRoundEntries];
#pragma unroll
                for (int k = 0; k < kRoundEntries; ++k) {
                    const int g = base + k * warp_lanes + lane;
                    const int o = owner[k];
                    ArenaMeshLeafEntry entry;
                    memcpy(&entry, &raw[k], sizeof(entry));
                    const uint32_t o01 = __shfl_sync(warp_mask, q01, o);
                    const uint32_t o23 = __shfl_sync(warp_mask, q23, o);
                    const uint32_t o45 = __shfl_sync(warp_mask, q45, o);
                    const uint16_t oqmin[3] = {static_cast<uint16_t>(o01), static_cast<uint16_t>(o01 >> 16), static_cast<uint16_t>(o23)};
                    const uint16_t oqmax[3] = {static_cast<uint16_t>(o23 >> 16), static_cast<uint16_t>(o45), static_cast<uint16_t>(o45 >> 16)};
                    const int o_tri_start = __shfl_sync(warp_mask, tri_start, o);
                    const int o_tri_count = __shfl_sync(warp_mask, tri_count, o);
                    const int local = entry.triangle_index;
                    const bool leaf = g < total &&
                        TestQuantizedAabbAgainstQuantizedAabb(oqmin, oqmax, entry.quantized_aabb_min, entry.quantized_aabb_max) &&
                        local >= 0 && local < o_tri_count;
                    tri[k] = leaf ? o_tri_start + local : -1;
                    if (leaf) {
                        const ArenaMeshTriangle& src = cfg.arena_mesh_tris[tri[k]];
                        tris[k].a = src.a;
                        tris[k].b = src.b;
                        tris[k].c = src.c;
                        tris[k].normal = src.normal;
                    }
                }
#pragma unroll
                for (int k = 0; k < kRoundEntries; ++k) {
                    const int round_first = base + k * warp_lanes;
                    if (round_first >= total) {
                        break;
                    }
                    const int o = owner[k];
                    int test = 0;
                    if (__any_sync(warp_mask, tri[k] >= 0)) {
                        CarWorldMeshQueryFrame of;
                        const auto shfl3 = [&](const Vec3& v) {
                            return Vec3(__shfl_sync(warp_mask, v.x, o), __shfl_sync(warp_mask, v.y, o), __shfl_sync(warp_mask, v.z, o));
                        };
                        of.axis_x = shfl3(frame.axis_x);
                        of.axis_y = shfl3(frame.axis_y);
                        of.axis_z = shfl3(frame.axis_z);
                        of.box_center_bt = shfl3(frame.box_center_bt);
                        of.box_half_extent_with_margin_bt = shfl3(frame.box_half_extent_with_margin_bt);
                        of.extent_bt = shfl3(frame.extent_bt);
                        of.aabb_min_bt = shfl3(frame.aabb_min_bt);
                        of.aabb_max_bt = shfl3(frame.aabb_max_bt);
                        const float o_threshold_bt = __shfl_sync(warp_mask, threshold_bt, o);
                        if (tri[k] >= 0) {
                            test = CarWorldTriangleQueryTest(
                                tris[k], of, o_threshold_bt,
                                cfg.mutator.cal_arena_local_tris != nullptr ? &cfg.mutator.cal_arena_local_tris[tri[k]] : nullptr);
                        }
                    }
                    const unsigned hit_bits = __ballot_sync(warp_mask, test >= 1);
                    const unsigned gather_bits = __ballot_sync(warp_mask, test == 2);
                    const int o_gathered_before = __shfl_sync(warp_mask, gathered, o);
                    const int o_lo = Max(__shfl_sync(warp_mask, prefix, o) - round_first, 0);
                    const unsigned below = (1u << lane) - 1u;
                    const int rank = o_gathered_before + __popc(gather_bits & below & ~((1u << o_lo) - 1u));
                    if (write) {
                        const int o_pair_start = __shfl_sync(warp_mask, pair_start, o);
                        const int o_car = __shfl_sync(warp_mask, car_global, o);
                        if (test == 2) {
                            pairs.car_idx[o_pair_start + rank] = o_car;
                            pairs.tri_idx[o_pair_start + rank] = tri[k];
                        }
                    } else {
                        const int at = buffered + __popc(gather_bits & below);
                        if (test == 2 && at < warp_buffer_size) {
                            warp_buffer[at] = (static_cast<uint32_t>(tri[k]) << 12) | (static_cast<uint32_t>(o) << 7) |
                                static_cast<uint32_t>(Min(rank, 127));
                        }
                        buffered += __popc(gather_bits);
                    }
                    const int lo = Max(prefix - round_first, 0);
                    const int hi = Min(prefix + scan_entries - round_first, warp_lanes);
                    const unsigned range = hi > lo ? ((hi - lo == 32 ? 0xffffffffu : ((1u << (hi - lo)) - 1u)) << lo) : 0u;
                    broadphase_hits += __popc(hit_bits & range);
                    gathered += __popc(gather_bits & range);
                }
            }
        };
        scan(on_grid ? entries : 0, false);
        const int need = on_grid && gathered > 0 && gathered <= kCarWorldMeshGatherTriangles ? gathered : 0;
        int need_before = need;
        for (int d = 1; d < 32; d <<= 1) {
            const int v = __shfl_up_sync(warp_mask, need_before, d);
            if (lane >= d) {
                need_before += v;
            }
        }
        const int need_total = __shfl_sync(warp_mask, need_before, warp_lanes - 1);
        need_before -= need;
        int warp_start = 0;
        if (lane == 0 && need_total > 0) {
            warp_start = atomicAdd(pairs.count, 0);
            while (true) {
                if (warp_start > pairs.capacity - need_total) {
                    warp_start = -1;
                    break;
                }
                const int observed = atomicCAS(pairs.count, warp_start, warp_start + need_total);
                if (observed == warp_start) {
                    break;
                }
                warp_start = observed;
            }
        }
        warp_start = __shfl_sync(warp_mask, warp_start, 0);
        if (on_grid) {
            CarWorldMeshCandidateStage& stage = stages[car_idx];
            CarWorldMeshCandidateStageMesh& staged = stage.meshes[CarWorldMeshStageSlot(stage.mesh_mask, mesh)];
            staged.broadphase_hits = broadphase_hits;
            staged.pair_count = -1;
            if (gathered <= kCarWorldMeshGatherTriangles) {
                const int start = gathered == 0 ? 0 : (warp_start >= 0 ? warp_start + need_before : -1);
                if (start >= 0) {
                    staged.pair_start = start;
                    staged.pair_count = gathered;
                    pair_start = start;
                }
            }
        }
        if (buffered <= warp_buffer_size) {
            __syncwarp(warp_mask);
            for (int base = 0; base < buffered; base += warp_lanes) {
                const int i = base + lane;
                const uint32_t v = i < buffered ? warp_buffer[i] : 0u;
                const int o = static_cast<int>((v >> 7) & 31u);
                const int o_pair_start = __shfl_sync(warp_mask, pair_start, o);
                const int o_car = __shfl_sync(warp_mask, car_global, o);
                if (i < buffered && o_pair_start >= 0) {
                    pairs.car_idx[o_pair_start + static_cast<int>(v & 127u)] = o_car;
                    pairs.tri_idx[o_pair_start + static_cast<int>(v & 127u)] = static_cast<int>(v >> 12);
                }
            }
            __syncwarp(warp_mask);
        } else {
            const bool write = on_grid && pair_start >= 0 && gathered > 0;
            gathered = 0;
            scan(write ? entries : 0, true);
        }
    }
}

ROCKETGPU_GLOBAL void CarWorldTriangleHitsKernel(
    const __grid_constant__ SimConfig cfg,
    const CarState* cars,
    const CarConfig* car_configs,
    CarWorldTrianglePairs pairs,
    bullet_port::btGjkEpaScratch* gjk_pool) {
    const int count = Min(*pairs.count, pairs.capacity);
    if (count <= 0) {
        return;
    }
    bullet_port::btGjkEpaScratch* const gjk_scratch = gjk_pool + (blockIdx.x * blockDim.x + threadIdx.x);
    for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); i < count; i += static_cast<int>(gridDim.x * blockDim.x)) {
        const int car_idx = pairs.car_idx[i];
        const CarConfig& car_cfg = car_configs[car_idx];
        pairs.hits[i] = ComputeCarWorldTriangleHit(
            cars[car_idx], car_cfg, cfg.mutator, cfg.arena_mesh_tris, pairs.tri_idx[i],
            gjk_scratch);
    }
}

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
        reinterpret_cast<const void*>(&StageWheelRaysLanesKernel<2>),
        reinterpret_cast<const void*>(&StageWheelRaysLanesKernel<4>),
        reinterpret_cast<const void*>(&StageWheelRaysLanesKernel<8>),
        reinterpret_cast<const void*>(&PrefilterCarWorldMeshCandidatesKernel),
        reinterpret_cast<const void*>(&StageCarWorldMeshCandidatesKernel),
        reinterpret_cast<const void*>(&StageCarWorldGatherKernel),
        reinterpret_cast<const void*>(&CarWorldTriangleHitsKernel),
        reinterpret_cast<const void*>(&StagedCarPreKernel),
        reinterpret_cast<const void*>(&StagedVehicleIndependenceKernel),
        reinterpret_cast<const void*>(&StagedWorldKernel<kWorldStepContacts>),
        reinterpret_cast<const void*>(&StagedWorldKernel<kWorldStepContactsCarsApart>),
        reinterpret_cast<const void*>(&StagedWorldKernel<kWorldStepContacts, kWorld1Lanes, false, 1>),
        reinterpret_cast<const void*>(&StagedWorldKernel<kWorldStepContactsCarsApart, 1, false, 2>),
        reinterpret_cast<const void*>(&StagedWorldKernel<kWorldStepContactsCarsApartPrefix>),
        reinterpret_cast<const void*>(&StagedWorldKernel<kWorldStepContactsCarsApartInfo>),
        reinterpret_cast<const void*>(&StagedWorldKernel<kWorldStepRowsCarsApart>),
        reinterpret_cast<const void*>(&StagedWorldKernel<kWorldStepRowsCarsApart, 1, true>),
        reinterpret_cast<const void*>(&StagedCarPostKernel),
        reinterpret_cast<const void*>(&StagedEnvTailKernel),
        reinterpret_cast<const void*>(&StagedPostFusedKernel),
        reinterpret_cast<const void*>(&StagedPostPreFusedKernel),
        reinterpret_cast<const void*>(&StagedPadPreKernel),
        reinterpret_cast<const void*>(&StagedCarWorldRowsKernel),
        reinterpret_cast<const void*>(&StagedCarWorldRowsSplitKernel),
        reinterpret_cast<const void*>(&StagedCarWorldRowsListKernel<true>),
        reinterpret_cast<const void*>(&StagedCarWorldRowsListKernel<false>),
        reinterpret_cast<const void*>(&StagedCarWorldRowsSplitPlanesKernel),
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

inline int GetCudaStepBlockSize(int env_count) {
    return env_count >= 64 * CudaSmCount() ? 64 : 32;
}

constexpr int kCudaRayKernelBlockSize = 128;

inline int SpreadBlockSize(long long threads) {
    const long long per_sm = (threads + CudaSmCount() - 1) / CudaSmCount();
    int bs = 8;
    while (bs < 64 && bs < per_sm) {
        bs *= 2;
    }
    return bs;
}
inline int World4BlockSize(long long envs, int spread_block_size) {
    const long long per_block = envs / (CudaSmCount() * 6ll);
    int bs = 1;
    while (bs * 2 <= per_block) {
        bs *= 2;
    }
    return bs <= 8 ? bs : spread_block_size;
}
inline unsigned PoolBlocks(size_t threads, size_t pool, int block_size) {
    const size_t bs = static_cast<size_t>(block_size);
    size_t blocks = (threads + bs - 1) / bs;
    if (blocks * bs > pool) {
        blocks = std::max<size_t>(1, pool / bs);
    }
    return static_cast<unsigned>(blocks);
}

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
inline size_t CarWorldStagingThreads() {
    return static_cast<size_t>(CudaSmCount()) * 2 * kCudaRayKernelBlockSize;
}
static bool StreamIsCapturing(cudaStream_t stream);
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
    int32_t* mesh_work = nullptr;
    int32_t* mesh_work_count = nullptr;
    CarWorldTrianglePairs pairs{};
    size_t capacity = 0;
    CarWorldMeshCandidateStage* published = nullptr;
    CarState* car_snapshots = nullptr;
    int32_t* row_cars = nullptr;
    uint32_t* row_reach = nullptr;
    CarWorldLocalRows* plane_rows = nullptr;
    CarWorldRowsPart* plane_parts = nullptr;
    size_t plane_capacity = 0;
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
            cudaFree(b.row_reach);
            cudaFree(b.mesh_work);
            cudaFree(b.pairs.car_idx);
            cudaFree(b.pairs.tri_idx);
            cudaFree(b.pairs.hits);
            b.stages = nullptr;
            b.mesh_work = nullptr;
            b.pairs = CarWorldTrianglePairs{};
        }
        if (b.mesh_work_count == nullptr) {
            CheckCuda(cudaMalloc(&b.mesh_work_count, (kCarWorldMeshStageMeshes + 4 + kRowsFullKeys) * sizeof(int32_t)), "cudaMalloc(car-world mesh work count)");
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
        CheckCuda(cudaMalloc(&b.row_cars, (1 + kRowsFullKeys) * count * sizeof(int32_t)), "cudaMalloc(car-world rows car lists)");
        CheckCuda(cudaMalloc(&b.row_reach, count * sizeof(uint32_t)), "cudaMalloc(car-world rows reach masks)");
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
inline bool EnsureCarWorldPlaneRows(cudaStream_t stream) {
    CarWorldStageBuffers& b = GetCarWorldStageBuffers();
    if (b.plane_capacity >= b.capacity) {
        return b.plane_rows != nullptr;
    }
    if (StreamIsCapturing(stream)) {
        return false;
    }
    CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(car-world plane rows grow)");
    if (b.plane_rows != nullptr) {
        cudaFree(b.plane_rows);
        cudaFree(b.plane_parts);
        b.plane_rows = nullptr;
        b.plane_parts = nullptr;
        b.plane_capacity = 0;
    }
    CheckCuda(cudaMalloc(&b.plane_rows, b.capacity * sizeof(CarWorldLocalRows)), "cudaMalloc(car-world plane rows)");
    CheckCuda(cudaMalloc(&b.plane_parts, b.capacity * sizeof(CarWorldRowsPart)), "cudaMalloc(car-world plane parts)");
    b.plane_capacity = b.capacity;
    ++StepResourceEpoch();
    return true;
}
inline BallWorldStage* EnsureBallWorldStageArmed(const SimConfig& cfg, cudaStream_t stream) {
    struct Buffer {
        BallWorldStage* data = nullptr;
        size_t capacity = 0;
        BallWorldStage* published = nullptr;
    };
    static Buffer b;
    const size_t count = static_cast<size_t>(cfg.env_count);
    if (count == 0) {
        return nullptr;
    }
    const bool capturing = StreamIsCapturing(stream);
    if (b.capacity < count) {
        if (capturing) {
            return nullptr;
        }
        CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(ball-world stage grow)");
        if (b.data != nullptr) {
            cudaFree(b.data);
            b.data = nullptr;
        }
        CheckCuda(cudaMalloc(&b.data, count * sizeof(BallWorldStage)), "cudaMalloc(ball-world stage)");
        CheckCuda(cudaMemset(b.data, 0, count * sizeof(BallWorldStage)), "cudaMemset(ball-world stage)");
        b.capacity = count;
        ++StepResourceEpoch();
    }
    if (b.published != b.data) {
        if (capturing) {
            return nullptr;
        }
        CheckCuda(cudaMemcpyToSymbolAsync(g_rocketgpu_ball_world_stage, &b.data, sizeof(b.data), 0,
                                          cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyToSymbolAsync(g_rocketgpu_ball_world_stage)");
        CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(ball-world stage publish)");
        b.published = b.data;
        ++StepResourceEpoch();
    }
    return b.data;
}
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
struct CarWorldStageSlice {
    int32_t* mesh_work = nullptr;
    int32_t* mesh_work_count = nullptr;
    int32_t* row_cars = nullptr;
    CarWorldTrianglePairs pairs{};
    bullet_port::btGjkEpaScratch* gjk_pool = nullptr;
    size_t threads = 0;
};
inline void LaunchStageCarWorldMeshCandidates(
    const SimConfig& cfg,
    const CarState* cars,
    const CarConfig* car_configs,
    const int32_t* env_car_count,
    cudaStream_t stream,
    cudaEvent_t prefilter_done = nullptr,
    const CarWorldStageSlice* slice = nullptr,
    bool counts_zeroed = false) {
    const auto mark_prefilter_done = [&] {
        if (prefilter_done != nullptr) {
            CheckCuda(cudaEventRecord(prefilter_done, stream), "cudaEventRecord(car-world prefilter)");
        }
    };
    bullet_port::btGjkEpaScratch* const gjk_pool =
        slice != nullptr ? slice->gjk_pool : EnsureGjkScratchPool(CarWorldStagingThreads(), stream);
    const size_t staging_threads = slice != nullptr ? slice->threads : CarWorldStagingThreads();
    if (gjk_pool == nullptr || (slice == nullptr && !EnsureCarWorldStageArmed(cfg, stream))) {
        if (!StreamIsCapturing(stream)) {
            PublishCarWorldStage(nullptr, 0, stream);
        }
        mark_prefilter_done();
        return;
    }
    CarWorldStageBuffers& b = GetCarWorldStageBuffers();
    int32_t* const mesh_work = slice != nullptr ? slice->mesh_work : b.mesh_work;
    int32_t* const mesh_work_count = slice != nullptr ? slice->mesh_work_count : b.mesh_work_count;
    const size_t car_slots = static_cast<size_t>(cfg.env_count) * static_cast<size_t>(cfg.max_cars_per_env);
    const int block_size = kCudaRayKernelBlockSize;
    if (!counts_zeroed) {
        ZeroInt32Kernel<<<1, 32, 0, stream>>>(mesh_work_count, kCarWorldMeshStageMeshes + 4 + kRowsFullKeys);
        CheckCuda(cudaGetLastError(), "ZeroInt32Kernel launch (car-world mesh work count)");
    }
    PrefilterCarWorldMeshCandidatesKernel<<<static_cast<unsigned>((car_slots + block_size - 1) / block_size),
                                            static_cast<unsigned>(block_size), 0, stream>>>(
        cfg, cars, car_configs, env_car_count, b.stages, mesh_work, mesh_work_count,
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
    CarWorldTrianglePairs pairs = slice != nullptr ? slice->pairs : b.pairs;
    pairs.car_stride = cfg.max_cars_per_env;
    const size_t stage_threads = std::min(pair_count, staging_threads);
    const unsigned stage_blocks = PoolBlocks(stage_threads, staging_threads, block_size);
    StageCarWorldGatherKernel<<<stage_blocks, static_cast<unsigned>(block_size), 0, stream>>>(
        cfg, cars, car_configs, b.stages, mesh_work, mesh_work_count, pairs, gjk_pool);
    CheckCuda(cudaGetLastError(), "StageCarWorldGatherKernel launch");
    const size_t hit_threads = std::min<size_t>(static_cast<size_t>(pairs.capacity), staging_threads);
    CarWorldTriangleHitsKernel<<<PoolBlocks(hit_threads, staging_threads, block_size),
                                 static_cast<unsigned>(block_size), 0, stream>>>(cfg, cars, car_configs, pairs, gjk_pool);
    CheckCuda(cudaGetLastError(), "CarWorldTriangleHitsKernel launch");
    StageCarWorldMeshCandidatesKernel<<<stage_blocks, static_cast<unsigned>(block_size), 0, stream>>>(
        cfg, cars, car_configs, b.stages, mesh_work, mesh_work_count, pairs, kTrianglePairsMerge, gjk_pool);
    CheckCuda(cudaGetLastError(), "StageCarWorldMeshCandidatesKernel merge launch");
}
inline int WheelRayLanes(long long rays) {
    if (rays <= 8192) return 8;
    if (rays <= 24576) return 4;
    if (rays <= 196608) return 2;
    return 1;
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
    cudaStream_t stream) {
    const long long rays = static_cast<long long>(cfg.env_count) * 4ll * static_cast<long long>(cfg.max_cars_per_env);
    if (rays <= 0) {
        return;
    }
    const int block_size = kCudaRayKernelBlockSize;
    const int lanes = WheelRayLanes(rays);
    const long long blocks = (rays * lanes + block_size - 1) / block_size;
    switch (lanes) {
        case 2:
            StageWheelRaysLanesKernel<2><<<static_cast<unsigned>(blocks), static_cast<unsigned>(block_size), 0, stream>>>(
                cfg, cars, car_configs, wheel_runtime_states, balls, env_states, solver_scratch, env_car_count);
            break;
        case 4:
            StageWheelRaysLanesKernel<4><<<static_cast<unsigned>(blocks), static_cast<unsigned>(block_size), 0, stream>>>(
                cfg, cars, car_configs, wheel_runtime_states, balls, env_states, solver_scratch, env_car_count);
            break;
        case 8:
            StageWheelRaysLanesKernel<8><<<static_cast<unsigned>(blocks), static_cast<unsigned>(block_size), 0, stream>>>(
                cfg, cars, car_configs, wheel_runtime_states, balls, env_states, solver_scratch, env_car_count);
            break;
        default:
            StageWheelRaysKernel<<<static_cast<unsigned>(blocks), static_cast<unsigned>(block_size), 0, stream>>>(
                cfg, cars, car_configs, wheel_runtime_states, balls, env_states, solver_scratch, env_car_count);
            break;
    }
    CheckCuda(cudaGetLastError(), "StageWheelRaysKernel launch");
}

constexpr int kStagedBlockSize = 64;
constexpr int kWorld1LanesMaxEnvs = 16384;
inline bool UseWorld1Lanes(const SimConfig& cfg) {
    return cfg.env_count <= kWorld1LanesMaxEnvs;
}
constexpr int kRaysInlineMaxEnvs = 4096;
inline bool UseRaysInline(const SimConfig& cfg) {
    return cfg.env_count <= kRaysInlineMaxEnvs;
}
constexpr int kRegisterIterationsMaxEnvs = 4096;
inline bool UseRegisterIterations(const SimConfig& cfg) {
    return cfg.env_count <= kRegisterIterationsMaxEnvs;
}
constexpr int kPostFusedMaxEnvs = 2048;
inline bool UsePostFused(const SimConfig& cfg) {
    return cfg.env_count <= kPostFusedMaxEnvs;
}
struct StagedStreams {
    cudaStream_t stage = nullptr;
    cudaStream_t dependent = nullptr;
    cudaStream_t ball = nullptr;
    cudaStream_t world3 = nullptr;
    cudaStream_t world1_contacts = nullptr;
    cudaStream_t world3_prefix = nullptr;
    cudaStream_t world3_info = nullptr;
    cudaEvent_t pre_done = nullptr;
    cudaEvent_t prefilter_done = nullptr;
    cudaEvent_t stage_done = nullptr;
    cudaEvent_t independence_done = nullptr;
    cudaEvent_t dependent_done = nullptr;
    cudaEvent_t rows_split_done = nullptr;
    cudaEvent_t rows_light_done = nullptr;
    cudaEvent_t ball_done = nullptr;
    cudaEvent_t world3_ready = nullptr;
    cudaEvent_t world3_done = nullptr;
    cudaEvent_t world1_contacts_done = nullptr;
    cudaEvent_t world3_prefix_done = nullptr;
    cudaEvent_t world3_info_done = nullptr;
};
inline StagedStreams MakeStagedStreams() {
    StagedStreams s;
    CheckCuda(cudaStreamCreateWithFlags(&s.stage, cudaStreamNonBlocking), "cudaStreamCreate(staged stage)");
    CheckCuda(cudaStreamCreateWithFlags(&s.dependent, cudaStreamNonBlocking), "cudaStreamCreate(staged dependent)");
    CheckCuda(cudaStreamCreateWithFlags(&s.ball, cudaStreamNonBlocking), "cudaStreamCreate(staged ball)");
    CheckCuda(cudaStreamCreateWithFlags(&s.world3, cudaStreamNonBlocking), "cudaStreamCreate(staged world3)");
    CheckCuda(cudaStreamCreateWithFlags(&s.world1_contacts, cudaStreamNonBlocking),
              "cudaStreamCreate(staged world1 contacts)");
    CheckCuda(cudaStreamCreateWithFlags(&s.world3_prefix, cudaStreamNonBlocking), "cudaStreamCreate(staged world3 prefix)");
    CheckCuda(cudaStreamCreateWithFlags(&s.world3_info, cudaStreamNonBlocking), "cudaStreamCreate(staged world3 info)");
    for (cudaEvent_t* e : {&s.pre_done, &s.prefilter_done, &s.stage_done, &s.independence_done, &s.dependent_done,
                          &s.rows_split_done, &s.rows_light_done, &s.ball_done, &s.world3_ready, &s.world3_done,
                          &s.world1_contacts_done, &s.world3_prefix_done, &s.world3_info_done}) {
        CheckCuda(cudaEventCreateWithFlags(e, cudaEventDisableTiming), "cudaEventCreate(staged)");
    }
    return s;
}
inline StagedStreams& GetStagedStreams() {
    static StagedStreams streams = MakeStagedStreams();
    return streams;
}

constexpr int kLiveEnvsMax = 32768;
constexpr long long kInterleaveFullCarsMax = 65536;
struct StagedEnvGroups {
    int groups = 0;
    int capacity = 0;
    int32_t* order = nullptr;
    int32_t* split = nullptr;
    int32_t* counts = nullptr;
    cudaEvent_t fork = nullptr;
    std::vector<cudaStream_t> main;
    std::vector<cudaEvent_t> join;
    std::vector<StagedStreams> side;
};
inline StagedEnvGroups& GetStagedEnvGroups() {
    static StagedEnvGroups groups;
    return groups;
}
ROCKETGPU_GLOBAL void InterleaveGroupsKernel(const int32_t* src, int32_t* dst, int count, int groups) {
    const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= count) {
        return;
    }
    const int g = j % groups;
    int first = 0;
    for (int h = 0; h < g; ++h) {
        first += (count - h + groups - 1) / groups;
    }
    dst[first + j / groups] = src != nullptr ? src[j] : j;
}
ROCKETGPU_GLOBAL void IotaInt32Kernel(int32_t* data, int count) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) {
        data[i] = i;
    }
}
inline int StagedEnvGroupCount(const SimConfig& cfg, cudaStream_t stream) {
    const long long slots = static_cast<long long>(cfg.env_count) * cfg.max_cars_per_env;
    const long long cars = cfg.env_car_total > 0 ? cfg.env_car_total : slots;
    const long long per = cars < slots ? cfg.env_count / 1024 : cars / 8192;
    int cap = 8;
    while (cap * 8192ll < cfg.env_count) {
        cap *= 2;
    }
    int wanted = 1;
    while (wanted < cap && wanted * 2 <= per) {
        wanted *= 2;
    }
    const int groups = std::min(wanted, cfg.env_count);
    if (groups <= 1 || !EnsureCarWorldStageArmed(cfg, stream)) {
        return 1;
    }
    StagedEnvGroups& eg = GetStagedEnvGroups();
    if (eg.groups >= groups && eg.capacity >= cfg.env_count) {
        return groups;
    }
    if (StreamIsCapturing(stream)) {
        return 1;
    }
    CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(env groups grow)");
    if (eg.capacity < cfg.env_count) {
        if (eg.order != nullptr) {
            cudaFree(eg.order);
        }
        CheckCuda(cudaMalloc(&eg.order, static_cast<size_t>(cfg.env_count) * sizeof(int32_t)), "cudaMalloc(env groups order)");
        if (eg.split != nullptr) {
            cudaFree(eg.split);
        }
        CheckCuda(cudaMalloc(&eg.split, static_cast<size_t>(cfg.env_count) * sizeof(int32_t)), "cudaMalloc(env groups split)");
        IotaInt32Kernel<<<(cfg.env_count + 255) / 256, 256, 0, stream>>>(eg.order, cfg.env_count);
        CheckCuda(cudaGetLastError(), "IotaInt32Kernel launch");
        eg.capacity = cfg.env_count;
    }
    if (eg.groups < groups) {
        if (eg.counts != nullptr) {
            cudaFree(eg.counts);
        }
        CheckCuda(cudaMalloc(&eg.counts, static_cast<size_t>(groups) * (kCarWorldMeshStageMeshes + 4 + kRowsFullKeys) * sizeof(int32_t)),
                  "cudaMalloc(env groups counters)");
        if (eg.fork == nullptr) {
            CheckCuda(cudaEventCreateWithFlags(&eg.fork, cudaEventDisableTiming), "cudaEventCreate(env groups fork)");
        }
        while (static_cast<int>(eg.main.size()) < groups) {
            cudaStream_t s = nullptr;
            cudaEvent_t e = nullptr;
            CheckCuda(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate(env group)");
            CheckCuda(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "cudaEventCreate(env group join)");
            eg.main.push_back(s);
            eg.join.push_back(e);
            eg.side.push_back(MakeStagedStreams());
        }
        eg.groups = groups;
    }
    CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(env groups ready)");
    ++StepResourceEpoch();
    return groups;
}

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
    EnsureCudaStepStackLimit(cfg.mutator.physics_events.HasNonGoalEvents());
    EnsureEpaScratchPool();
    const int kPreTickEnvsPerBlock = std::max(1, std::min(32, 1024 / (cfg.max_cars_per_env + 1)));
    const int pre_block_size = kPreTickEnvsPerBlock * (cfg.max_cars_per_env + 1);
    const int pre_blocks = (cfg.env_count + kPreTickEnvsPerBlock - 1) / kPreTickEnvsPerBlock;
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
    bullet_port::btGjkEpaScratch* step_gjk_pool = nullptr;
    if (lanes > 1) {
        step_gjk_pool = EnsureGjkScratchPool(std::max(static_cast<size_t>(blocks) * block_size, CarWorldStagingThreads()), stream);
        if (step_gjk_pool == nullptr) {
            lanes = 1;
        }
    }
    const bool local_state = lanes == 1 && cfg.max_cars_per_env <= kLocalStepMaxSlots &&
                             cfg.env_count > StepKernelResidentThreads(block_size);
    bullet_port::btGjkEpaScratch* const staged_gjk_pool =
        !cfg.mutator.physics_events.HasNonGoalEvents() ? EnsureGjkScratchPool(CarWorldStagingThreads(), stream) : nullptr;
    if (staged_gjk_pool != nullptr) {
      BallWorldStage* const ball_stages = EnsureBallWorldStageArmed(cfg, stream);
      const int spread_env_bs = SpreadBlockSize(cfg.env_count);
      const int world4_bs = World4BlockSize(cfg.env_count, spread_env_bs);
      const int spread_tail_bs = SpreadBlockSize(static_cast<long long>(cfg.env_count) * kEnvTailLanes);
      const long long all_cars =
          cfg.env_car_total > 0 ? cfg.env_car_total : static_cast<long long>(cfg.env_count) * cfg.max_cars_per_env;
      const int spread_car_bs = SpreadBlockSize(all_cars);
      const int world3_bs = all_cars >= 4ll * cfg.env_count ? std::min(spread_env_bs, 16) : spread_env_bs;
      const int spread_rows_full_bs = SpreadBlockSize(all_cars / 8);
      const int world1_lanes = UseWorld1Lanes(cfg) ? kWorld1Lanes : 1;
      const bool post_fused = UsePostFused(cfg);
      const bool rays_inline = UseRaysInline(cfg);
      const bool register_iterations = UseRegisterIterations(cfg);
      const auto run_group = [&](const SimConfig& cfg, StagedStreams& side, cudaStream_t stream,
                                 const CarWorldStageSlice* slice) {
        const int pre_blocks = (cfg.env_count + kPreTickEnvsPerBlock - 1) / kPreTickEnvsPerBlock;
        const long long car_slots = static_cast<long long>(cfg.env_count) * cfg.max_cars_per_env;
        struct Shape { unsigned blocks; unsigned threads; };
        const auto shape = [](long long threads, int bs = kStagedBlockSize) {
            return Shape{static_cast<unsigned>((threads + bs - 1) / bs), static_cast<unsigned>(bs)};
        };
        const Shape car_pre = shape(car_slots, spread_car_bs);
        const Shape car_post = shape(car_slots, spread_car_bs);
        const Shape env_k = shape(cfg.env_count, spread_env_bs);
        const Shape world1 = shape(static_cast<long long>(cfg.env_count) * world1_lanes, spread_env_bs * world1_lanes);
        const Shape world3 = shape(cfg.env_count, world3_bs);
        int pred_body_shift = 0;
        while ((1 << pred_body_shift) < std::min(cfg.max_cars_per_env + 1, 9)) {
            ++pred_body_shift;
        }
        const int pred_lanes = StagedPredictiveRaysLanes(pred_body_shift);
        const long long pred_threads = (static_cast<long long>(cfg.env_count) << pred_body_shift) * pred_lanes;
        const int pred_bs = std::max(32, (SpreadBlockSize(pred_threads) + 31) / 32 * 32);
        const Shape pred_rays{static_cast<unsigned>((pred_threads + pred_bs - 1) / pred_bs), static_cast<unsigned>(pred_bs)};
        const long long w3_proxy_threads = static_cast<long long>(cfg.env_count) << pred_body_shift;
        const int w3_proxy_bs = std::max(32, (SpreadBlockSize(w3_proxy_threads) + 31) / 32 * 32);
        const Shape w3_proxies{static_cast<unsigned>((w3_proxy_threads + w3_proxy_bs - 1) / w3_proxy_bs),
                               static_cast<unsigned>(w3_proxy_bs)};
        const Shape world4 = shape(cfg.env_count, world4_bs);
        const Shape env_tail = shape(static_cast<long long>(cfg.env_count) * kEnvTailLanes, spread_tail_bs);
        const Shape rows_split = shape(car_slots);
        const Shape rows_light = shape(car_slots);
        const bool pads_apart = cfg.enable_boost_pads && BoostPadsApart(cfg);
        const long long pad_pre_slots = cfg.enable_boost_pads ? static_cast<long long>(cfg.env_count) * BoostPadCount(cfg) : 0;
        const Shape pad_pre = shape(pad_pre_slots);
        const unsigned pad_pre_blocks = pad_pre.blocks;
        bullet_port::btGjkEpaScratch* const rows_gjk_pool = slice != nullptr ? slice->gjk_pool : staged_gjk_pool;
        const size_t rows_threads = slice != nullptr ? slice->threads : CarWorldStagingThreads();
        const int rows_full_bs = spread_rows_full_bs;
        const unsigned row_blocks = PoolBlocks(
            static_cast<size_t>(std::min<long long>(car_slots, static_cast<long long>(rows_threads))), rows_threads, rows_full_bs);
        const bool split_rows = EnsureCarWorldStageArmed(cfg, stream);
        const cudaStream_t stage_stream = side.stage;
        const cudaStream_t dep_stream = side.dependent;
        int32_t* const pre_zero_counts =
            slice != nullptr ? slice->mesh_work_count
            : split_rows && EnsureGjkScratchPool(CarWorldStagingThreads(), stream) != nullptr
                ? GetCarWorldStageBuffers().mesh_work_count : nullptr;
        const bool pad_pre_on_ball = ball_stages != nullptr;
        const int tail_envs_per_block = static_cast<int>(env_tail.threads) / kEnvTailLanes;
        const bool post_pre_fused = post_fused && slice == nullptr &&
                                    tail_envs_per_block * (cfg.max_cars_per_env + 1) <= static_cast<int>(env_tail.threads);
        for (int32_t t = 0; t < ticks; ++t) {
            if (!post_pre_fused || t == 0) {
                PreTickKernel<<<pre_blocks, pre_block_size, 0, stream>>>(
                    cfg, teams, env_car_count, cars, solver_scratch, controls, car_configs, wheel_runtime_states, balls,
                    env_states, pads,
                    controls_by_tick, control_stride, env_commands_by_tick, env_command_stride,
                    respawn_slots, writable_respawn_slots, respawn_rng, respawn_order, seed, t, pad_pre_blocks == 0,
                    pre_zero_counts, kCarWorldMeshStageMeshes + 4 + kRowsFullKeys);
                CheckCuda(cudaGetLastError(), "PreTickKernel launch");
            }
            CheckCuda(cudaEventRecord(side.pre_done, stream), "cudaEventRecord(pre tick)");
            CheckCuda(cudaStreamWaitEvent(stage_stream, side.pre_done, 0), "cudaStreamWaitEvent(stage)");
            if (pad_pre_blocks > 0 && !pad_pre_on_ball) {
                StagedPadPreKernel<<<pad_pre.blocks, pad_pre.threads, 0, stage_stream>>>(cfg, env_car_count, pads);
                CheckCuda(cudaGetLastError(), "StagedPadPreKernel launch");
            }
            if (ball_stages != nullptr) {
                CheckCuda(cudaStreamWaitEvent(side.ball, side.pre_done, 0), "cudaStreamWaitEvent(ball stage)");
                StagedBallWorldKernel<<<env_k.blocks, env_k.threads, 0, side.ball>>>(cfg, balls, env_states, ball_stages);
                CheckCuda(cudaGetLastError(), "StagedBallWorldKernel launch");
                if (pad_pre_blocks > 0) {
                    StagedPadPreKernel<<<pad_pre.blocks, pad_pre.threads, 0, side.ball>>>(cfg, env_car_count, pads);
                    CheckCuda(cudaGetLastError(), "StagedPadPreKernel launch");
                }
                CheckCuda(cudaEventRecord(side.ball_done, side.ball), "cudaEventRecord(ball stage)");
            }
            LaunchStageCarWorldMeshCandidates(cfg, cars, car_configs, env_car_count, stage_stream, side.prefilter_done,
                                              slice, pre_zero_counts != nullptr);
            CheckCuda(cudaEventRecord(side.stage_done, stage_stream), "cudaEventRecord(stage)");
            LaunchStageWheelRays(cfg, cars, car_configs, wheel_runtime_states, balls, env_states, solver_scratch,
                                 env_car_count, stream);
            if (!rays_inline) {
                StagedVehicleIndependenceKernel<<<env_k.blocks, env_k.threads, 0, stream>>>(cfg, env_car_count, solver_scratch);
                CheckCuda(cudaGetLastError(), "StagedVehicleIndependenceKernel launch");
            }
            CheckCuda(cudaEventRecord(side.independence_done, stream), "cudaEventRecord(independence)");
            CheckCuda(cudaStreamWaitEvent(dep_stream, side.independence_done, 0), "cudaStreamWaitEvent(dependent)");
            CheckCuda(cudaStreamWaitEvent(dep_stream, side.prefilter_done, 0), "cudaStreamWaitEvent(dependent prefilter)");
            StagedCarPreKernel<<<car_pre.blocks, car_pre.threads, 0, stream>>>(
                cfg, env_car_count, cars, wheel_runtime_states, solver_scratch, controls, car_configs, balls, rays_inline);
            CheckCuda(cudaGetLastError(), "StagedCarPreKernel launch");
            const bool world1_split = world1_lanes > 1;
            if (world1_split) {
                StagedWorldKernel<kWorldStepContacts, kWorld1Lanes, false, 1><<<world1.blocks, world1.threads, 0, dep_stream>>>(
                    cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls, car_configs,
                    balls, env_states, pads, seed, rays_inline);
            } else {
                StagedWorldKernel<kWorldStepContacts><<<world1.blocks, world1.threads, 0, dep_stream>>>(
                    cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls, car_configs,
                    balls, env_states, pads, seed, rays_inline);
            }
            CheckCuda(cudaGetLastError(), "StagedWorldKernel contacts launch");
            CheckCuda(cudaEventRecord(side.dependent_done, dep_stream), "cudaEventRecord(dependent)");
            if (world1_split) {
                CheckCuda(cudaStreamWaitEvent(side.world1_contacts, side.dependent_done, 0),
                          "cudaStreamWaitEvent(world1 contacts)");
                if (ball_stages != nullptr) {
                    CheckCuda(cudaStreamWaitEvent(side.world1_contacts, side.ball_done, 0),
                              "cudaStreamWaitEvent(world1 contacts ball stage)");
                }
                StagedW3ProxiesKernel<<<w3_proxies.blocks, w3_proxies.threads, 0, side.world1_contacts>>>(
                    cfg, env_car_count, cars, car_configs, balls, solver_scratch, pred_body_shift, rays_inline, true);
                CheckCuda(cudaGetLastError(), "StagedW3ProxiesKernel (World<1> split) launch");
                StagedWorldKernel<kWorldStepContactsCarsApart, 1, false, 2><<<env_k.blocks, env_k.threads, 0, side.world1_contacts>>>(
                    cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls, car_configs,
                    balls, env_states, pads, seed, rays_inline);
                CheckCuda(cudaGetLastError(), "StagedWorldKernel contacts (World<1> split) launch");
                CheckCuda(cudaEventRecord(side.world1_contacts_done, side.world1_contacts), "cudaEventRecord(world1 contacts)");
            }
            CheckCuda(cudaStreamWaitEvent(stream, side.prefilter_done, 0), "cudaStreamWaitEvent(prefilter)");
            if (ball_stages != nullptr) {
                CheckCuda(cudaStreamWaitEvent(stream, side.ball_done, 0), "cudaStreamWaitEvent(ball stage)");
            }
            const cudaStream_t world3_stream = side.world3;
            CheckCuda(cudaEventRecord(side.world3_ready, stream), "cudaEventRecord(world3 ready)");
            CheckCuda(cudaStreamWaitEvent(side.world3, side.world3_ready, 0), "cudaStreamWaitEvent(world3)");
            CheckCuda(cudaStreamWaitEvent(side.world3_prefix, side.world3_ready, 0), "cudaStreamWaitEvent(world3 prefix)");
            CheckCuda(cudaStreamWaitEvent(side.world3_info, side.world3_ready, 0), "cudaStreamWaitEvent(world3 info)");
            StagedWorldKernel<kWorldStepContactsCarsApartInfo><<<world3.blocks, world3.threads, 0, side.world3_info>>>(
                cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls, car_configs,
                balls, env_states, pads, seed, rays_inline);
            CheckCuda(cudaGetLastError(), "StagedWorldKernel contacts prefix info (cars apart) launch");
            CheckCuda(cudaEventRecord(side.world3_info_done, side.world3_info), "cudaEventRecord(world3 info)");
            StagedW3ProxiesKernel<<<w3_proxies.blocks, w3_proxies.threads, 0, side.world3_prefix>>>(
                cfg, env_car_count, cars, car_configs, balls, solver_scratch, pred_body_shift, rays_inline, false);
            CheckCuda(cudaGetLastError(), "StagedW3ProxiesKernel launch");
            StagedWorldKernel<kWorldStepContactsCarsApartPrefix><<<world3.blocks, world3.threads, 0, side.world3_prefix>>>(
                cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls, car_configs,
                balls, env_states, pads, seed, rays_inline);
            CheckCuda(cudaGetLastError(), "StagedWorldKernel contacts prefix (cars apart) launch");
            CheckCuda(cudaEventRecord(side.world3_prefix_done, side.world3_prefix), "cudaEventRecord(world3 prefix)");
            if (UseStagedPredictiveRays(cfg)) {
                LaunchStagedPredictiveRays(pred_lanes, pred_rays.blocks, pred_rays.threads, world3_stream, cfg, env_car_count,
                                           cars, car_configs, balls, solver_scratch, pred_body_shift, rays_inline);
            }
            CheckCuda(cudaStreamWaitEvent(world3_stream, side.world3_prefix_done, 0), "cudaStreamWaitEvent(world3 prefix)");
            CheckCuda(cudaStreamWaitEvent(world3_stream, side.world3_info_done, 0), "cudaStreamWaitEvent(world3 info)");
            StagedWorldKernel<kWorldStepContactsCarsApart><<<world3.blocks, world3.threads, 0, world3_stream>>>(
                cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls, car_configs,
                balls, env_states, pads, seed, rays_inline);
            CheckCuda(cudaGetLastError(), "StagedWorldKernel contacts (cars apart) launch");
            CheckCuda(cudaEventRecord(side.world3_done, side.world3), "cudaEventRecord(world3)");
            CheckCuda(cudaStreamWaitEvent(stream, side.dependent_done, 0), "cudaStreamWaitEvent(join dependent)");
            CheckCuda(cudaStreamWaitEvent(stream, side.stage_done, 0), "cudaStreamWaitEvent(join stage)");
            if (split_rows) {
                CarWorldStageBuffers& b = GetCarWorldStageBuffers();
                int32_t* const row_car_count =
                    (slice != nullptr ? slice->mesh_work_count : b.mesh_work_count) + kCarWorldMeshStageMeshes + 1;
                int32_t* const row_cars = slice != nullptr ? slice->row_cars : b.row_cars;
                StagedCarWorldRowsSplitKernel<<<rows_split.blocks, rows_split.threads, 0, stream>>>(
                    cfg, env_car_count, cars, car_configs, solver_scratch, row_cars, row_car_count, b.row_reach);
                CheckCuda(cudaGetLastError(), "StagedCarWorldRowsSplitKernel launch");
                const cudaStream_t light_stream = side.dependent;
                CheckCuda(cudaEventRecord(side.rows_split_done, stream), "cudaEventRecord(rows split)");
                CheckCuda(cudaStreamWaitEvent(light_stream, side.rows_split_done, 0), "cudaStreamWaitEvent(rows split)");
                StagedCarWorldRowsListKernel<true><<<rows_light.blocks, rows_light.threads, 0, light_stream>>>(
                    cfg, cars, car_configs, solver_scratch, row_pool, row_cars, row_car_count, rows_gjk_pool, b.row_reach);
                CheckCuda(cudaGetLastError(), "StagedCarWorldRowsListKernel<true> launch");
                if (rows_full_bs <= 16 && EnsureCarWorldPlaneRows(stream)) {
                    StagedCarWorldRowsSplitPlanesKernel<<<row_blocks, 64, 0, stream>>>(
                        cfg, cars, car_configs, solver_scratch, row_pool, row_cars, row_car_count, rows_gjk_pool,
                        b.row_reach, b.plane_rows, b.plane_parts, rows_full_bs);
                    CheckCuda(cudaGetLastError(), "StagedCarWorldRowsSplitPlanesKernel launch");
                } else {
                    StagedCarWorldRowsListKernel<false><<<row_blocks, static_cast<unsigned>(rows_full_bs), 0, stream>>>(
                        cfg, cars, car_configs, solver_scratch, row_pool, row_cars, row_car_count, rows_gjk_pool,
                        b.row_reach);
                    CheckCuda(cudaGetLastError(), "StagedCarWorldRowsListKernel<false> launch");
                }
                CheckCuda(cudaEventRecord(side.rows_light_done, light_stream), "cudaEventRecord(rows light)");
                CheckCuda(cudaStreamWaitEvent(stream, side.rows_light_done, 0), "cudaStreamWaitEvent(rows light)");
            } else {
                StagedCarWorldRowsKernel<<<row_blocks, static_cast<unsigned>(rows_full_bs), 0, stream>>>(
                    cfg, env_car_count, cars, car_configs, solver_scratch, row_pool, rows_gjk_pool);
                CheckCuda(cudaGetLastError(), "StagedCarWorldRowsKernel launch");
            }
            CheckCuda(cudaStreamWaitEvent(stream, side.world3_done, 0), "cudaStreamWaitEvent(join world3)");
            if (world1_split) {
                CheckCuda(cudaStreamWaitEvent(stream, side.world1_contacts_done, 0), "cudaStreamWaitEvent(join world1 contacts)");
            }
            if (register_iterations) {
                StagedWorldKernel<kWorldStepRowsCarsApart, 1, true><<<world4.blocks, world4.threads, 0, stream>>>(
                    cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls,
                    car_configs, balls, env_states, pads, seed);
            } else {
                StagedWorldKernel<kWorldStepRowsCarsApart><<<world4.blocks, world4.threads, 0, stream>>>(
                    cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls,
                    car_configs, balls, env_states, pads, seed);
            }
            CheckCuda(cudaGetLastError(), "StagedWorldKernel rows launch");
            if (post_pre_fused && t + 1 < ticks) {
                StagedPostPreFusedKernel<<<env_tail.blocks, env_tail.threads, 0, stream>>>(
                    cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, controls, car_configs, balls,
                    env_states, pads, pads_apart, tail_envs_per_block, controls_by_tick, control_stride,
                    env_commands_by_tick, env_command_stride, respawn_slots, writable_respawn_slots, respawn_rng,
                    respawn_order, seed, t + 1, pad_pre_blocks == 0, pre_zero_counts,
                    kCarWorldMeshStageMeshes + 4 + kRowsFullKeys);
                CheckCuda(cudaGetLastError(), "StagedPostPreFusedKernel launch");
                continue;
            }
            if (post_fused) {
                StagedPostFusedKernel<<<env_tail.blocks, env_tail.threads, 0, stream>>>(
                    cfg, env_car_count, cars, wheel_runtime_states, solver_scratch, controls, car_configs, balls,
                    env_states, pads, pads_apart, static_cast<int>(env_tail.threads) / kEnvTailLanes);
                CheckCuda(cudaGetLastError(), "StagedPostFusedKernel launch");
                continue;
            }
            StagedCarPostKernel<<<car_post.blocks, car_post.threads, 0, stream>>>(
                cfg, env_car_count, cars, wheel_runtime_states, solver_scratch, controls, car_configs);
            CheckCuda(cudaGetLastError(), "StagedCarPostKernel launch");
            StagedEnvTailKernel<<<env_tail.blocks, env_tail.threads, 0, stream>>>(
                cfg, env_car_count, cars, car_configs, balls, env_states, pads, pads_apart);
            CheckCuda(cudaGetLastError(), "StagedEnvTailKernel launch");
        }
      };
        const int groups = StagedEnvGroupCount(cfg, stream);
        if (groups <= 1) {
            run_group(cfg, GetStagedStreams(), stream, nullptr);
            return;
        }
        StagedEnvGroups& eg = GetStagedEnvGroups();
        CarWorldStageBuffers& b = GetCarWorldStageBuffers();
        const int stride = cfg.max_cars_per_env;
        const int per_group = (cfg.env_count + groups - 1) / groups;
        const bool interleave =
            groups >= 2 && cfg.env_car_total > 0 &&
            (cfg.env_car_total < static_cast<long long>(cfg.env_count) * cfg.max_cars_per_env ||
             cfg.env_car_total <= kInterleaveFullCarsMax);
        if (interleave) {
            InterleaveGroupsKernel<<<(cfg.env_count + 255) / 256, 256, 0, stream>>>(cfg.env_step_order, eg.split,
                                                                                    cfg.env_count, groups);
            CheckCuda(cudaGetLastError(), "InterleaveGroupsKernel launch");
        }
        const size_t group_threads =
            std::max<size_t>(kCudaRayKernelBlockSize, CarWorldStagingThreads() / groups / kCudaRayKernelBlockSize * kCudaRayKernelBlockSize);
        const int fit = std::max(1, kLiveEnvsMax / per_group);
        const int waves = (groups + fit - 1) / fit;
        const int live = (groups + waves - 1) / waves;
        CheckCuda(cudaEventRecord(eg.fork, stream), "cudaEventRecord(env groups fork)");
        int interleaved_first = 0;
        for (int g = 0; g < groups; ++g) {
            const int first = interleave ? interleaved_first : g * per_group;
            const int n = interleave ? (cfg.env_count - g + groups - 1) / groups : std::min(per_group, cfg.env_count - first);
            interleaved_first += n;
            if (n <= 0) {
                break;
            }
            SimConfig gcfg = cfg;
            gcfg.env_count = n;
            gcfg.env_step_order =
                (interleave ? eg.split : cfg.env_step_order != nullptr ? cfg.env_step_order : eg.order) + first;
            const size_t car_first = static_cast<size_t>(first) * stride;
            const size_t cars_n = static_cast<size_t>(n) * stride;
            CarWorldStageSlice slice;
            slice.mesh_work = b.mesh_work + car_first * kCarWorldMeshStageMeshes;
            slice.mesh_work_count = eg.counts + static_cast<size_t>(g) * (kCarWorldMeshStageMeshes + 4 + kRowsFullKeys);
            slice.row_cars = b.row_cars + (1 + kRowsFullKeys) * car_first;
            slice.pairs = b.pairs;
            slice.pairs.car_idx = b.pairs.car_idx + car_first * kCarWorldTrianglePairsPerCar;
            slice.pairs.tri_idx = b.pairs.tri_idx + car_first * kCarWorldTrianglePairsPerCar;
            slice.pairs.hits = b.pairs.hits + car_first * kCarWorldTrianglePairsPerCar;
            slice.pairs.count = slice.mesh_work_count + kCarWorldMeshStageMeshes;
            slice.pairs.capacity = static_cast<int32_t>(std::min<size_t>(cars_n * kCarWorldTrianglePairsPerCar, INT32_MAX));
            slice.gjk_pool = staged_gjk_pool + static_cast<size_t>(g) * group_threads;
            slice.threads = group_threads;
            const cudaStream_t gstream = eg.main[g];
            CheckCuda(cudaStreamWaitEvent(gstream, eg.fork, 0), "cudaStreamWaitEvent(env group fork)");
            if (live > 0 && g >= live) {
                CheckCuda(cudaStreamWaitEvent(gstream, eg.join[g - live], 0), "cudaStreamWaitEvent(env group wave)");
            }
            run_group(gcfg, eg.side[g], gstream, &slice);
            CheckCuda(cudaEventRecord(eg.join[g], gstream), "cudaEventRecord(env group join)");
            CheckCuda(cudaStreamWaitEvent(stream, eg.join[g], 0), "cudaStreamWaitEvent(env group join)");
        }
        return;
    }
    for (int32_t t = 0; t < ticks; ++t) {
        PreTickKernel<<<pre_blocks, pre_block_size, 0, stream>>>(
            cfg, teams, env_car_count, cars, solver_scratch, controls, car_configs, wheel_runtime_states, balls,
            env_states, pads,
            controls_by_tick, control_stride, env_commands_by_tick, env_command_stride,
            respawn_slots, writable_respawn_slots, respawn_rng, respawn_order, seed, t);
        CheckCuda(cudaGetLastError(), "PreTickKernel launch");
        StagedStreams& side = GetStagedStreams();
        CheckCuda(cudaEventRecord(side.pre_done, stream), "cudaEventRecord(pre tick)");
        CheckCuda(cudaStreamWaitEvent(side.stage, side.pre_done, 0), "cudaStreamWaitEvent(stage)");
        LaunchStageCarWorldMeshCandidates(cfg, cars, car_configs, env_car_count, side.stage);
        CheckCuda(cudaEventRecord(side.stage_done, side.stage), "cudaEventRecord(stage)");
        LaunchStageWheelRays(cfg, cars, car_configs, wheel_runtime_states, balls, env_states, solver_scratch,
                             env_car_count, stream);
        CheckCuda(cudaStreamWaitEvent(stream, side.stage_done, 0), "cudaStreamWaitEvent(join stage)");
        StepKernel<<<blocks, block_size, 0, stream>>>(
            cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls, car_configs,
            balls, env_states, pads, seed, lanes, step_gjk_pool, local_state);
        CheckCuda(cudaGetLastError(), "StepKernel launch");
    }
}

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
    if (ticks <= 0 || StreamIsCapturing(stream)) {
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
        constexpr size_t kMaxEntries = 32;
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
    const CarConfig* car_configs,
    cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    const int blocks = (cfg.env_count + kBlockSize - 1) / kBlockSize;
    ResetToKickoffKernel<<<blocks, kBlockSize, 0, stream>>>(
        cfg, teams, env_car_count, cars, controls, balls, env_states, pads, seed, respawn_rng, kickoff_seed_mode,
        car_configs);
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
    const CarConfig* car_configs,
    cudaStream_t stream) {
    if (command_count <= 0) {
        return;
    }
    constexpr int kBlockSize = 256;
    const int blocks = (command_count + kBlockSize - 1) / kBlockSize;
    ApplyEnvCommandsKernel<<<blocks, kBlockSize, 0, stream>>>(
        cfg, teams, env_car_count, cars, controls, solver_scratch, balls, env_states, pads, env_commands, command_count, respawn_rng,
        car_configs);
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
