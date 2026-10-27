#include "rocketgpu/detail/cuda_api.hpp"

#if defined(ROCKETGPU_USE_CUDA)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <thread>
#include <utility>
#include <vector>

#include "rocketgpu/detail/step.hpp"
#include "rocketgpu/detail/respawn_rng.hpp"

namespace rocketgpu {
namespace detail {
__device__ CarWorldMeshCandidateStage* g_rocketgpu_car_world_stage = nullptr;
__device__ int g_rocketgpu_car_world_stage_stride = 0;

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
    SimConfig cfg, const CarState* cars, const int32_t* counts,
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

ROCKETGPU_HD float CanonicalizeImportedLinearScalar(float value) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kBTToUU = 50.0f;
    return (value * kUUToBT) * kBTToUU;
}

ROCKETGPU_HD Vec3 CanonicalizeImportedLinearVec3(const Vec3& value) {
    return Vec3(
        CanonicalizeImportedLinearScalar(value.x),
        CanonicalizeImportedLinearScalar(value.y),
        CanonicalizeImportedLinearScalar(value.z));
}

ROCKETGPU_HD float ImportedCarLinearScalarBt(float value) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    return CanonicalizeImportedLinearScalar(value) * kUUToBT;
}

ROCKETGPU_HD Vec3 ImportedCarLinearVec3Bt(const Vec3& value) {
    return Vec3(
        ImportedCarLinearScalarBt(value.x),
        ImportedCarLinearScalarBt(value.y),
        ImportedCarLinearScalarBt(value.z));
}

ROCKETGPU_HD float ImportedBallLinearScalarBt(float value) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    return CanonicalizeImportedLinearScalar(value) * kUUToBT;
}

ROCKETGPU_HD Vec3 ImportedBallLinearVec3Bt(const Vec3& value) {
    return Vec3(
        ImportedBallLinearScalarBt(value.x),
        ImportedBallLinearScalarBt(value.y),
        ImportedBallLinearScalarBt(value.z));
}

ROCKETGPU_GLOBAL void ResetToKickoffKernel(
    SimConfig cfg,
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
        for (int manifold_idx = 0; manifold_idx < kMaxCarWorldMeshPersistentManifolds; ++manifold_idx) {
            solver_scratch[env_idx].car_world_mesh_manifolds[manifold_idx] = CarWorldMeshPersistentManifold{};
        }
        for (int manifold_idx = 0; manifold_idx < kMaxCarWorldPlaneRetainedManifolds; ++manifold_idx) {
            solver_scratch[env_idx].car_world_plane_manifolds[manifold_idx].count = 0;
        }
        for (int manifold_idx = 0; manifold_idx < kMaxCarCarRetainedManifolds; ++manifold_idx) {
            solver_scratch[env_idx].car_car_retained_manifolds[manifold_idx].count = 0;
        }
        for (int manifold_idx = 0; manifold_idx < kMaxBallCarRetainedManifolds; ++manifold_idx) {
            solver_scratch[env_idx].ball_car_retained_manifolds[manifold_idx].count = 0;
        }
    }
}

ROCKETGPU_GLOBAL void ApplyEnvCommandsKernel(
    SimConfig cfg,
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
    cars[idx].ResetTrainingHistory();
    cars[idx].pos_bt = ImportedCarLinearVec3Bt(cars[idx].pos);
    cars[idx].vel_bt = ImportedCarLinearVec3Bt(cars[idx].vel);
    cars[idx].pos = cars[idx].pos_bt * kCarBTToUU;
    cars[idx].vel = CanonicalizeImportedLinearVec3(cars[idx].vel);
    CanonicalizeImportedCarOrientation(cars[idx]);
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
}

ROCKETGPU_GLOBAL void StepControlStreamKernel(
    SimConfig cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch,
    WorldSolverScratchRowPool row_pool,
    CarControls* controls,
    const CarControls* controls_by_tick,
    int32_t control_stride,
    CarConfig* car_configs,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    uint64_t seed,
    int32_t ticks,
    RespawnRngState* respawn_rng,
    const int32_t* respawn_order,
    uint8_t* respawn_slots,
    int32_t phase,
    int32_t tick_offset) {
    int env_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (env_idx >= cfg.env_count) {
        return;
    }
    MapStepEnv(cfg, env_idx);
    for (int tick = 0; tick < ticks; ++tick) {
        const int tick_idx = tick_offset + tick;

        if (phase != kStepEnvPhasePost) {
            StageStreamControlsForEnv(env_idx, cfg, controls, controls_by_tick, control_stride, tick_idx);
            if (respawn_rng) {
                PrepareRespawnSlotsForEnv(env_idx, cfg, cars, env_car_count, env_states,
                    respawn_rng, respawn_order, respawn_slots);
            }
        }
        StepEnv(env_idx, cfg, teams, env_car_count, cars, wheel_runtime_states,
            solver_scratch, row_pool, controls, car_configs, balls, env_states, pads,
            respawn_slots, seed, 1, phase, tick_idx);
    }
}

ROCKETGPU_GLOBAL void StepCommandStreamsKernel(
    SimConfig cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch,
    WorldSolverScratchRowPool row_pool,
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
    RespawnRngState* respawn_rng,
    const int32_t* respawn_order,
    uint8_t* respawn_slots,
    int32_t phase,
    int32_t tick_offset) {
    int env_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (env_idx >= cfg.env_count) {
        return;
    }
    MapStepEnv(cfg, env_idx);
    for (int tick = 0; tick < ticks; ++tick) {
        const int tick_idx = tick_offset + tick;
        if (phase != kStepEnvPhasePost) {
            const EnvCommand command = env_commands_by_tick[
                static_cast<size_t>(tick_idx) * static_cast<size_t>(env_command_stride) + static_cast<size_t>(env_idx)];
            ApplyEnvCommandForEnv(env_idx, cfg, teams, env_car_count, cars, controls, solver_scratch,
                                  balls, env_states, pads, command, respawn_rng);
            StageStreamControlsForEnv(env_idx, cfg, controls, controls_by_tick, control_stride, tick_idx);
            if (respawn_rng) {
                PrepareRespawnSlotsForEnv(env_idx, cfg, cars, env_car_count, env_states,
                    respawn_rng, respawn_order, respawn_slots);
            }
        }
        StepEnv(env_idx, cfg, teams, env_car_count, cars, wheel_runtime_states,
            solver_scratch, row_pool, controls, car_configs, balls, env_states, pads,
            respawn_slots, seed, 1, phase, tick_idx);
    }
}

ROCKETGPU_GLOBAL void StepKernel(
    SimConfig cfg,
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
    const uint8_t* respawn_slot_overrides,
    uint64_t seed,
    int32_t ticks,
    RespawnRngState* respawn_rng,
    const int32_t* respawn_order,
    uint8_t* writable_respawn_slots,
    int32_t phase,
    int32_t tick_offset) {
    int env_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (env_idx >= cfg.env_count) {
        return;
    }
    MapStepEnv(cfg, env_idx);
    if (respawn_rng) {
        for (int tick = 0; tick < ticks; ++tick) {
            if (phase != kStepEnvPhasePost) {
                PrepareRespawnSlotsForEnv(env_idx, cfg, cars, env_car_count, env_states,
                    respawn_rng, respawn_order, writable_respawn_slots);
            }
            StepEnv(env_idx, cfg, teams, env_car_count, cars, wheel_runtime_states,
                solver_scratch, row_pool, controls, car_configs, balls, env_states, pads,
                writable_respawn_slots, seed, 1, phase, tick_offset + tick);
        }
    } else {
        StepEnv(env_idx, cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls, car_configs, balls, env_states, pads, respawn_slot_overrides, seed, ticks, phase, tick_offset);
    }
}

ROCKETGPU_GLOBAL void StageWheelRaysKernel(
    SimConfig cfg,
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

ROCKETGPU_GLOBAL void StageCarWorldMeshCandidatesKernel(
    SimConfig cfg,
    const CarState* cars,
    const CarConfig* car_configs,
    const int32_t* env_car_count,
    CarWorldMeshCandidateStage* stages,
    bullet_port::btGjkEpaScratch* gjk_scratch) {
    const int stride = cfg.max_cars_per_env;
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int env_idx = stride > 0 ? tid / stride : cfg.env_count;
    if (env_idx >= cfg.env_count) {
        return;
    }
    const int slot = tid - env_idx * stride;
    CarWorldMeshCandidateStage& stage = stages[tid];
    stage.valid = 0;
    if (slot >= env_car_count[env_idx]) {
        return;
    }
    const int car_idx = env_idx * stride + slot;
    const CarState& car = cars[car_idx];
    if (car.is_demoed || car.just_respawned || !cfg.mutator.arena_collision ||
        cfg.arena_mesh_tris == nullptr || cfg.arena_mesh_tri_count <= 0) {
        return;
    }
    const CarConfig& car_cfg = car_configs[car_idx];
    FillCarWorldMeshCandidateFingerprint(car, car_cfg, stage.fingerprint);
    ContactAccumulator acc;
    int split_row_count = 0;
    int normal_row_count = 0;
    int friction_row_count = 0;
    AppendCarWorldMeshContactRows(
        car, car_cfg, cfg.mutator, Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 0.0f), acc,
        nullptr, split_row_count, nullptr, normal_row_count, nullptr, friction_row_count, 0.0f,
        cfg.arena_mesh_tris, cfg.arena_mesh_tri_count,
        cfg.arena_mesh_bvh_quantization, cfg.arena_mesh_bvh_quantization_count,
        cfg.arena_mesh_bvh_meshes, cfg.arena_mesh_bvh_mesh_count,
        cfg.arena_mesh_bvh_nodes, cfg.arena_mesh_bvh_node_count,
        cfg.arena_mesh_bvh_subtrees, cfg.arena_mesh_bvh_subtree_count,
        nullptr, 0, env_idx, slot, &gjk_scratch[tid], nullptr, nullptr, nullptr, 0.0f,
        &stage, nullptr);
}

inline void EnsureCudaStepStackLimit() {
    constexpr size_t kMinRequiredStackBytes = 43008;
    constexpr size_t kMaxRequiredStackBytes = 65536;
    constexpr size_t kCallChainHeadroomBytes = 8192;
    static bool configured = false;
    if (configured) {
        return;
    }
    size_t required_stack_bytes = kMinRequiredStackBytes;
    const void* const step_kernels[] = {
        reinterpret_cast<const void*>(&StepKernel),
        reinterpret_cast<const void*>(&StepControlStreamKernel),
        reinterpret_cast<const void*>(&StepCommandStreamsKernel),
    };
    for (const void* kernel : step_kernels) {
        cudaFuncAttributes attr{};
        if (cudaFuncGetAttributes(&attr, kernel) != cudaSuccess) {
            cudaGetLastError();
            continue;
        }
        required_stack_bytes = std::max(required_stack_bytes, static_cast<size_t>(attr.localSizeBytes) + kCallChainHeadroomBytes);
    }
    required_stack_bytes = std::min(required_stack_bytes, kMaxRequiredStackBytes);
    size_t current_limit = 0;
    CheckCuda(cudaDeviceGetLimit(&current_limit, cudaLimitStackSize), "cudaDeviceGetLimit(cudaLimitStackSize)");
    if (current_limit < required_stack_bytes) {
        CheckCuda(cudaDeviceSetLimit(cudaLimitStackSize, required_stack_bytes), "cudaDeviceSetLimit(cudaLimitStackSize)");
    }
    configured = true;
}

// One thread per env; 64-thread blocks once there are 64+ envs per SM.
inline int GetCudaStepBlockSize(int env_count) {
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
    return env_count >= 64 * sm_count ? 64 : 32;
}

constexpr int kCudaRayKernelBlockSize = 128;

struct CarWorldStageBuffers {
    CarWorldMeshCandidateStage* stages = nullptr;
    bullet_port::btGjkEpaScratch* gjk_scratch = nullptr;
    size_t capacity = 0;
    CarWorldMeshCandidateStage* published = nullptr;
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
    CheckCuda(cudaMemcpyToSymbolAsync(g_rocketgpu_car_world_stage, &stages, sizeof(stages), 0,
                                      cudaMemcpyHostToDevice, stream),
              "cudaMemcpyToSymbolAsync(g_rocketgpu_car_world_stage)");
    CheckCuda(cudaMemcpyToSymbolAsync(g_rocketgpu_car_world_stage_stride, &stride, sizeof(stride), 0,
                                      cudaMemcpyHostToDevice, stream),
              "cudaMemcpyToSymbolAsync(g_rocketgpu_car_world_stage_stride)");
}
static bool StreamIsCapturing(cudaStream_t stream);

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
            b.stages = nullptr;
        }
        if (b.gjk_scratch != nullptr) {
            cudaFree(b.gjk_scratch);
            b.gjk_scratch = nullptr;
        }
        b.capacity = 0;
        b.published = nullptr;
        CheckCuda(cudaMalloc(&b.stages, count * sizeof(CarWorldMeshCandidateStage)), "cudaMalloc(car-world stage)");
        CheckCuda(cudaMalloc(&b.gjk_scratch, count * sizeof(bullet_port::btGjkEpaScratch)),
                  "cudaMalloc(car-world stage gjk scratch)");
        CheckCuda(cudaMemset(b.stages, 0, count * sizeof(CarWorldMeshCandidateStage)), "cudaMemset(car-world stage)");
        CheckCuda(cudaMemset(b.gjk_scratch, 0, count * sizeof(bullet_port::btGjkEpaScratch)),
                  "cudaMemset(car-world stage gjk scratch)");
        b.capacity = count;
    }
    if (b.published != b.stages || b.published_stride != cfg.max_cars_per_env) {
        if (capturing) {
            return false;
        }
        PublishCarWorldStage(b.stages, cfg.max_cars_per_env, stream);
    }
    return true;
}
void PrepareCarWorldStage(int env_count, int max_cars_per_env, cudaStream_t stream) {
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
    cudaStream_t stream) {
    if (!EnsureCarWorldStageArmed(cfg, stream)) {
        if (!StreamIsCapturing(stream)) {
            PublishCarWorldStage(nullptr, 0, stream);
        }
        return;
    }
    CarWorldStageBuffers& b = GetCarWorldStageBuffers();
    const size_t count = static_cast<size_t>(cfg.env_count) * static_cast<size_t>(cfg.max_cars_per_env);
    const int block_size = kCudaRayKernelBlockSize;
    const size_t blocks = (count + static_cast<size_t>(block_size) - 1) / static_cast<size_t>(block_size);
    StageCarWorldMeshCandidatesKernel<<<static_cast<unsigned>(blocks), static_cast<unsigned>(block_size), 0, stream>>>(
        cfg, cars, car_configs, env_car_count, b.stages, b.gjk_scratch);
    CheckCuda(cudaGetLastError(), "StageCarWorldMeshCandidatesKernel launch");
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
    const long long blocks = (rays + block_size - 1) / block_size;
    StageWheelRaysKernel<<<static_cast<unsigned>(blocks), static_cast<unsigned>(block_size), 0, stream>>>(
        cfg, cars, car_configs, wheel_runtime_states, balls, env_states, solver_scratch, env_car_count);
    CheckCuda(cudaGetLastError(), "StageWheelRaysKernel launch");
    LaunchStageCarWorldMeshCandidates(cfg, cars, car_configs, env_car_count, stream);
}

template <typename LaunchStep, typename LaunchRays>
inline void RunStepLaunches(int32_t ticks, const LaunchStep& launch, const LaunchRays& launch_rays) {
    for (int32_t t = 0; t < ticks; ++t) {
        launch(kStepEnvPhasePre, t);
        launch_rays();
        launch(kStepEnvPhasePost, t);
    }
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
    EnsureCudaStepStackLimit();
    const int block_size = GetCudaStepBlockSize(cfg.env_count);
    const int blocks = (cfg.env_count + block_size - 1) / block_size;
    const auto launch = [&](int32_t phase, int32_t tick_offset) {
        StepKernel<<<blocks, block_size, 0, stream>>>(
            cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls, car_configs, balls, env_states, pads, respawn_slot_overrides, seed, 1,
            respawn_rng, respawn_order, writable_respawn_slots,
            phase, tick_offset);
        CheckCuda(cudaGetLastError(), "StepKernel launch");
    };
    RunStepLaunches(ticks, launch, [&]() {
        LaunchStageWheelRays(cfg, cars, car_configs, wheel_runtime_states, balls, env_states, solver_scratch, env_car_count, stream);
    });
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
    EnsureCudaStepStackLimit();
    const int block_size = GetCudaStepBlockSize(cfg.env_count);
    const int blocks = (cfg.env_count + block_size - 1) / block_size;
    const auto launch = [&](int32_t phase, int32_t tick_offset) {
        StepControlStreamKernel<<<blocks, block_size, 0, stream>>>(
            cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls, controls_by_tick, control_stride, car_configs, balls, env_states, pads, seed, 1,
            respawn_rng, respawn_order, respawn_slots,
            phase, tick_offset);
        CheckCuda(cudaGetLastError(), "StepControlStreamKernel launch");
    };
    RunStepLaunches(ticks, launch, [&]() {
        LaunchStageWheelRays(cfg, cars, car_configs, wheel_runtime_states, balls, env_states, solver_scratch, env_car_count, stream);
    });
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
    EnsureCudaStepStackLimit();
    const int block_size = GetCudaStepBlockSize(cfg.env_count);
    const int blocks = (cfg.env_count + block_size - 1) / block_size;
    const auto launch = [&](int32_t phase, int32_t tick_offset) {
        StepCommandStreamsKernel<<<blocks, block_size, 0, stream>>>(
            cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls, controls_by_tick, control_stride,
            env_commands_by_tick, env_command_stride, car_configs, balls, env_states, pads, seed, 1,
            respawn_rng, respawn_order, respawn_slots,
            phase, tick_offset);
        CheckCuda(cudaGetLastError(), "StepCommandStreamsKernel launch");
    };
    RunStepLaunches(ticks, launch, [&]() {
        LaunchStageWheelRays(cfg, cars, car_configs, wheel_runtime_states, balls, env_states, solver_scratch, env_car_count, stream);
    });
}
}
}

#endif
