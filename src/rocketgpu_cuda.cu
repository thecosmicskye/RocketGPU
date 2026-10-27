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

// Tick start: a thread per car slot, a block holding whole envs. Each env's first thread applies the env's
// command, stream controls and respawn slots; then every car, and the ball and pads, start the tick in parallel.
ROCKETGPU_GLOBAL void PreTickKernel(
    SimConfig cfg,
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
    int32_t tick_idx) {
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
    if (!active) {
        return;
    }
    const int count = env_car_count[env_idx];
    if (slot < count) {
        StepEnvPreTickCar(env_idx, slot, cfg, teams, cars, solver_scratch, env_states[env_idx], respawn_slots, seed);
    }
    if (slot == 0) {
        StepEnvPreTickBallAndPads(env_idx, cfg, count, solver_scratch, balls, pads);
    }
}

// The rest of the tick, after PreTickKernel and the staged wheel rays: one thread per env.
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
    uint64_t seed,
    bool vehicles_done,
    int world_part) {
    int env_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (env_idx >= cfg.env_count) {
        return;
    }
    MapStepEnv(cfg, env_idx);
    StepEnv(env_idx, cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls,
            car_configs, balls, env_states, pads, nullptr, seed, 1,
            vehicles_done ? kStepEnvPhasePostVehiclesDone : kStepEnvPhasePost, 0, world_part);
}

// The car-world rows of the world step, a thread per car, between StepKernel's kWorldStepContacts and kWorldStepRows.
ROCKETGPU_GLOBAL void CarWorldRowsKernel(
    SimConfig cfg,
    const int32_t* env_car_count,
    CarState* cars,
    const CarConfig* car_configs,
    WorldSolverScratch* solver_scratch,
    WorldSolverScratchRowPool row_pool) {
    const int stride = cfg.max_cars_per_env;
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<long long>(cfg.env_count) * stride) {
        return;
    }
    const int env_idx = static_cast<int>(tid / stride);
    const int slot = static_cast<int>(tid - static_cast<long long>(env_idx) * stride);
    if (slot >= env_car_count[env_idx]) {
        return;
    }
    // Raw storage, as in StageCarWorldMeshCandidatesKernel.
    alignas(bullet_port::btGjkEpaScratch) unsigned char gjk_storage[sizeof(bullet_port::btGjkEpaScratch)];
    BuildCarWorldRowsForCar(env_idx, slot, cfg, cars, car_configs, solver_scratch[env_idx], row_pool.view(env_idx),
                            reinterpret_cast<bullet_port::btGjkEpaScratch*>(gjk_storage));
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

// Vehicle updates for envs whose cars are independent (VehicleUpdatesIndependent), a thread per car; StepKernel
// runs the others in visit order.
ROCKETGPU_GLOBAL void VehicleUpdateKernel(
    SimConfig cfg,
    const int32_t* env_car_count,
    CarState* cars,
    WheelRuntimeState* wheel_runtime_states,
    const WorldSolverScratch* solver_scratch,
    const CarControls* controls,
    CarConfig* car_configs,
    BallState* balls) {
    const int stride = cfg.max_cars_per_env;
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<long long>(cfg.env_count) * stride) {
        return;
    }
    const int env_idx = static_cast<int>(tid / stride);
    const int slot = static_cast<int>(tid - static_cast<long long>(env_idx) * stride);
    const int count = env_car_count[env_idx];
    if (slot >= count) {
        return;
    }
    const WheelRayStage* const staged = solver_scratch[env_idx].wheel_ray_stage;
    if (!VehicleUpdatesIndependent(cfg, staged, count)) {
        return;
    }
    uint32_t pushed_cars_mask = 0u;
    StepEnvVehicleUpdateCar(env_idx, slot, count, cfg, cars, wheel_runtime_states, car_configs, balls,
                            controls[env_idx * stride + slot], staged, pushed_cars_mask);
}

// Car-world mesh staging, in two passes. The prefilter runs a thread per car slot: it writes the stage header,
// marks the meshes the car's query AABB may touch, and lists each (car, overlapping mesh) pair under its
// mesh. The staging pass runs a thread per listed pair, mesh-major so a warp walks one mesh's BVH. Every pair writes
// only its own entry, so the list order does not affect the result.
ROCKETGPU_GLOBAL void PrefilterCarWorldMeshCandidatesKernel(
    SimConfig cfg,
    const CarState* cars,
    const CarConfig* car_configs,
    const int32_t* env_car_count,
    CarWorldMeshCandidateStage* stages,
    int32_t* mesh_work,
    int32_t* mesh_work_count) {
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
    FillCarWorldMeshCandidateFingerprint(car, car_cfg, stage.fingerprint);
    stage.mesh_count = have_bvh_traversal ? cfg.arena_mesh_bvh_mesh_count : 0;
    stage.valid = have_bvh_traversal ? 1 : 0;
    if (!have_bvh_traversal) {
        return;
    }
    const CarWorldMeshQueryFrame frame = ComputeCarWorldMeshQueryFrame(car, car_cfg);
    uint32_t mesh_mask = 0;
    for (int mesh = 0; mesh < cfg.arena_mesh_bvh_mesh_count; ++mesh) {
        const ArenaMeshBvhMeshInfo& mesh_info = meshes[mesh];
        if (CarWorldMeshInfoUsable(mesh_info, cfg.arena_mesh_bvh_quantization_count, cfg.arena_mesh_bvh_node_count) &&
            CarWorldMeshAabbMayOverlap(mesh_info, frame.aabb_min_bt, frame.aabb_max_bt)) {
            mesh_mask |= 1u << mesh;
            const int item = atomicAdd(&mesh_work_count[mesh], 1);
            mesh_work[static_cast<long long>(mesh) * car_slots + item] = car_idx;
        }
    }
    stage.mesh_mask = mesh_mask;
}

ROCKETGPU_GLOBAL void StageCarWorldMeshCandidatesKernel(
    SimConfig cfg,
    const CarState* cars,
    const CarConfig* car_configs,
    CarWorldMeshCandidateStage* stages,
    const int32_t* mesh_work,
    const int32_t* mesh_work_count,
    int spread) {
    const int stride = cfg.max_cars_per_env;
    const long long car_slots = static_cast<long long>(cfg.env_count) * stride;
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long span = car_slots * spread;
    const int mesh = static_cast<int>(tid / span);
    const long long lane = tid - mesh * span;
    const long long item = lane / spread;
    if (lane - item * spread != 0 || mesh >= cfg.arena_mesh_bvh_mesh_count || item >= mesh_work_count[mesh]) {
        return;
    }
    const int car_idx = mesh_work[static_cast<long long>(mesh) * car_slots + item];
    const int env_idx = car_idx / stride;
    const int slot = car_idx - env_idx * stride;
    // Raw storage: GJK/EPA initialize it on use, and constructing 28 KB per thread every tick dominates otherwise.
    alignas(bullet_port::btGjkEpaScratch) unsigned char gjk_storage[sizeof(bullet_port::btGjkEpaScratch)];
    bullet_port::btGjkEpaScratch* const gjk_scratch = reinterpret_cast<bullet_port::btGjkEpaScratch*>(gjk_storage);
    ContactAccumulator acc;
    int split_row_count = 0;
    int normal_row_count = 0;
    int friction_row_count = 0;
    AppendCarWorldMeshContactRows(
        cars[car_idx], car_configs[car_idx], cfg.mutator, Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 0.0f), acc,
        nullptr, split_row_count, nullptr, normal_row_count, nullptr, friction_row_count, 0.0f,
        cfg.arena_mesh_tris, cfg.arena_mesh_tri_count,
        cfg.arena_mesh_bvh_quantization, cfg.arena_mesh_bvh_quantization_count,
        cfg.arena_mesh_bvh_meshes, cfg.arena_mesh_bvh_mesh_count,
        cfg.arena_mesh_bvh_nodes, cfg.arena_mesh_bvh_node_count,
        cfg.arena_mesh_bvh_subtrees, cfg.arena_mesh_bvh_subtree_count,
        env_idx, slot, gjk_scratch, nullptr, nullptr, nullptr, 0.0f,
        &stages[car_idx], nullptr, mesh);
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
        reinterpret_cast<const void*>(&PreTickKernel),
        reinterpret_cast<const void*>(&VehicleUpdateKernel),
        reinterpret_cast<const void*>(&CarWorldRowsKernel),
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

struct CarWorldStageBuffers {
    CarWorldMeshCandidateStage* stages = nullptr;
    int32_t* mesh_work = nullptr;        // per mesh, the car slots whose query AABB may touch it
    int32_t* mesh_work_count = nullptr;  // kCarWorldMeshStageMeshes counters
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
            cudaFree(b.mesh_work);
            b.stages = nullptr;
            b.mesh_work = nullptr;
        }
        if (b.mesh_work_count == nullptr) {
            CheckCuda(cudaMalloc(&b.mesh_work_count, kCarWorldMeshStageMeshes * sizeof(int32_t)), "cudaMalloc(car-world mesh work count)");
        }
        b.capacity = 0;
        b.published = nullptr;
        CheckCuda(cudaMalloc(&b.mesh_work, count * kCarWorldMeshStageMeshes * sizeof(int32_t)), "cudaMalloc(car-world mesh work)");
        CheckCuda(cudaMalloc(&b.stages, count * sizeof(CarWorldMeshCandidateStage)), "cudaMalloc(car-world stage)");
        CheckCuda(cudaMemset(b.stages, 0, count * sizeof(CarWorldMeshCandidateStage)), "cudaMemset(car-world stage)");
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
    const size_t car_slots = static_cast<size_t>(cfg.env_count) * static_cast<size_t>(cfg.max_cars_per_env);
    const int block_size = kCudaRayKernelBlockSize;
    CheckCuda(cudaMemsetAsync(b.mesh_work_count, 0, kCarWorldMeshStageMeshes * sizeof(int32_t), stream),
              "cudaMemsetAsync(car-world mesh work count)");
    PrefilterCarWorldMeshCandidatesKernel<<<static_cast<unsigned>((car_slots + block_size - 1) / block_size),
                                            static_cast<unsigned>(block_size), 0, stream>>>(
        cfg, cars, car_configs, env_car_count, b.stages, b.mesh_work, b.mesh_work_count);
    CheckCuda(cudaGetLastError(), "PrefilterCarWorldMeshCandidatesKernel launch");
    const int meshes = std::max(0, std::min(cfg.arena_mesh_bvh_mesh_count, kCarWorldMeshStageMeshes));
    // Lanes per staged pair. Pairs packed densely serialize their divergent GJK work within a warp; that only
    // pays once there are enough pairs (about three per car) to fill the GPU, so small batches spread them out.
    const size_t lanes_to_fill = static_cast<size_t>(CudaSmCount()) * 32 * 32;
    const int spread = static_cast<int>(std::max<size_t>(1, std::min<size_t>(8, lanes_to_fill / std::max<size_t>(1, car_slots * 3))));
    const size_t count = car_slots * static_cast<size_t>(meshes) * static_cast<size_t>(spread);
    if (count == 0) {
        return;
    }
    StageCarWorldMeshCandidatesKernel<<<static_cast<unsigned>((count + block_size - 1) / block_size),
                                        static_cast<unsigned>(block_size), 0, stream>>>(
        cfg, cars, car_configs, b.stages, b.mesh_work, b.mesh_work_count, spread);
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

// Each tick: PreTickKernel, the staged wheel rays and car-world candidates, then StepKernel. controls_by_tick and
// env_commands_by_tick are optional per-tick streams; respawn_slots are the slots the step respawns cars into.
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
    EnsureCudaStepStackLimit();
    constexpr int kPreTickEnvsPerBlock = 32;
    const int pre_block_size = kPreTickEnvsPerBlock * cfg.max_cars_per_env;
    const int pre_blocks = (cfg.env_count + kPreTickEnvsPerBlock - 1) / kPreTickEnvsPerBlock;
    const int block_size = GetCudaStepBlockSize(cfg.env_count);
    const int blocks = (cfg.env_count + block_size - 1) / block_size;
    const int car_blocks = (cfg.env_count * cfg.max_cars_per_env + kCudaRayKernelBlockSize - 1) / kCudaRayKernelBlockSize;
    // A thread per car for the vehicle updates pays only while a thread per env leaves the GPU short of warps;
    // at full occupancy their divergent branches cost what the split saves.
    const bool vehicles_apart = static_cast<long long>(cfg.env_count) * cfg.max_cars_per_env <= CudaSmCount() * 32ll * 8;
    // So do the car-world rows below a quarter of that (their split costs two launches and a saved world step),
    // unless physics events are on: those must come out in env step order.
    const bool world_rows_apart = static_cast<long long>(cfg.env_count) * cfg.max_cars_per_env <= CudaSmCount() * 32ll * 2 &&
                                  !cfg.mutator.physics_events.windows;
    if (cfg.env_count <= 0 || pre_block_size <= 0) {
        return;
    }
    for (int32_t t = 0; t < ticks; ++t) {
        PreTickKernel<<<pre_blocks, pre_block_size, 0, stream>>>(
            cfg, teams, env_car_count, cars, solver_scratch, controls, balls, env_states, pads,
            controls_by_tick, control_stride, env_commands_by_tick, env_command_stride,
            respawn_slots, writable_respawn_slots, respawn_rng, respawn_order, seed, t);
        CheckCuda(cudaGetLastError(), "PreTickKernel launch");
        LaunchStageWheelRays(cfg, cars, car_configs, wheel_runtime_states, balls, env_states, solver_scratch, env_car_count, stream);
        if (vehicles_apart) {
            VehicleUpdateKernel<<<car_blocks, kCudaRayKernelBlockSize, 0, stream>>>(
                cfg, env_car_count, cars, wheel_runtime_states, solver_scratch, controls, car_configs, balls);
            CheckCuda(cudaGetLastError(), "VehicleUpdateKernel launch");
        }
        StepKernel<<<blocks, block_size, 0, stream>>>(
            cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls, car_configs,
            balls, env_states, pads, seed, vehicles_apart, world_rows_apart ? kWorldStepContacts : kWorldStepWhole);
        CheckCuda(cudaGetLastError(), "StepKernel launch");
        if (world_rows_apart) {
            CarWorldRowsKernel<<<car_blocks, kCudaRayKernelBlockSize, 0, stream>>>(
                cfg, env_car_count, cars, car_configs, solver_scratch, row_pool);
            CheckCuda(cudaGetLastError(), "CarWorldRowsKernel launch");
            StepKernel<<<blocks, block_size, 0, stream>>>(
                cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch, row_pool, controls, car_configs,
                balls, env_states, pads, seed, vehicles_apart, kWorldStepRows);
            CheckCuda(cudaGetLastError(), "StepKernel launch");
        }
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
