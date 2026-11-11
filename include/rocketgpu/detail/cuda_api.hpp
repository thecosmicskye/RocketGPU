#pragma once

#include <cstdint>

#include "rocketgpu/rocketgpu.h"
#include "rocketgpu/detail/car_ref.hpp"

namespace rocketgpu {
namespace detail {
struct WheelRuntimeState;
struct CarCarPairCachedContact;
struct WorldSolverScratch;
struct WorldSolverScratchRowPool;
struct RespawnRngState;

#if defined(ROCKETGPU_USE_CUDA)

int32_t AssignArenaMeshSlot(const ArenaMeshBvhMeshInfo* infos, int32_t count);

void LaunchPrepareRespawnSlots(
    const SimConfig& cfg, CarCSpanT cars, const int32_t* counts,
    const EnvState* envs, RespawnRngState* rng, const int32_t* order,
    uint8_t* slots, cudaStream_t stream);

void LaunchResetToKickoff(
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarSpanT cars,
    CarControls* controls,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    uint64_t seed,
    RespawnRngState* respawn_rng,
    uint8_t kickoff_seed_mode,
    const CarConfig* car_configs,
    cudaStream_t stream);

void LaunchApplyEnvCommands(
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarSpanT cars,
    CarControls* controls,
    WorldSolverScratch* solver_scratch,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    const EnvCommand* env_commands,
    int32_t command_count,
    RespawnRngState* respawn_rng,
    const CarConfig* car_configs,
    cudaStream_t stream);

void LaunchCanonicalizeImportedCars(
    CarSpanT cars,
    int32_t count,
    cudaStream_t stream);

void LaunchCanonicalizeImportedBalls(
    BallState* balls,
    int32_t count,
    WorldSolverScratch* solver_scratch,
    const MutatorConfig& mutator,
    float dt,
    cudaStream_t stream);

void LaunchStep(
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarSpanT cars,
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
    RespawnRngState* respawn_rng = nullptr,
    const int32_t* respawn_order = nullptr,
    uint8_t* writable_respawn_slots = nullptr);

void LaunchStepWithControlStream(
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarSpanT cars,
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
    RespawnRngState* respawn_rng = nullptr,
    const int32_t* respawn_order = nullptr,
    uint8_t* respawn_slots = nullptr);

void LaunchStepWithCommandStreams(
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarSpanT cars,
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
    RespawnRngState* respawn_rng = nullptr,
    const int32_t* respawn_order = nullptr,
    uint8_t* respawn_slots = nullptr);

void PrepareCarWorldStage(int env_count, int max_cars_per_env, cudaStream_t stream);

void LaunchZeroTickRows(const WorldSolverScratchRowPool& row_pool, size_t first, size_t count, cudaStream_t stream);

#endif
}
}
