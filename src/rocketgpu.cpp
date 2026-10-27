#include "rocketgpu/rocketgpu.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#if defined(_WIN32) && defined(_MSC_VER) && !defined(__clang__) && defined(_M_X64)
#include <intrin.h>
#endif

#include "rocketgpu/detail/arena/mesh_loader.hpp"
#include "rocketgpu/detail/arena/mesh_path.hpp"
#include "rocketgpu/detail/buffer.hpp"
#include "rocketgpu/detail/cuda_api.hpp"
#include "rocketgpu/detail/respawn_rng.hpp"
#include "rocketgpu/detail/step.hpp"
#include "rocketgpu/detail/native_sse_table.hpp"

namespace {
constexpr uint8_t kNoRespawnSlotOverride = 0xFFu;

bool PlatformBulletFmaSolver() {
#if defined(_WIN32) && defined(_MSC_VER) && !defined(__clang__) && defined(_M_X64)

    int cpu[4] = {};
    __cpuid(cpu, 1);
    constexpr int required = (1 << 12) | (1 << 19) | (1 << 27) | (1 << 28);
    return (cpu[2] & required) == required && (_xgetbv(0) & 6) == 6;
#else
    return false;
#endif
}

void CacheBallDamping(rocketgpu::MutatorConfig& mutator, float dt, rocketgpu::Mode mode) {
    if (mode != rocketgpu::Mode::Soccar) {
        mutator.cached_ball_damping_drag = -1.0f;
        mutator.cached_ball_damping_dt = -1.0f;
        mutator.cached_ball_damping_scale = -1.0f;
        return;
    }
    mutator.cached_ball_damping_drag = mutator.ball_drag;
    mutator.cached_ball_damping_dt = dt;
    mutator.cached_ball_damping_scale = mutator.ball_drag <= 0.0f ? 1.0f
        : ::powf(rocketgpu::Max(0.0f, 1.0f - mutator.ball_drag), dt);
}

#if defined(ROCKETGPU_USE_CUDA)

constexpr rocketgpu::detail::RespawnRngKind PlatformRespawnRngKind() {
    using Engine = std::default_random_engine;
    static_assert(std::is_same_v<Engine, std::minstd_rand0> ||
        std::is_same_v<Engine, std::minstd_rand> || std::is_same_v<Engine, std::mt19937>,
        "Port this standard library's default_random_engine before enabling CUDA respawn mirroring");
    if constexpr (std::is_same_v<Engine, std::mt19937>) return rocketgpu::detail::RespawnRngKind::Mt19937;
    if constexpr (std::is_same_v<Engine, std::minstd_rand>) return rocketgpu::detail::RespawnRngKind::MinStd;
    return rocketgpu::detail::RespawnRngKind::MinStd0;
}

struct RespawnOrderUpload {
    int32_t* data = nullptr;
    cudaEvent_t ready = nullptr;
    cudaStream_t stream = nullptr;
    bool submitted = false;
    bool recorded = false;

    ~RespawnOrderUpload() {
        if (submitted) {
            if (recorded) cudaEventSynchronize(ready);
            else cudaStreamSynchronize(stream);
        }
        if (ready) cudaEventDestroy(ready);
        if (data) cudaFreeHost(data);
    }
    bool complete() {
        if (!recorded) return false;
        const cudaError_t result = cudaEventQuery(ready);
        if (result == cudaErrorNotReady) return false;
        rocketgpu::CheckCuda(result, "cudaEventQuery respawn order upload");
        submitted = false;
        return true;
    }
};
#endif
}

namespace rocketgpu {
namespace {
bool SolverRowsInterleavedDefault() {
#if defined(ROCKETGPU_USE_CUDA)
    return true;
#else
    return false;
#endif
}

float CanonicalizeImportedLinearScalar(float value) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kBTToUU = 50.0f;
    return (value * kUUToBT) * kBTToUU;
}

Vec3 CanonicalizeImportedLinearVec3(const Vec3& value) {
    return Vec3(
        CanonicalizeImportedLinearScalar(value.x),
        CanonicalizeImportedLinearScalar(value.y),
        CanonicalizeImportedLinearScalar(value.z));
}

float ImportedBallLinearScalarBt(float value) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    return CanonicalizeImportedLinearScalar(value) * kUUToBT;
}

Vec3 ImportedBallLinearVec3Bt(const Vec3& value) {
    return Vec3(
        ImportedBallLinearScalarBt(value.x),
        ImportedBallLinearScalarBt(value.y),
        ImportedBallLinearScalarBt(value.z));
}

float ImportedCarLinearScalarBt(float value) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    return CanonicalizeImportedLinearScalar(value) * kUUToBT;
}

Vec3 ImportedCarLinearVec3Bt(const Vec3& value) {
    return Vec3(
        ImportedCarLinearScalarBt(value.x),
        ImportedCarLinearScalarBt(value.y),
        ImportedCarLinearScalarBt(value.z));
}

Vec3 BtVec3ToUuMirror(const Vec3& value_bt) {
    constexpr float kBTToUU = 50.0f;
    return Vec3(value_bt.x * kBTToUU, value_bt.y * kBTToUU, value_bt.z * kBTToUU);
}

void CanonicalizeImportedCarStatesInPlace(CarState* states, size_t count) {
    for (size_t idx = 0; idx < count; ++idx) {
        states[idx].ResetTrainingHistory();
        states[idx].pos_bt = ImportedCarLinearVec3Bt(states[idx].pos);

        states[idx].vel_bt = ImportedCarLinearVec3Bt(states[idx].vel);
        states[idx].pos = BtVec3ToUuMirror(states[idx].pos_bt);
        states[idx].vel = CanonicalizeImportedLinearVec3(states[idx].vel);

        detail::CanonicalizeImportedCarOrientation(states[idx]);
    }
}

void CanonicalizeImportedBallStatesInPlace(BallState* states, size_t count) {
    for (size_t idx = 0; idx < count; ++idx) {
        states[idx].pos_bt = ImportedBallLinearVec3Bt(states[idx].pos);
        states[idx].vel_bt = ImportedBallLinearVec3Bt(states[idx].vel);
        states[idx].pos = BtVec3ToUuMirror(states[idx].pos_bt);
        states[idx].vel = BtVec3ToUuMirror(states[idx].vel_bt);
    }
}
}

void CanonicalizeImportedCarState(CarState& state) {
    CanonicalizeImportedCarStatesInPlace(&state, 1);
}

void CanonicalizeImportedBallState(BallState& state) {
    CanonicalizeImportedBallStatesInPlace(&state, 1);
}

CarConfig OctaneConfig() {
    return CarConfig{};
}

namespace {
CarConfig MakePresetConfig(
    Vec3 hitbox_size, Vec3 hitbox_offset,
    float front_radius, float front_rest, Vec3 front_offset,
    float back_radius, float back_rest, Vec3 back_offset,
    bool three_wheels) {
    CarConfig c{};
    c.hitbox_size = hitbox_size;
    c.hitbox_offset = hitbox_offset;
    c.front_wheels_offset = front_offset;
    c.back_wheels_offset = back_offset;
    c.front_wheel_radius = front_radius;
    c.back_wheel_radius = back_radius;
    c.front_suspension_rest = front_rest;
    c.back_suspension_rest = back_rest;
    c.three_wheels = three_wheels ? 1 : 0;
    return c;
}
}

CarConfig DominusConfig() {
    return MakePresetConfig(Vec3(130.427f, 85.7799f, 33.8f), Vec3(9.0f, 0.0f, 15.75f),
                            12.00f, 33.95f, Vec3(50.30f, 31.10f, 15.75f),
                            13.50f, 33.85f, Vec3(-34.75f, 33.00f, 15.75f), false);
}
CarConfig PlankConfig() {
    return MakePresetConfig(Vec3(131.32f, 87.1704f, 31.8944f), Vec3(9.00857f, 0.0f, 12.0942f),
                            12.50f, 31.9242f, Vec3(49.97f, 27.80f, 12.0942f),
                            17.00f, 27.9242f, Vec3(-35.43f, 20.28f, 12.0942f), false);
}
CarConfig BreakoutConfig() {
    return MakePresetConfig(Vec3(133.992f, 83.021f, 32.8f), Vec3(12.5f, 0.0f, 11.75f),
                            13.50f, 29.7f, Vec3(51.50f, 26.67f, 11.75f),
                            15.00f, 29.666f, Vec3(-35.75f, 35.00f, 11.75f), false);
}
CarConfig HybridConfig() {
    return MakePresetConfig(Vec3(129.519f, 84.6879f, 36.6591f), Vec3(13.8757f, 0.0f, 20.755f),
                            12.50f, 38.755f, Vec3(51.25f, 25.90f, 20.755f),
                            15.00f, 37.055f, Vec3(-34.00f, 29.50f, 20.755f), false);
}
CarConfig MercConfig() {
    return MakePresetConfig(Vec3(123.22f, 79.2103f, 44.1591f), Vec3(11.3757f, 0.0f, 21.505f),
                            15.00f, 39.505f, Vec3(51.25f, 25.90f, 21.505f),
                            15.00f, 39.105f, Vec3(-33.75f, 29.50f, 21.505f), false);
}
CarConfig PsyclopsConfig() {
    return MakePresetConfig(Vec3(120.507f + 0.134f, 86.6994f + 0.134f, 38.6591f + 0.134f), Vec3(13.8757f, 0.0f, 15.0f),
                            12.50f, 33.0f, Vec3(51.25f, 5.000f, 15.000f),
                            15.00f, 31.3f, Vec3(-33.75f, 29.50f, 15.000f), true);
}

struct BatchSim::Impl {
    explicit Impl(const SimConfig& in_cfg) : cfg(in_cfg) {
        cfg.mutator.physics_events = {};
        cfg.mutator.native_sphere_rsqrt_table = nullptr;
        cfg.mutator.native_sphere_rsqrt_shift = 0;
        if (cfg.native_sphere_normalization) {
            const auto& native = detail::NativeSseHostTable();
            native_sphere_rsqrt.resize(native.values.size());
            native_sphere_rsqrt.copy_from_host(native.values.data(), native.values.size());
            cfg.mutator.native_sphere_rsqrt_table = native_sphere_rsqrt.data();
            cfg.mutator.native_sphere_rsqrt_shift = native.shift;
        }
        cfg.mutator.game_mode = static_cast<uint8_t>(cfg.mode);
        cfg.mutator.ball_no_rot = cfg.no_ball_rot;
        cfg.mutator.cal_ball_world = cfg.mode != Mode::Snowday ? 1 : 0;
        if (cfg.mode == Mode::TheVoid) {
            cfg.mutator.arena_collision = 0;
            cfg.enable_boost_pads = 0;
            cfg.enable_scoring = 0;
        } else if (cfg.mode == Mode::Snowday) {
            cfg.mutator.arena_collision = 1;

            const MutatorConfig soccar_defaults{};
            if (cfg.mutator.ball_radius == soccar_defaults.ball_radius) cfg.mutator.ball_radius = detail::SnowdayPuckRadiusUu();
            if (cfg.mutator.ball_mass == soccar_defaults.ball_mass) cfg.mutator.ball_mass = detail::SnowdayPuckMassBt();
            if (cfg.mutator.ball_friction == soccar_defaults.ball_friction) cfg.mutator.ball_friction = detail::SnowdayPuckFriction();
            if (cfg.mutator.ball_restitution == soccar_defaults.ball_restitution) cfg.mutator.ball_restitution = detail::SnowdayPuckRestitution();
            cfg.mutator.ball_shape = 1;
            detail::BuildSnowdayPuckPointsBt(cfg.mutator.ball_radius, cfg.mutator.puck_points_bt);
            cfg.mutator.ball_local_inertia_bt = detail::ComputeSnowdayPuckLocalInertiaBt(cfg.mutator.puck_points_bt, cfg.mutator.ball_mass);
            cfg.mutator.ball_contact_breaking_threshold_bt = detail::ComputeSnowdayPuckContactBreakingThresholdBt(cfg.mutator.puck_points_bt);
        } else if (cfg.mode != Mode::Soccar && cfg.mode != Mode::Heatseeker) {
            throw std::invalid_argument("BatchSim supports Soccar, Heatseeker, Snowday and TheVoid modes (Hoops/Dropshot are blocked on their arena meshes)");
        } else {
            cfg.mutator.arena_collision = 1;
        }
        if (cfg.env_count <= 0) {
            throw std::invalid_argument("env_count must be > 0");
        }
        if (cfg.max_cars_per_env <= 0) {
            throw std::invalid_argument("max_cars_per_env must be > 0");
        }
        CacheBallDamping(cfg.mutator, cfg.tick_dt, cfg.mode);
        cfg.mutator.solver_use_fma = cfg.mode == Mode::Soccar && PlatformBulletFmaSolver();

        if (!cfg.use_custom_boost_pads) {
            cfg.boost_pad_count = detail::BoostPadCountSoccar();
            for (int p = 0; p < cfg.boost_pad_count; ++p) {
                cfg.boost_pads[p].pos = detail::BoostPadLocSoccar(p);
                cfg.boost_pads[p].is_big = detail::BoostPadIsBigSoccar(p);
            }
        } else if (cfg.boost_pad_count < 0 || cfg.boost_pad_count > kMaxBoostPads) {
            throw std::invalid_argument("boost_pad_count must be within [0, kMaxBoostPads]");
        }

        const size_t car_capacity = static_cast<size_t>(cfg.env_count) * static_cast<size_t>(cfg.max_cars_per_env);
        const size_t pad_capacity = static_cast<size_t>(cfg.env_count) * static_cast<size_t>(cfg.boost_pad_count);

        car_states.resize(car_capacity);
        wheel_runtime_states.resize(car_capacity);
        solver_scratch.resize(static_cast<size_t>(cfg.env_count));
#if defined(ROCKETGPU_USE_CUDA)

        detail::PrepareCarWorldStage(cfg.env_count, cfg.max_cars_per_env, nullptr);
#endif

        allocate_solver_rows();
        respawn_slot_overrides.resize(car_capacity);
        controls.resize(car_capacity);
        car_configs.resize(car_capacity);
        teams.resize(car_capacity);
        ball_states.resize(static_cast<size_t>(cfg.env_count));
        env_states.resize(static_cast<size_t>(cfg.env_count));
        boost_pad_states.resize(pad_capacity);
        env_car_count.resize(static_cast<size_t>(cfg.env_count));
        env_command_scratch.resize(static_cast<size_t>(cfg.env_count));

        std::vector<CarState> host_cars(car_capacity);
        std::vector<detail::WheelRuntimeState> host_wheel_runtime(car_capacity);
        std::vector<CarControls> host_controls(car_capacity);
        std::vector<CarConfig> host_configs(car_capacity, cfg.default_car);
        std::vector<Team> host_teams(car_capacity, Team::Blue);
        std::vector<BallState> host_balls(static_cast<size_t>(cfg.env_count));
        std::vector<EnvState> host_envs(static_cast<size_t>(cfg.env_count));
        std::vector<BoostPadState> host_pads(pad_capacity);
        std::vector<int32_t> host_counts(static_cast<size_t>(cfg.env_count), 0);
        std::vector<uint8_t> host_respawn_overrides(car_capacity, kNoRespawnSlotOverride);

        car_states.copy_from_host(host_cars.data(), host_cars.size());
        wheel_runtime_states.copy_from_host(host_wheel_runtime.data(), host_wheel_runtime.size());
        controls.copy_from_host(host_controls.data(), host_controls.size());
        respawn_slot_overrides.copy_from_host(host_respawn_overrides.data(), host_respawn_overrides.size());
        car_configs.copy_from_host(host_configs.data(), host_configs.size());
        teams.copy_from_host(host_teams.data(), host_teams.size());
        ball_states.copy_from_host(host_balls.data(), host_balls.size());
        env_states.copy_from_host(host_envs.data(), host_envs.size());
        boost_pad_states.copy_from_host(host_pads.data(), host_pads.size());
        env_car_count.copy_from_host(host_counts.data(), host_counts.size());
        env_step_order.resize(static_cast<size_t>(cfg.env_count));
        rebuild_env_step_order(nullptr);

        if (cfg.arena_mesh_tris == nullptr || cfg.arena_mesh_tri_count <= 0) {
            std::vector<detail::ArenaMeshBvhQuantization> host_mesh_quantization;
            std::vector<detail::ArenaMeshBvhMeshInfo> host_mesh_infos;
            std::vector<detail::ArenaMeshBvhNode> host_mesh_nodes;
            std::vector<detail::ArenaMeshBvhSubtreeInfo> host_mesh_subtrees;

            const bool soccar_arena = cfg.mode == Mode::Soccar || cfg.mode == Mode::Snowday || cfg.mode == Mode::Heatseeker;
            if (soccar_arena && !detail::HasGoalMeshes()) {
                throw std::runtime_error("soccar arenas need the game's local goal meshes in <collision mesh dir>/soccar_goals/");
            }
            std::vector<detail::ArenaMeshTriangle> host_local_mesh;
            int local_goal_meshes = 0;
            std::vector<detail::ArenaMeshTriangle> host_mesh = LoadArenaMeshTriangles(
                &host_mesh_quantization,
                &host_mesh_infos,
                &host_mesh_nodes,
                &host_mesh_subtrees,
                soccar_arena ? &host_local_mesh : nullptr,
                &local_goal_meshes);
            if (soccar_arena) {
                if (local_goal_meshes == 0 || host_local_mesh.size() != host_mesh.size()) {
                    throw std::runtime_error("no local goal mesh in soccar_goals/ matches the arena meshes");
                }
                arena_mesh_local_triangles.resize(host_local_mesh.size());
                arena_mesh_local_triangles.copy_from_host(host_local_mesh.data(), host_local_mesh.size());
                cfg.mutator.cal_arena_local_tris = arena_mesh_local_triangles.data();
            }
            if (!host_mesh.empty()) {
                arena_mesh_triangles.resize(host_mesh.size());
                arena_mesh_triangles.copy_from_host(host_mesh.data(), host_mesh.size());
                cfg.arena_mesh_tris = arena_mesh_triangles.data();
                cfg.arena_mesh_tri_count = static_cast<int32_t>(arena_mesh_triangles.size());
                arena_mesh_bvh_quantization.resize(host_mesh_quantization.size());
                arena_mesh_bvh_quantization.copy_from_host(host_mesh_quantization.data(), host_mesh_quantization.size());
                cfg.arena_mesh_bvh_quantization = arena_mesh_bvh_quantization.data();
                cfg.arena_mesh_bvh_quantization_count = static_cast<int32_t>(arena_mesh_bvh_quantization.size());
                arena_mesh_bvh_meshes.resize(host_mesh_infos.size());
                arena_mesh_bvh_meshes.copy_from_host(host_mesh_infos.data(), host_mesh_infos.size());
                cfg.arena_mesh_bvh_meshes = arena_mesh_bvh_meshes.data();
                cfg.arena_mesh_bvh_mesh_count = static_cast<int32_t>(arena_mesh_bvh_meshes.size());
                arena_mesh_bvh_nodes.resize(host_mesh_nodes.size());
                arena_mesh_bvh_nodes.copy_from_host(host_mesh_nodes.data(), host_mesh_nodes.size());
                cfg.arena_mesh_bvh_nodes = arena_mesh_bvh_nodes.data();
                cfg.arena_mesh_bvh_node_count = static_cast<int32_t>(arena_mesh_bvh_nodes.size());
                arena_mesh_bvh_subtrees.resize(host_mesh_subtrees.size());
                arena_mesh_bvh_subtrees.copy_from_host(host_mesh_subtrees.data(), host_mesh_subtrees.size());
                cfg.arena_mesh_bvh_subtrees = arena_mesh_bvh_subtrees.data();
                cfg.arena_mesh_bvh_subtree_count = static_cast<int32_t>(arena_mesh_bvh_subtrees.size());

                std::vector<detail::ArenaMeshLeafGridInfo> host_leaf_grid_infos;
                std::vector<int32_t> host_leaf_cell_start;
                std::vector<detail::ArenaMeshLeafEntry> host_leaf_entries;
                detail::BuildArenaMeshLeafGrid(host_mesh_quantization, host_mesh_infos, host_mesh_nodes,
                                               host_mesh_subtrees, host_leaf_grid_infos, host_leaf_cell_start,
                                               host_leaf_entries);
                if (!host_leaf_entries.empty()) {
                    arena_mesh_leaf_grid_infos.resize(host_leaf_grid_infos.size());
                    arena_mesh_leaf_grid_infos.copy_from_host(host_leaf_grid_infos.data(), host_leaf_grid_infos.size());
                    arena_mesh_leaf_cell_start.resize(host_leaf_cell_start.size());
                    arena_mesh_leaf_cell_start.copy_from_host(host_leaf_cell_start.data(), host_leaf_cell_start.size());
                    arena_mesh_leaf_entries.resize(host_leaf_entries.size());
                    arena_mesh_leaf_entries.copy_from_host(host_leaf_entries.data(), host_leaf_entries.size());
                    cfg.arena_mesh_leaf_grid_infos = arena_mesh_leaf_grid_infos.data();
                    cfg.arena_mesh_leaf_cell_start = arena_mesh_leaf_cell_start.data();
                    cfg.arena_mesh_leaf_entries = arena_mesh_leaf_entries.data();
                }

                detail::ArenaBroadphaseGrid host_grid{};
                const std::vector<uint16_t> host_cell_mask =
                    detail::BuildArenaBroadphaseCellMask(host_mesh, host_mesh_infos, host_grid);
                if (!host_cell_mask.empty()) {
                    arena_broadphase_cell_mask.resize(host_cell_mask.size());
                    arena_broadphase_cell_mask.copy_from_host(host_cell_mask.data(), host_cell_mask.size());
                    cfg.arena_broadphase_cell_mask = arena_broadphase_cell_mask.data();
                    cfg.arena_broadphase_cell_count = static_cast<int32_t>(host_cell_mask.size());
                    cfg.arena_broadphase_min_bt = host_grid.min_bt;
                    cfg.arena_broadphase_max_bt = host_grid.max_bt;
                    cfg.arena_broadphase_cell_size_bt = host_grid.cell_size_bt;
                    cfg.arena_broadphase_cells_x = host_grid.cells_x;
                    cfg.arena_broadphase_cells_y = host_grid.cells_y;
                    cfg.arena_broadphase_cells_z = host_grid.cells_z;
                }
            }
        }

        if (cfg.mirror_official_respawn_rng) {
            respawn_rng = std::default_random_engine(static_cast<unsigned int>(cfg.respawn_rng_seed));
            respawn_rng_env.clear();
            respawn_rng_env.reserve(static_cast<size_t>(cfg.env_count));
            for (int env_idx = 0; env_idx < cfg.env_count; ++env_idx) {
                respawn_rng_env.emplace_back(RespawnRngSeedForEnv(cfg.respawn_rng_seed, env_idx));
            }
#if defined(ROCKETGPU_USE_CUDA)
            device_respawn_rng.resize(static_cast<size_t>(cfg.env_count));
            std::vector<detail::RespawnRngState> states(static_cast<size_t>(cfg.env_count));
            for (int env = 0; env < cfg.env_count; ++env) {
                detail::SeedRespawnRng(states[env], PlatformRespawnRngKind(),
                    RespawnRngSeedForEnv(cfg.respawn_rng_seed, env));
            }
            device_respawn_rng.copy_from_host(states.data(), states.size());

            CheckCuda(cudaStreamSynchronize(nullptr), "initialize respawn RNG");
            device_respawn_order.resize(car_capacity);
            host_respawn_order.resize(car_capacity, -1);
#else
            host_respawn_slot_overrides.assign(car_capacity, kNoRespawnSlotOverride);
            host_respawn_cars.resize(car_capacity);
            host_respawn_counts.resize(static_cast<size_t>(cfg.env_count), 0);
#endif
        }

        reset_to_kickoff(0, nullptr);
    }

    SimConfig cfg;
    uint64_t seed = 0;

    detail::Buffer<detail::ArenaMeshTriangle> arena_mesh_triangles;
    detail::Buffer<detail::ArenaMeshTriangle> arena_mesh_local_triangles;
    detail::Buffer<detail::ArenaMeshBvhQuantization> arena_mesh_bvh_quantization;
    detail::Buffer<detail::ArenaMeshBvhMeshInfo> arena_mesh_bvh_meshes;
    detail::Buffer<detail::ArenaMeshBvhNode> arena_mesh_bvh_nodes;
    detail::Buffer<detail::ArenaMeshBvhSubtreeInfo> arena_mesh_bvh_subtrees;
    detail::Buffer<detail::ArenaMeshLeafGridInfo> arena_mesh_leaf_grid_infos;
    detail::Buffer<int32_t> arena_mesh_leaf_cell_start;
    detail::Buffer<detail::ArenaMeshLeafEntry> arena_mesh_leaf_entries;
    detail::Buffer<uint16_t> arena_broadphase_cell_mask;
    detail::Buffer<uint32_t> native_sphere_rsqrt;
    detail::Buffer<CarState> car_states;
    detail::Buffer<detail::WheelRuntimeState> wheel_runtime_states;
    detail::Buffer<detail::WorldSolverScratch> solver_scratch;

    detail::Buffer<detail::BallSplitContactRow> rows_ball_split;
    detail::Buffer<detail::BallVelocityContactRow> rows_ball_velocity;
    detail::Buffer<detail::BallFrictionContactRow> rows_ball_friction;
    detail::Buffer<detail::CarSharedSplitContactRow> rows_pair_split;
    detail::Buffer<detail::CarSharedSplitContactRow> rows_world_split;
    detail::Buffer<detail::CarSharedVelocityContactRow> rows_pair_velocity;
    detail::Buffer<detail::CarSharedFrictionContactRow> rows_pair_friction;
    detail::Buffer<detail::CarSharedVelocityContactRow> rows_world_velocity;
    detail::Buffer<detail::CarSharedFrictionContactRow> rows_world_friction;
    detail::Buffer<detail::SharedContactOrderGroup> rows_shared_contact_order_groups;
    detail::Buffer<detail::BulletManifoldEntry> rows_bullet_manifolds;
    detail::Buffer<detail::ContactSortScratch> contact_sort_scratch;
    detail::WorldSolverScratchRowPool solver_row_pool;
    detail::Buffer<uint8_t> respawn_slot_overrides;
    detail::Buffer<CarControls> controls;
    detail::Buffer<CarConfig> car_configs;
    detail::Buffer<Team> teams;
    detail::Buffer<BallState> ball_states;
    detail::Buffer<EnvState> env_states;
    detail::Buffer<BoostPadState> boost_pad_states;
    detail::Buffer<int32_t> env_car_count;
    detail::Buffer<int32_t> env_step_order;
    detail::Buffer<EnvCommand> env_command_scratch;

    // Group envs by car count so a warp of the step kernels runs one per-car loop trip count (a random
    // 1v1/2v2/3v3 mix idles most lanes). Each env's physics is independent, so the order can't change results.
    // Below 128 envs a tick is a few lone warps' latency and a warp of only 3v3s is slower than a mixed one
    // (64 envs: -9% grouped), so keep identity order there.
    void rebuild_env_step_order(cudaStream_t stream) {
        const size_t n = env_car_count.size();
        if (n < 128) {
            cfg.env_step_order = nullptr;
            return;
        }
        std::vector<int32_t> counts(n);
        env_car_count.copy_to_host(counts.data(), n, stream);
#if defined(ROCKETGPU_USE_CUDA)
        CheckCuda(cudaStreamSynchronize(stream), "env step order counts");
#endif
        std::vector<int32_t> order(n);
        for (size_t i = 0; i < n; ++i) order[i] = static_cast<int32_t>(i);
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return counts[a] > counts[b]; });
        env_step_order.copy_from_host(order.data(), n, stream);
#if defined(ROCKETGPU_USE_CUDA)
        CheckCuda(cudaStreamSynchronize(stream), "env step order upload");
#endif
        cfg.env_step_order = env_step_order.data();
    }

    std::default_random_engine respawn_rng;

    std::vector<std::default_random_engine> respawn_rng_env;
    std::vector<CarState> host_respawn_cars;
    std::vector<int32_t> host_respawn_counts;
    std::vector<uint8_t> host_respawn_slot_overrides;
    std::vector<EnvState> host_respawn_envs;

#if defined(ROCKETGPU_USE_CUDA)
    detail::Buffer<detail::RespawnRngState> device_respawn_rng;
    detail::Buffer<int32_t> device_respawn_order;
    std::vector<int32_t> host_respawn_order;
    std::vector<int32_t> uploaded_respawn_order;
    std::vector<std::unique_ptr<RespawnOrderUpload>> respawn_order_uploads;

    const int32_t* prepare_respawn_order(cudaStream_t stream) {
        cudaStreamCaptureStatus capture;
        CheckCuda(cudaStreamIsCapturing(stream, &capture), "query respawn order capture");
        if (capture == cudaStreamCaptureStatusNone) {
            respawn_order_uploads.erase(std::remove_if(respawn_order_uploads.begin(), respawn_order_uploads.end(),
                [](const auto& upload) { return upload->complete(); }), respawn_order_uploads.end());
        }
        return nullptr;
    }
#endif

    bool prepare_respawn_slot_overrides(cudaStream_t stream, const int32_t* device_order = nullptr) {
        if (!cfg.mirror_official_respawn_rng) {
            return false;
        }

#if defined(ROCKETGPU_USE_CUDA)
        detail::LaunchPrepareRespawnSlots(cfg, car_states.data(), env_car_count.data(), env_states.data(),
            device_respawn_rng.data(), device_order, respawn_slot_overrides.data(), stream);
        return true;
#else
        (void)device_order;
        if (host_respawn_slot_overrides.empty()) {
            const size_t car_capacity = static_cast<size_t>(cfg.env_count) * static_cast<size_t>(cfg.max_cars_per_env);
            host_respawn_slot_overrides.assign(car_capacity, kNoRespawnSlotOverride);
            host_respawn_cars.resize(car_capacity);
            host_respawn_counts.resize(static_cast<size_t>(cfg.env_count), 0);
        }

        car_states.copy_to_host(host_respawn_cars.data(), host_respawn_cars.size(), stream);
        env_car_count.copy_to_host(host_respawn_counts.data(), host_respawn_counts.size(), stream);
        if (host_respawn_envs.size() != static_cast<size_t>(cfg.env_count)) {
            host_respawn_envs.resize(static_cast<size_t>(cfg.env_count));
        }
        env_states.copy_to_host(host_respawn_envs.data(), host_respawn_envs.size(), stream);
        std::fill(host_respawn_slot_overrides.begin(), host_respawn_slot_overrides.end(), kNoRespawnSlotOverride);

        for (int env_idx = 0; env_idx < cfg.env_count; ++env_idx) {
            const int base = env_idx * cfg.max_cars_per_env;
            const int count = host_respawn_counts[static_cast<size_t>(env_idx)];
            const int32_t* official_order = nullptr;
            const bool have_official_order =
                detail::TryGetHostValidationRespawnOrder(env_idx, official_order);
            for (int order_idx = 0; order_idx < count; ++order_idx) {
                const int i = (have_official_order && official_order[order_idx] >= 0 && official_order[order_idx] < count)
                    ? official_order[order_idx]
                    : order_idx;
                const CarState& car = host_respawn_cars[static_cast<size_t>(base + i)];
                if (!car.is_demoed) {
                    continue;
                }
                if (Max(0.0f, car.demo_respawn_timer - cfg.tick_dt) > 0.0f) {
                    continue;
                }
                host_respawn_slot_overrides[static_cast<size_t>(base + i)] =
                    static_cast<uint8_t>(respawn_rng_env[static_cast<size_t>(env_idx)]() % 4);
            }
        }

        respawn_slot_overrides.copy_from_host(
            host_respawn_slot_overrides.data(),
            host_respawn_slot_overrides.size(),
            stream);
        return true;
#endif
    }

    bool exact_host_validation = false;
    void mark_exact_host_dirty() {}
    void ensure_official_synced() {}
    void pull_all_official_to_buffers() {}
    void apply_exact_host_command(int, const EnvCommand&) {}
    void step_exact_host(int32_t) {}
    void step_exact_host_with_control_stream(const CarControls*, int32_t, int32_t) {}
    void step_exact_host_with_command_streams(const CarControls*, int32_t, const EnvCommand*, int32_t, int32_t) {}

    void sync_exact_host_for_download() {
    }

    DeviceView view() const noexcept {
        DeviceView v;
        v.car_states = const_cast<CarState*>(car_states.data());
        v.controls = const_cast<CarControls*>(controls.data());
        v.car_configs = const_cast<CarConfig*>(car_configs.data());
        v.teams = const_cast<Team*>(teams.data());
        v.ball_states = const_cast<BallState*>(ball_states.data());
        v.env_states = const_cast<EnvState*>(env_states.data());
        v.boost_pad_states = const_cast<BoostPadState*>(boost_pad_states.data());
        v.env_car_count = const_cast<int32_t*>(env_car_count.data());
        return v;
    }

#if !defined(ROCKETGPU_USE_CUDA)

    bool host_kickoff_order(int env_idx, uint64_t kickoff_seed, uint8_t kickoff_seed_mode, int* order) {
        if (!cfg.mirror_official_respawn_rng || static_cast<size_t>(env_idx) >= respawn_rng_env.size()) {
            return false;
        }
        for (int i = 0; i < 5; ++i) order[i] = i;
        const uint32_t engine_seed = kickoff_seed_mode == 1
            ? static_cast<uint32_t>(kickoff_seed)
            : detail::MixSeed(kickoff_seed, env_idx, 0x6b1d4f2a);
        if (kickoff_seed_mode == 2 || static_cast<int32_t>(engine_seed) == -1) {
            std::shuffle(order, order + 5, respawn_rng_env[static_cast<size_t>(env_idx)]);
        } else {
            std::default_random_engine engine(engine_seed);
            std::shuffle(order, order + 5, engine);
        }
        return true;
    }
#endif

    void allocate_solver_rows() {
        const size_t n = static_cast<size_t>(cfg.env_count);
        rows_ball_split.resize(n * detail::kMaxBallSplitContactRows);
        rows_ball_velocity.resize(n * detail::kMaxBallVelocityContactRows);
        rows_ball_friction.resize(n * detail::kMaxBallFrictionContactRows);
        rows_pair_split.resize(n * detail::kMaxCarPairSplitContactRows);
        rows_world_split.resize(n * detail::kMaxCarSharedSplitContactRows);
        rows_pair_velocity.resize(n * detail::kMaxCarPairSharedVelocityContactRows);
        rows_pair_friction.resize(n * detail::kMaxCarPairSharedFrictionContactRows);
        rows_world_velocity.resize(n * detail::kMaxCarWorldSharedVelocityContactRows);
        rows_world_friction.resize(n * detail::kMaxCarWorldSharedFrictionContactRows);
        rows_shared_contact_order_groups.resize(n * detail::kMaxSharedContactOrderGroups);
        rows_bullet_manifolds.resize(n * detail::kMaxBulletManifoldEntries);
        solver_row_pool = detail::WorldSolverScratchRowPool{};
#if defined(ROCKETGPU_USE_CUDA)
        contact_sort_scratch.resize(n);
        solver_row_pool.contact_sort_scratch = contact_sort_scratch.data();
        solver_row_pool.contact_sort_stride = sizeof(detail::ContactSortScratch);
#endif
        solver_row_pool.ball_split_rows = rows_ball_split.data();
        solver_row_pool.ball_velocity_rows = rows_ball_velocity.data();
        solver_row_pool.ball_friction_rows = rows_ball_friction.data();
        solver_row_pool.pair_split_rows = rows_pair_split.data();
        solver_row_pool.world_split_rows = rows_world_split.data();
        solver_row_pool.pair_velocity_rows = rows_pair_velocity.data();
        solver_row_pool.pair_friction_rows = rows_pair_friction.data();
        solver_row_pool.world_velocity_rows = rows_world_velocity.data();
        solver_row_pool.world_friction_rows = rows_world_friction.data();
        solver_row_pool.shared_contact_order_groups = rows_shared_contact_order_groups.data();
        solver_row_pool.bullet_manifolds = rows_bullet_manifolds.data();
        solver_row_pool.env_count = cfg.env_count;
        solver_row_pool.interleaved = SolverRowsInterleavedDefault() ? 1 : 0;
    }

    void memset_zero_solver_rows(cudaStream_t stream) {
        rows_ball_split.memset_zero(stream);
        rows_ball_velocity.memset_zero(stream);
        rows_ball_friction.memset_zero(stream);
        rows_pair_split.memset_zero(stream);
        rows_world_split.memset_zero(stream);
        rows_pair_velocity.memset_zero(stream);
        rows_pair_friction.memset_zero(stream);
        rows_world_velocity.memset_zero(stream);
        rows_world_friction.memset_zero(stream);
        rows_shared_contact_order_groups.memset_zero(stream);
        rows_bullet_manifolds.memset_zero(stream);
    }

    void copy_solver_rows_from(const Impl& source, cudaStream_t stream) {
        if (source.solver_row_pool.interleaved != solver_row_pool.interleaved ||
            source.solver_row_pool.env_count != solver_row_pool.env_count) {
            throw std::invalid_argument("copy_solver_rows_from: row pool layout mismatch");
        }
        rows_ball_split.copy_from_buffer(source.rows_ball_split, rows_ball_split.size(), stream);
        rows_ball_velocity.copy_from_buffer(source.rows_ball_velocity, rows_ball_velocity.size(), stream);
        rows_ball_friction.copy_from_buffer(source.rows_ball_friction, rows_ball_friction.size(), stream);
        rows_pair_split.copy_from_buffer(source.rows_pair_split, rows_pair_split.size(), stream);
        rows_world_split.copy_from_buffer(source.rows_world_split, rows_world_split.size(), stream);
        rows_pair_velocity.copy_from_buffer(source.rows_pair_velocity, rows_pair_velocity.size(), stream);
        rows_pair_friction.copy_from_buffer(source.rows_pair_friction, rows_pair_friction.size(), stream);
        rows_world_velocity.copy_from_buffer(source.rows_world_velocity, rows_world_velocity.size(), stream);
        rows_world_friction.copy_from_buffer(source.rows_world_friction, rows_world_friction.size(), stream);
        rows_shared_contact_order_groups.copy_from_buffer(source.rows_shared_contact_order_groups, rows_shared_contact_order_groups.size(), stream);
        rows_bullet_manifolds.copy_from_buffer(source.rows_bullet_manifolds, rows_bullet_manifolds.size(), stream);
    }

    template <typename T>
    void download_rows_one(const detail::Buffer<T>& src, T* dst, int max_rows, size_t first, size_t count, cudaStream_t stream) const {
        const size_t n = static_cast<size_t>(cfg.env_count);
        if (solver_row_pool.interleaved) {
#if defined(ROCKETGPU_USE_CUDA)
            CheckCuda(cudaMemcpy2DAsync(
                dst, count * sizeof(T),
                src.data() + first, n * sizeof(T),
                count * sizeof(T), static_cast<size_t>(max_rows),
                cudaMemcpyDeviceToHost, stream), "cudaMemcpy2DAsync D2H rows");
#else
            for (int i = 0; i < max_rows; ++i) {
                std::copy_n(src.data() + static_cast<size_t>(i) * n + first, count, dst + static_cast<size_t>(i) * count);
            }
            (void)stream;
#endif
        } else {
            src.copy_range_to_host(dst, first * static_cast<size_t>(max_rows), count * static_cast<size_t>(max_rows), stream);
        }
    }

    void download_solver_rows(detail::WorldSolverScratchRowPoolHost& out, size_t first, size_t count, cudaStream_t stream) const {
        out.reset(static_cast<int>(count), solver_row_pool.interleaved != 0);
        if (count == 0) {
            return;
        }
        download_rows_one(rows_ball_split, out.ball_split_rows.data(), detail::kMaxBallSplitContactRows, first, count, stream);
        download_rows_one(rows_ball_velocity, out.ball_velocity_rows.data(), detail::kMaxBallVelocityContactRows, first, count, stream);
        download_rows_one(rows_ball_friction, out.ball_friction_rows.data(), detail::kMaxBallFrictionContactRows, first, count, stream);
        download_rows_one(rows_pair_split, out.pair_split_rows.data(), detail::kMaxCarPairSplitContactRows, first, count, stream);
        download_rows_one(rows_world_split, out.world_split_rows.data(), detail::kMaxCarSharedSplitContactRows, first, count, stream);
        download_rows_one(rows_pair_velocity, out.pair_velocity_rows.data(), detail::kMaxCarPairSharedVelocityContactRows, first, count, stream);
        download_rows_one(rows_pair_friction, out.pair_friction_rows.data(), detail::kMaxCarPairSharedFrictionContactRows, first, count, stream);
        download_rows_one(rows_world_velocity, out.world_velocity_rows.data(), detail::kMaxCarWorldSharedVelocityContactRows, first, count, stream);
        download_rows_one(rows_world_friction, out.world_friction_rows.data(), detail::kMaxCarWorldSharedFrictionContactRows, first, count, stream);
        download_rows_one(rows_shared_contact_order_groups, out.shared_contact_order_groups.data(), detail::kMaxSharedContactOrderGroups, first, count, stream);
        download_rows_one(rows_bullet_manifolds, out.bullet_manifolds.data(), detail::kMaxBulletManifoldEntries, first, count, stream);
    }

    void reset_to_kickoff(uint64_t new_seed, cudaStream_t stream) {
        seed = new_seed;
        solver_scratch.memset_zero(stream);
        memset_zero_solver_rows(stream);
#if defined(ROCKETGPU_USE_CUDA)
        detail::LaunchResetToKickoff(
            cfg,
            teams.data(),
            env_car_count.data(),
            car_states.data(),
            controls.data(),
            ball_states.data(),
            env_states.data(),
            boost_pad_states.data(),
            seed,
            cfg.mirror_official_respawn_rng ? device_respawn_rng.data() : nullptr,
            static_cast<uint8_t>(0),
            stream);
#else
        {
            for (int env_idx = 0; env_idx < cfg.env_count; ++env_idx) {
                int kickoff_order[5];
                const bool have_order = host_kickoff_order(env_idx, seed, 0, kickoff_order);
                detail::ResetEnvToKickoff(
                    env_idx,
                    cfg,
                    teams.data(),
                    env_car_count.data(),
                    car_states.data(),
                    controls.data(),
                    ball_states.data(),
                    boost_pad_states.data(),
                    seed,
                    false,
                    env_states.data(),
                    0,
                    nullptr,
                    have_order ? kickoff_order : nullptr);
            }
        }
        (void)stream;
#endif
    }

    void apply_env_commands_host(const EnvCommand* host_commands, size_t count, cudaStream_t stream) {
        if (host_commands == nullptr) {
            throw std::invalid_argument("host_commands must not be null");
        }
        if (count > static_cast<size_t>(cfg.env_count)) {
            throw std::out_of_range("apply_env_commands_host count exceeds env_count");
        }
#if defined(ROCKETGPU_USE_CUDA)
        env_command_scratch.copy_from_host(host_commands, count, stream);
        detail::LaunchApplyEnvCommands(
            cfg,
            teams.data(),
            env_car_count.data(),
            car_states.data(),
            controls.data(),
            solver_scratch.data(),
            ball_states.data(),
            env_states.data(),
            boost_pad_states.data(),
            env_command_scratch.data(),
            static_cast<int32_t>(count),
            cfg.mirror_official_respawn_rng ? device_respawn_rng.data() : nullptr,
            stream);
#else
        {
            for (size_t env_idx = 0; env_idx < count; ++env_idx) {
                int kickoff_order[5];
                const bool have_order = host_commands[env_idx].reset_to_kickoff &&
                    host_kickoff_order(static_cast<int>(env_idx), host_commands[env_idx].kickoff_seed,
                                       host_commands[env_idx].kickoff_seed_mode, kickoff_order);
                detail::ApplyEnvCommand(
                    static_cast<int>(env_idx),
                    cfg,
                    teams.data(),
                    env_car_count.data(),
                    car_states.data(),
                    controls.data(),
                    ball_states.data(),
                    env_states.data(),
                    boost_pad_states.data(),
                    host_commands[env_idx],
                    nullptr,
                    have_order ? kickoff_order : nullptr);
                if (host_commands[env_idx].reset_to_kickoff) {
                    solver_scratch.data()[env_idx] = detail::WorldSolverScratch{};
                    detail::ZeroWorldSolverScratchRows(solver_row_pool.view(env_idx));
                }
            }
        }
        (void)stream;
#endif
    }

    void apply_env_commands_device(const EnvCommand* device_commands, size_t count, cudaStream_t stream) {
        if (device_commands == nullptr) {
            throw std::invalid_argument("device_commands must not be null");
        }
        if (count > static_cast<size_t>(cfg.env_count)) {
            throw std::out_of_range("apply_env_commands_device count exceeds env_count");
        }
#if defined(ROCKETGPU_USE_CUDA)
        detail::LaunchApplyEnvCommands(
            cfg,
            teams.data(),
            env_car_count.data(),
            car_states.data(),
            controls.data(),
            solver_scratch.data(),
            ball_states.data(),
            env_states.data(),
            boost_pad_states.data(),
            device_commands,
            static_cast<int32_t>(count),
            cfg.mirror_official_respawn_rng ? device_respawn_rng.data() : nullptr,
            stream);
#else
        {
            for (size_t env_idx = 0; env_idx < count; ++env_idx) {
                int kickoff_order[5];
                const bool have_order = device_commands[env_idx].reset_to_kickoff &&
                    host_kickoff_order(static_cast<int>(env_idx), device_commands[env_idx].kickoff_seed,
                                       device_commands[env_idx].kickoff_seed_mode, kickoff_order);
                detail::ApplyEnvCommand(
                    static_cast<int>(env_idx),
                    cfg,
                    teams.data(),
                    env_car_count.data(),
                    car_states.data(),
                    controls.data(),
                    ball_states.data(),
                    env_states.data(),
                    boost_pad_states.data(),
                    device_commands[env_idx],
                    nullptr,
                    have_order ? kickoff_order : nullptr);
            }
        }
        (void)stream;
#endif
    }

    void step(int32_t ticks, cudaStream_t stream) {
        if (ticks <= 0) {
            return;
        }
#if defined(ROCKETGPU_USE_CUDA)
        if (cfg.mirror_official_respawn_rng) {
            const int32_t* order = prepare_respawn_order(stream);
            detail::LaunchStep(
                    cfg,
                    teams.data(),
                    env_car_count.data(),
                    car_states.data(),
                    wheel_runtime_states.data(),
                    solver_scratch.data(),
                    solver_row_pool,
                    controls.data(),
                    car_configs.data(),
                    ball_states.data(),
                    env_states.data(),
                    boost_pad_states.data(),
                    respawn_slot_overrides.data(),
                    seed,
                    ticks,
                    stream,
                    device_respawn_rng.data(), order, respawn_slot_overrides.data());
        } else {
            detail::LaunchStep(
                cfg,
                teams.data(),
                env_car_count.data(),
                car_states.data(),
                wheel_runtime_states.data(),
                solver_scratch.data(),
                solver_row_pool,
                controls.data(),
                car_configs.data(),
                ball_states.data(),
                env_states.data(),
                boost_pad_states.data(),
                nullptr,
                seed,
                ticks,
                stream);
        }
#else
        {
            for (int32_t tick = 0; tick < ticks; ++tick) {
                const uint8_t* respawn_override_ptr = nullptr;
                if (cfg.mirror_official_respawn_rng) {
                    prepare_respawn_slot_overrides(stream);
                    respawn_override_ptr = host_respawn_slot_overrides.data();
                }
                for (int env_idx = 0; env_idx < cfg.env_count; ++env_idx) {
                    detail::StepEnv(
                        env_idx,
                        cfg,
                        teams.data(),
                        env_car_count.data(),
                        car_states.data(),
                        wheel_runtime_states.data(),
                        solver_scratch.data(),
                        solver_row_pool,
                        controls.data(),
                        car_configs.data(),
                        ball_states.data(),
                        env_states.data(),
                        boost_pad_states.data(),
                        respawn_override_ptr,
                        seed,
                        1);
                }
            }
        }
        (void)stream;
#endif
    }

    void step_with_device_controls(const CarControls* device_controls_by_tick, int32_t ticks, int32_t control_stride, cudaStream_t stream) {
        if (ticks <= 0) {
            return;
        }
        if (device_controls_by_tick == nullptr) {
            throw std::invalid_argument("device_controls_by_tick must not be null");
        }
        if (control_stride < cfg.max_cars_per_env * cfg.env_count) {
            throw std::invalid_argument("control_stride must be at least the total car capacity");
        }
#if defined(ROCKETGPU_USE_CUDA)
        const int32_t* order = cfg.mirror_official_respawn_rng ? prepare_respawn_order(stream) : nullptr;

        detail::LaunchStepWithControlStream(
            cfg,
            teams.data(),
            env_car_count.data(),
            car_states.data(),
            wheel_runtime_states.data(),
            solver_scratch.data(),
            solver_row_pool,
            controls.data(),
            device_controls_by_tick,
            control_stride,
            car_configs.data(),
            ball_states.data(),
            env_states.data(),
            boost_pad_states.data(),
            seed,
            ticks,
            stream,
            cfg.mirror_official_respawn_rng ? device_respawn_rng.data() : nullptr,
            order,
            cfg.mirror_official_respawn_rng ? respawn_slot_overrides.data() : nullptr);
#else
        {
            for (int32_t tick = 0; tick < ticks; ++tick) {
                const uint8_t* respawn_override_ptr = nullptr;
                if (cfg.mirror_official_respawn_rng) {
                    prepare_respawn_slot_overrides(stream);
                    respawn_override_ptr = host_respawn_slot_overrides.data();
                }
                for (int env_idx = 0; env_idx < cfg.env_count; ++env_idx) {
                    detail::StepEnvWithControlStream(
                        env_idx,
                        cfg,
                        teams.data(),
                        env_car_count.data(),
                        car_states.data(),
                        wheel_runtime_states.data(),
                        solver_scratch.data(),
                        solver_row_pool,
                        controls.data(),
                        device_controls_by_tick + static_cast<size_t>(tick) * static_cast<size_t>(control_stride),
                        control_stride,
                        car_configs.data(),
                        ball_states.data(),
                        env_states.data(),
                        boost_pad_states.data(),
                        respawn_override_ptr,
                        seed);
                }
            }

            std::memcpy(controls.data(),
                        device_controls_by_tick + static_cast<size_t>(ticks - 1) * static_cast<size_t>(control_stride),
                        static_cast<size_t>(cfg.env_count) * static_cast<size_t>(cfg.max_cars_per_env) * sizeof(CarControls));
        }
        (void)stream;
#endif
    }

    void step_with_device_commands(
        const CarControls* device_controls_by_tick,
        int32_t control_stride,
        const EnvCommand* device_env_commands_by_tick,
        int32_t env_command_stride,
        int32_t ticks,
        cudaStream_t stream) {
        if (ticks <= 0) {
            return;
        }
        if (device_controls_by_tick == nullptr) {
            throw std::invalid_argument("device_controls_by_tick must not be null");
        }
        if (device_env_commands_by_tick == nullptr) {
            throw std::invalid_argument("device_env_commands_by_tick must not be null");
        }
        if (control_stride < cfg.max_cars_per_env * cfg.env_count) {
            throw std::invalid_argument("control_stride must be at least the total car capacity");
        }
        if (env_command_stride < cfg.env_count) {
            throw std::invalid_argument("env_command_stride must be at least env_count");
        }
#if defined(ROCKETGPU_USE_CUDA)
        const int32_t* order = cfg.mirror_official_respawn_rng ? prepare_respawn_order(stream) : nullptr;

        detail::LaunchStepWithCommandStreams(
            cfg,
            teams.data(),
            env_car_count.data(),
            car_states.data(),
            wheel_runtime_states.data(),
            solver_scratch.data(),
            solver_row_pool,
            controls.data(),
            device_controls_by_tick,
            control_stride,
            device_env_commands_by_tick,
            env_command_stride,
            car_configs.data(),
            ball_states.data(),
            env_states.data(),
            boost_pad_states.data(),
            seed,
            ticks,
            stream,
            cfg.mirror_official_respawn_rng ? device_respawn_rng.data() : nullptr,
            order,
            cfg.mirror_official_respawn_rng ? respawn_slot_overrides.data() : nullptr);
#else
        {
            for (int32_t tick = 0; tick < ticks; ++tick) {
                const uint8_t* respawn_override_ptr = nullptr;
                if (cfg.mirror_official_respawn_rng) {
                    prepare_respawn_slot_overrides(stream);
                    respawn_override_ptr = host_respawn_slot_overrides.data();
                }
                for (int env_idx = 0; env_idx < cfg.env_count; ++env_idx) {
                    int kickoff_order[5];
                    const EnvCommand& tick_command = device_env_commands_by_tick[static_cast<size_t>(tick) * static_cast<size_t>(env_command_stride) + static_cast<size_t>(env_idx)];
                    const bool have_order = tick_command.reset_to_kickoff &&
                        host_kickoff_order(env_idx, tick_command.kickoff_seed, tick_command.kickoff_seed_mode, kickoff_order);
                    detail::StepEnvWithCommandStreams(
                        env_idx,
                        cfg,
                        teams.data(),
                        env_car_count.data(),
                        car_states.data(),
                        wheel_runtime_states.data(),
                        solver_scratch.data(),
                        solver_row_pool,
                        controls.data(),
                        device_controls_by_tick + static_cast<size_t>(tick) * static_cast<size_t>(control_stride),
                        control_stride,
                        device_env_commands_by_tick + static_cast<size_t>(tick) * static_cast<size_t>(env_command_stride),
                        env_command_stride,
                        car_configs.data(),
                        ball_states.data(),
                        env_states.data(),
                        boost_pad_states.data(),
                        respawn_override_ptr,
                        seed,
                        have_order ? kickoff_order : nullptr);
                }
            }

            std::memcpy(controls.data(),
                        device_controls_by_tick + static_cast<size_t>(ticks - 1) * static_cast<size_t>(control_stride),
                        static_cast<size_t>(cfg.env_count) * static_cast<size_t>(cfg.max_cars_per_env) * sizeof(CarControls));
        }
        (void)stream;
#endif
    }

    void synchronize(cudaStream_t stream) const {
#if defined(ROCKETGPU_USE_CUDA)
        CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
#else
        (void)stream;
#endif
    }
};

BatchSim::BatchSim(const SimConfig& config) : impl_(new Impl(config)) {}

BatchSim::~BatchSim() {
    delete impl_;
    impl_ = nullptr;
}

BatchSim::BatchSim(BatchSim&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

BatchSim& BatchSim::operator=(BatchSim&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

const SimConfig& BatchSim::config() const noexcept {
    return impl_->cfg;
}

void BatchSim::set_respawn_rng_seed(uint64_t seed) {
    impl_->cfg.respawn_rng_seed = seed;
    std::vector<int32_t> identity(static_cast<size_t>(impl_->cfg.env_count));
    for (int32_t i = 0; i < impl_->cfg.env_count; ++i) identity[static_cast<size_t>(i)] = i;
    set_respawn_rng_env_indices(identity.data(), identity.size());
}

void BatchSim::set_respawn_rng_env_indices(const int32_t* source_env_indices, size_t count) {
    if (!impl_->cfg.mirror_official_respawn_rng || !source_env_indices) {
        return;
    }
    for (size_t env_idx = 0; env_idx < count && env_idx < impl_->respawn_rng_env.size(); ++env_idx) {
        impl_->respawn_rng_env[env_idx] = std::default_random_engine(
            RespawnRngSeedForEnv(impl_->cfg.respawn_rng_seed, source_env_indices[env_idx]));
    }
#if defined(ROCKETGPU_USE_CUDA)
    const size_t selected = std::min(count, impl_->device_respawn_rng.size());
    if (selected) {
        std::vector<detail::RespawnRngState> states(selected);
        for (size_t env = 0; env < selected; ++env) {
            detail::SeedRespawnRng(states[env], PlatformRespawnRngKind(),
                RespawnRngSeedForEnv(impl_->cfg.respawn_rng_seed, source_env_indices[env]));
        }
        impl_->device_respawn_rng.copy_from_host(states.data(), selected);
        CheckCuda(cudaStreamSynchronize(nullptr), "reseed respawn RNG");
    }
#endif
}

void BatchSim::set_mutator_config(const MutatorConfig& mutator) {
    SimConfig& cfg = impl_->cfg;
    const uint8_t game_mode = cfg.mutator.game_mode;
    const uint8_t arena_collision = cfg.mutator.arena_collision;
    const auto* native_sphere_rsqrt_table = cfg.mutator.native_sphere_rsqrt_table;
    const auto native_sphere_rsqrt_shift = cfg.mutator.native_sphere_rsqrt_shift;
    const auto physics_events = cfg.mutator.physics_events;
    const auto cal_arena_local_tris = cfg.mutator.cal_arena_local_tris;
    const uint32_t ball_settle_epoch = cfg.mutator.ball_settle_epoch;
    cfg.mutator = mutator;
    cfg.mutator.ball_settle_epoch = ball_settle_epoch + 1u;
    cfg.mutator.native_sphere_rsqrt_table = native_sphere_rsqrt_table;
    cfg.mutator.cal_arena_local_tris = cal_arena_local_tris;
    cfg.mutator.native_sphere_rsqrt_shift = native_sphere_rsqrt_shift;
    cfg.mutator.physics_events = physics_events;
    cfg.mutator.game_mode = game_mode;
    cfg.mutator.arena_collision = arena_collision;
    cfg.mutator.ball_no_rot = cfg.no_ball_rot;
    cfg.mutator.cal_ball_world = cfg.mode != Mode::Snowday ? 1 : 0;
    CacheBallDamping(cfg.mutator, cfg.tick_dt, cfg.mode);
    cfg.mutator.solver_use_fma = cfg.mode == Mode::Soccar && PlatformBulletFmaSolver();
}

DeviceView BatchSim::device_view() const noexcept {
    return impl_->view();
}

void BatchSim::set_physics_event_stream(const PhysicsEventStream& stream) {
    const bool disabled = !stream.records && !stream.windows && !stream.capacity_per_env
        && !stream.env_count && !stream.cars_per_env;
    if (!disabled && (!stream.records || !stream.windows || !stream.capacity_per_env
        || stream.env_count != env_count()
        || stream.cars_per_env != static_cast<uint32_t>(impl_->cfg.max_cars_per_env)))
        throw std::invalid_argument("physics event stream must match the complete batch");
    if (!disabled && impl_->cfg.mode != Mode::Soccar)
        throw std::invalid_argument("detailed physics events currently require Soccar");
    impl_->cfg.mutator.physics_events = stream;
}

size_t BatchSim::car_capacity() const noexcept {
    return static_cast<size_t>(impl_->cfg.env_count) * static_cast<size_t>(impl_->cfg.max_cars_per_env);
}

size_t BatchSim::env_count() const noexcept {
    return static_cast<size_t>(impl_->cfg.env_count);
}

size_t BatchSim::boost_pad_capacity() const noexcept {
    return static_cast<size_t>(impl_->cfg.env_count) * static_cast<size_t>(impl_->cfg.boost_pad_count);
}

void BatchSim::set_env_car_count(const int32_t* host_counts, size_t count, cudaStream_t stream) {
    if (count > env_count()) {
        throw std::out_of_range("set_env_car_count count exceeds env_count");
    }
    impl_->env_car_count.copy_from_host(host_counts, count, stream);
    impl_->rebuild_env_step_order(stream);
    impl_->mark_exact_host_dirty();
}

void BatchSim::set_teams(const Team* host_teams, size_t count, cudaStream_t stream) {
    if (count > car_capacity()) {
        throw std::out_of_range("set_teams count exceeds car_capacity");
    }
    impl_->teams.copy_from_host(host_teams, count, stream);
    impl_->mark_exact_host_dirty();
}

void BatchSim::set_car_configs(const CarConfig* host_configs, size_t count, cudaStream_t stream) {
    if (count > car_capacity()) {
        throw std::out_of_range("set_car_configs count exceeds car_capacity");
    }
    impl_->car_configs.copy_from_host(host_configs, count, stream);
    impl_->mark_exact_host_dirty();
}

void BatchSim::upload_car_states(const CarState* host_states, size_t count, cudaStream_t stream,
    StateUploadMode mode) {
    if (count > car_capacity()) {
        throw std::out_of_range("upload_car_states count exceeds car_capacity");
    }
    impl_->car_states.copy_from_host(host_states, count, stream);
    if (mode == StateUploadMode::PublicState) {
#if defined(ROCKETGPU_USE_CUDA)
    detail::LaunchCanonicalizeImportedCars(impl_->car_states.data(), static_cast<int32_t>(count), stream);
#else
    CanonicalizeImportedCarStatesInPlace(impl_->car_states.data(), count);
#endif
    }
    impl_->mark_exact_host_dirty();
}

void BatchSim::upload_ball_states(const BallState* host_states, size_t count, cudaStream_t stream,
    StateUploadMode mode) {
    if (count > env_count()) {
        throw std::out_of_range("upload_ball_states count exceeds env_count");
    }
    impl_->ball_states.copy_from_host(host_states, count, stream);
    if (mode == StateUploadMode::PublicState) {
#if defined(ROCKETGPU_USE_CUDA)
    detail::LaunchCanonicalizeImportedBalls(impl_->ball_states.data(), static_cast<int32_t>(count), stream);
#else
    CanonicalizeImportedBallStatesInPlace(impl_->ball_states.data(), count);
#endif
    }
    impl_->mark_exact_host_dirty();
}

void BatchSim::upload_controls(const CarControls* host_controls, size_t count, cudaStream_t stream) {
    if (count > car_capacity()) {
        throw std::out_of_range("upload_controls count exceeds car_capacity");
    }
    impl_->controls.copy_from_host(host_controls, count, stream);
}

void BatchSim::upload_env_states(const EnvState* host_states, size_t count, cudaStream_t stream) {
    if (count > env_count()) {
        throw std::out_of_range("upload_env_states count exceeds env_count");
    }

    for (size_t env = 0; env < count; ++env) {
        const EnvState& state = host_states[env];
        if (state.car_visit_order_mode == 0 || state.car_visit_order_mode == 2) continue;
        if (state.car_visit_order_mode != 1) {
            throw std::invalid_argument("upload_env_states: car_visit_order_mode must be 0, 1 or 2");
        }
        const int cars = impl_->cfg.max_cars_per_env;
        uint32_t seen = 0;
        for (int k = 0; k < cars; ++k) {
            const int slot = CarVisitSlot(state, k, cars);
            if (slot >= cars || (seen & (1u << slot))) {
                throw std::invalid_argument("upload_env_states: car_visit_order_packed is not a permutation of the car slots");
            }
            seen |= 1u << slot;
        }
    }
    impl_->env_states.copy_from_host(host_states, count, stream);
    impl_->mark_exact_host_dirty();
}

void BatchSim::upload_boost_pad_states(const BoostPadState* host_states, size_t count, cudaStream_t stream) {
    if (count > boost_pad_capacity()) {
        throw std::out_of_range("upload_boost_pad_states count exceeds boost_pad_capacity");
    }
    impl_->boost_pad_states.copy_from_host(host_states, count, stream);
    impl_->mark_exact_host_dirty();
}

void BatchSim::download_car_states(CarState* host_states, size_t count, cudaStream_t stream) const {
    download_car_states_async(host_states, count, stream);
    synchronize(stream);
}

void BatchSim::download_car_states_async(CarState* host_states, size_t count, cudaStream_t stream) const {
    if (count > car_capacity()) {
        throw std::out_of_range("download_car_states count exceeds car_capacity");
    }
    if (count == 0) return;
    if (!host_states) {
        throw std::invalid_argument("download_car_states_async requires a destination");
    }
    impl_->sync_exact_host_for_download();
    impl_->car_states.copy_to_host(host_states, count, stream);
}

void BatchSim::upload_wheel_runtime_states(const detail::WheelRuntimeState* host_states, size_t count, cudaStream_t stream) {
    if (count > car_capacity()) {
        throw std::out_of_range("upload_wheel_runtime_states count exceeds car_capacity");
    }
    impl_->wheel_runtime_states.copy_from_host(host_states, count, stream);
    impl_->mark_exact_host_dirty();
}

void BatchSim::copy_state_from(const BatchSim& source, cudaStream_t stream) {
    if (impl_->cfg.native_sphere_normalization != source.impl_->cfg.native_sphere_normalization)
        throw std::invalid_argument("copy_state_from: native sphere math mismatch");
    if (source.env_count() != env_count() || source.car_capacity() != car_capacity() ||
        source.boost_pad_capacity() != boost_pad_capacity()) {
        throw std::invalid_argument("copy_state_from: shape mismatch");
    }
    if (impl_->cfg.mirror_official_respawn_rng != source.impl_->cfg.mirror_official_respawn_rng) {
        throw std::invalid_argument("copy_state_from: respawn RNG mode mismatch");
    }
    source.impl_->sync_exact_host_for_download();
#if defined(ROCKETGPU_USE_CUDA)
    impl_->device_respawn_rng.copy_from_buffer(source.impl_->device_respawn_rng,
        source.impl_->device_respawn_rng.size(), stream);
#endif
    impl_->car_states.copy_from_buffer(source.impl_->car_states, car_capacity(), stream);
    impl_->ball_states.copy_from_buffer(source.impl_->ball_states, env_count(), stream);
    impl_->env_states.copy_from_buffer(source.impl_->env_states, env_count(), stream);
    impl_->boost_pad_states.copy_from_buffer(source.impl_->boost_pad_states, boost_pad_capacity(), stream);
    impl_->controls.copy_from_buffer(source.impl_->controls, car_capacity(), stream);
    impl_->wheel_runtime_states.copy_from_buffer(source.impl_->wheel_runtime_states, car_capacity(), stream);
    impl_->solver_scratch.copy_from_buffer(source.impl_->solver_scratch, env_count(), stream);
    impl_->copy_solver_rows_from(*source.impl_, stream);

    impl_->seed = source.impl_->seed;
    impl_->respawn_rng = source.impl_->respawn_rng;
    impl_->respawn_rng_env = source.impl_->respawn_rng_env;
    impl_->mark_exact_host_dirty();
}

void BatchSim::copy_solver_scratch_from(const BatchSim& source, cudaStream_t stream) {
    if (source.env_count() != env_count()) {
        throw std::invalid_argument("copy_solver_scratch_from: env count mismatch");
    }
    impl_->solver_scratch.copy_from_buffer(source.impl_->solver_scratch, env_count(), stream);
    impl_->copy_solver_rows_from(*source.impl_, stream);
    impl_->mark_exact_host_dirty();
}

void BatchSim::copy_wheel_runtime_states_from(const BatchSim& source, cudaStream_t stream) {
    if (source.car_capacity() != car_capacity()) {
        throw std::invalid_argument("copy_wheel_runtime_states_from: car capacity mismatch");
    }
    impl_->wheel_runtime_states.copy_from_buffer(source.impl_->wheel_runtime_states, car_capacity(), stream);
    impl_->mark_exact_host_dirty();
}

void BatchSim::download_wheel_runtime_states(detail::WheelRuntimeState* host_states, size_t count, cudaStream_t stream) const {
    if (count > car_capacity()) {
        throw std::out_of_range("download_wheel_runtime_states count exceeds car_capacity");
    }
    impl_->wheel_runtime_states.copy_to_host(host_states, count, stream);
    synchronize(stream);
}

size_t BatchSim::respawn_rng_state_count() const noexcept {
#if defined(ROCKETGPU_USE_CUDA)
    return impl_->device_respawn_rng.size();
#else
    return 0;
#endif
}

void BatchSim::download_respawn_rng_states(detail::RespawnRngState* host_states, size_t count, cudaStream_t stream) const {
    if (count > respawn_rng_state_count()) {
        throw std::out_of_range("download_respawn_rng_states count exceeds respawn_rng_state_count");
    }
#if defined(ROCKETGPU_USE_CUDA)
    impl_->device_respawn_rng.copy_to_host(host_states, count, stream);
    synchronize(stream);
#else
    (void)host_states;
    (void)stream;
#endif
}

void BatchSim::download_ball_states(BallState* host_states, size_t count, cudaStream_t stream) const {
    download_ball_states_async(host_states, count, stream);
    synchronize(stream);
}

void BatchSim::download_ball_states_async(BallState* host_states, size_t count, cudaStream_t stream) const {
    if (count > env_count()) {
        throw std::out_of_range("download_ball_states count exceeds env_count");
    }
    if (count == 0) return;
    if (!host_states) {
        throw std::invalid_argument("download_ball_states_async requires a destination");
    }
    impl_->sync_exact_host_for_download();
    impl_->ball_states.copy_to_host(host_states, count, stream);
}

void BatchSim::download_env_states(EnvState* host_states, size_t count, cudaStream_t stream) const {
    download_env_states_async(host_states, count, stream);
    synchronize(stream);
}

void BatchSim::download_env_states_async(EnvState* host_states, size_t count, cudaStream_t stream) const {
    if (count > env_count()) {
        throw std::out_of_range("download_env_states count exceeds env_count");
    }
    if (count == 0) return;
    if (!host_states) {
        throw std::invalid_argument("download_env_states_async requires a destination");
    }
    impl_->sync_exact_host_for_download();
    impl_->env_states.copy_to_host(host_states, count, stream);
}

void BatchSim::download_boost_pad_states(BoostPadState* host_states, size_t count, cudaStream_t stream) const {
    download_boost_pad_states_async(host_states, count, stream);
    synchronize(stream);
}

void BatchSim::download_boost_pad_states_async(BoostPadState* host_states, size_t count, cudaStream_t stream) const {
    if (count > boost_pad_capacity()) {
        throw std::out_of_range("download_boost_pad_states count exceeds boost_pad_capacity");
    }
    if (count == 0) return;
    if (!host_states) {
        throw std::invalid_argument("download_boost_pad_states_async requires a destination");
    }
    impl_->sync_exact_host_for_download();
    impl_->boost_pad_states.copy_to_host(host_states, count, stream);
}

void BatchSim::download_solver_scratch(detail::WorldSolverScratch* host_states, size_t count, cudaStream_t stream) const {
    if (count > env_count()) {
        throw std::out_of_range("download_solver_scratch count exceeds env_count");
    }
    impl_->solver_scratch.copy_to_host(host_states, count, stream);
    synchronize(stream);
}

void BatchSim::download_solver_scratch_range(detail::WorldSolverScratch* host_states, size_t first, size_t count, cudaStream_t stream) const {
    if (first > env_count() || count > env_count() - first) {
        throw std::out_of_range("download_solver_scratch_range exceeds env_count");
    }
    impl_->solver_scratch.copy_range_to_host(host_states, first, count, stream);
    synchronize(stream);
}

void BatchSim::download_solver_scratch_rows(detail::WorldSolverScratchRowPoolHost& out, size_t first, size_t count, cudaStream_t stream) const {
    if (first > env_count() || count > env_count() - first) {
        throw std::out_of_range("download_solver_scratch_rows exceeds env_count");
    }
    impl_->download_solver_rows(out, first, count, stream);
    synchronize(stream);
}

void BatchSim::reset_to_kickoff(uint64_t seed, cudaStream_t stream) {
    impl_->reset_to_kickoff(seed, stream);
}

void BatchSim::apply_env_commands(const EnvCommand* host_commands, size_t count, cudaStream_t stream) {
    impl_->apply_env_commands_host(host_commands, count, stream);
}

void BatchSim::apply_env_commands_device(const EnvCommand* device_commands, size_t count, cudaStream_t stream) {
    impl_->apply_env_commands_device(device_commands, count, stream);
}

void BatchSim::step(int32_t ticks, cudaStream_t stream) {
    impl_->step(ticks, stream);
}

void BatchSim::step_with_device_controls(const CarControls* device_controls_by_tick, int32_t ticks, int32_t control_stride, cudaStream_t stream) {
    impl_->step_with_device_controls(device_controls_by_tick, ticks, control_stride, stream);
}

void BatchSim::step_with_device_commands(
    const CarControls* device_controls_by_tick,
    int32_t control_stride,
    const EnvCommand* device_env_commands_by_tick,
    int32_t env_command_stride,
    int32_t ticks,
    cudaStream_t stream) {
    impl_->step_with_device_commands(device_controls_by_tick, control_stride, device_env_commands_by_tick, env_command_stride, ticks, stream);
}

size_t SolverScratchBytesPerEnv() {
    return sizeof(detail::WorldSolverScratch) + detail::WorldSolverScratchRowBytesPerEnv();
}

void BatchSim::synchronize(cudaStream_t stream) const {
    impl_->synchronize(stream);
}
}
