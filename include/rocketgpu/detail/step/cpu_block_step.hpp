#pragma once
#if !defined(ROCKETGPU_USE_CUDA)
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(ROCKETGPU_CPU_ISPC)
#include "rocketgpu/detail/step/state_data_ranges.hpp"
struct RgPreTickParams {
    float dt, car_mass, flip_torque_time, supersonic_start_speed, supersonic_maintain_max_time,
        supersonic_maintain_min_speed, car_max_speed, car_max_ang_speed;
};
extern "C" void rg_pretick_cars(::rocketgpu::CarState* cars, const float* in, const int32_t* idx, const float* frame_frac,
                                int32_t n, const RgPreTickParams& p, float* out, uint32_t* wout);
extern "C" void rg_pretick_balls(const float* in, int32_t n, float max_speed, float max_ang_speed,
                                 bool heatseeker, bool damp, float damp_scale, float* out);
#endif

namespace rocketgpu {
namespace detail {

constexpr int kCpuBlockMaxEnvs = 64;

inline void CpuPrefetch(const void* p, int bytes) {
    const uintptr_t first = reinterpret_cast<uintptr_t>(p) & ~uintptr_t{63};
    const uintptr_t end = reinterpret_cast<uintptr_t>(p) + static_cast<uintptr_t>(bytes);
    for (uintptr_t a = first; a < end; a += 64) {
#if defined(_MSC_VER) && !defined(__clang__)
        _mm_prefetch(reinterpret_cast<const char*>(a), _MM_HINT_T0);
#else
        __builtin_prefetch(reinterpret_cast<const void*>(a));
#endif
    }
}

#if defined(ROCKETGPU_CPU_ISPC)
constexpr int kCpuBlockMinEnvs = 4;

constexpr int kBallOutFields = 43;
constexpr int kBallOutStartVel = 37;
constexpr int kBallOutPredictVel = 40;
inline bool SameVec3(const Vec3& a, const Vec3& b) { return std::memcmp(&a, &b, sizeof(Vec3)) == 0; }
inline Vec3 BallOutV3(const float* o, int f) {
    return Vec3(o[f * kCpuBlockMaxEnvs], o[(f + 1) * kCpuBlockMaxEnvs], o[(f + 2) * kCpuBlockMaxEnvs]);
}
constexpr int kCarOutLanes = 64;
constexpr int kCarOutFields = 60;
constexpr int kCarOutWords = 10;
constexpr int kCarOutStartVel = 49;
constexpr int kCarOutSnapRot = 52;
constexpr int kCarOutSnapRaw = 56;
inline Vec3 CarOutV3(const float* o, int f) {
    return Vec3(o[f * kCarOutLanes], o[(f + 1) * kCarOutLanes], o[(f + 2) * kCarOutLanes]);
}
inline Quat CarOutQ(const float* o, int f) {
    Quat q;
    q.w = o[f * kCarOutLanes];
    q.x = o[(f + 1) * kCarOutLanes];
    q.y = o[(f + 2) * kCarOutLanes];
    q.z = o[(f + 3) * kCarOutLanes];
    return q;
}
inline void StoreCarPreTickOut(CarState& c, WheelRuntimeState& w, const float* o, const uint32_t* wo) {
    constexpr int s = kCarOutLanes;
    c.tick_start_pos_bt = CarOutV3(o, 0);
    c.pos = CarOutV3(o, 3);
    c.tick_start_basis_forward = CarOutV3(o, 6);
    c.tick_start_basis_right = CarOutV3(o, 9);
    c.tick_start_basis_up = CarOutV3(o, 12);
    c.cal_tick_start_ang_vel = CarOutV3(o, 15);
    c.supersonic_time = o[18 * s];
    c.air_time = o[19 * s];
    c.air_time_since_jump = o[20 * s];
    c.cal_pre_forward_speed_uu = o[21 * s];
    c.cal_pre_pitchlock_restart_time = o[22 * s];
    c.pos_bt = CarOutV3(o, 23);
    c.vel = CarOutV3(o, 26);
    c.vel_bt = CarOutV3(o, 29);
    c.ang_vel = CarOutV3(o, 32);
    c.internal_flip_time = o[35 * s];
    c.rot = CarOutQ(o, 36);
    c.basis_forward = CarOutV3(o, 40);
    c.basis_right = CarOutV3(o, 43);
    c.basis_up = CarOutV3(o, 46);
    constexpr int kCarWordOff[8] = {120, 124, 128, 132, 136, 312, 608, 660};
    constexpr int kWrsWordOff[2] = {280, 288};
    for (int k = 0; k < 8; ++k) {
        std::memcpy(reinterpret_cast<unsigned char*>(&c) + kCarWordOff[k], &wo[k * s], sizeof(uint32_t));
    }
    for (int k = 0; k < 2; ++k) {
        std::memcpy(reinterpret_cast<unsigned char*>(&w) + kWrsWordOff[k], &wo[(8 + k) * s], sizeof(uint32_t));
    }
}
constexpr int kBallInFields = 23;
inline void LoadBallPreTickIn(const BallState& b, float* in) {
    constexpr int s = kCpuBlockMaxEnvs;
    const auto v3 = [&](int f, const Vec3& v) {
        in[f * s] = v.x;
        in[(f + 1) * s] = v.y;
        in[(f + 2) * s] = v.z;
    };
    v3(0, b.vel_bt);
    v3(3, b.ang_vel);
    v3(6, b.pos_bt);
    v3(9, b.basis_forward);
    v3(12, b.basis_right);
    v3(15, b.basis_up);
    in[18 * s] = b.rot.w;
    in[19 * s] = b.rot.x;
    in[20 * s] = b.rot.y;
    in[21 * s] = b.rot.z;
    in[22 * s] = b.basis_authoritative != 0 ? 1.0f : 0.0f;
}
inline void StoreBallPreTickOut(BallState& b, const float* o) {
    constexpr int s = kCpuBlockMaxEnvs;
    const auto v3 = [&](int f) { return BallOutV3(o, f); };
    b.pos_bt = v3(0);
    b.pos = v3(3);
    b.vel_bt = v3(6);
    b.vel = v3(9);
    b.ang_vel = v3(12);
    b.rot.w = o[15 * s];
    b.rot.x = o[16 * s];
    b.rot.y = o[17 * s];
    b.rot.z = o[18 * s];
    b.basis_forward = v3(19);
    b.basis_right = v3(22);
    b.basis_up = v3(25);
    b.basis_authoritative = 1;
    b.inertia_basis_forward = v3(28);
    b.inertia_basis_right = v3(31);
    b.inertia_basis_up = v3(34);
    b.inertia_basis_valid = 1;
}
constexpr int kCarInBlocks = 15;
constexpr int kCarInWrsBlock = 14;
constexpr int kCarInOff[kCarInBlocks] = {28, 40, 72, 92, 264, 276, 312, 328, 340, 496, 120, 136, 600, 660, 276};
constexpr uint8_t kCarInMask[kCarInBlocks] = {7, 7, 3, 1, 7, 7, 15, 7, 7, 3, 15, 1, 7, 1, 14};
static_assert(offsetof(CarState, vel) == 28 && offsetof(CarState, ang_vel) == 40 && offsetof(CarState, air_time) == 72 &&
                  offsetof(CarState, air_time_since_jump) == 76 && offsetof(CarState, supersonic_time) == 92 &&
                  offsetof(CarState, pos_bt) == 264 && offsetof(CarState, vel_bt) == 276 &&
                  offsetof(CarState, basis_forward) == 316 && offsetof(CarState, basis_right) == 328 &&
                  offsetof(CarState, basis_up) == 340 && offsetof(CarState, internal_flip_time) == 496 &&
                  offsetof(WheelRuntimeState, cal_jump_inactive_time) == 284 && sizeof(WheelRuntimeState) >= 292 &&
                  sizeof(CarState) >= 676,
              "rg_pretick.ispc RG_IN_* rows and kCarInOff");
#if defined(__clang__)
typedef uint32_t PreTickInU4 __attribute__((vector_size(16)));
#endif
inline const unsigned char* CarInSrc(const CarState* cars, const WheelRuntimeState* wrs, int ci, int b) {
    return b == kCarInWrsBlock ? reinterpret_cast<const unsigned char*>(&wrs[ci]) + kCarInOff[b]
                               : reinterpret_cast<const unsigned char*>(&cars[ci]) + kCarInOff[b];
}
inline void LoadCarPreTickIn(const CarState* cars, const WheelRuntimeState* wrs, const int32_t* idx, int n, uint32_t* in) {
    for (int j = 0; j < n; j += 4) {
        int ci[4];
        for (int c = 0; c < 4; ++c) {
            ci[c] = idx[j + c < n ? j + c : n - 1];
        }
#if defined(__clang__)
#pragma clang loop unroll(full)
#endif
        for (int b = 0; b < kCarInBlocks; ++b) {
#if defined(__clang__)
            PreTickInU4 v0, v1, v2, v3;
            std::memcpy(&v0, CarInSrc(cars, wrs, ci[0], b), 16);
            std::memcpy(&v1, CarInSrc(cars, wrs, ci[1], b), 16);
            std::memcpy(&v2, CarInSrc(cars, wrs, ci[2], b), 16);
            std::memcpy(&v3, CarInSrc(cars, wrs, ci[3], b), 16);
            const PreTickInU4 t0 = __builtin_shufflevector(v0, v1, 0, 4, 1, 5);
            const PreTickInU4 t1 = __builtin_shufflevector(v0, v1, 2, 6, 3, 7);
            const PreTickInU4 t2 = __builtin_shufflevector(v2, v3, 0, 4, 1, 5);
            const PreTickInU4 t3 = __builtin_shufflevector(v2, v3, 2, 6, 3, 7);
            const PreTickInU4 r[4] = {__builtin_shufflevector(t0, t2, 0, 1, 4, 5), __builtin_shufflevector(t0, t2, 2, 3, 6, 7),
                                   __builtin_shufflevector(t1, t3, 0, 1, 4, 5), __builtin_shufflevector(t1, t3, 2, 3, 6, 7)};
            for (int k = 0; k < 4; ++k) {
                if (kCarInMask[b] & (1 << k)) {
                    std::memcpy(&in[(4 * b + k) * kCarOutLanes + j], &r[k], 16);
                }
            }
#else
            for (int k = 0; k < 4; ++k) {
                if (kCarInMask[b] & (1 << k)) {
                    for (int c = 0; c < 4; ++c) {
                        std::memcpy(&in[(4 * b + k) * kCarOutLanes + j + c], CarInSrc(cars, wrs, ci[c], b) + 4 * k, 4);
                    }
                }
            }
#endif
        }
    }
}
inline bool CarPreTickInMatches(const CarState* cars, const WheelRuntimeState* wrs, const int32_t* idx, int n, const uint32_t* in) {
    for (int j = 0; j < n; ++j) {
        for (int b = 0; b < kCarInBlocks; ++b) {
            for (int k = 0; k < 4; ++k) {
                if ((kCarInMask[b] & (1 << k)) &&
                    std::memcmp(&in[(4 * b + k) * kCarOutLanes + j], CarInSrc(cars, wrs, idx[j], b) + 4 * k, 4) != 0) {
                    return false;
                }
            }
        }
    }
    return true;
}
#endif

static_assert(sizeof(WorldSolverScratchRows) ==
                  11 * sizeof(EnvStridedArray<int>) + sizeof(CarCarPairCacheView) + 5 * sizeof(void*) + 4 * sizeof(int),
              "CpuRowsView must set every WorldSolverScratchRows field");
inline WorldSolverScratchRows CpuRowsView(const WorldSolverScratchRowPool& p, int env_idx) {
    if (p.interleaved) {
        return p.view(env_idx);
    }
    const std::ptrdiff_t e = env_idx;
    const auto at = [e](auto* base, int per_env) { return base == nullptr ? base : base + e * per_env; };
    const WorldSolverScratchRowCaps& c = p.caps;
    WorldSolverScratchRows r;
    r.ball_split_rows = at(p.ball_split_rows, c.ball_split);
    r.ball_velocity_rows = at(p.ball_velocity_rows, c.ball_velocity);
    r.ball_friction_rows = at(p.ball_friction_rows, c.ball_friction);
    r.pair_split_rows = at(p.pair_split_rows, c.pair_split);
    r.world_split_rows = at(p.world_split_rows, c.world_split);
    r.pair_velocity_rows = at(p.pair_velocity_rows, c.pair_velocity);
    r.pair_friction_rows = at(p.pair_friction_rows, c.pair_friction);
    r.world_velocity_rows = at(p.world_velocity_rows, c.world_velocity);
    r.world_friction_rows = at(p.world_friction_rows, c.world_friction);
    r.shared_contact_order_groups = at(p.shared_contact_order_groups, c.order_groups);
    r.bullet_manifolds = at(p.bullet_manifolds, c.manifolds);
    r.pair_contact_cache.slots = at(p.pair_contact_cache, c.pair_cache_cars * c.pair_cache_cars);
    r.pair_contact_cache.cars = c.pair_cache_cars;
    r.car_world_plane_manifolds = at(p.car_world_plane_manifolds, c.car_world_plane);
    r.car_world_mesh_manifolds = at(p.car_world_mesh_manifolds, c.car_world_mesh);
    r.car_car_retained_manifolds = at(p.car_car_retained_manifolds, c.car_car_retained);
    r.ball_car_retained_manifolds = at(p.ball_car_retained_manifolds, c.ball_car_retained);
    r.car_world_local_rows = at(p.car_world_local_rows, c.car_world_local);
    r.car_world_plane_count = c.car_world_plane;
    r.car_world_mesh_count = c.car_world_mesh;
    r.car_car_retained_count = c.car_car_retained;
    r.ball_car_retained_count = c.ball_car_retained;
    return r;
}

inline int CpuBlockEnvs(int env_count, int workers) {
    static const int forced = [] {
        const char* s = std::getenv("ROCKETGPU_CPU_BLOCK");
        const int b = s != nullptr ? std::atoi(s) : 0;
        return s == nullptr ? 0 : (b < 1 ? 1 : (b > kCpuBlockMaxEnvs ? kCpuBlockMaxEnvs : b));
    }();
    if (forced > 0) {
        return forced;
    }
    const int w = workers < 1 ? 1 : workers;
    const int small = (env_count + 2 * w - 1) / (2 * w);
    const int lo = small < 1 ? 1 : (small < 4 ? small : 4);
    const int b = env_count / (16 * w);
    const int blk = b < lo ? lo : (b > 16 ? 16 : b);
    const int fit = env_count / (3 * w);
    return blk > fit ? (fit < 1 ? 1 : fit) : blk;
}

constexpr int kCpuTailEnvsPerWorker = 2;
inline int CpuTailEnvs(int env_count, int workers, int block) {
    const int tail = kCpuTailEnvsPerWorker * workers;
    if (workers < 2 || block < 2) {
        return 0;
    }
    return env_count >= 8 * tail ? tail : (env_count >= 4 * workers ? workers : 0);
}

inline bool CpuIspcOff() {
    static const bool off = std::getenv("ROCKETGPU_ISPC_OFF") != nullptr;
    return off;
}

inline bool CpuIspcCheck() {
    static const bool on = std::getenv("ROCKETGPU_ISPC_CHECK") != nullptr;
    return on;
}

struct CpuBlockRng {
    void* context;
    const uint8_t* (*prepare)(void*, int);
    bool (*kickoff)(void*, int, uint64_t, uint8_t, int*);
    bool (*report_each_tick)(void*, int, int);
};

template <bool kReportEachTick = false, typename ControlsFor>
inline void StepEnvBlockCpu(
    int env0,
    int env_n,
    const SimConfig& cfg,
    Team* teams,
    int32_t* env_car_count,
    CarState* cars,
    WheelRuntimeState* wheel_runtime_states,
    WorldSolverScratch* solver_scratch,
    const WorldSolverScratchRowPool& row_pool,
    const ControlsFor& controls_for,
    CarConfig* car_configs,
    BallState* balls,
    EnvState* env_states,
    BoostPadState* pads,
    uint64_t seed,
    int ticks,
    const EnvCommand* commands_by_tick = nullptr,
    int command_stride = 0,
    CarControls* controls_for_reset = nullptr,
    const CpuBlockRng* mirror_rng = nullptr) {
    constexpr int kMaxCars = kCpuBlockMaxEnvs * 8;
    if (env_n > 1 && env_n * cfg.max_cars_per_env > kMaxCars) {
        const int h = env_n / 2;
        StepEnvBlockCpu<kReportEachTick>(env0, h, cfg, teams, env_car_count, cars, wheel_runtime_states, solver_scratch,
                                         row_pool, controls_for, car_configs, balls, env_states, pads, seed, ticks,
                                         commands_by_tick, command_stride, controls_for_reset, mirror_rng);
        StepEnvBlockCpu<kReportEachTick>(env0 + h, env_n - h, cfg, teams, env_car_count, cars, wheel_runtime_states,
                                         solver_scratch, row_pool, controls_for, car_configs, balls, env_states, pads,
                                         seed, ticks, commands_by_tick, command_stride, controls_for_reset, mirror_rng);
        return;
    }
    bool mirror_reports_each_tick[kCpuBlockMaxEnvs];
    const uint8_t* respawn_overrides[kCpuBlockMaxEnvs];
    int pre_counts[kCpuBlockMaxEnvs];
    int32_t lane_idx[kMaxCars];
    int32_t lane_env[kMaxCars];
    int32_t lane_slot[kMaxCars];
    float lane_frac[kMaxCars];
    CarFrame snap_frame;
    const ArenaBroadphaseGrid grid = MakeArenaBroadphaseGrid(cfg);
    CarFrame car_frames[8];
    const CarShapeCacheEntry* car_shapes[kCpuBlockMaxEnvs];
    UninitArray<CarShapeCacheEntry, kCpuBlockMaxEnvs * 8> car_shapes_local;
    UninitArray<WorldSolverScratchRows, kCpuBlockMaxEnvs> env_rows;
    const CarControls* tick_controls[kCpuBlockMaxEnvs];
    int16_t tick_order[kMaxCars];
    UninitArray<CarControlsMemo, kCpuBlockMaxEnvs * 8> ctl_memos;
    for (int e = 0; e < env_n; ++e) {
        mirror_reports_each_tick[e] = mirror_rng && !kReportEachTick &&
            mirror_rng->report_each_tick(mirror_rng->context, env0 + e, ticks);
        const int count = env_car_count[env0 + e];
        car_shapes[e] = nullptr;
        if (count <= 8) {
            const CarConfig* const env_cfgs = car_configs + (env0 + e) * cfg.max_cars_per_env;
            if (solver_scratch != nullptr && mirror_rng == nullptr) {
                car_shapes[e] = CarShapeCacheFor(solver_scratch[env0 + e], env_cfgs, count, cfg.mutator.car_mass);
            } else {
                FillCarShapeCache(&car_shapes_local[e * 8], env_cfgs, count, cfg.mutator.car_mass);
                car_shapes[e] = &car_shapes_local[e * 8];
            }
        }
        new (&env_rows[e]) WorldSolverScratchRows(CpuRowsView(row_pool, env0 + e));
        for (int i = 0; i < 8; ++i) {
            new (&ctl_memos[e * 8 + i]) CarControlsMemo;
        }
    }
    const WorldStepHostArgs world_args{&cfg, &grid, teams, cars, wheel_runtime_states, car_configs};
#if defined(ROCKETGPU_CPU_ISPC)
    const MutatorConfig& m = cfg.mutator;
    const RgPreTickParams pre_params{cfg.tick_dt, m.car_mass, m.flip_torque_time, m.supersonic_start_speed,
                                     m.supersonic_maintain_max_time, m.supersonic_maintain_min_speed, m.car_max_speed,
                                     m.car_max_ang_speed};
    const bool lanes = m.native_sphere_rsqrt_table == nullptr && wheel_runtime_states != nullptr &&
                       !CpuIspcOff();
    const bool ball_lanes = m.native_sphere_rsqrt_table == nullptr && !CpuIspcOff();
    const bool ball_heatseeker = m.game_mode == static_cast<uint8_t>(Mode::Heatseeker);
    const bool ball_damp = ball_lanes && !ball_heatseeker ? !(m.ball_drag <= 0.0f) : m.ball_drag > 0.0f;
    const float ball_damp_scale =
        !ball_damp ? 1.0f
        : (m.cached_ball_damping_drag == m.ball_drag && m.cached_ball_damping_dt == cfg.tick_dt &&
           m.cached_ball_damping_scale >= 0.0f)
            ? m.cached_ball_damping_scale
            : RocketGpuUcrtPowf(Max(0.0f, 1.0f - m.ball_drag), cfg.tick_dt);
    alignas(64) float ball_out[kBallOutFields * kCpuBlockMaxEnvs];
    alignas(64) float ball_in[kBallInFields * kCpuBlockMaxEnvs];
#else
    const bool lanes = false;
    const bool ball_lanes = false;
#endif
    uint64_t pad_work[kCpuBlockMaxEnvs];
    uint64_t pad_prev[kCpuBlockMaxEnvs];
    bool pad_work_ok[kCpuBlockMaxEnvs];
    for (int e = 0; e < env_n; ++e) {
        pad_work_ok[e] = false;
    }
    const auto prefetch_env_cars = [&](int e) {
        const int env_idx = env0 + e;
        const int base = env_idx * cfg.max_cars_per_env;
        for (int k = 0; k < env_car_count[env_idx]; ++k) {
            const char* const c = reinterpret_cast<const char*>(&cars[base + k]);
            for (size_t off = 0; off < sizeof(CarState); off += 64) {
                __builtin_prefetch(c + off);
            }
            if (wheel_runtime_states != nullptr) {
                const char* const w = reinterpret_cast<const char*>(&wheel_runtime_states[base + k]);
                for (size_t off = 0; off < sizeof(WheelRuntimeState); off += 64) {
                    __builtin_prefetch(w + off);
                }
            }
        }
    };
    for (int t = 0; t < ticks; ++t) {
        int n = 0;
        for (int e = 0; e < env_n; ++e) {
            CpuPrefetch(controls_for(env0 + e).row(t), env_car_count[env0 + e] * static_cast<int>(sizeof(CarControls)));
        }
        for (int e = 0; e < env_n; ++e) {
            if (e + 1 < env_n) {
                prefetch_env_cars(e + 1);
            }
            const int env_idx = env0 + e;
            const int base = env_idx * cfg.max_cars_per_env;
            const auto controls = controls_for(env_idx);
            respawn_overrides[e] = mirror_rng && (kReportEachTick || mirror_reports_each_tick[e])
                ? mirror_rng->prepare(mirror_rng->context, env_idx) : nullptr;
            if constexpr (kReportEachTick) {
                const EnvCommand& command = commands_by_tick[static_cast<size_t>(t) * command_stride + env_idx];
                if (command.reset_to_kickoff || command.new_round_bodies) {
                    int kickoff_order[5];
                    const bool have_order = mirror_rng && command.reset_to_kickoff &&
                        mirror_rng->kickoff(mirror_rng->context, env_idx, command.kickoff_seed,
                                            command.kickoff_seed_mode, kickoff_order);
                    StepEnvPreTickEnv(env_idx, cfg, teams, env_car_count, cars, solver_scratch, controls_for_reset,
                                      balls, env_states, pads, command, have_order ? kickoff_order : nullptr, car_configs);
                    pad_work_ok[e] = false;
                } else {
                    StepEnvPreTickEnvState(env_idx, cfg, env_car_count, cars, balls, env_states);
                }
            } else {
                StepEnvPreTickEnvState(env_idx, cfg, env_car_count, cars, balls, env_states);
            }
            const int pre_count = env_car_count[env_idx];
            pre_counts[e] = pre_count;
            const CarControls* const env_controls = controls.row(t);
            tick_controls[e] = env_controls;
            int16_t* const env_order = &tick_order[e * cfg.max_cars_per_env];
            const float env_frac = FrameFrac(cfg, env_idx);
            for (int k = 0; k < pre_count; ++k) {
                env_order[k] = static_cast<int16_t>(CarVisitSlot(env_states[env_idx], k, pre_count));
            }
            for (int k = 0; k < pre_count; ++k) {
                const int slot = env_order[k];
                const CarState& car = cars[base + slot];
                const CarControls& c = env_controls[slot];
                if (lanes && !car.is_demoed && car.tick_count_since_update != 0 &&
                    !wheel_runtime_states[base + slot].cal_restored &&
                    !(c.jump && !car.driving_disabled && !car.last_controls.jump) && !car.is_auto_flipping &&
                    !car_configs[base + slot].three_wheels) {
                    lane_idx[n] = base + slot;
                    lane_env[n] = env_idx;
                    lane_slot[n] = slot;
                    lane_frac[n++] = env_frac;
                    continue;
                }
                StepEnvPreTickCar(env_idx, slot, cfg, teams, cars, solver_scratch, env_states[env_idx], respawn_overrides[e], seed, c,
                                  car_configs[base + slot],
                                  wheel_runtime_states ? &wheel_runtime_states[base + slot] : nullptr);
            }
        }
#if defined(ROCKETGPU_CPU_ISPC)
        for (int j0 = 0; j0 < n; j0 += kCarOutLanes) {
            const int ln = n - j0 < kCarOutLanes ? n - j0 : kCarOutLanes;
            alignas(64) float car_out[kCarOutFields * kCarOutLanes];
            alignas(64) uint32_t car_wout[kCarOutWords * kCarOutLanes];
            alignas(64) uint32_t car_in[4 * kCarInBlocks * kCarOutLanes];
            LoadCarPreTickIn(cars, wheel_runtime_states, lane_idx + j0, ln, car_in);
            rg_pretick_cars(cars, reinterpret_cast<const float*>(car_in), lane_idx + j0, lane_frac + j0, ln, pre_params, car_out,
                            car_wout);
            if (CpuIspcCheck()) {
                if (!CarPreTickInMatches(cars, wheel_runtime_states, lane_idx + j0, ln, car_in)) {
                    std::fprintf(stderr, "ROCKETGPU_ISPC_CHECK pre-tick input rows mismatch, lanes from car %d\n", lane_idx[j0]);
                }
                for (int jj = 0; jj < ln; ++jj) {
                    const int j = j0 + jj;
                    const float* const o = car_out + jj;
                    CarState got;
                    WheelRuntimeState got_w;
                    std::memcpy(static_cast<void*>(&got), &cars[lane_idx[j]], sizeof(CarState));
                    std::memcpy(static_cast<void*>(&got_w), &wheel_runtime_states[lane_idx[j]], sizeof(WheelRuntimeState));
                    StoreCarPreTickOut(got, got_w, o, car_wout + jj);
                    const int env_idx = lane_env[j], slot = lane_slot[j];
                    StepEnvPreTickCar(env_idx, slot, cfg, teams, cars, solver_scratch, env_states[env_idx], respawn_overrides[env_idx - env0],
                                      seed, controls_for(env_idx).get(slot, t), car_configs[lane_idx[j]],
                                      &wheel_runtime_states[lane_idx[j]]);
                    const Vec3 start_vel = CarOutV3(o, kCarOutStartVel);
                    if (!StateDataEqual(&got, &cars[lane_idx[j]], kCarStateData) ||
                        !StateDataEqual(&got_w, &wheel_runtime_states[lane_idx[j]], kWheelRuntimeStateData) ||
                        (solver_scratch != nullptr && slot < 8 &&
                         std::memcmp(&start_vel, &solver_scratch[env_idx].cal_start_vel[slot], sizeof(Vec3)) != 0)) {
                        std::fprintf(stderr, "ROCKETGPU_ISPC_CHECK pre-tick mismatch, car %d\n", lane_idx[j]);
                    }
                    const CarState& cc = cars[lane_idx[j]];
                    const Quat raw = GameGetRotationScalar(cc.basis_forward, cc.basis_right, cc.basis_up);
                    const Quat rot = GameRbStateQuantizeQuat(raw);
                    const Quat lane_raw = CarOutQ(o, kCarOutSnapRaw);
                    const Quat lane_rot = CarOutQ(o, kCarOutSnapRot);
                    if (std::memcmp(&raw, &lane_raw, sizeof(Quat)) != 0 || std::memcmp(&rot, &lane_rot, sizeof(Quat)) != 0) {
                        std::fprintf(stderr, "ROCKETGPU_ISPC_CHECK pre-tick snap rotation mismatch, car %d\n", lane_idx[j]);
                    }
                }
                continue;
            }
            for (int jj = 0; jj < ln; ++jj) {
                const int j = j0 + jj;
                const float* const o = car_out + jj;
                CarState& car = cars[lane_idx[j]];
                StoreCarPreTickOut(car, wheel_runtime_states[lane_idx[j]], o, car_wout + jj);
                const int slot = lane_slot[j];
                if (solver_scratch != nullptr && slot < 8) {
                    WorldSolverScratch& s = solver_scratch[lane_env[j]];
                    s.cal_start_vel[slot] = CarOutV3(o, kCarOutStartVel);
                    s.car_predict_vel[slot] = car.vel;
                    s.car_predict_vel_bt[slot] = car.vel_bt;
                    s.car_predict_ang_vel[slot] = car.ang_vel;
                    s.car_predict_valid[slot] = 1;
                    if (mirror_rng == nullptr) {
                        CarRbSnapMemo& snap = s.rb_snap[slot];
                        snap.fwd = car.basis_forward;
                        snap.right = car.basis_right;
                        snap.up = car.basis_up;
                        snap.rot = CarOutQ(o, kCarOutSnapRot);
                        snap.raw = CarOutQ(o, kCarOutSnapRaw);
                        snap.valid = 1;
                    }
                }
            }
        }
        if (ball_lanes) {
            for (int e = 0; e < env_n; ++e) {
                LoadBallPreTickIn(balls[env0 + e], ball_in + e);
            }
            if (CpuIspcCheck()) {
                static thread_local BallState start[kCpuBlockMaxEnvs];
                static thread_local BallState got[kCpuBlockMaxEnvs];
                std::memcpy(start, balls + env0, sizeof(BallState) * env_n);
                rg_pretick_balls(ball_in, env_n, m.ball_max_speed, m.ball_max_ang_speed, ball_heatseeker, ball_damp,
                                 ball_damp_scale, ball_out);
                for (int e = 0; e < env_n; ++e) {
                    StoreBallPreTickOut(balls[env0 + e], ball_out + e);
                }
                std::memcpy(got, balls + env0, sizeof(BallState) * env_n);
                std::memcpy(balls + env0, start, sizeof(BallState) * env_n);
                for (int e = 0; e < env_n; ++e) {
                    StepEnvPreTickBall(env0 + e, cfg, solver_scratch, balls);
                    if (!StateDataEqual(&got[e], &balls[env0 + e], kBallStateData) ||
                        (solver_scratch != nullptr &&
                         (!SameVec3(BallOutV3(ball_out + e, kBallOutStartVel), solver_scratch[env0 + e].cal_ball_start_vel) ||
                          (!ball_heatseeker && !SameVec3(BallOutV3(ball_out + e, kBallOutPredictVel),
                                                       solver_scratch[env0 + e].ball_predict_vel_bt))))) {
                        std::fprintf(stderr, "ROCKETGPU_ISPC_CHECK ball pre-tick mismatch, env %d\n", env0 + e);
                    }
                }
            } else {
                rg_pretick_balls(ball_in, env_n, m.ball_max_speed, m.ball_max_ang_speed, ball_heatseeker, ball_damp,
                                 ball_damp_scale, ball_out);
                for (int e = 0; e < env_n; ++e) {
                    StoreBallPreTickOut(balls[env0 + e], ball_out + e);
                }
                if (solver_scratch != nullptr) {
                    for (int e = 0; e < env_n; ++e) {
                        solver_scratch[env0 + e].cal_ball_start_vel = BallOutV3(ball_out + e, kBallOutStartVel);
                        if (!ball_heatseeker) {
                            solver_scratch[env0 + e].ball_predict_vel_bt = BallOutV3(ball_out + e, kBallOutPredictVel);
                        }
                    }
                }
            }
        }
#endif
        for (int e = 0; e < env_n; ++e) {
            const bool report_fields = kReportEachTick || t + 1 == ticks || mirror_reports_each_tick[e];
            const int env_idx = env0 + e;
            if (!ball_lanes) {
                StepEnvPreTickBall(env_idx, cfg, solver_scratch, balls);
            }
            if (e + 1 < env_n && cfg.enable_boost_pads) {
                const int pad_count = BoostPadCount(cfg);
                CpuPrefetch(pads + (env_idx + 1) * pad_count, pad_count * static_cast<int>(sizeof(BoostPadState)));
            }
            BoostPadTickMasks pad_masks;
            const bool pad_pre = cfg.enable_boost_pads && pre_counts[e] > 0;
            if (pad_pre) {
                const int pad_count = BoostPadCount(cfg);
                if (pad_work_ok[e]) {
                    PreTickBoostPadsHostWork(pads + env_idx * pad_count, pad_work[e], pad_prev[e], cfg.tick_dt,
                                             cfg.tick_dt > 0.0f, pad_masks);
                } else {
                    PreTickBoostPadsHostMasks(pads + env_idx * pad_count, pad_count, cfg.tick_dt, cfg.tick_dt > 0.0f,
                                              pad_masks);
                    pad_work[e] = pad_count >= 64 ? ~uint64_t{0} : (uint64_t{1} << pad_count) - 1u;
                }
            }
            const int base = env_idx * cfg.max_cars_per_env;
            const CarControls* const env_controls = tick_controls[e];
            const int16_t* const env_order = &tick_order[e * cfg.max_cars_per_env];
            const int count = env_car_count[env_idx];
            EnvState& env = env_states[env_idx];
            const uint64_t current_tick = static_cast<uint64_t>(env.tick);
            uint32_t cal_pushed_cars_mask = 0u;
            for (int k = 0; k < count; ++k) {
                const int i = env_order[k];
                CarFrame* const frame = i < 8 ? &car_frames[i] : &snap_frame;
                frame->rb_snap = mirror_rng == nullptr && solver_scratch != nullptr && i < 8
                    ? &solver_scratch[env_idx].rb_snap[i] : nullptr;
                frame->shape = count <= 8 && i < 8 ? &car_shapes[e][i] : nullptr;
                frame->ctl_memo = i < 8 ? &ctl_memos[e * 8 + i] : nullptr;
                StepEnvVehicleUpdateCar(env_idx, i, count, cfg, cars, wheel_runtime_states, car_configs, balls,
                                        env_controls[i], nullptr, cal_pushed_cars_mask,
                                        solver_scratch != nullptr ? &solver_scratch[env_idx].bp.sweep : nullptr,
                                        frame);
            }
            BallState& ball = balls[env_idx];
            if (cfg.mutator.game_mode == static_cast<uint8_t>(Mode::Heatseeker)) {
                HeatseekerBallPreTick(ball, cfg.tick_dt, [](BallState& b, const Vec3& v_bt) { SetBallLinearVelocityBt(b, v_bt); });
            }
            WorldStepSoccarWholeHost(world_args, env_idx, base, count, solver_scratch ? solver_scratch + env_idx : nullptr,
                                     env_rows[e], ball, current_tick, count <= 8 ? car_frames : nullptr,
                                     car_shapes[e]);
            if (e + 1 < env_n) {
                prefetch_env_cars(e + 1);
            }
            for (int k = 0; k < count; ++k) {
                const int i = env_order[k];
                StepEnvPostCar(env_idx, i, cfg, cars, wheel_runtime_states, solver_scratch, car_configs,
                               env_controls[i], false, report_fields);
            }
            StepEnvTail(env_idx, count, cfg, cars, car_configs, ball, env, pads, true, true,
                        pad_pre ? &pad_masks : nullptr,
                        solver_scratch != nullptr && count <= 8 ? solver_scratch[env_idx].pad_budget : nullptr);
            pad_work_ok[e] = pad_pre;
            if (pad_pre) {
                const int pad_count = BoostPadCount(cfg);
                BoostPadWorkAfterTick(pads + env_idx * pad_count, pad_masks, pad_work[e], pad_prev[e]);
            }
        }
    }
}

}
}
#endif
