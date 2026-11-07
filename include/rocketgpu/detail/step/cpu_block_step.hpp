#pragma once
#if !defined(ROCKETGPU_USE_CUDA)
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(ROCKETGPU_CPU_ISPC)
struct RgPreTickParams {
    float dt, car_mass, flip_torque_time, supersonic_start_speed, supersonic_maintain_max_time,
        supersonic_maintain_min_speed, car_max_speed, car_max_ang_speed;
};
extern "C" void rg_pretick_cars(::rocketgpu::CarState* cars, ::rocketgpu::detail::WheelRuntimeState* wrs,
                                const int32_t* idx, const float* frame_frac, int32_t n, const RgPreTickParams& p,
                                ::rocketgpu::Vec3* start_vel, ::rocketgpu::Quat* snap_rot, ::rocketgpu::Quat* snap_raw);
extern "C" void rg_pretick_balls(const ::rocketgpu::BallState* balls, int32_t n, float max_speed, float max_ang_speed,
                                 bool heatseeker, bool damp, float damp_scale, float* out);
#endif

namespace rocketgpu {
namespace detail {

constexpr int kCpuBlockMaxEnvs = 64;

#if defined(ROCKETGPU_CPU_ISPC)
constexpr int kCpuBlockMinEnvs = 4;

constexpr int kBallOutFields = 43;
constexpr int kBallOutStartVel = 37;
constexpr int kBallOutPredictVel = 40;
inline bool SameVec3(const Vec3& a, const Vec3& b) { return std::memcmp(&a, &b, sizeof(Vec3)) == 0; }
inline Vec3 BallOutV3(const float* o, int f) {
    return Vec3(o[f * kCpuBlockMaxEnvs], o[(f + 1) * kCpuBlockMaxEnvs], o[(f + 2) * kCpuBlockMaxEnvs]);
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
    return b < lo ? lo : (b > 16 ? 16 : b);
}

constexpr int kCpuTailEnvsPerWorker = 2;
inline int CpuTailEnvs(int env_count, int workers, int block) {
    const int tail = kCpuTailEnvsPerWorker * workers;
    return workers > 1 && block > 1 && env_count >= 8 * tail ? tail : 0;
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
    Vec3 lane_start_vel[kMaxCars];
    Quat lane_snap_rot[kMaxCars];
    Quat lane_snap_raw[kMaxCars];
    CarFrame snap_frame;
    const ArenaBroadphaseGrid grid = MakeArenaBroadphaseGrid(cfg);
    CarFrame car_frames[8];
    const CarShapeCacheEntry* car_shapes[kCpuBlockMaxEnvs];
    UninitArray<CarShapeCacheEntry, kCpuBlockMaxEnvs * 8> car_shapes_local;
    UninitArray<WorldSolverScratchRows, kCpuBlockMaxEnvs> env_rows;
    UninitArray<CarControls, kMaxCars> tick_controls;
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
#else
    const bool lanes = false;
    const bool ball_lanes = false;
#endif
    for (int t = 0; t < ticks; ++t) {
        int n = 0;
        for (int e = 0; e < env_n; ++e) {
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
                } else {
                    StepEnvPreTickEnvState(env_idx, cfg, env_car_count, cars, balls, env_states);
                }
            } else {
                StepEnvPreTickEnvState(env_idx, cfg, env_car_count, cars, balls, env_states);
            }
            const int pre_count = env_car_count[env_idx];
            pre_counts[e] = pre_count;
            CarControls* const env_controls = &tick_controls[e * cfg.max_cars_per_env];
            int16_t* const env_order = &tick_order[e * cfg.max_cars_per_env];
            const float env_frac = FrameFrac(cfg, env_idx);
            for (int k = 0; k < pre_count; ++k) {
                env_order[k] = static_cast<int16_t>(CarVisitSlot(env_states[env_idx], k, pre_count));
            }
            for (int k = 0; k < pre_count; ++k) {
                const int slot = env_order[k];
                const CarState& car = cars[base + slot];
                const CarControls c = controls.get(slot, t);
                env_controls[slot] = c;
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
        if (n > 0) {
            if (CpuIspcCheck()) {
                static thread_local CarState got[kMaxCars];
                static thread_local WheelRuntimeState got_w[kMaxCars];
                static thread_local CarState start[kMaxCars];
                static thread_local WheelRuntimeState start_w[kMaxCars];
                for (int j = 0; j < n; ++j) {
                    start[j] = cars[lane_idx[j]];
                    start_w[j] = wheel_runtime_states[lane_idx[j]];
                }
                rg_pretick_cars(cars, wheel_runtime_states, lane_idx, lane_frac, n, pre_params, lane_start_vel, lane_snap_rot,
                                lane_snap_raw);
                for (int j = 0; j < n; ++j) {
                    got[j] = cars[lane_idx[j]];
                    got_w[j] = wheel_runtime_states[lane_idx[j]];
                    cars[lane_idx[j]] = start[j];
                    wheel_runtime_states[lane_idx[j]] = start_w[j];
                    const int env_idx = lane_env[j], slot = lane_slot[j];
                    StepEnvPreTickCar(env_idx, slot, cfg, teams, cars, solver_scratch, env_states[env_idx], respawn_overrides[env_idx - env0],
                                      seed, controls_for(env_idx).get(slot, t), car_configs[lane_idx[j]],
                                      &wheel_runtime_states[lane_idx[j]]);
                    if (std::memcmp(&got[j], &cars[lane_idx[j]], sizeof(CarState)) != 0 ||
                        std::memcmp(&got_w[j], &wheel_runtime_states[lane_idx[j]], sizeof(WheelRuntimeState)) != 0 ||
                        (solver_scratch != nullptr && slot < 8 &&
                         std::memcmp(&lane_start_vel[j], &solver_scratch[env_idx].cal_start_vel[slot], sizeof(Vec3)) != 0)) {
                        std::fprintf(stderr, "ROCKETGPU_ISPC_CHECK pre-tick mismatch, car %d\n", lane_idx[j]);
                    }
                    const CarState& cc = cars[lane_idx[j]];
                    const Quat raw = GameGetRotationScalar(cc.basis_forward, cc.basis_right, cc.basis_up);
                    const Quat rot = GameRbStateQuantizeQuat(raw);
                    if (std::memcmp(&raw, &lane_snap_raw[j], sizeof(Quat)) != 0 ||
                        std::memcmp(&rot, &lane_snap_rot[j], sizeof(Quat)) != 0) {
                        std::fprintf(stderr, "ROCKETGPU_ISPC_CHECK pre-tick snap rotation mismatch, car %d\n", lane_idx[j]);
                    }
                }
            } else {
                rg_pretick_cars(cars, wheel_runtime_states, lane_idx, lane_frac, n, pre_params, lane_start_vel, lane_snap_rot,
                                lane_snap_raw);
                if (solver_scratch != nullptr) {
                    for (int j = 0; j < n; ++j) {
                        const int slot = lane_slot[j];
                        if (slot < 8) {
                            WorldSolverScratch& s = solver_scratch[lane_env[j]];
                            const CarState& car = cars[lane_idx[j]];
                            s.cal_start_vel[slot] = lane_start_vel[j];
                            s.car_predict_vel[slot] = car.vel;
                            s.car_predict_ang_vel[slot] = car.ang_vel;
                            s.car_predict_valid[slot] = 1;
                            if (mirror_rng == nullptr) {
                                CarRbSnapMemo& snap = s.rb_snap[slot];
                                snap.fwd = car.basis_forward;
                                snap.right = car.basis_right;
                                snap.up = car.basis_up;
                                snap.rot = lane_snap_rot[j];
                                snap.raw = lane_snap_raw[j];
                                snap.valid = 1;
                            }
                        }
                    }
                }
            }
        }
        if (ball_lanes) {
            if (CpuIspcCheck()) {
                static thread_local BallState start[kCpuBlockMaxEnvs];
                static thread_local BallState got[kCpuBlockMaxEnvs];
                std::memcpy(start, balls + env0, sizeof(BallState) * env_n);
                rg_pretick_balls(balls + env0, env_n, m.ball_max_speed, m.ball_max_ang_speed, ball_heatseeker, ball_damp,
                                 ball_damp_scale, ball_out);
                for (int e = 0; e < env_n; ++e) {
                    StoreBallPreTickOut(balls[env0 + e], ball_out + e);
                }
                std::memcpy(got, balls + env0, sizeof(BallState) * env_n);
                std::memcpy(balls + env0, start, sizeof(BallState) * env_n);
                for (int e = 0; e < env_n; ++e) {
                    StepEnvPreTickBall(env0 + e, cfg, solver_scratch, balls);
                    if (std::memcmp(&got[e], &balls[env0 + e], sizeof(BallState)) != 0 ||
                        (solver_scratch != nullptr &&
                         (!SameVec3(BallOutV3(ball_out + e, kBallOutStartVel), solver_scratch[env0 + e].cal_ball_start_vel) ||
                          (!ball_heatseeker && !SameVec3(BallOutV3(ball_out + e, kBallOutPredictVel),
                                                       solver_scratch[env0 + e].ball_predict_vel_bt))))) {
                        std::fprintf(stderr, "ROCKETGPU_ISPC_CHECK ball pre-tick mismatch, env %d\n", env0 + e);
                    }
                }
            } else {
                rg_pretick_balls(balls + env0, env_n, m.ball_max_speed, m.ball_max_ang_speed, ball_heatseeker, ball_damp,
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
            BoostPadTickMasks pad_masks;
            const bool pad_pre = cfg.enable_boost_pads && pre_counts[e] > 0;
            if (pad_pre) {
                const int pad_count = BoostPadCount(cfg);
                PreTickBoostPadsHostMasks(pads + env_idx * pad_count, pad_count, cfg.tick_dt, cfg.tick_dt > 0.0f,
                                          pad_masks);
            }
            const int base = env_idx * cfg.max_cars_per_env;
            const CarControls* const env_controls = &tick_controls[e * cfg.max_cars_per_env];
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
            for (int k = 0; k < count; ++k) {
                const int i = env_order[k];
                StepEnvPostCar(env_idx, i, cfg, cars, wheel_runtime_states, solver_scratch, car_configs,
                               env_controls[i], false, report_fields);
            }
            StepEnvTail(env_idx, count, cfg, cars, car_configs, ball, env, pads, true, true,
                        pad_pre ? &pad_masks : nullptr);
        }
    }
}

}
}
#endif
