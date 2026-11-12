#include <cstdlib>
#include <rocketgpu/rocketsim_compat_devbatch.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define ROCKETGPU_STEP_TYPES_ONLY 1
#include "rocketgpu/detail/step.hpp"

namespace rocketgpu {
namespace compat {

DevBatch::DevBatch(const Arena& tmpl, int n_envs) {
    if (n_envs < 1) throw std::invalid_argument("DevBatch: n_envs < 1");
    SimConfig cfg = tmpl.sim_->config();
    cap_ = static_cast<size_t>(cfg.max_cars_per_env);
    cfg.env_count = n_envs;
    cfg.arena_mesh_tris = nullptr;
    cfg.arena_mesh_tri_count = 0;
    cfg.arena_mesh_bvh_quantization = nullptr;
    cfg.arena_mesh_bvh_meshes = nullptr;
    cfg.arena_mesh_bvh_nodes = nullptr;
    cfg.arena_mesh_bvh_subtrees = nullptr;
    cfg.arena_mesh_leaf_grid_infos = nullptr;
    cfg.arena_mesh_leaf_cell_start = nullptr;
    cfg.arena_mesh_leaf_entries = nullptr;
    cfg.arena_broadphase_cell_mask = nullptr;
    cfg.arena_broadphase_cell_count = 0;
    cfg.env_step_order = nullptr;
    cfg.mutator.cal_arena_local_tris = nullptr;
    cfg.mutator.native_sphere_rsqrt_table = nullptr;
    if (std::getenv("ROCKETGPU_DEVBATCH_NO_MIRROR") != nullptr) cfg.mirror_official_respawn_rng = 0;
    sim_ =std::make_unique<BatchSim>(cfg);
    n_ = static_cast<size_t>(n_envs);
    pads_ = sim_->boost_pad_capacity() / n_;
    std::vector<int32_t> zero(n_, 0);
    sim_->set_respawn_rng_env_indices(zero.data(), zero.size());
    cars_.assign(n_ * cap_, CarState{});
    wheels_.assign(n_ * cap_, detail::WheelRuntimeState{});
    controls_.assign(n_ * cap_, CarControls{});
    configs_.assign(n_ * cap_, CarConfig{});
    teams_.assign(n_ * cap_, Team::Blue);
    balls_.assign(n_, BallState{});
    envs_.assign(n_, EnvState{});
    scratch_.assign(n_, detail::WorldSolverScratch{});
    pad_states_.assign(n_ * pads_, BoostPadState{});
    counts_.assign(n_, 0);
}

DevBatch::~DevBatch() = default;

void DevBatch::SetEnv(int i, const Arena& a) {
    if (i < 0 || static_cast<size_t>(i) >= n_) throw std::out_of_range("DevBatch::SetEnv: env index");
    BatchSim& s = *a.sim_;
    if (s.car_capacity() != cap_ || s.boost_pad_capacity() != pads_) {
        throw std::invalid_argument("DevBatch::SetEnv: arena shape differs from the template");
    }
    const size_t e = static_cast<size_t>(i);
    s.download_car_states(cars_.data() + e * cap_, cap_);
    s.download_wheel_runtime_states(wheels_.data() + e * cap_, cap_);
    s.download_ball_states(balls_.data() + e, 1);
    s.download_env_states(envs_.data() + e, 1);
    s.download_boost_pad_states(pad_states_.data() + e * pads_, pads_);
    const uint32_t id_base = static_cast<uint32_t>(e * cap_);
    for (size_t j = 0; j < cap_; ++j) {
        CarState& c = cars_[e * cap_ + j];
        if (c.car_contact.other_car_id != 0) c.car_contact.other_car_id += id_base;
        if (c.ball_hit_info.hit_car_id != 0) c.ball_hit_info.hit_car_id += id_base;
        for (int m = 0; m < 7; ++m) {
            if (c.pending_bump_hitter[m] != 0) c.pending_bump_hitter[m] += id_base;
        }
    }
    for (size_t p = 0; p < pads_; ++p) {
        BoostPadState& pad = pad_states_[e * pads_ + p];
        if (pad.prev_locked_car_id != 0) pad.prev_locked_car_id += id_base;
        if (pad.cur_locked_car_id != 0) pad.cur_locked_car_id += id_base;
    }
    s.download_solver_scratch_range(scratch_.data() + e, 0, 1);
    if (rows_.size() < 2) {
        rows_.resize(2);
    }
    if (!rows_[0]) {
        rows_[0] = std::make_unique<detail::WorldSolverScratchRowPoolHost>();
    }
    if (!rows_[1]) {
        rows_[1] = std::make_unique<detail::WorldSolverScratchRowPoolHost>();
    }
    s.download_solver_scratch_rows(*rows_[1], 0, 1);
    const size_t n_plane = rows_[1]->car_world_plane_manifolds.size();
    const size_t n_mesh = rows_[1]->car_world_mesh_manifolds.size();
    const size_t n_car_car = rows_[1]->car_car_retained_manifolds.size();
    const size_t n_ball_car = rows_[1]->ball_car_retained_manifolds.size();
    detail::WorldSolverScratchRowPoolHost& dst = *rows_[0];
    if (dst.car_world_plane_manifolds.size() != n_ * n_plane) {
        dst.car_world_plane_manifolds.assign(n_ * n_plane, detail::CarWorldPlaneRetainedManifold{});
        dst.car_world_mesh_manifolds.assign(n_ * n_mesh, detail::CarWorldMeshRetainedManifold{});
        dst.car_car_retained_manifolds.assign(n_ * n_car_car, detail::CarCarRetainedManifold{});
        dst.ball_car_retained_manifolds.assign(n_ * n_ball_car, detail::BallCarRetainedManifold{});
    }
    std::memcpy(dst.car_world_plane_manifolds.data() + e * n_plane,
                rows_[1]->car_world_plane_manifolds.data(),
                n_plane * sizeof(detail::CarWorldPlaneRetainedManifold));
    std::memcpy(dst.car_world_mesh_manifolds.data() + e * n_mesh,
                rows_[1]->car_world_mesh_manifolds.data(),
                n_mesh * sizeof(detail::CarWorldMeshRetainedManifold));
    std::memcpy(dst.car_car_retained_manifolds.data() + e * n_car_car,
                rows_[1]->car_car_retained_manifolds.data(),
                n_car_car * sizeof(detail::CarCarRetainedManifold));
    std::memcpy(dst.ball_car_retained_manifolds.data() + e * n_ball_car,
                rows_[1]->ball_car_retained_manifolds.data(),
                n_ball_car * sizeof(detail::BallCarRetainedManifold));
    std::memcpy(configs_.data() + e * cap_, a.configs_.data(), cap_ * sizeof(CarConfig));
    std::memcpy(teams_.data() + e * cap_, a.teams_.data(), cap_ * sizeof(Team));
    std::memcpy(controls_.data() + e * cap_, a.controls_.data(), cap_ * sizeof(CarControls));
    counts_[e] = a.car_count_;
}

void DevBatch::Commit() {
    BatchSim& s = *sim_;
    std::vector<int32_t> zero(n_, 0);
    s.set_respawn_rng_env_indices(zero.data(), zero.size());
    s.set_teams(teams_.data(), teams_.size());
    s.set_car_configs(configs_.data(), configs_.size());
    s.set_env_car_count(counts_.data(), counts_.size());
    s.upload_car_states(cars_.data(), cars_.size(), nullptr, BatchSim::StateUploadMode::PreserveInternal);
    s.upload_ball_states(balls_.data(), balls_.size(), nullptr, BatchSim::StateUploadMode::PreserveInternal);
    s.upload_env_states(envs_.data(), envs_.size());
    s.upload_wheel_runtime_states(wheels_.data(), wheels_.size());
    s.upload_boost_pad_states(pad_states_.data(), pad_states_.size());
    s.upload_controls(controls_.data(), controls_.size());
    s.upload_solver_scratch_range_dev(scratch_.data(), 0, n_);
    if (!rows_.empty() && rows_[0] && !rows_[0]->car_world_plane_manifolds.empty()) {
        s.upload_solver_retained_rows_dev(*rows_[0], 0, n_);
    }
    s.synchronize();
}

void DevBatch::Step(int ticks) { sim_->step(ticks); }

void DevBatch::SetFrameFrac(const std::vector<float>& frac) {
    if (frac.size() != n_) throw std::invalid_argument("DevBatch::SetFrameFrac: one value per env");
    const MutatorConfig m = sim_->config().mutator;
    sim_->set_mutator_config(m);
    sim_->set_env_frame_frac(frac.data(), frac.size());
}

void DevBatch::SetFrameStartDt(const std::vector<float>& dt) {
    if (dt.size() != n_) throw std::invalid_argument("DevBatch::SetFrameStartDt: one value per env");
    sim_->set_env_frame_start_dt(dt.data(), dt.size());
}

std::vector<uint8_t> DevBatch::GetRaw(const std::string& what) const {
    BatchSim& s = *sim_;
    if (what == "car_states") {
        std::vector<uint8_t> out(cars_.size() * sizeof(CarState));
        s.download_car_states(reinterpret_cast<CarState*>(out.data()), cars_.size());
        return out;
    }
    if (what == "ball_states") {
        std::vector<uint8_t> out(balls_.size() * sizeof(BallState));
        s.download_ball_states(reinterpret_cast<BallState*>(out.data()), balls_.size());
        return out;
    }
    if (what == "env_states") {
        std::vector<uint8_t> out(envs_.size() * sizeof(EnvState));
        s.download_env_states(reinterpret_cast<EnvState*>(out.data()), envs_.size());
        return out;
    }
    if (what == "boost_pads") {
        std::vector<uint8_t> out(pad_states_.size() * sizeof(BoostPadState));
        s.download_boost_pad_states(reinterpret_cast<BoostPadState*>(out.data()), pad_states_.size());
        return out;
    }
    if (what == "wheel_runtime") {
        std::vector<uint8_t> out(n_ * cap_ * sizeof(detail::WheelRuntimeState));
        s.download_wheel_runtime_states(reinterpret_cast<detail::WheelRuntimeState*>(out.data()), n_ * cap_);
        return out;
    }
    throw std::invalid_argument("DevBatch::GetRaw: unknown array " + what);
}

void DevBatch::SetRaw(const std::string& what, const std::vector<uint8_t>& b) {
    BatchSim& s = *sim_;
    auto check = [&](size_t count, size_t elem_size) {
        if (b.size() != count * elem_size) {
            throw std::invalid_argument("DevBatch::SetRaw " + what + ": expected " + std::to_string(count * elem_size) + " bytes");
        }
    };
    if (what == "controls") {
        check(controls_.size(), sizeof(CarControls));
        s.upload_controls(reinterpret_cast<const CarControls*>(b.data()), controls_.size());
    } else if (what == "car_states") {
        check(cars_.size(), sizeof(CarState));
        s.upload_car_states(reinterpret_cast<const CarState*>(b.data()), cars_.size(), nullptr, BatchSim::StateUploadMode::PreserveInternal);
    } else if (what == "ball_states") {
        check(balls_.size(), sizeof(BallState));
        s.upload_ball_states(reinterpret_cast<const BallState*>(b.data()), balls_.size(), nullptr, BatchSim::StateUploadMode::PreserveInternal);
    } else if (what == "env_states") {
        check(envs_.size(), sizeof(EnvState));
        s.upload_env_states(reinterpret_cast<const EnvState*>(b.data()), envs_.size());
    } else if (what == "boost_pads") {
        check(pad_states_.size(), sizeof(BoostPadState));
        s.upload_boost_pad_states(reinterpret_cast<const BoostPadState*>(b.data()), pad_states_.size());
    } else {
        throw std::invalid_argument("DevBatch::SetRaw: unknown array " + what);
    }
}

}
}
