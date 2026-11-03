// Dev only (full-state suite): N restored compat arenas' env 0 copied into one BatchSim and stepped together.
#pragma once

#include <rocketgpu/rocketsim_compat.h>

#include <memory>
#include <string>
#include <vector>

namespace rocketgpu {
namespace detail {
struct WheelRuntimeState;
struct WorldSolverScratch;
struct WorldSolverScratchRowPoolHost;
}
namespace compat {

class DevBatch {
public:
    // n_envs envs shaped like tmpl (its SimConfig: mode, mutators, car capacity, pads); respawn picks as env 0's.
    DevBatch(const Arena& tmpl, int n_envs);
    ~DevBatch();
    // Stage env i from arena a's env 0: car / ball / env / wheel / pad states, solver scratch and retained manifolds,
    // car configs, teams, controls and car count. Commit() uploads every staged env.
    void SetEnv(int i, const Arena& a);
    void Commit();
    void Step(int ticks = 1);
    // Each env's render frame frac for the next steps (one per env), as a single-env arena's per-tick
    // set_mutator_config(boost_pad_frame_frac = f) sets it; that call also bumps the settled-ball epoch, so this does.
    void SetFrameFrac(const std::vector<float>& frac);
    // Each env's frame_start_dt for the next steps (one per env; MutatorConfig.countdown_on_frame_start).
    void SetFrameStartDt(const std::vector<float>& dt);
    // Whole arrays over all envs: "car_states" (n * cap), "ball_states", "env_states" (n), "boost_pads" (n * pads), "wheel_runtime" (n * cap);
    // SetRaw also takes "controls" (n * cap CarControls).
    std::vector<uint8_t> GetRaw(const std::string& what) const;
    void SetRaw(const std::string& what, const std::vector<uint8_t>& bytes);
    int EnvCount() const { return static_cast<int>(n_); }
    int CarCapacity() const { return static_cast<int>(cap_); }
    int PadCount() const { return static_cast<int>(pads_); }

private:
    std::unique_ptr<BatchSim> sim_;
    size_t n_ = 0, cap_ = 0, pads_ = 0;
    mutable std::vector<CarState> cars_;
    std::vector<detail::WheelRuntimeState> wheels_;
    std::vector<CarControls> controls_;
    std::vector<CarConfig> configs_;
    std::vector<Team> teams_;
    mutable std::vector<BallState> balls_;
    mutable std::vector<EnvState> envs_;
    std::vector<detail::WorldSolverScratch> scratch_;
    std::vector<std::unique_ptr<detail::WorldSolverScratchRowPoolHost>> rows_;
    mutable std::vector<BoostPadState> pad_states_;
    std::vector<int32_t> counts_;
};

}  // namespace compat
}  // namespace rocketgpu
