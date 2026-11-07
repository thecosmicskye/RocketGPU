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
    DevBatch(const Arena& tmpl, int n_envs);
    ~DevBatch();
    void SetEnv(int i, const Arena& a);
    void Commit();
    void Step(int ticks = 1);
    void SetFrameFrac(const std::vector<float>& frac);
    void SetFrameStartDt(const std::vector<float>& dt);
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

}
}
