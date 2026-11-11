#include <rocketgpu/rocketgpu.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <sched.h>
#endif

#if defined(ROCKETGPU_USE_CUDA)
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)
#endif
#if defined(__CUDACC__)
#define HD __host__ __device__
#else
#define HD
#endif

using namespace rocketgpu;

namespace {
constexpr int kTickSkip = 8, kEpisodeSteps = 900, kGroundRows = 24, kActions = 90;
constexpr float kGroundFrac = 0.75f;
constexpr float kTeamSizeCum[2] = {0.5478515625f, 0.5478515625f + 0.271484375f};
constexpr uint32_t kOppWeights[6] = {446064, 20470, 500000, 11547, 11120, 10799};
CarConfig (*const kOppConfigs[6])() = {OctaneConfig, DominusConfig, HybridConfig, BreakoutConfig, PlankConfig, MercConfig};

HD inline uint64_t MulHi(uint64_t a, uint64_t b) {
#if defined(__CUDA_ARCH__)
    return __umul64hi(a, b);
#else
    return (uint64_t)(((unsigned __int128)a * b) >> 64);
#endif
}

struct Rng {
    uint64_t s;
    HD uint64_t u64() {
        s += 0x2d358dccaa6c78a5ull;
        const uint64_t b = s ^ 0x8bb84b93962eacc9ull;
        return (s * b) ^ MulHi(s, b);
    }
    HD uint32_t mod32(uint32_t n) {
        uint64_t m = (uint64_t)(uint32_t)u64() * n;
        if ((uint32_t)m < n) {
            const uint32_t t = (0u - n) % n;
            while ((uint32_t)m < t) m = (uint64_t)(uint32_t)u64() * n;
        }
        return (uint32_t)(m >> 32);
    }
    HD uint64_t mod64(uint64_t n) {
        uint64_t r = u64();
        if (r * n < n) {
            const uint64_t t = (0ull - n) % n;
            while (r * n < t) r = u64();
        }
        return MulHi(r, n);
    }
    HD float f32() {
        for (;;) {
            const float x = (float)(int64_t)(u64() >> 1) * (1.0f / 9223372036854775808.0f);
            if (x < 1.0f) return x;
        }
    }
};

struct Env {
    Rng rng;
    uint64_t seed;
    uint64_t resets;
    uint32_t age;
    int32_t cars;
    int8_t held[6];
    uint64_t base;
    uint32_t len, pos;
};

HD inline void DrawEnv(Env& v, int e, int n, const CarControls* table, CarControls* ctrl, bool fused) {
    for (int i = 0; i < v.cars; ++i) {
        const int row = v.rng.f32() < kGroundFrac ? (int)v.rng.mod64(kGroundRows)
                                                   : kGroundRows + (int)v.rng.mod64(kActions - kGroundRows);
        int onset = (int)v.rng.mod64(kTickSkip);
        if (fused) onset = 0;
        const CarControls held = v.held[i] < 0 ? CarControls{} : table[v.held[i]];
        for (int s = 0; s < kTickSkip; ++s) ctrl[((size_t)s * n + e) * 6 + i] = s >= onset ? table[row] : held;
        v.held[i] = (int8_t)row;
    }
}

HD inline void DrawReplay(Env& v, int e, int n, const CarControls* rows, const uint16_t* idx, CarControls* ctrl) {
    for (int s = 0; s < kTickSkip; ++s) {
        const uint16_t* t = idx + v.base + (size_t)((v.pos + s) % v.len) * v.cars;
        for (int i = 0; i < v.cars; ++i) ctrl[((size_t)s * n + e) * 6 + i] = rows[t[i]];
    }
    v.pos = (v.pos + kTickSkip) % v.len;
}

HD inline void DoneEnv(Env& v, bool scored, EnvCommand& cmd) {
    cmd = EnvCommand{};
    if (scored || ++v.age >= (uint32_t)kEpisodeSteps) {
        cmd.reset_to_kickoff = cmd.reset_scoreboard = cmd.reset_tick_count = 1;
        cmd.kickoff_seed = v.seed + ++v.resets * 9176;
        for (int i = 0; i < 6; ++i) v.held[i] = -1;
        v.age = 0;
    }
}

#if defined(ROCKETGPU_USE_CUDA)
__global__ void DrawKernel(Env* envs, int n, const CarControls* table, CarControls* ctrl, bool fused) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n) DrawEnv(envs[e], e, n, table, ctrl, fused);
}

__global__ void ReplayKernel(Env* envs, int n, const CarControls* rows, const uint16_t* idx, CarControls* ctrl) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n) DrawReplay(envs[e], e, n, rows, idx, ctrl);
}

__global__ void DoneKernel(Env* envs, int n, const EnvState* es, EnvCommand* cmds, unsigned long long* goals) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    const bool scored = es[e].blue_score > 0 || es[e].orange_score > 0;
    if (scored) atomicAdd(goals, 1ull);
    DoneEnv(envs[e], scored, cmds[e]);
}
#endif

std::vector<CarControls> ActionTable() {
    std::vector<CarControls> rows;
    auto add = [&](float throttle, float steer, float pitch, float yaw, float roll, bool jump, bool boost, bool hb) {
        CarControls c;
        c.throttle = throttle; c.steer = steer; c.pitch = pitch; c.yaw = yaw; c.roll = roll;
        c.jump = jump; c.boost = boost; c.handbrake = hb;
        rows.push_back(c);
    };
    for (float throttle : {-1.f, 0.f, 1.f})
        for (float steer : {-1.f, 0.f, 1.f})
            for (int boost : {0, 1})
                for (int hb : {0, 1})
                    if (!(boost && throttle != 1.f)) add(boost ? 1.f : throttle, steer, 0, steer, 0, false, boost, hb);
    for (float pitch : {-1.f, 0.f, 1.f})
        for (float yaw : {-1.f, 0.f, 1.f})
            for (float roll : {-1.f, 0.f, 1.f})
                for (int jump : {0, 1})
                    for (int boost : {0, 1}) {
                        if ((jump && yaw != 0.f) || (pitch == 0.f && roll == 0.f && !jump)) continue;
                        add(boost ? 1.f : 0.f, yaw, pitch, yaw, roll, jump, boost, jump && (pitch != 0.f || yaw != 0.f || roll != 0.f));
                    }
    if (rows.size() != kActions) std::abort();
    return rows;
}

class Pool {
public:
    explicit Pool(int count) : count_(count) {
        for (int i = 1; i < count; ++i) threads_.emplace_back([this, i] { work(i); });
    }
    ~Pool() {
        stop_ = true;
        gen_.fetch_add(1);
        gen_.notify_all();
        for (auto& t : threads_) t.join();
    }
    void run(const std::function<void(int)>& fn) {
        fn_ = &fn;
        done_.store(0);
        gen_.fetch_add(1);
        gen_.notify_all();
        fn(0);
        while (done_.load() != count_ - 1) std::this_thread::yield();
    }

private:
    void work(int i) {
        uint32_t seen = 0;
        for (;;) {
            gen_.wait(seen);
            seen = gen_.load();
            if (stop_) return;
            (*fn_)(i);
            done_.fetch_add(1);
        }
    }
    int count_;
    std::vector<std::thread> threads_;
    std::atomic<uint32_t> gen_{0};
    std::atomic<int> done_{0};
    std::atomic<bool> stop_{false};
    const std::function<void(int)>* fn_ = nullptr;
};

struct Replay {
    std::vector<CarControls> rows;
    std::vector<uint16_t> idx;
    std::vector<std::pair<uint64_t, uint32_t>> matches[4];
};

Replay LoadReplay(const char* path) {
    Replay r;
    FILE* f = std::fopen(path, "rb");
    auto u32 = [&] {
        uint32_t x;
        if (std::fread(&x, 4, 1, f) != 1) { std::fprintf(stderr, "%s: truncated\n", path); std::exit(1); }
        return x;
    };
    if (!f || u32() != 0x43524752u || u32() != 1) { std::fprintf(stderr, "%s: not a replay file\n", path); std::exit(1); }
    std::vector<float> v((size_t)u32() * 8);
    if (std::fread(v.data(), 4, v.size(), f) != v.size()) std::exit(1);
    for (size_t i = 0; i < v.size(); i += 8) {
        CarControls c;
        c.throttle = v[i]; c.steer = v[i + 1]; c.pitch = v[i + 2]; c.yaw = v[i + 3]; c.roll = v[i + 4];
        c.jump = v[i + 5] != 0; c.boost = v[i + 6] != 0; c.handbrake = v[i + 7] != 0;
        r.rows.push_back(c);
    }
    for (uint32_t m = u32(); m > 0; --m) {
        const uint32_t k = u32(), ticks = u32();
        if (k < 1 || k > 3) std::exit(1);
        r.matches[k].push_back({r.idx.size(), ticks});
        r.idx.resize(r.idx.size() + (size_t)ticks * 2 * k);
        if (std::fread(r.idx.data() + r.matches[k].back().first, 2, (size_t)ticks * 2 * k, f) != (size_t)ticks * 2 * k) std::exit(1);
    }
    std::fclose(f);
    for (int k = 1; k <= 3; ++k)
        if (r.matches[k].empty()) { std::fprintf(stderr, "%s: no %dv%d matches\n", path, k, k); std::exit(1); }
    return r;
}

const char* StrArg(int argc, char** argv, const char* name) {
    for (int i = 1; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], name)) return argv[i + 1];
    return nullptr;
}

long Arg(int argc, char** argv, const char* name, long def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], name)) return std::atol(argv[i + 1]);
    return def;
}

bool Flag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], name)) return true;
    return false;
}

void PinThreads() {
#if defined(__linux__)
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
    std::vector<int> cpus, tids;
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &allowed)) cpus.push_back(c);
    if (DIR* d = opendir("/proc/self/task")) {
        while (dirent* e = readdir(d))
            if (e->d_name[0] != '.') tids.push_back(std::atoi(e->d_name));
        closedir(d);
    }
    std::sort(tids.begin(), tids.end());
    for (size_t i = 0; i < tids.size(); ++i) {
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(cpus[i % cpus.size()], &one);
        sched_setaffinity(tids[i], sizeof(one), &one);
    }
#endif
}

double LoadAvg() {
    double l = -1;
    if (FILE* f = std::fopen("/proc/loadavg", "r")) {
        if (std::fscanf(f, "%lf", &l) != 1) l = -1;
        std::fclose(f);
    }
    return l;
}
}

int main(int argc, char** argv) {
    const auto t_start = std::chrono::steady_clock::now();
    const int n = (int)Arg(argc, argv, "--arenas", 1024);
    const int draw_threads = (int)std::clamp<long>(Arg(argc, argv, "--draw-threads", 1), 1, n);
    const int repeats = (int)Arg(argc, argv, "--repeats", 3);
    const int warmup = (int)Arg(argc, argv, "--warmup", 16);
    const long target = Arg(argc, argv, "--target-arena-ticks", 40000000);
    const long max_steps = Arg(argc, argv, "--max-steps", 6000);
    const bool fused = Flag(argc, argv, "--fused"), stats = Flag(argc, argv, "--stats"), pin = Flag(argc, argv, "--pin");
    const bool custom_arena = Flag(argc, argv, "--custom-arena");
    const char* replay_path = StrArg(argc, argv, "--replay");
    const Replay replay = replay_path ? LoadReplay(replay_path) : Replay{};
    const int steps = (int)std::clamp((target + (long)n * kTickSkip - 1) / ((long)n * kTickSkip), 8L, max_steps);

    const std::vector<CarControls> table = ActionTable();
    uint32_t opp_total = 0;
    for (uint32_t w : kOppWeights) opp_total += w;

    std::vector<Env> envs(n);
    std::vector<int32_t> car_count(n);
    std::vector<Team> teams((size_t)n * 6);
    std::vector<CarConfig> configs((size_t)n * 6, HybridConfig());
    int mix[4] = {};
    for (int e = 0; e < n; ++e) {
        Env& v = envs[e];
        v.seed = (uint64_t)e * 1000003 + 7;
        v.rng.s = v.seed;
        v.resets = 0;
        const float u = v.rng.f32();
        const int k = u < kTeamSizeCum[0] ? 1 : u < kTeamSizeCum[1] ? 2 : 3;
        ++mix[k];
        v.cars = car_count[e] = 2 * k;
        for (int s = 0; s < 6; ++s) teams[(size_t)e * 6 + s] = s < k ? Team::Blue : Team::Orange;
        for (int s = k; s < 2 * k; ++s) {
            uint32_t r = v.rng.mod32(opp_total);
            int c = 0;
            while (r >= kOppWeights[c]) r -= kOppWeights[c++];
            configs[(size_t)e * 6 + s] = kOppConfigs[c]();
        }
        v.age = v.rng.mod32(kEpisodeSteps);
        std::fill(v.held, v.held + 6, (int8_t)-1);
        if (replay_path) {
            const auto& ms = replay.matches[k];
            const auto& m = ms[v.rng.mod32((uint32_t)ms.size())];
            v.base = m.first;
            v.len = m.second;
            v.pos = v.rng.mod32(m.second);
        }
    }

    SimConfig cfg;
    cfg.env_count = n;
    cfg.max_cars_per_env = 6;
    ArenaLayout layout;
    if (custom_arena) {
        const char* mesh_root = std::getenv("ROCKETGPU_COLLISION_MESH_DIR");
        layout.mesh_dir = std::string(mesh_root ? mesh_root : ".") + "/soccar";
        layout.planes = ArenaLayout::OfficialPlanes();
        for (ArenaGoal& g : layout.goals) {
            g.min = Vec3(-893.0f, -1.0e30f, -1.0e30f);
            g.max = Vec3(893.0f, 1.0e30f, 642.775f);
        }
        BoostPadConfig pad;
        pad.pos = Vec3(0.0f, 2000.0f, 73.0f);
        pad.is_big = 1;
        layout.boost_pads.push_back(pad);
    }
    BatchSim sim = custom_arena ? BatchSim(cfg, layout) : BatchSim(cfg);
    sim.set_env_car_count(car_count.data(), n);
    sim.set_teams(teams.data(), teams.size());
    std::vector<EnvState> states(n);
    for (EnvState& s : states) s.car_visit_order_mode = 0;
    sim.upload_env_states(states.data(), n);
    sim.set_car_configs(configs.data(), configs.size());
    sim.reset_to_kickoff(0);
    std::vector<EnvCommand> cmds(n);
    for (int e = 0; e < n; ++e) {
        cmds[e].reset_to_kickoff = cmds[e].reset_scoreboard = cmds[e].reset_tick_count = 1;
        cmds[e].kickoff_seed = envs[e].seed;
    }
    sim.apply_env_commands(cmds.data(), n);

    const size_t ctrl_count = (size_t)kTickSkip * n * 6;
#if defined(ROCKETGPU_USE_CUDA)
    Env* envs_dev;
    CarControls *table_dev, *ctrl;
    EnvCommand* cmds_dev;
    unsigned long long* goals_dev;
    CK(cudaMalloc(&envs_dev, n * sizeof(Env)));
    CK(cudaMalloc(&table_dev, kActions * sizeof(CarControls)));
    CK(cudaMalloc(&ctrl, ctrl_count * sizeof(CarControls)));
    CK(cudaMalloc(&cmds_dev, n * sizeof(EnvCommand)));
    CK(cudaMalloc(&goals_dev, sizeof(unsigned long long)));
    CK(cudaMemcpy(envs_dev, envs.data(), n * sizeof(Env), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(table_dev, table.data(), kActions * sizeof(CarControls), cudaMemcpyHostToDevice));
    CK(cudaMemset(ctrl, 0, ctrl_count * sizeof(CarControls)));
    CK(cudaMemset(goals_dev, 0, sizeof(unsigned long long)));
    CarControls* rows_dev = nullptr;
    uint16_t* idx_dev = nullptr;
    if (replay_path) {
        CK(cudaMalloc(&rows_dev, replay.rows.size() * sizeof(CarControls)));
        CK(cudaMalloc(&idx_dev, replay.idx.size() * sizeof(uint16_t)));
        CK(cudaMemcpy(rows_dev, replay.rows.data(), replay.rows.size() * sizeof(CarControls), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(idx_dev, replay.idx.data(), replay.idx.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    }
    const EnvState* es_dev = sim.device_view().env_states;
    const int blocks = (n + 127) / 128;
    double phase[3] = {};
    auto env_step = [&] {
        if (replay_path) ReplayKernel<<<blocks, 128>>>(envs_dev, n, rows_dev, idx_dev, ctrl);
        else DrawKernel<<<blocks, 128>>>(envs_dev, n, table_dev, ctrl, fused);
        sim.step_with_device_controls(ctrl, kTickSkip, n * 6);
        DoneKernel<<<blocks, 128>>>(envs_dev, n, es_dev, cmds_dev, goals_dev);
        sim.apply_env_commands_device(cmds_dev, n);
    };
    auto fetch = [&](unsigned long long& goals) {
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(envs.data(), envs_dev, n * sizeof(Env), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&goals, goals_dev, sizeof(goals), cudaMemcpyDeviceToHost));
    };
#else
    std::vector<CarControls> ctrl(ctrl_count);
    Pool pool(draw_threads);
    const int chunk = (n + draw_threads - 1) / draw_threads;
    const std::function<void(int)> draw = [&](int t) {
        for (int e = t * chunk; e < std::min(n, (t + 1) * chunk); ++e) {
            if (replay_path) DrawReplay(envs[e], e, n, replay.rows.data(), replay.idx.data(), ctrl.data());
            else DrawEnv(envs[e], e, n, table.data(), ctrl.data(), fused);
        }
    };
    unsigned long long goals_host = 0;
    double phase[3] = {};
    auto env_step = [&] {
        const auto t0 = stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (draw_threads > 1) pool.run(draw);
        else draw(0);
        const auto t1 = stats ? std::chrono::steady_clock::now() : t0;
        sim.step_with_device_controls(ctrl.data(), kTickSkip, n * 6);
        const auto t2 = stats ? std::chrono::steady_clock::now() : t0;
        const EnvState* es = sim.device_view().env_states;
        for (int e = 0; e < n; ++e) {
            const bool scored = es[e].blue_score > 0 || es[e].orange_score > 0;
            goals_host += scored;
            DoneEnv(envs[e], scored, cmds[e]);
        }
        sim.apply_env_commands_device(cmds.data(), n);
        if (stats) {
            const auto t3 = std::chrono::steady_clock::now();
            phase[0] += std::chrono::duration<double>(t1 - t0).count();
            phase[1] += std::chrono::duration<double>(t2 - t1).count();
            phase[2] += std::chrono::duration<double>(t3 - t2).count();
        }
    };
    auto fetch = [&](unsigned long long& goals) { goals = goals_host; };
#endif

    std::vector<CarState> cs((size_t)n * 6);
    std::vector<BallState> bs(n);
    uint64_t last_resets = 0, last_goals = 0;
    auto print_stats = [&](const char* when) {
        unsigned long long goals;
        fetch(goals);
        sim.download_car_states(cs.data(), cs.size());
        uint64_t resets = 0;
        int cars = 0, contact = 0, ground = 0, demoed = 0;
        double speed = 0;
        for (int e = 0; e < n; ++e) {
            resets += envs[e].resets;
            for (int i = 0; i < envs[e].cars; ++i) {
                const CarState& c = cs[(size_t)e * 6 + i];
                ++cars;
                contact += c.car_contact.cooldown_timer > 0.0f;
                ground += c.is_on_ground;
                demoed += c.is_demoed;
                speed += std::sqrt(c.vel.x * c.vel.x + c.vel.y * c.vel.y + c.vel.z * c.vel.z);
            }
        }
        std::printf("    %s t=%.2fs load %.2f: resets +%llu goals +%llu | car-car cooldown %.2f%%, on ground %.1f%%, demoed %.2f%%, "
                    "mean speed %.0f | draw %.3fs step %.3fs check+reset %.3fs\n", when,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(), LoadAvg(),
                    (unsigned long long)(resets - last_resets), (unsigned long long)(goals - last_goals), 100.0 * contact / cars,
                    100.0 * ground / cars, 100.0 * demoed / cars, speed / cars, phase[0], phase[1], phase[2]);
        phase[0] = phase[1] = phase[2] = 0;
        last_resets = resets;
        last_goals = goals;
    };

    for (int i = 0; i < warmup; ++i) {
        env_step();
        if (pin && i == 0) PinThreads();
    }
    sim.synchronize();
    const char* sim_threads = std::getenv("ROCKETGPU_CPU_THREADS");
    std::printf("arenas %d (1v1 %d, 2v2 %d, 3v3 %d), %s, draw threads %d, ROCKETGPU_CPU_THREADS %s, %d env steps x %d ticks "
                "per repeat, warmup %d\n", n, mix[1], mix[2], mix[3], replay_path ? "replay" : fused ? "fused" : "training", draw_threads,
                sim_threads ? sim_threads : "unset", steps, kTickSkip, warmup);
    if (stats) print_stats("before");
    std::vector<double> rates;
    for (int r = 0; r < repeats; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < steps; ++i) env_step();
        sim.synchronize();
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        rates.push_back((double)n * steps * kTickSkip / dt);
        std::printf("  repeat %d: %.0f arena-ticks/s (%.2f s)\n", r + 1, rates.back(), dt);
        if (stats) print_stats("after");
    }
    const double first = rates[0];
    std::sort(rates.begin(), rates.end());
    const double med = rates[rates.size() / 2];

    unsigned long long goals;
    fetch(goals);
    sim.download_car_states(cs.data(), cs.size());
    sim.download_ball_states(bs.data(), n);
    uint64_t h = 0xcbf29ce484222325ull;
    auto mix_f = [&](const float* p, int k) {
        for (int i = 0; i < k; ++i) {
            uint32_t b;
            std::memcpy(&b, p + i, 4);
            h = (h ^ b) * 0x100000001b3ull;
        }
    };
    uint64_t resets = 0;
    for (int e = 0; e < n; ++e) {
        for (int i = 0; i < envs[e].cars; ++i) {
            const CarState& c = cs[(size_t)e * 6 + i];
            mix_f(&c.pos.x, 3); mix_f(&c.vel.x, 3); mix_f(&c.ang_vel.x, 3); mix_f(&c.rot.w, 4);
        }
        mix_f(&bs[e].pos.x, 3); mix_f(&bs[e].vel.x, 3); mix_f(&bs[e].ang_vel.x, 3);
        resets += envs[e].resets;
    }
    std::printf("median %.0f arena-ticks/s, spread %.1f%%, first %.0f, goals %llu, resets %llu, state hash %016llx\n", med,
                (rates.back() - rates.front()) / med * 100.0, first, goals, (unsigned long long)resets, (unsigned long long)h);
#if defined(ROCKETGPU_USE_CUDA)
    cudaFree(envs_dev);
    cudaFree(table_dev);
    cudaFree(ctrl);
    cudaFree(cmds_dev);
    cudaFree(goals_dev);
    cudaFree(rows_dev);
    cudaFree(idx_dev);
#endif
    return 0;
}
