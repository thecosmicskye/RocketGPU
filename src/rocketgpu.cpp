#include "rocketgpu/rocketgpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
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
#include <thread>
#include <type_traits>
#include <vector>
#if defined(__linux__)
#include <linux/futex.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include "rocketgpu/detail/arena/mesh_loader.hpp"
#include "rocketgpu/detail/arena/mesh_path.hpp"
#include "rocketgpu/detail/buffer.hpp"
#include "rocketgpu/detail/cuda_api.hpp"
#include "rocketgpu/detail/respawn_rng.hpp"
#include "rocketgpu/detail/step.hpp"
#include "rocketgpu/detail/step/cpu_block_step.hpp"
#if defined(ROCKETGPU_CPU_ISPC)
#include "cpu_ispc/rg_layout_check.inc"
#endif
#include "rocketgpu/detail/native_sse_table.hpp"

namespace {
constexpr uint8_t kNoRespawnSlotOverride = 0xFFu;

#if !defined(ROCKETGPU_USE_CUDA)
inline void CpuRelax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

inline int DefaultCpuHardwareThreads() {
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0) {
        const int count = CPU_COUNT(&cpuset);
        if (count > 0) {
            return count;
        }
    }
#elif defined(__APPLE__)
    int perf_cores = 0;
    size_t len = sizeof(perf_cores);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &perf_cores, &len, nullptr, 0) == 0 && perf_cores > 0) {
        return perf_cores;
    }
#endif
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
}

class CpuEnvPool {
public:
    explicit CpuEnvPool(int env_count) {
        int n = 0;
        if (const char* s = std::getenv("ROCKETGPU_CPU_THREADS")) {
            n = std::atoi(s);
        }
        if (n <= 0) {
            int hw = DefaultCpuHardwareThreads();
#if !defined(__APPLE__)
            if (hw >= 8) {
                hw -= 2;
            }
#endif
            n = std::min(hw, std::max(1, env_count / 4));
        }
        try {
            for (int i = 1; i < n; ++i) {
                workers_.emplace_back([this] { work(); });
            }
        } catch (...) {
            shutdown();
            throw;
        }
    }

    ~CpuEnvPool() { shutdown(); }

    int concurrency() const { return static_cast<int>(workers_.size()) + 1; }

    template <class F>
    void run(int count, F&& fn) {
        if (workers_.empty() || count < 2) {
            for (int i = 0; i < count; ++i) {
                fn(i);
            }
            return;
        }
        const int64_t t0 = NowNs();
        if (last_done_ns_ != 0) {
            gap_ns_ += (t0 - last_done_ns_ - gap_ns_) / 4;
        }
        call_ = [](void* ctx, int i) { (*static_cast<std::remove_reference_t<F>*>(ctx))(i); };
        ctx_ = &fn;
        count_ = count;
        chunk_ = std::max(1, count / (static_cast<int>(workers_.size() + 1) * 16));
        next_.store(0, std::memory_order_relaxed);
        done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_seq_cst);
        if (parked_.load(std::memory_order_seq_cst) != 0) {
            wake_all();
        }
        drain();
        int spins = 0;
        const int expected = static_cast<int>(workers_.size());
        while (done_.load(std::memory_order_acquire) != expected) {
            if (++spins < 4096) {
                CpuRelax();
            } else {
                std::this_thread::yield();
            }
        }
        const int64_t t1 = NowNs();
        run_ns_ += (t1 - t0 - run_ns_) / 4;
        last_done_ns_ = t1;
        spin_ns_.store(gap_ns_ <= kSpinGapMaxNs ? std::min(kSpinMaxNs, 2 * gap_ns_ + run_ns_) : 0,
                       std::memory_order_relaxed);
    }

private:
    void shutdown() {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            gen_.fetch_add(1, std::memory_order_seq_cst);
        }
        wake_all();
        for (std::thread& t : workers_) {
            t.join();
        }
    }

    void wake_all() {
#if defined(__linux__)
        syscall(SYS_futex, reinterpret_cast<uint32_t*>(&gen_), FUTEX_WAKE_PRIVATE, INT32_MAX, nullptr, nullptr, 0);
#else
        const std::lock_guard<std::mutex> lock(mutex_);
        cv_.notify_all();
#endif
    }

    void drain() {
        const int count = count_;
        const int chunk = chunk_;
        const auto call = call_;
        void* const ctx = ctx_;
        for (;;) {
            const int i0 = next_.fetch_add(chunk, std::memory_order_relaxed);
            if (i0 >= count) {
                return;
            }
            const int i1 = std::min(count, i0 + chunk);
            for (int i = i0; i < i1; ++i) {
                call(ctx, i);
            }
        }
    }

    void work() {
        uint32_t seen = 0;
        for (;;) {
            const int64_t spin_ns = spin_ns_.load(std::memory_order_relaxed);
            if (spin_ns > 0) {
                const int64_t end = NowNs() + spin_ns;
                for (int k = 1; gen_.load(std::memory_order_acquire) == seen; ++k) {
                    CpuRelax();
                    if ((k & 31) == 0 && NowNs() >= end) {
                        break;
                    }
                }
            }
            for (int k = 0; k < 4096 && gen_.load(std::memory_order_acquire) == seen; ++k) {
                CpuRelax();
            }
            if (gen_.load(std::memory_order_acquire) == seen) {
#if defined(__linux__)
                parked_.fetch_add(1, std::memory_order_seq_cst);
                while (gen_.load(std::memory_order_seq_cst) == seen) {
                    syscall(SYS_futex, reinterpret_cast<uint32_t*>(&gen_), FUTEX_WAIT_PRIVATE, seen, nullptr, nullptr, 0);
                }
#else
                std::unique_lock<std::mutex> lock(mutex_);
                parked_.fetch_add(1, std::memory_order_seq_cst);
                if (!stop_ && gen_.load(std::memory_order_seq_cst) == seen) {
                    cv_.wait(lock, [&] { return stop_ || gen_.load(std::memory_order_acquire) != seen; });
                }
#endif
                parked_.fetch_sub(1, std::memory_order_relaxed);
            }
            seen = gen_.load(std::memory_order_acquire);
            if (stop_) {
                return;
            }
            drain();
            done_.fetch_add(1, std::memory_order_release);
        }
    }

    static int64_t NowNs() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    static constexpr int64_t kSpinGapMaxNs = 100000;
    static constexpr int64_t kSpinMaxNs = 1000000;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_;
    alignas(128) std::atomic<uint32_t> gen_{0};
    alignas(128) std::atomic<int> next_{0};
    alignas(128) std::atomic<int> done_{0};
    alignas(128) std::atomic<int> parked_{0};
    alignas(128) std::atomic<int64_t> spin_ns_{0};
    int64_t last_done_ns_ = 0;
    int64_t gap_ns_ = 0;
    int64_t run_ns_ = 0;
    void (*call_)(void*, int) = nullptr;
    void* ctx_ = nullptr;
    int count_ = 0;
    int chunk_ = 1;
    bool stop_ = false;
};
#endif

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
        : RocketGpuUcrtPowf(rocketgpu::Max(0.0f, 1.0f - mutator.ball_drag), dt);
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

float ImportedLinearScalarBt(float value) {
    constexpr float kUUToBT = 1.0f / 50.0f;
    constexpr float kBTToUU = 50.0f;
    const float bt = value * kUUToBT;
    if (bt * kBTToUU == value) {
        return bt;
    }
    const float up = nextafterf(bt, INFINITY);
    const float down = nextafterf(bt, -INFINITY);
    if (up * kBTToUU == value) {
        return up;
    }
    if (down * kBTToUU == value) {
        return down;
    }
    return bt;
}

Vec3 ImportedCarLinearVec3Bt(const Vec3& value) {
    return Vec3(
        ImportedLinearScalarBt(value.x),
        ImportedLinearScalarBt(value.y),
        ImportedLinearScalarBt(value.z));
}

Vec3 ImportedBallLinearVec3Bt(const Vec3& value) {
    return ImportedCarLinearVec3Bt(value);
}

Vec3 BtVec3ToUuMirror(const Vec3& value_bt) {
    constexpr float kBTToUU = 50.0f;
    return Vec3(value_bt.x * kBTToUU, value_bt.y * kBTToUU, value_bt.z * kBTToUU);
}

void CanonicalizeImportedCarStatesInPlace(CarState* states, size_t count) {
    for (size_t idx = 0; idx < count; ++idx) {
        const uint8_t kept = states[idx].tick_count_since_update != 0 ? states[idx].RestoreWorkingFields() : 0;
        states[idx].ResetTrainingHistory();
        states[idx].pos_bt = ImportedCarLinearVec3Bt(states[idx].pos);

        states[idx].vel_bt = ImportedCarLinearVec3Bt(states[idx].vel);
        states[idx].pos = BtVec3ToUuMirror(states[idx].pos_bt);
        states[idx].vel = BtVec3ToUuMirror(states[idx].vel_bt);

        detail::CanonicalizeImportedCarOrientation(states[idx]);
        states[idx].ReReportFields(kept);
    }
}

void CanonicalizeImportedBallStatesInPlace(BallState* states, size_t count) {
    for (size_t idx = 0; idx < count; ++idx) {
        states[idx].pos_bt = ImportedBallLinearVec3Bt(states[idx].pos);
        states[idx].vel_bt = ImportedBallLinearVec3Bt(states[idx].vel);
        states[idx].pos = BtVec3ToUuMirror(states[idx].pos_bt);
        states[idx].vel = BtVec3ToUuMirror(states[idx].vel_bt);
        states[idx].sleeping = 2;
        states[idx].deactivation_time = 0.0f;
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
void SetSteerCurvesDeg(CarConfig& c, const float (&steer)[6], const float (&powerslide)[2]) {
    for (int i = 0; i < 6; ++i) c.steer_angle_curve_deg[i] = steer[i];
    for (int i = 0; i < 2; ++i) c.powerslide_steer_angle_curve_deg[i] = powerslide[i];
}

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
    CarConfig c = MakePresetConfig(Vec3(127.92678f + 2.5f, 83.27995f + 2.5f, 31.3f + 2.5f), Vec3(9.0f, 0.0f, 15.75f),
                            12.00f, 33.95f, Vec3(50.30f, 31.10f, 15.75f),
                            13.50f, 33.85f, Vec3(-34.75f, 33.00f, 15.75f), false);
    c.hitbox_published = Vec3(127.92678f, 83.27995f, 31.3f);
    SetSteerCurvesDeg(c, {29.923809051513672f, 18.018766403198242f, 10.329229354858398f, 6.019903182983398f,
                          4.85035514831543f, 1.9751166105270386f}, {22.08690071105957f, 7.173928260803223f});
    c.spawn_z = 36.01f;
    return c;
}
CarConfig PlankConfig() {
    CarConfig c = MakePresetConfig(Vec3(128.81978f + 2.5f, 84.670364f + 2.5f, 29.394402f + 2.5f), Vec3(9.008572f, 0.0f, 12.0942f),
                                   12.50f, 31.9242f, Vec3(49.97f, 27.799904f, 12.0942f),
                                   17.00f, 27.9242f, Vec3(-35.43f, 20.280005f, 12.0942f), false);
    c.hitbox_published = Vec3(128.81978f, 84.670364f, 29.394402f);
    SetSteerCurvesDeg(c, {30.342710494995117f, 18.1981258392334f, 10.394840240478516f, 6.043498516082764f,
                          4.865903377532959f, 1.9777874946594238f}, {22.341325759887695f, 7.206938743591309f});
    c.spawn_z = 38.84f;
    return c;
}
CarConfig BreakoutConfig() {
    CarConfig c = MakePresetConfig(Vec3(131.49236f + 2.5f, 80.521f + 2.5f, 30.3f + 2.5f), Vec3(12.5f, 0.0f, 11.75f),
                            13.50f, 29.7f, Vec3(51.50f, 26.67f, 11.75f),
                            15.00f, 29.66f, Vec3(-35.75f, 35.00f, 11.75f), false);
    c.hitbox_published = Vec3(131.49236f, 80.521f, 30.3f);
    SetSteerCurvesDeg(c, {30.55976104736328f, 18.289745330810547f, 10.42794418334961f, 6.055309772491455f,
                          4.873669147491455f, 1.9791138172149658f}, {22.472003936767578f, 7.2235002517700195f});
    c.spawn_z = 36.00f;
    return c;
}
CarConfig HybridConfig() {
    CarConfig c = MakePresetConfig(Vec3(127.01919f + 2.5f, 82.18787f + 2.5f, 34.159073f + 2.5f), Vec3(13.87566f, 0.0f, 20.754988f),
                            12.50f, 38.75499f, Vec3(51.25f, 25.90f, 20.754988f),
                            15.00f, 37.054989f, Vec3(-34.00f, 29.50f, 20.754988f), false);
    c.hitbox_published = Vec3(127.01919f, 82.18787f, 34.159073f);
    SetSteerCurvesDeg(c, {30.580455780029297f, 18.29843521118164f, 10.431071281433105f, 6.056422233581543f,
                          4.874399185180664f, 1.9792383909225464f}, {22.484424591064453f, 7.225060939788818f});
    c.spawn_z = 37.53f;
    return c;
}
CarConfig MercConfig() {
    CarConfig c = MakePresetConfig(Vec3(120.72023f + 2.5f, 76.71031f + 2.5f, 41.659073f + 2.5f), Vec3(11.37566f, 0.0f, 21.504988f),
                            15.00f, 39.50499f, Vec3(51.25f, 25.90f, 21.504988f),
                            15.00f, 39.104988f, Vec3(-33.75f, 29.50f, 21.504988f), false);
    c.hitbox_published = Vec3(120.72023f, 76.71031f, 41.659073f);
    c.spawn_z = 35.05f;
    return c;
}
CarConfig PsyclopsConfig() {
    CarConfig c = MakePresetConfig(Vec3(118.00738f + 2.5f, 84.19941f + 2.5f, 36.159073f + 2.5f), Vec3(13.87566f, 0.0f, 15.00499f),
                                   12.50f, 33.00499f, Vec3(51.25f, 5.000f, 15.00499f),
                                   15.00f, 31.304989f, Vec3(-34.25f, 29.50f, 15.00499f), true);
    c.powerslide_steer_angle_curve_deg[0] = 24.427291870117188f;
    c.powerslide_steer_angle_curve_deg[1] = 7.443766117095947f;
    c.spawn_z = 36.00f;
    return c;
}
CarConfig BatmobileConfig() {
    CarConfig c = MakePresetConfig(Vec3(128.81978f + 2.5f, 84.670364f + 2.5f, 29.394402f + 2.5f), Vec3(9.008572f, 0.0f, 12.0942f),
                                   12.50f, 31.9242f, Vec3(49.97f, 27.80f, 12.0942f),
                                   17.00f, 27.9242f, Vec3(-35.43f, 20.28f, 12.0942f), false);
    c.hitbox_published = Vec3(128.81978f, 84.670364f, 29.394402f);
    SetSteerCurvesDeg(c, {30.342700958251953f, 18.1981201171875f, 10.394838333129883f, 6.0434980392456055f,
                          4.865902900695801f, 1.9777874946594238f}, {22.341318130493164f, 7.206938743591309f});
    c.spawn_z = 36.00f;
    return c;
}

struct ArenaMeshHost {
    std::vector<detail::ArenaMeshTriangle> mesh, local_mesh;
    std::vector<detail::ArenaMeshBvhQuantization> quantization;
    std::vector<detail::ArenaMeshBvhMeshInfo> infos;
    std::vector<detail::ArenaMeshBvhNode> nodes;
    std::vector<detail::ArenaMeshBvhSubtreeInfo> subtrees;
    std::vector<detail::ArenaMeshLeafGridInfo> leaf_grid_infos;
    std::vector<int32_t> leaf_cell_start;
    std::vector<detail::ArenaMeshLeafEntry> leaf_entries;
    std::vector<detail::ArenaMeshLeafGridInfo> ray_grid_infos;
    std::vector<int32_t> ray_cell_start;
    std::vector<detail::ArenaMeshLeafEntry> ray_entries;
#if !defined(ROCKETGPU_USE_CUDA)
    std::vector<float> ray_entry_boxes;
    std::vector<detail::ArenaMeshRayBlock> ray_blocks;
    std::vector<uint8_t> clearance;
    Vec3 clearance_lo;
    float clearance_inv_cell = 0.0f;
    float clearance_unit = 0.0f;
    int32_t clearance_dims[3] = {0, 0, 0};
#endif
    detail::ArenaBroadphaseGrid grid{};
    std::vector<uint16_t> cell_mask;
};

const ArenaMeshHost& CachedArenaMeshHost(bool soccar_arena) {
    static std::mutex mutex;
    static std::map<std::pair<std::string, bool>, std::unique_ptr<ArenaMeshHost>> cache;
    const std::lock_guard<std::mutex> lock(mutex);
    std::error_code ec;
    const std::filesystem::path raw_root = detail::OfficialCollisionMeshRoot();
    const std::filesystem::path canon_root = std::filesystem::weakly_canonical(raw_root, ec);
    const std::string key_root = (ec || canon_root.empty()) ? raw_root.string() : canon_root.string();
    std::unique_ptr<ArenaMeshHost>& slot = cache[{key_root, soccar_arena}];
    if (slot) {
        return *slot;
    }
    auto h = std::make_unique<ArenaMeshHost>();
    int local_goal_meshes = 0;
    h->mesh = LoadArenaMeshTriangles(&h->quantization, &h->infos, &h->nodes, &h->subtrees,
                                     soccar_arena ? &h->local_mesh : nullptr, &local_goal_meshes);
    if (soccar_arena && (local_goal_meshes == 0 || h->local_mesh.size() != h->mesh.size())) {
        throw std::runtime_error("no goal mesh found in the soccar arena meshes");
    }
    if (!h->mesh.empty()) {
        detail::SetArenaMeshFreeBox(h->infos);
        Vec3 neg_x_min, neg_x_max;
        detail::BulletStaticPlaneShapeAabb(2, neg_x_min, neg_x_max);
        detail::SetArenaMeshRayOrder(h->infos, neg_x_max.x);
        detail::BuildArenaMeshLeafGrid(h->quantization, h->infos, h->nodes, h->subtrees, h->leaf_grid_infos,
                                       h->leaf_cell_start, h->leaf_entries);
#if defined(ROCKETGPU_USE_CUDA)
        detail::BuildArenaMeshLeafGrid(h->quantization, h->infos, h->nodes, h->subtrees, h->ray_grid_infos,
                                       h->ray_cell_start, h->ray_entries, 1.4f, 1);
#else
        detail::BuildArenaMeshLeafGrid(h->quantization, h->infos, h->nodes, h->subtrees, h->ray_grid_infos,
                                       h->ray_cell_start, h->ray_entries, 1.4f, 4);
        h->ray_entry_boxes.assign(h->ray_entries.size() * 8, 0.0f);
        for (size_t m = 0; m < h->ray_grid_infos.size(); ++m) {
            const detail::ArenaMeshLeafGridInfo& gi = h->ray_grid_infos[m];
            const int mesh_index = h->infos[m].mesh_index;
            if (gi.cell_offset < 0 || mesh_index < 0 || mesh_index >= static_cast<int>(h->quantization.size())) {
                continue;
            }
            const detail::ArenaMeshBvhQuantization& q = h->quantization[static_cast<size_t>(mesh_index)];
            const int cells = gi.cells[0] * gi.cells[1] * gi.cells[2];
            for (int e = h->ray_cell_start[gi.cell_offset]; e < h->ray_cell_start[gi.cell_offset + cells]; ++e) {
                const Vec3 lo = detail::UnquantizeArenaMeshBvhPoint(q, h->ray_entries[e].quantized_aabb_min);
                const Vec3 hi = detail::UnquantizeArenaMeshBvhPoint(q, h->ray_entries[e].quantized_aabb_max);
                float* b = &h->ray_entry_boxes[static_cast<size_t>(e) * 8];
                b[0] = lo.x; b[1] = lo.y; b[2] = lo.z; b[4] = hi.x; b[5] = hi.y; b[6] = hi.z;
            }
        }
        {
            const bool local = !h->local_mesh.empty();
            const std::vector<detail::ArenaMeshTriangle>& tris = local ? h->local_mesh : h->mesh;
            detail::ArenaMeshRayBlock nan_block;
            std::fill_n(&nan_block.lo[0][0], (sizeof(nan_block) - offsetof(detail::ArenaMeshRayBlock, lo)) / sizeof(float),
                        std::numeric_limits<float>::quiet_NaN());
            std::fill_n(nan_block.tri, 4, -1);
            std::fill_n(nan_block.mesh_index, 4, -1);
            h->ray_blocks.assign(h->ray_entries.size() / 4, nan_block);
            for (size_t e = 0; e < h->ray_entries.size(); ++e) {
                const detail::ArenaMeshLeafEntry& entry = h->ray_entries[e];
                detail::ArenaMeshRayBlock& blk = h->ray_blocks[e / 4];
                const int k = static_cast<int>(e % 4);
                blk.quant[0][k] = entry.quantized_aabb_min[0];
                blk.quant[0][4 + k] = entry.quantized_aabb_min[1];
                blk.quant[1][k] = entry.quantized_aabb_min[2];
                blk.quant[1][4 + k] = static_cast<uint16_t>(~entry.quantized_aabb_max[0]);
                blk.quant[2][k] = static_cast<uint16_t>(~entry.quantized_aabb_max[1]);
                blk.quant[2][4 + k] = static_cast<uint16_t>(~entry.quantized_aabb_max[2]);
            }
            for (size_t m = 0; m < h->ray_grid_infos.size(); ++m) {
                const detail::ArenaMeshLeafGridInfo& gi = h->ray_grid_infos[m];
                const detail::ArenaMeshBvhMeshInfo& mi = h->infos[m];
                if (gi.cell_offset < 0) {
                    continue;
                }
                const int cells = gi.cells[0] * gi.cells[1] * gi.cells[2];
                for (int e = h->ray_cell_start[gi.cell_offset]; e < h->ray_cell_start[gi.cell_offset + cells]; ++e) {
                    const int idx = h->ray_entries[e].triangle_index;
                    const int g = mi.triangle_start + idx;
                    if (idx < 0 || idx >= mi.triangle_count || g < 0 || g >= static_cast<int>(tris.size())) {
                        continue;
                    }
                    detail::ArenaMeshRayBlock& blk = h->ray_blocks[static_cast<size_t>(e) / 4];
                    const int k = e % 4;
                    const float* box = &h->ray_entry_boxes[static_cast<size_t>(e) * 8];
                    const detail::ArenaMeshTriangle& tri = tris[static_cast<size_t>(g)];
                    const bool hit_local = local && mi.cal_origin_y != 0.0f;
                    if (!hit_local && g >= static_cast<int>(h->mesh.size())) {
                        continue;
                    }
                    blk.tri[k] = g;
                    blk.mesh_index[k] =
                        static_cast<int32_t>((hit_local ? tri : h->mesh[static_cast<size_t>(g)]).mesh_index);
                    const Vec3 n = detail::Cross(detail::operator-(tri.b, tri.a), detail::operator-(tri.c, tri.a));
                    const Vec3* v[4] = {&tri.a, &tri.b, &tri.c, &n};
                    float(*dst[4])[4] = {blk.a, blk.b, blk.c, blk.n};
                    for (int a = 0; a < 3; ++a) {
                        blk.lo[a][k] = box[a];
                        blk.hi[a][k] = box[4 + a];
                        for (int j = 0; j < 4; ++j) {
                            dst[j][a][k] = a == 0 ? v[j]->x : a == 1 ? v[j]->y : v[j]->z;
                        }
                    }
                    blk.plane[k] = detail::Dot(tri.a, n);
                    blk.tol[k] = -0.0001f * detail::LengthSq(n);
                    blk.origin_y[k] = local ? static_cast<float>(tri.cal_origin_sign) * detail::kGoalMeshOriginYBt : 0.0f;
                }
            }
        }
        int clearance_threads = 0;
        if (const char* s = std::getenv("ROCKETGPU_CPU_THREADS")) {
            clearance_threads = std::atoi(s);
        }
        if (clearance_threads <= 0) {
            clearance_threads = std::min(DefaultCpuHardwareThreads(), 16);
        }
        detail::BuildArenaMeshClearance(h->mesh, h->local_mesh, h->clearance, h->clearance_lo, h->clearance_inv_cell,
                                        h->clearance_unit, h->clearance_dims, clearance_threads);
#endif
        h->cell_mask = detail::BuildArenaBroadphaseCellMask(h->mesh, h->infos, h->grid);
    }
    slot = std::move(h);
    return *slot;
}

struct BatchSim::Impl {
    explicit Impl(const SimConfig& in_cfg) : cfg(in_cfg), user_cfg(in_cfg) {
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
        if (!(cfg.tick_dt > 0.0f)) {
            throw std::invalid_argument("tick_dt must be > 0");
        }
        CacheBallDamping(cfg.mutator, cfg.tick_dt, cfg.mode);

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
#if !defined(ROCKETGPU_USE_CUDA)
        ray_hints.assign(car_capacity * 8, -1);
#endif
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

#if defined(ROCKETGPU_USE_CUDA)
        cfg.arena_mesh_ray_grid_infos = nullptr;
#endif
        if (cfg.arena_mesh_tris == nullptr || cfg.arena_mesh_tri_count <= 0) {
            const bool soccar_arena = cfg.mode == Mode::Soccar || cfg.mode == Mode::Snowday || cfg.mode == Mode::Heatseeker;
            const ArenaMeshHost& mh = CachedArenaMeshHost(soccar_arena);
#if !defined(ROCKETGPU_USE_CUDA)
            host_mesh = &mh;
#endif
            const auto& host_mesh_quantization = mh.quantization;
            const auto& host_mesh_infos = mh.infos;
            const auto& host_mesh_nodes = mh.nodes;
            const auto& host_mesh_subtrees = mh.subtrees;
            const auto& host_local_mesh = mh.local_mesh;
            const auto& host_mesh = mh.mesh;
            if (soccar_arena) {
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
                detail::BulletSweepHistory sweep_statics[2];
                for (int goal_floors = 0; goal_floors < 2; ++goal_floors) {
                    detail::BulletSweepAddStatics(sweep_statics[goal_floors], true, goal_floors != 0,
                                                  host_mesh_infos.data(), static_cast<int>(host_mesh_infos.size()));
                }
                arena_sweep_statics.resize(2);
                arena_sweep_statics.copy_from_host(sweep_statics, 2);
                cfg.arena_sweep_statics = arena_sweep_statics.data();
                arena_mesh_bvh_nodes.resize(host_mesh_nodes.size());
                arena_mesh_bvh_nodes.copy_from_host(host_mesh_nodes.data(), host_mesh_nodes.size());
                cfg.arena_mesh_bvh_nodes = arena_mesh_bvh_nodes.data();
                cfg.arena_mesh_bvh_node_count = static_cast<int32_t>(arena_mesh_bvh_nodes.size());
                arena_mesh_bvh_subtrees.resize(host_mesh_subtrees.size());
                arena_mesh_bvh_subtrees.copy_from_host(host_mesh_subtrees.data(), host_mesh_subtrees.size());
                cfg.arena_mesh_bvh_subtrees = arena_mesh_bvh_subtrees.data();
                cfg.arena_mesh_bvh_subtree_count = static_cast<int32_t>(arena_mesh_bvh_subtrees.size());

                const auto& host_leaf_grid_infos = mh.leaf_grid_infos;
                const auto& host_leaf_cell_start = mh.leaf_cell_start;
                const auto& host_leaf_entries = mh.leaf_entries;
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
#if defined(ROCKETGPU_USE_CUDA)
                if (!mh.ray_entries.empty() && mh.ray_grid_infos.size() == host_mesh_infos.size()) {
                    arena_mesh_ray_grid_infos.resize(mh.ray_grid_infos.size());
                    arena_mesh_ray_grid_infos.copy_from_host(mh.ray_grid_infos.data(), mh.ray_grid_infos.size());
                    arena_mesh_ray_cell_start.resize(mh.ray_cell_start.size());
                    arena_mesh_ray_cell_start.copy_from_host(mh.ray_cell_start.data(), mh.ray_cell_start.size());
                    arena_mesh_ray_entries.resize(mh.ray_entries.size());
                    arena_mesh_ray_entries.copy_from_host(mh.ray_entries.data(), mh.ray_entries.size());
                    cfg.arena_mesh_ray_grid_infos = arena_mesh_ray_grid_infos.data();
                    cfg.arena_mesh_ray_cell_start = arena_mesh_ray_cell_start.data();
                    cfg.arena_mesh_ray_entries = arena_mesh_ray_entries.data();
                }
#endif

                const detail::ArenaBroadphaseGrid& host_grid = mh.grid;
                const std::vector<uint16_t>& host_cell_mask = mh.cell_mask;
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

#if !defined(ROCKETGPU_USE_CUDA)
        BuildHostStepTables();
#endif
        reset_to_kickoff(0, nullptr);
    }

#if !defined(ROCKETGPU_USE_CUDA)
    void BuildHostStepTables() {
        using Tables = detail::HostStepTables;
        constexpr float kUUToBT = 1.0f / 50.0f;
        constexpr float kFar = 1.0e4f;
        host_tables = std::make_unique<Tables>();
        Tables& t = *host_tables;
        const auto fit = [kFar](Tables::Axis& a, float lo, float hi) {
            lo = std::max(lo, -kFar);
            hi = std::min(hi, kFar);
            a.lo = lo;
            a.inv = hi > lo ? static_cast<float>(Tables::kBins) / (hi - lo) : 0.0f;
        };
        const int pads = detail::BoostPadCount(cfg);
        if (pads >= 0 && pads <= 64) {
            Vec3 lo(kFar, kFar, kFar), hi(-kFar, -kFar, -kFar);
            bool pad_z_nan = false;
            for (int p = 0; p < pads; ++p) {
                const Vec3 bt = detail::operator*(detail::BoostPadLoc(cfg, p), kUUToBT);
                pad_z_nan |= std::isnan(bt.z);
                lo = detail::MinComponents(lo, bt);
                hi = detail::MaxComponents(hi, bt);
            }
            fit(t.pad_x.axis, lo.x, hi.x);
            fit(t.pad_y.axis, lo.y, hi.y);
            t.pad_min_z_bt = pad_z_nan ? -std::numeric_limits<float>::infinity() : lo.z;
            t.pad_max_z_bt = pad_z_nan ? std::numeric_limits<float>::infinity() : hi.z;
            for (int p = 0; p < pads; ++p) {
                const Vec3 bt = detail::operator*(detail::BoostPadLoc(cfg, p), kUUToBT);
                t.pad_x.Add(p, bt.x, bt.x);
                t.pad_y.Add(p, bt.y, bt.y);
            }
            t.pad_count = pads;
        }
        const detail::ArenaMeshBvhMeshInfo* meshes = cfg.arena_mesh_bvh_meshes;
        const int n = cfg.arena_mesh_bvh_mesh_count;
        if (meshes != nullptr && n > 0 && n <= 32) {
            const auto info = [&](int pos) -> const detail::ArenaMeshBvhMeshInfo& {
                return meshes[meshes[pos].ray_order >= 0 ? meshes[pos].ray_order : pos];
            };
            Vec3 lo(kFar, kFar, kFar), hi(-kFar, -kFar, -kFar);
            for (int pos = 0; pos < n; ++pos) {
                lo = detail::MinComponents(lo, info(pos).aabb_min);
                hi = detail::MaxComponents(hi, info(pos).aabb_max);
            }
            fit(t.mesh_axis[0].axis, lo.x, hi.x);
            fit(t.mesh_axis[1].axis, lo.y, hi.y);
            fit(t.mesh_axis[2].axis, lo.z, hi.z);
            for (int pos = 0; pos < n; ++pos) {
                const detail::ArenaMeshBvhMeshInfo& m = info(pos);
                t.mesh_axis[0].Add(pos, m.aabb_min.x, m.aabb_max.x);
                t.mesh_axis[1].Add(pos, m.aabb_min.y, m.aabb_max.y);
                t.mesh_axis[2].Add(pos, m.aabb_min.z, m.aabb_max.z);
            }
            t.meshes = meshes;
            t.mesh_count = n;
            if (host_mesh != nullptr && !host_mesh->ray_entries.empty() &&
                host_mesh->ray_grid_infos.size() == static_cast<size_t>(n)) {
                t.ray_grid.infos = host_mesh->ray_grid_infos.data();
                t.ray_grid.cell_start = host_mesh->ray_cell_start.data();
                t.ray_grid.entries = host_mesh->ray_entries.data();
                if (cfg.arena_mesh_bvh_quantization != nullptr &&
                    cfg.arena_mesh_bvh_quantization_count == static_cast<int32_t>(host_mesh->quantization.size()) &&
                    std::memcmp(cfg.arena_mesh_bvh_quantization, host_mesh->quantization.data(),
                                host_mesh->quantization.size() * sizeof(detail::ArenaMeshBvhQuantization)) == 0) {
                    t.ray_grid.entry_boxes = host_mesh->ray_entry_boxes.data();
                    t.ray_grid.box_quantization = cfg.arena_mesh_bvh_quantization;
                    const detail::ArenaMeshTriangle* local = cfg.mutator.cal_arena_local_tris;
                    const std::vector<detail::ArenaMeshTriangle>& built =
                        host_mesh->local_mesh.empty() ? host_mesh->mesh : host_mesh->local_mesh;
                    const detail::ArenaMeshTriangle* tris = local != nullptr ? local : cfg.arena_mesh_tris;
                    bool magic = true;
                    for (const detail::ArenaMeshLeafGridInfo& gi : host_mesh->ray_grid_infos) {
                        magic = magic && (gi.cell_offset < 0 ||
                                          (gi.cell_magic[0] != 0 && gi.cell_magic[1] != 0 && gi.cell_magic[2] != 0));
                    }
                    bool same_meshes = host_mesh->infos.size() == static_cast<size_t>(n) && cfg.arena_mesh_tris != nullptr &&
                                       cfg.arena_mesh_tri_count == static_cast<int32_t>(host_mesh->mesh.size()) &&
                                       std::memcmp(cfg.arena_mesh_tris, host_mesh->mesh.data(),
                                                   host_mesh->mesh.size() * sizeof(detail::ArenaMeshTriangle)) == 0;
                    for (int m = 0; same_meshes && m < n; ++m) {
                        same_meshes = meshes[m].triangle_start == host_mesh->infos[m].triangle_start &&
                                      meshes[m].cal_origin_y == host_mesh->infos[m].cal_origin_y;
                    }
                    if (magic && same_meshes && host_mesh->ray_blocks.size() * 4 == host_mesh->ray_entries.size() && tris != nullptr &&
                        (local != nullptr) == !host_mesh->local_mesh.empty() &&
                        cfg.arena_mesh_tri_count == static_cast<int32_t>(built.size()) &&
                        std::memcmp(tris, built.data(), built.size() * sizeof(detail::ArenaMeshTriangle)) == 0) {
                        t.ray_blocks = host_mesh->ray_blocks.data();
                        t.ray_cell_count = static_cast<int32_t>(host_mesh->ray_cell_start.size()) - 1;
                        t.ray_block_count = static_cast<int32_t>(host_mesh->ray_blocks.size());
                        if (ray_hints.size() == wheel_runtime_states.size() * 8) {
                            t.ray_hints = ray_hints.data();
                            t.ray_hint_base = wheel_runtime_states.data();
                            t.ray_hint_cars = static_cast<int32_t>(wheel_runtime_states.size());
                        }
                        t.ray_block_tris = local;
                        t.ray_block_mesh = cfg.arena_mesh_tris;
                    }
                }
            }
            if (host_mesh != nullptr && !host_mesh->clearance.empty() && cfg.arena_mesh_tris != nullptr &&
                cfg.arena_mesh_tri_count == static_cast<int32_t>(host_mesh->mesh.size())) {
                detail::ArenaMeshClearance& c = t.clearance;
                c.mesh = cfg.arena_mesh_tris;
                c.lb = host_mesh->clearance.data();
                c.lo = host_mesh->clearance_lo;
                c.inv_cell = host_mesh->clearance_inv_cell;
                c.unit = host_mesh->clearance_unit;
                c.nx = host_mesh->clearance_dims[0];
                c.ny = host_mesh->clearance_dims[1];
                c.nz = host_mesh->clearance_dims[2];
            }
        }
        if (cfg.arena_mesh_tris != nullptr && cfg.arena_mesh_tri_count > 0) {
            host_tri_normals.resize(static_cast<size_t>(cfg.arena_mesh_tri_count) * 4);
            for (int32_t i = 0; i < cfg.arena_mesh_tri_count; ++i) {
                const detail::ArenaMeshTriangle& tri = cfg.arena_mesh_tris[i];
                const Vec3 nrm = detail::Normalize(tri.normal, Vec3(0.0f, 0.0f, 1.0f));
                float* out = &host_tri_normals[static_cast<size_t>(i) * 4];
                out[0] = nrm.x;
                out[1] = nrm.y;
                out[2] = nrm.z;
                out[3] = detail::Dot(nrm, tri.a);
            }
            t.tri_normals = host_tri_normals.data();
            t.tri_normals_mesh = cfg.arena_mesh_tris;
        }
        t.owner = &cfg;
        cfg.host_tables = host_tables.get();
    }
    std::unique_ptr<detail::HostStepTables> host_tables;
    std::vector<int32_t> ray_hints;
    std::vector<float> host_tri_normals;
    const ArenaMeshHost* host_mesh = nullptr;
#endif

    SimConfig cfg;
    SimConfig user_cfg;
    uint64_t seed = 0;
    bool kickoff_settled_dirty = true;
    detail::Buffer<detail::KickoffSettledRow> kickoff_settled_rows;
    detail::Buffer<int64_t> kickoff_settled_entry;
    detail::Buffer<BallState> kickoff_settled_ball;
    struct KickoffSettledPair {
        CarConfig config;
        Team team;
    };
    std::vector<KickoffSettledPair> kickoff_settled_pairs;
    std::vector<detail::KickoffSettledRow> kickoff_settled_pair_rows;
    std::vector<BallState> kickoff_settled_pair_ball;

    void ensure_kickoff_settled(cudaStream_t stream) {
        if (!cfg.mutator.kickoff_settled) {
            cfg.kickoff_settled_rows = nullptr;
            cfg.kickoff_settled_entry = nullptr;
            cfg.kickoff_settled_ball = nullptr;
            cfg.kickoff_settled_wheels = nullptr;
            return;
        }
        cfg.kickoff_settled_wheels = wheel_runtime_states.data();
        if (!kickoff_settled_dirty) {
            return;
        }
        const size_t n = static_cast<size_t>(cfg.env_count) * static_cast<size_t>(cfg.max_cars_per_env);
        std::vector<CarConfig> configs(n);
        std::vector<Team> host_teams(n);
        car_configs.copy_to_host(configs.data(), n, stream);
        teams.copy_to_host(host_teams.data(), n, stream);
#if defined(ROCKETGPU_USE_CUDA)
        CheckCuda(cudaStreamSynchronize(stream), "kickoff_settled configs");
#endif
        std::vector<KickoffSettledPair>& pairs = kickoff_settled_pairs;
        std::vector<size_t> first;
        std::vector<int64_t> entry(n);
        const size_t old_pairs = pairs.size();
        const size_t old_rows = kickoff_settled_pair_rows.size();
        const size_t old_balls = kickoff_settled_pair_ball.size();
        const int spawns = std::max(cfg.max_cars_per_env, 5);
        try {
            for (size_t i = 0; i < n; ++i) {
                size_t e = 0;
                while (e < pairs.size() && !(pairs[e].team == host_teams[i] &&
                                             std::memcmp(&pairs[e].config, &configs[i], sizeof(CarConfig)) == 0)) {
                    ++e;
                }
                if (e == pairs.size()) {
                    pairs.push_back({configs[i], host_teams[i]});
                    first.push_back(i);
                }
                entry[i] = static_cast<int64_t>(e);
            }
            if (!first.empty()) {
                record_kickoff_settled(first, configs, host_teams, spawns);
            }
        } catch (...) {
            pairs.resize(old_pairs);
            kickoff_settled_pair_rows.resize(old_rows);
            kickoff_settled_pair_ball.resize(old_balls);
            throw;
        }
        const size_t m = kickoff_settled_pair_rows.size();
        kickoff_settled_rows.resize(m);
        kickoff_settled_rows.copy_from_host(kickoff_settled_pair_rows.data(), m, stream);
        kickoff_settled_entry.resize(n);
        kickoff_settled_entry.copy_from_host(entry.data(), n, stream);
        kickoff_settled_ball.resize(1);
        kickoff_settled_ball.copy_from_host(kickoff_settled_pair_ball.data(), 1, stream);
        cfg.kickoff_settled_rows = kickoff_settled_rows.data();
        cfg.kickoff_settled_entry = kickoff_settled_entry.data();
        cfg.kickoff_settled_ball = kickoff_settled_ball.data();
        cfg.kickoff_settled_spawns = spawns;
        kickoff_settled_dirty = false;
    }

    void record_kickoff_settled(const std::vector<size_t>& first, const std::vector<CarConfig>& configs,
                                const std::vector<Team>& host_teams, int spawns) {
        const size_t envs = first.size();
        SimConfig run_cfg = user_cfg;
        run_cfg.env_count = static_cast<int32_t>(envs);
        run_cfg.max_cars_per_env = spawns;
        run_cfg.mirror_official_respawn_rng = 0;
        run_cfg.mutator.kickoff_settled = 0;
        run_cfg.env_frame_frac = nullptr;
        run_cfg.env_frame_start_dt = nullptr;
        run_cfg.env_step_order = nullptr;
        const MutatorConfig frames_default{};
        if (run_cfg.mutator.countdown_on_frame_start ? run_cfg.mutator.frame_start_dt < 0.0f
                                                     : run_cfg.mutator.boost_pad_frame_frac < 0.0f) {
            run_cfg.mutator.countdown_on_frame_start = frames_default.countdown_on_frame_start;
            run_cfg.mutator.frame_start_dt = frames_default.frame_start_dt;
            run_cfg.mutator.boost_pad_frame_frac = frames_default.boost_pad_frame_frac;
        }
        BatchSim run(run_cfg);
        const size_t m = envs * static_cast<size_t>(spawns);
        std::vector<int32_t> counts(envs, spawns);
        std::vector<CarConfig> run_configs(m);
        std::vector<Team> run_teams(m);
        for (size_t i = 0; i < m; ++i) {
            run_configs[i] = configs[first[i / spawns]];
            run_teams[i] = host_teams[first[i / spawns]];
        }
        run.set_env_car_count(counts.data(), envs);
        run.set_teams(run_teams.data(), m);
        run.set_car_configs(run_configs.data(), m);
        EnvCommand cmd;
        cmd.reset_to_kickoff = 1;
        cmd.reset_scoreboard = 1;
        cmd.kickoff_seed_mode = 3;
        cmd.kickoff_countdown_ticks = kKickoffCountdownGameTimers;
        std::vector<EnvCommand> cmds(envs, cmd);
        run.apply_env_commands(cmds.data(), envs);
        run.step(kKickoffCountdownTicks);
        std::vector<EnvState> env_out(envs);
        for (int k = 0;; ++k) {
            run.download_env_states(env_out.data(), envs);
            if (std::all_of(env_out.begin(), env_out.end(), [](const EnvState& e) { return e.countdown_phase == 0; })) {
                break;
            }
            if (k == 16) {
                throw std::runtime_error("kickoff_settled: the countdown did not end");
            }
            run.step(1);
        }
        std::vector<CarState> cars(m);
        std::vector<detail::WheelRuntimeState> wheels(m);
        BallState ball;
        run.download_car_states(cars.data(), m);
        run.download_wheel_runtime_states(wheels.data(), m);
        run.download_ball_states(&ball, 1);
        const size_t base = kickoff_settled_pair_rows.size();
        kickoff_settled_pair_rows.resize(base + m);
        for (size_t i = 0; i < m; ++i) {
            kickoff_settled_pair_rows[base + i].car = cars[i];
            kickoff_settled_pair_rows[base + i].wheels = wheels[i];
        }
        if (kickoff_settled_pair_ball.empty()) {
            kickoff_settled_pair_ball.push_back(ball);
        }
    }

    detail::Buffer<detail::ArenaMeshTriangle> arena_mesh_triangles;
    detail::Buffer<detail::ArenaMeshTriangle> arena_mesh_local_triangles;
    detail::Buffer<detail::ArenaMeshBvhQuantization> arena_mesh_bvh_quantization;
    detail::Buffer<detail::ArenaMeshBvhMeshInfo> arena_mesh_bvh_meshes;
    detail::Buffer<detail::ArenaMeshBvhNode> arena_mesh_bvh_nodes;
    detail::Buffer<detail::ArenaMeshBvhSubtreeInfo> arena_mesh_bvh_subtrees;
    detail::Buffer<detail::ArenaMeshLeafGridInfo> arena_mesh_leaf_grid_infos;
    detail::Buffer<int32_t> arena_mesh_leaf_cell_start;
    detail::Buffer<detail::ArenaMeshLeafEntry> arena_mesh_leaf_entries;
#if defined(ROCKETGPU_USE_CUDA)
    detail::Buffer<detail::ArenaMeshLeafGridInfo> arena_mesh_ray_grid_infos;
    detail::Buffer<int32_t> arena_mesh_ray_cell_start;
    detail::Buffer<detail::ArenaMeshLeafEntry> arena_mesh_ray_entries;
#endif
    detail::Buffer<detail::BulletSweepHistory> arena_sweep_statics;
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
    detail::Buffer<detail::CarCarPairCachedContact> rows_pair_contact_cache;
    detail::Buffer<detail::CarWorldPlaneRetainedManifold> rows_car_world_plane;
    detail::Buffer<detail::CarWorldMeshRetainedManifold> rows_car_world_mesh;
    detail::Buffer<detail::CarCarRetainedManifold> rows_car_car_retained;
    detail::Buffer<detail::BallCarRetainedManifold> rows_ball_car_retained;
    detail::Buffer<detail::CarWorldLocalRows> rows_car_world_local;
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
    detail::Buffer<float> env_frame_frac;
    detail::Buffer<float> env_frame_start_dt;
    detail::Buffer<EnvCommand> env_command_scratch;

    void rebuild_env_step_order(cudaStream_t stream) {
        const size_t n = env_car_count.size();
        if (n < 128) {
            cfg.env_step_order = nullptr;
            cfg.env_car_total = 0;
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
        cfg.env_car_total = 0;
        for (const int32_t c : counts) cfg.env_car_total += c;
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

#if !defined(ROCKETGPU_USE_CUDA)
    void ensure_host_respawn_slot_overrides_sized() {
        if (!cfg.mirror_official_respawn_rng) {
            return;
        }
        const size_t car_capacity = static_cast<size_t>(cfg.env_count) * static_cast<size_t>(cfg.max_cars_per_env);
        if (host_respawn_slot_overrides.size() != car_capacity) {
            host_respawn_slot_overrides.assign(car_capacity, kNoRespawnSlotOverride);
        }
    }

    bool mirror_respawn_window_safe(int env_idx, int ticks, float window_dt, bool fresh_demo_safe) const {
        if (!fresh_demo_safe || ticks <= 1) {
            return false;
        }
        const int base = env_idx * cfg.max_cars_per_env;
        const int count = env_car_count.data()[static_cast<size_t>(env_idx)];
        const CarState* const cars_ptr = car_states.data();
        for (int i = 0; i < count; ++i) {
            const CarState& car = cars_ptr[static_cast<size_t>(base + i)];
            if (car.is_demoed && car.respawn_slot_override_plus1 < kKickoffSpawnOverridePlus1 &&
                !(car.demo_respawn_timer > window_dt + cfg.tick_dt)) {
                return false;
            }
        }
        return true;
    }

    const uint8_t* prepare_respawn_slot_overrides_env(int env_idx) {
        if (!cfg.mirror_official_respawn_rng) {
            return nullptr;
        }
        const int base = env_idx * cfg.max_cars_per_env;
        const int count = env_car_count.data()[static_cast<size_t>(env_idx)];
        bool any_respawn = false;
        for (int i = 0; i < count; ++i) {
            const CarState& car = car_states.data()[static_cast<size_t>(base + i)];
            if (!car.is_demoed || car.respawn_slot_override_plus1 >= kKickoffSpawnOverridePlus1) {
                continue;
            }
            if (Max(0.0f, car.demo_respawn_timer - cfg.tick_dt) > 0.0f) {
                continue;
            }
            if (!any_respawn) {
                std::fill_n(host_respawn_slot_overrides.data() + static_cast<size_t>(base),
                            static_cast<size_t>(cfg.max_cars_per_env), kNoRespawnSlotOverride);
                any_respawn = true;
            }
            host_respawn_slot_overrides[static_cast<size_t>(base + i)] =
                static_cast<uint8_t>(respawn_rng_env[static_cast<size_t>(env_idx)]() % 4);
        }
        return any_respawn ? host_respawn_slot_overrides.data() : nullptr;
    }
#endif

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
        if (kickoff_seed_mode == 3 || !cfg.mirror_official_respawn_rng || static_cast<size_t>(env_idx) >= respawn_rng_env.size()) {
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
        const detail::WorldSolverScratchRowCaps caps = detail::MakeWorldSolverScratchRowCaps(cfg.max_cars_per_env);
        rows_ball_split.resize(n * caps.ball_split);
        rows_ball_velocity.resize(n * caps.ball_velocity);
        rows_ball_friction.resize(n * caps.ball_friction);
        rows_pair_split.resize(n * caps.pair_split);
        rows_world_split.resize(n * caps.world_split);
        rows_pair_velocity.resize(n * caps.pair_velocity);
        rows_pair_friction.resize(n * caps.pair_friction);
        rows_world_velocity.resize(n * caps.world_velocity);
        rows_world_friction.resize(n * caps.world_friction);
        rows_shared_contact_order_groups.resize(n * caps.order_groups);
        rows_bullet_manifolds.resize(n * caps.manifolds);
        rows_pair_contact_cache.resize(n * caps.pair_cache_cars * caps.pair_cache_cars);
        rows_car_world_plane.resize(n * caps.car_world_plane);
        rows_car_world_mesh.resize(n * caps.car_world_mesh);
        rows_car_car_retained.resize(n * caps.car_car_retained);
        rows_ball_car_retained.resize(n * caps.ball_car_retained);
        rows_car_world_local.resize(n * caps.car_world_local);
        solver_row_pool = detail::WorldSolverScratchRowPool{};
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
        solver_row_pool.pair_contact_cache = rows_pair_contact_cache.data();
        solver_row_pool.car_world_plane_manifolds = rows_car_world_plane.data();
        solver_row_pool.car_world_mesh_manifolds = rows_car_world_mesh.data();
        solver_row_pool.car_car_retained_manifolds = rows_car_car_retained.data();
        solver_row_pool.ball_car_retained_manifolds = rows_ball_car_retained.data();
        solver_row_pool.car_world_local_rows = rows_car_world_local.data();
        solver_row_pool.env_count = cfg.env_count;
        solver_row_pool.interleaved = SolverRowsInterleavedDefault() ? 1 : 0;
        solver_row_pool.caps = caps;
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
        rows_pair_contact_cache.memset_zero(stream);
        rows_car_world_plane.memset_zero(stream);
        rows_car_world_mesh.memset_zero(stream);
        rows_car_car_retained.memset_zero(stream);
        rows_ball_car_retained.memset_zero(stream);
    }

    template <typename T>
    void reset_rows_one(detail::Buffer<T>& rows, int max_rows, size_t first, size_t count, cudaStream_t stream) {
        const size_t n = static_cast<size_t>(cfg.env_count);
        const size_t r = static_cast<size_t>(max_rows);
        const bool il = solver_row_pool.interleaved != 0;
#if defined(ROCKETGPU_USE_CUDA)
        if (il) {
            CheckCuda(cudaMemset2DAsync(rows.data() + first, n * sizeof(T), 0, count * sizeof(T), r, stream),
                      "cudaMemset2DAsync rows");
        } else {
            CheckCuda(cudaMemsetAsync(rows.data() + first * r, 0, count * r * sizeof(T), stream), "cudaMemsetAsync rows");
        }
#else
        for (size_t i = 0; i < (il ? r : 1); ++i) {
            T* p = il ? rows.data() + i * n + first : rows.data() + first * r;
            std::fill(p, p + (il ? count : count * r), T{});
        }
        (void)stream;
#endif
    }

    void reset_transient_rows(size_t first, size_t count, cudaStream_t stream) {
        const detail::WorldSolverScratchRowCaps& c = solver_row_pool.caps;
        reset_rows_one(rows_ball_split, c.ball_split, first, count, stream);
        reset_rows_one(rows_ball_velocity, c.ball_velocity, first, count, stream);
        reset_rows_one(rows_ball_friction, c.ball_friction, first, count, stream);
        reset_rows_one(rows_pair_split, c.pair_split, first, count, stream);
        reset_rows_one(rows_world_split, c.world_split, first, count, stream);
        reset_rows_one(rows_pair_velocity, c.pair_velocity, first, count, stream);
        reset_rows_one(rows_pair_friction, c.pair_friction, first, count, stream);
        reset_rows_one(rows_world_velocity, c.world_velocity, first, count, stream);
        reset_rows_one(rows_world_friction, c.world_friction, first, count, stream);
        reset_rows_one(rows_shared_contact_order_groups, c.order_groups, first, count, stream);
        reset_rows_one(rows_bullet_manifolds, c.manifolds, first, count, stream);
        reset_rows_one(rows_pair_contact_cache, c.pair_cache_cars * c.pair_cache_cars, first, count, stream);
    }

    void copy_solver_rows_from(const Impl& source, cudaStream_t stream) {
        if (source.solver_row_pool.interleaved != solver_row_pool.interleaved ||
            source.solver_row_pool.env_count != solver_row_pool.env_count ||
            source.cfg.max_cars_per_env != cfg.max_cars_per_env) {
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
        rows_pair_contact_cache.copy_from_buffer(source.rows_pair_contact_cache, rows_pair_contact_cache.size(), stream);
        rows_car_world_plane.copy_from_buffer(source.rows_car_world_plane, rows_car_world_plane.size(), stream);
        rows_car_world_mesh.copy_from_buffer(source.rows_car_world_mesh, rows_car_world_mesh.size(), stream);
        rows_car_car_retained.copy_from_buffer(source.rows_car_car_retained, rows_car_car_retained.size(), stream);
        rows_ball_car_retained.copy_from_buffer(source.rows_ball_car_retained, rows_ball_car_retained.size(), stream);
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
        out.reset(static_cast<int>(count), solver_row_pool.interleaved != 0, solver_row_pool.caps);
        if (count == 0) {
            return;
        }
        download_rows_one(rows_ball_split, out.ball_split_rows.data(), solver_row_pool.caps.ball_split, first, count, stream);
        download_rows_one(rows_ball_velocity, out.ball_velocity_rows.data(), solver_row_pool.caps.ball_velocity, first, count, stream);
        download_rows_one(rows_ball_friction, out.ball_friction_rows.data(), solver_row_pool.caps.ball_friction, first, count, stream);
        download_rows_one(rows_pair_split, out.pair_split_rows.data(), solver_row_pool.caps.pair_split, first, count, stream);
        download_rows_one(rows_world_split, out.world_split_rows.data(), solver_row_pool.caps.world_split, first, count, stream);
        download_rows_one(rows_pair_velocity, out.pair_velocity_rows.data(), solver_row_pool.caps.pair_velocity, first, count, stream);
        download_rows_one(rows_pair_friction, out.pair_friction_rows.data(), solver_row_pool.caps.pair_friction, first, count, stream);
        download_rows_one(rows_world_velocity, out.world_velocity_rows.data(), solver_row_pool.caps.world_velocity, first, count, stream);
        download_rows_one(rows_world_friction, out.world_friction_rows.data(), solver_row_pool.caps.world_friction, first, count, stream);
        download_rows_one(rows_shared_contact_order_groups, out.shared_contact_order_groups.data(), solver_row_pool.caps.order_groups, first, count, stream);
        download_rows_one(rows_bullet_manifolds, out.bullet_manifolds.data(), solver_row_pool.caps.manifolds, first, count, stream);
        download_rows_one(rows_pair_contact_cache, out.pair_contact_cache.data(),
                          solver_row_pool.caps.pair_cache_cars * solver_row_pool.caps.pair_cache_cars, first, count, stream);
        const detail::WorldSolverScratchRowCaps& caps = solver_row_pool.caps;
        rows_car_world_plane.copy_range_to_host(out.car_world_plane_manifolds.data(), first * caps.car_world_plane,
                                                count * caps.car_world_plane, stream);
        rows_car_world_mesh.copy_range_to_host(out.car_world_mesh_manifolds.data(), first * caps.car_world_mesh,
                                               count * caps.car_world_mesh, stream);
        rows_car_car_retained.copy_range_to_host(out.car_car_retained_manifolds.data(), first * caps.car_car_retained,
                                                 count * caps.car_car_retained, stream);
        rows_ball_car_retained.copy_range_to_host(out.ball_car_retained_manifolds.data(), first * caps.ball_car_retained,
                                                  count * caps.ball_car_retained, stream);
    }

    void upload_retained_rows(const detail::WorldSolverScratchRowPoolHost& in, size_t first, size_t count, cudaStream_t stream) {
        const detail::WorldSolverScratchRowCaps& caps = solver_row_pool.caps;
        if (in.car_world_plane_manifolds.size() < count * caps.car_world_plane ||
            in.car_world_mesh_manifolds.size() < count * caps.car_world_mesh ||
            in.car_car_retained_manifolds.size() < count * caps.car_car_retained ||
            in.ball_car_retained_manifolds.size() < count * caps.ball_car_retained) {
            throw std::invalid_argument("upload_retained_rows: host pool smaller than count");
        }
        rows_car_world_plane.copy_range_from_host(in.car_world_plane_manifolds.data(), first * caps.car_world_plane,
                                                  count * caps.car_world_plane, stream);
        rows_car_world_mesh.copy_range_from_host(in.car_world_mesh_manifolds.data(), first * caps.car_world_mesh,
                                                 count * caps.car_world_mesh, stream);
        rows_car_car_retained.copy_range_from_host(in.car_car_retained_manifolds.data(), first * caps.car_car_retained,
                                                   count * caps.car_car_retained, stream);
        rows_ball_car_retained.copy_range_from_host(in.ball_car_retained_manifolds.data(), first * caps.ball_car_retained,
                                                    count * caps.ball_car_retained, stream);
    }

    void reset_to_kickoff(uint64_t new_seed, cudaStream_t stream) {
        seed = new_seed;
        ensure_kickoff_settled(stream);
#if defined(ROCKETGPU_USE_CUDA)
        solver_scratch.memset_zero(stream);
        memset_zero_solver_rows(stream);
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
            car_configs.data(),
            stream);
#else
        cpu_envs().run(cfg.env_count, [&](int env_idx) {
            solver_scratch.data()[env_idx] = detail::WorldSolverScratch{};
            detail::ZeroWorldSolverScratchRows(solver_row_pool.view(static_cast<size_t>(env_idx)), solver_row_pool.caps);
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
                have_order ? kickoff_order : nullptr,
                false,
                car_configs.data());
        });
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
        size_t active_cmds = 0;
        bool settled_reset = false;
        for (size_t i = 0; i < count; ++i) {
            const EnvCommand& c = host_commands[i];
            active_cmds += static_cast<size_t>((c.reset_to_kickoff | c.reset_scoreboard | c.reset_tick_count | c.new_round_bodies) != 0);
            settled_reset = settled_reset || (c.reset_to_kickoff && c.kickoff_countdown_ticks == 0);
        }
        if (active_cmds == 0) {
            (void)stream;
            return;
        }
        if (settled_reset) {
            ensure_kickoff_settled(stream);
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
            car_configs.data(),
            stream);
#else
        auto apply_one = [&](int env) {
            const size_t env_idx = static_cast<size_t>(env);
            const EnvCommand& cmd = host_commands[env_idx];
            if ((cmd.reset_to_kickoff | cmd.reset_scoreboard | cmd.reset_tick_count | cmd.new_round_bodies) == 0) {
                return;
            }
            int kickoff_order[5];
            const bool have_order = cmd.reset_to_kickoff &&
                host_kickoff_order(static_cast<int>(env_idx), cmd.kickoff_seed,
                                   cmd.kickoff_seed_mode, kickoff_order);
            detail::ApplyEnvCommandBodies(solver_scratch.data()[env_idx], cmd,
                                          env_car_count.data()[env_idx],
                                          car_states.data() + env_idx * cfg.max_cars_per_env);
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
                cmd,
                nullptr,
                have_order ? kickoff_order : nullptr,
                car_configs.data());
            if (cmd.reset_to_kickoff) {
                const detail::BulletBroadphaseHistory bp = solver_scratch.data()[env_idx].bp;
                solver_scratch.data()[env_idx] = detail::WorldSolverScratch{};
                solver_scratch.data()[env_idx].bp = bp;
                detail::ZeroWorldSolverScratchRows(solver_row_pool.view(env_idx), solver_row_pool.caps);
            }
        };
        if (active_cmds < 16) {
            for (int env = 0; env < static_cast<int>(count); ++env) {
                apply_one(env);
            }
        } else {
            cpu_envs().run(static_cast<int>(count), apply_one);
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
        ensure_kickoff_settled(stream);
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
            car_configs.data(),
            stream);
#else
        apply_env_commands_host(device_commands, count, stream);
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
#if defined(ROCKETGPU_CPU_ISPC)
        if ((!cfg.mirror_official_respawn_rng || cfg.env_count >= 4) && cpu_blocks_pay()) {
            cpu_step_blocks(ticks, [&](int env_idx) {
                detail::InternalControlsAccessor a;
                a.controls = controls.data();
                a.base = env_idx * cfg.max_cars_per_env;
                return a;
            });
        } else
#endif
        if (!cfg.mirror_official_respawn_rng) {
            cpu_envs().run(cfg.env_count, [&](int env_idx) {
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
                    nullptr,
                    seed,
                    ticks);
            });
        } else {
            ensure_host_respawn_slot_overrides_sized();
            const float window_dt = static_cast<float>(ticks) * cfg.tick_dt;
            const bool fresh_demo_safe = ticks <= 16 && cfg.tick_dt > 0.0f && std::isnormal(cfg.tick_dt) &&
                                         cfg.mutator.demo_respawn_time > (ticks <= 4 ? window_dt : window_dt + cfg.tick_dt);
            cpu_envs().run(cfg.env_count, [&](int env_idx) {
                if (fresh_demo_safe && ticks > 1) {
                    const int base = env_idx * cfg.max_cars_per_env;
                    const int count = env_car_count.data()[static_cast<size_t>(env_idx)];
                    const CarState* const cars_ptr = car_states.data();
                    bool any_expiring_demo = false;
                    for (int i = 0; i < count; ++i) {
                        const CarState& car = cars_ptr[static_cast<size_t>(base + i)];
                        if (car.is_demoed && car.respawn_slot_override_plus1 < kKickoffSpawnOverridePlus1 &&
                            !(car.demo_respawn_timer > window_dt + cfg.tick_dt)) {
                            any_expiring_demo = true;
                            break;
                        }
                    }
                    if (!any_expiring_demo) {
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
                            nullptr,
                            seed,
                            ticks);
                        return;
                    }
                }
                for (int32_t tick = 0; tick < ticks; ++tick) {
                    const uint8_t* respawn_override_ptr = prepare_respawn_slot_overrides_env(env_idx);
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
            });
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
        if (control_stride != 0 && control_stride < cfg.max_cars_per_env * cfg.env_count) {
            throw std::invalid_argument("control_stride must be zero or at least the total car capacity");
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
#if defined(ROCKETGPU_CPU_ISPC)
        if ((!cfg.mirror_official_respawn_rng || cfg.env_count >= 4) && cpu_blocks_pay()) {
            cpu_step_blocks(ticks, [&](int env_idx) {
                detail::StreamControlsAccessor a;
                a.controls_by_tick = device_controls_by_tick;
                a.base = env_idx * cfg.max_cars_per_env;
                a.stride = control_stride;
                return a;
            });
        } else
#endif
        if (!cfg.mirror_official_respawn_rng) {
            cpu_envs().run(cfg.env_count, [&](int env_idx) {
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
                    device_controls_by_tick,
                    control_stride,
                    car_configs.data(),
                    ball_states.data(),
                    env_states.data(),
                    boost_pad_states.data(),
                    nullptr,
                    seed,
                    detail::kStepEnvPhaseFull,
                    ticks);
            });
        } else {
            ensure_host_respawn_slot_overrides_sized();
            const float window_dt = static_cast<float>(ticks) * cfg.tick_dt;
            const bool fresh_demo_safe = ticks <= 16 && cfg.tick_dt > 0.0f && std::isnormal(cfg.tick_dt) &&
                                         cfg.mutator.demo_respawn_time > (ticks <= 4 ? window_dt : window_dt + cfg.tick_dt);
            cpu_envs().run(cfg.env_count, [&](int env_idx) {
                if (fresh_demo_safe && ticks > 1) {
                    const int base = env_idx * cfg.max_cars_per_env;
                    const int count = env_car_count.data()[static_cast<size_t>(env_idx)];
                    const CarState* const cars_ptr = car_states.data();
                    bool any_expiring_demo = false;
                    for (int i = 0; i < count; ++i) {
                        const CarState& car = cars_ptr[static_cast<size_t>(base + i)];
                        if (car.is_demoed && car.respawn_slot_override_plus1 < kKickoffSpawnOverridePlus1 &&
                            !(car.demo_respawn_timer > window_dt + cfg.tick_dt)) {
                            any_expiring_demo = true;
                            break;
                        }
                    }
                    if (!any_expiring_demo) {
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
                            device_controls_by_tick,
                            control_stride,
                            car_configs.data(),
                            ball_states.data(),
                            env_states.data(),
                            boost_pad_states.data(),
                            nullptr,
                            seed,
                            detail::kStepEnvPhaseFull,
                            ticks);
                        return;
                    }
                }
                for (int32_t tick = 0; tick < ticks; ++tick) {
                    const uint8_t* respawn_override_ptr = prepare_respawn_slot_overrides_env(env_idx);
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
            });
        }
        std::memcpy(controls.data(),
                    device_controls_by_tick + static_cast<size_t>(ticks - 1) * static_cast<size_t>(control_stride),
                    static_cast<size_t>(cfg.env_count) * static_cast<size_t>(cfg.max_cars_per_env) * sizeof(CarControls));
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
        ensure_kickoff_settled(stream);
        if (control_stride != 0 && control_stride < cfg.max_cars_per_env * cfg.env_count) {
            throw std::invalid_argument("control_stride must be zero or at least the total car capacity");
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
#if defined(ROCKETGPU_CPU_ISPC)
            if ((!cfg.mirror_official_respawn_rng || cfg.env_count >= 4) && cpu_blocks_pay()) {
                cpu_step_blocks<true>(ticks, [&](int env_idx) {
                    detail::StreamControlsAccessor a;
                    a.controls_by_tick = device_controls_by_tick;
                    a.base = env_idx * cfg.max_cars_per_env;
                    a.stride = control_stride;
                    return a;
                }, device_env_commands_by_tick, env_command_stride);
            } else
#endif
            {
                const auto step_one = [&](int env_idx, int32_t tick, const uint8_t* respawn_override_ptr) {
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
                };
                cpu_step_envs(ticks, stream, step_one);
            }

            std::memcpy(controls.data(),
                        device_controls_by_tick + static_cast<size_t>(ticks - 1) * static_cast<size_t>(control_stride),
                        static_cast<size_t>(cfg.env_count) * static_cast<size_t>(cfg.max_cars_per_env) * sizeof(CarControls));
        }
        (void)stream;
#endif
    }

#if !defined(ROCKETGPU_USE_CUDA)
    CpuEnvPool& cpu_envs() {
        if (!cpu_pool) {
            cpu_pool = std::make_unique<CpuEnvPool>(cfg.env_count);
        }
        return *cpu_pool;
    }

#if defined(ROCKETGPU_CPU_ISPC)
    bool cpu_blocks_pay() const {
        if (cfg.env_count >= detail::kCpuBlockMinEnvs) {
            return true;
        }
        for (int e = 0; e < cfg.env_count; ++e) {
            if (env_car_count.data()[e] > 2) {
                return true;
            }
        }
        return false;
    }

    template <bool kReportEachTick = false, class ControlsFor>
    void cpu_step_blocks(int32_t ticks, const ControlsFor& controls_for,
                         const EnvCommand* commands_by_tick = nullptr, int command_stride = 0) {
        static const bool layout_ok = RgIspcLayoutOk();
        if (!layout_ok) {
            throw std::runtime_error("RocketGPU: ISPC struct layout differs from C++; run tools/gen_ispc_types.py src/cpu_ispc");
        }
        ensure_host_respawn_slot_overrides_sized();
        struct MirrorContext { Impl* impl; float window_dt; bool fresh_demo_safe; };
        MirrorContext mirror{this, 0.0f, false};
        if (cfg.mirror_official_respawn_rng && !kReportEachTick) {
            mirror.window_dt = static_cast<float>(ticks) * cfg.tick_dt;
            mirror.fresh_demo_safe = ticks <= 16 && cfg.tick_dt > 0.0f && std::isnormal(cfg.tick_dt) &&
                cfg.mutator.demo_respawn_time > (ticks <= 4 ? mirror.window_dt : mirror.window_dt + cfg.tick_dt);
        }
        const detail::CpuBlockRng rng{
            &mirror,
            [](void* ctx, int env) { return static_cast<MirrorContext*>(ctx)->impl->prepare_respawn_slot_overrides_env(env); },
            [](void* ctx, int env, uint64_t seed, uint8_t mode, int* order) {
                return static_cast<MirrorContext*>(ctx)->impl->host_kickoff_order(env, seed, mode, order);
            },
            [](void* ctx, int env, int ticks) {
                const auto* m = static_cast<MirrorContext*>(ctx);
                return !m->impl->mirror_respawn_window_safe(env, ticks, m->window_dt, m->fresh_demo_safe);
            }};
        CpuEnvPool& pool = cpu_envs();
        const int block = detail::CpuBlockEnvs(cfg.env_count, pool.concurrency());
        const int tail = detail::CpuTailEnvs(cfg.env_count, pool.concurrency(), block);
        const int head_envs = cfg.env_count - tail;
        const int head_blocks = head_envs > 0 ? 1 + (head_envs - 1) / block : 0;
        pool.run(head_blocks + tail, [&](int b) {
            const int env0 = b < head_blocks ? b * block : head_envs + (b - head_blocks);
            const int env_n = b < head_blocks ? std::min(block, head_envs - env0) : 1;
            detail::StepEnvBlockCpu<kReportEachTick>(env0, env_n, cfg, teams.data(), env_car_count.data(),
                                    car_states.data(), wheel_runtime_states.data(), solver_scratch.data(),
                                    solver_row_pool, controls_for, car_configs.data(), ball_states.data(),
                                    env_states.data(), boost_pad_states.data(), seed, ticks, commands_by_tick,
                                    command_stride, controls.data(), cfg.mirror_official_respawn_rng ? &rng : nullptr);
        });
    }

#endif

    template <class F>
    void cpu_step_envs(int32_t ticks, cudaStream_t stream, F&& one) {
        (void)stream;
        if (!cfg.mirror_official_respawn_rng) {
            cpu_envs().run(cfg.env_count, [&](int env_idx) {
                for (int32_t tick = 0; tick < ticks; ++tick) {
                    one(env_idx, tick, nullptr);
                }
            });
            return;
        }
        const size_t car_capacity = static_cast<size_t>(cfg.env_count) * static_cast<size_t>(cfg.max_cars_per_env);
        if (host_respawn_slot_overrides.size() != car_capacity) {
            host_respawn_slot_overrides.assign(car_capacity, kNoRespawnSlotOverride);
        }
        cpu_envs().run(cfg.env_count, [&](int env_idx) {
            for (int32_t tick = 0; tick < ticks; ++tick) {
                const uint8_t* overrides = prepare_respawn_slot_overrides_env(env_idx);
                one(env_idx, tick, overrides);
            }
        });
    }

    std::unique_ptr<CpuEnvPool> cpu_pool;
#endif

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
    impl_->user_cfg.mutator = mutator;
    impl_->kickoff_settled_dirty = true;
    impl_->kickoff_settled_pairs.clear();
    impl_->kickoff_settled_pair_rows.clear();
    impl_->kickoff_settled_pair_ball.clear();
}

void BatchSim::set_env_frame_frac(const float* host_values, size_t count, cudaStream_t stream) {
    if (count == 0) {
        impl_->cfg.env_frame_frac = nullptr;
        return;
    }
    if (count != env_count()) {
        throw std::invalid_argument("set_env_frame_frac: count must be env_count (or 0)");
    }
    if (impl_->env_frame_frac.size() != count) impl_->env_frame_frac.resize(count);
    impl_->env_frame_frac.copy_from_host(host_values, count, stream);
    impl_->cfg.env_frame_frac = impl_->env_frame_frac.data();
}

void BatchSim::set_env_frame_start_dt(const float* host_values, size_t count, cudaStream_t stream) {
    if (count == 0) {
        impl_->cfg.env_frame_start_dt = nullptr;
        return;
    }
    if (count != env_count()) {
        throw std::invalid_argument("set_env_frame_start_dt: count must be env_count (or 0)");
    }
    if (impl_->env_frame_start_dt.size() != count) impl_->env_frame_start_dt.resize(count);
    impl_->env_frame_start_dt.copy_from_host(host_values, count, stream);
    impl_->cfg.env_frame_start_dt = impl_->env_frame_start_dt.data();
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
    for (size_t i = 0; i < count; ++i) {
        if (host_counts[i] < 0 || host_counts[i] > impl_->cfg.max_cars_per_env) {
            throw std::out_of_range("set_env_car_count: car count outside [0, max_cars_per_env]");
        }
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
    impl_->kickoff_settled_dirty = true;
}

void BatchSim::set_car_configs(const CarConfig* host_configs, size_t count, cudaStream_t stream) {
    if (count > car_capacity()) {
        throw std::out_of_range("set_car_configs count exceeds car_capacity");
    }
    impl_->car_configs.copy_from_host(host_configs, count, stream);
    impl_->mark_exact_host_dirty();
    impl_->kickoff_settled_dirty = true;
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
    detail::LaunchCanonicalizeImportedBalls(impl_->ball_states.data(), static_cast<int32_t>(count),
                                            impl_->solver_scratch.data(), impl_->cfg.mutator, impl_->cfg.tick_dt, stream);
#else
    CanonicalizeImportedBallStatesInPlace(impl_->ball_states.data(), count);
    for (size_t idx = 0; idx < count; ++idx) {
        detail::PrimeImportedBallWorldPlanes(impl_->ball_states.data()[idx], impl_->cfg.mutator, impl_->cfg.tick_dt,
                                             impl_->solver_scratch.data()[idx]);
    }
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
        if (state.car_visit_order_mode != 1 && state.car_visit_order_mode != 3) {
            throw std::invalid_argument("upload_env_states: car_visit_order_mode must be 0, 1, 2 or 3");
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

void BatchSim::upload_solver_scratch_range_dev(const detail::WorldSolverScratch* host_states, size_t first, size_t count, cudaStream_t stream) {
    if (first > env_count() || count > env_count() - first) {
        throw std::out_of_range("upload_solver_scratch_range_dev exceeds env_count");
    }
    impl_->solver_scratch.copy_range_from_host(host_states, first, count, stream);
    impl_->reset_transient_rows(first, count, stream);
    impl_->mark_exact_host_dirty();
}

void BatchSim::upload_solver_retained_rows_dev(const detail::WorldSolverScratchRowPoolHost& in, size_t first, size_t count, cudaStream_t stream) {
    if (first > env_count() || count > env_count() - first) {
        throw std::out_of_range("upload_solver_retained_rows_dev exceeds env_count");
    }
    impl_->upload_retained_rows(in, first, count, stream);
    impl_->mark_exact_host_dirty();
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
