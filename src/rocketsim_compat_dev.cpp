#include <rocketgpu/rocketsim_compat.h>

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define ROCKETGPU_STEP_TYPES_ONLY 1
#include "rocketgpu/detail/step.hpp"

namespace rocketgpu {
namespace compat {
namespace {

template <class T>
std::vector<uint8_t> Bytes(const std::vector<T>& v, size_t n) {
    std::vector<uint8_t> out(n * sizeof(T));
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

template <class T>
void Fill(std::vector<T>& v, size_t n, const std::vector<uint8_t>& bytes, const std::string& what) {
    if (bytes.size() != n * sizeof(T)) {
        throw std::invalid_argument("DevSetRaw " + what + ": expected " + std::to_string(n * sizeof(T)) + " bytes");
    }
    v.resize(n);
    std::memcpy(v.data(), bytes.data(), bytes.size());
}

void CheckSize(size_t n, size_t elem_size, const std::vector<uint8_t>& bytes, const std::string& what) {
    if (bytes.size() != n * elem_size) {
        throw std::invalid_argument("DevSetRaw " + what + ": expected " + std::to_string(n * elem_size) + " bytes");
    }
}

}

std::vector<uint8_t> Arena::DevGetRaw(const std::string& what) const {
    BatchSim& s = *sim_;
    const size_t cars = s.car_capacity();
    if (what == "car_states") {
        std::vector<uint8_t> out(cars * sizeof(CarState));
        s.download_car_states(reinterpret_cast<CarState*>(out.data()), cars);
        return out;
    }
    if (what == "ball_states") {
        std::vector<uint8_t> out(sizeof(BallState));
        s.download_ball_states(reinterpret_cast<BallState*>(out.data()), 1);
        return out;
    }
    if (what == "env_states") {
        std::vector<uint8_t> out(sizeof(EnvState));
        s.download_env_states(reinterpret_cast<EnvState*>(out.data()), 1);
        return out;
    }
    if (what == "wheel_runtime") {
        std::vector<uint8_t> out(cars * sizeof(detail::WheelRuntimeState));
        s.download_wheel_runtime_states(reinterpret_cast<detail::WheelRuntimeState*>(out.data()), cars);
        return out;
    }
    if (what == "solver_scratch") {
        std::vector<uint8_t> out(sizeof(detail::WorldSolverScratch));
        s.download_solver_scratch_range(reinterpret_cast<detail::WorldSolverScratch*>(out.data()), 0, 1);
        return out;
    }
    thread_local detail::WorldSolverScratchRowPoolHost rows;
    s.download_solver_scratch_rows(rows, 0, 1);
    if (what == "car_world_plane") return Bytes(rows.car_world_plane_manifolds, rows.car_world_plane_manifolds.size());
    if (what == "car_world_mesh") return Bytes(rows.car_world_mesh_manifolds, rows.car_world_mesh_manifolds.size());
    if (what == "car_car") return Bytes(rows.car_car_retained_manifolds, rows.car_car_retained_manifolds.size());
    if (what == "ball_car") return Bytes(rows.ball_car_retained_manifolds, rows.ball_car_retained_manifolds.size());
    throw std::invalid_argument("DevGetRaw: unknown array " + what);
}

void Arena::DevSetRaw(const std::string& what, const std::vector<uint8_t>& bytes) {
    BatchSim& s = *sim_;
    const size_t cars = s.car_capacity();
    if (what == "car_states") {
        Fill(car_states_, cars, bytes, what);
        s.upload_car_states(car_states_.data(), cars, nullptr, BatchSim::StateUploadMode::PreserveInternal);
    } else if (what == "ball_states") {
        Fill(ball_states_, 1, bytes, what);
        s.upload_ball_states(ball_states_.data(), 1, nullptr, BatchSim::StateUploadMode::PreserveInternal);
    } else if (what == "env_states") {
        CheckSize(1, sizeof(EnvState), bytes, what);
        s.upload_env_states(reinterpret_cast<const EnvState*>(bytes.data()), 1);
    } else if (what == "wheel_runtime") {
        CheckSize(cars, sizeof(detail::WheelRuntimeState), bytes, what);
        s.upload_wheel_runtime_states(reinterpret_cast<const detail::WheelRuntimeState*>(bytes.data()), cars);
    } else if (what == "solver_scratch") {
        CheckSize(1, sizeof(detail::WorldSolverScratch), bytes, what);
        s.upload_solver_scratch_range_dev(reinterpret_cast<const detail::WorldSolverScratch*>(bytes.data()), 0, 1);
    } else {
        thread_local detail::WorldSolverScratchRowPoolHost rows;
        s.download_solver_scratch_rows(rows, 0, 1);
        if (what == "car_world_plane") Fill(rows.car_world_plane_manifolds, rows.car_world_plane_manifolds.size(), bytes, what);
        else if (what == "car_world_mesh") Fill(rows.car_world_mesh_manifolds, rows.car_world_mesh_manifolds.size(), bytes, what);
        else if (what == "car_car") Fill(rows.car_car_retained_manifolds, rows.car_car_retained_manifolds.size(), bytes, what);
        else if (what == "ball_car") Fill(rows.ball_car_retained_manifolds, rows.ball_car_retained_manifolds.size(), bytes, what);
        else throw std::invalid_argument("DevSetRaw: unknown array " + what);
        s.upload_solver_retained_rows_dev(rows, 0, 1);
    }
    dirty_ = true;
}

template <class S, class F>
void DevAdd(std::vector<Arena::DevField>& out, const char* name, F f) {
    if constexpr (std::is_invocable_v<F, S*>) {
        alignas(S) static unsigned char buf[sizeof(S)];
        S* s = reinterpret_cast<S*>(buf);
        const auto* m = f(s);
        out.push_back({name, static_cast<size_t>(reinterpret_cast<const unsigned char*>(m) - buf), sizeof(*m)});
    }
}

#define RG_DEV_F(S, path) DevAdd<S>(out, #S "." #path, [](auto* p) -> decltype(&p->path) { return &p->path; })
#define RG_DEV_S(S) out.push_back({#S, 0, sizeof(S)})

std::vector<Arena::DevField> Arena::DevLayouts() {
    static const std::vector<DevField> cached = [] {
        using namespace rocketgpu::detail;
        std::vector<DevField> out;
    RG_DEV_S(CarState);
    RG_DEV_F(CarState, pos); RG_DEV_F(CarState, rot); RG_DEV_F(CarState, vel); RG_DEV_F(CarState, ang_vel);
    RG_DEV_F(CarState, flip_rel_torque); RG_DEV_F(CarState, jump_time); RG_DEV_F(CarState, flip_time);
    RG_DEV_F(CarState, air_time); RG_DEV_F(CarState, air_time_since_jump); RG_DEV_F(CarState, boost);
    RG_DEV_F(CarState, time_since_boosted); RG_DEV_F(CarState, boosting_time); RG_DEV_F(CarState, supersonic_time);
    RG_DEV_F(CarState, handbrake_val); RG_DEV_F(CarState, auto_flip_timer); RG_DEV_F(CarState, auto_flip_torque_scale);
    RG_DEV_F(CarState, demo_respawn_timer); RG_DEV_F(CarState, tick_count_since_update);
    RG_DEV_F(CarState, ball_contact_last_tick); RG_DEV_F(CarState, ball_contact_this_tick);
    RG_DEV_F(CarState, is_on_ground); RG_DEV_F(CarState, wheels_with_contact); RG_DEV_F(CarState, has_jumped);
    RG_DEV_F(CarState, has_double_jumped); RG_DEV_F(CarState, has_flipped); RG_DEV_F(CarState, is_flipping);
    RG_DEV_F(CarState, is_jumping); RG_DEV_F(CarState, is_boosting); RG_DEV_F(CarState, is_supersonic);
    RG_DEV_F(CarState, is_auto_flipping); RG_DEV_F(CarState, is_demoed); RG_DEV_F(CarState, just_respawned);
    RG_DEV_F(CarState, respawn_slot_override_plus1);
    RG_DEV_F(CarState, world_contact.has_contact); RG_DEV_F(CarState, world_contact.contact_normal);
    RG_DEV_F(CarState, car_contact.other_car_id); RG_DEV_F(CarState, car_contact.cooldown_timer);
    RG_DEV_F(CarState, vel_impulse_cache); RG_DEV_F(CarState, last_controls); RG_DEV_F(CarState, pos_bt);
    RG_DEV_F(CarState, vel_bt); RG_DEV_F(CarState, vel_touched_solve_stage); RG_DEV_F(CarState, basis_forward);
    RG_DEV_F(CarState, basis_right); RG_DEV_F(CarState, basis_up); RG_DEV_F(CarState, basis_authoritative);
    RG_DEV_F(CarState, ext_force_impulse_bt_valid); RG_DEV_F(CarState, vel_impulse_cache_bt);
    RG_DEV_F(CarState, internal_jump_time); RG_DEV_F(CarState, internal_flip_time); RG_DEV_F(CarState, air_state);
    RG_DEV_F(CarState, dodge_timeout); RG_DEV_F(CarState, dodge_elapsed); RG_DEV_F(CarState, internal_vel);
    RG_DEV_F(CarState, internal_ang_vel); RG_DEV_F(CarState, reported_vel); RG_DEV_F(CarState, reported_ang_vel);
    RG_DEV_F(CarState, internal_has_jumped); RG_DEV_F(CarState, internal_has_double_jumped);
    RG_DEV_F(CarState, internal_has_flipped); RG_DEV_F(CarState, internal_is_supersonic);
    RG_DEV_F(CarState, reported_flags); RG_DEV_F(CarState, game_has_jumped); RG_DEV_F(CarState, double_jump_ticks);
    RG_DEV_F(CarState, jump_hold_broken); RG_DEV_F(CarState, supersonic_set_v3); RG_DEV_F(CarState, set_wheel_normals);
    RG_DEV_F(CarState, set_wheel_normals_mask); RG_DEV_F(CarState, cal_tick_start_ang_vel);
    RG_DEV_F(CarState, restored_inv_inertia_bt); RG_DEV_F(CarState, restored_inv_inertia_valid);
    RG_DEV_F(CarState, ball_hit_info.is_valid); RG_DEV_F(CarState, ball_hit_info.tick_count_when_hit);
    RG_DEV_F(CarState, ball_hit_info.tick_count_when_extra_impulse_applied); RG_DEV_F(CarState, ball_hit_info.extra_tick_is_previous);
    RG_DEV_F(CarState, ball_hit_info.hit_car_id); RG_DEV_F(CarState, latest_touch_tick);
    RG_DEV_F(CarState, driving_disabled); RG_DEV_F(CarState, boost_pad_touching);
    RG_DEV_F(CarState, pad_frame_pos_bt); RG_DEV_F(CarState, pad_frame_pos_valid);

    RG_DEV_S(BallState);
    RG_DEV_F(BallState, pos); RG_DEV_F(BallState, rot); RG_DEV_F(BallState, vel); RG_DEV_F(BallState, ang_vel);
    RG_DEV_F(BallState, tick_count_since_update); RG_DEV_F(BallState, pos_bt); RG_DEV_F(BallState, vel_bt);
    RG_DEV_F(BallState, basis_forward); RG_DEV_F(BallState, basis_right); RG_DEV_F(BallState, basis_up);
    RG_DEV_F(BallState, basis_authoritative); RG_DEV_F(BallState, vel_impulse_cache_bt);
    RG_DEV_F(BallState, sleeping); RG_DEV_F(BallState, deactivation_time); RG_DEV_F(BallState, constrained_2d);
    RG_DEV_F(BallState, tick_count_when_extra_impulse_applied);
    RG_DEV_F(BallState, restored_inv_inertia_bt); RG_DEV_F(BallState, restored_inv_inertia_valid);

    RG_DEV_S(EnvState);
    RG_DEV_F(EnvState, tick); RG_DEV_F(EnvState, car_visit_order_mode); RG_DEV_F(EnvState, car_visit_order_packed);
    RG_DEV_F(EnvState, countdown_phase); RG_DEV_F(EnvState, countdown_fires); RG_DEV_F(EnvState, frame_ticks);
    RG_DEV_F(EnvState, countdown_count); RG_DEV_F(EnvState, frame_frac_prev);

    RG_DEV_S(WheelRuntimeState);
    RG_DEV_F(WheelRuntimeState, event_world_ids); RG_DEV_F(WheelRuntimeState, event_world_normals);
    RG_DEV_F(WheelRuntimeState, steer_angle); RG_DEV_F(WheelRuntimeState, engine_force); RG_DEV_F(WheelRuntimeState, brake);
    RG_DEV_F(WheelRuntimeState, lat_friction); RG_DEV_F(WheelRuntimeState, long_friction);
    RG_DEV_F(WheelRuntimeState, extra_pushback); RG_DEV_F(WheelRuntimeState, cal_prev_sticky_dir);
    RG_DEV_F(WheelRuntimeState, cal_prev_wheel_normal_sum); RG_DEV_F(WheelRuntimeState, cal_prev_world_wheel_contacts);
    RG_DEV_F(WheelRuntimeState, cal_game_jumped); RG_DEV_F(WheelRuntimeState, cal_game_double_jumped);
    RG_DEV_F(WheelRuntimeState, cal_cached_ground); RG_DEV_F(WheelRuntimeState, cal_jump_inactive_time);
    RG_DEV_F(WheelRuntimeState, cal_frame_open); RG_DEV_F(WheelRuntimeState, cal_restored);

    RG_DEV_S(WorldSolverScratch);
    RG_DEV_F(WorldSolverScratch, bp.sweep.sap_handle_count); RG_DEV_F(WorldSolverScratch, bp.sweep.sap_static_count);
    RG_DEV_F(WorldSolverScratch, bp.sweep.sap_next_fresh); RG_DEV_F(WorldSolverScratch, bp.sweep.sap_free_count);
    RG_DEV_F(WorldSolverScratch, bp.sweep.sap_free); RG_DEV_F(WorldSolverScratch, bp.sweep.sap_body_handle);
    RG_DEV_F(WorldSolverScratch, bp.sweep.sap_obj_count); RG_DEV_F(WorldSolverScratch, bp.sweep.sap_obj);
    RG_DEV_F(WorldSolverScratch, bp.sweep.sap_edge_pos); RG_DEV_F(WorldSolverScratch, bp.sweep.sap_edge_handle);
    RG_DEV_F(WorldSolverScratch, bp.sweep.sap_min_edge); RG_DEV_F(WorldSolverScratch, bp.sweep.sap_max_edge);
    RG_DEV_F(WorldSolverScratch, bp.sweep.sap_static_id);
    RG_DEV_F(WorldSolverScratch, bp.sweep.sap_body_aabb); RG_DEV_F(WorldSolverScratch, bp.sweep.sap_body_aabb_valid);
    RG_DEV_F(WorldSolverScratch, bp.bp_pair_count); RG_DEV_F(WorldSolverScratch, bp.bp_manifold_count);
    RG_DEV_F(WorldSolverScratch, bp.bp_pairs); RG_DEV_F(WorldSolverScratch, bp.bp_manifolds);
    RG_DEV_F(WorldSolverScratch, bp.bp_predictive_count); RG_DEV_F(WorldSolverScratch, bp.bp_predictive);
    RG_DEV_F(WorldSolverScratch, ball_world_mesh_manifolds); RG_DEV_F(WorldSolverScratch, ball_world_plane_manifolds);
    RG_DEV_F(WorldSolverScratch, ball_mesh_maybe_retained); RG_DEV_F(WorldSolverScratch, pair_cache_maybe_valid);
    RG_DEV_F(WorldSolverScratch, retained_reset_pending); RG_DEV_F(WorldSolverScratch, car_world_reset);
    RG_DEV_F(WorldSolverScratch, predictive_hit_mask); RG_DEV_F(WorldSolverScratch, predictive_staged);
    RG_DEV_F(WorldSolverScratch, predictive_normal); RG_DEV_F(WorldSolverScratch, predictive_distance_bt);
    RG_DEV_F(WorldSolverScratch, ball_settled); RG_DEV_F(WorldSolverScratch, cal_start_vel);
    RG_DEV_F(WorldSolverScratch, cal_ball_start_vel); RG_DEV_F(WorldSolverScratch, ball_predict_vel_bt); RG_DEV_F(WorldSolverScratch, car_predict_vel); RG_DEV_F(WorldSolverScratch, car_predict_ang_vel); RG_DEV_F(WorldSolverScratch, car_predict_valid);
    RG_DEV_F(WorldSolverScratch, w3_prefix_done); RG_DEV_F(WorldSolverScratch, pair_cache_cars);

    RG_DEV_S(BallMeshPersistentManifold);
    RG_DEV_F(BallMeshPersistentManifold, count); RG_DEV_F(BallMeshPersistentManifold, points);
    RG_DEV_S(BallMeshManifoldPoint);
    RG_DEV_F(BallMeshManifoldPoint, normal); RG_DEV_F(BallMeshManifoldPoint, pen_uu);
    RG_DEV_F(BallMeshManifoldPoint, local_point_a_bt); RG_DEV_F(BallMeshManifoldPoint, local_point_b_bt);
    RG_DEV_F(BallMeshManifoldPoint, triangle_index); RG_DEV_F(BallMeshManifoldPoint, rel_pos1_length_bt);
    RG_DEV_F(BallMeshManifoldPoint, rel_pos1_bt); RG_DEV_F(BallMeshManifoldPoint, distance_bt);
    RG_DEV_S(CarWorldPlaneRetainedManifold);
    RG_DEV_F(CarWorldPlaneRetainedManifold, count); RG_DEV_F(CarWorldPlaneRetainedManifold, points);
    RG_DEV_S(CarWorldPlaneRetainedPoint);
    RG_DEV_F(CarWorldPlaneRetainedPoint, local_point_a_bt); RG_DEV_F(CarWorldPlaneRetainedPoint, point_on_b_bt);
    RG_DEV_F(CarWorldPlaneRetainedPoint, distance_bt);
    RG_DEV_S(CarWorldMeshRetainedManifold);
    RG_DEV_F(CarWorldMeshRetainedManifold, count); RG_DEV_F(CarWorldMeshRetainedManifold, mesh);
    RG_DEV_F(CarWorldMeshRetainedManifold, points);
    RG_DEV_S(CarWorldMeshRetainedPoint);
    RG_DEV_F(CarWorldMeshRetainedPoint, local_point_a_bt); RG_DEV_F(CarWorldMeshRetainedPoint, local_point_b_bt);
    RG_DEV_F(CarWorldMeshRetainedPoint, normal); RG_DEV_F(CarWorldMeshRetainedPoint, distance_bt);
    RG_DEV_S(CarCarRetainedManifold);
    RG_DEV_F(CarCarRetainedManifold, count); RG_DEV_F(CarCarRetainedManifold, points);
    RG_DEV_S(CarCarRetainedPoint);
    RG_DEV_F(CarCarRetainedPoint, local_a_bt); RG_DEV_F(CarCarRetainedPoint, local_b_bt);
    RG_DEV_F(CarCarRetainedPoint, pos_a_bt); RG_DEV_F(CarCarRetainedPoint, pos_b_bt);
    RG_DEV_F(CarCarRetainedPoint, normal); RG_DEV_F(CarCarRetainedPoint, distance_bt);
    RG_DEV_S(BallCarRetainedManifold);
    RG_DEV_F(BallCarRetainedManifold, count); RG_DEV_F(BallCarRetainedManifold, points);
    RG_DEV_S(BallCarRetainedPoint);
    RG_DEV_F(BallCarRetainedPoint, local_a_bt); RG_DEV_F(BallCarRetainedPoint, local_b_bt);
    RG_DEV_F(BallCarRetainedPoint, pos_a_bt); RG_DEV_F(BallCarRetainedPoint, pos_b_bt);
    RG_DEV_F(BallCarRetainedPoint, normal); RG_DEV_F(BallCarRetainedPoint, distance_bt);
    return out;
    }();
    return cached;
}

#undef RG_DEV_F
#undef RG_DEV_S

namespace {

constexpr std::pair<const char*, const char*> kFullStateArrays[] = {
    {"car_states", "CarState"}, {"ball_states", "BallState"}, {"env_states", "EnvState"},
    {"wheel_runtime", "WheelRuntimeState"}, {"solver_scratch", "WorldSolverScratch"},
    {"car_world_plane", "CarWorldPlaneRetainedManifold"}, {"car_world_mesh", "CarWorldMeshRetainedManifold"},
    {"car_car", "CarCarRetainedManifold"}, {"ball_car", "BallCarRetainedManifold"}};

// Fields that hold an array of another listed struct: written field by field, so the inner layout can change too.
constexpr std::pair<const char*, const char*> kNestedFields[] = {
    {"WorldSolverScratch.ball_world_mesh_manifolds", "BallMeshPersistentManifold"},
    {"WorldSolverScratch.ball_world_plane_manifolds", "BallMeshPersistentManifold"},
    {"BallMeshPersistentManifold.points", "BallMeshManifoldPoint"},
    {"CarWorldPlaneRetainedManifold.points", "CarWorldPlaneRetainedPoint"},
    {"CarWorldMeshRetainedManifold.points", "CarWorldMeshRetainedPoint"},
    {"CarCarRetainedManifold.points", "CarCarRetainedPoint"},
    {"BallCarRetainedManifold.points", "BallCarRetainedPoint"}};

struct FieldSlice {
    std::string key;
    std::vector<size_t> starts;   // byte offset of the field in each element, in order
    size_t size;
};

// Every leaf field of struct `type` in the elements starting at `starts`, named prefix + field path.
void FullStateFields(const std::string& type, const std::vector<size_t>& starts, const std::string& prefix,
                     std::vector<FieldSlice>& out) {
    const std::vector<Arena::DevField>& layouts = Arena::DevLayouts();
    const std::string head = type + ".";
    for (const Arena::DevField& f : layouts) {
        if (f.name.compare(0, head.size(), head) != 0) continue;
        const std::string path = f.name.substr(head.size());
        const char* nested = nullptr;
        for (const auto& [field, elem] : kNestedFields) {
            if (f.name == field) nested = elem;
        }
        size_t elem_size = 0;
        for (const Arena::DevField& g : layouts) {
            if (nested != nullptr && g.name == nested) elem_size = g.size;
        }
        if (elem_size == 0) {
            std::vector<size_t> at;
            at.reserve(starts.size());
            for (size_t s : starts) at.push_back(s + f.offset);
            out.push_back({prefix + path, std::move(at), f.size});
            continue;
        }
        std::vector<size_t> inner;
        for (size_t s : starts) {
            for (size_t k = 0; k < f.size / elem_size; ++k) inner.push_back(s + f.offset + k * elem_size);
        }
        FullStateFields(nested, inner, prefix + path + ".", out);
    }
}

std::vector<FieldSlice> ArrayFields(const char* array, const char* type, size_t bytes) {
    size_t size = 0;
    for (const Arena::DevField& g : Arena::DevLayouts()) {
        if (g.name == type) size = g.size;
    }
    std::vector<size_t> starts;
    for (size_t s = 0; size != 0 && s + size <= bytes; s += size) starts.push_back(s);
    std::vector<FieldSlice> out;
    FullStateFields(type, starts, std::string(array) + ".", out);
    return out;
}

}

std::map<std::string, std::vector<uint8_t>> Arena::GetFullState(const std::vector<std::string>& arrays) const {
    std::map<std::string, std::vector<uint8_t>> out;
    for (const std::string& a : arrays) {
        bool known = false;
        for (const auto& [array, type] : kFullStateArrays) known = known || a == array;
        if (!known) throw std::invalid_argument("GetFullState: unknown array " + a);
    }
    for (const auto& [array, type] : kFullStateArrays) {
        bool want = arrays.empty();
        for (const std::string& a : arrays) want = want || a == array;
        if (!want) continue;
        const std::vector<uint8_t> raw = DevGetRaw(array);
        for (const FieldSlice& f : ArrayFields(array, type, raw.size())) {
            std::vector<uint8_t>& v = out[f.key];
            v.reserve(f.starts.size() * f.size);
            for (size_t s : f.starts) v.insert(v.end(), raw.begin() + s, raw.begin() + s + f.size);
        }
    }
    return out;
}

void Arena::SetFullState(const std::map<std::string, std::vector<uint8_t>>& state) {
    struct Pending {
        const char* array;
        std::vector<uint8_t> raw;
        std::vector<std::pair<const FieldSlice*, const std::vector<uint8_t>*>> writes;
        std::vector<FieldSlice> fields;
    };
    std::vector<Pending> arrays;
    std::map<std::string, bool> seen;
    for (const auto& [array, type] : kFullStateArrays) {
        const std::string head = std::string(array) + ".";
        const auto first = state.lower_bound(head);
        if (first == state.end() || first->first.compare(0, head.size(), head) != 0) continue;
        Pending p{array, DevGetRaw(array), {}, {}};
        p.fields = ArrayFields(array, type, p.raw.size());
        arrays.push_back(std::move(p));
    }
    size_t used = 0;
    for (Pending& p : arrays) {
        for (const FieldSlice& f : p.fields) {
            const auto it = state.find(f.key);
            if (it == state.end()) continue;
            if (it->second.size() != f.starts.size() * f.size) {
                throw std::invalid_argument("SetFullState " + f.key + ": expected " +
                                            std::to_string(f.starts.size() * f.size) + " bytes, got " +
                                            std::to_string(it->second.size()));
            }
            p.writes.push_back({&f, &it->second});
            seen[f.key] = true;
            ++used;
        }
    }
    if (used != state.size()) {
        for (const auto& kv : state) {
            if (!seen.count(kv.first)) throw std::invalid_argument("SetFullState: unknown field " + kv.first);
        }
    }
    for (Pending& p : arrays) {
        for (const auto& [f, v] : p.writes) {
            for (size_t k = 0; k < f->starts.size(); ++k) {
                std::memcpy(p.raw.data() + f->starts[k], v->data() + k * f->size, f->size);
            }
        }
        DevSetRaw(p.array, p.raw);
    }
}

}
}
