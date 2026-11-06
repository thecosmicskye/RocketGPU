// CPU lane kernels for the car pre-tick (lanes = cars, gathered from the AoS CarState array).
#include "rg_types.isph"
#include "rg_math.isph"
#include "rg_layout.isph"

struct RgPreTickParams {
    float dt;
    float car_mass;
    float flip_torque_time;
    float supersonic_start_speed;
    float supersonic_maintain_max_time;
    float supersonic_maintain_min_speed;
    float car_max_speed;
    float car_max_ang_speed;
};

static const uniform float kUuToBt = 1.0f / 50.0f;

// uint8 fields stay in registers for the whole kernel: their aligned 32-bit words are gathered once at the start and
// the written ones scattered once at the end. Lanes are distinct cars, so each word belongs to one car.
#define RG_FIELD_OFF(T, f) ((uniform int32)((uniform int8 * uniform)(&((uniform T * uniform)0)->f) - (uniform int8 * uniform)0))
#define RG_CAR_WORDS 11
#define RG_WRS_WORDS 2
// CarState bytes 120..139, 312..315, 500..503, 600..611 and 660..663; WheelRuntimeState bytes 280..283 and 288..291.
static const uniform int32 kCarWordOff[RG_CAR_WORDS] = {120, 124, 128, 132, 136, 312, 500, 600, 604, 608, 660};
static const uniform int32 kWrsWordOff[RG_WRS_WORDS] = {280, 288};
struct RgFlagWords {
    uint32 c[RG_CAR_WORDS];
    uint32 w[RG_WRS_WORDS];
};
static inline uniform int32 CarWordSlot(const uniform int32 off) {
    return off < 140 ? ((off - 120) >> 2) : off < 316 ? 5 : off < 504 ? 6 : off < 612 ? 7 + ((off - 600) >> 2) : 10;
}
static inline uniform int32 WrsWordSlot(const uniform int32 off) { return off < 284 ? 0 : 1; }
static inline uint8 WordByte(const uint32 w, const uniform int32 off) { return (uint8)((w >> ((off & 3) * 8)) & 0xFFu); }
static inline void SetWordByte(uint32& w, const uniform int32 off, const int v) {
    const uniform uint32 shift = (off & 3) * 8;
    w = (w & ~(0xFFu << shift)) | (((uint32)v & 0xFFu) << shift);
}
#define CAR_GETB(a, i, f) WordByte(fl.c[CarWordSlot(RG_FIELD_OFF(CarState, f))], RG_FIELD_OFF(CarState, f))
#define CAR_SETB(a, i, f, v) SetWordByte(fl.c[CarWordSlot(RG_FIELD_OFF(CarState, f))], RG_FIELD_OFF(CarState, f), (v))
#define WRS_GETB(a, i, f) WordByte(fl.w[WrsWordSlot(RG_FIELD_OFF(WheelRuntimeState, f))], RG_FIELD_OFF(WheelRuntimeState, f))
#define WRS_SETB(a, i, f, v) SetWordByte(fl.w[WrsWordSlot(RG_FIELD_OFF(WheelRuntimeState, f))], RG_FIELD_OFF(WheelRuntimeState, f), (v))
static inline void LoadFlagWords(uniform CarState cars[], uniform WheelRuntimeState wrs[], const int ci, RgFlagWords& fl) {
    uniform int8 * uniform cb = (uniform int8 * uniform)cars;
    uniform int8 * uniform wb = (uniform int8 * uniform)wrs;
#pragma unroll
    for (uniform int k = 0; k < RG_CAR_WORDS; ++k) {
        fl.c[k] = *((const uniform uint32 * varying)(cb + ci * (uniform int32)sizeof(uniform CarState) + kCarWordOff[k]));
    }
#pragma unroll
    for (uniform int k = 0; k < RG_WRS_WORDS; ++k) {
        fl.w[k] = *((const uniform uint32 * varying)(wb + ci * (uniform int32)sizeof(uniform WheelRuntimeState) + kWrsWordOff[k]));
    }
}
// Words 500, 600 and 604 are only read.
static const uniform int32 kCarStoreSlots[8] = {0, 1, 2, 3, 4, 5, 9, 10};
static inline void StoreFlagWords(uniform CarState cars[], uniform WheelRuntimeState wrs[], const int ci, const RgFlagWords& fl) {
    uniform int8 * uniform cb = (uniform int8 * uniform)cars;
    uniform int8 * uniform wb = (uniform int8 * uniform)wrs;
#pragma unroll
    for (uniform int k = 0; k < 8; ++k) {
        *((uniform uint32 * varying)(cb + ci * (uniform int32)sizeof(uniform CarState) + kCarWordOff[kCarStoreSlots[k]])) = fl.c[kCarStoreSlots[k]];
    }
#pragma unroll
    for (uniform int k = 0; k < RG_WRS_WORDS; ++k) {
        *((uniform uint32 * varying)(wb + ci * (uniform int32)sizeof(uniform WheelRuntimeState) + kWrsWordOff[k])) = fl.w[k];
    }
}

// Car float fields the kernel reads more than once, kept in registers: loaded once, stored once at the end.
struct RgCarHot {
    Vec3 pos_bt;
    Vec3 vel;
    Vec3 vel_bt;
    Vec3 ang_vel;
    float internal_flip_time;
    Quat rot;
    Vec3 basis_forward;
    Vec3 basis_right;
    Vec3 basis_up;
};

// raycast_wheels_and_contact_prep.hpp QuantRoundPosV3
static inline float QuantRoundPosV3(float x, uniform float scale) {
    const float scaled = x * scale;
    return floor(scaled + 0.5f) * (1.0f / scale);
}

// raycast_wheels_and_contact_prep.hpp QuantTruncOffsetV3
static inline float QuantTruncOffsetV3(float x, uniform float scale) {
    const uniform float inv_scale = 1.0f / scale;
    const float k = (float)((int32)(x * scale));
    const float rounded = k * inv_scale;
    const uniform float offset_mag = 0.1f * inv_scale;
    const float sgn = rounded == 0.0f ? 0.0f : (rounded > 0.0f ? 1.0f : -1.0f);
    const float offset = sgn * offset_mag;
    return rounded + offset;
}

// CalTickStartLimitAndQuantizeCar for cars that are not demoed, without the native SSE rsqrt table
// (raycast_wheels_and_contact_prep.hpp), including QuantizeCarState(car, true, &vel_uu, true).
static inline void TickStartLimitAndQuantize(uniform CarState cars[], const int ci, RgFlagWords& fl, RgCarHot& h, uniform float car_max_speed,
                                             uniform float car_max_ang_speed) {
    {
        Vec3 vel_bt = h.vel_bt;
        Vec3 vel_uu = Scale(vel_bt, 50.0f);
        const float vel_uu_len_sq = LengthSq(vel_uu);
        if (vel_uu_len_sq > car_max_speed * car_max_speed) {
            const Vec3 vel_uu_normalized = Scale(vel_uu, RsqrtScalar(vel_uu_len_sq));
            vel_uu = Scale(vel_uu_normalized, car_max_speed);
            // SetCarLinearVelocityBt: overwritten below by the quantized velocity
        }
        const Vec3 ang_vel = ClampLength(h.ang_vel, car_max_ang_speed);

        // QuantizeCarState(v3_exact): SetCarOriginBt then pos = qp; SetCarLinearVelocityBt then vel = q
        const Vec3 p = Scale(h.pos_bt, 50.0f);
        const Vec3 qp = V3(QuantRoundPosV3(p.x, 100.0f), QuantRoundPosV3(p.y, 100.0f), QuantRoundPosV3(p.z, 100.0f));
        h.pos_bt = Scale(qp, kUuToBt);
        cars[ci].pos = qp;
        const Vec3 q = V3(QuantTruncOffsetV3(vel_uu.x, 100.0f), QuantTruncOffsetV3(vel_uu.y, 100.0f),
                          QuantTruncOffsetV3(vel_uu.z, 100.0f));
        h.vel_bt = Scale(q, kUuToBt);
        h.vel = q;
        h.ang_vel = V3(QuantTruncOffsetV3(ang_vel.x, 10000.0f), QuantTruncOffsetV3(ang_vel.y, 10000.0f),
                              QuantTruncOffsetV3(ang_vel.z, 10000.0f));
        Vec3 fwd, right, up;
        if (CAR_GETB(cars, ci, basis_authoritative)) {
            fwd = cars[ci].basis_forward;
            right = cars[ci].basis_right;
            up = cars[ci].basis_up;
        } else {
            GetBasisVectors(cars[ci].rot, fwd, right, up);
        }
        cars[ci].tick_start_basis_forward = fwd;
        cars[ci].tick_start_basis_right = right;
        cars[ci].tick_start_basis_up = up;
        h.rot = GameQuantizeBasisColumns(fwd, right, up);
        h.basis_forward = fwd;
        h.basis_right = right;
        h.basis_up = up;
        CAR_SETB(cars, ci, basis_authoritative, 1);
    }
}

static inline float Abs(float v) { return v >= 0.0f ? v : -v; }

// raycast_wheels_and_contact_prep.hpp CalFlipRestartTimeOnGround
static inline float CalFlipRestartTimeOnGround(uniform CarState cars[], int ci, const RgFlagWords& fl, const RgCarHot& h, uniform float flip_torque_time,
                                               uniform float dt) {
    if (CAR_GETB(cars, ci, is_flipping)) {
        return 0.0f;
    }
    const float ift = h.internal_flip_time;
    if (!CAR_GETB(cars, ci, has_flipped)) {
        return ift;
    }
    if (ift < flip_torque_time) {
        return 0.0f;
    }
    int k = (int)round(ift / dt) - (int)round(flip_torque_time / dt);
    if (k < 1) {
        k = 1;
    }
    float t = 0.0f;
    if (k > 64) {
        t = (float)k * dt;
    } else {
        for (int i = 0; i < k; ++i) {
            t += dt;
        }
    }
    return t;
}


// StepEnvPreTickCarFinish (env_step.hpp) after the cal_start_vel store, without the predict-velocity stores, on the
// rg_pretick_cars path: PreTickCarPrePhysics makes no frame, starts no jump or flip and applies no impulse.
static inline void PreTickFinish(uniform CarState cars[], uniform WheelRuntimeState wrs[], const int ci, RgFlagWords& fl, RgCarHot& h,
                                 const float frame_frac, uniform const RgPreTickParams& p) {
    const uniform float dt = p.dt;
    {
        // TickStartCarRbSnapshot
        const Quat snap_rot = h.rot;
        const Vec3 snap_vel = h.vel;
        // CalFlipZDampPrepass
        if (!CAR_GETB(cars, ci, is_on_ground) && CAR_GETB(cars, ci, is_flipping)) {
            const float flip_time = h.internal_flip_time + dt;
            if (flip_time >= 0.15f && flip_time <= p.flip_torque_time) {
                if (h.vel_bt.z < 0.0f || flip_time < 0.21f) {
                    const float dv_uu = (-h.vel.z) * 0.35f;
                    if (Abs(dv_uu) >= 1.0e-4f) {
                        const uniform float inv_mass = 1.0f / p.car_mass;
                        const float impulse = (dv_uu * 0.02f) * (1.0f / inv_mass);
                        const Vec3 v = h.vel_bt;
                        const Vec3 nv = V3(v.x, v.y, v.z + impulse * inv_mass);
                        h.vel_bt = nv;
                        h.vel = Scale(nv, 50.0f);
                    }
                }
            }
        }
        cars[ci].cal_tick_start_ang_vel = h.ang_vel;

        // PreTickCarPrePhysics
        CAR_SETB(cars, ci, cal_pre_flags, 0);
        const uint8 start_is_flipping = CAR_GETB(cars, ci, is_flipping);
        const uint8 start_has_flipped = CAR_GETB(cars, ci, has_flipped);
        const uint8 start_is_jumping = CAR_GETB(cars, ci, is_jumping);
        const float start_internal_flip_time = h.internal_flip_time;
        const int start_wheel_contacts = (CAR_GETB(cars, ci, wheels_with_contact[0]) != 0 ? 1 : 0) +
            (CAR_GETB(cars, ci, wheels_with_contact[1]) != 0 ? 1 : 0) + (CAR_GETB(cars, ci, wheels_with_contact[2]) != 0 ? 1 : 0) +
            (CAR_GETB(cars, ci, wheels_with_contact[3]) != 0 ? 1 : 0);
        CAR_SETB(cars, ci, demo_no_contact_response, 0);
        if (CAR_GETB(cars, ci, supersonic_set_v3)) {
            CAR_SETB(cars, ci, supersonic_set_v3, 0);
        } else {
            // UpdateSupersonicCalDemo
            const float speed_sq = LengthSq(h.vel);
            if (speed_sq >= (p.supersonic_start_speed * p.supersonic_start_speed)) {
                CAR_SETB(cars, ci, is_supersonic, 1);
                cars[ci].supersonic_time = 0.0f;
            } else if (CAR_GETB(cars, ci, is_supersonic)) {
                const float st = cars[ci].supersonic_time + dt;
                cars[ci].supersonic_time = st;
                if (st >= p.supersonic_maintain_max_time ||
                    speed_sq < (p.supersonic_maintain_min_speed * p.supersonic_maintain_min_speed)) {
                    CAR_SETB(cars, ci, is_supersonic, 0);
                    cars[ci].supersonic_time = 0.0f;
                }
            }
        }
        CAR_SETB(cars, ci, vel_touched_solve_stage, 0);
        const float cal_pitchlock_restart_time =
            (!start_has_flipped && !start_is_flipping && (start_internal_flip_time > 0.0f || CAR_GETB(cars, ci, flip_timer_live)))
            ? start_internal_flip_time : -1.0f;
        const bool cal_jump_g = start_wheel_contacts >= 3;
        WRS_SETB(wrs, ci, cal_restored, 0);
        if (!WRS_GETB(wrs, ci, cal_frame_open)) {
            if (cal_jump_g && !WRS_GETB(wrs, ci, cal_cached_ground)) {
                WRS_SETB(wrs, ci, cal_game_jumped, 0);
                WRS_SETB(wrs, ci, cal_game_double_jumped, 0);
            }
            WRS_SETB(wrs, ci, cal_cached_ground, cal_jump_g ? 1 : 0);
        }
        if (WRS_GETB(wrs, ci, cal_game_jumped) && cal_jump_g && !start_is_jumping && wrs[ci].cal_jump_inactive_time >= 0.125f) {
            WRS_SETB(wrs, ci, cal_game_jumped, 0);
            WRS_SETB(wrs, ci, cal_game_double_jumped, 0);
            WRS_SETB(wrs, ci, cal_cached_ground, 1);
        }
        WRS_SETB(wrs, ci, cal_frame_open, frame_frac < 0.0f ? 1 : 0);
        // SnapshotForwardSpeed
        const float w2 = snap_rot.w + snap_rot.w;
        const float s = snap_rot.w * snap_rot.w - ((snap_rot.y * snap_rot.y + snap_rot.x * snap_rot.x) + snap_rot.z * snap_rot.z);
        float d = (1.0f * snap_rot.x + 0.0f * snap_rot.y) + 0.0f * snap_rot.z;
        d = d + d;
        const Vec3 fwd = V3(((0.0f * snap_rot.y - 0.0f * snap_rot.z) * w2 + s * 1.0f) + d * snap_rot.x,
                            ((1.0f * snap_rot.z - 0.0f * snap_rot.x) * w2 + s * 0.0f) + d * snap_rot.y,
                            ((0.0f * snap_rot.x - 1.0f * snap_rot.y) * w2 + s * 0.0f) + d * snap_rot.z);
        const float forward_speed_uu = (fwd.x * snap_vel.x + fwd.y * snap_vel.y) + fwd.z * snap_vel.z;
        const bool air_jump_used_before = CAR_GETB(cars, ci, has_double_jumped) || CAR_GETB(cars, ci, has_flipped);
        // UpdateDoubleJumpOrFlipState without a jump press
        if (CAR_GETB(cars, ci, is_on_ground)) {
            CAR_SETB(cars, ci, has_double_jumped, 0);
            h.internal_flip_time = CalFlipRestartTimeOnGround(cars, ci, fl, h, p.flip_torque_time, dt);
            CAR_SETB(cars, ci, has_flipped, 0);
            cars[ci].air_time = 0.0f;
            cars[ci].air_time_since_jump = 0.0f;
        } else {
            cars[ci].air_time = cars[ci].air_time + dt;
            if (CAR_GETB(cars, ci, has_jumped) && !CAR_GETB(cars, ci, is_jumping)) {
                cars[ci].air_time_since_jump = cars[ci].air_time_since_jump + dt;
            } else {
                cars[ci].air_time_since_jump = 0.0f;
            }
        }
        if (CAR_GETB(cars, ci, is_flipping)) {
            h.internal_flip_time = h.internal_flip_time + dt;
        } else if (CAR_GETB(cars, ci, has_flipped) || CAR_GETB(cars, ci, flip_timer_live) || (h.internal_flip_time != 0.0f)) {
            h.internal_flip_time = h.internal_flip_time + dt;
        }
        CAR_SETB(cars, ci, cal_pre_flags, (uint8)((air_jump_used_before ? 8 : 0) | (start_is_flipping ? 32 : 0) |
                                         (start_is_jumping ? 64 : 0)));
        cars[ci].cal_pre_forward_speed_uu = forward_speed_uu;
        cars[ci].cal_pre_pitchlock_restart_time = cal_pitchlock_restart_time;

        // BodyDampingClampCar(car, mutator, true, true): velocity unchanged by the pre-physics on this path, so a
        // linear clamp only rewrites vel_bt
        const Vec3 v = h.vel_bt;
        const uniform float max_lin = p.car_max_speed * 0.02f;
        const float vsq = (v.x * v.x + v.y * v.y) + v.z * v.z;
        if (0.0f < max_lin && max_lin * max_lin < vsq) {
            const float sc = 1.0f / sqrt(vsq);
            h.vel_bt = V3((v.x * sc) * max_lin, (v.y * sc) * max_lin, (v.z * sc) * max_lin);
        }
        const Vec3 w = h.ang_vel;
        const uniform float max_ang = p.car_max_ang_speed;
        const float lsq = (w.x * w.x + w.y * w.y) + w.z * w.z;
        if (0.0f < max_ang && max_ang * max_ang < lsq) {
            const float sc = 1.0f / sqrt(lsq);
            h.ang_vel = V3((w.x * sc) * max_ang, (w.y * sc) * max_ang, (w.z * sc) * max_ang);
        }
    }
}

// StepEnvPreTickCarStart (env_step.hpp) for a car that is not demoed, with tick_count_since_update != 0.
static inline void PreTickStart(uniform CarState cars[], const int ci, RgFlagWords& fl, RgCarHot& h) {
    CAR_SETB(cars, ci, ball_contact_this_tick, 0);
    // CarState::RestoreWorkingFields
    const uint8 reported = CAR_GETB(cars, ci, reported_flags);
    if (reported & 128) {
        const Vec3 v = h.vel, rv = cars[ci].reported_vel;
        if (v.x == rv.x && v.y == rv.y && v.z == rv.z) {
            h.vel = cars[ci].internal_vel;
        }
        const Vec3 w = h.ang_vel, rw = cars[ci].reported_ang_vel;
        if (w.x == rw.x && w.y == rw.y && w.z == rw.z) {
            h.ang_vel = cars[ci].internal_ang_vel;
        }
        if ((CAR_GETB(cars, ci, has_jumped) != 0) == ((reported & 4) != 0)) {
            CAR_SETB(cars, ci, has_jumped, CAR_GETB(cars, ci, internal_has_jumped));
        }
        if ((CAR_GETB(cars, ci, has_double_jumped) != 0) == ((reported & 8) != 0)) {
            CAR_SETB(cars, ci, has_double_jumped, CAR_GETB(cars, ci, internal_has_double_jumped));
        }
        if ((CAR_GETB(cars, ci, has_flipped) != 0) == ((reported & 16) != 0)) {
            CAR_SETB(cars, ci, has_flipped, CAR_GETB(cars, ci, internal_has_flipped));
        }
        if ((CAR_GETB(cars, ci, is_supersonic) != 0) == ((reported & 32) != 0)) {
            CAR_SETB(cars, ci, is_supersonic, CAR_GETB(cars, ci, internal_is_supersonic));
        }
    }
    CAR_SETB(cars, ci, just_respawned, 0);
    cars[ci].tick_start_pos_bt = h.pos_bt;
}

// StepEnvPreTickCar (env_step.hpp) without the solver-scratch stores, for cars that are not demoed, have a
// WheelRuntimeState without cal_restored and tick_count_since_update != 0, press no jump
// (controls.jump && !driving_disabled && !last_controls.jump), are not auto-flipping and not three-wheeled, with no
// native SSE rsqrt table. start_vel[j] gets the cal_start_vel value.
export void rg_pretick_cars(uniform CarState cars[], uniform WheelRuntimeState wrs[], uniform const int32 idx[],
                            uniform const float frame_frac[], uniform int n, uniform const RgPreTickParams& p,
                            uniform Vec3 start_vel[], uniform Quat snap_rot[], uniform Quat snap_raw[]) {
    foreach (j = 0 ... n) {
        const int ci = idx[j];
        RgFlagWords fl;
        LoadFlagWords(cars, wrs, ci, fl);
        RgCarHot h;
        h.pos_bt = cars[ci].pos_bt;
        h.vel = cars[ci].vel;
        h.vel_bt = cars[ci].vel_bt;
        h.ang_vel = cars[ci].ang_vel;
        h.internal_flip_time = cars[ci].internal_flip_time;
        PreTickStart(cars, ci, fl, h);
        TickStartLimitAndQuantize(cars, ci, fl, h, p.car_max_speed, p.car_max_ang_speed);
        start_vel[j] = h.vel;
        PreTickFinish(cars, wrs, ci, fl, h, frame_frac[j], p);
        StoreFlagWords(cars, wrs, ci, fl);
        cars[ci].pos_bt = h.pos_bt;
        cars[ci].vel = h.vel;
        cars[ci].vel_bt = h.vel_bt;
        cars[ci].ang_vel = h.ang_vel;
        cars[ci].internal_flip_time = h.internal_flip_time;
        cars[ci].rot = h.rot;
        cars[ci].basis_forward = h.basis_forward;
        cars[ci].basis_right = h.basis_right;
        cars[ci].basis_up = h.basis_up;
        // FreshCarRbSnapshot's rotation for the basis the car leaves the kernel with, and its unquantized source.
        const Quat raw = GameGetRotationScalar(h.basis_forward, h.basis_right, h.basis_up);
        snap_raw[j] = raw;
        snap_rot[j] = GameRbStateQuantizeQuat(raw);
    }
}
