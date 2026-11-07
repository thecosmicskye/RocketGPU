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

#define RG_FIELD_OFF(T, f) ((uniform int32)((uniform int8 * uniform)(&((uniform T * uniform)0)->f) - (uniform int8 * uniform)0))
#define RG_CAR_WORDS 11
#define RG_WRS_WORDS 2
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
#define RG_IN_STRIDE 64
#define RG_IN_VEL 0
#define RG_IN_ANG_VEL 4
#define RG_IN_AIR_TIME 8
#define RG_IN_AIR_TIME_SINCE_JUMP 9
#define RG_IN_SUPERSONIC_TIME 12
#define RG_IN_POS_BT 16
#define RG_IN_VEL_BT 20
#define RG_IN_FWD 25
#define RG_IN_RIGHT 28
#define RG_IN_UP 32
#define RG_IN_INTERNAL_FLIP_TIME 36
#define RG_IN_CAL_JUMP_INACTIVE 58
static const uniform int32 kCarWordRow[RG_CAR_WORDS] = {40, 41, 42, 43, 44, 24, 37, 48, 49, 50, 52};
static const uniform int32 kWrsWordRow[RG_WRS_WORDS] = {57, 59};
static inline float In1(uniform const float rin[], const uniform int r, const int j) { return rin[r * RG_IN_STRIDE + j]; }
static inline Vec3 InV3(uniform const float rin[], const uniform int r, const int j) {
    return V3(In1(rin, r, j), In1(rin, r + 1, j), In1(rin, r + 2, j));
}
static inline void LoadFlagWords(uniform const float rin[], const int j, RgFlagWords& fl) {
#pragma unroll
    for (uniform int k = 0; k < RG_CAR_WORDS; ++k) {
        fl.c[k] = (uint32)intbits(In1(rin, kCarWordRow[k], j));
    }
#pragma unroll
    for (uniform int k = 0; k < RG_WRS_WORDS; ++k) {
        fl.w[k] = (uint32)intbits(In1(rin, kWrsWordRow[k], j));
    }
}
#define RG_CAR_OUT_STRIDE 64
static const uniform int32 kCarStoreSlots[8] = {0, 1, 2, 3, 4, 5, 9, 10};
static inline void StoreFlagWords(uniform uint32 wout[], const int j, const RgFlagWords& fl) {
#pragma unroll
    for (uniform int k = 0; k < 8; ++k) {
        wout[k * RG_CAR_OUT_STRIDE + j] = fl.c[kCarStoreSlots[k]];
    }
#pragma unroll
    for (uniform int k = 0; k < RG_WRS_WORDS; ++k) {
        wout[(8 + k) * RG_CAR_OUT_STRIDE + j] = fl.w[k];
    }
}

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
    Vec3 tick_start_pos_bt;
    Vec3 pos;
    Vec3 tick_start_basis_forward;
    Vec3 tick_start_basis_right;
    Vec3 tick_start_basis_up;
    Vec3 cal_tick_start_ang_vel;
    float supersonic_time;
    float air_time;
    float air_time_since_jump;
    float cal_pre_forward_speed_uu;
    float cal_pre_pitchlock_restart_time;
};

static inline float QuantRoundPosV3(float x, uniform float scale) {
    const float scaled = x * scale;
    return floor(scaled + 0.5f) * (1.0f / scale);
}

static inline float QuantTruncOffsetV3(float x, uniform float scale) {
    const uniform float inv_scale = 1.0f / scale;
    const float k = (float)((int32)(x * scale));
    const float rounded = k * inv_scale;
    const uniform float offset_mag = 0.1f * inv_scale;
    const float sgn = rounded == 0.0f ? 0.0f : (rounded > 0.0f ? 1.0f : -1.0f);
    const float offset = sgn * offset_mag;
    return rounded + offset;
}

static inline void TickStartLimitAndQuantize(uniform CarState cars[], uniform const float rin[], const int ci, const int j, RgFlagWords& fl,
                                             RgCarHot& h, uniform float car_max_speed, uniform float car_max_ang_speed) {
    {
        Vec3 vel_bt = h.vel_bt;
        Vec3 vel_uu = Scale(vel_bt, 50.0f);
        const float vel_uu_len_sq = LengthSq(vel_uu);
        if (vel_uu_len_sq > car_max_speed * car_max_speed) {
            const Vec3 vel_uu_normalized = Scale(vel_uu, RsqrtScalar(vel_uu_len_sq));
            vel_uu = Scale(vel_uu_normalized, car_max_speed);
        }
        const Vec3 ang_vel = ClampLength(h.ang_vel, car_max_ang_speed);

        const Vec3 p = Scale(h.pos_bt, 50.0f);
        const Vec3 qp = V3(QuantRoundPosV3(p.x, 100.0f), QuantRoundPosV3(p.y, 100.0f), QuantRoundPosV3(p.z, 100.0f));
        h.pos_bt = Scale(qp, kUuToBt);
        h.pos = qp;
        const Vec3 q = V3(QuantTruncOffsetV3(vel_uu.x, 100.0f), QuantTruncOffsetV3(vel_uu.y, 100.0f),
                          QuantTruncOffsetV3(vel_uu.z, 100.0f));
        h.vel_bt = Scale(q, kUuToBt);
        h.vel = q;
        h.ang_vel = V3(QuantTruncOffsetV3(ang_vel.x, 10000.0f), QuantTruncOffsetV3(ang_vel.y, 10000.0f),
                              QuantTruncOffsetV3(ang_vel.z, 10000.0f));
        Vec3 fwd, right, up;
        if (CAR_GETB(cars, ci, basis_authoritative)) {
            fwd = InV3(rin, RG_IN_FWD, j);
            right = InV3(rin, RG_IN_RIGHT, j);
            up = InV3(rin, RG_IN_UP, j);
        } else {
            GetBasisVectors(cars[ci].rot, fwd, right, up);
        }
        h.tick_start_basis_forward = fwd;
        h.tick_start_basis_right = right;
        h.tick_start_basis_up = up;
        h.rot = GameQuantizeBasisColumns(fwd, right, up);
        h.basis_forward = fwd;
        h.basis_right = right;
        h.basis_up = up;
        CAR_SETB(cars, ci, basis_authoritative, 1);
    }
}

static inline float Abs(float v) { return v >= 0.0f ? v : -v; }

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


static inline void PreTickFinish(uniform CarState cars[], uniform const float rin[], const int ci, const int j, RgFlagWords& fl, RgCarHot& h,
                                 const float frame_frac, uniform const RgPreTickParams& p) {
    const uniform float dt = p.dt;
    {
        const Quat snap_rot = h.rot;
        const Vec3 snap_vel = h.vel;
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
        h.cal_tick_start_ang_vel = h.ang_vel;

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
            const float speed_sq = LengthSq(h.vel);
            if (speed_sq >= (p.supersonic_start_speed * p.supersonic_start_speed)) {
                CAR_SETB(cars, ci, is_supersonic, 1);
                h.supersonic_time = 0.0f;
            } else if (CAR_GETB(cars, ci, is_supersonic)) {
                const float st = h.supersonic_time + dt;
                h.supersonic_time = st;
                if (st >= p.supersonic_maintain_max_time ||
                    speed_sq < (p.supersonic_maintain_min_speed * p.supersonic_maintain_min_speed)) {
                    CAR_SETB(cars, ci, is_supersonic, 0);
                    h.supersonic_time = 0.0f;
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
        if (WRS_GETB(wrs, ci, cal_game_jumped) && cal_jump_g && !start_is_jumping && In1(rin, RG_IN_CAL_JUMP_INACTIVE, j) >= 0.125f) {
            WRS_SETB(wrs, ci, cal_game_jumped, 0);
            WRS_SETB(wrs, ci, cal_game_double_jumped, 0);
            WRS_SETB(wrs, ci, cal_cached_ground, 1);
        }
        WRS_SETB(wrs, ci, cal_frame_open, frame_frac < 0.0f ? 1 : 0);
        const float w2 = snap_rot.w + snap_rot.w;
        const float s = snap_rot.w * snap_rot.w - ((snap_rot.y * snap_rot.y + snap_rot.x * snap_rot.x) + snap_rot.z * snap_rot.z);
        float d = (1.0f * snap_rot.x + 0.0f * snap_rot.y) + 0.0f * snap_rot.z;
        d = d + d;
        const Vec3 fwd = V3(((0.0f * snap_rot.y - 0.0f * snap_rot.z) * w2 + s * 1.0f) + d * snap_rot.x,
                            ((1.0f * snap_rot.z - 0.0f * snap_rot.x) * w2 + s * 0.0f) + d * snap_rot.y,
                            ((0.0f * snap_rot.x - 1.0f * snap_rot.y) * w2 + s * 0.0f) + d * snap_rot.z);
        const float forward_speed_uu = (fwd.x * snap_vel.x + fwd.y * snap_vel.y) + fwd.z * snap_vel.z;
        const bool air_jump_used_before = CAR_GETB(cars, ci, has_double_jumped) || CAR_GETB(cars, ci, has_flipped);
        if (CAR_GETB(cars, ci, is_on_ground)) {
            CAR_SETB(cars, ci, has_double_jumped, 0);
            h.internal_flip_time = CalFlipRestartTimeOnGround(cars, ci, fl, h, p.flip_torque_time, dt);
            CAR_SETB(cars, ci, has_flipped, 0);
            h.air_time = 0.0f;
            h.air_time_since_jump = 0.0f;
        } else {
            h.air_time = h.air_time + dt;
            if (CAR_GETB(cars, ci, has_jumped) && !CAR_GETB(cars, ci, is_jumping)) {
                h.air_time_since_jump = h.air_time_since_jump + dt;
            } else {
                h.air_time_since_jump = 0.0f;
            }
        }
        if (CAR_GETB(cars, ci, is_flipping)) {
            h.internal_flip_time = h.internal_flip_time + dt;
        } else if (CAR_GETB(cars, ci, has_flipped) || CAR_GETB(cars, ci, flip_timer_live) || (h.internal_flip_time != 0.0f)) {
            h.internal_flip_time = h.internal_flip_time + dt;
        }
        CAR_SETB(cars, ci, cal_pre_flags, (uint8)((air_jump_used_before ? 8 : 0) | (start_is_flipping ? 32 : 0) |
                                         (start_is_jumping ? 64 : 0)));
        h.cal_pre_forward_speed_uu = forward_speed_uu;
        h.cal_pre_pitchlock_restart_time = cal_pitchlock_restart_time;

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

static inline void PreTickStart(uniform CarState cars[], const int ci, RgFlagWords& fl, RgCarHot& h) {
    CAR_SETB(cars, ci, ball_contact_this_tick, 0);
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
    h.tick_start_pos_bt = h.pos_bt;
}

#define RG_CAR_OUT_TICK_START_POS_BT 0
#define RG_CAR_OUT_POS 3
#define RG_CAR_OUT_TICK_START_FWD 6
#define RG_CAR_OUT_TICK_START_RIGHT 9
#define RG_CAR_OUT_TICK_START_UP 12
#define RG_CAR_OUT_CAL_TICK_START_ANG_VEL 15
#define RG_CAR_OUT_SUPERSONIC_TIME 18
#define RG_CAR_OUT_AIR_TIME 19
#define RG_CAR_OUT_AIR_TIME_SINCE_JUMP 20
#define RG_CAR_OUT_CAL_PRE_FORWARD_SPEED 21
#define RG_CAR_OUT_CAL_PRE_PITCHLOCK 22
#define RG_CAR_OUT_POS_BT 23
#define RG_CAR_OUT_VEL 26
#define RG_CAR_OUT_VEL_BT 29
#define RG_CAR_OUT_ANG_VEL 32
#define RG_CAR_OUT_INTERNAL_FLIP_TIME 35
#define RG_CAR_OUT_ROT 36
#define RG_CAR_OUT_FWD 40
#define RG_CAR_OUT_RIGHT 43
#define RG_CAR_OUT_UP 46
#define RG_CAR_OUT_START_VEL 49
#define RG_CAR_OUT_SNAP_ROT 52
#define RG_CAR_OUT_SNAP_RAW 56
#define RG_CAR_OUT_FIELDS 60
static inline void CarOut1(uniform float out[], const uniform int f, const int j, const float v) {
    out[f * RG_CAR_OUT_STRIDE + j] = v;
}
static inline void CarOutV3(uniform float out[], const uniform int f, const int j, const Vec3 v) {
    CarOut1(out, f, j, v.x);
    CarOut1(out, f + 1, j, v.y);
    CarOut1(out, f + 2, j, v.z);
}
static inline void CarOutQ(uniform float out[], const uniform int f, const int j, const Quat q) {
    CarOut1(out, f, j, q.w);
    CarOut1(out, f + 1, j, q.x);
    CarOut1(out, f + 2, j, q.y);
    CarOut1(out, f + 3, j, q.z);
}
export void rg_pretick_cars(uniform CarState cars[], uniform const float rin[], uniform const int32 idx[],
                            uniform const float frame_frac[], uniform int n, uniform const RgPreTickParams& p,
                            uniform float out[], uniform uint32 wout[]) {
    foreach (j = 0 ... n) {
        const int ci = idx[j];
        RgFlagWords fl;
        LoadFlagWords(rin, j, fl);
        RgCarHot h;
        h.pos_bt = InV3(rin, RG_IN_POS_BT, j);
        h.vel = InV3(rin, RG_IN_VEL, j);
        h.vel_bt = InV3(rin, RG_IN_VEL_BT, j);
        h.ang_vel = InV3(rin, RG_IN_ANG_VEL, j);
        h.internal_flip_time = In1(rin, RG_IN_INTERNAL_FLIP_TIME, j);
        h.supersonic_time = In1(rin, RG_IN_SUPERSONIC_TIME, j);
        h.air_time = In1(rin, RG_IN_AIR_TIME, j);
        h.air_time_since_jump = In1(rin, RG_IN_AIR_TIME_SINCE_JUMP, j);
        PreTickStart(cars, ci, fl, h);
        TickStartLimitAndQuantize(cars, rin, ci, j, fl, h, p.car_max_speed, p.car_max_ang_speed);
        CarOutV3(out, RG_CAR_OUT_START_VEL, j, h.vel);
        PreTickFinish(cars, rin, ci, j, fl, h, frame_frac[j], p);
        StoreFlagWords(wout, j, fl);
        CarOutV3(out, RG_CAR_OUT_TICK_START_POS_BT, j, h.tick_start_pos_bt);
        CarOutV3(out, RG_CAR_OUT_POS, j, h.pos);
        CarOutV3(out, RG_CAR_OUT_TICK_START_FWD, j, h.tick_start_basis_forward);
        CarOutV3(out, RG_CAR_OUT_TICK_START_RIGHT, j, h.tick_start_basis_right);
        CarOutV3(out, RG_CAR_OUT_TICK_START_UP, j, h.tick_start_basis_up);
        CarOutV3(out, RG_CAR_OUT_CAL_TICK_START_ANG_VEL, j, h.cal_tick_start_ang_vel);
        CarOut1(out, RG_CAR_OUT_SUPERSONIC_TIME, j, h.supersonic_time);
        CarOut1(out, RG_CAR_OUT_AIR_TIME, j, h.air_time);
        CarOut1(out, RG_CAR_OUT_AIR_TIME_SINCE_JUMP, j, h.air_time_since_jump);
        CarOut1(out, RG_CAR_OUT_CAL_PRE_FORWARD_SPEED, j, h.cal_pre_forward_speed_uu);
        CarOut1(out, RG_CAR_OUT_CAL_PRE_PITCHLOCK, j, h.cal_pre_pitchlock_restart_time);
        CarOutV3(out, RG_CAR_OUT_POS_BT, j, h.pos_bt);
        CarOutV3(out, RG_CAR_OUT_VEL, j, h.vel);
        CarOutV3(out, RG_CAR_OUT_VEL_BT, j, h.vel_bt);
        CarOutV3(out, RG_CAR_OUT_ANG_VEL, j, h.ang_vel);
        CarOut1(out, RG_CAR_OUT_INTERNAL_FLIP_TIME, j, h.internal_flip_time);
        CarOutQ(out, RG_CAR_OUT_ROT, j, h.rot);
        CarOutV3(out, RG_CAR_OUT_FWD, j, h.basis_forward);
        CarOutV3(out, RG_CAR_OUT_RIGHT, j, h.basis_right);
        CarOutV3(out, RG_CAR_OUT_UP, j, h.basis_up);
        const Quat raw = GameGetRotationScalar(h.basis_forward, h.basis_right, h.basis_up);
        CarOutQ(out, RG_CAR_OUT_SNAP_RAW, j, raw);
        CarOutQ(out, RG_CAR_OUT_SNAP_ROT, j, GameRbStateQuantizeQuat(raw));
    }
}

#define RG_BALL_OUT_STRIDE 64
#define RG_BALL_OUT_POS_BT 0
#define RG_BALL_OUT_POS 3
#define RG_BALL_OUT_VEL_BT 6
#define RG_BALL_OUT_VEL 9
#define RG_BALL_OUT_ANG_VEL 12
#define RG_BALL_OUT_ROT 15
#define RG_BALL_OUT_FWD 19
#define RG_BALL_OUT_RIGHT 22
#define RG_BALL_OUT_UP 25
#define RG_BALL_OUT_INERTIA_FWD 28
#define RG_BALL_OUT_INERTIA_RIGHT 31
#define RG_BALL_OUT_INERTIA_UP 34
#define RG_BALL_OUT_START_VEL 37
#define RG_BALL_OUT_PREDICT_VEL 40
#define RG_BALL_OUT_FIELDS 43
static inline void StoreV3(uniform float out[], const uniform int f, const int j, const Vec3 v) {
    out[f * RG_BALL_OUT_STRIDE + j] = v.x;
    out[(f + 1) * RG_BALL_OUT_STRIDE + j] = v.y;
    out[(f + 2) * RG_BALL_OUT_STRIDE + j] = v.z;
}

#define RG_BALL_IN_VEL_BT 0
#define RG_BALL_IN_ANG_VEL 3
#define RG_BALL_IN_POS_BT 6
#define RG_BALL_IN_FWD 9
#define RG_BALL_IN_RIGHT 12
#define RG_BALL_IN_UP 15
#define RG_BALL_IN_ROT 18
#define RG_BALL_IN_AUTHORITATIVE 22
static inline Vec3 LoadV3(uniform const float bin[], const uniform int f, const int j) {
    return V3(bin[f * RG_BALL_OUT_STRIDE + j], bin[(f + 1) * RG_BALL_OUT_STRIDE + j], bin[(f + 2) * RG_BALL_OUT_STRIDE + j]);
}
export void rg_pretick_balls(uniform const float bin[], uniform int n, uniform float max_speed,
                             uniform float max_ang_speed, uniform bool heatseeker, uniform bool damp, uniform float damp_scale,
                             uniform float out[]) {
    foreach (j = 0 ... n) {
        Vec3 vel_uu = Scale(LoadV3(bin, RG_BALL_IN_VEL_BT, j), 50.0f);
        const float vel_uu_len_sq = LengthSq(vel_uu);
        if (vel_uu_len_sq > max_speed * max_speed) {
            vel_uu = Scale(Scale(vel_uu, RsqrtScalar(vel_uu_len_sq)), max_speed);
        }
        const Vec3 ang = ClampLength(LoadV3(bin, RG_BALL_IN_ANG_VEL, j), max_ang_speed);

        const Vec3 p = Scale(LoadV3(bin, RG_BALL_IN_POS_BT, j), 50.0f);
        const Vec3 qp = V3(QuantRoundPosV3(p.x, 100.0f), QuantRoundPosV3(p.y, 100.0f), QuantRoundPosV3(p.z, 100.0f));
        StoreV3(out, RG_BALL_OUT_POS_BT, j, Scale(qp, kUuToBt));
        StoreV3(out, RG_BALL_OUT_POS, j, qp);
        const Vec3 qv = V3(QuantTruncOffsetV3(vel_uu.x, 100.0f), QuantTruncOffsetV3(vel_uu.y, 100.0f),
                           QuantTruncOffsetV3(vel_uu.z, 100.0f));
        Vec3 vel_bt = Scale(qv, kUuToBt);
        Vec3 vel = qv;
        Vec3 w = V3(QuantTruncOffsetV3(ang.x, 10000.0f), QuantTruncOffsetV3(ang.y, 10000.0f),
                    QuantTruncOffsetV3(ang.z, 10000.0f));
        Vec3 fwd, right, up;
        if (bin[RG_BALL_IN_AUTHORITATIVE * RG_BALL_OUT_STRIDE + j] != 0.0f) {
            fwd = LoadV3(bin, RG_BALL_IN_FWD, j);
            right = LoadV3(bin, RG_BALL_IN_RIGHT, j);
            up = LoadV3(bin, RG_BALL_IN_UP, j);
        } else {
            const Quat q = Q4(bin[RG_BALL_IN_ROT * RG_BALL_OUT_STRIDE + j], bin[(RG_BALL_IN_ROT + 1) * RG_BALL_OUT_STRIDE + j],
                              bin[(RG_BALL_IN_ROT + 2) * RG_BALL_OUT_STRIDE + j], bin[(RG_BALL_IN_ROT + 3) * RG_BALL_OUT_STRIDE + j]);
            const float len_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
            if (len_sq <= 1e-12f) {
                fwd = V3(1.0f, 0.0f, 0.0f);
                right = V3(0.0f, 1.0f, 0.0f);
                up = V3(0.0f, 0.0f, 1.0f);
            } else {
                const float s = 2.0f / len_sq;
                const float xs = q.x * s;
                const float ys = q.y * s;
                const float zs = q.z * s;
                const float wx = q.w * xs;
                const float wy = q.w * ys;
                const float wz = q.w * zs;
                const float xx = q.x * xs;
                const float xy = q.x * ys;
                const float xz = q.x * zs;
                const float yy = q.y * ys;
                const float yz = q.y * zs;
                const float zz = q.z * zs;
                fwd = V3(1.0f - (yy + zz), xy + wz, xz - wy);
                right = V3(xy - wz, 1.0f - (xx + zz), yz + wx);
                up = V3(xz + wy, yz - wx, 1.0f - (xx + yy));
            }
        }
        StoreV3(out, RG_BALL_OUT_INERTIA_FWD, j, fwd);
        StoreV3(out, RG_BALL_OUT_INERTIA_RIGHT, j, right);
        StoreV3(out, RG_BALL_OUT_INERTIA_UP, j, up);
        const Quat rot = GameQuantizeBasisColumns(fwd, right, up);
        out[RG_BALL_OUT_ROT * RG_BALL_OUT_STRIDE + j] = rot.w;
        out[(RG_BALL_OUT_ROT + 1) * RG_BALL_OUT_STRIDE + j] = rot.x;
        out[(RG_BALL_OUT_ROT + 2) * RG_BALL_OUT_STRIDE + j] = rot.y;
        out[(RG_BALL_OUT_ROT + 3) * RG_BALL_OUT_STRIDE + j] = rot.z;
        StoreV3(out, RG_BALL_OUT_FWD, j, fwd);
        StoreV3(out, RG_BALL_OUT_RIGHT, j, right);
        StoreV3(out, RG_BALL_OUT_UP, j, up);
        StoreV3(out, RG_BALL_OUT_START_VEL, j, vel);

        if (!heatseeker) {
            const uniform float max_lin = max_speed * 0.02f;
            const float vsq = (vel_bt.x * vel_bt.x + vel_bt.y * vel_bt.y) + vel_bt.z * vel_bt.z;
            if (0.0f < max_lin && max_lin * max_lin < vsq) {
                const float s = 1.0f / sqrt(vsq);
                vel_bt = V3((vel_bt.x * s) * max_lin, (vel_bt.y * s) * max_lin, (vel_bt.z * s) * max_lin);
                vel = Scale(vel_bt, 50.0f);
            }
            const float lsq = (w.x * w.x + w.y * w.y) + w.z * w.z;
            if (0.0f < max_ang_speed && max_ang_speed * max_ang_speed < lsq) {
                const float s = 1.0f / sqrt(lsq);
                w = V3((w.x * s) * max_ang_speed, (w.y * s) * max_ang_speed, (w.z * s) * max_ang_speed);
            }
            if (damp) {
                vel_bt = Scale(vel_bt, damp_scale);
                vel = Scale(vel_bt, 50.0f);
            }
            StoreV3(out, RG_BALL_OUT_PREDICT_VEL, j, vel_bt);
        }
        StoreV3(out, RG_BALL_OUT_VEL_BT, j, vel_bt);
        StoreV3(out, RG_BALL_OUT_VEL, j, vel);
        StoreV3(out, RG_BALL_OUT_ANG_VEL, j, w);
    }
}
