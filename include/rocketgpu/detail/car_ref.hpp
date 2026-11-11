#pragma once

#include <cstddef>
#include <cstring>
#include <new>
#include <type_traits>

#include "rocketgpu/rocketgpu.h"

#if defined(ROCKETGPU_USE_CUDA) && !defined(ROCKETGPU_CAR_AOS)
#define ROCKETGPU_CAR_TILES 1
#endif

namespace rocketgpu {
namespace detail {

#if !defined(ROCKETGPU_CAR_TILES)

using CarRefT = CarState&;
using CarCRefT = const CarState&;
using CarSpanT = CarState*;
using CarCSpanT = const CarState*;
using CarPtrT = CarState*;
using CarCPtrT = const CarState*;
using CarArgT = CarState&;
using CarCArgT = const CarState&;

ROCKETGPU_HD constexpr size_t CarStoreBytes(size_t car_capacity) { return car_capacity * sizeof(CarState); }
ROCKETGPU_HD CarSpanT CarSpanFromBytes(void* p) { return static_cast<CarState*>(p); }
ROCKETGPU_HD void* CarSpanBytes(CarCSpanT s) { return const_cast<CarState*>(s); }
constexpr int kCarChunks8 = static_cast<int>(sizeof(CarState) / 8);
ROCKETGPU_HD unsigned long long* CarChunk8(CarSpanT s, long long idx, int k) {
    return reinterpret_cast<unsigned long long*>(s + idx) + k;
}
ROCKETGPU_HD const unsigned long long* CarChunk8(CarCSpanT s, long long idx, int k) {
    return reinterpret_cast<const unsigned long long*>(s + idx) + k;
}

#else

#include "rocketgpu/detail/car_tile_layout.inc"

constexpr int kCarTileLanes = 32;
constexpr int kCarTileWordBytes = 16;
constexpr size_t kCarTileWideStart = static_cast<size_t>(ROCKETGPU_CAR_TILE_WORDS) * kCarTileLanes * kCarTileWordBytes;
constexpr size_t kCarTileColdStart = kCarTileWideStart + static_cast<size_t>(kCarTileLanes) * ROCKETGPU_CAR_TILE_WIDE_BYTES;
constexpr size_t kCarTileBytes = kCarTileColdStart + static_cast<size_t>(kCarTileLanes) * ROCKETGPU_CAR_TILE_COLD_BYTES;
static_assert(sizeof(CarState) == ROCKETGPU_CAR_STATE_BYTES, "run tools/gen_car_tile_layout.py");
#define ROCKETGPU_CAR_TILE_CHECK(name, region, offset)                                                                  \
    static_assert((offset) % alignof(decltype(CarState::name)) == 0, "car tile field misaligned");                    \
    static_assert((region) != 0 || (offset) % kCarTileWordBytes + sizeof(decltype(CarState::name)) <= kCarTileWordBytes, \
                  "car tile hot field crosses a word");
ROCKETGPU_CAR_TILE_FIELDS(ROCKETGPU_CAR_TILE_CHECK)
#undef ROCKETGPU_CAR_TILE_CHECK

ROCKETGPU_HD constexpr size_t CarTileStoreBytes(size_t car_capacity) {
    return (car_capacity + kCarTileLanes - 1) / kCarTileLanes * kCarTileBytes;
}

template <class S>
struct BasicCarPtr;
template <class S>
struct BasicCarArg;

template <class S>
struct BasicCarRef {
    static constexpr bool kConst = std::is_const<S>::value;
    using Byte = typename std::conditional<kConst, const unsigned char, unsigned char>::type;
    template <class T>
    using Field = typename std::conditional<kConst, const T, T>::type;

    static constexpr int kMaxPendingBumps = CarState::kMaxPendingBumps;
    static constexpr uint8_t kReportedVel = CarState::kReportedVel, kReportedAngVel = CarState::kReportedAngVel,
                             kReportedHasJumped = CarState::kReportedHasJumped,
                             kReportedHasDoubleJumped = CarState::kReportedHasDoubleJumped,
                             kReportedHasFlipped = CarState::kReportedHasFlipped,
                             kReportedIsSupersonic = CarState::kReportedIsSupersonic,
                             kReportedValid = CarState::kReportedValid;

#define ROCKETGPU_CAR_REF_MEMBER(name, region, offset) Field<decltype(CarState::name)>& name;
    ROCKETGPU_CAR_TILE_FIELDS(ROCKETGPU_CAR_REF_MEMBER)
#undef ROCKETGPU_CAR_REF_MEMBER

    ROCKETGPU_HD BasicCarRef(Byte* tile, int lane)
        : BasicCarRef(tile + lane * kCarTileWordBytes, tile + kCarTileWideStart + lane * ROCKETGPU_CAR_TILE_WIDE_BYTES,
                      tile + kCarTileColdStart + lane * ROCKETGPU_CAR_TILE_COLD_BYTES, 0) {}
    ROCKETGPU_HD BasicCarRef(S& s)
        :
#define ROCKETGPU_CAR_REF_FROM_STATE(name, region, offset) name(s.name),
          ROCKETGPU_CAR_TILE_FIELDS(ROCKETGPU_CAR_REF_FROM_STATE)
#undef ROCKETGPU_CAR_REF_FROM_STATE
          unused_(0), org_state_(&s) {
    }
    ROCKETGPU_HD BasicCarRef(const BasicCarArg<S>& a);
    template <class S2, class = typename std::enable_if<kConst && !std::is_const<S2>::value>::type>
    ROCKETGPU_HD BasicCarRef(const BasicCarRef<S2>& o)
        :
#define ROCKETGPU_CAR_REF_FROM_REF(name, region, offset) name(o.name),
          ROCKETGPU_CAR_TILE_FIELDS(ROCKETGPU_CAR_REF_FROM_REF)
#undef ROCKETGPU_CAR_REF_FROM_REF
          unused_(0), org_w_(o.org_w_), org_wide_(o.org_wide_), org_cold_(o.org_cold_), org_state_(o.org_state_) {
    }
    BasicCarRef(const BasicCarRef&) = default;

    ROCKETGPU_HD operator CarState() const {
        CarState s;
        memset(static_cast<void*>(&s), 0, sizeof(s));
#define ROCKETGPU_CAR_REF_TO_STATE(name, region, offset) CopyField(s.name, name);
        ROCKETGPU_CAR_TILE_FIELDS(ROCKETGPU_CAR_REF_TO_STATE)
#undef ROCKETGPU_CAR_REF_TO_STATE
        return s;
    }
    template <class Src>
    ROCKETGPU_HD const BasicCarRef& Assign(const Src& o) const {
        static_assert(!kConst, "assignment through a const car view");
#define ROCKETGPU_CAR_REF_ASSIGN(name, region, offset) CopyField(name, o.name);
        ROCKETGPU_CAR_TILE_FIELDS(ROCKETGPU_CAR_REF_ASSIGN)
#undef ROCKETGPU_CAR_REF_ASSIGN
        return *this;
    }
    ROCKETGPU_HD const BasicCarRef& operator=(const CarState& o) const { return Assign(o); }
    ROCKETGPU_HD const BasicCarRef& operator=(const BasicCarRef& o) const { return Assign(o); }
    template <class S2>
    ROCKETGPU_HD const BasicCarRef& operator=(const BasicCarRef<S2>& o) const { return Assign(o); }

    ROCKETGPU_HD bool HasFlipOrJump() const {
        return is_on_ground || (!has_flipped && !has_double_jumped && air_time_since_jump < 1.25f);
    }
    ROCKETGPU_HD void ResetTrainingHistory() const {
        restored_inv_inertia_valid = 0;
        world_contact.wheel_ball_mask = 0;
        previous_ball_contact_start = last_ball_contact_start = last_world_surface_contact = ~0ULL;
        first_world_after_previous_ball = first_world_after_last_ball = ~0ULL;
        last_flip_or_jump_start = last_directional_flip_start = ~0ULL;
        ball_contact_last_tick = ball_contact_this_tick = last_ball_started_with_world = 0;
        flip_or_jump_last_tick = HasFlipOrJump();
    }
    ROCKETGPU_HD void RecordWorldSurface(bool body_contact) const {
        last_world_surface_contact = tick_count_since_update;
        if (body_contact && last_ball_contact_start == tick_count_since_update)
            last_ball_started_with_world = 1;
        if (last_ball_contact_start != ~0ULL && first_world_after_last_ball == ~0ULL
            && tick_count_since_update > last_ball_contact_start)
            first_world_after_last_ball = tick_count_since_update;
    }
    ROCKETGPU_HD void RecordBallContactStart() const {
        previous_ball_contact_start = last_ball_contact_start;
        first_world_after_previous_ball = first_world_after_last_ball;
        last_ball_contact_start = tick_count_since_update;
        first_world_after_last_ball = ~0ULL;
        last_ball_started_with_world = last_world_surface_contact == tick_count_since_update;
    }
    ROCKETGPU_HD uint8_t RestoreWorkingFields() const {
        if (!(reported_flags & kReportedValid)) {
            return 0;
        }
        uint8_t kept = kReportedValid;
        if (vel.x == reported_vel.x && vel.y == reported_vel.y && vel.z == reported_vel.z) {
            vel = internal_vel;
            kept |= kReportedVel;
        }
        if (ang_vel.x == reported_ang_vel.x && ang_vel.y == reported_ang_vel.y && ang_vel.z == reported_ang_vel.z) {
            ang_vel = internal_ang_vel;
            kept |= kReportedAngVel;
        }
        RestoreWorkingFlag(has_jumped, internal_has_jumped, kReportedHasJumped, kept);
        RestoreWorkingFlag(has_double_jumped, internal_has_double_jumped, kReportedHasDoubleJumped, kept);
        RestoreWorkingFlag(has_flipped, internal_has_flipped, kReportedHasFlipped, kept);
        RestoreWorkingFlag(is_supersonic, internal_is_supersonic, kReportedIsSupersonic, kept);
        return kept;
    }
    ROCKETGPU_HD void ReReportFields(uint8_t kept) const {
        if (!(kept & kReportedValid)) {
            return;
        }
        internal_vel = vel;
        internal_ang_vel = ang_vel;
        internal_has_jumped = has_jumped;
        internal_has_double_jumped = has_double_jumped;
        internal_has_flipped = has_flipped;
        internal_is_supersonic = is_supersonic;
        if (kept & kReportedVel) vel = reported_vel; else reported_vel = vel;
        if (kept & kReportedAngVel) ang_vel = reported_ang_vel; else reported_ang_vel = ang_vel;
        uint8_t flags = kReportedValid;
        ReReportFlag(has_jumped, kReportedHasJumped, kept, flags);
        ReReportFlag(has_double_jumped, kReportedHasDoubleJumped, kept, flags);
        ReReportFlag(has_flipped, kReportedHasFlipped, kept, flags);
        ReReportFlag(is_supersonic, kReportedIsSupersonic, kept, flags);
        reported_flags = flags;
    }
    ROCKETGPU_HD void RestoreWorkingFlag(Field<uint8_t>& field, uint8_t internal, uint8_t bit, uint8_t& kept) const {
        if ((field != 0) == ((reported_flags & bit) != 0)) {
            field = internal;
            kept |= bit;
        }
    }
    ROCKETGPU_HD void ReReportFlag(Field<uint8_t>& field, uint8_t bit, uint8_t kept, uint8_t& flags) const {
        if (kept & bit) {
            field = (reported_flags & bit) ? 1 : 0;
        }
        if (field) {
            flags |= bit;
        }
    }
    ROCKETGPU_HD BasicCarPtr<S> operator&() const;

private:
    template <class S2>
    friend struct BasicCarRef;
    ROCKETGPU_HD BasicCarRef(Byte* w, Byte* wide, Byte* cold, int)
        :
#define ROCKETGPU_CAR_REF_FROM_TILE(name, region, offset)                                                               \
    name(*reinterpret_cast<Field<decltype(CarState::name)>*>(((region) == 0 ? w : (region) == 1 ? wide : cold) + (offset))),
          ROCKETGPU_CAR_TILE_FIELDS(ROCKETGPU_CAR_REF_FROM_TILE)
#undef ROCKETGPU_CAR_REF_FROM_TILE
          unused_(0), org_w_(w), org_wide_(wide), org_cold_(cold) {
    }
    template <class T, size_t N, class U>
    ROCKETGPU_HD static void CopyField(T (&dst)[N], const U (&src)[N]) {
        for (size_t i = 0; i < N; ++i) dst[i] = src[i];
    }
    template <class T, class U>
    ROCKETGPU_HD static void CopyField(T& dst, const U& src) {
        dst = src;
    }
    int unused_;

public:
    Byte* org_w_ = nullptr;
    Byte* org_wide_ = nullptr;
    Byte* org_cold_ = nullptr;
    S* org_state_ = nullptr;
};

template <class S>
struct BasicCarArg {
    using Ref = BasicCarRef<S>;
    using Byte = typename Ref::Byte;
    Byte* w = nullptr;
    Byte* wide = nullptr;
    Byte* cold = nullptr;
    S* state = nullptr;

    BasicCarArg() = default;
    ROCKETGPU_HD BasicCarArg(const Ref& r) : w(r.org_w_), wide(r.org_wide_), cold(r.org_cold_), state(r.org_state_) { CheckTile(); }
    template <class S2, class = typename std::enable_if<Ref::kConst && !std::is_const<S2>::value>::type>
    ROCKETGPU_HD BasicCarArg(const BasicCarRef<S2>& r) : w(r.org_w_), wide(r.org_wide_), cold(r.org_cold_), state(r.org_state_) {
        CheckTile();
    }
    template <class S2, class = typename std::enable_if<Ref::kConst && !std::is_const<S2>::value>::type>
    ROCKETGPU_HD BasicCarArg(const BasicCarArg<S2>& o) : w(o.w), wide(o.wide), cold(o.cold), state(o.state) {}
#ifdef __CUDA_ARCH__
    BasicCarArg(S& s) = delete;
    ROCKETGPU_HD void CheckTile() {
        if (state != nullptr) __trap();
        state = nullptr;
    }
    ROCKETGPU_HD const void* id() const { return w; }
#else
    ROCKETGPU_HD BasicCarArg(S& s) : state(&s) {}
    ROCKETGPU_HD void CheckTile() {}
    ROCKETGPU_HD const void* id() const { return state != nullptr ? static_cast<const void*>(state) : static_cast<const void*>(w); }
#endif
};

template <class S>
ROCKETGPU_HD BasicCarRef<S>::BasicCarRef(const BasicCarArg<S>& a)
    :
#ifdef __CUDA_ARCH__
#define ROCKETGPU_CAR_REF_FROM_ARG(name, region, offset)                                                                \
    name(*reinterpret_cast<Field<decltype(CarState::name)>*>(((region) == 0 ? a.w : (region) == 1 ? a.wide : a.cold) + \
                                                              (offset))),
#else
#define ROCKETGPU_CAR_REF_FROM_ARG(name, region, offset)                                                                \
    name(a.state != nullptr ? a.state->name                                                                            \
                            : *reinterpret_cast<Field<decltype(CarState::name)>*>(                                     \
                                  ((region) == 0 ? a.w : (region) == 1 ? a.wide : a.cold) + (offset))),
#endif
      ROCKETGPU_CAR_TILE_FIELDS(ROCKETGPU_CAR_REF_FROM_ARG)
#undef ROCKETGPU_CAR_REF_FROM_ARG
#ifdef __CUDA_ARCH__
      unused_(0), org_w_(a.w), org_wide_(a.wide), org_cold_(a.cold), org_state_(nullptr) {
#else
      unused_(0), org_w_(a.w), org_wide_(a.wide), org_cold_(a.cold), org_state_(a.state) {
#endif
}

using CarRef = BasicCarRef<CarState>;
using CarCRef = BasicCarRef<const CarState>;

template <class S>
struct BasicCarSpan {
    using Ref = BasicCarRef<S>;
    using Byte = typename Ref::Byte;
    Byte* tiles = nullptr;
    long long first = 0;

    struct Arrow {
        Ref ref;
        ROCKETGPU_HD const Ref* operator->() const { return __builtin_addressof(ref); }
    };

    BasicCarSpan() = default;
    ROCKETGPU_HD BasicCarSpan(std::nullptr_t) {}
    ROCKETGPU_HD BasicCarSpan(Byte* t, long long f) : tiles(t), first(f) {}
    template <class S2, class = typename std::enable_if<Ref::kConst && !std::is_const<S2>::value>::type>
    ROCKETGPU_HD BasicCarSpan(const BasicCarSpan<S2>& o) : tiles(o.tiles), first(o.first) {}

    ROCKETGPU_HD Ref operator[](long long i) const {
        const long long c = first + i;
        return Ref(tiles + static_cast<size_t>(c >> 5) * kCarTileBytes, static_cast<int>(c & 31));
    }
    ROCKETGPU_HD Ref operator*() const { return (*this)[0]; }
    ROCKETGPU_HD Arrow operator->() const { return Arrow{(*this)[0]}; }
    ROCKETGPU_HD BasicCarSpan operator+(long long n) const { return BasicCarSpan(tiles, first + n); }
    ROCKETGPU_HD BasicCarSpan operator-(long long n) const { return BasicCarSpan(tiles, first - n); }
    ROCKETGPU_HD BasicCarSpan& operator+=(long long n) { first += n; return *this; }
    ROCKETGPU_HD BasicCarSpan& operator++() { ++first; return *this; }
    ROCKETGPU_HD long long operator-(const BasicCarSpan& o) const { return first - o.first; }
    ROCKETGPU_HD bool operator==(std::nullptr_t) const { return tiles == nullptr; }
    ROCKETGPU_HD bool operator!=(std::nullptr_t) const { return tiles != nullptr; }
    ROCKETGPU_HD bool operator==(const BasicCarSpan& o) const { return tiles == o.tiles && first == o.first; }
    ROCKETGPU_HD bool operator!=(const BasicCarSpan& o) const { return !(*this == o); }
    ROCKETGPU_HD explicit operator bool() const { return tiles != nullptr; }
};

template <class S>
struct BasicCarPtr {
    using Ref = BasicCarRef<S>;
    BasicCarArg<S> arg;
    bool valid = false;

    struct Arrow {
        Ref ref;
        ROCKETGPU_HD const Ref* operator->() const { return __builtin_addressof(ref); }
    };

    BasicCarPtr() = default;
    ROCKETGPU_HD BasicCarPtr(std::nullptr_t) {}
    ROCKETGPU_HD explicit BasicCarPtr(const Ref& r) : arg(r), valid(true) {}
    template <class S2, class = typename std::enable_if<Ref::kConst && !std::is_const<S2>::value>::type>
    ROCKETGPU_HD BasicCarPtr(const BasicCarPtr<S2>& o) : arg(o.arg), valid(o.valid) {}
    ROCKETGPU_HD Arrow operator->() const { return Arrow{Ref(arg)}; }
    ROCKETGPU_HD Ref operator*() const { return Ref(arg); }
    ROCKETGPU_HD explicit operator bool() const { return valid; }
    ROCKETGPU_HD bool operator==(std::nullptr_t) const { return !valid; }
    ROCKETGPU_HD bool operator!=(std::nullptr_t) const { return valid; }
    template <class S2>
    ROCKETGPU_HD bool operator==(const BasicCarPtr<S2>& o) const {
        return valid == o.valid && (!valid || arg.id() == o.arg.id());
    }
    template <class S2>
    ROCKETGPU_HD bool operator!=(const BasicCarPtr<S2>& o) const { return !(*this == o); }
};

using CarPtr = BasicCarPtr<CarState>;
using CarCPtr = BasicCarPtr<const CarState>;
template <class S>
ROCKETGPU_HD BasicCarPtr<S> BasicCarRef<S>::operator&() const {
    return BasicCarPtr<S>(*this);
}

using CarSpan = BasicCarSpan<CarState>;
using CarCSpan = BasicCarSpan<const CarState>;

ROCKETGPU_HD constexpr size_t CarStoreBytes(size_t car_capacity) { return CarTileStoreBytes(car_capacity); }
ROCKETGPU_HD CarSpan CarSpanFromBytes(void* p) { return CarSpan(static_cast<unsigned char*>(p), 0); }
ROCKETGPU_HD void* CarSpanBytes(const CarCSpan& s) { return const_cast<unsigned char*>(s.tiles); }
constexpr int kCarChunks8 = (ROCKETGPU_CAR_TILE_WORDS * kCarTileWordBytes + ROCKETGPU_CAR_TILE_WIDE_BYTES +
                             ROCKETGPU_CAR_TILE_COLD_BYTES) / 8;
ROCKETGPU_HD size_t CarTileChunkOffset(int lane, int k) {
    constexpr int kWordChunks = ROCKETGPU_CAR_TILE_WORDS * 2;
    constexpr int kWideChunks = ROCKETGPU_CAR_TILE_WIDE_BYTES / 8;
    return k < kWordChunks ? static_cast<size_t>(k >> 1) * kCarTileLanes * kCarTileWordBytes + lane * kCarTileWordBytes + (k & 1) * 8
           : k < kWordChunks + kWideChunks
               ? kCarTileWideStart + static_cast<size_t>(lane) * ROCKETGPU_CAR_TILE_WIDE_BYTES + (k - kWordChunks) * 8
               : kCarTileColdStart + static_cast<size_t>(lane) * ROCKETGPU_CAR_TILE_COLD_BYTES + (k - kWordChunks - kWideChunks) * 8;
}
template <class S>
ROCKETGPU_HD auto CarChunk8(const BasicCarSpan<S>& s, long long idx, int k) {
    using Word = typename std::conditional<std::is_const<S>::value, const unsigned long long, unsigned long long>::type;
    const long long c = s.first + idx;
    return reinterpret_cast<Word*>(s.tiles + static_cast<size_t>(c >> 5) * kCarTileBytes + CarTileChunkOffset(static_cast<int>(c & 31), k));
}

using CarRefT = CarRef;
using CarCRefT = CarCRef;
using CarSpanT = CarSpan;
using CarCSpanT = CarCSpan;
using CarPtrT = CarPtr;
using CarCPtrT = CarCPtr;
using CarArgT = BasicCarArg<CarState>;
using CarCArgT = BasicCarArg<const CarState>;

#endif

}
}
