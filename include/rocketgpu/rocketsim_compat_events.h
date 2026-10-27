#pragma once

#include <cstdint>
#include <functional>

#include "rocketgpu/rocketsim_compat.h"

namespace rocketgpu {
namespace compat {
using ShotEventFn = std::function<void(Arena* arena, Car* shooter, Car* passer, void* user_info)>;
using GoalEventFn = std::function<void(Arena* arena, Car* scorer, Car* passer, void* user_info)>;
using SaveEventFn = std::function<void(Arena* arena, Car* saver, void* user_info)>;

struct GameEventTrackerConfig {
    float shotMinSpeed = 1750.0f;
    float shotTouchMinDelay = 0.3f;
    float predScoreExtraMargin = 0.0f;
    float shotEventCooldown = 1.0f;
    float shotMinScoreTime = 2.0f;
    float goalMaxTouchTime = 4.0f;
    float passMaxTouchTime = 2.0f;
};

struct GameEventTracker {
    GameEventTrackerConfig config = {};

    void SetShotCallback(ShotEventFn callback_fn, void* user_info = nullptr) {
        shot_callback_ = std::move(callback_fn);
        shot_user_info_ = user_info;
    }
    void SetGoalCallback(GoalEventFn callback_fn, void* user_info = nullptr) {
        goal_callback_ = std::move(callback_fn);
        goal_user_info_ = user_info;
    }
    void SetSaveCallback(SaveEventFn callback_fn, void* user_info = nullptr) {
        save_callback_ = std::move(callback_fn);
        save_user_info_ = user_info;
    }

    void Update(Arena* arena);
    bool autoStateSetDetection = true;
    void ResetPersistentInfo();

private:
    ShotEventFn shot_callback_;
    void* shot_user_info_ = nullptr;
    GoalEventFn goal_callback_;
    void* goal_user_info_ = nullptr;
    SaveEventFn save_callback_;
    void* save_user_info_ = nullptr;

    float shot_cooldown_ = 0.0f;
    bool ball_shot_ = false;
    Team ball_shot_goal_team_ = Team::Blue;
    bool ball_scored_last_ = false;
    uint64_t last_ball_update_count_ = 0;
};
}
}

namespace rocketgpu {
namespace compat {
struct BallPredTracker {
    Arena* ballPredArena = nullptr;
    std::vector<BallStateCompat> predData;
    size_t numPredTicks = 0;
    int lastUpdateTickCount = 0;

    BallPredTracker(Arena* arena, size_t num_pred_ticks);
    ~BallPredTracker();
    BallPredTracker(const BallPredTracker&) = delete;
    BallPredTracker& operator=(const BallPredTracker&) = delete;

    void UpdatePredFromArena(Arena* arena);
    void UpdatePredManual(const BallStateCompat& cur_ball_state, int ticks_since_last_update);
    void ForceUpdateAllPred(const BallStateCompat& initial_ball_state);
    BallStateCompat GetBallStateForTime(float pred_time) const;
};

bool BallStateMatches(const BallStateCompat& a, const BallStateCompat& b, float margin_pos = 0.8f, float margin_vel = 0.4f, float margin_ang_vel = 0.02f);
}
}
