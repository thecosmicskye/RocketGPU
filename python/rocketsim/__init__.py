import rocketgpu_py as _rs
from rocketgpu_py import compat as _compat

Arena = _compat.Arena
Car = _compat.Car
Ball = _compat.Ball
GameMode = _compat.GameMode
RotMat = _compat.RotMat
CarState = _compat.CarState
BallState = _compat.BallState
GameEventTracker = _compat.GameEventTracker
GameEventTrackerConfig = _compat.GameEventTrackerConfig
BallPredTracker = _compat.BallPredTracker
ball_state_matches = _compat.ball_state_matches

Team = _rs.Team
Vec = _rs.Vec3
Vec3 = _rs.Vec3
CarControls = _rs.CarControls
CarConfig = _rs.CarConfig
MutatorConfig = _rs.MutatorConfig
BoostPadState = _rs.BoostPadState

_PRESETS = {
    "CAR_CONFIG_OCTANE": "OctaneConfig",
    "CAR_CONFIG_DOMINUS": "DominusConfig",
    "CAR_CONFIG_PLANK": "PlankConfig",
    "CAR_CONFIG_BREAKOUT": "BreakoutConfig",
    "CAR_CONFIG_HYBRID": "HybridConfig",
    "CAR_CONFIG_MERC": "MercConfig",
    "CAR_CONFIG_PSYCLOPS": "PsyclopsConfig",
}
for _name, _fn in _PRESETS.items():
    if hasattr(_rs, _fn):
        globals()[_name] = getattr(_rs, _fn)()

BatchSim = _rs.BatchSim
SimConfig = _rs.SimConfig

_initialized = False


def init(mesh_dir=None):
    global _initialized
    _initialized = True
    return None


def is_initialized():
    return _initialized


__all__ = [
    "Arena", "Car", "Ball", "GameMode", "RotMat", "CarState", "BallState", "GameEventTracker",
    "GameEventTrackerConfig", "BallPredTracker", "ball_state_matches", "Team", "Vec", "Vec3",
    "CarControls", "CarConfig", "MutatorConfig", "BoostPadState", "BatchSim", "SimConfig",
    "init", "is_initialized",
] + list(_PRESETS)
