# RocketGPU

Rocket League physics on a GPU.

## Build

Needs CMake 3.21+, a C++17 compiler, the CUDA toolkit and pybind11.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DROCKETGPU_USE_CUDA=ON -DROCKETGPU_BUILD_PYTHON=ON
```

```bash
cmake --build build -j
```

The Python module `rocketgpu_py` ends up in `build/python`, along with a RocketSim-style `rocketsim` package.
Without `-DROCKETGPU_USE_CUDA=ON` you get a CPU build.

## Collision meshes

Meshes are not included. Dump them from your own copy of Rocket League, then set:

```bash
export ROCKETGPU_COLLISION_MESH_DIR=/path/to/collision_meshes
```

The folder needs `soccar/` (the RocketSim arena meshes).

## Use

```python
import rocketgpu_py as rg

cfg = rg.SimConfig()
cfg.env_count = 8192
cfg.max_cars_per_env = 6
sim = rg.BatchSim(cfg)
sim.set_env_car_count([6] * cfg.env_count)
sim.set_teams(([rg.Team.Blue] * 3 + [rg.Team.Orange] * 3) * cfg.env_count)
sim.reset_to_kickoff(0)
controls = rg.CarControls()
controls.throttle = 1.0
sim.upload_controls([controls] * sim.car_capacity())
sim.step(1)
cars = sim.download_car_states(sim.car_capacity())
```

## Car state

Car fields that RLBot v5 exposes hold RLBot's value for the tick: `vel` and `ang_vel` after the speed caps,
`is_supersonic` (speed >= 2200), `has_jumped`, `has_double_jumped`, `has_flipped`, and the RLBot-only `air_state`,
`dodge_timeout`, `dodge_elapsed`, `dodge_dir_x`/`dodge_dir_y` and latest ball touch. The other timers follow RocketSim
v3 (`jump_time` is v3's `jump_ticks / 120`). Written states use the same meanings; set `tick_count_since_update = 0` on
every car you change. `include/rocketgpu/rocketgpu.h` documents each field.

For RocketSim v3's own conventions, the compat `Car::GetStateV3` / `SetStateV3` (Python `get_state_v3` /
`set_state_v3`) read and write `CarStateV3`, which mirrors v3's `CarState` field for field: velocities before the caps,
the step's own flags, `jump_ticks`, the grace-timer `is_supersonic`, `bump_cooldown_victim` (a car id) and
`last_extra_hit_tick` (`None` when unset). RocketGPU stores `jump_hold_broken` but its jump does not read it.

## Speed

RTX 5090, random controls, 55% 1v1, 27% 2v2 and 18% 3v3 arenas:

| Arenas | Arena-ticks/s |
|---|---|
| 1,024 | 1.15M |
| 8,192 | 4.20M |
| 16,384 | 6.12M |
| 32,768 | 7.25M |

## Accuracy

Compared against RocketSim v3 (latest).

**RLBot match suite**: real-game states captured through RLBot (lower is better)

| | RocketGPU | v3 (latest) |
|---|---|---|
| Mean position error after 1 s, all 2342 match windows (uu) | 0.577 | 2.409 |
| Mean position error, 393 plan trials (uu) | 0.0999 | 0.271 |

**v3's suite**: v3's real-game recordings, scored with its `rlpr_metric` (pass %: higher is better; error norm: lower is better)

| | RocketGPU | v3 (latest) |
|---|---|---|
| Games, pass % (709,990 car-ticks) | 98.74 | 92.29 |
| Games, mean error norm | 0.314 | 2.88 |
| Special cases, pass % (15,586 car-ticks) | 97.77 | 90.12 |
| Special cases, mean error norm | 0.181 | 1.82 |

## License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Credits

Built on [RocketSim](https://github.com/ZealanL/RocketSim) by ZealanL. Thanks to [VirxEC](https://github.com/VirxEC) for
guidance and accuracy suite.
