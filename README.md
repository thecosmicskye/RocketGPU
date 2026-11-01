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

Arena-ticks/s. RocketGPU on an RTX 5090; RocketSim v3 0.3.4 on a Ryzen 7 9800X3D with 16 threads. Random controls,
55% 1v1, 27% 2v2 and 18% 3v3 arenas:

| Arenas | RocketGPU | RocketSim v3 |
|---|---|---|
| 1,024 | 3.99M | 3.79M |
| 4,096 | 10.37M | 3.99M |
| 8,192 | 15.62M | 4.05M |
| 16,384 | 17.44M | 4.34M |

## Accuracy

Compared against RocketSim v3 0.3.4.

**RLBot match suite**: real-game states captured through RLBot (lower is better)

| | RocketGPU | v3 0.3.4 |
|---|---|---|
| Mean position error after 1 s, all 2342 match windows (uu) | 0.576 | 1.931 |
| Mean position error, 393 plan trials (uu) | 0.0998 | 0.271 |

**v3's suite**: v3's real-game recordings, scored with its `rlpr_metric` (pass %: higher is better; error norm: lower is better)

| | RocketGPU | v3 0.3.4 |
|---|---|---|
| Games, pass % (1,073,187 car-ticks) | 99.10 | 97.67 |
| Games, mean error norm | 0.244 | 0.706 |
| Special cases, pass % (15,595 car-ticks) | 98.03 | 93.15 |
| Special cases, mean error norm | 0.190 | 0.753 |

## License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Credits

Built on [RocketSim](https://github.com/ZealanL/RocketSim) by ZealanL. Thanks to [VirxEC](https://github.com/VirxEC) for
guidance and accuracy suite.
