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

`CarState` fields that RLBot v5 exposes hold RLBot's value for the tick:

- `vel` and `ang_vel`, after the speed caps
- `is_supersonic`: speed >= 2200 uu/s
- `has_jumped`, `has_double_jumped`, `has_flipped`
- RLBot-only: `air_state`, `dodge_timeout`, `dodge_elapsed`, `dodge_dir_x` / `dodge_dir_y` and the latest ball touch

The other timers follow RocketSim v3 (`jump_time` is v3's `jump_ticks / 120`). Written states use the same meanings;
set `tick_count_since_update = 0` on every car you change. `include/rocketgpu/rocketgpu.h` documents each field.

For RocketSim v3's own conventions, use `get_state_v3` / `set_state_v3` (C++ `Car::GetStateV3` / `SetStateV3`).
`CarStateV3` matches v3's `CarState` field for field:

- velocities before the caps
- the step's own flags and `jump_ticks`
- `is_supersonic` with v3's grace timer
- `bump_cooldown_victim` (a car id) and `last_extra_hit_tick` (`None` when unset)

`jump_hold_broken` is stored, but RocketGPU's jump does not read it.

`Arena.get_full_state()` returns the whole simulation state, including contact manifolds and timers, as
`{"array.field": bytes}`. `set_full_state` writes the fields it is given and keeps the rest.

## Speed

For CPU builds, `ROCKETGPU_CPU_THREADS` sets the worker count; set it before creating a batch. Unset or 0 uses the
available hardware threads (on Linux, those in the creating thread's CPU affinity), at most one per 16 environments.

Arena-ticks/s. RocketGPU on an RTX 5090; RocketSim v3 0.3.5 on a Ryzen 7 9800X3D with 16 threads. Random controls,
55% 1v1, 27% 2v2 and 18% 3v3 arenas:

| Arenas | RocketGPU | RocketSim v3 |
|---|---|---|
| 1,024 | 3.16M | 4.10M |
| 4,096 | 7.78M | 4.38M |
| 8,192 | 11.61M | 4.72M |
| 16,384 | 12.18M | 5.49M |

## Accuracy

121 real-game recordings: 1v1 to 3v3 matches, every car body, and scripted scenarios. Each 120-tick segment starts
from the game's state and replays the game's inputs; 22,557 segments, 12,038,088 body-ticks. v3 takes no hidden
state, so it starts from the public state.

| | RocketGPU | v3 0.3.5 |
|---|---|---|
| Bit-exact body-ticks (pos, vel, ang vel, rotation at 0 ulp) | 100% | 4.49% |
| Mean error after 1 tick | 0 | 0.80 |
| Mean error after 30 ticks | 0 | 0.64 |
| Mean error after 120 ticks | 0 | 1.65 |

Error: rlpr_metric's normalized error per car-tick (position / 2 uu, velocity / 3 uu/s, angular velocity / 0.05
rad/s, axes / 0.05, car plus ball), capped at 10.

## License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Credits

Built on [RocketSim](https://github.com/ZealanL/RocketSim) by ZealanL. Thanks to [VirxEC](https://github.com/VirxEC) for
guidance and accuracy suite.
