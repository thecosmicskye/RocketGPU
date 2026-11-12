# RocketGPU

Faster, bit-accurate Rocket League physics. Works best with [RocketChad3](https://github.com/thecosmicskye/RocketChad3).

## Speed

Arena-ticks/s from `bench_cpp --replay`: RocketSim v3's `arena_scale_v3` training workload (55% 1v1, 27% 2v2 and 18%
3v3 arenas, resets on goals and timeouts), with each arena playing the recorded controls of a real match from the
[fullstate suite](https://github.com/thecosmicskye/RocketGPU-fullstate-suite). RocketGPU CUDA on an RTX 5090; RocketGPU
CPU (profile-guided build) and RocketSim v3 0.3.5 (`arena_scale_v3`, same recorded controls) on a Ryzen 7 9800X3D with
14 threads:

| Arenas | RocketGPU CUDA | RocketGPU CPU | RocketSim v3 |
|---|---|---|---|
| 96 | 0.66M | 4.40M | 2.91M |
| 256 | 1.49M | 4.57M | 3.54M |
| 512 | 2.56M | 4.72M | 3.51M |
| 1,024 | 4.43M | 4.68M | 3.79M |
| 4,096 | 10.84M | 3.86M | 3.61M |
| 8,192 | 15.65M | 3.69M | 3.73M |
| 16,384 | 18.56M | 3.75M | 3.52M |

## Accuracy

[144 real-game recordings](https://github.com/thecosmicskye/RocketGPU-fullstate-suite): 1v1 to 3v3 matches, uneven
teams, every car body, boost pad tests and scripted scenarios. Each 120-tick segment starts from the game's state and
replays the game's inputs; 25,622 segments, 13,930,650 body-ticks. v3 takes no hidden state, so it starts from the
public state.

| | RocketGPU | v3 0.3.5 |
|---|---|---|
| Bit-exact body-ticks (pos, vel, ang vel, rotation at 0 ulp) | 100% | 4.32% |
| Mean error after 1 tick | 0 | 0.80 |
| Mean error after 30 ticks | 0 | 0.65 |
| Mean error after 120 ticks | 0 | 1.63 |

Error: rlpr_metric's normalized error per car-tick (position / 2 uu, velocity / 3 uu/s, angular velocity / 0.05
rad/s, axes / 0.05, car plus ball), capped at 10.

## Build

Needs CMake 3.21+ and a C++17 compiler. CUDA builds also need CUDA 12.1+; Python bindings need Python
development files and pybind11.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DROCKETGPU_USE_CUDA=ON -DROCKETGPU_BUILD_PYTHON=ON
```

```bash
cmake --build build -j
```

The Python module `rocketgpu_py` ends up in `build/python`, along with a RocketSim-style `rocketsim` package.
Without `-DROCKETGPU_USE_CUDA=ON` you get a CPU build, tuned for the machine that builds it; add
`-DROCKETGPU_NATIVE_ARCH=OFF` for a binary that runs on other machines. `ROCKETGPU_CPU_THREADS`, set before the first
CPU call, fixes the thread count.

CUDA builds compile the kernels as seven translation units by default, which needs Python 3 at configure time.
Without it, or with `-DROCKETGPU_CUDA_SPLIT=OFF`, they build as one file: slower, in about twice the memory, with
identical results.

The fastest CPU build is profile-guided (needs `ROCKETGPU_COLLISION_MESH_DIR` and numpy):

```bash
tools/pgo_build.sh build -DROCKETGPU_USE_CUDA=OFF -DCMAKE_CXX_COMPILER=clang++
```

CPU builds use ISPC kernels when `ispc` is installed; `-DROCKETGPU_CPU_ISPC=OFF` gives a scalar build.

`-DROCKETGPU_BUILD_BENCH=ON` also builds `bench_cpp`, a C++ throughput benchmark (RocketSim v3's `arena_scale_v3`
workload). It needs a C++20 compiler, and with CUDA, CMake 3.25.2+ and CUDA 12.1+:

```bash
python3 tools/replay_controls.py path/to/RocketGPU-fullstate-suite replay.bin
build/bench_cpp --arenas 1024 --replay replay.bin
```

Without `--replay`, controls are random.

## Collision meshes

Meshes are not included. [Dump them](https://github.com/ZealanL/RLArenaCollisionDumper) from your own copy of Rocket League, then set:

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

When controls for several ticks are known in advance, batch them in one call. For a CPU build:

```python
import numpy as np
import rocketgpu_py_cuda as views

controls_by_tick = np.zeros((4, sim.car_capacity()), dtype=views.state_dtype('controls'))
controls_by_tick['throttle'][2:] = 1.0
sim.step_with_device_controls(controls_by_tick.ctypes.data, 4, sim.car_capacity())
```

Each row holds one `CarControls` per car slot; the stride is in records and must be at least `sim.car_capacity()`.
Pass a control stride of 0 to hold one row of controls for every tick (also in `step_with_device_commands`).
On CUDA builds, pass a device pointer and keep its allocation alive until the chosen stream completes. Use separate
steps when controls or callbacks depend on intermediate physics states.

On CPU builds, a live NumPy view reads car state without per-car Python objects:

```python
cars = views.host_struct_array(sim, 'car_states')
positions = cars['pos'].copy()
```

Rows are `env * max_cars_per_env + slot`; the first `env_car_count` slots of each arena are active. The view is
read-only and changes as the simulator steps, so copy what you keep and set state through the state API.

On CUDA builds, `views.field_arrays(sim, 'car_states')` gives zero-copy device views shaped `(tiles, 32[, dims])`, car
`i` at `[i // 32, i % 32]`:

```python
pos = cp.asarray(views.field_arrays(sim, 'car_states')['pos']).reshape(-1, 3)[:sim.car_capacity()]  # copies
```

The raw car pointer (`DeviceView::car_states`, `sim.device_arrays()['car_states']`) is that tile store, described by
`views.state_layout('car_states')['device']`. To get plain `CarState` records on any build, use
`sim.download_state_array('car_states', host_ptr, count)` or `download_car_states`.

## Custom arenas

A soccar match (scoring, kickoffs, boost pads, demos) on your own geometry. A default `ArenaLayout` is the official
arena; change any part of it:

```python
layout = rg.ArenaLayout()
layout.mesh_dir = '/path/to/my_arena'          # every .cmf file in it
layout.meshes = [rg.ArenaMesh(vertices, indices)]  # and/or triangles: Vec3 vertices in uu, 3 indices per triangle
layout.planes = rg.ArenaLayout.official_planes()   # floor, ceiling and side walls; or your own ArenaPlane(point, normal)
pad = rg.BoostPadConfig()
pad.pos, pad.is_big = rg.Vec3(0, 2000, 73), 1
layout.boost_pads = layout.boost_pads + [pad]
goals = layout.goals                           # [blue, orange]: point, normal (into the goal), optional min/max box
goals[0].min, goals[0].max = rg.Vec3(-893, -1e30, -1e30), rg.Vec3(893, 1e30, 642.775)
layout.goals = goals
# layout.kickoff_spawns (2 x 5), layout.respawn_spawns (2 x 4): ArenaSpawn(pos, yaw); layout.ball_spawn
sim = rg.BatchSim(cfg, layout)
```

Spawns and goals are in world coordinates for each team. Every position must be within 11,500 uu of the origin on each
axis and every spawn yaw within 8π of zero. Meshes are merged into at most 16 groups. Custom geometry skips the
official arena's fast paths, so it steps slower than the official arena.

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
- the step's own flags and `jump_ticks`; `has_jumped` is also set by a double jump or dodge with no jump before it
- `is_supersonic` with v3's grace timer
- `bump_cooldown_victim` (a car id) and `last_extra_hit_tick` (`None`: no earlier hit)

`Arena.get_full_state()` returns the whole simulation state, including contact manifolds and timers, as
`{"array.field": bytes}`. `set_full_state` writes the fields it is given and keeps the rest.

## Usage & Legal Notice
RocketGPU does not contain any code from the Rocket League. This project is not affiliated with Epic Games or Psyonix.

RocketGPU is intended to be used for creating machine learning environments, as well as other hobby projects (provided they don't violate Rocket League's terms of use/service). The use of RocketGPU to create a playable Rocket League clone is heavily discouraged. Please do not do this!

### To Epic Games & Psyonix
If any of you have an issue with this project, let me know and we can resolve it! I will be happy to make changes if needed.

## License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Credits

Built on [RocketSim](https://github.com/ZealanL/RocketSim) by ZealanL. Thanks to [VirxEC](https://github.com/VirxEC) for
guidance. Thanks to akaAMAZING, creator of [RLMAX](https://github.com/akaAMAZING/RLMAX), for ideas that made RocketGPU faster.
