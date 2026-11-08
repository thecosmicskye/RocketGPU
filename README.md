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
| 96 | 0.59M | 4.17M | 2.91M |
| 256 | 1.31M | 4.26M | 3.54M |
| 512 | 2.24M | 4.43M | 3.51M |
| 1,024 | 3.79M | 4.38M | 3.79M |
| 4,096 | 9.03M | 3.65M | 3.61M |
| 8,192 | 11.48M | 3.61M | 3.73M |
| 16,384 | 12.42M | 3.68M | 3.52M |

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

Needs CMake 3.21+ and a C++17 compiler. CUDA builds also need the CUDA toolkit; Python bindings need Python
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

The fastest CPU build is profile-guided (needs `ROCKETGPU_COLLISION_MESH_DIR` and numpy):

```bash
tools/pgo_build.sh build -DROCKETGPU_USE_CUDA=OFF -DCMAKE_CXX_COMPILER=clang++
```

CPU builds use ISPC kernels when `ispc` is installed; `-DROCKETGPU_CPU_ISPC=OFF` gives a scalar build.

`-DROCKETGPU_BUILD_BENCH=ON` also builds `bench_cpp`, a C++ throughput benchmark (RocketSim v3's `arena_scale_v3`
workload). It needs a C++20 compiler, and with CUDA, CMake 3.25.2+ and CUDA 12+:

```bash
python3 tools/replay_controls.py path/to/RocketGPU-fullstate-suite replay.bin
build/bench_cpp --arenas 1024 --replay replay.bin
```

Without `--replay`, controls are random.

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
- `bump_cooldown_victim` (a car id) and `last_extra_hit_tick` (`None`: no earlier hit)

`jump_hold_broken` is stored, but RocketGPU's jump does not read it.

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
guidance.
