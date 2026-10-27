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

The folder needs `soccar/` (the RocketSim arena meshes) and `soccar_goals/` (the goal meshes `goal_obj8.cmf` to `goal_obj11.cmf`).

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

## Speed

RTX 5090, random controls, 55% 1v1, 27% 2v2 and 18% 3v3 arenas:

| Arenas | Arena-ticks/s |
|---|---|
| 1,024 | 1.01M |
| 8,192 | 4.50M |
| 16,384 | 6.57M |
| 32,768 | 8.67M |

## Accuracy

Compared against RocketSim v3 (latest), using RLBot recordings as ground truth.

**RLBot match suite** (lower is better)

| | RocketGPU | v3 (latest) |
|---|---|---|
| Mean position error after 1 s, all 2342 match windows (uu) | 1.152 | 2.409 |
| Mean position error, 393 plan trials (uu) | 0.133 | 0.271 |

**v3's suite** (pass %: higher is better; error norm: lower is better)

| | RocketGPU | v3 (latest) |
|---|---|---|
| Games, pass % (688,806 car-ticks) | 91.75 | 91.01 |
| Games, mean error norm | 8.38 | 9.09 |
| Special cases, pass % (14,356 car-ticks) | 91.13 | 90.34 |
| Special cases, mean error norm | 1.11 | 2.01 |

## License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Credits

Built on [RocketSim](https://github.com/ZealanL/RocketSim) by ZealanL.
