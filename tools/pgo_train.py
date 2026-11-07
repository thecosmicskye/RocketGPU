#!/usr/bin/env python3
import argparse, itertools, struct

import numpy as np
import rocketgpu_py as rs
import rocketgpu_py_cuda as views

TICK_SKIP, EPISODE_STEPS, GROUND_FRAC = 8, 900, 0.75
TEAM_SIZE_FRACTIONS = {1: 0.5478515625, 2: 0.271484375}
CAR_FIELDS = ('hitbox_size', 'hitbox_offset', 'front_wheels_offset', 'back_wheels_offset', 'front_wheel_radius',
              'back_wheel_radius', 'front_suspension_rest', 'back_suspension_rest')
CARS = {
    'OCTANE': ((120.507, 86.6994, 38.6591), (13.8757, 0, 20.755), (51.25, 25.9, 20.755), (-33.75, 29.5, 20.755), 12.5, 15, 38.755, 37.055),
    'DOMINUS': ((130.427, 85.7799, 33.8), (9, 0, 15.75), (50.3, 31.1, 15.75), (-34.75, 33, 15.75), 12, 13.5, 33.95, 33.85),
    'PLANK': ((131.32, 87.1704, 31.8944), (9.00857, 0, 12.0942), (49.97, 27.8, 12.0942), (-35.43, 20.28, 12.0942), 12.5, 17, 31.9242, 27.9242),
    'BREAKOUT': ((133.992, 83.021, 32.8), (12.5, 0, 11.75), (51.5, 26.67, 11.75), (-35.75, 35, 11.75), 13.5, 15, 29.7, 29.666),
    'HYBRID': ((129.519, 84.6879, 36.6591), (13.8757, 0, 20.755), (51.25, 25.9, 20.755), (-34, 29.5, 20.755), 12.5, 15, 38.755, 37.055),
    'MERC': ((123.22, 79.2103, 44.1591), (11.3757, 0, 21.505), (51.25, 25.9, 21.505), (-33.75, 29.5, 21.505), 15, 15, 39.505, 39.105),
}
OPPONENT_MIX = (('OCTANE', 446064), ('DOMINUS', 20470), ('HYBRID', 500000), ('BREAKOUT', 11547), ('PLANK', 11120), ('MERC', 10799))


def f32(x):
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]


def car_config(name):
    c = rs.CarConfig()
    for field, v in zip(CAR_FIELDS, CARS[name]):
        setattr(c, field, rs.Vec3(*map(f32, v)) if isinstance(v, tuple) else f32(v))
    c.dodge_deadzone = 0.5
    return c


def action_table():
    rows = []
    for throttle, steer, boost, handbrake in itertools.product((-1, 0, 1), (-1, 0, 1), (0, 1), (0, 1)):
        if not (boost and throttle != 1):
            rows.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
    for pitch, yaw, roll, jump, boost in itertools.product((-1, 0, 1), (-1, 0, 1), (-1, 0, 1), (0, 1), (0, 1)):
        if not (jump and yaw != 0) and not (pitch == roll == jump == 0):
            rows.append([boost, yaw, pitch, yaw, roll, jump, boost, int(jump and (pitch != 0 or yaw != 0 or roll != 0))])
    return np.array(rows, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arenas', type=int, default=1024)
    ap.add_argument('--ticks', type=int, default=8_000_000, help='arena-ticks to step')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    n, rng = args.arenas, np.random.default_rng(args.seed)
    cfg = rs.SimConfig()
    cfg.env_count, cfg.max_cars_per_env = n, 6
    cfg.enable_boost_pads = cfg.enable_car_car_collisions = cfg.enable_demos = cfg.enable_scoring = 1
    cfg.tick_dt = 1.0 / 120
    sim = rs.BatchSim(cfg)
    sizes = np.full(n, 3)
    perm, off = rng.permutation(n), 0
    for m, frac in TEAM_SIZE_FRACTIONS.items():
        k = int(n * frac) // 16 * 16 if n >= 1024 else int(n * frac)
        sizes[perm[off:off + k]] = m
        off += k
    sim.set_env_car_count([2 * int(k) for k in sizes], 0)
    sim.set_teams([t for k in sizes for t in [rs.Team.Blue] * k + [rs.Team.Orange] * (6 - k)], 0)
    states = [rs.EnvState() for _ in range(n)]
    for s in states:
        s.car_visit_order_mode = 0
    sim.upload_env_states(states, 0)
    built = {name: car_config(name) for name in CARS}
    w = np.array([x for _, x in OPPONENT_MIX], dtype=float)
    opp = rng.choice(len(OPPONENT_MIX), size=3 * n, p=w / w.sum())
    configs, j = [], 0
    for k in sizes:
        for slot in range(6):
            if k <= slot < 2 * k:
                configs.append(built[OPPONENT_MIX[opp[j]][0]]); j += 1
            else:
                configs.append(built['HYBRID'])
    sim.set_car_configs(configs, 0)
    sim.reset_to_kickoff(args.seed, 0)
    state = views.host_struct_array(sim, 'env_states')

    table = action_table()
    zero = len(table)
    t24 = np.zeros((zero + 1, 24), dtype=np.uint8)
    t24[:zero].view(np.float32)[:, :5] = table[:, :5]
    t24[:zero, 20:23] = table[:, 5:8]
    t24 = t24.view(np.uint64)
    present = np.array([slot < 2 * k for k in sizes for slot in range(6)])
    held = np.full(n * 6, zero, dtype=np.intp)
    buf = np.zeros((TICK_SKIP, n * 6, 6), dtype=np.float32)
    age = rng.integers(0, EPISODE_STEPS, size=n)
    resets = np.zeros(n, dtype=np.int64)
    commands = np.zeros((n, 2), dtype=np.int64)
    env_ids = np.arange(n, dtype=np.int64)
    for _ in range(-(-args.ticks // (n * TICK_SKIP))):
        u = rng.random(n * 6)
        act = np.where(u < GROUND_FRAC, rng.integers(0, 24, size=n * 6), rng.integers(24, zero, size=n * 6))
        delays = rng.integers(0, TICK_SKIP, size=n * 6)
        idx = np.where(delays <= np.arange(TICK_SKIP)[:, None], np.where(present, act, zero), held)
        np.take(t24, idx, axis=0, out=buf.view(np.uint8).reshape(TICK_SKIP, n * 6, 24).view(np.uint64))
        held = idx[TICK_SKIP - 1]
        sim.step_with_device_controls(buf.ctypes.data, TICK_SKIP, n * 6, 0)
        age += 1
        done = (state['blue_score'] > 0) | (state['orange_score'] > 0) | (age >= EPISODE_STEPS)
        resets += done
        commands[:, 0] = done.astype(np.int64) * 65793
        commands[:, 1] = (args.seed + env_ids * 1000003 + resets * 9176) & 9223372036854775807
        sim.apply_env_commands_device(commands.ctypes.data, n, 0)
        age[done] = 0
        held.reshape(n, 6)[done] = zero


if __name__ == '__main__':
    main()
