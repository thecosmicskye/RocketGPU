

import glob
import json
import os
import re
import struct
import sys

import numpy as np


def matches(suite):
    for path in sorted(glob.glob(os.path.join(suite, "data", "*.npz"))):
        name = os.path.basename(path)
        m = re.search(r"_(\d)v(\d)", name)
        if not m or m[1] != m[2] or int(m[1]) > 3 or "_scn_" in name:
            continue
        z = np.load(path)
        meta = json.loads(z["meta"].tobytes())
        if meta.get("controls_only"):
            continue
        ctl, k = z["ctl"], int(m[1])
        at = {int(f): i for i, f in enumerate(z["frame"])}
        order = {}
        for seg in meta["segments"]:
            teams = [c[1][0]["value"] for c in meta["cars"][seg["cars"]] if c[0] == "add_car"]
            slots = [s for s, t in enumerate(teams) if t == 0] + [s for s, t in enumerate(teams) if t == 1]
            if len(slots) != 2 * k or sum(t == 0 for t in teams) != k:
                continue
            for j in range(seg["ticks"]):
                order.setdefault(at[seg["start"] + j], slots)
        rows = sorted(order)
        yield name, k, np.stack([ctl[i, order[i]] for i in rows])


def main():
    suite, out = sys.argv[1:3]
    found = list(matches(suite))
    table, inv = np.unique(np.concatenate([c.reshape(-1, 8) for _, _, c in found]), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    with open(out, "wb") as f:
        f.write(b"RGRC" + struct.pack("<II", 1, len(table)) + table.astype("<f4").tobytes())
        f.write(struct.pack("<I", len(found)))
        at = 0
        for name, k, c in found:
            n = c.shape[0] * 2 * k
            f.write(struct.pack("<II", k, c.shape[0]) + inv[at:at + n].astype("<u2").tobytes())
            at += n
    for k in (1, 2, 3):
        ks = [c.shape[0] for _, kk, c in found if kk == k]
        print(f"{k}v{k}: {len(ks)} matches, {sum(ks)} ticks")
    print(f"{len(table)} distinct control rows -> {out}")


if __name__ == "__main__":
    main()
