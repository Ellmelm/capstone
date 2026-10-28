# -*- coding: utf-8 -*-
"""Step 05 - Build a per-cell movement cost field for a user profile.

Usage:
    python pipeline/05_cost_field.py <walkable.npz> <semantic.npz> <out_dir>
                                     [--profile wheelchair|elderly|general]

Output:
    cost_<profile>.npz
    cost_<profile>_summary.json

cost semantics:
    cost >= 1.0   multiplier on distance when stepping onto the cell
                  (1.0 = easy flat ground, larger = harder)
    cost == inf   the cell cannot be used by this profile
                  (stairs for wheelchair, too steep, too narrow, obstacle...)

This step is STATIC: it does not depend on the start/goal point, so it is
run once per scan and profile. Start/goal are handled by 06_route_effort.py.

All thresholds are heuristics. Slope comes from point normals and is noisy;
clearance is measured from the cell centre to the nearest blocked cell, so
a value of 0.5 m means roughly 1 m of free width. Tune with the --slope-*
and --clear-* options and verify against the applicable standard.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

# semantic.npz label codes (see 03_semantic_label.py)
FLOOR, RAMP, STAIRS, OBSTACLE, DOOR, UNKNOWN, UNREACHABLE = 1, 2, 3, 4, 5, 6, 7

PROFILES = {
    "wheelchair": dict(
        slope_comfort=3.0, slope_hard=6.0,
        clear_hard=0.40, clear_soft=0.90,
        ramp_pen=1.0, stairs_pen=None, door_pen=0.5,
        w_slope=4.0, w_clear=3.0,
    ),
    "elderly": dict(
        slope_comfort=5.0, slope_hard=10.0,
        clear_hard=0.30, clear_soft=0.60,
        ramp_pen=0.5, stairs_pen=6.0, door_pen=0.3,
        w_slope=3.0, w_clear=2.0,
    ),
    "general": dict(
        slope_comfort=8.0, slope_hard=20.0,
        clear_hard=0.25, clear_soft=0.50,
        ramp_pen=0.2, stairs_pen=2.0, door_pen=0.1,
        w_slope=2.0, w_clear=1.0,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="Build movement cost field.")
    p.add_argument("walkable", help="walkable.npz from 02_walkable.py")
    p.add_argument("semantic", help="semantic.npz from 03_semantic_label.py")
    p.add_argument("out_dir")
    p.add_argument("--profile", choices=sorted(PROFILES), default="general")
    p.add_argument("--slope-comfort", type=float, default=None)
    p.add_argument("--slope-hard", type=float, default=None)
    p.add_argument("--clear-hard", type=float, default=None)
    p.add_argument("--clear-soft", type=float, default=None)
    return p.parse_args()


def load_npz(path, required):
    path = Path(path).resolve()
    if not path.is_file():
        sys.exit("ERROR: file not found: {}".format(path))
    with np.load(path) as d:
        missing = [k for k in required if k not in d.files]
        if missing:
            sys.exit("ERROR: {} is missing {}. Re-run the earlier steps."
                     .format(path.name, missing))
        return {k: d[k] for k in d.files}


def main():
    args = parse_args()

    prm = dict(PROFILES[args.profile])
    for key, val in (("slope_comfort", args.slope_comfort),
                     ("slope_hard", args.slope_hard),
                     ("clear_hard", args.clear_hard),
                     ("clear_soft", args.clear_soft)):
        if val is not None:
            prm[key] = float(val)

    if prm["slope_hard"] <= prm["slope_comfort"]:
        sys.exit("ERROR: slope hard limit must exceed comfort slope.")
    if prm["clear_soft"] <= prm["clear_hard"]:
        sys.exit("ERROR: clear soft must exceed clear hard.")

    w = load_npz(args.walkable, ["reach", "valid", "slope", "clearance",
                                 "traversable", "blocked", "floor_z",
                                 "height", "mins", "cell"])
    s = load_npz(args.semantic, ["label", "room_id"])

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cell = float(np.asarray(w["cell"]).item())
    label = s["label"].astype(np.int64)
    if label.shape != w["reach"].shape:
        sys.exit("ERROR: semantic.npz and walkable.npz grid sizes differ.")

    slope = ndi.median_filter(w["slope"].astype(np.float64), size=3)
    clearance = w["clearance"].astype(np.float64)

    # ---- hard limits (cell unusable) ---------------------------------
    impassable = ~w["traversable"].astype(bool)
    impassable |= slope > prm["slope_hard"]
    impassable |= clearance < prm["clear_hard"]
    impassable |= np.isin(label, [OBSTACLE, UNKNOWN, UNREACHABLE])
    if prm["stairs_pen"] is None:
        impassable |= label == STAIRS

    # ---- soft penalties (cell usable but harder) ---------------------
    span = prm["slope_hard"] - prm["slope_comfort"]
    pen_slope = np.where(
        slope > prm["slope_comfort"],
        prm["w_slope"] * ((slope - prm["slope_comfort"]) / span) ** 2,
        0.0,
    )

    span = prm["clear_soft"] - prm["clear_hard"]
    pen_clear = np.where(
        clearance < prm["clear_soft"],
        prm["w_clear"] * (prm["clear_soft"] - clearance) / span,
        0.0,
    )

    pen_sem = np.zeros_like(slope)
    pen_sem[label == RAMP] += prm["ramp_pen"]
    pen_sem[label == DOOR] += prm["door_pen"]
    if prm["stairs_pen"] is not None:
        pen_sem[label == STAIRS] += prm["stairs_pen"]

    passable = ~impassable
    cost = 1.0 + pen_slope + pen_clear + pen_sem
    cost[~passable] = np.inf

    surface = w["valid"].astype(bool) | w["reach"].astype(bool)
    surface_z = np.where(np.isfinite(w["height"]), w["height"], w["floor_z"])

    out_npz = out_dir / "cost_{}.npz".format(args.profile)
    np.savez_compressed(
        str(out_npz),
        cost=cost,
        passable=passable,
        surface=surface,
        blocked=w["blocked"].astype(bool),
        reach=w["reach"].astype(bool),
        valid=w["valid"].astype(bool),
        surface_z=surface_z,
        label=label.astype(np.uint8),
        room_id=s["room_id"],
        mins=w["mins"],
        cell=np.asarray(cell),
        profile=np.asarray(args.profile),
    )

    reach = w["reach"].astype(bool)
    finite = cost[passable]
    summary = {
        "profile": args.profile,
        "parameters": prm,
        "cell_size_m": cell,
        "cells": {
            "reach": int(reach.sum()),
            "passable": int(passable.sum()),
            "reach_but_unusable_for_profile": int((reach & ~passable).sum()),
            "unreachable_surface": int((surface & ~reach).sum()),
        },
        "cost_stats": {
            "min": float(finite.min()) if finite.size else None,
            "mean": float(finite.mean()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
        },
        "note": "Heuristic cost field. Thresholds are not a substitute "
                "for an accessibility standard or on-site inspection.",
    }
    with (out_dir / "cost_{}_summary.json".format(args.profile)).open(
            "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Profile: {}".format(args.profile))
    print("Reach cells: {:,} | passable for profile: {:,} | "
          "reach but unusable: {:,}".format(
              int(reach.sum()), int(passable.sum()),
              int((reach & ~passable).sum())))
    if finite.size:
        print("Cost min/mean/max: {:.2f} / {:.2f} / {:.2f}".format(
            finite.min(), finite.mean(), finite.max()))
    print("Saved {}".format(out_npz))


if __name__ == "__main__":
    main()