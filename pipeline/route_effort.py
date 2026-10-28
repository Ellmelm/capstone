# -*- coding: utf-8 -*-
"""Step 06 - Effort heatmap from a chosen start point (and optional goal).

Usage:
    python pipeline/06_route_effort.py <cost_<profile>.npz> <out_dir>
            --start X Y [--goal X Y] [--low 30 --high 60 | --percentile]

X, Y are metres in the aligned point-cloud frame.

This step is light: re-run it whenever the start/goal changes. Steps
01-05 do not need to be repeated.

Effort = shortest-path cost from the start (Dijkstra, 8 neighbours).
One unit equals walking 1 m on easy flat ground, so effort 40 means
"as tiring as ~40 m of easy walking" for the chosen profile.

Colours:
    green  : effort <= low
    yellow : low < effort <= high
    red    : effort > high
    grey   : surface exists but cannot be reached (raised platform,
             disconnected area, or unusable for this profile)

Outputs:
    effort.npz
    effort_cells.csv        x,y,z,effort,category,r,g,b
    effort_heatmap.pcd      coloured cell centres (open in CloudCompare)
    effort_path.csv         only when --goal is given
    effort_summary.json
"""

import sys
import json
import heapq
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

CAT_NONE, CAT_LOW, CAT_MED, CAT_HIGH, CAT_UNREACH = 0, 1, 2, 3, 4
RGB = {
    CAT_LOW: (0, 200, 0),
    CAT_MED: (255, 215, 0),
    CAT_HIGH: (220, 0, 0),
    CAT_UNREACH: (128, 128, 128),
}
NAMES = {CAT_LOW: "low", CAT_MED: "medium",
         CAT_HIGH: "high", CAT_UNREACH: "unreachable"}

SQ2 = 2.0 ** 0.5
NBR = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
       (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2)]


def parse_args():
    p = argparse.ArgumentParser(description="Effort heatmap from a start point.")
    p.add_argument("cost", help="cost_<profile>.npz from 05_cost_field.py")
    p.add_argument("out_dir")
    p.add_argument("--start", type=float, nargs=2, required=True,
                   metavar=("X", "Y"))
    p.add_argument("--goal", type=float, nargs=2, default=None,
                   metavar=("X", "Y"))
    p.add_argument("--low", type=float, default=30.0,
                   help="effort at or below this = green (absolute mode)")
    p.add_argument("--high", type=float, default=60.0,
                   help="effort above this = red (absolute mode)")
    p.add_argument("--percentile", action="store_true",
                   help="use 33/66 percentiles of reachable cells instead")
    p.add_argument("--snap-max", type=float, default=3.0,
                   help="max distance (m) to snap a point to a passable cell")
    return p.parse_args()


def snap(point, passable, mins, cell, snap_max, name):
    ni, nj = passable.shape
    i = int(np.floor((point[0] - mins[0]) / cell))
    j = int(np.floor((point[1] - mins[1]) / cell))
    if not (0 <= i < ni and 0 <= j < nj):
        sys.exit("ERROR: {} point is outside the grid.".format(name))
    if passable[i, j]:
        return (i, j), 0.0
    dist, ind = distance_transform_edt(
        ~passable, return_distances=True, return_indices=True)
    d = float(dist[i, j]) * cell
    if d > snap_max:
        sys.exit("ERROR: {} point is {:.1f} m from the nearest usable cell "
                 "(limit {:.1f} m).".format(name, d, snap_max))
    return (int(ind[0, i, j]), int(ind[1, i, j])), d


def dijkstra(cost, passable, cell, start):
    ni, nj = cost.shape
    cost_l = cost.tolist()
    pass_l = passable.tolist()
    eff = [[float("inf")] * nj for _ in range(ni)]
    parent = {}
    eff[start[0]][start[1]] = 0.0
    pq = [(0.0, start[0], start[1])]

    while pq:
        d, r, c = heapq.heappop(pq)
        if d > eff[r][c]:
            continue
        for dr, dc, step in NBR:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < ni and 0 <= nc < nj) or not pass_l[nr][nc]:
                continue
            # no corner cutting through blocked cells
            if dr and dc and not (pass_l[r + dr][c] and pass_l[r][c + dc]):
                continue
            nd = d + step * cell * 0.5 * (cost_l[r][c] + cost_l[nr][nc])
            if nd < eff[nr][nc]:
                eff[nr][nc] = nd
                parent[(nr, nc)] = (r, c)
                heapq.heappush(pq, (nd, nr, nc))

    return np.array(eff, dtype=np.float64), parent


def main():
    args = parse_args()
    cost_path = Path(args.cost).resolve()
    if not cost_path.is_file():
        sys.exit("ERROR: file not found: {}".format(cost_path))

    with np.load(cost_path) as d:
        cost = d["cost"]
        passable = d["passable"].astype(bool)
        surface = d["surface"].astype(bool)
        blocked = d["blocked"].astype(bool)
        surface_z = d["surface_z"]
        mins = d["mins"]
        cell = float(np.asarray(d["cell"]).item())
        profile = str(np.asarray(d["profile"]))

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not passable.any():
        sys.exit("ERROR: no usable cells for profile '{}'.".format(profile))

    start, d_start = snap(args.start, passable, mins, cell,
                          args.snap_max, "start")
    print("Start snapped {:.2f} m to cell {}".format(d_start, start))

    effort, parent = dijkstra(cost, passable, cell, start)
    reached = np.isfinite(effort)

    if args.percentile:
        low, high = np.percentile(effort[reached], [33.0, 66.0])
    else:
        low, high = args.low, args.high
    if high <= low:
        sys.exit("ERROR: --high must be greater than --low.")

    # surface that is not blocked by an object but cannot be reached
    grey = surface & ~blocked & ~reached
    category = np.zeros(effort.shape, dtype=np.uint8)
    category[reached & (effort <= low)] = CAT_LOW
    category[reached & (effort > low) & (effort <= high)] = CAT_MED
    category[reached & (effort > high)] = CAT_HIGH
    category[grey] = CAT_UNREACH

    np.savez_compressed(
        str(out_dir / "effort.npz"),
        effort=effort, category=category, start=np.asarray(start),
        mins=mins, cell=np.asarray(cell), profile=np.asarray(profile),
    )

    ii, jj = np.nonzero(category)
    xs = mins[0] + (ii + 0.5) * cell
    ys = mins[1] + (jj + 0.5) * cell
    zs = surface_z[ii, jj]
    cats = category[ii, jj]
    colors = np.array([RGB[c] for c in cats], dtype=np.uint8)

    with (out_dir / "effort_cells.csv").open("w", encoding="utf-8",
                                             newline="") as f:
        f.write("x,y,z,effort,category,r,g,b\n")
        for x, y, z, e, c, rgb in zip(xs, ys, zs, effort[ii, jj], cats, colors):
            f.write("{:.3f},{:.3f},{:.3f},{},{},{},{},{}\n".format(
                x, y, z, "" if not np.isfinite(e) else "{:.2f}".format(e),
                NAMES[int(c)], rgb[0], rgb[1], rgb[2]))

    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.column_stack([xs, ys, zs + 0.05]))
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        o3d.io.write_point_cloud(str(out_dir / "effort_heatmap.pcd"), pcd)
    except ImportError:
        print("open3d not available; effort_heatmap.pcd skipped.")

    summary = {
        "profile": profile,
        "start_requested_xy": list(args.start),
        "start_snapped_cell": [int(start[0]), int(start[1])],
        "start_snap_distance_m": d_start,
        "thresholds": {"low": float(low), "high": float(high),
                       "mode": "percentile" if args.percentile else "absolute"},
        "area_m2": {NAMES[c]: float((category == c).sum() * cell * cell)
                    for c in NAMES},
        "max_effort": float(effort[reached].max()),
    }

    if args.goal is not None:
        goal, d_goal = snap(args.goal, passable, mins, cell,
                            args.snap_max, "goal")
        if not np.isfinite(effort[goal]):
            summary["goal"] = {"reachable": False}
            print("Goal is NOT reachable for profile '{}'.".format(profile))
        else:
            path, node = [goal], goal
            while node != start:
                node = parent[node]
                path.append(node)
            path.reverse()
            length = sum(np.hypot(a[0] - b[0], a[1] - b[1])
                         for a, b in zip(path[:-1], path[1:])) * cell
            with (out_dir / "effort_path.csv").open(
                    "w", encoding="utf-8", newline="") as f:
                f.write("x,y,z\n")
                for (i, j) in path:
                    f.write("{:.3f},{:.3f},{:.3f}\n".format(
                        mins[0] + (i + 0.5) * cell,
                        mins[1] + (j + 0.5) * cell, surface_z[i, j]))
            summary["goal"] = {
                "reachable": True,
                "effort": float(effort[goal]),
                "path_length_m": float(length),
                "effort_per_metre": float(effort[goal] / max(length, 1e-9)),
            }
            print("Goal: effort {:.1f} | path length {:.1f} m".format(
                effort[goal], length))

    with (out_dir / "effort_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Thresholds: low {:.1f}, high {:.1f}".format(low, high))
    for c in NAMES:
        print("  {:12s}: {:8.1f} m2".format(
            NAMES[c], (category == c).sum() * cell * cell))
    print("Saved effort.npz, effort_cells.csv, effort_heatmap.pcd")


if __name__ == "__main__":
    main()