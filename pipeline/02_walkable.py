# -*- coding: utf-8 -*-
"""Step 02 - extract a connected walkable floor surface + obstacle/clearance grid.

Usage:
    python pipeline/02_walkable.py <aligned.pcd> <out_dir> [--no-walls]
                                   [--service-points service_points.csv]

service_points.csv (optional): columns x,y[,label]   (metres, aligned frame)

Optional criteria.yaml block (defaults shown):
    obstacle:
      min_height_m: 0.20          # ignore things lower than this above floor
      max_height_m: 1.80          # ignore ceiling / things above this
      min_points_per_cell: 3

Outputs:
    walkable.npz   (mask, reach, floor_z, slope, mins, cell,
                    blocked, obstacle, traversable, clearance, service,
                    service_labels)
    walkable.pcd
    walkable_summary.json
    surface_cells.csv
    context_cells.csv

Grid meaning (all shaped [ni, nj]):
    reach       walkable surface connected to the seed floor
    valid       cells that look like floor (before connectivity filtering).
                valid & ~reach = floor-like but unreachable (e.g. raised
                platform above max_step_m) -> grey in the effort heatmap
    height      each cell's own surface height (NaN where not valid);
                floor_z is nearest-filled, so use height for unreachable cells
    mask        reach + small gap filling (used for meshing)
    blocked     something solid between floor+min_h and floor+max_h (wall/furniture)
    obstacle    blocked AND inside the walkable surface (object standing on floor)
    traversable reach AND NOT blocked  -> the cells a person can actually stand on
    clearance   distance (m) from each traversable cell to the nearest
                non-traversable cell. Used for door width / corridor width
                and for the clearance penalty in the cost field.
    service     int id of service points (0 = none), names in service_labels
"""

import sys
import csv
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.ndimage import (
    binary_closing,
    binary_opening,
    distance_transform_edt,
)
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from common import load_criteria, write_pcd


def shift_slices(n, d):
    if d >= 0:
        return slice(d, n), slice(0, n - d)
    return slice(0, n + d), slice(-d, n)


def load_point_cloud(path):
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        sys.exit("ERROR: Cannot read points from {}".format(path))

    pts = np.asarray(cloud.points, dtype=np.float64)
    if len(pts) < 50:
        sys.exit("ERROR: Too few points in input PCD.")

    return cloud, pts


def estimate_normals(cloud):
    print("Estimating point normals...", flush=True)
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.20,
            max_nn=30,
        )
    )

    normals = np.asarray(cloud.normals, dtype=np.float64).copy()

    # Orient normals upward where possible.
    flip = normals[:, 2] < 0
    normals[flip] *= -1.0

    return normals


def nearest_fill(values, valid):
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)

    if not valid.any():
        return np.zeros_like(values)

    indices = distance_transform_edt(
        ~valid,
        return_distances=False,
        return_indices=True,
    )
    return values[tuple(indices)]


def reachable_cells(height, valid, seeds, max_step, radius):
    """Keep valid cells connected to at least one seed."""
    ni, nj = valid.shape
    ids = -np.ones((ni, nj), dtype=np.int64)
    nv = int(valid.sum())

    if nv == 0:
        return np.zeros_like(valid, dtype=bool)

    ids[valid] = np.arange(nv)
    rows, cols = [], []

    for dx in range(0, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy <= 0:
                continue

            di, si = shift_slices(ni, dx)
            dj, sj = shift_slices(nj, dy)

            limit = max_step * max(abs(dx), abs(dy))

            ok = (
                valid[di, dj]
                & valid[si, sj]
                & (
                    np.abs(height[di, dj] - height[si, sj])
                    <= limit
                )
            )

            if np.any(ok):
                rows.append(ids[si, sj][ok])
                cols.append(ids[di, dj][ok])

    if rows:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        graph = coo_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)),
            shape=(nv, nv),
        )
    else:
        graph = coo_matrix((nv, nv), dtype=np.int8)

    _, labels = connected_components(graph, directed=False)
    seed_ids = ids[seeds & valid]

    if seed_ids.size == 0:
        return np.zeros_like(valid, dtype=bool)

    seed_labels = np.unique(labels[seed_ids])
    reach = np.zeros_like(valid, dtype=bool)
    reach[valid] = np.isin(labels, seed_labels)

    return reach


def compute_blocked(pts, floor_z, ni, nj, mins, cell,
                    min_h, max_h, min_points):
    """Cells that contain solid points between floor+min_h and floor+max_h.

    Walls, furniture, columns, counters... all end up here. A person
    cannot stand in these cells.
    """
    ij = np.floor((pts[:, :2] - mins) / cell).astype(np.int64)
    inside = (
        (ij[:, 0] >= 0) & (ij[:, 0] < ni)
        & (ij[:, 1] >= 0) & (ij[:, 1] < nj)
    )
    ii = ij[inside, 0]
    jj = ij[inside, 1]
    dz = pts[inside, 2] - floor_z[ii, jj]

    above = (dz >= min_h) & (dz <= max_h)
    occ = np.bincount(
        ii[above] * nj + jj[above], minlength=ni * nj
    ).reshape(ni, nj)

    return occ >= min_points


def load_service_points(path, mins, cell, shape):
    """Read service points (x,y[,label]) into an int grid. 0 = none."""
    grid = np.zeros(shape, dtype=np.int16)
    labels = []

    if path is None:
        return grid, labels

    path = Path(path)
    if not path.is_file():
        print("WARNING: service points file not found: {}".format(path))
        return grid, labels

    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                x = float(row["x"])
                y = float(row["y"])
            except (KeyError, TypeError, ValueError):
                continue

            i = int(np.floor((x - mins[0]) / cell))
            j = int(np.floor((y - mins[1]) / cell))

            if 0 <= i < shape[0] and 0 <= j < shape[1]:
                labels.append(
                    (row.get("label") or "").strip()
                    or "service_{}".format(len(labels) + 1)
                )
                grid[i, j] = len(labels)
            else:
                print("WARNING: service point outside grid: ({}, {})".format(x, y))

    return grid, labels


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def write_surface_csv(path, mask, floor_z, slope, mins, cell):
    ii, jj = np.nonzero(mask)
    xs = mins[0] + (ii + 0.5) * cell
    ys = mins[1] + (jj + 0.5) * cell

    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("x,y,z,slope\n")
        for x, y, z, s in zip(
            xs, ys, floor_z[ii, jj], slope[ii, jj]
        ):
            f.write(
                "{:.3f},{:.3f},{:.3f},{:.2f}\n".format(
                    x, y, z, s
                )
            )


def main(aligned_path, out_dir, with_walls=True, service_points=None):
    cfg = load_criteria()
    w = cfg["walkable"]
    obs_cfg = cfg.get("obstacle", {}) or {}

    configured_cell = float(cfg["cell_size_m"])

    if configured_cell <= 0:
        raise ValueError("cell_size_m must be greater than 0.")

    cell = configured_cell

    min_height = float(w["min_height_m"])
    max_height = float(w["max_height_m"])
    normal_min_z = float(w["normal_min_z"])
    max_var = float(cfg["align"]["max_surface_variation"])

    obs_min_h = float(obs_cfg.get("min_height_m", 0.20))
    obs_max_h = float(obs_cfg.get("max_height_m", 1.80))
    obs_min_pts = int(obs_cfg.get("min_points_per_cell", 3))

    aligned_path = Path(aligned_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cloud, pts = load_point_cloud(aligned_path)
    normals = estimate_normals(cloud)

    n_all = len(pts)
    z = pts[:, 2]

    normal_ok = (
        np.isfinite(normals).all(axis=1)
        & (normals[:, 2] >= normal_min_z)
    )

    print("Checking local surface variation...", flush=True)
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    variation = np.ones(n_all, dtype=np.float64)
    k = min(12, n_all)
    batch_size = 10000

    for start in range(0, n_all, batch_size):
        end = min(start + batch_size, n_all)
        _, indices = tree.query(pts[start:end], k=k)

        if k == 1:
            indices = indices[:, None]

        local = normals[indices]
        mean_n = local.mean(axis=1)
        length = np.linalg.norm(mean_n, axis=1)
        mean_n /= np.maximum(length[:, None], 1e-12)

        dot = np.abs(
            np.sum(normals[start:end] * mean_n, axis=1)
        )
        variation[start:end] = np.clip(1.0 - dot, 0.0, 1.0)

    smooth_ok = (
        normal_ok
        & np.isfinite(variation)
        & (variation <= max_var)
    )

    in_band = (z >= min_height) & (z <= max_height)
    candidate = smooth_ok & in_band

    print(
        "Input points: {:,} | upward: {:,} | candidates: {:,}".format(
            n_all, int(normal_ok.sum()), int(candidate.sum())
        ),
        flush=True,
    )

    if candidate.sum() < 50:
        sys.exit(
            "ERROR: Too few floor candidates. Check the point cloud "
            "alignment and criteria.yaml."
        )

    mins = pts[:, :2].min(axis=0) - cell
    maxs = pts[:, :2].max(axis=0) + cell
    ni, nj = np.ceil((maxs - mins) / cell).astype(int) + 1

    candidate_indices = np.flatnonzero(candidate)
    xy = pts[candidate, :2]
    zc = z[candidate]
    nc = normals[candidate]

    ij = np.floor((xy - mins) / cell).astype(np.int64)
    cell_id = ij[:, 0] * nj + ij[:, 1]
    total_cells = ni * nj

    count = np.bincount(
        cell_id, minlength=total_cells
    ).astype(np.int64)

    z_order = np.lexsort((zc, cell_id))
    sorted_cells = cell_id[z_order]
    sorted_z = zc[z_order]

    unique_cells, starts, counts = np.unique(
        sorted_cells, return_index=True, return_counts=True
    )

    cell_height = np.full(total_cells, np.nan, dtype=np.float64)
    cell_slope = np.full(total_cells, np.nan, dtype=np.float64)

    # Low percentile helps avoid letting a few high points dominate.
    for cid, start, n in zip(unique_cells, starts, counts):
        values = sorted_z[start:start + n]
        cell_height[cid] = np.percentile(values, 20)

    # Estimate slope from averaged upward normals per grid cell.
    sum_nx = np.bincount(
        cell_id, weights=nc[:, 0], minlength=total_cells
    )
    sum_ny = np.bincount(
        cell_id, weights=nc[:, 1], minlength=total_cells
    )
    sum_nz = np.bincount(
        cell_id, weights=nc[:, 2], minlength=total_cells
    )

    norm_mag = np.sqrt(sum_nx**2 + sum_ny**2 + sum_nz**2)
    cos_slope = np.clip(
        sum_nz / np.maximum(norm_mag, 1e-12), 0.0, 1.0
    )
    cell_slope[:] = np.degrees(np.arccos(cos_slope))

    min_count = max(3, int(w["min_points_per_cell"]))
    valid = (
        (count >= min_count)
        & np.isfinite(cell_height)
        & (cell_height >= min_height)
        & (cell_height <= max_height)
    ).reshape(ni, nj)

    height = cell_height.reshape(ni, nj)
    slope = cell_slope.reshape(ni, nj)

    seed_band = float(w["seed_band_m"])
    seeds = valid & (np.abs(height) <= seed_band)
    notes = [
        "Grid resolution used: {:.2f} m.".format(cell),
        "Surface candidates use upward normals and local variation.",
        "Cell height is the 20th percentile of candidate Z values.",
        "Surface mask is a candidate map, not a semantic floor classifier.",
        "Obstacles = solid points between floor+{:.2f} m and floor+{:.2f} m.".format(
            obs_min_h, obs_max_h
        ),
    ]

    if not valid.any():
        sys.exit(
            "ERROR: No valid surface cells. Lower min_points_per_cell "
            "or verify the point cloud."
        )

    if not seeds.any():
        lowest_z = np.nanmin(np.where(valid, height, np.nan))
        seeds = valid & (height <= lowest_z + seed_band)
        notes.append(
            "No seed near Z=0; used the lowest populated cells."
        )

    reach = reachable_cells(
        height,
        valid,
        seeds,
        float(w["max_step_m"]),
        max(1, int(w["link_radius_cells"])),
    )

    if not reach.any():
        sys.exit("ERROR: No connected surface cells were found.")

    mask = binary_closing(reach, iterations=1) | reach
    mask = binary_opening(mask, iterations=1) | reach

    floor_z = nearest_fill(np.where(reach, height, 0.0), reach)
    slope_fill = nearest_fill(np.where(reach, slope, 0.0), reach)

    # ------------------------------------------------------------------
    # NEW: obstacle / traversable / clearance / service grids
    # ------------------------------------------------------------------
    print("Computing obstacle and clearance grids...", flush=True)
    blocked = compute_blocked(
        pts, floor_z, ni, nj, mins, cell,
        obs_min_h, obs_max_h, obs_min_pts,
    )
    obstacle = blocked & mask
    traversable = reach & ~blocked

    # Distance (m) to the nearest non-traversable cell.
    clearance = distance_transform_edt(traversable) * cell

    service, service_labels = load_service_points(
        service_points, mins, cell, (ni, nj)
    )

    reachable_flat = reach.reshape(-1)
    point_cell_reachable = reachable_flat[cell_id]
    walk_pts_sel = np.zeros(n_all, dtype=bool)
    walk_pts_sel[candidate_indices[point_cell_reachable]] = True

    if not walk_pts_sel.any():
        sys.exit("ERROR: Surface grid found but no points were selected.")

    np.savez_compressed(
        str(out_dir / "walkable.npz"),
        mask=mask,
        reach=reach,
        valid=valid,
        height=height,
        floor_z=floor_z,
        slope=slope_fill,
        mins=mins,
        cell=np.asarray(cell),
        blocked=blocked,
        obstacle=obstacle,
        traversable=traversable,
        clearance=clearance,
        service=service,
        service_labels=np.asarray(service_labels, dtype=str),
    )

    write_pcd(out_dir / "walkable.pcd", pts[walk_pts_sel])

    write_surface_csv(
        out_dir / "surface_cells.csv",
        mask,
        floor_z,
        slope_fill,
        mins,
        cell,
    )

    context_path = out_dir / "context_cells.csv"
    wall_count = 0

    if with_walls:
        try:
            from walls import wall_cells, write_wall_csv

            wall_cfg = cfg["walls"]
            cells = wall_cells(
                pts,
                mask,
                floor_z,
                mins,
                cell,
                size=float(wall_cfg["size_m"]),
                h_min=float(wall_cfg["min_height_m"]),
                h_max=float(wall_cfg["max_height_m"]),
                min_bins=int(wall_cfg["min_bins"]),
                max_dist=float(wall_cfg["max_distance_m"]),
            )
            write_wall_csv(context_path, cells)
            wall_count = len(cells)
        except ImportError:
            notes.append(
                "walls.py not found; wall context generation skipped."
            )
            context_path.write_text(
                "x,y,z,height\n", encoding="utf-8"
            )
    else:
        context_path.write_text(
            "x,y,z,height\n", encoding="utf-8"
        )

    walk_pts = pts[walk_pts_sel]
    n_reach = int(reach.sum())
    n_mask = int(mask.sum())
    n_trav = int(traversable.sum())

    save_json(out_dir / "walkable_summary.json", {
        "mode": "coarse_height_map_reachability",
        "configured_cell_size_m": configured_cell,
        "effective_cell_size_m": cell,
        "criteria": {k: float(v) for k, v in w.items()},
        "obstacle_criteria": {
            "min_height_m": obs_min_h,
            "max_height_m": obs_max_h,
            "min_points_per_cell": obs_min_pts,
        },
        "counts": {
            "input_points": int(n_all),
            "upward_normal_points": int(normal_ok.sum()),
            "candidate_points": int(candidate.sum()),
            "walkable_points": int(walk_pts_sel.sum()),
        },
        "cells": {
            "valid_candidate_cells": int(valid.sum()),
            "seed_cells": int(seeds.sum()),
            "reachable_cells": n_reach,
            "walkable_cells": n_mask,
            "obstacle_cells": int(obstacle.sum()),
            "traversable_cells": n_trav,
            "service_points": len(service_labels),
            "wall_cells": int(wall_count),
        },
        "walkable_area_m2": float(n_mask * cell * cell),
        "traversable_area_m2": float(n_trav * cell * cell),
        "walkable_percentage": round(
            100.0 * float(walk_pts_sel.sum()) / max(n_all, 1), 2
        ),
        "bounds": {
            "min_xyz": walk_pts.min(axis=0).tolist(),
            "max_xyz": walk_pts.max(axis=0).tolist(),
        },
        "notes": notes,
    })

    print(
        "Grid: {} x {} | cell: {:.2f} m".format(ni, nj, cell),
        flush=True,
    )
    print(
        "Valid cells: {:,} | seeds: {:,} | reachable: {:,}".format(
            int(valid.sum()), int(seeds.sum()), n_reach
        ),
        flush=True,
    )
    print(
        "Walkable cells: {:,} | obstacle cells: {:,} | traversable: {:,}".format(
            n_mask, int(obstacle.sum()), n_trav
        ),
        flush=True,
    )
    print(
        "Walkable area: {:.2f} m2 | traversable area: {:.2f} m2".format(
            n_mask * cell * cell, n_trav * cell * cell
        ),
        flush=True,
    )
    print("Service points: {}".format(len(service_labels)), flush=True)
    print("Saved walkable.npz, walkable.pcd, surface_cells.csv", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("aligned", help="Aligned input .pcd")
    parser.add_argument("out_dir")
    parser.add_argument("--no-walls", action="store_true")
    parser.add_argument(
        "--service-points",
        default=None,
        help="Optional CSV with columns x,y[,label]",
    )
    args = parser.parse_args()

    main(
        args.aligned,
        args.out_dir,
        with_walls=not args.no_walls,
        service_points=args.service_points,
    )