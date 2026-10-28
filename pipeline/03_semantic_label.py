# -*- coding: utf-8 -*-
"""
Step 03 - Semantic labeling guided by the walkable surface pipeline.

Required inputs:
    aligned.pcd
    walkable.pcd
    walkable.npz        (from the updated 02_walkable.py)

Outputs:
    semantic.npz            <- NEW: grid-level semantics used by the cost field
    semantic_labeled.pcd    per-point colours
    semantic_labels.csv     per-point labels
    semantic_summary.json

semantic.npz fields (all [ni, nj] grids aligned with walkable.npz):
    label        uint8   0 none, 1 floor, 2 ramp, 3 stairs,
                         4 obstacle, 5 door, 6 unknown (too steep)
    room_id      int32   0 = not walkable, 1..N = connected open region
    region_type  uint8   0 none, 1 room, 2 corridor
    mins, cell, label_names, label_codes

Example:
    python pipeline/03_semantic_label.py aligned.pcd walkable.pcd walkable.npz --out-dir <run_dir>

Important:
    Geometry-based heuristic. This is not AI semantic segmentation.
    Door / room / corridor detection is limited by the grid resolution:
    with a 0.5 m cell a door can only be resolved to about +-0.5 m.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy import ndimage as ndi
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    maximum_filter,
    minimum_filter,
)
from scipy.spatial import cKDTree


# Grid-level codes (stored in semantic.npz)
CODE = {
    "none": 0,
    "floor": 1,
    "ramp": 2,
    "stairs": 3,
    "obstacle": 4,
    "door": 5,
    "unknown": 6,
    "unreachable": 7,   # floor-like surface that is not connected (e.g. raised platform)
}
CODE_TO_NAME = {v: k for k, v in CODE.items()}

# Point-level labels
LABELS = [
    "floor",
    "wall",
    "ramp",
    "stairs",
    "door",
    "obstacle",
    "overhead",
    "unknown",
    "unreachable",
]

POINT_CODE = {name: i + 1 for i, name in enumerate(LABELS)}

COLORS = {
    "floor":    [0.10, 0.80, 0.20],  # Green
    "wall":     [0.10, 0.35, 0.95],  # Blue
    "ramp":     [1.00, 0.65, 0.05],  # Orange
    "stairs":   [0.95, 0.15, 0.15],  # Red
    "door":     [0.10, 0.85, 0.90],  # Cyan
    "obstacle": [0.85, 0.10, 0.85],  # Purple
    "overhead": [0.80, 0.80, 0.55],  # Khaki (roof / canopy above walkway)
    "unknown":  [0.55, 0.55, 0.55],  # Gray
    "unreachable": [0.65, 0.40, 0.15],  # Brown (floor-like but not reachable)
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Semantic labeling guided by walkable surface results."
    )

    parser.add_argument("aligned", help="Full aligned point cloud (.pcd)")
    parser.add_argument("walkable", help="02_walkable.py output walkable.pcd")
    parser.add_argument("grid", help="02_walkable.py output walkable.npz")

    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--voxel", type=float, default=0.05)
    parser.add_argument("--normal-radius", type=float, default=0.20)
    parser.add_argument("--floor-tolerance", type=float, default=0.18)
    parser.add_argument("--wall-normal-z", type=float, default=0.30)

    # surface type
    parser.add_argument("--ramp-min-deg", type=float, default=8.0,
                        help="Normal slope above this = ramp candidate")
    parser.add_argument("--ramp-max-deg", type=float, default=20.0,
                        help="Normal slope above this = unknown/too steep")
    parser.add_argument("--stairs-grad-deg", type=float, default=25.0,
                        help="Flat treads (low normal slope) but height "
                             "gradient above this = stairs")

    parser.add_argument("--min-surface-cells", type=int, default=6,
                        help="Ramp/stairs clusters smaller than this many "
                             "cells are treated as noise and become floor")

    # obstacles (points)
    parser.add_argument("--obstacle-min-height", type=float, default=0.30)
    parser.add_argument("--obstacle-max-height", type=float, default=1.80)

    # rooms / doors / corridors
    parser.add_argument("--core-clearance", type=float, default=None,
                        help="Clearance (m) for a cell to count as open "
                             "space. Default: 1.5 x cell size")
    parser.add_argument("--min-core-cells", type=int, default=4)
    parser.add_argument("--corridor-max-width", type=float, default=2.5)
    parser.add_argument("--detect-doors", action="store_true",
                        help="Enable door detection (unreliable when the "
                             "grid cell is larger than ~0.3 m)")
    parser.add_argument("--door-search", type=float, default=1.0,
                        help="Radius (m) used to find narrow cells that "
                             "touch two different regions")

    return parser.parse_args()


def read_cloud(path):
    path = Path(path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    cloud = o3d.io.read_point_cloud(str(path))

    if cloud.is_empty():
        raise ValueError(f"Point cloud is empty or unreadable: {path}")

    points = np.asarray(cloud.points, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1)

    return path, points, valid


def load_walkable_grid(path):
    path = Path(path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Walkable grid not found: {path}")

    with np.load(path) as data:
        required = {"reach", "floor_z", "slope", "mins", "cell"}
        missing = required - set(data.files)

        if missing:
            raise ValueError(
                f"{path} is missing required fields: {sorted(missing)}"
            )

        reach = np.asarray(data["reach"], dtype=bool)
        floor_z = np.asarray(data["floor_z"], dtype=np.float64)
        slope = np.asarray(data["slope"], dtype=np.float64)
        mins = np.asarray(data["mins"], dtype=np.float64)
        cell = float(np.asarray(data["cell"]).item())

        # New fields from the updated 02_walkable.py (fall back if absent)
        if "blocked" in data.files:
            blocked = np.asarray(data["blocked"], dtype=bool)
        else:
            print("WARNING: walkable.npz has no 'blocked'. "
                  "Re-run the updated 02_walkable.py.")
            blocked = np.zeros_like(reach)

        if "valid" in data.files:
            valid = np.asarray(data["valid"], dtype=bool)
        else:
            print("WARNING: walkable.npz has no 'valid'. "
                  "Re-run the updated 02_walkable.py.")
            valid = reach.copy()

        if "height" in data.files:
            height = np.asarray(data["height"], dtype=np.float64)
        else:
            height = np.where(reach, floor_z, np.nan)

        if "traversable" in data.files:
            traversable = np.asarray(data["traversable"], dtype=bool)
        else:
            traversable = reach & ~blocked

        if "clearance" in data.files:
            clearance = np.asarray(data["clearance"], dtype=np.float64)
        else:
            clearance = distance_transform_edt(traversable) * cell

    if cell <= 0:
        raise ValueError("Grid cell size must be positive.")

    if reach.shape != floor_z.shape or reach.shape != slope.shape:
        raise ValueError("reach, floor_z and slope have inconsistent shapes.")

    return {
        "reach": reach,
        "floor_z": floor_z,
        "slope": slope,
        "mins": mins,
        "cell": cell,
        "blocked": blocked,
        "traversable": traversable,
        "clearance": clearance,
        "valid": valid,
        "height": height,
    }


# ----------------------------------------------------------------------
# Grid-level semantics (NEW)
# ----------------------------------------------------------------------
def segment_regions(traversable, clearance, cell, core_clearance,
                    min_core_cells, corridor_max_width):
    """Split traversable cells into rooms / corridors.

    1. 'Core' = cells far from walls (clearance >= core_clearance).
       Doorways are narrow, so they drop out of the core and the cores
       of two rooms separate.
    2. Each connected core gets an id, then grows back over all
       traversable cells (nearest core wins).
    3. A region is a corridor if it is narrow and long, else a room.
    """
    shape = traversable.shape
    room_id = np.zeros(shape, dtype=np.int32)
    region_type = np.zeros(shape, dtype=np.uint8)

    core = traversable & (clearance >= core_clearance)
    core_lab, n = ndi.label(core)

    if n == 0:
        return room_id, region_type, 0

    sizes = np.bincount(core_lab.ravel(), minlength=n + 1)
    keep = sizes >= min_core_cells
    keep[0] = False
    n_keep = int(keep.sum())

    if n_keep == 0:
        return room_id, region_type, 0

    remap = np.zeros(n + 1, dtype=np.int32)
    remap[keep] = np.arange(1, n_keep + 1)
    core_lab = remap[core_lab]

    idx = distance_transform_edt(
        core_lab == 0, return_distances=False, return_indices=True
    )
    grown = core_lab[tuple(idx)]
    room_id = np.where(traversable, grown, 0).astype(np.int32)

    for rid, sl in enumerate(ndi.find_objects(room_id), start=1):
        if sl is None:
            continue

        m = room_id[sl] == rid
        if not m.any():
            continue

        extents = [(s.stop - s.start) * cell for s in sl]
        width = 2.0 * float(clearance[sl][m].max())
        length = max(extents)

        is_corridor = (
            width <= corridor_max_width
            and length >= 3.0 * max(width, cell)
        )
        region_type[sl][m] = 2 if is_corridor else 1

    return room_id, region_type, n_keep


def detect_doors(traversable, clearance, room_id, cell,
                 core_clearance, search_m):
    """Narrow traversable cells that sit between two different regions."""
    narrow = traversable & (clearance < core_clearance)

    radius = max(1, int(np.ceil(search_m / cell)))
    size = 2 * radius + 1

    big = np.iinfo(np.int32).max
    rid_max = maximum_filter(room_id, size=size)
    rid_min = minimum_filter(np.where(room_id > 0, room_id, big), size=size)

    return (
        narrow
        & (rid_max > 0)
        & (rid_min < big)
        & (rid_max != rid_min)
    )


def drop_small(mask, min_cells):
    """Remove 8-connected clusters smaller than min_cells."""
    lab, n = ndi.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = sizes >= min_cells
    keep[0] = False
    return keep[lab]


def build_grid_semantics(grid, args):
    reach = grid["reach"]
    floor_z = grid["floor_z"]
    slope = grid["slope"]
    cell = grid["cell"]
    blocked = grid["blocked"]
    traversable = grid["traversable"]
    clearance = grid["clearance"]

    core_clearance = (
        args.core_clearance
        if args.core_clearance is not None
        else 1.5 * cell
    )

    label = np.zeros(reach.shape, dtype=np.uint8)
    label[reach] = CODE["floor"]
    unreachable = grid["valid"] & ~reach
    label[unreachable] = CODE["unreachable"]

    # Ramp vs stairs:
    #   ramp   : the surface normal itself is tilted
    #   stairs : treads are flat (small normal slope) but the height
    #            keeps changing from cell to cell
    interior = binary_erosion(reach, iterations=1)

    # Median-smooth to suppress per-cell scan noise before thresholding.
    slope = ndi.median_filter(slope, size=3)
    floor_z = ndi.median_filter(floor_z, size=3)
    gy, gx = np.gradient(floor_z, cell)
    grad_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

    ramp = interior & (slope > args.ramp_min_deg) & (slope <= args.ramp_max_deg)
    steep = reach & (slope > args.ramp_max_deg)
    stairs = (
        interior
        & (slope <= args.ramp_min_deg)
        & (grad_deg >= args.stairs_grad_deg)
    )

    ramp = drop_small(ramp, args.min_surface_cells)
    stairs = drop_small(stairs, args.min_surface_cells)

    label[ramp] = CODE["ramp"]
    label[steep] = CODE["unknown"]
    label[stairs] = CODE["stairs"]
    label[blocked & reach] = CODE["obstacle"]

    room_id, region_type, n_regions = segment_regions(
        traversable,
        clearance,
        cell,
        core_clearance,
        args.min_core_cells,
        args.corridor_max_width,
    )

    if args.detect_doors:
        doors = detect_doors(
            traversable, clearance, room_id, cell,
            core_clearance, args.door_search,
        )
    else:
        doors = np.zeros_like(traversable, dtype=bool)
    label[doors & ~stairs] = CODE["door"]

    stats = {
        "regions": int(n_regions),
        "rooms": int(len(np.unique(room_id[region_type == 1]))),
        "corridors": int(len(np.unique(room_id[region_type == 2]))),
        "door_cells": int(doors.sum()),
        "stairs_cells": int(stairs.sum()),
        "ramp_cells": int(ramp.sum()),
        "obstacle_cells": int((label == CODE["obstacle"]).sum()),
        "unreachable_cells": int(unreachable.sum()),
        "core_clearance_m": float(core_clearance),
    }

    return label, room_id, region_type, stats


# ----------------------------------------------------------------------
# Point-level labels
# ----------------------------------------------------------------------
def estimate_sample_normals(points, voxel, radius):
    """Estimate normals on a downsampled cloud to reduce computation."""
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(points)

    sampled = source.voxel_down_sample(voxel)

    if len(sampled.points) < 10:
        raise ValueError(
            "Too few points after downsampling. Try a smaller --voxel."
        )

    print(f"Normal-estimation samples: {len(sampled.points):,}")
    print("Estimating normals...")

    sampled.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=40,
        )
    )

    sample_points = np.asarray(sampled.points)
    normals = np.asarray(sampled.normals)

    return sample_points, normals


def classify_points(points, grid, label_grid, sample_points, sample_normals,
                    floor_tolerance, wall_normal_z, obs_min_h, obs_max_h):
    """
    Label points.

    1. Points near the walkable surface take the label of their grid cell
       (floor / ramp / stairs / door). Floor under an obstacle stays floor.
    2. Points standing above a reachable cell (obs_min_h..obs_max_h) are
       obstacles.
    3. Remaining near-vertical points are walls.
    """
    n = len(points)
    labels = np.full(n, "unknown", dtype=object)

    reach = grid["reach"]
    floor_z = grid["floor_z"]
    slope = grid["slope"]
    mins = grid["mins"]
    cell = grid["cell"]

    ij = np.floor((points[:, :2] - mins[:2]) / cell).astype(np.int64)

    inside = (
        (ij[:, 0] >= 0)
        & (ij[:, 0] < reach.shape[0])
        & (ij[:, 1] >= 0)
        & (ij[:, 1] < reach.shape[1])
    )

    rows = np.flatnonzero(inside)
    ii = ij[inside, 0]
    jj = ij[inside, 1]

    reachable_cell = reach[ii, jj]
    estimated_z = floor_z[ii, jj]
    cell_slope = slope[ii, jj]

    dz = points[rows, 2] - estimated_z
    finite = np.isfinite(estimated_z) & np.isfinite(cell_slope)

    near_surface = reachable_cell & finite & (np.abs(dz) <= floor_tolerance)

    # grid label -> point label
    codes = label_grid[ii, jj].astype(np.int64)
    codes = np.where(codes == CODE["obstacle"], CODE["floor"], codes)

    names = np.array(
        [CODE_TO_NAME.get(c, "unknown") for c in range(max(CODE.values()) + 1)],
        dtype=object,
    )
    labels[rows[near_surface]] = names[codes[near_surface]]

    above = (
        reachable_cell
        & finite
        & ~near_surface
        & (dz >= obs_min_h)
        & (dz <= obs_max_h)
    )
    # Floor-like surface that exists but is not connected to the main floor
    # (raised platform, isolated slab, ...). Uses the cell's own height.
    own_h = grid["height"][ii, jj]
    unreachable_pts = (
        grid["valid"][ii, jj]
        & ~reachable_cell
        & np.isfinite(own_h)
        & (np.abs(points[rows, 2] - own_h) <= floor_tolerance)
        & (labels[rows] == "unknown")
    )
    labels[rows[unreachable_pts]] = "unreachable"

    # Solid points in cells on the rim of the walkable area are most
    # likely wall bases / railings; interior ones are real obstacles.
    interior_cell = binary_erosion(reach, iterations=1)[ii, jj]
    labels[rows[above & interior_cell]] = "obstacle"
    labels[rows[above & ~interior_cell]] = "wall"

    # Points far above the nearest floor level: candidates for roof / canopy.
    # floor_z is nearest-filled, so this also covers roofs that extend
    # past the edge of the walkable area.
    over_reach = np.zeros(n, dtype=bool)
    over_reach[rows[finite & (dz > obs_max_h)]] = True

    remaining = labels == "unknown"

    if np.any(remaining):
        print("Mapping nearby normals to remaining points...")
        normal_tree = cKDTree(sample_points)
        remaining_indices = np.flatnonzero(remaining)

        batch_size = 100000

        for start in range(0, len(remaining_indices), batch_size):
            batch_indices = remaining_indices[start:start + batch_size]
            _, nearest = normal_tree.query(
                points[batch_indices],
                k=1,
                workers=-1,
            )

            nz = np.abs(sample_normals[nearest, 2])
            near_vertical = nz <= wall_normal_z

            labels[batch_indices[near_vertical]] = "wall"

            roof = (~near_vertical) & over_reach[batch_indices]
            labels[batch_indices[roof]] = "overhead"

            print(
                "  Checked normals: {:,} / {:,}".format(
                    min(start + batch_size, len(remaining_indices)),
                    len(remaining_indices),
                )
            )

    # ---- why are points still 'unknown'? ---------------------------
    unk = labels == "unknown"
    unk_in_grid = unk[rows]
    diag = {
        "unknown_total": int(unk.sum()),
        "unknown_outside_grid": int(unk.sum() - unk_in_grid.sum()),
        "unknown_in_non_reach_cell": int(
            (unk_in_grid & ~reachable_cell).sum()
        ),
        "unknown_in_reach_cell_below_floor": int(
            (unk_in_grid & reachable_cell & (dz < -floor_tolerance)).sum()
        ),
        "unknown_in_reach_cell_above_floor": int(
            (unk_in_grid & reachable_cell & (dz > floor_tolerance)).sum()
        ),
        "unknown_in_steep_reach_cell": int(
            (unk_in_grid & reachable_cell & (np.abs(dz) <= floor_tolerance)).sum()
        ),
    }

    return labels, diag


def main():
    args = parse_args()

    if args.voxel <= 0:
        raise ValueError("--voxel must be positive.")

    if args.normal_radius <= 0:
        raise ValueError("--normal-radius must be positive.")

    if args.floor_tolerance <= 0:
        raise ValueError("--floor-tolerance must be positive.")

    if not 0.0 <= args.wall_normal_z <= 1.0:
        raise ValueError("--wall-normal-z must be between 0 and 1.")

    if args.ramp_min_deg >= args.ramp_max_deg:
        raise ValueError("--ramp-min-deg must be below --ramp-max-deg.")

    aligned_path, all_points, valid_mask = read_cloud(args.aligned)
    walkable_path, walkable_points, walkable_valid = read_cloud(args.walkable)
    grid = load_walkable_grid(args.grid)

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else aligned_path.parent
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_indices = np.flatnonzero(valid_mask)
    valid_points = all_points[valid_mask]

    if len(valid_points) == 0:
        raise ValueError("Aligned point cloud contains no valid points.")

    walkable_points = walkable_points[walkable_valid]

    if len(walkable_points) == 0:
        raise ValueError("walkable.pcd contains no valid points.")

    print(f"Aligned points:  {len(all_points):,}")
    print(f"Valid points:    {len(valid_points):,}")
    print(f"Walkable points: {len(walkable_points):,}")
    print(f"Grid cell size:  {grid['cell']:.3f} m")
    print(f"Floor tolerance: {args.floor_tolerance:.3f} m")

    # ---- grid semantics -------------------------------------------------
    print("Building grid semantics (floor/ramp/stairs/door/room)...")
    label_grid, room_id, region_type, grid_stats = build_grid_semantics(
        grid, args
    )

    semantic_npz = out_dir / "semantic.npz"
    np.savez_compressed(
        str(semantic_npz),
        label=label_grid,
        room_id=room_id,
        region_type=region_type,
        mins=grid["mins"],
        cell=np.asarray(grid["cell"]),
        label_names=np.array(list(CODE.keys()), dtype=str),
        label_codes=np.array(list(CODE.values()), dtype=np.int32),
    )

    print(
        "Regions: {regions} (rooms {rooms}, corridors {corridors}) | "
        "unreachable cells: {unreachable_cells} | "
        "door cells: {door_cells} | stairs: {stairs_cells} | "
        "ramp: {ramp_cells} | obstacle: {obstacle_cells}".format(**grid_stats)
    )

    # ---- point semantics ------------------------------------------------
    sample_points, sample_normals = estimate_sample_normals(
        valid_points,
        args.voxel,
        args.normal_radius,
    )

    print("Classifying points from walkable grid and normals...")

    valid_labels, unknown_diag = classify_points(
        valid_points,
        grid,
        label_grid,
        sample_points,
        sample_normals,
        args.floor_tolerance,
        args.wall_normal_z,
        args.obstacle_min_height,
        args.obstacle_max_height,
    )

    all_labels = np.full(len(all_points), "unknown", dtype=object)
    all_labels[valid_indices] = valid_labels

    output_cloud = o3d.geometry.PointCloud()
    output_cloud.points = o3d.utility.Vector3dVector(valid_points)

    colors = np.empty((len(valid_points), 3), dtype=np.float64)

    for label, color in COLORS.items():
        colors[valid_labels == label] = color

    output_cloud.colors = o3d.utility.Vector3dVector(colors)

    labeled_path = out_dir / "semantic_labeled.pcd"
    if not o3d.io.write_point_cloud(str(labeled_path), output_cloud):
        raise RuntimeError(f"Failed to write {labeled_path}")

    csv_path = out_dir / "semantic_labels.csv"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["point_index", "x", "y", "z", "label", "label_code"])

        for original_index, point, label in zip(
            valid_indices, valid_points, valid_labels
        ):
            writer.writerow([
                int(original_index),
                float(point[0]),
                float(point[1]),
                float(point[2]),
                label,
                POINT_CODE.get(label, 0),
            ])

    counts = {
        label: int(np.count_nonzero(valid_labels == label))
        for label in LABELS
    }

    summary = {
        "input_aligned": str(aligned_path),
        "input_walkable": str(walkable_path),
        "input_grid": str(Path(args.grid).resolve()),
        "total_points": int(len(all_points)),
        "valid_points": int(len(valid_points)),
        "walkable_input_points": int(len(walkable_points)),
        "sampled_points": int(len(sample_points)),
        "grid_cell_size_m": float(grid["cell"]),
        "floor_tolerance_m": float(args.floor_tolerance),
        "normal_radius_m": float(args.normal_radius),
        "voxel_size_m": float(args.voxel),
        "method": "walkable_grid_height_clearance_regions_and_sampled_normals",
        "label_counts": counts,
        "grid_semantics": grid_stats,
        "unknown_breakdown": unknown_diag,
        "colors_rgb": COLORS,
        "warning": (
            "Geometry-based heuristic only. Floor labels depend on "
            "02_walkable.py. Wall labels may include vertical sides of "
            "objects. Obstacle labels are candidates, not verified objects. "
            "Stairs/door/room/corridor detection is approximate and limited "
            "by grid resolution."
        ),
    }

    summary_path = out_dir / "semantic_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nLabel counts:")
    for label in LABELS:
        count = counts[label]
        percent = 100.0 * count / max(len(valid_points), 1)
        print(f"  {label:10s}: {count:>10,} ({percent:6.2f}%)")

    print("\nWhy points are still 'unknown':")
    for key, value in unknown_diag.items():
        print(f"  {key:36s}: {value:>10,}")

    print("\nSaved files:")
    print(f"  {semantic_npz}")
    print(f"  {labeled_path}")
    print(f"  {csv_path}")
    print(f"  {summary_path}")

    if args.view:
        print("Opening 3D viewer...")
        o3d.visualization.draw_geometries(
            [output_cloud],
            window_name="Semantic Point Cloud",
            width=1400,
            height=900,
        )


if __name__ == "__main__":
    main()