
# -*- coding: utf-8 -*-
"""Step 03 - calculate accessibility scores.

Usage:
    python pipeline/03_score.py <aligned.pcd> <walkable.npz> <out_dir>

Outputs:
    scores.npz
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.ndimage import binary_dilation, distance_transform_edt, maximum_filter

from common import load_criteria


def main(aligned_path, walkable_path, out_dir):
    cfg = load_criteria()
    measure_cfg = cfg["measure"]
    score_cfg = cfg["score"]

    aligned_path = Path(aligned_path).resolve()
    walkable_path = Path(walkable_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not aligned_path.is_file():
        sys.exit("ERROR: aligned.pcd not found: {}".format(aligned_path))

    if not walkable_path.is_file():
        sys.exit("ERROR: walkable.npz not found: {}".format(walkable_path))

    cloud = o3d.io.read_point_cloud(str(aligned_path))
    if cloud.is_empty():
        sys.exit("ERROR: Cannot read points from aligned.pcd.")

    pts = np.asarray(cloud.points, dtype=np.float64)

    with np.load(str(walkable_path)) as data:
        mask = np.asarray(data["mask"], dtype=bool)
        floor_z = np.asarray(data["floor_z"], dtype=np.float64)
        slope = np.asarray(data["slope"], dtype=np.float64)
        mins = np.asarray(data["mins"], dtype=np.float64).reshape(-1)
        cell = float(np.asarray(data["cell"]).reshape(-1)[0])

    if not mask.any():
        sys.exit("ERROR: No walkable cells found in walkable.npz.")

    ni, nj = mask.shape

    # Map every point to its grid cell.
    ij = np.floor((pts[:, :2] - mins) / cell).astype(np.int64)

    inside = (
        (ij[:, 0] >= 0) & (ij[:, 0] < ni)
        & (ij[:, 1] >= 0) & (ij[:, 1] < nj)
    )

    ij = np.clip(ij, [0, 0], [ni - 1, nj - 1])

    # Estimate obstacles above the local walkable floor.
    rel_height = pts[:, 2] - floor_z[ij[:, 0], ij[:, 1]]

    obstacle_hit = (
        inside
        & (rel_height > float(measure_cfg["obstacle_z_min_m"]))
        & (rel_height < float(measure_cfg["obstacle_z_max_m"]))
    )

    obstacle = np.zeros((ni, nj), dtype=bool)
    obstacle[ij[obstacle_hit, 0], ij[obstacle_hit, 1]] = True
    obstacle = binary_dilation(obstacle, iterations=1)

    width_cap = float(measure_cfg["width_cap_m"])

    if obstacle.any():
        clearance = distance_transform_edt(~obstacle) * cell

        window = max(
            1,
            int(round(float(measure_cfg["spread_window_m"]) / cell)),
        )
        if window % 2 == 0:
            window += 1

        width = 2.0 * maximum_filter(clearance, size=window)
    else:
        # No obstacles detected: cap the estimate at configured maximum.
        width = np.full((ni, nj), width_cap, dtype=np.float64)

    width = np.minimum(width, width_cap)

    # Convert slope and width into scores from 0 to 100.
    slope_ok = float(score_cfg["slope_ok_deg"])
    slope_fail = float(score_cfg["slope_fail_deg"])
    width_ok = float(score_cfg["width_ok_m"])
    width_fail = float(score_cfg["width_fail_m"])

    if slope_fail <= slope_ok:
        sys.exit("ERROR: slope_fail_deg must be greater than slope_ok_deg.")

    if width_ok <= width_fail:
        sys.exit("ERROR: width_ok_m must be greater than width_fail_m.")

    slope_score = np.clip(
        (slope_fail - slope) / (slope_fail - slope_ok),
        0.0,
        1.0,
    ) * 100.0

    width_score = np.clip(
        (width - width_fail) / (width_ok - width_fail),
        0.0,
        1.0,
    ) * 100.0

    score = np.minimum(slope_score, width_score)

    valid = mask & np.isfinite(score)
    score = np.where(valid, score, np.nan)
    slope_score = np.where(valid, slope_score, np.nan)
    width_score = np.where(valid, width_score, np.nan)

    np.savez_compressed(
        str(out_dir / "scores.npz"),
        score=score,
        score_raw=score,
        slope_score=slope_score,
        width_score=width_score,
        mask=valid,
        mins=mins,
        cell=np.asarray(cell),
        floor_z=floor_z,
        slope=slope,
        width=width,
    )

    area = float(valid.sum() * cell * cell)

    print("=" * 50, flush=True)
    print("ACCESSIBILITY SCORE", flush=True)
    print("Walkable area: {:.2f} m2".format(area), flush=True)
    print("Valid cells: {:,}".format(int(valid.sum())), flush=True)

    if valid.any():
        print(
            "Mean score: {:.1f}/100".format(float(np.nanmean(score[valid]))),
            flush=True,
        )
        print(
            "Median slope: {:.2f} degrees".format(
                float(np.median(slope[valid]))
            ),
            flush=True,
        )
        print(
            "Median estimated width: {:.2f} m".format(
                float(np.median(width[valid]))
            ),
            flush=True,
        )

    print("Saved: {}".format(out_dir / "scores.npz"), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(
            "Usage: python 03_score.py "
            "<aligned.pcd> <walkable.npz> <out_dir>"
        )

    main(*sys.argv[1:4])
