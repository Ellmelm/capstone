# -*- coding: utf-8 -*-
"""Align a point cloud to a horizontal floor plane.

Input:
    01_align.py <input.pcd> <output_directory>

Outputs:
    aligned.pcd
    alignment_transform.npy
    alignment_inverse_transform.npy
    alignment_info.json

Floor selection:
    The floor is the LOWEST horizontal plane among the planes that are
    big enough compared with the largest horizontal plane. This avoids
    picking a small stray/noise plane (scattered points, ground outside
    the building) just because it happens to be lower.
"""

import sys
import json
import numpy as np
import open3d as o3d

from pathlib import Path


VOXEL_SIZE_M = 0.05
OUTLIER_NEIGHBORS = 20
OUTLIER_STD_RATIO = 2.0

PLANE_DISTANCE_M = 0.05
PLANE_RANSAC_N = 3
PLANE_ITERATIONS = 2000

# Walls are found (and removed) before the floor in many scans, so allow
# more planes than before.
MAX_PLANES = 20
MIN_PLANE_POINTS = 3000
MIN_POINTS_TO_SEARCH = 1000
HORIZONTAL_NORMAL_Z_MIN = 0.85

# A horizontal plane can be the floor only if it has at least this
# fraction of the points of the largest horizontal plane.
FLOOR_MIN_FRACTION = 0.4


def rot_a_to_b(a, b):
    """Return a 3x3 rotation matrix mapping vector a onto b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na < 1e-12 or nb < 1e-12:
        raise ValueError("Cannot align a zero-length vector.")

    a = a / na
    b = b / nb

    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    s = np.linalg.norm(v)

    if s < 1e-8:
        if c > 0:
            return np.eye(3)

        # 180-degree rotation around X.
        return np.diag([1.0, -1.0, -1.0])

    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])

    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def make_transform(rotation, translation):
    """Create a homogeneous 4x4 transform: p_out = R*p_in + t."""
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def main(src, out_dir):
    src = Path(src).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        sys.exit("Input point cloud does not exist: {}".format(src))

    print("Loading point cloud:", src, flush=True)

    pcd = o3d.io.read_point_cloud(str(src))

    if pcd.is_empty():
        sys.exit(
            "Open3D could not read any points from: {}".format(src)
        )

    original_count = len(pcd.points)
    print("Original points: {:,}".format(original_count), flush=True)

    # Reduce point density before plane detection.
    pcd = pcd.voxel_down_sample(VOXEL_SIZE_M)

    if len(pcd.points) < MIN_POINTS_TO_SEARCH:
        sys.exit(
            "Too few points after downsampling: {:,}".format(
                len(pcd.points)
            )
        )

    # Remove isolated points.
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=OUTLIER_NEIGHBORS,
        std_ratio=OUTLIER_STD_RATIO
    )

    if len(pcd.points) < MIN_POINTS_TO_SEARCH:
        sys.exit(
            "Too few points after outlier removal: {:,}".format(
                len(pcd.points)
            )
        )

    print(
        "After downsampling/outlier removal: {:,}".format(
            len(pcd.points)
        ),
        flush=True
    )

    # Search for horizontal planes. Keep the source point indices
    # in the current downsampled cloud.
    rest = pcd
    candidates = []

    for i in range(MAX_PLANES):
        if len(rest.points) < MIN_POINTS_TO_SEARCH:
            break

        plane, inliers = rest.segment_plane(
            distance_threshold=PLANE_DISTANCE_M,
            ransac_n=PLANE_RANSAC_N,
            num_iterations=PLANE_ITERATIONS
        )

        if len(inliers) == 0:
            break

        normal = np.asarray(plane[:3], dtype=float)
        normal_length = np.linalg.norm(normal)

        if normal_length < 1e-12:
            rest = rest.select_by_index(inliers, invert=True)
            continue

        normal /= normal_length

        inlier_points = np.asarray(rest.points)[inliers]
        mean_z = float(np.mean(inlier_points[:, 2]))

        print(
            "Plane {}: {:,} points, normal={}, mean z={:.3f} m".format(
                i,
                len(inliers),
                np.round(normal, 3).tolist(),
                mean_z
            ),
            flush=True
        )

        if (
            abs(normal[2]) >= HORIZONTAL_NORMAL_Z_MIN
            and len(inliers) >= MIN_PLANE_POINTS
        ):
            candidates.append({
                "mean_z": mean_z,
                "normal": normal.copy(),
                "points": inlier_points.copy(),
                "count": len(inliers),
            })

        rest = rest.select_by_index(inliers, invert=True)

    if not candidates:
        sys.exit(
            "No sufficiently large horizontal plane found. "
            "Check the input cloud and units."
        )

    # Floor = lowest horizontal plane that is big enough relative to the
    # largest horizontal plane (ignores small stray planes below it).
    largest = max(item["count"] for item in candidates)
    eligible = [
        item for item in candidates
        if item["count"] >= FLOOR_MIN_FRACTION * largest
    ]
    eligible.sort(key=lambda item: item["mean_z"])
    floor = eligible[0]

    print(
        "Horizontal candidates (z, points):",
        [
            (round(item["mean_z"], 3), item["count"])
            for item in sorted(candidates, key=lambda c: c["mean_z"])
        ],
        flush=True
    )
    print(
        "Selected floor: z={:.3f} m, {:,} points "
        "(largest horizontal plane: {:,})".format(
            floor["mean_z"], floor["count"], largest
        ),
        flush=True
    )

    normal = floor["normal"].copy()

    # Point the floor normal upward before rotating to +Z.
    if normal[2] < 0:
        normal = -normal

    rotation = rot_a_to_b(normal, np.array([0.0, 0.0, 1.0]))

    # Compute floor elevation after rotation.
    rotated_floor = floor["points"] @ rotation.T
    z_floor = float(np.median(rotated_floor[:, 2]))

    translation = np.array([0.0, 0.0, -z_floor])

    # Homogeneous transform:
    # p_aligned = rotation @ p_original + translation
    forward = make_transform(rotation, translation)
    inverse = np.linalg.inv(forward)

    # Apply the exact same transformation to the complete filtered cloud.
    pcd.rotate(rotation, center=(0.0, 0.0, 0.0))
    pcd.translate(translation)

    aligned_path = out_dir / "aligned.pcd"
    if not o3d.io.write_point_cloud(str(aligned_path), pcd):
        sys.exit("Failed to write: {}".format(aligned_path))

    forward_path = out_dir / "alignment_transform.npy"
    inverse_path = out_dir / "alignment_inverse_transform.npy"

    np.save(str(forward_path), forward)
    np.save(str(inverse_path), inverse)

    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()

    info = {
        "input": str(src),
        "original_point_count": int(original_count),
        "aligned_point_count": int(len(pcd.points)),
        "floor_mean_z_before_alignment_m": float(floor["mean_z"]),
        "floor_z_after_rotation_m": z_floor,
        "floor_plane_point_count": int(floor["count"]),
        "floor_min_fraction": FLOOR_MIN_FRACTION,
        "horizontal_candidates": [
            {
                "mean_z_m": float(item["mean_z"]),
                "point_count": int(item["count"]),
            }
            for item in sorted(candidates, key=lambda c: c["mean_z"])
        ],
        "voxel_size_m": VOXEL_SIZE_M,
        "aligned_extent_m": [float(v) for v in extent],
        "transform_convention": (
            "p_aligned = alignment_transform @ [x,y,z,1]"
        ),
        "inverse_transform_convention": (
            "p_original = alignment_inverse_transform @ [x,y,z,1]"
        ),
    }

    info_path = out_dir / "alignment_info.json"
    with open(str(info_path), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("Saved:", aligned_path, flush=True)
    print("Saved:", forward_path, flush=True)
    print("Saved:", inverse_path, flush=True)
    print("Saved:", info_path, flush=True)
    print("Aligned extent (m):", np.round(extent, 3), flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(
            "Usage: python 01_align.py <input.pcd> <output_directory>"
        )

    main(sys.argv[1], sys.argv[2])