
# -*- coding: utf-8 -*-
"""Shared helper functions for the accessibility-capstone pipeline."""

import json
from pathlib import Path

import numpy as np
import yaml


def load_criteria(path="config/criteria.yaml"):
    """Load project criteria from YAML."""
    path = Path(path)

    if not path.is_absolute() and not path.exists():
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / path

    if not path.exists():
        raise FileNotFoundError(f"Criteria file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def save_json(path, data):
    """Save data as a UTF-8 JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def nearest_fill(values, valid):
    """Fill invalid grid cells using the nearest valid cell."""
    from scipy.ndimage import distance_transform_edt

    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)

    if values.shape != valid.shape:
        raise ValueError("values and valid must have the same shape")

    if not np.any(valid):
        return np.zeros_like(values, dtype=np.float64)

    indices = distance_transform_edt(
        ~valid,
        return_distances=False,
        return_indices=True,
    )

    return values[tuple(indices)]


def write_pcd(path, points, colors=None, normals=None):
    """Write an Nx3 point array or Open3D point cloud to a PCD file."""
    import open3d as o3d

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(points, o3d.geometry.PointCloud):
        cloud = points
    else:
        points = np.asarray(points, dtype=np.float64)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"Expected points with shape (N, 3), got {points.shape}"
            )

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            colors = np.asarray(colors, dtype=np.float64)
            if colors.shape == points.shape:
                cloud.colors = o3d.utility.Vector3dVector(
                    np.clip(colors, 0.0, 1.0)
                )

        if normals is not None:
            normals = np.asarray(normals, dtype=np.float64)
            if normals.shape == points.shape:
                cloud.normals = o3d.utility.Vector3dVector(normals)

    success = o3d.io.write_point_cloud(str(path), cloud)

    if not success:
        raise IOError(f"Failed to write point cloud: {path}")

    return path
