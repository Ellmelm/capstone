
# # -*- coding: utf-8 -*-
# """Estimate walkable area, floor elevation, slope and clearance width.

# Usage:
#   python pipeline/05_measure.py aligned.pcd walkable.pcd slope.npy output_dir
# """

# import sys
# from pathlib import Path

# import numpy as np
# import open3d as o3d
# from scipy.ndimage import (
#     distance_transform_edt,
#     maximum_filter,
#     binary_closing,
#     binary_dilation,
# )

# from common import load_criteria


# def read_points(path, label):
#     """Read a point cloud and return finite XYZ points."""
#     path = Path(path).resolve()

#     if not path.is_file():
#         raise FileNotFoundError("{} not found: {}".format(label, path))

#     cloud = o3d.io.read_point_cloud(str(path))
#     points = np.asarray(cloud.points)

#     if points.size == 0:
#         raise ValueError("{} contains no points".format(label))

#     points = points[np.isfinite(points).all(axis=1)]

#     if len(points) == 0:
#         raise ValueError("{} contains no finite points".format(label))

#     return points


# def main(aligned, walkable, slope_file, out_dir):
#     cfg = load_criteria()
#     cell = float(cfg["cell_size_m"])
#     measure = cfg["measure"]

#     if cell <= 0:
#         raise ValueError("cell_size_m must be greater than zero")

#     out_dir = Path(out_dir).resolve()
#     out_dir.mkdir(parents=True, exist_ok=True)

#     full = read_points(aligned, "aligned point cloud")
#     walk = read_points(walkable, "walkable point cloud")

#     slope_path = Path(slope_file).resolve()
#     if not slope_path.is_file():
#         raise FileNotFoundError("Slope file not found: {}".format(slope_path))

#     slope = np.asarray(np.load(str(slope_path)), dtype=float).reshape(-1)

#     if len(slope) != len(walk):
#         raise ValueError(
#             "slope.npy has {} values but walkable.pcd has {} points. "
#             "Regenerate 04_segment.py output before running this step."
#             .format(len(slope), len(walk))
#         )

#     if not np.isfinite(slope).all():
#         raise ValueError("slope.npy contains NaN or infinite values")

#     # Create a grid around the aligned cloud.
#     mins = full[:, :2].min(axis=0) - cell
#     maxs = full[:, :2].max(axis=0) + cell
#     shape = tuple(
#         int(v) for v in (np.ceil((maxs - mins) / cell).astype(int) + 1)
#     )

#     if min(shape) < 2:
#         raise ValueError("Point cloud is too small to create a measurement grid")

#     def to_cell(points):
#         ij = np.floor((points[:, :2] - mins) / cell).astype(int)
#         return ij[:, 0], ij[:, 1]

#     # 1. Grid the walkable points.
#     wi, wj = to_cell(walk)
#     count = np.zeros(shape, dtype=np.float64)
#     sum_z = np.zeros(shape, dtype=np.float64)
#     sum_slope = np.zeros(shape, dtype=np.float64)

#     np.add.at(count, (wi, wj), 1)
#     np.add.at(sum_z, (wi, wj), walk[:, 2])
#     np.add.at(sum_slope, (wi, wj), slope)

#     has_floor = count > 0

#     if not has_floor.any():
#         raise ValueError("No walkable grid cells were generated")

#     # Fill only small holes in the walkable mask.
#     mask = binary_closing(has_floor, iterations=2)

#     # Use the nearest observed floor cell to fill elevation and slope.
#     nearest = distance_transform_edt(
#         ~has_floor,
#         return_distances=False,
#         return_indices=True,
#     )

#     floor_observed = np.zeros(shape, dtype=np.float64)
#     slope_observed = np.zeros(shape, dtype=np.float64)

#     floor_observed[has_floor] = (
#         sum_z[has_floor] / count[has_floor]
#     )
#     slope_observed[has_floor] = (
#         sum_slope[has_floor] / count[has_floor]
#     )

#     floor_fill = floor_observed[nearest[0], nearest[1]]
#     slope_fill = slope_observed[nearest[0], nearest[1]]

#     # 2. Estimate obstacle cells from points above the nearest floor.
#     fi, fj = to_cell(full)
#     relative_height = full[:, 2] - floor_fill[fi, fj]

#     z_min = float(measure["obstacle_z_min_m"])
#     z_max = float(measure["obstacle_z_max_m"])

#     if z_min < 0 or z_max <= z_min:
#         raise ValueError(
#             "Invalid obstacle height range in criteria.yaml"
#         )

#     selected = (
#         (relative_height > z_min)
#         & (relative_height < z_max)
#     )

#     obstacle = np.zeros(shape, dtype=bool)
#     obstacle[fi[selected], fj[selected]] = True

#     # Expand slightly to account for sparse point-cloud sampling.
#     obstacle = binary_dilation(obstacle, iterations=1)

#     # 3. Estimate clearance width.
#     # This is an approximation, not a guaranteed clear passage width.
#     clearance = distance_transform_edt(~obstacle) * cell

#     spread_m = float(measure["spread_window_m"])
#     width_cap = float(measure["width_cap_m"])

#     if spread_m <= 0 or width_cap <= 0:
#         raise ValueError(
#             "spread_window_m and width_cap_m must be greater than zero"
#         )

#     window = max(1, int(round(spread_m / cell)))
#     if window % 2 == 0:
#         window += 1

#     width = 2.0 * maximum_filter(clearance, size=window)
#     width = np.minimum(width, width_cap)

#     # Only cells in the walkable mask are evaluated downstream.
#     floor_fill[~mask] = 0.0
#     slope_fill[~mask] = 0.0
#     width[~mask] = 0.0

#     output_path = out_dir / "measure.npz"

#     np.savez(
#         str(output_path),
#         mask=mask,
#         floor_z=floor_fill,
#         slope=slope_fill,
#         width=width,
#         mins=mins,
#         cell=cell,
#     )

#     measured_widths = width[mask]

#     print("Walkable grid cells: {:,}".format(int(mask.sum())))
#     print("Approx. walkable area: {:.2f} m2".format(mask.sum() * cell**2))
#     print(
#         "Approx. width (m): min {:.2f}, 5th percentile {:.2f}, "
#         "median {:.2f}".format(
#             measured_widths.min(),
#             np.percentile(measured_widths, 5),
#             np.median(measured_widths),
#         )
#     )
#     print("Saved:", output_path)


# if __name__ == "__main__":
#     if len(sys.argv) != 5:
#         sys.exit(
#             "Usage: python 05_measure.py "
#             "<aligned.pcd> <walkable.pcd> <slope.npy> <out_dir>"
#         )

#     try:
#         main(*sys.argv[1:5])
#     except (KeyError, ValueError, FileNotFoundError, OSError) as exc:
#         sys.exit("ERROR: {}".format(exc))
