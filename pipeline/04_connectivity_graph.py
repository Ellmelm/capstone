# -*- coding: utf-8 -*-
"""
Step 04 - Connect points from Step 03 without changing their positions or RGB colors.

Input:
    semantic_labeled.pcd

Outputs:
    connectivity_graph.ply   PLY containing original XYZ/RGB vertices and graph edges
    graph_nodes.csv          Original point coordinates and RGB values
    graph_edges.csv          Edges and their lengths
    connectivity_graph.json  Graph summary and settings
    edge_diagnostics.json    Connectivity diagnostics

Example:
    python pipeline\\04_connectivity_graph.py output\\RUN_ID\\semantic_labeled.pcd
    python pipeline\\04_connectivity_graph.py output\\RUN_ID\\semantic_labeled.pcd --max-distance 0.25 --max-neighbors 6 --view

Notes:
    - No points are filtered by semantic label or color.
    - XYZ and RGB are read from semantic_labeled.pcd and are not recomputed.
    - Each point is connected to up to K nearest points within max-distance.
    - The PLY is written as ASCII PLY with vertex RGB and edge elements.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Connect nearby points in semantic_labeled.pcd while preserving XYZ and RGB."
    )
    parser.add_argument(
        "input_pcd",
        type=Path,
        help="Path to Step 03 semantic_labeled.pcd",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the input PCD's directory.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.25,
        help="Maximum 3D distance in meters for an edge (default: 0.25).",
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=6,
        help="Maximum nearest neighbors per point before edge deduplication (default: 6).",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open an Open3D preview showing original-colored points and gray edges.",
    )
    return parser.parse_args()


def load_point_cloud(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input PCD does not exist: {path}")

    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)

    if points.size == 0:
        raise ValueError(f"No points were loaded from: {path}")

    # Keep RGB as read from the PCD. Do not recolor or classify any point.
    colors = np.asarray(cloud.colors, dtype=np.float64)
    if colors.shape != points.shape:
        print(
            "WARNING: Input PCD has no complete RGB color array. "
            "The script will preserve XYZ, but cannot preserve missing RGB values.",
            file=sys.stderr,
        )
        colors = np.zeros_like(points)

    # Do not remove or reorder points. This keeps original vertex indices stable.
    return cloud, points, colors


def build_edges(points: np.ndarray, max_distance: float, max_neighbors: int):
    """
    Create undirected edges between nearby points.

    A point is connected to its nearest neighbors only when their 3D distance
    is <= max_distance. An edge is stored once as (smaller_index, larger_index).
    No semantic labels or RGB values are used in this function.
    """
    if cKDTree is None:
        raise ImportError(
            "scipy is required for efficient neighbor search. "
            "Install it in the active environment with: pip install scipy"
        )

    n_points = len(points)
    if n_points < 2:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float64)

    tree = cKDTree(points)
    k = min(max_neighbors + 1, n_points)

    # query returns each point itself as the first neighbor (distance ~ 0).
    distances, indices = tree.query(points, k=k, workers=-1)

    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    edge_distances = {}
    for source in range(n_points):
        for distance, target in zip(distances[source], indices[source]):
            target = int(target)
            distance = float(distance)

            if target == source or not np.isfinite(distance):
                continue
            if distance > max_distance:
                continue

            a, b = (source, target) if source < target else (target, source)
            # If either endpoint selected the other, keep one undirected edge.
            edge_distances[(a, b)] = distance

    if not edge_distances:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float64)

    sorted_items = sorted(edge_distances.items())
    edges = np.asarray([item[0] for item in sorted_items], dtype=np.int64)
    lengths = np.asarray([item[1] for item in sorted_items], dtype=np.float64)
    return edges, lengths


def write_colored_graph_ply(
    output_path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    edges: np.ndarray,
):
    """
    Write an ASCII PLY containing:
      - vertex elements with original XYZ and RGB
      - edge elements referencing the original vertex indices

    RGB values are converted to PLY's 0..255 representation for writing.
    This conversion does not change the source PCD or the point-cloud colors
    used for the Open3D preview.
    """
    rgb_u8 = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)

    with output_path.open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment Generated by pipeline/04_connectivity_graph.py\n")
        f.write("comment Vertex XYZ and RGB are copied from semantic_labeled.pcd\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property double x\n")
        f.write("property double y\n")
        f.write("property double z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for xyz, rgb in zip(points, rgb_u8):
            f.write(
                f"{xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )

        # Neutral gray edges; vertex RGB remains untouched.
        edge_rgb = (150, 150, 150)
        for a, b in edges:
            f.write(
                f"{int(a)} {int(b)} "
                f"{edge_rgb[0]} {edge_rgb[1]} {edge_rgb[2]}\n"
            )


def write_nodes_csv(path: Path, points: np.ndarray, colors: np.ndarray):
    rgb_u8 = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x", "y", "z", "r", "g", "b"])
        for i, (xyz, rgb) in enumerate(zip(points, rgb_u8)):
            writer.writerow([
                i,
                f"{xyz[0]:.9f}",
                f"{xyz[1]:.9f}",
                f"{xyz[2]:.9f}",
                int(rgb[0]),
                int(rgb[1]),
                int(rgb[2]),
            ])


def write_edges_csv(path: Path, edges: np.ndarray, lengths: np.ndarray):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "distance_m"])
        for (a, b), distance in zip(edges, lengths):
            writer.writerow([int(a), int(b), f"{distance:.9f}"])


def connected_component_stats(n_points: int, edges: np.ndarray):
    """Compute component sizes using a simple union-find structure."""
    parent = np.arange(n_points, dtype=np.int64)
    size = np.ones(n_points, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    def union(a, b):
        ra, rb = find(int(a)), find(int(b))
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for a, b in edges:
        union(a, b)

    roots = [find(i) for i in range(n_points)]
    unique_roots, counts = np.unique(roots, return_counts=True)
    component_sizes = sorted((int(c) for c in counts), reverse=True)
    return {
        "component_count": int(len(unique_roots)),
        "largest_component_size": int(component_sizes[0]) if component_sizes else 0,
        "isolated_node_count": int(np.count_nonzero(counts == 1)),
        "component_sizes_descending": component_sizes,
    }


def main():
    args = parse_args()

    if args.max_distance <= 0:
        raise ValueError("--max-distance must be greater than 0.")
    if args.max_neighbors < 1:
        raise ValueError("--max-neighbors must be at least 1.")

    input_path = args.input_pcd.resolve()
    out_dir = (args.out_dir.resolve() if args.out_dir else input_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Step 04] Input: {input_path}")
    cloud, points, colors = load_point_cloud(input_path)
    print(f"[Step 04] Loaded points: {len(points):,}")
    print("[Step 04] Using original XYZ and RGB; no label/color filtering.")

    edges, lengths = build_edges(
        points,
        max_distance=args.max_distance,
        max_neighbors=args.max_neighbors,
    )
    print(f"[Step 04] Nodes: {len(points):,}")
    print(f"[Step 04] Edges: {len(edges):,}")

    ply_path = out_dir / "connectivity_graph.ply"
    nodes_path = out_dir / "graph_nodes.csv"
    edges_path = out_dir / "graph_edges.csv"
    graph_path = out_dir / "connectivity_graph.json"
    diagnostics_path = out_dir / "edge_diagnostics.json"

    write_colored_graph_ply(ply_path, points, colors, edges)
    write_nodes_csv(nodes_path, points, colors)
    write_edges_csv(edges_path, edges, lengths)

    component_stats = connected_component_stats(len(points), edges)
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)

    graph_summary = {
        "input_pcd": str(input_path),
        "output_ply": str(ply_path),
        "node_count": int(len(points)),
        "edge_count": int(len(edges)),
        "max_distance_m": float(args.max_distance),
        "max_neighbors": int(args.max_neighbors),
        "coordinate_source": "semantic_labeled.pcd",
        "color_source": "semantic_labeled.pcd",
        "points_filtered_by_label_or_color": False,
        "coordinates_recomputed": False,
        "node_colors_reassigned": False,
        "xyz_min": xyz_min.tolist(),
        "xyz_max": xyz_max.tolist(),
        **component_stats,
    }
    with graph_path.open("w", encoding="utf-8") as f:
        json.dump(graph_summary, f, indent=2, ensure_ascii=False)

    diagnostics = {
        "max_distance_m": float(args.max_distance),
        "max_neighbors": int(args.max_neighbors),
        "node_count": int(len(points)),
        "edge_count": int(len(edges)),
        "degree_min": 0,
        "degree_max": 0,
        "degree_mean": 0.0,
        "edge_length_min_m": None,
        "edge_length_max_m": None,
        "edge_length_mean_m": None,
        "isolated_node_count": component_stats["isolated_node_count"],
        "component_count": component_stats["component_count"],
    }

    if len(edges):
        degree = np.zeros(len(points), dtype=np.int64)
        np.add.at(degree, edges[:, 0], 1)
        np.add.at(degree, edges[:, 1], 1)
        diagnostics.update({
            "degree_min": int(degree.min()),
            "degree_max": int(degree.max()),
            "degree_mean": float(degree.mean()),
            "edge_length_min_m": float(lengths.min()),
            "edge_length_max_m": float(lengths.max()),
            "edge_length_mean_m": float(lengths.mean()),
        })

    with diagnostics_path.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)

    print("\n[Step 04] Outputs:")
    print(f"  - {ply_path}")
    print(f"  - {nodes_path}")
    print(f"  - {edges_path}")
    print(f"  - {graph_path}")
    print(f"  - {diagnostics_path}")

    if args.view:
        # The original point cloud object is retained for the preview.
        # The separate LineSet adds only neutral gray edges.
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(edges.astype(np.int32))
        if len(edges):
            line_set.colors = o3d.utility.Vector3dVector(
                np.tile(np.array([[0.58, 0.58, 0.58]]), (len(edges), 1))
            )
        o3d.visualization.draw_geometries(
            [cloud, line_set],
            window_name="Step 04 - Connectivity Graph (original point colors)",
            width=1280,
            height=800,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
