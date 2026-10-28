
# -*- coding: utf-8 -*-
"""Export accessibility score grid for Revit surface heatmap.

Input:
    scores.npz

Outputs:
    heatmap_cells.csv
    heatmap_surface.png (if matplotlib is installed)

CSV format remains compatible with AccessibilityAnalyzer.py:
    x,y,z,score

Coordinates are grid-cell centres in the aligned point-cloud frame.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", help="Path to scores.npz")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.35)
    return parser.parse_args()


def load_data(path):
    with np.load(str(path), allow_pickle=False) as data:
        score_key = (
            "score_raw" if "score_raw" in data.files else "score"
        )

        required = ["mask", "mins", "cell", "floor_z", score_key]
        missing = [k for k in required if k not in data.files]

        if missing:
            raise ValueError(
                "Missing keys {}. Available keys: {}".format(
                    missing, list(data.files)
                )
            )

        score = np.asarray(data[score_key], dtype=np.float64)
        mask = np.asarray(data["mask"], dtype=bool)
        mins = np.asarray(data["mins"], dtype=np.float64).reshape(-1)
        cell = float(np.asarray(data["cell"]).reshape(-1)[0])
        floor_z = np.asarray(data["floor_z"], dtype=np.float64)

    if score.ndim != 2:
        raise ValueError("Score grid must be 2D; got {}".format(score.shape))

    if not (score.shape == mask.shape == floor_z.shape):
        raise ValueError(
            "score, mask and floor_z must have matching shapes."
        )

    if len(mins) < 2:
        raise ValueError("mins must contain at least X and Y origins.")

    if not np.isfinite(cell) or cell <= 0:
        raise ValueError("cell must be a positive finite number.")

    valid = mask & np.isfinite(score) & np.isfinite(floor_z)

    if not valid.any():
        raise ValueError("No valid score cells to export.")

    score = np.where(valid, np.clip(score, 0.0, 100.0), np.nan)

    return score, valid, mins, cell, floor_z


def smooth_score(score, valid, cell, sigma_m):
    """Smooth for visual display only; do not modify scores.npz."""
    if sigma_m <= 0:
        return score.copy()

    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        print(
            "WARNING: scipy unavailable; using unsmoothed scores.",
            flush=True
        )
        return score.copy()

    sigma_cells = sigma_m / cell

    weights = valid.astype(np.float64)
    values = np.where(valid, score, 0.0)

    numerator = gaussian_filter(
        values, sigma=sigma_cells, mode="nearest"
    )
    denominator = gaussian_filter(
        weights, sigma=sigma_cells, mode="nearest"
    )

    output = np.full(score.shape, np.nan, dtype=np.float64)
    use = valid & (denominator > 1e-10)

    output[use] = numerator[use] / denominator[use]
    output[use] = np.clip(output[use], 0.0, 100.0)

    return output


def export_cells(score, valid, mins, cell, floor_z, stride):
    """Export cell centres while preserving the score grid's X/Y indexing."""
    nx, ny = score.shape
    cells = []

    for i in range(0, nx, stride):
        x = float(mins[0] + (i + 0.5) * cell)

        for j in range(0, ny, stride):
            if not valid[i, j]:
                continue

            y = float(mins[1] + (j + 0.5) * cell)
            z = float(floor_z[i, j])
            s = float(score[i, j])

            if not all(np.isfinite(v) for v in (x, y, z, s)):
                continue

            cells.append((x, y, z, s))

    return cells


def write_csv(path, cells):
    with open(str(path), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "score"])
        writer.writerows(cells)


def write_preview(path, score, valid, mins, cell):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib unavailable; PNG skipped.", flush=True)
        return False

    nx, ny = score.shape
    masked = np.ma.masked_where(~valid, score)

    x0 = float(mins[0])
    y0 = float(mins[1])
    x1 = x0 + nx * cell
    y1 = y0 + ny * cell

    fig, ax = plt.subplots(figsize=(10, 8))

    # The score grid is indexed [X, Y], so transpose for display.
    image = ax.imshow(
        masked.T,
        origin="lower",
        extent=(x0, x1, y0, y1),
        interpolation="nearest",
        aspect="equal",
        cmap="RdYlGn",
        vmin=0,
        vmax=100
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Accessibility Heatmap")
    fig.colorbar(image, ax=ax, label="Accessibility score (0-100)")
    fig.tight_layout()
    fig.savefig(str(path), dpi=180)
    plt.close(fig)

    return True


def main():
    args = parse_args()
    source = Path(args.scores).resolve()

    if not source.is_file():
        sys.exit("scores.npz not found: {}".format(source))

    if args.stride < 1:
        sys.exit("--stride must be >= 1")

    if args.sigma < 0:
        sys.exit("--sigma must be >= 0")

    try:
        score, valid, mins, cell, floor_z = load_data(source)

        print("=" * 50)
        print("EXPORT HEATMAP GRID")
        print("Input       :", source)
        print("Grid shape  :", score.shape)
        print("Cell size   : {:.3f} m".format(cell))
        print("Valid cells :", int(valid.sum()))
        print("Stride      :", args.stride)
        print("Sigma       : {:.3f} m".format(args.sigma))

        # Smoothing is for visualisation only. The source scores.npz
        # remains unchanged, and invalid cells stay invalid.
        display_score = smooth_score(
            score, valid, cell, args.sigma
        )

        cells = export_cells(
            display_score,
            valid,
            mins,
            cell,
            floor_z,
            args.stride
        )

        if not cells:
            sys.exit("No heatmap cells were exported.")

        output_dir = source.parent
        csv_path = output_dir / "heatmap_cells.csv"
        png_path = output_dir / "heatmap_surface.png"

        write_csv(csv_path, cells)
        png_created = write_preview(
            png_path, display_score, valid, mins, cell
        )

        print("Exported cells:", format(len(cells), ","))
        print("Saved:", csv_path)

        if png_created:
            print("Saved:", png_path)

        print("Done.")

    except Exception as exc:
        sys.exit("Heatmap export failed: {}".format(exc))


if __name__ == "__main__":
    main()
