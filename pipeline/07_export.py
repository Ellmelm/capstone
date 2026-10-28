
import sys
import csv
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.ndimage import label, binary_dilation

from common import load_criteria


LEVEL_NAME = {
    "critical": "Critical Low",
    "very_low": "Very Low",
    "low": "Low",
    "high": "High",
}

# ใช้สีชุดเดียวกันทุกภาพ เพื่อให้ตีความคะแนนตรงกัน
LEVEL_COLORS = [
    "#E53935",  # Critical Low
    "#FB8C00",  # Very Low
    "#FDD835",  # Low
    "#43A047",  # High
]

BACKGROUND = "#202832"
TEXT_COLOR = "#263238"


def level_key(s, b):
    if s <= b["critical"]:
        return "critical"
    if s <= b["very_low"]:
        return "very_low"
    if s <= b["low"]:
        return "low"
    return "high"


def make_cmap(bands):
    """สร้าง color map สำหรับคะแนน โดยไม่เปลี่ยนเกณฑ์เดิม"""
    cmap = ListedColormap(LEVEL_COLORS)
    cmap.set_bad((0, 0, 0, 0))

    norm = BoundaryNorm(
        [
            0,
            bands["critical"],
            bands["very_low"],
            bands["low"],
            100.01,
        ],
        cmap.N,
    )
    return cmap, norm


def save_topview(out_dir, img, ext, cmap, norm,
                 overall, low_pct, cell):
    """ผังคะแนนแบบมีแกนพิกัด สเกล และคำอธิบายสี"""
    fig, ax = plt.subplots(figsize=(10, 9))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F1F3F5")

    ax.imshow(
        img.T,
        origin="lower",
        extent=ext,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_title(
        "Accessibility Assessment",
        fontsize=17,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=18,
    )

    ax.text(
        0.5,
        1.01,
        "Score: %.1f / 100    |    Low-access area: %.1f%%"
        % (overall, low_pct),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        color=TEXT_COLOR,
    )

    ax.set_xlabel("X coordinate (m)", fontsize=10)
    ax.set_ylabel("Y coordinate (m)", fontsize=10)
    ax.tick_params(labelsize=9)

    ax.grid(
        True,
        color="#AAB2BD",
        linewidth=0.45,
        alpha=0.35,
    )

    legend_items = [
        Patch(facecolor=LEVEL_COLORS[0], label="Critical Low (0-15)"),
        Patch(facecolor=LEVEL_COLORS[1], label="Very Low (16-30)"),
        Patch(facecolor=LEVEL_COLORS[2], label="Low (31-50)"),
        Patch(facecolor=LEVEL_COLORS[3], label="High (51-100)"),
    ]

    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    ax.set_aspect("equal", adjustable="box")

    fig.text(
        0.99,
        0.015,
        "Grid cell: %.2f m" % cell,
        ha="right",
        fontsize=8,
        color="#667085",
    )

    fig.tight_layout(rect=[0, 0.07, 1, 0.95])

    fig.savefig(
        out_dir / "topview.png",
        dpi=180,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(fig)


def save_heatmap(out_dir, img, ext, cmap, norm,
                 mins, cell, mask, recs):
    """Heatmap พื้นหลังโปร่งใสสำหรับแสดงใน Revit"""
    ii, jj = np.nonzero(mask)

    x0 = mins[0] + ii.min() * cell
    x1 = mins[0] + (ii.max() + 1) * cell
    y0 = mins[1] + jj.min() * cell
    y1 = mins[1] + (jj.max() + 1) * cell

    # เพิ่มขอบรอบพื้นที่เพื่อไม่ให้จุดขอบถูกตัด
    pad = max(1.0, cell * 2.0)

    width = x1 - x0 + 2 * pad
    height = y1 - y0 + 2 * pad

    # ใช้ภาพขนาดคงที่ ป้องกันรูปยาวมากหรือใหญ่เกินจำเป็น
    aspect = width / max(height, 1e-6)
    aspect = float(np.clip(aspect, 0.5, 2.0))

    fig_height = 7.0
    fig_width = fig_height * aspect

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height)
    )

    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))

    ax.imshow(
        img.T,
        origin="lower",
        extent=ext,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)

    # แสดงหมายเลข Recommendation ให้เด่น แต่ไม่ใหญ่จนบัง Heatmap
    for rec in recs:
        ax.scatter(
            [rec["x"]],
            [rec["y"]],
            s=125,
            facecolors="#17202A",
            edgecolors="white",
            linewidths=1.4,
            zorder=5,
        )

        ax.text(
            rec["x"],
            rec["y"],
            str(rec["rank"]),
            color="white",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            zorder=6,
        )

    ax.set_axis_off()

    fig.savefig(
        out_dir / "heatmap.png",
        dpi=180,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.06,
    )
    plt.close(fig)


def save_problem_areas(out_dir, img, ext, cmap, norm,
                       low_mask, recs, cell):
    """ภาพสำหรับเน้นบริเวณคะแนนต่ำและจุดแนะนำ"""
    ii, jj = np.nonzero(low_mask)

    if len(ii) == 0:
        return

    x0, x1 = ext[0], ext[1]
    y0, y1 = ext[2], ext[3]

    pad = max(1.0, cell * 2.0)

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F1F3F5")

    # แสดงเฉพาะบริเวณคะแนนต่ำ
    low_img = np.where(low_mask, img, np.nan)

    ax.imshow(
        low_img.T,
        origin="lower",
        extent=ext,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_xlim(
        max(x0, x0 + ii.min() * cell - pad),
        min(x1, x0 + (ii.max() + 1) * cell + pad),
    )
    ax.set_ylim(
        max(y0, y0 + jj.min() * cell - pad),
        min(y1, y0 + (jj.max() + 1) * cell + pad),
    )

    for rec in recs:
        ax.scatter(
            [rec["x"]],
            [rec["y"]],
            s=180,
            color="#17202A",
            edgecolors="white",
            linewidths=1.5,
            zorder=5,
        )
        ax.text(
            rec["x"],
            rec["y"],
            str(rec["rank"]),
            color="white",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=6,
        )

    ax.set_title(
        "Areas Requiring Accessibility Review",
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=14,
    )
    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        out_dir / "problem_areas.png",
        dpi=180,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(fig)


def main(scores_file, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_criteria()
    sc = cfg["score"]
    b = sc["bands"]

    rc = cfg.get("recommend", {})
    min_area = float(rc.get("min_cluster_area_m2", 2.0))
    max_items = int(rc.get("max_items", 5))
    gap_m = float(rc.get("merge_gap_m", 0.5))

    d = np.load(scores_file)

    score = d["score"]
    ss = d["slope_score"]
    ws = d["width_score"]
    mask = d["mask"].astype(bool)
    mins = d["mins"]
    cell = float(d["cell"])
    floor_z = d["floor_z"]

    slope = d["slope"] if "slope" in d.files else None
    width = d["width"] if "width" in d.files else None

    if score.shape != mask.shape:
        sys.exit("score และ mask มีขนาดไม่ตรงกัน")

    if ss.shape != mask.shape or ws.shape != mask.shape:
        sys.exit("slope_score หรือ width_score มีขนาดไม่ตรงกับ mask")

    if floor_z.shape != mask.shape:
        sys.exit("floor_z มีขนาดไม่ตรงกับ mask")

    if cell <= 0:
        sys.exit("ค่า cell ต้องมากกว่า 0")

    if not np.isfinite(score[mask]).all():
        sys.exit("พบคะแนนที่ไม่ใช่ตัวเลขในพื้นที่เดินได้")

    if not mask.any():
        sys.exit("ไม่มีช่องพื้นที่เดินได้ในไฟล์ scores.npz")

    unit = cell * cell
    total = float(mask.sum() * unit)
    overall = float(score[mask].mean())

    low_mask = mask & (score <= b["low"])
    low_area = float(low_mask.sum() * unit)
    low_pct = 100.0 * low_area / total

    def area(m):
        return round(float(m.sum() * unit), 1)

    bands_area = {
        "critical": area(mask & (score <= b["critical"])),
        "very_low": area(
            mask & (score > b["critical"])
            & (score <= b["very_low"])
        ),
        "low": area(
            mask & (score > b["very_low"])
            & (score <= b["low"])
        ),
        "high": area(mask & (score > b["low"])),
    }

    # ---- Low-access cells: schema เดิมสำหรับ Revit --------------------
    cells = []

    for i, j in np.argwhere(low_mask):
        cells.append({
            "x": round(float(mins[0] + (i + 0.5) * cell), 2),
            "y": round(float(mins[1] + (j + 0.5) * cell), 2),
            "z": round(float(floor_z[i, j]), 2),
            "score": round(float(score[i, j]), 1),
            "reason": "slope" if ss[i, j] <= ws[i, j] else "width",
        })

    # ---- Heuristic recommendations: ใช้เกณฑ์เดิม ---------------------
    cands = []

    if low_mask.any():
        iters = max(1, int(round(gap_m / cell)))

        merged = binary_dilation(
            low_mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=iters,
        )

        lab, n = label(
            merged,
            structure=np.ones((3, 3), dtype=bool),
        )

        lab = np.where(low_mask, lab, 0)

        counts = np.bincount(
            lab.ravel(),
            minlength=n + 1,
        )
        counts[0] = 0

        for cid in np.argsort(-counts)[:60]:
            if cid == 0:
                continue

            if counts[cid] * unit < min_area:
                break

            ii, jj = np.nonzero(lab == cid)

            if len(ii) == 0:
                continue

            ar = len(ii) * unit
            ms = float(score[ii, jj].mean())

            slope_dom = (
                float((ss[ii, jj] <= ws[ii, jj]).mean()) >= 0.5
            )

            cx = float(mins[0] + (ii.mean() + 0.5) * cell)
            cy = float(mins[1] + (jj.mean() + 0.5) * cell)
            cz = float(floor_z[ii, jj].mean())

            where = "near (%.0f, %.0f)" % (cx, cy)

            if slope_dom:
                typ, title = "slope", "Reduce the gradient"

                if slope is not None:
                    detail = (
                        "About %.0f m² %s averages %.1f° slope "
                        "(target ≤ %.0f°). Consider a gentler ramp, "
                        "a level landing, or an alternative route."
                    ) % (
                        ar, where,
                        float(slope[ii, jj].mean()),
                        sc["slope_ok_deg"],
                    )
                else:
                    detail = (
                        "About %.0f m² %s scores low on slope "
                        "(average %.0f/100)."
                    ) % (ar, where, float(ss[ii, jj].mean()))

            else:
                typ, title = "width", "Widen the passage"

                if width is not None:
                    detail = (
                        "About %.0f m² %s has a clear width of roughly "
                        "%.1f m (target ≥ %.1f m). Check for obstacles "
                        "or widen the opening."
                    ) % (
                        ar, where,
                        float(width[ii, jj].mean()),
                        sc["width_ok_m"],
                    )
                else:
                    detail = (
                        "About %.0f m² %s scores low on width "
                        "(average %.0f/100)."
                    ) % (ar, where, float(ws[ii, jj].mean()))

            cands.append({
                "priority": ar * (100.0 - ms),
                "level": level_key(ms, b),
                "type": typ,
                "title": title,
                "detail": detail,
                "x": round(cx, 2),
                "y": round(cy, 2),
                "z": round(cz, 2),
                "area_m2": round(ar, 1),
                "score": round(ms, 1),
            })

    cands.sort(key=lambda c: -c["priority"])

    recs = []

    for k, c in enumerate(cands[:max_items], 1):
        c.pop("priority")
        c["rank"] = k
        recs.append(c)

    # ---- Notes ---------------------------------------------------------
    notes = []

    if low_pct > 50:
        notes.append({
            "level": "warning",
            "text": (
                "More than half of the walkable area scores low. Check that "
                "vegetation, roofs or other non-walkable surfaces were not "
                "counted (run the pipeline with --traj to filter by the walked path)."
            ),
        })

    if slope is None or width is None:
        notes.append({
            "level": "info",
            "text": (
                "Re-run 06_score.py to include slope and width values "
                "in the suggestions."
            ),
        })

    if not recs:
        notes.append({
            "level": "info",
            "text": (
                "No concentrated problem area larger than %.0f m² was found."
                % min_area
            ),
        })

    # ---- Summary: schema เดิม -----------------------------------------
    overall_level = level_key(overall, b)

    summary = {
        "overall_score": round(overall, 1),
        "overall_level": overall_level,
        "overall_level_name": LEVEL_NAME[overall_level],
        "total_walkable_area_m2": round(total, 1),
        "low_access_area_m2": round(low_area, 1),
        "low_access_percent": round(low_pct, 1),
        "bands_area_m2": bands_area,
        "cell_size_m": cell,
        "bands": b,
    }

    with open(
        out_dir / "result.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "summary": summary,
                "recommendations": recs,
                "notes": notes,
                "low_access_cells": cells,
            },
            f,
            ensure_ascii=False,
            indent=1,
        )

    with open(
        out_dir / "summary.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)

        for k, v in summary.items():
            if not isinstance(v, dict):
                w.writerow([k, v])

        for r in recs:
            w.writerow([
                "rec_%d" % r["rank"],
                r["title"] + " | " + r["detail"],
            ])

    # ---- Shared heatmap data ------------------------------------------
    cmap, norm = make_cmap(b)

    img = np.where(mask, score, np.nan)

    ext = [
        float(mins[0]),
        float(mins[0] + img.shape[0] * cell),
        float(mins[1]),
        float(mins[1] + img.shape[1] * cell),
    ]

    # 1. Report / top view
    save_topview(
        out_dir, img, ext, cmap, norm,
        overall, low_pct, cell,
    )

    # 2. Transparent image for the Revit panel
    save_heatmap(
        out_dir, img, ext, cmap, norm,
        mins, cell, mask, recs,
    )

    # 3. Dedicated problem-area image
    save_problem_areas(
        out_dir, img, ext, cmap, norm,
        low_mask, recs, cell,
    )

    print(
        "คะแนนรวม %.1f (%s) | พื้นที่เดินได้ %.1f m² | "
        "พื้นที่คะแนนต่ำ %.1f%%"
        % (
            overall,
            LEVEL_NAME[overall_level],
            total,
            low_pct,
        )
    )

    for r in recs:
        print(
            "  #%d [%s] %s - %s"
            % (r["rank"], r["level"], r["title"], r["detail"])
        )

    for note in notes:
        print("  หมายเหตุ:", note["text"])

    print("บันทึก result.json, summary.csv")
    print("บันทึก topview.png, heatmap.png, problem_areas.png")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(
            "Usage: python 07_export.py <scores.npz> <out_dir>"
        )

    main(sys.argv[1], sys.argv[2])
