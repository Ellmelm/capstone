# -*- coding: utf-8 -*-
"""Run the accessibility pipeline using a unique output folder per run.

Order:
    01_align -> 02_walkable -> 03_semantic_label -> 04_score
    -> 07_export -> 08_export_heatmap_mesh -> 10_reconstruct_mesh

05_cost_field.py is not run here yet (add it as a step once its
arguments are final).
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def create_unique_run_dir(output_root):
    """Create output/runs/YYYYMMDD_HHMMSS without overwriting prior runs."""
    runs_root = output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / timestamp
    suffix = 1

    while run_dir.exists():
        run_dir = runs_root / "{}_{:02d}".format(timestamp, suffix)
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run accessibility analysis in a new output folder."
    )

    parser.add_argument("pcd", help="Input point cloud .pcd")
    parser.add_argument("out_dir", help="Output root directory")

    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.35)

    # Mesh edge limit must accommodate the walkable grid.
    parser.add_argument("--max-edge", type=float, default=0.75)
    parser.add_argument("--no-walls", action="store_true")
    parser.add_argument("--service-points", default=None,
                        help="Optional CSV (x,y[,label]) passed to 02_walkable")
    parser.add_argument("--skip-semantic", action="store_true",
                        help="Skip 03_semantic_label.py")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    pipeline_dir = Path(__file__).resolve().parent
    python_exe = sys.executable

    input_pcd = Path(args.pcd).resolve()
    output_root = Path(args.out_dir).resolve()

    if not input_pcd.is_file():
        sys.exit("ERROR: Input PCD does not exist: {}".format(input_pcd))

    if args.stride < 1:
        sys.exit("ERROR: --stride must be at least 1.")

    if args.sigma < 0:
        sys.exit("ERROR: --sigma cannot be negative.")

    if args.max_edge <= 0:
        sys.exit("ERROR: --max-edge must be positive.")

    required_scripts = [
        "01_align.py",
        "02_walkable.py",
        "04_score.py",
        "07_export.py",
        "08_export_heatmap_mesh.py",
        "10_reconstruct_mesh.py",
    ]
    if not args.skip_semantic:
        required_scripts.append("03_semantic_label.py")

    missing = [
        name for name in required_scripts
        if not (pipeline_dir / name).is_file()
    ]

    if missing:
        sys.exit(
            "ERROR: Missing required pipeline scripts:\n  - "
            + "\n  - ".join(missing)
        )

    try:
        out = create_unique_run_dir(output_root)
    except Exception as exc:
        sys.exit("ERROR: Cannot create output folder: {}".format(exc))

    print("=" * 65, flush=True)
    print("ACCESSIBILITY PIPELINE", flush=True)
    print("=" * 65, flush=True)
    print("Input  : {}".format(input_pcd), flush=True)
    print("Output : {}".format(out), flush=True)
    print("=" * 65, flush=True)

    walk_args = [str(out / "aligned.pcd"), str(out)]

    if args.no_walls:
        walk_args.append("--no-walls")

    if args.service_points:
        walk_args += ["--service-points", str(Path(args.service_points).resolve())]

    reconstruct_args = [
        str(out / "aligned.pcd"),
        str(out),
        "--walkable",
        str(out / "walkable.npz"),
        "--max-edge",
        str(args.max_edge),
    ]

    if args.no_walls:
        reconstruct_args.append("--no-walls")

    steps = [
        ("01_align.py", [str(input_pcd), str(out)]),
        ("02_walkable.py", walk_args),
    ]

    if not args.skip_semantic:
        steps.append((
            "03_semantic_label.py",
            [
                str(out / "aligned.pcd"),
                str(out / "walkable.pcd"),
                str(out / "walkable.npz"),
                "--out-dir", str(out),
            ],
        ))

    steps += [
        (
            "04_score.py",
            [
                str(out / "aligned.pcd"),
                str(out / "walkable.npz"),
                str(out),
            ],
        ),
        ("07_export.py", [str(out / "scores.npz"), str(out)]),
        (
            "08_export_heatmap_mesh.py",
            [
                str(out / "scores.npz"),
                "--stride", str(args.stride),
                "--sigma", str(args.sigma),
            ],
        ),
        ("10_reconstruct_mesh.py", reconstruct_args),
    ]

    log_path = out / "pipeline.log"

    with open(log_path, "w", encoding="utf-8") as log:
        log.write("Input: {}\nOutput: {}\n".format(input_pcd, out))

        for index, (script_name, script_args) in enumerate(steps, start=1):
            script_path = pipeline_dir / script_name
            command = [python_exe, str(script_path)] + script_args

            print("\n[{}/{}] {}".format(index, len(steps), script_name), flush=True)
            print("Command:", " ".join(command), flush=True)

            log.write("\n[{}/{}] {}\n".format(index, len(steps), script_name))
            log.write("Command: {}\n".format(" ".join(command)))
            log.flush()

            try:
                result = subprocess.run(
                    command,
                    cwd=str(project_root),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            except Exception as exc:
                print("ERROR: Could not start {}: {}".format(script_name, exc), flush=True)
                print("See log: {}".format(log_path), flush=True)
                sys.exit(1)

            log.flush()

            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as reader:
                    lines = reader.readlines()
                print("".join(lines[-100:]), flush=True)
            except OSError:
                pass

            if result.returncode != 0:
                print(
                    "\nERROR: {} failed with exit code {}.".format(
                        script_name, result.returncode
                    ),
                    flush=True,
                )
                print("Partial results kept at: {}".format(out), flush=True)
                print("Full log: {}".format(log_path), flush=True)
                sys.exit(result.returncode)

    print("\n" + "=" * 65, flush=True)
    print("PIPELINE COMPLETED", flush=True)
    print("Output folder: {}".format(out), flush=True)
    print("Log file: {}".format(log_path), flush=True)
    print("=" * 65, flush=True)

    print("\nCheck these outputs if they were generated:", flush=True)
    for filename in [
        "aligned.pcd",
        "walkable.pcd",
        "walkable.npz",
        "surface_cells.csv",
        "semantic.npz",
        "semantic_labeled.pcd",
        "scores.npz",
        "result.json",
        "heatmap_cells.csv",
        "floor_mesh.ply",
        "wall_mesh.ply",
        "reconstruction.json",
    ]:
        path = out / filename
        status = "[OK]" if path.is_file() else "[--]"
        print("  {} {}".format(status, filename), flush=True)


if __name__ == "__main__":
    main()