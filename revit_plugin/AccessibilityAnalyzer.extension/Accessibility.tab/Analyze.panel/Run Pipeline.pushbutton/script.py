
# -*- coding: utf-8 -*-
"""Run Accessibility Pipeline from the PCD selected in Revit.

Reads:
    data/selected_pcd.txt

Runs:
    pipeline/run_pipeline.py <input.pcd> output

Writes:
    output/runs/<timestamp>/
    data/latest_run.txt
"""

import os
import io
import subprocess
from datetime import datetime

from pyrevit import forms, script


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = r"D:\Projects\accessibility-capstone"

PYTHON_EXE = (
    r"C:\Users\USER\anaconda3\envs\access\python.exe"
)

PIPELINE_SCRIPT = os.path.join(
    PROJECT_ROOT, "pipeline", "run_pipeline.py"
)

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")
RUNS_ROOT = os.path.join(OUTPUT_ROOT, "runs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

SELECTED_FILE_TXT = os.path.join(
    DATA_DIR, "selected_pcd.txt"
)

LATEST_RUN_TXT = os.path.join(
    DATA_DIR, "latest_run.txt"
)

LAUNCH_LOG = os.path.join(
    OUTPUT_ROOT, "pipeline_launcher.log"
)


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    forms.alert(
        message,
        title="Run Accessibility Pipeline",
        exitscript=True
    )


def read_selected_pcd():
    if not os.path.isfile(SELECTED_FILE_TXT):
        fail(
            "No selected PCD path was found.\n\n"
            "Please click Import PCD first.\n\n"
            "Expected file:\n{}".format(SELECTED_FILE_TXT)
        )

    with io.open(SELECTED_FILE_TXT, "r", encoding="utf-8") as f:
        pcd_path = f.read().strip()

    if not pcd_path:
        fail(
            "The selected PCD path is empty.\n"
            "Please click Import PCD again."
        )

    # The Import PCD button saves paths using forward slashes.
    pcd_path = os.path.abspath(pcd_path.replace("/", os.sep))

    if not os.path.isfile(pcd_path):
        fail(
            "The selected PCD file cannot be found.\n\n"
            "Path:\n{}".format(pcd_path)
        )

    if os.path.splitext(pcd_path)[1].lower() != ".pcd":
        fail(
            "The selected file is not a .pcd file.\n\n"
            "Path:\n{}".format(pcd_path)
        )

    return pcd_path


def get_existing_run_dirs():
    existing = set()

    if os.path.isdir(RUNS_ROOT):
        for name in os.listdir(RUNS_ROOT):
            path = os.path.join(RUNS_ROOT, name)
            if os.path.isdir(path):
                existing.add(os.path.normcase(os.path.abspath(path)))

    return existing


def find_new_run_dir(existing_dirs):
    if not os.path.isdir(RUNS_ROOT):
        return None

    candidates = []

    for name in os.listdir(RUNS_ROOT):
        path = os.path.join(RUNS_ROOT, name)

        if not os.path.isdir(path):
            continue

        normalized = os.path.normcase(os.path.abspath(path))

        if normalized not in existing_dirs:
            candidates.append(path)

    if not candidates:
        return None

    return max(candidates, key=os.path.getmtime)


def save_latest_run(run_dir):
    with io.open(LATEST_RUN_TXT, "w", encoding="utf-8") as f:
        f.write(os.path.abspath(run_dir).replace("\\", "/"))
        f.write(u"\n")


# ============================================================
# MAIN
# ============================================================

if not os.path.isdir(PROJECT_ROOT):
    fail("Project folder not found:\n{}".format(PROJECT_ROOT))

if not os.path.isfile(PYTHON_EXE):
    fail(
        "Python environment was not found.\n\n"
        "Expected:\n{}\n\n"
        "Check the Python executable for your access environment."
        .format(PYTHON_EXE)
    )

if not os.path.isfile(PIPELINE_SCRIPT):
    fail(
        "Pipeline script was not found:\n{}".format(PIPELINE_SCRIPT)
    )

if not os.path.isdir(OUTPUT_ROOT):
    os.makedirs(OUTPUT_ROOT)

if not os.path.isdir(RUNS_ROOT):
    os.makedirs(RUNS_ROOT)

pcd_path = read_selected_pcd()

# Remember the existing runs so we can identify the new one.
existing_runs = get_existing_run_dirs()

output = script.get_output()
output.print_md("# Accessibility Pipeline")
output.print_md("**Input PCD:** `{}`".format(pcd_path))
output.print_md("**Python:** `{}`".format(PYTHON_EXE))
output.print_md("**Pipeline:** `{}`".format(PIPELINE_SCRIPT))
output.print_md("")
output.print_md("Running the pipeline. Please wait...")

command = [
    PYTHON_EXE,
    PIPELINE_SCRIPT,
    pcd_path,
    OUTPUT_ROOT
]

started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
return_code = None
launch_error = None

try:
    with open(LAUNCH_LOG, "w") as log_file:
        log_file.write("Started: {}\n".format(started_at))
        log_file.write("Input PCD: {}\n".format(pcd_path))
        log_file.write("Command: {}\n\n".format(
            " ".join(['"{}"'.format(x) for x in command])
        ))
        log_file.flush()

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        process = subprocess.Popen(
            [PYTHON_EXE, PIPELINE_SCRIPT, pcd_path, OUTPUT_ROOT],
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env
            )

        return_code = process.wait()

except Exception as exc:
    launch_error = str(exc)


# Find the output folder created during this run.
run_dir = find_new_run_dir(existing_runs)

if launch_error:
    forms.alert(
        "Could not start the pipeline.\n\n"
        "Error:\n{}\n\n"
        "Launcher log:\n{}".format(launch_error, LAUNCH_LOG),
        title="Run Pipeline",
        exitscript=True
    )

if return_code != 0:
    forms.alert(
        "Pipeline failed.\n\n"
        "Exit code: {}\n\n"
        "Please inspect:\n{}\n\n"
        "Launcher log:\n{}".format(
            return_code,
            os.path.join(run_dir, "pipeline.log")
            if run_dir else RUNS_ROOT,
            LAUNCH_LOG
        ),
        title="Run Pipeline",
        exitscript=True
    )

if not run_dir:
    fail(
        "Pipeline finished, but the new output folder could not be "
        "identified.\n\nCheck:\n{}".format(RUNS_ROOT)
    )


# Verify the outputs needed by the Revit buttons.
required_outputs = [
    "result.json",
    "heatmap_cells.csv",
    "floor_mesh.ply"
]

missing_outputs = []

for filename in required_outputs:
    if not os.path.isfile(os.path.join(run_dir, filename)):
        missing_outputs.append(filename)

if missing_outputs:
    fail(
        "Pipeline returned exit code 0, but some expected output files "
        "are missing.\n\n"
        "Run folder:\n{}\n\n"
        "Missing files:\n{}\n\n"
        "Check pipeline.log for details.".format(
            run_dir,
            "\n".join(missing_outputs)
        )
    )


# Save the output folder for Generate Surface to use.
try:
    save_latest_run(run_dir)
except Exception as exc:
    fail(
        "Pipeline succeeded, but latest_run.txt could not be saved.\n\n"
        "Run folder:\n{}\n\nError:\n{}".format(run_dir, exc)
    )


# Show the final result.
output.print_md("")
output.print_md("## Pipeline completed successfully")
output.print_md("**Run folder:** `{}`".format(run_dir))
output.print_md("**Mesh:** `{}`".format(
    os.path.join(run_dir, "floor_mesh.ply")
))
output.print_md("**Heatmap data:** `{}`".format(
    os.path.join(run_dir, "heatmap_cells.csv")
))
output.print_md("**Evaluation result:** `{}`".format(
    os.path.join(run_dir, "result.json")
))
output.print_md("**Latest run pointer:** `{}`".format(LATEST_RUN_TXT))

forms.alert(
    "Pipeline completed successfully!\n\n"
    "Input:\n{}\n\n"
    "Output folder:\n{}\n\n"
    "Created:\n"
    "- floor_mesh.ply\n"
    "- heatmap_cells.csv\n"
    "- result.json\n\n"
    "Next: Generate Surface, then Evaluate.".format(
        os.path.basename(pcd_path),
        run_dir
    ),
    title="Run Pipeline"
)
