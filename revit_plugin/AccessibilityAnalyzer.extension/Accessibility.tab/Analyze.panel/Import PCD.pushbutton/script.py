
# -*- coding: utf-8 -*-
"""Import / Select PCD
Select a PCD file, copy it into the project, and save its path
for the accessibility pipeline to use later.

Designed for pyRevit / Revit 2027.
"""

import os
import io
import shutil
from datetime import datetime

from pyrevit import forms, script


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = r"D:\Projects\accessibility-capstone"

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMPORT_DIR = os.path.join(DATA_DIR, "imports")

# The pipeline integration can read this file in a later step.
SELECTED_FILE_TXT = os.path.join(DATA_DIR, "selected_pcd.txt")


# ============================================================
# HELPERS
# ============================================================

def stop_with_message(message):
    forms.alert(message, title="Import PCD", exitscript=True)


def read_pcd_header(file_path):
    """Read the PCD header without loading the entire point cloud."""

    header_lines = []

    with open(file_path, "rb") as source:
        for _ in range(100):
            raw_line = source.readline()

            if not raw_line:
                break

            try:
                line = raw_line.decode("ascii", "replace").strip()
            except Exception:
                line = str(raw_line).strip()

            header_lines.append(line)

            if line.upper().startswith("DATA "):
                break

    return header_lines


def validate_pcd(file_path):
    """Basic PCD header validation. Supports ASCII and binary PCD."""

    if not os.path.isfile(file_path):
        return False, "The selected file does not exist."

    if os.path.getsize(file_path) == 0:
        return False, "The selected file is empty."

    header = read_pcd_header(file_path)

    fields = None
    points = None
    data_format = None
    has_version = False

    for line in header:
        upper = line.upper()

        if upper.startswith("VERSION "):
            has_version = True

        elif upper.startswith("FIELDS "):
            fields = upper.split()[1:]

        elif upper.startswith("POINTS "):
            try:
                points = int(upper.split()[1])
            except Exception:
                points = None

        elif upper.startswith("DATA "):
            parts = upper.split()
            if len(parts) >= 2:
                data_format = parts[1]

    if not has_version or fields is None or data_format is None:
        return False, (
            "The file does not appear to have a complete PCD header.\n"
            "Please select a valid .pcd file."
        )

    if not all(axis in fields for axis in ("X", "Y", "Z")):
        return False, (
            "The PCD file must contain X, Y, and Z fields.\n"
            "Detected fields: {}".format(fields)
        )

    if data_format not in ("ASCII", "BINARY", "BINARY_COMPRESSED"):
        return False, (
            "Unsupported PCD DATA format: {}".format(data_format)
        )

    if points is not None and points <= 0:
        return False, "The PCD header reports zero points."

    return True, {
        "points": points,
        "fields": fields,
        "data_format": data_format,
        "size_bytes": os.path.getsize(file_path)
    }


def unique_destination(source_path):
    """Avoid overwriting an existing imported file."""

    filename = os.path.basename(source_path)
    destination = os.path.join(IMPORT_DIR, filename)

    if not os.path.exists(destination):
        return destination

    stem, extension = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    destination = os.path.join(
        IMPORT_DIR,
        "{}_{}{}".format(stem, timestamp, extension)
    )

    counter = 1

    while os.path.exists(destination):
        destination = os.path.join(
            IMPORT_DIR,
            "{}_{}_{}{}".format(stem, timestamp, counter, extension)
        )
        counter += 1

    return destination


def format_size(size_bytes):
    if size_bytes >= 1024 * 1024 * 1024:
        return "{:.2f} GB".format(
            float(size_bytes) / (1024 * 1024 * 1024)
        )

    if size_bytes >= 1024 * 1024:
        return "{:.2f} MB".format(
            float(size_bytes) / (1024 * 1024)
        )

    if size_bytes >= 1024:
        return "{:.2f} KB".format(
            float(size_bytes) / 1024
        )

    return "{} bytes".format(size_bytes)


# ============================================================
# MAIN
# ============================================================

if not os.path.isdir(PROJECT_ROOT):
    stop_with_message(
        "Project folder was not found:\n\n{}".format(PROJECT_ROOT)
    )

if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.isdir(IMPORT_DIR):
    os.makedirs(IMPORT_DIR)


# 1. Select the source PCD.
source_path = forms.pick_file(
    file_ext="pcd",
    multi_file=False,
    title="Select a Point Cloud (.pcd)"
)

if not source_path:
    script.exit()


# 2. Validate the selected file.
is_valid, info = validate_pcd(source_path)

if not is_valid:
    stop_with_message(
        "Cannot import this file.\n\n{}".format(info)
    )


# 3. Copy the file into the project's imports folder.
source_abs = os.path.normcase(os.path.abspath(source_path))

if os.path.dirname(source_abs) == os.path.normcase(
        os.path.abspath(IMPORT_DIR)):
    # Already inside the import folder; use it directly.
    destination_path = source_path
else:
    destination_path = unique_destination(source_path)

    try:
        shutil.copy2(source_path, destination_path)
    except Exception as ex:
        stop_with_message(
            "Could not copy the PCD file.\n\n{}".format(ex)
        )


# 4. Save the selected file path for future pipeline integration.
try:
    with io.open(SELECTED_FILE_TXT, "w", encoding="utf-8") as output_file:
        output_file.write(
            os.path.abspath(destination_path).replace("\\", "/")
        )
        output_file.write(u"\n")
except Exception as ex:
    stop_with_message(
        "The PCD was imported, but its path could not be saved.\n\n"
        "{}".format(ex)
    )


# 5. Report the result.
output = script.get_output()

output.print_md("# Import PCD — Complete")
output.print_md(
    "**Source:** `{}`".format(source_path)
)
output.print_md(
    "**Imported file:** `{}`".format(destination_path)
)
output.print_md(
    "**Point count in header:** {}".format(
        "{:,}".format(info["points"])
        if info["points"] is not None
        else "Not specified"
    )
)
output.print_md(
    "**Format:** `{}`".format(info["data_format"])
)
output.print_md(
    "**File size:** {}".format(format_size(info["size_bytes"]))
)
output.print_md(
    "**Saved path:** `{}`".format(SELECTED_FILE_TXT)
)

forms.alert(
    "PCD imported successfully!\n\n"
    "File: {}\n\n"
    "Format: {}\n"
    "Header point count: {}\n"
    "Size: {}\n\n"
    "The selected path has been saved for pipeline integration."
    .format(
        os.path.basename(destination_path),
        info["data_format"],
        "{:,}".format(info["points"])
        if info["points"] is not None
        else "Not specified",
        format_size(info["size_bytes"])
    ),
    title="Import PCD"
)
