# -*- coding: utf-8 -*-
"""Accessibility Analyzer for Revit.

Click       : loads the newest analysis result automatically.
Shift+Click : choose another result.json manually.
Shows the overall score, walkable-area heatmap, area breakdown, and practical
recommendations in a compact side panel.
"""
__persistentengine__ = True   # keeps the modeless side panel alive

import os
import io
import csv
import json
import math
import clr
clr.AddReference("PresentationFramework")
clr.AddReference("PresentationCore")
clr.AddReference("WindowsBase")
from System.IO import File, MemoryStream
from System.Collections.Generic import List
from System.Windows import (SystemParameters, WindowStartupLocation, Thickness,
                            CornerRadius, GridLength, GridUnitType, TextWrapping,
                            FontWeights, Visibility, HorizontalAlignment,
                            VerticalAlignment)
from System.Windows.Controls import Grid, ColumnDefinition, Border, TextBlock, StackPanel
from System.Windows.Media import BrushConverter
from System.Windows.Media.Imaging import BitmapImage, BitmapCacheOption
from pyrevit import revit, forms, script
from Autodesk.Revit.DB import (XYZ, Color, FilteredElementCollector, DirectShape,
    BuiltInCategory, Material, TessellatedShapeBuilder, TessellatedShapeBuilderTarget,
    TessellatedShapeBuilderFallback, TessellatedFace, ElementId, GeometryObject)
from Autodesk.Revit.DB import Transaction, Category
# ---- settings -------------------------------------------------------------
RESULT_ROOT = r"D:\Projects\accessibility-capstone\output"
DRAW_SLABS = False   # False = ใช้ point cloud ที่ย้อมสีแทน, True = วาดแผ่นสีเพิ่ม
ROT_DEG = 0.0
OFF_X, OFF_Y, OFF_Z = 0.0, 0.0, 0.0
GRID_M = 0.5  # mesh cell size in meters; smaller = more detail but heavier Revit model
CHUNK = 500
M_TO_FT = 3.280839895
STYLE_NAME = "Accessibility score"
BAR_WIDTH = 300.0
# ---------------------------------------------------------------------------

LEVEL_COLOR = {"critical": "#E51A1A", "very_low": "#FF8C00",
               "low": "#F2E61A", "high": "#1ACC33"}
LEVEL_NAME = {"critical": "Critical Low", "very_low": "Very Low",
              "low": "Low", "high": "High"}

output = script.get_output()
doc = revit.doc
view = revit.active_view

if doc is None:
    forms.alert("Open a project first.", exitscript=True)

try:
    shift = bool(__shiftclick__)
except NameError:
    shift = False

_bc = BrushConverter()


def brush(hexcolor):
    return _bc.ConvertFromString(hexcolor)


def newest_result(root):
    """Find the newest result.json anywhere below RESULT_ROOT."""
    if not os.path.isdir(root):
        return None
    files = []
    for current_root, dirs, filenames in os.walk(root):
        if "result.json" in filenames:
            files.append(os.path.join(current_root, "result.json"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def transform(x, y, z):
    r = math.radians(ROT_DEG)
    xr = x * math.cos(r) - y * math.sin(r)
    yr = x * math.sin(r) + y * math.cos(r)
    return XYZ((xr + OFF_X) * M_TO_FT, (yr + OFF_Y) * M_TO_FT,
               (z + OFF_Z + 0.05) * M_TO_FT)


def get_or_create_material(name, rgb, transparency=28):
    """Create/reuse a Revit material for a heatmap band."""
    for mat in FilteredElementCollector(doc).OfClass(Material):
        if mat.Name == name:
            mat.Color = Color(rgb[0], rgb[1], rgb[2])
            mat.Transparency = transparency
            return mat.Id
    mid = Material.Create(doc, name)
    mat = doc.GetElement(mid)
    mat.Color = Color(rgb[0], rgb[1], rgb[2])
    mat.Transparency = transparency
    return mid


def band_for_score(score):
    if score <= 15.0:
        return "critical"
    if score <= 30.0:
        return "very_low"
    if score <= 50.0:
        return "low"
    return "high"


def generic_model_cat_id():
    return Category.GetCategory(doc, BuiltInCategory.OST_GenericModel).Id


def build_shape(faces, material_id):
    """Build mesh geometry from square cells (one connected face set per cell)."""
    builder = TessellatedShapeBuilder()
    builder.Target = TessellatedShapeBuilderTarget.Mesh
    builder.Fallback = TessellatedShapeBuilderFallback.Mesh
    for points in faces:
        builder.OpenConnectedFaceSet(False)
        pts = List[XYZ]()
        for p in points:
            pts.Add(p)
        builder.AddFace(TessellatedFace(pts, material_id))
        builder.CloseConnectedFaceSet()
    builder.Build()
    geometry = builder.GetBuildResult().GetGeometricalObjects()
    if geometry is None or geometry.Count == 0:
        raise Exception("builder returned no geometry")
    return geometry


def place_shape(geometry, band_name, index):
    ds = DirectShape.CreateElement(doc, generic_model_cat_id())
    ds.ApplicationId = "AccessibilityAnalyzer"
    ds.ApplicationDataId = "AccessibilityHeatmap-%s-%d" % (band_name, index)
    ds.SetShape(geometry)
    try:
        ds.Name = "Accessibility Heatmap - %s %d" % (
            band_name.replace("_", " ").title(), index)
    except Exception:
        pass
    return ds

# ---- choose the file ------------------------------------------------------
path = None
if not shift:
    path = newest_result(RESULT_ROOT)
if path is None:
    path = forms.pick_file(file_ext="json", title="Select result.json")
if not path:
    script.exit()

result_dir = os.path.dirname(path)
scan_name = os.path.basename(result_dir)

try:
    with io.open(path, encoding="utf-8") as f:
        data = json.load(f)
except Exception as e:
    forms.alert("Cannot read result.json:\n%s\n\n%s" % (path, e), exitscript=True)

summary = data.get("summary", {})
recs = data.get("recommendations", [])
notes = data.get("notes", [])
if not isinstance(summary, dict):
    forms.alert("Invalid result.json: summary must be an object.", exitscript=True)
if not isinstance(recs, list):
    recs = []
if not isinstance(notes, list):
    notes = []

output.print_md("### Accessibility Analyzer - Run details")
output.print_md("**Result file:** `%s`" % path)
output.print_md("**Result folder:** `%s`" % result_dir)
output.print_md("**Heatmap image exists:** `%s`" % os.path.exists(os.path.join(result_dir, "heatmap.png")))

# Prefer the full set of walkable cells exported from scores.npz by 07_export.py.
# If the CSV is missing, fall back to the low-access-only cells in result.json.
heatmap_csv = os.path.join(result_dir, "heatmap_cells.csv")
cells = []
if os.path.exists(heatmap_csv):
    try:
        # IronPython 2.7's csv module expects binary mode.
        with open(heatmap_csv, "rb") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    cells.append({"x": float(row["x"]), "y": float(row["y"]),
                                  "z": float(row["z"]), "score": float(row["score"]),
                                  "reason": ""})
                except (KeyError, TypeError, ValueError):
                    continue
        output.print_md("**Heatmap source:** `heatmap_cells.csv`")
        output.print_md("**Full heatmap cells loaded:** `%s`" % len(cells))
    except Exception as e:
        output.print_md("**CSV read error:** `%s`" % e)
        cells = []
else:
    output.print_md("**Heatmap source:** `result.json / low_access_cells`")
    output.print_md("**Note:** Full heatmap CSV was not found, so only low-access cells will be shown. Run the export step to enable the full heatmap.")
    cells = data.get("low_access_cells", [])

if not isinstance(cells, list):
    cells = []
valid_cells = []
for c in cells:
    try:
        x, y, z, score = float(c["x"]), float(c["y"]), float(c["z"]), float(c["score"])
        if any([math.isnan(v) or math.isinf(v) for v in (x, y, z, score)]):
            continue
        valid_cells.append({"x": x, "y": y, "z": z,
                            "score": max(0.0, min(100.0, score)),
                            "reason": c.get("reason", "")})
    except (KeyError, TypeError, ValueError):
        continue
cells = valid_cells

required = ["total_walkable_area_m2", "low_access_area_m2", "low_access_percent"]
missing = [key for key in required if key not in summary]
if missing:
    forms.alert("Missing fields in result.json: %s\nPlease re-run 07_export.py." % ", ".join(missing), exitscript=True)
try:
    total_area = max(0.0, float(summary["total_walkable_area_m2"]))
    low_total = max(0.0, float(summary["low_access_area_m2"]))
    low_percent = max(0.0, min(100.0, float(summary["low_access_percent"])))
except Exception as e:
    forms.alert("Invalid area values in result.json:\n%s" % e, exitscript=True)

# ---- area per band --------------------------------------------------------
bands = summary.get("bands", {"critical": 15, "very_low": 30, "low": 50, "high": 100})
cs = float(summary.get("cell_size_m", 0.1))
if cs <= 0:
    cs = 0.1
unit = cs * cs
n_crit = n_vl = n_low = 0
for c in cells:
    score = c["score"]
    if score <= float(bands.get("critical", 15)):
        n_crit += 1
    elif score <= float(bands.get("very_low", 30)):
        n_vl += 1
    elif score <= float(bands.get("low", 50)):
        n_low += 1
area_crit, area_vl, area_low = n_crit * unit, n_vl * unit, n_low * unit

# Prefer exact exported band areas when available (they include all walkable cells).
band_areas = summary.get("bands_area_m2", {})
if isinstance(band_areas, dict) and band_areas:
    area_crit = float(band_areas.get("critical", area_crit))
    area_vl = float(band_areas.get("very_low", area_vl))
    area_low = float(band_areas.get("low", area_low))
    area_high = float(band_areas.get("high", max(total_area - low_total, 0.0)))
else:
    area_high = max(total_area - low_total, 0.0)

# ---- merge cells into tiles, then into big rectangles ----------------------
from Autodesk.Revit.DB import (Transaction, CurveLoop, Line,
                               GeometryCreationUtilities, GeometryObject,
                               FillPatternElement, OverrideGraphicSettings, Category)

CELL_M = 0.25        # size of one heatmap tile (metres)
SLAB_FT = 0.04       # slab thickness (feet, ~1.2 cm)
MAX_SOLIDS = 150     # solids per DirectShape element
TRANSPARENCY = 35    # 0 = opaque, 100 = invisible

mesh_colors = {
    "critical": (229, 26, 26),
    "very_low": (255, 140, 0),
    "low": (242, 230, 26),
    "high": (26, 204, 51),
}

acc = {}
for c in cells:
    key = (int(math.floor(c["x"] / CELL_M)),
           int(math.floor(c["y"] / CELL_M)),
           int(math.floor(c["z"] / CELL_M)))
    a = acc.get(key)
    if a is None:
        acc[key] = [c["z"], c["score"], 1]
    else:
        a[0] += c["z"]
        a[1] += c["score"]
        a[2] += 1

tiles = {}   # (band, z-level) -> {(ix, iy): z}
items = []
for (ix, iy, iz), a in acc.items():
    n = float(a[2])
    z = a[0] / n
    s = a[1] / n
    tiles.setdefault((band_for_score(s), iz), {})[(ix, iy)] = z
    items.append(((ix + 0.5) * CELL_M, (iy + 0.5) * CELL_M, z, s))
output.print_md("**Heatmap tiles (%.2f m):** `%s`" % (CELL_M, len(items)))


def merge_rects(cellmap):
    """{(ix, iy): z} -> [(ix0, ix1, iy0, iy1, z)] with ix1/iy1 exclusive."""
    rows = {}
    for (ix, iy) in cellmap:
        rows.setdefault(iy, []).append(ix)
    runs = {}
    for iy, xs in rows.items():
        xs.sort()
        start = prev = xs[0]
        lst = []
        for x in xs[1:]:
            if x == prev + 1:
                prev = x
            else:
                lst.append((start, prev + 1))
                start = prev = x
        lst.append((start, prev + 1))
        runs[iy] = lst

    rects = []
    open_runs = {}

    def close(keys, end_iy):
        for k in keys:
            y0 = open_runs.pop(k)
            rects.append((k[0], k[1], y0, end_iy))

    last_iy = None
    for iy in sorted(runs):
        if last_iy is not None and iy != last_iy + 1:
            close(list(open_runs.keys()), last_iy + 1)
        cur = set(runs[iy])
        close([k for k in list(open_runs.keys()) if k not in cur], iy)
        for k in cur:
            if k not in open_runs:
                open_runs[k] = iy
        last_iy = iy
    if last_iy is not None:
        close(list(open_runs.keys()), last_iy + 1)

    out = []
    for (x0, x1, y0, y1) in rects:
        zs = [cellmap[(ix, iy)] for ix in range(x0, x1) for iy in range(y0, y1)
              if (ix, iy) in cellmap]
        out.append((x0, x1, y0, y1, sum(zs) / len(zs)))
    return out


def make_slab(x0, x1, y0, y1, z):
    a = transform(x0 * CELL_M, y0 * CELL_M, z)
    b = transform(x1 * CELL_M, y0 * CELL_M, z)
    c = transform(x1 * CELL_M, y1 * CELL_M, z)
    d = transform(x0 * CELL_M, y1 * CELL_M, z)
    loop = CurveLoop()
    loop.Append(Line.CreateBound(a, b))
    loop.Append(Line.CreateBound(b, c))
    loop.Append(Line.CreateBound(c, d))
    loop.Append(Line.CreateBound(d, a))
    loops = List[CurveLoop]()
    loops.Add(loop)
    return GeometryCreationUtilities.CreateExtrusionGeometry(loops, XYZ.BasisZ, SLAB_FT)


def solid_fill_id():
    for fp in FilteredElementCollector(doc).OfClass(FillPatternElement):
        try:
            if fp.GetFillPattern().IsSolidFill:
                return fp.Id
        except Exception:
            pass
    return ElementId.InvalidElementId


def make_override(rgb, fill_id):
    ogs = OverrideGraphicSettings()
    if fill_id != ElementId.InvalidElementId:
        ogs.SetSurfaceForegroundPatternId(fill_id)
    ogs.SetSurfaceForegroundPatternColor(Color(rgb[0], rgb[1], rgb[2]))
    ogs.SetSurfaceTransparency(TRANSPARENCY)
    return ogs

def remove_old_slabs():
    t = Transaction(doc, "Remove old heatmap slabs")
    t.Start()
    try:
        ids = [e.Id for e in FilteredElementCollector(doc).OfClass(DirectShape)
               if e.ApplicationId == "AccessibilityAnalyzer"]
        for oid in ids:
            doc.Delete(oid)
        t.Commit()
        output.print_md("Removed %d old heatmap slabs" % len(ids))
    except Exception as e:
        output.print_md("Could not remove old slabs: `%s`" % e)
        try:
            t.RollBack()
        except Exception:
            pass
        
# ---- build the coloured slabs in the 3D view -------------------------------
if items:
    ORDER = ("critical", "very_low", "low", "high")
    created = 0
    failed = 0
    override_failed = 0
    first_err = None
    total_rects = 0

    t = Transaction(doc, "Accessibility Heatmap")
    t.Start()
    try:
        # remove slabs from earlier runs (collect ids first, then delete)
        try:
            old_ids = [e.Id for e in FilteredElementCollector(doc).OfClass(DirectShape)
                       if e.ApplicationId == "AccessibilityAnalyzer"]
            for oid in old_ids:
                doc.Delete(oid)
            output.print_md("Removed %d old heatmap elements" % len(old_ids))
        except Exception as e:
            output.print_md("**Could not remove old elements:** `%s`" % e)

        fill_id = solid_fill_id()
        gen_cat = Category.GetCategory(doc, BuiltInCategory.OST_GenericModel).Id

        for band_name in ORDER:
            rects = []
            for (b, iz), cmap in tiles.items():
                if b == band_name:
                    rects.extend(merge_rects(cmap))
            total_rects += len(rects)
            ogs = make_override(mesh_colors[band_name], fill_id)
            for s0 in range(0, len(rects), MAX_SOLIDS):
                try:
                    geom = List[GeometryObject]()
                    for r in rects[s0:s0 + MAX_SOLIDS]:
                        geom.Add(make_slab(r[0], r[1], r[2], r[3], r[4]))
                    ds = DirectShape.CreateElement(doc, gen_cat)
                    ds.ApplicationId = "AccessibilityAnalyzer"
                    ds.ApplicationDataId = "AccessibilityHeatmap-%s-%d" % (band_name, s0)
                    ds.SetShape(geom)
                    created += 1
                    try:
                        view.SetElementOverrides(ds.Id, ogs)
                    except Exception as e:
                        override_failed += 1
                        if first_err is None:
                            first_err = "override: %s" % e
                except Exception as e:
                    failed += 1
                    if first_err is None:
                        first_err = "%s chunk %d: %s" % (band_name, s0, e)
            output.print_md("- %s: %d rectangles" % (band_name, len(rects)))

        output.print_md("**Rectangles:** %d  |  **Elements created:** %d, failed: %d, "
                        "colour override failed: %d" % (total_rects, created, failed,
                                                        override_failed))
        if first_err:
            output.print_md("First error: `%s`" % first_err)
        t.Commit()
    except Exception as e:
        output.print_md("**Heatmap build error:** `%s`" % e)
        try:
            t.RollBack()
        except Exception:
            pass
else:
    output.print_md("No valid heatmap cells were found. The summary panel will still be shown.")
# ---- side panel -----------------------------------------------------------
# NOTE: keep this XAML ASCII-only (IronPython 2.7 str); use XML entities.
XAML = """
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="Accessibility Analyzer" Width="380" SizeToContent="Height"
        WindowStyle="None" AllowsTransparency="True" Background="Transparent"
        ResizeMode="NoResize" ShowInTaskbar="False" FontFamily="Segoe UI">
  <Window.Resources>

  </Window.Resources>

  <Border Margin="12" CornerRadius="14" Background="#10151C"
          BorderBrush="#263241" BorderThickness="1">
    <Border.Effect>
      <DropShadowEffect BlurRadius="18" ShadowDepth="2" Opacity="0.55" Color="Black"/>
    </Border.Effect>
    <DockPanel>

      <Border x:Name="Header" DockPanel.Dock="Top" Background="#151C26"
              CornerRadius="13,13,0,0" Padding="18,14,14,14">
        <Grid>
          <StackPanel>
            <TextBlock Text="ACCESSIBILITY OVERVIEW" FontSize="13" FontWeight="Bold"
                       Foreground="#4FC3F7"/>
            <TextBlock x:Name="TxtScan" FontSize="11" Foreground="#7D8A99"
                       Margin="0,3,0,0"/>
          </StackPanel>
          <TextBlock x:Name="BtnClose" Text="&#x2715;" FontSize="14" Foreground="#7D8A99"
                     HorizontalAlignment="Right" VerticalAlignment="Top" Cursor="Hand"/>
        </Grid>
      </Border>

      <ScrollViewer VerticalScrollBarVisibility="Auto">
        <StackPanel Margin="18,16,18,0">

          <TextBlock Text="YOUR ACCESSIBILITY SCORE" FontSize="10"
                     FontWeight="SemiBold" Foreground="#7D8A99"/>
          <StackPanel Orientation="Horizontal" Margin="0,2,0,0">
            <TextBlock x:Name="TxtOverall" FontSize="44" FontWeight="Bold"
                       Foreground="#E6EDF3"/>
            <TextBlock Text="/100" FontSize="16" Foreground="#7D8A99"
                       VerticalAlignment="Bottom" Margin="4,0,0,10"/>
            <Border x:Name="ChipLevel" CornerRadius="10" Padding="10,3"
                    VerticalAlignment="Bottom" Margin="14,0,0,12" Background="#3A4656">
              <TextBlock x:Name="TxtLevel" FontSize="11" FontWeight="SemiBold"
                         Foreground="#10151C"/>
            </Border>
          </StackPanel>
          <Grid Width="300" Height="14" HorizontalAlignment="Left" Margin="0,8,0,0">
            <Border Height="8" CornerRadius="4" VerticalAlignment="Center">
              <Border.Background>
                <LinearGradientBrush StartPoint="0,0" EndPoint="1,0">
                  <GradientStop Color="#E51A1A" Offset="0"/>
                  <GradientStop Color="#E51A1A" Offset="0.12"/>
                  <GradientStop Color="#FF8C00" Offset="0.18"/>
                  <GradientStop Color="#FF8C00" Offset="0.27"/>
                  <GradientStop Color="#F2E61A" Offset="0.33"/>
                  <GradientStop Color="#F2E61A" Offset="0.46"/>
                  <GradientStop Color="#1ACC33" Offset="0.55"/>
                  <GradientStop Color="#1ACC33" Offset="1"/>
                </LinearGradientBrush>
              </Border.Background>
            </Border>
            <Border x:Name="ScoreMarker" Width="4" Height="14" CornerRadius="2"
                    Background="White" HorizontalAlignment="Left"/>
          </Grid>
          <TextBlock x:Name="TxtLow" FontSize="12" Foreground="#C5CED8"
                     Margin="0,10,0,0" TextWrapping="Wrap"/>

          <TextBlock Text="WALKABLE-AREA HEATMAP" FontSize="10" FontWeight="SemiBold"
                     Foreground="#7D8A99" Margin="0,20,0,0"/>
          <Border x:Name="BorderHeat" Background="#0B0F14" CornerRadius="10"
                  Padding="6" Margin="0,8,0,0">
            <Image x:Name="ImgHeat" Stretch="Uniform" MaxHeight="220"/>
          </Border>
          <TextBlock x:Name="TxtHeatCaption" FontSize="10" Foreground="#566273"
                     Margin="0,6,0,0" TextWrapping="Wrap"/>

          <TextBlock Text="AREA BREAKDOWN" FontSize="10" FontWeight="SemiBold"
                     Foreground="#7D8A99" Margin="0,18,0,8"/>
          <Grid Margin="0,0,0,7">
            <Grid.ColumnDefinitions>
              <ColumnDefinition Width="22"/><ColumnDefinition Width="*"/><ColumnDefinition Width="Auto"/>
            </Grid.ColumnDefinitions>
            <Ellipse Width="12" Height="12" Fill="#E51A1A" HorizontalAlignment="Left"/>
            <StackPanel Grid.Column="1" Orientation="Horizontal">
              <TextBlock Text="Critical Low" Foreground="#E6EDF3" FontSize="12"/>
              <TextBlock Text="  0-15%" Foreground="#6B7888" FontSize="11" VerticalAlignment="Center"/>
            </StackPanel>
            <TextBlock x:Name="TxtCrit" Grid.Column="2" Foreground="#E6EDF3" FontSize="12" FontWeight="SemiBold"/>
          </Grid>
          <Grid Margin="0,0,0,7">
            <Grid.ColumnDefinitions>
              <ColumnDefinition Width="22"/><ColumnDefinition Width="*"/><ColumnDefinition Width="Auto"/>
            </Grid.ColumnDefinitions>
            <Ellipse Width="12" Height="12" Fill="#FF8C00" HorizontalAlignment="Left"/>
            <StackPanel Grid.Column="1" Orientation="Horizontal">
              <TextBlock Text="Very Low" Foreground="#E6EDF3" FontSize="12"/>
              <TextBlock Text="  16-30%" Foreground="#6B7888" FontSize="11" VerticalAlignment="Center"/>
            </StackPanel>
            <TextBlock x:Name="TxtVLow" Grid.Column="2" Foreground="#E6EDF3" FontSize="12" FontWeight="SemiBold"/>
          </Grid>
          <Grid Margin="0,0,0,7">
            <Grid.ColumnDefinitions>
              <ColumnDefinition Width="22"/><ColumnDefinition Width="*"/><ColumnDefinition Width="Auto"/>
            </Grid.ColumnDefinitions>
            <Ellipse Width="12" Height="12" Fill="#F2E61A" HorizontalAlignment="Left"/>
            <StackPanel Grid.Column="1" Orientation="Horizontal">
              <TextBlock Text="Low" Foreground="#E6EDF3" FontSize="12"/>
              <TextBlock Text="  31-50%" Foreground="#6B7888" FontSize="11" VerticalAlignment="Center"/>
            </StackPanel>
            <TextBlock x:Name="TxtLowBand" Grid.Column="2" Foreground="#E6EDF3" FontSize="12" FontWeight="SemiBold"/>
          </Grid>
          <Grid>
            <Grid.ColumnDefinitions>
              <ColumnDefinition Width="22"/><ColumnDefinition Width="*"/><ColumnDefinition Width="Auto"/>
            </Grid.ColumnDefinitions>
            <Ellipse Width="12" Height="12" Fill="#1ACC33" HorizontalAlignment="Left"/>
            <StackPanel Grid.Column="1" Orientation="Horizontal">
              <TextBlock Text="High" Foreground="#E6EDF3" FontSize="12"/>
              <TextBlock Text="  51-100%" Foreground="#6B7888" FontSize="11" VerticalAlignment="Center"/>
            </StackPanel>
            <TextBlock x:Name="TxtHigh" Grid.Column="2" Foreground="#E6EDF3" FontSize="12" FontWeight="SemiBold"/>
          </Grid>

          <Border BorderBrush="#243040" BorderThickness="0,1,0,0" Margin="0,16,0,0"
                  Padding="0,10,0,0">
            <Grid>
              <TextBlock Text="Total walkable area" Foreground="#7D8A99" FontSize="12"/>
              <TextBlock x:Name="TxtTotal" HorizontalAlignment="Right"
                         Foreground="#E6EDF3" FontSize="12" FontWeight="SemiBold"/>
            </Grid>
          </Border>

          <TextBlock Text="SUGGESTED IMPROVEMENTS" FontSize="10" FontWeight="SemiBold"
                     Foreground="#7D8A99" Margin="0,20,0,2"/>
          <TextBlock Text="Priority areas to review. Confirm each suggestion against the applicable accessibility standards."
                     FontSize="10" Foreground="#566273" TextWrapping="Wrap" Margin="0,0,0,10"/>
          <StackPanel x:Name="RecPanel"/>
          <StackPanel x:Name="NotePanel" Margin="0,0,0,4"/>

          <Border Background="#151C26" BorderBrush="#263241" BorderThickness="1"
                  CornerRadius="8" Padding="10" Margin="0,14,0,16">
            <TextBlock Text="Prototype assessment: scores are estimates. Use them to guide review, not as a substitute for an accessibility inspection."
                       FontSize="10" Foreground="#9AA7B5" TextWrapping="Wrap"/>
          </Border>
        </StackPanel>
      </ScrollViewer>
    </DockPanel>
  </Border>
</Window>
"""


def fmt_area(v):
    return u"%.1f m\u00b2" % v


def level_from_score(s):
    if s <= bands["critical"]:
        return "critical"
    if s <= bands["very_low"]:
        return "very_low"
    if s <= bands["low"]:
        return "low"
    return "high"


class ResultPanel(forms.WPFWindow):
    def __init__(self, folder, scan, summ, n_points):
        forms.WPFWindow.__init__(self, XAML, literal_string=True)
        self.folder = folder

        short = scan if len(scan) <= 34 else scan[:31] + u"..."
        self.TxtScan.Text = u"Analysis: " + short

        # overall score
        overall = summ.get("overall_score")
        if overall is None:
            self.TxtOverall.Text = u"--"
            self.TxtLevel.Text = u"n/a"
            self.ScoreMarker.Visibility = Visibility.Collapsed
        else:
            overall = float(overall)
            lvl = summ.get("overall_level") or level_from_score(overall)
            self.TxtOverall.Text = u"%.0f" % overall
            self.TxtLevel.Text = LEVEL_NAME.get(lvl, lvl)
            self.ChipLevel.Background = brush(LEVEL_COLOR.get(lvl, "#3A4656"))
            self.TxtOverall.Foreground = brush(LEVEL_COLOR.get(lvl, "#E6EDF3"))
            x = max(0.0, min(BAR_WIDTH - 4.0, BAR_WIDTH * overall / 100.0 - 2.0))
            self.ScoreMarker.Margin = Thickness(x, 0, 0, 0)

        self.TxtLow.Text = u"Needs improvement: %s of %s walkable area (%.1f%%)" % (
            fmt_area(float(summ["low_access_area_m2"])),
            fmt_area(float(summ["total_walkable_area_m2"])),
            float(summ["low_access_percent"]))

        # heatmap image
        heat = os.path.join(folder, "heatmap.png")
        heat_ok = False
        if os.path.exists(heat):
            try:
                ms = MemoryStream(File.ReadAllBytes(heat))
                bmp = BitmapImage()
                bmp.BeginInit()
                bmp.CacheOption = BitmapCacheOption.OnLoad
                bmp.StreamSource = ms
                bmp.EndInit()
                bmp.Freeze()
                ms.Close()
                self.ImgHeat.Source = bmp
                heat_ok = True
            except Exception as e:
                output.print_md("Could not load heatmap.png: `%s`" % e)
        if heat_ok:
            self.TxtHeatCaption.Text = (u"Red = lower accessibility score; yellow = moderate; "
                                        u"green = higher score. Recommendations are listed below.")
        else:
            self.BorderHeat.Visibility = Visibility.Collapsed
            self.TxtHeatCaption.Text = u"Heatmap preview is unavailable. Re-run the export step to create heatmap.png."

        # breakdown
        self.TxtCrit.Text = fmt_area(area_crit)
        self.TxtVLow.Text = fmt_area(area_vl)
        self.TxtLowBand.Text = fmt_area(area_low)
        self.TxtHigh.Text = fmt_area(area_high)
        self.TxtTotal.Text = fmt_area(float(summ["total_walkable_area_m2"]))

        # recommendations
        for rec in recs:
            self.add_rec(rec)
        if overall is None:
            notes_all = [{"level": "info", "text":
                          "Re-run 06_score.py and 07_export.py to get the overall "
                          "score and recommendations."}] + list(notes)
        else:
            notes_all = notes
        for n in notes_all:
            self.add_note(n)

        # position
        self.MaxHeight = SystemParameters.WorkArea.Height - 20
        self.WindowStartupLocation = WindowStartupLocation.Manual
        self.Left = SystemParameters.WorkArea.Right - self.Width - 4
        self.Top = SystemParameters.WorkArea.Top + 10

        self.Header.MouseLeftButtonDown += self.on_drag
        self.BtnClose.MouseLeftButtonUp += self.on_close

    def add_rec(self, rec):
        color = LEVEL_COLOR.get(rec.get("level", "low"), "#F2E61A")
        row = Grid()
        row.ColumnDefinitions.Add(ColumnDefinition(Width=GridLength(30)))
        row.ColumnDefinitions.Add(ColumnDefinition(Width=GridLength(1, GridUnitType.Star)))
        badge_txt = TextBlock(Text=str(rec.get("rank", "")), FontSize=11,
                              FontWeight=FontWeights.Bold,
                              Foreground=brush("#10151C"),
                              HorizontalAlignment=HorizontalAlignment.Center,
                              VerticalAlignment=VerticalAlignment.Center)
        badge = Border(Width=22, Height=22, CornerRadius=CornerRadius(11),
                       Background=brush(color), Child=badge_txt,
                       HorizontalAlignment=HorizontalAlignment.Left,
                       VerticalAlignment=VerticalAlignment.Top)
        row.Children.Add(badge)
        body = StackPanel()
        body.Children.Add(TextBlock(Text=rec.get("title", ""), FontSize=12,
                                    FontWeight=FontWeights.SemiBold,
                                    Foreground=brush("#E6EDF3")))
        body.Children.Add(TextBlock(Text=rec.get("detail", ""), FontSize=11,
                                    Foreground=brush("#9AA7B5"),
                                    TextWrapping=TextWrapping.Wrap,
                                    Margin=Thickness(0, 3, 0, 0)))
        Grid.SetColumn(body, 1)
        row.Children.Add(body)
        card = Border(Background=brush("#161D28"), CornerRadius=CornerRadius(8),
                      Padding=Thickness(10), Margin=Thickness(0, 0, 0, 8), Child=row)
        self.RecPanel.Children.Add(card)

    def add_note(self, note):
        warn = note.get("level") == "warning"
        txt = TextBlock(Text=u"\u2022 " + note.get("text", ""), FontSize=11,
                        Foreground=brush("#E3B341" if warn else "#7D8A99"),
                        TextWrapping=TextWrapping.Wrap,
                        Margin=Thickness(0, 0, 0, 6))
        self.NotePanel.Children.Add(txt)

    def on_drag(self, sender, args):
        try:
            self.DragMove()
        except Exception:
            pass

    def on_close(self, sender, args):
        self.Close()



try:
    panel = ResultPanel(result_dir, scan_name, summary, len(items))
    panel.show(modal=False)
except Exception as e:
    forms.alert("Side panel failed: %s\n\nLow access: %s m2 (%s%%)" % (
        e, summary["low_access_area_m2"], summary["low_access_percent"]))