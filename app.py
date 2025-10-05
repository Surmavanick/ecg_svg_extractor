import io
import math
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file
import xml.etree.ElementTree as ET
from svg.path import parse_path

app = Flask(__name__)

# ECG calibration (from page footer): 25 mm/s, 10 mm/mV
MM_PER_S = 25.0
MM_PER_MV = 10.0

# Output sampling rate (uniform grid)
TARGET_FS = 500  # Hz (შეგიძლია შეცვალო 250/500-ზე)

# -------------------- Helpers --------------------

def _parse_float_with_unit(val):
    """Return (value, unit) for attributes like '210mm' or '595'."""
    val = str(val).strip()
    num = ''.join(ch for ch in val if (ch.isdigit() or ch in '.-'))
    unit = val[len(num):].strip().lower()
    return float(num), unit

def get_px_per_mm(svg_root):
    """Compute px/mm from <svg width=.. height=.. viewBox=..>."""
    viewBox = svg_root.get("viewBox")
    if not viewBox:
        # Conservative fallback: assume 96 dpi => ~3.7795 px/mm
        return 3.7795275591

    vb = [float(x) for x in viewBox.strip().split()]
    if len(vb) != 4:
        return 3.7795275591
    vb_w = vb[2]  # in "user px"

    width_attr = svg_root.get("width")
    if width_attr:
        width_val, width_unit = _parse_float_with_unit(width_attr)
        if width_unit == "mm":
            # px/mm = viewBox_width_px / width_mm
            return vb_w / max(width_val, 1e-9)
        elif width_unit == "px" or width_unit == "":
            # width given in px; if we also have height in mm we could use that,
            # otherwise assume A4 width=210mm as a reasonable default.
            return (width_val / 210.0)
    # height in mm?
    height_attr = svg_root.get("height")
    if height_attr:
        h_val, h_unit = _parse_float_with_unit(height_attr)
        vb_h = svg_root.get("viewBox")
        if h_unit == "mm":
            vb = [float(x) for x in viewBox.strip().split()]
            vb_h_val = vb[3]
            return vb_h_val / max(h_val, 1e-9)

    # Final fallback
    return 3.7795275591  # 96 dpi

def sample_path_points(d_attr, samples_per_segment=60):
    """Sample a <path d='...'> into Nx2 array of [x_px, y_px]."""
    try:
        path = parse_path(d_attr)
    except Exception:
        return np.empty((0, 2))
    pts = []
    for seg in path:
        for i in range(samples_per_segment):
            t = i / (samples_per_segment - 1)
            z = seg.point(t)
            pts.append([float(z.real), float(z.imag)])
    return np.array(pts)

def pick_12_leads(candidates, view_w):
    """
    candidates: list of dicts with keys:
      'xy' (Nx2), 'y_mean', 'x_min', 'x_range', 'y_std'
    Strategy:
      - take top signals by x_range and y_std (avoid grid/flat lines)
      - order by y_mean (top→bottom) into 4 rows, 3 per row
      - inside each row order by x_min (left→right)
    """
    if len(candidates) <= 12:
        chosen = candidates
    else:
        # Score: prioritize long and variable traces
        scores = []
        for c in candidates:
            score = (c["x_range"] / max(view_w, 1e-6)) + (c["y_std"])
            scores.append(score)
        idx = np.argsort(scores)[-12:]
        chosen = [candidates[i] for i in idx]

    # sort by vertical position
    chosen.sort(key=lambda c: c["y_mean"])
    # chunk into rows of 3
    rows = [chosen[i:i+3] for i in range(0, len(chosen), 3)]
    # if for some reason not multiple of 3, pad last row
    while len(rows) < 4:
        rows.append([])
    rows = rows[:4]
    for r in rows:
        r.sort(key=lambda c: c["x_min"])

    # Map rows to standard names
    lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    ordered = []
    for r in rows:
        ordered.extend(r)
    # If fewer than 12 managed to pass, pad with empties
    while len(ordered) < 12:
        ordered.append(None)
    return ordered, lead_names

def to_seconds_and_mV(xy_px, px_per_mm):
    """Convert [x_px, y_px] -> (t_sec, v_mV) using calibration 25mm/s, 10mm/mV."""
    if xy_px.size == 0:
        return np.array([]), np.array([])
    # Time: px -> mm -> s
    t_sec = (xy_px[:, 0] / px_per_mm) / MM_PER_S
    # Amplitude: baseline ~ median y, px -> mm -> mV (note: SVG y grows downward)
    baseline_px = np.median(xy_px[:, 1])
    v_mV = -((xy_px[:, 1] - baseline_px) / px_per_mm) / MM_PER_MV
    return t_sec, v_mV

def resample_uniform(t, v, fs=TARGET_FS):
    """Resample to uniform time grid at fs (Hz) via linear interpolation."""
    if len(t) < 2:
        return np.array([]), np.array([])
    t0 = float(t.min())
    t1 = float(t.max())
    # Build uniform grid
    dt = 1.0 / fs
    n = int((t1 - t0) / dt) + 1
    t_uni = t0 + np.arange(n) * dt
    # Remove any non-monotonic segments before interp
    order = np.argsort(t)
    t_sorted = t[order]
    v_sorted = v[order]
    # Unique times for np.interp
    t_sorted, unique_idx = np.unique(t_sorted, return_index=True)
    v_sorted = v_sorted[unique_idx]
    v_uni = np.interp(t_uni, t_sorted, v_sorted)
    return t_uni, v_uni

# -------------------- Core extraction --------------------

def extract_csv_from_svg(svg_text, target_fs=TARGET_FS):
    root = ET.fromstring(svg_text)
    px_per_mm = get_px_per_mm(root)

    # get viewBox width for scoring
    vb = [float(x) for x in root.get("viewBox").split()]
    view_w = vb[2] if len(vb) == 4 else 1000.0

    # Collect candidate paths
    ns = {"svg": "http://www.w3.org/2000/svg"}
    paths = root.findall(".//svg:path", ns)

    candidates = []
    for p in paths:
        d = p.get("d")
        if not d:
            continue
        xy = sample_path_points(d, samples_per_segment=60)
        if xy.shape[0] < 30:
            continue
        x_rng = float(xy[:,0].max() - xy[:,0].min())
        y_std = float(np.std(xy[:,1]))
        if x_rng < 0.10 * view_w or y_std < 0.5:  # heuristics: skip tiny/flat shapes (grid, ticks)
            continue
        candidates.append({
            "xy": xy,
            "y_mean": float(xy[:,1].mean()),
            "x_min": float(xy[:,0].min()),
            "x_range": x_rng,
            "y_std": y_std
        })

    ordered, lead_names = pick_12_leads(candidates, view_w)

    # Build uniform, equal-length arrays
    waveforms = []
    min_len = math.inf
    for c in ordered:
        if c is None:
            waveforms.append(np.array([]))
            continue
        t, v = to_seconds_and_mV(c["xy"], px_per_mm)
        t_u, v_u = resample_uniform(t, v, fs=target_fs)
        waveforms.append(v_u)
        if len(v_u) > 0:
            min_len = min(min_len, len(v_u))

    # Truncate all to the shortest length so CSV columns align
    if min_len is math.inf:
        min_len = 0
    data = np.full((min_len, 12), np.nan, dtype=float)
    for j, v in enumerate(waveforms):
        if len(v) >= min_len and min_len > 0:
            data[:, j] = v[:min_len]

    df = pd.DataFrame(data, columns=lead_names)

    mem = io.BytesIO()
    df.to_csv(mem, index=False, float_format="%.5f")
    mem.seek(0)
    return mem

# -------------------- Flask routes --------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "svg_file" not in request.files:
        return "No file part", 400
    f = request.files["svg_file"]
    if not f.filename.lower().endswith(".svg"):
        return "Invalid file type", 400

    svg_text = f.read().decode("utf-8", errors="ignore")
    csv_mem = extract_csv_from_svg(svg_text, target_fs=TARGET_FS)
    out_name = f"{f.filename.rsplit('.',1)[0]}_leads.csv"
    return send_file(csv_mem, as_attachment=True, download_name=out_name, mimetype="text/csv")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
