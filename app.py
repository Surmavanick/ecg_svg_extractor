import io
import json
import math
import re
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file
import xml.etree.ElementTree as ET
from svg.path import parse_path

app = Flask(__name__)

# --- ნაგულისხმევი კალიბრაცია ---
MM_PER_S_DEFAULT = 25.0
MM_PER_MV_DEFAULT = 10.0
TARGET_FS = 500  # Hz

LEAD_LABELS_STD = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

# -------------------- SVG utilities --------------------

def _parse_float_with_unit(val):
    val = str(val).strip()
    num = ''.join(ch for ch in val if (ch.isdigit() or ch in '.-'))
    unit = val[len(num):].strip().lower()
    return float(num), unit

def _sample_path_points(d_attr, samples_per_segment=80):
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

def _get_viewbox(root):
    vb = root.get("viewBox")
    if not vb:
        return (0, 0, 1000, 1000)
    a = [float(x) for x in vb.strip().split()]
    return tuple(a) if len(a) == 4 else (0,0,1000,1000)

# -------------------- Grid detection --------------------

def _median_neighbor_spacing(values):
    v = np.sort(np.unique(np.round(values, 3)))
    if len(v) < 2: return None
    deltas = np.diff(v)
    q1, q3 = np.percentile(deltas, [25, 75])
    iqr = q3 - q1
    lo, hi = max(1e-6, q1 - 1.5*iqr), q3 + 1.5*iqr
    core = deltas[(deltas >= lo) & (deltas <= hi)]
    if len(core) == 0: core = deltas
    return float(np.median(core))

def detect_px_per_mm_from_grid(root):
    ns = {"svg": "http://www.w3.org/2000/svg"}
    xs, ys = [], []

    for ln in root.findall(".//svg:line", ns):
        x1 = ln.get("x1"); x2 = ln.get("x2")
        y1 = ln.get("y1"); y2 = ln.get("y2")
        try:
            x1 = float(x1); x2 = float(x2)
            y1 = float(y1); y2 = float(y2)
        except:
            continue
        if abs(x2 - x1) < 0.5 and abs(y2 - y1) > 5:
            xs.append(x1)
        if abs(y2 - y1) < 0.5 and abs(x2 - x1) > 5:
            ys.append(y1)

    px_step_x = _median_neighbor_spacing(xs) if len(xs) > 0 else None
    px_step_y = _median_neighbor_spacing(ys) if len(ys) > 0 else None
    steps = [s for s in [px_step_x, px_step_y] if s]
    if not steps:
        vb = _get_viewbox(root)
        width_attr = root.get("width")
        if width_attr:
            w_val, w_unit = _parse_float_with_unit(width_attr)
            if w_unit == "mm":
                return vb[2] / max(w_val, 1e-9)
        return 3.7795275591  # fallback

    step = float(np.median(steps))
    per_mm_guess = step / 5.0
    if 2.0 <= per_mm_guess <= 6.0:
        return per_mm_guess
    return step

# -------------------- Text-based metadata --------------------

def parse_text_meta_and_labels(root):
    ns = {"svg": "http://www.w3.org/2000/svg"}
    texts = root.findall(".//svg:text", ns)
    meta = {"mm_per_s": None, "mm_per_mV": None, "band_hz": None}
    label_pos = {}

    mmps_re  = re.compile(r'(\d+(?:\.\d+)?)\s*mm/s', re.I)
    mmpmV_re = re.compile(r'(\d+(?:\.\d+)?)\s*mm/mV', re.I)
    band_re  = re.compile(r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*Hz', re.I)

    for t in texts:
        txt = "".join(t.itertext()).strip()
        if not txt:
            continue

        m1 = mmps_re.search(txt)
        if m1: meta["mm_per_s"] = float(m1.group(1))
        m2 = mmpmV_re.search(txt)
        if m2: meta["mm_per_mV"] = float(m2.group(1))
        m3 = band_re.search(txt)
        if m3: meta["band_hz"] = (float(m3.group(1)), float(m3.group(2)))

        clean = txt.replace(" ", "").upper()
        if clean in [s.upper() for s in LEAD_LABELS_STD]:
            x_attr = t.get("x"); y_attr = t.get("y")
            if (x_attr is None or y_attr is None) and len(list(t)):
                ts = list(t)[0]
                x_attr = x_attr or ts.get("x")
                y_attr = y_attr or ts.get("y")
            try:
                x = float(x_attr); y = float(y_attr)
                for std in LEAD_LABELS_STD:
                    if clean == std.replace(" ", "").upper():
                        label_pos[std] = (x, y)
                        break
            except:
                pass

    return meta, label_pos

# -------------------- Lead picking --------------------

def pick_12_leads_from_paths(root, px_per_mm):
    ns = {"svg": "http://www.w3.org/2000/svg"}
    paths = root.findall(".//svg:path", ns)
    vb = _get_viewbox(root)
    view_w = vb[2]

    candidates = []
    for p in paths:
        d = p.get("d")
        if not d: continue
        xy = _sample_path_points(d, 80)
        if xy.shape[0] < 50: continue
        x_rng = float(xy[:,0].max() - xy[:,0].min())
        y_std = float(np.std(xy[:,1]))
        if x_rng < 0.15*view_w or y_std < 0.8:
            continue
        candidates.append({
            "xy": xy,
            "y_mean": float(xy[:,1].mean()),
            "x_min": float(xy[:,0].min()),
            "x_range": x_rng,
            "y_std": y_std
        })

    meta, label_pos = parse_text_meta_and_labels(root)
    label_y = {k: v[1] for k, v in label_pos.items()}
    have_labels = len(label_y) >= 8

    ordered = [None]*12
    names = LEAD_LABELS_STD.copy()

    if have_labels and len(candidates) > 0:
        unused = set(range(len(candidates)))
        for li, name in enumerate(names):
            if name not in label_y:
                continue
            want_y = label_y[name]
            best, best_idx = None, None
            for ci in list(unused):
                dy = abs(candidates[ci]["y_mean"] - want_y)
                if (best is None) or (dy < best):
                    best, best_idx = dy, ci
            if best_idx is not None:
                ordered[li] = candidates[best_idx]
                unused.remove(best_idx)
        rest = [candidates[i] for i in unused]
        rest.sort(key=lambda c: (c["y_mean"], c["x_min"]))
        for li in range(12):
            if ordered[li] is None and rest:
                ordered[li] = rest.pop(0)
    else:
        chosen = candidates
        if len(chosen) > 12:
            scores = np.array([(c["x_range"]/view_w) + 0.5*np.log1p(c["y_std"]) for c in chosen])
            idx = np.argsort(scores)[-12:]
            chosen = [chosen[i] for i in idx]
        chosen.sort(key=lambda c: c["y_mean"])
        rows = [chosen[i:i+3] for i in range(0, len(chosen), 3)]
        while len(rows) < 4:
            rows.append([])
        for r in rows:
            r.sort(key=lambda c: c["x_min"])
        ordered = []
        for r in rows:
            ordered.extend(r)
        while len(ordered) < 12:
            ordered.append(None)

    mm_per_s  = meta["mm_per_s"]  if meta["mm_per_s"]  else MM_PER_S_DEFAULT
    mm_per_mV = meta["mm_per_mV"] if meta["mm_per_mV"] else MM_PER_MV_DEFAULT

    return ordered, names, mm_per_s, mm_per_mV

# -------------------- Validation --------------------

def validate_backprojection(xy_px, px_per_mm, baseline_px, v_u, fs=TARGET_FS):
    if xy_px.size == 0 or len(v_u) == 0:
        return None

    t_sec = (xy_px[:,0] / px_per_mm)
    order = np.argsort(t_sec)
    t = t_sec[order]; y = xy_px[:,1][order]
    t, uniq = np.unique(t, return_index=True); y = y[uniq]

    dt = 1.0/fs
    t0, t1 = float(t.min()), float(t.max())
    n = int((t1 - t0)/dt) + 1
    t_u = t0 + np.arange(n)*dt
    y_interp = np.interp(t_u, t, y)
    y_hat = baseline_px - (v_u * (px_per_mm * MM_PER_MV_DEFAULT))

    n_min = min(len(y_hat), len(y_interp))
    if n_min < 10:
        return None
    y_hat, y_interp = y_hat[:n_min], y_interp[:n_min]

    err = y_hat - y_interp
    rmse = float(np.sqrt(np.mean(err**2)))
    rng = float(np.sqrt(np.mean((y_interp - np.median(y_interp))**2)))
    if rng < 1e-6: return 0.0
    acc = max(0.0, 1.0 - rmse / (rng + 1e-9))
    return 100.0 * acc

# -------------------- Extraction --------------------

def extract_csv_and_report(svg_text, target_fs=TARGET_FS):
    root = ET.fromstring(svg_text)
    px_per_mm = detect_px_per_mm_from_grid(root)
    ordered, lead_names, mm_per_s, mm_per_mV = pick_12_leads_from_paths(root, px_per_mm)

    waveforms, accs = [], []
    min_len = math.inf

    for c in ordered:
        if c is None:
            waveforms.append(np.array([])); accs.append(None); continue
        xy = c["xy"]
        t_sec = (xy[:,0] / px_per_mm) / mm_per_s
        baseline = np.median(xy[:,1])
        v_mV = -((xy[:,1] - baseline) / px_per_mm) / mm_per_mV
        if len(t_sec) >= 2:
            order = np.argsort(t_sec)
            t = t_sec[order]; v = v_mV[order]
            t, uniq = np.unique(t, return_index=True); v = v[uniq]
            dt = 1.0/target_fs
            t0, t1 = float(t.min()), float(t.max())
            n = int((t1 - t0)/dt) + 1
            t_u = t0 + np.arange(n)*dt
            v_u = np.interp(t_u, t, v)
        else:
            v_u = np.array([])
        waveforms.append(v_u)
        acc = validate_backprojection(xy, px_per_mm, baseline, v_u, fs=target_fs)
        accs.append(acc if acc is not None else None)
        if len(v_u) > 0:
            min_len = min(min_len, len(v_u))

    if min_len is math.inf:
        min_len = 0

    data = np.full((min_len, 12), np.nan, dtype=float)
    for j, v in enumerate(waveforms):
        if len(v) >= min_len and min_len > 0:
            data[:, j] = v[:min_len]

    df = pd.DataFrame(data, columns=lead_names)
    valid_accs = [a for a in accs if isinstance(a, (int, float))]
    overall = float(np.mean(valid_accs)) if valid_accs else 0.0

    header = {
        "px_per_mm": round(px_per_mm, 3),
        "mm_per_s": round(mm_per_s, 2),
        "mm_per_mV": round(mm_per_mV, 2),
        "overall_accuracy_percent": round(overall, 2)
    }
    if overall < 80:
        header["warning"] = "low signal match (<80%)"

    # CSV memory
    mem = io.StringIO()
    # --- ჩაწერა report-ის თავში ---
    for k, v in header.items():
        mem.write(f"{k},{v}\n")
    mem.write("\n")  # გამყოფი ხაზით
    df.to_csv(mem, index=False, float_format="%.5f")
    mem.seek(0)

    return io.BytesIO(mem.getvalue().encode("utf-8"))

# -------------------- Flask --------------------

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
    csv_mem = extract_csv_and_report(svg_text, target_fs=TARGET_FS)
    base = f.filename.rsplit(".",1)[0]

    return send_file(csv_mem, as_attachment=True,
                     download_name=f"{base}_leads.csv",
                     mimetype="text/csv")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
