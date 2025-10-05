import io
import re
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file
import xml.etree.ElementTree as ET
from svg.path import parse_path

app = Flask(__name__)

# --- კონფიგურაცია ---
PX_PER_MM = 2.833  # SVG calibration (viewBox 595 / width 210mm)
MM_PER_MV = 10
MM_PER_S = 25
PX_PER_MV = PX_PER_MM * MM_PER_MV  # 28.33 px/mV
MAX_PATHS = 50


def sample_path_points(d_attr, samples=50):
    """Sample SVG <path> d attribute into [x, y] წერტილები."""
    try:
        path = parse_path(d_attr)
    except Exception:
        return np.empty((0, 2))

    coords = []
    for seg in path:
        for i in range(samples):
            t = i / (samples - 1)
            pt = seg.point(t)
            coords.append([float(pt.real), float(pt.imag)])
    return np.array(coords)


def extract_leads_from_svg(svg_text):
    """ECG lead-ების გამოყოფა SVG-დან და CSV ფორმატში მომზადება."""
    root = ET.fromstring(svg_text)
    ns = {"svg": "http://www.w3.org/2000/svg"}
    paths = root.findall(".//svg:path", ns)

    all_signals = []
    for p in paths[:MAX_PATHS]:
        d = p.get("d")
        if not d:
            continue
        pts = sample_path_points(d)
        if len(pts) > 10:
            all_signals.append(pts)

    # y-ს საშუალო პოზიციით დავახარისხოთ ბლოკებად
    ys = [np.mean(sig[:, 1]) for sig in all_signals]
    sorted_indices = np.argsort(ys)
    leads = [all_signals[i] for i in sorted_indices[:12]]

    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    max_len = max(len(sig) for sig in leads)
    data = np.full((max_len, len(leads)), np.nan)

    for i, sig in enumerate(leads):
        y = sig[:, 1]
        y = -(y - np.mean(y)) / PX_PER_MV  # კონვერტაცია mV-ში
        data[:len(y), i] = y

    df = pd.DataFrame(data, columns=lead_names)
    mem = io.BytesIO()
    df.to_csv(mem, index=False, float_format="%.5f")
    mem.seek(0)
    return mem


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

    csv_mem = extract_leads_from_svg(svg_text)
    csv_name = f"{f.filename.rsplit('.',1)[0]}_leads.csv"

    return send_file(
        csv_mem,
        as_attachment=True,
        download_name=csv_name,
        mimetype="text/csv"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
