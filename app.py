import io
import json
import re
from flask import Flask, render_template, request, send_file
import xml.etree.ElementTree as ET
from svg.path import parse_path

app = Flask(__name__)

# --- Limits to prevent OOM on Render ---
MAX_POINTS_PER_PATH = 300
MAX_TOTAL_POINTS = 20000


def sample_path_points(d_attr, samples=50, max_points=MAX_POINTS_PER_PATH):
    """Sample SVG <path> d attribute into list of [x,y] points."""
    try:
        path = parse_path(d_attr)
    except Exception:
        return []

    coords = []
    for seg in path:
        n = min(samples, max_points)
        for i in range(n):
            t = i / (n - 1)
            pt = seg.point(t)
            coords.append([float(pt.real), float(pt.imag)])
            if len(coords) >= max_points:
                return coords
    return coords


def parse_poly_points(points_str):
    """Convert points='x1,y1 x2,y2 ...' to list of [x,y]."""
    if not points_str:
        return []
    nums = [float(x) for x in re.split(r"[,\s]+", points_str.strip()) if x]
    return [[nums[i], nums[i + 1]] for i in range(0, len(nums) - 1, 2)]


def extract_coordinates(svg_text):
    """Extract all coordinates from SVG <path>, <polyline>, and <line>."""
    try:
        root = ET.fromstring(svg_text)
    except Exception:
        return {"error": "Invalid SVG format"}

    ns = {"svg": "http://www.w3.org/2000/svg"}
    paths = root.findall(".//svg:path", ns)
    polylines = root.findall(".//svg:polyline", ns)
    lines = root.findall(".//svg:line", ns)

    all_coords = []
    for p in paths:
        d = p.get("d")
        if d:
            pts = sample_path_points(d)
            all_coords.extend(pts)
            if len(all_coords) >= MAX_TOTAL_POINTS:
                break

    for pl in polylines:
        pts = parse_poly_points(pl.get("points"))
        all_coords.extend(pts)
        if len(all_coords) >= MAX_TOTAL_POINTS:
            break

    for ln in lines:
        try:
            x1 = float(ln.get("x1", "0"))
            y1 = float(ln.get("y1", "0"))
            x2 = float(ln.get("x2", "0"))
            y2 = float(ln.get("y2", "0"))
            all_coords.extend([[x1, y1], [x2, y2]])
        except Exception:
            continue
        if len(all_coords) >= MAX_TOTAL_POINTS:
            break

    return {"count": len(all_coords), "coordinates": all_coords}


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

    result = extract_coordinates(svg_text)
    result["source_svg"] = f.filename

    mem = io.BytesIO()
    mem.write(json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8"))
    mem.seek(0)

    json_name = f"{f.filename.rsplit('.',1)[0]}_coords.json"
    return send_file(mem, as_attachment=True, download_name=json_name, mimetype="application/json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
