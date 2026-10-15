import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.service import analyze_patient, get_overview

DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
app = Flask(__name__, static_folder=str(DIST_DIR) if DIST_DIR.exists() else None)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/overview")
def overview():
    try:
        return jsonify(get_overview())
    except Exception as error:  # noqa: BLE001
        return jsonify({"error": str(error)}), 500


@app.post("/api/analyze")
def analyze():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a valid JSON object."}), 400

    try:
        model_type = str(data.get("model_type", "continuous")).lower().strip()
        result = analyze_patient(data.get("inputs", {}), model_type=model_type)
        return jsonify(result)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:  # noqa: BLE001
        return jsonify({"error": str(error)}), 500


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if DIST_DIR.exists():
        target = DIST_DIR / path
        if target.exists() and target.is_file():
            return send_from_directory(DIST_DIR, path)
        return send_from_directory(DIST_DIR, "index.html")
    return jsonify({"message": "ACTG175 Causal Backend Live. Frontend build not present."}), 200


if __name__ == "__main__":
    print("[backend] Starting ACTG175 Causal API on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
