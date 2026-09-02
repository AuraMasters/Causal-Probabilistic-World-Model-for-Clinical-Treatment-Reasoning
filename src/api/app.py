import sys
from pathlib import Path

from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.service import (
    analyze_patient,
    get_overview,
)

app = Flask(__name__)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/overview")
def overview():
    try:
        payload = get_overview()
        return jsonify(payload)
    except Exception as error:  # noqa: BLE001
        return jsonify({"error": str(error)}), 500


@app.post("/api/analyze")
def analyze():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify(
            {"error": "Request body must be a JSON object."}
        ), 400

    try:
        payload = analyze_patient(data.get("inputs", {}))
        return jsonify(payload)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:  # noqa: BLE001
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
