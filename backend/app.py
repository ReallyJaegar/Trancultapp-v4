"""
Cultural Image Transformer - Backend
Pipeline: GPT-4o Vision → GroundingDINO (HuggingFace) + SAM → Stable Diffusion Inpainting
"""

import os
import sys
import logging
from pathlib import Path
from flask import Flask, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = Path(__file__).parent / "uploads"
app.config["OUTPUT_FOLDER"] = Path(__file__).parent / "outputs"
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)
app.config["OUTPUT_FOLDER"].mkdir(exist_ok=True)

from routes.analyze import analyze_bp
from routes.segment import segment_bp
from routes.inpaint import inpaint_bp

app.register_blueprint(analyze_bp, url_prefix="/api/analyze")
app.register_blueprint(segment_bp, url_prefix="/api/segment")
app.register_blueprint(inpaint_bp, url_prefix="/api/inpaint")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Cultural Transformer API running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
