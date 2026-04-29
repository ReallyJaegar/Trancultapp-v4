"""
Route: POST /api/analyze/
Step 1 — GPT-4o Vision: identify culturally salient objects and suggest swaps
"""

import base64
import logging
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from services.gpt4o_analyzer import analyze_cultural_objects

logger = logging.getLogger(__name__)
analyze_bp = Blueprint("analyze", __name__)

ALLOWED = {"png", "jpg", "jpeg", "webp"}


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


@analyze_bp.route("/", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file           = request.files["image"]
    source_country = request.form.get("source_country", "").strip()
    target_country = request.form.get("target_country", "").strip()
    api_key        = request.form.get("openai_api_key", "").strip()

    if not source_country or not target_country:
        return jsonify({"error": "source_country and target_country are required"}), 400
    if not api_key:
        return jsonify({"error": "openai_api_key is required"}), 400
    if not allowed(file.filename):
        return jsonify({"error": "Invalid file type. Use PNG, JPG, JPEG or WEBP"}), 400

    filename    = secure_filename(file.filename)
    upload_path = current_app.config["UPLOAD_FOLDER"] / filename
    file.save(str(upload_path))

    with open(upload_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    ext        = filename.rsplit(".", 1)[1].lower()
    media_type = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    try:
        swaps = analyze_cultural_objects(
            image_b64=image_b64,
            media_type=media_type,
            source_country=source_country,
            target_country=target_country,
            api_key=api_key,
        )
        return jsonify({
            "status": "ok",
            "image_filename": filename,
            "source_country": source_country,
            "target_country": target_country,
            "swaps": swaps,
        })
    except Exception as e:
        logger.error(f"GPT-4o analysis failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
