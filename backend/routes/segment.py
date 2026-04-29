"""
Route: POST /api/segment/
Step 2 — GroundingDINO (HuggingFace) + SAM: detect and mask each object
"""

import logging
from flask import Blueprint, request, jsonify, current_app
from services.segmentation_service import segment_objects

logger = logging.getLogger(__name__)
segment_bp = Blueprint("segment", __name__)


@segment_bp.route("/", methods=["POST"])
def segment():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    image_filename = data.get("image_filename")
    swaps          = data.get("swaps", [])

    if not image_filename:
        return jsonify({"error": "image_filename required"}), 400
    if not swaps:
        return jsonify({"error": "swaps list required"}), 400

    image_path = current_app.config["UPLOAD_FOLDER"] / image_filename
    if not image_path.exists():
        return jsonify({"error": f"Image not found: {image_filename}"}), 404

    try:
        enriched = segment_objects(str(image_path), swaps)
        return jsonify({"status": "ok", "image_filename": image_filename, "swaps": enriched})
    except Exception as e:
        logger.error(f"Segmentation failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
