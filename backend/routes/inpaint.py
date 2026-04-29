"""
Route: POST /api/inpaint/
Step 3 — Stable Diffusion Inpainting: apply all swaps sequentially
"""

import logging
from flask import Blueprint, request, jsonify, current_app
from services.inpainting_service import run_inpainting

logger = logging.getLogger(__name__)
inpaint_bp = Blueprint("inpaint", __name__)


@inpaint_bp.route("/", methods=["POST"])
def inpaint():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    image_filename = data.get("image_filename")
    swaps          = data.get("swaps", [])
    target_country = data.get("target_country", "")

    if not image_filename:
        return jsonify({"error": "image_filename required"}), 400

    image_path = current_app.config["UPLOAD_FOLDER"] / image_filename
    if not image_path.exists():
        return jsonify({"error": f"Image not found: {image_filename}"}), 404

    try:
        result = run_inpainting(
            image_path=str(image_path),
            swaps=swaps,
            target_country=target_country,
            output_dir=str(current_app.config["OUTPUT_FOLDER"]),
        )
        return jsonify({"status": "ok", **result})
    except Exception as e:
        logger.error(f"Inpainting failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
