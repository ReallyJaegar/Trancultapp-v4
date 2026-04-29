"""
Service: Segmentation using GroundingDINO + SAM
Detects each object and generates a binary mask
"""

import logging
import base64
import io
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _load_models():
    """Lazily load GroundingDINO and SAM models."""
    # GroundingDINO
    try:
        from groundingdino.util.inference import load_model, predict
        from groundingdino.util import box_ops
        import groundingdino.datasets.transforms as T
        import torch
        import os

        dino_config = os.environ.get(
            "GROUNDING_DINO_CONFIG",
            "groundingdino/config/GroundingDINO_SwinT_OGC.py",
        )
        dino_weights = os.environ.get(
            "GROUNDING_DINO_WEIGHTS",
            "weights/groundingdino_swint_ogc.pth",
        )
        dino_model = load_model(dino_config, dino_weights)
        logger.info("GroundingDINO loaded")
    except Exception as e:
        logger.warning(f"GroundingDINO not available: {e}. Using fallback bbox.")
        dino_model = None

    # SAM
    try:
        from segment_anything import sam_model_registry, SamPredictor
        import torch
        import os

        sam_checkpoint = os.environ.get("SAM_CHECKPOINT", "weights/sam_vit_h_4b8939.pth")
        sam_model_type = os.environ.get("SAM_MODEL_TYPE", "vit_h")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)
        logger.info(f"SAM loaded on {device}")
    except Exception as e:
        logger.warning(f"SAM not available: {e}. Using fallback mask generation.")
        sam_predictor = None

    return dino_model, sam_predictor


# Module-level model cache
_dino_model = None
_sam_predictor = None
_models_loaded = False


def _ensure_models():
    global _dino_model, _sam_predictor, _models_loaded
    if not _models_loaded:
        _dino_model, _sam_predictor = _load_models()
        _models_loaded = True
    return _dino_model, _sam_predictor


def _mask_to_b64(mask: np.ndarray) -> str:
    """Convert boolean numpy mask to base64 PNG."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    img = Image.fromarray(mask_uint8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _fallback_bbox_mask(image: Image.Image, bbox_fraction=(0.2, 0.2, 0.5, 0.5)) -> tuple:
    """
    Fallback when models unavailable: return a center-crop mask.
    Returns (mask_b64, bbox, confidence=0.0)
    """
    w, h = image.size
    x1 = int(bbox_fraction[0] * w)
    y1 = int(bbox_fraction[1] * h)
    x2 = int(bbox_fraction[2] * w)
    y2 = int(bbox_fraction[3] * h)
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return _mask_to_b64(mask), [x1, y1, x2, y2], 0.0


def _grounding_dino_detect(dino_model, image_path: str, prompt: str):
    """Run GroundingDINO to get bounding boxes for the prompt."""
    import torch
    from groundingdino.util.inference import predict, load_image
    from groundingdino.util import box_ops

    image_source, image_tensor = load_image(image_path)
    h, w = image_source.shape[:2]

    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_tensor,
        caption=prompt,
        box_threshold=0.30,
        text_threshold=0.25,
    )

    if len(boxes) == 0:
        return None, 0.0, (h, w)

    # Take highest-confidence detection
    best_idx = logits.argmax().item()
    box = boxes[best_idx]  # cx, cy, w, h normalized
    conf = logits[best_idx].item()

    # Convert to pixel coords
    cx, cy, bw, bh = box
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)

    return [x1, y1, x2, y2], conf, (h, w)


def _sam_mask_from_bbox(sam_predictor, image_array: np.ndarray, bbox: list) -> np.ndarray:
    """Use SAM to generate precise mask from bbox."""
    import numpy as np

    sam_predictor.set_image(image_array)
    x1, y1, x2, y2 = bbox
    input_box = np.array([x1, y1, x2, y2])

    masks, scores, _ = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=True,
    )

    # Pick highest-score mask
    best = scores.argmax()
    return masks[best]


def segment_objects(image_path: str, swaps: list) -> list:
    """
    For each swap, detect the object using GroundingDINO and
    produce a binary mask using SAM.

    Returns enriched swaps list with mask_b64, bbox, confidence.
    """
    dino_model, sam_predictor = _ensure_models()
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)
    w, h = image.size

    enriched = []
    for i, swap in enumerate(swaps):
        prompt = swap.get("detection_prompt", swap.get("object", "object"))
        logger.info(f"Segmenting swap {i+1}: '{prompt}'")

        try:
            if dino_model is not None:
                bbox, conf, _ = _grounding_dino_detect(dino_model, image_path, prompt)
            else:
                bbox = None
                conf = 0.0

            if bbox is None:
                # Spread fallbacks across image quadrants
                offsets = [
                    (0.05, 0.05, 0.45, 0.45),
                    (0.55, 0.05, 0.95, 0.45),
                    (0.05, 0.55, 0.45, 0.95),
                    (0.55, 0.55, 0.95, 0.95),
                    (0.25, 0.25, 0.75, 0.75),
                    (0.10, 0.10, 0.60, 0.60),
                ]
                frac = offsets[i % len(offsets)]
                mask_b64, bbox, conf = _fallback_bbox_mask(image, frac)
                logger.warning(f"GroundingDINO found nothing for '{prompt}', using fallback bbox")
            else:
                if sam_predictor is not None:
                    mask = _sam_mask_from_bbox(sam_predictor, image_array, bbox)
                    mask_b64 = _mask_to_b64(mask)
                else:
                    # Bbox-only mask
                    x1, y1, x2, y2 = bbox
                    mask = np.zeros((h, w), dtype=bool)
                    mask[y1:y2, x1:x2] = True
                    mask_b64 = _mask_to_b64(mask)

            enriched.append({
                **swap,
                "mask_b64": mask_b64,
                "bbox": bbox,
                "confidence": round(conf, 3),
            })

        except Exception as e:
            logger.error(f"Failed to segment '{prompt}': {e}", exc_info=True)
            mask_b64, bbox, conf = _fallback_bbox_mask(image)
            enriched.append({
                **swap,
                "mask_b64": mask_b64,
                "bbox": bbox,
                "confidence": 0.0,
                "segment_error": str(e),
            })

    return enriched
