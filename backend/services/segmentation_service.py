"""
Service: Segmentation using HuggingFace GroundingDINO + SAM
Uses: IDEA-Research/grounding-dino-base  (transformers, zero compilation)
      segment-anything pip package for precise masks
"""

import logging
import base64
import io
import os
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_gdino_processor = None
_gdino_model     = None
_sam_predictor   = None
_models_loaded   = False


def _load_gdino():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    import torch
    model_id = os.environ.get("GDINO_MODEL_ID", "IDEA-Research/grounding-dino-base")
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading GroundingDINO ({model_id}) on {device}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model     = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    logger.info("GroundingDINO loaded ✅")
    return processor, model


def _load_sam():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from segment_anything import sam_model_registry, SamPredictor
        checkpoint = os.environ.get("SAM_CHECKPOINT", "/content/weights/sam_vit_h_4b8939.pth")
        model_type = os.environ.get("SAM_MODEL_TYPE", "vit_h")
        if not os.path.exists(checkpoint):
            logger.warning(f"SAM checkpoint not found at {checkpoint}")
            return None
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        logger.info(f"SAM loaded on {device} ✅")
        return predictor
    except Exception as e:
        logger.warning(f"SAM not available: {e}")
        return None


def _ensure_models():
    global _gdino_processor, _gdino_model, _sam_predictor, _models_loaded
    if not _models_loaded:
        try:
            _gdino_processor, _gdino_model = _load_gdino()
        except Exception as e:
            logger.warning(f"GroundingDINO load failed: {e} — bbox fallback mode")
            _gdino_processor = _gdino_model = None
        _sam_predictor = _load_sam()
        _models_loaded = True
    return _gdino_processor, _gdino_model, _sam_predictor


def _mask_to_b64(mask: np.ndarray) -> str:
    mask_uint8 = (mask.astype(bool) * 255).astype(np.uint8)
    img = Image.fromarray(mask_uint8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _bbox_mask(image: Image.Image, bbox_frac=(0.2, 0.2, 0.5, 0.5)):
    w, h = image.size
    x1 = int(bbox_frac[0] * w); y1 = int(bbox_frac[1] * h)
    x2 = int(bbox_frac[2] * w); y2 = int(bbox_frac[3] * h)
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return _mask_to_b64(mask), [x1, y1, x2, y2], 0.0


def _gdino_detect(processor, model, image: Image.Image, text_prompt: str):
    import torch
    device = next(model.parameters()).device
    prompt = text_prompt.strip()
    if not prompt.endswith("."):
        prompt += "."
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    w, h = image.size
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.30,
        text_threshold=0.25,
        target_sizes=[(h, w)],
    )[0]
    boxes  = results["boxes"]
    scores = results["scores"]
    if len(scores) == 0:
        return None, 0.0
    best  = scores.argmax().item()
    bbox  = [int(v) for v in boxes[best].tolist()]
    return bbox, round(scores[best].item(), 3)


def _sam_mask_from_bbox(predictor, image_array: np.ndarray, bbox: list) -> np.ndarray:
    predictor.set_image(image_array)
    input_box = np.array(bbox)
    masks, scores, _ = predictor.predict(
        point_coords=None, point_labels=None,
        box=input_box[None, :], multimask_output=True,
    )
    return masks[scores.argmax()]


FALLBACK_FRACS = [
    (0.05, 0.05, 0.45, 0.45), (0.55, 0.05, 0.95, 0.45),
    (0.05, 0.55, 0.45, 0.95), (0.55, 0.55, 0.95, 0.95),
    (0.20, 0.20, 0.80, 0.80), (0.10, 0.10, 0.60, 0.60),
]


def segment_objects(image_path: str, swaps: list) -> list:
    processor, gdino_model, sam_predictor = _ensure_models()
    image       = Image.open(image_path).convert("RGB")
    image_array = np.array(image)
    w, h        = image.size
    enriched    = []

    for i, swap in enumerate(swaps):
        prompt = swap.get("detection_prompt", swap.get("object", "object"))
        logger.info(f"Segmenting [{i+1}/{len(swaps)}]: '{prompt}'")
        try:
            bbox, conf = None, 0.0
            if gdino_model is not None:
                bbox, conf = _gdino_detect(processor, gdino_model, image, prompt)

            if bbox is None:
                logger.warning(f"  No detection for '{prompt}' — fallback bbox")
                mask_b64, bbox, conf = _bbox_mask(image, FALLBACK_FRACS[i % len(FALLBACK_FRACS)])
            elif sam_predictor is not None:
                mask     = _sam_mask_from_bbox(sam_predictor, image_array, bbox)
                mask_b64 = _mask_to_b64(mask)
            else:
                x1, y1, x2, y2 = bbox
                mask = np.zeros((h, w), dtype=bool)
                mask[y1:y2, x1:x2] = True
                mask_b64 = _mask_to_b64(mask)

            enriched.append({**swap, "mask_b64": mask_b64, "bbox": bbox, "confidence": conf})

        except Exception as e:
            logger.error(f"  Failed for '{prompt}': {e}", exc_info=True)
            mask_b64, bbox, conf = _bbox_mask(image, FALLBACK_FRACS[i % len(FALLBACK_FRACS)])
            enriched.append({**swap, "mask_b64": mask_b64, "bbox": bbox,
                             "confidence": 0.0, "segment_error": str(e)})

    return enriched
