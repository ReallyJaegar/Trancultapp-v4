"""
Service: Stable Diffusion Inpainting
Intentionally uses lower-quality settings (20 steps, SD 1.x) for raw pipeline look.
Applies swaps sequentially — each output feeds into the next.
"""

import logging
import base64
import io
import os
import uuid
from pathlib import Path
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SD_INPAINT_MODEL = os.environ.get("SD_INPAINT_MODEL", "runwayml/stable-diffusion-inpainting")

_pipe = None


def _load_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe
    try:
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32
        logger.info(f"Loading SD inpainting model: {SD_INPAINT_MODEL} on {device}")
        _pipe = StableDiffusionInpaintPipeline.from_pretrained(
            SD_INPAINT_MODEL, torch_dtype=dtype, safety_checker=None,
        ).to(device)
        logger.info("SD pipeline loaded ✅")
    except Exception as e:
        logger.warning(f"SD pipeline not available: {e}")
        _pipe = "UNAVAILABLE"
    return _pipe


def _b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def _mask_b64_to_pil(b64_str: str, size: tuple) -> Image.Image:
    from PIL import ImageFilter
    mask = Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    return mask.filter(ImageFilter.MaxFilter(5))


def _pil_to_b64(img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _placeholder_inpaint(image: Image.Image, mask: Image.Image, swap_name: str) -> Image.Image:
    from PIL import ImageDraw
    result   = image.copy()
    mask_arr = np.array(mask)
    ys, xs   = np.where(mask_arr > 128)
    if len(xs) == 0:
        return result
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    draw = ImageDraw.Draw(result, "RGBA")
    draw.rectangle([x1, y1, x2, y2], fill=(255, 165, 0, 180))
    draw.text((x1 + 4, y1 + 4), f"→ {swap_name}", fill=(255, 255, 255))
    return result


def run_inpainting(image_path: str, swaps: list, target_country: str, output_dir: str) -> dict:
    pipe          = _load_pipeline()
    current_image = Image.open(image_path).convert("RGB")
    orig_size     = current_image.size
    SD_SIZE       = (512, 512)
    current_512   = current_image.resize(SD_SIZE, Image.LANCZOS)
    step_images   = []
    output_dir    = Path(output_dir)

    for i, swap in enumerate(swaps):
        inpaint_prompt = swap.get("inpaint_prompt", f"a {swap.get('swap','object')}, photorealistic")
        mask_b64       = swap.get("mask_b64")
        swap_name      = swap.get("swap", "object")
        obj_name       = swap.get("object", "object")

        logger.info(f"Inpainting [{i+1}/{len(swaps)}]: {obj_name} → {swap_name}")

        if not mask_b64:
            step_images.append({"swap_index": i, "image_b64": _pil_to_b64(current_512), "swap": swap_name})
            continue

        mask_pil       = _mask_b64_to_pil(mask_b64, SD_SIZE)
        full_prompt    = (
            f"{inpaint_prompt}, {target_country} cultural style, "
            "realistic, detailed, natural lighting, high quality photograph"
        )
        negative_prompt = (
            "blurry, distorted, deformed, ugly, low quality, watermark, "
            "text, signature, out of frame, duplicate"
        )

        try:
            if pipe == "UNAVAILABLE":
                result_img = _placeholder_inpaint(current_512, mask_pil, swap_name)
            else:
                import torch
                output     = pipe(
                    prompt=full_prompt,
                    negative_prompt=negative_prompt,
                    image=current_512,
                    mask_image=mask_pil,
                    num_inference_steps=20,   # intentionally low — raw pipeline look
                    guidance_scale=7.5,
                    strength=0.85,
                    generator=torch.Generator().manual_seed(42 + i),
                )
                result_img = output.images[0]

            current_512 = result_img
            step_images.append({"swap_index": i, "image_b64": _pil_to_b64(result_img), "swap": swap_name})

        except Exception as e:
            logger.error(f"Inpainting step {i} failed: {e}", exc_info=True)
            step_images.append({"swap_index": i, "image_b64": _pil_to_b64(current_512),
                                 "swap": swap_name, "error": str(e)})

    final_image     = current_512.resize(orig_size, Image.LANCZOS)
    output_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
    final_image.save(str(output_dir / output_filename), format="JPEG", quality=92)

    return {
        "result_image_b64": _pil_to_b64(final_image),
        "step_images": step_images,
        "output_filename": output_filename,
    }
