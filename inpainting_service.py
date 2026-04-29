"""
Service: Stable Diffusion Inpainting
Intentionally uses a lower-quality model (runwayml/stable-diffusion-inpainting)
to produce visible "pipeline struggle" artifacts.
Applies swaps sequentially: each swap's output feeds as input to the next.
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

# Intentionally lower-quality model — shows "pipeline effort"
# vs. runwayml/stable-diffusion-inpainting which is cleaner
SD_INPAINT_MODEL = os.environ.get(
    "SD_INPAINT_MODEL",
    "runwayml/stable-diffusion-inpainting",  # swap to "stabilityai/stable-diffusion-2-inpainting" for slightly better
)

_pipe = None


def _load_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    try:
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info(f"Loading SD inpainting model: {SD_INPAINT_MODEL} on {device}")
        _pipe = StableDiffusionInpaintPipeline.from_pretrained(
            SD_INPAINT_MODEL,
            torch_dtype=dtype,
            safety_checker=None,   # Disable safety checker for speed
        )
        _pipe = _pipe.to(device)

        # Reduce quality intentionally: fewer inference steps, higher guidance
        # This makes the outputs look more "raw pipeline" style
        logger.info("SD inpainting pipeline loaded")

    except Exception as e:
        logger.warning(f"SD pipeline not available: {e}. Using placeholder compositing.")
        _pipe = "UNAVAILABLE"

    return _pipe


def _b64_to_pil(b64_str: str) -> Image.Image:
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _mask_b64_to_pil(b64_str: str, size: tuple) -> Image.Image:
    data = base64.b64decode(b64_str)
    mask = Image.open(io.BytesIO(data)).convert("L")
    # Resize mask to match image if needed
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    # Dilate mask slightly for cleaner inpainting
    from PIL import ImageFilter
    mask = mask.filter(ImageFilter.MaxFilter(5))
    return mask


def _pil_to_b64(img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _placeholder_inpaint(image: Image.Image, mask: Image.Image, prompt: str, swap_obj: str) -> Image.Image:
    """
    Fallback when SD is not available: draw a colored rectangle with label.
    """
    from PIL import ImageDraw, ImageFont
    result = image.copy()
    mask_arr = np.array(mask)
    ys, xs = np.where(mask_arr > 128)
    if len(xs) == 0:
        return result
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    draw = ImageDraw.Draw(result, "RGBA")
    draw.rectangle([x1, y1, x2, y2], fill=(255, 165, 0, 180))
    draw.text((x1 + 4, y1 + 4), f"→ {swap_obj}", fill=(255, 255, 255))
    return result


def run_inpainting(
    image_path: str,
    swaps: list,
    target_country: str,
    output_dir: str,
) -> dict:
    """
    Sequentially apply each swap to the image via inpainting.

    Returns:
    {
        result_image_b64: str,
        step_images: [{ swap_index: int, image_b64: str }, ...]
    }
    """
    pipe = _load_pipeline()

    current_image = Image.open(image_path).convert("RGB")
    orig_size = current_image.size

    # Resize to 512x512 for SD (required by SD 1.x inpainting)
    SD_SIZE = (512, 512)
    current_512 = current_image.resize(SD_SIZE, Image.LANCZOS)

    step_images = []
    output_dir = Path(output_dir)

    for i, swap in enumerate(swaps):
        inpaint_prompt = swap.get("inpaint_prompt", f"a {swap['swap']}, photorealistic")
        mask_b64 = swap.get("mask_b64")
        swap_name = swap.get("swap", "object")
        obj_name = swap.get("object", "object")

        logger.info(f"Inpainting swap {i+1}/{len(swaps)}: {obj_name} → {swap_name}")

        if not mask_b64:
            logger.warning(f"No mask for swap {i}, skipping")
            step_images.append({"swap_index": i, "image_b64": _pil_to_b64(current_512)})
            continue

        mask_pil = _mask_b64_to_pil(mask_b64, SD_SIZE)

        # Enhance prompt with country context
        full_prompt = (
            f"{inpaint_prompt}, {target_country} cultural style, "
            "realistic, detailed, natural lighting, high quality photograph"
        )
        negative_prompt = (
            "blurry, distorted, deformed, ugly, low quality, watermark, "
            "text, signature, out of frame, duplicate, extra limbs"
        )

        try:
            if pipe == "UNAVAILABLE":
                result_img = _placeholder_inpaint(current_512, mask_pil, full_prompt, swap_name)
            else:
                import torch
                output = pipe(
                    prompt=full_prompt,
                    negative_prompt=negative_prompt,
                    image=current_512,
                    mask_image=mask_pil,
                    num_inference_steps=20,   # Intentionally low — shows pipeline effort
                    guidance_scale=7.5,
                    strength=0.85,
                    generator=torch.Generator().manual_seed(42 + i),
                )
                result_img = output.images[0]

            current_512 = result_img
            b64 = _pil_to_b64(result_img)
            step_images.append({"swap_index": i, "image_b64": b64, "swap": swap_name})

        except Exception as e:
            logger.error(f"Inpainting step {i} failed: {e}", exc_info=True)
            step_images.append({
                "swap_index": i,
                "image_b64": _pil_to_b64(current_512),
                "swap": swap_name,
                "error": str(e),
            })

    # Upscale back to original size for final result
    final_image = current_512.resize(orig_size, Image.LANCZOS)

    # Save final output
    output_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
    final_path = output_dir / output_filename
    final_image.save(str(final_path), format="JPEG", quality=92)

    return {
        "result_image_b64": _pil_to_b64(final_image),
        "step_images": step_images,
        "output_filename": output_filename,
    }
