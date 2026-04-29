"""
Service: GPT-4o Vision Cultural Analyzer
Sends image + countries → returns 5-6 culturally salient object swaps
"""

import json
import re
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a cultural anthropologist and visual AI assistant specializing in identifying
culturally specific objects in images and suggesting culturally appropriate replacements.

When given an image and a source/target country pair, you must:
1. Identify 5-6 objects in the image that are CULTURALLY SALIENT to the source country
   (food, clothing, architecture details, vehicles, symbols, decor, etc.)
2. For each object, suggest a culturally equivalent replacement from the target country
3. Provide brief reasoning for each swap
4. Provide a detection_prompt (short phrase) suitable for an object detection model
5. Provide an inpaint_prompt suitable for Stable Diffusion inpainting

Respond ONLY with a valid JSON array. No prose, no markdown fences, no explanation outside the JSON.

Format:
[
  {
    "object": "taco",
    "swap": "samosa",
    "reasoning": "Tacos are iconic Mexican street food; samosas are the Indian equivalent",
    "detection_prompt": "taco on plate",
    "inpaint_prompt": "a golden crispy samosa on a small plate, Indian street food, photorealistic"
  }
]

Rules:
- Pick objects that are VISIBLE and IDENTIFIABLE in the image
- Keep detection_prompt SHORT and specific (2-5 words)
- Make inpaint_prompt descriptive for Stable Diffusion
- Return EXACTLY 5 or 6 items
- Return ONLY the JSON array, nothing else
"""


def analyze_cultural_objects(
    image_b64: str,
    media_type: str,
    source_country: str,
    target_country: str,
    api_key: str,
) -> list:
    client = OpenAI(api_key=api_key)

    user_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{image_b64}",
                "detail": "high",
            },
        },
        {
            "type": "text",
            "text": (
                f"Source country: {source_country}\n"
                f"Target country: {target_country}\n\n"
                f"Identify 5-6 culturally salient objects visible in this image and suggest "
                f"culturally appropriate replacements from {target_country}. "
                "Return ONLY the JSON array."
            ),
        },
    ]

    logger.info(f"GPT-4o analysis: {source_country} → {target_country}")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=1500,
        temperature=0.4,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    swaps = json.loads(raw)

    if not isinstance(swaps, list):
        raise ValueError("GPT-4o did not return a JSON array")

    required = {"object", "swap", "reasoning", "detection_prompt", "inpaint_prompt"}
    for i, s in enumerate(swaps):
        missing = required - s.keys()
        if missing:
            raise ValueError(f"Swap {i} missing keys: {missing}")

    return swaps[:6]
