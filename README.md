# 🌍 Cultural Image Transformer

Transform images from one cultural context to another using a 3-stage ML pipeline.

## Pipeline

```
Image + Countries
       │
       ▼
┌─────────────────────────────────┐
│  STEP 1 — GPT-4o Vision         │
│  · Identifies 5-6 culturally    │
│    salient objects               │
│  · Suggests target-country swap  │
│  · Provides reasoning + prompts  │
└──────────────┬──────────────────┘
               │  swap list
               ▼
┌─────────────────────────────────┐
│  STEP 2 — GroundingDINO + SAM   │
│  · Detects each object w/ DINO  │
│  · Generates precise SAM masks  │
└──────────────┬──────────────────┘
               │  masks + bboxes
               ▼
┌─────────────────────────────────┐
│  STEP 3 — SD Inpainting         │
│  · runwayml/stable-diffusion-   │
│    inpainting (intentionally    │
│    lower quality for raw feel)  │
│  · Sequential per-object swaps  │
│  · Step-by-step intermediate    │
│    images saved                  │
└──────────────┬──────────────────┘
               │
               ▼
       Final Transformed Image
```

## Project Structure

```
cultural-transformer/
├── backend/
│   ├── app.py                        # Flask app, route registration
│   ├── routes/
│   │   ├── analyze.py                # POST /api/analyze/ — GPT-4o Vision
│   │   ├── segment.py                # POST /api/segment/ — GroundingDINO+SAM
│   │   └── inpaint.py                # POST /api/inpaint/ — SD Inpainting
│   ├── services/
│   │   ├── gpt4o_analyzer.py         # GPT-4o call + JSON parsing
│   │   ├── segmentation_service.py   # GroundingDINO detection + SAM masking
│   │   └── inpainting_service.py     # Diffusers SD pipeline
│   ├── uploads/                      # Uploaded images (auto-created)
│   ├── outputs/                      # Final results (auto-created)
│   ├── weights/                      # Model checkpoints (see setup.sh)
│   └── requirements.txt
├── frontend/
│   └── index.html                    # Single-file SPA
├── setup.sh                          # Install deps + download weights
├── run.sh                            # Start backend + frontend servers
└── README.md
```

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA GPU recommended (8GB+ VRAM for SAM + SD)
- OpenAI API key (GPT-4o access)

### Installation

```bash
git clone <this-repo>
cd cultural-transformer
chmod +x setup.sh run.sh
./setup.sh
```

### Running

```bash
./run.sh
# Open http://localhost:8080
```

### Usage
1. Enter source country (e.g. `Mexico`)
2. Enter target country (e.g. `India`)
3. Paste your OpenAI API key
4. Upload any image
5. Click **Run Pipeline**

The UI shows:
- GPT-4o's identified cultural swaps with reasoning
- GroundingDINO + SAM masks overlaid on the original image
- Each inpainting step as a thumbnail strip
- Final before/after comparison with download

## API Endpoints

### POST /api/analyze/
```
multipart/form-data:
  image            — image file
  source_country   — string
  target_country   — string
  openai_api_key   — string

Response:
  { swaps: [{ object, swap, reasoning, detection_prompt, inpaint_prompt }] }
```

### POST /api/segment/
```json
{
  "image_filename": "photo.jpg",
  "swaps": [...]
}
Response:
  { swaps: [..., { mask_b64, bbox, confidence }] }
```

### POST /api/inpaint/
```json
{
  "image_filename": "photo.jpg",
  "swaps": [...],
  "target_country": "India"
}
Response:
  { result_image_b64, step_images: [{ swap_index, image_b64 }] }
```

## Configuration

Environment variables (set before running):
```bash
GROUNDING_DINO_CONFIG   # path to GD config file
GROUNDING_DINO_WEIGHTS  # path to GD checkpoint
SAM_CHECKPOINT          # path to SAM checkpoint
SAM_MODEL_TYPE          # vit_h | vit_l | vit_b
SD_INPAINT_MODEL        # HuggingFace model ID for inpainting
```

## Model Choices

| Model | Purpose | Why this choice |
|-------|---------|-----------------|
| `gpt-4o` | Cultural analysis | Best VLM for nuanced cultural understanding |
| `GroundingDINO SwinT` | Object detection | Open-vocab detection from text prompts |
| `SAM ViT-H` | Precise masking | State-of-art segmentation from bbox prompt |
| `runwayml/stable-diffusion-inpainting` | Image generation | Intentionally "lower quality" SD 1.x model — shows raw pipeline effort |

## Notes

- The SD inpainting model is **intentionally** set to 20 inference steps (vs typical 50) with the lower-quality `runwayml` v1 model to create visible "pipeline effort" in outputs
- Models are lazily loaded on first request — first inference will be slow
- The frontend runs purely as a static HTML file with no build step
