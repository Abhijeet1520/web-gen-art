import os
import io
import base64
import argparse
from typing import Dict, Any, Optional, List

import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLAdapterPipeline,
    StableDiffusionImg2ImgPipeline,
)

# Define request/response models
class GenRequest(BaseModel):
    model: str = "sdxl_base"
    prompt: str
    negative_prompt: Optional[str] = None
    steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.7
    width: int = 1024
    height: int = 768
    image_b64: str  # webcam PNG/JPEG

class GenResponse(BaseModel):
    image: str  # data URL

class SuggestRequest(BaseModel):
    partial_prompt: str

class SuggestResponse(BaseModel):
    suggestions: List[str]

# Initialize FastAPI app
app = FastAPI(title="Stable Diffusion Image-to-Image API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Model registry
MODELS: Dict[str, Dict[str, Any]] = {
    "sdxl_base": {
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "class": StableDiffusionXLImg2ImgPipeline,
        "refiner": None,
    },
    "sdxl_base+refiner": {
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "class": StableDiffusionXLImg2ImgPipeline,
        "refiner": "stabilityai/stable-diffusion-xl-refiner-1.0",
    },
    "sd_v1_4": {
        "repo": "CompVis/stable-diffusion-v1-4",
        "class": StableDiffusionImg2ImgPipeline,
        "refiner": None,
    },
    "sd_v1_5": {
        "repo": "runwayml/stable-diffusion-v1-5",
        "class": StableDiffusionImg2ImgPipeline,
        "refiner": None,
    },
}

# Cache for loaded models
CACHE: Dict[str, Any] = {}

# Example prompts for suggestions
EXAMPLE_PROMPTS = [
    "cyber-punk cat detective, neon rain, ultra-detailed",
    "steampunk airship over Victorian London, dusk lighting",
    "hyper-realistic portrait, 85 mm lens, soft rim light",
    "fantasy landscape with mountains, trending on artstation",
    "cosmic entity made of light and stars, ethereal, mystical",
    "photorealistic cityscape, golden hour, atmospheric",
    "dark forest with glowing mushrooms, fantasy concept art",
    "anime style character, vibrant colors, detailed",
    "underwater world with bioluminescent creatures, surreal",
    "abstract fluid art, colorful swirls, high-resolution render"
]

def get_pipeline(name: str, device: str, torch_dtype):
    """Lazy-load model pipelines when needed"""
    if name in CACHE:
        return CACHE[name]

    if name not in MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{name}' not found. Available models: {list(MODELS.keys())}")

    spec = MODELS[name]
    print(f"Loading model: {name} from {spec['repo']}...")

    try:
        # Load base model
        pipe = spec["class"].from_pretrained(
            spec["repo"],
            torch_dtype=torch_dtype,
            use_safetensors=True
        ).to(device)

        # Enable optimizations if available
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()

        if hasattr(pipe, "to") and torch.cuda.is_available():
            pipe.to(memory_format=torch.channels_last)

        # Load refiner if specified
        refiner = None
        if spec["refiner"]:
            refiner = StableDiffusionXLAdapterPipeline.from_pretrained(
                spec["refiner"],
                torch_dtype=torch_dtype,
                use_safetensors=True
            ).to(device)

            if hasattr(refiner, "enable_xformers_memory_efficient_attention"):
                refiner.enable_xformers_memory_efficient_attention()

            if hasattr(refiner, "to") and torch.cuda.is_available():
                refiner.to(memory_format=torch.channels_last)

        CACHE[name] = (pipe, refiner)
        return CACHE[name]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model '{name}': {str(e)}"
        )

@app.get("/")
async def root():
    return {"message": "Stable Diffusion Image-to-Image API is running"}

@app.get("/models", response_model=List[str])
async def list_models():
    return list(MODELS.keys())

@app.post("/generate", response_model=GenResponse)
async def generate(req: GenRequest):
    try:
        # Handle data URLs (they start with "data:")
        if req.image_b64.startswith("data:"):
            # Extract the base64 part
            b64_data = req.image_b64.split(",")[1]
        else:
            b64_data = req.image_b64

        # Decode base64 to image
        im = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
        im = im.resize((req.width, req.height))

        # Get the appropriate pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        pipe, refiner = get_pipeline(req.model, device, torch_dtype)

        # Process with base model
        base_out = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image=im,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            strength=req.strength,
            denoising_end=0.8 if refiner else 1.0
        ).images[0]

        # Process with refiner if available
        final_img = base_out
        if refiner:
            final_img = refiner(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                image=base_out,
                num_inference_steps=10,
                guidance_scale=req.guidance_scale,
                denoising_start=0.8
            ).images[0]

        # Encode result as data URL
        buf = io.BytesIO()
        final_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {"image": f"data:image/png;base64,{b64}"}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating image: {str(e)}"
        )

@app.post("/suggest", response_model=SuggestResponse)
async def suggest(req: SuggestRequest):
    try:
        p = req.partial_prompt.lower()
        sug = [s for s in EXAMPLE_PROMPTS if p in s.lower()][:5]
        return {"suggestions": sug or EXAMPLE_PROMPTS[:3]}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error suggesting prompts: {str(e)}"
        )

def main():
    parser = argparse.ArgumentParser(description="Run Stable Diffusion Image-to-Image API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--log-level", type=str, default="info", choices=["debug", "info", "warning", "error", "critical"], help="Logging level")

    args = parser.parse_args()

    # Print welcome message
    print(f"Starting Stable Diffusion Image-to-Image API server on {args.host}:{args.port}")
    print(f"Available models: {list(MODELS.keys())}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Start the server
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)

if __name__ == "__main__":
    main()
