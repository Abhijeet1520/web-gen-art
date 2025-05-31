import os
import io
import base64
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

import torch
from diffusers import StableDiffusionImg2ImgPipeline, LMSDiscreteScheduler

# --- Model config: Change here to add more models later
MODEL_CONFIG = {
    "sd_v1_4": {
        "repo": "CompVis/stable-diffusion-v1-4",
        "name": "Stable Diffusion v1.4"
    }
}
DEFAULT_MODEL = "sd_v1_4"

# --- Load pipeline at startup (single model for now)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = MODEL_CONFIG[DEFAULT_MODEL]["repo"]
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)
# Use LMSDiscreteScheduler as in the Colab
pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)

# --- FastAPI app
app = FastAPI(title="Simple SD Img2Img Server")

class GenerateRequest(BaseModel):
    prompt: str
    image_b64: str         # expects data:image/png;base64,.... or just base64 string
    strength: float = 0.75
    guidance_scale: float = 7.5
    seed: Optional[int] = 1024
    model: Optional[str] = DEFAULT_MODEL   # ready for extension

class GenerateResponse(BaseModel):
    image: str   # base64 PNG

@app.get("/")
async def root():
    return {
        "status": "ok",
        "available_models": list(MODEL_CONFIG.keys()),
        "device": device,
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    # Only allow the configured model for now
    if req.model != DEFAULT_MODEL:
        raise HTTPException(400, f"Only model '{DEFAULT_MODEL}' is supported right now.")

    # Decode image from base64
    if req.image_b64.startswith("data:"):
        b64_data = req.image_b64.split(",")[1]
    else:
        b64_data = req.image_b64
    try:
        input_img = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
        # Notebook uses 768x512
        input_img = input_img.resize((768, 512))
    except Exception as e:
        raise HTTPException(400, f"Could not decode input image: {str(e)}")

    # Run the model
    generator = torch.Generator(device=device).manual_seed(req.seed or 1024)
    result = pipe(
        prompt=req.prompt,
        image=input_img,
        strength=req.strength,
        guidance_scale=req.guidance_scale,
        generator=generator
    ).images[0]

    # Encode result to base64
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    b64_result = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"image": "data:image/png;base64," + b64_result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
