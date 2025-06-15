import os
import io
import base64
import asyncio
import json
import time
import sys
from typing import Optional, List, Set
import logging

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline, LMSDiscreteScheduler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sd_basic_server")

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# --- Model config: Simple single model setup
MODEL_CONFIG = {
    "sd_v1_4": {
        "repo": "CompVis/stable-diffusion-v1-4",
        "name": "Stable Diffusion v1.4",
        "description": "Classic stable diffusion model",
        "size_mb": 4000,
        "speed": "fast",
        "quality": "medium"
    }
}
DEFAULT_MODEL = "sd_v1_4"

# Keep track of active WebSocket connections
active_connections: Set[WebSocket] = set()

# --- Progress callback for reporting generation progress
class ProgressCallback:
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.completed_steps = 0
        self.start_time = time.time()

    def __call__(self, step: int, timestep: int, latents: torch.FloatTensor):
        self.completed_steps = step
        progress_pct = (step / self.total_steps) * 100
        elapsed = time.time() - self.start_time

        # Estimate remaining time
        if step > 0:
            time_per_step = elapsed / step
            remaining = time_per_step * (self.total_steps - step)
        else:
            remaining = 0

        # Log progress to console
        logger.info(f"Progress: {step}/{self.total_steps} ({progress_pct:.1f}%) - ETA: {remaining:.1f}s")

        # Store the progress data to be sent via WebSocket
        progress_data = {
            "type": "progress",
            "step": step,
            "total": self.total_steps,
            "percentage": progress_pct,
            "elapsed_seconds": round(elapsed, 1),
            "remaining_seconds": round(remaining, 1)
        }

        # Broadcast progress via websocket
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(broadcast_message(json.dumps(progress_data)))
        except Exception as e:
            logger.warning(f"Could not send progress via websocket: {e}")

# Broadcast a message to all connected WebSocket clients
async def broadcast_message(message: str):
    disconnected = set()
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except Exception:
            disconnected.add(connection)

    # Remove disconnected clients
    for conn in disconnected:
        active_connections.remove(conn)

# --- Load pipeline at startup - UPDATED to match working code exactly
logger.info("Initializing Stable Diffusion pipeline...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = MODEL_CONFIG[DEFAULT_MODEL]["repo"]
logger.info(f"Loading model: {model_path} on {device}")

# Match the working code's loading exactly
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
pipe = pipe.to(device)

# Use LMSDiscreteScheduler as in the working notebook
lms = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = lms

logger.info(f"Pipeline loaded successfully on {device}")

# --- FastAPI app with CORS
app = FastAPI(title="Simple SD Img2Img Server", description="Basic Stable Diffusion Image-to-Image API")

# Add CORS middleware to allow web requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# --- Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.75
    width: int = 512
    height: int = 512
    image_b64: str
    model: Optional[str] = DEFAULT_MODEL

class GenerateResponse(BaseModel):
    image: str

class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    size_mb: Optional[float] = None
    speed: str = "medium"
    quality: str = "medium"
    available: bool = True
    has_refiner: bool = False

class ServerStatusResponse(BaseModel):
    status: str
    models: List[str]
    model_info: List[ModelInfo]
    available_models: List[str]
    cuda_available: bool
    cuda_device: Optional[str] = None
    cuda_memory_gb: Optional[float] = None
    xformers_available: bool = True

class SuggestRequest(BaseModel):
    partial_prompt: str

class SuggestResponse(BaseModel):
    suggestions: List[str]

# Example prompts for texture transformations
TEXTURE_PROMPTS = [
    "transform into horror style, dark shadows, eerie lighting, abandoned atmosphere, create a haunting and unsettling environment with ominous tones",
    "convert to medieval aesthetic, stone castle, torch-lit walls, wooden beams, transform into an ancient fortress with weathered textures and rustic elements",
    "change to caveman style, primitive cave, stone age, tribal elements, turn into prehistoric scenery with natural pigments and crude markings",
    "transform into cyberpunk world, neon lights, digital interfaces, high-tech city, convert to futuristic urban dystopia with glowing elements and holographic displays",
    "remake as steampunk, brass gears, copper pipes, victorian style, change to alternate history with mechanical contraptions and ornate industrial designs",
    "convert to fantasy realm, magical elements, enchanted forest, mystical creatures, transform into a world of wonder with ethereal lighting and supernatural features",
    "change to sci-fi environment, futuristic technology, space station, alien world, remake with advanced technological elements and extraterrestrial landscapes",
    "transform into underwater scene, bubbles, coral reefs, seaweed, aquatic life, convert to submerged world with refracted light and flowing organic forms",
    "change to winter wonderland, snow-covered, icicles, frosted surfaces, soft blue lighting, create a frigid atmosphere with crystalline structures and muted palette",
    "convert to tropical paradise, lush jungle, vibrant flowers, exotic wildlife, transform into verdant rainforest with saturated colors and dense vegetation",
    "transform into desert landscape, sand dunes, cacti, heat waves, barren terrain, change to arid environment with sun-bleached textures and vast open spaces",
    "remake as celestial space, stars, nebulae, cosmic clouds, space background, convert to astronomical vista with interstellar phenomena and cosmic radiation",
    "change to industrial setting, factory interior, machinery, metal surfaces, transform into mechanized environment with utilitarian structures and raw materials",
    "convert to retro 80s style, synthwave, neon pink and blue, arcade aesthetic, remake with nostalgic elements and bold geometric patterns from the 1980s era",
    "transform into gothic architecture, ornate stonework, stained glass, gargoyles, change to medieval-inspired design with intricate details and dramatic verticality",
    "change to anime style, cel-shaded, vibrant colors, exaggerated features, convert to Japanese animation aesthetic with distinctive line work and stylized proportions",
    "transform into cartoon world, bold outlines, primary colors, simplified style, remake with exaggerated forms and playful, non-realistic elements",
    "convert to watercolor painting, paint texture, soft edges, artistic rendering, change to hand-painted appearance with pigment variations and gentle color bleeding",
    "transform into pixelated graphics, 8-bit style, retro game aesthetic, blocky textures, remake with deliberately limited resolution and nostalgic video game look",
    "change to post-apocalyptic wasteland, ruins, overgrown vegetation, decay, transform into devastated landscape with abandoned structures reclaimed by nature",
    "convert to vaporwave aesthetic, pastel colors, glitch effects, surreal elements, remake with internet-inspired nostalgia and deliberately artificial design",
    "transform into film noir, black and white, high contrast, dramatic shadows, deep blacks, change to cinematic style with mysterious mood and stark lighting",
    "change to minimalist design, clean lines, uncluttered space, monochromatic palette, convert to simplified elegant aesthetic with emphasis on negative space",
    "transform into baroque style, ornate details, gold accents, dramatic lighting, complex patterns, remake with elaborate decorative elements and rich textures",
    "convert to art deco, geometric patterns, luxury materials, symmetrical design, change to 1920s-inspired elegance with bold shapes and decorative opulence",
    "change to grunge texture, weathered surfaces, urban decay, worn materials, peeling paint, transform into distressed appearance with layers of history",
    "transform into surrealist dreamscape, impossible physics, strange combinations, floating elements, convert to subconscious-inspired imagery with unexpected juxtapositions",
    "change to isometric perspective, 45-degree angle view, game-like geometry, consistent depth, remake with technical precision and uniform directional representation",
    "convert to abstract composition, non-representational, shape-focused, color-field, transform into non-figurative arrangement with emphasis on form and color relationships",
    "change to luxury environment, gold and marble, ornate decorations, classical elements, transform into opulent setting with premium materials and sophisticated details"
]

# --- API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Simple Stable Diffusion Image-to-Image API is running",
        "available_models": [DEFAULT_MODEL],
        "model_info": [get_model_info()],
        "device": device,
    }

@app.get("/status", response_model=ServerStatusResponse)
async def server_status():
    """Get server status and available resources"""
    cuda = torch.cuda.is_available()

    response = {
        "status": "running",
        "models": [DEFAULT_MODEL],
        "model_info": [get_model_info()],
        "available_models": [DEFAULT_MODEL],
        "cuda_available": cuda,
        "xformers_available": True,  # Assuming it's available since we tried to enable it
    }

    if cuda:
        response["cuda_device"] = torch.cuda.get_device_name(0)
        response["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return response

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """Return detailed information about available models"""
    return [get_model_info()]

def get_model_info() -> ModelInfo:
    """Get model information"""
    config = MODEL_CONFIG[DEFAULT_MODEL]
    return ModelInfo(
        id=DEFAULT_MODEL,
        name=config["name"],
        description=config["description"],
        size_mb=config["size_mb"],
        speed=config["speed"],
        quality=config["quality"],
        available=True,
        has_refiner=False
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate image using img2img pipeline"""
    try:
        # Send initial progress message via WebSocket
        await broadcast_message(json.dumps({
            "type": "progress",
            "step": 0,
            "total": req.steps,
            "percentage": 0,
            "status": "starting",
            "message": f"Starting generation with model {req.model}"
        }))

        # Handle data URLs (they start with "data:")
        if req.image_b64.startswith("data:"):
            b64_data = req.image_b64.split(",")[1]
        else:
            b64_data = req.image_b64

        # Decode base64 to image - Match working code's processing
        try:
            input_img = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
            input_img = input_img.resize((768, 512))  # Exact size from working notebook
        except Exception as e:
            raise HTTPException(400, f"Could not decode input image: {str(e)}")

        try:
            generator = torch.Generator(device=device)

            # First try without callback to match notebook approach
            result = pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                image=input_img,
                strength=req.strength,
                guidance_scale=req.guidance_scale,
                num_inference_steps=req.steps,
                generator=generator
            ).images[0]

        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}")
            raise HTTPException(500, f"Error during image generation: {str(e)}")

        # Send progress updates
        await broadcast_message(json.dumps({
            "type": "progress",
            "step": req.steps,
            "total": req.steps,
            "percentage": 100,
            "status": "completed",
            "message": "Generation complete"
        }))

        # Encode result as data URL
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        b64_result = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {"image": f"data:image/png;base64,{b64_result}"}

    except Exception as e:
        logger.error(f"Error in generate: {str(e)}")
        import traceback
        traceback.print_exc()

        # Send error message via WebSocket
        await broadcast_message(json.dumps({
            "type": "error",
            "message": f"Error during generation: {str(e)}"
        }))

        raise HTTPException(500, f"Error generating image: {str(e)}")

@app.post("/suggest", response_model=SuggestResponse)
async def suggest(req: SuggestRequest):
    """Get prompt suggestions based on partial input"""
    try:
        p = req.partial_prompt.lower()
        sug = [s for s in TEXTURE_PROMPTS if p in s.lower()][:5]
        return {"suggestions": sug or TEXTURE_PROMPTS[:3]}
    except Exception as e:
        raise HTTPException(500, f"Error suggesting prompts: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time progress updates"""
    logger.info(f"New WebSocket connection request from {websocket.client}")
    try:
        await websocket.accept()
        logger.info(f"WebSocket connection accepted for {websocket.client}")
        active_connections.add(websocket)

        # Send initial status message
        status_msg = {
            "type": "status",
            "message": "Connected to server",
            "models": [DEFAULT_MODEL],
            "model_info": [get_model_info().model_dump()],
            "available_models": [DEFAULT_MODEL],
            "cuda_available": torch.cuda.is_available(),
            "xformers_available": True
        }

        if torch.cuda.is_available():
            status_msg["cuda_device"] = torch.cuda.get_device_name(0)
            status_msg["cuda_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)

        await websocket.send_text(json.dumps(status_msg))

        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {str(e)}")
                pass
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {websocket.client}")
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if websocket in active_connections:
            active_connections.remove(websocket)

# --- Main entry point
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Simple Stable Diffusion Image-to-Image API server")
    logger.info(f"Available models: {list(MODEL_CONFIG.keys())}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
