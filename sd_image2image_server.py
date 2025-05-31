import os
import io
import base64
import argparse
import asyncio
import json
import time
import socket
from typing import Dict, Any, Optional, List, Set

import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLAdapterPipeline,  # Changed from StableDiffusionXLRefinerPipeline
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)

# Keep track of active WebSocket connections
active_connections: Set[WebSocket] = set()

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

class ServerStatusResponse(BaseModel):
    status: str
    models: List[str]
    cuda_available: bool
    cuda_device: Optional[str] = None
    cuda_memory_gb: Optional[float] = None

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

# Custom callback for reporting generation progress
class ProgressCallback:
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.completed_steps = 0
        self.start_time = time.time()

    async def __call__(self, step: int, timestep: int, latents: torch.FloatTensor):
        self.completed_steps = step
        progress_pct = (step / self.total_steps) * 100
        elapsed = time.time() - self.start_time

        # Estimate remaining time
        if step > 0:
            time_per_step = elapsed / step
            remaining = time_per_step * (self.total_steps - step)
        else:
            remaining = 0

        # Broadcast progress to all connected clients
        message = {
            "type": "progress",
            "step": step,
            "total": self.total_steps,
            "percentage": progress_pct,
            "elapsed_seconds": round(elapsed, 1),
            "remaining_seconds": round(remaining, 1)
        }

        await broadcast_message(json.dumps(message))

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

        # Use DPM++ scheduler for better quality/speed
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        # Enable optimizations if available
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()

        if hasattr(pipe, "to") and torch.cuda.is_available():
            pipe.to(memory_format=torch.channels_last)

        # Load refiner if specified
        refiner = None
        if spec["refiner"]:
            refiner = StableDiffusionXLAdapterPipeline.from_pretrained(  # Changed to StableDiffusionXLAdapterPipeline
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

@app.get("/status", response_model=ServerStatusResponse)
async def server_status():
    """Get server status and available resources"""
    response = {
        "status": "running",
        "models": list(MODELS.keys()),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        response["cuda_device"] = torch.cuda.get_device_name(0)
        response["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return response

@app.get("/models", response_model=List[str])
async def list_models():
    return list(MODELS.keys())

@app.post("/generate", response_model=GenResponse)
async def generate(req: GenRequest, background_tasks: BackgroundTasks):
    try:
        # Send initial progress message
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

        # Create callback for progress reporting
        callback = ProgressCallback(req.steps)

        # Process with base model
        base_out = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image=im,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            strength=req.strength,
            denoising_end=0.8 if refiner else 1.0,
            callback=callback,
            callback_steps=1
        ).images[0]

        # Process with refiner if available
        final_img = base_out
        if refiner:
            await broadcast_message(json.dumps({
                "type": "progress",
                "step": req.steps,
                "total": req.steps + 10,  # Add refiner steps
                "percentage": (req.steps / (req.steps + 10)) * 100,
                "status": "refining",
                "message": "Starting refinement"
            }))

            refiner_callback = ProgressCallback(10)  # 10 steps for refiner

            # Modification for StableDiffusionXLAdapterPipeline
            # Since the API might be different, we need to adapt our usage
            try:
                final_img = refiner(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    image=base_out,
                    num_inference_steps=10,
                    guidance_scale=req.guidance_scale,
                    adapter_conditioning_scale=0.8,  # Using this instead of denoising_start
                    callback=refiner_callback,
                    callback_steps=1
                ).images[0]
            except Exception as refiner_error:
                print(f"Refiner error: {str(refiner_error)}")
                # If the adapter approach fails, fall back to using base_out
                final_img = base_out
                await broadcast_message(json.dumps({
                    "type": "warning",
                    "message": f"Refinement failed, using base output: {str(refiner_error)}"
                }))

        # Encode result as data URL
        buf = io.BytesIO()
        final_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        # Send completion message
        await broadcast_message(json.dumps({
            "type": "progress",
            "step": req.steps + (10 if refiner else 0),
            "total": req.steps + (10 if refiner else 0),
            "percentage": 100,
            "status": "completed",
            "message": "Generation complete"
        }))

        return {"image": f"data:image/png;base64,{b64}"}

    except Exception as e:
        # Send error message to websocket clients
        await broadcast_message(json.dumps({
            "type": "error",
            "message": f"Error during generation: {str(e)}"
        }))

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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)

    # Send initial status message
    status_msg = {
        "type": "status",
        "message": "Connected to server",
        "models": list(MODELS.keys()),
        "cuda_available": torch.cuda.is_available()
    }

    if torch.cuda.is_available():
        status_msg["cuda_device"] = torch.cuda.get_device_name(0)
        status_msg["cuda_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)

    await websocket.send_text(json.dumps(status_msg))

    try:
        while True:
            # Keep connection alive and handle incoming messages if needed
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except:
                pass
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        if websocket in active_connections:
            active_connections.remove(websocket)

def find_available_port(start_port=8000, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            # Try to create a socket on the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

def main():
    parser = argparse.ArgumentParser(description="Run Stable Diffusion Image-to-Image API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--auto-port", action="store_true", help="Automatically find an available port if specified port is in use")
    parser.add_argument("--log-level", type=str, default="info", choices=["debug", "info", "warning", "error", "critical"], help="Logging level")

    args = parser.parse_args()

    # Try to find an available port if requested
    port = args.port
    if args.auto_port:
        try:
            port = find_available_port(start_port=port)
        except RuntimeError as e:
            print(f"Error: {str(e)}")
            return 1

    # Print welcome message
    print(f"Starting Stable Diffusion Image-to-Image API server on {args.host}:{port}")
    print(f"Available models: {list(MODELS.keys())}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Start the server
    try:
        uvicorn.run(app, host=args.host, port=port, log_level=args.log_level)
    except OSError as e:
        if "address already in use" in str(e).lower():
            print(f"\nError: Port {port} is already in use.")
            print("Use --auto-port to automatically find an available port or specify a different port with --port.")
            return 1
        raise

if __name__ == "__main__":
    main()
