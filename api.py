from functools import lru_cache
from hashlib import sha256

import torch
from diffusers import FluxPipeline
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

app = FastAPI()


class GenerationRequest(BaseModel):
    prompt: str
    height: int = 1024
    width: int = 1024
    num_images: int = 1
    guidance_scale: float = 4.5
    num_inference_steps: int = 40
    seed: int | None = None


@lru_cache()
def load_model():
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    pipe.load_lora_weights("./weights", adapter_name="default", weight_name="lora.safetensors")

    return pipe


@app.post("/generate")
def generate_image(request: GenerationRequest):
    pipe = load_model()

    prompt = request.prompt
    seed = request.seed

    if seed is not None:
        seed = (seed * 7057 + 57793) % 65213

        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    images: list[Image.Image] = pipe(
        prompt,
        height=request.height,
        width=request.width,
        num_images_per_prompt=request.num_images,
        generator=generator, 
        guidance_scale=request.guidance_scale,
        num_inference_steps=request.num_inference_steps,
    ).images

    image_paths = []
    for image in images:
        # calculate hash of image
        image_hash = sha256(image.tobytes()).hexdigest()
        image_path = f"./generated_images/{image_hash}.jpg"
        image.save(image_path)
        image_paths.append(image_path)

    return {"images": image_paths}
