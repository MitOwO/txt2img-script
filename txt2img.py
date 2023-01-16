#!/usr/bin/python
# Alencia Diffusion txt2img python engine
import base64
from io import BytesIO
import diffusers
import torch
import argparse
from torch import autocast

parser = argparse.ArgumentParser()

parser.add_argument('--prompt', type=str, required=True, help='Text for Stable Diffusion to generate Image')
parser.add_argument('--negative-prompt', type=str, required=False, help='Negative Prompt for image generation/prompt engineering')
parser.add_argument('--cfg-scale', type=float, required=False, help='Guidance Scale')
parser.add_argument('--sampler', type=str, required=False, help='Image Sampler')
parser.add_argument('--model', type=str, required=False, help='SD 1.x model name')
parser.add_argument('--width', type=int, required=False, help='Image Width')
parser.add_argument('--height', type=int, required=False, help='Image Height')
parser.add_argument('--steps', type=int, required=False, help='Inference Steps')
parser.add_argument('--xformers', type=bool, required=False, help='Xformers optimization attention')
# parser.add_argument('--nsfw', type=bool, required=False)

args = parser.parse_args()
# Configuration variables
prompt = args.prompt
model = args.model
cfgscale = args.cfg_scale
width = args.width
height = args.height
negative_prompt = args.negative_prompt
steps = args.steps
scheduler = args.sampler

# End of Configuration Variables

if model is None:
    model = "Linaqruf/anything-v3.0"
if cfgscale is None:
    cfgscale = 8.5
if width is None:
    width = 512
if height is None:
    height = 512
if negative_prompt is None:
    negative_prompt = "bad anatomy, lowres, medium resolution, ugly, blurred, letterbox, signature, text"
if steps is None:
    steps = 32
if scheduler is None:
    scheduler = "ddim"

alencia_pipeline = diffusers.StableDiffusionPipeline.from_pretrained(model)

def choose_scheduler(schedulertype):
    if schedulertype == "euler":
        return diffusers.EulerDiscreteScheduler.from_config(alencia_pipeline.scheduler.config)
    elif schedulertype == "euler_a":
        return diffusers.EulerAncestralDiscreteScheduler.from_config(alencia_pipeline.scheduler.config)
    else:
        return diffusers.DDIMScheduler.from_config(alencia_pipeline.scheduler.config)

alencia_pipeline.scheduler = choose_scheduler(scheduler)

nsfw_image = False

def override_safety_checker(images, clip_output):
    nsfw_image = True
    return images, False
alencia_pipeline.safety_checker = override_safety_checker

alencia_pipeline.enable_attention_slicing(1)
alencia_pipeline.enable_vae_slicing()
# alencia_pipeline.enable_sequential_cpu_offload()

if args.xformers is True:
    alencia_pipeline.enable_xformers_memory_efficient_attention()

alencia_pipeline = alencia_pipeline.to("cuda")
def img_2_b64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    return img_str

def generate_image():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    with autocast("cuda"):
        image = alencia_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfgscale
        ).images[0]
        print(img_2_b64(image))
