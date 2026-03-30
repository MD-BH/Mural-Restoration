import argparse
import os
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image, ImageFilter
from safetensors.torch import load_model
from transformers import CLIPTextModel
from diffusers import UniPCMultistepScheduler
# Add ControlNet imports
from diffusers.pipelines.controlnet.pipeline_controlnet import ControlNetModel
from powerpaint.pipelines.pipeline_PowerPaint_ControlNet import (
    StableDiffusionControlNetInpaintPipeline as ControlNetPipeline,
)
from controlnet_aux import HEDdetector, OpenposeDetector 
# Standard imports
from powerpaint.models.BrushNet_CA import BrushNetModel
from powerpaint.models.unet_2d_condition import UNet2DConditionModel
from powerpaint.pipelines.pipeline_PowerPaint import StableDiffusionInpaintPipeline as Pipeline
from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import StableDiffusionPowerPaintBrushNetPipeline
from powerpaint.utils.utils import TokenizerWrapper, add_tokens

def add_task(prompt, negative_prompt, control_type, version):
    pos_prefix = neg_prefix = ""
    # Task Prompt Handling
    if control_type == "object-removal" or control_type == "image-outpainting":
        if version == "ppt-v1":
            pos_prefix = "empty scene blur " + prompt
            neg_prefix = negative_prompt
        promptA = pos_prefix + " P_ctxt"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + " P_obj"
        negative_promptB = neg_prefix + " P_obj"
    elif control_type == "context-aware":
        # Custom mode for repair/restoration using P_ctxt (Context) but without "empty scene" forcing
        if version == "ppt-v1":
             pos_prefix = prompt
             neg_prefix = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry "
        promptA = pos_prefix + " P_ctxt"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + " P_obj"
        negative_promptB = neg_prefix + " P_obj"
    elif control_type == "shape-guided":
        if version == "ppt-v1":
            pos_prefix = prompt
            neg_prefix = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry "
        promptA = pos_prefix + " P_shape"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + "P_shape"
        negative_promptB = neg_prefix + "P_ctxt"
    else: # text-guided (default)
        if version == "ppt-v1":
            pos_prefix = prompt
            neg_prefix = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry "
        promptA = pos_prefix + " P_obj"
        promptB = pos_prefix + " P_obj"
        negative_promptA = neg_prefix + "P_obj"
        negative_promptB = neg_prefix + "P_obj"

    return promptA, promptB, negative_promptA, negative_promptB

class PowerPaintBatchInference:
    def __init__(self, weight_dtype="float16", checkpoint_dir="./checkpoints/ppt-v1", version="ppt-v1", device="cuda"):
        self.version = version
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Initialize ControlNet attributes
        self.control_pipe = None
        self.control_type = None
        self.hed_detector = None
        self.openpose_detector = None

        # Determine dtype
        self.torch_dtype = torch.float16 if weight_dtype == "float16" else torch.float32
        
        # Determine device (handle macos mps)
        if self.device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                print("CUDA not available, using MPS")
                self.device = "mps"
            else:
                print("CUDA not available, using CPU")
                self.device = "cpu"
                
        print(f"Loading model ({version}) on {self.device}...")
        self._load_model()
        
    def _load_model(self):
        if self.version == "ppt-v1":
            # Initialize Pipeline
            self.pipe = Pipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=self.torch_dtype,
                local_files_only=False,
                safety_checker=None,
            )
            self.pipe.tokenizer = TokenizerWrapper(
                from_pretrained="runwayml/stable-diffusion-v1-5",
                subfolder="tokenizer",
                revision=None,
                torch_type=self.torch_dtype,
                local_files_only=False,
            )
            
            # Add tokens
            add_tokens(
                tokenizer=self.pipe.tokenizer,
                text_encoder=self.pipe.text_encoder,
                placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
                initialize_tokens=["a", "a", "a"],
                num_vectors_per_token=10,
            )

            # Load weights
            load_model(self.pipe.unet, os.path.join(self.checkpoint_dir, "unet/unet.safetensors"))
            load_model(self.pipe.text_encoder, os.path.join(self.checkpoint_dir, "text_encoder/text_encoder.safetensors"))
            self.pipe = self.pipe.to(self.device)
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

    def load_controlnet(self, control_type="canny"):
        if self.control_pipe is None or self.control_type != control_type:
            print(f"Loading ControlNet: {control_type}...")
            
            # Create local cache directory for ControlNet models
            local_cache_dir = Path(self.checkpoint_dir).resolve().parent / "controlnet"
            local_cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"ControlNet cache dir: {local_cache_dir}")

            # Automatically download from HuggingFace if not present in cache
            if control_type == "canny":
                base_control = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-canny", torch_dtype=self.torch_dtype, cache_dir=str(local_cache_dir)
                )
            elif control_type == "pose":
                base_control = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-openpose", torch_dtype=self.torch_dtype, cache_dir=str(local_cache_dir)
                )
                print("Loading OpenPose Detector...")
                self.openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet", cache_dir=str(local_cache_dir))
            elif control_type == "hed":
                base_control = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-hed", torch_dtype=self.torch_dtype, cache_dir=str(local_cache_dir)
                )
                if self.hed_detector is None:
                    print("Loading HED Detector...")
                    self.hed_detector = HEDdetector.from_pretrained("lllyasviel/ControlNet", cache_dir=str(local_cache_dir))
            elif control_type == "depth":
                base_control = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-depth", torch_dtype=self.torch_dtype, cache_dir=str(local_cache_dir)
                )

            # Create ControlNet Pipeline reusing components
            self.control_pipe = ControlNetPipeline(
                self.pipe.vae,
                self.pipe.text_encoder,
                self.pipe.tokenizer,
                self.pipe.unet,
                base_control,
                self.pipe.scheduler,
                None,
                None,
                False,
            )
            self.control_pipe = self.control_pipe.to(self.device)
            self.control_type = control_type

    def predict(self, input_image, input_mask, prompt, task, fitting_degree=1.0, ddim_steps=45, scale=7.5, seed=42, negative_prompt="", enable_control=False, control_type="canny", controlnet_conditioning_scale=0.5):
        # Ensure input images are RGB
        input_image = input_image.convert("RGB")
        input_mask = input_mask.convert("RGB")

        # Resize logic
        size1, size2 = input_image.size
        # Ensure divisible by 8
        W = int(size1 - size1 % 8)
        H = int(size2 - size2 % 8)
        input_image = input_image.resize((W, H))
        input_mask = input_mask.resize((W, H))

        # Handle ControlNet pre-processing
        control_image = input_image
        if enable_control:
            self.load_controlnet(control_type)
            if control_type == "hed":
                  # Use full-image HED map directly as ControlNet condition.
                  control_image = self.hed_detector(input_image)
                 
            elif control_type == "pose":
                 control_image = self.openpose_detector(input_image)
            # dim check for canny/depth normally needed but simplified here
        
        # Task specific prompt modification

        if self.version != "ppt-v1":

             if task == "image-outpainting":
                prompt = prompt + " empty scene"
             if task == "object-removal":
                prompt = prompt + " empty scene blur"

        promptA, promptB, negative_promptA, negative_promptB = add_task(prompt, negative_prompt, task, self.version)
        print(f"Running Task: {task}")
        print(f"Prompt A: {promptA}")

        # Set seed
        generator = torch.Generator(self.device).manual_seed(seed)

        if self.version == "ppt-v1":
             if enable_control and self.control_pipe:
                print(f"Running Inference with ControlNet: {control_type} (scale={controlnet_conditioning_scale})")
                
                # Check control image type needed
                # For HED/Canny, controlnet expects processed image usually.
                # app.py uses HEDdetector/OpenposeDetector which returns PIL Image
                
                result = self.control_pipe(
                    promptA=promptA,
                    promptB=promptB,
                    tradoff=fitting_degree,
                    tradoff_nag=fitting_degree,
                    negative_promptA=negative_promptA,
                    negative_promptB=negative_promptB,
                    image=input_image,
                    mask=input_mask,
                    control_image=control_image, # Passed to ControlNet
                    width=W,
                    height=H,
                    guidance_scale=scale,
                    num_inference_steps=ddim_steps,
                    generator=generator,
                    controlnet_conditioning_scale=controlnet_conditioning_scale
                ).images[0]
                return result
             
             # Standard PowerPaint Inference without ControlNet
             result = self.pipe(
                promptA=promptA,
                promptB=promptB,
                tradoff=fitting_degree,
                tradoff_nag=fitting_degree,
                negative_promptA=negative_promptA,
                negative_promptB=negative_promptB,
                image=input_image,
                mask=input_mask,
                width=W,
                height=H,
                guidance_scale=scale,
                num_inference_steps=ddim_steps,
                generator=generator
            ).images[0]
            
             return result
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PowerPaint Batch Inference with ControlNet")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the input image (with damage or blank)")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the mask image")
    parser.add_argument("--output_path", type=str, default="output.jpg", help="Path to save the result")
    parser.add_argument("--prompt", type=str, default="restored mural painting, high quality", help="Text prompt for inpainting")
    parser.add_argument("--scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--control_scale", type=float, default=0.5, help="ControlNet conditioning scale (0.0 - 1.0)")
    parser.add_argument("--control_type", type=str, default="hed", choices=["canny", "hed", "depth", "pose"], help="ControlNet type")
    
    args = parser.parse_args()

    # Initialize Inference
    inference = PowerPaintBatchInference(checkpoint_dir="./checkpoints/ppt-v1", version="ppt-v1")
    
    # Load Images
    if not os.path.exists(args.img_path):
        print(f"Error: Image not found at {args.img_path}")
        exit(1)
    if not os.path.exists(args.mask_path):
        print(f"Error: Mask not found at {args.mask_path}")
        exit(1)

    print(f"Processing: {args.img_path}")
    img = Image.open(args.img_path).convert("RGB")
    mask = Image.open(args.mask_path).convert("RGB") # Can be L or RGB

    # Run Inference with ControlNet (Scheme A: Complete Image -> ControlNet)
    result = inference.predict(
        input_image=img,          
        input_mask=mask,          
        prompt=args.prompt, 
        task="text-guided",       
        enable_control=True,      # Enable ControlNet to see unmasked image details
        control_type=args.control_type,     
        controlnet_conditioning_scale=args.control_scale
    )

    if result:
        result.save(args.output_path)
        print(f"Saved result toward: {args.output_path}")
    else:
        print("Inference failed.")
