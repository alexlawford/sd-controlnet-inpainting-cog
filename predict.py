from cog import BasePredictor, Input, Path
import os
import torch
from PIL import Image
from diffusers import {
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel,
}

CONTROL_NAME = "lllyasviel/ControlNet"
OPENPOSE_NAME = "thibaud/controlnet-openpose-sdxl-1.0"
MODEL_NAME = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
CONTROL_CACHE = "control-cache"
POSE_CACHE = "pose-cache"
MODEL_CACHE = "model-cache"

class Predictor(BasePredictor):
    def setup(self):
        controlnet = ControlNetModel.from_pretrained(
            POSE_CACHE,
            torch_dtype=torch.float16
        )

        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            MODEL_CACHE,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

        self.pipe = pipe.to("cudad")

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An athletic man, white studio",
        ),
        image: Path = Input(description="The image to be inpainted"),
        mask_image: Path = Input(description="Mask for inpainting"),
        control_image: Path = Input(description="Control image"),
        seed: int = Input(description="Random seed. Set to 0 to randomize the seed", default=0),
    ) -> Path:
        # API guide:
        # https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet_sdxl#diffusers.StableDiffusionXLControlNetInpaintPipeline.__call__
        if (seed == 0) or (seed == None):
            seed = int.from_bytes(os.urandom(2), byteorder='big')
        generator = torch.Generator('cuda').manual_seed(seed)
        print("Using seed:", seed)

        # load each of our images. IF FOR UNCONTROLLED images input,
        # should add resize image, see original respository
        # all three images should be same size
        # likewise, convert mask to RGB.
        image = Image.open(image)
        mask_image = Image.open(mask_image)
        control_image = Image.open(control_image)
        
        output = self.pipe(
            image=image,
            mask_image=mask_image,
            control_image=control_image,
            prompt=prompt,
            seed=seed
        )

        output_path = "./output.png"
        output_image = output.images[0]
        output_image.save(output_path)

        return Path(output_path)