# pylint: disable=no-member


from math import sqrt, ceil

from diffusers.training_utils import set_seed
from PIL import Image
import torch
from torch import autocast
from torchvision import transforms

from img2img import StableDiffusionImg2ImgPipeline


def show_image_grid(imgs):
    rows = int(sqrt(len(imgs)))
    cols = int(ceil(len(imgs) / rows))
    width, height = imgs[0].size
    grid = Image.new("RGB", size=(cols * width, rows * height))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * width, i // cols * height))

    grid.show()


def init_model(version: str = "3", **kwargs):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        f"CompVis/stable-diffusion-v1-{version}",
        torch_dtype=torch.float16,
        revision="fp16",
        use_auth_token=True,
        **kwargs,
    )
    pipe = pipe.to("cuda")
    return pipe


def create_blank_tensor(pipe, height, width):
    latents = torch.randn(
        (1, pipe.unet.in_channels, height // 8, width // 8),
        device="cuda",
        dtype=torch.float16,
    )
    latents = 1 / 0.18215 * latents
    with autocast("cuda"):
        with torch.no_grad():
            tensor = pipe.vae.decode(latents)
    tensor = (tensor / 2 + 0.5).clamp(0, 1)
    return tensor


def render(
    prompt: str,
    init_image: Image = None,
    num: int = 1,
    height: int = None,
    width: int = None,
    seed: int = 1337,
    iters: int = 50,
    scale: float = 6.0,
    strength: float = None,
    show: bool = True,
    pipe: StableDiffusionImg2ImgPipeline = None,
    **kwargs,
):
    set_seed(seed)
    if pipe is None:
        pipe = init_model()
    if num > 1:
        prompt = [prompt] * num
        if width is None or height is None:
            height = 256
            width = 256
    elif width is None or height is None:
        height = 512
        width = 512
    if init_image is not None:
        init_image = init_image.resize((width, height))
        init_tensor = transforms.ToTensor()(init_image).unsqueeze_(0)
        if strength is None:
            strength = 0.75
    else:
        #  init_tensor = torch.rand((1, 3, height, width))
        init_tensor = create_blank_tensor(pipe, height, width)
        if strength is None:
            strength = 1.0

    with autocast("cuda"):
        images = pipe(
            prompt,
            guidance_scale=scale,
            num_inference_steps=iters,
            init_image=init_tensor,
            strength=strength,
            **kwargs,
        )

    if show:
        if num > 1:
            show_image_grid(images)
        else:
            images[0].show()

    return images
