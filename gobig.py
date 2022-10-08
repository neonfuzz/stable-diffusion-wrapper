"""
Tools for intelligently uscaling images.

Functions:
    upscale - apply RealESRGAN
    gobig - upscale and fill in details with Diffusers


Some contents of this file are copyright (c) 2022 Jeffrey Quesnelle and
fall under the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import gc
from typing import List, Tuple
from tqdm import tqdm
from warnings import warn

# pylint: disable=import-error, no-name-in-module
from basicsr.archs.rrdbnet_arch import RRDBNet
from diffusers import StableDiffusionImg2ImgPipeline
from gfpgan import GFPGANer
import numpy as np
from PIL import Image, ImageDraw
from realesrgan import RealESRGANer
from torch import autocast, cuda


ESRGAN_MODELS = dict(
    rdb=RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    ),
)
ESRGAN_MODELS["upsampler"] = RealESRGANer(
    scale=4,
    model_path="/home/addie/opt/Real-ESRGAN/experiments/pretrained_models/"
    "RealESRGAN_x4plus.pth",
    model=ESRGAN_MODELS["rdb"],
    tile=512,
    tile_pad=128,
    pre_pad=0,
    half=False,
    gpu_id=0,
)
ESRGAN_MODELS["face_enhancer"] = GFPGANer(
    model_path="https://github.com/TencentARC/GFPGAN/releases/download/"
    "v1.3.0/GFPGANv1.3.pth",
    upscale=2,
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=ESRGAN_MODELS["upsampler"],
)


def upscale(
    img: Image.Image, scale: int = 2, face_enhance: bool = True
) -> Image.Image:
    """Upscale an image using RealESRGAN.

    Args:
        img (Image.Image or np.array): image to upscale
        scale (int): amount to scale, default: 2
        face_enhance (bool): apply GFPGAN to reconstruct faces, default: True

    Returns:
        Image.Image: upscaled image
    """

    to_pil = False
    if isinstance(img, Image.Image):
        to_pil = True
        img = np.array(img)[..., ::-1]

    if face_enhance:
        face_enhancer = ESRGAN_MODELS["face_enhancer"]
        face_enhancer.upscale = scale
        face_enhancer.bg_upsampler.model.to("cuda:0")
        face_enhancer.gfpgan.to("cuda:0")
        _, _, output = face_enhancer.enhance(
            img, has_aligned=False, only_center_face=False, paste_back=True
        )
        face_enhancer.bg_upsampler.model.to("cpu")
        face_enhancer.gfpgan.to("cpu")
    else:
        upsampler = ESRGAN_MODELS["upsampler"]
        upsampler.model.to("cuda:0")
        output, _ = upsampler.enhance(img, outscale=scale)
        upsampler.model.to("cpu")
    gc.collect()
    cuda.empty_cache()

    if to_pil is True:
        return Image.fromarray(output[..., ::-1])

    return output


def _grid_coords(target, original, overlap):
    # pylint: disable=invalid-name
    # generate a list of coordinate tuples for our sections, in order of how
    #   they'll be rendered
    # target should be the size for the gobig result, original is the size of
    #   each chunk being rendered
    center = []
    target_x, target_y = target
    center_x = int(target_x / 2)
    center_y = int(target_y / 2)
    original_x, original_y = original
    x = center_x - int(original_x / 2)
    y = center_y - int(original_y / 2)
    center.append((x, y))  # center chunk
    uy = y  # up
    uy_list = []
    dy = y  # down
    dy_list = []
    lx = x  # left
    lx_list = []
    rx = x  # right
    rx_list = []
    while uy > 0:  # center row vertical up
        uy = uy - original_y + overlap
        uy_list.append((lx, uy))
    while (dy + original_y) <= target_y:  # center row vertical down
        dy = dy + original_y - overlap
        dy_list.append((rx, dy))
    while lx > 0:
        lx = lx - original_x + overlap
        lx_list.append((lx, y))
        uy = y
        while uy > 0:
            uy = uy - original_y + overlap
            uy_list.append((lx, uy))
        dy = y
        while (dy + original_y) <= target_y:
            dy = dy + original_y - overlap
            dy_list.append((lx, dy))
    while (rx + original_x) <= target_x:
        rx = rx + original_x - overlap
        rx_list.append((rx, y))
        uy = y
        while uy > 0:
            uy = uy - original_y + overlap
            uy_list.append((rx, uy))
        dy = y
        while (dy + original_y) <= target_y:
            dy = dy + original_y - overlap
            dy_list.append((rx, dy))
    # calculate a new size that will fill the canvas, which will be optionally
    # used in grid_slice and go_big
    last_coordx, last_coordy = dy_list[-1:][0]
    render_edgey = (
        last_coordy + original_y
    )  # outer bottom edge of the render canvas
    render_edgex = (
        last_coordx + original_x
    )  # outer side edge of the render canvas
    scalarx = render_edgex / target_x
    scalary = render_edgey / target_y
    if scalarx <= scalary:
        new_edgex = int(target_x * scalarx)
        new_edgey = int(target_y * scalarx)
    else:
        new_edgex = int(target_x * scalary)
        new_edgey = int(target_y * scalary)
    # now put all the chunks into one master list of coordinates
    # (essentially reverse of how we calculated them so that the central slices
    # will be on top)
    result = []
    for coords in dy_list[::-1]:
        result.append(coords)
    for coords in uy_list[::-1]:
        result.append(coords)
    for coords in rx_list[::-1]:
        result.append(coords)
    for coords in lx_list[::-1]:
        result.append(coords)
    result.append(center[0])
    return result, (new_edgex, new_edgey)


# Chop our source into a grid of images that each equal the size of the
# original render
def _grid_slice(
    source: Image.Image, overlap: int, og_size: Tuple[int, int]
) -> Tuple[List[Tuple[Image.Image, int, int]], int]:
    # pylint: disable=invalid-name
    width, height = og_size  # size of the slices to be rendered
    coordinates, new_size = _grid_coords(source.size, og_size, overlap)
    # loc_width and loc_height are the center point of the goal size, and we'll
    # start there and work our way out
    slices = []
    for coordinate in coordinates:
        x, y = coordinate
        slices.append(((source.crop((x, y, x + width, y + height))), x, y))
    return slices, new_size


# Stitch slices back together.
def _stitch(
    img: Image.Image,
    slices: List,
    overlap: int,
    slice_size: Tuple[int, int],
) -> Image.Image:
    # Make alpha mask with faded gradient.
    alpha = Image.new("L", slice_size, color=0xFF)
    alpha_gradient = ImageDraw.Draw(alpha)
    i = 0
    shape = (slice_size, (0, 0))
    while i < overlap:
        alpha_gradient.rectangle(shape, fill=i * 4)
        i += 1
        shape = ((slice_size[0] - i, slice_size[1] - i), (i, i))

    # Apply mask to each chunk and merge into big image.
    img = img.convert("RGBA")
    for chunk, xpos, ypos in slices:
        chunk.putalpha(alpha)
        img.alpha_composite(chunk, (xpos, ypos))

    return img.convert("RGB")


def gobig(
    img: Image.Image,
    prompt: str,
    pipe: StableDiffusionImg2ImgPipeline,
    **kwargs,
) -> Image.Image:
    """Upscale an image with Diffusers.

    The image will be upscaled with RealESRGAN and optionally GFPGAN (see
    args). Pieces of the image will be re-rendered with a Diffusers pipeline,
    and then all images will be stitched back together.

    Args:
        img (Image.Image): the image to upscale
        prompt (str): text prompt used to guide diffusion
        pipe (StableDiffusionImg2ImgPipeline): diffusion pipeline
        overlap (int): degree to which image pieces overlap, default=128
        face_enhance (bool): use GFPGAN to enhance faces, default=True
        strength (float): amount to change underlying image with diffusers;
            0 is no change, 1 is complete change; default=0.3
        cfg (float): classifier free guidance, default=6.0
        diffuse_iters (int): number of diffusion iterations, default=50
        piece_size (Tuple[int]): size of each image chunk, default=(512, 512)

    Returns:
        Image.Image: upscaled image
    """
    overlap = kwargs.pop("overlap", 128)
    face_enhance = kwargs.pop("face_enhance", True)
    strength = kwargs.pop("strength", 0.3)
    cfg = kwargs.pop("cfg", 6.0)
    diffuse_iters = kwargs.pop("diffuse_iters", 50)
    piece_size = kwargs.pop("piece_size", (512, 512))

    pipe.to("cpu")
    cuda.empty_cache()
    img = upscale(img, face_enhance=face_enhance)
    pipe.to("cuda:0")
    cuda.empty_cache()

    better_slices = []
    slices, _ = _grid_slice(img, overlap, piece_size)
    try:
        for (chunk, coord_x, coord_y) in tqdm(
            slices, desc="batch progress", position=0
        ):
            with autocast("cuda"):
                result = pipe(
                    prompt,
                    init_image=chunk,
                    strength=strength,
                    guidance_scale=cfg,
                    num_inference_steps=diffuse_iters,
                )["images"][0]
            better_slices.append((result, coord_x, coord_y))
            gc.collect()
            cuda.empty_cache()
    except RuntimeError as err:
        warn(f"Full upscaling did not complete.\n\n{err}")
        gc.collect()
        cuda.empty_cache()
        return img

    return _stitch(img, better_slices, overlap, piece_size)
