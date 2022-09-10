"""Tools to make interactive use of Stable Diffusion (SD) easier.

Classes:
    StableSettings - contain settings for SD
    StablePrompt - contain prompt for SD
    StableImage - contain SD image and its generation information
    StableWorkshop - use SD in an interactive fashion

Functions:
    show_image_grid - display images in a grid
"""

# TODO: memory-saving tricks lead to multiple images generated per batch?
# TODO: trim brainstormed
# TODO: draft (with faster noise scheduler)
# TODO: garbage collection on the gpu
# TODO: average images in latent space
# TODO: "working" image, which can be set or loaded from file
# TODO: interactive inpainting?

# pylint: disable=no-member

import os
from copy import copy
from math import sqrt, ceil
from typing import Callable, Iterable, Union
import yaml

# pylint: disable=no-name-in-module
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.training_utils import set_seed
from PIL import Image
import torch
from torch import autocast
from torchvision import transforms


def show_image_grid(imgs, rows=None, cols=None):
    """Display multiple images at once, in a grid format."""
    if isinstance(imgs[0], StableImage):
        imgs = [i.image for i in imgs]
    rows = rows or int(sqrt(len(imgs)))
    cols = cols or int(ceil(len(imgs) / rows))
    width, height = imgs[0].size
    grid = Image.new("RGB", size=(cols * width, rows * height))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * width, i // cols * height))

    grid.show()


# TODO: sampling method (k_lms, k_ddim, k_euler_a, k_dpl_2_a)
# https://www.reddit.com/r/StableDiffusion/comments/x41n87/how_to_get_images_that_dont_suck_a/
class StableSettings:
    """Container for holding stable diffusion settings.

    Instance Attributes:
        height (int): image height in pixels, default=512
        width (int): image width in pixels, default=512
        seed (int): random seed for generating images, default=1337
        iters (int): number of diffusion steps for generation, default=50
        cfg (float): classifier free guidance, default=6.0
        strength (float): maintain original image, default=1.0
        dict (dict): represent setting as a dictionary

    Note that `height` and `width` must be multiples of 8. Weird results occur
    if both `height` and `width` are over 512. It is recommended to keep one
    value at 512 and vary the other.

    Guidelines for `cfg`:
        Classifier free guidance controls how closely the AI will match the
        image output to your text prompt.

        2-5: Allow the AI to hallucinate.
        6-10: Dynamic balance between human and AI.
        11-15: Strong inclusion of prompt. Only use for well-crafted prompts.
        16-20: Force prompt. Not recommended.

    Guidelines for `strength`:
        Strength controls how closely the AI will match the image output to the
        image input (as with tuning). A strength of 0.0 maintains the original
        image. 1.0 completely changes the image (and should be used for
        brainstorming and hallucinating.)

        0.0-0.5: Stay very close to the original input. Not recommended.
        0.6-0.7: Keep structure from the input, but change details.
        0.8-0.9: Change many details and some structure from the input.
        1.0: Use input as a jumping off point, but allow anything to change.
    """

    def __init__(self, **kwargs):
        self.height = kwargs.pop("height", 512)
        self.width = kwargs.pop("width", 512)
        self.seed = kwargs.pop("seed", 1337)
        self.iters = kwargs.pop("iters", 50)
        self.cfg = kwargs.pop("cfg", 6.0)
        self.strength = kwargs.pop("strength", 1.0)

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, idx):
        return self.__dict__[idx]

    def __setitem__(self, idx, value):
        self.__dict__[idx] = value

    @property
    def dict(self):
        """Access settings as a dictionary."""
        return self.__dict__


# TODO: init settings for different modes
class StablePrompt:
    """Container for holding Stable Diffusion Prompts.

    Instance Attributes:
        medium (str): describe type of image, default="oil painting on canvas"
        subject (str): composition subject, default="a fantasy landscape"
        artists (list of str): artist names to guide style,
            default=["Tyler Edlin", "Michael Whelan"]
        details (list of str): additional details to render in the image,
            default=["blue sky", "grass", "river"]
        modifiers (list of str): keywords which will make your image better,
            default=[
                "oil on canvas",
                "intricate",
                "4k resolution",
                "trending on artstation"
            ]
        dict (dict): represent prompt as a dictionary

    Read-Only Attributes:
        artist_str (str): represent `artists` as a string
        details_str (str): represent `artists` as a string
        modifiers_str (str): represent `artists` as a string
    """

    def __init__(
        self,
        medium="oil painting on canvas",
        subject="a fantasy landscape",
        **kwargs,
    ):
        self.medium = medium
        self.subject = subject
        self.artists = kwargs.pop("artists", ["Tyler Edlin", "Michael Whelan"])
        self.details = kwargs.pop("details", ["blue sky", "grass", "river"])
        self.modifiers = kwargs.pop(
            "modifiers",
            [
                "intricate",
                "4k resolution",
                "trending on artstation",
            ],
        )

    def __repr__(self):
        return (
            f"{self.medium} of {self.subject}{self.artist_str}"
            f"{self.details_str}{self.modifiers_str}"
        )

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    @property
    def dict(self):
        """Access prompt as a dictionary."""
        return self.__dict__

    @property
    def artist_str(self):
        """Convert list of artists into a prompt string."""
        if self.artists:
            artists = " and ".join(self.artists)
            return f" by {artists}"
        return ""

    @property
    def details_str(self):
        """Convert list of details into a prompt string."""
        details = [""] + self.details
        return ", ".join(details)

    @property
    def modifiers_str(self):
        """Convert list of modifiers into a string."""
        modifiers = [""] + self.modifiers
        return ", ".join(modifiers)


class StableImage:
    """Contain a generated image and the information used to generate it.

    Read-Only Instance Attributes:
        prompt (StablePrompt): the prompt used to generate the image
        settings (StableSettings): the settings used to generate the image
        init (StableImage): the seed image, if it exists
        hash (str): a unique hash for identifying the image

    Methods:
        show: display the image
        save: save the image and settings to file
    """

    def __init__(
        self,
        prompt: StablePrompt,
        settings: StableSettings,
        image: Image,
        init=None,
    ):
        self._prompt = copy(prompt)
        self._settings = copy(settings) or StableSettings()
        self._image = copy(image)
        self._init = init

    def __repr__(self):
        return f"StableImage: {self.hash}"

    def show(self):
        """Show the image."""
        self._image.show()

    def save(self):
        """Save the image and its associated settings.

        Image saves to generated/`hash`.png.
        Settings and prompt append to generated/logs.yaml.
        If `init` is not none, `init` saves as well.
        """
        try:
            os.mkdir("generated")
        except FileExistsError:
            pass

        self.image.save(f"generated/{self.hash}.png")
        with open("generated/logs.yaml", "a", encoding="utf-8") as logfile:
            logfile.write(
                yaml.dump(
                    {
                        self.hash: {
                            "prompt": str(self.prompt),
                            "settings": self.settings,
                            "init": str(self.init),
                        }
                    }
                )
            )

        if self.init:
            self.init.save()

    prompt = property(fget=lambda self: self._prompt)
    settings = property(fget=lambda self: self._settings)
    image = property(fget=lambda self: self._image)
    init = property(fget=lambda self: self._init)
    hash = property(
        fget=lambda self: f"{hash(self.image.tobytes()):x}".strip("-")
    )


class StableWorkshop:
    """An interactive tool for generating Stable Diffusion images.

    Instance Attributes:
        prompt (StablePrompt): the prompt used for generating
        settings (StableSettings): the settings used for generating
        generated (list of StableImage): all images that have been generated
        brainstormed (list of StableImage): all brainstormed images

        Additionally, all attributes of `prompt` and `settings` are accessible
        as attributes within the StableWorkshop class.

    Methods:
        reset - reset workshop to default values
        show_brainstormed - display brainstormed images
        show_generated - display generated images
        brainstorm - generate many low-quality images
        hallucinate - generate an image from scratch
        tune - generate an image using a `brainstorm`ed image as a template
        refine - generate an image using a `generated` image as a template
        grid_search - search across multiple idxs and seeds
        save - save all `generated` images
    """

    def __init__(self, **kwargs):
        self._init_model()
        self.prompt = StablePrompt(**kwargs)
        self.settings = StableSettings(**kwargs)
        self.generated = []
        self.brainstormed = []
        for key in self.prompt.dict:
            fget = lambda self, k=key: self.prompt[k]
            fset = lambda self, value, k=key: setattr(self.prompt, k, value)
            setattr(self.__class__, key, property(fget=fget, fset=fset))
        for key in self.settings.dict:
            fget = lambda self, k=key: self.settings[k]
            fset = lambda self, value, k=key: setattr(self.settings, k, value)
            setattr(self.__class__, key, property(fget=fget, fset=fset))

    def _init_model(self, **kwargs):
        self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-3",
            torch_dtype=torch.float16,
            revision="fp16",
            use_auth_token=True,
            **kwargs,
        )
        self._pipe = self._pipe.to("cuda")
        self._pipe.enable_attention_slicing()

    def _init_image(self):
        latents = torch.randn(
            (
                1,
                self._pipe.unet.in_channels,
                self.settings.height // 8,
                self.settings.width // 8,
            ),
            device="cuda",
            dtype=torch.float16,
        )
        latents = 1 / 0.18215 * latents
        with autocast("cuda"):
            with torch.no_grad():
                init_tensor = self._pipe.vae.decode(latents)
        init_tensor = (init_tensor["sample"] / 2 + 0.5).clamp(0, 1)
        return transforms.ToPILImage()(init_tensor[0])

    def _render(self, init_image: Image = None, num: int = 1):
        set_seed(self.settings.seed)
        prompt = [str(self.prompt)] * num
        if init_image is None:
            init_image = self._init_image()
        with autocast("cuda"):
            result = self._pipe(
                prompt,
                init_image=init_image,
                strength=self.settings.strength,
                guidance_scale=self.settings.cfg,
                num_inference_steps=self.settings.iters,
            )
        return result["images"]

    def _update_settings(self, **kwargs):
        for key, value in kwargs.items():
            self.settings[key] = value

    def reset(self, **kwargs):
        """Reset the Workshop to default values.

        Accepts keyword arguments that can be passed to
        `StablePrompt` and `StableSettings`.

        Zeroes out `generated` and `brainstormed`
        """
        self.prompt = StablePrompt(**kwargs)
        self.settings = StableSettings(**kwargs)
        self.generated = []
        self.brainstormed = []

    def show_brainstormed(self):
        """Show brainstormed images in a grid."""
        show_image_grid([bs.image for bs in self.brainstormed])

    def show_generated(self):
        """Show all generated images in a grid."""
        show_image_grid([gn.image for gn in self.generated])

    def brainstorm(self, num: int = 12, show: bool = True, **kwargs):
        """Generate many small images that can be used for `tune`ing.

        Args:
            num (int): number to generate, default=12
            show (bool): show a grid after generation, default=True

            Additional kwargs are used at render time for this call only.

        Images are stored in `brainstormed` and will be overwritten
        if `brainstorm` is called additional times.
        """
        kwargs["height"] = kwargs.pop("height", 256)
        kwargs["width"] = kwargs.pop("width", 256)
        settings = copy(self.settings)
        self._update_settings(**kwargs)

        images = self._render(num=num)
        self.brainstormed = [
            StableImage(prompt=self.prompt, settings=self.settings, image=i)
            for i in images
        ]

        self.settings = settings
        if show is True:
            self.show_brainstormed()

    def hallucinate(self, show: bool = True, **kwargs):
        """Generate an image from scratch.

        Args:
            show (bool): show the image after generation, default=True

            Additional kwargs (except strength) are updated in settings and
            will persist after calling this method.

        `strength` will be temporarily set to 1.0 for this method call,
        regardless of internal settings or kwargs.

        Any generated images will be added to `generated`.
        """
        strength = copy(self.settings.strength)
        self._update_settings(**kwargs)
        self.settings.strength = 1.0
        image = self._render()[0]
        self.generated.append(
            StableImage(
                prompt=str(self.prompt), settings=self.settings, image=image
            )
        )
        self.settings.strength = strength
        if show is True:
            image.show()

    def tune(self, idx: int, show: bool = True, **kwargs):
        """Tune a `brainstorm`ed image into a (hopefully) better image.

        Can only be run after `brainstorm`.

        Args:
            idx (int): index of `brainstormed`
            show (bool): show the image after generation, default=True

            Additional kwargs are updated in settings and
            will persist after calling this method.

        Any generated images will be added to `generated`.
        """
        if not self.brainstormed:
            raise RuntimeError("Cannot tune until we've `brainstorm`ed.")
        self._update_settings(**kwargs)
        init_image = self.brainstormed[idx].image.resize(
            (self.settings.width, self.settings.height)
        )
        image = self._render(init_image=init_image)[0]
        self.generated.append(
            StableImage(
                prompt=str(self.prompt),
                settings=self.settings,
                image=image,
                init=self.brainstormed[idx],
            )
        )
        if show is True:
            image.show()

    def refine(self, idx: int, show: bool = True, **kwargs):
        """Refine a `generated` image.

        Can only be run after at least one of `hallucinate` or `tune`.

        Args:
            idx (int): index of `generated`
            show (bool): show the image after generation, default=True

            Additional kwars are updated in settings and
            will persist after calling this method.

        Any generated images will be added to `generated`.
        """
        self._update_settings(**kwargs)
        init_image = self.generated[idx].image.resize(
            (self.settings.width, self.settings.height)
        )
        image = self._render(init_image=init_image)[0]
        self.generated.append(
            StableImage(
                prompt=str(self.prompt),
                settings=self.settings,
                image=image,
                init=self.generated[idx],
            )
        )
        if show is True:
            image.show()

    def grid_search(
        self,
        idxs: Union[Iterable[int], int],
        seeds: Union[Iterable[int], int],
        func: Callable = None,
        **kwargs,
    ):
        """Generate across multiple seeds and indices.

        Args:
            idxs (int or list of int): indices to search across
            seeds (int or list of int): seeds to search across,
                default: [271, 314159, 42, 57721, 60221023]
            func (callable): function e.g. `tune` or `refine`,
                default: `tune`

            Additional kwargs are used at render time for this call only.

        Any generated images will be added to `generated`.
        """
        self._update_settings(**kwargs)
        if isinstance(idxs, int):
            idxs = [idxs]
        if isinstance(seeds, int):
            seeds = [seeds]
        seeds = seeds or [271, 314159, 42, 57721, 60221023]
        func = func or self.tune
        for seed in seeds:
            self.settings.seed = seed
            for idx in idxs:
                func(idx)

    def save(self):
        """Save all `generated` images. See `StableImage.save`."""
        for image in self.generated:
            image.save()
