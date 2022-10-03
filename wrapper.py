"""Tools to make interactive use of Stable Diffusion (SD) easier.

Classes:
    StableSettings - contain settings for SD
    StableImage - contain SD image and its generation information
    StableWorkshop - use SD in an interactive fashion

Functions:
    show_image_grid - display images in a grid
    load_learned_embed_in_clip - load a learned embedding
"""

# TODO: memory-saving tricks lead to multiple images generated per batch?
# TODO: average images in latent space
# TODO: "working" image, which can be set or loaded from file
# TODO: interactive inpainting?

# pylint: disable=no-member

import os
from copy import copy
import gc
from math import sqrt, ceil
from typing import Callable, Iterable, Union
import warnings
import yaml

# pylint: disable=no-name-in-module
from diffusers import (
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionImg2ImgPipeline,
)
from diffusers.training_utils import set_seed
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from torch import autocast, cuda
from torchvision import transforms

from gobig import upscale, gobig
from prompt import StablePrompt


SEEDS = [1337, 271, 314159, 41245, 59017, 61023]


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


def load_learned_embed_in_clip(
    learned_embeds_path: str,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    token: str = None,
):
    """Load (in place) a learned embedding into a CLIP model.

    Args:
        learned_embeds_path (str): path to the '.bin' file for learned weights
        text_encoder (CLIPTextModel): CLIP text encoder
        tokenizer (CLIPTokenizer): CLIP tokenizer
        token (str): string used to represent the token;
            default: load from file
    """
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

    # separate token and the embeds
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]

    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)

    # add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {token}. Please pass a "
            "different `token` that is not already in the tokenizer."
        )

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds

    # log to screen
    print(f"Added token '{token}' to the CLIP model.")


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
        image. 1.0 completely changes the image.

        0.0-0.5: Stay very close to the original input. Not recommended.
        0.6-0.7: Keep structure from the input, but change details.
        0.8-0.9: Change many details and some structure from the input.
        1.0: Use input as a jumping off point, but allow anything to change.
    """

    def __init__(self, **kwargs):
        self.height = kwargs.pop("height", 512)
        self.width = kwargs.pop("width", 512)
        self.seed = kwargs.pop("seed", SEEDS[0])
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
        open: open an image from file
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

    @classmethod
    def open(cls, fpath: str):
        """Open an image from file.

        Args:
            fpath (str): path to file
        """
        return StableImage(
            prompt=fpath, settings=None, image=Image.open(fpath)
        )

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
        drafted (list of StableImage): all drafted images

        Additionally, all attributes of `prompt` and `settings` are accessible
        as attributes within the StableWorkshop class.

    Methods:
        load_token - load a custom-trained token
        reset - reset workshop to default values
        draft_mode - enable drafting
        draft_off - disable drafting
        show - display generated images
        show_drafted - display drafted images
        hallucinate - generate an image from scratch
        tune - generate an image using a `draft_mode` image as a template
        refine - generate an image using a `generated` image as a template
        upscale - make a generated image larger, with science
        grid_search - search across multiple idxs and seeds
        save - save all `generated` images

    When indexed:
        returns corresponding item in `generated`
    """

    def __init__(self, version="3", **kwargs):
        self._init_model(version)
        self.prompt = StablePrompt(**kwargs)
        self.settings = StableSettings(**kwargs)
        self.generated = []
        self.drafted = []
        self._draft = False
        for key in self.prompt.dict:
            fget = lambda self, k=key: self.prompt[k]
            fset = lambda self, value, k=key: setattr(self.prompt, k, value)
            setattr(self.__class__, key, property(fget=fget, fset=fset))
        for key in self.settings.dict:
            fget = lambda self, k=key: self.settings[k]
            fset = lambda self, value, k=key: setattr(self.settings, k, value)
            setattr(self.__class__, key, property(fget=fget, fset=fset))

    def __len__(self):
        return len(self.generated)

    def __getitem__(self, idx):
        return self.generated[idx]

    def __setitem__(self, idx, new):
        self.generated[idx] = new

    def _init_model(self, version, **kwargs):
        version = str(version)
        if version not in [str(i) for i in range(1, 5)]:
            raise ValueError(f"`version` needs to be 1-4, not {version}")
        self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            f"CompVis/stable-diffusion-v1-{version}",
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
        gc.collect()
        cuda.empty_cache()
        return result["images"]

    def _update_settings(self, **kwargs):
        for key, value in kwargs.items():
            self.settings[key] = value

    def load_token(self, fpath: str, token: str = None):
        """Load a trained token into the model.

        Args:
            fpath (str): path to the '.bin' file for learned weights
            token (str): string used to represent the token;
                default: load from file
        """
        load_learned_embed_in_clip(
            fpath, self._pipe.text_encoder, self._pipe.tokenizer, token
        )

    def reset(self, **kwargs):
        """Reset the Workshop to default values.

        Accepts keyword arguments that can be passed to
        `StablePrompt` and `StableSettings`.

        Zeroes out `generated` and `drafted`.

        Exits draft mode.
        """
        self.draft_off()
        self.prompt = StablePrompt(**kwargs)
        self.settings = StableSettings(**kwargs)
        self.generated = []
        self.drafted = []

    def draft_mode(self, iters: int = 10):
        """Enable draft mode.

        When draft mode is enabled, the LMS (rather than PNDM) noise scheduler
        is used. This scheduler converges much more quickly (~10 iterations),
        but gives less-good results. It's best used for getting the feel of a
        seed and fine-tuning a prompt before committing to a longer run time.

        When draft mode is enabled, outputs are saved to `drafted` rather than
        `generated`.

        Args:
            iters (int): number of iterations to draft with
        """
        self._draft = True
        self._pipe.scheduler = LMSDiscreteScheduler(
            beta_start=self._pipe.scheduler.beta_start,
            beta_end=self._pipe.scheduler.beta_end,
            beta_schedule=self._pipe.scheduler.beta_schedule,
        )
        self.settings.iters = iters

    def draft_off(self, iters: int = 50):
        """Disable draft mode.

        When draft mode is disabled, the PNDM (rather than LMS) noise scheduler
        is used. The scheduler takes longer to converge (~50 iterations), but
        generally produces better results. It's best used when you've finished
        fine-tuning and are ready to generate your final result.

        Args:
            iters (int): number of iterations to generate with
        """
        self._draft = False
        self._pipe.scheduler = PNDMScheduler(
            beta_start=self._pipe.scheduler.beta_start,
            beta_end=self._pipe.scheduler.beta_end,
            beta_schedule=self._pipe.scheduler.beta_schedule,
            skip_prk_steps=True,
        )
        self.settings.iters = iters

    def show(self, **kwargs):
        """Show all generated images in a grid."""
        show_image_grid([gn.image for gn in self.generated], **kwargs)

    def show_drafted(self, **kwargs):
        """Show drafted images in a grid."""
        show_image_grid([bs.image for bs in self.drafted], **kwargs)

    def hallucinate(self, show: bool = True, **kwargs):
        """Generate an image from scratch.

        Args:
            show (bool): show the image after generation, default=True

        Additional kwargs (except strength) are updated in settings and will
        persist after calling this method.

        `strength` will be temporarily set to 1.0 for this method call,
        regardless of internal settings or kwargs.

        Any generated images will be added to `generated`.
        """
        strength = copy(self.settings.strength)
        self._update_settings(**kwargs)
        self.settings.strength = 1.0
        image = self._render()[0]
        image = StableImage(
            prompt=str(self.prompt), settings=self.settings, image=image
        )
        if self._draft:
            self.drafted.append(image)
        else:
            self.generated.append(image)
        self.settings.strength = strength
        if show is True:
            image.show()

    def tune(self, idx: int, show: bool = True, **kwargs):
        """Tune a `draft_mode` image into a (hopefully) better image.

        Args:
            idx (int): index of `drafted`
            show (bool): show the image after generation, default=True

        Additional kwargs are updated in settings and will persist after
        calling this method.

        Any generated images will be added to `generated`.
        """
        if not self.drafted:
            raise RuntimeError(
                "Draft something with `draft_mode` before tuning."
            )
        self._update_settings(**kwargs)
        if self.drafted[idx].settings.seed == self.settings.seed:
            warnings.warn(
                "The current seed and the seed used to generate the image are "
                'the same. This can lead to undesired effects, like "burn-in".'
            )
        init_image = self.drafted[idx].image.resize(
            (self.settings.width, self.settings.height)
        )
        image = self._render(init_image=init_image)[0]
        image = StableImage(
            prompt=str(self.prompt),
            settings=self.settings,
            image=image,
            init=self.drafted[idx],
        )
        if self._draft:
            self.drafted.append(image)
        else:
            self.generated.append(image)
        if show is True:
            image.show()

    def refine(self, idx: int, show: bool = True, **kwargs):
        """Refine a `generated` image.

        Can only be run after at least one of `hallucinate` or `tune`.

        Args:
            idx (int): index of `generated`
            show (bool): show the image after generation, default=True

        Additional kwargs are updated in settings and will persist after
        calling this method.

        Any generated images will be added to `generated`.
        """
        if not self.generated:
            raise RuntimeError(
                "Generate something with `draft_off` before tuning."
            )
        self._update_settings(**kwargs)
        if self.generated[idx].settings.seed == self.settings.seed:
            warnings.warn(
                "The current seed and the seed used to generate the image are "
                'the same. This can lead to undesired effects, like "burn-in".'
            )
        init_image = self.generated[idx].image.resize(
            (self.settings.width, self.settings.height)
        )
        image = self._render(init_image=init_image)[0]
        image = StableImage(
            prompt=str(self.prompt),
            settings=self.settings,
            image=image,
            init=self.generated[idx],
        )
        if self._draft:
            self.drafted.append(image)
        else:
            self.generated.append(image)
        if show is True:
            image.show()

    def upscale(
        self,
        idx: int,
        show: bool = True,
        render_more: bool = True,
        face_enhance: bool = True,
        **kwargs,
    ):
        """Upscale a `generated` image.

        Can only be run after at least one of `hallucinate` or `tune`.

        Args:
            idx (int): index of `generated`
            show (bool): show the image after generation, default=True
            render_more (bool): apply more passes with Stable Diffusion to the
                upscaled image, default=True
            face_enhance (bool): apply a face enhancement algorithm,
                default=True

        Additional kwargs are passed to `gobig`

        Any generated images will be added to `generated`.
        """
        if not self.generated:
            raise RuntimeError(
                "Generate something with `draft_off` before upscaling."
            )
        if self._draft:
            raise RuntimeError("You cannot upscale while in draft mode.")

        init_image = self.generated[idx].image
        if render_more:
            image = gobig(
                init_image,
                prompt=str(self.prompt),
                pipe=self._pipe,
                face_enhance=face_enhance,
                **kwargs,
            )
        else:
            image = upscale(init_image, face_enhance=face_enhance, **kwargs)
        image = StableImage(
            prompt=str(self.prompt),
            settings=self.settings,
            image=image,
            init=self.generated[idx],
        )
        self.generated.append(image)
        if show is True:
            image.show()

    def grid_search(
        self,
        idxs: Union[Iterable[int], int] = None,
        seeds: Union[Iterable[int], int] = None,
        func: Callable = None,
        hallucinate: bool = False,
        **kwargs,
    ):
        """Generate across multiple seeds and indices.

        Args:
            idxs (int or list of int): indices to search across
            seeds (int or list of int): seeds to search across,
                default: `SEEDS`
            func (callable): function e.g. `tune` or `refine`,
                default: `tune`
            hallucinate (bool): whether or not to hallucinate in addition
                to `func`, default: False

            Additional kwargs are used at render time for this call only.

        Any generated images will be added to `generated`.
        """
        start_seed = self.settings.seed
        if isinstance(idxs, int):
            idxs = [idxs]
        if isinstance(seeds, int):
            seeds = [seeds]
        seeds = seeds or SEEDS
        func = func or self.tune
        for seed in seeds:
            self.settings.seed = seed
            if hallucinate:
                self.hallucinate(**kwargs)
            for idx in idxs:
                func(idx, **kwargs)
        self.settings.seed = start_seed

    def save(self):
        """Save all `generated` images. See `StableImage.save`."""
        for image in self.generated:
            image.save()
