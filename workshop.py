"""Tools to make interactive use of Stable Diffusion (SD) easier.

Classes:
    StableWorkshop - use SD in an interactive fashion

Functions:
    load_learned_embed_in_clip - load a learned embedding
"""


# bug-fix and easy
# TODO: option to skip same seed during grid search
# TODO: when upscaling images, make sure the metadata is traceable
# TODO: when loading images, set the hash to the loaded one?

# long-term
# TODO: average images in latent space
# TODO: "working" image, which can be set or loaded from file
# TODO: interactive inpainting?

# pylint: disable=no-member, no-name-in-module
from copy import copy
import gc
from typing import Iterable, Union
import warnings

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
from image import StableImage, StableGallery, show_image_grid
from prompt import StablePrompt
from settings import SEEDS, StableSettings


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
        draft_on - enable drafting
        draft_off - disable drafting
        show - display generated images
        show_drafted - display drafted images
        hallucinate - generate an image from scratch
        tune - generate an image using a `draft_on` image as a template
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
        self.generated = StableGallery()
        self.drafted = StableGallery()
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

    def draft_on(self, iters: int = 10):
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
        """Show all generated images in a grid, with index labels."""
        self.generated.show(**kwargs)

    def show_drafted(self, **kwargs):
        """Show drafted images in a grid, with index labels."""
        self.drafted.show(**kwargs)

    def hallucinate(
        self, seeds: Union[int, Iterable] = None, show: bool = True, **kwargs
    ):
        """Generate an image from scratch.

        Args:
            seeds (int or iterable): seed(s) with which to hallucinate,
                default=`settings.seed`
            show (bool): show the image after generation, default=True

        Additional kwargs (except strength) are updated in settings and will
        persist after calling this method.

        `strength` will be temporarily set to 1.0 for this method call,
        regardless of internal settings or kwargs.

        Any generated images will be added to `generated` or to `drafted`,
        depending on the use of `draft_on/draft_off`.
        """
        strength = copy(self.settings.strength)
        seed = copy(self.settings.seed)
        self._update_settings(**kwargs)
        self.settings.strength = 1.0
        seeds = seeds or seed
        if isinstance(seeds, int):
            seeds = [seeds]

        for seed_ in seeds:
            self.settings.seed = seed_
            image = self._render()[0]
            image = StableImage(
                prompt=self.prompt, settings=self.settings, image=image
            )
            if self._draft:
                self.drafted.append(image)
            else:
                self.generated.append(image)
            if show is True:
                image.show()

        self.settings.strength = strength
        self.settings.seed = seed

    def tune(
        self,
        idxs: Union[int, Iterable],
        seeds: Union[int, Iterable] = None,
        skip_same: bool = True,
        show: bool = True,
        **kwargs,
    ):
        """Tune a `draft_on` image into a (hopefully) better image.

        Args:
            idxs (int or iterable): inde(x/ces) of `drafted`
            seeds (int or iterable): seed(s) with which to tune,
                default=`settings.seed`
            skip_same (bool): skip init images with the same seed,
                default: True
            show (bool): show the image after generation, default=True

        Additional kwargs are updated in settings and will persist after
        calling this method.

        Any generated images will be added to `generated` or to `drafted`,
        depending on the use of `draft_on/draft_off`.
        """
        if not self.drafted:
            raise RuntimeError(
                "Draft something with `draft_on` before tuning."
            )
        self._update_settings(**kwargs)
        if isinstance(idxs, int):
            idxs = [idxs]
        seed = copy(self.settings.seed)
        seeds = seeds or seed
        if isinstance(seeds, int):
            seeds = [seeds]

        for seed_ in seeds:
            self.settings.seed = seed_
            for idx in idxs:
                if self.drafted[idx].settings.seed == self.settings.seed:
                    if skip_same:
                        continue
                    warnings.warn(
                        "The current seed and the seed used to generate the "
                        "image are the same. This can lead to undesired "
                        'effects, like "burn-in".'
                    )
                init_image = self.drafted[idx].image.resize(
                    (self.settings.width, self.settings.height)
                )
                image = self._render(init_image=init_image)[0]
                image = StableImage(
                    prompt=self.prompt,
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

        self.settings.seed = seed

    def refine(
        self,
        idxs: Union[int, Iterable],
        seeds: Union[int, Iterable] = None,
        skip_same: bool = True,
        show: bool = True,
        **kwargs,
    ):
        """Refine a `generated` image.

        Can only be run after at least one of `hallucinate` or `tune`.

        Args:
            idxs (int or iterable): inde(x/ces) of `generated`
            seeds (int or iterable): seed(s) with which to tune,
                default=`settings.seed`
            skip_same (bool): skip init images with the same seed,
                default: True
            show (bool): show the image after generation, default=True

        Additional kwargs are updated in settings and will persist after
        calling this method.

        Any generated images will be added to `generated` or to `drafted`,
        depending on the use of `draft_on/draft_off`.
        """
        if not self.generated:
            raise RuntimeError(
                "Generate something with `draft_off` before tuning."
            )
        self._update_settings(**kwargs)
        if isinstance(idxs, int):
            idxs = [idxs]
        seed = copy(self.settings.seed)
        seeds = seeds or seed
        if isinstance(seeds, int):
            seeds = [seeds]

        for seed_ in seeds:
            self.settings.seed = seed_
            for idx in idxs:
                if self.drafted[idx].settings.seed == self.settings.seed:
                    if skip_same:
                        continue
                    warnings.warn(
                        "The current seed and the seed used to generate the "
                        "image are the same. This can lead to undesired "
                        'effects, like "burn-in".'
                    )
                init_image = self.generated[idx].image.resize(
                    (self.settings.width, self.settings.height)
                )
                image = self._render(init_image=init_image)[0]
                image = StableImage(
                    prompt=self.prompt,
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
        idxs: Union[int, Iterable],
        show: bool = True,
        render_more: bool = True,
        face_enhance: bool = True,
        **kwargs,
    ):
        """Upscale a `generated` image.

        Can only be run after at least one of `hallucinate` or `tune`.

        Args:
            idxs (int or iterable): inde(x/ces) of `generated`
            show (bool): show the image after generation, default=True
            render_more (bool): apply more passes with Stable Diffusion to the
                upscaled image, default=True
            face_enhance (bool): apply a face enhancement algorithm,
                default=True

        Additional kwargs are passed to `gobig`

        Any generated images will be added to `generated` or to `drafted`,
        depending on the use of `draft_on/draft_off`.
        """
        if not self.generated:
            raise RuntimeError(
                "Generate something with `draft_off` before upscaling."
            )
        if self._draft:
            raise RuntimeError("You cannot upscale while in draft mode.")

        if isinstance(idxs, int):
            idxs = [idxs]

        for idx in idxs:
            init_image = self.generated[idx].image
            if render_more:
                ws.draft_off()
                image = gobig(
                    init_image,
                    prompt=str(self.prompt),
                    pipe=self._pipe,
                    face_enhance=face_enhance,
                    **kwargs,
                )
            else:
                image = upscale(
                    init_image, face_enhance=face_enhance, **kwargs
                )
            settings = copy(self.settings)
            settings.width, settings.height = image.size
            settings.iters = kwargs.get("diffuse_iters", 50)
            settings.cfg = kwargs.get("cfg", 6.0)
            settings.strength = kwargs.get("strength", 0.3)
            image = StableImage(
                prompt=self.prompt,
                settings=settings,
                image=image,
                init=self.generated[idx],
            )
            self.generated.append(image)
            if show is True:
                image.show()

    def save(self):
        """Save all `generated` images. See `StableImage.save`."""
        for image in self.generated:
            image.save()
