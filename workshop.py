"""Tools to make interactive use of Stable Diffusion (SD) easier.

Classes:
    StableWorkshop - use SD in an interactive fashion

Functions:
    load_learned_embed_in_clip - load a learned embedding
    make_seeds - generate random seeds
"""

# bug-fix and easy
# TODO: clean up "undraft"
# TODO: implement seed in upscaling
# TODO: when upscaling images, make sure the metadata is traceable

# long-term
# TODO: when loading images, set the hash to the loaded one?
# TODO: average images in latent space
# TODO: "working" image, which can be set or loaded from file

# pylint: disable=no-member, no-name-in-module
from copy import copy
import gc
from typing import Iterable, Union
import warnings

from diffusers import LMSDiscreteScheduler, PNDMScheduler
from diffusers.training_utils import set_seed
import numpy as np
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from torch import autocast, cuda
from torchvision import transforms
from tqdm import tqdm

from gobig import upscale, gobig
from image import StableImage, StableGallery
from prompt import StablePrompt
from settings import SEEDS, StableSettings
from pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline


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


def make_seeds(n: int = 6):
    # pylint: disable=invalid-name
    """Make `n` random seeds."""
    return list(np.random.randint(0, 10000000, n))


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
        self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
            f"CompVis/stable-diffusion-v1-{version}",
            torch_dtype=torch.float16,
            revision="fp16",
            use_auth_token=True,
            **kwargs,
        )
        self._pipe.set_progress_bar_config(leave=False, position=1)
        self._pipe = self._pipe.to("cuda")
        self._pipe.enable_attention_slicing()

    def _init_image(self, settings: StableSettings) -> Image.Image:
        latents = torch.randn(
            (
                1,
                self._pipe.unet.in_channels,
                settings.height // 8,
                settings.width // 8,
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

    def _render(
        self,
        settings: StableSettings,
        init_image: Union[Image.Image, StableImage] = None,
        num: int = 1,
    ) -> Iterable[Image.Image]:
        set_seed(settings.seed)
        prompt = [str(self.prompt)] * num
        if init_image is None:
            init_image = self._init_image(settings)
            mask = Image.new(
                "L", (settings.width, settings.height), 255
            )
        elif isinstance(init_image, StableImage):
            mask = init_image.mask
            init_image = init_image.image
        with autocast("cuda"):
            result = self._pipe(
                prompt,
                init_image=init_image,
                mask_image=mask,
                strength=settings.strength,
                guidance_scale=settings.cfg,
                num_inference_steps=settings.iters,
            )
        gc.collect()
        cuda.empty_cache()
        return result["images"]

    def _render_loop(
        self,
        inits: Union[StableImage, StableGallery] = None,
        seeds: Union[int, Iterable] = None,
        skip_same: bool = True,
        show: bool = True,
        **kwargs,
    ):
        settings = self.settings.copy(**kwargs)
        if isinstance(seeds, int):
            seeds = [seeds]
        seeds = seeds or [settings.seed]
        if isinstance(inits, StableImage):
            inits = [inits]
        elif inits is None:
            inits = [None]

        pbar = tqdm(
            total=len(inits) * len(seeds),
            desc="batch progress",
            position=0,
            leave=True,
        )
        for seed in seeds:
            settings.seed = int(seed)
            for init in inits:
                if init is not None and init.settings.seed == settings.seed:
                    if skip_same:
                        warnings.warn(
                            "The current seed and the seed used to generate "
                            "this image are the same. Because `skip_same` is "
                            "True, this generation will be skipped."
                        )
                        pbar.update(1)
                        continue
                    warnings.warn(
                        "The current seed and the seed used to generate this "
                        "image are the same. This can lead to undesired "
                        'effects, like "burn-in"'
                    )
                image = self._render(settings=settings, init_image=init)[0]
                image = StableImage(
                    prompt=self.prompt,
                    settings=settings,
                    image=image,
                    init=init,
                )
                if self._draft:
                    self.drafted.append(image)
                else:
                    self.generated.append(image)
                if show is True:
                    image.show()
                pbar.update(1)

        pbar.close()

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
        self.generated = StableGallery()
        self.drafted = StableGallery()

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

        Additional kwargs will be used for this method call only.

        `strength` will be temporarily set to 1.0 for this method call,
        regardless of internal settings or kwargs.

        Any generated images will be added to `generated` or to `drafted`,
        depending on the use of `draft_on/draft_off`.
        """
        kwargs.pop("strength", None)
        self._render_loop(
            inits=None, seeds=seeds, show=show, strength=1.0, **kwargs
        )

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

        Additional kwargs will be used for this method call only.

        Any generated images will be added to `generated` or to `drafted`,
        depending on the use of `draft_on/draft_off`.
        """
        if not self.drafted:
            raise RuntimeError(
                "Draft something with `draft_on` before tuning."
            )
        inits = self.drafted[idxs]
        self._render_loop(
            inits=inits, seeds=seeds, skip_same=skip_same, show=show, **kwargs
        )


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

        Additional kwargs will be used for this method call only.

        Any generated images will be added to `generated` or to `drafted`,
        depending on the use of `draft_on/draft_off`.
        """
        if not self.generated:
            raise RuntimeError(
                "Generate something with `draft_off` before refining."
            )
        inits = self.generated[idxs]
        self._render_loop(
            inits=inits, seeds=seeds, skip_same=skip_same, show=show, **kwargs
        )

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
                self.draft_off()
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
        self.generated.save()

