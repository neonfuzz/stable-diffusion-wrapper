"""Inpainting pipeline tools.

Adapted from the HuggingFace Diffusers 0.3.0 implementation of
`pipeline_stable_diffusion_inpaint.py`.
"""


# pylint: disable=no-name-in-module, no-member
# pylint: disable=too-many-arguments, too-many-statements, too-many-locals, too-many-branches
import inspect
import re
from typing import List, Optional, Union

import numpy as np
import torch

import PIL

from diffusers.schedulers import LMSDiscreteScheduler
from diffusers.utils import logging
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionInpaintPipeline,
    StableDiffusionPipelineOutput,
)


logger = logging.get_logger(__name__)


def preprocess_image(image: PIL.Image.Image) -> torch.tensor:
    """Convert a PIL image into a tensor ready to use with `StablePipe`."""
    width, height = image.size
    width, height = map(
        lambda x: x - x % 32, (width, height)
    )  # Resize to integer multiple of 32.
    image = image.resize((width, height), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def preprocess_mask(mask: PIL.Image.Image) -> torch.tensor:
    """Convert a PIL mask into a tensor ready to use with `StablePipe`."""
    mask = mask.convert("L")
    width, height = mask.size
    width, height = map(
        lambda x: x - x % 32, (width, height)
    )  # Resize to integer multiple of 32.
    mask = mask.resize((width // 8, height // 8), resample=PIL.Image.NEAREST)
    mask = np.array(mask) / 255.0
    mask = np.tile(mask, (1, 4, 1, 1))
    mask = 1 - mask  # Repaint white, keep black.
    mask = torch.from_numpy(mask).to(torch.float16)
    return mask


class StablePipe(StableDiffusionInpaintPipeline):
    """Run the Stable Diffusion Pipeline from start to finish.

    Subclassed from StableDiffusion's inpainting pipeline (via diffusers).

    Additional methods:
        get_embeddings_with_modifiers: create weighted embeddings

    Differences when called:
        - Prompt modifiers allow you to control the contribution of certain
            modifiers by enclosing them in curly braces. For instance,
            "a cityscape {by Leonid Afremov:0.25}" will include
            "by Leonid Afremov" at 25% of the strength of the rest of the
            prompt. Separate multiple modifiers with the "|" character
            e.g., "a cityscape {by Leonid Afremov:0.25|by Lisa Frank:2}"
        - New kwarg `neg_input` allows you to control the conditional embedding
            (i.e., subtract it from the prompt)
        - Saftey check is not performed.
        - Classifier free guidance is always performed, allowing you to query
            negative strengths (i.e., opposite of the prompt).
    """

    @torch.no_grad()
    def get_embeddings_with_modifiers(
        self, prompt: Union[str, List[str]]
    ) -> torch.tensor:
        """Project a text prompt into the latent space.

        Prompt modifiers allow you to control the contribution of certain
        modifiers by enclosing them in curly braces. For instance,
        "a cityscape {by Leonid Afremov:0.25}" will include "by Leonid Afremov"
        at 25% of the strength of the rest of the prompt. Separate multiple
        modifiers with the "|" character
        e.g., "a cityscape {by Leonid Afremov:0.25|by Lisa Frank:2}"

        Args:
            prompt (str or list of str): text prompt(s)

        Returns:
            torch.tensor: embeddings
        """
        if isinstance(prompt, List):
            return torch.vstack(
                [self.get_embeddings_with_modifiers(p) for p in prompt]
            )
        # Extract modifiers and weights using regexes.
        try:
            modifier_string = "|".join(re.search(r"{(.*)}", prompt).groups())
        except AttributeError:
            texts = []
            weights = []
        else:
            modifiers = modifier_string.split("|")
            texts = [re.search(r"(.*):", m).groups()[0] for m in modifiers]
            weights = [
                float(re.search(r".*: *([-\d\.]+)", m).groups()[0])
                for m in modifiers
            ]
            for weight in weights:
                if weight <= 0:
                    raise NotImplementedError(
                        "Negative modifiers not yet implemented."
                    )

        # Include prompt without any modifiers with a weight of 1.0.
        texts.insert(0, re.sub(r"({.*})", "", prompt))
        weights.insert(0, 1)

        # Encode.
        text_input = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_input.input_ids.to(self.device)
        raw_embeddings = self.text_encoder(input_ids)[0]

        # Apply weights and return to correct shape.
        weights = torch.tensor(weights)[:, None, None].to(self.device)
        embeddings = (raw_embeddings * weights).sum(axis=0) / weights.sum()
        embeddings = embeddings[None, ...]
        return embeddings

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        init_image: Union[torch.FloatTensor, PIL.Image.Image],
        mask_image: Union[torch.FloatTensor, PIL.Image.Image],
        neg_input: Union[str, List[str]] = "",
        strength: float = 1.0,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 6,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        # pylint: disable=line-too-long
        # URLs in the markdown make it untenable to have shorter lines.
        r"""Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            init_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be
                used as the starting point for the process. This is the image
                whose masked region will be inpainted.
            mask_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask
                `init_image`. White pixels in the mask will be replaced by
                noise and therefore repainted, while black pixels will be
                preserved. The mask image will be converted to a single channel
                (luminance) before use.
            neg_input (`str` or `List[str]`, *optional*, defaults to ""):
                Input used to condition the image generation. Effectively
                subtracts this value from `prompt`.
            strength (`float`, *optional*, defaults to 1.0):
                Conceptually, indicates how much to inpaint the masked area.
                Must be between 0 and 1. When `strength` is 1, the denoising
                process will be run on the masked area for the full number of
                iterations specified in `num_inference_steps`. `init_image`
                will be used as a reference for the masked area, adding more
                noise to that region the larger the `strength`. If `strength`
                is 0, no inpainting will occur.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The reference number of denoising steps. More denoising steps
                usually lead to a higher quality image at the expense of slower
                inference. This parameter will be modulated by `strength`, as
                explained above.
            guidance_scale (`float`, *optional*, defaults to 6):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of
                [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf).
                Guidance scale is enabled by setting `guidance_scale > 1`.
                Higher guidance scale encourages to generate images that are
                closely linked to the text `prompt`, usually at the expense of
                lower image quality.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper:
                https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/):
                `PIL.Image.Image` or `nd.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a
                [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`]
                instead of a plain tuple.

        Returns:
            `StableDiffusionPipelineOutput` or `tuple`:
                `StableDiffusionPipelineOutput` if `return_dict` is True,
                otherwise a `tuple.
            When returning a tuple, the first element is a list with the
                generated images, and the second element is a list of `False`s.
        """
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                "`prompt` has to be of type `str` or `list` but is "
                f"{type(prompt)}"
            )

        if strength < 0 or strength > 1:
            raise ValueError(
                f"The value of strength should in [0.0, 1.0] but is {strength}"
            )

        # Set timesteps.
        accepts_offset = "offset" in set(
            inspect.signature(self.scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        offset = 0
        if accepts_offset:
            offset = 1
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        if not isinstance(init_image, torch.FloatTensor):
            init_image = preprocess_image(init_image).to(self.device)

        # Encode the init image into latents and scale the latents.
        init_latent_dist = self.vae.encode(
            init_image.to(self.device)
        ).latent_dist
        init_latents = init_latent_dist.sample(generator=generator)
        init_latents = 0.18215 * init_latents

        # Expand init_latents for batch_size.
        init_latents = torch.cat([init_latents] * batch_size)
        init_latents_orig = init_latents

        if not isinstance(mask_image, torch.FloatTensor):
            mask = preprocess_mask(mask_image)
        mask = mask.to(torch.long).to(self.device)
        mask = torch.cat([mask] * batch_size)

        # Check sizes.
        if not mask.shape == init_latents.shape:
            raise ValueError(
                "The mask and init_image should be the same size!"
            )

        # Get the original timestep using init_timestep.
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            timesteps = torch.tensor(
                [num_inference_steps - init_timestep] * batch_size,
                dtype=torch.long,
                device=self.device,
            )
        else:
            timesteps = self.scheduler.timesteps[-init_timestep]
            timesteps = torch.tensor(
                [timesteps] * batch_size, dtype=torch.long, device=self.device
            )

        # Add noise to latents using the timesteps.
        noise = torch.randn(
            init_latents.shape, generator=generator, device=self.device
        )
        init_latents = self.scheduler.add_noise(
            init_latents, noise, timesteps
        ).to(self.device)

        # Get prompt text embeddings.
        text_embeddings = self.get_embeddings_with_modifiers(prompt)
        uncond_embeddings = self.get_embeddings_with_modifiers(neg_input)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a
        # single batch to avoid doing two forward passes.
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Prepare extra kwargs for the scheduler step, since not all schedulers
        # have the same signature.
        # eta (η) is only used with the DDIMScheduler, it will be ignored for
        # other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latents = init_latents

        t_start = max(num_inference_steps - init_timestep + offset, 0)
        for i, timestep in enumerate(
            self.progress_bar(self.scheduler.timesteps[t_start:])
        ):
            t_index = t_start + i

            # Expand the latents if we are doing classifier free guidance.
            latent_model_input = torch.cat([latents] * 2)

            # If we use LMSDiscreteScheduler, let's make sure latents are
            # mulitplied by sigmas.
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[t_index]
                # The model input needs to be scaled to match the continuous
                # ODE formulation in K-LMS.
                latent_model_input = latent_model_input / (
                    (sigma**2 + 1) ** 0.5
                )
                latent_model_input = latent_model_input.to(self.unet.dtype)
                timestep = timestep.to(self.unet.dtype)

            # Predict the noise residual.
            noise_pred = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeddings,
            ).sample

            # Perform guidance.
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # Compute the previous noisy sample x_t -> x_t-1.
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(
                    noise_pred, t_index, latents, **extra_step_kwargs
                ).prev_sample
                init_latents_proper = self.scheduler.add_noise(
                    init_latents_orig, noise, i + 1
                )
            else:
                latents = self.scheduler.step(
                    noise_pred, timestep, latents, **extra_step_kwargs
                ).prev_sample
                init_latents_proper = self.scheduler.add_noise(
                    init_latents_orig, noise, timestep.to(torch.long)
                )
            # Masking.
            latents = (init_latents_proper * mask) + (latents * (1 - mask))

        # Scale and decode the image latents with vae.
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents.to(self.vae.dtype)).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, False)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=False
        )
