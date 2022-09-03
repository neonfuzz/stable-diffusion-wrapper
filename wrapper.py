# pylint: disable=no-member

from copy import copy
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


class StableSettings:
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
        return self.__dict__


class StablePrompt:
    def __init__(
        self,
        subject="a fantasy landscape",
        context="a beautiful painting",
        artists=None,
        details=None,
        modifiers=None,
        **kwargs,
    ):
        self.subject = subject
        self.context = context

        if isinstance(artists, str):
            self.artists = [artists]
        else:
            self.artists = artists or ["Tyler Edlin", "Michael Whelan"]

        if isinstance(details, str):
            self.details = [details]
        else:
            self.details = details or ["blue sky", "grass", "river"]

        if isinstance(modifiers, str):
            self.modifiers = [modifiers]
        else:
            self.modifiers = modifiers or [
                "oil on canvas",
                "intricate",
                "4k resolution",
                "trending on artstation",
            ]

    def __repr__(self):
        return (
            f"{self.context} of {self.subject}{self.artist_str}"
            f"{self.details_str}{self.modifiers_str}"
        )

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    @property
    def dict(self):
        return self.__dict__

    @property
    def artist_str(self):
        if self.artists:
            artists = " and ".join(self.artists)
            return f" by {artists}"
        return ""

    @property
    def details_str(self):
        details = [""] + self.details
        return ", ".join(details)

    @property
    def modifiers_str(self):
        modifiers = [""] + self.modifiers
        return ", ".join(modifiers)


class StableImage:
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

    def show(self):
        self._image.show()

    # TODO
    def save(self):
        raise NotImplementedError

    prompt = property(fget=lambda self: self._prompt)
    settings = property(fget=lambda self: self._settings)
    image = property(fget=lambda self: self._image)
    init = property(fget=lambda self: self._init)
    hash = property(
        fget=lambda self: f"{hash(self.prompt):x}{hash(self.settings):x}"
    )


class StableWorkshop:
    def __init__(self, **kwargs):
        self._init_model()
        self.prompt = StablePrompt(**kwargs)
        self.settings = StableSettings(**kwargs)
        self.generated = []
        self._brainstorm = []
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

    def _init_tensor(self):
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
        init_tensor = (init_tensor / 2 + 0.5).clamp(0, 1)
        return init_tensor

    def _render(self, init_image: Image = None, num: int = 1):
        set_seed(self.settings.seed)
        prompt = [str(self.prompt)] * num
        if init_image is None:
            init_tensor = self._init_tensor()
        else:
            init_tensor = transforms.ToTensor()(
                init_image.resize((self.settings.width, self.settings.height))
            ).unsqueeze_(0)
        with autocast("cuda"):
            result = self._pipe(
                prompt,
                init_image=init_tensor,
                strength=self.settings.strength,
                guidance_scale=self.settings.cfg,
                num_inference_steps=self.settings.iters,
            )
        return result

    def _update_settings(self, **kwargs):
        for key, value in kwargs.items():
            self.settings[key] = value

    def show_brainstorm(self):
        show_image_grid([bs.image for bs in self._brainstorm])

    def show_generated(self):
        show_image_grid([gn.image for gn in self.generated])

    def brainstorm(self, num=6, show=True, **kwargs):
        kwargs["height"] = kwargs.pop("height", 256)
        kwargs["width"] = kwargs.pop("width", 256)
        self._update_settings(**kwargs)

        images = self._render(num=num)
        self._brainstorm = [
            StableImage(prompt=self.prompt, settings=self.settings, image=i)
            for i in images
        ]
        if show is True:
            self.show_brainstorm()

    def hallucinate(self, show=True, **kwargs):
        self._update_settings(**kwargs)
        image = self._render()[0]
        self.generated.append(
            StableImage(
                prompt=str(self.prompt), settings=self.settings, image=image
            )
        )
        if show is True:
            image.show()

    def tune(self, idx: int, show=True, **kwargs):
        if not self._brainstorm:
            raise RuntimeError("Cannot tune until we've `brainstorm`ed.")
        self._update_settings(**kwargs)
        image = self._render(init_image=self._brainstorm[idx].image)[0]
        self.generated.append(
            StableImage(
                prompt=str(self.prompt),
                settings=self.settings,
                image=image,
                init=self._brainstorm[idx],
            )
        )
        if show is True:
            image.show()

    # TODO
    def save(self):
        raise NotImplementedError
