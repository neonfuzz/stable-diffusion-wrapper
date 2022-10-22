"""
Handle image storage for interactive use of Stable Diffusion (SD).

Classes:
    StableImage - contain SD image and its generation information
    StableGallery - contain multiple StableImages

Functions:
    show_image_grid = display images in a grid
"""


from copy import deepcopy as copy
from math import ceil, sqrt
from pathlib import Path
from typing import Iterable
import yaml

from PIL import Image, ImageDraw, ImageFont

from mask import StableMasker
from prompt import StablePrompt
from settings import StableSettings


_FONT = ImageFont.truetype("Sarabun-Medium.ttf", size=40)


def _add_label(img, label):
    img = copy(img)
    draw = ImageDraw.Draw(img)
    draw.text((9, 9), str(label), fill=(0, 0, 0), font=_FONT)
    draw.text((10, 10), str(label), fill=(255, 255, 255), font=_FONT)
    return img


def show_image_grid(
    imgs: Iterable, rows: int = None, cols: int = None, labels: Iterable = None
):
    """Display multiple images at once, in a grid format."""
    if isinstance(imgs[0], StableImage):
        imgs = [i.image for i in imgs]
    if labels:
        imgs = [_add_label(im, lab) for im, lab in zip(imgs, labels)]
    if cols:
        rows = rows or int(ceil(len(imgs) / cols))
    else:
        rows = rows or int(sqrt(len(imgs)))
        cols = int(ceil(len(imgs) / rows))
    width = max([i.width for i in imgs])
    height = max([i.height for i in imgs])
    grid = Image.new("RGB", size=(cols * width, rows * height))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * width, i // cols * height))

    grid.show()


class StableGallery(list):
    """Contain a gallery of `StableImage`s.

    Subclassed from `list`.

    Additional behavior:
        Can be indexed with a tuple or a list

    Additional methods:
        show - display the images to screen
        save - save all images
    """

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return StableGallery(super().__getitem__(idx))
        if isinstance(idx, (tuple, list)):
            return StableGallery([self.__getitem__(i) for i in idx])
        return super().__getitem__(idx)

    def __add__(self, other_obj):
        return StableGallery(super().__add__(other_obj))

    def show(self, **kwargs):
        """Show the gallery.

        Args:
            labels (list): manually add labels; overrides `label` if passed

        Additional kwargs are passed to `show_image_grid`.
        """
        labels = range(len(self))
        kwargs["labels"] = kwargs.pop("labels", labels)
        show_image_grid(self, **kwargs)

    def save(self):
        """Save all images in the gallery. See `StableImage.save`."""
        for image in self:
            image.save()


class StableImage:
    """Contain a generated image and the information used to generate it.

    Read-Only Instance Attributes:
        prompt (StablePrompt): the prompt used to generate the image
        settings (StableSettings): the settings used to generate the image
        image (Image.Image): the raw image data
        mask (Image.Image): the infill mask
        init (StableImage): the seed image, if it exists
        hash (str): a unique hash for identifying the image

    Methods:
        show: display the image
        edit_mask: edit the infill mask
        reset_mask: set infill mask to default
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
        self._mask = None
        self._init = copy(init)

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
        If `mask` has been edited, `mask` saves as well.
        """
        Path("generated").mkdir(parents=True, exist_ok=True)
        Path("generated/logs.yaml").touch()
        with open("generated/logs.yaml", "r", encoding="utf-8") as infile:
            logs = yaml.safe_load(infile)
        if logs is None:
            logs = {}
        if self.hash in logs:
            return

        self.image.save(f"generated/{self.hash}.png")
        logs[self.hash] = {
            "prompt": str(self.prompt),
            "neg": self.prompt.neg
            if isinstance(self.prompt, StablePrompt)
            else "",
            "settings": self.settings.dict,
            "mask": str(self._mask),
            "init": str(self.init),
        }
        with open("generated/logs.yaml", "w", encoding="utf-8") as outfile:
            yaml.safe_dump(
                logs,
                outfile,
                explicit_start=True,
                explicit_end=True,
                width=66,
            )

        if self._mask:
            self._mask.save()

        if self.init:
            self.init.save()

    def edit_mask(self):
        """Interactively edit the infill mask.

        Displays `image` to screen in an editable window. Left click adds mask,
        right click erases mask. Scroll up/down changes brush size. "Escape" or
        "Return" to finalize mask. "R" to erase mask and start over.
        """
        masker = StableMasker()
        mask = masker(self.image)
        self._mask = StableImage(prompt="mask", settings={}, image=mask)

    def reset_mask(self):
        """Reset the mask to default."""
        self._mask = None

    @property
    def mask(self):
        """Return mask."""
        if self._mask is not None:
            return self._mask.image
        return Image.new("L", (self.settings.width, self.settings.height), 255)

    prompt = property(fget=lambda self: self._prompt)
    settings = property(fget=lambda self: self._settings)
    image = property(fget=lambda self: self._image)
    init = property(fget=lambda self: self._init)
    hash = property(
        fget=lambda self: f"{hash(self.image.tobytes()):x}".strip("-")
    )
