"""
Handle image storage for interactive use of Stable Diffusion (SD).

Classes:
    StableImage - contain SD image and its generation information
"""


from copy import copy
import os
import yaml

from PIL import Image

from prompt import StablePrompt
from settings import StableSettings


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
