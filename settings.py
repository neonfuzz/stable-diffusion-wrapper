"""Handle settings for interactive use of Stable Diffusion (SD).

Classes:
    StableSettings - contain settings for SD

Variables:
    SEEDS (list of int) - favorite starting seeds for generation
"""


from copy import copy


SEEDS = [1337, 271, 314159, 41245, 59017, 61023]


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

    Methods:
        copy: return a copy of the StableSettings object
        update: update attributes in place

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
        """Initialize.

        Args:
            height (int): image height in pixels, default=512
            width (int): image width in pixels, default=512
            seed (int): random seed for generating images, default=1337
            iters (int): number of diffusion steps for generation, default=50
            cfg (float): classifier free guidance, default=6.0
            strength (float): maintain original image, default=1.0
        """
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

    def copy(self, **kwargs):
        """Return a copy of the StableSettings object.

        kwargs are updated in the copy.
        """
        ret = copy(self)
        ret.update(**kwargs)
        return ret

    def update(self, **kwargs):
        """Update attributes in place via kwargs."""
        for key, value in kwargs.items():
            self[key] = value

    @property
    def dict(self):
        """Access settings as a dictionary."""
        return self.__dict__
