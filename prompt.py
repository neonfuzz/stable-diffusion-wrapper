"""
Prompt-crafting tools.

Classes:
    StablePrompt - contain prompt for SD
"""

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

    Methods:
        painting: set defaults to emulate painting
        photo: set defaults to emulate photography
        render: set defaults to emulate 3d graphics
        rainbow: set defaults to be very colorful
        manga: set defaults to emulate anime
        scifi: set defaults to emulate science fiction movies
        portrait: set defaults to emulate a painted portrait
        wildlife: set defaults to emulate wildlife photography
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
            f"{self.subject}{self.medium_str}{self.artist_str}"
            f"{self.details_str}{self.modifiers_str}"
        )

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def painting(self):
        """Set medium and modifiers for painting."""
        self.medium = "oil painting on canvas"
        self.modifiers = [
            "intricate",
            "4k resolution",
            "trending on artstation",
        ]

    def photo(self):
        """Set medium and modifiers for photography."""
        self.medium = "a photograph"
        self.modifiers = [
            "intricate",
            "4k resolution",
            "trending on flickr",
        ]

    def render(self):
        """Set medium, artists, and modifiers for 3d rendering."""
        self.painting()
        self.medium = "a 3d render"
        self.artists = ["Pixar"]
        self.modifiers = [
            "raytracing",
            "octane render",
            "unreal engine",
        ] + self.modifiers

    def rainbow(self):
        """Set medium, artists, and modifiers for psychadelic colors."""
        self.painting()
        self.artists = ["Lisa Frank", "Thomas Kinkade", "Georgia O'Keefe"]
        self.modifiers.insert(0, "kaliedoscope")

    def manga(self):
        """Set medium, artists, and modifiers for manga/anime."""
        self.painting()
        self.medium = "manga"
        self.artists = ["Studio Ghibli"]

    def scifi(self):
        """Set medium, artists, and modifiers for science fiction scenes."""
        self.photo()
        self.medium = "a film still"
        self.artists = ["Ridley Scott", "Simon Stalenhag"]

    def portrait(self):
        """Set all fields for a painterly portrait of a woman."""
        self.painting()
        self.subject = "a woman"
        self.artists = ["Jesper Ejsing", "Annie Leibovitz"]
        self.details = [
            "fantastic eyes",
            "highly-detailed and symmetric face",
            "professional lighting",
            "studio portrait",
            "studio lighting",
            "well-lit",
        ]

    def wildlife(self):
        """Set all fields for wildlife photography of a jaguar."""
        self.photo()
        self.medium = "wildlife photography"
        self.subject = "a jaguar"
        self.artists = ["Marsel Van Oosten"]
        self.details = [
            "telephoto lens",
            "sigma 500mm",
            "f/5",
            "shot from afar",
        ]

    @property
    def dict(self):
        """Access prompt as a dictionary."""
        return self.__dict__

    @property
    def medium_str(self):
        """Convert medium into a prompt string."""
        if self.medium:
            return ", " + self.medium
        return ""

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
