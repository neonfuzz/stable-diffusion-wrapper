"""
Prompt-crafting tools.

Classes:
    StablePrompt - contain prompt for SD
"""


class StablePrompt:
    """Container for holding Stable Diffusion Prompts.

    Instance Attributes:
        subject (str): composition subject, default="a fantasy landscape"
        medium (str): describe type of image,
            default="a detailed matte painting"
        details (list of str): additional details to render in the image,
            default=["blue sky", "grass", "river"]
        artists (list of str): artist names to guide style,
            default=["Tyler Edlin", "Michael Whelan"]
        trend_type (str with "{}"): type of trend, default "trending on {}"
        trending (str): where it's trending, default "artstation"
        movement (str): overall style, default "fantasy art"
        flavors (list of str): style guides,
            default=[
                "matte painting",
                "matte drawing",
                "reimagined by industrial light and magic",
                ]

    Read-Only Attributes:
        dict (dict): represent prompt as a dictionary
        details_str (str): represent `artists` as a string
        medium_str (str): represent `medium` as a string
        artist_str (str): represent `artists` as a string
        trending_str (str): represent `trend_type` and `trending` as a string
        movement_str (str): represent `movement` as a string
        flavor_str (str): represent `flavor` as a string

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
        subject="a fantasy landscape",
        medium="a detailed matte painting",
        **kwargs,
    ):
        self.subject = subject
        self.details = kwargs.pop("details", ["blue sky", "grass", "river"])
        self.medium = medium
        self.artists = kwargs.pop("artists", ["Tyler Edlin", "Michael Whelan"])
        self.trend_type = kwargs.pop("trend_type", "trending on {}")
        self.trending = kwargs.pop("trending", "artstation")
        self.movement = kwargs.pop("movement", "fantasy art")
        self.flavors = kwargs.pop(
            "flavors",
            [
                "matte painting",
                "matte drawing",
                "reimagined by industrial light and magic",
            ],
        )

    def __repr__(self):
        return (
            f"{self.subject}{self.details_str}{self.medium_str}"
            f"{self.artist_str}{self.trending_str}{self.movement_str}"
            f"{self.flavors_str}"
        )

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def painting(self):
        """Set attributes for a painting."""
        self.subject = "a fantasy landscape"
        self.details = ["blue sky", "grass", "river"]
        self.medium = "a detailed matte painting"
        self.artists = ["Tyler Edlin", "Michael Whelan"]
        self.trending = "artstation"
        self.movement = "fantasy art"
        self.flavors = ["matte painting", "matte drawing"]

    def photo(self):
        """Set attributes for photography."""
        self.subject = "a dramatic landscape"
        self.details = ["blue sky", "grass", "river"]
        self.medium = "a photograph"
        self.artists = ["Ansel Adams"]
        self.trending = "flickr"
        self.movement = "photorealism"
        self.flavors = [
            "national geographic photo",
            "sense of awe",
            "ambient occlusion",
        ]

    def render(self):
        """Set attributes for 3d rendering."""
        self.subject = "a robot"
        self.details = ["toonami", "shiny"]
        self.medium = "a computer rendering"
        self.artists = ["Pixar"]
        self.trending = "cg society"
        self.movement = "plasticien"
        self.flavors = ["quantum wavetracing", "vray tracing", "unreal engine"]

    def rainbow(self):
        """Set attributes for psychadelic colors."""
        self.painting()
        self.artists = ["Lisa Frank", "Thomas Kinkade", "Georgia O'Keefe"]
        self.trending = "behance"
        self.movement = "psychadelic art"
        self.flavors = ["kaliedoscope", "vaporwave", "maximalist"]

    def manga(self):
        """Set attributes for manga/anime."""
        self.painting()
        self.artists = ["Studio Ghibli", "Hayao Mikazaki"]
        self.movement = ""
        self.flavors = ["anime aesthetic"]

    def scifi(self):
        """Set attributes for science fiction scenes."""
        self.subject = "a cityscape"
        self.details = ["rain", "reflections"]
        self.medium = "a movie still"
        self.artists = ["Ridley Scott", "Simon Stalenhag"]
        self.trending = "cg society"
        self.movement = "retrofuturism"
        self.flavors = ["dystopian art, sci-fi", "futuristic"]

    def portrait(self):
        """Set attributes for a painterly portrait."""
        self.subject = "a person"
        self.details = [
            "fantastic eyes",
            "highly-detailed and symmetric face",
            "professional lighting",
            "studio lighting",
            "well-lit",
        ]
        self.medium = "oil painting on canvas"
        self.artists = ["Jesper Ejsing", "Annie Leibovitz"]
        self.trending = "artstation"
        self.movement = "renaissance"
        self.flavors = ["studio portrait", "dutch golden age", "elegant"]

    def wildlife(self):
        """Set attributes for wildlife photography."""
        self.photo()
        self.subject = "a jaguar"
        self.details = [
            "telephoto lens",
            "sigma 500mm",
            "f/5",
            "shot from afar",
        ]
        self.medium = "wildlife photography"
        self.artists = ["Marsel Van Oosten"]
        self.trending = "shutterstock"

    @property
    def dict(self):
        """Access prompt as a dictionary."""
        return self.__dict__

    @property
    def details_str(self):
        """Convert list of details into a prompt string."""
        details = [""] + self.details
        return ", ".join(details)

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
    def trending_str(self):
        """Convert trending into a prompt string."""
        if self.trending:
            return ", " + self.trend_type.format(self.trending)
        return ""

    @property
    def movement_str(self):
        """Convert movement into a prompt string."""
        if self.movement:
            return ", " + self.movement
        return ""

    @property
    def flavors_str(self):
        """Convert list of flavors into a prompt string."""
        flavors = [""] + self.flavors
        return ", ".join(flavors)
