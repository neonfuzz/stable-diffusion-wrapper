"""
Prompt-crafting tools.

Classes:
    StablePrompt - contain prompt for SD
"""

import textwrap

# pylint: disable=import-error
import openai
import questionary


def _select_result(choices):
    choices = ["\n   ".join(textwrap.wrap(c)) + "\n" for c in choices]
    result = questionary.select(
        "Which subject should we choose?", choices=choices
    ).ask()
    return result.replace("\n", " ").replace("    ", " ").strip()


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
        hallucinate: use GPT-3 to improve your prompt
        painting: set defaults to emulate painting
        photo: set defaults to emulate photography
        manga: set defaults to emulate anime
        portrait: set defaults to emulate a painted portrait
        rainbow: set defaults to be very colorful
        render: set defaults to emulate 3d graphics
        scifi: set defaults to emulate science fiction movies
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

    def hallucinate(
        self, genre: str = "epic fantasy", subject: str = None, **kwargs
    ):
        """Improve your prompt with GPT-3.

        Requires an OpenAI API key. Set environment variable "OPENAI_API_KEY".
        https://beta.openai.com/account/api-keys

        Will replace `subject` with the new prompt.

        Args:
            genre (str): what kind of "movie" should GPT-3 describe?
                default: 'epic fantasy'
            subject (str): topic to send to GPT-3, default `subject`
            model (str): OpenAI model name, default: 'text-davinci-002'
            max_tokens (int): maximum tokens to generate, default: 50
            top_p (float): sampling probability; 0=deterministic 1=random,
                default: 0.5
            n (int): number of descriptions to generate, default: 5

        Additional kwargs are passed to OpenAI's `Completion`.
        """
        subject = subject or self.subject
        prompt = (
            "You are an extremely creative, award-winning writer for a new "
            f"{genre} movie. Your task is to come up with interesting, "
            "surprising scene ideas for a macro shot of a described scene, "
            "based on curious exploring and logic around what sort of macro "
            "filmable items may be found nearby and at that described scene."
            f"The described scene is: {subject}. What interesting "
            "macro scene can be found here that would drive the plot forward? "
            "Describe only one and do not preface "
            "your description with the previous location. Get right into it!"
        )
        result = openai.Completion.create(
            model=kwargs.pop("model", "text-davinci-002"),
            prompt=prompt,
            max_tokens=kwargs.pop("max_tokens", 50),
            top_p=kwargs.pop("top_p", 0.75),
            n=kwargs.pop("n", 5),
            **kwargs,
        )
        choices = [c["text"].strip() for c in result["choices"]]
        choices.append(subject)
        self.subject = _select_result(choices)

    def zero(self):
        self.subject = ""
        self.details = []
        self.medium = ""
        self.artists = []
        self.trending = ""
        self.movement = ""
        self.flavors = []

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

    def manga(self):
        """Set attributes for manga/anime."""
        self.painting()
        self.artists = ["Studio Ghibli", "Hayao Mikazaki"]
        self.movement = ""
        self.flavors = ["anime aesthetic"]

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
        self.medium = "a portrait"
        self.artists = ["William Forsyth", "Richard Schmid", "Michael Garmash"]
        self.trending = "artstation"
        self.movement = "figurative art"
        self.flavors = [
            "detailed painting",
            "studio portrait",
            "oil on canvas",
        ]

    def rainbow(self):
        """Set attributes for psychadelic colors."""
        self.painting()
        self.artists = ["Lisa Frank", "Thomas Kinkade", "Georgia O'Keefe"]
        self.trending = "behance"
        self.movement = "psychadelic art"
        self.flavors = ["kaliedoscope", "vaporwave", "maximalist"]

    def render(self):
        """Set attributes for 3d rendering."""
        self.subject = "a robot"
        self.details = ["toonami", "shiny"]
        self.medium = "a computer rendering"
        self.artists = ["Pixar"]
        self.trending = "cg society"
        self.movement = "plasticien"
        self.flavors = ["quantum wavetracing", "vray tracing", "unreal engine"]

    def scifi(self):
        """Set attributes for science fiction scenes."""
        self.subject = "a cityscape"
        self.details = ["rain", "reflections"]
        self.medium = "a movie still"
        self.artists = ["Ridley Scott", "Simon Stalenhag"]
        self.trending = "cg society"
        self.movement = "retrofuturism"
        self.flavors = ["dystopian art, sci-fi", "futuristic"]

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
        self.flavors = [
            "national geographic photo",
            "majestic",
            "uhd image",
        ]

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
