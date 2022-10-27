"""Tools to reverse engineer a prompt from Internet images.

Variables:
    DATA_DIR (str): where the lists of modifiers are saved
    BATCH_SIZE (int): maximum number of modifiers to embed at a time
    DEVICE (torch.device): cuda if available, else cpu
    CLIP_MODEL_NAME (str): huggingface clip model name
    CLIP_PROCESSOR (CLIPProcessor): clip preprocess pipeline
    CLIP_MODEL(CLIPModel): clip embedding model

Classes:
    ModifierEmbedding: hold modifiers and their embeddings

functions:
    embed_text: convert text to CLIP embeddings
    embed_images: convert image(s) to CLIP embeddings
    download_image: get an image from the Internet
    ask_best: ask the user to select the best modifiers
    reverse_engineer_prompt: get best modifiers across all types
"""

# pylint: disable=no-member, import-error

from copy import copy
import gc
import hashlib
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import questionary
import requests
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

from image import show_thumbs
from prompt import StablePrompt

from utils import LazyLoad
import data


DATA_DIR = data.__path__[0]
BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
CLIP_PROCESSOR = LazyLoad(AutoProcessor.from_pretrained, CLIP_MODEL_NAME)
CLIP_MODEL = LazyLoad(AutoModel.from_pretrained, CLIP_MODEL_NAME)


class ModifierEmbedding:
    """Hold modifier and embedding information.

    Read-only instance attributes:
        desc (str): description of modifier type
        embeds (torch.tensor): CLIP embeddings
        hash (str): unique hash used to cache results between sessions
        modifiers (list of str): modifier texts

    Methods:
        get_most_similar: query against a(n/ set of) image(s)
    """

    def __init__(self, modifiers: List[str], desc: str):
        """Initialize.

        Args:
            modifiers (List[str]): modifiers texts
            desc (str): description of modifier type
        """
        self._desc = desc
        self._modifiers = modifiers
        self._embeds = None

    def __repr__(self):
        return f"ModifierEmbedding: {self.desc.title()}"

    def __len__(self):
        return len(self._modifiers)

    def __getitem__(self, mod):
        idx = self.modifiers.index(mod)
        return self.embeds[idx]

    def _load(self):
        fname = f"./.cache/{self.desc}_{self.hash}.pkl"
        if Path(fname).exists():
            self._embeds = torch.load(fname)
            return True
        return False

    def _save(self):
        fname = f"./.cache/{self.desc}_{self.hash}.pkl"
        Path("./.cache").mkdir(parents=True, exist_ok=True)
        torch.save(self.embeds, fname)

    def get_most_similar(
        self,
        image_embeds: torch.tensor,
        topk: int = 3,
    ) -> List[Tuple[str, float]]:
        """Query most similar modifiers, given image embeddings.

        Args:
            image_embeds (torch.tensor): image to query, embedded
            topk (int): number of results to return, default=3

        Returns:
            pd.DataFrame: with columns `desc` and `desc`_score
        """
        topk = min(topk, self.__len__())
        similarity = image_embeds @ self.embeds.T
        top_val, top_idx = similarity.norm(dim=0).topk(topk)
        result = pd.DataFrame(
            {
                self.desc: np.array(self.modifiers)[top_idx],
                f"{self.desc}_score": top_val.numpy() * 100,
            }
        )
        return (
            result.sort_values(f"{self.desc}_score", ascending=False)
            .drop_duplicates(self.desc)
            .head(topk)
        )

    @property
    def embeds(self):
        """Lazy load/calculate embeddings."""
        if self._embeds is None and not self._load():
            self._embeds = embed_text(
                self.modifiers, message=f"Preprocessing {self.desc}"
            )
            self._save()
        return self._embeds

    desc = property(fget=lambda self: self._desc)
    hash = property(
        fget=lambda self: hashlib.md5(
            ", ".join(self.modifiers).encode()
        ).hexdigest()[:15]
    )
    modifiers = property(fget=lambda self: self._modifiers)


@torch.no_grad()
def embed_text(
    text: Union[str, Iterable],
    batch_size: int = BATCH_SIZE,
    message: str = None,
) -> torch.tensor:
    """Embed text into CLIP space.

    Args:
        text (Union[str, Iterable]): text to embed
        batch_size (int): maximum size per iteration, default=`BATCH_SIZE`
        message (str): description to show in progress bar, default=None

    Returns:
        torch.tensor: embedded text
    """
    if isinstance(text, str):
        text = [text]

    CLIP_PROCESSOR.load()
    CLIP_MODEL.load()
    CLIP_MODEL.to(DEVICE)
    batches = np.array_split(text, ceil(len(text) / batch_size))
    result = []
    for batch in tqdm(batches, desc=message):
        # pylint: disable=not-callable
        tokens = CLIP_PROCESSOR(
            text=list(batch),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)
        text_features = CLIP_MODEL.get_text_features(**tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.float().cpu()
        result.append(text_features)
    CLIP_MODEL.cpu()

    torch.cuda.empty_cache()
    gc.collect()
    return torch.concat(result)


@torch.no_grad()
def embed_images(
    images: Union[Image.Image, List[Image.Image]]
) -> torch.tensor:
    """Embed images into CLIP space.

    Args:
        images (Union[Image.Image, List[Image.Image]]): images to embed

    Returns:
        torch.tensor: embedded images
    """
    CLIP_PROCESSOR.load()
    CLIP_MODEL.load()
    CLIP_MODEL.to(DEVICE)
    # pylint: disable=not-callable
    images = CLIP_PROCESSOR(images=images, return_tensors="pt").to(DEVICE)
    image_features = CLIP_MODEL.get_image_features(**images)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.float().cpu()
    CLIP_MODEL.cpu()

    torch.cuda.empty_cache()
    gc.collect()
    return image_features


def _load_list(fpath: str) -> List[str]:
    """Load a text list from file."""
    with open(fpath, "r", encoding="utf-8", errors="replace") as infile:
        return [line.strip() for line in infile.readlines()]


def download_image(url: str) -> Image.Image:
    """Download image from url."""
    return Image.open(requests.get(url, stream=True).raw).convert("RGB")


def _init_mods(data_dir: str = DATA_DIR) -> Dict[str, ModifierEmbedding]:
    """Load a whole bunch of modifiers.

    Args:
        data_dir (str): directory where modifiers are saved

    Returns:
        Dict[ModifierEmbedding]: modifier embedding objects
    """
    sites = [
        "Artstation",
        "behance",
        "cg society",
        "cgsociety",
        "deviantart",
        "dribble",
        "flickr",
        "instagram",
        "pexels",
        "pinterest",
        "pixabay",
        "pixiv",
        "polycount",
        "reddit",
        "shutterstock",
        "tumblr",
        "unsplash",
        "zbrush central",
    ]
    trends = copy(sites)
    trends.extend([f"trending on {site}" for site in sites])
    trends.extend([f"featured on {site}" for site in sites])
    trends.extend([f"{site} contest winner" for site in sites])

    artists = ModifierEmbedding(
        _load_list(f"{data_dir}/artists.txt"), "artists"
    )
    flavors = ModifierEmbedding(
        _load_list(f"{data_dir}/flavors.txt"), "flavors"
    )
    mediums = ModifierEmbedding(
        _load_list(f"{data_dir}/mediums.txt"), "mediums"
    )
    movements = ModifierEmbedding(
        _load_list(f"{data_dir}/movements.txt"), "movements"
    )
    trends = ModifierEmbedding(trends, "trends")

    return {
        "mediums": mediums,
        "artists": artists,
        "trends": trends,
        "movements": movements,
        "flavors": flavors,
    }


def ask_best(
    modifier: ModifierEmbedding,
    image_embeds: torch.tensor,
    topk: int = 5,
    finalk: int = 1,
) -> str:
    """Ask the user which is the best (set of) modifier(s) to choose.

    Args:
        modifier (ModifierEmbedding): modifier embedding object
        image_embeds (torch.tensor): images in CLIP embedding space
        topk (int): number of results to preset to the user
        finalk (int): number of results to highlight;
            if `finalk`==1, only one result will be allowed to be selected

    Returns:
        str or list of str: selected options
    """
    best = modifier.get_most_similar(image_embeds, topk)
    desc = modifier.desc if finalk > 1 else modifier.desc.strip("s")
    choices = [
        questionary.Choice(title=f"{mod} ({score:0.1f}%)", value=mod)
        for mod, score in best.values
    ]
    if finalk == 1:
        choices.append(questionary.Choice(title="<none>", value=""))
        return questionary.select(
            f"What {desc} should we choose?", choices=choices
        ).ask()

    for i, choice in enumerate(choices):
        choice.checked = True
        if i >= finalk - 1:
            break
    return questionary.checkbox(
        f"What {desc} should be choose?", choices=choices
    ).ask()


def reverse_engineer_prompt(urls: Union[str, Iterable[str]]) -> StablePrompt:
    """Given a collection of URLs, reverse engineer the style in CLIP.

    Interacts with user to select best output.

    Args:
        urls (Union[str, Iterable[str]]): URL(s) to query

    Returns:
        StablePrompt: selected prompt with style modifiers filled in
    """
    if isinstance(urls, str):
        urls = [urls]
    # print("Fetching urls...")
    images = [download_image(u) for u in urls]
    show_thumbs(images)
    embeds = embed_images(images)
    modifiers = _init_mods()
    return StablePrompt(
        subject="",
        details=[],
        medium=ask_best(modifiers["mediums"], embeds),
        artists=ask_best(modifiers["artists"], embeds, 5, 3),
        trend_type="{}",
        trending=ask_best(modifiers["trends"], embeds),
        movement=ask_best(modifiers["movements"], embeds),
        flavors=ask_best(modifiers["flavors"], embeds, 25, 5),
    )
