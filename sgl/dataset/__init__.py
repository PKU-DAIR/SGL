from .acm import Acm
from .actor import Actor
from .airports import Airports
from .amazon import Amazon
from .amazon_product import AmazonProduct
from .aminer import Aminer
from .coauthor import Coauthor
from .dblp import Dblp
from .facebook import Facebook
from .flickr import Flickr
from .github import Github
from .imdb import Imdb
from .karateclub import KarateClub
from .linkx_dataset import LINKXDataset
from .nell import Nell
from .ogbn import Ogbn
from .ogbn_mag import OgbnMag
from .planetoid import Planetoid
from .reddit import Reddit
from .twitch import Twitch
from .webkb import WebKB
from .wikics import Wikics

from .custom_dataset import Custom_Hetero, Custom_Homo

__all__ = [
    "Acm",
    "Actor",
    "Airports",
    "AmazonProduct",
    "Amazon",
    "Aminer",
    "Coauthor",
    "Dblp",
    "Facebook",
    "Flickr",
    "Github",
    "Imdb",
    "KarateClub",
    "LINKXDataset",
    "Nell",
    "OgbnMag",
    "Ogbn",
    "Planetoid",
    "Reddit",
    "Twitch",
    "WebKB",
    "Wikics",
    "Custom_Hetero", 
    "Custom_Homo",
]
