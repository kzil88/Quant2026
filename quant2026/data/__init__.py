"""Data layer: acquisition, cleaning, storage for China A-shares."""

from .base import DataProvider
from .akshare_provider import AkShareProvider
from .cache import CachedProvider

__all__ = ["DataProvider", "AkShareProvider", "CachedProvider"]
