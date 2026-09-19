"""jsort: sort by meaning, on TypeSafe's Jev decision model."""

__version__ = "0.1.2"

from .engine import Ranking, arank, rank

__all__ = ["Ranking", "arank", "rank"]
