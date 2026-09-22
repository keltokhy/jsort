"""jsort's view of the JevKit runtime: which providers it offers, and its names for the shared pieces."""

from __future__ import annotations

from jevkit_core import (
    DEFAULT_PRICE_PER_MTOK,
    AnswerStore as Cache,
    Backend,
    Client as Jev,
    JevBudgetExceeded,
    JevError,
    JevFatal,
    Meter,
    Settings,
    catalog,
    resolve,
)

PROVIDERS = catalog("typesafe", "openrouter", "gateway")


def resolve_backend(name: str | None = None, *, model: str | None = None, require_key: bool = True) -> Backend:
    return resolve(PROVIDERS, name, model=model, require_key=require_key)


__all__ = [
    "DEFAULT_PRICE_PER_MTOK",
    "PROVIDERS",
    "Backend",
    "Cache",
    "Jev",
    "JevBudgetExceeded",
    "JevError",
    "JevFatal",
    "Meter",
    "Settings",
    "resolve_backend",
]
