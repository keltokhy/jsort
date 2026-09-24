"""jsort's view of the JevKit runtime: the providers it offers, in the order it prefers them."""

from jevkit_runtime import catalog

PROVIDERS = catalog("typesafe", "openrouter", "gateway", "diffusiongemma", "laya", "gliner")
