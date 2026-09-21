"""A saved scale: the anchors later texts are placed against, and what was asked of whom to get them.

An ordinary run fits a scale that belongs to that run. Saving it keeps a few of its texts as anchors,
each with the score and standard error the fit gave it, together with everything that decides what Jev
is asked: the question, the model and endpoint, and how much of a text is shown. A later run places new
texts by comparing them with the anchors only (placement.py), so their scores are on the saved scale.

The file holds the anchors and not every fitted text. The run's own output already carries every
score, a scale built on thousands of documents stays small enough to keep with a project, and
`anchors` can be raised to the number of texts when all of them should be available.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

SCHEMA_VERSION = 1
# A default -k 10 placement spends six comparisons near its running estimate, where an anchor within a
# logit still carries four-fifths of the most a comparison can. Thirty anchors put about six within a
# logit of any point on a scale ten logits long, the longest in the README, and more on a shorter one.
DEFAULT_ANCHORS = 30


class ScaleError(ValueError):
    """The scale cannot be used: unreadable, malformed, or built on another model's answers."""


@dataclass(frozen=True)
class Anchor:
    text: str            # what Jev was shown, so already cut to the scale's max_chars
    score: float
    se: float
    comparisons: int


@dataclass(frozen=True)
class Scale:
    description: str
    question: dict             # the question exactly as it was asked; placement asks it again verbatim
    model: dict                # api, endpoint and requested, which the cache keys on; answered, a count of answers
                               # by the model the API named; unknown_answers, those that came with no name
    input: dict                # max_chars, and the unit and field the texts were read as
    fit: dict                  # texts, comparisons, rounds, per_item, seed, reliability, lean, gamma, ridge
    anchors: tuple[Anchor, ...]   # highest first
    created: str | None = None
    jsort_version: str | None = None

    @property
    def gamma(self) -> float:
        return float(self.fit["gamma"])

    @property
    def max_chars(self) -> int:
        return int(self.input["max_chars"])

    @property
    def answered_by(self) -> str | None:
        """The one model every answer behind the scale named. None when they named several, or some named none."""
        named = self.model["answered"]
        return next(iter(named)) if len(named) == 1 and not self.model["unknown_answers"] else None

    @property
    def span(self) -> tuple[float, float]:
        """The lowest and highest anchor scores. A text placed outside them is flagged as beyond the scale."""
        return self.anchors[-1].score, self.anchors[0].score

    def to_json(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, "jsort_version": self.jsort_version, "created": self.created,
                "description": self.description, "question": self.question, "model": self.model,
                "input": self.input, "fit": self.fit,
                "anchors": [{"text": a.text, "score": a.score, "se": a.se, "comparisons": a.comparisons}
                            for a in self.anchors]}

    def save(self, path) -> None:
        """Write the file whole or not at all, so a failed write cannot leave half a scale behind."""
        path = os.fspath(path)
        partial = f"{path}.{os.getpid()}.part"
        try:
            with open(partial, "w", encoding="utf-8") as f:
                json.dump(self.to_json(), f, ensure_ascii=False, indent=1)
                f.write("\n")
            os.replace(partial, path)
        except OSError:
            if os.path.exists(partial):
                os.unlink(partial)
            raise

    @classmethod
    def load(cls, path) -> "Scale":
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except OSError as e:
            raise ScaleError(f"{path}: {e.strerror or e}") from None
        except ValueError:
            raise ScaleError(f"{path}: not a scale file: it is not valid JSON") from None
        try:
            return cls.from_json(data)
        except ScaleError as e:
            raise ScaleError(f"{path}: {e}") from None

    @classmethod
    def from_json(cls, data) -> "Scale":
        if not isinstance(data, dict) or "schema_version" not in data:
            raise ScaleError("not a scale file: no schema_version")
        version = data["schema_version"]
        if version != SCHEMA_VERSION:
            raise ScaleError(f"scale schema {version!r} is not one this jsort reads (it reads {SCHEMA_VERSION}); "
                             "upgrade jsort or save the scale again")
        description, question = data.get("description"), data.get("question")
        if not isinstance(description, str) or not description.strip():
            raise ScaleError("the scale has no description")
        if (not isinstance(question, dict) or question.get("type") != "noul"
                or not isinstance(question.get("instructions"), str)):
            raise ScaleError("the scale's question is not a noul with instructions")
        if description not in question["instructions"]:
            raise ScaleError("the scale's question does not contain its description; placement asks the question, "
                             "so a description edited by hand would change nothing")
        model, inputs, fitted = data.get("model"), data.get("input"), data.get("fit")
        if not isinstance(model, dict) or not all(isinstance(model.get(k), str) and model[k]
                                                  for k in ("api", "endpoint", "requested")):
            raise ScaleError("the scale does not say which api, endpoint and model it was built with")
        named = model.get("answered")
        if (not isinstance(named, dict) or not all(isinstance(k, str) and k and _whole(v, 1) for k, v in named.items())
                or not _whole(model.get("unknown_answers"), 0)):
            raise ScaleError("the scale does not count its answers by the model that gave them")
        if not isinstance(inputs, dict) or not _whole(inputs.get("max_chars"), 1):
            raise ScaleError("the scale does not say how many characters of a text Jev was shown")
        if not isinstance(fitted, dict) or not _finite(fitted.get("gamma")):
            raise ScaleError("the scale does not record the first-position lean it was fitted with")
        rows = data.get("anchors")
        if not isinstance(rows, list) or len(rows) < 2:
            raise ScaleError("a scale needs at least two anchors")
        anchors = []
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip()
                    or not _finite(row.get("score")) or not _finite(row.get("se")) or row["se"] < 0
                    or not _whole(row.get("comparisons", 0), 0)):
                raise ScaleError("an anchor needs a text, a finite score and a standard error")
            anchors.append(Anchor(row["text"], float(row["score"]), float(row["se"]), int(row.get("comparisons", 0))))
        if len({a.text for a in anchors}) != len(anchors):
            raise ScaleError("two anchors have the same text")
        anchors.sort(key=lambda a: -a.score)
        return cls(description, question, model, inputs, fitted, tuple(anchors),
                   created=data.get("created"), jsort_version=data.get("jsort_version"))

    def check_model(self, api: str, endpoint: str, requested: str) -> None:
        """Refuse a client the cache would not treat as the one that built the scale, and a scale that vouches for no model.

        Scores from different models are not comparable, and neither are answers the cache keeps apart.
        Asking for the model that answered by its own ID is as good as asking for the alias that reached it.
        """
        built = self.model
        if self.answered_by is None:
            raise ScaleError(f"the scale cannot say which model it was built on: {_unvouched(built)}. Texts placed on it "
                             "could not be checked against that model. Build the scale again from answers that name one "
                             "model (--no-cache, or a pinned --model), or pass --any-model to place anyway")
        differs = [f"{what} {mine!r}, not {theirs!r}" for what, mine, theirs in (
            ("the API is", api, built["api"]), ("the endpoint is", endpoint, built["endpoint"])) if mine != theirs]
        if requested not in (built["requested"], self.answered_by):
            differs.append(f"the model is {requested!r}, not {built['requested']!r}")
        if differs:
            raise ScaleError(f"this run would not ask the model the scale was built with: {'; '.join(differs)}. "
                             "Scores from different models are not comparable. Match the scale with --api and "
                             "--model, or pass --any-model to place anyway")

    def check_answer(self, answered: str | None, source: str) -> bool:
        """Whether an answer can be vouched for. Raises when it names a model other than the scale's: a moved alias
        shows this way, on the first live reply or on a cached answer that recorded who gave it."""
        built = self.answered_by
        if answered and built and answered != built:
            where = "the cache holds an answer from" if source == "cache" else "the answers now come from"
            raise ScaleError(f"the scale was built on answers from {built}, and {where} {answered}. Scores from different "
                             f"models are not comparable. Pin it with --model {built}"
                             + (" and --no-cache" if source == "cache" else "") + ", or pass --any-model to place anyway")
        return bool(answered)


def _unvouched(model: dict) -> str:
    named, unknown = model["answered"], model["unknown_answers"]
    if len(named) > 1:
        return "its answers came from " + " and ".join(f"{m} ({n:,})" for m, n in sorted(named.items()))
    return f"{unknown:,} of the {unknown + sum(named.values()):,} answers behind it does not say which model gave it" if unknown == 1 \
        else f"{unknown:,} of the {unknown + sum(named.values()):,} answers behind it do not say which model gave them"


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _whole(x, minimum: int) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= minimum


def choose(score, se, count: int) -> list[int]:
    """Which texts to keep as anchors, highest first: spread over the score range, the best measured of each stretch.

    The highest and the lowest text are always kept, so the anchors span everything the scale was fitted
    on and "beyond the anchors" means beyond all of it. Between them the range is cut into `count`
    stretches of equal width and the text with the smallest standard error is taken from each. A stretch
    with no text in it leaves a place over; those go round again, to the next best of each stretch.
    """
    usable = [i for i in range(len(score)) if math.isfinite(score[i]) and math.isfinite(se[i])]
    if len(usable) > count:
        low, high = min(usable, key=lambda i: (score[i], i)), max(usable, key=lambda i: (score[i], -i))
        width = (score[high] - score[low]) / count or 1.0
        stretches: list[list[int]] = [[] for _ in range(count)]
        for i in usable:
            stretches[min(count - 1, int((score[i] - score[low]) / width))].append(i)
        for stretch in stretches:
            stretch.sort(key=lambda i: (i not in (low, high), se[i], i))
        usable, depth = [], 0
        while len(usable) < count:
            layer = sorted((s[depth] for s in stretches if len(s) > depth),
                           key=lambda i: (i not in (low, high), se[i], i))
            usable += layer[:count - len(usable)]
            depth += 1
    return sorted(usable, key=lambda i: (-score[i], i))


def identity(jev, responders: dict) -> dict:
    """Who was asked, as the cache records it (endpoint and requested model), and who gave the answers the fit
    used: a count for each model the API named, and a count of answers that came with no name."""
    backend = getattr(jev, "backend", None)
    return {"api": getattr(backend, "name", None), "endpoint": getattr(jev, "url", None),
            "requested": getattr(jev, "model", None),
            "answered": {m: n for m, n in sorted(responders.items(), key=lambda kv: str(kv[0])) if m},
            "unknown_answers": sum(n for m, n in responders.items() if not m)}


def build(ranking, anchors: int = DEFAULT_ANCHORS, *, unit: str | None = None, field: str | None = None,
          any_model: bool = False) -> Scale:
    """The scale a finished ranking defines. `unit` and `field` record how the texts were read.

    A scale names the model it was built on, and later placements are checked against that name, so it is not
    saved on answers that named two models or on answers that named none, unless any_model says to. What the
    file records is the count either way, never one name standing for answers that did not give it.
    """
    from . import __version__
    from .model import RIDGE

    if isinstance(anchors, bool) or not isinstance(anchors, (int, np.integer)) or anchors < 2:
        raise ValueError("anchors must be an integer of at least 2")
    run = ranking.run
    if run is None:
        raise ScaleError("nothing was compared, so there is no scale to save")
    if run.get("top") is not None:
        raise ScaleError("a --top run stops asking about texts that are out of the running, so the far end of its "
                         "scale rests on two or three comparisons, and a scale would keep that end as an anchor and "
                         "treat it as exact. Sort without --top to save a scale")
    named, unknown = run["model"]["answered"], run["model"]["unknown_answers"]
    if len(named) > 1 and not any_model:
        raise ScaleError("the fit mixes answers from " + " and ".join(f"{m} ({n:,})" for m, n in sorted(named.items()))
                         + ": the model ID asked for reached different models at different times, and the cache "
                         "kept the earlier answers. That is not one model's scale. Sort again with --no-cache or with "
                         "--model pinned to one of them, or pass --any-model to save it as it is")
    if unknown and not any_model:
        raise ScaleError(f"{unknown:,} of the {unknown + sum(named.values()):,} answers in this fit did not say which "
                         "model gave them (cache entries written before jsort recorded it, or an API that does not "
                         "name its model), so the scale cannot name the model it was built on. Sort again with "
                         "--no-cache, or pass --any-model to save it unverified; --scale will then need --any-model too")
    first: dict[str, int] = {}
    for i, text in enumerate(run["texts"]):   # identical texts share a score, so one of them stands for all
        if text.strip() and math.isfinite(ranking.score[i]) and math.isfinite(ranking.se[i]):
            first.setdefault(text, i)
    scored = len(first)
    # Placement takes an anchor's score as exact, so a text the run barely measured cannot be one. A full sort gives
    # every text about -k comparisons; one the budget or a refusal cut short may have left some with the opening two.
    needed = max(2, int(run["per_item"]) // 2)
    index = [i for i in first.values() if ranking.comparisons[i] >= needed]
    if scored < 2:
        raise ScaleError("fewer than two texts were scored, so there is nothing to anchor a scale on")
    if len(index) < 2:
        raise ScaleError(f"the run stopped before two texts had the {needed} comparisons an anchor needs at -k "
                         f"{run['per_item']}, so there is nothing to anchor a scale on; raise --budget and sort again")
    kept = [index[j] for j in choose([float(ranking.score[i]) for i in index],
                                     [float(ranking.se[i]) for i in index], int(anchors))]
    return Scale(
        description=run["description"], question=run["question"], model=run["model"],
        input={"max_chars": int(run["max_chars"]), "unit": unit, "field": field},
        fit={"texts": scored, "eligible": len(index), "anchor_min_comparisons": needed,
             "comparisons": int(ranking.asked), "rounds": int(ranking.rounds),
             "per_item": int(run["per_item"]), "seed": int(run["seed"]),
             "reliability": None if ranking.reliability is None else round(float(ranking.reliability), 4),
             "lean": round(float(ranking.lean), 6), "gamma": round(float(ranking.gamma), 6), "ridge": RIDGE,
             "over_budget": bool(ranking.over_budget), "failed": len(ranking.errors)},
        anchors=tuple(Anchor(run["texts"][i], round(float(ranking.score[i]), 6), round(float(ranking.se[i]), 6),
                             int(ranking.comparisons[i])) for i in kept),
        created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), jsort_version=__version__)
