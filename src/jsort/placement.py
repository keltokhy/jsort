"""Placing new texts on a saved scale: each one against the anchors only, whose scores stay fixed.

A text's score is the one-parameter version of the fit the scale came from (model.place). Which anchors
it meets is chosen as a computerised adaptive test chooses items: a first round spread across the scale,
then rounds of the anchors nearest the running estimate, where a comparison says the most. A round's
questions go out together, so a text is placed in about three round trips.

Nothing a text is asked depends on any other text in the input. Its random choices are drawn from a
generator seeded with the seed and the text itself, so a text gets the same questions, and so the same
score, alone or among a million others, and texts can be placed as they arrive.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np

from .core import Jev, JevError, JevFatal
from .engine import Ranking
from .model import place as locate
from .scale import Scale, ScaleError

ROUNDS = 3   # one spread across the scale and two near the estimate: the round trips a text waits for


@dataclass
class Placement(Ranking):
    """A Ranking whose scores are on the saved scale. `lean` is the scale's and there is no reliability:
    that belongs to the run that fitted the scale."""
    beyond: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))   # +1 above every anchor, -1 below, else 0
    unverified: int = 0            # answers that did not say which model gave them, so could not be checked against the scale's


class _Purse:
    """The budget, shared by every text being placed at once.

    arank sizes each batch of requests from the largest charge seen so far. Placement has no batches, so
    each request reserves that charge before it goes out, and until a charge has been seen one request
    goes alone to learn it. As there, the last request can take spending past the limit.
    """

    def __init__(self, budget: float):
        self.budget, self.spent, self.price, self.held, self.flying = budget, 0.0, 0.0, 0.0, 0
        self.over = False
        self.changed = asyncio.Condition()

    def charge(self, cost: float) -> None:
        self.spent += cost
        self.price = max(self.price, cost)

    def remaining(self) -> float:
        # Roundoff at the limit must not buy an extra request.
        return 0.0 if math.isclose(self.spent, self.budget, rel_tol=1e-12) else self.budget - self.spent

    async def admit(self, halted) -> float | None:
        """Wait for room. Returns what was reserved, or None when the run has stopped or the budget is spent."""
        if not self.budget:
            return None if halted() else 0.0
        async with self.changed:
            while not halted():
                remaining = self.remaining()
                if remaining <= 0:
                    self.over = True
                    self.changed.notify_all()
                    break
                alone = self.flying == 0
                if alone or (self.price and self.held + self.price <= remaining * (1 + 1e-12)):
                    self.flying += 1
                    self.held += self.price
                    return self.price
                await self.changed.wait()
        return None

    async def release(self, reserved: float) -> None:
        if self.budget:
            async with self.changed:
                self.flying -= 1
                self.held -= reserved
                self.changed.notify_all()


class Placer:
    """One run's shared state: the scale, the client, the budget and what went wrong. `place` takes one text."""

    def __init__(self, scale: Scale, jev: Jev, *, per_item: int = 10, se_target: float | None = None, seed: int = 0,
                 budget: float | None = None, max_chars: int | None = None, concurrency: int = 32,
                 any_model: bool = False):
        max_chars = scale.max_chars if max_chars is None else max_chars
        for name, value, minimum in (("per_item", per_item, 2), ("concurrency", concurrency, 1),
                                     ("max_chars", max_chars, 1), ("seed", seed, 0)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if se_target is not None and (isinstance(se_target, bool) or not isinstance(se_target, Real)
                                      or not math.isfinite(se_target) or se_target <= 0):
            raise ValueError("se_target must be finite and greater than 0")
        if budget is None:
            budget = float(os.environ.get("JSORT_BUDGET") or 1.0)
        if not math.isfinite(budget) or budget < 0:
            raise ValueError("budget must be finite and nonnegative")
        if not any_model:
            scale.check_model(jev.backend.name, jev.url, jev.model)
        self.scale, self.jev, self.any_model = scale, jev, any_model
        self.per_item, self.se_target, self.seed, self.max_chars = int(per_item), se_target, int(seed), int(max_chars)
        self.scores = np.array([a.score for a in scale.anchors])     # highest first, as the scale keeps them
        self.known = {a.text: a for a in scale.anchors}
        self.purse, self.sem = _Purse(budget), asyncio.Semaphore(concurrency)
        self.asked = self.rounds = self.unverified = 0
        self.errors: list[str] = []
        self.fatal: str | None = None

    @property
    def over_budget(self) -> bool:
        return self.purse.over

    @property
    def halted(self) -> bool:
        return bool(self.fatal) or self.purse.over

    async def _compare(self, shown: str, a: int, leads: bool) -> float | None:
        anchor = self.scale.anchors[a].text
        async with self.sem:
            reserved = await self.purse.admit(lambda: self.halted)
            if reserved is None:
                return None
            try:
                state = {"A": shown, "B": anchor} if leads else {"A": anchor, "B": shown}
                origin: dict = {}
                answer = await self.jev.ask(state, {"q": self.scale.question}, on_cost=self.purse.charge, provenance=origin)
                origin = origin.get("q") or {}
                # Each answer is checked, the cached ones too: the cache keys on the ID asked for, not on who replied.
                if not self.any_model and not self.scale.check_answer(origin.get("resolved_model"), origin.get("source", "api")):
                    self.unverified += 1
                return float(answer["q"]["noul"])
            except JevError as e:
                self.errors.append(str(e))
            except (JevFatal, ScaleError) as e:
                self.fatal = self.fatal or str(e)
            finally:
                await self.purse.release(reserved)
        return None

    def _spread(self, size: int, rng: np.random.Generator) -> list[int]:
        """One anchor drawn from each of `size` equal stretches of the scale."""
        stretches = np.array_split(np.arange(len(self.scores)), min(size, len(self.scores)))
        return [int(rng.choice(stretch)) for stretch in stretches]

    def _nearest(self, estimate: float, size: int, met: dict[int, list[bool]]) -> list[int]:
        """The anchors closest to the estimate, those not met yet first. An anchor is met twice at most, once in each position."""
        open_ = [a for a in range(len(self.scores)) if len(met.get(a, ())) < 2]
        open_.sort(key=lambda a: (len(met.get(a, ())), abs(self.scores[a] - estimate), a))
        return open_[:size]

    async def place(self, text: str) -> tuple[float, float, int, int]:
        """(score, standard error, comparisons, beyond) for one text. An unplaced text has a score of nan."""
        shown = text[:self.max_chars]
        if not shown.strip() or self.halted:
            return math.nan, math.nan, 0, 0
        if shown in self.known:    # an anchor is where the scale says it is, and comparing a text with itself says nothing
            anchor = self.known[shown]
            return anchor.score, anchor.se, anchor.comparisons, 0

        digest = hashlib.sha256(shown.encode("utf-8", "surrogatepass")).digest()
        rng = np.random.default_rng([self.seed, int.from_bytes(digest[:8], "big")])
        leads = bool(rng.integers(2))      # positions alternate from a random start, so the lean has both to work with
        met: dict[int, list[bool]] = {}
        against, led, ys = [], [], []
        estimate, se = 0.0, math.nan
        sizes = [self.per_item // ROUNDS + (r < self.per_item % ROUNDS) for r in range(ROUNDS)]
        for r, size in enumerate(s for s in sizes if s):
            picks = self._spread(size, rng) if r == 0 else self._nearest(estimate, size, met)
            if not picks or self.halted:
                break
            asks = []
            for a in picks:
                position = (not met[a][0]) if a in met else leads
                leads = not leads
                met.setdefault(a, []).append(position)
                asks.append((a, position))
            self.rounds = max(self.rounds, r + 1)
            for (a, position), y in zip(asks, await asyncio.gather(*(self._compare(shown, a, p) for a, p in asks))):
                if y is not None:
                    against.append(self.scores[a])
                    led.append(position)
                    ys.append(y)
                    self.asked += 1
            if ys:
                estimate, se = locate(against, led, ys, self.scale.gamma, start=estimate)
            # The first round's answers are mostly lopsided and few, so its standard error is not one to stop on.
            if self.se_target is not None and r >= 1 and ys and se <= self.se_target:
                break
        if not ys:
            return math.nan, math.nan, 0, 0
        low, high = self.scale.span
        return estimate, se, len(ys), int(estimate > high) - int(estimate < low)


async def aplace(texts: list[str], scale: Scale | str | os.PathLike, jev: Jev, *, concurrency: int = 32,
                 progress=None, **options) -> Placement:
    """Place texts on a saved scale, given a Scale or the path of one.

    per_item bounds the comparisons each text gets, all of them with anchors; se_target stops a text
    early, after its second round at the soonest, once its standard error is that small. max_chars
    defaults to the scale's. A scale built with another API, endpoint or model raises ScaleError unless
    any_model is set. budget is as in arank. Blank texts get no score; an anchor's own text gets the
    score the scale gave it.
    """
    if not isinstance(scale, Scale):
        scale = Scale.load(scale)
    placer = Placer(scale, jev, concurrency=concurrency, **options)
    results: list = [None] * len(texts)
    waiting = iter(enumerate(texts))
    done = 0

    async def worker() -> None:     # as many texts in hand as calls in flight, however long the input
        nonlocal done
        for k, text in waiting:
            results[k] = await placer.place(text)
            done += 1
            if progress:
                progress(done, len(texts))

    await asyncio.gather(*(worker() for _ in range(min(concurrency, len(texts)))))
    return collect(results, scale, placer)


def collect(results: list[tuple[float, float, int, int]], scale: Scale, placer: Placer | None = None) -> Placement:
    """One Placement from each text's (score, se, comparisons, beyond). Without a placer, nothing had to be asked."""
    columns = list(zip(*results)) or [(), (), (), ()]
    out = Placement(np.array(columns[0], dtype=float), np.array(columns[1], dtype=float),
                    np.array(columns[2], dtype=int), lean=float(scale.fit.get("lean") or 0.0), gamma=scale.gamma,
                    beyond=np.array(columns[3], dtype=int))
    if placer is not None:
        out.asked, out.rounds, out.errors, out.unverified = placer.asked, placer.rounds, placer.errors, placer.unverified
        out.over_budget, out.fatal = placer.over_budget, placer.fatal
    return out


def place(texts: list[str], scale: Scale | str | os.PathLike, *, api: str | None = None, model: str | None = None,
          cache: bool = True, timeout: float = 15.0, transport=None, **options) -> Placement:
    """aplace from ordinary code, as rank is to arank. The API and model default to the ones the scale was built with.

        scale = jsort.rank(statements, "more hawkish about inflation").scale()
        scale.save("hawkish.json")
        p = jsort.place(new_statements, "hawkish.json")
        p.score, p.se, p.beyond
    """
    import concurrent.futures

    from .core import Cache

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and greater than 0")
    if not isinstance(scale, Scale):
        scale = Scale.load(scale)

    async def go() -> Placement:
        backend, key, requested = client_for(scale, api, model)
        jev = Jev(key, backend, model=requested, timeout=timeout, concurrency=options.get("concurrency", 32),
                  cache=Cache() if cache else None, transport=transport)
        try:
            return await aplace(texts, scale, jev, **options)
        finally:
            await jev.close()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(go())
    with concurrent.futures.ThreadPoolExecutor(1) as pool:   # a notebook already has a loop running
        return pool.submit(asyncio.run, go()).result()


def client_for(scale: Scale, api: str | None, model: str | None):
    """The backend, key and model to ask: what was given, else the environment's, else the scale's own.

    A model ID belongs to its API, so the scale's is only borrowed when the API is the scale's too.
    """
    from .core import resolve_backend

    backend, key = resolve_backend(api or os.environ.get("JEV_API") or scale.model["api"])
    if not (model or os.environ.get("JEV_MODEL")) and backend.name == scale.model["api"]:
        model = scale.model["requested"]
    return backend, key, model
