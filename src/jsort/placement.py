"""Placing new texts on a saved scale: each one against the anchors only, whose scores stay fixed.

A text's score is the one-parameter version of the fit the scale came from (model.place). Which anchors
it meets is chosen as a computerised adaptive test chooses items: a first round spread across the scale,
then rounds of the anchors nearest the running estimate, where a comparison says the most. A round's
questions go out together, so a text is placed in about three round trips. Every text gets all -k of its
comparisons; nothing stops on a standard error that looks small, which would select for small estimates of it.

Nothing a text is asked depends on any other text in the input. Its random choices are drawn from a
generator seeded with the seed and the text itself, so a text gets the same questions, and so the same
score, alone or among a million others, and texts can be placed as they arrive.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from .core import PRICE_PER_MTOK, Jev, JevError, JevFatal
from .engine import Ranking
from .model import place as locate
from .scale import Scale, ScaleError

ROUNDS = 3   # one spread across the scale and two near the estimate: the round trips a text waits for
SE_MARGIN = 2.0   # logits past an end anchor: saturation makes the reported error misleading


@dataclass
class Placement(Ranking):
    """A Ranking whose scores are on the saved scale. `lean` is the scale's and there is no reliability:
    that belongs to the run that fitted the scale."""
    beyond: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))   # +1 above every anchor, -1 below, else 0
    unverified: int = 0            # answers that did not say which model gave them, so could not be checked against the scale's
    partial: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))   # placed on fewer comparisons than were planned
    model_refused: bool = False    # none of a batch's scores may print after a model mismatch


def _tokens(model: str, state: dict, question: dict) -> int:
    """What a request will bill, near enough to reserve for it: jgrep's estimate, a token for every four bytes
    of the request and 270 of overhead."""
    payload = json.dumps({"model": model, "state": state, "questions": {"q": question}}, ensure_ascii=False)
    return math.ceil(len(payload.encode("utf-8")) / 4) + 270


class _Purse:
    """The budget, shared by every text being placed at once.

    Nothing is reserved from what earlier replies happened to cost: a short probe says nothing about the long
    documents behind it. A request is priced before it goes, from its own size, at the dearest rate per
    estimated token seen so far, and until a charge has been seen at MARGIN times the list price.

    A text is admitted only when the budget covers every comparison it may ask for, and texts are admitted in
    input order. So when the money runs out some texts are placed in full and the rest not at all, and a score
    does not depend on what else was in the input. Only a rise in price can cut a text short once it has begun,
    and such a text is reported as partial.

    While nothing is known the first request goes alone, and the number of texts in hand then doubles with each
    text that finishes without the rate rising, from one up to -j; a rise sends it back to one. What can still
    take spending past the limit is the calls in the air at the moment a price rises.
    """

    MARGIN = 1.5     # on the list price, while no charge has been seen
    RISE = 1.25      # a rate this much above the one reserved at is a rise, not jitter in the estimate

    def __init__(self, budget: float, width: int):
        self.budget, self.width, self.spent = budget, width, 0.0
        self.rate = 0.0                    # dollars per estimated token: the dearest seen
        self.held = self.flying = 0        # tokens reserved by the texts in hand, and how many texts that is
        self.flight = 0                    # tokens of the requests in the air
        self.window, self.rises = 1, 0
        self.tickets = self.serving = 0    # texts are admitted in the order they asked
        self.gone: set[int] = set()
        self.over = False
        self.changed = asyncio.Condition()
        self.probe = asyncio.Lock()

    def price(self, tokens: int) -> float:
        return tokens * (self.rate or PRICE_PER_MTOK / 1e6 * self.MARGIN)

    def remaining(self) -> float:
        # Roundoff at the limit must not buy an extra request.
        return 0.0 if math.isclose(self.spent, self.budget, rel_tol=1e-12) else self.budget - self.spent

    def charge(self, cost: float, tokens: int) -> None:
        self.spent += cost
        if cost / tokens > self.RISE * (self.rate or PRICE_PER_MTOK / 1e6 * self.MARGIN):
            self.rises += 1
            self.window = 1
        self.rate = max(self.rate, cost / tokens)

    async def admit(self, tokens: int, stopped) -> int | None:
        """Wait in line for room for a whole text. Returns the state of `rises` on admission, or None: the run has
        stopped, or the budget cannot cover this text and nothing in hand will give any of it back."""
        if not self.budget:
            return None if stopped() else 0
        ticket, self.tickets = self.tickets, self.tickets + 1
        async with self.changed:
            try:
                while not (stopped() or self.over):
                    if ticket == self.serving:
                        if self.flying < self.window and self.price(self.held + tokens) <= self.remaining():
                            self.held, self.flying = self.held + tokens, self.flying + 1
                            return self.rises
                        if not self.flying:
                            self.over = True
                            break
                    await self.changed.wait()
                return None
            finally:       # admitted, refused or cancelled, the text leaves the line and the next one is served
                self.gone.add(ticket)
                while self.serving in self.gone:
                    self.gone.discard(self.serving)
                    self.serving += 1
                self.changed.notify_all()

    def take(self, tokens: int) -> bool:
        """Whether a request may go, counting what is already in the air at today's rate. Its text's admission
        covered it unless the price has risen since; then the request is not sent and the text is cut short."""
        if self.budget:
            if self.price(self.flight + tokens) > self.remaining():
                self.over = True
                return False
            self.flight += tokens
        return True

    def give(self, tokens: int) -> None:
        if self.budget:
            self.flight -= tokens

    async def leave(self, tokens: int, admitted_at: int) -> None:
        """A text is done with what is left of its reservation."""
        if self.budget:
            async with self.changed:
                self.held, self.flying = self.held - tokens, self.flying - 1
                if self.rises == admitted_at:
                    self.window = min(2 * self.window, self.width)
                self.changed.notify_all()


class Placer:
    """One run's shared state: the scale, the client, the budget and what went wrong. `place` takes one text."""

    def __init__(self, scale: Scale, jev: Jev, *, per_item: int = 10, seed: int = 0, budget: float | None = None,
                 max_chars: int | None = None, concurrency: int = 32, any_model: bool = False):
        max_chars = scale.max_chars if max_chars is None else max_chars
        for name, value, minimum in (("per_item", per_item, 2), ("concurrency", concurrency, 1),
                                     ("max_chars", max_chars, 1), ("seed", seed, 0)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if budget is None:
            budget = float(os.environ.get("JSORT_BUDGET") or 1.0)
        if not math.isfinite(budget) or budget < 0:
            raise ValueError("budget must be finite and nonnegative")
        if not any_model:
            scale.check_model(jev.backend.name, jev.url, jev.model)
        self.scale, self.jev, self.any_model = scale, jev, any_model
        self.per_item, self.seed, self.max_chars = int(per_item), int(seed), int(max_chars)
        self.scores = np.array([a.score for a in scale.anchors])     # highest first, as the scale keeps them
        self.known = {a.text: a for a in scale.anchors}
        self.purse, self.sem = _Purse(budget, concurrency), asyncio.Semaphore(concurrency)
        self.model = getattr(jev, "model", None) or ""      # only to size a request; a stand-in judge may have none
        self.longest = max((a.text for a in scale.anchors), key=len)
        self.planned = min(self.per_item, 2 * len(scale.anchors))     # an anchor is met twice at most
        self.asked = self.rounds = self.unverified = 0
        self.errors: list[str] = []
        self.fatal: str | None = None
        self.probed = self.model_refused = False
        self.placed: dict[bytes, asyncio.Future] = {}   # one result per shown text, even without the answer cache

    @property
    def over_budget(self) -> bool:
        return self.purse.over

    @property
    def halted(self) -> bool:
        return bool(self.fatal) or self.purse.over

    async def _ask(self, state: dict, tokens: int, origin: dict) -> dict | None:
        """One request, if the budget has room for it beside those already in the air. None when it does not."""
        if not self.purse.take(tokens):
            return None
        try:
            return await self.jev.ask(state, {"q": self.scale.question}, provenance=origin,
                                      on_cost=lambda cost: self.purse.charge(cost, tokens))
        finally:
            self.purse.give(tokens)

    def _state(self, shown: str, anchor: str, leads: bool) -> dict:
        return {"A": shown, "B": anchor} if leads else {"A": anchor, "B": shown}

    async def _compare(self, state: dict, tokens: int) -> float | None:
        async with self.sem:
            if self.fatal:
                return None
            if not self.probed:
                async with self.purse.probe:
                    if self.fatal:
                        return None
                    if not self.probed:
                        # Check identity before releasing the lock. A cached answer cannot confirm today's alias,
                        # and a zero-cost reply can; the first live request goes alone even with --budget 0.
                        return await self._answer(state, tokens)
            return await self._answer(state, tokens)

    async def _answer(self, state: dict, tokens: int) -> float | None:
        try:
            origin: dict = {}
            answer = await self._ask(state, tokens, origin)
            if answer is None:
                return None
            origin = origin.get("q") or {}
            verified = bool(origin.get("resolved_model"))
            # Each answer is checked, the cached ones too: the cache keys on the ID asked for, not on who replied.
            if not self.any_model:
                verified = self.scale.check_answer(origin.get("resolved_model"), origin.get("source", "api"))
                self.unverified += not verified
            if origin.get("source", "api") == "api" and (verified or self.any_model):
                self.probed = True
            return float(answer["q"]["noul"])
        except JevError as e:
            self.errors.append(str(e))
        except (JevFatal, ScaleError) as e:
            self.model_refused |= isinstance(e, ScaleError)
            self.fatal = self.fatal or str(e)
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

    async def place(self, text: str) -> tuple[float, float, int, int, bool]:
        """(score, standard error, comparisons, beyond, partial) for one text. An unplaced text has a score of nan."""
        nowhere = (math.nan, math.nan, 0, 0, False)
        shown = text[:self.max_chars]
        if not shown.strip():
            return nowhere
        if shown in self.known:    # an anchor is where the scale says it is, and comparing a text with itself says nothing
            anchor = self.known[shown]
            return anchor.score, anchor.se, anchor.comparisons, 0, False
        digest = hashlib.sha256(shown.encode("utf-8", "surrogatepass")).digest()
        if digest in self.placed:
            return await asyncio.shield(self.placed[digest])
        if self.halted:
            return nowhere
        result = self.placed[digest] = asyncio.get_running_loop().create_future()
        try:
            # Room for every comparison the text may ask for, each priced as if it met the longest anchor.
            reserve = self.planned * _tokens(self.model, self._state(shown, self.longest, True), self.scale.question)
            admitted_at = await self.purse.admit(reserve, lambda: self.halted)
            if admitted_at is None:
                result.set_result(nowhere)
            else:
                try:
                    result.set_result(await self._place(shown))
                finally:
                    await self.purse.leave(reserve, admitted_at)
            return result.result()
        finally:
            if not result.done():
                result.cancel()       # wake any duplicate waiting on a cancelled placement

    async def _place(self, shown: str) -> tuple[float, float, int, int, bool]:
        digest = hashlib.sha256(shown.encode("utf-8", "surrogatepass")).digest()
        rng = np.random.default_rng([self.seed, int.from_bytes(digest[:8], "big")])
        leads = bool(rng.integers(2))      # positions alternate from a random start, so the lean has both to work with
        met: dict[int, list[bool]] = {}
        against, led, ys = [], [], []
        estimate, se, planned = 0.0, math.nan, 0
        sizes = [self.per_item // ROUNDS + (r < self.per_item % ROUNDS) for r in range(ROUNDS)]
        for r, size in enumerate(s for s in sizes if s):
            picks = self._spread(size, rng) if r == 0 else self._nearest(estimate, size, met)
            if not picks:
                break
            planned += len(picks)
            asks = []
            for a in picks:
                position = (not met[a][0]) if a in met else leads
                leads = not leads
                met.setdefault(a, []).append(position)
                state = self._state(shown, self.scale.anchors[a].text, position)
                asks.append((a, position, state, _tokens(self.model, state, self.scale.question)))
            self.rounds = max(self.rounds, r + 1)
            answers = [] if self.fatal else await asyncio.gather(*(self._compare(ask[2], ask[3]) for ask in asks))
            for (a, position, _, _), y in zip(asks, answers):
                if y is not None:
                    against.append(self.scores[a])
                    led.append(position)
                    ys.append(y)
                    self.asked += 1
            if ys:
                estimate, se = locate(against, led, ys, self.scale.gamma, start=estimate, ridge=self.scale.ridge)
            if self.fatal or (self.purse.over and len(answers) > sum(y is not None for y in answers)):
                planned += sum(sizes[r + 1:])          # cut short: the rounds that will not be asked were planned too
                break
        if not ys:
            return math.nan, math.nan, 0, 0, False
        low, high = self.scale.span
        if len(ys) < 3 or estimate < low - SE_MARGIN or estimate > high + SE_MARGIN:
            se = math.nan
        return estimate, se, len(ys), int(estimate > high) - int(estimate < low), len(ys) < min(planned, self.planned)


async def aplace(texts: list[str], scale: Scale | str | os.PathLike, jev: Jev, *, concurrency: int = 32,
                 progress=None, **options) -> Placement:
    """Place texts on a saved scale, given a Scale or the path of one.

    per_item is the comparisons each text gets, all of them with anchors. Every text gets them all: there is
    no stopping once a standard error looks small, because a rule that stops on the reported error selects for
    small estimates of it. max_chars defaults to the scale's. A scale that cannot name one model, or that was
    built with another API, endpoint or model, raises ScaleError unless any_model is set. budget is the
    dollars the run may spend: a text is placed in full or not at all, and `partial` marks one cut short.
    Blank texts get no score; an anchor's own text gets the score the scale gave it.
    Repeated shown texts share one result within the run. A new text's se is nan below three successful
    comparisons or more than two logits past an end anchor, where it would suggest misleading precision.
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


def collect(results: list[tuple[float, float, int, int, bool]], scale: Scale, placer: Placer | None = None) -> Placement:
    """One Placement from each text's (score, se, comparisons, beyond, partial). Without a placer, nothing had to be asked."""
    columns = list(zip(*results)) or [(), (), (), (), ()]
    out = Placement(np.array(columns[0], dtype=float), np.array(columns[1], dtype=float),
                    np.array(columns[2], dtype=int), lean=float(scale.fit.get("lean") or 0.0), gamma=scale.gamma,
                    beyond=np.array(columns[3], dtype=int), partial=np.array(columns[4], dtype=bool))
    if placer is not None:
        out.asked, out.rounds, out.errors, out.unverified = placer.asked, placer.rounds, placer.errors, placer.unverified
        out.over_budget, out.fatal = placer.over_budget, placer.fatal
        out.model_refused = placer.model_refused
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
