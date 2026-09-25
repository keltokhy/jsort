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
import math
import os
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from jevkit_runtime import Answers, Budget, Client, JevBudgetExceeded, JevError, JevFatal, estimate_tokens, request_body
from jevkit_runtime.cli import run_sync
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


def _tokens(model: str, state: dict, question) -> int:
    """What a comparison will bill, near enough to set money aside for: the runtime's estimate."""
    return estimate_tokens(request_body(model, state, {"q": question}))


class _Admission:
    """Which texts are placed when the money runs out: whole texts, in input order.

    A text is admitted only with a share of the budget covering every comparison it may ask for, and texts
    are admitted in the order they came, each joining the budget's line before the next. So when the money
    runs out some texts are placed in full and the rest not at all, and a score does not depend on what else
    was in the input. Only a rise in price can cut a text short once it has begun, and such a text is
    reported as partial.

    While nothing is known the first text goes alone, and the number of texts in hand then doubles with each
    text that finishes without the price rising, from one up to -j; a rise sends it back to one. Without a
    limit there is nothing to protect, and -j texts go at once.
    """

    def __init__(self, budget: Budget, width: int):
        self.budget, self.width = budget, width
        self.window = width if budget.unlimited else 1
        self.flying = self.tickets = self.serving = 0
        self.gone: set[int] = set()
        self.over = False
        self.room = asyncio.Condition()

    async def admit(self, price, stopped) -> tuple[Budget, int] | None:
        """Wait in line for a whole text's share: the share and the rises seen at admission, or None when the
        run has stopped or the budget cannot cover the text even once everything in hand has come back.
        `price()` is asked at admission, so a charge seen while the text waited sets what it reserves."""
        ticket, self.tickets = self.tickets, self.tickets + 1
        async with self.room:
            try:
                await self.room.wait_for(
                    lambda: stopped() or self.over or (ticket == self.serving and self.flying < self.window))
                if stopped() or self.over:
                    return None
                self.flying += 1
            finally:       # admitted, refused or cancelled, the text leaves the line and the next one is served
                self.gone.add(ticket)
                while self.serving in self.gone:
                    self.gone.discard(self.serving)
                    self.serving += 1
                self.room.notify_all()
        try:
            # A cache-only run (a zero budget) still places whatever the cache holds, text by text.
            share = await self.budget.allot(price() if self.budget.limit > 0 else 0.0)
        except BaseException as refused:
            if isinstance(refused, JevBudgetExceeded):
                self.over = True
            async with self.room:
                self.flying -= 1
                self.room.notify_all()
            if isinstance(refused, JevBudgetExceeded):
                return None
            raise
        return share, self.budget.rises

    async def leave(self, share: Budget, admitted_at: int) -> None:
        """A text is done with what is left of its share."""
        share.close()
        async with self.room:
            self.flying -= 1
            if not self.budget.unlimited:
                self.window = min(2 * self.window, self.width) if self.budget.rises == admitted_at else 1
            self.room.notify_all()


class Placer:
    """One run's shared state: the scale, the client, the budget and what went wrong. `place` takes one text."""

    def __init__(self, scale: Scale, jev: Client, *, per_item: int = 10, seed: int = 0,
                 max_chars: int | None = None, concurrency: int = 32, any_model: bool = False,
                 budget: Budget | None = None):
        max_chars = scale.max_chars if max_chars is None else max_chars
        for name, value, minimum in (("per_item", per_item, 2), ("concurrency", concurrency, 1),
                                     ("max_chars", max_chars, 1), ("seed", seed, 0)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if not any_model:
            scale.check_model(jev.backend.name, jev.url, jev.model)
        self.scale, self.jev, self.any_model = scale, jev, any_model
        self.per_item, self.seed, self.max_chars = int(per_item), int(seed), int(max_chars)
        self.scores = np.array([a.score for a in scale.anchors])     # highest first, as the scale keeps them
        self.known = {a.text: a for a in scale.anchors}
        self.asked_question = scale.asked
        # This run's budget, else the client's; a stand-in judge may have none, which is no limit.
        self.budget = budget or getattr(jev, "budget", None) or Budget()
        self.admission, self.sem = _Admission(self.budget, concurrency), asyncio.Semaphore(concurrency)
        self.model = getattr(jev, "model", None) or ""      # only to size a request; a stand-in judge may have none
        self.longest = max((a.text for a in scale.anchors), key=len)
        self.planned = min(self.per_item, 2 * len(scale.anchors))     # an anchor is met twice at most
        self.asked = self.rounds = self.unverified = 0
        self.errors: list[str] = []
        self.fatal: str | None = None
        self.probed = self.model_refused = False
        self.probe = asyncio.Lock()
        self.placed: dict[bytes, asyncio.Future] = {}   # one result per shown text, even without the answer cache

    @property
    def over_budget(self) -> bool:
        return self.admission.over

    @property
    def halted(self) -> bool:
        return bool(self.fatal) or self.admission.over

    async def _ask(self, state: dict, share: Budget) -> Answers | None:
        """One comparison, drawn on the text's share. None when the share has no room for it: prices rose
        after the text was admitted, or a cache-only run meets a comparison it has not seen."""
        try:
            return await self.jev.ask(state, {"q": self.asked_question}, budget=share)
        except JevBudgetExceeded:
            if self.budget.limit > 0:
                self.admission.over = True
            return None

    def _state(self, shown: str, anchor: str, leads: bool) -> dict:
        return {"A": shown, "B": anchor} if leads else {"A": anchor, "B": shown}

    async def _compare(self, state: dict, share: Budget) -> float | None:
        async with self.sem:
            if self.fatal:
                return None
            if not self.probed:
                async with self.probe:
                    if self.fatal:
                        return None
                    if not self.probed:
                        # Check identity before releasing the lock. A cached answer cannot confirm today's model,
                        # and a live reply can; the first live request goes alone.
                        return await self._answer(state, share)
            return await self._answer(state, share)

    async def _answer(self, state: dict, share: Budget) -> float | None:
        try:
            answer = await self._ask(state, share)
            if answer is None:
                return None
            origin = answer.origins["q"]
            verified = bool(origin.get("resolved_model"))
            # Each answer is checked, the cached ones too: the cache keys on the ID asked for, not on who replied.
            if not self.any_model:
                verified = self.scale.check_answer(origin.get("resolved_model"), origin.get("source", "api"))
                self.unverified += not verified
            if origin.get("source", "api") == "api" and (verified or self.any_model):
                self.probed = True
            return self.asked_question.value(answer["q"])
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
            tokens = self.planned * _tokens(self.model, self._state(shown, self.longest, True), self.asked_question)
            backend = getattr(self.jev, "backend", None)
            price = (lambda: self.budget.price(backend, tokens)) if backend is not None else (lambda: 0.0)
            admitted = await self.admission.admit(price, lambda: self.halted)
            if admitted is None:
                result.set_result(nowhere)
            else:
                share, admitted_at = admitted
                try:
                    result.set_result(await self._place(shown, share))
                finally:
                    await self.admission.leave(share, admitted_at)
            return result.result()
        finally:
            if not result.done():
                result.cancel()       # wake any duplicate waiting on a cancelled placement

    async def _place(self, shown: str, share: Budget) -> tuple[float, float, int, int, bool]:
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
                asks.append((a, position, state))
            self.rounds = max(self.rounds, r + 1)
            answers = [] if self.fatal else await asyncio.gather(*(self._compare(ask[2], share) for ask in asks))
            for (a, position, _), y in zip(asks, answers):
                if y is not None:
                    against.append(self.scores[a])
                    led.append(position)
                    ys.append(y)
                    self.asked += 1
            if ys:
                estimate, se = locate(against, led, ys, self.scale.gamma, start=estimate, ridge=self.scale.ridge)
            if self.fatal or (self.admission.over and len(answers) > sum(y is not None for y in answers)):
                planned += sum(sizes[r + 1:])          # cut short: the rounds that will not be asked were planned too
                break
        if not ys:
            return math.nan, math.nan, 0, 0, False
        low, high = self.scale.span
        if len(ys) < 3 or estimate < low - SE_MARGIN or estimate > high + SE_MARGIN:
            se = math.nan
        return estimate, se, len(ys), int(estimate > high) - int(estimate < low), len(ys) < min(planned, self.planned)


async def aplace(texts: list[str], scale: Scale | str | os.PathLike, jev: Client, *, concurrency: int = 32,
                 progress=None, **options) -> Placement:
    """Place texts on a saved scale, given a Scale or the path of one.

    per_item is the comparisons each text gets, all of them with anchors. Every text gets them all: there is
    no stopping once a standard error looks small, because a rule that stops on the reported error selects for
    small estimates of it. max_chars defaults to the scale's. A scale that cannot name one model, or that was
    built with another API, endpoint or model, raises ScaleError unless any_model is set. Spending is the
    client's budget: a text is placed in full or not at all, and `partial` marks one cut short.
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
          cache: bool = True, timeout: float = 15.0, budget: float | Budget | None = None, transport=None,
          **options) -> Placement:
    """aplace from ordinary code, as rank is to arank. The API and model default to the ones the scale was built with.

        scale = jsort.rank(statements, "more hawkish about inflation").scale()
        scale.save("hawkish.json")
        p = jsort.place(new_statements, "hawkish.json")
        p.score, p.se, p.beyond

    `budget` is as rank's: dollars, math.inf for no limit, 0 for the cache only, or a Budget; by default
    $JEV_BUDGET, else 1.00.
    """
    from jevkit_runtime import AnswerStore
    from .engine import DEFAULT_BUDGET

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and greater than 0")
    if not isinstance(scale, Scale):
        scale = Scale.load(scale)
    budget = budget if isinstance(budget, Budget) else (
        Budget.from_settings(DEFAULT_BUDGET) if budget is None else Budget(budget))

    async def go() -> Placement:
        jev = Client(client_for(scale, api, model), timeout=timeout, concurrency=options.get("concurrency", 32),
                     store=AnswerStore() if cache else None, budget=budget, transport=transport)
        try:
            return await aplace(texts, scale, jev, **options)
        finally:
            await jev.close()

    return run_sync(go())


def client_for(scale: Scale, api: str | None, model: str | None):
    """The backend to ask: what was given, else the environment's, else the scale's own.

    A model ID belongs to its API, so the scale's is only borrowed when the API is the scale's too.
    """
    from dataclasses import replace

    from jevkit_runtime import Settings, resolve
    from .core import PROVIDERS

    settings = Settings.from_env()
    backend = resolve(PROVIDERS, api or settings.api or scale.model["api"], model=model)
    if not (model or settings.model) and backend.name == scale.model["api"]:
        backend = replace(backend, model=scale.model["requested"])
    return backend
