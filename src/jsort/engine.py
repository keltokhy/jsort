"""Rounds of comparisons: ask a batch of pairs, refit the scale, choose the next batch."""

from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from jevkit_runtime import Client, JevError, JevFatal
from .model import Fit, fit, information, reliability, shortfall, standard_errors
from .scale import DEFAULT_ANCHORS, Scale, build, identity, question  # noqa: F401  (question is part of this module's API)
from .schedule import Schedule

SETTLE = 3    # comparisons a text must have before --top may stop asking about it
OUT = 4.0     # and how much worse the fit must get, in log-likelihood, were the text moved up into the top


@dataclass
class Ranking:
    """Scores line up with the texts passed in. A text that was never compared has a score of nan."""
    score: np.ndarray
    se: np.ndarray
    comparisons: np.ndarray
    lean: float = 0.0              # how far the first-shown text's chance sits from 0.5 in an even matchup
    reliability: float | None = None
    asked: int = 0
    rounds: int = 0
    errors: list[str] = field(default_factory=list)
    over_budget: bool = False
    fatal: str | None = None
    gamma: float = 0.0             # the same lean in logits, as the model holds it; a saved scale places with it
    run: dict | None = field(default=None, repr=False)   # what was asked, of whom, about which texts: what scale() saves

    @classmethod
    def unscored(cls, n: int) -> "Ranking":
        return cls(np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=int))

    def order(self, reverse: bool = False) -> list[int]:
        """Indices from highest score to lowest (lowest first when reversed); unscored texts go last, in input order."""
        scored = [i for i in range(len(self.score)) if not math.isnan(self.score[i])]
        scored.sort(key=lambda i: self.score[i] if reverse else -self.score[i])
        return scored + [i for i in range(len(self.score)) if math.isnan(self.score[i])]

    def scale(self, anchors: int = DEFAULT_ANCHORS, *, unit: str | None = None, field: str | None = None,
              any_model: bool = False) -> Scale:
        """This run's scale, with `anchors` of its texts kept for placing later ones: `r.scale().save(path)`.

        Raises ScaleError when the answers behind the fit do not all name one model, unless any_model is set.
        """
        return build(self, anchors, unit=unit, field=field, any_model=any_model)


async def arank(texts: list[str], description: str, jev: Client, *, per_item: int = 10, top: int | None = None,
                lowest: bool = False, seed: int = 0, budget: float | None = None, max_chars: int = 8000,
                concurrency: int = 32, progress=None) -> Ranking:
    """Place texts on a scale by asking Jev about pairs of them.

    per_item is the number of comparisons each text takes part in, so the whole run asks about
    len(texts) * per_item / 2 questions. With top set, texts that are clearly outside the top stop being
    asked about once that is clear, and the remaining questions go to the ones still in contention.
    With lowest as well, it is the bottom of the scale that is wanted, and the hunt runs the other way.
    Identical and blank texts are handled here: duplicates share one score and blanks get none.
    budget defaults to $JSORT_BUDGET, else 1.00, per run; 0 disables the limit.
    """
    if not description.strip():
        raise ValueError("description must not be empty")
    for name, value, minimum in (("per_item", per_item, 2), ("concurrency", concurrency, 1),
                                 ("max_chars", max_chars, 1), ("top", top, 1)):
        if name == "top" and value is None:
            continue
        if not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if budget is None:
        budget = float(os.environ.get("JSORT_BUDGET") or 1.0)
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and nonnegative")
    spent = max_call_cost = 0.0
    shown = [t[:max_chars] for t in texts]
    unique: dict[str, int] = {}
    member = [unique.setdefault(t, len(unique)) if t.strip() else -1 for t in shown]
    items = list(unique)
    n = len(items)

    q = question(description)
    schedule = Schedule(n, seed)
    first: list[int] = []
    second: list[int] = []
    ys: list[float] = []
    out = Ranking.unscored(len(texts))
    sem = asyncio.Semaphore(concurrency)
    total = math.ceil(n * per_item / 2)   # the opening ring alone is two comparisons per text

    def record_cost(cost: float) -> None:
        nonlocal spent, max_call_cost
        spent += cost
        max_call_cost = max(max_call_cost, cost)

    def remaining_budget() -> float:
        # Roundoff at the limit must not buy an extra request.
        return 0.0 if math.isclose(spent, budget, rel_tol=1e-12) else budget - spent

    async def compare(i: int, j: int):
        async with sem:
            if out.fatal or out.over_budget:
                return None
            if budget and remaining_budget() <= 0:
                out.over_budget = True
                return None
            try:
                answer = await jev.ask({"A": items[i], "B": items[j]}, {"q": q}, on_cost=record_cost)
                return i, j, float(answer["q"]["noul"]), answer.origins["q"].get("resolved_model")
            except JevError as e:
                out.errors.append(str(e))
            except JevFatal as e:
                out.fatal = out.fatal or str(e)
            return None

    fitted = Fit(np.zeros(n), 0.0, True)
    responders: dict[str | None, int] = {}     # which model gave each answer the fit uses; None where that is not known
    active = np.ones(n, dtype=bool)
    scheduled = 0
    while scheduled < total and not (out.fatal or out.over_budget):
        room = total - scheduled
        if out.rounds == 0:
            pairs = schedule.ring(room)
        else:
            spread = 1 / np.sqrt(information(n, first, second, fitted))
            if top is not None and top < n:
                settled = schedule.count >= SETTLE
                # For the bottom, turn the scale over: negate it and swap the two positions, which leaves
                # every fitted probability as it was.
                turned = Fit(-fitted.theta, fitted.gamma, True) if lowest else fitted
                a, b = (second, first) if lowest else (first, second)
                bar = np.sort(turned.theta - spread)[-top]
                active = ~settled | (shortfall(n, a, b, ys, turned, bar) < OUT)
            # The questions saved on texts that are out of the running go to the ones still in it.
            cap = per_item if top is None else 2 * per_item
            pairs = schedule.neighbours(fitted.theta, spread, active, room, cap)
        if not pairs:
            break
        out.rounds += 1
        scheduled += len(pairs)
        offset = 0
        while offset < len(pairs) and not (out.fatal or out.over_budget):
            size = len(pairs) - offset
            if budget:
                remaining = remaining_budget()
                if remaining <= 0:
                    out.over_budget = True
                    break
                # Learn the price with one request, then reserve the largest observed charge per
                # in-flight request. A final request or an unexpected price increase can still overshoot.
                cost = max_call_cost
                size = min(concurrency, max(1, int(remaining / cost))) if cost else 1
            batch = pairs[offset:offset + size]
            offset += len(batch)
            for result in await asyncio.gather(*(compare(i, j) for i, j in batch)):
                if result is not None:
                    first.append(result[0])
                    second.append(result[1])
                    ys.append(result[2])
                    responders[result[3]] = responders.get(result[3], 0) + 1
        fitted = fit(n, first, second, ys, start=fitted)
        if progress:
            progress(scheduled, total)

    out.asked = len(ys)
    if ys:
        compared = np.bincount(first, minlength=n) + np.bincount(second, minlength=n)
        theta = np.where(compared > 0, fitted.theta, np.nan)
        theta -= np.nanmean(theta)
        se = np.where(compared > 0, standard_errors(n, first, second, ys, fitted), np.nan)
        for k, m in enumerate(member):
            if m >= 0:
                out.score[k], out.se[k], out.comparisons[k] = theta[m], se[m], compared[m]
        out.lean = float(1 / (1 + math.exp(-fitted.gamma)) - 0.5)
        out.gamma = float(fitted.gamma)
        out.reliability = reliability(n, first, second, ys, seed=seed)
        out.run = {"description": description, "question": q, "model": identity(jev, responders), "max_chars": max_chars,
                   "per_item": per_item, "seed": seed, "top": top, "texts": shown}
    return out


def rank(texts: list[str], description: str, *, api: str | None = None, model: str | None = None,
         cache: bool = True, timeout: float = 15.0, transport=None, **options) -> Ranking:
    """The same as the command line, from Python. Keys are found the way the command finds them.

        import jsort
        r = jsort.rank(statements, "more hawkish about inflation")
        for i in r.order()[:5]:
            print(f"{r.score[i]:6.2f} ±{r.se[i]:.2f}  {statements[i]}")

    Spending stops at `budget` dollars, which defaults as the command's does: $JSORT_BUDGET, else 1.00, and
    0 for no limit. `r.over_budget` says whether it was reached. Other options are arank's.
    """
    import concurrent.futures

    from jevkit_runtime import AnswerStore, resolve
    from .core import PROVIDERS

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and greater than 0")

    async def go() -> Ranking:
        backend = resolve(PROVIDERS, api, model=model)
        jev = Client(backend, timeout=timeout, concurrency=options.get("concurrency", 32),
                  store=AnswerStore() if cache else None, transport=transport)
        try:
            return await arank(texts, description, jev, **options)
        finally:
            await jev.close()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(go())
    with concurrent.futures.ThreadPoolExecutor(1) as pool:   # a notebook already has a loop running
        return pool.submit(asyncio.run, go()).result()
