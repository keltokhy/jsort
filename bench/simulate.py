"""Does the design work? Offline: a simulated judge with a known scale, no network and no key.

The judge answers p = sigmoid(theta_a - theta_b + gamma + e_ab). e_ab is a fixed quirk of that ordered
pair, so the judge is deterministic, as Jev nearly is, but does not sit exactly on one scale. The
target is the scale a fit to every ordered pair would give; the questions are how close the
comparisons jsort chooses get to it, whether the intervals cover it, and whether the reported
reliability matches the truth.

    uv run python bench/simulate.py
"""

import asyncio
import math
import zlib

import numpy as np

from jevkit_runtime import Answers, Meter
from jsort.engine import arank
from jsort.model import fit
from jsort.placement import aplace


class SimulatedJev:
    def __init__(self, theta, gamma=-0.12, quirk=0.8):
        self.theta, self.gamma, self.quirk, self.meter = theta, gamma, quirk, Meter()

    def p(self, a: int, b: int) -> float:
        e = np.random.default_rng(zlib.crc32(f"{a},{b}".encode())).standard_normal() * self.quirk
        return 1 / (1 + math.exp(-(self.theta[a] - self.theta[b] + self.gamma + e)))

    async def ask(self, state, questions, **_):
        answer = {"q": {"type": "noul", "noul": self.p(int(state["A"]), int(state["B"]))}}
        # a saved scale names the model that answered, so the judge has to have a name
        return Answers(answer, {"q": {"resolved_model": "simulated-judge", "source": "api"}})


def spearman(a, b) -> float:
    rank = lambda v: np.argsort(np.argsort(v))
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def target(judge: SimulatedJev, n: int) -> np.ndarray:
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    first, second = (np.array(c) for c in zip(*pairs))
    theta = fit(n, first, second, [judge.p(a, b) for a, b in pairs]).theta
    return theta - theta.mean()


async def main() -> None:
    n = 300
    texts = [str(i) for i in range(n)]
    print(f"{n} texts; judge quirk sd 0.8 logits, first-position lean built in\n")
    print(f"{'per item':>8} {'asked':>6} {'spearman':>9} {'top-10 hit':>10} {'cover 95%':>9} "
          f"{'reliab.':>8} {'true r^2':>8} {'lean':>6}")
    for per_item in (4, 6, 8, 10, 14, 20, 30):
        rows = []
        for rep in range(5):
            theta = np.random.default_rng(100 + rep).normal(0, 1.5, n)
            judge = SimulatedJev(theta)
            goal = target(judge, n)
            r = await arank(texts, "x", judge, per_item=per_item, seed=rep)
            cover = float(np.mean(np.abs(r.score - goal) <= 1.96 * r.se))
            hit = len(set(np.argsort(-r.score)[:10]) & set(np.argsort(-goal)[:10])) / 10
            rows.append((r.asked, spearman(r.score, goal), hit, cover, r.reliability,
                         float(np.corrcoef(r.score, goal)[0, 1]) ** 2, r.lean))
        m = np.mean(np.array(rows, dtype=float), axis=0)
        print(f"{per_item:>8} {m[0]:>6.0f} {m[1]:>9.3f} {m[2]:>10.2f} {m[3]:>9.2f} {m[4]:>8.3f} {m[5]:>8.3f} {m[6]:>+6.3f}")

    # A larger set, where --top has something to retire. Fitting every ordered pair of 2,000 texts is out
    # of reach, so here the target is the judge's own underlying scale.
    n, reps = 2000, 4
    texts = [str(i) for i in range(n)]
    print(f"\n--top 10 against a full sort, {n} texts at 10 per item")
    for label, top in (("full", None), ("top 10", 10)):
        rows = []
        for rep in range(reps):
            theta = np.random.default_rng(100 + rep).normal(0, 1.5, n)
            r = await arank(texts, "x", SimulatedJev(theta), per_item=10, top=top, seed=rep)
            found = set(np.argsort(-np.nan_to_num(r.score, nan=-99))[:10])
            rows.append((r.asked, len(found & set(np.argsort(-theta)[:10])) / 10))
        m = np.mean(np.array(rows, dtype=float), axis=0)
        print(f"{label:>8}: {m[0]:,.0f} questions, {m[1]:.2f} of the true top ten found")

    # Placing on a saved scale. 200 texts are sorted at the default and the scale is saved; 100 the fit
    # never saw are then placed against its anchors only. The target is the score a fit to every ordered
    # pair of all 300 would give them, on the saved scale's zero.
    base, held, reps = 200, 100, 8
    texts = [str(i) for i in range(base + held)]
    print(f"\nplacing {held} new texts on a scale saved from {base} (-k 10)\n")
    print(f"{'anchors':>8} {'per item':>8} {'asked':>6} {'pearson':>8} {'rmse':>6} {'cover 95%':>9} {'beyond':>7}   the fit's own texts")
    for anchors, per_item in ((15, 10), (30, 6), (30, 10), (30, 16), (60, 10)):
        rows = []
        for rep in range(reps):
            theta = np.random.default_rng(100 + rep).normal(0, 1.5, base + held)
            judge = SimulatedJev(theta)
            goal = target(judge, base + held)
            goal -= goal[:base].mean()
            r = await arank(texts[:base], "x", judge, per_item=10, seed=rep)
            p = await aplace(texts[base:], r.scale(anchors), judge, per_item=per_item, seed=rep, any_model=True)
            miss, fitted_miss = p.score - goal[base:], r.score - goal[:base]
            rows.append((p.asked / held, float(np.corrcoef(p.score, goal[base:])[0, 1]), float(np.sqrt(np.mean(miss ** 2))),
                         float(np.mean(np.abs(miss) <= 1.96 * p.se)), float(np.mean(p.beyond != 0)),
                         float(np.sqrt(np.mean(fitted_miss ** 2))), float(np.mean(np.abs(fitted_miss) <= 1.96 * r.se))))
        m = np.mean(np.array(rows, dtype=float), axis=0)
        print(f"{anchors:>8} {per_item:>8} {m[0]:>6.1f} {m[1]:>8.3f} {m[2]:>6.3f} {m[3]:>9.2f} {m[4]:>7.2f}   rmse {m[5]:.3f}, cover {m[6]:.2f}")

asyncio.run(main())
