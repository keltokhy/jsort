"""Which pairs to ask next.

All n(n-1)/2 pairs are never needed. A lopsided pair says almost nothing, because the answer was
predictable; a pair of near-neighbours on the current scale says the most. So the first round is a
random ring, which connects every text to two others and shows each one once in either position, and
every later round pairs texts that currently sit close together, as a Swiss-system tournament does.

Closeness is judged on a jittered copy of the scale, each score perturbed in proportion to how
uncertain it still is, so a text the comparisons have not pinned down yet keeps meeting new
neighbours. A pair can be asked twice at most, once in each order. Everything is drawn from one seeded
generator: the same input asks the same questions, and a rerun is answered from the cache.
"""

from __future__ import annotations

import numpy as np

WINDOW = 12  # how far down the ordering to look for a partner that has not been met yet


class Schedule:
    def __init__(self, n: int, seed: int = 0):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.asked: set[tuple[int, int]] = set()
        self.as_first = np.zeros(n, dtype=int)
        self.count = np.zeros(n, dtype=int)

    def _orient(self, i: int, j: int) -> tuple[int, int] | None:
        """Put the text that has led less often first, unless that order has been asked already."""
        options = [pair for pair in ((i, j), (j, i)) if pair not in self.asked]
        if not options:
            return None
        if len(options) == 2:
            lead_i, lead_j = self.as_first[i], self.as_first[j]
            if lead_i == lead_j:
                return options[int(self.rng.integers(2))]
            return (i, j) if lead_i < lead_j else (j, i)
        return options[0]

    def _take(self, pair: tuple[int, int]) -> tuple[int, int]:
        self.asked.add(pair)
        self.as_first[pair[0]] += 1
        self.count[pair[0]] += 1
        self.count[pair[1]] += 1
        return pair

    def ring(self, limit: int) -> list[tuple[int, int]]:
        """A random cycle through every text. With two texts this is the pair in both orders."""
        if self.n < 2:
            return []
        order = self.rng.permutation(self.n)
        pairs = []
        for t in range(self.n):
            pair = (int(order[t % self.n]), int(order[(t + 1) % self.n]))
            if len(pairs) < limit and pair not in self.asked:
                pairs.append(self._take(pair))
        return pairs

    def neighbours(self, theta: np.ndarray, spread: np.ndarray, active: np.ndarray, limit: int,
                   per_item: int) -> list[tuple[int, int]]:
        """One round: every active text that still needs comparisons meets the nearest text it has not met."""
        jittered = np.nan_to_num(theta) + self.rng.standard_normal(self.n) * spread
        order = [int(i) for i in np.argsort(-jittered, kind="stable") if active[i]]
        matched: set[int] = set()
        pairs = []
        for pos, i in enumerate(order):
            if len(pairs) >= limit:
                break
            if i in matched or self.count[i] >= per_item:
                continue
            for j in order[pos + 1: pos + 1 + WINDOW]:
                if j in matched:
                    continue
                pair = self._orient(i, j)
                if pair is not None:
                    matched.update((i, j))
                    pairs.append(self._take(pair))
                    break
        return pairs
