"""Which pairs get asked. No network."""

import numpy as np

from jsort.schedule import Schedule


def test_ring_meets_every_text_twice_once_in_each_position():
    s = Schedule(9, seed=1)
    pairs = s.ring(limit=100)
    assert len(pairs) == 9
    assert sorted(a for a, _ in pairs) == list(range(9)) == sorted(b for _, b in pairs)
    assert np.all(s.count == 2) and np.all(s.as_first == 1)


def test_ring_of_two_asks_both_orders_and_of_one_asks_nothing():
    assert sorted(Schedule(2).ring(10)) == [(0, 1), (1, 0)]
    assert Schedule(1).ring(10) == []
    assert len(Schedule(9).ring(limit=4)) == 4


def test_neighbours_pairs_adjacent_texts_and_never_repeats_an_ordered_pair():
    n = 40
    s = Schedule(n, seed=0)
    theta = np.arange(n, dtype=float)
    seen = set(s.ring(1000))
    for _ in range(12):
        pairs = s.neighbours(theta, np.full(n, 0.01), np.ones(n, bool), limit=1000, per_item=1000)
        assert pairs and not (set(pairs) & seen)
        assert len({i for p in pairs for i in p}) == 2 * len(pairs)      # one comparison per text per round
        assert max(abs(a - b) for a, b in pairs) <= 12
        seen |= set(pairs)


def test_neighbours_respects_the_limit_the_cap_and_the_active_set():
    n = 20
    theta = np.arange(n, dtype=float)
    s = Schedule(n, seed=0)
    assert len(s.neighbours(theta, np.zeros(n), np.ones(n, bool), limit=3, per_item=99)) == 3
    s = Schedule(n, seed=0)
    active = np.zeros(n, bool)
    active[:6] = True
    pairs = s.neighbours(theta, np.zeros(n), active, limit=99, per_item=99)
    assert pairs and all(a < 6 and b < 6 for a, b in pairs)
    s = Schedule(n, seed=0)
    s.count[:] = 5
    assert s.neighbours(theta, np.zeros(n), np.ones(n, bool), limit=99, per_item=5) == []


def test_same_seed_same_questions():
    def questions(seed):
        s = Schedule(30, seed=seed)
        theta = np.linspace(-1, 1, 30)
        return s.ring(99) + s.neighbours(theta, np.full(30, 0.3), np.ones(30, bool), 99, 99)
    assert questions(7) == questions(7)
    assert questions(7) != questions(8)


def test_a_small_set_runs_out_of_pairs():
    s = Schedule(3, seed=0)
    asked = s.ring(99)
    for _ in range(10):
        asked += s.neighbours(np.zeros(3), np.ones(3), np.ones(3, bool), 99, 99)
    assert sorted(asked) == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
