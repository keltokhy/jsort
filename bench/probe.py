"""Which question shape should a comparison use? Live; costs well under a cent.

Twelve sentences with a known order of urgency, every ordered pair asked three ways:
a choice between A and B, a noul that A ranks higher, and a noul with the texts in a plain string.
Reports accuracy on the hard decision, how far p(A,B) + p(B,A) strays from 1, and the first-position lean.
"""

import asyncio
import itertools
import statistics

from jsort.core import Jev, resolve_backend

ITEMS = [  # least to most urgent
    "No rush at all, just curious whether dark mode is planned someday.",
    "When you get a chance, could you update my billing address?",
    "The export button is a little slow, not a big deal.",
    "I'd like to change my plan before the next billing cycle.",
    "The dashboard has been showing stale numbers since this morning.",
    "I can't log in and I have a report due this afternoon.",
    "Our whole team has been locked out for two hours, please help.",
    "Checkout is failing for every customer and we are losing sales right now.",
    "Production is down, customers are churning, we need someone on this immediately.",
    "Patient records are inaccessible in the ER, this is a life-safety emergency.",
]
DESC = "more urgent"

SHAPES = {
    "choice/object": (lambda a, b: {"A": a, "B": b},
                      {"type": "choice", "instructions": f'Which text ranks higher on this criterion: "{DESC}"?',
                       "criteria": {"A": "Text A ranks higher.", "B": "Text B ranks higher."}}),
    "noul/object": (lambda a, b: {"A": a, "B": b},
                    {"type": "noul", "instructions": f'Text A ranks higher than text B on this criterion: "{DESC}"'}),
    "noul/string": (lambda a, b: f"Text A:\n{a}\n\nText B:\n{b}",
                    {"type": "noul", "instructions": f'Text A ranks higher than text B on this criterion: "{DESC}"'}),
}


def p_first(answer: dict) -> float:
    if "noul" in answer:
        return float(answer["noul"])
    return float(answer["probabilities"]["A"])


async def main() -> None:
    backend, key = resolve_backend()
    jev = Jev(key, backend, cache=None)
    pairs = list(itertools.permutations(range(len(ITEMS)), 2))
    sem = asyncio.Semaphore(16)

    async def one(shape, i, j):
        build, q = SHAPES[shape]
        async with sem:
            ans = await jev.ask(build(ITEMS[i], ITEMS[j]), {"q": q})
        return shape, i, j, ans["q"]

    results = await asyncio.gather(*(one(s, i, j) for s in SHAPES for i, j in pairs))
    await jev.close()
    print("sample choice answer:", next(a for s, *_, a in results if s == "choice/object"))
    for shape in SHAPES:
        p = {(i, j): p_first(a) for s, i, j, a in results if s == shape}
        acc = statistics.mean((p[i, j] > 0.5) == (i > j) for i, j in pairs)
        gap = statistics.mean(abs(p[i, j] + p[j, i] - 1) for i, j in pairs if i < j)
        lean = statistics.mean(p.values()) - 0.5
        adjacent = statistics.mean((p[i, j] > 0.5) == (i > j) for i, j in pairs if abs(i - j) == 1)
        mid = statistics.mean(0.1 < v < 0.9 for v in p.values())
        print(f"{shape:14s} accuracy {acc:.3f}  adjacent-pair accuracy {adjacent:.3f}  "
              f"|p(a,b)+p(b,a)-1| {gap:.3f}  first-position lean {lean:+.3f}  share of p in (0.1,0.9) {mid:.2f}")
    print(jev.meter.summary())


asyncio.run(main())
