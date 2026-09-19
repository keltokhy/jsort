"""Does jsort's scale agree with people's? Live; about ten cents.

The CommonLit Ease of Readability corpus (Crossley et al. 2022, CC BY-NC-SA 4.0) has 4,724 excerpts,
each with an easiness score that is itself a Bradley-Terry fit, to teachers' pairwise judgments of
which of two excerpts is easier. That makes it the natural yardstick: the same method, with Jev in the
teachers' chair. The corpus is downloaded on first use and is not redistributed here.

    uv run --group bench python bench/readability.py prepare     # download and draw the sample
    uv run --group bench python bench/readability.py run         # jsort at several -k, against the alternatives

Compared on the same sample:
  jsort at several -k                      the installed command, uncached on its first pass
  one question per text                    the probability that a text is "easy to read", which is `jgrep -o | sort`
  a five-level rubric, one call per text   Jev's score primitive, as the level chosen and as the expected level
  readability formulas                     Flesch-Kincaid and the rest, as shipped with the corpus
"""

import asyncio
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from jsort.core import Cache, Jev, resolve_backend

OUT = Path(__file__).parent / "out"
SOURCE = "https://raw.githubusercontent.com/scrosseye/CLEAR-Corpus/main/CLEAR_corpus_final.xlsx"
SAMPLE, SEED = 300, 20260919
DESCRIPTION = "easier to read"
FORMULAS = ["Flesch-Reading-Ease", "Flesch-Kincaid-Grade-Level", "Automated Readability Index", "SMOG Readability",
            "New Dale-Chall Readability Formula", "CAREC", "CML2RI"]
RUBRIC = {"type": "score", "instructions": "How easy is this text to read?",
          "criteria": ["Very hard: suited to readers at the end of high school or beyond.",
                       "Hard: suited to high school readers.",
                       "Moderate: suited to middle school readers.",
                       "Easy: suited to upper elementary school readers.",
                       "Very easy: suited to children in the first years of school."]}
DIRECT = {"type": "noul", "instructions": 'The text fits this description: "easy to read"'}


def spearman(a, b) -> float:
    def ranks(v):
        v = np.asarray(v, dtype=float)
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for value in np.unique(v):                 # average the ranks of ties
            tie = v == value
            r[tie] = r[tie].mean()
        return r
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def prepare() -> None:
    import openpyxl

    OUT.mkdir(exist_ok=True)
    book = OUT / "CLEAR_corpus_final.xlsx"
    if not book.exists():
        urllib.request.urlretrieve(SOURCE, book)
    rows = openpyxl.load_workbook(book, read_only=True)["Data"].iter_rows(values_only=True)
    header = next(rows)
    table = [dict(zip(header, r)) for r in rows if r[header.index("Excerpt")] and r[header.index("BT_easiness")] is not None]
    pick = np.random.default_rng(SEED).choice(len(table), SAMPLE, replace=False)
    with open(OUT / "clear.jsonl", "w") as f:
        for i in sorted(pick):
            r = table[i]
            rec = {"id": r["ID"], "excerpt": " ".join(str(r["Excerpt"]).split()), "easiness": float(r["BT_easiness"]),
                   "easiness_se": float(r["s.e."])} | {k: float(r[k]) for k in FORMULAS}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {SAMPLE} of {len(table):,} excerpts to {OUT / 'clear.jsonl'}")


async def one_call_per_text(texts: list[str]) -> dict:
    backend, key = resolve_backend()
    jev = Jev(key, backend, cache=Cache())
    sem = asyncio.Semaphore(32)

    async def ask(t):
        async with sem:
            return await jev.ask(t, {"direct": DIRECT, "rubric": RUBRIC})

    t0 = time.perf_counter()
    answers = await asyncio.gather(*(ask(t) for t in texts))
    await jev.close()
    levels = np.arange(len(RUBRIC["criteria"]))
    expected = []
    for a in answers:
        p = a["rubric"].get("probabilities") or {}
        ps = np.array([float(v) for v in p.values()]) if len(p) == len(levels) else None
        expected.append(float(ps @ levels / ps.sum()) if ps is not None and ps.sum() > 0 else float(a["rubric"]["score"]))
    return {"direct": [float(a["direct"]["noul"]) for a in answers],
            "rubric_level": [float(a["rubric"]["score"]) for a in answers], "rubric_expected": expected,
            "seconds": time.perf_counter() - t0, "cost": jev.meter.cost, "calls": jev.meter.calls,
            "sample_rubric_answer": answers[0]["rubric"]}


def run() -> None:
    data = [json.loads(l) for l in open(OUT / "clear.jsonl")]
    truth = np.array([d["easiness"] for d in data])
    ceiling = 1 - float(np.mean([d["easiness_se"] ** 2 for d in data])) / float(truth.var())
    print(f"{len(data)} excerpts; the teachers' scale has a reliability of about {ceiling:.2f}, so no measure can be "
          f"expected to correlate with it above about {ceiling ** 0.5:.2f}\n")
    results = {"n": len(data), "ceiling": ceiling ** 0.5, "rows": []}

    def row(label, values, **extra):
        r, rho = float(np.corrcoef(values, truth)[0, 1]), spearman(values, truth)
        results["rows"].append({"measure": label, "pearson": r, "spearman": rho, "distinct": len(set(values))} | extra)
        note = "  ".join(f"{k} {v}" for k, v in extra.items())
        print(f"{label:44s} r {r:+.3f}  rho {rho:+.3f}  distinct values {len(set(values)):4d}  {note}")

    for k in (16, 10, 6, 4):   # the largest first: the smaller runs ask a prefix of its questions and come from the cache
        t0 = time.perf_counter()
        done = subprocess.run(["jsort", DESCRIPTION, str(OUT / "clear.jsonl"), "--jsonl", "--field", "excerpt", "-o",
                               "--keep-order", "-k", str(k), "--stats", "--budget", "0"],
                              capture_output=True, text=True)
        if done.returncode:
            sys.exit(done.stderr)
        scored = [json.loads(l) for l in done.stdout.splitlines()]
        stats = done.stderr.strip().splitlines()[-1]
        row(f'jsort -k {k} "{DESCRIPTION}"', [s["jsort_score"] for s in scored], seconds=round(time.perf_counter() - t0, 1))
        print(f"    {stats}")
        results["rows"][-1]["stats"] = stats
        if k == 10:
            measures = {"jsort -k 10": np.array([s["jsort_score"] for s in scored])}
            results["median_se"] = float(np.median([s["jsort_se"] for s in scored]))

    single = asyncio.run(one_call_per_text([d["excerpt"] for d in data]))
    print(f"\none call per text, both questions in it: {single['calls']} calls, ${single['cost']:.4f}, {single['seconds']:.1f}s")
    print(f"    a rubric answer looks like {json.dumps(single['sample_rubric_answer'])[:200]}")
    row('one question: p("easy to read")', single["direct"])
    row("five-level rubric: level chosen", single["rubric_level"])
    row("five-level rubric: expected level", single["rubric_expected"])
    measures |= {"rubric": np.array(single["rubric_level"]), 'p("easy")': np.array(single["direct"])}

    # Where should comparisons help most? On texts that are close. Agreement with the teachers on which
    # of two excerpts is easier, by how far apart the teachers put them.
    print("\nagreement with the teachers on random pairs, by the gap between the two on the teachers' scale")
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(data), 40000), rng.integers(0, len(data), 40000)
    gap = np.abs(truth[i] - truth[j])
    results["pairs"] = []
    for lo, hi in ((0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99.0)):
        m = (gap >= lo) & (gap < hi)
        agree = {name: float(np.mean(np.sign(v[i[m]] - v[j[m]]) == np.sign(truth[i[m]] - truth[j[m]])))
                 for name, v in measures.items()}
        results["pairs"].append({"gap": [lo, hi], "pairs": int(m.sum())} | agree)
        print(f"  gap {lo:4.2f} to {hi:5.2f} ({int(m.sum()):6,d} pairs)  " + "  ".join(f"{k} {v:.3f}" for k, v in agree.items()))
    print()
    for name in FORMULAS:
        values = [d[name] for d in data]
        sign = np.sign(np.corrcoef(values, truth)[0, 1])  # grade-level formulas run the other way; report the size
        row(f"formula: {name}", list(sign * np.array(values)))
    (OUT / "readability.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    {"prepare": prepare, "run": run}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
