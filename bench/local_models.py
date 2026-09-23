"""DiffusionGemma and Laya, on this machine, against the Jev runs already on file: YC one-liners and the Fed.

Nothing here calls a hosted model. Every sort runs the installed `jsort` with `--api diffusiongemma` or
`--api laya` named on the command line and in JEV_API, an empty answer cache, an empty config directory
(so no hosted key can be found even by mistake) and `--budget 0`. The Jev column is read from files:
bench/out/yc/yc-scifi.csv, bench/out/fed/fed.json, and for the sentences of the latest statement a
replay of jsort 0.1.2 over the answers it cached on 2026-09-19, which reproduces run.log to the digit.

    uv run python bench/local_models.py prepare                       # the YC sample and Jev's sentence scores; no model calls
    uv run python bench/local_models.py run --model diffusiongemma --bench fed-whole --statements 5 --k 2 --tag probe
    uv run python bench/local_models.py run --model diffusiongemma --bench yc
    uv run python bench/local_models.py run --model diffusiongemma --bench fed-latest
    uv run python bench/local_models.py run --model diffusiongemma --bench fed-whole [--statements N --seed S]
    uv run python bench/local_models.py run --model laya --bench yc          # likewise fed-latest, fed-whole
    uv run python bench/local_models.py freeze                        # docs/benchmarks/local-models-2026-09-22.json

Benchmarks, all with jsort's defaults unless named (seed 0, -k 10), as the Jev runs had them:
  yc          "sounds more like science fiction" on 500 YC one-liners drawn from yc.csv with seed 20260922,
              the `text` field Jev saw; against Jev's scores for the same companies from its 6,081-company sort
  fed-latest  "more hawkish about inflation" on the 51 sentences of latest.txt, fed.py's first sort
  fed-whole   the same description on the 95 opening statements, --whole --max-chars 16000, fed.py's second
              sort; scored as fed.py scores it, and against Jev's statement scores from fed.json

Each run writes <model>-<bench>[-<tag>][-tryN].{json,log,stdout,stderr} under bench/out/local-2026-09-22/ and
uses cache-<same stem>/ there as XDG_CACHE_HOME; a stem whose cache directory exists is never reused, so a
second attempt gets -try2. Laya's refusals are counted from its audit log (the server's --audit file, named by
the LAYA_AUDIT environment variable), by the lines it gained during the run.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from fed import DESCRIPTION as HAWKISH, OUT as FED, checks, fedlock, spearman

BENCH = Path(__file__).parent
YC = BENCH / "out" / "yc"
WORK = BENCH / "out" / "local-2026-09-22"
FROZEN = BENCH.parent / "docs" / "benchmarks" / "local-models-2026-09-22.json"
SCIFI = "sounds more like science fiction"
YC_SAMPLE, YC_SEED = 500, 20260922
NY = ZoneInfo("America/New_York")
LAYA_AUDIT = Path(os.environ["LAYA_AUDIT"]) if os.environ.get("LAYA_AUDIT") else None  # the Laya server's --audit file
MODELS = {
    "diffusiongemma": {"api": "diffusiongemma", "model": "openjev-0.1", "url_env": "JEV_DIFFUSIONGEMMA_URL",
                       "url": "http://127.0.0.1:8080/v1/systemone", "health": "http://127.0.0.1:8080/health",
                       "jobs": 4, "timeout": 300},
    "laya": {"api": "laya", "model": "laya-421m", "url_env": "JEV_LAYA_URL",
             "url": "http://127.0.0.1:8081/v1/systemone", "health": "http://127.0.0.1:8081/health",
             "jobs": 4, "timeout": 120},
}
BENCHMARKS = ("yc", "fed-latest", "fed-whole")
JEV_012 = "2897d18"     # the jsort commit that made fed.json and run.log on 2026-09-19
JEV_012_MODEL = "~typesafe/jev-latest"


def now() -> str:
    return datetime.now(NY).isoformat(timespec="seconds")


def pearson(a, b) -> float:
    return float(np.corrcoef(np.asarray(a, dtype=float), np.asarray(b, dtype=float))[0, 1])


def correlations(ours: dict, theirs: dict) -> dict:
    keys = [k for k in ours if ours[k] is not None and theirs.get(k) is not None]
    if len(keys) < 3:
        return {"n": len(keys), "spearman": None, "pearson": None}
    a, b = [ours[k] for k in keys], [theirs[k] for k in keys]
    return {"n": len(keys), "spearman": spearman(a, b), "pearson": pearson(a, b)}


def top(scores: dict, n: int) -> list:
    return sorted((k for k in scores if scores[k] is not None), key=lambda k: -scores[k])[:n]


# ---- inputs and the Jev column; no model calls ----

def yc_sample() -> list[dict]:
    rows = list(csv.DictReader(open(YC / "yc.csv", newline="")))
    pick = sorted(np.random.default_rng(YC_SEED).choice(len(rows), YC_SAMPLE, replace=False).tolist())
    return [rows[i] | {"row": i} for i in pick]


def jev_yc() -> dict:
    """Jev's score for every company, by the text it was shown, from the 6,081-company sort of 2026-09-20."""
    return {r["text"]: float(r["scifi_score"]) for r in csv.DictReader(open(YC / "yc-scifi.csv", newline=""))
            if r["scifi_score"]}


def jev_whole() -> dict:
    """Jev's statement scores and fed.py's outcomes, by date, from fed.json."""
    table = json.loads((FED / "fed.json").read_text())["table"]
    return {d: {"chair": c, "score": s, "se": se, "move": m, "ahead": a} for d, c, s, se, m, a in table}


REPLAY = r'''
import asyncio, hashlib, json, sqlite3, sys
from pathlib import Path
import jsort.engine as engine
assert engine.__file__.startswith(sys.argv[1]), engine.__file__
db = sqlite3.connect("file:" + sys.argv[2] + "?mode=ro&immutable=1", uri=True)
class Meter:
    cost = 0.0
class CacheOnly:
    """Answers from the 2026-09-19 cache, keyed as jsort 0.1.2 keyed them. A miss is an error, never a call."""
    meter = Meter()
    async def ask(self, state, questions):
        out = {}
        for qid, q in questions.items():
            key = hashlib.sha256(json.dumps([sys.argv[3], state, q], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            row = db.execute("SELECT answer FROM answers WHERE key = ?", (key,)).fetchone()
            if row is None:
                raise SystemExit("cache miss: the replay needs every answer the 2026-09-19 run had")
            out[qid] = json.loads(row[0])
        return out
lines = [l for l in Path(sys.argv[4]).read_text().splitlines() if l.strip()]
r = asyncio.run(engine.arank(lines, sys.argv[5], CacheOnly()))
print(json.dumps({"asked": r.asked, "rounds": r.rounds, "reliability": r.reliability, "lean": r.lean,
                  "rows": [{"line": i + 1, "text": t, "score": float(r.score[i]), "se": float(r.se[i])}
                           for i, t in enumerate(lines)]}))
'''


def replay_jev_latest() -> dict:
    """Jev's scores for all 51 sentences. run.log prints ten; the rest come from replaying the 0.1.2 sort
    over the answers it cached, read-only, with a client that fails on a miss rather than call anything."""
    cache = Path.home() / ".cache" / "jev" / "answers.sqlite"
    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.run(["git", "-C", str(BENCH.parent), "archive", JEV_012, "src/jsort"],
                                 capture_output=True, check=True).stdout
        tarfile.open(fileobj=io.BytesIO(archive)).extractall(tmp, filter="data")
        src = str(Path(tmp) / "src")
        env = {k: v for k, v in os.environ.items() if not k.startswith(("JEV_", "OPENROUTER", "TYPESAFE"))}
        done = subprocess.run([sys.executable, "-c", REPLAY, src, str(cache), JEV_012_MODEL, str(FED / "latest.txt"),
                               HAWKISH], capture_output=True, text=True, env=env | {"PYTHONPATH": src})
    if done.returncode:
        sys.exit(done.stderr)
    out = json.loads(done.stdout)
    # the check: every score run.log printed, to two places
    printed = [l.split("\t") for l in (FED / "run.log").read_text().splitlines()[1:12] if l.count("\t") == 2]
    ours = {r["text"]: (f"{r['score']:.2f}", f"{r['se']:.2f}") for r in out["rows"]}
    mismatched = [p for p in printed if ours.get(p[2]) != (p[0], p[1])]
    out |= {"source": f"jsort {JEV_012} replayed over ~/.cache/jev/answers.sqlite (read-only), model {JEV_012_MODEL}",
            "checked_against_run_log": len(printed), "mismatched": mismatched}
    return out


def prepare() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    sample = yc_sample()
    path = WORK / "yc-sample500.csv"
    body = io.StringIO()
    w = csv.DictWriter(body, ["name", "batch", "status", "text"], extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(sample)
    if path.exists() and path.read_text() != body.getvalue():
        sys.exit(f"{path} exists and differs from the seeded draw; leave it and investigate")
    path.write_text(body.getvalue())
    jev = jev_yc()
    missing = [r["text"] for r in sample if r["text"] not in jev]
    (WORK / "yc-sample500.json").write_text(json.dumps(
        {"source": str(YC / "yc.csv"), "seed": YC_SEED, "n": len(sample), "rows": [r["row"] for r in sample],
         "draw": "numpy.random.default_rng(seed).choice(6081, 500, replace=False), sorted, 0-based data rows",
         "missing_jev_score": missing}, indent=1))
    print(f"{path}: {len(sample)} companies, seed {YC_SEED}; {len(sample) - len(missing)} have a Jev score")
    latest = replay_jev_latest()
    (WORK / "jev-latest.json").write_text(json.dumps(latest, indent=1, ensure_ascii=False))
    print(f"{WORK / 'jev-latest.json'}: {latest['asked']} comparisons, reliability {latest['reliability']:.2f}; "
          f"{latest['checked_against_run_log']} scores checked against run.log, {len(latest['mismatched'])} mismatched")


# ---- one local run ----

def environment(model: str, stem: str, outdir: Path) -> dict:
    """The child's environment: no hosted keys, no config files, one named local API, an empty cache."""
    cfg = MODELS[model]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("JEV_", "OPENROUTER", "TYPESAFE", "JSORT_"))}
    return env | {"JEV_API": cfg["api"], "JEV_MODEL": cfg["model"], cfg["url_env"]: cfg["url"],
                  "XDG_CACHE_HOME": str(outdir / f"cache-{stem}"), "XDG_CONFIG_HOME": str(outdir / f"config-{stem}")}


def audit_count() -> int:
    return sum(1 for _ in open(LAYA_AUDIT)) if LAYA_AUDIT and LAYA_AUDIT.exists() else 0


def audit_since(before: int) -> list[dict]:
    if LAYA_AUDIT is None or not LAYA_AUDIT.exists():
        return []
    with open(LAYA_AUDIT) as f:
        return [json.loads(line) for i, line in enumerate(f) if i >= before and line.strip()]


def provenance(cache: Path) -> dict:
    """Who answered, from the metadata the runtime stores with each answer in the run's own cache."""
    db_path = cache / "jev" / "answers.sqlite"
    if not db_path.exists():
        return {"answers": 0, "origins": {}}
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    origins: dict = {}
    for (meta,) in db.execute("SELECT metadata FROM answers"):
        m = json.loads(meta) if meta else {}
        k = f"{m.get('provider')}|{m.get('requested_model')}|{m.get('resolved_model')}"
        origins[k] = origins.get(k, 0) + 1
    db.close()
    return {"answers": sum(origins.values()), "origins": origins}


def parse_stats(stderr: str) -> dict:
    lines = stderr.splitlines()
    stats = next((l for l in reversed(lines) if "comparisons in" in l), "")
    num = lambda pattern: int(m.group(1).replace(",", "")) if (m := re.search(pattern, stats)) else None
    flt = lambda pattern: float(m.group(1)) if (m := re.search(pattern, stats)) else None
    failed = sum(1 for l in lines if l.startswith("jsort: a comparison failed:"))
    failed += sum(int(m.group(1).replace(",", "")) for l in lines if (m := re.search(r"and ([\d,]+) more failed", l)))
    return {"stats": stats, "texts": num(r"([\d,]+) texts"), "comparisons": num(r"([\d,]+) comparisons in"),
            "rounds": num(r"in (\d+) rounds"), "reliability": flt(r"reliability ([\d.-]+)"),
            "lean": flt(r"lean ([+\-\d.]+)"), "calls": num(r"([\d,]+) calls"), "cached": num(r"([\d,]+) cached"),
            "retries": num(r"([\d,]+) retries"), "tokens": num(r"([\d,]+) tokens"), "cost": flt(r"\$([\d.]+)"),
            "jsort_seconds": flt(r"; ([\d.]+)s$"), "failed_comparisons": failed,
            "notes": [l for l in lines if l.startswith("jsort:") and l != stats and "comparison failed" not in l
                      and "more failed comparisons" not in l][:10],
            "sample_errors": [l for l in lines if l.startswith("jsort: a comparison failed:")][:3]}


def health(model: str) -> str:
    with urllib.request.urlopen(MODELS[model]["health"], timeout=10) as r:
        return r.read().decode()


def inputs(bench: str, a, run_dir: Path, stem: str) -> tuple[list[str], dict]:
    """The jsort arguments after the description, and what was sorted."""
    if bench == "yc":
        sample = WORK / "yc-sample500.csv"
        if not sample.exists():
            sys.exit(f"{sample} is missing: run `prepare` first")
        path = sample
        if a.limit:
            rows = sample.read_text().splitlines()
            path = run_dir / f"{stem}-input.csv"
            path.write_text("\n".join(rows[:a.limit + 1]) + "\n")
        return [str(path), "--csv", "--field", "text", "--json", "--keep-order"], {"file": str(path)}
    if bench == "fed-latest":
        path = FED / "latest.txt"
        if a.limit:
            lines = [l for l in path.read_text().splitlines() if l.strip()]
            path = run_dir / f"{stem}-input.txt"
            path.write_text("\n".join(lines[:a.limit]) + "\n")
        return [str(path), "--json", "--keep-order"], {"file": str(path)}
    files = sorted(FED.glob("statements/*.txt"))
    chosen = {"of": len(files)}
    if a.statements and a.statements < len(files):
        pick = sorted(np.random.default_rng(a.seed).choice(len(files), a.statements, replace=False).tolist())
        files = [files[i] for i in pick]
        chosen |= {"statements": a.statements, "seed": a.seed,
                   "draw": "numpy.random.default_rng(seed).choice(95, statements, replace=False), sorted by date"}
    chosen["dates"] = [f.stem for f in files]
    return ["--whole", "--json", "--keep-order", "--max-chars", "16000", *map(str, files)], chosen


def score(bench: str, rows: list[dict], result: dict) -> dict:
    if bench == "yc":
        jev = jev_yc()
        ours = {r["record"]["text"]: r["score"] for r in rows}
        theirs = {t: jev.get(t) for t in ours}
        top20 = top(ours, 20)
        jev20 = top(theirs, 20)
        return {"vs_jev": correlations(ours, theirs), "top20_overlap": len(set(top20) & set(jev20)),
                "top20": top20, "jev_top20": jev20, "scores": ours}
    if bench == "fed-latest":
        jev = {r["text"]: r["score"] for r in json.loads((WORK / "jev-latest.json").read_text())["rows"]}
        ours = {r["text"]: r["score"] for r in rows}
        theirs = {t: jev.get(t) for t in ours}
        return {"vs_jev": correlations(ours, theirs), "top10_overlap": len(set(top(ours, 10)) & set(top(theirs, 10))),
                "top6": top(ours, 6), "jev_top6": top(theirs, 6), "scores": ours}
    jev = jev_whole()
    ours = {Path(r["file"]).stem: r["score"] for r in rows}
    out = {"vs_jev": correlations(ours, {d: jev[d]["score"] for d in ours}), "scores": ours}
    for name, field in (("same_day", "move"), ("next_180_days", "ahead")):
        both = [d for d in ours if jev[d][field] is not None]
        scored = [d for d in both if ours[d] is not None]
        mine = spearman([ours[d] for d in scored], [jev[d][field] for d in scored]) if len(scored) > 2 else None
        # Jev on exactly the statements this model scored, and on every statement in this run's subset
        jev_same = spearman([jev[d]["score"] for d in scored], [jev[d][field] for d in scored]) if len(scored) > 2 else None
        jev_all = spearman([jev[d]["score"] for d in both], [jev[d][field] for d in both]) if len(both) > 2 else None
        out[name] = {"n": len(scored), "of": len(both), "spearman": mine, "jev_same_statements": jev_same,
                     "jev_all_statements_in_run": jev_all}
    return out


def run(a) -> None:
    cfg = MODELS[a.model]
    outdir = a.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    base = f"{a.model}-{a.bench}" + (f"-{a.tag}" if a.tag else "")
    attempt = 1
    while (outdir / f"cache-{base}{'' if attempt == 1 else f'-try{attempt}'}").exists() or \
            (outdir / f"{base}{'' if attempt == 1 else f'-try{attempt}'}.json").exists():
        attempt += 1
    stem = base + ("" if attempt == 1 else f"-try{attempt}")
    env = environment(a.model, stem, outdir)
    Path(env["XDG_CONFIG_HOME"]).mkdir()
    args, chosen = inputs(a.bench, a, outdir, stem)
    description = SCIFI if a.bench == "yc" else HAWKISH
    jobs, timeout = a.jobs or cfg["jobs"], a.timeout or cfg["timeout"]
    cmd = [str(Path(sys.executable).parent / "jsort"), description, *args, "-k", str(a.k), "--stats",
           "--budget", "0", "-j", str(jobs), "--timeout", str(timeout), "--api", cfg["api"], "--model", cfg["model"]]
    log = open(outdir / f"{stem}.log", "w")

    def say(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(f"{now()}  {stem}: {cfg['api']} {cfg['model']} at {cfg['url']}; health {health(a.model).strip()}")
    say(f"  cache {env['XDG_CACHE_HOME']} (new), -j {jobs}, --timeout {timeout}, -k {a.k}")
    say("  $ " + " ".join(cmd[:2]) + " ... " + " ".join(c for c in cmd[2:] if not c.startswith(str(FED / "statements"))))
    before = audit_count() if a.model == "laya" else None
    started = now()
    t0 = time.perf_counter()
    done = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=BENCH.parent)
    seconds = time.perf_counter() - t0
    ended = now()
    (outdir / f"{stem}.stdout").write_text(done.stdout)
    (outdir / f"{stem}.stderr").write_text(done.stderr)
    rows = [json.loads(l) for l in done.stdout.splitlines() if l.strip()]
    stats = parse_stats(done.stderr)
    result = {"model": a.model, "api": cfg["api"], "requested_model": cfg["model"], "url": cfg["url"],
              "bench": a.bench, "tag": a.tag, "stem": stem, "description": description, "k": a.k, "jobs": jobs,
              "timeout": timeout, "input": chosen, "command": cmd, "cache": env["XDG_CACHE_HOME"],
              "started": started, "ended": ended, "seconds": round(seconds, 1), "exit": done.returncode,
              "texts": len(rows), "scored": sum(1 for r in rows if r["score"] is not None)} | stats
    result["provenance"] = provenance(Path(env["XDG_CACHE_HOME"]))
    if a.model == "laya":
        audit = audit_since(before)
        result["laya_requests"] = len(audit)
        result["laya_rejected"] = sum(1 for x in audit if x.get("status") == "context_rejected")
        result["laya_statuses"] = {s: sum(1 for x in audit if x.get("status") == s) for s in {x.get("status") for x in audit}}
    if rows:
        result |= score(a.bench, rows, result)
    comparisons = result["comparisons"] or 0
    if a.bench == "fed-whole" and comparisons:
        per = seconds / comparisons
        result["seconds_per_comparison"] = round(per, 2)
        result["projected_minutes_all_95"] = round(475 * per / 60, 1)
        result["statements_within_60_minutes"] = min(95, int(3600 / (5 * per)))
    (outdir / f"{stem}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    say(f"{ended}  exit {done.returncode}; {seconds:.1f}s wall-clock; {stats['stats']}")
    say(f"  scored {result['scored']}/{result['texts']}; failed comparisons {stats['failed_comparisons']}; "
        f"answers by {result['provenance']['origins']}")
    if stats["sample_errors"]:
        say("  " + "\n  ".join(stats["sample_errors"]))
    if a.model == "laya":
        say(f"  laya audit: {result['laya_requests']} requests, {result['laya_rejected']} rejected for context")
    if "vs_jev" in result:
        v = result["vs_jev"]
        say(f"  against Jev on {v['n']}: spearman {v['spearman']}  pearson {v['pearson']}")
    for key in ("top20_overlap", "top10_overlap"):
        if key in result:
            say(f"  {key} with Jev: {result[key]}")
    for key in ("same_day", "next_180_days"):
        if key in result:
            say(f"  {key}: {result[key]}")
    if "projected_minutes_all_95" in result:
        say(f"  {result['seconds_per_comparison']} s per comparison; all 95 statements at -k 10 would take about "
            f"{result['projected_minutes_all_95']} minutes; {result['statements_within_60_minutes']} fit in 60")
    say(f"  wrote {outdir / (stem + '.json')}")
    wrong = [k for k in result["provenance"]["origins"] if not k.startswith(cfg["api"] + "|")]
    if wrong:
        say(f"  WRONG PROVIDER in the cache: {wrong}")
        sys.exit(3)


# ---- the frozen file ----

def freeze(a) -> None:
    """The latest untagged attempt of each (model, benchmark), without per-text scores, and the Jev column."""
    runs = {}
    for path in sorted(WORK.glob("*.json")):
        r = json.loads(path.read_text())
        if "bench" not in r or r.get("tag"):
            continue
        key = f"{r['model']}/{r['bench']}"
        runs.setdefault(key, []).append(r)
    chosen = {}
    rel = lambda p: str(Path(p).relative_to(BENCH.parent))

    def relative(value):
        """Paths inside the repository, wherever they sit in a run's record, relative to its root."""
        if isinstance(value, dict):
            return {k: relative(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relative(v) for v in value]
        if isinstance(value, str) and value.startswith(str(BENCH.parent) + "/"):
            return rel(value)
        return value

    for key, attempts in sorted(runs.items()):
        last = max(attempts, key=lambda r: r["started"])
        chosen[key] = relative({k: v for k, v in last.items() if k not in ("scores", "command")}) | \
            {"attempts": [r["stem"] for r in attempts], "command": " ".join(last["command"][1:2] + ["..."]),
             "cache": rel(last["cache"]),
             "raw": {ext: rel(WORK / f"{last['stem']}.{ext}") for ext in ("json", "log", "stdout", "stderr")}
             | {"console": rel(WORK / f"console-{last['model']}-{last['bench']}.txt")},
             "other_server": "loaded but idle (runs strictly serialized, one model and one benchmark at a time)"}
    yc_log = (YC / "yc-scifi.log").read_text().strip()
    fed = json.loads((FED / "fed.json").read_text())
    # fed.py check: the same two after-the-fact checks on each local model's statement scores, against the
    # FedLock files Jev was checked against, so the columns share a file date.
    outcomes = {d: {"move": move, "ahead": ahead} for d, _, _, _, move, ahead in fed["table"]}
    for key, entry in chosen.items():
        if entry["bench"] == "fed-whole":
            scores = max(runs[key], key=lambda r: r["started"])["scores"]
            entry["checks"] = {day: checks(scores, outcomes, fedlock(day)[0]) for day in fed.get("checks", {})}
    latest = json.loads((WORK / "jev-latest.json").read_text())
    probes = {p.stem: {k: v for k, v in json.loads(p.read_text()).items()
                       if k in ("started", "seconds", "comparisons", "k", "jobs", "input", "tokens", "cost",
                                "seconds_per_comparison", "projected_minutes_all_95", "statements_within_60_minutes")}
              | {"raw": str((WORK / f"{p.stem}.json").relative_to(BENCH.parent))}
              for p in sorted(WORK.glob("*-probe*.json"))}
    report = {
        "date": "2026-09-22", "timezone": "America/New_York",
        "hardware": "Apple M3 Ultra, 96 GiB unified memory; both servers local, loaded throughout",
        "servers": {
            "diffusiongemma": {"url": MODELS["diffusiongemma"]["url"], "server": "OpenJev", "server_commit": "e04794a",
                               "api": "diffusiongemma", "model": "openjev-0.1",
                               "weights": "mlx-community/diffusiongemma-26B-A4B-it-4bit", "weights_revision": "a7a8140",
                               "note": "executes model work serially on the GPU"},
            "laya": {"url": MODELS["laya"]["url"], "server": "jevkit-core scripts/laya_server.py",
                     "laya_mlx_commit": "fc1df62", "api": "laya", "model": "laya-421m",
                     "checkpoint": "aac6fef/laya-mlx", "checkpoint_revision": "0476785",
                     "context_window_tokens": 512,
                     "note": "requests whose state would be cropped are rejected with HTTP 422 (audit status "
                             "context_rejected); jsort counts them as failed comparisons, not answers",
                     "audit_log": "the server's --audit file, named by LAYA_AUDIT; each run counts the lines it added"},
        },
        "concurrency_and_deadlines": {m: {"jobs_in_flight_default": c["jobs"], "timeout_seconds": c["timeout"]}
                                      for m, c in MODELS.items()}
        | {"note": "per-run jobs and timeout actually used are in each run's `jobs` and `timeout`"},
        "cost": "--budget 0 on every run; local calls are metered at $0; no hosted model was called",
        "cache": "every run used a new empty XDG_CACHE_HOME (the run's `cache`) and an empty XDG_CONFIG_HOME",
        "wall_clock": "`seconds` is perf_counter around the jsort subprocess; `jsort_seconds` is jsort's own --stats "
                      "figure; `started`/`ended` are America/New_York",
        "probes": {"runs": probes, "note": "tagged probe runs decided fed-whole full vs subsample; left out of `runs`"},
        "jev": {
            "model": "Jev 1.13 (hosted; not called for this benchmark, every number read from files)",
            "yc": {"source": "bench/out/yc/yc-scifi.csv (scores) and bench/out/yc/yc-scifi.log (stats), 2026-09-20",
                   "stats": yc_log, "comparisons": 30401, "reliability": 0.97, "cost_usd": 0.412,
                   "seconds": 503.9, "jobs": 32,
                   "sample": "the 500 companies of bench/out/local-2026-09-22/yc-sample500.csv (yc.csv rows, seed "
                             f"{YC_SEED}), joined on the exact `text` field Jev saw",
                   "caveat": "Jev's scores come from one fit over all 6,081 companies, not a 500-company run; the "
                             "local correlations are agreement with Jev's full-population scale. No like-for-like "
                             "Jev wall-clock or reliability exists for a 500-company sort."},
            "fed-latest": {"source": "bench/out/local-2026-09-22/jev-latest.json: " + latest["source"]
                                     + "; answers from the 2026-09-19 run of bench/fed.py",
                           "stats": "255 comparisons in 10 rounds; reliability 0.96; lean -0.07 (bench/out/fed/run.log, "
                                    "2026-09-19)",
                           "reliability": round(latest["reliability"], 4),
                           "checked_against_run_log": latest["checked_against_run_log"],
                           "mismatched": len(latest["mismatched"]),
                           "caveat": "run.log prints 10 of the 51 scores; the other 41 are reconstructed by the replay. "
                                     "The first-run wall-clock was never logged (run.log shows a cached rerun, 0.1 s)."},
            "fed-whole": {"source": "bench/out/fed/fed.json and bench/out/fed/run.log, 2026-09-19", "stats": fed["stats"],
                          "same_day": "+0.46 (95 statements)", "next_180_days": "+0.37 (91 statements)",
                          "checks": fed.get("checks", {}),
                          "note": "68 of 475 answers came from an earlier cache, so the 9.2 s is not a clean "
                                  "wall-clock. Each run's same_day/next_180_days also carry Jev's Spearman on exactly "
                                  "the statements that model scored (jev_same_statements). `checks` is fed.py check: "
                                  "action days alone, means by decision, and FedLock, keyed by the FedLock file's date."},
        },
        "fedlock": {"note": "FedLock (https://jnathan9.github.io/fedlock/) is a running tournament; its scores move "
                            "between releases, so every FedLock number is keyed by the date of the file it was "
                            "computed from. The 2026-09-20 file is the snapshot archived by fedjev-bench "
                            "(https://github.com/maybern-tripp-smith/fedjev-bench, data/raw/fedlock/data.json); "
                            "the 2026-09-23 file was downloaded by fed.py check.",
                    "files": {day: c["fedlock_file"] for day, c in fed.get("checks", {}).items()}},
        "runs": chosen,
    }
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"wrote {FROZEN}: {', '.join(chosen)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=MODELS)
    r.add_argument("--bench", required=True, choices=BENCHMARKS)
    r.add_argument("--k", type=int, default=10)
    r.add_argument("--jobs", type=int, help="calls in flight (default 4)")
    r.add_argument("--timeout", type=float, help="seconds per comparison (default 300 DiffusionGemma, 120 Laya)")
    r.add_argument("--statements", type=int, help="fed-whole: a seeded subsample of this many statements")
    r.add_argument("--seed", type=int, default=20260922, help="fed-whole: the subsample's seed")
    r.add_argument("--limit", type=int, help="rehearsal only: the first N records (fed-whole: use --statements)")
    r.add_argument("--tag", default="", help="suffix for the file names; tagged runs are left out of `freeze`")
    r.add_argument("--outdir", type=Path, default=WORK)
    sub.add_parser("freeze")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "run":
        run(a)
    else:
        freeze(a)


if __name__ == "__main__":
    main()
