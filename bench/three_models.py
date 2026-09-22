"""Three decision models on the same readability sample: hosted Jev, and DiffusionGemma and Laya on this machine.

    uv run --group bench python bench/three_models.py [--models jev,diffusiongemma,laya] [--k 10] [--workdir DIR]

Runs the installed `jsort` once per model on the sample from `readability.py prepare`, each with an empty answer
cache so every comparison is bought, then asks each model the one-question alternative. Reports agreement with
the teachers' scale, coverage where a model could not read a pair, and cost. Local servers must already be running:
see the jevkit-runtime docs for OpenJev (port 8080) and Laya (port 8081). Set LAYA_AUDIT to the file given
to the Laya server's --audit to count the pairs it refused.
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from jevkit_runtime import Client, JevError, JevFatal, resolve
from jsort.core import PROVIDERS
from readability import DESCRIPTION, DIRECT, OUT, spearman

MODELS = {
    "jev": {"env": {"JEV_API": "openrouter", "JEV_MODEL": "typesafe/jev-1.13"}, "jobs": 32, "timeout": 60},
    "diffusiongemma": {"env": {"JEV_API": "diffusiongemma", "JEV_MODEL": "openjev-0.1"}, "jobs": 2, "timeout": 300},
    "laya": {"env": {"JEV_API": "laya"}, "jobs": 4, "timeout": 120},
}
LAYA_AUDIT = Path(p) if (p := os.environ.get("LAYA_AUDIT")) else None


def audit_lines() -> list[dict]:
    if LAYA_AUDIT is None or not LAYA_AUDIT.exists():
        return []
    return [json.loads(line) for line in LAYA_AUDIT.read_text().splitlines() if line.strip()]


def environment(model: str, workdir: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("JEV_URL", "JEV_API", "JEV_MODEL")}
    env |= MODELS[model]["env"] | {"XDG_CACHE_HOME": str(workdir / f"cache-{model}")}
    return env


def sort(model: str, k: int, workdir: Path) -> dict:
    cfg = MODELS[model]
    before = len(audit_lines()) if model == "laya" else 0
    cmd = ["jsort", DESCRIPTION, str(OUT / "clear.jsonl"), "--jsonl", "--field", "excerpt", "-o", "--keep-order",
           "-k", str(k), "--stats", "--budget", "0", "-j", str(cfg["jobs"]), "--timeout", str(cfg["timeout"])]
    t0 = time.perf_counter()
    done = subprocess.run(cmd, capture_output=True, text=True, env=environment(model, workdir))
    seconds = time.perf_counter() - t0
    (workdir / f"{model}-sort.jsonl").write_text(done.stdout)
    (workdir / f"{model}-sort.stderr").write_text(done.stderr)
    rows = [json.loads(line) for line in done.stdout.splitlines() if line.strip()]
    lines = done.stderr.splitlines()
    stats = next((line for line in reversed(lines) if "comparisons in" in line), "")
    errors = [line for line in lines if line.startswith("jsort:") and line is not stats]
    more = sum(int(m.group(1).replace(",", "")) - 1 for line in errors if (m := re.search(r"and ([\d,]+) more", line)))
    number = lambda pattern: int(m.group(1).replace(",", "")) if (m := re.search(pattern, stats)) else None
    cost = float(m.group(1)) if (m := re.search(r"\$([\d.]+)", stats)) else None
    result = {"exit": done.returncode, "seconds": round(seconds, 1), "stats": stats, "rows": rows,
              "failed_comparisons": len(errors) + more, "comparisons": number(r"([\d,]+) comparisons"),
              "calls": number(r"([\d,]+) calls"), "cost": cost}
    if model == "laya":
        audit = audit_lines()[before:]
        result["laya_requests"] = len(audit)
        result["laya_rejected"] = sum(1 for a in audit if a.get("status") == "context_rejected")
    return result


async def direct(model: str, texts: list[str], workdir: Path) -> dict:
    cfg = MODELS[model]
    env = environment(model, workdir)
    os.environ.update({k: env[k] for k in ("JEV_API",) if k in env})
    os.environ.pop("JEV_MODEL", None)
    if "JEV_MODEL" in cfg["env"]:
        os.environ["JEV_MODEL"] = cfg["env"]["JEV_MODEL"]
    before = len(audit_lines()) if model == "laya" else 0
    client = Client(resolve(PROVIDERS), timeout=cfg["timeout"], concurrency=cfg["jobs"])
    sem = asyncio.Semaphore(cfg["jobs"])

    async def ask(text):
        async with sem:
            try:
                return float((await client.ask(text, {"direct": DIRECT}))["direct"]["noul"])
            except (JevError, JevFatal) as exc:
                return str(exc)

    t0 = time.perf_counter()
    try:
        answers = await asyncio.gather(*(ask(t) for t in texts))
    finally:
        await client.close()
    result = {"values": [a if isinstance(a, float) else None for a in answers],
              "failed": sum(1 for a in answers if not isinstance(a, float)),
              "sample_error": next((a for a in answers if not isinstance(a, float)), None),
              "seconds": round(time.perf_counter() - t0, 1), "calls": client.meter.calls, "cost": client.meter.cost,
              "resolved_model": client.meter.model}
    if model == "laya":
        audit = audit_lines()[before:]
        result["laya_rejected"] = sum(1 for a in audit if a.get("status") == "context_rejected")
    return result


def agreement(values: np.ndarray, truth: np.ndarray, present: np.ndarray) -> list[dict]:
    """Agreement with the teachers on random pairs both scored, by the gap between them on the teachers' scale."""
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(truth), 40000), rng.integers(0, len(truth), 40000)
    keep = present[i] & present[j] & (i != j)
    i, j = i[keep], j[keep]
    gap = np.abs(truth[i] - truth[j])
    out = []
    for lo, hi in ((0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99.0)):
        m = (gap >= lo) & (gap < hi)
        agree = float(np.mean(np.sign(values[i[m]] - values[j[m]]) == np.sign(truth[i[m]] - truth[j[m]]))) if m.any() else None
        out.append({"gap": [lo, hi], "pairs": int(m.sum()), "agree": agree})
    return out


def correlate(values: list, truth: np.ndarray) -> dict:
    present = np.array([v is not None for v in values])
    if present.sum() < 3:
        return {"n": int(present.sum()), "pearson": None, "spearman": None, "distinct": 0}
    v = np.array([x for x in values if x is not None], dtype=float)
    t = truth[present]
    return {"n": int(present.sum()), "pearson": float(np.corrcoef(v, t)[0, 1]), "spearman": spearman(v, t),
            "distinct": int(len(set(v.tolist())))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="laya,jev,diffusiongemma")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--workdir", type=Path, default=OUT / "three-models")
    ap.add_argument("--no-direct", action="store_true", help="skip the one-question alternative")
    a = ap.parse_args()
    a.workdir.mkdir(parents=True, exist_ok=True)
    data = [json.loads(line) for line in open(OUT / "clear.jsonl")]
    truth = np.array([d["easiness"] for d in data])
    texts = [d["excerpt"] for d in data]
    ceiling = (1 - float(np.mean([d["easiness_se"] ** 2 for d in data])) / float(truth.var())) ** 0.5
    report = {"date": time.strftime("%Y-%m-%d"), "n": len(data), "k": a.k, "ceiling": ceiling, "models": {}}
    out_path = a.workdir / "three-models.json"
    if out_path.exists():
        report["models"] = json.loads(out_path.read_text()).get("models", {})
    for model in a.models.split(","):
        print(f"\n=== {model}: jsort -k {a.k}", flush=True)
        s = sort(model, a.k, a.workdir)
        values = [r.get("jsort_score") for r in s["rows"]]
        if len(values) != len(data):
            print(f"  {len(values)} rows for {len(data)} texts; exit {s['exit']}", flush=True)
            print("  " + "\n  ".join(s["rows"][:0] or open(a.workdir / f"{model}-sort.stderr").read().splitlines()[-5:]))
        entry = {"sort": {k: v for k, v in s.items() if k != "rows"} | correlate(values, truth)}
        present = np.array([v is not None for v in values] + [False] * (len(data) - len(values)))
        if present.any():
            filled = np.array([v if v is not None else np.nan for v in values] + [np.nan] * (len(data) - len(values)))
            entry["sort"]["agreement"] = agreement(filled, truth, present)
        print(f"  {s['stats']}\n  failed comparisons {s['failed_comparisons']}; scored texts {entry['sort']['n']}/{len(data)}; "
              f"r {entry['sort']['pearson']}  rho {entry['sort']['spearman']}  ({s['seconds']}s)", flush=True)
        if "laya_rejected" in s:
            print(f"  laya: {s['laya_rejected']} of {s['laya_requests']} requests rejected for context", flush=True)
        if not a.no_direct:
            print(f"=== {model}: one question per text", flush=True)
            d = asyncio.run(direct(model, texts, a.workdir))
            entry["direct"] = {k: v for k, v in d.items() if k != "values"} | correlate(d["values"], truth)
            print(f"  answered {entry['direct']['n']}/{len(data)}; r {entry['direct']['pearson']}  rho {entry['direct']['spearman']}"
                  f"; {d['calls']} calls ${d['cost']:.4f} ({d['seconds']}s); model {d['resolved_model']!r}", flush=True)
            if d["sample_error"]:
                print(f"  sample error: {d['sample_error'][:160]}", flush=True)
        entry["sort_values"] = values
        entry["direct_values"] = d["values"] if not a.no_direct else None
        report["models"][model] = entry
        out_path.write_text(json.dumps(report, indent=1))
    # the texts every run scored, for a like-for-like number
    scored = [m for m in report["models"] if report["models"][m]["sort"]["n"]]
    if len(scored) > 1:
        common = np.all([[v is not None for v in report["models"][m]["sort_values"]] + [False] * (len(data) - len(report["models"][m]["sort_values"])) for m in scored], axis=0)
        report["common"] = {"n": int(common.sum())}
        for m in scored:
            vals = report["models"][m]["sort_values"] + [None] * (len(data) - len(report["models"][m]["sort_values"]))
            report["common"][m] = correlate([v if c else None for v, c in zip(vals, common)], truth)
        print(f"\n=== on the {int(common.sum())} texts every model scored: " + "  ".join(
            f"{m} r {report['common'][m]['pearson']:+.3f}" for m in scored), flush=True)
    out_path.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
