"""The README's example: how hawkish is the Fed chair? Live; about seven cents.

Transcripts of FOMC press conferences come from federalreserve.gov (public domain) and the target range
for the federal funds rate from FRED. Needs `pdftotext` (poppler) on the path.

    uv run python bench/fed.py prepare     # download every transcript since the first, in 2011, and cut out each opening statement
    uv run python bench/fed.py run         # sort one statement's sentences, then all the statements

Two sorts, both on "more hawkish about inflation":
  the sentences of the latest opening statement, which is the example at the top of the README
  every opening statement since April 2011 as a whole, scored and set against what the Committee then did

The check on the second was fixed before it was run: the rank correlation between a statement's score
and the change in the top of the target range from the day before the press conference to 180 days
after it, and with the move announced that day. Statements too recent to have 180 days behind them
are left out of the first.
"""

import csv
import io
import json
import re
import subprocess
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

OUT = Path(__file__).parent / "out" / "fed"
CALENDAR = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
TRANSCRIPT = "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{}.pdf"
TARGET = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFEDTARU"
DESCRIPTION = "more hawkish about inflation"
FURNITURE = re.compile(r"^(Page \d+ of \d+|PRELIMINARY|FINAL|[A-Z][a-z]+ \d{1,2}, \d{4}|"
                       r"(Transcript of )?Chair\w* \w+[’']s Press Conference)$")
SPEAKER = re.compile(r"^((?:[A-Z][A-Z’'\-]+\.? ){1,3}[A-Z][A-Z’'\-]+)[.:] ")   # CHAIR POWELL. / MICHELLE SMITH. / CHAIRMAN POWELL:
ABBREVIATIONS = ("U.S.", "Mr.", "Ms.", "Mrs.", "Dr.", "St.", "vs.", "i.e.", "e.g.", "a.m.", "p.m.")


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (jsort benchmark)"})
    with urllib.request.urlopen(request, timeout=60) as r:
        return r.read()


def opening_statement(raw: str) -> tuple[str, str]:
    """The chair's first turn, as one string, and the chair's label. Page furniture and line wraps are removed."""
    lines = [l.strip() for l in raw.splitlines()]
    lines = [l for l in lines if l and not FURNITURE.match(l)]
    start = next(i for i, l in enumerate(lines) if (m := SPEAKER.match(l)) and m.group(1).startswith("CHAIR"))
    chair = SPEAKER.match(lines[start]).group(1)
    end = next((i for i in range(start + 1, len(lines)) if SPEAKER.match(lines[i])), len(lines))
    text = " ".join(lines[start:end])[len(chair) + 2:]
    return re.sub(r"\s+", " ", text).strip(), chair.title()


def sentences(text: str) -> list[str]:
    for a in ABBREVIATIONS:
        text = text.replace(a, a.replace(".", "․"))
    parts = re.split(r"(?<=[.?!])[”\"]?\s+(?=[A-Z“\"])", text)
    return [p.replace("․", ".").strip() for p in parts if len(p.split()) >= 6]


def prepare() -> None:
    (OUT / "statements").mkdir(parents=True, exist_ok=True)
    page = fetch(CALENDAR).decode("utf-8", "replace")
    # The Fed spells the page both ways (fomcpresconf, fomcpressconf). The calendar page starts in 2012;
    # the three press conferences of 2011, the first year there were any, are listed by hand.
    days = sorted(set(re.findall(r"fomcpress?conf(\d{8})\.htm", page)) | {"20110427", "20110622", "20111102"})
    index = []
    for day in days:
        pdf = OUT / f"{day}.pdf"
        if not pdf.exists():
            try:
                pdf.write_bytes(fetch(TRANSCRIPT.format(day)))
            except OSError as e:
                print(f"{day}: no transcript ({e})")
                continue
            time.sleep(0.5)
        raw = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
        text, chair = opening_statement(raw)
        iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        (OUT / "statements" / f"{iso}.txt").write_text(text + "\n")
        index.append({"date": iso, "chair": chair, "words": len(text.split())})
        print(f"{iso}  {chair:16s} {len(text.split()):5d} words")
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    latest = index[-1]["date"]
    (OUT / "latest.txt").write_text("\n".join(sentences((OUT / "statements" / f"{latest}.txt").read_text())) + "\n")
    (OUT / "target.csv").write_bytes(fetch(TARGET))
    print(f"{len(index)} opening statements; the latest, {latest}, is in {OUT / 'latest.txt'} one sentence per line")


def jsort(*argv: str) -> tuple[str, str]:
    done = subprocess.run(["jsort", DESCRIPTION, *argv, "--stats", "--budget", "0"], capture_output=True, text=True)
    if done.returncode:
        sys.exit(done.stderr)
    return done.stdout, done.stderr.strip().splitlines()[-1]


def spearman(a, b) -> float:
    import numpy as np

    def ranks(v):
        v = np.asarray(v, dtype=float)
        r = np.empty(len(v))
        r[np.argsort(v, kind="stable")] = np.arange(len(v))
        for value in np.unique(v):
            r[v == value] = r[v == value].mean()
        return r
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def run() -> None:
    index = json.loads((OUT / "index.json").read_text())
    print(f'$ jsort -o "{DESCRIPTION}" latest.txt        # {index[-1]["date"]}, {index[-1]["chair"]}')
    out, stats = jsort("-o", str(OUT / "latest.txt"))
    lines = out.splitlines()
    print("\n".join(lines[:6]) + "\n...\n" + "\n".join(lines[-4:]) + f"\n{stats}\n")

    files = sorted(str(p) for p in (OUT / "statements").glob("*.txt"))
    out, stats = jsort("--whole", "--json", "--max-chars", "16000", *files)
    scored = {Path(o["file"]).stem: o for o in map(json.loads, out.splitlines())}
    print(f"every opening statement since {index[0]['date']}, as a whole\n{stats}\n")

    rows = list(csv.reader(io.StringIO((OUT / "target.csv").read_text())))[1:]
    target = {date.fromisoformat(d): float(v) for d, v in rows if v not in ("", ".")}
    last = max(target)

    def top_of_range(day: date) -> float:
        while day not in target:
            day -= timedelta(days=1)
        return target[day]

    table, later, same_day = [], [], []
    for entry in index:
        day = date.fromisoformat(entry["date"])
        s = scored[entry["date"]]
        # The new range takes effect the day after the announcement.
        move = round((top_of_range(min(day + timedelta(days=1), last)) - top_of_range(day - timedelta(days=1))) * 100)
        ahead = None
        if day + timedelta(days=180) <= last:
            ahead = round((top_of_range(day + timedelta(days=180)) - top_of_range(day - timedelta(days=1))) * 100)
            later.append((s["score"], ahead))
        same_day.append((s["score"], move))
        table.append((entry["date"], entry["chair"], s["score"], s["se"], move, ahead))
    print(f"{'date':10s}  {'chair':15s} {'score':>6s} {'se':>5s}  {'move that day':>13s}  {'next 180 days':>13s}")
    for d, chair, score, se, move, ahead in table:
        print(f"{d}  {chair:15s} {score:6.2f} {se:5.2f}  {move:+10d} bp  " + (f"{ahead:+10d} bp" if ahead is not None else f"{'':>13s}"))
    print(f"\nrank correlation of the score with the change in the target over the next 180 days: "
          f"{spearman(*zip(*later)):+.2f} ({len(later)} statements)")
    print(f"rank correlation of the score with the move announced that day: {spearman(*zip(*same_day)):+.2f} "
          f"({len(same_day)} statements)")
    (OUT / "fed.json").write_text(json.dumps({"table": table, "stats": stats}, indent=1))


if __name__ == "__main__":
    {"prepare": prepare, "run": run}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
