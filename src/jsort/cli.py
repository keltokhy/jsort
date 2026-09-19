"""jsort: put lines in order along a dimension you describe.

    jsort "more urgent" tickets.txt | head
    jsort -o "more hawkish about inflation" statements.txt
    jsort --csv --field narrative -o --name breadth "drew broader military participation" coups.csv

Jev is asked which of two texts ranks higher, for a few pairs per text, and a Bradley-Terry scale is
fitted to its answers. The highest text prints first. Exit status: 0 when sorted, 2 on any error.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import time

from . import __version__
from .core import BACKENDS, Cache, Jev, JevFatal, config_dir, resolve_backend
from .engine import Ranking, arank
from .inputs import Record, read

MAX_ERRORS_SHOWN = 10
DEFAULT_BUDGET = 1.0  # dollars, as in jgrep: a command that bills per question needs a seat belt
SHAKY = 0.8           # below this split-half reliability the order is reported as unsteady


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="jsort", formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="jsort [options] DESCRIPTION [FILE ...]",
        description="Sort lines along a plain-English dimension, from pairwise comparisons judged by TypeSafe's Jev model.",
        epilog='examples:\n'
               '  jsort "more urgent" tickets.txt | head\n'
               '  jsort -o "more hawkish about inflation" statements.txt\n'
               '  jsort --top 5 "a more serious safety problem" complaints.txt\n'
               '  jsort --whole -r "easier for a newcomer to read" abstracts/*.txt\n'
               '  jsort --csv --field narrative -o --keep-order "drew broader participation" events.csv\n\n'
               "Jev is reached through TypeSafe's API (TYPESAFE_API_KEY), OpenRouter (OPENROUTER_API_KEY) or a\n"
               "System One gateway of your own (JEV_GATEWAY_URL and JEV_GATEWAY_API_KEY).\n"
               f"Keys can also live in {config_dir()}/typesafe.key, openrouter.key or gateway.key.")
    ap.add_argument("args", nargs="*", help=argparse.SUPPRESS)
    ap.add_argument("-k", "--per-item", type=int, default=10, metavar="N",
                    help="comparisons each text takes part in (default 10); the run asks about half that many "
                         "questions per text")
    ap.add_argument("--top", type=int, metavar="N",
                    help="print only the top N, and stop asking about texts that are clearly out of the running")
    ap.add_argument("-r", "--reverse", action="store_true", help="lowest first")
    ap.add_argument("-o", "--score", action="store_true",
                    help="put the score and its standard error in the first two tab-separated columns; "
                         "with --csv or --jsonl, add them to each record")
    ap.add_argument("--name", default="jsort", metavar="NAME",
                    help="with -o on --csv or --jsonl, call the new fields NAME_score, NAME_se and NAME_n (default jsort)")
    ap.add_argument("--keep-order", action="store_true",
                    help="print in input order; with -o this adds scores to a file without rearranging it")
    ap.add_argument("-n", "--line-number", action="store_true", help="prefix each line with its line number")
    ap.add_argument("-H", "--with-filename", action="store_true", help="prefix each line with its file name")
    ap.add_argument("--json", action="store_true", help="print one JSON object per text, with rank, score and standard error")
    formats = ap.add_mutually_exclusive_group()
    formats.add_argument("--jsonl", action="store_true", help="read JSON objects, comparing only --field and returning full records")
    formats.add_argument("--csv", action="store_true", help="read CSV with a header, comparing only --field and returning full rows")
    ap.add_argument("--field", metavar="NAME", help="JSON field (dotted paths supported) or CSV column to compare")
    ap.add_argument("--para", action="store_true", help="sort paragraphs (separated by blank lines), not lines")
    ap.add_argument("--whole", action="store_true", help="sort whole files and print their names")
    ap.add_argument("--seed", type=int, default=0, metavar="N",
                    help="seed for the choice of pairs (default 0); the same seed asks the same questions, so a rerun is cached")
    ap.add_argument("-j", "--concurrency", type=int, default=32, metavar="N", help="calls in flight (default 32)")
    ap.add_argument("--timeout", type=float, default=15.0, metavar="SECONDS",
                    help="give up on a comparison after this long, retries included (default 15)")
    ap.add_argument("--budget", type=float, default=None, metavar="DOLLARS",
                    help="stop asking once this much has been spent and sort on what is known (default 1.00, or "
                         "$JSORT_BUDGET; 0 for no limit)")
    ap.add_argument("--max-chars", type=int, default=8000, metavar="N",
                    help="show Jev only the first N characters of a text (default 8000)")
    ap.add_argument("--no-cache", action="store_true", help="do not read or write the answer cache")
    ap.add_argument("--api", choices=list(BACKENDS), help="which API to call (default: whichever has a key)")
    ap.add_argument("--model", metavar="ID", help="model ID to request (default: the API's latest Jev)")
    ap.add_argument("--stats", action=argparse.BooleanOptionalAction, default=None,
                    help="print comparisons, reliability, tokens and cost to stderr at the end (default: when stderr is a terminal)")
    ap.add_argument("--version", action="version", version=f"jsort {__version__}")
    return ap


def _number(x: float, places: int) -> str:
    return "" if math.isnan(x) else f"{x + 0.0:.{places}f}".replace("-0." + "0" * places, "0." + "0" * places)


def write(records: list[Record], header: list[str] | None, ranking: Ranking, order: list[int], args, out) -> None:
    ranks = {i: r for r, i in enumerate(ranking.order(), 1) if not math.isnan(ranking.score[i])}   # 1 is the highest, with or without -r
    fields = [f"{args.name}_score", f"{args.name}_se", f"{args.name}_n"]

    def scores(i: int) -> dict:
        unscored = math.isnan(ranking.score[i])
        return {fields[0]: None if unscored else round(float(ranking.score[i]), 4),
                fields[1]: None if unscored else round(float(ranking.se[i]), 4),
                fields[2]: int(ranking.comparisons[i])}

    if args.json:
        for i in order:
            rec, s = records[i], scores(i)
            obj = {"rank": ranks.get(i), "score": s[fields[0]], "se": s[fields[1]], "comparisons": s[fields[2]],
                   "file": rec.file, "line": rec.lineno}
            if not args.whole:
                obj["text"] = rec.text if args.csv else rec.original
            if rec.data is not None:
                obj["record"], obj["field"] = rec.data, args.field
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    elif args.csv:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow((header or []) + (fields if args.score else []))
        for i in order:
            data = records[i].data
            row = ["" if data.get(h) is None else data[h] for h in header or []]
            if args.score:
                row += ["" if v is None else v for v in scores(i).values()]
            # A row longer than the header keeps its surplus values, after the named columns.
            writer.writerow(row + list(data.get(None) or []))
    else:
        for i in order:
            rec = records[i]
            if args.jsonl and args.score and isinstance(rec.data, dict):
                body = json.dumps(rec.data | scores(i), ensure_ascii=False)
            else:
                body = ((f"{rec.file}:" if args.with_filename and not args.whole else "")
                        + (f"{rec.lineno}:" if args.line_number and not args.whole else "") + rec.original)
                if args.score:
                    body = f"{_number(ranking.score[i], 2)}\t{_number(ranking.se[i], 2)}\t{body}"
            out.write(body + ("\n\n" if args.para else "\n"))
    out.flush()


def summary(records: list[Record], ranking: Ranking, jev: Jev) -> str:
    parts = [f"{len(records):,} texts, {ranking.asked:,} comparisons in {ranking.rounds} rounds"]
    if ranking.reliability is not None:
        parts.append(f"reliability {ranking.reliability:.2f}")
    if ranking.asked:
        parts.append(f"first-position lean {ranking.lean:+.2f}")
    return "; ".join(parts + [jev.meter.summary()])


def main(argv: list[str] | None = None, *, transport=None, out=None, err=None) -> int:
    out, err = out or sys.stdout, err or sys.stderr
    ap = parser()
    args = ap.parse_intermixed_args(argv)
    if not args.args:
        ap.print_usage(err)
        return 2
    description, files = args.args[0], args.args[1:]
    if args.budget is None:
        try:
            args.budget = float(os.environ.get("JSORT_BUDGET") or DEFAULT_BUDGET)
        except ValueError:
            print(f"jsort: JSORT_BUDGET must be a number of dollars; got {os.environ['JSORT_BUDGET']!r}", file=err)
            return 2
    for valid, message in (
        (bool(description.strip()), "the description is empty"),
        (args.per_item >= 2, "-k takes 2 or more comparisons per text; the first round alone gives every text two"),
        (args.top is None or args.top >= 1, "--top takes 1 or more"),
        (args.concurrency >= 1, "-j takes 1 or more concurrent calls"),
        (math.isfinite(args.budget) and args.budget >= 0, "--budget / JSORT_BUDGET must be finite and nonnegative"),
        (math.isfinite(args.timeout) and args.timeout > 0, "--timeout must be finite and greater than 0"),
        (args.max_chars > 0, "--max-chars must be greater than 0"),
        (not (args.jsonl or args.csv) or bool(args.field), "--jsonl and --csv require --field"),
        (not args.field or args.jsonl or args.csv, "--field requires --jsonl or --csv"),
        (not (args.jsonl or args.csv) or not (args.para or args.whole), "structured input cannot be combined with --para or --whole"),
        (not (args.whole and args.para), "--whole and --para cannot be combined"),
        (not args.whole or bool(files), "--whole sorts files, so it needs file names"),
        (not (args.top and args.keep_order), "--top and --keep-order cannot be combined"),
        (args.name.isidentifier(), "--name must be a plain word, since it becomes part of a column name"),
    ):
        if not valid:
            print(f"jsort: {message}", file=err)
            return 2

    records, header, problems = read(files, args)
    if args.csv and args.score and header and (clash := {f"{args.name}_{s}" for s in ("score", "se", "n")} & set(header)):
        print(f"jsort: the file already has a column named {sorted(clash)[0]}; choose another prefix with --name", file=err)
        return 2
    for message in problems[:MAX_ERRORS_SHOWN]:
        print(f"jsort: {message}", file=err)
    if len(problems) > MAX_ERRORS_SHOWN:
        print(f"jsort: and {len(problems) - MAX_ERRORS_SHOWN:,} more input errors", file=err)
    if args.jsonl and args.score and (clash := {f"{args.name}_{x}" for x in ("score", "se", "n")}
                                      & {k for r in records if isinstance(r.data, dict) for k in r.data}):
        print(f"jsort: the records already have a field named {sorted(clash)[0]}; choose another prefix with --name", file=err)
        return 2
    if not records:
        if args.csv and header is not None and not args.json:
            write([], header, Ranking.unscored(0), [], args, out)     # a header with no rows is still a CSV
        return 2 if problems else 0

    texts = [r.text for r in records]
    truncated = sum(len(t) > args.max_chars for t in texts)
    show_stats = args.stats or (args.stats is None and err.isatty())
    jev = None
    t0 = time.perf_counter()
    if len({t[:args.max_chars] for t in texts if t.strip()}) < 2:
        ranking = Ranking.unscored(len(texts))   # nothing to compare, so no key is needed either
    else:
        try:
            backend, key = resolve_backend(args.api)
        except JevFatal as e:
            print(f"jsort: {e}", file=err)
            return 2
        jev = Jev(key, backend, model=args.model, timeout=args.timeout, concurrency=args.concurrency,
                  cache=None if args.no_cache else Cache(), transport=transport)

        def progress(done: int, total: int) -> None:
            if show_stats and err.isatty():
                print(f"\rjsort: {done:,} of about {total:,} comparisons", end="", file=err, flush=True)

        async def go() -> Ranking:
            try:
                return await arank(texts, description, jev, per_item=args.per_item, top=args.top, lowest=args.reverse,
                                   seed=args.seed, budget=args.budget, max_chars=args.max_chars,
                                   concurrency=args.concurrency, progress=progress)
            finally:
                await jev.close()

        try:
            ranking = asyncio.run(go())
        except KeyboardInterrupt:
            print(f"\njsort: interrupted; {jev.meter.summary()}", file=err)
            return 130
        if show_stats and err.isatty():
            print("\r\033[K", end="", file=err)

    if ranking.fatal:
        print(f"jsort: {ranking.fatal}", file=err)
        if not ranking.asked:
            return 2
        # Answers that were paid for before the API refused are still worth a sort.
        print(f"jsort: sorted on the {ranking.asked:,} comparisons answered before that", file=err)
    for message in ranking.errors[:MAX_ERRORS_SHOWN]:
        print(f"jsort: a comparison failed: {message}", file=err)
    if len(ranking.errors) > MAX_ERRORS_SHOWN:
        print(f"jsort: and {len(ranking.errors) - MAX_ERRORS_SHOWN:,} more failed comparisons", file=err)

    order = list(range(len(records))) if args.keep_order else ranking.order(args.reverse)
    if args.top:
        order = [i for i in order if not math.isnan(ranking.score[i])][:args.top]
    try:
        write(records, header, ranking, order, args, out)
    except BrokenPipeError:
        pass

    if args.csv and (ragged := sum(1 for r in records if r.data.get(None))):
        print(f"jsort: {ragged:,} rows have more values than the header has columns; the surplus is kept at the end "
              "of each row", file=err)
    if truncated:
        print(f"jsort: compared only the first {args.max_chars:,} characters of {truncated:,} texts; raise --max-chars",
              file=err)
    if ranking.over_budget:
        print(f"jsort: stopped asking at the ${args.budget:.2f} budget after {ranking.asked:,} comparisons and sorted on "
              "those; raise it with --budget, and the answers so far come back from the cache", file=err)
    unscored = sum(math.isnan(s) and bool(t.strip()) for s, t in zip(ranking.score, texts))
    if unscored and jev is not None:
        print(f"jsort: {unscored:,} texts were never compared and are listed last", file=err)
    if ranking.reliability is not None and ranking.reliability < SHAKY:
        print(f"jsort: reliability {ranking.reliability:.2f}: two halves of the comparisons give different orders. "
              "Raise -k, or reword the description so that any two of these texts can be compared on it", file=err)
    if show_stats and jev is not None:
        print(f"jsort: {summary(records, ranking, jev)}; {time.perf_counter() - t0:.1f}s", file=err)
    return 2 if problems or ranking.errors or ranking.over_budget or ranking.fatal else 0


def cli() -> None:
    code = main()
    try:
        sys.stdout.flush()
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    sys.stderr.flush()
    sys.exit(code)


if __name__ == "__main__":
    cli()
