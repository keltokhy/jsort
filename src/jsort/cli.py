"""jsort: put lines in order along a dimension you describe.

    jsort "more urgent" tickets.txt | head
    jsort -o "more hawkish about inflation" statements.txt
    jsort --csv --field narrative -o --name breadth "drew broader military participation" coups.csv
    jsort --save-scale hawkish.json "more hawkish about inflation" statements.txt
    tail -f captions.txt | jsort --scale hawkish.json -o --keep-order

Jev is asked which of two texts ranks higher, for a few pairs per text, and a Bradley-Terry scale is
fitted to its answers. The highest text prints first. A saved scale places later texts by comparing
them with its anchors only. Exit status: 0 when sorted, 2 on any error.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import CancelledError

from . import __version__
from .core import BACKENDS, Cache, Jev, JevFatal, config_dir, resolve_backend
from .engine import Ranking, arank
from .inputs import Record, read, records as read_records
from .placement import Placement, Placer, aplace, client_for, collect
from .scale import DEFAULT_ANCHORS, Scale, ScaleError

MAX_ERRORS_SHOWN = 10
DEFAULT_BUDGET = 1.0  # dollars, as in jgrep: a command that bills per question needs a seat belt
SHAKY = 0.8           # below this split-half reliability the order is reported as unsteady


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="jsort", formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="jsort [options] DESCRIPTION [FILE ...]\n       jsort [options] --scale SCALE [FILE ...]",
        description="Sort lines along a plain-English dimension, from pairwise comparisons judged by TypeSafe's Jev model.",
        epilog='examples:\n'
               '  jsort "more urgent" tickets.txt | head\n'
               '  jsort -o "more hawkish about inflation" statements.txt\n'
               '  jsort --top 5 "a more serious safety problem" complaints.txt\n'
               '  jsort --whole -r "easier for a newcomer to read" abstracts/*.txt\n'
               '  jsort --csv --field narrative -o --keep-order "drew broader participation" events.csv\n'
               '  jsort --save-scale hawkish.json "more hawkish about inflation" statements.txt\n'
               '  tail -f captions.txt | jsort --scale hawkish.json -o --keep-order\n\n'
               "Jev is reached through TypeSafe's API (TYPESAFE_API_KEY), OpenRouter (OPENROUTER_API_KEY) or a\n"
               "System One gateway of your own (JEV_GATEWAY_URL and JEV_GATEWAY_API_KEY).\n"
               f"Keys can also live in {config_dir()}/typesafe.key, openrouter.key or gateway.key.")
    ap.add_argument("args", nargs="*", help=argparse.SUPPRESS)
    ap.add_argument("-k", "--per-item", type=int, default=10, metavar="N",
                    help="comparisons each text takes part in (default 10); the run asks about half that many "
                         "questions per text. With --scale, the most comparisons a new text gets, all with anchors")
    ap.add_argument("--top", type=int, metavar="N",
                    help="print only the top N, and stop asking about texts that are clearly out of the running")
    ap.add_argument("-r", "--reverse", action="store_true", help="lowest first")
    ap.add_argument("-o", "--score", action="store_true",
                    help="put the score and its standard error in the first two tab-separated columns; "
                         "with --csv or --jsonl, add them to each record")
    ap.add_argument("--name", default="jsort", metavar="NAME",
                    help="with -o on --csv or --jsonl, call the new fields NAME_score, NAME_se and NAME_n (default jsort)")
    ap.add_argument("--keep-order", action="store_true",
                    help="print in input order; with -o this adds scores to a file without rearranging it. "
                         "With --scale, each text prints as soon as it and those before it are placed")
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
    ap.add_argument("--save-scale", metavar="FILE",
                    help="also write the fitted scale to FILE: the description, the model, and anchors with their scores")
    ap.add_argument("--anchors", type=int, default=None, metavar="N",
                    help=f"with --save-scale, how many texts to keep as anchors (default {DEFAULT_ANCHORS})")
    ap.add_argument("--scale", metavar="FILE",
                    help="place the input on the scale saved in FILE, which supplies the description. Each text is "
                         "compared with the scale's anchors only, so its score does not depend on the rest of the input")
    ap.add_argument("--se-target", type=float, default=None, metavar="SE",
                    help="with --scale, stop asking about a text once its standard error is this small")
    ap.add_argument("--unordered", action="store_true",
                    help="with --scale, print each text as soon as it is placed, not in input order")
    ap.add_argument("--any-model", action="store_true",
                    help="with --scale, place even though the API, endpoint or model is not the one the scale was built "
                         "with; with --save-scale, save even though the answers do not all name one model")
    ap.add_argument("-j", "--concurrency", type=int, default=32, metavar="N", help="calls in flight (default 32)")
    ap.add_argument("--timeout", type=float, default=15.0, metavar="SECONDS",
                    help="give up on a comparison after this long, retries included (default 15)")
    ap.add_argument("--budget", type=float, default=None, metavar="DOLLARS",
                    help="stop asking once this much has been spent and sort on what is known (default 1.00, or "
                         "$JSORT_BUDGET; 0 for no limit)")
    ap.add_argument("--max-chars", type=int, default=None, metavar="N",
                    help="show Jev only the first N characters of a text (default 8000; with --scale, the scale's)")
    ap.add_argument("--no-cache", action="store_true", help="do not read or write the answer cache")
    ap.add_argument("--api", choices=list(BACKENDS), help="which API to call (default: whichever has a key)")
    ap.add_argument("--model", metavar="ID", help="model ID to request (default: the API's latest Jev)")
    ap.add_argument("--stats", action=argparse.BooleanOptionalAction, default=None,
                    help="print comparisons, reliability, tokens and cost to stderr at the end (default: when stderr is a terminal)")
    ap.add_argument("--version", action="version", version=f"jsort {__version__}")
    return ap


def _number(x: float, places: int) -> str:
    return "" if math.isnan(x) else f"{x + 0.0:.{places}f}".replace("-0." + "0" * places, "0." + "0" * places)


class Writer:
    """Output, one text at a time, so that a text placed on a saved scale can print the moment it is known."""

    def __init__(self, args, out, placing: bool = False):
        self.args, self.out, self.placing = args, out, placing
        # A placed text also says whether it fell beyond the scale's anchors, where its score is an extrapolation.
        self.fields = [f"{args.name}_{s}" for s in ("score", "se", "n") + (("beyond",) if placing else ())]
        self.header: list[str] | None = None
        self.rows = csv.writer(out, lineterminator="\n") if args.csv and not args.json else None

    def clash(self, names) -> str | None:
        found = set(self.fields) & set(names) if self.args.score else set()
        return sorted(found)[0] if found else None

    def start(self, header: list[str] | None) -> None:
        self.header = header
        if self.rows is not None:
            self.rows.writerow((header or []) + (self.fields if self.args.score else []))

    def text(self, rec: Record, rank: int | None, score: float, se: float, n: int, beyond: int = 0) -> None:
        args, unscored = self.args, math.isnan(score)
        s = dict(zip(self.fields, (None if unscored else round(float(score), 4), None if unscored else round(float(se), 4),
                                   int(n), {1: "above", -1: "below"}.get(int(beyond)))))
        if args.json:
            obj = {"rank": rank, "score": s[self.fields[0]], "se": s[self.fields[1]], "comparisons": s[self.fields[2]]}
            if self.placing:
                obj["beyond"] = s[self.fields[3]]
            obj |= {"file": rec.file, "line": rec.lineno}
            if not args.whole:
                obj["text"] = rec.text if args.csv else rec.original
            if rec.data is not None:
                obj["record"], obj["field"] = rec.data, args.field
            self.out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        elif args.csv:
            row = ["" if rec.data.get(h) is None else rec.data[h] for h in self.header or []]
            if args.score:
                row += ["" if v is None else v for v in s.values()]
            # A row longer than the header keeps its surplus values, after the named columns.
            self.rows.writerow(row + list(rec.data.get(None) or []))
        else:
            if args.jsonl and args.score and isinstance(rec.data, dict):
                body = json.dumps(rec.data | s, ensure_ascii=False)
            else:
                body = ((f"{rec.file}:" if args.with_filename and not args.whole else "")
                        + (f"{rec.lineno}:" if args.line_number and not args.whole else "") + rec.original)
                if args.score:
                    body = f"{_number(score, 2)}\t{_number(se, 2)}\t{body}"
            self.out.write(body + ("\n\n" if args.para else "\n"))


def write(records: list[Record], header: list[str] | None, ranking: Ranking, order: list[int], args, out) -> None:
    ranks = {i: r for r, i in enumerate(ranking.order(), 1) if not math.isnan(ranking.score[i])}   # 1 is the highest, with or without -r
    placing = isinstance(ranking, Placement)
    writer = Writer(args, out, placing)
    writer.start(header)
    for i in order:
        writer.text(records[i], ranks.get(i), ranking.score[i], ranking.se[i], ranking.comparisons[i],
                    ranking.beyond[i] if placing else 0)
    out.flush()


def summary(records: list[Record], ranking: Ranking, jev: Jev) -> str:
    parts = [f"{len(records):,} texts, {ranking.asked:,} comparisons in {ranking.rounds} rounds"]
    if ranking.reliability is not None:
        parts.append(f"reliability {ranking.reliability:.2f}")
    if ranking.asked:
        parts.append(f"first-position lean {ranking.lean:+.2f}")
    return "; ".join(parts + [jev.meter.summary()])


def closing_notes(args, scale: Scale | None, *, truncated: int, ragged: int, over_budget: bool, asked: int,
                  unscored: int, beyond: tuple[int, int], unverified: int = 0, err) -> None:
    """What the user should know about the run, after the output: the same lines whether it sorted or placed."""
    if ragged:
        print(f"jsort: {ragged:,} rows have more values than the header has columns; the surplus is kept at the end "
              "of each row", file=err)
    if truncated:
        print(f"jsort: compared only the first {args.max_chars:,} characters of {truncated:,} texts; raise --max-chars",
              file=err)
    if over_budget:
        print(f"jsort: stopped asking at the ${args.budget:.2f} budget after {asked:,} comparisons and "
              f"{'placed' if scale else 'sorted'} on those; raise it with --budget, and the answers so far come back "
              "from the cache", file=err)
    if unscored:
        print(f"jsort: {unscored:,} texts " + ("could not be placed and have no score" if scale else
                                               "were never compared and are listed last"), file=err)
    if unverified:
        print(f"jsort: {unverified:,} of the {asked:,} answers did not say which model gave them (cache entries written "
              f"before jsort recorded it, or an API that does not name its model), so they could not be checked against "
              f"the scale's, {scale.answered_by}; --no-cache asks again", file=err)
    if sum(beyond):
        low, high = scale.span
        print(f"jsort: {sum(beyond):,} texts fell beyond the scale's anchors ({beyond[0]:,} above {_number(high, 2)}, "
              f"{beyond[1]:,} below {_number(low, 2)}); their scores are extrapolations", file=err)


def main(argv: list[str] | None = None, *, transport=None, out=None, err=None) -> int:
    out, err = out or sys.stdout, err or sys.stderr
    ap = parser()
    args = ap.parse_intermixed_args(argv)
    scale = None
    if args.scale:
        try:
            scale = Scale.load(args.scale)
        except ScaleError as e:
            print(f"jsort: {e}", file=err)
            return 2
        description, files = scale.description, list(args.args)
        if files and files[0] == description:
            files = files[1:]
        elif files and files[0] != "-" and not os.path.exists(files[0]):
            # The scale is its description. Another one on the command line is a mistake, never an override.
            print(f"jsort: {files[0]!r} is neither a file nor the description of {args.scale}, which is "
                  f"{description!r}; with --scale the description comes from the scale", file=err)
            return 2
    elif not args.args:
        ap.print_usage(err)
        return 2
    else:
        description, files = args.args[0], args.args[1:]
    if args.budget is None:
        try:
            args.budget = float(os.environ.get("JSORT_BUDGET") or DEFAULT_BUDGET)
        except ValueError:
            print(f"jsort: JSORT_BUDGET must be a number of dollars; got {os.environ['JSORT_BUDGET']!r}", file=err)
            return 2
    given_max_chars = args.max_chars
    if args.max_chars is None:
        args.max_chars = scale.max_chars if scale else 8000
    for valid, message in (
        (bool(description.strip()), "the description is empty"),
        (args.per_item >= 2, "-k takes 2 or more comparisons per text; the first round alone gives every text two"),
        (args.top is None or args.top >= 1, "--top takes 1 or more"),
        (args.concurrency >= 1, "-j takes 1 or more concurrent calls"),
        (args.seed >= 0, "--seed takes 0 or more"),
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
        (not (args.scale and args.save_scale), "--save-scale saves the scale a run fits, and a --scale run fits none"),
        (args.anchors is None or bool(args.save_scale), "--anchors goes with --save-scale"),
        (args.anchors is None or args.anchors >= 2, "--anchors takes 2 or more"),
        (bool(args.scale) or not (args.se_target is not None or args.unordered), "--se-target and --unordered go with --scale"),
        (bool(args.scale or args.save_scale) or not args.any_model, "--any-model goes with --scale or --save-scale"),
        (args.se_target is None or (math.isfinite(args.se_target) and args.se_target > 0),
         "--se-target must be finite and greater than 0"),
        (not (args.unordered and (args.keep_order or args.top)), "--unordered cannot be combined with --keep-order or --top"),
        (not args.save_scale or os.path.isdir(os.path.dirname(os.path.abspath(args.save_scale))),
         f"--save-scale: no directory to write {args.save_scale} in"),
    ):
        if not valid:
            print(f"jsort: {message}", file=err)
            return 2
    if scale and given_max_chars not in (None, scale.max_chars):
        print(f"jsort: the scale was built showing Jev the first {scale.max_chars:,} characters of a text; this run "
              f"shows {args.max_chars:,}", file=err)
    show_stats = args.stats or (args.stats is None and err.isatty())

    def client() -> Jev:
        """The scale's API and model unless others were named. Made inside the running loop, where it is closed."""
        backend, key, model = client_for(scale, args.api, args.model)
        return Jev(key, backend, model=model, timeout=args.timeout, concurrency=args.concurrency,
                   cache=None if args.no_cache else Cache(), transport=transport)

    if scale and (args.keep_order or args.unordered):
        return stream(args, scale, files, client, show_stats, out, err)

    records, header, problems = read(files, args)
    placing = bool(scale)
    if args.csv and header and (clash := Writer(args, out, placing).clash(header)):
        print(f"jsort: the file already has a column named {clash}; choose another prefix with --name", file=err)
        return 2
    for message in problems[:MAX_ERRORS_SHOWN]:
        print(f"jsort: {message}", file=err)
    if len(problems) > MAX_ERRORS_SHOWN:
        print(f"jsort: and {len(problems) - MAX_ERRORS_SHOWN:,} more input errors", file=err)
    if args.jsonl and (clash := Writer(args, out, placing).clash(k for r in records if isinstance(r.data, dict) for k in r.data)):
        print(f"jsort: the records already have a field named {clash}; choose another prefix with --name", file=err)
        return 2
    if not records:
        if args.csv and header is not None and not args.json:
            write([], header, Placement.unscored(0) if placing else Ranking.unscored(0), [], args, out)     # a header with no rows is still a CSV
        return 2 if problems else 0

    texts = [r.text for r in records]
    shown = [t[:args.max_chars] for t in texts]
    truncated = sum(len(t) > args.max_chars for t in texts)
    jev = None
    t0 = time.perf_counter()

    def progress(done: int, total: int) -> None:
        if show_stats and err.isatty():
            print(f"\rjsort: {done:,} of {total:,} texts placed" if placing else
                  f"\rjsort: {done:,} of about {total:,} comparisons", end="", file=err, flush=True)

    if placing:
        anchored = {a.text for a in scale.anchors}
        if not any(t.strip() and t not in anchored for t in shown):
            work = None       # nothing to ask, so no key is needed either: an anchor's score is in the file
        else:
            async def work() -> Ranking:
                nonlocal jev
                jev = client()
                try:
                    return await aplace(texts, scale, jev, per_item=args.per_item, se_target=args.se_target,
                                        seed=args.seed, budget=args.budget, max_chars=args.max_chars,
                                        concurrency=args.concurrency, any_model=args.any_model, progress=progress)
                finally:
                    await jev.close()
    elif len({t for t in shown if t.strip()}) < 2:
        work = None           # nothing to compare, so no key is needed either
    else:
        try:
            backend, key = resolve_backend(args.api)
        except JevFatal as e:
            print(f"jsort: {e}", file=err)
            return 2
        jev = Jev(key, backend, model=args.model, timeout=args.timeout, concurrency=args.concurrency,
                  cache=None if args.no_cache else Cache(), transport=transport)

        async def work() -> Ranking:
            try:
                return await arank(texts, description, jev, per_item=args.per_item, top=args.top, lowest=args.reverse,
                                   seed=args.seed, budget=args.budget, max_chars=args.max_chars,
                                   concurrency=args.concurrency, progress=progress)
            finally:
                await jev.close()

    if work is None and placing:
        known = {a.text: a for a in scale.anchors}
        ranking = collect([(known[t].score, known[t].se, known[t].comparisons, 0) if t in known
                           else (math.nan, math.nan, 0, 0) for t in shown], scale)
    elif work is None:
        ranking = Ranking.unscored(len(texts))
    else:
        try:
            ranking = asyncio.run(work())
        except (JevFatal, ScaleError) as e:     # no key for the scale's API, or a model the scale was not built with
            print(f"jsort: {e}", file=err)
            return 2
        except KeyboardInterrupt:
            print(f"\njsort: interrupted; {jev.meter.summary()}" if jev else "\njsort: interrupted", file=err)
            return 130
        if show_stats and err.isatty():
            print("\r\033[K", end="", file=err)

    verb = "placed" if placing else "sorted"
    if ranking.fatal:
        print(f"jsort: {ranking.fatal}", file=err)
        if not ranking.asked:
            return 2
        # Answers that were paid for before the API refused are still worth a sort.
        print(f"jsort: {verb} on the {ranking.asked:,} comparisons answered before that", file=err)
    for message in ranking.errors[:MAX_ERRORS_SHOWN]:
        print(f"jsort: a comparison failed: {message}", file=err)
    if len(ranking.errors) > MAX_ERRORS_SHOWN:
        print(f"jsort: and {len(ranking.errors) - MAX_ERRORS_SHOWN:,} more failed comparisons", file=err)

    saved = None
    if args.save_scale:
        try:
            saved = ranking.scale(args.anchors or DEFAULT_ANCHORS, field=args.field, any_model=args.any_model,
                                  unit="whole" if args.whole else "para" if args.para else "field" if args.field else "line")
            saved.save(args.save_scale)
        except (ScaleError, OSError) as e:
            saved = None
            print(f"jsort: {args.save_scale} was not written: {getattr(e, 'strerror', None) or e}", file=err)

    order = list(range(len(records))) if args.keep_order else ranking.order(args.reverse)
    if args.top:
        if jev is not None or placing:   # failed comparisons are unranked; a singleton or all-identical input is already ordered
            order = [i for i in order if not math.isnan(ranking.score[i])]
        else:
            order = [i for i in order if shown[i].strip()]
        order = order[:args.top]
    try:
        write(records, header, ranking, order, args, out)
    except BrokenPipeError:
        pass

    unscored = sum(math.isnan(s) and bool(t.strip()) for s, t in zip(ranking.score, texts))
    beyond = (int((ranking.beyond > 0).sum()), int((ranking.beyond < 0).sum())) if placing else (0, 0)
    closing_notes(args, scale, truncated=truncated, ragged=sum(1 for r in records if r.data.get(None)) if args.csv else 0,
                  over_budget=ranking.over_budget, asked=ranking.asked,
                  unscored=unscored if jev is not None or placing else 0, beyond=beyond,
                  unverified=ranking.unverified if placing else 0, err=err)
    if ranking.reliability is not None and ranking.reliability < SHAKY:
        print(f"jsort: reliability {ranking.reliability:.2f}: two halves of the comparisons give different orders. "
              "Raise -k, or reword the description so that any two of these texts can be compared on it", file=err)
    if saved is not None and show_stats:
        print(f"jsort: saved {len(saved.anchors):,} anchors from {_number(saved.span[0], 2)} to "
              f"{_number(saved.span[1], 2)} in {args.save_scale}", file=err)
    if show_stats and jev is not None:
        line = (f"{len(records):,} texts placed against {len(scale.anchors):,} anchors, {ranking.asked:,} comparisons; "
                f"{jev.meter.summary()}" if placing else summary(records, ranking, jev))
        print(f"jsort: {line}; {time.perf_counter() - t0:.1f}s", file=err)
    failed = problems or ranking.errors or ranking.over_budget or ranking.fatal or (args.save_scale and saved is None)
    return 2 if failed else 0


def stream(args, scale: Scale, files: list[str], client, show_stats: bool, out, err) -> int:
    """--scale with --keep-order or --unordered: read as the input arrives and print each text once it is placed.

    A placed text's score owes nothing to the rest of the input, so nothing has to wait for the end of it.
    The shape is jgrep's: a thread reads, -j bounds the texts in hand, and with ordered output a slot
    stays taken until its text has printed, so a slow first text cannot let the rest run ahead.
    """
    writer = Writer(args, out, placing=True)
    known = {a.text: a for a in scale.anchors}
    s = {"next": 0, "seen": 0, "problems": 0, "fatal": None, "broken_pipe": False, "truncated": 0, "ragged": 0,
         "unscored": 0, "above": 0, "below": 0, "jev": None, "placer": None}
    t0 = time.perf_counter()

    def complain(message: str) -> None:
        s["problems"] += 1
        if s["problems"] <= MAX_ERRORS_SHOWN:
            print(f"jsort: {message}", file=err)

    async def go() -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency)
        # A slot covers both a text being placed and its result waiting for ordered output.
        sem = asyncio.Semaphore(args.concurrency)
        stop, halt = threading.Event(), asyncio.Event()
        finished: dict[int, tuple] = {}
        tasks: set[asyncio.Task] = set()
        feeding = {"put": None}

        def feed() -> None:
            def enqueue(item) -> bool:
                if stop.is_set():
                    return False
                put = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
                feeding["put"] = put
                # Stop may race with creation of the pending put. Either the consumer
                # or this check must cancel it so a full queue cannot strand a reader.
                if stop.is_set():
                    put.cancel()
                put.result()
                return not stop.is_set()

            try:
                for item in read_records(files, args, stop):
                    if not enqueue(item):
                        return
            except Exception as e:
                if not stop.is_set():
                    try:
                        enqueue(f"input reader: {type(e).__name__}: {e}")
                    except (CancelledError, RuntimeError):
                        pass
            finally:
                # Even an unexpected reader failure must wake the consumer.
                if not stop.is_set():
                    try:
                        enqueue(None)
                    except (CancelledError, RuntimeError):
                        pass

        def emit(rec: Record, score: float, se: float, n: int, beyond: int) -> None:
            s["seen"] += 1
            s["unscored"] += math.isnan(score) and bool(rec.text[:args.max_chars].strip())
            if beyond:
                s["above" if beyond > 0 else "below"] += 1
                if s["above"] + s["below"] <= MAX_ERRORS_SHOWN:   # a live stream has no end at which to say so
                    low, high = scale.span
                    print(f"jsort: {rec.file}:{rec.lineno}: " + (f"above every anchor (the highest is {_number(high, 2)})"
                          if beyond > 0 else f"below every anchor (the lowest is {_number(low, 2)})")
                          + f"; its score of {_number(score, 2)} is an extrapolation", file=err)
            try:
                writer.text(rec, None, score, se, n, beyond)
                out.flush()
            except BrokenPipeError:
                s["broken_pipe"] = True
                halt.set()

        def deliver(seq: int, rec: Record, result: tuple) -> None:
            if args.unordered:
                sem.release()
                return None if s["broken_pipe"] or s["fatal"] else emit(rec, *result)
            finished[seq] = (rec, *result)
            while s["next"] in finished and not (s["broken_pipe"] or s["fatal"]):
                emit(*finished.pop(s["next"]))
                s["next"] += 1
                sem.release()

        async def judge(seq: int, rec: Record) -> None:
            s["truncated"] += len(rec.text) > args.max_chars
            result, shown = (math.nan, math.nan, 0, 0), rec.text[:args.max_chars]
            try:
                if shown in known:              # an anchor's score is in the file
                    result = (known[shown].score, known[shown].se, known[shown].comparisons, 0)
                elif shown.strip():
                    if s["placer"] is None:     # the first text worth asking about is what needs a key
                        s["jev"] = s["jev"] or client()
                        s["placer"] = Placer(scale, s["jev"], per_item=args.per_item, se_target=args.se_target,
                                             seed=args.seed, budget=args.budget, max_chars=args.max_chars,
                                             concurrency=args.concurrency, any_model=args.any_model)
                    result = await s["placer"].place(rec.text)
            except (JevFatal, ScaleError) as e:
                s["fatal"] = s["fatal"] or str(e)
            except Exception as e:
                # Every text needs a result so one failure cannot leave a permanent gap in ordered output.
                complain(f"{rec.file}:{rec.lineno}: {type(e).__name__}: {e}")
            placer = s["placer"]
            if placer is not None and placer.fatal:
                s["fatal"] = s["fatal"] or placer.fatal
            deliver(seq, rec, result)
            if s["fatal"] or (placer is not None and placer.over_budget):
                halt.set()

        def completed(task: asyncio.Task) -> None:
            tasks.discard(task)
            if not task.cancelled() and (error := task.exception()) is not None:
                s["fatal"] = s["fatal"] or f"{type(error).__name__}: {error}"
                halt.set()

        threading.Thread(target=feed, daemon=True).start()
        halted = asyncio.ensure_future(halt.wait())
        seq = 0
        try:
            while not halt.is_set():
                get = asyncio.ensure_future(queue.get())
                await asyncio.wait({get, halted}, return_when=asyncio.FIRST_COMPLETED)
                if not get.done():
                    get.cancel()
                    break
                item = get.result()
                if item is None:
                    break
                if isinstance(item, str):
                    complain(item)
                    continue
                if isinstance(item, list):
                    if clash := writer.clash(item):
                        s["fatal"] = f"the file already has a column named {clash}; choose another prefix with --name"
                        break
                    writer.start(item)
                    continue
                if args.jsonl and isinstance(item.data, dict) and (clash := writer.clash(item.data)):
                    s["fatal"] = f"the records already have a field named {clash}; choose another prefix with --name"
                    break
                s["ragged"] += bool(args.csv and item.data.get(None))
                slot = asyncio.ensure_future(sem.acquire())
                await asyncio.wait({slot, halted}, return_when=asyncio.FIRST_COMPLETED)
                if halt.is_set():
                    slot.cancel()
                    await asyncio.gather(slot, return_exceptions=True)
                    break
                task = asyncio.create_task(judge(seq, item))
                seq += 1
                tasks.add(task)
                task.add_done_callback(completed)

            stop.set()
            if feeding["put"] is not None:
                feeding["put"].cancel()
            # At the budget the texts in hand are placed on what they have and printed; nothing new is asked.
            # A refusal or a closed pipe ends the printing, so what is still in the air is dropped.
            pending = list(tasks)
            if s["fatal"] or s["broken_pipe"]:
                for t in pending:
                    t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            stop.set()
            halted.cancel()
            await asyncio.gather(halted, return_exceptions=True)
            if s["jev"] is not None:
                await s["jev"].close()

    try:
        asyncio.run(go())
    except KeyboardInterrupt:
        print(f"\njsort: interrupted; {s['jev'].meter.summary()}" if s["jev"] else "\njsort: interrupted", file=err)
        return 130

    placer = s["placer"]
    if s["problems"] > MAX_ERRORS_SHOWN:
        print(f"jsort: and {s['problems'] - MAX_ERRORS_SHOWN:,} more input errors", file=err)
    if s["fatal"]:
        print(f"jsort: {s['fatal']}", file=err)
    errors = placer.errors if placer else []
    for message in errors[:MAX_ERRORS_SHOWN]:
        print(f"jsort: a comparison failed: {message}", file=err)
    if len(errors) > MAX_ERRORS_SHOWN:
        print(f"jsort: and {len(errors) - MAX_ERRORS_SHOWN:,} more failed comparisons", file=err)
    closing_notes(args, scale, truncated=s["truncated"], ragged=s["ragged"], over_budget=bool(placer and placer.over_budget),
                  asked=placer.asked if placer else 0, unscored=s["unscored"], beyond=(s["above"], s["below"]),
                  unverified=placer.unverified if placer else 0, err=err)
    if show_stats and placer is not None:
        print(f"jsort: {s['seen']:,} texts placed against {len(scale.anchors):,} anchors, {placer.asked:,} comparisons; "
              f"{s['jev'].meter.summary()}; {time.perf_counter() - t0:.1f}s", file=err)
    return 2 if s["problems"] or s["fatal"] or errors or (placer and placer.over_budget) else 0


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
