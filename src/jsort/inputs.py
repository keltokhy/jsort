"""Reading what is to be sorted. Like sort(1), everything is read before anything is printed."""

from __future__ import annotations

import csv
import io
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass

STDIN = "(standard input)"
csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))   # the default of 131,072 characters is smaller than --max-chars allows


@dataclass
class Record:
    text: str                    # what Jev is shown
    original: str                # what is printed: the line, the paragraph, the JSON line or the file name
    file: str
    lineno: int
    data: dict | None = None     # the parsed record, for --csv and --jsonl


class InputError(Exception):
    pass


def lookup(obj, path: str):
    """A JSON field by name; failing that, by dotted path such as event.message or events.0.message."""
    if isinstance(obj, dict) and path in obj:
        return obj[path]
    for part in path.split("."):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        elif isinstance(obj, list) and part.isdigit() and int(part) < len(obj):
            obj = obj[int(part)]
        else:
            raise KeyError(path)
    return obj


def as_text(value) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


@contextmanager
def _open(name: str):
    if name == "-":
        stream = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8-sig", errors="replace", newline="")
        try:
            yield stream
        finally:
            stream.detach()   # the caller owns stdin's buffer
    else:
        with open(name, encoding="utf-8-sig", errors="replace", newline="") as stream:
            yield stream     # -sig drops the byte-order mark Excel writes


def read(files: list[str], args) -> tuple[list[Record], list[str] | None, list[str]]:
    """Records, the CSV header if there is one, and the problems met along the way."""
    records: list[Record] = []
    header: list[str] | None = None
    problems: list[str] = []
    for name in files or ["-"]:
        label = STDIN if name == "-" else name
        try:
            with _open(name) as f:
                if args.csv:
                    reader = csv.DictReader(f)
                    if reader.fieldnames is None:
                        continue
                    if len(set(reader.fieldnames)) != len(reader.fieldnames):
                        raise InputError("its header repeats a column name, so rows could not be returned intact")
                    if args.field not in reader.fieldnames:
                        raise InputError(f"no column named {args.field!r}")
                    if header is not None and list(reader.fieldnames) != header:
                        raise InputError("its header differs from the first file's; sort files with one header at a time")
                    header = list(reader.fieldnames)
                    for row in reader:
                        records.append(Record(as_text(row.get(args.field)), "", label, reader.line_num, row))
                elif args.jsonl:
                    for lineno, line in enumerate(f, 1):
                        line = line.rstrip("\r\n")
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                            if not isinstance(obj, dict):
                                problems.append(f"{label}:{lineno}: expected a JSON object")
                                continue
                            records.append(Record(as_text(lookup(obj, args.field)), line, label, lineno, obj))
                        except ValueError:
                            problems.append(f"{label}:{lineno}: not valid JSON")
                        except KeyError:
                            problems.append(f"{label}:{lineno}: no field {args.field!r}")
                elif args.whole:
                    records.append(Record(f.read(), label, label, 1))
                elif args.para:
                    block: list[str] = []
                    start = 1
                    for lineno, line in enumerate(f, 1):
                        line = line.rstrip("\r\n")
                        if line.strip():
                            if not block:
                                start = lineno
                            block.append(line)
                        elif block:
                            records.append(Record("\n".join(block), "\n".join(block), label, start))
                            block = []
                    if block:
                        records.append(Record("\n".join(block), "\n".join(block), label, start))
                else:
                    for lineno, line in enumerate(f, 1):
                        line = line.rstrip("\r\n")
                        if line.strip():
                            records.append(Record(line, line, label, lineno))
        except (OSError, InputError, csv.Error) as e:
            problems.append(f"{label}: {e.strerror if isinstance(e, OSError) and e.strerror else e}")
    return records, header, problems
