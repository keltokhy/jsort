"""Client for TypeSafe's Jev decision model: two backends, an answer cache and a cost meter.

Jev can be reached through TypeSafe's own API or through OpenRouter. Both take one state and any
number of questions per call and return one typed answer per question. Answers are cached per
(endpoint, model, state, question), so packing questions into a call and rerunning a command are both cheap.

Based on the client shared by jgrep and jlink.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

RETRYABLE = {408, 429, 500, 502, 503, 504, 529}
FATAL = {401, 402, 403}
# TypeSafe's API reports tokens but not cost; OpenRouter reports both.
PRICE_PER_MTOK = float(os.environ.get("JEV_PRICE_PER_MTOK", 0.042))


class JevError(Exception):
    """One request failed; the rest of the run can continue."""


class JevFatal(Exception):
    """Nothing will work until the user fixes something, such as a bad key or no credits."""


@dataclass(frozen=True)
class Backend:
    name: str
    url: str
    model: str
    key_env: str
    # Set for a backend whose URL is not fixed: the variable (or a `<name>.url` config file) that names it.
    url_env: str | None = None

    @property
    def key_file(self) -> Path:
        return config_dir() / f"{self.name}.key"

    @property
    def url_file(self) -> Path:
        return config_dir() / f"{self.name}.url"

    def key(self) -> str | None:
        if os.environ.get(self.key_env):
            return os.environ[self.key_env].strip()
        return self.key_file.read_text().strip() if self.key_file.exists() else None

    def configured_url(self) -> str | None:
        """The URL to call, or None for a gateway whose URL was never given."""
        if self.url:
            return self.url
        if self.url_env and os.environ.get(self.url_env):
            return os.environ[self.url_env].strip()
        return self.url_file.read_text().strip() if self.url_file.exists() else None


# Order matters: with keys for several, the first one here is used.
BACKENDS = {
    "typesafe": Backend("typesafe", "https://api.typesafe.ai/v1/systemone", "jev-latest", "TYPESAFE_API_KEY"),
    "openrouter": Backend("openrouter", "https://openrouter.ai/api/alpha/decisions", "~typesafe/jev-latest",
                          "OPENROUTER_API_KEY"),
    # Anything that speaks System One and takes its own key: an LLM gateway such as LiteLLM or Ramp Router,
    # a corporate proxy, a mock. The URL is the full endpoint, for example https://gateway.example.com/v1/systemone.
    "gateway": Backend("gateway", "", "jev-latest", "JEV_GATEWAY_API_KEY", url_env="JEV_GATEWAY_URL"),
}


def config_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "jev"


def cache_path() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "jev" / "answers.sqlite"


def resolve_backend(name: str | None = None) -> tuple[Backend, str]:
    """The API to use and its key. A name (or JEV_API) wins; otherwise the first backend with a key."""
    name = name or os.environ.get("JEV_API")
    if name:
        if name not in BACKENDS:
            raise JevFatal(f"unknown API {name!r}; choose from {', '.join(BACKENDS)}")
        backend = BACKENDS[name]
        if not (key := backend.key()):
            raise JevFatal(f"no key for {name}. Set {backend.key_env} or put the key in {backend.key_file}")
        if backend.url_env and not backend.configured_url() and not os.environ.get("JEV_URL"):
            raise JevFatal(f"no URL for {name}. Set {backend.url_env} to the full System One endpoint "
                           f"(for example https://gateway.example.com/v1/systemone) or put it in {backend.url_file}")
        return backend, key
    for backend in BACKENDS.values():
        if (key := backend.key()) and (not backend.url_env or backend.configured_url()):
            return backend, key
    options = " or ".join(b.key_env for b in BACKENDS.values())
    raise JevFatal(f"no API key. Set {options}, or put a key in {config_dir()}/<api>.key")


class Cache:
    """Answers on disk, keyed on the endpoint, exact model, state and question.

    The file is shared with jgrep, jlink and jcol, so the `answers` table and its keys never change. Who gave
    an answer is kept beside it, in the `answer_metadata` table jlink introduced: joined on the key and on the
    time the answer was written, so an answer an older tool has since overwritten does not inherit what was
    recorded about the one before. An answer with no such row is simply of unknown origin.
    """

    def __init__(self, path: Path | None = None):
        path = path or cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None, check_same_thread=False)
        # WAL lets two tools in one pipeline share the file.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS answers "
                        "(key TEXT PRIMARY KEY, answer TEXT NOT NULL, at REAL NOT NULL) WITHOUT ROWID")
        self.db.execute("CREATE TABLE IF NOT EXISTS answer_metadata "
                        "(key TEXT PRIMARY KEY, at REAL NOT NULL, metadata TEXT NOT NULL) WITHOUT ROWID")

    @staticmethod
    def key(model: str, state, question: dict, *, endpoint: str) -> str:
        blob = json.dumps([endpoint, model, state, question], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        row = self.db.execute("SELECT answer FROM answers WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def get_entry(self, key: str) -> tuple[dict, dict] | None:
        """An answer and what was recorded about who gave it; {} for an answer that does not say."""
        row = self.db.execute("SELECT a.answer, m.metadata FROM answers a LEFT JOIN answer_metadata m "
                              "ON a.key = m.key AND a.at = m.at WHERE a.key = ?", (key,)).fetchone()
        return (json.loads(row[0]), json.loads(row[1]) if row[1] else {}) if row else None

    def put(self, key: str, answer: dict, *, metadata: dict | None = None) -> None:
        at = time.time()
        self.db.execute("BEGIN IMMEDIATE")      # the answer and who gave it are written together or not at all
        try:
            self.db.execute("INSERT OR REPLACE INTO answers VALUES (?, ?, ?)", (key, json.dumps(answer), at))
            if metadata is not None:
                self.db.execute("INSERT OR REPLACE INTO answer_metadata VALUES (?, ?, ?)", (key, at, json.dumps(metadata)))
            else:
                self.db.execute("DELETE FROM answer_metadata WHERE key = ?", (key,))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise


@dataclass
class Meter:
    calls: int = 0
    cached: int = 0
    retries: int = 0
    input_tokens: int = 0
    cost: float = 0.0
    model: str = ""  # the model the API says answered, which resolves aliases like jev-latest
    latencies: list[float] = field(default_factory=list)
    max_call_cost: float = 0.0

    def summary(self) -> str:
        parts = [f"{self.calls:,} calls, {self.cached:,} cached"]
        if self.retries:
            parts.append(f"{self.retries:,} retries")
        if self.calls:
            parts.append(f"{self.input_tokens:,} tokens")
            parts.append(f"${self.cost:.4f}")
        return "; ".join(parts)


class Jev:
    def __init__(self, key: str, backend: Backend | str = "openrouter", *, model: str | None = None,
                 timeout: float = 15.0, attempts: int = 4, concurrency: int = 32, cache: Cache | None = None,
                 transport=None):
        self.backend = BACKENDS[backend] if isinstance(backend, str) else backend
        self.model = model or os.environ.get("JEV_MODEL") or self.backend.model
        self.url = os.environ.get("JEV_URL") or self.backend.configured_url() or self.backend.url
        self.timeout, self.attempts, self.cache = timeout, attempts, cache
        self.meter = Meter()
        self._flights: dict[str, asyncio.Task] = {}
        self.http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {key}", "X-Title": "jev tools"},
            limits=httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4),
            transport=transport,
        )

    async def close(self) -> None:
        await self.http.aclose()

    async def ask(self, state, questions: dict[str, dict], *, on_cost=None, provenance: dict | None = None) -> dict[str, dict]:
        """Answer every question about one state. Only questions missing from the cache are sent.

        on_cost receives charges for requests started by this call; cached and shared answers are free.
        provenance, if given, receives for each question who answered it: `resolved_model` is the model
        the API named, or None where that was never recorded, and `source` is api, cache or shared.
        """
        keys = {qid: Cache.key(self.model, state, q, endpoint=self.url) for qid, q in questions.items()}
        answers, origins = {}, {}
        if self.cache:
            for qid, k in keys.items():
                if (hit := self.cache.get_entry(k)) is not None:
                    _validate_answer(qid, questions[qid], hit[0])
                    answers[qid], origins[qid] = hit[0], hit[1] | {"source": "cache"}
        misses = {qid: q for qid, q in questions.items() if qid not in answers}
        if provenance is not None:
            provenance.update(origins)
        if not misses:
            self.meter.cached += 1
            return answers

        # Identical requests already in the air share one call; logs repeat themselves a lot.
        flight = "|".join(sorted(keys[qid] for qid in misses))
        task = self._flights.get(flight)
        source = "api" if task is None else "shared"
        if task is None:
            task = asyncio.ensure_future(self._call(state, misses, on_cost=on_cost))
            self._flights[flight] = task
            task.add_done_callback(lambda _: self._flights.pop(flight, None))
        else:
            self.meter.cached += 1
        by_key, metadata = await task
        if provenance is not None:
            provenance.update({qid: metadata | {"source": source} for qid in misses})
        return answers | {qid: by_key[keys[qid]] for qid in misses}

    async def _call(self, state, questions: dict[str, dict], *, on_cost=None) -> tuple[dict[str, dict], dict]:
        """One request, retried inside a total time budget. Returns answers by cache key, and who gave them."""
        body = {"model": self.model, "state": state, "questions": questions}
        deadline = time.monotonic() + self.timeout
        last = "no attempt made"
        for attempt in range(self.attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            t0 = time.perf_counter()
            try:
                # HTTPX limits each network wait, not the whole response. Bound the
                # complete request, including a body that keeps arriving in small chunks.
                r = await asyncio.wait_for(
                    self.http.post(self.url, json=body, timeout=remaining), timeout=remaining)
            except asyncio.TimeoutError:
                last = "deadline exceeded"
                break
            except httpx.TransportError as e:
                last = type(e).__name__
            else:
                data = _json(r)
                if r.status_code == 200 and "answers" in data:
                    return self._record(state, questions, data, time.perf_counter() - t0, on_cost=on_cost)
                detail = _detail(data) or r.text[:200]
                if r.status_code in FATAL:
                    raise JevFatal(f"{self.backend.name} said {r.status_code}: {detail}")
                if r.status_code != 200 and r.status_code not in RETRYABLE:
                    raise JevError(f"HTTP {r.status_code}: {detail}")
                last = f"HTTP {r.status_code}"
            if attempt + 1 < self.attempts:
                self.meter.retries += 1
                pause = 0.2 * 2 ** attempt + random.random() * 0.1
                await asyncio.sleep(max(0.0, min(pause, deadline - time.monotonic())))
        raise JevError(f"gave up after {self.timeout:g}s ({last})")

    def _record(self, state, questions: dict, data: dict, seconds: float, *, on_cost=None) -> tuple[dict[str, dict], dict]:
        usage = data.get("usage") or {}
        tokens = usage.get("input_tokens") or 0
        cost = usage.get("cost")
        self.meter.calls += 1
        self.meter.input_tokens += tokens
        cost = tokens * PRICE_PER_MTOK / 1e6 if cost is None else cost
        self.meter.cost += cost
        self.meter.max_call_cost = max(self.meter.max_call_cost, cost)
        if on_cost is not None:
            on_cost(cost)
        self.meter.latencies.append(seconds)
        self.meter.model = data.get("model") or self.model
        answers = data["answers"]
        if not isinstance(answers, dict):
            raise JevError("invalid answers returned: expected an object")
        out = {}
        for qid, q in questions.items():
            if qid not in answers:
                raise JevError(f"no answer returned for question {qid!r}")
            _validate_answer(qid, q, answers[qid])
            k = Cache.key(self.model, state, q, endpoint=self.url)
            out[k] = answers[qid]
        # The model the API names, literally. The meter's falls back to the one requested; this must not, since a
        # saved scale vouches for it. The fields are jlink's, so either tool can read what the other recorded.
        resolved = data.get("model")
        metadata = {"version": 1, "provider": self.backend.name, "requested_model": self.model,
                    "resolved_model": resolved if isinstance(resolved, str) and resolved.strip() else None,
                    "answered_at": time.time()}
        # Validate the entire response before storing any part of it.
        if self.cache:
            for k, answer in out.items():
                self.cache.put(k, answer, metadata=metadata)
        return out, metadata


def _validate_answer(qid: str, question: dict, answer) -> None:
    if not isinstance(answer, dict):
        raise JevError(f"invalid answer returned for question {qid!r}: expected an object")
    if question.get("type") == "noul":
        p = answer.get("noul")
        if (isinstance(p, bool) or not isinstance(p, (int, float))
                or not math.isfinite(p) or not 0 <= p <= 1):
            raise JevError(f"invalid answer returned for question {qid!r}: noul must be a probability from 0 to 1")


def _json(r: httpx.Response) -> dict:
    try:
        data = r.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _detail(data: dict) -> str:
    """The human-readable part of an error body. OpenRouter nests it under `error`, TypeSafe under `detail`,
    and `detail` may itself be a string, an object or a list of validation problems."""
    found = data.get("error", data.get("detail"))
    if isinstance(found, list):
        found = "; ".join(_detail({"detail": item}) for item in found)
    elif isinstance(found, dict):
        found = found.get("message") or found.get("msg") or json.dumps(found)
    return " ".join(str(found or "").split())[:200]
