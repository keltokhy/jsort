"""jsort compatibility adapter for the shared JevKit implementation.

Prompts, cache identity, answer reuse, and budget policy retain their existing contracts.
Transport, configuration, storage, validation, and usage parsing come from jevkit_core.
"""

from __future__ import annotations

from dataclasses import dataclass

from jevkit_core import (
    FATAL,
    RETRYABLE,
    PRICE_PER_MTOK,
    AnswerCache,
    Backend as _Backend,
    DecisionClient,
    JevBudgetExceeded,
    JevError,
    JevFatal,
    Meter as _Meter,
    answer_provenance,
    backend_catalog,
    cache_path,
    config_dir,
    digest,
    parse_usage,
    resolve_backend as _resolve_backend,
    validate_answer as _validate_answer,
)


Backend = _Backend


BACKENDS = backend_catalog("typesafe", "openrouter", "gateway")


def resolve_backend(name: str | None = None):
    return _resolve_backend(BACKENDS, name)


class Cache(AnswerCache):
    """Keep the existing cache identity while sharing its storage implementation."""

    metadata = True

    @staticmethod
    def key(model: str, state, question: dict, *, endpoint: str) -> str:
        return digest([endpoint, model, state, question])


@dataclass
class Meter(_Meter):
    max_call_cost: float = 0.0


class Jev(DecisionClient):
    def __init__(
        self,
        key: str,
        backend: Backend | str = "openrouter",
        *,
        model: str | None = None,
        timeout: float = 15.0,
        attempts: int = 4,
        concurrency: int = 32,
        cache: Cache | None = None,
        transport=None,
    ):
        backend = BACKENDS[backend] if isinstance(backend, str) else backend
        meter = Meter()
        super().__init__(
            key,
            backend,
            model=model,
            timeout=timeout,
            attempts=attempts,
            concurrency=concurrency,
            cache=cache,
            transport=transport,
            meter=meter,
        )

    async def ask(
        self,
        state,
        questions: dict[str, dict],
        *,
        on_cost=None,
        provenance: dict | None = None,
    ) -> dict[str, dict]:
        """Answer every question about one state. Only questions missing from the cache are sent.

        on_cost receives charges for requests started by this call; cached and shared answers are free.
        provenance, if given, receives for each question who answered it: `resolved_model` is the model
        the API named, or None where that was never recorded, and `source` is api, cache or shared.
        """
        keys = {
            qid: Cache.key(self.model, state, q, endpoint=self.url)
            for qid, q in questions.items()
        }
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

        task, started = self.share_request(
            (keys[qid] for qid in misses),
            lambda: self._call(state, misses, on_cost=on_cost),
        )
        source = "api" if started else "shared"
        by_key, metadata = await task
        if provenance is not None:
            provenance.update({qid: metadata | {"source": source} for qid in misses})
        return answers | {qid: by_key[keys[qid]] for qid in misses}

    def _record(
        self, state, questions: dict, data: dict, seconds: float, *, on_cost=None
    ) -> tuple[dict[str, dict], dict]:
        usage = parse_usage(data.get("usage"), price_per_mtok=PRICE_PER_MTOK)

        def charge(cost):
            self.meter.max_call_cost = max(self.meter.max_call_cost, cost)
            if on_cost is not None:
                on_cost(cost)

        self.meter.record(
            usage, seconds, model=data.get("model") or self.model, on_cost=charge
        )
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
        metadata = answer_provenance(
            provider=self.backend.name,
            requested_model=self.model,
            resolved_model=data.get("model"),
        )
        # Validate the entire response before storing any part of it.
        if self.cache:
            for k, answer in out.items():
                self.cache.put(k, answer, metadata=metadata)
        return out, metadata


__all__ = [
    "BACKENDS",
    "Backend",
    "Cache",
    "Meter",
    "Jev",
    "JevError",
    "JevFatal",
    "JevBudgetExceeded",
    "PRICE_PER_MTOK",
    "RETRYABLE",
    "FATAL",
    "cache_path",
    "config_dir",
    "resolve_backend",
]
