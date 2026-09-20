"""Budget and concurrency regressions against an offline Decisions endpoint."""

import asyncio
import json

import httpx
import pytest

from jsort.core import Jev
from jsort.engine import arank


class ChargedEndpoint:
    def __init__(self, cost):
        self.cost = cost
        self.calls = self.active = self.peak = 0

    async def __call__(self, request):
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0)
            body = json.loads(request.content)
            return httpx.Response(200, json={
                "answers": {qid: {"noul": 0.5} for qid in body["questions"]},
                "usage": {"cost": self.cost},
            })
        finally:
            self.active -= 1


def run(endpoint, **options):
    async def go():
        jev = Jev("test-key", transport=httpx.MockTransport(endpoint))
        try:
            result = await arank([str(i) for i in range(40)], "higher", jev, **options)
            return result, jev.meter.cost
        finally:
            await jev.close()
    return asyncio.run(go())


def test_concurrent_requests_respect_the_remaining_budget():
    endpoint = ChargedEndpoint(0.01)
    result, spent = run(endpoint, budget=0.02)
    assert result.over_budget and result.asked == endpoint.calls == 2
    assert spent == pytest.approx(0.02)


def test_budget_keeps_parallelism_when_there_is_room():
    endpoint = ChargedEndpoint(0.001)
    result, spent = run(endpoint, budget=1.0, per_item=2)
    assert not result.over_budget and result.asked == 40
    assert spent == pytest.approx(0.04) and endpoint.peak > 1


@pytest.mark.parametrize("configured,expected", [(None, 1.0), ("0.4", 0.4)])
def test_arank_uses_the_default_or_environment_budget(monkeypatch, configured, expected):
    monkeypatch.delenv("JSORT_BUDGET", raising=False)
    if configured is not None:
        monkeypatch.setenv("JSORT_BUDGET", configured)
    endpoint = ChargedEndpoint(0.2)
    result, spent = run(endpoint)
    assert result.over_budget and spent == pytest.approx(expected)


def test_explicit_zero_budget_is_unlimited(monkeypatch):
    monkeypatch.setenv("JSORT_BUDGET", "0.4")
    endpoint = ChargedEndpoint(0.2)
    result, spent = run(endpoint, budget=0, per_item=2)
    assert not result.over_budget and result.asked == 40
    assert spent == pytest.approx(8.0) and endpoint.peak > 1


def test_budget_is_per_run_when_reusing_a_client():
    async def go():
        endpoint = ChargedEndpoint(0.01)
        jev = Jev("test-key", transport=httpx.MockTransport(endpoint))
        try:
            for _ in range(5):
                result = await arank([str(i) for i in range(10)], "higher", jev, budget=0.02)
                assert result.over_budget and result.asked == 2
            assert jev.meter.cost == pytest.approx(0.10)
        finally:
            await jev.close()
    asyncio.run(go())


def test_zero_concurrency_raises_instead_of_hanging():
    async def go():
        endpoint = ChargedEndpoint(0.01)
        jev = Jev("test-key", transport=httpx.MockTransport(endpoint))
        try:
            with pytest.raises(ValueError, match="concurrency"):
                await asyncio.wait_for(arank(["a", "b"], "higher", jev, concurrency=0), timeout=0.1)
            assert endpoint.calls == 0
        finally:
            await jev.close()
    asyncio.run(go())


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf")])
def test_invalid_budgets_fail_before_any_request(budget):
    endpoint = ChargedEndpoint(0.01)
    with pytest.raises(ValueError, match="budget"):
        run(endpoint, budget=budget)
    assert endpoint.calls == 0
