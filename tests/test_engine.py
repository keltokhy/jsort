"""Budget and concurrency regressions against an offline Decisions endpoint."""

import asyncio
import json

import httpx
import pytest

import math

from jevkit_runtime import Backend, Budget, Client
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


def run(endpoint, budget=None, **options):
    async def go():
        jev = Client(Backend("openrouter", "https://fixture.invalid/decisions", "jev-1.13.0", key="test-key"),
                     budget=budget or Budget(), transport=httpx.MockTransport(endpoint))
        try:
            result = await arank([str(i) for i in range(40)], "higher", jev, **options)
            return result, jev.meter.cost
        finally:
            await jev.close()
    return asyncio.run(go())


def test_the_first_charge_is_learned_alone_and_no_request_goes_past_the_budget():
    endpoint = ChargedEndpoint(0.01)
    result, spent = run(endpoint, budget=Budget(0.02))
    assert result.over_budget and result.asked == endpoint.calls == 2
    assert spent == pytest.approx(0.02)


def test_budget_keeps_parallelism_when_there_is_room():
    endpoint = ChargedEndpoint(0.001)
    result, spent = run(endpoint, budget=Budget(1.0), per_item=2)
    assert not result.over_budget and result.asked == 40
    assert spent == pytest.approx(0.04) and endpoint.peak > 1


@pytest.mark.parametrize("configured,expected", [(None, 1.0), ("0.4", 0.4)])
def test_rank_uses_the_default_or_environment_budget(monkeypatch, tmp_path, configured, expected):
    import jsort

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for name in ("TYPESAFE_API_KEY", "JEV_API", "JEV_BUDGET"):
        monkeypatch.delenv(name, raising=False)
    if configured is not None:
        monkeypatch.setenv("JEV_BUDGET", configured)
    endpoint = ChargedEndpoint(0.2)
    result = jsort.rank([str(i) for i in range(40)], "higher", transport=httpx.MockTransport(endpoint), cache=False)
    assert result.over_budget and endpoint.calls * 0.2 == pytest.approx(expected)


def test_an_unlimited_budget_spends_what_the_run_takes():
    endpoint = ChargedEndpoint(0.2)
    result, spent = run(endpoint, budget=Budget(math.inf), per_item=2)
    assert not result.over_budget and result.asked == 40
    assert spent == pytest.approx(8.0) and endpoint.peak > 1


def test_a_run_can_bring_its_own_budget_to_a_shared_client():
    async def go():
        endpoint = ChargedEndpoint(0.01)
        jev = Client(Backend("openrouter", "https://fixture.invalid/decisions", "jev-1.13.0", key="test-key"),
                     transport=httpx.MockTransport(endpoint))
        try:
            for _ in range(5):
                result = await arank([str(i) for i in range(10)], "higher", jev, budget=Budget(0.02))
                assert result.over_budget and result.asked == 2
            assert jev.meter.cost == pytest.approx(0.10)
        finally:
            await jev.close()
    asyncio.run(go())


def test_concurrent_rankings_have_independent_budgets():
    async def go():
        endpoint = ChargedEndpoint(0.01)
        jev = Client(Backend("openrouter", "https://fixture.invalid/decisions", "jev-1.13.0", key="test-key"),
                     transport=httpx.MockTransport(endpoint))
        try:
            results = await asyncio.gather(*(
                arank([f"{prefix}{i}" for i in range(10)], "higher", jev, budget=Budget(0.02))
                for prefix in ("a", "b")
            ))
            assert all(r.over_budget and r.asked == 2 for r in results)
            assert endpoint.calls == 4 and endpoint.peak > 1
            assert jev.meter.cost == pytest.approx(0.04)
        finally:
            await jev.close()
    asyncio.run(go())


@pytest.mark.parametrize("options", [
    {"top": -4}, {"top": 0}, {"top": 1.5}, {"per_item": 1}, {"per_item": 2.5},
    {"max_chars": 0}, {"max_chars": 1.5}, {"concurrency": 1.5},
])
def test_invalid_ranking_options_fail_before_any_request(options):
    endpoint = ChargedEndpoint(0.01)
    with pytest.raises(ValueError):
        run(endpoint, **options)
    assert endpoint.calls == 0


def test_zero_concurrency_raises_instead_of_hanging():
    async def go():
        endpoint = ChargedEndpoint(0.01)
        jev = Client(Backend("openrouter", "https://fixture.invalid/decisions", "jev-latest", key="test-key"), transport=httpx.MockTransport(endpoint))
        try:
            with pytest.raises(ValueError, match="concurrency"):
                await asyncio.wait_for(arank(["a", "b"], "higher", jev, concurrency=0), timeout=0.1)
            assert endpoint.calls == 0
        finally:
            await jev.close()
    asyncio.run(go())


@pytest.mark.parametrize("budget", [-1, float("nan"), True])
def test_invalid_budgets_fail_before_any_request(budget):
    with pytest.raises(ValueError, match="budget"):
        Budget(budget)
