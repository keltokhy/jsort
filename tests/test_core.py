"""Cache isolation and per-request charge attribution. No network."""

import asyncio

import httpx
import pytest

from jsort.core import Backend, Cache, Jev, JevError
from jsort.engine import question


def test_cache_is_scoped_to_the_endpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("JEV_URL", raising=False)
    cache = Cache(tmp_path / "answers.sqlite")
    calls = []

    async def fake(request):
        calls.append(request.url.host)
        probability = 0.9 if request.url.host == "first.example" else 0.1
        return httpx.Response(200, json={"answers": {"q": {"noul": probability}}, "usage": {"cost": 0.01}})

    async def go():
        for host, expected in [("first", 0.9), ("second", 0.1), ("first", 0.9), ("second", 0.1)]:
            backend = Backend("gateway", f"https://{host}.example/decisions", "jev-latest", "UNUSED")
            jev = Jev("test-key", backend, model="jev-latest", cache=cache, transport=httpx.MockTransport(fake))
            try:
                answers = await jev.ask({"A": "a", "B": "b"}, {"q": question("higher")})
                assert answers["q"]["noul"] == expected
            finally:
                await jev.close()
    try:
        asyncio.run(go())
        assert calls == ["first.example", "second.example"]
    finally:
        cache.db.close()


def test_only_the_request_owner_is_charged_and_cache_hits_are_free(tmp_path):
    cache = Cache(tmp_path / "answers.sqlite")
    charges = [[], [], []]

    async def fake(request):
        await asyncio.sleep(0)
        return httpx.Response(200, json={"answers": {"q": {"noul": 0.5}}, "usage": {"cost": 0.01}})

    async def go():
        jev = Jev("test-key", cache=cache, transport=httpx.MockTransport(fake))
        try:
            state, questions = {"A": "a", "B": "b"}, {"q": question("higher")}
            await asyncio.gather(*(jev.ask(state, questions, on_cost=charges[i].append) for i in range(2)))
            await jev.ask(state, questions, on_cost=charges[2].append)
            assert sum(map(sum, charges)) == pytest.approx(jev.meter.cost)
            assert jev.meter.calls == 1
        finally:
            await jev.close()
    try:
        asyncio.run(go())
        assert charges == [[0.01], [], []]
    finally:
        cache.db.close()


def test_a_billed_invalid_answer_still_reports_its_charge():
    charges = []

    async def go():
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
            "answers": {"q": {"noul": 2}}, "usage": {"cost": 0.01},
        }))
        jev = Jev("test-key", transport=transport)
        try:
            with pytest.raises(JevError):
                await jev.ask({"A": "a", "B": "b"}, {"q": question("higher")}, on_cost=charges.append)
        finally:
            await jev.close()
    asyncio.run(go())
    assert charges == [0.01]
