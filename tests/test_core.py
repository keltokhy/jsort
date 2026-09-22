"""Cache scoping, per-request charge attribution, and who-answered provenance. No network."""

import asyncio

import httpx
import pytest

from jevkit_runtime import AnswerStore, Backend, Client, JevError
from jsort.engine import question


def fixture(name="openrouter", url="https://fixture.invalid/decisions", model="jev-latest"):
    return Backend(name, url, model, key="test-key")


def test_cache_is_scoped_to_the_endpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("JEV_URL", raising=False)
    cache = AnswerStore(tmp_path / "answers.sqlite")
    calls = []

    async def fake(request):
        calls.append(request.url.host)
        probability = 0.9 if request.url.host == "first.example" else 0.1
        return httpx.Response(200, json={"answers": {"q": {"noul": probability}}, "usage": {"cost": 0.01}})

    async def go():
        for host, expected in [("first", 0.9), ("second", 0.1), ("first", 0.9), ("second", 0.1)]:
            jev = Client(fixture("gateway", f"https://{host}.example/decisions"), store=cache,
                      transport=httpx.MockTransport(fake))
            try:
                answers = await jev.ask({"A": "a", "B": "b"}, {"q": question("higher")})
                assert answers["q"]["noul"] == expected
            finally:
                await jev.close()
    try:
        asyncio.run(go())
        assert calls == ["first.example", "second.example"]
    finally:
        cache.close()


def test_only_the_request_owner_is_charged_and_cache_hits_are_free(tmp_path):
    cache = AnswerStore(tmp_path / "answers.sqlite")
    charges = [[], [], []]

    async def fake(request):
        await asyncio.sleep(0)
        return httpx.Response(200, json={"answers": {"q": {"noul": 0.5}}, "usage": {"cost": 0.01}})

    async def go():
        jev = Client(fixture(), store=cache, transport=httpx.MockTransport(fake))
        try:
            state, questions = {"A": "a", "B": "b"}, {"q": question("higher")}
            await asyncio.gather(*(jev.ask(state, questions, on_cost=charges[i].append) for i in range(2)))
            await jev.ask(state, questions, on_cost=charges[2].append)
            assert sum(map(sum, charges)) == pytest.approx(jev.meter.cost)
            assert jev.meter.calls == 1 and jev.meter.max_call_cost == 0.01
        finally:
            await jev.close()
    try:
        asyncio.run(go())
        assert charges == [[0.01], [], []]
    finally:
        cache.close()


def test_a_billed_invalid_answer_still_reports_its_charge():
    charges = []

    async def go():
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
            "answers": {"q": {"noul": 2}}, "usage": {"cost": 0.01},
        }))
        jev = Client(fixture(), transport=transport)
        try:
            with pytest.raises(JevError):
                await jev.ask({"A": "a", "B": "b"}, {"q": question("higher")}, on_cost=charges.append)
        finally:
            await jev.close()
    asyncio.run(go())
    assert charges == [0.01]


def test_who_answered_is_kept_beside_the_answers(tmp_path):
    async def fake(request):
        return httpx.Response(200, json={"model": "jev-1.13", "answers": {"q": {"noul": 0.75}}, "usage": {"cost": 0.01}})

    async def go():
        jev = Client(fixture("gateway", "https://gateway.example/decisions"), store=AnswerStore(tmp_path / "answers.sqlite"),
                  transport=httpx.MockTransport(fake))
        try:
            state, q = {"A": "old a", "B": "old b"}, question("higher")
            jev.store.put(jev.key(state, q), {"noul": 0.25})          # an answer nobody attributed
            old = await jev.ask(state, {"q": q})
            assert old["q"] == {"noul": 0.25}
            assert old.origins["q"]["source"] == "cache" and old.origins["q"].get("resolved_model") is None
            fresh = {"A": "new a", "B": "new b"}
            new = await jev.ask(fresh, {"q": q})
            assert new["q"] == {"noul": 0.75}
            origin = new.origins["q"]
            assert (origin["source"], origin["resolved_model"], origin["provider"]) == ("api", "jev-1.13", "gateway")
            again = await jev.ask(fresh, {"q": q})
            assert again["q"] == {"noul": 0.75}
            assert (again.origins["q"]["source"], again.origins["q"]["resolved_model"]) == ("cache", "jev-1.13")
            assert jev.meter.calls == 1
            assert jev.store.entry(jev.key(fresh, q)).metadata["provider"] == "gateway"
        finally:
            await jev.close()
            jev.store.close()

    asyncio.run(go())
