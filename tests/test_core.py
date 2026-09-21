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


def test_who_answered_is_kept_beside_the_answers_and_older_tools_are_undisturbed(tmp_path):
    import sqlite3

    path = tmp_path / "answers.sqlite"
    legacy = sqlite3.connect(path, isolation_level=None)          # a cache as jsort 0.1.3 and jgrep create and fill it
    legacy.execute("CREATE TABLE IF NOT EXISTS answers "
                   "(key TEXT PRIMARY KEY, answer TEXT NOT NULL, at REAL NOT NULL) WITHOUT ROWID")
    state, q = {"A": "old a", "B": "old b"}, question("higher")
    old_key = Cache.key("jev-latest", state, q, endpoint="https://gateway.example/decisions")
    legacy.execute("INSERT OR REPLACE INTO answers VALUES (?, ?, ?)", (old_key, '{"noul": 0.25}', 5.0))

    async def fake(request):
        return httpx.Response(200, json={"model": "jev-1.13", "answers": {"q": {"noul": 0.75}}, "usage": {"cost": 0.01}})

    async def go():
        backend = Backend("gateway", "https://gateway.example/decisions", "jev-latest", "UNUSED")
        jev = Jev("test-key", backend, cache=Cache(path), transport=httpx.MockTransport(fake))
        try:
            old, new, again = {}, {}, {}
            assert (await jev.ask(state, {"q": q}, provenance=old))["q"] == {"noul": 0.25}
            assert old["q"]["source"] == "cache" and old["q"].get("resolved_model") is None   # unknown stays unknown
            fresh = {"A": "new a", "B": "new b"}
            assert (await jev.ask(fresh, {"q": q}, provenance=new))["q"] == {"noul": 0.75}
            assert (new["q"]["source"], new["q"]["resolved_model"], new["q"]["provider"]) == ("api", "jev-1.13", "gateway")
            assert (await jev.ask(fresh, {"q": q}, provenance=again))["q"] == {"noul": 0.75}
            assert (again["q"]["source"], again["q"]["resolved_model"]) == ("cache", "jev-1.13") and jev.meter.calls == 1
            return Cache.key("jev-latest", fresh, q, endpoint="https://gateway.example/decisions")
        finally:
            await jev.close()
            jev.cache.db.close()

    new_key = asyncio.run(go())
    # The table older tools read is as it was: same three columns, and the new answer is there for them.
    assert [row[1] for row in legacy.execute("PRAGMA table_info(answers)")] == ["key", "answer", "at"]
    assert legacy.execute("SELECT answer FROM answers WHERE key = ?", (new_key,)).fetchone() == ('{"noul": 0.75}',)
    # An older tool overwrites that answer. What was recorded about the old one must not be read as the new one's.
    legacy.execute("INSERT OR REPLACE INTO answers VALUES (?, ?, ?)", (new_key, '{"noul": 0.5}', 9.0))
    legacy.close()
    cache = Cache(path)
    try:
        assert cache.get(new_key) == {"noul": 0.5} and cache.get_entry(new_key) == ({"noul": 0.5}, {})
    finally:
        cache.db.close()
