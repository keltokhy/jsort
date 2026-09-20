"""jsort against a fake Decisions endpoint. No network, no key."""

import csv
import io
import json
import math
import re

import httpx
import pytest

import jsort
from jsort.cli import main


class Fake:
    """Every text carries a number, `v=3.5`; the first text ranks higher with probability sigmoid(a - b - 0.1)."""

    def __init__(self, *, status=None, cost=0.0001, healthy_for=0):
        self.bodies, self.status, self.cost, self.healthy_for = [], status, cost, healthy_for

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.status and len(self.bodies) > self.healthy_for:
            return httpx.Response(self.status, json={"error": {"message": f"scripted {self.status}"}})
        a, b = body["state"]["A"], body["state"]["B"]
        if "POISON" in a + b:
            return httpx.Response(400, json={"error": {"message": "bad state", "code": 400}})
        value = lambda t: float(re.search(r"v=(-?[\d.]+)", t).group(1))
        p = 1 / (1 + math.exp(-(value(a) - value(b) - 0.1)))
        answers = {qid: {"type": "noul", "noul": p} for qid in body["questions"]}
        usage = {"input_tokens": 300, "output_tokens": 1, "cost": self.cost}
        return httpx.Response(200, json={"model": "typesafe/jev-1.13", "answers": answers, "usage": usage})


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for name in ("TYPESAFE_API_KEY", "JEV_API", "JEV_MODEL", "JEV_URL", "JSORT_BUDGET",
                 "JEV_GATEWAY_URL", "JEV_GATEWAY_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def run(argv, fake=None, **kw):
    fake = fake or Fake(**kw)
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, transport=httpx.MockTransport(fake), out=out, err=err)
    return code, out.getvalue(), err.getvalue(), fake


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


VALUES = [0.5, -2.0, 3.0, 1.5, -0.5, 2.2, -3.1, 0.9, -1.2, 4.0, 2.8, -2.6]
LINES = [f"ticket {i} v={v}" for i, v in enumerate(VALUES)]
SORTED = [l for _, l in sorted(zip(VALUES, LINES), reverse=True)]


def test_sorts_highest_first_and_reversed(tmp_path):
    f = write(tmp_path, "a.txt", "\n".join(LINES) + "\n")
    code, out, _, fake = run(["more urgent", f])
    assert code == 0 and out.splitlines() == SORTED
    assert run(["more urgent", f, "-r"])[1].splitlines() == SORTED[::-1]
    body = fake.bodies[0]
    assert set(body["state"]) == {"A", "B"}
    (q,) = body["questions"].values()
    assert q["type"] == "noul" and '"more urgent"' in q["instructions"]
    assert len(fake.bodies) <= len(LINES) * 10 / 2


def test_score_columns_blank_lines_and_line_numbers(tmp_path):
    f = write(tmp_path, "a.txt", "low v=-2\n\nhigh v=2\n   \nmid v=0\n")
    code, out, _, _ = run(["x", f, "-o", "-n"])
    rows = [l.split("\t") for l in out.splitlines()]
    assert code == 0 and [r[2] for r in rows] == ["3:high v=2", "5:mid v=0", "1:low v=-2"]
    scores = [float(r[0]) for r in rows]
    assert scores == sorted(scores, reverse=True) and abs(sum(scores)) < 0.02
    assert all(float(r[1]) >= 0 for r in rows)


def test_duplicates_share_a_score_and_cost_nothing_extra(tmp_path):
    f = write(tmp_path, "a.txt", "b v=1\na v=2\nb v=1\nc v=0\nb v=1\n")
    code, out, _, fake = run(["x", f, "-o"])
    lines = out.splitlines()
    assert [l.split("\t")[2] for l in lines] == ["a v=2", "b v=1", "b v=1", "b v=1", "c v=0"]
    assert len({l.split("\t")[0] for l in lines[1:4]}) == 1
    assert len(fake.bodies) <= 6                      # three distinct texts have six ordered pairs


def test_top_prints_only_the_top(tmp_path):
    f = write(tmp_path, "a.txt", "\n".join(LINES) + "\n")
    code, out, _, _ = run(["x", f, "--top", "3"])
    assert code == 0 and out.splitlines() == SORTED[:3]


def test_rerun_is_answered_from_the_cache(tmp_path):
    f = write(tmp_path, "a.txt", "\n".join(LINES) + "\n")
    _, first, _, fake = run(["x", f, "-o"])
    asked = len(fake.bodies)
    _, second, err, fake = run(["x", f, "-o", "--stats"], fake=fake)
    assert second == first and len(fake.bodies) == asked
    assert "0 calls" in err and "reliability" in err and "first-position lean -0.0" in err
    run(["x", f, "--seed", "5"], fake=fake)
    assert len(fake.bodies) > asked                   # another seed asks other pairs


def test_csv_gains_named_columns_and_can_keep_its_order(tmp_path):
    rows = [{"id": str(i), "note": f"n, {i} v={v}"} for i, v in enumerate(VALUES[:6])]
    buf = io.StringIO()
    w = csv.DictWriter(buf, ["id", "note"])
    w.writeheader()
    w.writerows(rows)
    f = write(tmp_path, "a.csv", buf.getvalue())

    code, out, _, _ = run(["x", f, "--csv", "--field", "note"])
    got = list(csv.DictReader(io.StringIO(out)))
    assert code == 0 and [r["id"] for r in got] == ["2", "5", "3", "0", "4", "1"] and list(got[0]) == ["id", "note"]

    code, out, _, _ = run(["x", f, "--csv", "--field", "note", "-o", "--keep-order", "--name", "urgency"])
    got = list(csv.DictReader(io.StringIO(out)))
    assert [r["id"] for r in got] == [r["id"] for r in rows] and got[0]["note"] == rows[0]["note"]
    assert list(got[0]) == ["id", "note", "urgency_score", "urgency_se", "urgency_n"]
    by_score = sorted(got, key=lambda r: -float(r["urgency_score"]))
    assert [r["id"] for r in by_score] == ["2", "5", "3", "0", "4", "1"]

    repeated = write(tmp_path, "r.csv", "note,note\nv=1,v=2\nv=3,v=0\n")
    code, out, err, _ = run(["x", repeated, "--csv", "--field", "note"])
    assert (code, out) == (2, "") and "repeats a column name" in err

    clash = write(tmp_path, "b.csv", "note,jsort_score\nv=1,9\nv=2,9\n")
    code, _, err, _ = run(["x", clash, "--csv", "--field", "note", "-o"])
    assert code == 2 and "--name" in err


def test_jsonl_records_come_back_whole_and_json_output_has_ranks(tmp_path):
    objs = [{"id": i, "event": {"message": f"m v={v}"}} for i, v in enumerate(VALUES[:5])]
    f = write(tmp_path, "a.jsonl", "".join(json.dumps(o) + "\n" for o in objs) + "not json\n")
    code, out, err, _ = run(["x", f, "--jsonl", "--field", "event.message"])
    assert code == 2 and "not valid JSON" in err       # the bad line is reported, the rest still sorts
    assert [json.loads(l)["id"] for l in out.splitlines()] == [2, 3, 0, 4, 1]
    assert out.splitlines()[0] == json.dumps(objs[2])

    _, out, _, _ = run(["x", f, "--jsonl", "--field", "event.message", "-o"])
    top = json.loads(out.splitlines()[0])
    assert top["id"] == 2 and top["jsort_score"] > 0 and top["jsort_n"] > 0

    _, out, _, _ = run(["x", f, "--jsonl", "--field", "event.message", "--json"])
    got = [json.loads(l) for l in out.splitlines()]
    assert [g["rank"] for g in got] == [1, 2, 3, 4, 5] and got[0]["record"]["id"] == 2 and got[0]["line"] == 3
    _, out, _, _ = run(["x", f, "--jsonl", "--field", "event.message", "--json", "-r"])
    assert [json.loads(l)["rank"] for l in out.splitlines()] == [5, 4, 3, 2, 1]      # rank 1 stays the highest


def test_whole_files_and_paragraphs(tmp_path):
    names = [write(tmp_path, f"{c}.txt", f"title\n\nbody v={v}\n") for c, v in zip("abc", (0, 2, 1))]
    assert run(["x", "--whole", *names])[1].splitlines() == [names[1], names[2], names[0]]
    f = write(tmp_path, "p.txt", "one v=1\nstill one\n\n\ntwo v=2\n")
    assert run(["x", "--para", f])[1] == "two v=2\n\none v=1\nstill one\n\n"


def test_one_text_needs_no_key_and_no_call(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    f = write(tmp_path, "a.txt", "only v=1\n")
    code, out, _, fake = run(["x", f, "-o"])
    assert (code, out, fake.bodies) == (0, "\t\tonly v=1\n", [])
    assert run(["x", write(tmp_path, "e.txt", "")])[:2] == (0, "")


@pytest.mark.parametrize("lines", [["only v=1"], ["same v=1"] * 3])
@pytest.mark.parametrize("reverse", [False, True])
def test_top_keeps_texts_when_no_comparison_is_needed(tmp_path, monkeypatch, lines, reverse):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    f = write(tmp_path, "a.txt", "\n".join(lines) + "\n")
    code, out, _, fake = run(["x", f, "--top", "1"] + (["-r"] if reverse else []))
    assert (code, out, fake.bodies) == (0, lines[0] + "\n", [])


def test_top_does_not_return_texts_whose_comparisons_failed(tmp_path):
    f = write(tmp_path, "a.txt", "POISON v=1\nPOISON v=2\n")
    code, out, err, _ = run(["x", f, "--top", "1"])
    assert code == 2 and out == "" and "a comparison failed" in err


def test_budget_stops_the_questions_but_still_sorts(tmp_path):
    f = write(tmp_path, "a.txt", "\n".join(LINES) + "\n")
    code, out, err, fake = run(["x", f, "--budget", "0.02", "-j", "1"], cost=0.001)
    assert code == 2 and "budget" in err
    assert len(fake.bodies) == 20 and sorted(out.splitlines()) == sorted(LINES)
    assert out.splitlines()[0] == SORTED[0]


def test_a_bad_key_prints_nothing_and_a_bad_pair_is_skipped(tmp_path):
    f = write(tmp_path, "a.txt", "\n".join(LINES) + "\n")
    code, out, err, _ = run(["x", f], status=401)
    assert (code, out) == (2, "") and "401" in err

    g = write(tmp_path, "b.txt", "\n".join(LINES + ["POISON v=0"]) + "\n")
    code, out, err, _ = run(["x", g])
    assert code == 2 and "a comparison failed" in err and "never compared" in err
    assert out.splitlines()[:-1] == SORTED and out.splitlines()[-1] == "POISON v=0"


def test_out_of_credit_midway_still_sorts_on_what_was_paid_for(tmp_path):
    f = write(tmp_path, "a.txt", "\n".join(LINES) + "\n")
    code, out, err, fake = run(["x", f, "-j", "1"], status=402, healthy_for=30)
    assert code == 2 and "402" in err and "sorted on the 30 comparisons" in err
    assert sorted(out.splitlines()) == sorted(LINES) and out.splitlines()[0] == SORTED[0]
    assert len(fake.bodies) == 31                       # the refusal stops the asking


def test_reverse_top_hunts_the_bottom(tmp_path):
    values = [round(-6 + 0.2 * i, 1) for i in range(60)]
    lines = [f"t{i} v={v}" for i, v in enumerate(values)]
    f = write(tmp_path, "a.txt", "\n".join(lines) + "\n")
    code, out, _, _ = run(["x", f, "-r", "--top", "5"])
    assert code == 0 and out.splitlines() == lines[:5]
    _, out, _, _ = run(["x", f, "-r", "--top", "5", "--json", "--seed", "1"])
    bottom = [json.loads(l)["comparisons"] for l in out.splitlines()]
    _, out, _, _ = run(["x", f, "--top", "5", "--json", "--seed", "1"])
    top = [json.loads(l)["comparisons"] for l in out.splitlines()]
    assert min(bottom) >= 10 and min(top) >= 10           # whichever end is wanted keeps being asked about


def test_awkward_csv_files(tmp_path):
    ragged = write(tmp_path, "r.csv", "id,note\n1,v=1\n2,v=3,surplus,more\n3,v=2\n")
    code, out, err, _ = run(["x", ragged, "--csv", "--field", "note", "-o", "--name", "s"])
    rows = list(csv.reader(io.StringIO(out)))
    assert code == 0 and "more values than the header" in err
    assert rows[0] == ["id", "note", "s_score", "s_se", "s_n"]
    assert rows[1][:2] == ["2", "v=3"] and rows[1][5:] == ["surplus", "more"] and len(rows[2]) == 5

    bom = write(tmp_path, "b.csv", "\ufeffnote,id\nv=1,1\nv=2,2\n")
    code, out, _, _ = run(["x", bom, "--csv", "--field", "note"])
    assert code == 0 and out == "note,id\nv=2,2\nv=1,1\n"

    long = write(tmp_path, "l.csv", f"id,note\n1,v=1 {'x' * 200_000}\n2,v=2\n")
    code, out, err, _ = run(["x", long, "--csv", "--field", "note", "--max-chars", "300000"])
    assert code == 0 and [r[0] for r in csv.reader(io.StringIO(out))] == ["id", "2", "1"]

    empty = write(tmp_path, "e.csv", "id,note\n")
    assert run(["x", empty, "--csv", "--field", "note", "-o"])[:2] == (0, "id,note,jsort_score,jsort_se,jsort_n\n")


def test_jsonl_refuses_to_overwrite_a_field(tmp_path):
    f = write(tmp_path, "a.jsonl", '{"m": "v=1", "jsort_se": 7}\n{"m": "v=2"}\n')
    code, out, err, fake = run(["x", f, "--jsonl", "--field", "m", "-o"])
    assert (code, out, fake.bodies) == (2, "", []) and "--name" in err
    assert run(["x", f, "--jsonl", "--field", "m", "-o", "--name", "u"])[0] == 0


def test_usage_errors(tmp_path):
    f = write(tmp_path, "a.txt", "a v=1\nb v=2\n")
    assert run([])[0] == 2
    for argv in (["x", f, "-k", "1"], ["x", f, "--csv"], ["x", f, "--field", "a"], ["x", "--whole"],
                 ["x", f, "--top", "2", "--keep-order"], ["x", f, "-o", "--name", "two words"], ["  ", f]):
        assert run(argv)[0] == 2, argv
    assert run(["x", str(tmp_path / "missing.txt")])[0] == 2


def test_python_api(tmp_path):
    fake = Fake()
    r = jsort.rank(LINES, "more urgent", transport=httpx.MockTransport(fake), seed=1)
    assert [LINES[i] for i in r.order()] == SORTED
    assert [LINES[i] for i in r.order(reverse=True)] == SORTED[::-1]
    assert r.asked == len(fake.bodies) and r.reliability > 0.9 and abs(r.lean + 0.025) < 0.02

    # The command's seat belt applies here too, and the options the command has are accepted.
    r = jsort.rank(LINES, "x", transport=httpx.MockTransport(Fake(cost=0.2)), timeout=5, cache=False, concurrency=1)
    assert r.over_budget and r.asked == 5
    r = jsort.rank(LINES, "x", transport=httpx.MockTransport(Fake(cost=0.2)), budget=0, cache=False)
    assert not r.over_budget and r.asked > 5

    async def inside_a_running_loop():                    # as in a notebook
        return jsort.rank(LINES, "more urgent", transport=httpx.MockTransport(Fake()))
    import asyncio
    assert [LINES[i] for i in asyncio.run(inside_a_running_loop()).order()] == SORTED

    with pytest.raises(ValueError, match="concurrency"):
        jsort.rank(LINES, "x", concurrency=0, transport=httpx.MockTransport(Fake()))
