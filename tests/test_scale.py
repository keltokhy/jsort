"""Saving a scale and placing new texts on it, against a fake Decisions endpoint. No network, no key."""

import asyncio
import csv
import io
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import numpy as np
import pytest

import jsort
from jsort.cli import main
from jsort.core import Jev
from jsort.placement import aplace
from jsort.scale import Scale, ScaleError, choose


class Oracle:
    """A judge with a latent scale. Every text carries its score, `v=1.5`, and the first text ranks higher with
    probability sigmoid(a - b - 0.1 + e), where e is a fixed quirk of that ordered pair: deterministic, as Jev
    nearly is, but not sitting exactly on one scale, so the standard errors have something to measure."""

    def __init__(self, *, quirk=0.5, cost=0.0001, model="typesafe/jev-1.13", gate=None):
        self.bodies, self.quirk, self.cost, self.model, self.gate = [], quirk, cost, model, gate

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.gate is not None:
            await self.gate(body)
        a, b = body["state"]["A"], body["state"]["B"]
        value = lambda t: float(re.search(r"v=(-?[\d.]+)", t).group(1))
        e = np.random.default_rng(zlib.crc32(f"{a}|{b}".encode())).standard_normal() * self.quirk
        p = 1 / (1 + math.exp(-(value(a) - value(b) - 0.1 + e)))
        answers = {qid: {"type": "noul", "noul": p} for qid in body["questions"]}
        cost = self.cost(len(self.bodies)) if callable(self.cost) else self.cost     # a price may change as the run goes on
        self.spent = getattr(self, "spent", 0.0) + cost
        return httpx.Response(200, json={"model": self.model, "answers": answers,
                                         "usage": {"input_tokens": 300, "output_tokens": 1, "cost": cost}})


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for name in ("TYPESAFE_API_KEY", "JEV_API", "JEV_MODEL", "JEV_URL", "JSORT_BUDGET",
                 "JEV_GATEWAY_URL", "JEV_GATEWAY_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def run(argv, oracle=None, out=None, **kw):
    oracle = oracle or Oracle(**kw)
    out, err = out or io.StringIO(), io.StringIO()
    code = main(argv, transport=httpx.MockTransport(oracle), out=out, err=err)
    return code, out.getvalue(), err.getvalue(), oracle


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


LATENT = np.round(np.random.default_rng(7).normal(0, 1.8, 140), 2)
BASE = [f"statement {i} v={v}" for i, v in enumerate(LATENT[:90])]
HELD = [f"later {i} v={v}" for i, v in enumerate(LATENT[90:])]
ON_SCALE = LATENT[90:] - LATENT[:90].mean()        # a fitted scale is centred on the texts it was fitted to
DESCRIPTION = "more hawkish about inflation"


@pytest.fixture
def scale(tmp_path):
    """A scale saved by an ordinary run, as a path."""
    path = str(tmp_path / "hawkish.json")
    code, _, err, _ = run([DESCRIPTION, write(tmp_path, "base.txt", "\n".join(BASE) + "\n"), "--save-scale", path])
    assert code == 0, err
    return path


def placed(out):
    return {o["text"]: o for o in map(json.loads, out.splitlines())}


def test_saving_writes_what_was_asked_of_whom_and_anchors_across_the_range(tmp_path, scale):
    data = json.loads(open(scale).read())
    assert data["schema_version"] == 1 and data["description"] == DESCRIPTION
    assert data["question"] == {"type": "noul", "instructions": f'Text A ranks higher than text B on this criterion: "{DESCRIPTION}"'}
    assert data["model"] == {"api": "openrouter", "endpoint": "https://openrouter.ai/api/alpha/decisions",
                             "requested": "~typesafe/jev-latest", "answered": {"typesafe/jev-1.13": 450}, "unknown_answers": 0}
    assert data["input"] == {"max_chars": 8000, "unit": "line", "field": None}
    fit = data["fit"]
    assert fit["texts"] == 90 and fit["comparisons"] == 450 and fit["reliability"] > 0.9 and abs(fit["lean"] + 0.025) < 0.02
    assert abs(fit["gamma"] + 0.1) < 0.08

    # The sort itself is what it was before: saving changes nothing about it.
    base = str(tmp_path / "base.txt")
    _, with_save, _, _ = run([DESCRIPTION, base, "-o", "--save-scale", str(tmp_path / "again.json")])
    assert with_save == run([DESCRIPTION, base, "-o"])[1]
    scores = {l.split("\t")[2]: float(l.split("\t")[0]) for l in with_save.splitlines()}
    anchors = data["anchors"]
    assert len(anchors) == 30 and all(abs(scores[a["text"]] - a["score"]) < 0.006 for a in anchors)
    assert [a["score"] for a in anchors] == sorted((a["score"] for a in anchors), reverse=True)
    assert anchors[0]["score"] == pytest.approx(max(scores.values()), abs=0.006)      # the ends of the fit are kept
    assert anchors[-1]["score"] == pytest.approx(min(scores.values()), abs=0.006)
    gaps = -np.diff([a["score"] for a in anchors])
    assert np.median(gaps) < 1.5 * (anchors[0]["score"] - anchors[-1]["score"]) / len(anchors)   # spread out, not bunched

    code, _, _, _ = run([DESCRIPTION, base, "--save-scale", str(tmp_path / "few.json"), "--anchors", "8"])
    assert code == 0 and len(json.loads(open(tmp_path / "few.json").read())["anchors"]) == 8


def test_anchor_choice_prefers_small_reported_errors_and_keeps_both_ends():
    score = [0.0, 0.1, 0.2, 5.0, 5.1, 9.9, 10.0, 4.9]
    se = [0.9, 0.1, 0.5, 0.2, 0.1, 0.1, 0.8, 0.7]
    assert choose(score, se, 4) == [6, 4, 7, 0]          # one from each quarter: both ends whatever their errors, else the smallest
    assert choose(score, se, 50) == [6, 5, 4, 3, 7, 2, 1, 0]
    assert choose([1.0, float("nan"), 2.0], [0.1, 0.1, float("nan")], 2) == [0]


def test_placed_scores_track_the_latent_scale_within_a_few_standard_errors(tmp_path, scale):
    code, out, err, oracle = run(["--scale", scale, write(tmp_path, "held.txt", "\n".join(HELD) + "\n"), "--json"])
    assert code == 0, err
    got = placed(out)
    score = np.array([got[t]["score"] for t in HELD])
    se = np.array([got[t]["se"] for t in HELD])
    within = np.array([got[t]["beyond"] is None for t in HELD])
    assert np.corrcoef(score, ON_SCALE)[0, 1] > 0.97
    z = np.abs(score - ON_SCALE)[within] / se[within]
    assert np.mean(z < 2) > 0.8 and np.mean(z < 4) > 0.95 and np.all(se[within] < 0.8)
    assert [o["rank"] for o in map(json.loads, out.splitlines())] == list(range(1, len(HELD) + 1))   # sorted, highest first
    assert all(o["comparisons"] <= 10 for o in got.values()) and len(oracle.bodies) <= 10 * len(HELD)
    # Every question set one new text against one anchor, and asked what the scale's run had asked.
    anchors = {a.text for a in Scale.load(scale).anchors}
    for body in oracle.bodies:
        a, b = body["state"]["A"], body["state"]["B"]
        assert (a in anchors) != (b in anchors) and list(body["questions"].values()) == [Scale.load(scale).question]
    new_first = np.mean([body["state"]["A"] not in anchors for body in oracle.bodies])
    assert 0.45 < new_first < 0.55                       # positions are balanced, as they are in the fit


def test_placement_never_changes_the_scale(tmp_path, scale):
    before, loaded = open(scale, "rb").read(), Scale.load(scale)
    held = write(tmp_path, "held.txt", "\n".join(HELD[:10]) + "\n")
    for extra in ([], ["--keep-order"], ["--unordered"]):
        assert run(["--scale", scale, held, "-o", *extra])[0] == 0
    assert open(scale, "rb").read() == before and Scale.load(scale) == loaded
    with pytest.raises(AttributeError):
        loaded.anchors[0].score = 0.0                    # nor can code holding the scale move an anchor


def test_a_text_scores_the_same_alone_or_in_company(tmp_path, scale):
    texts = HELD[:8]
    _, together, _, _ = run(["--scale", scale, write(tmp_path, "all.txt", "\n".join(texts) + "\n"), "--json", "--no-cache"])
    together = placed(together)
    other = write(tmp_path, "other.txt", "\n".join(reversed(HELD[3:30])) + "\n")
    company = placed(run(["--scale", scale, other, "--json", "--no-cache", "--keep-order"])[1])
    for i, text in enumerate(texts):
        _, alone, _, oracle = run(["--scale", scale, write(tmp_path, f"one{i}.txt", text + "\n"), "--json", "--no-cache"])
        (alone,) = placed(alone).values()
        assert len(oracle.bodies) == alone["comparisons"] == 10
        for key in ("score", "se", "comparisons", "beyond"):
            assert alone[key] == together[text][key]
            assert text not in company or alone[key] == company[text][key]


def test_streaming_prints_a_text_before_the_input_ends(monkeypatch, scale):
    r, w = os.pipe()
    source = io.TextIOWrapper(os.fdopen(r, "rb"))
    monkeypatch.setattr("sys.stdin", source)
    first_line_printed, saw_it_before_eof = threading.Event(), []

    class Out(io.StringIO):
        def write(self, s):
            n = super().write(s)
            if "\n" in self.getvalue():
                first_line_printed.set()
            return n

    def captions():
        with os.fdopen(w, "w") as pipe:
            pipe.write(HELD[0] + "\n")
            pipe.flush()
            saw_it_before_eof.append(first_line_printed.wait(10))    # the pipe is still open here
            pipe.write(HELD[1] + "\n")

    speaker = threading.Thread(target=captions)
    speaker.start()
    try:
        code, out, err, _ = run(["--scale", scale, "-o", "--keep-order"], out=Out())
    finally:
        speaker.join()
        source.close()
    assert code == 0 and saw_it_before_eof == [True], err
    assert [l.split("\t")[2] for l in out.splitlines()] == HELD[:2]


def test_stopping_while_the_pipe_is_still_open_exits_cleanly(scale):
    # The real command on a real pipe. The only endpoint the child knows is a server on this machine that refuses
    # it, slowly, so that the thread reading stdin is back waiting on the pipe when the run stops. A reader holding
    # sys.stdin's lock at that moment takes the interpreter down with it: "Fatal Python error", exit 134.
    class Refuse(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            time.sleep(0.3)
            body = b'{"error": {"message": "scripted 401"}}'
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Refuse)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = os.environ | {"JEV_API": "gateway", "JEV_GATEWAY_API_KEY": "none",
                        "JEV_GATEWAY_URL": f"http://127.0.0.1:{server.server_port}/v1/systemone"}
    child = subprocess.Popen([sys.executable, "-m", "jsort", "--scale", scale, "-o", "--keep-order", "--any-model", "--no-cache"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    try:
        child.stdin.write(HELD[0] + "\n")
        child.stdin.flush()                                  # and the pipe stays open, as under tail -f
        assert child.wait(timeout=30) == 2
        err = child.stderr.read()
        assert "401" in err and "Fatal Python error" not in err and child.stdout.read() == ""
    finally:
        child.kill()
        for pipe in (child.stdin, child.stdout, child.stderr):
            pipe.close()
        server.shutdown()
        server.server_close()


def test_keep_order_holds_the_input_order_and_unordered_does_not(tmp_path, scale):
    async def slow_first(body):
        if HELD[0] in body["state"].values():
            await asyncio.sleep(0.05)

    apart = [HELD[0]]                                        # texts far enough apart that their order is not in doubt
    for text in HELD[1:]:
        if all(abs(LATENT[90 + HELD.index(text)] - LATENT[90 + HELD.index(t)]) > 0.8 for t in apart) and len(apart) < 6:
            apart.append(text)
    by_latent = sorted(apart, key=lambda t: -LATENT[90 + HELD.index(t)])
    held = write(tmp_path, "held.txt", "\n".join(apart) + "\n")
    _, out, _, _ = run(["--scale", scale, held, "--keep-order"], gate=slow_first)
    assert out.splitlines() == apart and apart != by_latent
    # (No budget here: under one, the first text goes alone until its price is known, so it would also finish first.)
    _, out, _, _ = run(["--scale", scale, held, "--unordered", "--no-cache", "--budget", "0"], gate=slow_first)
    assert sorted(out.splitlines()) == sorted(apart) and out.splitlines()[-1] == HELD[0]
    _, out, _, _ = run(["--scale", scale, held])             # neither: a sort, which has to wait for the end
    assert out.splitlines() == by_latent
    _, out, _, _ = run(["--scale", scale, held, "--top", "2", "-r"])
    assert out.splitlines() == by_latent[::-1][:2]


def test_ordered_streaming_bounds_the_texts_in_hand(tmp_path, scale):
    started, in_hand = set(), []
    first_calls = 0

    async def slow_first(body):
        nonlocal first_calls
        started.add(next(t for t in body["state"].values() if t.startswith("later")))
        first_calls += HELD[0] in body["state"].values()
        if HELD[0] in body["state"].values() and first_calls > 1 and not in_hand:
            # Let the identity probe finish first, then hold the first text while its neighbours can finish.
            in_hand.append(0)
            await asyncio.sleep(0.1)                         # long enough for the rest to run ahead, were they allowed to
            in_hand[0] = len(started)

    held = write(tmp_path, "held.txt", "\n".join(HELD[:12]) + "\n")
    code, out, _, _ = run(["--scale", scale, held, "--keep-order", "-j", "2", "--budget", "0"], gate=slow_first)
    assert code == 0 and out.splitlines() == HELD[:12]
    assert in_hand == [2]                                    # the slow text and one more: a finished text keeps its slot until it prints


def test_a_conflicting_description_is_an_error_and_a_matching_one_is_not(tmp_path, scale):
    held = write(tmp_path, "held.txt", "\n".join(HELD[:3]) + "\n")
    code, out, err, oracle = run(["--scale", scale, "more dovish about inflation", held])
    assert (code, out, oracle.bodies) == (2, "", []) and "more dovish about inflation" in err and DESCRIPTION in err
    assert run(["--scale", scale, DESCRIPTION, held])[:2] == run(["--scale", scale, held])[:2]
    assert run(["--scale", scale, str(tmp_path / "missing.txt")])[0] == 2


def test_another_model_is_refused_unless_asked_for(tmp_path, scale, monkeypatch):
    held = write(tmp_path, "held.txt", "\n".join(HELD[:3]) + "\n")
    code, out, err, oracle = run(["--scale", scale, held, "--model", "typesafe/jev-2"])
    assert (code, out, oracle.bodies) == (2, "", [])          # refused before anything is asked
    assert "typesafe/jev-2" in err and "~typesafe/jev-latest" in err and "--any-model" in err
    assert run(["--scale", scale, held, "--model", "typesafe/jev-2", "--any-model"])[0] == 0

    # The alias the scale asked for now reaches a newer model. Only the API's reply can show that.
    code, out, err, oracle = run(["--scale", scale, held, "--no-cache", "-j", "1"], model="typesafe/jev-1.14")
    assert code == 2 and out == "" and len(oracle.bodies) == 1
    assert "typesafe/jev-1.13" in err and "the answers now come from typesafe/jev-1.14" in err
    assert run(["--scale", scale, held, "--no-cache", "--any-model"], model="typesafe/jev-1.14")[0] == 0
    for extra in (["--keep-order"], ["--unordered"]):
        code, out, err, _ = run(["--scale", scale, held, "--no-cache", *extra], model="typesafe/jev-1.14")
        assert code == 2 and out == "" and "the answers now come from typesafe/jev-1.14" in err
    # Pinning the model that answered is as good as the alias that reached it.
    assert run(["--scale", scale, held, "--no-cache", "--model", "typesafe/jev-1.13"])[0] == 0

    monkeypatch.setenv("TYPESAFE_API_KEY", "another-key")
    assert run(["--scale", scale, held])[0] == 0              # the scale's API is used, not the first with a key
    code, _, err, oracle = run(["--scale", scale, held, "--api", "typesafe"])
    assert code == 2 and oracle.bodies == [] and "the API is 'typesafe', not 'openrouter'" in err
    monkeypatch.delenv("OPENROUTER_API_KEY")
    code, _, err, _ = run(["--scale", scale, held, "--no-cache"])
    assert code == 2 and "no key for openrouter" in err


def test_a_scale_file_round_trips_and_a_bad_one_is_refused(tmp_path, scale):
    data = json.loads(open(scale).read())
    loaded = Scale.load(scale)
    assert loaded.to_json() == data and loaded.description == DESCRIPTION and len(loaded.anchors) == 30
    again = str(tmp_path / "again.json")
    loaded.save(again)
    assert open(again, "rb").read() == open(scale, "rb").read() and Scale.load(again) == loaded

    held = write(tmp_path, "held.txt", HELD[0] + "\n")
    for damage, complaint in (
        (lambda d: d.update(schema_version=2), "schema 2"),
        (lambda d: d.pop("schema_version"), "not a scale file"),
        (lambda d: d.update(anchors=d["anchors"][:1]), "at least two anchors"),
        (lambda d: d["anchors"][0].update(score=float("nan")), "score is not a number"),
        (lambda d: d.update(description="more dovish"), "question is not the one jsort asks"),
        (lambda d: d["fit"].pop("gamma"), "first-position lean"),
        (lambda d: d["model"].pop("endpoint"), "which api, endpoint and model"),
        (lambda d: d["input"].update(max_chars=0), "how many characters"),
    ):
        broken = json.loads(json.dumps(data))
        damage(broken)
        path = write(tmp_path, "broken.json", json.dumps(broken))
        with pytest.raises(ScaleError, match=complaint):
            Scale.load(path)
        code, out, err, oracle = run(["--scale", path, held])
        assert (code, out, oracle.bodies) == (2, "", []) and path in err
    assert run(["--scale", write(tmp_path, "junk.json", "not json"), held])[0] == 2
    assert run(["--scale", str(tmp_path / "missing.json"), held])[0] == 2


def test_seed_makes_placement_repeatable(tmp_path, scale):
    held = write(tmp_path, "held.txt", "\n".join(HELD[:5]) + "\n")
    asked = lambda oracle: sorted(json.dumps(b["state"], sort_keys=True) for b in oracle.bodies)
    _, first, _, a = run(["--scale", scale, held, "-o", "--no-cache", "--seed", "3"])
    _, second, _, b = run(["--scale", scale, held, "-o", "--no-cache", "--seed", "3"])
    assert first == second and asked(a) == asked(b)
    _, _, _, c = run(["--scale", scale, held, "-o", "--no-cache", "--seed", "4"])
    assert asked(c) != asked(a)
    _, cached, err, d = run(["--scale", scale, held, "-o", "--seed", "3", "--stats"])
    _, cached, err, d = run(["--scale", scale, held, "-o", "--seed", "3", "--stats"], oracle=d)
    assert cached == first and len(d.bodies) == 50 and "0 calls, 50 cached" in err    # a rerun is free
    assert "5 texts placed against 30 anchors, 50 comparisons" in err


def test_texts_off_either_end_are_finite_and_flagged(tmp_path, scale):
    # Against v=99 every answer is exactly 1.0: the case that would put a maximum-likelihood score at infinity.
    lines = ["hotter than anything v=99", HELD[0], "colder than anything v=-99"]
    held = write(tmp_path, "held.txt", "\n".join(lines) + "\n")
    code, out, err, _ = run(["--scale", scale, held, "--json", "--keep-order"], quirk=0.0)
    got = [json.loads(l) for l in out.splitlines()]
    low, high = Scale.load(scale).span
    assert code == 0 and [g["beyond"] for g in got] == ["above", None, "below"]
    assert all(math.isfinite(g["score"]) for g in got)
    assert got[0]["se"] is None and got[2]["se"] is None and math.isfinite(got[1]["se"])
    assert high < got[0]["score"] < high + 10 and low - 10 < got[2]["score"] < low
    assert f"{held}:1: above every anchor" in err and f"{held}:3: below every anchor" in err
    assert "2 texts fell beyond the scale's anchors (1 above" in err and "bounds toward the scale" in err

    _, out, err, _ = run(["--scale", scale, held, "-o"], quirk=0.0)       # plain columns stay numeric; stderr says which
    assert [float(l.split("\t")[0]) for l in out.splitlines()] == sorted((round(g["score"], 2) for g in got), reverse=True)
    assert "2 texts fell beyond" in err


def test_k_bounds_the_comparisons_and_nothing_stops_a_text_early(tmp_path, scale):
    held = write(tmp_path, "held.txt", "\n".join(HELD[:10]) + "\n")
    for k in (2, 7, 16):
        _, out, _, oracle = run(["--scale", scale, held, "--json", "-k", str(k), "--no-cache"])
        assert {o["comparisons"] for o in placed(out).values()} == {k} and len(oracle.bodies) == 10 * k
    # Stopping once the reported standard error looks small enough would select for small estimates of it:
    # the option that did so is gone, from the command and from Python.
    with pytest.raises(SystemExit) as unknown_option:
        run(["--scale", scale, held, "--se-target", "0.3"])
    assert unknown_option.value.code == 2
    with pytest.raises(TypeError):
        jsort.place(HELD[:2], scale, se_target=0.3, transport=httpx.MockTransport(Oracle()))

    # With five anchors a text can meet each twice at most, once in each position.
    few = str(tmp_path / "few.json")
    run([DESCRIPTION, str(tmp_path / "base.txt"), "--save-scale", few, "--anchors", "5"])
    _, out, _, oracle = run(["--scale", few, held, "--json", "-k", "30", "--no-cache"])
    assert {o["comparisons"] for o in placed(out).values()} == {10}
    states = [json.dumps(b["state"], sort_keys=True) for b in oracle.bodies]
    assert len(set(states)) == len(states) == 100


def test_an_anchor_keeps_its_score_and_needs_no_key(tmp_path, scale, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    anchors = Scale.load(scale).anchors
    lines = [anchors[3].text, "   ", anchors[0].text]
    code, out, _, oracle = run(["--scale", scale, write(tmp_path, "a.txt", "\n".join(lines) + "\n"), "-o", "--keep-order"])
    assert (code, oracle.bodies) == (0, [])
    assert out == "".join(f"{a.score:.2f}\t{a.se:.2f}\t{a.text}\n" for a in (anchors[3], anchors[0]))
    assert run(["--scale", scale, str(tmp_path / "a.txt"), "-o"])[1].splitlines()[0].endswith(anchors[0].text)


def test_csv_and_jsonl_gain_a_beyond_field(tmp_path, scale):
    rows = [{"id": str(i), "note": t} for i, t in enumerate(HELD[:4] + ["off the top v=99"])]
    buf = io.StringIO()
    w = csv.DictWriter(buf, ["id", "note"])
    w.writeheader()
    w.writerows(rows)
    f = write(tmp_path, "new.csv", buf.getvalue())
    for extra in (["--keep-order"], []):
        code, out, _, _ = run(["--scale", scale, f, "--csv", "--field", "note", "-o", "--name", "hawk", *extra])
        got = list(csv.DictReader(io.StringIO(out)))
        assert code == 0 and list(got[0]) == ["id", "note", "hawk_score", "hawk_se", "hawk_n", "hawk_beyond", "hawk_partial"]
        assert {r["id"]: r["hawk_beyond"] for r in got} == {"0": "", "1": "", "2": "", "3": "", "4": "above"}
        assert [r["id"] for r in got] == (["0", "1", "2", "3", "4"] if extra else
                                          [r["id"] for r in sorted(got, key=lambda r: -float(r["hawk_score"]))])
    clash = write(tmp_path, "clash.csv", "note,jsort_beyond\nv=1,x\n")
    for extra in ([], ["--keep-order"]):
        code, out, err, oracle = run(["--scale", scale, clash, "--csv", "--field", "note", "-o", *extra])
        assert (code, out, oracle.bodies) == (2, "", []) and "jsort_beyond" in err and "--name" in err
    assert run(["--scale", scale, write(tmp_path, "e.csv", "id,note\n"), "--csv", "--field", "note", "-o", "--keep-order"])[:2] \
        == (0, "id,note,jsort_score,jsort_se,jsort_n,jsort_beyond,jsort_partial\n")

    j = write(tmp_path, "new.jsonl", "".join(json.dumps(r) + "\n" for r in rows) + "not json\n")
    code, out, err, _ = run(["--scale", scale, j, "--jsonl", "--field", "note", "-o", "--keep-order"])
    got = [json.loads(l) for l in out.splitlines()]
    assert code == 2 and "not valid JSON" in err and [g["id"] for g in got] == ["0", "1", "2", "3", "4"]
    assert got[4]["jsort_beyond"] == "above" and got[0]["jsort_beyond"] is None and got[0]["jsort_n"] == 10
    assert got[0]["jsort_partial"] is False


def test_max_chars_comes_from_the_scale(tmp_path):
    long = [f"v={v} " + "x" * 40 for v in (1.0, -1.0, 0.0, 2.0)]
    path = str(tmp_path / "short.json")
    code, _, _, oracle = run(["x", write(tmp_path, "b.txt", "\n".join(long) + "\n"), "--save-scale", path, "--max-chars", "12"])
    assert code == 0 and all(len(t) <= 12 for body in oracle.bodies for t in body["state"].values())
    assert Scale.load(path).max_chars == 12 and all(len(a.text) <= 12 for a in Scale.load(path).anchors)
    new = write(tmp_path, "n.txt", "v=0.5 " + "y" * 40 + "\n")
    _, _, err, oracle = run(["--scale", path, new])
    assert all(len(t) <= 12 for body in oracle.bodies for t in body["state"].values()) and "first 12 characters" in err
    _, _, err, oracle = run(["--scale", path, new, "--max-chars", "20", "--no-cache"])
    assert max(len(t) for body in oracle.bodies for t in body["state"].values()) == 20
    assert "the scale was built showing Jev the first 12 characters of a text; this run shows 20" in err


def test_the_budget_stops_placement_between_texts(tmp_path, scale):
    held = write(tmp_path, "held.txt", "\n".join(HELD[:10]) + "\n")
    for extra in ([], ["--keep-order", "--no-cache"]):
        code, out, err, oracle = run(["--scale", scale, held, "--budget", "0.025", "-o", *extra], cost=0.001)
        # Two texts at ten comparisons each fit in the budget and a third does not, so the third is not begun.
        assert code == 2 and len(oracle.bodies) == 20 and oracle.spent <= 0.025
        assert "budget" in err and "in full or not at all" in err
        rows = [l.split("\t") for l in out.splitlines()]
        scored = [r[2] for r in rows if r[0]]
        assert sorted(scored) == sorted(HELD[:2]) and "fewer comparisons" not in err
        if extra:
            assert [r[2] for r in rows] == HELD[:len(rows)] and 2 <= len(rows) <= 10    # a stream stops reading at the budget
        else:
            assert sorted(r[2] for r in rows) == sorted(HELD[:10]) and "8 texts could not be placed" in err


def test_concurrent_placement_respects_the_remaining_budget(scale):
    async def go(budget, cost=0.01):
        oracle = Oracle(cost=cost)
        jev = Jev("test-key", transport=httpx.MockTransport(oracle))
        try:
            result = await aplace(HELD[:10], scale, jev, budget=budget, any_model=True)
            return result, len(oracle.bodies), jev.meter.cost
        finally:
            await jev.close()

    # The first reply shows that the budget does not cover the rest of the first text. It is cut short and says so.
    result, calls, spent = asyncio.run(go(0.035))
    assert result.over_budget and result.asked == calls and spent <= 0.035 and calls >= 1
    assert list(result.partial) == [True] + [False] * 9 and np.isnan(result.score[1:]).all()
    result, calls, spent = asyncio.run(go(0.25))
    assert result.over_budget and result.asked == calls == 20 and spent == pytest.approx(0.20) and not result.partial.any()
    result, calls, spent = asyncio.run(go(0))
    assert not result.over_budget and result.asked == calls == 100


def test_a_failed_comparison_is_reported_and_the_rest_still_place(tmp_path, scale):
    async def poison(body):
        if "POISON" in "".join(body["state"].values()):
            raise httpx.ConnectError("scripted failure")

    held = write(tmp_path, "held.txt", f"{HELD[0]}\nPOISON v=0\n{HELD[1]}\n")
    for extra in ([], ["--keep-order"]):
        code, out, err, _ = run(["--scale", scale, held, "-o", "--timeout", "0.05", *extra], gate=poison)
        assert code == 2 and "a comparison failed" in err and "1 texts could not be placed" in err
        rows = {l.split("\t")[2]: l.split("\t")[0] for l in out.splitlines()}
        assert rows["POISON v=0"] == "" and rows[HELD[0]] != "" and rows[HELD[1]] != ""


def test_a_bad_key_prints_nothing(tmp_path, scale):
    async def refuse(request):
        return httpx.Response(401, json={"error": {"message": "scripted 401"}})

    held = write(tmp_path, "held.txt", "\n".join(HELD[:3]) + "\n")
    for extra in ([], ["--keep-order"], ["--unordered"]):
        code, out, err, _ = run(["--scale", scale, held, *extra], oracle=refuse)
        assert (code, out) == (2, "") and "401" in err


def test_usage_errors(tmp_path, scale):
    f = write(tmp_path, "a.txt", "a v=1\nb v=2\n")
    for argv in (["--scale", scale, f, "--save-scale", str(tmp_path / "s.json")], ["x", f, "--anchors", "5"],
                 ["x", f, "--save-scale", str(tmp_path / "s.json"), "--anchors", "1"], ["x", f, "--unordered"],
                 ["x", f, "--any-model"],
                 ["--scale", scale, f, "--unordered", "--keep-order"], ["--scale", scale, f, "--unordered", "--top", "2"],
                 ["--scale", scale, f, "--keep-order", "--top", "2"], ["x", f, "--seed", "-1"],
                 ["x", f, "--save-scale", str(tmp_path / "no" / "such" / "dir.json")], ["--scale", scale, "--whole"]):
        code, out, _, oracle = run(argv)
        assert (code, out, oracle.bodies) == (2, "", []), argv
    # One scored text, or none, is not a scale.
    code, _, err, _ = run(["x", write(tmp_path, "one.txt", "only v=1\n"), "--save-scale", str(tmp_path / "one.json")])
    assert code == 2 and "was not written" in err and not (tmp_path / "one.json").exists()


def test_python_api(tmp_path):
    transport = lambda **kw: httpx.MockTransport(Oracle(**kw))
    r = jsort.rank(BASE, DESCRIPTION, transport=transport())
    scale = r.scale(anchors=20, unit="line")
    assert isinstance(scale, jsort.Scale) and len(scale.anchors) == 20 and scale.description == DESCRIPTION
    path = tmp_path / "hawkish.json"
    scale.save(path)
    assert jsort.Scale.load(path) == scale

    p = jsort.place(HELD[:12] + ["", "over the top v=99"], path, transport=transport(), seed=2)
    assert isinstance(p, jsort.Placement) and len(p.score) == len(p.se) == len(p.beyond) == 14
    assert np.corrcoef(p.score[:12], ON_SCALE[:12])[0, 1] > 0.95 and list(p.beyond) == [0] * 13 + [1]
    assert math.isnan(p.score[12]) and p.comparisons[12] == 0 and p.order()[0] == 13 and p.order()[-1] == 12
    assert p.asked == 130 and p.reliability is None and p.lean == scale.fit["lean"]
    again = jsort.place(HELD[:12], scale, transport=transport(), seed=2, cache=False)
    assert np.array_equal(again.score, p.score[:12])

    with pytest.raises(jsort.ScaleError, match="not comparable"):
        jsort.place(HELD[:2], scale, model="typesafe/jev-2", transport=transport())
    assert jsort.place(HELD[:2], scale, model="typesafe/jev-2", any_model=True, transport=transport()).asked == 20
    with pytest.raises(ValueError, match="per_item"):
        jsort.place(HELD[:2], scale, per_item=1, transport=transport())
    with pytest.raises(jsort.ScaleError, match="nothing was compared"):
        jsort.Ranking.unscored(3).scale()

    async def inside_a_running_loop():                    # as in a notebook
        return jsort.place(HELD[:3], scale, transport=transport())
    assert asyncio.run(inside_a_running_loop()).asked == 30


# ---- Who answered: the scale may only name a model it can vouch for ------------------------------------------------

def forget_who_answered(tmp_path):
    """Rewrite every cached answer the way jsort 0.1.3 and jgrep write them: the answers table only, a new time."""
    import sqlite3
    db = sqlite3.connect(tmp_path / "cache" / "jev" / "answers.sqlite", isolation_level=None)
    rows = db.execute("SELECT key, answer FROM answers").fetchall()
    for key, answer in rows:
        db.execute("INSERT OR REPLACE INTO answers VALUES (?, ?, ?)", (key, answer, 1.0))
    db.close()
    return len(rows)


def test_a_fit_that_mixes_two_models_is_not_saved_as_one(tmp_path):
    base = write(tmp_path, "base.txt", "\n".join(BASE[:12]) + "\n")
    path = tmp_path / "mixed.json"
    assert run(["x", base, "-k", "2"], model="typesafe/jev-1.13")[0] == 0        # twelve answers, cached under the alias
    # The alias has moved. A higher -k replays those from the cache and asks the new model for the rest.
    code, out, err, oracle = run(["x", base, "-k", "4", "--save-scale", str(path)], model="typesafe/jev-1.14")
    assert code == 2 and not path.exists() and len(oracle.bodies) == 12
    assert "typesafe/jev-1.13 (12)" in err and "typesafe/jev-1.14 (12)" in err and "--no-cache" in err
    assert sorted(out.splitlines()) == sorted(BASE[:12])                           # the sort itself still prints

    assert run(["x", base, "-k", "4", "--save-scale", str(path), "--no-cache"], model="typesafe/jev-1.14")[0] == 0
    assert json.loads(path.read_text())["model"]["answered"] == {"typesafe/jev-1.14": 24}

    forced = tmp_path / "forced.json"
    code, _, err, _ = run(["x", base, "-k", "4", "--save-scale", str(forced), "--any-model"], model="typesafe/jev-1.14")
    assert code == 0 and json.loads(forced.read_text())["model"]["answered"] == {"typesafe/jev-1.13": 12, "typesafe/jev-1.14": 12}
    held = write(tmp_path, "held.txt", HELD[0] + "\n")
    code, out, err, oracle = run(["--scale", str(forced), held])                 # and such a scale vouches for no model
    assert (code, out, oracle.bodies) == (2, "", []) and "typesafe/jev-1.13" in err and "--any-model" in err
    assert run(["--scale", str(forced), held, "--any-model"])[0] == 0


def test_answers_that_do_not_say_who_gave_them_are_not_vouched_for(tmp_path):
    base = write(tmp_path, "base.txt", "\n".join(BASE[:12]) + "\n")
    path = tmp_path / "legacy.json"
    assert run(["x", base])[0] == 0
    asked = forget_who_answered(tmp_path)
    code, _, err, oracle = run(["x", base, "--save-scale", str(path)])
    assert code == 2 and not path.exists() and oracle.bodies == []
    assert f"{asked} of the {asked} answers" in err and "--no-cache" in err and "--any-model" in err

    code, _, _, _ = run(["x", base, "--save-scale", str(path), "--any-model"])
    model = json.loads(path.read_text())["model"]
    assert code == 0 and model["answered"] == {} and model["unknown_answers"] == asked
    held = write(tmp_path, "held.txt", HELD[0] + "\n")
    code, out, err, oracle = run(["--scale", str(path), held])
    assert (code, out, oracle.bodies) == (2, "", []) and f"{asked} of the {asked} answers behind it do not say" in err
    assert "--any-model" in err

    # An API that never names its model leaves every answer unknown, cache or no cache.
    code, _, err, _ = run(["x", base, "--save-scale", str(path), "--no-cache"], model=None)
    assert code == 2 and "answers" in err and "--any-model" in err


def test_a_cached_answer_from_another_model_is_caught_without_a_call(tmp_path, scale):
    held = write(tmp_path, "held.txt", HELD[0] + "\n")
    assert run(["--scale", scale, held, "--any-model"], model="typesafe/jev-1.14")[0] == 0     # cached, and the cache says by whom
    code, out, err, oracle = run(["--scale", scale, held])
    assert (code, out, oracle.bodies) == (2, "", [])
    assert "typesafe/jev-1.13" in err and "typesafe/jev-1.14" in err and "cache" in err

    forget_who_answered(tmp_path)                                  # the same answers, no longer saying who gave them
    code, out, err, oracle = run(["--scale", scale, held, "-o"])
    assert code == 0 and oracle.bodies == [] and out.count("\n") == 1
    assert "10 of the 10 answers did not say which model gave them" in err


# ---- The budget: what a placement costs is reserved before it starts ------------------------------------------------

def test_a_price_rise_does_not_overshoot_the_budget_by_a_wave_of_calls(tmp_path, scale):
    held = write(tmp_path, "held.txt", "\n".join(HELD[:40]) + "\n")
    # One cheap reply, then every reply ten times dearer: a probe says nothing about what follows it.
    for extra in ([], ["--keep-order"], ["--unordered"]):
        code, _, err, oracle = run(["--scale", scale, held, "--budget", "0.05", "-j", "32", "--no-cache", *extra],
                                   cost=lambda n: 0.001 if n == 1 else 0.01)
        assert code == 2 and "budget" in err
        assert oracle.spent <= 0.05, (extra, oracle.spent)          # it was $0.311 when the dearest reply so far was the reserve
    # A rise once the run is at full width can only cost the calls already in the air, 32 of them at most here.
    code, _, _, oracle = run(["--scale", scale, held, "--budget", "0.05", "-j", "32", "--no-cache"],
                             cost=lambda n: 0.0001 if n <= 150 else 0.001)
    assert code == 2 and oracle.spent <= 0.05 + 32 * 0.001


def test_a_text_is_placed_in_full_or_not_at_all_when_the_budget_runs_out(tmp_path, scale):
    one = write(tmp_path, "one.txt", HELD[0] + "\n")
    two = write(tmp_path, "two.txt", HELD[0] + "\n" + HELD[1] + "\n")
    _, alone, _, _ = run(["--scale", scale, one, "--json", "--budget", "0.105", "--no-cache"], cost=0.01)
    (alone,) = placed(alone).values()
    assert alone["comparisons"] == 10 and alone["partial"] is False
    for extra in ([], ["--keep-order"]):
        code, out, err, oracle = run(["--scale", scale, two, "--json", "--budget", "0.105", "-j", "2", "--no-cache", *extra], cost=0.01)
        got = placed(out)
        # The budget reaches one text. It gets all its comparisons and the score it has alone; the other gets none.
        assert {k: got[HELD[0]][k] for k in ("score", "se", "comparisons", "partial")} == \
               {k: alone[k] for k in ("score", "se", "comparisons", "partial")}
        assert got[HELD[1]]["score"] is None and got[HELD[1]]["comparisons"] == 0 and got[HELD[1]]["partial"] is None
        assert code == 2 and len(oracle.bodies) == 10 and "1 texts could not be placed" in err and "budget" in err


def test_a_placement_cut_short_is_flagged(tmp_path, scale):
    async def one_fails(body):
        if HELD[1] in body["state"].values() and not failed:
            failed.append(1)
            raise httpx.ConnectError("scripted failure")

    held = write(tmp_path, "held.txt", "\n".join(HELD[:3]) + "\n")
    for extra in ([], ["--keep-order"]):
        failed = []
        code, out, err, _ = run(["--scale", scale, held, "--json", "--no-cache", "--timeout", "0.05", *extra], gate=one_fails)
        got = placed(out)
        assert code == 2 and [got[t]["partial"] for t in HELD[:3]] == [False, True, False]
        assert got[HELD[1]]["comparisons"] == 9 and got[HELD[1]]["score"] is not None
        assert "1 texts were placed on fewer comparisons than -k asked for" in err
    failed = []
    rows = write(tmp_path, "held.csv", "note\n" + "\n".join(HELD[:3]) + "\n")
    _, out, _, _ = run(["--scale", scale, rows, "--csv", "--field", "note", "-o", "--keep-order", "--no-cache", "--timeout", "0.05"],
                       gate=one_fails)
    got = list(csv.DictReader(io.StringIO(out)))
    assert list(got[0])[-2:] == ["jsort_beyond", "jsort_partial"] and [r["jsort_partial"] for r in got] == ["0", "1", "0"]
    # Too few anchors to give -k comparisons is the scale's doing, not a placement cut short.
    few = str(tmp_path / "few.json")
    run([DESCRIPTION, str(tmp_path / "base.txt"), "--save-scale", few, "--anchors", "3"])
    _, out, err, _ = run(["--scale", few, held, "--json", "--no-cache"])
    assert {(o["comparisons"], o["partial"]) for o in placed(out).values()} == {(6, False)} and "fewer comparisons" not in err


# ---- What may become an anchor -------------------------------------------------------------------------------------

def test_a_top_run_is_not_saved_as_a_scale(tmp_path):
    base = write(tmp_path, "base.txt", "\n".join(BASE) + "\n")
    path = tmp_path / "top.json"
    code, out, err, oracle = run(["x", base, "--top", "5", "--save-scale", str(path)])
    assert (code, out, oracle.bodies) == (2, "", []) and not path.exists()       # refused before anything is asked
    assert "--top" in err and "stops asking" in err
    r = jsort.rank(BASE, "x", top=5, transport=httpx.MockTransport(Oracle()))
    with pytest.raises(ScaleError, match="top"):
        r.scale()


def test_a_text_the_run_barely_measured_is_not_an_anchor(tmp_path):
    base = write(tmp_path, "base.txt", "\n".join(BASE) + "\n")
    path = tmp_path / "cut.json"
    # The budget stops this sort at 200 of its 450 comparisons: some texts have five comparisons, the rest four.
    code, _, err, oracle = run(["x", base, "--save-scale", str(path), "--budget", "0.02"], cost=0.0001)
    data = json.loads(path.read_text())
    assert code == 2 and "budget" in err and len(oracle.bodies) < 450
    assert data["fit"]["over_budget"] is True and data["fit"]["anchor_min_comparisons"] == 5
    assert 2 <= data["fit"]["eligible"] < data["fit"]["texts"] == 90
    assert min(a["comparisons"] for a in data["anchors"]) >= 5
    assert f"{data['fit']['eligible']} of the 90 texts" in err and "5 comparisons" in err

    # Stopped earlier still, no text has been measured well enough, and there is nothing to anchor a scale on.
    early = tmp_path / "early.json"
    code, _, err, _ = run(["x", base, "--save-scale", str(early), "--budget", "0.012", "--no-cache"], cost=0.0001)
    assert code == 2 and not early.exists() and "5 comparisons" in err and "was not written" in err
    # A complete sort leaves every text eligible, whatever -k is.
    for k in ("2", "10"):
        whole = tmp_path / f"whole{k}.json"
        assert run(["x", base, "--save-scale", str(whole), "-k", k, "--no-cache"])[0] == 0
        fit = json.loads(whole.read_text())["fit"]
        assert fit["eligible"] == fit["texts"] == 90 and fit["anchor_min_comparisons"] == max(2, int(k) // 2)


# ---- A scale file is checked as closely as it is trusted -----------------------------------------------------------

def test_a_hand_edited_scale_is_refused(tmp_path, scale):
    data = json.loads(open(scale).read())
    held = write(tmp_path, "held.txt", HELD[0] + "\n")
    nan = float("nan")
    for damage, complaint in (
        # The question is what Jev is asked. Reversing it while the description stays would measure the opposite.
        (lambda d: d["question"].update(instructions=f'Text A ranks LOWER, not higher, on this criterion: "{DESCRIPTION}"'), "question"),
        (lambda d: d["question"].update(extra="x"), "question"),
        (lambda d: d["fit"].update(ridge="corrupt"), "ridge"),
        (lambda d: d["fit"].update(ridge=0), "ridge"),
        (lambda d: d["fit"].update(ridge=50), "ridge"),
        (lambda d: d["fit"].update(lean=nan), "lean"),
        (lambda d: d["fit"].update(lean=0.3), "lean"),                        # not the lean its gamma implies
        (lambda d: d["fit"].update(gamma=nan), "lean"),
        (lambda d: d["fit"].update(gamma=40), "lean"),
        (lambda d: d["fit"].update(reliability=1.5), "reliability"),
        (lambda d: d["fit"].update(reliability="high"), "reliability"),
        (lambda d: d["fit"].update(comparisons=-3), "comparisons"),
        (lambda d: d["fit"].update(per_item=1.5), "per_item"),
        (lambda d: d["fit"].update(texts=True), "texts"),
        (lambda d: d["fit"].update(over_budget="no"), "over_budget"),
        (lambda d: d["anchors"][0].update(text="x" * (d["input"]["max_chars"] + 1)), "longer than"),
        (lambda d: d["anchors"][0].update(se=-0.1), "standard error"),
        (lambda d: d["anchors"][0].update(se=nan), "standard error"),
        (lambda d: d["anchors"][0].update(score=1e9), "score"),
        (lambda d: d["anchors"][0].update(comparisons=0), "comparisons"),
        (lambda d: d["anchors"][1].update(text=d["anchors"][0]["text"]), "same text"),
        (lambda d: d["input"].update(unit="chapter"), "unit"),
        (lambda d: d["input"].update(max_chars=2.5), "how many characters"),
        (lambda d: d["model"].update(unknown_answers=-1), "answers"),
        (lambda d: d["model"].update(answered={"typesafe/jev-1.13": 0}), "answers"),
        (lambda d: d["model"].update(answered="typesafe/jev-1.13"), "answers"),
    ):
        broken = json.loads(json.dumps(data))
        damage(broken)
        path = write(tmp_path, "broken.json", json.dumps(broken))
        with pytest.raises(ScaleError, match=complaint):
            Scale.load(path)
        code, out, err, oracle = run(["--scale", path, held])
        assert (code, out, oracle.bodies) == (2, "", []) and path in err, complaint
    assert Scale.load(scale).question == jsort.scale.question(DESCRIPTION) == jsort.engine.question(DESCRIPTION)


def test_placement_uses_the_ridge_the_scale_was_fitted_with(tmp_path, scale):
    data = json.loads(open(scale).read())
    assert data["fit"]["ridge"] == 0.01
    stiff = json.loads(json.dumps(data))
    stiff["fit"]["ridge"] = 0.5
    stiff_path = write(tmp_path, "stiff.json", json.dumps(stiff))
    held = write(tmp_path, "held.txt", "far beyond v=99\n")                 # certain answers: only the ridge holds it
    as_fitted = placed(run(["--scale", scale, held, "--json"], quirk=0.0)[1])["far beyond v=99"]["score"]
    stiffer = placed(run(["--scale", stiff_path, held, "--json"], quirk=0.0)[1])["far beyond v=99"]["score"]
    assert stiffer < as_fitted - 0.5


def test_a_closed_pipe_ends_a_stream_quietly_even_before_the_first_row(tmp_path, scale):
    class Closed(io.StringIO):
        def write(self, s):
            raise BrokenPipeError

    empty = write(tmp_path, "empty.csv", "id,note\n")
    rows = write(tmp_path, "rows.csv", "id,note\n1," + HELD[0] + "\n")
    for f in (empty, rows):                  # the header is the first thing a CSV stream writes
        for extra in (["--keep-order"], ["--unordered"], []):
            code, _, err, oracle = run(["--scale", scale, f, "--csv", "--field", "note", "-o", *extra], out=Closed())
            assert code == 0 and "Traceback" not in err
            assert not extra or oracle.bodies == []          # a stream whose reader has gone asks nothing more
    assert run(["x", empty, "--csv", "--field", "note"], out=Closed())[0] == 0      # an ordinary sort's header as well


def test_a_stand_in_judge_can_sort_save_and_place():
    # bench/simulate.py drives arank and aplace with an object that has only `ask` and `meter`. Keep that working.
    from jsort.core import Meter
    from jsort.engine import arank

    class Judge:
        meter = Meter()

        async def ask(self, state, questions, *, on_cost=None, provenance=None):
            if provenance is not None:
                provenance["q"] = {"resolved_model": "stand-in", "source": "api"}
            value = lambda t: float(re.search(r"v=(-?[\d.]+)", t).group(1))
            return {"q": {"noul": 1 / (1 + math.exp(-(value(state["A"]) - value(state["B"]))))}}

    async def go():
        r = await arank(BASE[:30], "x", Judge())
        built = r.scale(10)
        assert built.model["answered"] == {"stand-in": r.asked} and built.answered_by == "stand-in"
        for budget in (0, 1.0):
            p = await aplace(HELD[:5], built, Judge(), budget=budget, any_model=True)
            assert p.asked == 50 and not np.isnan(p.score).any() and not p.partial.any()

    asyncio.run(go())


@pytest.mark.parametrize("extra", [[], ["--keep-order"], ["--unordered"]])
@pytest.mark.parametrize("budget", ["0", "1"])
def test_a_moved_alias_sends_one_probe_and_a_cached_rerun_prints_nothing(tmp_path, scale, extra, budget):
    async def delay(body):
        await asyncio.sleep(0.002)       # a concurrent wave can start before the first answer arrives

    held = write(tmp_path, "held.txt", "\n".join(HELD[:40]) + "\n")
    argv = ["--scale", scale, held, "--json", "--budget", budget, "-j", "32", *extra]
    code, out, err, oracle = run(argv, model="typesafe/jev-1.14", gate=delay)
    assert (code, out) == (2, "") and "answers now come from typesafe/jev-1.14" in err
    assert len(oracle.bodies) == 1
    code, out, err, oracle = run(argv, model="typesafe/jev-1.14", gate=delay)
    assert (code, out, oracle.bodies) == (2, "", []) and "cache holds an answer from" in err


def test_cached_matching_answers_do_not_confirm_the_first_live_model(tmp_path, scale):
    # A completed placement is cached from the old model. The first uncached answer must still go alone.
    one = write(tmp_path, "one.txt", HELD[0] + "\n")
    assert run(["--scale", scale, one])[0] == 0

    async def delay(body):
        await asyncio.sleep(0.002)

    held = write(tmp_path, "held.txt", "\n".join(HELD[:20]) + "\n")
    for width in ("1", "32"):
        code, out, err, oracle = run(["--scale", scale, held, "--json", "--budget", "0", "-j", width],
                                    model="typesafe/jev-1.14", gate=delay)
        assert code == 2 and out == "" and "typesafe/jev-1.14" in err
        assert len(oracle.bodies) == (1 if width == "1" else 0)


@pytest.mark.parametrize("extra", [[], ["--keep-order"], ["--unordered"]])
@pytest.mark.parametrize("reason", ["far", "few", "failed"])
def test_unreliable_placed_errors_are_empty_in_every_format(tmp_path, scale, extra, reason):
    text = "outside v=99" if reason == "far" else HELD[0]
    k = "2" if reason == "few" else "3" if reason == "failed" else "10"
    for format_ in ("plain", "json", "csv", "jsonl"):
        failed = []

        async def fail_once(body):
            if reason == "failed" and not failed:
                failed.append(1)
                raise httpx.ConnectError("scripted failure")

        data = "note\n" + text + "\n" if format_ == "csv" else json.dumps({"note": text}) + "\n" if format_ == "jsonl" else text + "\n"
        source = write(tmp_path, "input.txt", data)
        flags = ["--" + format_, "--field", "note", "-o"] if format_ in ("csv", "jsonl") else ["--json"] if format_ == "json" else ["-o"]
        code, out, err, _ = run(["--scale", scale, source, "-k", k, "--no-cache", "--timeout", "0.01", *flags, *extra], gate=fail_once)
        assert code == (2 if reason == "failed" else 0), err
        if format_ == "plain":
            score, se, _ = out.rstrip("\n").split("\t")
            assert math.isfinite(float(score)) and se == ""
        else:
            row = next(csv.DictReader(io.StringIO(out))) if format_ == "csv" else json.loads(out)
            prefix = "" if format_ == "json" else "jsort_"
            assert math.isfinite(float(row[prefix + "score"]))
            assert row[prefix + "se"] == ("" if format_ == "csv" else None)
            assert row[prefix + "beyond"] == ("above" if reason == "far" else "" if format_ == "csv" else None)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("margin", [1.99, 2.0, 2.01])
def test_placed_error_is_suppressed_only_more_than_two_logits_beyond(scale, monkeypatch, sign, margin):
    low, high = Scale.load(scale).span
    score = (high if sign > 0 else low) + sign * margin
    monkeypatch.setattr("jsort.placement.locate", lambda *a, **kw: (score, 0.4))
    result = jsort.place([HELD[0]], scale, cache=False, transport=httpx.MockTransport(Oracle()))
    assert result.score[0] == score and result.beyond[0] == sign
    assert np.isnan(result.se[0]) if margin > 2 else result.se[0] == 0.4


@pytest.mark.parametrize("extra", [[], ["--keep-order"], ["--unordered"]])
@pytest.mark.parametrize("over_budget,failed", [(True, 0), (False, 2), (True, 2), (False, 0)])
def test_loading_an_incomplete_scale_warns_and_still_places(tmp_path, scale, extra, over_budget, failed):
    data = json.loads(open(scale).read())
    data["fit"].update(over_budget=over_budget, failed=failed)
    path = write(tmp_path, "incomplete.json", json.dumps(data))
    held = write(tmp_path, "held.txt", HELD[0] + "\n")
    code, out, err, _ = run(["--scale", path, held, "--json", *extra])
    assert code == 0 and placed(out)[HELD[0]]["score"] is not None
    assert ("saved from an incomplete fit" in err) == bool(over_budget or failed)
    if over_budget or failed:
        assert path in err and "rebuild" in err
        assert ("over budget" in err) == over_budget
        assert ("2 failed comparisons" in err) == bool(failed)


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_schema_version_is_an_integer(tmp_path, scale, version):
    data = json.loads(open(scale).read())
    data["schema_version"] = version
    with pytest.raises(ScaleError, match="schema"):
        Scale.from_json(data)


def test_placer_uses_the_saved_lean(tmp_path, scale):
    data = json.loads(open(scale).read())
    gamma, target = 1.8, 0.7
    data["fit"].update(gamma=gamma, lean=1 / (1 + math.exp(-gamma)) - 0.5)
    saved = Scale.from_json(data)
    values = {a.text: a.score for a in saved.anchors} | {"new text": target}

    async def exact(request):
        body = json.loads(request.content)
        state = body["state"]
        p = 1 / (1 + math.exp(-(values[state["A"]] - values[state["B"]] + gamma)))
        return httpx.Response(200, json={"model": saved.answered_by, "answers": {"q": {"type": "noul", "noul": p}}})

    result = jsort.place(["new text"], saved, per_item=3, cache=False, transport=httpx.MockTransport(exact))
    assert result.comparisons[0] == 3 and abs(result.score[0] - target) < 0.04


@pytest.mark.parametrize("extra", [[], ["--keep-order"], ["--unordered"]])
@pytest.mark.parametrize("width", ["1", "8"])
def test_duplicate_shown_texts_share_one_placement_without_a_cache(tmp_path, scale, extra, width):
    class Jitter(Oracle):
        async def __call__(self, request):
            response = await super().__call__(request)
            payload = response.json()
            p = payload["answers"]["q"]["noul"]
            payload["answers"]["q"]["noul"] = 1 / (1 + math.exp(-(math.log(p / (1 - p)) + 0.05 * len(self.bodies))))
            await asyncio.sleep(0.001)
            return httpx.Response(200, json=payload)

    text = HELD[0]
    held = write(tmp_path, "copies.txt", "\n".join(text + f" unseen suffix {i}" for i in range(6)) + "\n")
    code, out, err, oracle = run(["--scale", scale, held, "--json", "--no-cache", "--budget", "0", "-j", width,
                                "--max-chars", str(len(text)), *extra], oracle=Jitter())
    rows = list(map(json.loads, out.splitlines()))
    assert code == 0 and len(rows) == 6, err
    assert len(oracle.bodies) == 10
    assert len({(r["score"], r["se"], r["comparisons"], r["beyond"], r["partial"]) for r in rows}) == 1


def test_zero_cost_replies_release_the_first_request_gate(scale):
    flying, peak = 0, 0

    async def delay(body):
        nonlocal flying, peak
        flying += 1
        peak = max(peak, flying)
        await asyncio.sleep(0.002)
        flying -= 1

    result = jsort.place(HELD[:8], scale, cache=False, concurrency=8, transport=httpx.MockTransport(Oracle(cost=0, gate=delay)))
    assert result.asked == 80 and peak > 1
