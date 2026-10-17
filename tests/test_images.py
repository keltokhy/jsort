"""Sorting pictures: PNG and JPEG files compared as images, against a fake Decisions endpoint."""

import base64
import io
import json
import math

import httpx
import pytest
from PIL import Image as Pil

from jsort.cli import main


def picture(path, red):
    Pil.new("RGB", (400, 300), (red, 40, 40)).save(path, "PNG")
    return str(path)


def redness(part) -> int:
    data = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
    return Pil.open(io.BytesIO(data)).convert("RGB").getpixel((200, 150))[0]


class Fake:
    """A ranks higher the redder its picture is than B's."""

    def __init__(self):
        self.bodies = []

    async def __call__(self, request):
        body = json.loads(request.content)
        self.bodies.append(body)
        gap = (redness(body["state"]["A"]) - redness(body["state"]["B"])) / 40
        answers = {qid: {"type": "noul", "noul": round(1 / (1 + math.exp(-gap)), 4)} for qid in body["questions"]}
        usage = {"input_tokens": 800, "output_tokens": 1, "cost": 0.0001}
        return httpx.Response(200, json={"model": "typesafe/jev-1.13", "answers": answers, "usage": usage})


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for name in ("TYPESAFE_API_KEY", "JEV_API", "JEV_MODEL", "JEV_URL", "JEV_BUDGET"):
        monkeypatch.delenv(name, raising=False)


def jsort(argv):
    fake = Fake()
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, transport=httpx.MockTransport(fake), out=out, err=err)
    return code, out.getvalue(), err.getvalue(), fake


def test_image_files_sort_by_what_their_pictures_show(tmp_path):
    reds = (60, 220, 140, 100, 180)
    files = [picture(tmp_path / f"p{red}.png", red) for red in reds]
    code, out, err, fake = jsort(["redder", *files])
    assert code == 0, err
    expected = [f for _, f in sorted(zip(reds, files), reverse=True)]
    assert out.splitlines() == expected
    question = fake.bodies[0]["questions"]["q"]["instructions"]
    assert question == 'The picture in A ranks higher than the picture in B on this criterion: "redder"'


def test_scores_print_beside_file_names(tmp_path):
    files = [picture(tmp_path / f"p{red}.png", red) for red in (60, 220, 140)]
    code, out, _, _ = jsort(["-o", "redder", *files])
    rows = [line.split("\t") for line in out.splitlines()]
    assert [row[2] for row in rows] == [files[1], files[2], files[0]]
    assert float(rows[0][0]) > float(rows[-1][0])


def test_images_and_text_cannot_be_sorted_together(tmp_path):
    image = picture(tmp_path / "p.png", 100)
    notes = tmp_path / "notes.txt"
    notes.write_text("a line\nanother\n")
    code, _, err, fake = jsort(["redder", image, str(notes)])
    assert code == 2 and "texts or images, not both" in err and fake.bodies == []


def test_saved_scales_are_refused_for_images(tmp_path):
    files = [picture(tmp_path / f"p{red}.png", red) for red in (60, 220)]
    code, _, err, _ = jsort(["--save-scale", str(tmp_path / "s.json"), "redder", *files])
    assert code == 2 and "do not work with images yet" in err


def test_a_provider_without_images_stops_with_a_clear_error(tmp_path):
    files = [picture(tmp_path / f"p{red}.png", red) for red in (60, 220)]
    code, _, err, fake = jsort(["--api", "laya", "redder", *files])
    assert code == 2 and "laya does not read images" in err and fake.bodies == []
