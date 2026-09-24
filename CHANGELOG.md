# Changelog

## Unreleased

- Add `--api gliner`, a local [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide)
  server (`JEV_GLINER_URL`, port 8082). Requires `jevkit-runtime>=0.3.2`.

## 0.3.1

- The tool itself is unchanged from 0.3.0; this release brings its README on PyPI up to date.
- Compare the two local servers with Jev 1.13 in `docs/benchmarks/local-models-2026-09-22.md`,
  summarized in the README. The readability comparison moves there from `docs/three-models-2026-09-22.md`.
- Add `bench/local_models.py`, which runs the YC and Fed sorts on a local server and scores them
  beside Jev's recorded runs.
- `bench/fed.py check` sets the saved statement scores against the rate move on action days alone
  and against FedLock's published scores, without a model call; the README's Fed paragraph now
  reports both.

## 0.3.0

- Add `--api diffusiongemma` and `--api laya` for System One servers running on your own machine,
  through `jevkit-runtime` 0.3: chosen only by name, no key needed, and metered at zero API fees
  unless `JEV_PRICE_PER_MTOK` is set. The runtime's `docs/` explain how to run the servers.

## 0.2.0

- Move transport, configuration, the answer cache and metering to the shared `jevkit-runtime` 0.2. Answers are
  keyed by provider, endpoint and model and stored beside who answered them; the cache written by earlier
  versions is reset on first use and re-asked.
- Save a run's scale with `--save-scale FILE`: the description, the question as it was asked, the API, endpoint
  and model ID that were asked, a count of the fit's answers by the model that gave them, `--max-chars`, a summary
  of the fit and the anchors, each a text with its score and standard error. `--anchors N` (default 30) keeps the
  highest and lowest texts and, between them, the text with the smallest reported error in each equal stretch
  of the score range. Anchor texts are stored verbatim as shown to Jev; sharing the file shares those texts.
- Only a text with at least half of `-k` comparisons can be an anchor, which excludes nothing in a complete sort.
  `--save-scale` cannot be combined with `--top`, whose far end is barely measured, and a run the budget cut
  short records how many texts qualified and says so.
- Record which model gave each answer beside the answer in the shared cache. A scale is saved only when every answer behind it names the same model; one that
  mixes two, or rests on answers that name none, is refused with the reason. `--any-model` saves it anyway, with
  the counts as they are.
- Place new texts on a saved scale with `--scale FILE`. Each text is compared with anchors only, chosen
  adaptively over about three rounds, and scored by the one-parameter version of the fit, with the anchors, the
  first-position lean and the ridge held where the scale left them. Every text gets all `-k` of its comparisons.
  The description comes from the file; a different one on the command line is an error.
- A text placed in full scores the same alone or in any company, so `--scale` can stream: `--keep-order` prints
  each text once it and those before it are placed, with `-j` bounding the texts in hand as in jgrep, and
  `--unordered` prints as texts are placed. Sorted output still waits for the end of the input.
- Under `--scale` the budget prices each request from its size before sending it, sends the first alone, widens
  from one text in hand to `-j` as prices hold, and begins a text only if it can pay for all of it. When the money
  runs out some texts are placed in full and the rest not at all. Calls in the air when a price rises can still
  exceed it.
- Flag a text placed beyond the anchors (`beyond`) and one cut short of its comparisons (`partial`): in `--json`,
  as `NAME_beyond` and `NAME_partial` with `-o` on CSV and JSONL, and on stderr.
- Refuse a scale built with another API, endpoint or model, or one that cannot name its model, before any call.
  Every answer used is then checked against the scale's model, cached answers included, which is how an alias
  that has moved is caught. `--any-model` overrides.
- A scale file is validated field by field, and its question must be the one jsort asks for its description.
- Leave a new text's placement standard error empty below three successful comparisons or more than two logits
  beyond an end anchor; keep its score and flags. Empty errors are blank in plain output and CSV, null in JSON
  and JSONL. Document censoring beyond the anchors, noisy errors at small `-k`, and the limits of interval coverage.
- Send the first uncached placement request alone until the answering model is confirmed, including with
  `--budget 0`. Refuse batch output on a model mismatch even after valid cached answers. A confirmed reply with
  zero reported cost releases the gate; budget reservations still use the list-price estimate.
- Reuse a placement for duplicate shown texts within the run, including with `--no-cache`. This retains a text
  hash and result for each distinct shown text, including while streaming.
- Warn when `--scale` loads a fit marked over budget or with failed comparisons. Require an integer schema version.
- From Python: `Ranking.scale()`, `jsort.Scale`, `jsort.place` and `jsort.aplace`; `Jev.ask(..., provenance={})`
  reports who gave each answer.
- `bench/simulate.py` also measures placement against a fit to every pair, offline.
- A negative `--seed` is a usage error, not a traceback. A closed pipe no longer raises while a CSV header is written.

## 0.1.3

- Isolate cached answers by provider, endpoint, and model, and reset per-run budgets.
- Handle exhausted budgets, trivial top results, and comparison caps without unnecessary calls.
- Validate ranking inputs, including malformed or non-finite values, before scheduling comparisons.

## 0.1.2

- The Fed example now covers every FOMC press conference, 95 of them from April 2011: the scraper had
  missed one page the Fed spells differently and the three conferences of 2011.

## 0.1.1

- Rewrite the README around what jsort is for: the benchmark discussion, tips and cost notes.

## 0.1.0

- Sort lines, paragraphs (`--para`), whole files (`--whole`), CSV rows and JSONL records (`--field`) along a
  plain-English dimension, from pairwise comparisons judged by Jev.
- Fit a Bradley-Terry scale to Jev's probabilities as a fractional logit, with a term for the lean toward
  the first-shown text. Report a score and a robust standard error per text (`-o`), split-half reliability
  and the first-position lean.
- Choose pairs adaptively: a random ring, then near neighbours on the current scale. `-k` sets the
  comparisons per text; `--top N` retires texts that cannot reach the top and spends their questions on
  the contenders.
- Add scores to a dataset without reordering it (`-o --keep-order --name NAME`), emit JSON (`--json`), and
  rank from Python (`jsort.rank`).
- Seeded pair selection and the shared answer cache make a rerun free; `--budget` stops the questions and
  sorts on what is known.
- Benchmarks: a simulated judge (`bench/simulate.py`), teachers' pairwise readability judgments
  (`bench/readability.py`) and FOMC opening statements against the policy rate (`bench/fed.py`).
