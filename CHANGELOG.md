# Changelog

## Unreleased

- Save a run's scale with `--save-scale FILE`: the description, the question as it was asked, the API, endpoint
  and model, `--max-chars`, a summary of the fit and the anchors, each a text with its score and standard error.
  `--anchors N` (default 30) keeps the highest and lowest texts and, between them, the best measured text of each
  equal stretch of the score range.
- Place new texts on a saved scale with `--scale FILE`. Each text is compared with anchors only, chosen
  adaptively over about three rounds, and scored by the one-parameter version of the fit, with the anchors and
  the first-position lean held where the scale left them. `-k` bounds the comparisons per text and
  `--se-target` stops early. The description comes from the file; a different one on the command line is an error.
- A placed text's score does not depend on the rest of the input, so `--scale` can stream: `--keep-order` prints
  each text once it and those before it are placed, with `-j` bounding the texts in hand as in jgrep, and
  `--unordered` prints as texts are placed. Sorted output still waits for the end of the input.
- Flag a text placed beyond the anchors: `beyond` in `--json`, `NAME_beyond` with `-o` on CSV and JSONL, and a
  note on stderr. Its score is a finite extrapolation.
- Refuse a scale built with another API, endpoint or model: before any call where the IDs differ, and on the
  first reply when an alias now reaches another model. `--any-model` overrides.
- From Python: `Ranking.scale()`, `jsort.Scale`, `jsort.place` and `jsort.aplace`.
- `bench/simulate.py` also measures placement against a fit to every pair, offline.
- A negative `--seed` is a usage error, not a traceback.

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
