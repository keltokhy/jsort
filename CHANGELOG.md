# Changelog

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
