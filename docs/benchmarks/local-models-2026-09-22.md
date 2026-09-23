# Jev, DiffusionGemma, and Laya on jsort's benchmarks

Run September 22, 2026, on an Apple M3 Ultra with 96 GiB of unified memory, on four sorts: the
readability sample, 300 CommonLit excerpts on "easier to read"; 500 YC company one-liners on "sounds
more like science fiction"; and, on "more hawkish about inflation", the 51 sentences of the latest
FOMC opening statement and the 95 opening statements since April 2011, whole. Each local model
sorted through the installed `jsort` command at `-k 10`, with the provider named on the command
line, an empty answer cache and `--budget 0`, one model and one benchmark at a time. Jev
1.13 was not called again: its column is read from its recorded runs of September 19 and 20. The
readability section is the comparison `bench/three_models.py` ran earlier in the day, unchanged,
with its own frozen output. `bench/local_models.py` ran the other three from 21:45 to 22:34
America/New_York and scores them; [local-models-2026-09-22.json](local-models-2026-09-22.json) is
their frozen output.

## Readability

Run September 22, 2026, on the 300-excerpt sample that `bench/readability.py prepare` draws from the
CommonLit Ease of Readability corpus with seed 20260919. Each model sorted the same excerpts on
"easier to read" with `jsort -k 10`, through the installed command, with an empty answer cache, and
then answered the one-question alternative, p("easy to read"), once per excerpt. `bench/three_models.py`
runs it; [three-models-2026-09-22.json](../three-models-2026-09-22.json) is the frozen output.

The teachers' scale has a reliability of about 0.78 on this sample, so no measure can correlate with
it above about 0.88.

| | Jev 1.13 (`typesafe/jev-1.13`, OpenRouter) | DiffusionGemma (OpenJev, `openjev-0.1`, MLX) | Laya (`laya-421m`, MLX) |
|---|---:|---:|---:|
| `jsort -k 10`, Pearson r | **0.824** | 0.783 | 0.032 |
| `jsort -k 10`, Spearman ρ | **0.841** | 0.799 | 0.045 |
| Reliability (two halves of the comparisons agree) | 0.98 | 0.92 | 0.42 |
| Comparisons answered | 1,500 of 1,500 | 1,500 of 1,500 | 1,149 of 1,500 |
| Excerpts scored | 300 | 300 | 289 |
| One question per excerpt, Pearson r | 0.754 | 0.655 | −0.092 |
| One question per excerpt, distinct values | 73 | 300 | 282 |
| Wall-clock for the sort | under a minute | 823 s | 43 s |
| API cost for the sort | $0.046 | $0 | $0 |

Agreement with the teachers on which of two excerpts is easier, on random pairs both scored, by how
far apart the teachers put them (about 5,000 to 13,000 pairs per row):

| Gap on the teachers' scale | Jev | DiffusionGemma | Laya |
|---|---:|---:|---:|
| 0.25 to 0.5 | 0.675 | 0.627 | 0.511 |
| 0.5 to 1.0 | 0.771 | 0.749 | 0.520 |
| 1.0 to 2.0 | 0.925 | 0.900 | 0.527 |
| 2.0 and above | 0.990 | 0.982 | 0.512 |

On the 289 excerpts every model scored, DiffusionGemma's r is 0.789 and Laya's 0.032.

### What the table says

- **DiffusionGemma does the job, a few points below Jev.** Four points of r, two of ρ, and it
  keeps the ordering property that matters for a sort: agreement rises with the gap and reaches
  98% on clearly different excerpts. Its reliability of 0.92 is below Jev's 0.98, so a second run
  would reorder more close neighbours. Its one-question scores are far more spread out than Jev's
  (300 distinct values against 73) but correlate less, which is the pattern of a model that gives
  confident, noisy probabilities. Pairwise comparison recovers most of the gap, as it did for Jev.
- **Laya cannot do this task.** On the pairs it could read it agrees with the teachers at chance in
  every gap bucket, including excerpts the teachers put more than two logits apart, and reliability
  of 0.42 says two halves of its own comparisons disagree. The one-question variant is no better.
  This is not the context limit: 23% of pairs were refused for exceeding its 512-token window, and
  the excerpts it did read fare the same. It is consistent with the jgrep comparison, where Laya
  classified short news well and did little else.
- **Cost and speed.** Jev's sort cost five cents and finished in under a minute at 32 requests in
  flight. DiffusionGemma ran two requests at a time against a server that runs model work one call
  at a time on the GPU, 0.55 s per comparison, so fourteen minutes for 300 excerpts; a full corpus
  of 4,724 excerpts would take about four hours. Laya is fast and free but the answers are not
  usable here.

## YC one-liners

"sounds more like science fiction" on 500 of the 6,081 company one-liners in `bench/out/yc/yc.csv`,
drawn with seed 20260922, each as the `text` field Jev saw. Jev's scores for the same 500 companies
come from its sort of all 6,081 on September 20, so each local model is compared with Jev's scale
for the whole directory, not with a 500-company Jev run.

| | Jev 1.13 (OpenRouter, recorded run) | DiffusionGemma (OpenJev, `openjev-0.1`, MLX) | Laya (`laya-421m`, MLX) |
|---|---:|---:|---:|
| Spearman ρ with Jev's scores, same 500 companies | | 0.654 | −0.129 |
| Pearson r with Jev's scores | | 0.657 | −0.115 |
| Jev's top 20 found in the model's top 20 | | 4 | 0 |
| Reliability (two halves of the comparisons agree) | 0.97* | 0.62 | 0.45 |
| Companies scored | 6,081* | 500 of 500 | 500 of 500 |
| Comparisons answered | 30,401* | 2,500 of 2,500 | 2,500 of 2,500 |
| Requests refused for exceeding the context window | | | 0 of 2,500 |
| Wall-clock for the sort | 503.9 s* | 774 s | 51 s |
| API cost for the sort | $0.412* | $0 | $0 |

*The sort of all 6,081 companies at `-j 32`, of which these 500 are a sample. No 500-company Jev
run exists.

## Fed: the sentences of the latest statement

The first sort in `bench/fed.py`: "more hawkish about inflation" on the 51 sentences of
`bench/out/fed/latest.txt`, the opening statement sorted at the top of the README. Jev's column is
its September 19 run, rebuilt from the answers it cached (see the caveats).

| | Jev 1.13 (OpenRouter, recorded run) | DiffusionGemma (OpenJev, `openjev-0.1`, MLX) | Laya (`laya-421m`, MLX) |
|---|---:|---:|---:|
| Spearman ρ with Jev's scores, 51 sentences | | 0.522 | −0.110 |
| Pearson r with Jev's scores | | 0.652 | −0.148 |
| Jev's top 10 found in the model's top 10 | | 7 | 1 |
| Reliability (two halves of the comparisons agree) | 0.96 | 0.39 | 0.37 |
| Sentences scored | 51 | 51 of 51 | 51 of 51 |
| Comparisons answered | 255 | 254 of 254 | 254 of 254 |
| Requests refused for exceeding the context window | | | 0 of 254 |
| Wall-clock and API cost for the sort | not logged | 108 s, $0 | 15 s, $0 |

## Fed: the 95 statements, whole

The second sort in `bench/fed.py`: the same description on all 95 opening statements from April 2011
to September 2026, with `--whole --max-chars 16000`, scored as `bench/fed.py` scores Jev: the rank
correlation of each statement's score with the move in the top of the federal funds target range
announced that day, and with the change over the next 180 days, which 91 of the statements have.

| | Jev 1.13 (OpenRouter, recorded run) | DiffusionGemma (OpenJev, `openjev-0.1`, MLX) | Laya (`laya-421m`, MLX) |
|---|---:|---:|---:|
| Rank correlation with the move that day (95 statements) | +0.46 | +0.60 | none scored |
| Rank correlation with the move that day, action days only (32) | +0.80 | +0.92 | none scored |
| Rank correlation with the next 180 days (91 statements) | +0.37 | +0.55 | none scored |
| Mean score: cuts (11) / holds (63) / hikes (21) | −1.03 / −0.70 / +2.65 | −2.02 / −0.73 / +3.25 | none scored |
| Spearman ρ with FedLock's raw scores, file of 2026-09-23 (89 matched) | +0.85 | +0.86 | none scored |
| Spearman ρ with FedLock's raw scores, file of 2026-09-20 (92 matched) | +0.93 | +0.91 | none scored |
| Spearman ρ with Jev's scores | | 0.893 | |
| Pearson r with Jev's scores | | 0.943 | |
| Reliability (two halves of the comparisons agree) | 0.98 | 0.92 | |
| Statements scored | 95 | 95 of 95 | 0 of 95 |
| Comparisons answered | 475 | 475 of 475 | 0 of 474 |
| Requests refused for exceeding the context window | | | 474 of 474 |
| Wall-clock for the sort | 9.2 s* | 948 s | 1.8 s |
| API cost for the sort | $0.0568* | $0 | $0 |

*407 calls; the other 68 of the 475 answers came from an earlier cache, so 9.2 s is not a clean
wall-clock.

## What the YC and Fed tables say

- **On whole statements DiffusionGemma agrees with Jev and follows the Fed at least as closely.**
  Its scores for the 95 statements correlate 0.893 (Spearman) and 0.943 (Pearson) with Jev's.
  Scored the way `bench/fed.py` scores Jev, its rank correlation with the move announced that day
  is +0.60 against Jev's +0.46, +0.92 against +0.80 on the 32 meetings that moved the rate, and
  with the change over the next 180 days +0.55 against +0.37. Against FedLock, an independent
  scale, the two are level: +0.86 and +0.85 with the file of 2026-09-23, +0.91 and +0.93 with the
  file of 2026-09-20. That is one run of each on one description: it shows DiffusionGemma keeping
  up with Jev on documents of this length, not that it reads the Fed better. Its reliability is
  0.92 against Jev's 0.98.
- **On single lines it agrees with Jev in part.** Its order has a Spearman of 0.654 with Jev's on
  the 500 one-liners, with 4 of Jev's top 20 in its own, and 0.522 on the 51 sentences, with 7 of
  Jev's top 10. Both put "The plain fact is that inflation is too high and has been for too long."
  first. Its own reliability was 0.62 and 0.39, against Jev's 0.97 over all 6,081 companies and
  0.96 on the same sentences, and jsort printed its low-reliability warning on both runs. jsort's
  advice there, a higher `-k`, was not tried.
- **Laya's order does not follow Jev's on short texts.** Spearman −0.129 on the one-liners, with
  none of Jev's top 20, and −0.110 on the sentences, with 1 of Jev's top 10; reliability 0.45 and
  0.37. None of its 2,500 and 254 requests there was refused for context, so it read every text
  whole. It put "Today’s policy action will support a timelier return to the Committee’s 2 percent
  goal." first. This matches its readability result above, where it agreed with the teachers at
  chance.
- **Laya reads at most 512 tokens, so it scored none of the 95 statements.** All 474 of its
  requests were refused with HTTP 422, and jsort exited with status 2 and listed the 95 statements
  as never compared. Nothing was truncated or otherwise worked around; that coverage is the result.
- **Speed and cost.** At 4 requests in flight, on a server that runs model work one call at a time,
  DiffusionGemma took 774 s for the 2,500 one-liner comparisons, 108 s for the 254 sentence
  comparisons and 948 s for the 475 statement comparisons, 2.0 s each. Laya took 51 s and 15 s for
  the first two. Neither cost anything in API fees. Jev's recorded sort of all 6,081 companies,
  30,401 comparisons at 32 in flight, took 503.9 s and cost $0.412; its two Fed runs have no clean
  wall-clock.

## Provenance and caveats

### Readability

- The Jev column is the September 19 run from `bench/out/readability.json`, on the identical sample,
  seed, and setting; the OpenRouter key hit its spending limit on September 22 before a same-day
  rerun could start, and the paired rerun will replace this column when it runs. Model
  `typesafe/jev-1.13` was pinned both times.
- Local servers: DiffusionGemma is OpenJev at commit `e04794a`, serving `openjev-0.1` from
  `mlx-community/diffusiongemma-26B-A4B-it-4bit` revision `a7a8140`, which runs model work one call
  at a time on the GPU. Laya is laya-mlx at commit `fc1df62` with `aac6fef/laya-mlx` revision
  `0476785`, served as `laya-421m` through jevkit-core's `scripts/laya_server.py`, which refuses
  with HTTP 422 any request whose state would be cropped to fit its 512-token window, question
  included. Both on an Apple M3 Ultra with 96 GiB of unified memory.
- Laya's refused pairs are counted from the server's audit log, 351 of 1,500 requests; jsort recorded
  353 failed comparisons, the other two being deadline errors. Refused pairs are skipped, not
  truncated, so Laya is scored on what it read.
- One sample, one description, one `-k`. Nothing was tuned on these results.

### YC and the Fed

- **Jev column.** Jev 1.13 made no calls for these benchmarks; every number is read from a dated
  file. YC: `bench/out/yc/yc-scifi.csv` (scores) and `yc-scifi.log` (stats), September 20, one sort
  of all 6,081 companies: 30,401 comparisons, reliability 0.97, $0.4120, 503.9 s at `-j 32`. The 500
  sampled companies are joined to it on the exact text Jev saw, so the local correlations measure
  agreement with Jev's scale for the whole directory, and Jev's reliability, wall-clock and cost in
  that table are for all 6,081. Sentences: `bench/out/local-2026-09-22/jev-latest.json`, a read-only
  replay of jsort 0.1.2 (commit `2897d18`) over the answers `bench/fed.py` cached on September 19,
  keyed to `~typesafe/jev-latest`, the OpenRouter alias that run requested. It reproduces all 10
  scores that `bench/out/fed/run.log` prints, with no mismatch; the other 41 are reconstructed, and
  the first run's wall-clock and cost were never logged (run.log shows a cached rerun). Statements:
  `bench/out/fed/fed.json` and `run.log`, September 19; the runner recomputes +0.46 and +0.37 from
  them on the same statements.
- **Action days and FedLock.** `bench/fed.py check`, added September 23, computes two more checks
  from the saved scores without a model call, and `freeze` applies the same code to each local
  model's statement scores. The same-day move is zero on every hold, 63 of the 95 meetings, so the
  action-day row keeps the 32 meetings that moved the rate. [FedLock](https://jnathan9.github.io/fedlock/)
  (Joe Weisenthal) is a running pairwise tournament over about 4,000 Fed speeches, scored with an
  open-weights model and aggregated with TrueSkill; its published `data.json` carries a raw mean
  `m` and an era-adjusted `ma` per speech, and its press conferences are matched to the statements
  by date, within two days (FedLock usually dates a press conference the day after the meeting).
  The rows use `m`; `ma` is in the JSON. FedLock's scores move between its releases: on the 93
  press conferences in both, its file of 2026-09-23 (93 press conferences; SHA-256 `aec9e832…`,
  downloaded by `check`) and its file of 2026-09-20 (96; the snapshot archived by
  [fedjev-bench](https://github.com/maybern-tripp-smith/fedjev-bench) at
  `data/raw/fedlock/data.json`) have a rank correlation of +0.88 raw and +0.66 era-adjusted, and
  the three latest conferences are absent from the later file. So agreement with any one file is
  bounded by that, and both files are reported. FedLock scores full press conferences where jsort
  scores the openings.
- **Local servers.** DiffusionGemma: OpenJev at commit `e04794a`, serving `openjev-0.1` from
  `mlx-community/diffusiongemma-26B-A4B-it-4bit` revision `a7a8140`, which runs model work one call
  at a time on the GPU. Laya: laya-mlx at commit `fc1df62` with `aac6fef/laya-mlx` revision
  `0476785`, served as `laya-421m` through jevkit-core's `scripts/laya_server.py`, which refuses
  with HTTP 422 any request whose state would be cropped to fit its 512-token window, question
  included; jsort counts a refusal as a failed comparison, not an answer. Both on an Apple M3 Ultra
  with 96 GiB of unified memory and loaded throughout. Runs were serialized, all DiffusionGemma and
  then all Laya, so during each run the other server was loaded but idle.
- **Concurrency, deadlines and cache.** `-j 4` on every run, with `--timeout 300` for DiffusionGemma
  and `--timeout 120` for Laya. There were no deadline errors, so concurrency was never lowered,
  and each benchmark ran once. Every run had a new empty `XDG_CACHE_HOME` and `XDG_CONFIG_HOME` and
  `--budget 0`; local calls are metered at $0, and no hosted model was called.
- **Laya refusals** are counted from the adapter's audit log, by the lines each run added: 0 of
  2,500 requests on the one-liners, 0 of 254 on the sentences, and 474 of 474 on the statements,
  all with status `context_rejected`.
- **Samples.** The YC sample is 500 of the 6,081 companies, drawn with
  `numpy.random.default_rng(20260922)` and kept in `bench/out/local-2026-09-22/yc-sample500.csv`.
  The sentences and the statements are complete. Before the statements, a probe of 5 statements
  drawn with the same seed, at `-k 2`, took 7.16 s per comparison on DiffusionGemma, which
  projected 56.7 minutes for all 95, inside the 60 minutes allowed, so neither model was given a
  subsample. The probe was a single round of five comparisons on statements from December 2022 on,
  and it overstated the cost: the full run took 2.0 s per comparison, 15.8 minutes.
- **Comparison counts.** Both local models ran 254 comparisons on the sentences, where Jev's run
  had 255, and Laya's statement run drew 474 pairs where DiffusionGemma's drew 475. That is jsort's
  pair schedule, not failures.
- **Nothing was tuned.** Descriptions, `-k 10` and the seed for pairs are those of the Jev runs,
  and each benchmark is one run of one description on each model.
- **Raw outputs** are in `bench/out/local-2026-09-22/`, which is not committed, as
  `<model>-<bench>.{json,log,stdout,stderr}` with a console log each; `RESULTS-NOTES.md` there
  records the session. To reproduce: `uv run python bench/local_models.py prepare`, then
  `run --model diffusiongemma --bench yc`, `fed-latest` and `fed-whole`, the same with
  `--model laya`, then `uv run python bench/fed.py check` for each FedLock file date, and `freeze`.
