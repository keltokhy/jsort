# Jev, DiffusionGemma, and Laya on the readability sample

Run September 22, 2026, on the 300-excerpt sample that `bench/readability.py prepare` draws from the
CommonLit Ease of Readability corpus with seed 20260919. Each model sorted the same excerpts on
"easier to read" with `jsort -k 10`, through the installed command, with an empty answer cache, and
then answered the one-question alternative, p("easy to read"), once per excerpt. `bench/three_models.py`
runs it; [three-models-2026-09-22.json](three-models-2026-09-22.json) is the frozen output.

The teachers' scale has a reliability of about 0.78 on this sample, so no measure can correlate with
it above about 0.88.

| | Jev (`typesafe/jev-1.13`, OpenRouter) | DiffusionGemma (OpenJev, `openjev-0.1`, MLX) | Laya (`laya-421m`, MLX) |
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

## What the table says

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
  flight. DiffusionGemma ran two requests at a time against a server that executes serially on the
  GPU, 0.55 s per comparison, so fourteen minutes for 300 excerpts; a full corpus of 4,724 excerpts
  would take about four hours. Laya is fast and free but the answers are not usable here.

## Provenance and caveats

- The Jev column is the September 19 run from `bench/out/readability.json`, on the identical sample,
  seed, and setting; the OpenRouter key hit its spending limit on September 22 before a same-day
  rerun could start, and the paired rerun will replace this column when it runs. Model
  `typesafe/jev-1.13` was pinned both times.
- Local servers: OpenJev at commit `e04794a` with `mlx-community/diffusiongemma-26B-A4B-it-4bit`
  revision `a7a8140`; laya-mlx at commit `fc1df62` with `aac6fef/laya-mlx` revision `0476785`, through
  `scripts/laya_server.py` from jevkit-runtime with truncation rejection on. Both on an M3 Ultra.
- Laya's refused pairs are counted from the server's audit log, 351 of 1,500 requests; jsort recorded
  353 failed comparisons, the other two being deadline errors. Refused pairs are skipped, not
  truncated, so Laya is scored on what it read.
- One sample, one description, one `-k`. Nothing was tuned on these results.
