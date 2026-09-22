# jsort

sort, but the key is a description.

```console
$ jsort -o "more hawkish about inflation" presser.txt | head -6
2.18	0.12	The plain fact is that inflation is too high and has been for too long.
2.18	0.16	I defined the standard for action: We must be confident that underlying inflation is moving to our objective, clearly and at sufficient speed.
1.76	0.19	Yet for more than five years, inflation has been running above target.
1.68	0.36	Too many categories are still posting increases above 3 percent, on both a 6- and 12-month basis.
1.52	0.38	In the meeting just concluded, the FOMC decided to raise the target range for the federal funds rate by ¼ percentage point to 3¾ to 4 percent, in support of the Federal Reserve’s dual mandate.
1.39	0.22	The committee’s unanimous vote shows our resolve to achieve price stability on a timelier basis.
jsort: 51 texts, 255 comparisons in 10 rounds; reliability 0.96; first-position lean -0.07; 255 calls, 0 cached; 87,187 tokens; $0.0037; 7.2s

$ jsort -r "more hawkish about inflation" presser.txt | head -3      # the other end, from the cache
And with that, I’ll take a few of your questions.
New hiring, private-sector earnings, business capital investment—each of these markers has improved in recent months and is pointing in a good direction.
Those who are least well-off have the most to gain from a durable expansion, a solid labor market, and stable prices.
```

`presser.txt` is the chair's opening statement at the FOMC press conference of September 16, 2026, one
sentence per line. The first column is the score and the second its standard error.

jsort shows [Jev](https://docs.typesafe.ai), TypeSafe's decision model, two texts at a time and asks
which ranks higher on the dimension you described. Jev answers with a probability in about 200 ms.
jsort asks about a few pairs per text, not all of them, fits a Bradley-Terry scale to the answers
and prints the texts from the top of the scale down. With `-o` each line carries its score and a
standard error, so the order can be used as a measurement and not only as a ranking.

The same command sorts whole documents. Every opening statement since the first FOMC press
conference, 95 of them from April 2011 to this week, as files named by date:

```console
$ jsort --whole -o --max-chars 16000 "more hawkish about inflation" statements/*.txt | sed -n '1,4p;92,95p'
6.91	0.35	statements/2022-11-02.txt
6.27	0.49	statements/2022-09-21.txt
6.06	0.53	statements/2022-07-27.txt
5.64	0.41	statements/2022-06-15.txt
-3.01	0.23	statements/2020-11-05.txt
-3.05	0.22	statements/2020-04-29.txt
-3.10	0.22	statements/2020-07-29.txt
-3.46	0.17	statements/2020-06-10.txt
jsort: 95 texts, 475 comparisons in 10 rounds; reliability 0.98; first-position lean -0.04; 0 calls, 475 cached; 0.1s
```

That run came from the cache. The first time it was under half a minute and about seven cents, for
statements of about 1,250 words each. The top four are 2022 hikes of 75 points. The bottom is the
first year of the pandemic. This week's statement, a 25-point hike, ranks 15th of the 95. By chair,
the average score is -2.19 for Bernanke (12 statements), -1.29 for Yellen (16), 0.62 for Powell (64)
and 2.32 for Warsh (3). `bench/fed.py` builds both files from federalreserve.gov and checks the
scale against what the Committee did; the results are [below](#how-well-does-it-work).

A thousand short lines take about 5,000 comparisons: roughly a minute and seven cents.

## Install

```bash
uv tool install jev-sort        # the command it installs is jsort
```

jsort finds a key the way [jgrep](https://github.com/keltokhy/jgrep) does: `TYPESAFE_API_KEY`,
`OPENROUTER_API_KEY`, or a System One gateway (`JEV_GATEWAY_URL` and `JEV_GATEWAY_API_KEY`), from the
environment or from `~/.config/jev/typesafe.key`, `openrouter.key` or `gateway.key`. Force a choice
with `--api` or `JEV_API`. The two tools share one cache file. jsort keys answers by endpoint as well
as model and question, so switching gateways cannot reuse another gateway's answers. Older entries
without endpoint information remain on disk but are not reused. jsort also records which model gave
each answer, in the `answer_metadata` table jlink introduced beside `answers`. The `answers` table
and its keys are unchanged, so the other tools and older versions read and write the file as before.

## Use

```bash
jsort "more urgent" tickets.txt | head                       # the ten most urgent
jsort -r "more urgent" tickets.txt | head                    # the ten least
jsort -o "more hawkish about inflation" statements.txt       # with scores and standard errors
jsort --top 5 "a more serious safety problem" complaints.txt
jsort --para "makes a stronger causal claim" paper.txt       # paragraphs, not lines
jsort --whole "of more general interest" abstracts/*.txt     # whole files; prints their names
jsort --jsonl --field event.message "angrier" events.jsonl   # full records come back, sorted
jsort --csv --field narrative -o --keep-order --name breadth \
      "drew broader participation" events.csv > scored.csv    # add a variable, keep the row order
jsort --json "more urgent" tickets.txt                       # rank, score, se and source line
jsort --save-scale hawkish.json "more hawkish about inflation" statements.txt   # sort, and keep the scale
jsort --scale hawkish.json -o september.txt                  # place new texts on the saved scale
```

| Option | Meaning |
|---|---|
| `-k N` | Comparisons each text takes part in. Default 10, minimum 2. The run asks about N/2 questions per text. With `--scale`, the comparisons each new text gets, all of them with anchors. |
| `--top N` | Print only the top N. Texts that are clearly out of the running stop being asked about, and their questions go to the contenders. With `-r` it is the bottom N that is hunted and printed. |
| `-r` | Lowest first. |
| `-o` | Put the score and its standard error in the first two tab-separated columns. With `--csv` or `--jsonl`, add `jsort_score`, `jsort_se` and `jsort_n` to each record. |
| `--name NAME` | Call those fields `NAME_score`, `NAME_se` and `NAME_n`, so that one file can carry several scales. |
| `--keep-order` | Print in input order. With `-o` this adds scores to a file without rearranging it. With `--scale`, each text prints as soon as it and the texts before it are placed. |
| `-n`, `-H` | Prefix each line with its line number or file name. |
| `--json` | One JSON object per text: `rank`, `score`, `se`, `comparisons`, `file`, `line`, `text`. |
| `--jsonl --field NAME`, `--csv --field NAME` | Compare one field and return the complete records. JSON fields can be dotted paths. |
| `--para`, `--whole` | Sort paragraphs or whole files in place of lines. |
| `--seed N` | Seed for the choice of pairs. Default 0. The same seed asks the same questions, so a rerun comes from the cache. |
| `--save-scale FILE` | Also write the fitted scale to FILE, for `--scale` to place later texts on. Not with `--top`. See [Anchored scales](#anchored-scales). |
| `--anchors N` | With `--save-scale`, how many texts to keep as anchors. Default 30, minimum 2. |
| `--scale FILE` | Place the input on the scale saved in FILE. The description comes from the file. Each text is compared with the scale's anchors only. |
| `--unordered` | With `--scale`, print each text as soon as it is placed, not in input order. |
| `--any-model` | With `--scale`, place even though the API, endpoint or model is not the one the scale was built with, or the scale cannot name one. With `--save-scale`, save even though the answers behind the fit do not all name one model. |
| `--budget DOLLARS` | Stop asking once this much is spent and sort on what is known. Default 1.00, or `$JSORT_BUDGET`; 0 for no limit. |
| `--max-chars N` | Show Jev only the first N characters of a text. Default 8000; with `--scale`, the scale's. |
| `-j N`, `--timeout`, `--no-cache`, `--api`, `--model`, `--stats` | As in jgrep. |

Blank lines are dropped. Identical texts are compared once and share a score. Several files are
sorted together, as `sort` does. Exit status is 0 when the input was sorted and 2 on any error:
an unreadable file, a failed comparison, a stop at the budget, the API refusing further calls because
the credit ran out. In each of those cases whatever could be sorted from the answers already paid
for is still printed.

CSV needs a header with unique column names. A byte-order mark is ignored, a row with more values
than the header keeps them at its end, and a header with no rows comes back as a header. With `-o`,
jsort refuses to write over a column or JSON field that already exists; pick another prefix with
`--name`.

Write the description as a comparative: "more urgent", "easier to read", "drew broader military
participation". It goes into one question, `Text A ranks higher than text B on this criterion:
"..."`, so anything that completes that sentence sensibly will work.

## What the numbers mean

**Score.** The position on the scale, in logit units, centred on zero. A gap of 1.0 between two
texts means Jev gives the higher one about 73% in a head-to-head; a gap of 3 means about 95%.
Scores are relative to the other texts in the same run. They do not carry over to another file or
another description, unless the run's scale is saved and the other file is placed on it
([Anchored scales](#anchored-scales)).

**Standard error.** How well the comparisons pin the score down. Jev's answer is a probability, not
a win or a loss, so the model is a fractional logit and the errors are the robust (sandwich) kind,
with a leverage correction because there is one parameter per text. Where Jev's answers line up on
one scale the errors are small. Where they contradict each other the errors grow. In simulation,
intervals of 1.96 standard errors cover the target 94 to 95% of the time from `-k 6` up
(`bench/simulate.py`). Above 4,000 distinct texts the same errors are estimated by random probing,
to within about 7% each, because the matrix the exact ones need no longer fits.

**Reliability.** The comparisons are dealt into two halves, a scale is fitted to each, and the
correlation between the two is stepped up to full length (Spearman-Brown). Near 1, the order does
not depend on which pairs happened to be asked. Below 0.8 jsort says so: raise `-k`, or the
description is not one these texts can be ranked on. In simulation it tracks the true figure closely
(0.955 reported for 0.964 at the default) and errs on the cautious side.

**First-position lean.** How far the first-shown text's chance sits from a half in an even matchup.
Positions are randomised and balanced and the lean is estimated with the scale, as home advantage
is in a sports model, so it does not tilt the scores. It has been a few points either way.

## Anchored scales

Every sort fits its own scale, so two runs cannot be compared and a new text cannot be scored without
sorting everything again. That rules out a panel, where next month's texts have to land on the scale
the earlier months are on, and it rules out measuring anything as it happens. A saved scale allows both.

```bash
jsort --whole --max-chars 16000 --save-scale hawkish.json \
      "more hawkish about inflation" statements/*.txt          # an ordinary sort that also saves its scale
jsort --whole --scale hawkish.json -o statements/2026-10-28.txt   # a later statement, on the same scale
jsort --csv --field narrative --scale breadth.json -o --keep-order --name breadth new_events.csv > scored.csv
tail -f captions.txt | jsort --scale hawkish.json -o --keep-order  # each line scored as it is spoken
```

These are commands only. Output from live runs of them is still to be added.

**Saving.** `--save-scale FILE` changes nothing about the sort. It writes a JSON file holding the
description, the question exactly as Jev was asked it, the API, endpoint and model ID that were asked,
a count of the fit's answers by the model that gave them, `--max-chars` and how the texts were read, a
summary of the fit (texts, comparisons, reliability, the first-position lean, the ridge) and the
anchors: a text, its score and its standard error each. The anchors are 30 of the sorted texts by
default (`--anchors N`).
The highest and the lowest are always kept. Between them the range of scores is cut into equal
stretches and the text with the smallest reported standard error is taken from each. This spreads
anchors over the scale and favours small reported errors; it does not establish that those texts
are better measured. Thirty puts about six anchors within a logit of
any point on a scale ten logits long, which is what the two adaptive rounds of a default placement
look for. The file holds the anchors and not every sorted text: the sort's own output already has
every score, and a scale built on thousands of documents stays small enough to keep with a project.
`--anchors` can be raised to the number of texts. Anchor texts are stored verbatim as Jev saw them,
after truncation to `--max-chars`. Sharing the scale file shares those texts, including any private
information in them.

Placement takes an anchor's score as exact, so jsort is careful about what becomes one. `--save-scale`
cannot be combined with `--top`: that run stops asking about texts that are out of the running, and
the far end of its scale, which would be kept as an anchor, rests on two or three comparisons. A text
is eligible only if it took part in at least half of `-k` comparisons. A complete sort leaves every
text eligible. One that stopped at the budget may not: the file records how many texts qualified,
stderr says so, and if fewer than two did no scale is written.
Loading a scale with `--scale` also warns if its fit summary records an over-budget run or failed
comparisons. Such a scale can still be used; rebuild it for a complete fit.

A scale also has to be able to say which model it was built on, because that is what later
placements are checked against. jsort records in the cache which model gave each answer, as the API
named it, and a fit counts its answers by model. If the answers name two models (an alias such as
`jev-latest` moved between two runs and the cache kept the earlier answers), or if some name none
(cache entries written before jsort recorded it, or an API that does not name its model), the scale
is not saved and jsort says why. `--no-cache`, or a pinned `--model`, gives a fit from one model.
`--any-model` saves anyway; the file then records the counts as they are, never one name standing
for answers that did not give it, and `--scale` will need `--any-model` as well.

**Placing.** `jsort --scale FILE` takes the description from the file. A different description on the
command line is an error, not an override. Each new text is compared with anchors and with nothing
else. Its score is the one-parameter version of the fit the scale came from: the anchors stay at their
saved scores, the first-position lean stays at the value that fit estimated, Jev's probabilities are
used as they are, and the ridge that fit used keeps the score finite. The standard error is the same
robust kind, from the text's own comparisons. Anchors are chosen as a computerised adaptive test
chooses items. The first round draws anchors from across the whole scale. The second and third take
the anchors nearest the running estimate, where a comparison says the most. A round's questions go
out together, so a text is placed in about three round trips. The new text is shown first in about
half of its comparisons. Every text gets all `-k` of its comparisons. There is no option to stop once
a standard error looks small: a rule that stops on the reported error selects for small estimates of
it, which shortens the intervals, to save one round trip at most. A text that is itself an anchor
gets its saved score and costs nothing.

**Independence.** What a text is asked depends on the scale, the seed and the text, and on nothing
else in the input, and placing it never changes the scale file. So a text placed in full, on all the
comparisons `-k` asks for, gets the same score alone, in a file of a million, or on a pipe. What the
rest of the input can change is whether the budget reaches a text at all. jsort admits a text only
when the budget covers its whole placement, and admits texts in input order, so when the money runs
out some texts are placed in full and the rest get no score. A text is placed on fewer comparisons
than `-k` asked for only if something cuts it short once it has begun: a comparison that fails, a
refusal from the API, or a rise in price. Such a score is not the one a full placement gives, and it
is marked: `"partial": true` in `--json`, `NAME_partial` with `-o` on CSV and JSONL, and a line on
stderr. Within one placement run, copies of the same shown text share one result, even with
`--no-cache`; differences after `--max-chars` do not count. Across runs, uncached scores can still
differ if Jev gives different answers.

Since a score owes nothing to the rest of the input, `--scale` does not have to read everything first:

| Output | Behaviour |
|---|---|
| default, `-r`, `--top N` | Sorted, as jsort always prints. A sort has to wait for the end of the input. `--top` only trims the output here. |
| `--keep-order` | Streams. A text prints as soon as it and the texts before it are placed. |
| `--unordered` | Streams. A text prints as soon as it is placed. |

Streaming follows jgrep. With ordered output `-j N` bounds the texts in hand, those being placed and
those waiting for an earlier text to print, so a slow text cannot let the rest of the input run ahead.
Remembering duplicates retains one text hash and result per distinct shown text for the run, so that
memory grows with the number of distinct texts, including in a stream.
`--json` has no `rank` to give while streaming and prints `null`. In a sorted run the rank is among
the texts of that input.

**Beyond the anchors.** A text that scores above the highest anchor or below the lowest is outside
the range of its eligible anchors. Far outside that range, answers saturate and the ridge pulls
the score inward: the score is censored toward the scale and is best read as a bound, and its
standard error is not meaningful. A small reported error does not establish precision there.
Rebuild the scale with such texts included. For a score more than two logits past either end anchor,
jsort leaves the standard error empty. This margin is a reporting rule, not a guarantee that errors
closer to the anchors are calibrated. The score stays finite and flagged. `--json` carries `"beyond": "above"` or `"below"`,
`-o` on CSV and JSONL adds a `NAME_beyond` field, and stderr says how many texts fell outside and
where the anchors end. Scores and errors stay numeric or empty: blank in plain `-o` and CSV, `null`
in `--json` and JSONL. While streaming, stderr names each such
text as it prints, up to ten of them.

**The same model.** Scores from different models are not comparable. jsort asks the scale's API and
model unless `--api`, `--model` or the environment name others, and refuses before any call when the
API, the endpoint or the model ID differs from the scale's, or when the scale cannot name one model.
It then checks every answer it uses against the model the scale names, the cached answers too, since
the cache keys on the ID that was asked for and not on who replied. If the scale was built through
an alias such as `jev-latest` and that alias now reaches a newer model, the first uncached answer
shows it and the run stops. That request goes alone until its answering model is confirmed, even
with `--budget 0`; cached answers cannot confirm where the alias points today. A model refusal
exits with code 2 and prints no batch scores; a stream stops printing when it detects the refusal.
Pinning the model that answered, by the ID the message gives, is accepted. An
answer that names no model cannot be checked, and stderr says how many there were. `--any-model`
overrides all of these checks.

In simulation (`bench/simulate.py`, a judge with a known scale), 100 new texts placed at the default
`-k 10` on a scale saved from 200 land 0.28 logits (root mean square) from where a fit to every pair
of all 300 would put them. The 200 sorted texts themselves are 0.26 from it. Intervals of 1.96
standard errors cover that target 91% of the time, against 95% for the sorted texts, and 92% at
`-k 16`. No live benchmark of placement has been run yet.

Limits. The standard error treats anchors as exact and omits their errors, which are shared by texts
placed against them. An error estimate built on about ten residuals is itself noisy, so multiplying
it by 1.96 also understates interval uncertainty. Coverage is lower at small `-k`. For a new text
with fewer than three successful comparisons, jsort leaves the standard error empty, even when it
can give a score. An offline review found that using a t quantile with n-1 degrees of freedom
(n successful comparisons) together with an anchor-error term restored nominal coverage in its
simulation. That correction is not applied to the reported errors, and it is not evidence of
coverage on real texts.

The model check is only as good as the name the API gives: if a provider reports
an alias as the answering model, jsort cannot see the revision behind it, so pin the model with
`--model` when building a scale that has to last. jsort cannot tell whether new texts are the kind the scale was built on: sentences can be placed
on a scale of whole statements, and whether that means anything is for the user to judge. A placed
text costs up to `-k` comparisons of its own, twice its share of a sort, where every comparison is
shared by two texts.

## From Python

```python
import jsort

r = jsort.rank(statements, "more hawkish about inflation", per_item=10)
for i in r.order()[:5]:
    print(f"{r.score[i]:6.2f} ±{r.se[i]:.2f}  {statements[i]}")
r.reliability, r.lean, r.asked
```

`r.score`, `r.se` and `r.comparisons` are arrays aligned with the input. The command's seat belt
applies here too: spending stops at `budget=` dollars, by default `$JSORT_BUDGET` or 1.00, and
`r.over_budget` says whether it was reached. It works inside a notebook. `jsort.arank` is the same
thing as a coroutine, for a client you already hold. Each concurrent ranking has its own budget;
a shared request is charged to the ranking that starts it, and cached answers are free.

```python
scale = r.scale(anchors=30)                  # the run's scale, as --save-scale writes it
scale.save("hawkish.json")

p = jsort.place(new_statements, "hawkish.json", per_item=10)   # or a jsort.Scale; aplace is the coroutine
p.score, p.se, p.comparisons                 # aligned with new_statements, on the saved scale
p.beyond                                     # +1 above every anchor, -1 below, 0 within
p.partial                                    # True where a text was cut short of its comparisons
```

`r.scale()` raises `jsort.ScaleError` for a run that used `top`, for one that stopped before two texts
were measured well enough to anchor anything, and for a fit whose answers do not all name one model.
`jsort.place` raises it for a scale it cannot read, one that cannot name its model, or a model the
scale was not built with. `any_model=True` is the override for the model checks in both. `seed=` and
`budget=` are as on the command line.

## How it chooses pairs

All n(n-1)/2 pairs are never needed. A lopsided pair says almost nothing, because the answer was
predictable. A pair of near neighbours says the most. The first round is a random ring: every text
meets two others, once in each position, which connects everything. After each round the scale is
refitted, and the next round pairs texts that currently sit next to each other, as a Swiss-system
tournament does. Each score is jittered in proportion to how uncertain it still is, so a text that
is not pinned down yet keeps meeting new neighbours. A pair is asked at most twice, once in each
order. With a handful of texts jsort simply runs out of pairs and stops early.

`--top N` adds one rule. Once a text has three comparisons, jsort asks how much worse the fit would
get if that text were moved up to the edge of the top N. If the answer is "much worse", the text
gets no more questions. A text that lost 0.03 to 0.97 against a middling opponent is out, and no
further comparison will change that. In simulation this is the same cost as a full sort or a little
less, and finds more of the true top ten (0.85 against 0.77 of them, among 2,000 texts).

## Cost

A comparison bills about 270 tokens of overhead plus both texts and the description. For short lines
that is about 320 tokens, or $0.0000135 at $0.042 per million, and the default `-k 10` asks five
questions per text: about 7 cents per thousand lines. For 170-word excerpts a comparison is about
735 tokens and a thousand texts cost about 15 cents. A run is about ten rounds, each as slow as its
slowest call, so a small sort takes five to ten seconds. A large one manages about 90 comparisons a
second through OpenRouter.

jsort stops asking at `--budget`, one dollar by default, and sorts on what it has. Nothing is lost:
every answer is cached in `~/.cache/jev/answers.sqlite`, and the choice of pairs is seeded, so a rerun
with a higher budget, or a higher `-k`, starts by replaying the same questions from the cache and
only pays for the new ones.

With a budget, jsort starts with one request and sizes later concurrent batches using the largest
charge observed so far. Costs are reported after completion, so a final request or an unexpected
increase in request cost can still take spending above the threshold. `--budget 0` disables this limit.

The budget covers a `--scale` run too, more carefully, because texts arrive one by one and a short
first text says nothing about the price of a long one behind it. Each request is priced before it is
sent, from its own size (a token for every four bytes and 270 of overhead, as `jgrep --estimate`
counts), at the dearest rate per estimated token seen so far, and at one and a half times the list
price until a charge has been seen. Requests already in the air count at today's rate. The first
request of a run goes alone, and the number of texts being placed at once then doubles with each
text that finishes without the rate rising, from one up to `-j`, which is five doublings at the
default `-j 32`; a rise sends it back to one. A text is begun only if the budget covers all of its comparisons, so the
budget stops a run between texts, and since the reservation is a cautious estimate a little of the
budget is usually left unspent. It is still not a hard cap: cost is known only when a reply arrives,
so the calls in the air at the moment a price rises can take spending past the limit, by at most
`-j` requests at the new price. A stream that reaches the budget stops reading, finishes the texts
it has begun, and exits with status 2. For a long-lived `tail -f`, set your own default once with
`export JSORT_BUDGET=20`, or `0` for no limit.

## How well does it work

Two checks, run on 2026-09-19 with Jev 1.13 through OpenRouter. Both run the installed `jsort`
command, uncached.

**Against people making the same comparisons.** `bench/readability.py`. The CommonLit Ease of
Readability corpus (Crossley et al. 2022) has 4,724 excerpts written for
grades 3 to 12. Each has an easiness score that is itself a Bradley-Terry fit, to teachers'
judgments of which of two excerpts is easier. So this is the same method with Jev in the teachers'
chair. On a random 300 excerpts, the teachers' scale has a reliability of about 0.78, which means no
measure can be expected to correlate with it above about 0.88.

| Measure | Pearson r | Spearman ρ | Calls | Cost |
|---|---:|---:|---:|---:|
| `jsort -k 16 "easier to read"` | 0.824 | 0.840 | 2,400 | $0.074 |
| **`jsort "easier to read"`** (`-k 10`) | **0.824** | **0.841** | 1,500 | $0.046 |
| `jsort -k 6` | 0.813 | 0.833 | 900 | $0.028 |
| `jsort -k 4` | 0.803 | 0.826 | 600 | $0.018 |
| One question per text: the probability it is "easy to read" (`jgrep -o`) | 0.754 | 0.804 | 300 | $0.004 |
| Best readability formula shipped with the corpus (SMOG) | 0.661 | 0.647 | | |
| Flesch-Kincaid grade level | 0.598 | 0.595 | | |

Two things to take from it.

jsort gets most of the way to the ceiling from a two-word description. It beats every readability
formula shipped with the corpus by a wide margin, and it beats the other thing you can do with two
words, which is to ask for a probability per text and sort on that. That probability bunches up (73
distinct values among 300 texts), and it is weakest exactly where a sort is needed, on texts that
are close. On pairs the teachers put between 0.25 and 0.5 apart, jsort agrees with them 67.5% of the
time against 63.2%.

The default is already on the plateau. `-k 10` and `-k 16` give the same answer, `-k 6` is nearly
there, and the reliability of 0.98 says the same. Five cents' worth of comparisons orders 300 passages
at r = 0.82, against a ceiling of 0.88.

**Against what the Fed then did.** `bench/fed.py`. The 95 opening statements above, sorted whole on
"more hawkish about inflation", against the top of the target range for the federal funds rate (FRED
series DFEDTARU). Both tests were written into the script before the sort was run. The rank
correlation between a statement's score and the move announced that day is +0.46 (95 statements).
With the change in the target over the following 180 days it is +0.37 (91 statements). And the scale
picks up what the rate decision alone does not: the holds of June, September and November 2023
all rank among the thirteen most hawkish, because the chair held rates and talked tough. It gets
the shape of fifteen years right. The six statements that announced hikes of 50 or 75 points in
2022 are the top six, seven of 2023's eight statements fill ranks 7 to 13, and the bottom eight are
all from March 2020 to January 2021.

## Tips

- Write the description as a comparative and keep it to one dimension: "more urgent", not "more
  urgent and more polite". Stance ("more hawkish"), urgency and difficulty
  ("easier to read") are the dimensions shown above.
- jsort shines where levels are hard to name. You do not need a rubric or labeled examples, only a
  way to say what "more" means, and it keeps discriminating among texts that sit close together.
- When the scale is going into a model, use `-o` to keep the reported standard errors with the scores.
  A sort also reports reliability; placement errors have the limits described above and can be empty.
- For results that must reproduce, pin the model with `--model` (for example `typesafe/jev-1.13` on
  OpenRouter) and keep the cache file with the project. The cache makes a rerun exact and free.
- For long documents raise `--max-chars`, or cut them to the passages that bear on the dimension
  first (`jgrep --para` is one way). Counts and dates are better computed in code.

## Development

```bash
uv sync && uv run pytest              # offline: a fake API, no key
uv run python bench/simulate.py       # offline: a simulated judge with a known scale, sorting and placing
uv run python bench/probe.py          # live, under a cent: which question shape to use
uv run --group bench python bench/readability.py prepare && uv run --group bench python bench/readability.py run
```

`src/jsort/model.py` is the scale: the fit, the standard errors, reliability, the test `--top`
uses and the one-parameter fit that places a text. `schedule.py` chooses pairs. `engine.py` runs the
rounds and is the Python API. `scale.py` is the saved scale: the file, the choice of anchors and the
model check. `placement.py` places texts on one. `inputs.py` reads lines, paragraphs, files, CSV and
JSONL, as a generator so that placement can stream. `core.py` is based on jgrep's client: backends,
retries inside a time budget, the cache, in-flight deduplication and the cost meter, with per-call
cost tracking for the budget scheduler and, per answer, the model that gave it.

Why a noul and not a choice: `bench/probe.py` asks every ordered pair of ten texts both ways. A
two-option `choice` and a `noul` ("A ranks higher than B") made the same decisions, 95.6% correct,
but the choice put 93% of its answers below 0.1 or above 0.9, where the noul left 28% of its answers
between the two. Those in-between answers are what a scale is fitted from.

MIT license.

## Shared JevKit development

This tool uses [`jevkit-runtime`](https://github.com/keltokhy/jevkit-core), imported
as `jevkit_runtime`. Clone that repository beside this one as `../jevkit-core`, then
run `uv sync`. Core Python edits apply on the next invocation of this tool;
restart long-lived Python processes after editing.

The distribution name is `jevkit-runtime` because `jevkit-core` on PyPI belongs
to a different project. The runtime is [available on PyPI](https://pypi.org/project/jevkit-runtime/).
Use the sibling checkout for shared development, or `uv sync --no-sources` for a
standalone source checkout. Existing published versions of this tool are
unaffected by this source migration.

From the core checkout, `python scripts/dev.py setup`, `check`, and `wheel-check`
set up and validate all five consumers in separate environments. For isolated
worktrees such as `jgrep-jevkit`, add `--suffix=-jevkit` before the subcommand.
CI checks out core tag `v0.2.0`. Prompts, question construction, and budget policies
remain in this repository; answer identity, the answer store, transport, and metering
are the runtime's. Runtime 0.2 keys and stores answers differently from 0.1, so a cache
written by an earlier version is re-asked once after upgrading.
