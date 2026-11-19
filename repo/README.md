# Two-tier rescue for unanswered items in programmatic table reasoning

Anonymous code and result release for double-blind review.

The paper studies items that a program-executing table pipeline leaves **unanswered** after its
complete first pass, and measures how much of that class a strong model recovers when the pipeline
escalates only those items.

## What the method does

1. **Trigger.** An item is escalated iff the first pass produced no usable answer. The decision reads
   only the pipeline's own output — no labels, no confidence threshold, no extra small-model samples.
2. **Tier 1.** One strong-model call re-derives the answer: a fresh program is generated from the
   *original* question and a positionally indexed copy of the *original* table, bypassing the
   sanitised intermediate representation.
3. **Acceptance.** The tier-1 answer is used iff execution succeeded *and* an answer was extracted.
   Both are properties of the run, never of the reference answer.
4. **Tier 2.** Only a rejected tier-1 call falls back to a full strong-model pipeline rerun.

## Layout

```
scoring/     S^2 answer scoring (strict, extractor never sees the reference) + unit tests
rescue/      tier-1 prompt construction, execution, acceptance rule, escalation driver
analysis/    cluster bootstrap, per-item score dump, table/figure generators
results/     final results: per-item scores and the bootstrap outputs behind every table
```

### `results/` — what to read first

| File | Contents |
|---|---|
| `per_item_scores.json` | the main deliverable: per-item S^2 for `first_pass`, `tier1_only`, `two_tier`, `full_rerun`, plus `escalated` and the `table_uid` used as the bootstrap cluster |
| `bootstrap_ci.json` | cluster-bootstrap intervals at matched budget: scores, cost, gap vs the consistency cascade |
| `bootstrap_tighter.json` | the same at the tighter (tier-1-only) operating point |
| `bootstrap_macro.json` | interval for the three-dataset mean gap |
| `table2_unrounded.json` | table 2 values before rounding |
| `efficiency.json` | measured rescue efficiency ratios and acceptance rates |
| `s2_results.json` | aggregate S^2 per configuration |

`per_item_scores.json` alone is enough to recompute every headline number:

| Dataset | first pass | tier-1 only | **two-tier** | full rerun |
|---|---|---|---|---|
| TAT-QA (736) | 35.69 | 55.93 | **59.13** | 62.50 |
| CALTAB151 (151) | 25.17 | 34.77 | **37.09** | 36.75 |
| TableBench (493) | 47.62 | 50.96 | **52.18** | 52.79 |

```bash
python analysis/make_per_item_scores.py   # regenerates results/per_item_scores.json
```

### Reproducing the numbers without rerunning any model

`analysis/make_per_item_scores.py` reads the first-pass outputs and the rescue records and writes
`per_item_scores.json`. The means printed at the end should match the table above. The bootstrap
scripts then consume the same files to produce the intervals. See `REPRODUCING.md` for the exact
command order and the directory layout they expect.

## Scope and honest limitations

- The study fixes the strong model (Qwen2.5-72B-Instruct) and varies only the small model, over four
  backbones from different vendors. It does not claim transfer to a different strong model.
- The advantage over the consistency cascade is established **at the two operating budgets reported**
  (roughly 5%–13% of a full strong-model run), and holds across a measured budget window. It is not a
  claim about arbitrary budgets.
- On TableBench the gain over the consistency cascade is not statistically significant; the paper says
  so and the interval is reported.
- The method narrows the gap to running the strong model on every item; it does not close it.

## What is not included

- **The TabDSR / TableMaster runtime.** It is third-party code and is not redistributed; obtain it
  from its official release. `rescue/run_rescue.py` is the escalation driver that wraps it, so it
  imports that runtime and will not run without it.
- **A model-serving adapter and credentials.** The driver calls a hosted endpoint; the adapter used
  here is not shipped and **no key or credential is included anywhere in this release**. Point the
  driver at your own endpoint and supply the credential through your environment.
- **Benchmark files and model weights** (see below).

The analysis path — `analysis/make_per_item_scores.py` and the three bootstrap scripts — runs on the
files in `results/` and needs none of the above.

## Data

The evaluation subsets follow the TabDSR protocol and are **not redistributed here**. Obtain them from
the original releases of TAT-QA, TableBench and TabDSR, then place the pipeline outputs under the
layout described in `REPRODUCING.md`.

## License

MIT. See `LICENSE`.
