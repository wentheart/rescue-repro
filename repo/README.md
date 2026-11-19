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
| `bootstrap_ci_seed777.json` | the same bootstrap under a different seed, as a check that the intervals are not seed-dependent |
| `consistency_cascade.json` | the consistency-cascade baseline: its operating points and matched-budget score |
| `cost_efficiency.json` | rescue quality on the escalated set (0.625) and the consistency-cascade cost/accuracy curve |
| `paired_tests.json` | paired tests behind the significance statements: per-dataset delta, interval, discordant counts and p-value |
| `table2_unrounded.json` | table 2 values before rounding |
| `efficiency.json` | measured rescue efficiency ratios and acceptance rates |
| `union_sets.json` | escalation-set sizes per backbone and their union, used to schedule the cross-vendor runs |
| `s2_results.json` | aggregate S^2 per configuration |

`per_item_scores.json` alone is enough to recompute the four headline means:

| Dataset | first pass | tier-1 only | **two-tier** | full rerun |
|---|---|---|---|---|
| TAT-QA (736) | 35.69 | 55.93 | **59.13** | 62.50 |
| CALTAB151 (151) | 25.17 | 34.77 | **37.09** | 36.75 |
| TableBench (493) | 47.62 | 50.96 | **52.18** | 52.79 |

Each record carries the item score, whether the item entered the rescue, and the `table_uid` that the
cluster bootstrap resamples, so group statistics can be recomputed directly from this file.

### What can and cannot be recomputed here

| Task | Possible from this repository |
|---|---|
| Read any item's score under all four configurations | yes |
| Recompute the four means, escalation rates and per-dataset aggregates | yes |
| Re-derive the reported intervals | the frozen intervals are in `results/bootstrap_*.json`; recomputing them needs the raw outputs |
| Regenerate the raw outputs from scratch | no — see "What is not included" |

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

- **The raw per-item model outputs** (prompts, responses, token counts). They are not part of this
  release; they are available from the authors on request. Everything derived from them — the item
  scores, the aggregates and the frozen intervals — is in `results/`.
- **The TabDSR / TableMaster runtime.** It is third-party code and is not redistributed; obtain it
  from its official release. `rescue/run_rescue.py` is the escalation driver that wraps it, so it
  imports that runtime and will not run without it.
- **A model-serving adapter and credentials.** The driver calls a hosted endpoint; the adapter used
  here is not shipped and **no key or credential is included anywhere in this release**. Point the
  driver at your own endpoint and supply the credential through your environment.
- **Benchmark files and model weights** (see below).

The scripts in `analysis/` and `baselines/` are the ones that produced the published numbers from the
raw outputs; they are included so the analysis can be audited line by line, not so that the pipeline
can be re-run without the items above.

## Data

The evaluation subsets follow the TabDSR protocol and are **not redistributed here**. Obtain them from
the original releases of TAT-QA, TableBench and TabDSR. The raw outputs that the analysis scripts
consume are available from the authors on request.

## License

MIT. See `LICENSE`.
