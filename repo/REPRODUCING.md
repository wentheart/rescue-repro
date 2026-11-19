# Reproducing the reported numbers

What is available here, and what it lets you check.

## What the repository supports

From `results/per_item_scores.json`:

- any single item's score under the four configurations;
- the four means, the escalation rate and per-dataset aggregates;
- the split of escalated items into full credit, partial credit and unanswered.

The frozen intervals are in `results/bootstrap_*.json`, as produced by the scripts in `analysis/`.

## What needs the raw outputs

Regenerating `per_item_scores.json`, the intervals, the frontier and the table dumps requires the raw
per-item model outputs. Those are **not part of this release**; they are available from the authors on
request. If you obtain them, the scripts expect this layout:

```
$PROJECT_ROOT/
  _g4_scoring_20261102/     scoring.py, recompute.py
  _g8_plan_rescue_20261109/ rescue_missing.jsonl
  _g9_crossmodel_20261112/  out_full/, rescue_union.jsonl
  tabdsr_exp/out/           first-pass output shards
```

```bash
export PROJECT_ROOT=/path/to/raw             # Windows: set PROJECT_ROOT=...
python analysis/make_per_item_scores.py      # per-item scores -> results/
python analysis/bootstrap_ci.py              # matched-budget intervals
python analysis/bootstrap_tighter.py         # tighter operating point
python analysis/bootstrap_macro.py           # three-dataset mean interval
```

When `PROJECT_ROOT` is unset the scripts use the current working directory.

## What the per-item record contains

```json
{"id": 13, "table_uid": "b3d6…", "escalated": true,
 "first_pass": 0.0, "tier1_only": 0.0, "two_tier": 0.0, "full_rerun": 0.0}
```

- `escalated` — the first pass produced no usable answer, so this item entered the rescue
- `first_pass` — S² of the small-model pipeline alone
- `tier1_only` — first pass, and tier-1 on escalated items, no fallback
- `two_tier` — the method: tier-1, falling back to a full rerun when tier-1 is rejected
- `full_rerun` — every escalated item rerun with the strong model (the expensive endpoint)
- `table_uid` — cluster identifier resampled by the bootstrap

S² is the mean item score times 100, where an item score is the fraction of its reference values the
extracted answer matches: exact integers, a relative tolerance of 1e-4 otherwise, sign-sensitive, and
0 when nothing was extracted.

## Unit tests

```bash
python scoring/test_scoring.py
```
