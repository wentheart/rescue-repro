# Reproducing the reported numbers

Two routes, in increasing cost. The first needs no API access.

## Route A — from the shipped results (seconds)

```bash
python analysis/make_per_item_scores.py      # writes results/per_item_scores.json
```

The script prints one line per dataset; the `two_tier` column must reproduce
59.13 / 37.09 / 52.18 for TAT-QA / CALTAB151 / TableBench.

The bootstrap scripts consume the same inputs and write the intervals:

```bash
python analysis/bootstrap_ci.py        # matched-budget intervals, gap vs the cascade
python analysis/bootstrap_tighter.py   # the tighter operating point
python analysis/bootstrap_macro.py     # three-dataset mean interval
```

All three resample **tables**, not items (`table_uid`), 2000 replicates, fixed seed. The seed is
recorded in each script; the intervals are quoted in the paper.

## Route B — from the raw model outputs (needs the appendix archive)

The scripts expect the working tree they were written against:

```
$PROJECT_ROOT/
  _g4_scoring_20261102/         scoring.py, recompute.py
  _g8_plan_rescue_20261109/     rescue_missing.jsonl, rescue_union.jsonl
  _g9_crossmodel_20261112/      out_full/     (cross-vendor full runs)
  tabdsr_exp/out/               first-pass outputs, one JSON per shard
```

Set the root and run:

```bash
export PROJECT_ROOT=/path/to/extracted/appendix     # Windows: set PROJECT_ROOT=...
python analysis/make_per_item_scores.py
```

Without `PROJECT_ROOT` the scripts fall back to the current working directory.

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
- `table_uid` — cluster identifier for the bootstrap

S² is the mean item score times 100, where an item score is the fraction of its reference values the
extracted answer matches: exact integers, a relative tolerance of 1e-4 otherwise, sign-sensitive,
and 0 when nothing was extracted.

## Unit tests

```bash
python scoring/test_scoring.py
```
