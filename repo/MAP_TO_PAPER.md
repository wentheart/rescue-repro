# How this repository maps to the paper

Which script produces each number and figure, and what input it needs. Every required input is in the
appendix archive under `raw/`, or is a JSON already published in `results/`.

## Table 1 — unanswered failures and recoverability

| Content | Script | Input |
|---|---|---|
| Unanswered rate, share of errors | `analysis/make_per_item_scores.py` | `raw/tabdsr_exp/out` |
| Strong-model recoverable counts (strong ans./cor.) | `rescue/analyze_two_tier.py` | `raw/_g8.../rescue_missing.jsonl` |
| F / N / A split | `analysis/verify_fna.py` | same |

## Table 2 — comparison at matched budget

| Row | Script | Input |
|---|---|---|
| TabDSR first pass | `analysis/make_per_item_scores.py` | `raw/tabdsr_exp/out` |
| + SC@3 (majority over three samples) | `scoring/recompute.py` | same (`*_sc3_*` shards) |
| + naive cascade | `baselines/matched_cost.py` | `results/per_item_scores.json` |
| + consistency cascade | `baselines/consistency_cascade.py` | `raw/tabdsr_exp/out` (`*_sc2_*`, `*_sc3_*`) |
| + two-tier rescue (ours) | `analysis/make_per_item_scores.py` | same |
| Difference intervals | `analysis/bootstrap_ci.py` | same |
| Tighter operating point | `analysis/bootstrap_tighter.py` | same |
| Mean-over-datasets interval | `analysis/bootstrap_macro.py` | same |

The frozen output of the consistency cascade ships with the repository:
`results/consistency_cascade.json`.

## Table 3 — cross-vendor replication

| Content | Script | Input |
|---|---|---|
| The twelve backbone–dataset results | `analysis/gen_table3_blocks.py` | `raw/_g9.../out_full`, `raw/tabdsr_exp/out` |
| Whole table in one pass | `analysis/gen_tables.py` | same |

## Table 4 — component ablation

| Content | Script | Input |
|---|---|---|
| Δ for dropping the fallback / dropping tier 1 | `analysis/ablation_deltas.py` | `results/table2_unrounded.json` |
| The table itself | `analysis/gen_tables.py` | `raw/...` |

## Figure 2 — frontier and dominance window

| Content | Script | Input |
|---|---|---|
| Three-dataset frontier, dominance-window endpoints, peak gap | `analysis/frontier_all.py` | `results/consistency_cascade.json`, `raw/...` |

## Proposition 2 — efficiency ratio and acceptance threshold

| Content | Script | Input |
|---|---|---|
| Measured efficiency ratios, α*, observed acceptance | `analysis/rescue_efficiency.py` | `results/efficiency.json` and the raw runs |

## Expected input layout

```
$PROJECT_ROOT/
  _g4_scoring_20261102/     scoring.py, recompute.py
  _g8_plan_rescue_20261109/ rescue_missing.jsonl
  _g9_crossmodel_20261112/  out_full/, rescue_union.jsonl
  tabdsr_exp/out/           first-pass output shards
```

When `PROJECT_ROOT` is unset, the scripts use the current working directory as the root.

## What the repository cannot do on its own

The first-pass generation itself — the TabDSR pipeline that calls the models — is third-party code and
is not redistributed. Everything downstream of it is here: the raw outputs ship in the appendix, so all
tables, intervals and figures can be recomputed, but the runs cannot be regenerated without the
pipeline and an API endpoint.
