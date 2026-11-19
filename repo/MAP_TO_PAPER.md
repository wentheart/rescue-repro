# How this repository maps to the paper

Which script produces each number and figure, and what input it needs.

Two kinds of input appear below:

- `results/*.json` — published in this repository.
- raw inputs — the per-item model outputs. They are **not part of this release**; they are available
  from the authors on request. Scripts that need them are included so the analysis can be audited,
  and they run once `PROJECT_ROOT` points at the raw tree.

## Table 1 — unanswered failures and recoverability

| Content | Script | Input |
|---|---|---|
| Unanswered rate, share of errors | `analysis/make_per_item_scores.py` | raw first-pass outputs |
| Strong-model recoverable counts (strong ans./cor.) | `rescue/analyze_two_tier.py` | raw rescue records |
| F / N / A split | `analysis/verify_fna.py` | raw rescue records |

## Table 2 — comparison at matched budget

| Row | Script | Input |
|---|---|---|
| TabDSR first pass | `analysis/make_per_item_scores.py` | raw first-pass outputs |
| + SC@3 (majority over three samples) | `scoring/recompute.py` | same (`*_sc3_*` shards) |
| + naive cascade | `baselines/matched_cost.py` | `results/per_item_scores.json` |
| + consistency cascade | `baselines/consistency_cascade.py` | raw first-pass outputs |
| + two-tier rescue (ours) | `analysis/make_per_item_scores.py` | same |
| Difference intervals | `analysis/bootstrap_ci.py` | raw first-pass outputs |
| Tighter operating point | `analysis/bootstrap_tighter.py` | same |
| Mean-over-datasets interval | `analysis/bootstrap_macro.py` | same |

The frozen output of the consistency cascade ships with the repository:
`results/consistency_cascade.json`.

## Table 3 — cross-vendor replication

| Content | Script | Input |
|---|---|---|
| The twelve backbone–dataset results | `analysis/gen_table3_blocks.py` | raw cross-model and first-pass outputs |
| Whole table in one pass | `analysis/gen_tables.py` | same |

## Setup and repeat runs (§4.1)

| Content | Script | Input |
|---|---|---|
| Run-to-run variance of tier 1 (repeats on the fixed escalation set) | `analysis/variance_check.py` | raw rescue records |
| Second end-to-end run on the second first-pass escalation set | `analysis/rescue_rep2.py`, `analysis/end_to_end_rep2.py` | raw first-pass and rescue records |

## Table 4 — component ablation

| Content | Script | Input |
|---|---|---|
| Δ for dropping the fallback / dropping tier 1 | `analysis/ablation_deltas.py` | `results/table2_unrounded.json` |
| The table itself | `analysis/gen_tables.py` | raw outputs |

## §4.4 — sanitiser compliance

| Content | Script | Input |
|---|---|---|
| Share of items the pipeline's own sanitiser handles cleanly, per backbone | `analysis/recompute_compliance.py` | raw first-pass outputs |
| The same breakdown for the unanswered items | `analysis/recompute_compliance.py` | same |

## §4.6 — the two case studies

| Content | Script | Input |
|---|---|---|
| Select the accepted case and dump its records | `analysis/pick_case.py` | raw first-pass and rescue records |
| Re-verify both cases against the frozen records | `analysis/verify_cases.py` | same |

## Figure 2 — frontier and dominance window

| Content | Script | Input |
|---|---|---|
| Three-dataset frontier, dominance-window endpoints, peak gap | `analysis/frontier_all.py` | `results/consistency_cascade.json`, raw outputs |

## Proposition 2 — efficiency ratio and acceptance threshold

| Content | Script | Input |
|---|---|---|
| Measured efficiency ratios, α*, observed acceptance | `analysis/rescue_efficiency.py` | `results/efficiency.json`, raw outputs |

## Expected raw-input layout

```
$PROJECT_ROOT/
  _g4_scoring_20261102/     scoring.py, recompute.py
  _g8_plan_rescue_20261109/ rescue_missing.jsonl
  _g9_crossmodel_20261112/  out_full/, rescue_union.jsonl
  tabdsr_exp/out/           first-pass output shards
```

When `PROJECT_ROOT` is unset, the scripts use the current working directory as the root.

## What the repository cannot do on its own

The raw outputs and the first-pass generation are not included. The TabDSR pipeline that calls the
models is third-party code and is not redistributed. Everything needed to read, aggregate and audit the
published numbers is here; regenerating the runs is not.
