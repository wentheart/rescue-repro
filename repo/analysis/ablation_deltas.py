"""Unrounded accuracies for the three rescue actions, and the macro-average deltas relative to the
complete two-tier rescue."""
import json
import os
import sys

ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
G8 = os.path.join(ROOT, "_g8_plan_rescue_20261109")
G9 = os.path.join(ROOT, "_g9_crossmodel_20261112")
sys.path.insert(0, os.path.join(ROOT, "_g4_scoring_20261102"))
from scoring import extract, score, is_missing  # noqa: E402
from recompute import load  # noqa: E402

SETS = {"TAT-QA": ("TatQa_b8b-full_123_", "TatQa_b72-full_123_", "TAT-QA"),
        "CALTAB151": ("CalTab151_ct8b_123_", "CalTab151_b72-v2_123_", "CALTAB151"),
        "TableBench": ("TableBench_tb8b_123_", "TableBench_b72-v2_123_", "TableBench")}

rescue = {}
for p in (os.path.join(G8, "rescue_missing.jsonl"), os.path.join(G9, "rescue_union.jsonl")):
    if os.path.exists(p):
        for line in open(p, encoding="utf-8", errors="replace"):
            line = line.strip()
            if line:
                r = json.loads(line)
                rescue[r["key"]] = r["rawpot"]

accepted = lambda rec: rec and rec.get("status") == "ok" and not is_missing(rec.get("answer"))


def golds(r):
    a = r.get("Answer") or []
    return a if isinstance(a, list) else [a]


rows = {}
for ds, (sp, bp, key) in SETS.items():
    S, B = load(sp), load(bp)
    ids = [i for i in S if i in B and not S[i].get("pipeline_error") and not B[i].get("pipeline_error")]
    n = len(ids)
    esc = {i for i in ids if is_missing(S[i].get("ExtractResult"))}
    acc = {"tier1": 0.0, "two": 0.0, "full": 0.0}
    for i in ids:
        g = golds(S[i])
        s = score(extract(S[i].get("ExtractResult")), g)
        if i not in esc:
            acc["tier1"] += s
            acc["two"] += s
            acc["full"] += s
            continue
        rec = rescue.get(f"{key}|{i}")
        # tier-1 only
        acc["tier1"] += rec["score"] if accepted(rec) else 0.0
        # two-tier
        acc["two"] += rec["score"] if accepted(rec) else score(extract(B[i].get("ExtractResult")), g)
        # full rerun
        acc["full"] += score(extract(B[i].get("ExtractResult")), g)
    rows[ds] = {k: v / n * 100 for k, v in acc.items()}

for ds, v in rows.items():
    print(f"{ds:<11} tier1 {v['tier1']:.4f}  two {v['two']:.4f}  full {v['full']:.4f}")

mean = {k: sum(rows[ds][k] for ds in rows) / len(rows) for k in ("tier1", "two", "full")}
print(f"\nmean      tier1 {mean['tier1']:.4f}  two {mean['two']:.4f}  full {mean['full']:.4f}")
print(f"delta  -fallback : {mean['tier1'] - mean['two']:+.4f}")
print(f"delta  -tier1    : {mean['full'] - mean['two']:+.4f}")
