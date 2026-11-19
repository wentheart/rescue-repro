"""Emit the per-item final scores behind the paper's main configurations.

Output: results/per_item_scores.json

For every evaluation item it records the score of four configurations under S^2 and the table_uid
used as the cluster-bootstrap unit:
    first_pass   the small-model pipeline on its own
    tier1_only   first pass, and tier-1 re-derivation on the escalated items, no rerun
    two_tier     first pass, tier-1 on escalated items, full rerun when tier-1 is rejected
    full_rerun   first pass, and a full strong-model rerun on every escalated item

The escalation set is fixed by the first pass (empty output), so all four share it.
Run with PROJECT_ROOT pointing at the extracted appendix tree.
"""
import json
import os
import sys

ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "_g4_scoring_20261102"))
from scoring import extract, score, is_missing  # noqa: E402
import recompute  # noqa: E402

SETS = {
    "TAT-QA": "TatQa_b8b-full_123_",
    "CALTAB151": "CalTab151_ct8b_123_",
    "TableBench": "TableBench_tb8b_123_",
}
RESCUE = os.path.join(ROOT, "_g8_plan_rescue_20261109", "rescue_missing.jsonl")
OUTFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                       "per_item_scores.json")


def accepted(rec):
    return rec.get("status") == "ok" and not is_missing(rec.get("answer"))


rescue = {}
with open(RESCUE, encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rescue[rec["key"]] = rec

out = {"note": "S^2 per item; 0 = unanswered or wrong; table_uid is the bootstrap cluster",
       "datasets": {}}

for ds, prefix in SETS.items():
    rows = recompute.load(prefix)
    items = []
    for i in sorted(rows, key=lambda x: int(x)):
        r = rows[i]
        s_first = score(extract(r.get("ExtractResult")), recompute.golds(r))
        escalated = is_missing(r.get("ExtractResult"))
        rec = rescue.get(f"{ds}|{i}")
        s_t1, s_full = 0.0, 0.0
        if escalated and rec:
            s_t1 = rec["rawpot"]["score"] if accepted(rec["rawpot"]) else 0.0
            s_full = rec["full_score"]
        items.append({
            "id": int(i),
            "table_uid": recompute.cluster(r),
            "escalated": bool(escalated),
            "first_pass": round(s_first, 6),
            "tier1_only": round(s_first if not escalated else s_t1, 6),
            "two_tier": round(s_first if not escalated else (s_t1 if rec and accepted(rec["rawpot"]) else s_full), 6),
            "full_rerun": round(s_first if not escalated else s_full, 6),
        })
    means = {k: 100 * sum(it[k] for it in items) / len(items)
             for k in ("first_pass", "tier1_only", "two_tier", "full_rerun")}
    out["datasets"][ds] = {"n": len(items), "means": {k: round(v, 2) for k, v in means.items()},
                           "items": items}
    print(f"{ds:<12} n={len(items):<4} " + "  ".join(f"{k}={v:6.2f}" for k, v in means.items()))

os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
with open(OUTFILE, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("wrote", os.path.abspath(OUTFILE))
