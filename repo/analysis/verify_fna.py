"""Verify the F/N/A split: among usable-but-not-full-credit items, how many score zero?"""
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


for ds, (sp, bp, key) in SETS.items():
    S, B = load(sp), load(bp)
    ids = [i for i in S if i in B and not S[i].get("pipeline_error") and not B[i].get("pipeline_error")]
    esc = [i for i in ids if is_missing(S[i].get("ExtractResult"))]
    full = zero = positive = absent = 0
    for i in esc:
        rec = rescue.get(f"{key}|{i}")
        if accepted(rec):
            s = rec["score"]
        else:
            if is_missing(B[i].get("ExtractResult")):
                absent += 1
                continue
            s = score(extract(B[i].get("ExtractResult")), golds(S[i]))
        if s >= 1 - 1e-9:
            full += 1
        elif s > 0:
            positive += 1
        else:
            zero += 1
    n = len(esc)
    print(f"{ds:<11} n={n:<4} F={full:<4} N={positive + zero:<4} (positive {positive}, zero {zero})  A={absent}")
