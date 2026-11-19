"""Pick one real TAT-QA item for the case study: base pipeline unanswered with a column-name
error, tier-1 accepted and fully correct. Print everything needed to write the listing."""
import json
import os
import sys

ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
G8 = os.path.join(ROOT, "_g8_plan_rescue_20261109")
sys.path.insert(0, os.path.join(ROOT, "_g4_scoring_20261102"))
from scoring import extract, score, is_missing  # noqa: E402
from recompute import load  # noqa: E402

S = load("TatQa_b8b-full_123_")
rescue = {}
for line in open(os.path.join(G8, "rescue_missing.jsonl"), encoding="utf-8", errors="replace"):
    line = line.strip()
    if line:
        r = json.loads(line)
        rescue[r["key"]] = r["rawpot"]

def golds(r):
    a = r.get("Answer") or []
    return a if isinstance(a, list) else [a]

cands = []
for i, r in S.items():
    if not is_missing(r.get("ExtractResult")):
        continue
    rec = rescue.get(f"TAT-QA|{i}")
    if not rec or rec.get("status") != "ok" or is_missing(rec.get("answer")):
        continue
    if score(extract(rec["answer"]), golds(r)) < 1 - 1e-9:
        continue
    ev = r.get("_exec_events") or []
    if not any("KeyError" in str(e.get("etype", "")) for e in ev):
        continue
    # prefer a short table and a short tier-1 program
    cands.append((len(str(r.get("Table"))), len(str(rec.get("code", ""))), i, r, rec, ev))

cands.sort(key=lambda t: (t[0], t[1]))
print(f"{len(cands)} candidate(s)\n")
if cands:
    _, _, i, r, rec, ev = cands[0]
    print("id:", i)
    print("\nQUESTION:", r.get("Question"))
    tbl = r.get("Table")
    if isinstance(tbl, str):
        try:
            tbl = json.loads(tbl)
        except Exception:  # noqa: BLE001
            pass
    print("\nTABLE columns:", tbl.get("columns") if isinstance(tbl, dict) else str(tbl)[:200])
    if isinstance(tbl, dict):
        print("first rows:", tbl.get("data")[:3])
    print("\nGOLD:", golds(r))
    print("\n--- pipeline error events ---")
    for e in ev[:2]:
        print(" ", e.get("etype"), "|", str(e.get("msg"))[:160])
        print("  stage:", e.get("stage"))
    print("\n--- tier-1 program (first 700 chars) ---")
    print(str(rec.get("code"))[:700])
    print("\n--- tier-1 printed ---")
    print(str(rec.get("answer"))[:200])
