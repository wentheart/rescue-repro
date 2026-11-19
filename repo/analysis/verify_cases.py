"""Verify the two case items against the frozen records before rewriting them."""
import json
import os
import sys

ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
G8 = os.path.join(ROOT, "_g8_plan_rescue_20261109")
sys.path.insert(0, os.path.join(ROOT, "_g4_scoring_20261102"))
from recompute import load  # noqa: E402

S = load("TatQa_b8b-full_123_")
rescue = {}
for line in open(os.path.join(G8, "rescue_missing.jsonl"), encoding="utf-8", errors="replace"):
    line = line.strip()
    if line:
        r = json.loads(line)
        rescue[r["key"]] = r


def show(i):
    r = S[i]
    print(f"\n{'=' * 70}\nitem {i}")
    print("Question:", r.get("Question"))
    print("Answer:", r.get("Answer"), "| ExtractResult:", repr(r.get("ExtractResult")))
    t = r.get("Table")
    if isinstance(t, str):
        t = json.loads(t)
    print("columns:", t.get("columns"))
    for k, row in enumerate(t.get("data")[:5]):
        print(f"  row {k}: {row}")
    a3 = r.get("agent3") or {}
    code = a3.get("output") or a3.get("output1") or ""
    print(f"--- agent3.output ({len(str(code))} chars) ---")
    txt = str(code)
    for line2 in txt.splitlines():
        if "table_df[" in line2 or "Change" in line2 or "2019" in line2 or "2018" in line2:
            print("   ", line2.strip()[:120])
    print("--- _exec_events ---")
    print(json.dumps(r.get("_exec_events"), ensure_ascii=False)[:600])
    rec = rescue.get(f"TAT-QA|{i}")
    if rec:
        rp = rec["rawpot"]
        print("--- tier-1 status:", rp.get("status"), "| answer:", repr(rp.get("answer")))
        print("--- tier-1 code ---")
        print(rp.get("code"))
        print("--- tier-1 tokens:",
              (rp.get("prompt") or 0) + (rp.get("completion") or 0))


for i in ("201", "22"):
    show(i)
