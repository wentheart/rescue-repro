"""Recompute Agent2 compliance the way the pipeline actually does it.

`agent2.parseError` refers to the REFLECTION reply, not to the initial cleaning. The pipeline's
initial step is: find the last ```json ... ``` block in output1 and parse it; if there is none
or it does not parse, it falls back to the original table.

Reported per model: compliance count, and how many unanswered items still had a successful
first cleaning (i.e. the failure was downstream, in code execution).
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "_g4_scoring_20261102"))
from scoring import extract, score, is_missing  # noqa: E402

OUT = os.path.join(HERE, "out_pilot")
MODELS = ["hunyuan", "ling", "glm"]
DATASETS = ["TatQa", "TableBench", "CalTab151"]


def pipeline_clean(txt):
    """Exactly the pipeline's initial cleaning step (Agent2.py:88-92).

    parse the LAST ```json block, read columns/data, and build a DataFrame. A dimension
    mismatch (len(columns) != len(row)) raises and the pipeline falls back to the raw table.
    """
    blocks = re.findall(r"```json(.*?)```", str(txt or ""), re.IGNORECASE | re.DOTALL)
    if not blocks:
        return None
    try:
        obj = json.loads(blocks[-1].strip())
        obj["columns"]
        obj["data"]
        import pandas as pd
        pd.DataFrame(obj["data"], columns=obj["columns"])
    except Exception:  # noqa: BLE001
        return None
    return obj


def golds(r):
    a = r.get("Answer") or []
    return a if isinstance(a, list) else [a]


rows = []
for tag in MODELS:
    for ds in DATASETS:
        p = os.path.join(OUT, f"{ds}_{tag}_s0.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p, encoding="utf-8")):
            clean_ok = pipeline_clean((r.get("agent2") or {}).get("output1")) is not None
            ev = r.get("_exec_events") or []
            ans = r.get("ExtractResult")
            rows.append({"model": tag, "dataset": ds, "id": r.get("id"),
                         "clean_ok": clean_ok, "exec": bool(ev),
                         "missing": is_missing(ans),
                         "correct": score(extract(ans), golds(r)) >= 1 - 1e-9})

print(f"{'model':<9}{'n':>4}{'clean ok':>10}{'compliance':>12}{'unanswered':>12}"
      f"{'unans w/ clean ok':>19}{'correct':>9}")
for tag in MODELS:
    rs = [r for r in rows if r["model"] == tag]
    n = len(rs)
    ck = sum(1 for r in rs if r["clean_ok"])
    un = [r for r in rs if r["missing"]]
    print(f"{tag:<9}{n:>4}{ck:>10}{ck / n:>11.1%}{len(un):>12}"
          f"{sum(1 for r in un if r['clean_ok']):>19}"
          f"{sum(1 for r in rs if r['correct']):>9}")

print("\n=== unanswered items: cleaning status x execution ===")
for tag in MODELS:
    rs = [r for r in rows if r["model"] == tag and r["missing"]]
    both = sum(1 for r in rs if r["clean_ok"] and r["exec"])
    only_exec = sum(1 for r in rs if not r["clean_ok"] and r["exec"])
    print(f"  {tag:<9} unanswered {len(rs):>2} | clean ok & exec fail {both:>2} | "
          f"clean fallback & exec fail {only_exec:>2}")
