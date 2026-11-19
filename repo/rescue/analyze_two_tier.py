"""Offline scoring of the two-tier rescue, with the accounting formula agreed in review.

Acceptance rule (frozen, uses only runtime information):
    accept  iff  status == "ok" and not is_missing(answer)
    otherwise fall back to the cached FULL answer for that item

Cost (per dataset):
    numerator   = sum of RAW-PoT tokens over all escalation items
                + sum of FULL tokens over the items that fell back
    denominator = sum of FULL tokens over ALL items of the dataset
    cost        = numerator / denominator

The trigger is unchanged (first-pass missing), so the escalation set is fixed.

SC reference below 22% is NOT a measured operating point: it is the linear mixture of two
fixed policies (SC@3 vote-only at zero strong-model cost, and the tau=1/3 point). It is
reported as such.
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "_g4_scoring_20261102"))
from scoring import is_missing  # noqa: E402
from recompute import load  # noqa: E402

SETS = {"TAT-QA": ("TatQa_b8b-full_123_", "TatQa_b72-full_123_"),
        "CALTAB151": ("CalTab151_ct8b_123_", "CalTab151_b72-v2_123_"),
        "TableBench": ("TableBench_tb8b_123_", "TableBench_b72-v2_123_")}

rows = {}
for line in open(os.path.join(HERE, "rescue_missing.jsonl"), encoding="utf-8",
                 errors="replace"):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
        rows[r["key"]] = r
    except Exception:  # noqa: BLE001
        pass

print(f"items with a rescue record: {len(rows)}")

cc = json.load(open(os.path.join(ROOT, "_g4_scoring_20261102", "consistency_cascade.json"),
                    encoding="utf-8"))
s2 = json.load(open(os.path.join(ROOT, "_g4_scoring_20261102", "s2_results.json"),
                    encoding="utf-8"))


def accept(rec):
    return rec.get("status") == "ok" and not is_missing(rec.get("answer"))


def toks(d):
    """Works for both shapes: rescue records use prompt/completion, cached runs use _ptok/_ctok."""
    if "prompt" in d:
        return (d.get("prompt") or 0) + (d.get("completion") or 0)
    return (d.get("_ptok") or 0) + (d.get("_ctok") or 0)


print(f"\n{'dataset':<12}{'n':>4}{'accept%':>9}{'fallback':>9}{'rescue':>8}"
      f"{'cost%':>7}{'cascade':>9}")
summary = {}
for ds, (sp, bp) in SETS.items():
    S, B = load(sp), load(bp)
    all_ids = [i for i in S if i in B and not S[i].get("pipeline_error")
               and not B[i].get("pipeline_error")]
    denom = sum(toks(B[i]) for i in all_ids)
    mine = [rows[k] for k in rows if k.startswith(ds + "|")]
    if not mine:
        continue
    n = len(mine)
    acc = [r for r in mine if accept(r["rawpot"])]
    fb = [r for r in mine if not accept(r["rawpot"])]
    rescue = sum((r["rawpot"]["score"] if accept(r["rawpot"]) else r["full_score"])
                 for r in mine) / n
    num = sum(toks(r["rawpot"]) for r in mine) + sum(r["full_tok"]["prompt"] +
                                                     r["full_tok"]["completion"] for r in fb)
    cost = num / denom
    small = s2[ds]["small"] * 100
    esc = len(mine) / len(all_ids)
    casc = small + esc * rescue * 100
    summary[ds] = (n, rescue, cost * 100, casc, len(acc) / n, len(fb))
    print(f"{ds:<12}{n:>4}{len(acc) / n:>9.0%}{len(fb):>9}{rescue:>8.3f}"
          f"{cost * 100:>7.2f}{casc:>9.2f}")

# ---- comparison against FULL and the SC reference
print("\n=== same escalation set ===")
for ds, (sp, bp) in SETS.items():
    if ds not in summary:
        continue
    mine = [r for k, r in rows.items() if k.startswith(ds + "|")]
    full_rescue = sum(r["full_score"] for r in mine) / len(mine)
    print(f"{ds:<12} FULL rescue {full_rescue:.3f}   two-tier {summary[ds][1]:.3f}")

print("\n=== vs the consistency cascade ===")
for ds in summary:
    n, rescue, cost, casc, ar, nfb = summary[ds]
    pts = [(0.0, s2[ds]["sc3"] * 100)]
    pts += sorted((v["cost"] * 100, v["acc"] * 100) for v in cc[ds]["consistency"])
    if cost <= pts[0][0]:
        ref, note = pts[0][1], "at/below zero-cost endpoint"
    elif cost >= pts[-1][0]:
        ref, note = pts[-1][1], "at/above highest measured point"
    else:
        ref = next(y0 + (y1 - y0) * (cost - x0) / (x1 - x0)
                   for (x0, y0), (x1, y1) in zip(pts, pts[1:]) if x0 <= cost <= x1)
        note = "mixed endpoints" if cost < pts[1][0] else "between measured points"
    print(f"{ds:<12} ours {casc:6.2f} @ {cost:5.2f}%  |  SC {ref:6.2f}  -> {casc - ref:+.2f}"
          f"   [{note}]")

# ---- failure breakdown
print("\n=== rawpot status ===")
for ds in summary:
    c = Counter(r["rawpot"]["status"] for k, r in rows.items() if k.startswith(ds + "|"))
    print(f"{ds:<12}" + ", ".join(f"{k}x{v}" for k, v in c.most_common()))
print("\n=== accepted but wrong (these cap the quality) ===")
for ds in summary:
    mine = [r for k, r in rows.items() if k.startswith(ds + "|")]
    aw = [r for r in mine if accept(r["rawpot"]) and r["rawpot"]["score"] < 1 - 1e-9]
    ok = [r for r in mine if accept(r["rawpot"])]
    print(f"{ds:<12} {len(aw)}/{len(ok)} accepted answers are not fully correct"
          f"  (mean score {sum(r['rawpot']['score'] for r in ok) / max(1, len(ok)):.3f})")
