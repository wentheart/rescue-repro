"""Run-to-run variance of the two-tier rescue.

Rep1 and rep2 re-run only the first tier (the strong model's single call on the source table);
the fallback answers are cached, so any difference between the two runs is the method's own
sampling noise.

Reported per dataset:
  * rescue score and cascade accuracy in each run, and the difference
  * acceptance rate in each run
  * per-item flip rate (agreement of accepted/not-accepted and of the final answer)
  * the gap against SC in each run and its difference
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "_g4_scoring_20261102"))
from scoring import extract, score, is_missing, vote_key  # noqa: E402
from recompute import load  # noqa: E402

SETS = {"TAT-QA": ("TatQa_b8b-full_123_", "TatQa_b72-full_123_"),
        "CALTAB151": ("CalTab151_ct8b_123_", "CalTab151_b72-v2_123_"),
        "TableBench": ("TableBench_tb8b_123_", "TableBench_b72-v2_123_")}
cc = json.load(open(os.path.join(ROOT, "_g4_scoring_20261102", "consistency_cascade.json"),
                    encoding="utf-8"))
s2 = json.load(open(os.path.join(ROOT, "_g4_scoring_20261102", "s2_results.json"),
                    encoding="utf-8"))


def load_run(name):
    out = {}
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return out
    for line in open(p, encoding="utf-8", errors="replace"):
        line = line.strip()
        if line:
            try:
                r = json.loads(line)
                out[r["key"]] = r
            except Exception:  # noqa: BLE001
                pass
    return out


R1 = load_run("rescue_missing.jsonl")
R2 = load_run("rescue_missing_rep2.jsonl")
print(f"rep1 items {len(R1)}   rep2 items {len(R2)}")

accepted = lambda rec: rec.get("status") == "ok" and not is_missing(rec.get("answer"))


def toks(d):
    if "prompt" in d:
        return (d.get("prompt") or 0) + (d.get("completion") or 0)
    return (d.get("_ptok") or 0) + (d.get("_ctok") or 0)


def hull_eval(pts, c):
    pts = sorted(pts)
    if c <= pts[0][0]:
        return pts[0][1]
    if c >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= c <= x1:
            return y0 + (y1 - y0) * (c - x0) / (x1 - x0)


def upper_hull(points):
    pts = sorted(set(points))
    keep = [p for p in pts if not any(q[0] <= p[0] and q[1] >= p[1] and q != p for q in pts)]
    hull = []
    for p in keep:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def sc_points(ds, n, tot_big, items_by_id):
    pts = [(0.0, s2[ds]["sc3"] * 100)]
    for v in cc[ds]["consistency"]:
        pts.append((v["cost"] * 100, v["acc"] * 100))
    return pts


for ds, (sp, bp) in SETS.items():
    keys = [k for k in R1 if k.startswith(ds + "|") and k in R2]
    if not keys:
        continue
    S, B = load(sp), load(bp)
    all_ids = [i for i in S if i in B and not S[i].get("pipeline_error")
               and not B[i].get("pipeline_error")]
    denom = sum(toks(B[i]) for i in all_ids)
    N = len(all_ids)
    esc = len(keys) / N
    small = s2[ds]["small"] * 100

    print(f"\n=== {ds} (n={len(keys)}, escalation {esc:.1%}) ===")
    out = {}
    for name, R in (("rep1", R1), ("rep2", R2)):
        rescue = sum((R[k]["rawpot"]["score"] if accepted(R[k]["rawpot"])
                      else R[k]["full_score"]) for k in keys) / len(keys)
        acc_rate = sum(1 for k in keys if accepted(R[k]["rawpot"])) / len(keys)
        c = (sum(toks(R[k]["rawpot"]) for k in keys)
             + sum(R[k]["full_tok"]["prompt"] + R[k]["full_tok"]["completion"]
                   for k in keys if not accepted(R[k]["rawpot"]))) / denom * 100
        casc = small + esc * rescue * 100
        out[name] = (rescue, acc_rate, c, casc)
        print(f"  {name}: rescue {rescue:.4f}  accept {acc_rate:.1%}  cost {c:.2f}%  "
              f"cascade {casc:.2f}")

    dr = out["rep2"][0] - out["rep1"][0]
    da = out["rep2"][1] - out["rep1"][1]
    dc = out["rep2"][3] - out["rep1"][3]
    print(f"  delta: rescue {dr:+.4f}  accept {da:+.1%}  cascade {dc:+.2f}")

    # per-item agreement
    same_accept = sum(1 for k in keys
                      if accepted(R1[k]["rawpot"]) == accepted(R2[k]["rawpot"]))
    same_answer = sum(1 for k in keys
                      if vote_key(extract(R1[k]["rawpot"]["answer"])) ==
                      vote_key(extract(R2[k]["rawpot"]["answer"])))
    same_final = 0
    for k in keys:
        a1 = R1[k]["rawpot"]["score"] if accepted(R1[k]["rawpot"]) else R1[k]["full_score"]
        a2 = R2[k]["rawpot"]["score"] if accepted(R2[k]["rawpot"]) else R2[k]["full_score"]
        same_final += 1 if abs(a1 - a2) < 1e-9 else 0
    n = len(keys)
    print(f"  per-item: acceptance agrees {same_accept / n:.1%}, "
          f"tier-1 answer agrees {same_answer / n:.1%}, final score agrees {same_final / n:.1%}")
