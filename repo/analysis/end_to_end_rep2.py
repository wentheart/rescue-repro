"""End-to-end second run of the whole method.

Pass 1 is the second full 8B run (its own escalation set, 287 items); the rescue is re-run on
that set. This measures the variance of the complete method, not just of the rescue step.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "_g4_scoring_20261102"))
from scoring import extract, score, is_missing  # noqa: E402
from recompute import load  # noqa: E402

S1, S2, B = load("TatQa_b8b-full_123_"), load("TatQa_rep2_123_"), load("TatQa_b72-full_123_")
cc = json.load(open(os.path.join(ROOT, "_g4_scoring_20261102", "consistency_cascade.json"),
                    encoding="utf-8"))
s2res = json.load(open(os.path.join(ROOT, "_g4_scoring_20261102", "s2_results.json"),
                       encoding="utf-8"))


def golds(r):
    a = r.get("Answer") or []
    return a if isinstance(a, list) else [a]


def toks(d):
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


R2 = {}
for line in open(os.path.join(HERE, "rescue_rep2firstpass.jsonl"), encoding="utf-8",
                 errors="replace"):
    line = line.strip()
    if line:
        r = json.loads(line)
        R2[r["key"]] = r

ids = [i for i in S1 if i in S2 and i in B and not S1[i].get("pipeline_error")
       and not S2[i].get("pipeline_error") and not B[i].get("pipeline_error")]
N = len(ids)
tot_big = sum(toks(B[i]) for i in ids)
small = s2res["TAT-QA"]["small"] * 100
sc = upper_hull([(0.0, s2res["TAT-QA"]["sc3"] * 100)] +
                [(v["cost"] * 100, v["acc"] * 100) for v in cc["TAT-QA"]["consistency"]])

runs = {}
for name, S in (("run1", S1), ("run2", S2)):
    E = [i for i in ids if is_missing(S[i].get("ExtractResult"))]
    if name == "run1":
        files = os.path.join(HERE, "rescue_missing.jsonl")
    else:
        files = os.path.join(HERE, "rescue_rep2firstpass.jsonl")
    R = {}
    for line in open(files, encoding="utf-8", errors="replace"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r["key"].startswith("TAT-QA|"):
                R[r["key"]] = r
    base = sum(score(extract(S[i].get("ExtractResult")), golds(S[i])) for i in ids) / N
    q = []
    cost_num = 0.0
    for i in E:
        k = f"TAT-QA|{i}"
        rec = R.get(k)
        if rec is None:
            continue
        ok = rec["rawpot"]["status"] == "ok" and not is_missing(rec["rawpot"]["answer"])
        sc_i = rec["rawpot"]["score"] if ok else rec["full_score"]
        q.append(sc_i)
        cost_num += ((rec["rawpot"].get("prompt") or 0) +
                     (rec["rawpot"].get("completion") or 0))
        if not ok:
            cost_num += rec["full_tok"]["prompt"] + rec["full_tok"]["completion"]
    # accuracy = small-model scores on the non-escalated items + rescue scores on E
    nonE = [i for i in ids if i not in set(E)]
    acc = (sum(score(extract(S[i].get("ExtractResult")), golds(S[i])) for i in nonE)
           + sum(q)) / N * 100
    cost = cost_num / tot_big * 100
    runs[name] = (acc, cost, len(E), len(q))
    print(f"{name}: 8B alone {base * 100:.2f}  escalate {len(E)}/{N} ({len(E) / N:.1%})  "
          f"two-tier {acc:.2f} @ {cost:.2f}%")

a1, c1, _, _ = runs["run1"]
a2, c2, _, _ = runs["run2"]
print(f"\ndelta: accuracy {a2 - a1:+.2f}, cost {c2 - c1:+.2f}")

for name in runs:
    acc, cost, _, _ = runs[name]
    gap = acc - hull_eval(sc, cost)
    print(f"{name}: gap vs SC at {cost:.2f}% = {gap:+.2f}")
print(f"gap delta: {(a2 - hull_eval(sc, c2)) - (a1 - hull_eval(sc, c1)):+.2f}")
