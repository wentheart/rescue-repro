"""Frontier comparison on all three datasets, using each dataset's own frozen SC points."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "_g4_scoring_20261102"))
from scoring import is_missing  # noqa: E402
from recompute import load  # noqa: E402

SETS = {"TAT-QA": ("TatQa_b8b-full_123_", "TatQa_b72-full_123_"),
        "CALTAB151": ("CalTab151_ct8b_123_", "CalTab151_b72-v2_123_"),
        "TableBench": ("TableBench_tb8b_123_", "TableBench_b72-v2_123_")}

cc = json.load(open(os.path.join(ROOT, "_g4_scoring_20261102", "consistency_cascade.json"),
                    encoding="utf-8"))
s2 = json.load(open(os.path.join(ROOT, "_g4_scoring_20261102", "s2_results.json"),
                    encoding="utf-8"))

rows = {}
for line in open(os.path.join(HERE, "rescue_missing.jsonl"), encoding="utf-8",
                 errors="replace"):
    line = line.strip()
    if line:
        try:
            r = json.loads(line)
            rows[r["key"]] = r
        except Exception:  # noqa: BLE001
            pass


def toks(d):
    if "prompt" in d:
        return (d.get("prompt") or 0) + (d.get("completion") or 0)
    return (d.get("_ptok") or 0) + (d.get("_ctok") or 0)


acc = lambda rec: rec.get("status") == "ok" and not is_missing(rec.get("answer"))


def hull(pts, c):
    pts = sorted(pts)
    if c <= pts[0][0]:
        return pts[0][1]
    if c >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= c <= x1:
            return y0 + (y1 - y0) * (c - x0) / (x1 - x0)


def crossings(p, q, lo=0.01, hi=60.0, steps=300000):
    out, prev = [], hull(p, lo) - hull(q, lo)
    for k in range(1, steps + 1):
        c = lo + (hi - lo) * k / steps
        cur = hull(p, c) - hull(q, c)
        if (prev > 0) != (cur > 0):
            a, b = c - (hi - lo) / steps, c
            for _ in range(60):
                m = (a + b) / 2
                if (hull(p, m) - hull(q, m) > 0) == (prev > 0):
                    a = m
                else:
                    b = m
            out.append((a + b) / 2)
        prev = cur
    return out


for ds, (sp, bp) in SETS.items():
    S, B = load(sp), load(bp)
    mine = {k: r for k, r in rows.items() if k.startswith(ds + "|")}
    if not mine:
        continue
    all_ids = [i for i in S if i in B and not S[i].get("pipeline_error")
               and not B[i].get("pipeline_error")]
    denom = sum(toks(B[i]) for i in all_ids)
    n = len(mine)
    esc = n / len(all_ids)
    small = s2[ds]["small"] * 100

    rB = sum(r["rawpot"]["score"] for r in mine.values()) / n
    cB = sum(toks(r["rawpot"]) for r in mine.values()) / denom
    rC = sum((r["rawpot"]["score"] if acc(r["rawpot"]) else r["full_score"])
             for r in mine.values()) / n
    cC = (sum(toks(r["rawpot"]) for r in mine.values())
          + sum(r["full_tok"]["prompt"] + r["full_tok"]["completion"]
                for r in mine.values() if not acc(r["rawpot"]))) / denom
    rD = sum(r["full_score"] for r in mine.values()) / n
    cD = sum(r["full_tok"]["prompt"] + r["full_tok"]["completion"]
             for r in mine.values()) / denom

    ours = [(0.0, small), (cB * 100, small + esc * rB * 100),
            (cC * 100, small + esc * rC * 100), (cD * 100, small + esc * rD * 100)]
    theirs = [(0.0, s2[ds]["sc3"] * 100)]
    theirs += sorted((v["cost"] * 100, v["acc"] * 100) for v in cc[ds]["consistency"])
    # keep only points on the upper hull of theirs
    theirs = [(x, y) for x, y in sorted(theirs)
              if y >= max((yy for xx, yy in theirs if xx <= x), default=-1)]

    print(f"\n=== {ds} (escalation {esc:.1%}, n={n}) ===")
    print(f"  {'policy':<14}{'cost%':>8}{'cascade':>9}")
    for name, (c, a) in zip("ABCD", ours):
        print(f"  {name:<14}{c:>8.2f}{a:>9.2f}")
    xs = crossings(ours, theirs)
    if xs:
        lo, hi = xs[0], xs[-1]
        best = max(((c, hull(ours, c) - hull(theirs, c)) for c in
                    [lo + (hi - lo) * i / 200 for i in range(201)]), key=lambda z: z[1])
        print(f"  dominance window: {lo:.2f}% .. {hi:.2f}%   peak diff "
              f"{best[1]:+.2f} at {best[0]:.2f}%")
    else:
        print(f"  no crossing in range (ours - SC at 5%: "
              f"{hull(ours, 5) - hull(theirs, 5):+.2f}, at 20%: "
              f"{hull(ours, 20) - hull(theirs, 20):+.2f})")
