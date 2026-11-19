"""Joint bootstrap: macro-average gap against SC, plus a non-inferiority check.

Each replicate resamples tables *within* every dataset, recomputes that dataset's gap at its
own observed cost, then averages the three gaps (macro). This gives one interval for the
aggregate claim, instead of reading three separate intervals.

Non-inferiority: report the largest negative excursion the data still allow, so a margin can
be judged against it rather than asserted after the fact.
"""
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "_g4_scoring_20261102"))
from scoring import extract, score, is_missing, vote_key  # noqa: E402
from recompute import load  # noqa: E402

SETS = {"TAT-QA": ("TatQa_b8b-full_123_", "TatQa_b72-full_123_", "TatQa_sc2_123_",
                   "TatQa_sc3_123_"),
        "CALTAB151": ("CalTab151_ct8b_123_", "CalTab151_b72-v2_123_", "CalTab151_sc2_123_",
                      "CalTab151_sc3_123_"),
        "TableBench": ("TableBench_tb8b_123_", "TableBench_b72-v2_123_", "TableBench_sc2_123_",
                       "TableBench_sc3_123_")}
ITERS, SEED = 2000, 20261111


def toks(r):
    return (r.get("_ptok") or 0) + (r.get("_ctok") or 0)


def golds(r):
    a = r.get("Answer") or []
    return a if isinstance(a, list) else [a]


def majority(preds):
    keys = [vote_key(p) for p in preds]
    live = [k for k in keys if k]
    if not live:
        return None, 0.0
    c = Counter(live)
    return c.most_common(1)[0][0], max(c.values()) / len(preds)


rescue = {}
for line in open(os.path.join(HERE, "rescue_missing.jsonl"), encoding="utf-8",
                 errors="replace"):
    line = line.strip()
    if line:
        try:
            r = json.loads(line)
            rescue[r["key"]] = r
        except Exception:  # noqa: BLE001
            pass
accepted = lambda rec: rec.get("status") == "ok" and not is_missing(rec.get("answer"))


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


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(p * len(xs))))]


data = {}
for ds, (sp, bp, s2p, s3p) in SETS.items():
    S, B, SC2, SC3 = load(sp), load(bp), load(s2p), load(s3p)
    ids = [i for i in S if i in B and i in SC2 and i in SC3
           and not S[i].get("pipeline_error") and not B[i].get("pipeline_error")]
    items = []
    for i in ids:
        g = golds(S[i])
        preds = [extract(S[i].get("ExtractResult")), extract(SC2[i].get("ExtractResult")),
                 extract(SC3[i].get("ExtractResult"))]
        key, agree = majority(preds)
        maj = 0.0
        for p in preds:
            if vote_key(p) == key:
                maj = score(p, g)
                break
        rec = rescue.get(f"{ds}|{i}")
        items.append({"uid": str(((S[i].get("MetaData") or {}).get("table_uid")) or S[i].get("rawid") or i),
                      "s_base": score(extract(S[i].get("ExtractResult")), g),
                      "s_big": score(extract(B[i].get("ExtractResult")), g),
                      "tok_big": toks(B[i]), "maj": maj, "agree": agree,
                      "maj_none": key is None, "esc": rec is not None,
                      "acc": accepted(rec["rawpot"]) if rec else False,
                      "r_full": rec["full_score"] if rec else 0.0,
                      "r_acc": (rec["rawpot"]["score"] if rec and accepted(rec["rawpot"])
                                else 0.0),
                      "tok_raw": (((rec["rawpot"].get("prompt") or 0) +
                                   (rec["rawpot"].get("completion") or 0)) if rec else 0),
                      "tok_full": ((rec["full_tok"]["prompt"] + rec["full_tok"]["completion"])
                                   if rec else 0)})
    clusters = {}
    for idx, x in enumerate(items):
        clusters.setdefault(x["uid"], []).append(idx)
    data[ds] = {"items": items, "clusters": clusters, "uids": list(clusters)}


def gap_for(ds, sample):
    items = data[ds]["items"]
    xs = [items[k] for k in sample]
    n = len(xs)
    tb = sum(x["tok_big"] for x in xs)
    two = sum((x["r_acc"] if x["acc"] else x["r_full"]) if x["esc"] else x["s_base"]
              for x in xs) / n
    c_two = sum(x["tok_raw"] + (0 if x["acc"] else x["tok_full"])
                for x in xs if x["esc"]) / tb
    ours = [(0.0, sum(x["s_base"] for x in xs) / n * 100),
            (c_two * 100, two * 100)]
    sc = [(0.0, sum(x["maj"] for x in xs) / n * 100)]
    for tau in (1.0, 2 / 3, 1 / 3):
        esc = [x for x in xs if x["maj_none"] or x["agree"] < tau - 1e-9]
        es = set(id(x) for x in esc)
        acc = sum((x["s_big"] if id(x) in es else x["maj"]) for x in xs) / n
        cost = sum(x["tok_big"] for x in esc) / tb
        sc.append((cost * 100, acc * 100))
    oh = upper_hull(ours)
    sh = upper_hull(sc)
    return hull_eval(oh, c_two * 100) - hull_eval(sh, c_two * 100), c_two * 100


base = {}
for ds in data:
    full = list(range(len(data[ds]["items"])))
    g, c = gap_for(ds, full)
    base[ds] = (g, c)
    print(f"{ds:<12} gap {g:+.2f} @ our cost {c:.2f}%")

macro_base = sum(v[0] for v in base.values()) / len(base)
rng = random.Random(SEED)
draws = {"macro": [], "min": [], **{ds: [] for ds in data}}
for _ in range(ITERS):
    gaps = []
    for ds in data:
        d = data[ds]
        sample = []
        for _ in range(len(d["uids"])):
            sample.extend(d["clusters"][d["uids"][rng.randrange(len(d["uids"]))]])
        g, _ = gap_for(ds, sample)
        gaps.append(g)
        draws[ds].append(g)
    draws["macro"].append(sum(gaps) / len(gaps))
    draws["min"].append(min(gaps))

print(f"\nmacro-average gap {macro_base:+.2f} "
      f"[{q(draws['macro'], .025):+.2f}, {q(draws['macro'], .975):+.2f}]")
print(f"per-dataset 95% CIs:")
for ds in data:
    print(f"  {ds:<12} {base[ds][0]:+.2f} [{q(draws[ds], .025):+.2f}, "
          f"{q(draws[ds], .975):+.2f}]")
print(f"\nworst-case dataset gap in 2.5% of replicates: {q(draws['min'], .025):+.2f}")
print(f"  -> at a non-inferiority margin of 2 points, "
      f"{'PASS' if q(draws['min'], .025) > -2 else 'FAIL'}")
json.dump({"base": {k: list(v) for k, v in base.items()},
           "macro": macro_base,
           "macro_ci": [q(draws["macro"], .025), q(draws["macro"], .975)],
           "per_dataset_ci": {ds: [q(draws[ds], .025), q(draws[ds], .975)] for ds in data},
           "worst_case": q(draws["min"], .025)},
          open(os.path.join(HERE, "bootstrap_macro.json"), "w", encoding="utf-8"), indent=1)
print("\n-> bootstrap_macro.json")
