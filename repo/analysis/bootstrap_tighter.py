"""Cluster bootstrap for the gap at the TIGHTER budget (the tier-1-only operating point).

Same resampling scheme as bootstrap_ci.py: unit is the table, everything is recomputed inside each
replicate, and the gap is evaluated at the observed tier-1-only cost so the interval describes the
number the paper reports.
"""
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.join(os.environ.get("PROJECT_ROOT", os.getcwd()), "_g8_plan_rescue_20261109")
ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "_g4_scoring_20261102"))
from scoring import extract, score, is_missing, vote_key  # noqa: E402
from recompute import load  # noqa: E402

SETS = {"TAT-QA": ("TatQa_b8b-full_123_", "TatQa_b72-full_123_",
                   "TatQa_sc2_123_", "TatQa_sc3_123_"),
        "CALTAB151": ("CalTab151_ct8b_123_", "CalTab151_b72-v2_123_",
                      "CalTab151_sc2_123_", "CalTab151_sc3_123_"),
        "TableBench": ("TableBench_tb8b_123_", "TableBench_b72-v2_123_",
                       "TableBench_sc2_123_", "TableBench_sc3_123_")}
ITERS, SEED = 2000, 20261110


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


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(q * len(xs))))]


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


rescue = {}
for line in open(os.path.join(HERE, "rescue_missing.jsonl"), encoding="utf-8", errors="replace"):
    line = line.strip()
    if line:
        r = json.loads(line)
        rescue[r["key"]] = r

accepted = lambda rec: rec.get("status") == "ok" and not is_missing(rec.get("answer"))

out = {}
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
        items.append({
            "uid": str(((S[i].get("MetaData") or {}).get("table_uid")) or S[i].get("rawid") or i),
            "s_base": score(extract(S[i].get("ExtractResult")), g),
            "s_big": score(extract(B[i].get("ExtractResult")), g),
            "tok_big": toks(B[i]),
            "maj": maj, "agree": agree, "maj_none": key is None,
            "esc": rec is not None,
            "acc": accepted(rec["rawpot"]) if rec else False,
            "r_raw": rec["rawpot"]["score"] if rec else 0.0,
            "r_full": rec["full_score"] if rec else 0.0,
            "r_acc_score": (rec["rawpot"]["score"] if rec and accepted(rec["rawpot"]) else 0.0),
            "tok_raw": (((rec["rawpot"].get("prompt") or 0) +
                         (rec["rawpot"].get("completion") or 0)) if rec else 0),
            "tok_full": (rec["full_tok"]["prompt"] + rec["full_tok"]["completion"]) if rec else 0,
        })
    N = len(items)
    clusters = {}
    for idx, x in enumerate(items):
        clusters.setdefault(x["uid"], []).append(idx)
    uids = list(clusters)

    def metrics(sample_idx):
        xs = [items[k] for k in sample_idx]
        n = len(xs)
        tb = sum(x["tok_big"] for x in xs)
        two = sum((x["r_acc_score"] if x["acc"] else x["r_full"]) if x["esc"] else x["s_base"]
                  for x in xs) / n
        raw = sum((x["r_raw"] if x["esc"] else x["s_base"]) for x in xs) / n
        full = sum((x["r_full"] if x["esc"] else x["s_base"]) for x in xs) / n
        c_two = sum(x["tok_raw"] + (0 if x["acc"] else x["tok_full"]) for x in xs if x["esc"]) / tb
        c_raw = sum(x["tok_raw"] for x in xs if x["esc"]) / tb
        c_full = sum(x["tok_full"] for x in xs if x["esc"]) / tb
        sc_pts = [(0.0, sum(x["maj"] for x in xs) / n * 100)]
        for tau in (1.0, 2 / 3, 1 / 3):
            esc = [x for x in xs if x["maj_none"] or x["agree"] < tau - 1e-9]
            es = set(id(x) for x in esc)
            acc = sum((x["s_big"] if id(x) in es else x["maj"]) for x in xs) / n
            cost = sum(x["tok_big"] for x in esc) / tb
            sc_pts.append((cost * 100, acc * 100))
        ours_pts = [(0.0, sum(x["s_base"] for x in xs) / n * 100),
                    (c_raw * 100, raw * 100), (c_two * 100, two * 100), (c_full * 100, full * 100)]
        return {"ours_pts": ours_pts, "sc_pts": sc_pts, "c_raw": c_raw * 100,
                "ours_raw": raw * 100}

    base = metrics(list(range(N)))
    c_t = base["c_raw"]
    oh_b, sh_b = upper_hull(base["ours_pts"]), upper_hull(base["sc_pts"])
    gap_pt = hull_eval(oh_b, c_t) - hull_eval(sh_b, c_t)
    ours_pt = hull_eval(oh_b, c_t)
    sc_pt = hull_eval(sh_b, c_t)

    rng = random.Random(SEED)
    gaps = []
    for _ in range(ITERS):
        sample = []
        for _ in range(len(uids)):
            sample.extend(clusters[uids[rng.randrange(len(uids))]])
        m = metrics(sample)
        oh, sh = upper_hull(m["ours_pts"]), upper_hull(m["sc_pts"])
        gaps.append(hull_eval(oh, c_t) - hull_eval(sh, c_t))

    lo, hi = quantile(gaps, 0.025), quantile(gaps, 0.975)
    out[ds] = dict(cost=c_t, ours=ours_pt, sc=sc_pt, gap=gap_pt, lo=lo, hi=hi)
    print(f"{ds:<11} budget {c_t:>5.2f}%  ours {ours_pt:>6.2f}  sc {sc_pt:>6.2f}  "
          f"gap {gap_pt:+.2f} [{lo:+.2f}, {hi:+.2f}]")

json.dump(out, open(os.path.join(HERE, "bootstrap_tighter.json"), "w", encoding="utf-8"), indent=1)
print("\n-> bootstrap_tighter.json")
