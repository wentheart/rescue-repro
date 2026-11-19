"""Cluster bootstrap for the two-tier rescue results.

Resampling unit: table_uid (items from the same table are correlated).
Everything is recomputed inside each replicate: our rescue scores, our cost, and the SC
operating points -- so the gap against SC also gets an interval instead of a point estimate.

Reported per dataset:
  * two-tier rescue score
  * rescue policy variants (rawpot-only, full rerun)
  * cascade accuracy and strong-token cost
  * the gap against the SC frontier, evaluated at our observed cost
  * the dominance window endpoints and the peak gap
"""
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "_g4_scoring_20261102"))
from scoring import extract, score, is_missing, vote_key  # noqa: E402
from recompute import load  # noqa: E402
from collections import Counter  # noqa: E402

SETS = {"TAT-QA": ("TatQa_b8b-full_123_", "TatQa_b72-full_123_", "TatQa_sc2_123_",
                   "TatQa_sc3_123_"),
        "CALTAB151": ("CalTab151_ct8b_123_", "CalTab151_b72-v2_123_", "CalTab151_sc2_123_",
                      "CalTab151_sc3_123_"),
        "TableBench": ("TableBench_tb8b_123_", "TableBench_b72-v2_123_", "TableBench_sc2_123_",
                       "TableBench_sc3_123_")}
ITERS = 2000
SEED = 20261110


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
    top = max(c.values())
    return c.most_common(1)[0][0], top / len(preds)


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


def crossings(ours, theirs, hi=80.0, steps=200):
    out, prev = [], hull_eval(ours, 0.01) - hull_eval(theirs, 0.01)
    for k in range(1, steps + 1):
        c = 0.01 + (hi - 0.01) * k / steps
        cur = hull_eval(ours, c) - hull_eval(theirs, c)
        if (prev > 0) != (cur > 0):
            a, b = c - hi / steps, c
            for _ in range(40):
                m = (a + b) / 2
                if (hull_eval(ours, m) - hull_eval(theirs, m) > 0) == (prev > 0):
                    a = m
                else:
                    b = m
            out.append((a + b) / 2)
        prev = cur
    return out


report = {}
for ds, (sp, bp, s2p, s3p) in SETS.items():
    S, B, SC2, SC3 = load(sp), load(bp), load(s2p), load(s3p)
    ids = [i for i in S if i in B and i in SC2 and i in SC3
           and not S[i].get("pipeline_error") and not B[i].get("pipeline_error")]
    # per-item arrays
    items = []
    for i in ids:
        g = golds(S[i])
        preds = [extract(S[i].get("ExtractResult")), extract(SC2[i].get("ExtractResult")),
                 extract(SC3[i].get("ExtractResult"))]
        key, agree = majority(preds)
        maj_score = 0.0
        for p in preds:
            if vote_key(p) == key:
                maj_score = score(p, g)
                break
        rec = rescue.get(f"{ds}|{i}")
        items.append({
            "uid": str(((S[i].get("MetaData") or {}).get("table_uid")) or S[i].get("rawid") or i),
            "s_base": score(extract(S[i].get("ExtractResult")), g),
            "s_big": score(extract(B[i].get("ExtractResult")), g),
            "tok_big": toks(B[i]),
            "maj": maj_score, "agree": agree, "maj_none": key is None,
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
    tot_big = sum(x["tok_big"] for x in items)
    clusters = {}
    for idx, x in enumerate(items):
        clusters.setdefault(x["uid"], []).append(idx)
    uids = list(clusters)

    def metrics(sample_idx):
        """All quantities for one resample (indices into items)."""
        xs = [items[k] for k in sample_idx]
        n = len(xs)
        tb = sum(x["tok_big"] for x in xs)
        # ours
        two_tier = sum((x["r_acc_score"] if x["acc"] else x["r_full"]) if x["esc"] else x["s_base"]
                       for x in xs) / n
        raw_only = sum((x["r_raw"] if x["esc"] else x["s_base"]) for x in xs) / n
        full_only = sum((x["r_full"] if x["esc"] else x["s_base"]) for x in xs) / n
        c_two = sum(x["tok_raw"] + (0 if x["acc"] else x["tok_full"]) for x in xs if x["esc"]) / tb
        c_raw = sum(x["tok_raw"] for x in xs if x["esc"]) / tb
        c_full = sum(x["tok_full"] for x in xs if x["esc"]) / tb
        # SC
        sc_pts = [(0.0, sum(x["maj"] for x in xs) / n * 100)]
        for tau in (1.0, 2 / 3, 1 / 3):
            esc = [x for x in xs if x["maj_none"] or x["agree"] < tau - 1e-9]
            es = set(id(x) for x in esc)
            acc = sum((x["s_big"] if id(x) in es else x["maj"]) for x in xs) / n
            cost = sum(x["tok_big"] for x in esc) / tb
            sc_pts.append((cost * 100, acc * 100))
        ours_pts = [(0.0, sum(x["s_base"] for x in xs) / n * 100),
                    (c_raw * 100, raw_only * 100), (c_two * 100, two_tier * 100),
                    (c_full * 100, full_only * 100)]
        return {"two_tier": two_tier * 100, "c_two": c_two * 100,
                "raw_only": raw_only * 100, "c_raw": c_raw * 100,
                "full_only": full_only * 100, "c_full": c_full * 100,
                "ours_pts": ours_pts, "sc_pts": sc_pts,
                "rescue": (sum((x["r_acc_score"] if x["acc"] else x["r_full"])
                               for x in xs if x["esc"]) /
                           max(1, sum(1 for x in xs if x["esc"]))) * 100}

    base = metrics(list(range(N)))
    c0 = base["c_two"]
    # point estimates for the derived quantities (gap / peak / window)
    oh_b, sh_b = upper_hull(base["ours_pts"]), upper_hull(base["sc_pts"])
    point = dict(base)
    point["gap0"] = hull_eval(oh_b, c0) - hull_eval(sh_b, c0)
    point["peak"] = max(((c, hull_eval(oh_b, c) - hull_eval(sh_b, c))
                         for c in [i * 0.5 for i in range(2, 200)]), key=lambda z: z[1])[1]
    cr_b = crossings(oh_b, sh_b)
    point["win_lo"] = cr_b[0] if cr_b else float("nan")
    point["win_hi"] = cr_b[-1] if cr_b else float("nan")
    rng = random.Random(SEED)
    draws = {k: [] for k in ("two_tier", "c_two", "raw_only", "c_raw", "full_only", "c_full",
                             "rescue", "gap0", "peak", "win_lo", "win_hi")}
    for _ in range(ITERS):
        sample = []
        for _ in range(len(uids)):
            sample.extend(clusters[uids[rng.randrange(len(uids))]])
        m = metrics(sample)
        for k in ("two_tier", "c_two", "raw_only", "c_raw", "full_only", "c_full", "rescue"):
            draws[k].append(m[k])
        oh, sh = upper_hull(m["ours_pts"]), upper_hull(m["sc_pts"])
        draws["gap0"].append(hull_eval(oh, c0) - hull_eval(sh, c0))
        pk = max(((c, hull_eval(oh, c) - hull_eval(sh, c))
                  for c in [i * 0.5 for i in range(2, 200)]), key=lambda z: z[1])
        draws["peak"].append(pk[1])
        cr = crossings(oh, sh)
        if len(cr) >= 2:
            draws["win_lo"].append(cr[0])
            draws["win_hi"].append(cr[-1])

    def ci(key, fmt="{:.2f}"):
        v = draws[key]
        if not v:
            return f"{fmt.format(point[key])} [n/a]"
        return f"{fmt.format(point[key])} [{quantile(v, 0.025):.2f}, {quantile(v, 0.975):.2f}]"

    print(f"\n=== {ds} (n={N}, clusters={len(uids)}, our cost {c0:.2f}%) ===")
    for k in ("two_tier", "raw_only", "full_only", "c_two", "c_raw", "c_full"):
        print(f"  {k:<10} {ci(k)}")
    print(f"  rescue     {ci('rescue')}")
    print(f"  gap@cost   {ci('gap0')}")
    print(f"  peak       {ci('peak')}")
    print(f"  win_lo     {ci('win_lo')}")
    print(f"  win_hi     {ci('win_hi')}")
    report[ds] = {"n": N, "clusters": len(uids), "cost0": c0,
                  "mean": {k: base[k] for k in ("two_tier", "raw_only", "full_only",
                                                "c_two", "c_raw", "c_full", "rescue")},
                  "ci": {k: [quantile(draws[k], 0.025), quantile(draws[k], 0.975)]
                         for k in draws if draws[k]}}

json.dump(report, open(os.path.join(HERE, "bootstrap_ci.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n-> bootstrap_ci.json")
