"""Recompute everything under S2: main table, self-consistency with normalised voting,
and paired cluster-bootstrap intervals on the same metric the table reports.

Fixes the two P0 items of the second review:
  * P0-1: extraction never sees the reference (see scoring.py);
  * P0-2: SC votes on the extracted answer, not on the raw string.
"""
import glob
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
OUT = os.path.join(ROOT, "tabdsr_exp", "out")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tabdsr_exp"))
from scoring import extract, score, is_missing, vote  # noqa: E402

SETS = [
    ("TAT-QA", "TatQa_b8b-full_123_", "TatQa_b72-full_123_", "TatQa_sc2_123_", "TatQa_sc3_123_"),
    ("CALTAB151", "CalTab151_ct8b_123_", "CalTab151_b72-v2_123_", "CalTab151_sc2_123_",
     "CalTab151_sc3_123_"),
    ("TableBench", "TableBench_tb8b_123_", "TableBench_b72-v2_123_", "TableBench_sc2_123_",
     "TableBench_sc3_123_"),
]


def load(prefix):
    rows = {}
    for f in sorted(glob.glob(os.path.join(OUT, prefix + "*.json"))):
        try:
            for r in json.load(open(f, encoding="utf-8")):
                i = str(r["id"])
                old = rows.get(i)
                bad_new = bool(str(r.get("pipeline_error") or "").strip())
                bad_old = bool(str((old or {}).get("pipeline_error") or "").strip())
                if old is None or (bad_old and not bad_new):
                    rows[i] = r
        except Exception:  # noqa: BLE001
            continue
    return rows


def golds(r):
    a = r.get("Answer") or []
    return a if isinstance(a, list) else [a]


def cluster(r):
    md = r.get("MetaData")
    if isinstance(md, str):
        try:
            md = json.loads(md)
        except Exception:  # noqa: BLE001
            md = {}
    if isinstance(md, dict) and md.get("table_uid"):
        return md["table_uid"]
    return "id:" + str(r.get("id"))


def boot(pairs, iters=2000, seed=0):
    by = {}
    for cl, d in pairs:
        by.setdefault(cl, []).append(d)
    keys = list(by)
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        flat = [v for _ in keys for v in by[rng.choice(keys)]]
        vals.append(sum(flat) / len(flat) * 100)
    vals.sort()
    return vals[int(0.025 * iters)], vals[int(0.975 * iters)]


def main():
    print(f"{'dataset':<12}{'n':>5}{'trig':>6}{'8B':>8}{'casc':>8}{'72B':>8}"
          f"{'SC@2':>8}{'SC@3':>8}   {'casc-8B [95% CI]':>26}{'casc-SC3':>10}{'casc-72B':>10}")
    out = {}
    for label, sp, bp, s2p, s3p in SETS:
        S, B = load(sp), load(bp)
        ids = [i for i in S if i in B and not S[i].get("pipeline_error")
               and not B[i].get("pipeline_error")]
        n = len(ids)
        small = {i: score(extract(S[i].get("ExtractResult")), golds(S[i])) for i in ids}
        big = {i: score(extract(B[i].get("ExtractResult")), golds(B[i])) for i in ids}
        casc = {i: (big[i] if is_missing(S[i].get("ExtractResult")) else small[i]) for i in ids}
        base = sum(small.values()) / n
        c = sum(casc.values()) / n
        b = sum(big.values()) / n

        sc2, sc3 = load(s2p), load(s3p)
        common = [i for i in ids if i in sc2 and i in sc3]
        sc_acc = {}
        for k in (2, 3):
            tot = 0.0
            for i in common:
                preds = [extract(S[i].get("ExtractResult"))]
                if k >= 2:
                    preds.append(extract(sc2[i].get("ExtractResult")))
                if k >= 3:
                    preds.append(extract(sc3[i].get("ExtractResult")))
                tot += score(vote(preds), golds(S[i]))
            sc_acc[k] = tot / len(common)

        diffs = [(cluster(S[i]), casc[i] - small[i]) for i in ids]
        lo, hi = boot(diffs)
        d_sc = sum(casc[i] - score(vote([extract(S[i].get("ExtractResult")),
                                         extract(sc2[i].get("ExtractResult")),
                                         extract(sc3[i].get("ExtractResult"))]), golds(S[i]))
                    for i in common) / len(common)
        print(f"{label:<12}{n:>5}{sum(1 for i in ids if is_missing(S[i].get('ExtractResult'))):>6}"
              f"{base*100:>8.2f}{c*100:>8.2f}{b*100:>8.2f}"
              f"{sc_acc[2]*100:>8.2f}{sc_acc[3]*100:>8.2f}   "
              f"{(c-base)*100:+8.2f} [{lo:+6.2f},{hi:+6.2f}]  "
              f"{d_sc*100:+9.2f}{(c-b)*100:+10.2f}")
        out[label] = {"n": n, "small": base, "cascade": c, "big": b,
                      "sc2": sc_acc[2], "sc3": sc_acc[3],
                      "gain_ci": [lo / 100, hi / 100], "n_sc": len(common)}

    json.dump(out, open(os.path.join(HERE, "s2_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nwritten s2_results.json")
    print("note: SC@k columns are computed on the items covered by all k samples, "
          "so they can differ slightly from the full-set SC numbers")


if __name__ == "__main__":
    main()
