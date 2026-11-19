"""Consistency-triggered cascade, built offline from the three existing small-model samples.

This is the baseline the third review asked for (its §6.2): the ICLR-2024 style cascade
samples the weak model K times, and *escalates when the samples disagree*. Our earlier
SC@3 is NOT that baseline -- it has no escalation stage. Here both cascades are compared on
the same items, the same S2 metric and the same big-model outputs.

Definitions fixed before running:
  * K = 3 small-model pipeline runs per item (base, sc2, sc3);
  * agreement s = (votes for the majority answer) / K, where answers are compared with the
    same vote_key as everywhere else; a missing answer never forms a majority;
  * if the majority answer is missing, or s < tau -> escalate to the big model;
  * cost = escalation_rate x big-model tokens on those items, relative to always-big.

Usage: python consistency_cascade.py
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
OUT = os.path.join(ROOT, "tabdsr_exp", "out")
sys.path.insert(0, HERE)
from scoring import extract, score, is_missing, vote_key  # noqa: E402

SETS = [
    ("TAT-QA", "TatQa_b8b-full_123_", "TatQa_b72-full_123_", "TatQa_sc2_123_", "TatQa_sc3_123_"),
    ("CALTAB151", "CalTab151_ct8b_123_", "CalTab151_b72-v2_123_", "CalTab151_sc2_123_",
     "CalTab151_sc3_123_"),
    ("TableBench", "TableBench_tb8b_123_", "TableBench_b72-v2_123_", "TableBench_sc2_123_",
     "TableBench_sc3_123_"),
]
PRICE = 4.13 / 1e6


def load(prefix):
    rows = {}
    for f in sorted(os.listdir(OUT)):
        if not (f.startswith(prefix) and f.endswith(".json")):
            continue
        try:
            for r in json.load(open(os.path.join(OUT, f), encoding="utf-8")):
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


def toks(r):
    return (r.get("_ptok") or 0) + (r.get("_ctok") or 0)


def majority(preds):
    keys = [vote_key(p) for p in preds]
    live = [k for k in keys if k]
    if not live:
        return None, 0.0
    c = Counter(live)
    top = max(c.values())
    return c.most_common(1)[0][0], top / len(preds)


def main():
    out = {}
    print(f"{'dataset':<12}{'tau':>6}{'escalate':>10}{'acc':>8}{'cost/72B':>10}   "
          f"{'missing-trigger cascade':>26}")
    for label, sp, bp, s2p, s3p in SETS:
        S, B = load(sp), load(bp)
        sc2, sc3 = load(s2p), load(s3p)
        ids = [i for i in S if i in B and i in sc2 and i in sc3
               and not S[i].get("pipeline_error") and not B[i].get("pipeline_error")]
        n = len(ids)
        t_big_all = sum(toks(B[i]) for i in ids)
        big = {i: score(extract(B[i].get("ExtractResult")), golds(B[i])) for i in ids}
        small1 = {i: score(extract(S[i].get("ExtractResult")), golds(S[i])) for i in ids}

        # our cascade on the same ids: escalate when the first pass has no answer
        miss = [i for i in ids if is_missing(S[i].get("ExtractResult"))]
        ours_acc = sum(big[i] if i in miss else small1[i] for i in ids) / n
        ours_cost = sum(toks(B[i]) for i in miss) / max(1, t_big_all)

        # consistency cascade over K = 3
        preds = {i: [extract(S[i].get("ExtractResult")), extract(sc2[i].get("ExtractResult")),
                     extract(sc3[i].get("ExtractResult"))] for i in ids}
        maj = {}
        for i in ids:
            key, s = majority(preds[i])
            maj[i] = (key, s)
        # the answer behind the majority key, for scoring
        def maj_pred(i):
            key, _ = maj[i]
            for p in preds[i]:
                if vote_key(p) == key:
                    return p
            return ("missing", None, "")

        rows = []
        for tau in (1.0, 2 / 3, 1 / 3):
            esc = [i for i in ids if maj[i][0] is None or maj[i][1] < tau - 1e-9]
            acc = sum(big[i] if i in esc else score(maj_pred(i), golds(S[i])) for i in ids) / n
            cost = sum(toks(B[i]) for i in esc) / max(1, t_big_all)
            rows.append((tau, len(esc) / n, acc, cost))
            print(f"{label:<12}{tau:>6.2f}{len(esc) / n:>10.1%}{acc * 100:>8.2f}{cost:>10.1%}"
                  f"   ours {ours_acc * 100:>6.2f} @ {ours_cost:.1%}"
                  f"  ({'ours better' if ours_acc >= acc else 'consistency better'})")
        out[label] = {"n": n, "ours": {"acc": ours_acc, "cost": ours_cost},
                      "consistency": [{"tau": t, "esc": e, "acc": a, "cost": c} for t, e, a, c in rows]}
        print()
    json.dump(out, open(os.path.join(HERE, "consistency_cascade.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
