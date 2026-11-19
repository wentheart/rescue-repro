"""Rescue-efficiency criterion, derived and verified numerically.

Model
-----
Trigger: a fixed escalation set E (rate rho), chosen by the first pass returning nothing.
Baseline (escalate nothing): every item in E scores 0, so

    acc0 = (1 - rho) * q_S                      cost 0

Rescue action A: mean score q_A on E, spending tau_A strong-model tokens per escalated item,
measured in units of "one full rerun on one item". Then

    cost(A) = rho * tau_A
    acc(A)  = (1 - rho) * q_S + rho * q_A

Every action's point lies on a ray from (0, acc0) with slope

    e_A = q_A / tau_A          <-- rescue efficiency: quality bought per unit of budget

Proposition 1. For a fixed trigger rate rho, if e_A > e_B then A's operating point lies
strictly above the chord from the baseline to B's operating point.

Proof. At A's cost the chord to B reaches acc0 + rho*q_B*(tau_A/tau_B). A's point is
acc0 + rho*q_A. The former is smaller iff q_A/tau_A > q_B/tau_B.                 []

Two-tier model: tier 1 answers a fraction alpha of E with mean score q1 (over the accepted
items), the rest fall back to the full rerun (score q_f, cost 1 per item):

    q_C  = q_f + alpha * (q1 - q_f)
    tau_C = tau_1 + (1 - alpha)

Proposition 2. Tiering beats the pure full rerun iff

    alpha > alpha* = q_f * tau_1 / q1

so the crossover depends only on the full rerun's quality and cost and on how good the cheap
tier is on the items it accepts -- not on the trigger size.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "_g4_scoring_20261102"))
from scoring import extract, score, is_missing  # noqa: E402
from recompute import load  # noqa: E402

SETS = {"TAT-QA": ("TatQa_b8b-full_123_", "TatQa_b72-full_123_"),
        "CALTAB151": ("CalTab151_ct8b_123_", "CalTab151_b72-v2_123_"),
        "TableBench": ("TableBench_tb8b_123_", "TableBench_b72-v2_123_")}


def toks(d, kind="big"):
    if kind == "big":
        return (d.get("_ptok") or 0) + (d.get("_ctok") or 0)
    return (d.get("prompt") or 0) + (d.get("completion") or 0)


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
accepted = lambda rec: rec.get("status") == "ok" and not is_missing(rec.get("answer"))


def golds(r):
    a = r.get("Answer") or []
    return a if isinstance(a, list) else [a]


out = {}
for ds, (sp, bp) in SETS.items():
    S, B = load(sp), load(bp)
    ids = [i for i in S if i in B and not S[i].get("pipeline_error")
           and not B[i].get("pipeline_error")]
    N = len(ids)
    E = [i for i in ids if is_missing(S[i].get("ExtractResult"))]
    nonE = [i for i in ids if not is_missing(S[i].get("ExtractResult"))]
    rho = len(E) / N
    q_S = sum(score(extract(S[i].get("ExtractResult")), golds(S[i])) for i in nonE) / len(nonE)
    q_full = sum(score(extract(B[i].get("ExtractResult")), golds(B[i])) for i in E) / len(E)
    tok_item = sum(toks(B[i]) for i in ids) / N

    mine = {k: r for k, r in rows.items() if k.startswith(ds + "|")}
    keys = list(mine)
    ok = [k for k in keys if accepted(mine[k]["rawpot"])]
    alpha = len(ok) / len(keys)
    q1 = sum(mine[k]["rawpot"]["score"] for k in ok) / len(ok)      # over accepted items
    q_tier = sum((mine[k]["rawpot"]["score"] if accepted(mine[k]["rawpot"])
                  else mine[k]["full_score"]) for k in keys) / len(keys)
    q_raw = sum(mine[k]["rawpot"]["score"] for k in keys) / len(keys)

    tau_full = (sum(mine[k]["full_tok"]["prompt"] + mine[k]["full_tok"]["completion"]
                    for k in keys) / len(keys)) / tok_item
    tau_raw = (sum(toks(mine[k]["rawpot"], "raw") for k in keys) / len(keys)) / tok_item
    tau_tier = (sum(toks(mine[k]["rawpot"], "raw") +
                    (0 if accepted(mine[k]["rawpot"])
                     else mine[k]["full_tok"]["prompt"] + mine[k]["full_tok"]["completion"])
                    for k in keys) / len(keys)) / tok_item

    e_full, e_tier, e_raw = q_full / tau_full, q_tier / tau_tier, q_raw / tau_raw
    acc0 = (1 - rho) * q_S
    out[ds] = dict(rho=rho, q_S=q_S, q_full=q_full, q_tier=q_tier, q_raw=q_raw, q1=q1,
                   alpha=alpha, tau_full=tau_full, tau_tier=tau_tier, tau_raw=tau_raw,
                   e_full=e_full, e_tier=e_tier, e_raw=e_raw, acc0=acc0)

    print(f"\n=== {ds} ===")
    print(f"  trigger rate rho {rho:.3f} | baseline acc {acc0:.4f} | "
          f"q_S(non-escalated) {q_S:.4f}")
    print(f"  {'action':<12}{'q_A':>8}{'tau_A':>8}{'e_A':>9}")
    print(f"  {'full':<12}{q_full:>8.4f}{tau_full:>8.4f}{e_full:>9.4f}")
    print(f"  {'two-tier':<12}{q_tier:>8.4f}{tau_tier:>8.4f}{e_tier:>9.4f}")
    print(f"  {'tier-1 only':<12}{q_raw:>8.4f}{tau_raw:>8.4f}{e_raw:>9.4f}")
    print(f"  efficiency ratio two-tier / full = {e_tier / e_full:.2f}x")
    print(f"  acceptance alpha {alpha:.3f}, tier-1 quality on accepted q1 {q1:.4f}")

print("\n=== Proposition 1: dominance over the naive-cascade chord ===")
print(f"{'dataset':<12}{'tier acc':>10}{'chord':>9}{'margin':>9}{'verdict':>8}")
for ds, v in out.items():
    c_tier = v["rho"] * v["tau_tier"]
    acc_tier = v["acc0"] + v["rho"] * v["q_tier"]
    chord = v["acc0"] + (v["rho"] * v["q_full"]) / (v["rho"] * v["tau_full"]) * c_tier
    print(f"{ds:<12}{acc_tier:>10.4f}{chord:>9.4f}{acc_tier - chord:>+9.4f}"
          f"{'ABOVE' if acc_tier > chord else 'BELOW':>8}")

print("\n=== Proposition 2: crossover acceptance rate ===")
print(f"{'dataset':<12}{'q_f':>8}{'tau_1':>8}{'q1':>8}{'alpha*':>9}{'alpha':>8}{'ratio':>8}")
for ds, v in out.items():
    a_star = v["q_full"] * v["tau_raw"] / v["q1"]
    v["alpha_star"] = a_star
    print(f"{ds:<12}{v['q_full']:>8.4f}{v['tau_raw']:>8.4f}{v['q1']:>8.4f}"
          f"{a_star:>9.3f}{v['alpha']:>8.3f}{v['alpha'] / a_star:>7.1f}x")

print("\n=== numeric check of Proposition 2 (sweep alpha, rebuild the tier point) ===")
for ds, v in out.items():
    cross, prev = None, None
    for i in range(1, 2000):
        al = i / 2000
        q_c = v["q_full"] + al * (v["q1"] - v["q_full"])
        tau_c = v["tau_raw"] + (1 - al) * v["tau_full"]
        diff = q_c / tau_c - v["q_full"] / v["tau_full"]
        if prev is not None and (prev > 0) != (diff > 0):
            cross = al
        prev = diff
    print(f"  {ds:<12} numeric alpha* = {cross:.4f}   formula = {v['alpha_star']:.4f}   "
          f"gap {abs(cross - v['alpha_star']):.2e}")

print("\n=== where the efficiency comes from: cost per point of quality on E ===")
for ds, v in out.items():
    print(f"  {ds:<12} full {v['tau_full'] / v['q_full']:.4f} | "
          f"two-tier {v['tau_tier'] / v['q_tier']:.4f} | "
          f"tier-1 {v['tau_raw'] / v['q_raw']:.4f}   (budget per unit of quality on E)")

json.dump(out, open(os.path.join(HERE, "efficiency.json"), "w", encoding="utf-8"), indent=1)
print("\n-> efficiency.json")
