"""Emit two complete LaTeX tables (environment included) from the frozen records.

Writing whole table environments, rather than row bodies pulled in with \\input inside a tabular,
avoids TeX's "Misplaced \\noalign" when a file boundary falls between rows.
"""
import json
import os
import sys

ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
G8 = os.path.join(ROOT, "_g8_plan_rescue_20261109")
G9 = os.path.join(ROOT, "_g9_crossmodel_20261112")
OUT = os.path.join(ROOT, "paper")
sys.path.insert(0, os.path.join(ROOT, "_g4_scoring_20261102"))
from scoring import extract, score, is_missing  # noqa: E402
from recompute import load  # noqa: E402

TAG = {"TAT-QA": "TAT-QA", "CALTAB151": "CALTAB151", "TableBench": "TableBench"}
DSFILE = {"TAT-QA": "TatQa", "CALTAB151": "CalTab151", "TableBench": "TableBench"}
PREF = {"TAT-QA": ("TatQa_b8b-full_123_", "TatQa_b72-full_123_"),
        "CALTAB151": ("CalTab151_ct8b_123_", "CalTab151_b72-v2_123_"),
        "TableBench": ("TableBench_tb8b_123_", "TableBench_b72-v2_123_")}
BACKBONES = [("Qwen3-8B", "Alibaba", None),
             ("GLM-4-9B", "Zhipu", "glm"),
             ("Hunyuan-A13B", "Tencent", "hunyuan"),
             ("Ling-mini", "Ant", "ling")]
DATASETS = ("TAT-QA", "CALTAB151", "TableBench")

rescue = {}
for p in (os.path.join(G8, "rescue_missing.jsonl"), os.path.join(G9, "rescue_union.jsonl")):
    if os.path.exists(p):
        for line in open(p, encoding="utf-8", errors="replace"):
            line = line.strip()
            if line:
                rec = json.loads(line)
                rescue[rec["key"]] = rec["rawpot"]


def golds(r):
    a = r.get("Answer") or []
    return a if isinstance(a, list) else [a]


def toks(d):
    return (d.get("_ptok") or 0) + (d.get("_ctok") or 0)


def backbone_rows(ds, prefix):
    if prefix is None:
        return load(PREF[ds][0])
    rows = {}
    stem = DSFILE[ds]
    d = os.path.join(G9, "out_full")
    for f in os.listdir(d):
        if f.startswith(f"{stem}_{prefix}_"):
            for r in json.load(open(os.path.join(d, f), encoding="utf-8")):
                if not r.get("pipeline_error"):
                    rows[str(r.get("id"))] = r
    return rows


def measure(ds, prefix):
    R, B72 = backbone_rows(ds, prefix), load(PREF[ds][1])
    ids = [i for i in R if i in B72 and not R[i].get("pipeline_error")]
    n = len(ids)
    big_tok = sum(toks(B72[i]) for i in B72)   # all items of the dataset, as in the compilation
    esc = {i for i in ids if is_missing(R[i].get("ExtractResult"))}
    base = sum(score(extract(R[i].get("ExtractResult")), golds(B72[i])) for i in ids) / n * 100

    acc = t1 = full = 0.0
    miss_after = full_items = 0
    for i in ids:
        s = score(extract(R[i].get("ExtractResult")), golds(B72[i]))
        if i not in esc:
            acc += s
            continue
        rec = rescue.get(f"{TAG[ds]}|{i}")
        if rec:
            t1 += (rec.get("prompt") or 0) + (rec.get("completion") or 0)
        if rec and rec.get("status") == "ok" and not is_missing(rec.get("answer")):
            acc += rec["score"]
        else:
            full_items += 1
            full += toks(B72[i])
            acc += score(extract(B72[i].get("ExtractResult")), golds(B72[i]))
            if is_missing(B72[i].get("ExtractResult")):
                miss_after += 1
    return dict(n=n, miss_before=len(esc) / n * 100, miss_after=miss_after / n * 100,
                base=base, acc=acc / n * 100, delta=acc / n * 100 - base,
                cost=(t1 + full) / max(1, big_tok) * 100,
                full_items=full_items, t1=t1, full=full, big_tok=big_tok)


# ---------------------------------------------------------------- Table 3
rows3 = []
for ds in DATASETS:
    for name, vendor, prefix in BACKBONES:
        m = measure(ds, prefix)
        rows3.append(f"{name} ({vendor}) & {ds} & {m['n']} & "
                     f"{m['miss_before']:.1f} $\\rightarrow$ {m['miss_after']:.1f} & "
                     f"{m['base']:.2f} $\\rightarrow$ {m['acc']:.2f} & "
                     f"{m['delta']:+.2f} & {m['cost']:.1f} \\\\")

tab3 = "\n".join([
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Cross-vendor replication on all three benchmarks. \emph{Unans.} is the unanswered",
    r"rate before $\rightarrow$ after the rescue, in percent. \emph{Budget} is the strong-model",
    r"token budget as a fraction of a full strong-model run on all items.}",
    r"\label{tab:vendor}",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{llccccc}",
    r"\toprule",
    r"Backbone (vendor) & Dataset & $n$ & Unans. & S$^2$ & $\Delta$S$^2$ & Budget (\%) \\",
    r"\midrule",
] + rows3 + [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
])
open(os.path.join(OUT, "tab_vendor.tex"), "w", encoding="utf-8").write(tab3 + "\n")

# ---------------------------------------------------------------- Table 4
rows4 = []
for ds in DATASETS:
    m = measure(ds, None)
    S, B72 = load(PREF[ds][0]), load(PREF[ds][1])
    ids = [i for i in S if i in B72 and not S[i].get("pipeline_error")]
    n = len(ids)
    esc = {i for i in ids if is_missing(S[i].get("ExtractResult"))}

    acc, miss, t1tok = 0.0, 0, 0.0
    for i in ids:
        s = score(extract(S[i].get("ExtractResult")), golds(B72[i]))
        if i not in esc:
            acc += s
            continue
        rec = rescue.get(f"{TAG[ds]}|{i}")
        if rec:
            t1tok += (rec.get("prompt") or 0) + (rec.get("completion") or 0)
        if rec and rec.get("status") == "ok" and not is_missing(rec.get("answer")):
            acc += rec["score"]
        else:
            miss += 1
    row_tier1 = dict(acc=acc / n * 100, miss=miss / n * 100, full_items=0, t1=t1tok,
                     full=0.0, cost=t1tok / max(1, m["big_tok"]) * 100)

    acc2, full, miss2 = 0.0, 0.0, 0
    for i in ids:
        s = score(extract(S[i].get("ExtractResult")), golds(B72[i]))
        if i not in esc:
            acc2 += s
            continue
        full += toks(B72[i])
        acc2 += score(extract(B72[i].get("ExtractResult")), golds(B72[i]))
        if is_missing(B72[i].get("ExtractResult")):
            miss2 += 1
    row_full = dict(acc=acc2 / n * 100, miss=miss2 / n * 100, full_items=len(esc), t1=0.0,
                    full=full, cost=full / max(1, m["big_tok"]) * 100)

    row_two = dict(acc=m["acc"], miss=m["miss_after"], full_items=m["full_items"],
                   t1=m["t1"], full=m["full"], cost=m["cost"])

    for label, r in (("Tier-1 only", row_tier1), ("Two-tier", row_two), ("Full rerun", row_full)):
        rows4.append(f"{ds} & {label} & {r['acc']:.2f} & {r['miss']:.1f} & {r['full_items']} & "
                     f"{r['t1'] / 1000:.0f}k & {r['full'] / 1000:.0f}k & {r['cost']:.1f} \\\\")

tab4 = "\n".join([
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Rescue actions on the same escalation set, with cost decomposition. \emph{Final} is",
    r"the unanswered rate after the action. \emph{FULL} is the number of items that fell back to a",
    r"full rerun. Token columns are strong-model tokens in thousands. \emph{Budget} is the share of",
    r"a full strong-model run over all items.}",
    r"\label{tab:ablation}",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{llcccccc}",
    r"\toprule",
    r"Dataset & Action & S$^2$ & Final & FULL & Tier-1 tok. & Rerun tok. & Budget (\%) \\",
    r"\midrule",
] + rows4 + [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
])
open(os.path.join(OUT, "tab_ablation.tex"), "w", encoding="utf-8").write(tab4 + "\n")

print("ok")


