"""Regenerate tab_vendor.tex with the block layout: one block per dataset, active-parameter column
instead of the sample count."""
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
# name, vendor, directory prefix, active parameters in billions
BACKBONES = [("Qwen3-8B", "Alibaba", None, "8.2"),
             ("GLM-4-9B", "Zhipu", "glm", "9.0"),
             ("Hunyuan-A13B", "Tencent", "hunyuan", "13.0"),
             ("Ling-mini", "Ant", "ling", "1.4")]
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
    big_tok = sum(toks(B72[i]) for i in B72)          # full-dataset denominator
    esc = {i for i in ids if is_missing(R[i].get("ExtractResult"))}
    base = sum(score(extract(R[i].get("ExtractResult")), golds(B72[i])) for i in ids) / n * 100
    acc = t1 = full = 0.0
    miss_after = 0
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
            full += toks(B72[i])
            acc += score(extract(B72[i].get("ExtractResult")), golds(B72[i]))
            if is_missing(B72[i].get("ExtractResult")):
                miss_after += 1
    return dict(n=n, miss_before=len(esc) / n * 100, miss_after=miss_after / n * 100,
                base=base, acc=acc / n * 100, delta=acc / n * 100 - base,
                cost=(t1 + full) / max(1, big_tok) * 100)


rows = []
for ds in DATASETS:
    rows.append(r"\midrule" if ds != "TAT-QA" else "")
    rows.append(r"\multicolumn{6}{l}{\emph{%s}} \\" % ds)
    for name, vendor, prefix, params in BACKBONES:
        m = measure(ds, prefix)
        rows.append(f"{name} ({vendor}) & {params} & "
                    f"{m['miss_before']:.1f} $\\rightarrow$ {m['miss_after']:.1f} & "
                    f"{m['base']:.2f} $\\rightarrow$ {m['acc']:.2f} & "
                    f"{m['delta']:+.2f} & {m['cost']:.1f} \\\\")

tab = "\n".join([
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Cross-vendor replication. \emph{Unans.} is the unanswered rate before %s after the rescue,",
])
tab = tab % r"$\rightarrow$"
tab += "\n" + "\n".join([
    r"in percent; \emph{Budget} is the strong-model token budget as a fraction of a full run on all",
    r"items. Parameter counts are the approximate sizes reported by the vendors, in billions of",
    r"parameters.}",
    r"\label{tab:vendor}",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{llcccc}",
    r"\toprule",
    r"Backbone (vendor) & Active params.\,(B) & Unans.\ (\%) & S$^2$ & $\Delta$S$^2$ & Budget (\%) \\",
] + rows + [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
])
open(os.path.join(OUT, "tab_vendor.tex"), "w", encoding="utf-8").write(tab + "\n")
print("written")
