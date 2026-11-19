# Escalation driver for the two-tier rescue.
# Requires the TabDSR / TableMaster runtime (third-party, not redistributed here) and a
# model-serving adapter; no credential is included in this release. The analysis scripts in
# ../analysis run without either, on the files in ../results.
"""Rescue-action runner (pilot and full).

Fixes from review:
  1. the table is no longer spliced into Python source as JSON (which wrote `null` and made the
     driver fail before the model's code ever ran); it is embedded as a *string literal* and
     parsed with json.loads at run time;
  2. full provenance per item: raw model response, extracted code, exception text, tokens;
  3. token accounting is exact (sequential execution, per-call deltas);
  4. every record carries the fields needed to re-score offline without re-calling the API.

Actions
  rawpot : 72B writes Python against the raw rows            (1 call)
  plan   : 72B emits a restricted JSON plan, CPU executes    (1 call)

The escalation trigger is unchanged: these actions replace only the RESCUE step.
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
REPO = os.path.join(ROOT, "repos", "TabDSR-main")
sys.path.insert(0, os.path.join(ROOT, "_pylibs"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(ROOT, "repos", "TableMaster-main"))
sys.path.insert(0, os.path.join(ROOT, "_g4_scoring_20261102"))
sys.path.insert(0, os.path.join(ROOT, "tabdsr_exp"))
sys.path.insert(0, HERE)

import sf_api  # noqa: E402
from scoring import extract, score, is_missing  # noqa: E402
from recompute import load  # noqa: E402
import run_shard  # noqa: E402
import rescue_lib as RL  # noqa: E402

MODEL = "Qwen/Qwen2.5-72B-Instruct"
SETS = {"TAT-QA": ("TatQa_b8b-full_123_", "TatQa_b72-full_123_"),
        "CALTAB151": ("CalTab151_ct8b_123_", "CalTab151_b72-v2_123_"),
        "TableBench": ("TableBench_tb8b_123_", "TableBench_b72-v2_123_")}


def stat():
    return dict(sf_api.STATS)


def delta(a, b):
    return {"prompt": b["prompt_tokens"] - a["prompt_tokens"],
            "completion": b["completion_tokens"] - a["completion_tokens"]}


def extract_code(resp):
    """Pull the python code block out of a response.

    Preference order (per review): a block explicitly labelled python/py, then an untagged
    block, then the whole reply. Several python candidates -> keep the first.
    """
    t = str(resp or "")
    blocks = re.findall(r"```([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", t, re.S)
    labelled = [b for lang, b in blocks if lang.lower() in ("python", "py")]
    if labelled:
        return labelled[0].strip()
    untagged = [b for lang, b in blocks if lang == ""]
    if untagged:
        return untagged[0].strip()
    return t.strip()


def build_program(code, rows):
    """Embed the table as a STRING literal and parse it at run time.

    Two earlier failures are fixed here:
      * splicing json.dumps(rows) into the source wrote JSON `null`, which is not a Python
        name, so initialisation died before the model's code ever ran;
      * double-encoding with ensure_ascii=False could split non-BMP characters (emoji) into
        surrogate pairs. repr() of an ASCII-only JSON string is a safe Python literal.
    """
    payload = repr(json.dumps(rows, ensure_ascii=True))
    return ("import json\n"
            f"table = json.loads({payload})\n"
            "_TABLE_LEN = len(table)\n"
            f"{code}\n")


def run_rawpot(question, table):
    snap = stat()
    prompt = RL.render(RL.USER_RAW, table, question)
    rec = {"prompt_text_len": len(prompt)}
    try:
        resp = sf_api.get_openai_llm_response(prompt, model=MODEL, temperature=0.1)
    except Exception as e:  # noqa: BLE001
        rec.update({"raw": "", "code": "", "answer": "", "err": f"api:{type(e).__name__}:{e}",
                    "status": "api_error", **delta(snap, stat())})
        return rec
    rec.update(delta(snap, stat()))
    code = extract_code(resp)
    rec.update({"raw": resp, "code": code})
    if not code:
        rec.update({"answer": "", "err": "empty response", "status": "empty_response"})
        return rec
    try:
        out = run_shard.simple_exec(build_program(code, table.get("data") or []))
    except Exception as e:  # noqa: BLE001
        rec.update({"answer": "", "err": str(e)[:400], "status": "exec_error"})
        return rec
    out = (out or "").strip()
    if not out:
        rec.update({"answer": "", "err": "no stdout", "status": "empty_output"})
        return rec
    rec.update({"answer": out.splitlines()[-1].strip(), "err": "", "status": "ok"})
    return rec


def run_plan(question, table):
    snap = stat()
    prompt = RL.render(RL.USER_PLAN, table, question)
    rec = {"prompt_text_len": len(prompt)}
    try:
        resp = sf_api.get_openai_llm_response(prompt, model=MODEL, temperature=0.1)
    except Exception as e:  # noqa: BLE001
        rec.update({"raw": "", "answer": "", "err": f"api:{type(e).__name__}:{e}",
                    "status": "api_error", **delta(snap, stat())})
        return rec
    rec.update(delta(snap, stat()))
    rec["raw"] = resp
    try:
        ans, detail = RL.run_plan(resp, table)
    except Exception as e:  # noqa: BLE001
        rec.update({"answer": "", "err": str(e)[:400], "status": "plan_error"})
        return rec
    rec.update({"answer": ans, "err": "", "status": "ok"})
    return rec


def build_jobs(population):
    jobs = []
    for ds, (sp, bp) in SETS.items():
        S, B = load(sp), load(bp)
        ids = [i for i in S if i in B and not S[i].get("pipeline_error")
               and not B[i].get("pipeline_error") and isinstance(S[i].get("Table"), dict)]
        if population == "missing":
            ids = [i for i in ids if is_missing(S[i].get("ExtractResult"))]
        for i in ids:
            g = S[i].get("Answer")
            g = g if isinstance(g, list) else [g]
            jobs.append({"key": f"{ds}|{i}", "dataset": ds, "id": i,
                         "question": S[i]["Question"], "table": S[i]["Table"], "gold": g,
                         "full_score": score(extract(B[i].get("ExtractResult")), g),
                         "full_tok": {"prompt": B[i].get("_ptok") or 0,
                                      "completion": B[i].get("_ctok") or 0}})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="missing", choices=["missing", "all"])
    ap.add_argument("--actions", default="rawpot,plan")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sf_api.GLOBAL_MODEL = MODEL
    os.environ.setdefault("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")
    actions = [a.strip() for a in args.actions.split(",") if a.strip()]
    out = args.out or os.path.join(HERE, f"rescue_{args.population}.jsonl")
    if not os.path.isabs(out):
        # run_shard chdir()s into the repo at import time, so anchor relative paths
        out = os.path.join(HERE, out)

    done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8", errors="replace"):
            try:
                done.add(json.loads(line)["key"])
            except Exception:  # noqa: BLE001
                pass
    jobs = build_jobs(args.population)
    if args.limit:
        jobs = jobs[:args.limit]
    todo = [j for j in jobs if j["key"] not in done]
    print(f"population={args.population} total={len(jobs)} done={len(done)} "
          f"todo={len(todo)} actions={actions} calls={len(todo) * len(actions)}", flush=True)

    f = open(out, "a", encoding="utf-8")
    t0 = time.time()
    n = 0
    for j in todo:
        rec = {k: j[k] for k in ("key", "dataset", "id", "question", "gold", "full_score",
                                 "full_tok")}
        for act in actions:
            r = run_rawpot(j["question"], j["table"]) if act == "rawpot" \
                else run_plan(j["question"], j["table"])
            r["score"] = score(extract(r["answer"]), j["gold"])
            rec[act] = r
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        n += 1
        if n % 10 == 0:
            print(f"  {n}/{len(todo)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    f.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
