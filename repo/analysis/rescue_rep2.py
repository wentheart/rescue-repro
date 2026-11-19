"""End-to-end second run: rescue applied to the SECOND first-pass run's escalation set.

The earlier repeat only re-ran the rescue on the first run's escalation set, so it measured the
rescue's own variance. This run uses the second full 8B pipeline run, whose escalation set
differs (287 items instead of 276), so the whole method is re-executed end to end.
"""
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "_g4_scoring_20261102"))
sys.path.insert(0, HERE)

import run_rescue as R  # noqa: E402  (imports the repo, sets paths)
from scoring import is_missing  # noqa: E402
from recompute import load  # noqa: E402

OUT = os.path.join(HERE, "rescue_rep2firstpass.jsonl")
FP = "TatQa_rep2_123_"
BP = "TatQa_b72-full_123_"


def main():
    S, B = load(FP), load(BP)
    jobs = []
    for i, r in S.items():
        if r.get("pipeline_error") or i not in B or B[i].get("pipeline_error"):
            continue
        if not is_missing(r.get("ExtractResult")):
            continue
        if not isinstance(r.get("Table"), dict):
            continue
        g = r.get("Answer")
        g = g if isinstance(g, list) else [g]
        jobs.append({"key": f"TAT-QA|{i}", "dataset": "TAT-QA", "id": i,
                     "question": r["Question"], "table": r["Table"], "gold": g,
                     "full_score": R.score(R.extract(B[i].get("ExtractResult")), g),
                     "full_tok": {"prompt": B[i].get("_ptok") or 0,
                                  "completion": B[i].get("_ctok") or 0}})

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8", errors="replace"):
            try:
                done.add(json.loads(line)["key"])
            except Exception:  # noqa: BLE001
                pass
    todo = [j for j in jobs if j["key"] not in done]
    print(f"{len(jobs)} escalation items in the second run; done {len(done)}; "
          f"todo {len(todo)}", flush=True)

    f = open(OUT, "a", encoding="utf-8")
    t0 = time.time()
    for n, j in enumerate(todo, 1):
        rec = {k: j[k] for k in ("key", "dataset", "id", "question", "gold", "full_score",
                                 "full_tok")}
        r = R.run_rawpot(j["question"], j["table"])
        r["score"] = R.score(R.extract(r["answer"]), j["gold"])
        rec["rawpot"] = r
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        if n % 20 == 0:
            print(f"  {n}/{len(todo)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    f.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
