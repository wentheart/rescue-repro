"""Tests for S2 extraction/scoring and voting (v4).

Every case asserts an EXACT score. Covers both directions: correct answers must not be
rejected, and near-miss answers must not be credited.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scoring import extract, score, vote, vote_key  # noqa: E402

CASES = [
    # --- must be rejected (false-positive traps) -----------------------------
    (["0"], "-", 0.0),
    (["10"], "1", 0.0),
    (["-12"], "12", 0.0),
    (["12"], "123", 0.0),
    (["2019"], "The question asks about 2019", 0.0),
    (["2019"], "2019.2", 0.0),
    (["2019"], "Profit in 2019 is 1,234", 0.0),          # thousands separator
    (["Paris"], "Sub-query 1: Paris\nSub-query 2: Final answer: London", 0.0),
    (["yes"], "Final answer: No, not yes.", 0.0),        # negation
    (["10"], "Sub-query 1: 10; Sub-query 2: Final answer is 20", 0.0),
    (["0"], "0    8.98 dtype: float64", 0.0),
    # --- must be credited ----------------------------------------------------
    (["2019"], "2019", 1.0),
    (["2019"], "final answer: 2019", 1.0),
    (["8.98"], "0    8.98 dtype: float64", 1.0),
    (["-9.81"], "-9.81", 1.0),
    (["283.8"], "Sub-query 1: The maximum group operating profit in 2019 was 283.8 (€m)", 1.0),
    (["3432"], "Total net interest revenue in 2008 is 3432", 1.0),
    (["1,234"], "answer: 1,234", 1.0),                   # thousands separator parsed
    (["3500000"], "3.5 million", 1.0),                   # unit normalisation
    (["3.5 million"], "3.5 million", 1.0),               # symmetric gold-side unit
    (["2000000", "3000"], "2 million, 3 thousand", 1.0), # per-answer scales
    (["yes"], "Final answer: yesterday", 0.0),           # word boundary, not substring
    # numeric reference inside a text-typed answer: only labelled or leading positions count
    # number-typed prediction: only the trailing value counts, never a mid-line number
    (["71"], "Total podiums: 71 Total wins: 37", 0.0),
    (["71", "higher"], "Total podiums: 71 and podiums are higher than wins", 1.0),
    (["800 m"], "800 m", 1.0),
    (["2019-2020"], "2019-2020", 1.0),
    (["2020"], "Final answer: 2020", 1.0),
    (["Annual Incentive Plan", "Performance Share Plan"],
     "Sub-query 1: <StringArray> ['Annual Incentive Plan', 'Performance Share Plan']", 1.0),
    # --- multi-answer and partial credit ------------------------------------
    (["2019", "2018"], "2019, 2018", 1.0),
    (["2019", "2018"], "2018, 2019", 1.0),
    (["2019", "2018"], "2019", 0.5),
    (["2019", "2018"], "2019, 2018, 2017", 1.0),         # extra answers do not subtract
]

VOTE_CASES = [
    ("id=826", ["0.04", "0    8.98 dtype: float64", "8.98"], ["8.98"], 1.0),
    ("order", ["2019, 2018", "2018, 2019", "2019, 2018"], ["2019", "2018"], 1.0),
    ("majority", ["5", "7", "7"], ["7"], 1.0),
    ("scaled", ["3.5 million", "3500000", "3.5 million"], ["3500000"], 1.0),
]


def main():
    bad = []
    print("== scoring ==")
    for gold, out, expect in CASES:
        got = score(extract(out), gold)
        ok = abs(got - expect) < 1e-9
        print(f"  {'PASS' if ok else 'FAIL'}  expect={expect:.2f} got={got:.2f}  "
              f"gold={gold} out={out[:46]!r}")
        if not ok:
            bad.append(("score", gold, out, expect, got))

    print("\n== voting ==")
    for name, samples, gold, expect in VOTE_CASES:
        preds = [extract(s) for s in samples]
        got = score(vote(preds), gold)
        ok = abs(got - expect) < 1e-9
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<8} keys={[vote_key(p) for p in preds]} -> {got:.2f}")
        if not ok:
            bad.append(("vote", name, samples, expect, got))

    total = len(CASES) + len(VOTE_CASES)
    print(f"\n{total - len(bad)}/{total} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
