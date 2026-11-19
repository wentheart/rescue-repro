"""Two derived numbers used in the docs, computed instead of hand-estimated:

1. price crossover p between our cascade and the best consistency operating point (tau=0.33);
2. the consistency cascade's accuracy *at our cost*, linearly interpolated between its
   neighbouring tau points, and the difference against us.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "cost_efficiency.json"), encoding="utf-8"))


def interp(points, x):
    """points: [(cost, acc)] sorted by cost; linear interpolation, clamped at the ends."""
    pts = sorted(points)
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


print(f"{'dataset':<12}{'our acc':>9}{'our cost':>9}{'cons@our cost':>14}{'diff':>8}"
      f"{'crossover p':>13}")
for ds, r in d.items():
    ts = r["ts_tl"]
    o, cons = r["ours"], r["cons"]
    pts = [(v["big_cost"], v["acc"]) for v in cons.values()]
    at = interp(pts, o["big_cost"])
    c33 = cons["0.33"]
    num = o["acc"] * c33["big_cost"] - c33["acc"] * o["big_cost"]
    den = c33["acc"] * ts - 3 * o["acc"] * ts
    p = num / den if den else float("nan")
    print(f"{ds:<12}{o['acc'] * 100:>9.2f}{o['big_cost'] * 100:>9.1f}"
          f"{at * 100:>14.2f}{(o['acc'] - at) * 100:>+8.2f}{p:>13.3f}")

print("\n'diff' = our accuracy minus the consistency cascade interpolated to OUR cost.")
print("'crossover p' = small-model price ratio above which our 1-run method is cheaper.")
