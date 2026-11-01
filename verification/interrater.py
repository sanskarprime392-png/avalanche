"""Inter-rater agreement between the authors' rule-based pass and an independent second rater.

The second rater worked from rate.html: the same three-criterion rule shown on screen, the chips
presented in a scrambled but reproducible order, and no sight of the first rater's verdicts. This
answers the reviewer's request that part of the verification be performed independently, and it
turns the label quality of the inventory from an assertion into a measured quantity with an
interval attached.

Usage:
    python verification/interrater.py path/to/verdicts_rater2.json
"""
import json
import math
import os
import sys

CATS = ["avalanche", "meltout", "unclear"]
HERE = os.path.dirname(os.path.abspath(__file__))


def wilson(k, n, z=1.96):
    """Score interval for a proportion; the normal approximation is unusable at these counts."""
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def kappa(a, b, cats=CATS):
    """Cohen's kappa: agreement above what the two raters' marginals would give by chance."""
    n = len(a)
    if not n:
        return float("nan")
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    exp = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (obs - exp) / (1 - exp) if exp < 1 else float("nan")


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    d = d.get("verdicts", d)
    return {int(k): v for k, v in d.items()}


def main(path2):
    r1 = load(os.path.join(HERE, "verdicts_rulebased.json"))
    r2 = load(path2)
    keys = sorted(set(r1) & set(r2))
    if len(keys) < len(r1):
        print(f"note: {len(r1) - len(keys)} chips missing from the second rater's file\n")

    a = [r1[k] for k in keys]
    b = [r2[k] for k in keys]
    n = len(keys)

    print(f"Chips rated by both:  {n}\n")
    print(f"{'':<12}" + "".join(f"{c:>12}" for c in CATS) + f"{'total':>9}")
    print(f"{'':<12}" + "-" * (12 * len(CATS) + 9))
    for c1 in CATS:
        row = [sum(1 for x, y in zip(a, b) if x == c1 and y == c2) for c2 in CATS]
        print(f"{c1:<12}" + "".join(f"{v:>12}" for v in row) + f"{sum(row):>9}")
    print(f"{'total':<12}" + "".join(f"{b.count(c):>12}" for c in CATS) + f"{n:>9}")
    print("\n(rows: authors' rule-based pass   columns: independent rater)\n")

    agree = sum(1 for x, y in zip(a, b) if x == y)
    lo, hi = wilson(agree, n)
    print(f"Raw agreement        {agree}/{n} = {agree / n:.1%}  [{lo:.1%}, {hi:.1%}]")
    print(f"Cohen's kappa        {kappa(a, b):.2f}      (all three categories)")

    # Collapsing to the decision the inventory actually turns on.
    ja = [(x, y) for x, y in zip(a, b) if "unclear" not in (x, y)]
    if ja:
        aj = [x == "avalanche" for x, _ in ja]
        bj = [y == "avalanche" for _, y in ja]
        ag = sum(1 for x, y in zip(aj, bj) if x == y)
        lo2, hi2 = wilson(ag, len(ja))
        k2 = kappa([str(x) for x in aj], [str(y) for y in bj], cats=["True", "False"])
        print(f"\nJudgeable chips only ({len(ja)} where neither rater said 'can't tell'):")
        print(f"  agreement          {ag}/{len(ja)} = {ag / len(ja):.1%}  [{lo2:.1%}, {hi2:.1%}]")
        print(f"  kappa              {k2:.2f}      (avalanche vs not)")

    print("\nPrecision implied by each rater, counting 'unclear' as not verified:")
    for name, v in (("authors (rule-based)", a), ("independent rater", b)):
        k = v.count("avalanche")
        lo3, hi3 = wilson(k, n)
        print(f"  {name:<22} {k}/{n} = {k / n:.1%}  [{lo3:.1%}, {hi3:.1%}]")

    dis = [k for k in keys if r1[k] != r2[k]]
    if dis:
        print(f"\nDisagreements ({len(dis)}) — worth looking at together:")
        for k in dis:
            print(f"  cand_{k:03d}   authors: {r1[k]:<10} rater 2: {r2[k]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python verification/interrater.py verdicts_rater2.json")
    main(sys.argv[1])
