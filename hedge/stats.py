"""Small statistics toolkit: no scipy, no sklearn.

Everything here is exact/closed-form and unit-tested in tests/test_core.py.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n.

    Preferred over the normal approximation because our n is small and the rates
    sit near 0 or 1, exactly where the naive interval misbehaves (it can even
    produce bounds outside [0, 1]).
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return max(0.0, c - h), min(1.0, c + h)


def fmt_rate(k: int, n: int) -> str:
    """'k/n [lo-hi]' -- the reporting format used throughout the log and report."""
    lo, hi = wilson(k, n)
    if n == 0 or math.isnan(lo):
        return f"{k}/{n} [n/a]"
    return f"{k}/{n} [{lo:.2f}-{hi:.2f}]"


def auroc(pos: Tensor, neg: Tensor) -> float:
    """AUROC = P(random positive scores higher than random negative); 0.5 = chance.

    Ties count 1/2. Exact, no thresholds involved.
    """
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    diff = pos.reshape(-1)[:, None] - neg.reshape(-1)[None, :]
    return float((diff > 0).float().mean() + 0.5 * (diff == 0).float().mean())


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on a paired 2x2 table (discordant cells b, c).

    This is the honest test for "did the intervention flip outcomes": the same
    prompts are measured before and after, so the comparison is paired, and the
    discordant pairs are what carry the evidence. Exact binomial avoids the
    chi-square approximation, which is unreliable for the small b+c we see here.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def paired_diff_ci(before: Sequence[bool], after: Sequence[bool],
                   z: float = 1.96) -> tuple[float, float, float]:
    """Change in a paired proportion, rate(after) - rate(before), with a 95% interval.

    Newcombe's (1998) method 10, built from the two Wilson intervals: unlike the simple
    Wald interval it does not collapse to zero width when nothing changes or the rate
    sits at 0% / 100%. The interval is what supports "the intervention changed the rate
    by at most X": a narrow interval around 0 is evidence of no effect; a wide one is
    merely absence of evidence.
    """
    pairs = list(zip(before, after))
    n = len(pairs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    a = sum(x and y for x, y in pairs)            # yes before, yes after
    b = sum(x and not y for x, y in pairs)        # yes before, no after
    c = sum(y and not x for x, y in pairs)        # no before, yes after
    d = n - a - b - c
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(den) if den else 0.0
    diff = p2 - p1
    lo = diff - math.sqrt(max((p2 - l2) ** 2 - 2 * phi * (p2 - l2) * (u1 - p1) + (u1 - p1) ** 2, 0.0))
    hi = diff + math.sqrt(max((u2 - p2) ** 2 - 2 * phi * (u2 - p2) * (p1 - l1) + (p1 - l1) ** 2, 0.0))
    return diff, max(-1.0, lo), min(1.0, hi)


def permutation_p(observed: float, null: Sequence[float]) -> float:
    """One-sided permutation p-value: P(null >= observed), with the +1 correction.

    `(1 + #{null >= observed}) / (1 + N)` counts the observed statistic as one draw
    from the null, so p is never 0 and N shuffles can give at best p = 1/(N+1).
    """
    null = list(null)
    if not null:
        return float("nan")
    return (1 + sum(float(x) >= float(observed) for x in null)) / (1 + len(null))


def mean_sd(xs: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (0.0 for n < 2)."""
    xs = [float(x) for x in xs if not math.isnan(float(x))]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def summarize_across_seeds(rates: Sequence[float]) -> dict:
    m, sd = mean_sd(rates)
    return {"n_seeds": len(rates), "mean": m, "sd": sd,
            "min": min(rates) if rates else float("nan"),
            "max": max(rates) if rates else float("nan")}
