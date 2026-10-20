"""The labeler: turns an answer into direct / hedge / disclaimer.

Identical in spirit to the notebook's keyword judge, with the pattern lists in
`hedge.settings` so they can be extended without editing code. This is a *proxy*
for hedging -- it is noisy, and every step dumps the raw completions precisely so
the labels can be audited by hand (`labels_to_check.csv`, `completions.jsonl`).
"""
from __future__ import annotations

import re

from .language import breakdown, other_language
from .stats import wilson

CLASSES = ("direct", "hedge", "disclaimer")


class Judge:
    def __init__(self, s):
        """`s` is a hedge.Settings (patterns, order, opening length, degeneracy limits)."""
        self.max_chars = int(s.judge_max_chars)
        self.order = list(s.judge_order)
        self.disclaimer = [re.compile(p) for p in s.disclaimer_patterns]
        self.hedge = [re.compile(p) for p in s.hedge_patterns]
        self.min_words = int(s.degenerate_min_words)
        self.min_unique_ratio = float(s.degenerate_min_unique)
        self.language = s.answer_language

    # ------------------------------------------------------------------ labels

    def label(self, text: str) -> str:
        """Only the opening of an answer is judged -- hedging is front-loaded."""
        t = (text or "").lower()[: self.max_chars]
        for cls in self.order:
            if cls == "disclaimer" and any(p.search(t) for p in self.disclaimer):
                return "disclaimer"
            if cls == "hedge" and any(p.search(t) for p in self.hedge):
                return "hedge"
            if cls == "direct":
                return "direct"
        raise ValueError(f"judge.order does not end in 'direct': {self.order}")

    def looks_degenerate(self, text: str) -> bool:
        """Crude coherence check: empty output, or heavy word repetition.

        Catches the failure mode where an over-strong intervention destroys the
        model and the resulting gibberish happens to match no pattern (and would
        therefore be scored as 'direct').
        """
        words = re.findall(r"\w+", (text or "").lower())[:60]
        if not words:
            return True
        if len(words) >= self.min_words and len(set(words)) / len(words) < self.min_unique_ratio:
            return True
        return False

    # --------------------------------------------------------------- summaries

    def summarize(self, texts) -> dict:
        """Per-class counts with 95% Wilson intervals, plus degenerate and other-language counts.

        `other_language` counts answers confidently in a language other than the expected
        one; `languages` says which (e.g. {"chinese": 18}). The patterns are English, so
        such answers are labelled 'direct' -- the counts make that visible.
        """
        texts = list(texts)
        labs = [self.label(t) for t in texts]
        n = len(labs)
        s = {"n": n, "degenerate": sum(self.looks_degenerate(t) for t in texts),
             "other_language": sum(other_language(t, self.language) for t in texts),
             "languages": breakdown(texts, self.language),
             "labels": labs}
        for c in CLASSES:
            k = labs.count(c)
            lo, hi = wilson(k, n)
            s[c] = {"k": k, "ci95": [round(lo, 2), round(hi, 2)]}
        return s


def rate_str(summary: dict, cls: str) -> str:
    """Format one class of a summary as 'k/n [lo-hi]'."""
    if not summary or summary.get("n", 0) == 0:
        return "0/0 [n/a]"
    c = summary[cls]
    return f"{c['k']}/{summary['n']} [{c['ci95'][0]:.2f}-{c['ci95'][1]:.2f}]"


def is_refusal(text: str, patterns) -> bool:
    """Arditi et al.'s refusal check: any refusal substring, case-insensitive.

    Separate from the hedge judge on purpose: "As an AI" counts as a refusal here and as a
    disclaimer there, and the report says how often the two overlap.
    """
    t = (text or "").lower()
    return any(p.lower() in t for p in patterns)


def slim(summary: dict | None) -> dict | None:
    """A summary without its per-example label list (for results.json)."""
    return None if summary is None else {k: v for k, v in summary.items() if k != "labels"}


def rate(summary: dict, cls: str) -> float:
    return summary[cls]["k"] / summary["n"] if summary and summary.get("n") else float("nan")
