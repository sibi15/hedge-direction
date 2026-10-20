"""Prompts, hand labels and the train/val/test split.

A prompt source is

* `.txt`: one prompt per line; blank lines and lines starting with `#` are ignored, or
* `.json`: a list of prompt strings, or of objects whose `statement` (or `prompt`,
  `question`, `instruction`, `goal`) field is the prompt -- other fields are ignored. A JSON *object* holds
  several such lists by name, and `file.json#name` selects one: that is how
  `data/prompts.json#questions` and `data/prompts.json#factual` share one file.

Relative paths resolve against the current directory, then the repository root.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from .judging import CLASSES
from .settings import REPO_ROOT, HedgeError

# "instruction" / "goal": the field names in Arditi et al.'s and JailbreakBench's files
JSON_PROMPT_FIELDS = ("statement", "prompt", "question", "instruction", "goal", "Goal")


# ----------------------------------------------------------------------------- prompts

def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p if (REPO_ROOT / p).exists() else p


def _split_section(spec: str) -> tuple[str, str | None]:
    """'data/prompts.json#factual' -> ('data/prompts.json', 'factual')."""
    head, sep, section = str(spec).rpartition("#")
    if sep and head.lower().endswith(".json") and section:
        return head, section
    return str(spec), None


def _read_json(p: Path, section: str | None) -> list[str]:
    try:
        items = json.loads(p.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        raise HedgeError(f"{p}: invalid JSON ({e})") from None
    if isinstance(items, dict):
        lists = sorted(k for k, v in items.items() if isinstance(v, list))
        if section is None:
            raise HedgeError(f"{p} holds several prompt lists {lists}; pick one with "
                             f"'{p.name}#<name>'")
        if section not in items:
            raise HedgeError(f"{p}: no section {section!r}; available: {lists}")
        items = items[section]
    elif section is not None:
        raise HedgeError(f"{p} is a plain list; drop the '#{section}' suffix")
    if not isinstance(items, list):
        raise HedgeError(f"{p}: expected a JSON list of prompts or prompt objects")
    out = []
    for i, item in enumerate(items):
        if isinstance(item, dict):
            key = next((k for k in JSON_PROMPT_FIELDS if k in item), None)
            if key is None:
                raise HedgeError(f"{p}: item {i} has none of the fields {JSON_PROMPT_FIELDS}")
            item = item[key]
        if not isinstance(item, str):
            raise HedgeError(f"{p}: item {i} is not a string prompt")
        s = " ".join(item.split())   # one line, single spaces
        if s:
            out.append(s)
    return out


def read_prompts(spec) -> list[str]:
    """Prompts from a path, 'file.json#section', or a list of those."""
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        return [p for one in spec for p in read_prompts(one)]
    path, section = _split_section(spec)
    p = _resolve(path)
    if not p.exists():
        raise HedgeError(f"prompt file not found: {path}")
    if p.suffix.lower() == ".json":
        return _read_json(p, section)
    if section is not None:
        raise HedgeError(f"{p}: '#{section}' sections only apply to .json files")
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def spread(items: list, k: int | None) -> list:
    """`k` items spaced evenly through the list (all of them if k is None or too big).

    Evenly rather than the first k: prompt files are usually grouped (by type, by
    category), so "first k" would cover only the first few groups. Deterministic.
    """
    if not k or int(k) >= len(items):
        return list(items)
    k = int(k)
    if k == 1:
        return [items[len(items) // 2]]
    idx = sorted({round(i * (len(items) - 1) / (k - 1)) for i in range(k)})
    return [items[i] for i in idx]


def load_questions(s) -> list[str]:
    """The question bank, deduplicated, optionally capped evenly across the bank."""
    qs = list(dict.fromkeys(read_prompts(s.questions)))
    if not qs:
        raise HedgeError(f"no questions found in {s.questions!r}")
    return spread(qs, max(2, int(s.max_prompts)) if s.max_prompts else None)


# ------------------------------------------------------------------------------- split

def class_pools(questions, labels) -> dict[str, list[str]]:
    pools: dict[str, list[str]] = {c: [] for c in CLASSES}
    for q, lab in zip(questions, labels):
        if lab not in pools:
            raise HedgeError(f"unknown label {lab!r} for question {q!r}; expected {CLASSES}")
        pools[lab].append(q)
    return pools


def split3(xs, val_frac: float, test_frac: float, seed: int,
           min_val: int = 3, min_test: int = 3) -> tuple[list, list, list]:
    """Deterministic shuffle, then (train, val, test).

    train builds the direction, val chooses the layer and coefficient, and test is only
    used by the final test -- so the reported effect never chose its own configuration.
    """
    xs = list(xs)
    n = len(xs)
    if n < 3:
        raise HedgeError(f"a class with {n} prompts cannot be split into train/val/test "
                         f"(need >= 3); add questions or turn off `quick`")
    random.Random(seed).shuffle(xs)
    k_test = max(int(min_test), int(n * test_frac), 1)
    k_val = max(int(min_val), int(n * val_frac), 1)
    while k_test + k_val > n - 1:   # never leave the train side empty
        if k_test >= k_val and k_test > 1:
            k_test -= 1
        else:
            k_val -= 1
    return xs[k_test + k_val:], xs[k_test:k_test + k_val], xs[:k_test]


# ------------------------------------------------------------------------- hand labels

def _read_hand_labels(path: Path, validate: bool = True) -> dict[str, str]:
    hand: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab = (row.get("hand_label") or "").strip().lower()
            q = (row.get("question") or "").strip()
            if lab and q:
                hand[q] = lab
    bad = {v for v in hand.values() if v not in CLASSES} if validate else set()
    if bad:
        raise HedgeError(f"{path}: unknown hand labels {sorted(bad)}; use {list(CLASSES)}")
    return hand


def write_labels_to_check(path: Path, questions, auto_labels, answers) -> None:
    """The hand-check worksheet. Hand labels already filled in survive a rewrite."""
    path = Path(path)
    kept = _read_hand_labels(path, validate=False) if path.exists() else {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "auto_label", "hand_label", "answer"])
        for q, lab, a in zip(questions, auto_labels, answers):
            w.writerow([q, lab, kept.get(q, ""), a])


def apply_hand_labels(s, run_dir: Path, questions, auto_labels) -> tuple[list, dict | None]:
    """Overlay hand labels on the judge's labels; also report how often they agree.

    Looked for at `s.hand_labels`, then <run>/labels_checked.csv, then the
    <run>/labels_to_check.csv worksheet (once any hand_label cell is filled in).
    """
    if s.hand_labels:
        p = _resolve(s.hand_labels)
        if not p.exists():
            raise HedgeError(f"hand_labels={s.hand_labels!r} not found")
        candidates = [p]
    else:
        candidates = [Path(run_dir) / "labels_checked.csv", Path(run_dir) / "labels_to_check.csv"]
    path, hand = None, {}
    for p in candidates:
        if p.exists():
            hand = _read_hand_labels(p)
            if hand:
                path = p
                break
    if not hand:
        return list(auto_labels), None
    labels = [hand.get(q, a) for q, a in zip(questions, auto_labels)]
    pairs = [(hand[q], a) for q, a in zip(questions, auto_labels) if q in hand]
    agree = sum(h == a for h, a in pairs)
    return labels, {"path": str(path), "n_hand": len(pairs), "n_agree": agree,
                    "rate": agree / len(pairs) if pairs else None}
