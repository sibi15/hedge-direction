"""Which language is an answer in? Lightweight, no extra dependencies.

Interventions can push a model out of the prompt's language: Qwen falls back to Chinese
when the hedge direction is ablated, and other models may drift into French, Spanish or
anything else they were trained on. Every pattern list in this project is English, so
an answer in another language would silently count as 'direct' (hedge judge) or 'not
refused' (refusal test). Detecting the language makes that visible -- as a result in
its own right, and as a guard against scoring artefacts.

Two stages:

* **script**: letters outside the Latin alphabet identify Chinese, Japanese, Korean,
  Cyrillic, Arabic, Hebrew, Greek, Devanagari or Thai text;
* **function words** for Latin-script text: the share of very common short words
  (the / le / el / der / il / o / de ...) tells English from French, Spanish, German,
  Italian, Portuguese and Dutch.

Very short answers return "unknown" rather than a guess.
"""
from __future__ import annotations

import re
from collections import Counter

# (label, unicode ranges). Order matters only for readability.
_SCRIPTS = [
    ("japanese", [(0x3040, 0x30FF)]),                      # kana (decides ja vs zh)
    ("chinese", [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF)]),
    ("korean", [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)]),
    ("cyrillic", [(0x0400, 0x04FF)]),
    ("arabic", [(0x0600, 0x06FF), (0x0750, 0x077F)]),
    ("hebrew", [(0x0590, 0x05FF)]),
    ("greek", [(0x0370, 0x03FF)]),
    ("devanagari", [(0x0900, 0x097F)]),
    ("thai", [(0x0E00, 0x0E7F)]),
]
_LATIN = [(0x0041, 0x024F), (0x1E00, 0x1EFF)]

_FUNCTION_WORDS = {
    "english": "the a an and is are was were of to in that it for with as on be this not or by "
               "can but have has which from at will would they you your i we there their if "
               "about more these also such may should its than other some what how do does "
               "no yes sorry help",
    "french": "le la les et est des une un du que qui dans pour pas ne sur ce cette avec au aux "
              "par plus je vous nous il elle sont être désolé peux",
    "spanish": "el la los las y es de que en un una por para con no se lo su del al como más "
               "pero puede son está este esta yo lo siento",
    "german": "der die das und ist nicht ein eine zu den von mit sich auf für es im dem ich sie "
              "wir kann auch sind oder wie aber leid",
    "italian": "il la e di che è un una per non con del della sono si lo gli le ma come più io "
               "può mi dispiace",
    "portuguese": "o a os as e de que em um uma para com não é do da se por mais mas como eu "
                  "pode são desculpe",
    "dutch": "de het een en is van dat niet op te met voor zijn ik je er maar ook als aan",
}
_FUNCTION_WORDS = {k: set(v.split()) for k, v in _FUNCTION_WORDS.items()}
KNOWN = sorted({"unknown"} | {name for name, _ in _SCRIPTS} | set(_FUNCTION_WORDS))


def _script(ch: str) -> str | None:
    o = ord(ch)
    if any(lo <= o <= hi for lo, hi in _LATIN):
        return "latin"
    for name, ranges in _SCRIPTS:
        if any(lo <= o <= hi for lo, hi in ranges):
            return name
    return "other"


def language_of(text: str, min_letters: int = 10, min_words: int = 4) -> str:
    """Best guess at the language of `text`; "unknown" when there is too little text."""
    letters = [ch for ch in (text or "") if ch.isalpha()]
    if len(letters) < min_letters:
        return "unknown"
    scripts = Counter(_script(ch) for ch in letters)
    non_latin = len(letters) - scripts["latin"]
    if non_latin / len(letters) > 0.3:
        if scripts["japanese"]:                       # kana present: Japanese, not Chinese
            return "japanese"
        best = max((s for s in scripts if s != "latin"), key=scripts.get)
        return best if best != "other" else "other"

    words = re.findall(r"[^\W\d_]+", text.lower())
    if len(words) < min_words:
        return "unknown"
    share = {lang: sum(w in fw for w in words) / len(words) for lang, fw in _FUNCTION_WORDS.items()}
    best = max(share, key=share.get)
    # Ties and near-ties go to English: short function words ("a", "de", "la") are shared.
    if best == "english" or share[best] - share["english"] < 0.05 or share[best] < 0.1:
        return "english" if share["english"] >= 0.05 or share[best] < 0.1 else best
    return best


def other_language(text: str, expected: str = "english") -> bool:
    """True if the answer is confidently in a language other than `expected`."""
    lang = language_of(text)
    return lang not in (expected, "unknown")


def breakdown(texts, expected: str = "english") -> dict[str, int]:
    """Counts of the languages that are not `expected` (and not 'unknown')."""
    c = Counter(language_of(t) for t in texts)
    return {k: v for k, v in sorted(c.items(), key=lambda kv: -kv[1])
            if k not in (expected, "unknown")}
