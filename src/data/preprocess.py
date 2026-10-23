"""Stage 3: clean maintenance-log text: case, field shorthand, asset tags, and typos learned from the plant's own text."""
from __future__ import annotations

import re

SHORTHAND = {
    r"\btemp\b": "temperature", r"\bvibn?\b": "vibration", r"\bbrg\b": "bearing", r"\bmech\b": "mechanical",
    r"\bpress\b": "pressure", r"\bincr\b": "increasing", r"\brepl\b": "replaced", r"\bnrml\b": "normal",
    r"\bhi\b": "high", r"\blo\b": "low", r"\blkg\b": "leak", r"\bvlv\b": "valve",
}
_ASSET_RE = re.compile(r"\b[a-z]{1,2}-\d{3,4}a?\b")


def clean(text: str, fixes: dict | None = None) -> str:
    t = str(text).lower()
    for pat, rep in SHORTHAND.items():
        t = re.sub(pat, rep, t)
    t = _ASSET_RE.sub(" ", t)  # asset tags carry no meaning for the model
    if fixes:
        t = re.sub(r"[a-z]+", lambda m: fixes.get(m.group(0), m.group(0)), t)
    return re.sub(r"\s+", " ", t).strip()


def learn_typo_fixes(texts, min_ratio: float = 20.0) -> dict:
    """Map a word to a much more frequent word that differs by one swapped pair of adjacent letters.
    Learned from the plant's own text, so local vocabulary ('positioner', 'impeller') is respected."""
    from collections import Counter
    freq = Counter(w for t in texts for w in re.findall(r"[a-z]+", clean(t)))
    fixes = {}
    for w, n in freq.items():
        if len(w) < 5:
            continue
        for i in range(len(w) - 1):
            cand = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            if cand != w and freq.get(cand, 0) >= min_ratio * n:
                fixes[w] = cand
                break
    return fixes
