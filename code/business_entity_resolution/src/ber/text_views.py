"""Cached, reversible comparison views; the persisted index format is unchanged."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .normalize import core_name, folded


@lru_cache(maxsize=50000)
def ranking_name(text):
    """Only fields needed to rank the much larger raw retrieval pool."""
    text = fold_text(text)
    return text, " ".join(sorted(set(text.split())))


@lru_cache(maxsize=50000)
def ranking_address(text):
    text, ordered = ranking_name(text)
    return text, ordered, frozenset(re.findall(r"\d+", text))


@lru_cache(maxsize=60000)
def fold_text(text):
    # Original folded() loops in Python over every character, including ASCII.
    return text if text.isascii() else folded(text)


@dataclass(frozen=True, slots=True)
class View:
    text: str
    tokens: frozenset
    numbers: frozenset
    digits: frozenset
    sorted_text: str
    core: str
    postal: frozenset
    first_number: str
    variant: str
    initials: str


@lru_cache(maxsize=50000)
def view(text):
    text = fold_text(text)
    words = text.split()
    tokens = frozenset(words)
    numbers = frozenset(t for t in tokens if any(c.isdigit() for c in t))
    digits = frozenset(re.findall(r"\d+", text))
    # Generic string variants only; preserve the original as a separate feature.
    aliases = {"road": "rd", "street": "st", "avenue": "ave", "corporation": "corp",
               "limited": "ltd", "private": "pvt", "incorporated": "inc"}
    variant = " ".join(aliases.get(t, t) for t in words)
    core = core_name(text)
    initials = "".join(t[0] for t in core.split() if t not in {"and", "the", "of"})
    if len(core.split()) == 1:
        initials = core
    return View(text, tokens, numbers, digits, " ".join(sorted(tokens)), core,
                frozenset(t for t in words if t.isdigit() and len(t) in (5, 6)),
                next((t for t in words if any(c.isdigit() for c in t)), ""), variant, initials)
