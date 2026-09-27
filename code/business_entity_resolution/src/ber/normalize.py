from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from .common import stable_int

SUFFIXES = {"ltd", "limited", "inc", "incorporated", "corp", "corporation", "llc", "pvt", "private"}
STOP = SUFFIXES | {"the", "and", "of", "a", "an"}


def normalize(value):
    text = unicodedata.normalize("NFKC", value).casefold().replace("&", " and ")
    # Preserve combining marks (e.g. Indic vowels), not only letters/digits.
    text = "".join(c if c.isalnum() or unicodedata.category(c).startswith("M") else " " for c in text)
    return " ".join(text.split())


def folded(value):
    result, latin_base = [], False
    for char in unicodedata.normalize("NFKD", value):
        if unicodedata.category(char).startswith("M"):
            if not latin_base:
                result.append(char)
        else:
            result.append(char)
            latin_base = "LATIN" in unicodedata.name(char, "")
    return "".join(result)


@dataclass(frozen=True)
class Record:
    rid: int
    entity_id: str
    source: int
    country: str
    name: str
    address: str

    @property
    def tokens(self):
        return frozenset(self.name.split())

    @property
    def address_tokens(self):
        return frozenset(self.address.split())


def core_name(name):
    words = name.split()
    while words and words[-1] in SUFFIXES:
        words.pop()
    return " ".join(words)


def key_strings(record, cfg):
    name, address = folded(record.name), folded(record.address)
    keys = set()
    if name:
        keys.add("n:" + name)
    if name and address:
        keys.add("e:" + name + "|" + address)
    core = core_name(name)
    if core and core != name:
        keys.add("n:" + core)
    tokens = sorted({t for t in core.split() if t not in STOP and len(t) > 1}, key=lambda t: (-len(t), t))
    keys.update("t:" + t for t in tokens[:cfg["name_tokens"]])
    compact = name.replace(" ", "")
    grams = {compact[i:i + 3] for i in range(max(0, len(compact) - 2))}
    keys.update("g:" + g for g in sorted(grams, key=stable_int)[:cfg["name_grams"]])
    # Address keys use a name token/initials, never address-only identity evidence.
    atokens = sorted({t for t in address.split() if len(t) > 2}, key=lambda t: (-len(t), t))[:cfg["address_tokens"]]
    if tokens:
        keys.update("a:" + tokens[0] + "|" + a for a in atokens)
    initials = "".join(t[0] for t in core.split() if t not in STOP)
    if len(core.split()) == 1 and 2 <= len(core) <= 8:
        initials = core
    if 2 <= len(initials) <= 8:
        keys.update("i:" + initials + "|" + a for a in atokens[:2])
    return keys


def index_keys(record, source, cfg):
    # Global, source-specific index: country is evidence, never a closed-set filter.
    return [(stable_int(f"{source}|{key}"), key[:1]) for key in sorted(key_strings(record, cfg))]
