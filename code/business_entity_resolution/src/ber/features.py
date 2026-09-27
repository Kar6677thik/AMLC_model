from __future__ import annotations

from rapidfuzz.fuzz import ratio, token_set_ratio, token_sort_ratio

from .normalize import core_name, folded

FEATURES = [
    "name_exact", "name_ratio", "name_token_sort", "name_token_set", "name_jaccard",
    "name_containment", "name_length_ratio", "core_name_exact", "address_exact",
    "address_ratio", "address_token_sort", "address_token_set", "address_jaccard",
    "address_containment", "address_length_ratio", "number_jaccard", "number_conflict",
    "number_exact", "both_have_numbers", "country_equal", "country_conflict",
    "country_missing", "name_missing", "address_missing", "name_token_count",
    "target_name_token_count", "address_token_count", "target_address_token_count",
    "target_is_s3", "retrieval_score", "retrieval_key_hits", "candidate_count",
]


def overlap(a, b):
    return len(a & b) / len(a | b) if a or b else 0.0


def contain(a, b):
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def length_ratio(a, b):
    return min(len(a), len(b)) / max(len(a), len(b)) if a and b else 0.0


def pair_features(anchor, candidate, count):
    target = candidate.record
    an, bn, aa, ba = map(folded, (anchor.name, target.name, anchor.address, target.address))
    ant, bnt, aat, bat = (set(t.split()) for t in (an, bn, aa, ba))
    anum, bnum = ({t for t in ts if any(c.isdigit() for c in t)} for ts in (aat, bat))
    name_ok, address_ok = bool(an and bn), bool(aa and ba)
    country_ok = bool(anchor.country and target.country)
    cn, ct = core_name(an), core_name(bn)
    values = [
        name_ok and an == bn,
        ratio(an, bn)/100 if name_ok else 0,
        token_sort_ratio(an, bn)/100 if name_ok else 0,
        token_set_ratio(an, bn)/100 if name_ok else 0,
        overlap(ant, bnt), contain(ant, bnt), length_ratio(an, bn), bool(cn and ct and cn == ct),
        address_ok and aa == ba,
        ratio(aa, ba)/100 if address_ok else 0,
        token_sort_ratio(aa, ba)/100 if address_ok else 0,
        token_set_ratio(aa, ba)/100 if address_ok else 0,
        overlap(aat, bat), contain(aat, bat), length_ratio(aa, ba),
        overlap(anum, bnum), bool(anum and bnum and not (anum & bnum)),
        bool(anum and bnum and anum == bnum), bool(anum and bnum),
        country_ok and anchor.country == target.country,
        country_ok and anchor.country != target.country,
        not country_ok, not name_ok, not address_ok,
        len(ant), len(bnt), len(aat), len(bat), target.source == 3,
        candidate.retrieval_score, candidate.key_hits, count,
    ]
    return [float(v) for v in values]


def rule_score(features):
    """Conservative engineering baseline, not a calibrated probability."""
    f = dict(zip(FEATURES, features))
    if f["name_missing"] or f["address_missing"] or f["country_conflict"]:
        return 0.0
    if f["name_exact"] and f["address_exact"]:
        return 1.0
    score = .55*f["name_ratio"] + .35*f["address_ratio"] + .10*f["number_exact"]
    if f["number_conflict"]:
        score -= .25
    return max(0.0, min(1.0, score))
