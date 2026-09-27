"""Auditable, deletion-only refinements of a completed BER run (standard library only)."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import time
import zipfile

VERSION = 1
MATCH = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE = ["source1_entity_id", "candidate_entity_ids"]
LEDGER = ["source1_entity_id", "candidate_entity_ids", "scores"]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def log(message):
    print(time.strftime("[%H:%M:%S] ") + message, flush=True)


def connect(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA cache_size=-65536")
    return db


def fresh(path):
    path = Path(path)
    if path.exists():
        raise ValueError(f"Preserve existing results; choose a new directory: {path}")
    path.mkdir(parents=True)
    return path


def fetch(db, values, column="entity_id"):
    if column not in ("entity_id", "rid"):
        raise ValueError(column)
    values = list(set(values))
    result = {}
    for start in range(0, len(values), 700):
        batch = values[start:start + 700]
        for row in db.execute(f"SELECT * FROM records WHERE {column} IN ({','.join('?' for _ in batch)})", batch):
            result[row[column]] = dict(row)
    if len(result) != len(values):
        raise ValueError("Unknown record ID in score ledger")
    return result


def build_context(db, path):
    """One S1 scan; all indices live in a separate sidecar, never in the BER index."""
    context = sqlite3.connect(path)
    context.execute("PRAGMA cache_size=-65536")
    context.execute("PRAGMA temp_store=FILE")
    context.execute("CREATE TABLE s1 (id TEXT PRIMARY KEY,country TEXT,name TEXT,address TEXT)")
    context.executemany("INSERT INTO s1 VALUES (?,?,?,?)", db.execute(
        "SELECT entity_id,country,name,address FROM records WHERE source=1 ORDER BY rid"))
    context.execute("CREATE INDEX s1_address ON s1(country,address,name)")
    context.execute("CREATE TABLE names AS SELECT country,name,COUNT(*) n FROM s1 WHERE name!='' GROUP BY country,name")
    context.execute("CREATE UNIQUE INDEX names_key ON names(country,name)")
    context.commit()
    log("Built Source-1 ambiguity index")
    return context


# Parse only explicit, bounded street spans. Unknown layouts abstain. These are
# syntax markers, not business-name stop words or country-specific identity rules.
FR_ROADS = {"rue", "avenue", "boulevard", "chemin", "impasse", "allee", "allée", "route", "place", "quai"}
EN_ROADS = {"street", "st", "road", "rd", "avenue", "ave", "boulevard", "blvd", "lane", "ln", "drive", "dr"}
CONNECTORS = {"de", "du", "des", "la", "le", "les", "d", "l", "the", "of"}


def street(address):
    tokens = address.split()
    # French: a street marker followed by name and then a 5-digit postal code.
    # City text before a postal code cannot be isolated safely, so only accept a
    # short span; require disjoint names for the subsequent conflict flag.
    for i, token in enumerate(tokens):
        if token in FR_ROADS:
            end = next((j for j in range(i + 1, len(tokens)) if re.fullmatch(r"\d{5}", tokens[j])), None)
            if end is not None and 1 <= end - i - 1 <= 5:
                words = frozenset(t for t in tokens[i + 1:end] if not t.isdigit() and t not in CONNECTORS)
                return words if words else None
    # English/Indian: initial house number, then street name, ending at road type.
    if tokens and re.fullmatch(r"\d+[a-z]?", tokens[0]):
        for j in range(2, min(len(tokens), 7)):
            if tokens[j] in EN_ROADS:
                words = frozenset(t for t in tokens[1:j] if not t.isdigit() and t not in CONNECTORS)
                return words if words else None
    return None


def evidence(anchor, target, context, rules=None):
    country = bool(anchor["country"] and target["country"] and anchor["country"] != target["country"])
    # Exact normalized full names only: no unverified removal of legal/filler words.
    ambiguous = False
    if (rules is None or rules["missing_address"]) and (not anchor["address"] or not target["address"]):
        for name in {anchor["name"], target["name"]} - {""}:
            count = context.execute("SELECT n FROM names WHERE country=? AND name=?", (anchor["country"], name)).fetchone()
            ambiguous |= bool(count and count[0] > 1)
    left, right = (street(anchor["address"]), street(target["address"])) if rules is None or rules["street"] else (None, None)
    conflict = bool(left and right and left.isdisjoint(right))
    rival = False
    if (rules is None or rules["rival_name"]) and anchor["address"] and anchor["address"] == target["address"] and target["name"] and anchor["name"] != target["name"]:
        rival = context.execute("SELECT 1 FROM s1 WHERE country=? AND address=? AND name=? AND id!=? LIMIT 1",
            (anchor["country"], anchor["address"], target["name"], anchor["entity_id"])).fetchone() is not None
    return {"country": country, "missing_address": ambiguous, "street": conflict, "rival_name": rival}


def local_keep(row, policy):
    eligible = row["country"] in policy["learned_countries"]
    top = max((edge[1] for edge in row["edges"]), default=0.0)
    selected = []
    for target, score, flags in row["edges"]:
        if policy["country"] and flags["country"]:
            continue
        if eligible and any(policy[k] and flags[k] for k in ("missing_address", "street", "rival_name")):
            continue
        if eligible and policy["gap"] is not None and score < top - policy["gap"]:
            continue
        selected.append((target, score))
    return selected


def predictions(rows, policy):
    chosen = [local_keep(row, policy) for row in rows]
    if policy["ownership"]:
        best = {}
        for i, edges in enumerate(chosen):
            for target, score in edges:
                if target not in best or score > best[target][0]:
                    best[target] = (score, i)
                elif score == best[target][0]:
                    best[target] = (score, None)  # equal best scores: abstain on all ties
        chosen = [[(t, s) for t, s in edges if best[t][1] == i] for i, edges in enumerate(chosen)]
    return [set(t for t, _ in edges) for edges in chosen]


def f05(gold, predicted):
    return 5 * len(gold & predicted) / (4 * len(predicted) + len(gold)) if gold else float(not predicted)


def summarize(rows, chosen, split=None):
    values = []
    countries = defaultdict(list)
    tp = fp = fn = empty = empty_fp = 0
    for row, selected in zip(rows, chosen):
        if split is not None and row["split"] != split:
            continue
        score = f05(row["truth"], selected)
        values.append(score)
        countries[row["country"]].append(score)
        tp += len(row["truth"] & selected)
        fp += len(selected - row["truth"])
        fn += len(row["truth"] - selected)
        empty += not row["truth"]
        empty_fp += not row["truth"] and bool(selected)
    return {"anchors": len(values), "macro_f05": sum(values) / max(1, len(values)),
        "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
        "singleton_false_positive_rate": empty_fp / max(1, empty),
        "countries": {c: {"anchors": len(v), "macro_f05": sum(v) / len(v)} for c, v in countries.items()}}


def changed_links(rows, baseline, selected):
    removed_true = removed_false = affected = 0
    by_country = defaultdict(Counter)
    for row, before, after in zip(rows, baseline, selected):
        removed = before - after
        affected += bool(removed)
        good = len(removed & row["truth"])
        bad = len(removed - row["truth"])
        removed_true += good
        removed_false += bad
        by_country[row["country"]].update(removed_true=good, removed_false=bad, affected_anchors=int(bool(removed)))
    return {"removed_true": removed_true, "removed_false": removed_false, "affected_anchors": affected,
        "countries": {c: dict(v) for c, v in by_country.items()}}


def study(args):
    if not 0 < args.min_gain <= 1 or not 0 <= args.country_tolerance <= 1:
        raise ValueError("Require min-gain in (0,1] and country-tolerance in [0,1]")
    work, run = Path(args.work), Path(args.run)
    manifest, meta = read(work / "train_manifest.json"), read(run / "run.json")
    if meta["train_manifest"]["signature"] != manifest["signature"]:
        raise ValueError("Training data signature differs")
    report = read(run / "dev_report.json")
    if digest(run / "dev_scores.jsonl") != report["scores_sha256"]:
        raise ValueError("Development score cache changed")
    threshold = read(run / "decision.json")["threshold"]
    out = fresh(args.output)
    db = connect(work / "train.sqlite")
    context = build_context(db, out / "context.sqlite")
    rows = []
    with open(run / "dev_scores.jsonl", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            targets, scores = item["targets"], item["scores"]
            if len(targets) != len(scores) or len(set(targets)) != len(targets) or any(not math.isfinite(s) for s in scores):
                raise ValueError("Invalid development score row")
            anchor = fetch(db, [item["source1_entity_id"]])[item["source1_entity_id"]]
            candidates = [(t, s) for t, s in zip(targets, scores) if s >= threshold]
            records = fetch(db, [t for t, _ in candidates], "rid")
            rows.append({"id": anchor["entity_id"], "country": anchor["country"], "truth": set(item["truth"]),
                "split": hashlib.sha256(("refinement-v1|" + anchor["entity_id"]).encode()).digest()[0] % 2,
                "edges": [(t, s, evidence(anchor, records[t], context)) for t, s in candidates]})
            if len(rows) % 2000 == 0:
                log(f"Annotated {len(rows):,} development anchors")
    context.close()
    db.close()
    if len(rows) < 100 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Insufficient or duplicate development anchors")
    base = dict(country=False, ownership=False, missing_address=False, street=False, rival_name=False,
        gap=None, learned_countries=sorted({r["country"] for r in rows if r["country"]}))
    policies = {"baseline": base}
    stats = manifest["stats"]
    # Contradictions or absent evidence block structural rules.
    for rule, supported in [("country", stats.get("positive_country_mismatches") == 0),
                            ("ownership", stats.get("shared_component_anchors") == 0),
                            ("missing_address", True), ("street", True), ("rival_name", True)]:
        if supported:
            policies[rule] = dict(base, **{rule: True})
    for gap in (.4, .2, .1):
        policies[f"set_gap_{gap}"] = dict(base, gap=gap)
    if "country" in policies and "ownership" in policies:
        policies["country_ownership"] = dict(base, country=True, ownership=True)
    results = {}
    baseline = predictions(rows, base)
    if abs(summarize(rows, baseline)["macro_f05"] - report["metrics"]["macro_f05"]) > 1e-9:
        raise ValueError("Baseline reproduction differs from recorded development metric")
    for name, policy in policies.items():
        pred = predictions(rows, policy)
        results[name] = {"tune": summarize(rows, pred, 0), "check": summarize(rows, pred, 1),
            "all": summarize(rows, pred), "changes": changed_links(rows, baseline, pred)}
    # Choose ONCE on tune, gate once on check. Never search the check split for a fallback.
    winner = max(policies, key=lambda k: (results[k]["tune"]["macro_f05"], k == "baseline"))
    reasons = []
    for split in ("tune", "check"):
        gain = results[winner][split]["macro_f05"] - results["baseline"][split]["macro_f05"]
        if gain < args.min_gain:
            reasons.append(f"{split} gain {gain:.6f} below {args.min_gain}")
        for country, value in results["baseline"][split]["countries"].items():
            delta = results[winner][split]["countries"][country]["macro_f05"] - value["macro_f05"]
            if delta < -args.country_tolerance:
                reasons.append(f"{split} country {country} regressed by {-delta:.6f}")
    selected = "baseline" if reasons else winner
    policy = {"version": VERSION, "name": selected, "threshold": threshold, "rules": policies[selected],
        "run_sha256": digest(run / "run.json"), "decision_sha256": digest(run / "decision.json"),
        "dev_scores_sha256": report["scores_sha256"], "source_sha256": digest(__file__),
        "minimum_gain": args.min_gain, "country_tolerance": args.country_tolerance,
        "selection_note": "Individual ablations plus explicit country+ownership combination; reused development data, not an untouched holdout. No France accuracy claim."}
    save(out / "policy.json", policy)
    save(out / "study.json", {"selected": selected, "tune_winner": winner, "rejection_reasons": reasons,
        "ablations": results, "policies": policies, "policy_sha256": digest(out / "policy.json"),
        "caveats": ["Existing model/threshold were already selected on this development sample.",
            "Ownership competition is observed only among cached development anchors; test competition is global.",
            "Only deletions are possible; candidate recall and its oracle cannot increase.",
            "Raw model scores are ranking scores, not calibrated probabilities."]})
    log(f"Selected: {selected}; report: {out / 'study.json'}")


def tsv(path, header, compressed=False):
    opener = gzip.open if compressed else open
    csv.field_size_limit(64 * 1024 * 1024)
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t", strict=True)
        if next(reader, None) != header:
            raise ValueError(f"Unexpected header: {path}")
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"Invalid row: {path}")
            yield row


def ids(text):
    result = text.split(",") if text else []
    if result != sorted(set(result)) or any(not t or t.strip() != t for t in result):
        raise ValueError("Expected sorted unique IDs")
    return result


def apply(args):
    work, run, original = Path(args.work), Path(args.run), Path(args.original)
    policy = read(args.policy)
    if policy["version"] != VERSION or policy["source_sha256"] != digest(__file__):
        raise ValueError("Policy belongs to a different refinement implementation")
    for key, path in [("run_sha256", run / "run.json"), ("decision_sha256", run / "decision.json")]:
        if policy[key] != digest(path):
            raise ValueError(f"Policy mismatch: {key}")
    if policy["name"] == "baseline":
        raise ValueError("No measured refinement passed. Use the completed baseline output unchanged.")
    prediction = read(original / "prediction.json")  # Requires completed inference, never in-progress shards.
    if prediction["threshold"] != policy["threshold"]:
        raise ValueError("Frozen threshold differs")
    if any(prediction["identity"][key] != policy[key] for key in ("run_sha256", "decision_sha256")):
        raise ValueError("Original prediction belongs to another model/decision")
    if prediction["identity"]["test_signature"] != read(work / "test_manifest.json")["signature"]:
        raise ValueError("Test data signature differs")
    for name, expected in prediction["hashes"].items():
        if digest(original / name) != expected:
            raise ValueError(f"Original output changed: {name}")
    validation = read(original / "validation.json")
    if not validation.get("passed") or validation.get("hashes") != prediction["hashes"] or "target_membership" not in validation.get("checks", []):
        raise ValueError("Run ber validate on original outputs first; candidate membership must be verified")
    out = fresh(args.output)
    db = connect(work / "test.sqlite")
    context = build_context(db, out / "decisions.sqlite")
    context.execute("CREATE TABLE edges(target TEXT,anchor TEXT,score REAL,PRIMARY KEY(target,anchor)) WITHOUT ROWID")
    context.execute("CREATE TABLE seen(id TEXT PRIMARY KEY)")
    changes, country_changes = Counter(), defaultdict(Counter)
    streams = [tsv(original / "matching_results.tsv", MATCH), tsv(original / "candidate_pairs.tsv", CANDIDATE),
               tsv(original / "scoring_ledger.tsv.gz", LEDGER, True),
               db.execute("SELECT * FROM records WHERE source=1 ORDER BY rid")]
    count = candidate_count = 0
    for match, candidate, ledger, raw_anchor in itertools.zip_longest(*streams):
        if any(x is None for x in (match, candidate, ledger, raw_anchor)):
            raise ValueError("Missing/extra rows in original outputs")
        anchor = dict(raw_anchor)
        if not match[0] == candidate[0] == ledger[0] == anchor["entity_id"]:
            raise ValueError("Original outputs are not in Source-1 order")
        targets, scored = ids(candidate[1]), ids(ledger[1])
        scores = list(map(float, ledger[2].split(","))) if ledger[2] else []
        if targets != scored or len(targets) != len(scores) or any(not math.isfinite(s) for s in scores):
            raise ValueError("Candidate/score alignment failure")
        above = [(t, s) for t, s in zip(targets, scores) if s >= policy["threshold"]]
        if ids(match[1]) != [t for t, _ in above]:
            raise ValueError("Original threshold decision cannot be reproduced")
        # Unchanged candidate hashes inherit the original validator's membership
        # check. Fetch text only for provisional matches, avoiding ~100M lookups.
        records = fetch(db, [t for t, _ in above])
        if any(r["source"] not in (2, 3) for r in records.values()):
            raise ValueError("Candidate is not Source 2/3")
        row = {"country": anchor["country"], "edges": [(t, s, evidence(anchor, records[t], context, policy["rules"])) for t, s in above]}
        for _, _, flags in row["edges"]:
            changes.update({"flag_" + k: int(v) for k, v in flags.items()})
        selected = local_keep(row, policy["rules"])
        changes["removed_local"] += len(above) - len(selected)
        country_changes[anchor["country"]]["removed_local"] += len(above) - len(selected)
        country_changes[anchor["country"]]["original_matches"] += len(above)
        context.executemany("INSERT INTO edges VALUES (?,?,?)", ((t, anchor["entity_id"], s) for t, s in selected))
        context.execute("INSERT INTO seen VALUES (?)", (anchor["entity_id"],))
        count += 1
        candidate_count += len(targets)
        if count % 10000 == 0:
            context.commit()
            log(f"Refinement: {count:,} anchors")
    context.commit()
    if count != prediction["anchors"] or candidate_count != prediction["scored_pairs"]:
        raise ValueError("Prediction totals differ")
    if policy["rules"]["ownership"]:
        context.execute("CREATE INDEX edge_score ON edges(target,score)")
        # Materialize removals before mutation: simultaneous ties and losers must
        # not become winners merely because a previous row was deleted.
        context.execute("CREATE TABLE losers AS SELECT e.target,e.anchor FROM edges e WHERE EXISTS "
            "(SELECT 1 FROM edges other WHERE other.target=e.target AND other.anchor!=e.anchor AND other.score>=e.score)")
        changes["removed_ownership"] = context.execute("SELECT COUNT(*) FROM losers").fetchone()[0]
        for country, n in context.execute("SELECT s.country,COUNT(*) FROM losers l JOIN s1 s ON s.id=l.anchor GROUP BY s.country"):
            country_changes[country]["removed_ownership"] += n
        context.execute("DELETE FROM edges WHERE (target,anchor) IN (SELECT target,anchor FROM losers)")
    context.execute("CREATE INDEX edges_anchor ON edges(anchor,target)")
    context.commit()
    with open(out / "matching_results.tsv.tmp", "w", encoding="utf-8", newline="") as stream:
        stream.write("\t".join(MATCH) + "\n")
        for row in db.execute("SELECT entity_id FROM records WHERE source=1 ORDER BY rid"):
            selected = [r[0] for r in context.execute("SELECT target FROM edges WHERE anchor=? ORDER BY target", (row[0],))]
            stream.write(row[0] + "\t" + ",".join(selected) + "\n")
    (out / "matching_results.tsv.tmp").replace(out / "matching_results.tsv")
    shutil.copyfile(original / "candidate_pairs.tsv", out / "candidate_pairs.tsv")
    if digest(out / "candidate_pairs.tsv") != prediction["hashes"]["candidate_pairs.tsv"]:
        raise ValueError("Candidate file was altered")
    context.close()
    db.close()
    shutil.copyfile(args.policy, out / "policy.json")
    save(out / "refinement.json", {"version": VERSION, "policy_sha256": digest(args.policy),
        "source_sha256": digest(__file__), "original_prediction_sha256": digest(original / "prediction.json"),
        "original_validation_sha256": digest(original / "validation.json"),
        "original_prediction": prediction, "anchors": count, "candidate_pairs": candidate_count,
        "changes": dict(changes), "country_changes": {c: dict(v) for c, v in country_changes.items()},
        "hashes": {n: digest(out / n) for n in ("matching_results.tsv", "candidate_pairs.tsv")},
        "checks": "Full source-order/ID/threshold/ledger validation; deletion-only; candidates byte-identical; frozen policy applied globally."})
    log(f"Refined outputs complete: {out}")


def audit(args):
    """Same-address differences are negative evidence only; never auto-learn filler."""
    if not 2 <= args.max_group <= 100:
        raise ValueError("max-group must be between 2 and 100")
    out = fresh(args.output)
    db = connect(Path(args.work) / "test.sqlite")
    context = build_context(db, out / "context.sqlite")
    db.close()
    frequencies = defaultdict(Counter)
    separators, substitutions = defaultdict(Counter), defaultdict(Counter)
    examples = defaultdict(list)
    counts = Counter()
    rows = context.execute("SELECT country,address,name FROM s1 WHERE address!='' AND name!='' ORDER BY country,address,name")
    for (country, address), group in itertools.groupby(rows, key=lambda x: x[:2]):
        # Bounded memory even for pathological very large shared addresses.
        names = list(itertools.islice((r[2] for r in group), args.max_group + 1))
        if len(names) > args.max_group:
            counts[f"{country}|oversized_groups_skipped"] += 1
            continue
        names = sorted(set(names))
        for name in names:
            frequencies[country].update(set(name.split()))
        if len(names) < 2:
            continue
        counts[f"{country}|shared_address_groups"] += 1
        for left, right in itertools.combinations(names, 2):
            a, b = set(left.split()), set(right.split())
            if len(a & b) < 1 or len(a ^ b) > 2:
                continue
            counts[f"{country}|near_name_pairs"] += 1
            separators[country].update(a ^ b)
            if len(a - b) == len(b - a) == 1:
                change = " <-> ".join(sorted([next(iter(a - b)), next(iter(b - a))]))
                substitutions[country][change] += 1
            if len(examples[country]) < 30:
                examples[country].append({"address": address, "left": left, "right": right,
                    "distinguishing_tokens": sorted(a ^ b)})
    context.close()
    save(out / "audit.json", {"test_manifest_sha256": digest(Path(args.work) / "test_manifest.json"),
        "source_sha256": digest(__file__), "max_group": args.max_group, "counts": dict(counts),
        "countries": {c: {"distinguishing_tokens": [{"token": t, "separating_pair_count": n,
            "containing_names_in_bounded_groups": frequencies[c][t]} for t, n in separators[c].most_common(150)],
            "repeated_substitutions": substitutions[c].most_common(100), "examples": examples[c]} for c in separators},
        "interpretation": ["Distinct S1 names at the same address supply evidence against indiscriminate token removal.",
            "Pair counts are not probabilities; pairs and co-located businesses are dependent.",
            "Frequent edits may be natural or generated. Frequency alone does not identify decoys.",
            "No filler words, positive labels, or France veto rules are automatically inferred.",
            "All countries are separate; full normalized names and exact normalized addresses only."]})
    log(f"Structural audit: {out / 'audit.json'}")


def package(args):
    """Extend a verified original BER package, retaining its model and reproduction assets."""
    output, original_zip = Path(args.output), Path(args.baseline_zip)
    meta, policy = read(output / "refinement.json"), read(output / "policy.json")
    if meta["source_sha256"] != digest(__file__) or meta["policy_sha256"] != digest(output / "policy.json"):
        raise ValueError("Refinement source/policy changed")
    for name, sha in meta["hashes"].items():
        if digest(output / name) != sha:
            raise ValueError(f"Refined output changed: {name}")
    study_path = Path(args.study) / "study.json"
    if read(study_path)["policy_sha256"] != meta["policy_sha256"]:
        raise ValueError("Study belongs to another policy")
    destination = Path(args.destination)
    if destination.exists():
        raise ValueError("Archive already exists; use a new destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = "code/business_entity_resolution/assets/"
    with zipfile.ZipFile(original_zip) as source:
        if len(set(source.namelist())) != len(source.namelist()):
            raise ValueError("Duplicate archive entries")
        prediction = json.loads(source.read(prefix + "prediction.json"))
        if prediction != meta["original_prediction"]:
            raise ValueError("Baseline archive belongs to a different prediction")
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            h = hashlib.sha256()
            with source.open("output/" + name) as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != prediction["hashes"][name]:
                raise ValueError("Baseline archive TSV hash mismatch")
        with zipfile.ZipFile(str(destination) + ".tmp", "w", zipfile.ZIP_DEFLATED, compresslevel=1) as target:
            for entry in source.infolist():
                if entry.filename in ("output/matching_results.tsv", "Documentation_template.md"):
                    continue
                name = entry.filename
                if name in (prefix + "prediction.json", prefix + "validation.json"):
                    name = name.replace("assets/", "assets/baseline_")
                with source.open(entry) as old, target.open(name, "w", force_zip64=True) as new:
                    shutil.copyfileobj(old, new, 4 * 1024 * 1024)
            target.write(output / "matching_results.tsv", "output/matching_results.tsv")
            for name in ("policy.json", "refinement.json"):
                target.write(output / name, "code/refinement/assets/" + name)
            target.write(study_path, "code/refinement/assets/study.json")
            for name in ("refine.py", "README.md", "test_refine.py"):
                target.write(Path(__file__).with_name(name), "code/refinement/" + name)
            original_doc = source.read("Documentation_template.md").decode("utf-8")
            target.writestr("Baseline_methodology.md", original_doc)
            target.writestr("Documentation_template.md", "# RestoreBuildRun: refined submission\n\n"
                "The preserved Baseline_methodology.md describes candidate retrieval and GPU tree scoring. "
                "Its global threshold creates provisional matches. The final submitted matches additionally use "
                f"the frozen deletion-only policy **{policy['name']}** recorded in code/refinement/assets/policy.json. "
                "No new candidates are introduced; candidate_pairs.tsv is byte-identical to the original. "
                "The original prediction/validation manifests are explicitly named baseline_* and describe only "
                "the provisional stage. Final hashes and validation are in code/refinement/assets/refinement.json.\n\n"
                "## Selection and limits\n\nOne ablation was selected on a deterministic half of the existing development "
                "sample and gated once on the other half. Both halves were previously used for the original threshold "
                "selection, so neither is an untouched holdout. Learned text rules apply only to represented countries. "
                "No France accuracy or 0.99 leaderboard score is claimed. Study metrics and removed true/false links "
                "are in study.json. Ownership, when selected, resolves competition globally and abstains on tied best scores.\n\n"
                "## Reproduction\n\nFollow the original model README to regenerate provisional predictions, then follow "
                "code/refinement/README.md to apply the bundled policy to that ledger. Supply the organizer dataset separately. "
                "The refinement uses Python's standard library. Rebuilding the trees can change scores; re-evaluate instead "
                "of claiming identical results after refitting. Clean reproduction and portal acceptance remain unverified.\n")
    Path(str(destination) + ".tmp").replace(destination)
    save(destination.with_suffix(".release.json"), {"sha256": digest(destination), "archive": destination.name,
        "baseline_archive_sha256": digest(original_zip), "refinement": meta, "portal_submitted": False,
        "clean_reproduction_verified": False})
    log(f"Packaged: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("study")
    p.add_argument("--work", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-gain", type=float, default=.0005)
    p.add_argument("--country-tolerance", type=float, default=.002)
    p.set_defaults(function=study)
    p = commands.add_parser("apply")
    for key in ("work", "run", "original", "policy", "output"):
        p.add_argument("--" + key, required=True)
    p.set_defaults(function=apply)
    p = commands.add_parser("audit")
    for key in ("work", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--max-group", type=int, default=32)
    p.set_defaults(function=audit)
    p = commands.add_parser("package")
    for key in ("baseline-zip", "output", "study", "destination"):
        p.add_argument("--" + key, required=True)
    p.set_defaults(function=package)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
