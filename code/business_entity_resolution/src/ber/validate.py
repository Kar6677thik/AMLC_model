from __future__ import annotations

import csv
import gzip
import itertools
import math
import time
from collections import Counter
from pathlib import Path

from .common import log, read_json, save_json, sha256
from .database import connect
from .io import CANDIDATE_HEADER, MATCH_HEADER, id_list, tsv_rows
from .predict import LEDGER_HEADER, OUTPUT_NAMES


def ledger_rows(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t", strict=True)
        if next(reader, None) != LEDGER_HEADER:
            raise ValueError("Scoring ledger header is invalid")
        for row in reader:
            if len(row) != 3:
                raise ValueError("Scoring ledger must have three columns")
            yield row


def validate(work, output):
    work, output = Path(work), Path(output)
    started = time.perf_counter()
    manifest = read_json(output / "prediction.json")
    if manifest["identity"]["test_signature"] != read_json(work / "test_manifest.json")["signature"]:
        raise ValueError("Outputs and validation test index differ")
    for name in OUTPUT_NAMES:
        if sha256(output / name) != manifest["hashes"][name]:
            raise ValueError(f"Output was modified or corrupted: {name}")
    threshold = manifest["threshold"]
    db = connect(work / "test.sqlite")
    rows = pairs = matched_pairs = 0
    countries, empty = Counter(), Counter()
    try:
        expected = db.execute("SELECT entity_id,country FROM records WHERE source=1 ORDER BY rid")
        streams = (expected, tsv_rows(output / OUTPUT_NAMES[0], MATCH_HEADER),
                   tsv_rows(output / OUTPUT_NAMES[1], CANDIDATE_HEADER), ledger_rows(output / OUTPUT_NAMES[2]))
        for item in itertools.zip_longest(*streams):
            if any(row is None for row in item):
                raise ValueError("S1 coverage or row counts differ across input, outputs, and ledger")
            original, matching, candidate, ledger = item
            sid, country = original
            if not (sid == matching[0] == candidate[0] == ledger[0]):
                raise ValueError(f"S1 rows are missing, duplicated, reordered, or invalid near {sid}")
            mids, cids, lids = id_list(matching[1]), id_list(candidate[1]), id_list(ledger[1])
            if cids != sorted(cids) or mids != sorted(mids):
                raise ValueError(f"Noncanonical list ordering for {sid}")
            if cids != lids:
                raise ValueError(f"Exported candidates differ from scored candidates for {sid}")
            if not set(mids) <= set(cids):
                raise ValueError(f"Matched targets not in candidates for {sid}")
            if any(not target.startswith(("S2-", "S3-")) for target in cids):
                raise ValueError(f"Invalid target prefix for {sid}")
            for start in range(0, len(cids), 800):
                chunk = cids[start:start+800]
                sql = "SELECT COUNT(*) FROM records WHERE source IN (2,3) AND entity_id IN (" + ",".join("?" for _ in chunk) + ")"
                if db.execute(sql, chunk).fetchone()[0] != len(chunk):
                    raise ValueError(f"Unknown target ID for {sid}")
            scores = [float(v) for v in ledger[2].split(",")] if ledger[2] else []
            if len(scores) != len(cids) or any(not math.isfinite(v) for v in scores):
                raise ValueError(f"Invalid score ledger for {sid}")
            selected = [target for target, score in zip(cids, scores) if score >= threshold]
            if selected != mids:
                raise ValueError(f"Predictions do not reproduce the frozen threshold for {sid}")
            rows += 1
            pairs += len(cids)
            matched_pairs += len(mids)
            countries[country or "<missing>"] += 1
            empty["matching"] += not mids
            empty["candidate"] += not cids
            if rows % 100000 == 0:
                log(f"Strict validation: {rows:,} anchors")
    finally:
        db.close()
    if rows != manifest["anchors"] or pairs != manifest["scored_pairs"]:
        raise ValueError("Manifest row/pair totals disagree with files")
    report = {"passed": True, "anchors": rows, "candidate_pairs": pairs, "matched_pairs": matched_pairs,
              "countries": dict(countries), "empty_rows": dict(empty), "seconds": time.perf_counter()-started,
              "hashes": manifest["hashes"], "checks": ["exact_s1_coverage", "target_membership", "unique_ids",
              "matched_subset", "scored_candidate_equality", "threshold_reproduction", "file_hashes"]}
    save_json(output / "validation.json", report)
    log(f"STRICT PASS: {rows:,} anchors, {pairs:,} candidates, {matched_pairs:,} matches")
    return report
