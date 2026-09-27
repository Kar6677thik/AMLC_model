from __future__ import annotations

import gzip
import math
import os
import shutil
import time
from pathlib import Path

from .common import log, read_json, save_json, sha256
from .database import anchors, connect
from .io import CANDIDATE_HEADER, MATCH_HEADER, write_header, write_row
from .model import Scorer

OUTPUT_NAMES = ("matching_results.tsv", "candidate_pairs.tsv", "scoring_ledger.tsv.gz")
LEDGER_HEADER = ["source1_entity_id", "candidate_entity_ids", "scores"]


def predict(work, run, output):
    work, run, output = Path(work), Path(run), Path(output)
    scorer = Scorer(run)
    decision = read_json(run / "decision.json")
    threshold = decision["threshold"]
    test_manifest = read_json(work / "test_manifest.json")
    signature = {"run_sha256": sha256(run / "run.json"), "decision_sha256": sha256(run / "decision.json"),
                 "test_signature": test_manifest["signature"]}
    output.mkdir(parents=True, exist_ok=True)
    identity = output / "identity.json"
    if identity.exists():
        if read_json(identity) != signature:
            raise ValueError("Output directory belongs to another model/data/decision; use a new directory")
    else:
        if any(output.iterdir()):
            raise ValueError("Output directory is not empty and has no identity manifest")
        save_json(identity, signature)
    shard_root = output / ".shards"
    shard_root.mkdir(exist_ok=True)
    completed = []
    for path in sorted(shard_root.glob("*/complete.json")):
        if path.parent.name != f"{len(completed):06d}":
            raise ValueError("Prediction shards are not contiguous")
        meta = read_json(path)
        for name, digest in meta["hashes"].items():
            if sha256(path.parent / name) != digest:
                raise ValueError(f"Corrupted prediction shard: {path.parent / name}")
        completed.append(meta)
    last = completed[-1]["last_rid"] if completed else 0
    seen = sum(item["anchors"] for item in completed)
    total_pairs = sum(item["pairs"] for item in completed)
    log(f"Predicting test; resuming after {seen:,} completed anchors")
    db = connect(work / "test.sqlite")
    expected = db.execute("SELECT COUNT(*) FROM records WHERE source=1").fetchone()[0]
    started = time.perf_counter()
    handles = None
    shard_count = shard_pairs = 0
    shard_dir = None

    def finish_shard():
        nonlocal handles, shard_count, shard_pairs
        for handle in handles:
            handle.close()
        for name in OUTPUT_NAMES:
            os.replace(shard_dir / (name + ".tmp"), shard_dir / name)
        metadata = {"anchors": shard_count, "pairs": shard_pairs, "last_rid": last,
                    "hashes": {name: sha256(shard_dir / name) for name in OUTPUT_NAMES}}
        save_json(shard_dir / "complete.json", metadata)
        completed.append(metadata)
        handles = None
        shard_count = shard_pairs = 0

    try:
        for anchor, candidates, scores in scorer.batches(db, anchors(db, after=last)):
            if handles is None:
                shard_dir = shard_root / f"{len(completed):06d}"
                shard_dir.mkdir(exist_ok=True)
                handles = [
                    (shard_dir / (OUTPUT_NAMES[0] + ".tmp")).open("w", encoding="utf-8", newline=""),
                    (shard_dir / (OUTPUT_NAMES[1] + ".tmp")).open("w", encoding="utf-8", newline=""),
                    gzip.open(shard_dir / (OUTPUT_NAMES[2] + ".tmp"), "wt", encoding="utf-8", newline="", compresslevel=1),
                ]
                if not completed:
                    for handle, header in zip(handles, (MATCH_HEADER, CANDIDATE_HEADER, LEDGER_HEADER)):
                        write_header(handle, header)
            pairs = sorted((c.record.entity_id, score) for c, score in zip(candidates, scores))
            if len(pairs) != len(candidates) or any(not math.isfinite(s) for _, s in pairs):
                raise ValueError("Invalid matcher scores")
            candidate_ids = [entity_id for entity_id, _ in pairs]
            matched_ids = [entity_id for entity_id, score in pairs if score >= threshold]
            write_row(handles[0], anchor.entity_id, matched_ids)
            write_row(handles[1], anchor.entity_id, candidate_ids)
            handles[2].write(anchor.entity_id + "\t" + ",".join(candidate_ids) + "\t" + ",".join(repr(s) for _, s in pairs) + "\n")
            last = anchor.rid
            seen += 1
            shard_count += 1
            shard_pairs += len(pairs)
            total_pairs += len(pairs)
            if seen % 1000 == 0:
                log(f"Prediction: {seen:,}/{expected:,} anchors, {total_pairs:,} pairs")
            if shard_count >= scorer.cfg["prediction_shard_anchors"]:
                finish_shard()
        if handles is not None:
            finish_shard()
    finally:
        db.close()
        if handles is not None:
            for handle in handles:
                handle.close()
    if seen != expected:
        raise ValueError(f"Incomplete output: {seen} vs {expected} expected anchors")
    if not completed:
        raise ValueError("Test S1 is empty; investigate the input")
    # Concatenated gzip members are supported by gzip.open; only shard zero has a header.
    for name in OUTPUT_NAMES:
        temporary = output / (name + ".tmp")
        with temporary.open("wb") as dest:
            for idx in range(len(completed)):
                with (shard_root / f"{idx:06d}" / name).open("rb") as src:
                    shutil.copyfileobj(src, dest, length=4*1024*1024)
        os.replace(temporary, output / name)
    report = {"identity": signature, "anchors": seen, "scored_pairs": total_pairs, "threshold": threshold,
        "invocation_seconds": time.perf_counter()-started, "shards": len(completed),
        "blocking_this_invocation": getattr(scorer, "blocking_stats", {}),
        "performance_this_invocation": getattr(scorer, "performance", {}),
        "hashes": {name: sha256(output / name) for name in OUTPUT_NAMES}}
    save_json(output / "prediction.json", report)
    log(f"Both outputs complete: {seen:,} anchors, {total_pairs:,} scored pairs")
    return report
