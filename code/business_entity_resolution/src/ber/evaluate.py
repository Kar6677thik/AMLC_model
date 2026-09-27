from __future__ import annotations

import bisect
import json
import time
from collections import defaultdict
from pathlib import Path

from .common import log, read_json, save_json, sha256
from .database import anchors, connect, truth_for
from .metrics import Metrics
from .model import Scorer


def evaluate(work, run, partition="dev", limit=None):
    work, run = Path(work), Path(run)
    scorer = Scorer(run)
    current_manifest = read_json(work / "train_manifest.json")
    if current_manifest["signature"] != scorer.meta["train_manifest"]["signature"]:
        raise ValueError("Training index differs from this run's dataset/split signature")
    if partition not in ("dev", "holdout"):
        raise ValueError("Evaluate only dev or holdout; training metrics cannot select a release")
    report_path = run / f"{partition}_report.json"
    if report_path.exists():
        raise ValueError(f"{partition} was already evaluated for this run; preserve its report and use a new run for experiments")
    if limit is None:
        limit = scorer.cfg["dev_anchors"] if partition == "dev" else 0
    thresholds = sorted(set([i / 100 for i in range(5, 100)] + [.001, .005, .01, .02, .025, .995, .999, 1.000001]))
    if partition == "holdout":
        thresholds = [read_json(run / "decision.json")["threshold"]]
    totals = [0.0] * len(thresholds)
    prediction_counts = [0] * len(thresholds)
    cache = run / f"{partition}_scores.jsonl"
    db = connect(work / "train.sqlite")
    started = time.perf_counter()
    n = 0
    try:
        with cache.open("w", encoding="utf-8", newline="") as stream:
            for anchor, candidates, scores in scorer.batches(db, anchors(db, partition, limit)):
                gold = truth_for(db, anchor.rid)
                ordered = sorted(zip(scores, [c.record.rid for c in candidates]), reverse=True)
                neg_scores, prefix = [], [0]
                for score, rid in ordered:
                    neg_scores.append(-score)
                    prefix.append(prefix[-1] + (rid in gold))
                for idx, threshold in enumerate(thresholds):
                    count = bisect.bisect_right(neg_scores, -threshold)
                    tp = prefix[count]
                    totals[idx] += (5*tp/(4*count+len(gold)) if gold else float(count == 0))
                    prediction_counts[idx] += count
                stream.write(json.dumps({"source1_entity_id": anchor.entity_id, "country": anchor.country,
                    "truth": sorted(gold), "targets": [c.record.rid for c in candidates], "scores": scores}) + "\n")
                n += 1
                if n % 1000 == 0:
                    log(f"Evaluated {n:,} {partition} anchors")
    finally:
        db.close()
    if not n:
        raise ValueError(f"No {partition} anchors; cannot calibrate/evaluate")
    best = max(range(len(thresholds)), key=lambda i: (totals[i], -prediction_counts[i], thresholds[i]))
    threshold = thresholds[best]
    overall, slices = Metrics(), defaultdict(Metrics)
    with cache.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            prediction = [rid for rid, score in zip(row["targets"], row["scores"]) if score >= threshold]
            args = row["truth"], prediction, row["targets"]
            overall.add(*args)
            slices["country:" + (row["country"] or "<missing>")].add(*args)
            slices["multiplicity:" + ("0" if not row["truth"] else "1" if len(row["truth"]) == 1 else "2+")].add(*args)
    report = {"partition": partition, "requested_limit": limit, "threshold": threshold,
        "metrics": overall.report(), "slices": {k: v.report() for k, v in slices.items()},
        "seconds": time.perf_counter()-started, "blocking": scorer.blocking_stats,
        "performance": scorer.performance,
        "threshold_search": [{"threshold": t, "macro_f05": s/n, "predicted_pairs": p}
            for t, s, p in zip(thresholds, totals, prediction_counts)],
        "scores_sha256": sha256(cache)}
    if partition == "dev":
        save_json(run / "decision.json", {"threshold": threshold, "selected_on": "dev",
            "anchors": n, "macro_f05": totals[best]/n, "comparison": "score >= threshold"})
    save_json(report_path, report)
    log(f"{partition}: macro F0.5={overall.score/n:.6f}, threshold={threshold}, anchors={n:,}")
    return report
