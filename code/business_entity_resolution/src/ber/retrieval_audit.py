from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path

from .common import DEFAULTS, log, save_json
from .database import anchors, connect, truth_for
from .metrics import Metrics
from .parallel import feature_batches


def retrieval_audit(work, cfg, output, limit=1000):
    if limit < 1:
        raise ValueError("Audit limit must be positive")
    db = connect(Path(work) / "train.sqlite")
    # Identical held-out anchors and complete target index for every comparison.
    selected = list(anchors(db, "dev", limit))
    truth = {a.rid: truth_for(db, a.rid) for a in selected}
    results = []
    baseline = {**DEFAULTS, **{key: cfg[key] for key in
        ("seed", "name_tokens", "address_tokens", "name_grams", "feature_workers", "feature_batch_anchors")}}
    variants = [("baseline-v1-20", baseline)]
    variants += [(f"v2-{cap}", {**cfg, "candidates_per_source": cap}) for cap in (20, 32, 48)]
    try:
        for name, variant in variants:
            started = time.perf_counter()
            overall, countries, stats = Metrics(), defaultdict(Metrics), Counter()
            for groups, _, batch_stats, _ in feature_batches(db, iter(selected), variant):
                stats.update(batch_stats)
                for anchor, candidates in groups:
                    ids = {c.record.rid for c in candidates}
                    gold = truth[anchor.rid]
                    overall.add(gold, gold & ids, ids)
                    countries[anchor.country].add(gold, gold & ids, ids)
            report = {"name": name, "config": variant, "metrics": overall.report(),
                "countries": {key: metric.report() for key, metric in countries.items()},
                "seconds": time.perf_counter()-started, "blocking": dict(stats)}
            results.append(report)
            log(f"{name}: oracle={overall.report()['oracle_macro_f05']:.6f}, pair recall={overall.report()['candidate_micro_recall']:.4f}")
    finally:
        db.close()
    result = {"anchors": len(selected), "partition": "dev", "results": results,
        "note": "Oracle scores, not trained-model scores. Includes feature computation in timing. No validation positives were injected."}
    save_json(output, result)
    return result
