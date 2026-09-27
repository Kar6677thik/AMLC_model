from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path

from .common import log, save_json
from .database import connect
from .model import Scorer
from .normalize import Record


def benchmark(work, run, limit=1000):
    """Time evenly spaced S1 queries against the complete test index."""
    if limit < 1:
        raise ValueError("Benchmark limit must be positive")
    scorer = Scorer(run)
    db = connect(Path(work) / "test.sqlite")
    count, pairs, countries = 0, 0, Counter()
    started = time.perf_counter()
    try:
        total = db.execute("SELECT COUNT(*) FROM records WHERE source=1").fetchone()[0]
        stride = max(1, math.ceil(total/limit))
        rows = db.execute("""SELECT rid,entity_id,source,country,name,address FROM records
            WHERE source=1 AND (rid-1)%?=0 ORDER BY rid LIMIT ?""", (stride, limit))
        selected = (Record(*row) for row in rows)
        for anchor, candidates, _ in scorer.batches(db, selected):
            count += 1
            pairs += len(candidates)
            countries[anchor.country or "<missing>"] += 1
    finally:
        db.close()
    seconds = time.perf_counter()-started
    rate = count / max(seconds, 1e-9)
    report = {"sampled_anchors": count, "sampling": "evenly spaced by input row; inspect country coverage",
        "pairs": pairs, "countries": dict(countries), "seconds": seconds, "queries_per_second": rate,
        "total_test_anchors": total, "extrapolated_scoring_seconds": total/rate if rate else None,
        "conservative_scoring_seconds_2x": 2*total/rate if rate else None,
        "exclusions": "Does not include index build, full output writing, validation, packaging or upload.",
        "blocking": scorer.blocking_stats, "performance": scorer.performance,
        "backend": scorer.cfg["model_backend"], "device": scorer.cfg["device"]}
    save_json(Path(run) / "benchmark_test.json", report)
    log(f"Benchmark: {rate:.2f} queries/sec; scoring-only estimate {total/rate/3600 if rate else 0:.2f} hours (before 2x margin)")
    return report
