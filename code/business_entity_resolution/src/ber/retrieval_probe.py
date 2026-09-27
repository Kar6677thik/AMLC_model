"""Small, paired retrieval experiments before spending time on model training."""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from pathlib import Path

from .common import load_config, log, read_json, save_json, sha256, source_hash
from .database import anchors, connect, truth_for
from .metrics import Metrics
from .normalize import Record
from .parallel import feature_batches


def variants(cfg):
    control = dict(cfg, retrieval_version="v2", query_keys=10, posting_limit=400,
                   query_expansion=3, intersection_budget=2, candidates_per_source=32)
    adaptive = dict(control, retrieval_version="v3")
    balanced = dict(adaptive, query_keys=6, posting_limit=200, query_expansion=2,
                    intersection_budget=1)
    lean = dict(balanced, query_keys=4, posting_limit=120, candidates_per_source=20)
    return [("v2-control-32", control), ("adaptive-32", adaptive),
            ("balanced-32", balanced), ("balanced-20", dict(balanced, candidates_per_source=20)),
            ("lean-20", lean)]


def measure(db, selected, cfg, truth=None):
    started = time.perf_counter()
    overall, countries = Metrics(), defaultdict(Metrics)
    stats, times, coverage = Counter(), Counter(), Counter()
    count = pairs = 0
    for groups, _, batch_stats, batch_times in feature_batches(db, iter(selected), cfg):
        stats.update(batch_stats); times.update(batch_times)
        for anchor, candidates in groups:
            count += 1; pairs += len(candidates)
            coverage[anchor.country or "<missing>"] += 1
            if truth is not None:
                ids = {candidate.record.rid for candidate in candidates}
                gold = truth[anchor.rid]
                overall.add(gold, gold & ids, ids)
                countries[anchor.country or "<missing>"].add(gold, gold & ids, ids)
    elapsed = time.perf_counter()-started
    result = {"anchors": count, "pairs": pairs, "country_coverage": dict(coverage),
              "seconds": elapsed, "queries_per_second": count/max(elapsed, 1e-9),
              "blocking": dict(stats), "worker_timings": dict(times)}
    if truth is not None:
        result.update(metrics=overall.report(), countries={k: m.report() for k, m in countries.items()})
    return result


def select_variant(results, min_rate=150.0, oracle_loss=.005, country_loss=.01, recall_loss=.01):
    """Use development labels for quality, unlabeled test text only for timing."""
    reference = results[0]["dev"]
    eligible = []
    for result in results[1:]:
        reasons = []
        if result["test"]["queries_per_second"] < min_rate:
            reasons.append("Test retrieval/feature throughput below minimum")
        dev = result["dev"]
        for key, tolerance in (("oracle_macro_f05", oracle_loss), ("candidate_micro_recall", recall_loss)):
            current, base = dev["metrics"][key], reference["metrics"][key]
            if current is None or base is None or current < base-tolerance:
                reasons.append(f"Development {key} loss exceeds {tolerance}")
        for country, base in reference["countries"].items():
            current = dev["countries"].get(country)
            if current is None or current["oracle_macro_f05"] < base["oracle_macro_f05"]-country_loss:
                reasons.append(f"{country} oracle loss exceeds {country_loss}")
            elif (base["candidate_micro_recall"] is not None and
                  (current["candidate_micro_recall"] is None or
                   current["candidate_micro_recall"] < base["candidate_micro_recall"]-recall_loss)):
                reasons.append(f"{country} candidate recall loss exceeds {recall_loss}")
        result["rejection_reasons"] = reasons
        if not reasons:
            eligible.append(result)
    if not eligible:
        return None
    # All survivors satisfy explicit quality tolerances; prioritize finishing on time.
    return max(eligible, key=lambda r: (r["test"]["queries_per_second"], -r["dev"]["metrics"]["mean_candidates"]))


def verify_probe(work, output):
    work, output = Path(work), Path(output)
    report = read_json(output / "probe.json")
    if not report.get("complete") or not report.get("recommended"):
        raise ValueError("Probe found no qualifying configuration; inspect probe.json before training")
    if report["source_sha256"] != source_hash():
        raise ValueError("Source changed since probe; use a fresh probe directory")
    if sha256(output / "selected_config.json") != report["selected_config_sha256"]:
        raise ValueError("Selected config changed since probe; rerun the probe")
    for split in ("train", "test"):
        if report["index_signatures"][split] != read_json(work / f"{split}_manifest.json")["signature"]:
            raise ValueError(f"{split} index differs from probe")
    return load_config(output / "selected_config.json")


def retrieval_probe(work, cfg, output, dev_limit=2000, test_limit=3000, min_rate=150.0):
    if dev_limit < 1 or test_limit < 1 or not math.isfinite(min_rate) or min_rate <= 0:
        raise ValueError("Probe sample sizes and minimum throughput must be positive")
    work, output = Path(work), Path(output)
    if output.exists():
        raise ValueError("Probe directory already exists; use a new output directory")
    signatures = {split: read_json(work / f"{split}_manifest.json")["signature"] for split in ("train", "test")}
    for split, signature in signatures.items():
        for key, value in signature["index_config"].items():
            if cfg[key] != value:
                raise ValueError(f"{split} index configuration mismatch: {key}")
    train_db = connect(work / "train.sqlite")
    test_db = None
    try:
        test_db = connect(work / "test.sqlite")
        dev = list(anchors(train_db, "dev", dev_limit))
        truth = {anchor.rid: truth_for(train_db, anchor.rid) for anchor in dev}
        total = test_db.execute("SELECT COUNT(*) FROM records WHERE source=1").fetchone()[0]
        stride = max(1, math.ceil(total/test_limit))
        test = [Record(*row) for row in test_db.execute("""SELECT rid,entity_id,source,country,name,address
            FROM records WHERE source=1 AND (rid-1)%?=0 ORDER BY rid LIMIT ?""", (stride, test_limit))]
        if not dev or not test:
            raise ValueError("Probe requires nonempty development and test samples")
        report = {"complete": False, "source_sha256": source_hash(), "index_signatures": signatures,
                  "minimum_queries_per_second": min_rate, "results": [], "recommended": None,
                  "quality_tolerances": {"oracle_loss": .005, "country_oracle_loss": .01, "recall_loss": .01},
                  "note": "Paired development oracle comparison, not trained-model accuracy. Test timing includes retrieval/features/process startup, excludes model, output and validation. Later variants may benefit from OS cache warming; confirm with full model benchmark."}
        for name, variant in variants(cfg):
            log(f"Retrieval probe: {name}, dev={len(dev):,}, test={len(test):,}")
            result = {"name": name, "config": variant, "dev": measure(train_db, dev, variant, truth),
                      "test": measure(test_db, test, variant)}
            rate = result["test"]["queries_per_second"]
            result["test"]["extrapolated_retrieval_feature_seconds"] = total/rate
            report["results"].append(result)
            save_json(output / "probe.json", report)
            log(f"{name}: oracle={result['dev']['metrics']['oracle_macro_f05']:.6f}, test={rate:.2f} anchors/s")
        selected = select_variant(report["results"], min_rate)
        if selected:
            report["recommended"] = selected["name"]
            save_json(output / "selected_config.json", selected["config"])
            report["selected_config_sha256"] = sha256(output / "selected_config.json")
        report["complete"] = True
        save_json(output / "probe.json", report)
        log(f"Selected: {report['recommended'] or 'NONE; inspect rejection reasons before training'}")
        return report
    finally:
        train_db.close()
        if test_db is not None:
            test_db.close()
