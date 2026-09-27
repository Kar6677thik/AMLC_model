from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .common import DEFAULTS, environment, log, read_json, save_json, sha256, source_hash
from .database import anchors, connect, truth_for
from .features import FEATURES, rule_score
from .parallel import feature_batches
from .trees import TreeModel, gpu_check


def train(work, run, cfg, mode="learned"):
    work, run = Path(work), Path(run)
    if run.exists():
        raise ValueError(f"Run directory already exists; use a new --run: {run}")
    gpu_report = gpu_check() if mode == "learned" and cfg.get("device", "cpu").startswith("cuda") else None
    run.mkdir(parents=True)
    metadata = {"config": cfg, "mode": mode, "features": FEATURES, "environment": environment(),
                "source_sha256": source_hash(), "train_manifest": read_json(work / "train_manifest.json")}
    if gpu_report:
        metadata["gpu_check"] = gpu_report
    save_json(run / "run.json", metadata)
    try:
        frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        # Exclude editable/local project entries; the shipped package is installed separately.
        lines = [line for line in frozen.splitlines() if not line.startswith(("-e ", "#")) and "restorebuildrun" not in line.lower()]
        (run / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except (OSError, subprocess.CalledProcessError):
        (run / "environment.txt").write_text("# pip freeze unavailable; resolve before release\n", encoding="utf-8")
    if mode == "rules":
        save_json(run / "training.json", {"mode": mode, "trained_parameters": 0})
        return

    started = time.perf_counter()
    db = connect(work / "train.sqlite")
    try:
        available = db.execute("SELECT COUNT(*) FROM partitions WHERE part='fit'").fetchone()[0]
        nanchors = min(cfg["train_anchors"], available)
        if not nanchors:
            raise ValueError("No fit anchors; dataset is too small or split invalid")
        capacity = nanchors * 2 * cfg["candidates_per_source"]
        x = np.lib.format.open_memmap(run / "train_x.npy", mode="w+", dtype="float32", shape=(capacity, len(FEATURES)))
        y = np.lib.format.open_memmap(run / "train_y.npy", mode="w+", dtype="int8", shape=(capacity,))
        w = np.lib.format.open_memmap(run / "train_w.npy", mode="w+", dtype="float32", shape=(capacity,))
        stats, timings = Counter(), Counter()
        offset = positives = seen = 0
        train_ids_path = run / "training_anchors.tsv"
        with train_ids_path.open("w", encoding="utf-8", newline="") as ids:
            ids.write("source1_entity_id\n")
            for groups, matrix, batch_stats, batch_times in feature_batches(db, anchors(db, "fit", nanchors), cfg, fit=True):
                stats.update(batch_stats); timings.update(batch_times)
                x[offset:offset+len(matrix)] = matrix
                for anchor, candidates in groups:
                    gold = truth_for(db, anchor.rid)
                    ids.write(anchor.entity_id + "\n")
                    labels = [c.record.rid in gold for c in candidates]
                    size = len(labels)
                    y[offset:offset+size] = labels
                    w[offset:offset+size] = 1/max(1, size)
                    positives += sum(labels); offset += size; seen += 1
                if seen % 1024 < cfg.get("feature_batch_anchors", 128) or seen == nanchors:
                    log(f"Training features: {seen:,}/{nanchors:,} anchors, {offset:,} pairs")
        x.flush(); y.flush(); w.flush()
        if positives == 0 or positives == offset:
            raise ValueError("Training needs both positive and negative retrieved pairs; inspect audit/blocking or increase sample")
        feature_seconds = time.perf_counter()-started
        log(f"Fitting {cfg.get('model_backend', 'lightgbm')} on {offset:,} pairs; device={cfg.get('device', 'cpu')}")
        model = TreeModel(cfg)
        fit_start = time.perf_counter()
        model.fit(x[:offset], y[:offset], w[:offset])
        fit_seconds = time.perf_counter()-fit_start
        temporary = run / ("temporary-"+model.filename)
        model.save(temporary)
        os.replace(temporary, run / model.filename)
        metadata["model_file"] = model.filename
        metadata["model_sha256"] = sha256(run / model.filename)
        save_json(run / "run.json", metadata)
        save_json(run / "training.json", {"anchors": seen, "pairs": offset, "positive_pairs": int(positives),
            "seconds": time.perf_counter()-started, "feature_wall_seconds": feature_seconds,
            "fit_seconds": fit_seconds, "worker_timings": dict(timings), "blocking": dict(stats),
            "feature_importance_gain": model.importance(), "backend": model.backend, "device": cfg.get("device", "cpu")})
        del x, y, w
        # Retain memmaps for reproducibility/debugging; they are not packaged.
    finally:
        db.close()


class Scorer:
    def __init__(self, run):
        self.run = Path(run)
        self.meta = read_json(self.run / "run.json")
        if self.meta["features"] != FEATURES:
            raise ValueError("Model feature schema differs from current code")
        if self.meta["source_sha256"] != source_hash():
            raise ValueError("Source changed since training; create a new run rather than mixing code/model versions")
        from importlib.metadata import version
        dependencies = ["numpy", "rapidfuzz", "lightgbm"]
        if self.meta["config"].get("model_backend") == "xgboost" and self.meta["mode"] == "learned":
            dependencies.append("xgboost")
        for dependency in dependencies:
            if version(dependency) != self.meta["environment"][dependency]:
                raise ValueError(f"{dependency} version changed; restore the run's environment.txt before inference")
        if not (self.run / "training.json").exists():
            raise ValueError("Training did not complete")
        self.cfg = {**DEFAULTS, **self.meta["config"]}
        self.model = None
        if self.meta["mode"] == "learned":
            filename = self.meta.get("model_file", "model.txt")
            if sha256(self.run / filename) != self.meta["model_sha256"]:
                raise ValueError("Model hash mismatch")
            self.model = TreeModel(self.cfg, self.run / filename)

    def batches(self, db, selected, fit=False):
        batch, matrices = [], []
        stats, timings = Counter(), Counter()
        started = time.perf_counter()
        def score_batch():
            matrix = np.concatenate(matrices, axis=0) if matrices else np.empty((0, len(FEATURES)), dtype="float32")
            tick = time.perf_counter()
            values = (self.model.predict(matrix) if self.model else [rule_score(v) for v in matrix]) if len(matrix) else []
            timings["model_wall_seconds"] += time.perf_counter()-tick
            if len(values) != len(matrix) or not np.isfinite(values).all():
                raise ValueError("Matcher returned missing or non-finite scores")
            offset = 0
            for anchor, candidates in batch:
                scores = [float(s) for s in values[offset:offset+len(candidates)]]
                offset += len(candidates)
                yield anchor, candidates, scores
        for groups, matrix, batch_stats, batch_times in feature_batches(db, selected, self.cfg, fit=fit):
            batch.extend(groups); matrices.append(matrix)
            stats.update(batch_stats); timings.update(batch_times)
            if len(batch) >= self.cfg["prediction_batch_anchors"]:
                yield from score_batch()
                batch.clear(); matrices.clear()
        if batch:
            yield from score_batch()
        self.blocking_stats = dict(stats)
        self.performance = {**dict(timings), "pipeline_wall_seconds": time.perf_counter()-started,
            "feature_workers": self.cfg["feature_workers"], "prediction_batch_anchors": self.cfg["prediction_batch_anchors"],
            "note": "Worker seconds are summed CPU work, not elapsed time. Pipeline wall includes consumer work."}
