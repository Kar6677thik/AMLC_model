from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .blocking import Blocker
from .common import environment, log, read_json, save_json, sha256, source_hash
from .database import anchors, connect, truth_for
from .features import FEATURES, pair_features, rule_score


def train(work, run, cfg, mode="learned"):
    work, run = Path(work), Path(run)
    if run.exists():
        raise ValueError(f"Run directory already exists; use a new --run: {run}")
    run.mkdir(parents=True)
    metadata = {"config": cfg, "mode": mode, "features": FEATURES, "environment": environment(),
                "source_sha256": source_hash(), "train_manifest": read_json(work / "train_manifest.json")}
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

    import lightgbm as lgb
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
        blocker = Blocker(db, cfg, fit=True)
        offset = positives = seen = 0
        train_ids_path = run / "training_anchors.tsv"
        with train_ids_path.open("w", encoding="utf-8", newline="") as ids:
            ids.write("source1_entity_id\n")
            for anchor in anchors(db, "fit", nanchors):
                candidates = blocker.retrieve(anchor)
                gold = truth_for(db, anchor.rid)
                ids.write(anchor.entity_id + "\n")
                for candidate in candidates:
                    x[offset] = pair_features(anchor, candidate, len(candidates))
                    label = candidate.record.rid in gold
                    y[offset] = label
                    w[offset] = 1 / max(1, len(candidates))
                    positives += label
                    offset += 1
                seen += 1
                if seen % 1000 == 0:
                    log(f"Training features: {seen:,}/{nanchors:,} anchors, {offset:,} pairs")
        x.flush(); y.flush(); w.flush()
        if positives == 0 or positives == offset:
            raise ValueError("Training needs both positive and negative retrieved pairs; inspect audit/blocking or increase sample")
        log(f"Fitting LightGBM on {offset:,} pairs ({positives:,} positives)")
        dataset = lgb.Dataset(x[:offset], label=y[:offset], weight=w[:offset], feature_name=FEATURES)
        model = lgb.train({"objective": "binary", "learning_rate": cfg["learning_rate"],
            "num_leaves": cfg["num_leaves"], "min_data_in_leaf": cfg["min_child_samples"],
            "num_threads": cfg["threads"], "seed": cfg["seed"], "deterministic": True,
            "force_col_wise": True, "verbosity": -1}, dataset, num_boost_round=cfg["trees"])
        temporary = run / "model.tmp.txt"
        model.save_model(str(temporary))
        os.replace(temporary, run / "model.txt")
        metadata["model_sha256"] = sha256(run / "model.txt")
        save_json(run / "run.json", metadata)
        save_json(run / "training.json", {"anchors": seen, "pairs": offset, "positive_pairs": int(positives),
            "seconds": time.perf_counter()-started, "blocking": dict(blocker.stats),
            "feature_importance_gain": dict(zip(FEATURES, map(float, model.feature_importance(importance_type="gain"))))})
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
        for dependency in ("numpy", "rapidfuzz", "lightgbm"):
            if version(dependency) != self.meta["environment"][dependency]:
                raise ValueError(f"{dependency} version changed; restore the run's environment.txt before inference")
        if not (self.run / "training.json").exists():
            raise ValueError("Training did not complete")
        self.cfg = self.meta["config"]
        self.model = None
        if self.meta["mode"] == "learned":
            import lightgbm as lgb
            if sha256(self.run / "model.txt") != self.meta["model_sha256"]:
                raise ValueError("Model hash mismatch")
            self.model = lgb.Booster(model_file=str(self.run / "model.txt"))

    def batches(self, db, selected, fit=False):
        blocker = Blocker(db, self.cfg, fit=fit)
        batch, vectors = [], []
        def score_batch():
            if vectors:
                values = (self.model.predict(np.asarray(vectors, dtype="float32"), num_threads=self.cfg["threads"])
                          if self.model else [rule_score(v) for v in vectors])
            else:
                values = []
            if len(values) != len(vectors) or not np.isfinite(values).all():
                raise ValueError("Matcher returned missing or non-finite scores")
            offset = 0
            for anchor, candidates in batch:
                scores = [float(s) for s in values[offset:offset+len(candidates)]]
                offset += len(candidates)
                yield anchor, candidates, scores
        for anchor in selected:
            candidates = blocker.retrieve(anchor)
            batch.append((anchor, candidates))
            vectors.extend(pair_features(anchor, c, len(candidates)) for c in candidates)
            if len(batch) >= self.cfg["prediction_batch_anchors"]:
                yield from score_batch()
                batch.clear(); vectors.clear()
        if batch:
            yield from score_batch()
        self.blocking_stats = dict(blocker.stats)
