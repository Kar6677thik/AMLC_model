"""Bounded Windows-spawn workers for SQLite retrieval and feature extraction."""
from __future__ import annotations

import itertools
import multiprocessing
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .blocking import Blocker, Candidate
from .database import connect
from .features import FEATURES, pair_features
from .normalize import Record

_worker_db = _worker_blocker = None


def _initialize(path, cfg, fit):
    global _worker_db, _worker_blocker
    _worker_db = connect(path)
    _worker_blocker = Blocker(_worker_db, cfg, fit=fit)


def _extract(blocker, chunk):
    before = blocker.stats.copy()
    before_times = blocker.timings.copy()
    groups, vectors = [], []
    retrieval_time = feature_time = 0.0
    for anchor in chunk:
        started = time.perf_counter()
        candidates = blocker.retrieve(anchor)
        retrieval_time += time.perf_counter()-started
        started = time.perf_counter()
        vectors.extend(pair_features(anchor, c, len(candidates)) for c in candidates)
        feature_time += time.perf_counter()-started
        # Text is no longer needed downstream; avoid repeatedly pickling large addresses.
        compact = [Candidate(Record(c.record.rid, c.record.entity_id, c.record.source,
            c.record.country, "", ""), c.retrieval_score, c.key_hits) for c in candidates]
        groups.append((anchor, compact))
    matrix = np.asarray(vectors, dtype="float32").reshape((-1, len(FEATURES)))
    stats = {key: count-before.get(key, 0) for key, count in blocker.stats.items()
             if count != before.get(key, 0)}
    times = {"retrieval_worker_seconds": retrieval_time, "feature_worker_seconds": feature_time}
    times.update({key: value-before_times.get(key, 0) for key, value in blocker.timings.items()})
    return groups, matrix, stats, times


def _worker_extract(chunk):
    return _extract(_worker_blocker, chunk)


def feature_batches(db, selected, cfg, fit=False):
    size = cfg.get("feature_batch_anchors", 128)
    workers = cfg.get("feature_workers", 1)
    iterator = iter(selected)
    def chunks():
        while True:
            chunk = list(itertools.islice(iterator, size))
            if not chunk:
                break
            yield chunk
    if workers == 1:
        blocker = Blocker(db, cfg, fit=fit)
        for chunk in chunks():
            yield _extract(blocker, chunk)
        return
    path = db.execute("PRAGMA database_list").fetchone()[2]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize, initargs=(path, cfg, fit)) as pool:
        source, pending = iter(chunks()), deque()
        for chunk in itertools.islice(source, workers*2):
            pending.append(pool.submit(_worker_extract, chunk))
        while pending:
            # Ordered results keep shard/S1 order deterministic across worker counts.
            yield pending.popleft().result()
            chunk = next(source, None)
            if chunk is not None:
                pending.append(pool.submit(_worker_extract, chunk))
