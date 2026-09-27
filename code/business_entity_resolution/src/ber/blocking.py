from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

from rapidfuzz.fuzz import ratio

from .normalize import Record, folded, index_keys


@dataclass(frozen=True)
class Candidate:
    record: Record
    retrieval_score: float
    key_hits: int


class Blocker:
    def __init__(self, db, cfg, fit=False):
        self.db, self.cfg, self.fit = db, cfg, fit
        self.stats = Counter()

    @lru_cache(maxsize=100000)
    def frequency(self, key):
        row = self.db.execute("SELECT n FROM key_counts WHERE key=?", (key,)).fetchone()
        return row[0] if row else 0

    def _records(self, ids):
        ids = sorted(ids)
        for start in range(0, len(ids), 800):
            batch = ids[start:start+800]
            marks = ",".join("?" for _ in batch)
            sql = f"SELECT rid,entity_id,source,country,name,address FROM records WHERE rid IN ({marks})"
            if self.fit:
                sql += " AND rid NOT IN (SELECT rid FROM reserved_targets)"
            for row in self.db.execute(sql, batch):
                yield Record(*row)

    def retrieve(self, anchor):
        result = []
        self.stats["queries"] += 1
        for source in (2, 3):
            key_info = [(self.frequency(key), key, kind) for key, kind in index_keys(anchor, source, self.cfg)]
            live = sorted((n, key, kind) for n, key, kind in key_info if n)
            small = [item for item in live if item[0] <= self.cfg["posting_limit"]]
            # Exact/composite channels are prioritized; other channels prefer rare postings.
            small.sort(key=lambda item: (0 if item[2] in ("e", "n") else 1, item[0], item[1]))
            selected = small[:self.cfg["query_keys"]]
            pool = Counter()
            for _, key, _ in selected:
                for (rid,) in self.db.execute("SELECT rid FROM postings WHERE key=?", (key,)):
                    pool[rid] += 1
                    self.stats["posting_visits"] += 1
            oversized = [item for item in live if item[0] > self.cfg["posting_limit"]]
            self.stats["oversized_keys"] += len(oversized)
            # Never cut a common posting by ID order. Try bounded intersections instead.
            if not pool:
                bounded = [item for item in oversized if item[0] <= self.cfg["posting_limit"] * 50][:3]
                for (_, ka, _), (_, kb, _) in combinations(bounded, 2):
                    rows = self.db.execute("""SELECT a.rid FROM postings a JOIN postings b ON a.rid=b.rid
                        WHERE a.key=? AND b.key=? LIMIT ?""", (ka, kb, self.cfg["posting_limit"]+1)).fetchall()
                    self.stats["intersection_queries"] += 1
                    if len(rows) <= self.cfg["posting_limit"]:
                        for (rid,) in rows:
                            pool[rid] += 2
                    else:
                        self.stats["saturated_intersections"] += 1
            aname, aaddr = folded(anchor.name), folded(anchor.address)
            ranked = []
            for target in self._records(pool):
                name = ratio(aname, folded(target.name)) / 100 if aname and target.name else 0
                addr = ratio(aaddr, folded(target.address)) / 100 if aaddr and target.address else 0
                country_conflict = bool(anchor.country and target.country and anchor.country != target.country)
                score = .65 * name + .30 * addr + .05 * min(pool[target.rid], 3) / 3 - .10 * country_conflict
                ranked.append(Candidate(target, score, pool[target.rid]))
            ranked.sort(key=lambda c: (-c.retrieval_score, c.record.entity_id))
            self.stats["precap_records"] += len(ranked)
            if len(ranked) > self.cfg["candidates_per_source"]:
                self.stats["capped_source_lists"] += 1
            result.extend(ranked[:self.cfg["candidates_per_source"]])
        self.stats["empty_queries"] += not result
        self.stats["emitted_pairs"] += len(result)
        return result
