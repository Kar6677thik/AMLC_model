from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import time

from rapidfuzz.fuzz import ratio, token_set_ratio

from .normalize import Record, folded, index_keys
from .text_views import view, ranking_name, ranking_address


@dataclass(frozen=True)
class Candidate:
    record: Record
    retrieval_score: float
    key_hits: int


class Blocker:
    def __init__(self, db, cfg, fit=False):
        self.db, self.cfg, self.fit = db, cfg, fit
        self.stats = Counter()
        self.timings = Counter()
        self.count_cache = OrderedDict()

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
        if self.cfg.get("retrieval_version", "v1") in ("v2", "v3"):
            return self._retrieve_v2(anchor)
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

    def _counts(self, keys):
        missing = [key for key in keys if key not in self.count_cache]
        for start in range(0, len(missing), 800):
            batch = missing[start:start+800]
            sql = "SELECT key,n FROM key_counts WHERE key IN (" + ",".join("?" for _ in batch) + ")"
            values = dict(self.db.execute(sql, batch))
            for key in batch:
                self.count_cache[key] = values.get(key, 0)
        result = {key: self.count_cache[key] for key in keys}
        for key in keys:
            self.count_cache.move_to_end(key)
        while len(self.count_cache) > 100000:
            self.count_cache.popitem(last=False)
        return result

    def _retrieve_v2(self, anchor):
        self.stats["queries"] += 1
        result = []
        an, aa = view(anchor.name), view(anchor.address)
        adaptive = self.cfg.get("retrieval_version") == "v3"
        query_cfg = dict(self.cfg)
        expansion = self.cfg.get("query_expansion", 3)
        for key in ("name_tokens", "address_tokens", "name_grams"):
            query_cfg[key] *= expansion
        for source in (2, 3):
            tick = time.perf_counter()
            keys = dict(index_keys(anchor, source, query_cfg))
            counts = self._counts(keys)
            live = sorted((counts[k], k, kind) for k, kind in keys.items() if counts[k])
            small = [item for item in live if item[0] <= self.cfg["posting_limit"]]
            # Channel diversity prevents rare accidental grams from exhausting the query budget.
            selected, used = [], set()
            def take(items, quota):
                for item in items:
                    if item[1] not in used and quota and len(selected) < self.cfg["query_keys"]:
                        selected.append(item); used.add(item[1]); quota -= 1
            if adaptive:
                # Round-robin coverage still reaches grams with a four-key budget.
                channels = [tuple(x for x in small if x[2] in kinds)
                            for kinds in (("e", "n"), ("a", "i"), ("t",), ("g",))]
                for _ in range(self.cfg["query_keys"]):
                    previous_count = len(selected)
                    for channel in channels:
                        take(channel, 1)
                    if len(selected) >= self.cfg["query_keys"] or len(selected) == previous_count:
                        break
            else:
                take((x for x in small if x[2] in ("e", "n")), 3)
                take((x for x in small if x[2] in ("a", "i")), 2)
                take((x for x in small if x[2] == "t"), 3)
                take((x for x in small if x[2] == "g"), 2)
            take(small, self.cfg["query_keys"])
            pool = Counter()
            if selected:
                sql = "SELECT key,rid FROM postings WHERE key IN (" + ",".join("?" for _ in selected) + ")"
                for _, rid in self.db.execute(sql, [x[1] for x in selected]):
                    pool[rid] += 1
                    self.stats["posting_visits"] += 1
            self.timings["retrieval_keys_postings_worker_seconds"] += time.perf_counter()-tick
            ranked = self._rank(pool, an, aa, anchor) if adaptive else []
            needs_intersections = (not adaptive or len(ranked) < self.cfg.get("intersection_min_pool", 8)
                or max((row[0].retrieval_score for row in ranked), default=0) < self.cfg.get("intersection_min_score", .70))
            wide = [x for x in live if self.cfg["posting_limit"] < x[0] <= self.cfg["posting_limit"]*50]
            self.stats["oversized_keys"] += sum(n > self.cfg["posting_limit"] for n, _, _ in live)
            pairs = list(combinations(wide[:5], 2)) if needs_intersections else []
            if adaptive:
                self.stats["intersection_fallback_sources" if needs_intersections else "intersection_skipped_sources"] += 1
            pairs.sort(key=lambda p: (p[0][2] == p[1][2], p[0][0] + p[1][0]))
            extra = Counter()
            tick = time.perf_counter()
            for left, right in pairs[:self.cfg.get("intersection_budget", 2)]:
                rows = self.db.execute("""SELECT a.rid FROM postings a JOIN postings b ON a.rid=b.rid
                    WHERE a.key=? AND b.key=? LIMIT ?""", (left[1], right[1], self.cfg["posting_limit"]+1)).fetchall()
                self.stats["intersection_queries"] += 1
                if len(rows) <= self.cfg["posting_limit"]:
                    for (rid,) in rows:
                        extra[rid] += 2
                else:
                    self.stats["saturated_intersections"] += 1
            self.timings["retrieval_intersections_worker_seconds"] += time.perf_counter()-tick
            if not adaptive or extra:
                pool.update(extra)
                if adaptive:
                    # Re-rank only changed/new targets, preserving identical full-union scores.
                    ranked = [row for row in ranked if row[0].record.rid not in extra]
                    ranked.extend(self._rank({rid: pool[rid] for rid in extra}, an, aa, anchor))
                else:
                    ranked = self._rank(pool, an, aa, anchor)
            tick = time.perf_counter()
            ranked.sort(key=lambda x: (-x[0].retrieval_score, x[0].record.entity_id))
            self.stats["precap_records"] += len(ranked)
            cap = self.cfg["candidates_per_source"]
            self.stats["capped_source_lists"] += len(ranked) > cap
            keep, seen = [], set()
            quota = max(1, cap//8)
            def preserve(items, budget):
                for row in items:
                    if len(keep) >= cap or budget <= 0:
                        break
                    candidate = row[0]
                    if candidate.record.rid not in seen:
                        keep.append(candidate); seen.add(candidate.record.rid); budget -= 1
            preserve(ranked, max(1, cap-2*quota))
            preserve(sorted((r for r in ranked if r[1] >= .4), key=lambda r: (-r[2], -r[1], r[0].record.entity_id)), quota)
            preserve(sorted(ranked, key=lambda r: (-r[1], -r[2], r[0].record.entity_id)), quota)
            preserve(ranked, cap)
            keep.sort(key=lambda c: (-c.retrieval_score, c.record.entity_id))
            result.extend(keep)
            self.timings["retrieval_selection_worker_seconds"] += time.perf_counter()-tick
        self.stats["empty_queries"] += not result
        self.stats["emitted_pairs"] += len(result)
        return result

    def _rank(self, pool, an, aa, anchor):
        tick = time.perf_counter()
        records = list(self._records(pool))
        self.timings["retrieval_record_fetch_worker_seconds"] += time.perf_counter()-tick
        tick = time.perf_counter()
        ranked = []
        for target in records:
            if self.cfg.get("ranking_backend", "compact") == "legacy":
                bn, ba = view(target.name), view(target.address)
                name, name_ordered = bn.text, bn.sorted_text
                address, address_ordered, digits = ba.text, ba.sorted_text, ba.digits
            else:
                name, name_ordered = ranking_name(target.name)
                address, address_ordered, digits = ranking_address(target.address)
            name_score = (max(ratio(an.text, name), ratio(an.sorted_text, name_ordered)) / 100
                          if an.text and name else 0)
            address_score = token_set_ratio(aa.text, address)/100 if aa.text and address else 0
            addr_order = ratio(aa.sorted_text, address_ordered)/100 if aa.text and address else 0
            nums = len(aa.digits & digits)/len(aa.digits | digits) if aa.digits or digits else 0
            conflict = bool(anchor.country and target.country and anchor.country != target.country)
            score = .40*name_score + .30*address_score + .15*addr_order + .10*nums + .05*min(pool[target.rid], 3)/3
            score -= .20*conflict
            ranked.append((Candidate(target, score, pool[target.rid]), name_score, address_score + .2*nums))
        self.timings["retrieval_rank_worker_seconds"] += time.perf_counter()-tick
        return ranked
