from __future__ import annotations

import itertools
import os
import sqlite3
import time
from collections import Counter
from pathlib import Path

from .common import SCHEMA_VERSION, log, read_json, save_json, sha256, stable_int
from .io import MATCH_HEADER, SOURCE_HEADER, data_files, id_list, tsv_rows
from .normalize import Record, index_keys, normalize


def connect(path, readonly=True):
    path = Path(path).resolve()
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) if readonly else sqlite3.connect(path)
    db.execute("PRAGMA cache_size=-131072")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def source_manifest(dataset, split):
    return [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in data_files(dataset, split)]


def build_truth(db, path, seed):
    db.executescript("""
        CREATE TABLE raw_anchors (entity_id TEXT PRIMARY KEY);
        CREATE TABLE raw_truth (s1 TEXT, target TEXT, PRIMARY KEY(s1,target)) WITHOUT ROWID;
        CREATE TABLE truth (s1 INTEGER, target INTEGER, PRIMARY KEY(s1,target)) WITHOUT ROWID;
    """)
    anchors, edges = [], []
    for sid, raw_ids in tsv_rows(path, MATCH_HEADER):
        anchors.append((sid,))
        edges.extend((sid, target) for target in id_list(raw_ids))
        if len(anchors) >= 10000:
            db.executemany("INSERT INTO raw_anchors VALUES (?)", anchors)
            db.executemany("INSERT INTO raw_truth VALUES (?,?)", edges)
            db.commit()
            anchors.clear()
            edges.clear()
    db.executemany("INSERT INTO raw_anchors VALUES (?)", anchors)
    db.executemany("INSERT INTO raw_truth VALUES (?,?)", edges)
    bad = db.execute("""SELECT COUNT(*) FROM raw_anchors g LEFT JOIN records r ON r.entity_id=g.entity_id
                         WHERE r.rid IS NULL OR r.source != 1""").fetchone()[0]
    missing = db.execute("""SELECT COUNT(*) FROM records r LEFT JOIN raw_anchors g ON r.entity_id=g.entity_id
                             WHERE r.source=1 AND g.entity_id IS NULL""").fetchone()[0]
    if bad or missing:
        raise ValueError(f"Ground-truth anchor coverage invalid: {bad} unknown/wrong-source, {missing} missing")
    db.execute("""INSERT INTO truth SELECT a.rid, b.rid FROM raw_truth t
                  JOIN records a ON a.entity_id=t.s1 AND a.source=1
                  JOIN records b ON b.entity_id=t.target AND b.source IN (2,3)""")
    total = db.execute("SELECT COUNT(*) FROM raw_truth").fetchone()[0]
    if total != db.execute("SELECT COUNT(*) FROM truth").fetchone()[0]:
        raise ValueError("Ground truth references nonexistent or non-S2/S3 target IDs")
    db.execute("CREATE INDEX truth_target ON truth(target,s1)")

    # Only shared-target components need a union structure in memory.
    parents = {}
    def root(x):
        parents.setdefault(x, x)
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    def union(a, b):
        a, b = root(a), root(b)
        parents[max(a, b)] = min(a, b)
    shared = db.execute("""SELECT t.target,t.s1 FROM truth t JOIN
        (SELECT target FROM truth GROUP BY target HAVING COUNT(*)>1) d ON d.target=t.target
        ORDER BY t.target,t.s1""")
    for _, rows in itertools.groupby(shared, key=lambda row: row[0]):
        first = None
        for _, sid in rows:
            if first is None:
                first = sid
            else:
                union(first, sid)
    db.execute("CREATE TABLE partitions (s1 INTEGER PRIMARY KEY, part TEXT, sample_key INTEGER)")
    group_ids = {}
    batch = []
    for rid, entity_id in db.execute("SELECT rid,entity_id FROM records WHERE source=1 ORDER BY rid"):
        representative = root(rid) if rid in parents else rid
        if representative != rid:
            if representative not in group_ids:
                group_ids[representative] = db.execute("SELECT entity_id FROM records WHERE rid=?", (representative,)).fetchone()[0]
            entity_id = group_ids[representative]
        sample = stable_int(f"{seed}|{entity_id}")
        bucket = sample % 10000
        part = "fit" if bucket < 7000 else "dev" if bucket < 8500 else "holdout"
        batch.append((rid, part, sample))
        if len(batch) >= 10000:
            db.executemany("INSERT INTO partitions VALUES (?,?,?)", batch)
            batch.clear()
    db.executemany("INSERT INTO partitions VALUES (?,?,?)", batch)
    db.execute("CREATE INDEX partition_sample ON partitions(part,sample_key,s1)")
    db.execute("CREATE TABLE reserved_targets (rid INTEGER PRIMARY KEY)")
    db.execute("""INSERT INTO reserved_targets SELECT DISTINCT t.target FROM truth t
                  JOIN partitions p ON p.s1=t.s1 WHERE p.part!='fit'""")
    # Quarantine exact normalized duplicates of held-out positives from supervised negatives.
    db.execute("CREATE INDEX record_text ON records(name,address,country)")
    db.execute("""INSERT OR IGNORE INTO reserved_targets
        SELECT b.rid FROM records b JOIN records a
          ON a.name=b.name AND a.address=b.address AND a.country=b.country
        JOIN truth t ON t.target=a.rid JOIN partitions p ON p.s1=t.s1
        WHERE p.part!='fit' AND b.source IN (2,3) AND a.name!=''""")
    db.execute("DROP INDEX record_text")
    db.execute("DROP TABLE raw_truth")
    db.execute("DROP TABLE raw_anchors")
    db.commit()
    return {"positive_pairs": total, "shared_component_anchors": len(parents)}


def prepare(dataset, work, split, cfg):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    path = work / f"{split}.sqlite"
    manifest_path = work / f"{split}_manifest.json"
    log(f"Hashing {split} inputs")
    signature = {"schema": SCHEMA_VERSION, "files": source_manifest(dataset, split),
                 "index_config": {k: cfg[k] for k in ("name_tokens", "address_tokens", "name_grams", "seed")},
                 "database_builder_sha256": sha256(Path(__file__)),
                 "normalizer_sha256": sha256(Path(__file__).with_name("normalize.py"))}
    if path.exists() or manifest_path.exists():
        if path.exists() and manifest_path.exists() and read_json(manifest_path)["signature"] == signature:
            log(f"Reusing verified {split} index: {path}")
            return path
        raise ValueError(f"Incompatible/incomplete cache at {work}; use a new --work directory")
    temporary = path.with_suffix(".building.sqlite")
    if temporary.exists():
        raise ValueError(f"Incomplete build exists: {temporary}. Remove only this temporary file before retrying.")
    started = time.perf_counter()
    db = connect(temporary, readonly=False)
    db.executescript("""
        PRAGMA journal_mode=DELETE;
        CREATE TABLE records (
            rid INTEGER PRIMARY KEY, entity_id TEXT UNIQUE NOT NULL,
            source INTEGER NOT NULL, country TEXT NOT NULL, name TEXT NOT NULL, address TEXT NOT NULL
        );
        CREATE TABLE postings (key INTEGER NOT NULL, rid INTEGER NOT NULL);
    """)
    stats, countries = Counter(), Counter()
    records, postings = [], []
    rid = 0
    try:
        for source, source_path in enumerate(data_files(dataset, split)[:3], 1):
            log(f"Reading/indexing {source_path.name}")
            for entity_id, name, address, country in tsv_rows(source_path, SOURCE_HEADER):
                if not entity_id.startswith(f"S{source}-") or any(c.isspace() or c in ',"' for c in entity_id):
                    raise ValueError(f"Invalid source ID: {entity_id!r}")
                rid += 1
                record = Record(rid, entity_id, source, normalize(country), normalize(name), normalize(address))
                records.append((rid, entity_id, source, record.country, record.name, record.address))
                if source != 1:
                    postings.extend((key, rid) for key in sorted({k for k, _ in index_keys(record, source, cfg)}))
                stats[f"source{source}_rows"] += 1
                stats[f"source{source}_empty_name"] += not record.name
                stats[f"source{source}_empty_address"] += not record.address
                countries[f"S{source}|{record.country or '<missing>'}"] += 1
                if len(records) >= 10000:
                    db.executemany("INSERT INTO records VALUES (?,?,?,?,?,?)", records)
                    db.executemany("INSERT INTO postings VALUES (?,?)", postings)
                    db.commit()
                    records.clear()
                    postings.clear()
                    if rid % 100000 == 0:
                        log(f"Indexed {rid:,} records in {time.perf_counter()-started:.0f}s")
        db.executemany("INSERT INTO records VALUES (?,?,?,?,?,?)", records)
        db.executemany("INSERT INTO postings VALUES (?,?)", postings)
        db.commit()
        log("Building disk-backed posting indexes and key frequencies")
        db.execute("CREATE UNIQUE INDEX posting_key ON postings(key,rid)")
        db.execute("CREATE INDEX record_source ON records(source,rid)")
        db.execute("CREATE TABLE key_counts (key INTEGER PRIMARY KEY, n INTEGER)")
        db.execute("INSERT INTO key_counts SELECT key,COUNT(*) FROM postings GROUP BY key")
        db.commit()
        if split == "train":
            log("Validating labels and building grouped split")
            stats.update(build_truth(db, data_files(dataset, split)[3], cfg["seed"]))
            stats["singleton_anchors"] = db.execute("""SELECT COUNT(*) FROM records r WHERE source=1
                AND NOT EXISTS (SELECT 1 FROM truth t WHERE t.s1=r.rid)""").fetchone()[0]
            stats["positive_country_mismatches"] = db.execute("""SELECT COUNT(*) FROM truth t
                JOIN records a ON a.rid=t.s1 JOIN records b ON b.rid=t.target
                WHERE a.country!=b.country""").fetchone()[0]
            partitions = dict(db.execute("SELECT part,COUNT(*) FROM partitions GROUP BY part"))
        else:
            partitions = {}
        db.execute("ANALYZE")
        db.commit()
    finally:
        db.close()
    os.replace(temporary, path)
    save_json(manifest_path, {"signature": signature, "stats": dict(stats), "countries": dict(countries),
                              "partitions": partitions, "seconds": time.perf_counter()-started,
                              "database_bytes": path.stat().st_size})
    log(f"Prepared {split}: {rid:,} records, {time.perf_counter()-started:.0f}s")
    return path


def anchors(db, part=None, limit=None, after=0):
    if part:
        sql = """SELECT r.rid,r.entity_id,r.source,r.country,r.name,r.address FROM records r
                 JOIN partitions p ON p.s1=r.rid WHERE p.part=? ORDER BY p.sample_key,r.rid"""
        params = [part]
    else:
        sql = "SELECT rid,entity_id,source,country,name,address FROM records WHERE source=1 AND rid>? ORDER BY rid"
        params = [after]
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    for row in db.execute(sql, params):
        yield Record(*row)


def truth_for(db, rid):
    return {row[0] for row in db.execute("SELECT target FROM truth WHERE s1=?", (rid,))}
