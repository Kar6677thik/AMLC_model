from __future__ import annotations

import csv
import gzip
import importlib.util
import tempfile
import unittest
from unittest.mock import Mock
import zipfile
import numpy as np
from pathlib import Path

from ber.common import load_config, read_json, save_json, sha256
from ber.database import anchors, connect, prepare
from ber.evaluate import evaluate
from ber.io import MATCH_HEADER, SOURCE_HEADER, id_list
from ber.metrics import Metrics, f05
from ber.model import train
from ber.normalize import folded, normalize
from ber.package import package
from ber.predict import predict
from ber.validate import validate
from ber.features import FEATURES
from ber.parallel import feature_batches
from ber.text_views import view
from ber.trees import assert_cuda


def write_tsv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def fixture(root, split, count):
    offset = 0 if split == "train" else 10000
    rows = {source: [] for source in (1, 2, 3)}
    truth = []
    for i in range(count):
        num = offset + i
        country = ("US" if i % 2 else "India") if split == "train" else "France"
        name, address = f"Restore Atelier {num}", f"{2000+num} Rue Example District {i}"
        for source in (1, 2, 3):
            source_name = name if source == 1 or i % 4 else f"Different Business {num}"
            source_address = address if source == 1 or i % 4 else f"{9000+num} Other Road"
            rows[source].append([f"S{source}-{num:05d}", source_name, source_address, country])
        positives = [] if i % 4 == 0 else [f"S2-{num:05d}", f"S3-{num:05d}"]
        # Shared-target components must stay in a single supervised partition.
        if i == 2:
            positives.append(f"S2-{offset+1:05d}")
        truth.append([f"S1-{num:05d}", ",".join(positives)])
    for source in (1, 2, 3):
        write_tsv(root / split / f"{split}_source{source}.tsv", SOURCE_HEADER, rows[source])
    if split == "train":
        write_tsv(root / split / "train_ground_truth.tsv", MATCH_HEADER, truth)


class MetricTests(unittest.TestCase):
    def test_hand_calculated_cases(self):
        for truth, predicted, expected in [([], [], 1), ([], [1], 0), ([1], [], 0),
                ([1], [1], 1), ([1], [2], 0), ([1, 2], [1], 5/6), ([1, 2], [1, 2, 3], 5/7)]:
            self.assertAlmostEqual(f05(truth, predicted), expected)

    def test_macro_and_oracle_include_singletons(self):
        metrics = Metrics()
        metrics.add([], [], [])
        metrics.add([1, 2], [1], [1])
        self.assertAlmostEqual(metrics.report()["macro_f05"], 11/12)
        self.assertAlmostEqual(metrics.report()["oracle_macro_f05"], 11/12)

    def test_text_and_id_contracts(self):
        self.assertEqual(normalize("École & Sons"), "école and sons")
        self.assertEqual(folded(normalize("École")), "ecole")
        self.assertEqual(normalize("भारत"), "भारत")
        self.assertEqual(folded("भारत"), "भारत")
        self.assertEqual(id_list(""), [])
        for bad in ("S2-1,S2-1", "S2-1,", " S2-1", '"S2-1"'):
            with self.assertRaises(ValueError):
                id_list(bad)

    def test_cached_views_preserve_digits_and_unicode(self):
        self.assertEqual(view("भारत").text, "भारत")
        self.assertEqual(view("école").text, "ecole")
        self.assertEqual(view("17 road 560001").variant, "17 rd 560001")
        self.assertEqual(view("17 road 560001").postal, frozenset({"560001"}))
        self.assertEqual(view("17 road 560001").first_number, "17")
        self.assertEqual(len(FEATURES), 46)

    def test_cuda_request_rejects_cpu_fallback(self):
        model = Mock()
        model.save_config.return_value = '{"learner":{"generic_param":{"device":"cpu"}}}'
        with self.assertRaisesRegex(ValueError, "CUDA requested"):
            assert_cuda(model)
        model.save_config.return_value = '{"learner":{"generic_param":{"device":"cuda:0"}}}'
        self.assertEqual(assert_cuda(model), "cuda:0")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ber-fixture-")
        self.root = Path(self.temp.name)
        self.data, self.work = self.root / "dataset", self.root / "work"
        fixture(self.data, "train", 96)
        fixture(self.data, "test", 7)
        self.cfg = load_config()
        self.cfg.update(candidates_per_source=4, posting_limit=300, train_anchors=50,
                        dev_anchors=20, prediction_batch_anchors=3, prediction_shard_anchors=2,
                        trees=5, min_child_samples=2, threads=1)
        prepare(self.data, self.work, "train", self.cfg)
        prepare(self.data, self.work, "test", self.cfg)

    def tearDown(self):
        self.temp.cleanup()

    def run_pipeline(self, mode):
        run, output = self.root / mode, self.root / (mode + "_output")
        train(self.work, run, self.cfg, mode)
        evaluate(self.work, run)
        predict(self.work, run, output)
        self.assertTrue(validate(self.work, output)["passed"])
        return run, output

    def test_grouping_and_reserved_targets(self):
        db = connect(self.work / "train.sqlite")
        try:
            parts = dict(db.execute("SELECT r.entity_id,p.part FROM records r JOIN partitions p ON p.s1=r.rid"))
            self.assertEqual(parts["S1-00001"], parts["S1-00002"])
            leaked = db.execute("""SELECT COUNT(*) FROM truth t JOIN partitions p ON p.s1=t.s1
                WHERE p.part!='fit' AND t.target NOT IN (SELECT rid FROM reserved_targets)""").fetchone()[0]
            self.assertEqual(leaked, 0)
        finally:
            db.close()

    def test_rules_resume_validation_and_package(self):
        run, output = self.run_pipeline("rules")
        initial = [sha256(output / name) for name in ("matching_results.tsv", "candidate_pairs.tsv")]
        # Force only the final shard to be regenerated, simulating interrupted work.
        final_marker = sorted((output / ".shards").glob("*/complete.json"))[-1]
        final_marker.unlink()
        predict(self.work, run, output)
        self.assertEqual(initial, [sha256(output / name) for name in ("matching_results.tsv", "candidate_pairs.tsv")])
        report = validate(self.work, output)
        self.assertEqual(report["anchors"], 7)
        self.assertEqual(report["countries"], {"france": 7})
        archive = package(self.work, run, output, self.root / "dist", "Synthetic Test Member")
        with zipfile.ZipFile(archive) as z:
            self.assertIn("Documentation_template.md", z.namelist())
            self.assertIn("code/business_entity_resolution/src/ber/cli.py", z.namelist())
            self.assertEqual(z.read("output/matching_results.tsv"), (output / "matching_results.tsv").read_bytes())
        # Corrupted output must fail, even if formatting still looks valid.
        with (output / "candidate_pairs.tsv").open("a", encoding="utf-8") as stream:
            stream.write("S1-extra\t\n")
        with self.assertRaises(ValueError):
            validate(self.work, output)

    def test_learned_model_and_locked_holdout(self):
        run, output = self.run_pipeline("learned")
        before = sha256(run / "decision.json")
        report = evaluate(self.work, run, "holdout")
        self.assertEqual(before, sha256(run / "decision.json"))
        self.assertGreater(report["metrics"]["anchors"], 0)
        with self.assertRaises(ValueError):
            evaluate(self.work, run, "holdout")

    def test_incompatible_cache_is_rejected(self):
        changed = dict(self.cfg, name_grams=7)
        with self.assertRaises(ValueError):
            prepare(self.data, self.work, "train", changed)

    def test_unknown_target_is_rejected_beyond_hash_checks(self):
        _, output = self.run_pipeline("rules")
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            path = output / name
            lines = path.read_text(encoding="utf-8").splitlines()
            sid = lines[1].split("\t")[0]
            lines[1] = sid + "\tS2-nonexistent"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
        ledger = output / "scoring_ledger.tsv.gz"
        with gzip.open(ledger, "rt", encoding="utf-8") as stream:
            lines = stream.read().splitlines()
        sid = lines[1].split("\t")[0]
        lines[1] = sid + "\tS2-nonexistent\t1.0"
        with gzip.open(ledger, "wt", encoding="utf-8", newline="") as stream:
            stream.write("\n".join(lines) + "\n")
        manifest = read_json(output / "prediction.json")
        manifest["hashes"] = {name: sha256(output / name) for name in manifest["hashes"]}
        save_json(output / "prediction.json", manifest)
        with self.assertRaisesRegex(ValueError, "Unknown target ID"):
            validate(self.work, output)

    def test_parallel_v2_features_preserve_order_and_values(self):
        db = connect(self.work / "test.sqlite")
        cfg = dict(self.cfg, retrieval_version="v2", query_expansion=3, feature_batch_anchors=2)
        try:
            selected = list(anchors(db))
            serial = list(feature_batches(db, iter(selected), dict(cfg, feature_workers=1)))
            parallel = list(feature_batches(db, iter(selected), dict(cfg, feature_workers=2)))
            self.assertEqual(len(serial), len(parallel))
            for a, b in zip(serial, parallel):
                self.assertEqual(a[0], b[0])
                np.testing.assert_array_equal(a[1], b[1])
                self.assertEqual(a[2], b[2])
                for _, candidates in a[0]:
                    self.assertLessEqual(len(candidates), 2*cfg["candidates_per_source"])
                    self.assertEqual(len(candidates), len({c.record.rid for c in candidates}))
        finally:
            db.close()

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "Optional XGBoost is not installed")
    def test_xgboost_cpu_roundtrip_and_package(self):
        self.cfg.update(model_backend="xgboost", device="cpu", retrieval_version="v2")
        run, output = self.run_pipeline("learned")
        self.assertTrue((run / "model.ubj").exists())
        archive = package(self.work, run, output, self.root / "xgb-dist", "Synthetic Test Member")
        with zipfile.ZipFile(archive) as z:
            self.assertIn("code/business_entity_resolution/assets/model.ubj", z.namelist())


if __name__ == "__main__":
    unittest.main()
