"""Run on the compute PC: python -m unittest discover -s code/refinement -v."""
import argparse
import gzip
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zipfile

import refine as r


def policy(**changes):
    result = dict(country=False, ownership=False, missing_address=False, street=False,
        rival_name=False, gap=None, learned_countries=["us", "india"])
    result.update(changes)
    return result


def edge(target, score, **flags):
    result = dict(country=False, missing_address=False, street=False, rival_name=False)
    result.update(flags)
    return target, score, result


class Decisions(unittest.TestCase):
    def test_macro_metric_and_empty_credit(self):
        self.assertEqual(r.f05(set(), set()), 1)
        self.assertEqual(r.f05(set(), {1}), 0)
        self.assertAlmostEqual(r.f05({1}, {1, 2}), 5 / 9)
        self.assertAlmostEqual(r.f05({1, 2}, {1}), 5 / 6)

    def test_ownership_is_global_but_not_one_target_per_anchor(self):
        rows = [dict(country="us", edges=[edge(1, .9), edge(2, .9), edge(3, .8)]),
                dict(country="us", edges=[edge(1, .95), edge(2, .9), edge(4, .9)])]
        self.assertEqual(r.predictions(rows, policy(ownership=True)), [{3}, {1, 4}])

    def test_text_rules_do_not_transfer_to_unlabeled_france(self):
        rows = [dict(country="france", edges=[edge(1, .9, missing_address=True)]),
                dict(country="us", edges=[edge(1, .9, missing_address=True)])]
        self.assertEqual(r.predictions(rows, policy(missing_address=True)), [{1}, set()])

    def test_country_filter_requires_actual_conflict(self):
        rows = [dict(country="france", edges=[edge(1, .9, country=True), edge(2, .8)])]
        self.assertEqual(r.predictions(rows, policy(country=True)), [{2}])

    def test_adaptive_cutoff_and_street_parser(self):
        row = dict(country="us", edges=[edge(1, .99), edge(2, .9), edge(3, .7)])
        self.assertEqual(r.local_keep(row, policy(gap=.2)), [(1, .99), (2, .9)])
        self.assertEqual(r.street("12 maple road springfield 12345"), {"maple"})
        self.assertEqual(r.street("20 oak road springfield 12345"), {"oak"})
        self.assertEqual(r.street("12 rue victor hugo 75001 paris"), {"victor", "hugo"})
        self.assertIsNone(r.street("springfield 12345"))


class Integration(unittest.TestCase):
    def test_study_selects_measured_rule_and_rejects_insufficient_gain(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            work, run = root / "work", root / "run"
            work.mkdir()
            run.mkdir()
            db = sqlite3.connect(work / "train.sqlite")
            db.execute("CREATE TABLE records(rid INTEGER PRIMARY KEY,entity_id TEXT UNIQUE,source INTEGER,country TEXT,name TEXT,address TEXT)")
            cached = []
            for i in range(1, 121):
                db.executemany("INSERT INTO records VALUES (?,?,?,?,?,?)", [
                    (i, f"S1-{i:03d}", 1, "us", "shared name", "1 main road"),
                    (120 + i, f"S2-{i:03d}", 2, "us", "shared name", "1 main road"),
                    (240 + i, f"S3-{i:03d}", 3, "us", "shared name", "")])
                cached.append({"source1_entity_id": f"S1-{i:03d}", "country": "us", "truth": [120 + i],
                    "targets": [120 + i, 240 + i], "scores": [.95, .8]})
            db.commit()
            db.close()
            manifest = {"signature": {"fixture": True}, "stats": {"positive_country_mismatches": 0, "shared_component_anchors": 0}}
            r.save(work / "train_manifest.json", manifest)
            r.save(run / "run.json", {"train_manifest": manifest})
            r.save(run / "decision.json", {"threshold": .59})
            (run / "dev_scores.jsonl").write_text("\n".join(json.dumps(item) for item in cached) + "\n", encoding="utf-8")
            r.save(run / "dev_report.json", {"scores_sha256": r.digest(run / "dev_scores.jsonl"), "metrics": {"macro_f05": 5 / 9}})
            args = argparse.Namespace(work=work, run=run, output=root / "study", min_gain=.0005, country_tolerance=.002)
            r.study(args)
            selected = r.read(args.output / "policy.json")
            self.assertEqual(selected["name"], "missing_address")
            self.assertEqual(selected["rules"]["learned_countries"], ["us"])
            args.output = root / "strict-study"
            args.min_gain = .5
            r.study(args)
            self.assertEqual(r.read(args.output / "policy.json")["name"], "baseline")

    def fixture(self, root):
        work, run, original = (root / n for n in ("work", "run", "original"))
        for path in (work, run, original):
            path.mkdir()
        db = sqlite3.connect(work / "test.sqlite")
        db.execute("CREATE TABLE records(rid INTEGER PRIMARY KEY,entity_id TEXT UNIQUE,source INTEGER,country TEXT,name TEXT,address TEXT)")
        db.executemany("INSERT INTO records VALUES (?,?,?,?,?,?)", [
            (1, "S1-a", 1, "us", "alpha", "12 maple road"),
            (2, "S1-b", 1, "us", "beta", "22 oak road"),
            (3, "S1-c", 1, "france", "gamma", ""),
            (4, "S2-a", 2, "us", "alpha", "12 maple road"),
            (5, "S2-b", 2, "us", "beta", "22 oak road"),
            (6, "S3-c", 3, "france", "gamma", "")])
        db.commit()
        db.close()
        r.save(work / "test_manifest.json", {"signature": {"fixture": 1}})
        r.save(run / "run.json", {"fixture": 1})
        r.save(run / "decision.json", {"threshold": .59})
        matches = "source1_entity_id\tmatched_entity_ids\nS1-a\tS2-a,S2-b\nS1-b\tS2-a,S2-b\nS1-c\t\n"
        candidates = matches.replace("matched_entity_ids", "candidate_entity_ids").replace("S1-c\t\n", "S1-c\tS3-c\n")
        (original / "matching_results.tsv").write_text(matches, encoding="utf-8")
        (original / "candidate_pairs.tsv").write_text(candidates, encoding="utf-8")
        with gzip.open(original / "scoring_ledger.tsv.gz", "wt", encoding="utf-8") as stream:
            stream.write("\t".join(r.LEDGER) + "\nS1-a\tS2-a,S2-b\t0.9,0.9\nS1-b\tS2-a,S2-b\t0.95,0.9\nS1-c\tS3-c\t0.2\n")
        hashes = {n: r.digest(original / n) for n in ("matching_results.tsv", "candidate_pairs.tsv", "scoring_ledger.tsv.gz")}
        r.save(original / "prediction.json", {"identity": {"run_sha256": r.digest(run / "run.json"),
            "decision_sha256": r.digest(run / "decision.json"), "test_signature": {"fixture": 1}},
            "hashes": hashes, "threshold": .59, "anchors": 3, "scored_pairs": 5})
        r.save(original / "validation.json", {"passed": True, "hashes": hashes, "checks": ["target_membership"]})
        frozen = dict(version=r.VERSION, source_sha256=r.digest(r.__file__), name="ownership", rules=policy(ownership=True),
            run_sha256=r.digest(run / "run.json"), decision_sha256=r.digest(run / "decision.json"), threshold=.59)
        r.save(root / "policy.json", frozen)
        return argparse.Namespace(work=work, run=run, original=original, policy=root / "policy.json", output=root / "refined")

    def test_full_ledger_refinement_and_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.fixture(root)
            r.apply(args)
            self.assertEqual(list(r.tsv(args.output / "matching_results.tsv", r.MATCH)), [["S1-a", ""], ["S1-b", "S2-a"], ["S1-c", ""]])
            self.assertEqual(r.digest(args.output / "candidate_pairs.tsv"), r.digest(args.original / "candidate_pairs.tsv"))
            study = root / "study"
            study.mkdir()
            r.save(study / "study.json", {"policy_sha256": r.digest(args.policy)})
            baseline = root / "baseline.zip"
            with zipfile.ZipFile(baseline, "w") as archive:
                for name in ("prediction.json", "validation.json"):
                    archive.write(args.original / name, "code/business_entity_resolution/assets/" + name)
                for name in ("matching_results.tsv", "candidate_pairs.tsv"):
                    archive.write(args.original / name, "output/" + name)
                archive.writestr("Documentation_template.md", "Baseline test")
            destination = root / "release.zip"
            r.package(argparse.Namespace(output=args.output, baseline_zip=baseline, study=study, destination=destination))
            with zipfile.ZipFile(destination) as archive:
                self.assertEqual(archive.read("output/matching_results.tsv"), (args.output / "matching_results.tsv").read_bytes())
                self.assertIn("code/business_entity_resolution/assets/baseline_prediction.json", archive.namelist())
                self.assertIn("code/refinement/refine.py", archive.namelist())

    def test_tampered_candidate_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            with open(args.original / "candidate_pairs.tsv", "a") as stream:
                stream.write("S1-invented\tS2-a\n")
            with self.assertRaisesRegex(ValueError, "Original output changed"):
                r.apply(args)

    def test_same_address_rival_and_missing_address_ambiguity(self):
        with tempfile.TemporaryDirectory() as temp:
            db = sqlite3.connect(":memory:")
            db.execute("CREATE TABLE records(rid INTEGER,entity_id TEXT,source INTEGER,country TEXT,name TEXT,address TEXT)")
            db.executemany("INSERT INTO records VALUES (?,?,?,?,?,?)", [
                (1, "S1-a", 1, "france", "cafe nord", "1 rue a"),
                (2, "S1-b", 1, "france", "cafe sud", "1 rue a"),
                (3, "S1-c", 1, "france", "cafe nord", "2 rue b")])
            context = r.build_context(db, Path(temp) / "context.sqlite")
            a = dict(entity_id="S1-a", country="france", name="cafe nord", address="1 rue a")
            b = dict(country="france", name="cafe sud", address="1 rue a")
            self.assertTrue(r.evidence(a, b, context)["rival_name"])
            self.assertTrue(r.evidence(a, dict(b, name="cafe nord", address=""), context)["missing_address"])
            context.close()
            db.close()


if __name__ == "__main__":
    unittest.main()
