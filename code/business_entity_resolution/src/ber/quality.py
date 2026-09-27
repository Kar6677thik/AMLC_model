"""Refit preserved fit-only features and compare full-quality CUDA matchers."""
from __future__ import annotations

import gc
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import DEFAULTS, environment, log, read_json, save_json, sha256, source_hash
from .features import FEATURES
from .trees import TreeModel, gpu_check

# Reviewed optimized-002 source: same 46 features and v2 retrieval semantics.
REUSABLE_V2_SOURCE = "abb83b712cfa69fe041d57b753de8e5fc55067e0ef7b0b2c6a6320c34abb77a8"
FIT_ONLY_OPTIONS = {"trees", "max_depth", "max_bin", "learning_rate", "num_leaves", "min_child_samples",
                   "model_backend", "device", "threads", "feature_workers", "feature_batch_anchors",
                   "prediction_batch_anchors", "ranking_backend"}


def load_training_features(work, source):
    source = Path(source)
    meta, training = read_json(source / "run.json"), read_json(source / "training.json")
    if meta["source_sha256"] not in {REUSABLE_V2_SOURCE, source_hash()}:
        raise ValueError("Feature import requires reviewed optimized-002 or current source; unknown feature provenance")
    if meta["features"] != FEATURES or meta["mode"] != "learned" or meta["config"].get("retrieval_version") != "v2":
        raise ValueError("Feature import requires a completed learned v2 run with the current 46-feature schema")
    if meta["train_manifest"]["signature"] != read_json(Path(work) / "train_manifest.json")["signature"]:
        raise ValueError("Saved training features belong to another dataset/split")
    current_env = environment()
    for dependency in ("numpy", "rapidfuzz"):
        if meta["environment"][dependency] != current_env[dependency]:
            raise ValueError(f"Restore source feature dependency: {dependency}=={meta['environment'][dependency]}")
    arrays = [np.load(source / name, mmap_mode="r", allow_pickle=False)
              for name in ("train_x.npy", "train_y.npy", "train_w.npy")]
    x, y, w = arrays
    n = training["pairs"]
    if not isinstance(n, int) or n <= 0 or x.ndim != 2 or x.shape[1] != len(FEATURES):
        raise ValueError("Invalid saved feature dimensions/pair count")
    if x.dtype != np.float32 or y.dtype != np.int8 or w.dtype != np.float32:
        raise ValueError("Saved feature dtypes differ from the training contract")
    if y.ndim != 1 or w.ndim != 1 or not (len(x) == len(y) == len(w)) or n > len(x):
        raise ValueError("Saved training arrays have incompatible lengths")
    positives = 0
    for start in range(0, n, 131072):
        end = min(n, start+131072)
        if not np.isfinite(x[start:end]).all() or not np.isfinite(w[start:end]).all():
            raise ValueError("Non-finite saved features or weights")
        if not np.isin(y[start:end], [0, 1]).all() or (w[start:end] <= 0).any():
            raise ValueError("Invalid saved labels or weights")
        positives += int(y[start:end].sum())
    if positives != training["positive_pairs"] or not 0 < positives < n:
        raise ValueError("Saved positive labels disagree with training report")
    provenance = {"source_run_sha256": sha256(source / "run.json"), "source_sha256": meta["source_sha256"],
                  "training_report_sha256": sha256(source / "training.json"),
                  "files": {name: sha256(source / name) for name in
                            ("train_x.npy", "train_y.npy", "train_w.npy", "training_anchors.tsv")},
                  "note": "Only original fit-partition features/labels are reused; development labels are not fitted."}
    return meta, training, (x[:n], y[:n], w[:n]), provenance


def init_run(run, cfg, source_meta, provenance):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=False)
    meta = {"config": cfg, "mode": "learned", "features": FEATURES, "environment": environment(),
            "source_sha256": source_hash(), "train_manifest": source_meta["train_manifest"],
            "feature_provenance": provenance}
    save_json(run / "run.json", meta)
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    lines = [line for line in frozen.splitlines() if not line.startswith(("-e ", "#")) and "restorebuildrun" not in line.lower()]
    (run / "environment.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return meta


def refit(run, cfg, source, loaded):
    source_meta, training, arrays, provenance = loaded
    old_cfg = {**DEFAULTS, **source_meta["config"]}
    for key in DEFAULTS:
        if key not in FIT_ONLY_OPTIONS and cfg[key] != old_cfg[key]:
            raise ValueError(f"Cannot reuse features after changing {key}")
    run = Path(run)
    meta = init_run(run, cfg, source_meta, provenance)
    shutil.copyfile(Path(source) / "training_anchors.tsv", run / "training_anchors.tsv")
    log(f"GPU refit: {run.name}, {training['pairs']:,} preserved pairs, depth={cfg['max_depth']}, trees={cfg['trees']}")
    tick = time.perf_counter()
    model = TreeModel(cfg)
    model.fit(*arrays)
    model.save(run / model.filename)
    meta.update(model_file=model.filename, model_sha256=sha256(run / model.filename))
    save_json(run / "run.json", meta)
    report = {"anchors": training["anchors"], "pairs": training["pairs"], "positive_pairs": training["positive_pairs"],
              "fit_seconds": time.perf_counter()-tick, "feature_extraction_reused": True,
              "feature_importance_gain": model.importance(), "backend": cfg["model_backend"], "device": cfg["device"]}
    save_json(run / "training.json", report)
    del model
    gc.collect()


def ensemble(run, members):
    run = Path(run)
    source_meta = read_json(Path(members[0]) / "run.json")
    cfg = dict(source_meta["config"], model_backend="xgboost_ensemble")
    meta = init_run(run, cfg, source_meta, source_meta["feature_provenance"])
    manifest = {"members": []}
    for idx, member in enumerate(members):
        member = Path(member)
        member_meta = read_json(member / "run.json")
        if member_meta["config"]["model_backend"] != "xgboost" or member_meta["features"] != FEATURES:
            raise ValueError("Ensemble requires XGBoost members with matching features")
        for key in DEFAULTS:
            if key not in FIT_ONLY_OPTIONS and member_meta["config"][key] != cfg[key]:
                raise ValueError("Ensemble member retrieval configuration mismatch")
        if member_meta["feature_provenance"] != source_meta["feature_provenance"]:
            raise ValueError("Ensemble members must use the same fit data")
        model_path = member / member_meta["model_file"]
        if sha256(model_path) != member_meta["model_sha256"]:
            raise ValueError("Ensemble source model hash mismatch")
        filename = f"member-{idx}.ubj"
        shutil.copyfile(model_path, run / filename)
        manifest["members"].append({"filename": filename, "weight": 1/len(members),
                                    "sha256": sha256(run / filename), "config": member_meta["config"]})
    save_json(run / "ensemble.json", manifest)
    meta.update(model_file="ensemble.json", model_sha256=sha256(run / "ensemble.json"))
    save_json(run / "run.json", meta)
    save_json(run / "training.json", {"backend": "xgboost_ensemble", "device": cfg["device"],
              "members": manifest["members"], "note": "Equal weights; member models were fitted on identical fit-only data."})
    shutil.copyfile(Path(members[0]) / "training_anchors.tsv", run / "training_anchors.tsv")


def assert_same_candidates(reference_scores, actual_scores):
    import json
    from itertools import zip_longest
    count = 0
    with open(reference_scores, encoding="utf-8") as left, open(actual_scores, encoding="utf-8") as right:
        for a, b in zip_longest(left, right):
            if a is None or b is None:
                raise ValueError("Reference and new development samples have different lengths")
            a, b = json.loads(a), json.loads(b)
            if any(a[key] != b[key] for key in ("source1_entity_id", "truth", "targets")):
                raise ValueError(f"Full-quality retrieval changed for {a['source1_entity_id']}; do not reuse its fit features")
            count += 1
    return count


def remaining_seconds(deadline):
    stamp = datetime.fromisoformat(deadline)
    if stamp.tzinfo is None:
        raise ValueError("Deadline requires an explicit timezone offset")
    return (stamp-datetime.now(timezone.utc)).total_seconds()


def quality_sweep(dataset, work, source, output, deadline, reserve_minutes=60, margin=1.35, workers=6):
    from .database import prepare
    from .evaluate import evaluate
    from .benchmark import benchmark
    output, source = Path(output), Path(source)
    if output.exists():
        raise ValueError("Study directory exists; preserve it and use a new StudyId")
    if (remaining_seconds(deadline) <= reserve_minutes*60 or not math.isfinite(margin)
            or margin < 1 or workers < 1 or reserve_minutes < 0):
        raise ValueError("Invalid timing settings or deadline has no usable time remaining")
    meta = read_json(source / "run.json")
    cfg = {**DEFAULTS, **meta["config"], "model_backend": "xgboost", "device": "cuda:0",
           "ranking_backend": "compact", "feature_workers": workers}
    prepare(dataset, work, "train", cfg)
    prepare(dataset, work, "test", cfg)
    gpu = gpu_check()
    log("Validating and hashing saved fit arrays; feature extraction will not repeat")
    loaded = load_training_features(work, source)
    output.mkdir(parents=True)
    report = {"complete": False, "source_sha256": source_hash(), "gpu_check": gpu, "trials": [],
              "selected": None, "deadline": deadline, "timing_margin": margin, "reserve_minutes": reserve_minutes}
    save_json(output / "quality.json", report)

    control = output / "control"
    refit(control, cfg, source, loaded)
    dev = evaluate(work, control)
    # Full sample comparison protects the reuse boundary, not just an aggregate oracle score.
    report["candidate_equivalence_anchors"] = assert_same_candidates(source / "dev_scores.jsonl", control / "dev_scores.jsonl")
    bench = benchmark(work, control, 5000)
    report["trials"].append({"name": "control", "dev": dev["metrics"], "threshold": dev["threshold"]})
    report["control_benchmark"] = bench
    save_json(output / "quality.json", report)
    # Leave ten minutes for optional refit/evaluation, plus inference and release.
    room = remaining_seconds(deadline) - reserve_minutes*60 - margin*bench["extrapolated_scoring_seconds"]
    if room > 600:
        deep = output / "deep"
        try:
            deeper_cfg = dict(cfg, max_depth=8, trees=900, max_bin=128)
            refit(deep, deeper_cfg, source, loaded)
            deep_dev = evaluate(work, deep)
            report["trials"].append({"name": "deep", "dev": deep_dev["metrics"], "threshold": deep_dev["threshold"]})
        except Exception as exc:
            # Preserve the completed control when an optional larger GPU model fails.
            report["deep_failure"] = f"{type(exc).__name__}: {exc}"
            log(f"Optional deep trial failed; control is preserved: {exc}")
        save_json(output / "quality.json", report)
        room = remaining_seconds(deadline) - reserve_minutes*60 - margin*bench["extrapolated_scoring_seconds"]
        if any(t["name"] == "deep" for t in report["trials"]) and room > 300:
            combined = output / "ensemble"
            try:
                ensemble(combined, [control, deep])
                combined_dev = evaluate(work, combined)
                report["trials"].append({"name": "ensemble", "dev": combined_dev["metrics"], "threshold": combined_dev["threshold"]})
            except Exception as exc:
                report["ensemble_failure"] = f"{type(exc).__name__}: {exc}"
                log(f"Optional ensemble failed; completed models are preserved: {exc}")
    else:
        report["optional_trials_skipped"] = "Control ETA leaves insufficient time for optional model experiments"
    del loaded
    gc.collect()
    # Compare every finished model's F0.5; a deeper model/ensemble is not assumed better.
    ordered = sorted(report["trials"], key=lambda item: (item["dev"]["macro_f05"], item["name"] == "control"), reverse=True)
    report["best_model"] = ordered[0]["name"]
    report["minimum_dev_score"] = max(.8404988085983199, read_json(source / "dev_report.json")["metrics"]["macro_f05"]-.002)
    report["release_checks"] = []
    for trial in ordered:
        selected = output / trial["name"]
        final_bench = bench if trial["name"] == "control" else benchmark(work, selected, 5000)
        score_ok = trial["dev"]["macro_f05"] >= report["minimum_dev_score"]
        time_ok = margin*final_bench["extrapolated_scoring_seconds"]+reserve_minutes*60 <= remaining_seconds(deadline)
        report["release_checks"].append({"name": trial["name"], "score_passed": score_ok,
                                        "deadline_passed": time_ok, "benchmark": final_bench})
        if "benchmark" not in report:
            report["benchmark"] = final_bench
        if score_ok and time_ok:
            report["selected"] = trial["name"]
            report["benchmark"] = final_bench
            report["selected_run_sha256"] = sha256(selected / "run.json")
            report["selected_decision_sha256"] = sha256(selected / "decision.json")
            report["selected_benchmark_sha256"] = sha256(selected / "benchmark_test.json")
            break
    report["score_passed"] = any(item["score_passed"] for item in report["release_checks"])
    report["deadline_passed"] = report["selected"] is not None
    report["complete"] = True
    save_json(output / "quality.json", report)
    log(f"Best development score={ordered[0]['dev']['macro_f05']:.6f}; release selection={report['selected'] or 'NONE (see score/deadline gates)'}")
    return report


def verify_quality(output):
    output = Path(output)
    report = read_json(output / "quality.json")
    if not report.get("complete") or report.get("selected") not in {"control", "deep", "ensemble"}:
        raise ValueError("No release-ready model was selected; inspect quality.json")
    if report["source_sha256"] != source_hash():
        raise ValueError("Source changed after the quality sweep")
    run = output / report["selected"]
    for filename, key in (("run.json", "selected_run_sha256"), ("decision.json", "selected_decision_sha256"),
                          ("benchmark_test.json", "selected_benchmark_sha256")):
        if sha256(run / filename) != report[key]:
            raise ValueError(f"Selected {filename} changed after selection")
    needed = report["timing_margin"]*report["benchmark"]["extrapolated_scoring_seconds"] + report["reserve_minutes"]*60
    if needed > remaining_seconds(report["deadline"]):
        raise ValueError("Selected model no longer fits the remaining deadline/release buffer")
    return run


def rebuild_ensemble(dataset, work, assets, output):
    """Rebuild a shipped equal-weight ensemble from organizer data, without old .npy files."""
    from .database import prepare
    from .model import train
    from .evaluate import evaluate
    assets, output = Path(assets), Path(output)
    if output.exists():
        raise ValueError("Rebuild requires a fresh output directory")
    meta = read_json(assets / "run.json")
    if meta["source_sha256"] != source_hash() or meta["config"]["model_backend"] != "xgboost_ensemble":
        raise ValueError("Rebuild requires the ensemble's packaged source and assets")
    checked = TreeModel(meta["config"], assets / "ensemble.json")
    del checked
    members = read_json(assets / "ensemble.json")["members"]
    if any(abs(m["weight"]-1/len(members)) > 1e-9 for m in members):
        raise ValueError("Rebuild currently supports equal-weight ensembles")
    cfg = members[0]["config"]
    prepare(dataset, work, "train", cfg)
    feature_run = output / "features"
    train(work, feature_run, cfg)
    loaded = load_training_features(work, feature_run)
    paths = []
    for idx, member in enumerate(members):
        path = output / f"member-{idx}"
        refit(path, member["config"], feature_run, loaded)
        paths.append(path)
    final = output / "final"
    ensemble(final, paths)
    evaluate(work, final)
    del loaded
    return final
