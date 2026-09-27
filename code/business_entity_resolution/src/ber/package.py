from __future__ import annotations

import json
import re
import zipfile
from datetime import date
from pathlib import Path

from .common import read_json, save_json, sha256, source_hash
from .validate import validate


def package(work, run, output, destination, members, team="RestoreBuildRun"):
    work, run, output, destination = map(Path, (work, run, output, destination))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", team):
        raise ValueError("Team must contain only letters, digits, underscore, or hyphen")
    if not members.strip():
        raise ValueError("Supply team member names for the methodology document")
    meta = read_json(run / "run.json")
    if meta["source_sha256"] != source_hash():
        raise ValueError("Cannot package different source from the trained run")
    prediction = read_json(output / "prediction.json")
    if prediction["identity"]["run_sha256"] != sha256(run / "run.json"):
        raise ValueError("Output belongs to a different run")
    if prediction["identity"]["decision_sha256"] != sha256(run / "decision.json"):
        raise ValueError("Output belongs to a different decision rule")
    validation = validate(work, output)
    environment_lock = (run / "environment.txt").read_text(encoding="utf-8").lower()
    for dependency in ("numpy", "rapidfuzz", "lightgbm"):
        if not any(line.startswith(dependency + "==") for line in environment_lock.splitlines()):
            raise ValueError(f"Exact {dependency} pin missing from environment.txt; resolve before packaging")
    dev = read_json(run / "dev_report.json")
    holdout = read_json(run / "holdout_report.json") if (run / "holdout_report.json").exists() else None
    source_dir = Path(__file__).resolve().parents[2]
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / f"{team}_submission.zip"
    if archive_path.exists():
        raise ValueError(f"Archive exists; use a new --destination: {archive_path}")
    prefix = "code/business_entity_resolution/"
    body = f"""# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** {team}  
**Team Members:** {members}  
**Submission Date:** {date.today().isoformat()}

## 1. Executive Summary
The submitted system retrieves bounded lexical candidates using a disk-backed inverted index and scores name/address identity evidence with a {meta['mode']} matcher. A global threshold selected on grouped development data produces zero, one, or multiple matches per Source 1 entity. All scored candidates are exported and strict validation verifies ID membership and reproduces decisions from the score ledger.

## 2. Methodology
### 2.1 Problem Analysis
Input checks validate unique IDs, label coverage and target references. Country labels are treated as open strings; normalization retains Unicode and additional accent-folded matching views. The full measured input audit is in assets/train_manifest.json and assets/test_manifest.json; no France accuracy is claimed without labels.

### 2.2 Solution Strategy
The supervised split is a stable hash split of labeled connected components (approximately 70/15/15). Shared targets keep anchors together. Held-out positive targets and exact normalized duplicate target records are excluded from supervised fitting. Retrieval index frequencies use the unlabeled target corpus, as available at inference, rather than labels. The baseline uses supplied task records only; no external identity lookup or task-data augmentation is used.

## 3. Candidate Generation (Blocking)
Source-specific global keys cover exact name/address, normalized name, informative name tokens, selected name trigrams, and name/address or initials/address combinations. Posting lists exceeding the configured limit are skipped or intersected within a bounded fallback. Candidates are ranked by lexical similarity and capped separately per source; there is no closed country filter.

- Final scored/exported pairs: {validation['candidate_pairs']}
- Mean candidates per S1: {validation['candidate_pairs']/max(1, validation['anchors']):.4f}
- Development candidate recall and oracle ceiling: see the measured report below.
- Source-specific candidate cap: {meta['config']['candidates_per_source']}
- The budget constrains attainable recall; it does not guarantee complete recovery for high-multiplicity entities.

## 4. Matching Model
Features comprise name/address string and token similarity, numeric agreement/conflict, missingness, country agreement/conflict, source, retrieval evidence, and candidate counts. Learned mode uses LightGBM with per-anchor normalized pair weights on sampled complete anchor groups. Rules mode uses a conservative fixed lexical scoring function. The selected mode is **{meta['mode']}**, with frozen decision threshold **{dev['threshold']}**.

The metric is per-S1 macro F0.5, including singleton credit. The threshold grid is selected on development only; an optional holdout assessment never retunes it. No neural checkpoint is deployed by this baseline. Code and trained baseline artifacts are provided under MIT; third-party notices are included.

## 5. Results & Error Analysis
Development evaluation used **{dev['metrics']['anchors']} anchors**. These are development results used for threshold selection, not an unbiased final score. Exact figures and country/multiplicity slices are preserved in assets/dev_report.json.

```json
{json.dumps(dev['metrics'], indent=2)}
```

Holdout evaluation: {('available in assets/holdout_report.json; macro F0.5 = ' + str(holdout['metrics']['macro_f05'])) if holdout else 'not performed for this release; no held-out performance claim.'}

Known structural limitations include high-frequency posting suppression, truncation of high-multiplicity candidate sets, difficult transliterations, same-name branch ambiguity, and unmeasured France generalization. These are potential limitations, not a claim that manual error analysis was completed. Measured error slices appear in the reports; add any team-reviewed examples before final submission if available.

## 6. Conclusion
This release prioritizes a complete, reproducible baseline and an auditable candidate boundary. Runtime and hardware observations are in the bundled manifests. The archive must still be tested in an extracted clean directory and submitted by the team; packaging itself does not establish reproduction or portal acceptance.

## Appendix
### A. Code Artefacts
Source is under src/ber; README.md gives exact setup and stage commands. configs/final.json and assets contain the frozen configuration, model (if learned), decision, environment and audit reports. The original organizer data must be supplied separately. Output hashes are in assets/prediction.json and assets/validation.json.

### B. Additional Results
All development threshold comparisons, candidate counts, and slices are in the bundled reports. Training and inference timings are measured per stage/invocation; resumed-run invocation timings are not represented as total end-to-end runtime.
"""
    temporary = archive_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            archive.write(output / name, "output/" + name)
        for name in ("README.md", "pyproject.toml", "LICENSE", "THIRD_PARTY.md"):
            archive.write(source_dir / name, prefix + name)
        for path in sorted((source_dir / "src").rglob("*.py")):
            archive.write(path, prefix + path.relative_to(source_dir).as_posix())
        for path in sorted((source_dir / "tests").glob("*.py")):
            archive.write(path, prefix + path.relative_to(source_dir).as_posix())
        archive.writestr(prefix + "configs/final.json", json.dumps(meta["config"], indent=2) + "\n")
        archive.write(run / "environment.txt", prefix + "requirements.txt")
        for name in ("run.json", "training.json", "decision.json", "model.txt", "dev_report.json", "holdout_report.json", "benchmark_test.json", "environment.txt", "training_anchors.tsv"):
            if (run / name).exists():
                archive.write(run / name, prefix + "assets/" + name)
        for name in ("train_manifest.json", "test_manifest.json"):
            archive.write(work / name, prefix + "assets/" + name)
        for name in ("prediction.json", "validation.json"):
            archive.write(output / name, prefix + "assets/" + name)
        archive.writestr("Documentation_template.md", body)
    temporary.replace(archive_path)
    save_json(destination / "release.json", {"archive": archive_path.name, "sha256": sha256(archive_path),
        "output_hashes": prediction["hashes"], "team": team, "clean_reproduction_verified": False,
        "portal_submitted": False, "notes": "Review methodology and perform extracted-package reproduction before submission."})
    return archive_path
