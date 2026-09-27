# 05 — Repository and compute-PC execution plan

## 1. Division between machines

| Machine | Responsibility |
| --- | --- |
| Current workspace | Planning, source/config edits, review, Git synchronization |
| Windows ThinkPad P16 | Dataset audit, environment checks, tests, features/indexes, training, evaluation, inference, packaging |

Confirmed compute-PC details: Windows, Intel Ultra 7, 32 GB RAM, ample disk. GPU model, VRAM, driver, and exact free disk are unknown. No assumption of CUDA until checked. All commands below are for the compute PC unless explicitly marked otherwise.

## 2. First compute-PC preflight

Run in PowerShell and retain a compact report:

```powershell
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
Get-PSDrive -PSProvider FileSystem
Get-Command python, py, git -ErrorAction SilentlyContinue
nvidia-smi
```

If `nvidia-smi` is unavailable, identify the adapter/driver before selecting a GPU environment; Windows display-adapter memory reports are not a reliable VRAM measurement. The CPU baseline must still run. Once PyTorch is installed, verify actual GPU detection with a tiny allocation and timed forward pass.

Use AC power and confirm the intended Windows performance/sleep settings before long runs. Avoid competing jobs on a thermally constrained laptop. Benchmark sustained throughput, not only the first warm batch.

Use the existing working Python environment if suitable. Otherwise choose a supported Python version with native Windows wheels, provisionally Python 3.11. Verify package compatibility on that PC and freeze exact versions only after a successful smoke run. Do not invent a tested lockfile during planning.

For tonight, do not make WSL, Docker, a compiler toolchain, or a new GPU driver a prerequisite. Prefer native Windows wheels and a CPU fallback. An existing working WSL environment may be reused, but a late migration is not part of the critical path.

## 3. Git and data transfer

The configured remote is [Kar6677thik/AMLC_model](https://github.com/Kar6677thik/AMLC_model); local `main` had no commits when inspected. Before the first push, inspect remote history and visibility. Reconcile any existing remote content normally; do not force-push to initialize it.

Commit source, configs, plans, dependency specifications, tests, small aggregate reports, and license/provenance manifests. **Do not use `git add .` on the current untracked tree.** The TSVs are large, and several individual files are hundreds of MB. Use an explicit file allowlist after creating ignore rules.

Planned `.gitignore` coverage before first code commit:

```gitignore
student_resource/dataset/
data/
artifacts/
runs/
output/
dist/
models/
.venv/
__pycache__/
__MACOSX/
.env
*.log
```

Retain organizer README/template/validator files as references if redistribution is permitted. Ignore only the data subtree, not all documentation. Never commit credentials, access tokens, full row-level error dumps, or model binaries by accident.

Transfer original supplied data separately to the authorized compute PC by existing local/network storage and compare SHA-256 manifests there. Do not publish challenge data. Store large checkpoints, indexes, and run outputs outside Git; synchronize only necessary artifacts through approved private storage or local transfer.

Before a run, check out a recorded commit and confirm the tree is clean. After a run, return the compact report, config, commit hash, timings, and artifact hashes. Do not edit a live run's configuration in place. If the compute PC needs a code fix, commit/synchronize it before starting a replacement run.

## 4. Proposed repository layout

This is the target implementation structure, not a claim that these files currently exist.

```text
AMLC/
├── README.md
├── docs/                          # current planning documents
├── student_resource/              # unchanged organizer resources; data ignored
├── code/business_entity_resolution/
│   ├── README.md                   # exact setup/reproduction instructions
│   ├── pyproject.toml
│   ├── requirements.txt           # exact tested dependency pins
│   ├── configs/
│   │   ├── baseline.yaml
│   │   ├── compact_multilingual.yaml
│   │   └── final.yaml
│   ├── src/ber/
│   │   ├── cli.py                 # stage orchestration and run manifests
│   │   ├── io.py                  # TSV contracts and record maps
│   │   ├── audit.py
│   │   ├── normalize.py
│   │   ├── splits.py
│   │   ├── metrics.py
│   │   ├── blocking.py
│   │   ├── embeddings.py           # optional
│   │   ├── features.py
│   │   ├── train.py
│   │   ├── decisions.py
│   │   ├── predict.py
│   │   ├── validate.py
│   │   └── package.py
│   └── tests/                     # synthetic fixtures; run on compute PC
├── runs/<run_id>/                 # ignored; immutable manifests/reports
├── artifacts/                     # ignored; normalized tables/indexes/models
├── output/<release_id>/           # ignored; both TSVs from one run
└── dist/                          # ignored; final zip
```

Start with a small number of cohesive modules; do not spend deadline time creating empty abstractions. All implementation needed in the final package belongs under `code/business_entity_resolution/`. No runtime dependency on a notebook, absolute path on this PC, or unshipped helper.

## 5. Planned command contract

**These `ber` commands are interfaces to implement, not commands available now.** Prefer `python -m ber` with explicit paths and a config file. PowerShell environment variables and argument quoting must support paths containing spaces.

```powershell
$env:AMLC_DATA_DIR = 'D:\AMLC\student_resource\dataset'
$env:AMLC_ARTIFACT_DIR = 'D:\AMLC\artifacts'

python -m ber audit --config configs/baseline.yaml
python -m ber split --config configs/baseline.yaml
python -m ber train --config configs/baseline.yaml --run-id baseline-001
python -m ber evaluate --run-id baseline-001 --partition dev
python -m ber predict --run-id baseline-001 --partition test --output-dir output/baseline-001
python -m ber validate --output-dir output/baseline-001 --strict
python -m ber package --run-id baseline-001 --team RestoreBuildRun
```

Paths above are illustrative; use the actual data location. The package README must supply a real one-command end-to-end reproduction path once implemented. `train` and `predict` can orchestrate smaller resumable stages; users should not need to know internal shard filenames.

Configuration owns seeds, paths, split version, normalization, retrieval channels/budgets, feature list, model settings, thresholds, batching, worker count, and optional checkpoints. A resolved copy is saved in each run. Fail clearly on missing data, unsupported schema, incompatible cache, or an unverified model revision.

## 6. Dependencies and installation strategy

Baseline dependency candidates: NumPy; one tabular/columnar stack such as Polars/PyArrow; SciPy/scikit-learn where needed; RapidFuzz; LightGBM; YAML parser; pytest. Keep only dependencies actually used. Standard-library SQLite is a possible disk-backed index store if a native search dependency fails, but benchmark its query path before committing to it.

GPU extras: PyTorch matched to the detected hardware/driver, Transformers or Sentence Transformers, and one compatible ANN implementation. Do not install multiple heavyweight retrieval stacks speculatively. GPU training for LightGBM is not required; use CPU training to simplify setup.

Record Python, package versions, OS, driver, CUDA runtime where relevant, model revision, and deterministic settings. Capture a complete environment lock/export alongside a minimal pinned requirements file. Preserve package/model licenses. Hardware-specific setup instructions must explain the successful installation actually used.

## 7. Memory and storage plan for 32 GB RAM

Target peak process working memory below approximately **22–24 GiB**, leaving room for Windows, filesystem cache, and monitoring; adjust after observing the actual machine. Never assume the dataset's 2.52 GB on disk predicts its Python-object memory use.

- Stream TSV ingestion; use compact columnar intermediates and integer record maps.
- Avoid holding train and test indexes together unless measured memory permits.
- Partition indexes by source/country and process query shards with deterministic joins.
- Store feature shards without repeated text. Train on an intentional sample when all candidate features do not fit memory.
- Limit native thread pools and worker count to avoid oversubscription and duplicated arrays.
- Windows multiprocessing must use a guarded entry point; start with zero/few data-loader workers rather than forking assumptions.
- Flush intermediate shards and release large objects between stages. A failed memory allocation is a signal to reduce working set, not raise the candidate cap.
- Measure free disk before building caches; reserve space for original data, converted tables, indexes, two releases, checkpoints, temporary merges, and the zip. Actual requirements depend on measured pair count.

Embedding memory formula: `records × dimensions × bytes_per_value`, plus index/metadata/workspace and possible float32 conversion. Candidate storage scales with total emitted pairs, not only number of S1 entities. Export TSV IDs by streaming grouped integer pairs instead of materializing huge Python dictionaries of string sets.

## 8. Runtime estimation and stop gates

Benchmark representative query shards spanning countries, common names, missing addresses, and high candidate counts against full-size indexes. Include warmup, disk I/O, tokenization, feature generation, and output serialization. A tiny index benchmark will overstate full-scale throughput.

```text
T_total ≈ T_read + T_normalize + T_index
        + N1 / retrieval_queries_per_second
        + M / feature_pairs_per_second
        + M / matcher_pairs_per_second
        + T_write + T_validate + T_package
```

For dense retrieval add encoding and ANN build/query costs. For reranking add routed pairs divided by measured reranker throughput. Use a conservative safety factor based on shard variability and laptop thermal behavior. Keep the baseline if the new pipeline cannot finish, validate, and be rerun within the protected deadline window.

Optional GPU tiers after preflight: under 8 GB or unsupported GPU means CPU baseline/frozen compact experiments only; 8–12 GB can justify compact embedding inference if timed; larger VRAM permits bigger batches but does not override today's deadline. These are planning tiers, not guarantees of model fit.

## 9. Reproducibility and recovery

Every run records input hashes, Git SHA, resolved config, split IDs, feature schema, model hashes, checkpoint revisions/licenses, random seeds, environment, stage timings, peak RAM/VRAM, metrics, candidate counts, and output hashes.

Cache keys include all dependencies: data identity, normalization version, tokenizer/model revision, dimensions/dtype, index settings, and code/config hashes. A model or preprocessing change invalidates dependent embeddings/features; never reuse a cache solely because its filename exists.

Write temporary shard files and atomically rename only after successful completion. Resume only completed compatible shards. Merge in deterministic S1 order with stable target-ID sorting, detect duplicate shards, and verify full coverage. Keep the best completed release immutable while trying improvements.

Exact neural bitwise reproduction may depend on hardware. Record deterministic settings and establish tolerance/output agreement; investigate changed match decisions rather than claiming bitwise determinism without evidence.
