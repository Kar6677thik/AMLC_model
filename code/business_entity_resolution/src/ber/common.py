from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

DEFAULTS = {
    "seed": 42, "candidates_per_source": 20, "posting_limit": 300,
    "query_keys": 6, "name_tokens": 6, "address_tokens": 3, "name_grams": 6,
    "train_anchors": 25000, "dev_anchors": 10000,
    "prediction_batch_anchors": 128, "prediction_shard_anchors": 10000, "trees": 250, "learning_rate": 0.05,
    "num_leaves": 31, "min_child_samples": 50, "threads": 8,
    "retrieval_version": "v1", "model_backend": "lightgbm", "device": "cpu",
    "feature_workers": 1, "feature_batch_anchors": 128,
    "query_expansion": 1, "intersection_budget": 3,
    "max_depth": 6, "max_bin": 128,
    "intersection_min_pool": 8, "intersection_min_score": 0.70,
}
SCHEMA_VERSION = 1


def load_config(path=None):
    cfg = dict(DEFAULTS)
    if path:
        supplied = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        unknown = supplied.keys() - cfg.keys()
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        cfg.update(supplied)
    for key, default in DEFAULTS.items():
        if isinstance(default, str):
            allowed = {"retrieval_version": {"v1", "v2", "v3"}, "model_backend": {"lightgbm", "xgboost"},
                       "device": {"cpu", "cuda:0"}}
            if cfg[key] not in allowed[key]:
                raise ValueError(f"{key} must be one of {sorted(allowed[key])}")
        elif isinstance(default, int):
            if type(cfg[key]) is not int or cfg[key] < (0 if key in ("seed", "intersection_budget") else 1):
                raise ValueError(f"{key} must be a positive integer (seed/intersection_budget may be zero)")
        elif not isinstance(cfg[key], (int, float)) or not 0 < cfg[key] <= 1:
            raise ValueError(f"{key} must lie in (0, 1]")
    if cfg["device"].startswith("cuda") and cfg["model_backend"] != "xgboost":
        raise ValueError("CUDA requires model_backend=xgboost; LightGBM remains the CPU fallback")
    return cfg


def stable_int(value):
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big") & ((1 << 63) - 1)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def environment():
    from importlib.metadata import version
    result = {"python": sys.version, "platform": platform.platform()}
    for name in ("numpy", "rapidfuzz", "lightgbm", "xgboost"):
        try:
            result[name] = version(name)
        except Exception:
            result[name] = None
    try:
        result["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        result["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        result["git_sha"] = None
    return result


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
