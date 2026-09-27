"""Explicit CPU/GPU tree backends; CUDA requests never silently fall back."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
import numpy as np
from .features import FEATURES
from .common import read_json, sha256


def assert_cuda(model):
    actual = json.loads(model.save_config())["learner"]["generic_param"]["device"]
    if not actual.startswith("cuda"):
        raise ValueError(f"CUDA requested but XGBoost selected {actual!r}. Check driver/wheel or explicitly use optimized-cpu.json.")
    return actual


def gpu_check():
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ValueError("Install GPU extras: pip install -e 'code/business_entity_resolution[gpu]'") from exc
    build = xgb.build_info()
    if not build.get("USE_CUDA", False):
        raise ValueError("XGBoost has no CUDA support; install xgboost, not xgboost-cpu")
    x = np.random.default_rng(42).normal(size=(256, 8)).astype("float32")
    data = xgb.QuantileDMatrix(x, label=(x[:, 0] > 0).astype("float32"), max_bin=128, nthread=2)
    model = xgb.train({"device": "cuda:0", "tree_method": "hist", "objective": "binary:logistic",
        "max_depth": 2, "max_bin": 128, "nthread": 2}, data, num_boost_round=2)
    actual = assert_cuda(model)
    scores = model.predict(xgb.DMatrix(x, nthread=2))
    assert_cuda(model)
    if len(scores) != len(x) or not np.isfinite(scores).all():
        raise ValueError("GPU smoke check produced invalid results")
    try:
        hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
            "--format=csv"], text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    except (OSError, subprocess.SubprocessError):
        hardware = "nvidia-smi unavailable; actual CUDA execution verified by XGBoost"
    return {"passed": True, "device": actual, "xgboost": xgb.__version__, "build": build, "hardware": hardware}


class TreeModel:
    def __init__(self, cfg, path=None):
        self.cfg, self.model = cfg, None
        self.members = []
        self.backend = cfg.get("model_backend", "lightgbm")
        self.gpu = cfg.get("device", "cpu").startswith("cuda")
        if path:
            if self.backend == "xgboost_ensemble":
                manifest = read_json(path)
                for entry in manifest["members"]:
                    filename = entry["filename"]
                    if Path(filename).name != filename or not filename.endswith(".ubj"):
                        raise ValueError("Invalid ensemble member filename")
                    member_path = Path(path).parent / filename
                    if sha256(member_path) != entry["sha256"]:
                        raise ValueError("Ensemble member hash mismatch")
                    weight = float(entry["weight"])
                    if not np.isfinite(weight) or weight <= 0:
                        raise ValueError("Ensemble weights must be finite and positive")
                    member = TreeModel(dict(cfg, model_backend="xgboost"), member_path)
                    self.members.append((weight, member))
                if not self.members or abs(sum(w for w, _ in self.members)-1) > 1e-9:
                    raise ValueError("Ensemble weights must sum to one")
            elif self.backend == "xgboost":
                import xgboost as xgb
                self.model = xgb.Booster()
                self.model.load_model(path)
                self.model.set_param({"device": cfg["device"], "nthread": cfg["threads"]})
                if self.gpu:
                    assert_cuda(self.model)
            else:
                import lightgbm as lgb
                self.model = lgb.Booster(model_file=str(path))

    @property
    def filename(self):
        if self.backend == "xgboost_ensemble":
            return "ensemble.json"
        return "model.ubj" if self.backend == "xgboost" else "model.txt"

    def fit(self, x, y, weights):
        cfg = self.cfg
        if self.backend == "xgboost_ensemble":
            raise ValueError("Fit individual members, then build an ensemble")
        if self.backend == "xgboost":
            import xgboost as xgb
            data = xgb.QuantileDMatrix(x, label=y, weight=weights, feature_names=FEATURES,
                max_bin=cfg["max_bin"], nthread=cfg["threads"])
            self.model = xgb.train({"objective": "binary:logistic", "tree_method": "hist",
                "device": cfg["device"], "max_depth": cfg["max_depth"], "max_bin": cfg["max_bin"],
                "eta": cfg["learning_rate"], "seed": cfg["seed"], "nthread": cfg["threads"],
                "min_child_weight": 1, "reg_lambda": 2, "subsample": .9, "colsample_bytree": .9},
                data, num_boost_round=cfg["trees"])
            if self.gpu:
                assert_cuda(self.model)
        else:
            import lightgbm as lgb
            data = lgb.Dataset(x, label=y, weight=weights, feature_name=FEATURES)
            self.model = lgb.train({"objective": "binary", "learning_rate": cfg["learning_rate"],
                "num_leaves": cfg["num_leaves"], "min_data_in_leaf": cfg["min_child_samples"],
                "num_threads": cfg["threads"], "seed": cfg["seed"], "deterministic": True,
                "force_col_wise": True, "verbosity": -1}, data, num_boost_round=cfg["trees"])

    def predict(self, matrix):
        if self.backend == "xgboost_ensemble":
            return sum(weight * member.predict(matrix).astype("float64") for weight, member in self.members)
        if self.backend == "xgboost":
            import xgboost as xgb
            result = self.model.predict(xgb.DMatrix(matrix, feature_names=FEATURES, nthread=self.cfg["threads"]))
            if self.gpu:
                assert_cuda(self.model)
            return result
        return self.model.predict(matrix, num_threads=self.cfg["threads"])

    def save(self, path):
        self.model.save_model(str(path))

    def importance(self):
        if self.backend == "xgboost":
            gains = self.model.get_score(importance_type="total_gain")
            return {name: float(gains.get(name, 0)) for name in FEATURES}
        return dict(zip(FEATURES, map(float, self.model.feature_importance(importance_type="gain"))))
