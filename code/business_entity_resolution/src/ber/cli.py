from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import load_config, log, read_json
from .database import prepare


def parser():
    root = argparse.ArgumentParser(description="RestoreBuildRun business entity resolution baseline")
    subs = root.add_subparsers(dest="command", required=True)
    subs.add_parser("check-gpu", help="Verify actual CUDA training and inference on a tiny synthetic dataset")
    def add(name, dataset=False, run=False, output=False, config=False):
        p = subs.add_parser(name)
        p.add_argument("--work", required=True, type=Path, help="Persistent local index/artifact directory")
        if dataset:
            p.add_argument("--dataset", required=True, type=Path, help="Directory containing train/ and test/")
        if run:
            p.add_argument("--run", required=True, type=Path, help="Immutable model run directory")
        if output:
            p.add_argument("--output", required=True, type=Path)
        if config:
            p.add_argument("--config", type=Path)
        return p
    p = add("prepare", dataset=True, config=True)
    p.add_argument("--split", choices=["train", "test", "both"], default="train")
    p = add("train", dataset=True, run=True, config=True)
    p.add_argument("--mode", choices=["learned", "rules"], default="learned")
    p = add("evaluate", run=True)
    p.add_argument("--partition", choices=["dev", "holdout"], default="dev")
    p.add_argument("--limit", type=int, default=None, help="Optional evaluation cap; 0 means all. Recorded in report.")
    add("predict", dataset=True, run=True, output=True)
    p = add("benchmark", dataset=True, run=True)
    p.add_argument("--limit", type=int, default=1000)
    p = add("retrieval-audit", config=True, output=True)
    p.add_argument("--limit", type=int, default=1000)
    p = add("retrieval-probe", config=True, output=True)
    p.add_argument("--dev-limit", type=int, default=2000)
    p.add_argument("--test-limit", type=int, default=3000)
    p.add_argument("--min-rate", type=float, default=150.0)
    p.add_argument("--check", action="store_true", help="Verify a completed probe before using its selected config")
    add("validate", output=True)
    p = add("package", run=True, output=True)
    p.add_argument("--destination", required=True, type=Path)
    p.add_argument("--members", required=True)
    p.add_argument("--team", default="RestoreBuildRun")
    p = add("run", dataset=True, run=True, output=True, config=True)
    p.add_argument("--mode", choices=["learned", "rules"], default="learned")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if getattr(args, "limit", None) is not None and args.limit < 0:
            raise ValueError("--limit must be nonnegative")
        if args.command == "check-gpu":
            from .trees import gpu_check
            print(json.dumps(gpu_check(), indent=2))
        elif args.command == "retrieval-audit":
            from .retrieval_audit import retrieval_audit
            retrieval_audit(args.work, load_config(args.config), args.output, args.limit)
        elif args.command == "retrieval-probe":
            from .retrieval_probe import retrieval_probe, verify_probe
            if args.check:
                verify_probe(args.work, args.output)
                log("Probe source, index signatures and selected configuration verified")
            else:
                retrieval_probe(args.work, load_config(args.config), args.output,
                                args.dev_limit, args.test_limit, args.min_rate)
        elif args.command == "prepare":
            cfg = load_config(args.config)
            for split in (["train", "test"] if args.split == "both" else [args.split]):
                prepare(args.dataset, args.work, split, cfg)
        elif args.command in ("train", "run"):
            from .model import train
            cfg = load_config(args.config)
            prepare(args.dataset, args.work, "train", cfg)
            train(args.work, args.run, cfg, args.mode)
            if args.command == "run":
                from .evaluate import evaluate
                from .predict import predict
                from .validate import validate
                evaluate(args.work, args.run)
                prepare(args.dataset, args.work, "test", cfg)
                predict(args.work, args.run, args.output)
                validate(args.work, args.output)
        elif args.command == "evaluate":
            from .evaluate import evaluate
            evaluate(args.work, args.run, args.partition, args.limit)
        elif args.command == "predict":
            from .predict import predict
            cfg = read_json(args.run / "run.json")["config"]
            prepare(args.dataset, args.work, "test", cfg)
            predict(args.work, args.run, args.output)
        elif args.command == "benchmark":
            from .benchmark import benchmark
            cfg = read_json(args.run / "run.json")["config"]
            prepare(args.dataset, args.work, "test", cfg)
            benchmark(args.work, args.run, args.limit)
        elif args.command == "validate":
            from .validate import validate
            validate(args.work, args.output)
        elif args.command == "package":
            from .package import package
            print(package(args.work, args.run, args.output, args.destination, args.members, args.team))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        log(f"ERROR: {exc}")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
