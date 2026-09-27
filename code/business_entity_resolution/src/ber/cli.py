from __future__ import annotations

import argparse
from pathlib import Path

from .common import load_config, log, read_json
from .database import prepare


def parser():
    root = argparse.ArgumentParser(description="RestoreBuildRun business entity resolution baseline")
    subs = root.add_subparsers(dest="command", required=True)
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
        if args.command == "prepare":
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
