"""`uv run python -m veritrade_evals <command>`: dataset, split and (later) evaluation commands."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from veritrade_evals.datasets import (
    SPECS,
    DatasetError,
    Example,
    class_counts,
    fetch_all,
    finentity_single,
    fpb_allagree,
    twitter_fns,
)
from veritrade_evals.splits import SPLIT_NAMES, SplitError, load_split, make_splits, read_split_file

PLANNED = {
    "signal": "signal accuracy run over a split",
    "grid": "prompt grid per model",
    "freeze": "write frozen prompt versions and thresholds into models.toml",
    "extraction": "parser x extractor matrix",
    "latency": "latency suite",
    "summary": "summary evaluation",
    "select": "pick per-endpoint winners",
    "holdout": "one-shot holdout run",
    "report": "benchmarks.json and results README",
    "articles": "Opus-labelled article set",
    "opus": "Opus subagent prompts and judgements",
}


def _balance(examples: Sequence[Example]) -> str:
    counts = class_counts(examples)
    return " ".join(f"{k}={v} ({v / len(examples):.1%})" for k, v in counts.items())


def cmd_datasets_fetch(args: argparse.Namespace) -> int:
    lock = fetch_all(update=args.update)
    for name, entry in lock.items():
        print(f"{name}: {SPECS[name].repo_id}@{entry.revision[:12]} ({entry.license})")
    loaded = {
        "tfns train": twitter_fns("train"),
        "tfns validation": twitter_fns("validation"),
        "fpb allagree": fpb_allagree(),
        "finentity single": finentity_single(),
    }
    for label, examples in loaded.items():
        print(f"  {label}: n={len(examples)} {_balance(examples)}")
    return 0


def cmd_splits_make(args: argparse.Namespace) -> int:
    for name, path in make_splits().items():
        payload = read_split_file(name)
        print(f"{name}: n={payload['n']} {payload['class_counts']} -> {path.name}")
        if "class_weights" in payload:
            print(f"  train_prior={payload['train_prior']} weights={payload['class_weights']}")
            print(f"  excluded={payload['excluded']}")
    return 0


def cmd_splits_verify(args: argparse.Namespace) -> int:
    failed = 0
    for name in SPLIT_NAMES:
        try:
            split = load_split(name)
        except (SplitError, DatasetError) as exc:
            failed += 1
            print(f"FAIL {exc}", file=sys.stderr)
            continue
        print(f"ok {name}: n={len(split.examples)} {_balance(split.examples)}")
    return 1 if failed else 0


def cmd_planned(args: argparse.Namespace) -> int:
    print(f"`{args.command}` is not implemented yet ({PLANNED[args.command]})", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="veritrade_evals", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    datasets = commands.add_parser("datasets", help="public dataset downloads").add_subparsers(
        dest="action", required=True
    )
    fetch = datasets.add_parser(
        "fetch", help="download pinned revisions and record them in datasets.lock.json"
    )
    fetch.add_argument(
        "--update", action="store_true", help="re-resolve refs instead of reusing locked commits"
    )
    fetch.set_defaults(func=cmd_datasets_fetch)

    splits = commands.add_parser("splits", help="committed split files").add_subparsers(
        dest="action", required=True
    )
    splits.add_parser("make", help="write evals/data/splits/*.json").set_defaults(func=cmd_splits_make)
    splits.add_parser("verify", help="re-read datasets and check split hashes").set_defaults(
        func=cmd_splits_verify
    )

    for name, help_text in PLANNED.items():
        planned = commands.add_parser(name, help=f"{help_text} (not implemented yet)")
        planned.set_defaults(func=cmd_planned)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    # Planned commands accept anything so scripts written against them fail with the clear message below.
    if extra and args.func is not cmd_planned:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
