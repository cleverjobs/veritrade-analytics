"""Committed split files hold ids, normalised-text sha1 and labels only; licences keep the text out."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np

from veritrade.inference.types import SENTIMENTS, Sentiment
from veritrade_evals.datasets import (
    DatasetName,
    Example,
    Store,
    class_counts,
    finentity_single,
    fpb_allagree,
    text_hash,
    twitter_fns,
)
from veritrade_evals.paths import SPLITS_DIR

SplitName = Literal["dev300", "tfns_val", "fpb_allagree", "finentity_single"]
SPLIT_NAMES: tuple[SplitName, ...] = get_args(SplitName)
HOLDOUT_SPLITS: tuple[SplitName, ...] = ("tfns_val", "fpb_allagree", "finentity_single")

DEV_SEED = 20261018
DEV_PER_CLASS = 100
# Stored in every split file so a changed normaliser fails `splits verify` instead of silently re-keying.
NORMALISATION = "nfkc-casefold-nourl-ws"


class SplitError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SplitSource:
    dataset: DatasetName
    source_split: str
    selection: str
    load: Callable[[Store], list[Example]]


SOURCES: dict[SplitName, SplitSource] = {
    "dev300": SplitSource(
        "tfns",
        "train",
        f"stratified {DEV_PER_CLASS} per class (seed {DEV_SEED}) from train after keeping one row per "
        "normalised text and excluding texts also in validation or labelled inconsistently in train",
        lambda store: twitter_fns("train", store=store),
    ),
    "tfns_val": SplitSource(
        "tfns", "validation", "all rows", lambda store: twitter_fns("validation", store=store)
    ),
    "fpb_allagree": SplitSource(
        "fpb",
        "all",
        "Sentences_AllAgree; exact duplicates collapsed, sentences with conflicting labels dropped",
        lambda store: fpb_allagree(store=store),
    ),
    "finentity_single": SplitSource(
        "finentity",
        "all",
        "rows with >= 1 entity annotation and exactly one distinct label",
        lambda store: finentity_single(store=store),
    ),
}


@dataclass(frozen=True, slots=True)
class Split:
    name: SplitName
    examples: tuple[Example, ...]
    class_weights: dict[Sentiment, float] | None = None

    def example_weights(self) -> list[float] | None:
        """Per-example prevalence weights for metrics, or None when the split already has the true prior."""
        if self.class_weights is None:
            return None
        return [self.class_weights[e.label] for e in self.examples]


# --- selection (pure) ------------------------------------------------------------------------


def dev_pool(train: Sequence[Example], validation: Sequence[Example]) -> tuple[list[Example], dict[str, int]]:
    """Train rows eligible for dev: no validation overlap, one row per normalised text, no label conflicts."""
    val_hashes = {text_hash(e.text) for e in validation}
    groups: dict[str, list[Example]] = defaultdict(list)
    overlap = 0
    for e in train:
        h = text_hash(e.text)
        if h in val_hashes:
            overlap += 1
        else:
            groups[h].append(e)
    pool: list[Example] = []
    duplicates = conflicts = 0
    for members in groups.values():
        if len({m.label for m in members}) > 1:
            conflicts += len(members)
            continue
        duplicates += len(members) - 1
        pool.append(members[0])
    order = {e.id: i for i, e in enumerate(train)}
    pool.sort(key=lambda e: order[e.id])
    excluded = {"validation_overlap": overlap, "train_duplicates": duplicates, "label_conflicts": conflicts}
    return pool, excluded


def stratified_sample(pool: Sequence[Example], per_class: int, seed: int) -> list[Example]:
    """Same pool and seed give the same ids; output keeps pool order."""
    rng = np.random.default_rng(seed)
    order = {e.id: i for i, e in enumerate(pool)}
    chosen: list[Example] = []
    for label in SENTIMENTS:
        members = [e for e in pool if e.label == label]
        if len(members) < per_class:
            raise SplitError(f"only {len(members)} {label} examples, need {per_class}")
        picks = rng.choice(len(members), size=per_class, replace=False)
        chosen.extend(members[int(i)] for i in picks)
    return sorted(chosen, key=lambda e: order[e.id])


def prevalence_weights(
    population: Sequence[Example], sample: Sequence[Example]
) -> tuple[dict[Sentiment, float], dict[Sentiment, float]]:
    """(population prior, weight = prior / sample prior) so balanced-sample metrics estimate the real mix."""
    pop, smp = class_counts(population), class_counts(sample)
    prior = {c: pop[c] / len(population) for c in SENTIMENTS}
    weights = {c: prior[c] / (smp[c] / len(sample)) for c in SENTIMENTS if smp[c]}
    return prior, weights


# --- split files -----------------------------------------------------------------------------


def split_payload(
    name: SplitName, examples: Sequence[Example], *, revision: str, extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    source = SOURCES[name]
    return {
        "name": name,
        "dataset": source.dataset,
        "source_split": source.source_split,
        "revision": revision,
        "selection": source.selection,
        "normalisation": NORMALISATION,
        "n": len(examples),
        "class_counts": class_counts(examples),
        **(extra or {}),
        "examples": [{"id": e.id, "sha1": text_hash(e.text), "label": e.label} for e in examples],
    }


def dumps_split(payload: Mapping[str, Any]) -> str:
    """Header pretty-printed, one example per line, so diffs stay readable."""
    head = {k: v for k, v in payload.items() if k != "examples"}
    lines = json.dumps(head, indent=2).splitlines()[:-1]
    lines[-1] += ","
    rows = [json.dumps(row) for row in payload["examples"]]
    lines += ['  "examples": [', *(f"    {row}," for row in rows[:-1]), *(f"    {row}" for row in rows[-1:])]
    return "\n".join([*lines, "  ]", "}"]) + "\n"


def split_path(name: SplitName, splits_dir: Path = SPLITS_DIR) -> Path:
    return splits_dir / f"{name}.json"


def read_split_file(name: SplitName, splits_dir: Path = SPLITS_DIR) -> dict[str, Any]:
    path = split_path(name, splits_dir)
    if not path.exists():
        raise SplitError(f"no split file {path}; run `uv run python -m veritrade_evals splits make`")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def resolve_split(payload: Mapping[str, Any], available: Mapping[str, Example]) -> Split:
    """Rebuild a split from dataset rows, failing on any missing id, text drift or label drift."""
    name: SplitName = payload["name"]
    if payload.get("normalisation") != NORMALISATION:
        raise SplitError(f"{name}: normalisation {payload.get('normalisation')!r} != {NORMALISATION!r}")
    problems: list[str] = []
    examples: list[Example] = []
    for row in payload["examples"]:
        example = available.get(row["id"])
        if example is None:
            problems.append(f"{row['id']}: missing from dataset")
        elif text_hash(example.text) != row["sha1"]:
            problems.append(f"{row['id']}: text hash mismatch")
        elif example.label != row["label"]:
            problems.append(f"{row['id']}: label {example.label} != {row['label']}")
        else:
            examples.append(example)
    if problems:
        raise SplitError(f"{name}: {len(problems)} bad rows, e.g. {'; '.join(problems[:3])}")
    weights = payload.get("class_weights")
    return Split(name, tuple(examples), dict(weights) if weights is not None else None)


# --- orchestration ---------------------------------------------------------------------------


def build_split(name: SplitName, store: Store, cache: dict[SplitName, list[Example]]) -> dict[str, Any]:
    revision = store.entry(SOURCES[name].dataset).revision
    if name != "dev300":
        return split_payload(name, _load(name, store, cache), revision=revision)
    train = _load("dev300", store, cache)
    pool, excluded = dev_pool(train, _load("tfns_val", store, cache))
    sample = stratified_sample(pool, DEV_PER_CLASS, DEV_SEED)
    prior, weights = prevalence_weights(train, sample)
    extra = {"seed": DEV_SEED, "train_prior": prior, "class_weights": weights, "excluded": excluded}
    return split_payload(name, sample, revision=revision, extra=extra)


def make_splits(*, store: Store | None = None, splits_dir: Path = SPLITS_DIR) -> dict[SplitName, Path]:
    store = store or Store()
    cache: dict[SplitName, list[Example]] = {}
    splits_dir.mkdir(parents=True, exist_ok=True)
    written: dict[SplitName, Path] = {}
    for name in SPLIT_NAMES:
        path = split_path(name, splits_dir)
        path.write_text(dumps_split(build_split(name, store, cache)), encoding="utf-8")
        written[name] = path
    return written


def load_split(name: SplitName, *, store: Store | None = None, splits_dir: Path = SPLITS_DIR) -> Split:
    store = store or Store()
    payload = read_split_file(name, splits_dir)
    locked = store.entry(SOURCES[name].dataset).revision
    if payload["revision"] != locked:
        raise SplitError(
            f"{name} was made at {payload['revision']} but the lock pins {locked}; re-run splits make"
        )
    return resolve_split(payload, {e.id: e for e in SOURCES[name].load(store)})


def _load(name: SplitName, store: Store, cache: dict[SplitName, list[Example]]) -> list[Example]:
    if name not in cache:
        cache[name] = SOURCES[name].load(store)
    return cache[name]
