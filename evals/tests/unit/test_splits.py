from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from veritrade.inference.types import Sentiment
from veritrade_evals.datasets import Example, Store, class_counts, twitter_fns
from veritrade_evals.splits import (
    DEV_PER_CLASS,
    SPLIT_NAMES,
    SplitError,
    build_split,
    dev_pool,
    dumps_split,
    load_split,
    make_splits,
    prevalence_weights,
    resolve_split,
    split_path,
    stratified_sample,
)

from .synthetic import ex


def _pool(counts: dict[Sentiment, int]) -> list[Example]:
    labels = [label for label, k in counts.items() for _ in range(k)]
    return [ex(i, label) for i, label in enumerate(labels)]


def test_dev_pool_drops_validation_overlap_duplicates_and_conflicts() -> None:
    train = [
        ex(0, "bullish", "Apple up"),
        ex(1, "bullish", "apple UP https://t.co/x"),
        ex(2, "neutral", "Flat day"),
        ex(3, "bearish", "flat  DAY"),
        ex(4, "bearish", "Oil falls"),
        ex(5, "neutral", "Seen in val"),
    ]
    validation = [Example("v:0", "seen in VAL https://t.co/q", "neutral", "tfns", "validation")]
    pool, excluded = dev_pool(train, validation)
    assert [e.id for e in pool] == ["x:0", "x:4"]
    assert excluded == {"validation_overlap": 1, "train_duplicates": 1, "label_conflicts": 2}


def test_stratified_sample_is_balanced_deterministic_and_ordered() -> None:
    pool = _pool({"bullish": 30, "bearish": 20, "neutral": 50})
    first = stratified_sample(pool, 10, seed=7)
    assert class_counts(first) == {"bullish": 10, "bearish": 10, "neutral": 10}
    assert [e.id for e in stratified_sample(pool, 10, seed=7)] == [e.id for e in first]
    assert [e.id for e in stratified_sample(pool, 10, seed=8)] != [e.id for e in first]
    positions = [pool.index(e) for e in first]
    assert positions == sorted(positions)


def test_stratified_sample_needs_enough_per_class() -> None:
    with pytest.raises(SplitError, match="only 5 bearish"):
        stratified_sample(_pool({"bullish": 20, "bearish": 5, "neutral": 20}), 10, seed=1)


def test_prevalence_weights_restore_the_population_prior() -> None:
    population = _pool({"neutral": 60, "bullish": 30, "bearish": 10})
    sample = _pool({"bullish": 10, "bearish": 10, "neutral": 10})
    prior, weights = prevalence_weights(population, sample)
    assert prior == pytest.approx({"bullish": 0.3, "bearish": 0.1, "neutral": 0.6})
    assert weights == pytest.approx({"bullish": 0.9, "bearish": 0.3, "neutral": 1.8})


def test_make_splits_end_to_end(store: Store, tmp_path: Path) -> None:
    written = make_splits(store=store, splits_dir=tmp_path / "splits")
    assert set(written) == set(SPLIT_NAMES)

    dev = load_split("dev300", store=store, splits_dir=tmp_path / "splits")
    assert class_counts(dev.examples) == dict.fromkeys(("bullish", "bearish", "neutral"), DEV_PER_CLASS)
    excluded_rows = {"tfns:train:225", "tfns:train:425", "tfns:train:426", "tfns:train:427"}
    assert not excluded_rows & {e.id for e in dev.examples}
    assert dev.class_weights == pytest.approx(
        {"bullish": 122 / 428 * 3, "bearish": 106 / 428 * 3, "neutral": 200 / 428 * 3}
    )
    weights = dev.example_weights()
    assert weights is not None
    assert len(weights) == 300

    payload = json.loads(written["dev300"].read_text(encoding="utf-8"))
    assert payload["excluded"] == {"validation_overlap": 1, "train_duplicates": 1, "label_conflicts": 2}
    assert payload["seed"] == 20261018

    for name in ("tfns_val", "fpb_allagree", "finentity_single"):
        split = load_split(name, store=store, splits_dir=tmp_path / "splits")
        assert split.class_weights is None
        assert split.example_weights() is None
    assert len(load_split("tfns_val", store=store, splits_dir=tmp_path / "splits").examples) == 21


def test_split_files_carry_no_text_and_are_reproducible(store: Store, tmp_path: Path) -> None:
    a = make_splits(store=store, splits_dir=tmp_path / "a")
    b = make_splits(store=store, splits_dir=tmp_path / "b")
    for name in SPLIT_NAMES:
        text = a[name].read_text(encoding="utf-8")
        assert text == b[name].read_text(encoding="utf-8")
        assert "headline" not in text.lower()
        assert "Gamma" not in text
        rows = json.loads(text)["examples"]
        assert all(set(row) == {"id", "sha1", "label"} for row in rows)


def test_resolve_split_reports_missing_and_drifted_rows(store: Store) -> None:
    payload = build_split("tfns_val", store, {})
    available = {e.id: e for e in twitter_fns("validation", store=store)}
    resolve_split(payload, available)

    gone = dict(available)
    del gone["tfns:validation:0"]
    with pytest.raises(SplitError, match="missing from dataset"):
        resolve_split(payload, gone)

    edited = dict(available)
    edited["tfns:validation:1"] = dataclasses.replace(available["tfns:validation:1"], text="something else")
    with pytest.raises(SplitError, match="text hash mismatch"):
        resolve_split(payload, edited)

    relabelled = dict(available)
    relabelled["tfns:validation:1"] = dataclasses.replace(available["tfns:validation:1"], label="neutral")
    with pytest.raises(SplitError, match="label neutral != bearish"):
        resolve_split(payload, relabelled)

    with pytest.raises(SplitError, match="normalisation"):
        resolve_split({**payload, "normalisation": "other"}, available)


def test_load_split_refuses_a_different_revision(store: Store, tmp_path: Path) -> None:
    make_splits(store=store, splits_dir=tmp_path)
    path = split_path("fpb_allagree", tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(dumps_split({**payload, "revision": "b" * 40}), encoding="utf-8")
    with pytest.raises(SplitError, match="re-run splits make"):
        load_split("fpb_allagree", store=store, splits_dir=tmp_path)


def test_load_split_without_file_points_at_make(store: Store, tmp_path: Path) -> None:
    with pytest.raises(SplitError, match="splits make"):
        load_split("dev300", store=store, splits_dir=tmp_path)
