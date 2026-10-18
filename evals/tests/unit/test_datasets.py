from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from veritrade_evals import datasets
from veritrade_evals.datasets import (
    FINENTITY_LABELS,
    FPB_LABELS,
    TWITTER_LABELS,
    DatasetError,
    Store,
    class_counts,
    finentity_examples,
    finentity_single,
    fpb_allagree,
    normalise,
    parse_fpb,
    read_fpb_zip,
    read_lock,
    text_hash,
    twitter_examples,
    twitter_fns,
)

from .synthetic import FINENTITY_ROWS, FPB_LINES, write_fpb_zip


def test_label_maps_match_upstream_cards() -> None:
    assert TWITTER_LABELS == {0: "bearish", 1: "bullish", 2: "neutral"}
    assert FPB_LABELS == {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}
    assert FINENTITY_LABELS == {"Positive": "bullish", "Negative": "bearish", "Neutral": "neutral"}


def test_twitter_examples_ids_labels_and_strip() -> None:
    examples = twitter_examples([" a ", "b", "c"], [0, 1, 2], "validation")
    assert [e.id for e in examples] == ["tfns:validation:0", "tfns:validation:1", "tfns:validation:2"]
    assert [e.label for e in examples] == ["bearish", "bullish", "neutral"]
    assert examples[0].text == "a"
    assert {(e.dataset, e.split) for e in examples} == {("tfns", "validation")}


def test_parse_fpb_splits_on_last_at_sign() -> None:
    (example,) = parse_fpb("Shares rose 5 % @ EUR 3.10 each@positive\n")
    assert example.text == "Shares rose 5 % @ EUR 3.10 each"
    assert example.label == "bullish"
    digest = hashlib.sha1(example.text.encode("utf-8"), usedforsecurity=False).hexdigest()
    assert example.id == f"fpb:{digest[:12]}"


def test_parse_fpb_dedupes_exact_duplicates_and_drops_conflicts() -> None:
    examples = parse_fpb("\n".join(FPB_LINES))
    by_text = {e.text: e.label for e in examples}
    assert by_text == {
        "Alpha rose 5 % @ EUR 1.2": "bullish",
        "Beta fell": "bearish",
        "Gamma flat": "neutral",
        "Café stable": "neutral",
    }


@pytest.mark.parametrize("line", ["no label here", "text@bullish", "@positive", "@"])
def test_parse_fpb_rejects_malformed_lines(line: str) -> None:
    with pytest.raises(DatasetError, match="line 1"):
        parse_fpb(line)


def test_read_fpb_zip_decodes_latin1_member(tmp_path: Path) -> None:
    raw = read_fpb_zip(write_fpb_zip(tmp_path / "fpb.zip", ["Café sales grew@positive"]))
    assert parse_fpb(raw)[0].text == "Café sales grew"


def test_finentity_keeps_only_single_label_rows() -> None:
    examples = finentity_examples(FINENTITY_ROWS)
    assert [(e.id, e.label) for e in examples] == [
        ("fin:1", "bullish"),
        ("fin:2", "bullish"),
        ("fin:4", "neutral"),
    ]
    assert examples[-1].text == "Acme holds"


def test_normalise_ignores_case_whitespace_and_urls() -> None:
    assert normalise("  $AAPL  UP\tbig https://t.co/abc ") == "$aapl up big"
    assert text_hash("Apple UP https://t.co/a") == text_hash("apple   up http://t.co/b")
    assert text_hash("apple up") != text_hash("apple down")


def test_class_counts_lists_every_class() -> None:
    assert class_counts(twitter_examples(["a"], [1], "train")) == {"bullish": 1, "bearish": 0, "neutral": 0}


def test_loaders_read_pinned_files(store: Store) -> None:
    train = twitter_fns("train", store=store)
    assert len(train) == 428
    assert class_counts(train) == {"bullish": 122, "bearish": 106, "neutral": 200}
    assert len(twitter_fns("validation", store=store)) == 21
    assert len(fpb_allagree(store=store)) == 4
    assert [e.id for e in finentity_single(store=store)] == ["fin:1", "fin:2", "fin:4"]


def test_store_rejects_a_file_that_drifted_from_the_lock(store: Store, hub_files: dict[str, Path]) -> None:
    hub_files["FinEntity.json"].write_text("[]", encoding="utf-8")
    with pytest.raises(DatasetError, match="locked sha256"):
        finentity_single(store=store)


def test_missing_lock_points_at_fetch(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="datasets fetch"):
        twitter_fns("train", store=Store(lock_path=tmp_path / "absent.json", cache_dir=tmp_path))


def test_fetch_all_records_revision_and_reuses_it(
    store: Store, hub_files: dict[str, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved: list[str] = []

    def fake_resolve(spec: datasets.DatasetSpec) -> str:
        resolved.append(spec.repo_id)
        return "a" * 40

    monkeypatch.setattr(datasets, "resolve_revision", fake_resolve)
    lock_path = tmp_path / "fresh.lock.json"
    datasets.fetch_all(lock_path=lock_path, cache_dir=tmp_path)
    assert len(resolved) == 3
    lock = read_lock(lock_path)
    assert lock["fpb"].revision == "a" * 40
    assert lock["tfns"].files["default/train/0000.parquet"].sha256 == datasets._sha256(
        hub_files["default/train/0000.parquet"]
    )
    datasets.fetch_all(lock_path=lock_path, cache_dir=tmp_path)
    assert len(resolved) == 3
