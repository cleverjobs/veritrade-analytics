"""Synthetic stand-ins for the public datasets: same file formats, invented text."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from veritrade.inference.types import Sentiment
from veritrade_evals.datasets import FPB_MEMBER, DatasetName, Example

TWITTER_IDS: dict[Sentiment, int] = {"bearish": 0, "bullish": 1, "neutral": 2}
REVISION = "a" * 40


def ex(i: int, label: Sentiment, text: str | None = None, dataset: DatasetName = "tfns") -> Example:
    return Example(f"x:{i}", text if text is not None else f"{label} headline {i}", label, dataset, "train")


def twitter_train_rows() -> list[tuple[str, int]]:
    rows = [(f"bear headline {i}", 0) for i in range(105)]
    rows += [(f"bull headline {i}", 1) for i in range(120)]
    rows += [(f"neutral headline {i}", 2) for i in range(200)]
    rows += [("bull headline 0", 1), ("mixed headline", 1), ("Mixed  HEADLINE", 0)]
    return rows


def twitter_validation_rows() -> list[tuple[str, int]]:
    rows = [("Neutral Headline 0 https://t.co/abc", 2)]
    rows += [(f"val bear {i}", 0) for i in range(5)]
    rows += [(f"val bull {i}", 1) for i in range(5)]
    rows += [(f"val neutral {i}", 2) for i in range(10)]
    return rows


FPB_LINES = [
    "Alpha rose 5 % @ EUR 1.2@positive",
    "Beta fell@negative",
    "Gamma flat@neutral",
    "Gamma flat@neutral",
    "Delta@positive",
    "Delta@negative",
    "Café stable@neutral",
]

FINENTITY_ROWS: list[dict[str, Any]] = [
    {"content": "no entities here", "annotations": []},
    {"content": "Acme beats", "annotations": [{"label": "Positive"}]},
    {"content": "Acme and Bolt beat", "annotations": [{"label": "Positive"}, {"label": "Positive"}]},
    {"content": "Acme up, Bolt down", "annotations": [{"label": "Positive"}, {"label": "Negative"}]},
    {"content": " Acme holds ", "annotations": [{"label": "Neutral"}]},
]


def write_parquet(path: Path, rows: Sequence[tuple[str, int]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"text": [t for t, _ in rows], "label": pa.array([lab for _, lab in rows], pa.int64())})
    pq.write_table(table, path)
    return path


def write_fpb_zip(path: Path, lines: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("__MACOSX/FinancialPhraseBank-v1.0/._Sentences_AllAgree.txt", b"junk")
        zf.writestr(FPB_MEMBER, ("\r\n".join(lines) + "\r\n").encode("iso-8859-1"))
    return path


def write_json(path: Path, rows: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path
