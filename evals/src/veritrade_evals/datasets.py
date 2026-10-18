"""Public sentiment datasets pinned to Hugging Face commits; the text only ever lives in the private cache."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from veritrade.inference.types import SENTIMENTS, Sentiment
from veritrade_evals.paths import DATASETS_LOCK, HF_CACHE_DIR

DatasetName = Literal["tfns", "fpb", "finentity"]
TwitterSplit = Literal["train", "validation"]


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    text: str
    label: Sentiment
    dataset: DatasetName
    split: str


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    repo_id: str
    ref: str
    files: tuple[str, ...]
    license: str


SPECS: dict[DatasetName, DatasetSpec] = {
    "tfns": DatasetSpec(
        repo_id="zeroshot/twitter-financial-news-sentiment",
        ref="refs/convert/parquet",
        files=("default/train/0000.parquet", "default/validation/0000.parquet"),
        license="MIT",
    ),
    "fpb": DatasetSpec(
        repo_id="takala/financial_phrasebank",
        ref="main",
        files=("data/FinancialPhraseBank-v1.0.zip",),
        license="CC BY-NC-SA 3.0",
    ),
    "finentity": DatasetSpec(
        repo_id="yixuantt/FinEntity",
        ref="main",
        files=("FinEntity.json",),
        license="ODC-BY",
    ),
}

TWITTER_LABELS: dict[int, Sentiment] = {0: "bearish", 1: "bullish", 2: "neutral"}
FPB_LABELS: dict[str, Sentiment] = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}
FINENTITY_LABELS: dict[str, Sentiment] = {"Positive": "bullish", "Negative": "bearish", "Neutral": "neutral"}
FPB_MEMBER = "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"
FETCH_HINT = "run `uv run python -m veritrade_evals datasets fetch` first"


class DatasetError(RuntimeError):
    pass


# --- text identity ---------------------------------------------------------------------------

_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """NFKC + casefold with URLs and whitespace runs dropped: reposts of a headline differ only in links."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _WS.sub(" ", _URL.sub(" ", folded)).strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(normalise(text).encode("utf-8"), usedforsecurity=False).hexdigest()


def class_counts(examples: Iterable[Example]) -> dict[Sentiment, int]:
    counts = Counter(e.label for e in examples)
    return {label: counts.get(label, 0) for label in SENTIMENTS}


# --- revision lock ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LockedFile:
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class LockEntry:
    repo_id: str
    ref: str
    revision: str
    license: str
    files: dict[str, LockedFile]


def read_lock(path: Path = DATASETS_LOCK) -> dict[DatasetName, LockEntry]:
    if not path.exists():
        raise DatasetError(f"no dataset lock at {path}; {FETCH_HINT}")
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))["datasets"]
    lock: dict[DatasetName, LockEntry] = {}
    for name in SPECS:
        if name in raw:
            entry = raw[name]
            files = {fname: LockedFile(**meta) for fname, meta in entry["files"].items()}
            lock[name] = LockEntry(
                repo_id=entry["repo_id"],
                ref=entry["ref"],
                revision=entry["revision"],
                license=entry["license"],
                files=files,
            )
    return lock


def write_lock(lock: Mapping[DatasetName, LockEntry], path: Path = DATASETS_LOCK) -> None:
    payload = {"datasets": {name: asdict(lock[name]) for name in SPECS if name in lock}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(spec: DatasetSpec, filename: str, revision: str, cache_dir: Path) -> Path:
    # A 40-char commit revision with a warm cache returns without touching the network.
    return Path(
        hf_hub_download(spec.repo_id, filename, repo_type="dataset", revision=revision, cache_dir=cache_dir)
    )


def resolve_revision(spec: DatasetSpec) -> str:
    sha = HfApi().dataset_info(spec.repo_id, revision=spec.ref).sha
    if not sha:
        raise DatasetError(f"could not resolve {spec.repo_id}@{spec.ref} to a commit")
    return sha


def fetch_all(
    *, lock_path: Path = DATASETS_LOCK, cache_dir: Path = HF_CACHE_DIR, update: bool = False
) -> dict[DatasetName, LockEntry]:
    """Download every dataset file at its locked revision (resolving refs only when unlocked or on update)."""
    previous = read_lock(lock_path) if lock_path.exists() else {}
    lock: dict[DatasetName, LockEntry] = {}
    for name, spec in SPECS.items():
        prev = previous.get(name)
        revision = prev.revision if prev is not None and not update else resolve_revision(spec)
        files: dict[str, LockedFile] = {}
        for filename in spec.files:
            path = _download(spec, filename, revision, cache_dir)
            digest = _sha256(path)
            locked = prev.files.get(filename) if prev is not None and prev.revision == revision else None
            if locked is not None and locked.sha256 != digest:
                raise DatasetError(f"{name}/{filename} sha256 differs from the lock at {revision}")
            files[filename] = LockedFile(sha256=digest, bytes=path.stat().st_size)
        lock[name] = LockEntry(spec.repo_id, spec.ref, revision, spec.license, files)
    write_lock(lock, lock_path)
    return lock


@dataclass(frozen=True, slots=True)
class Store:
    """Where the lock and the downloaded files live; tests point it at a temp dir."""

    lock_path: Path = DATASETS_LOCK
    cache_dir: Path = HF_CACHE_DIR

    def entry(self, name: DatasetName) -> LockEntry:
        entry = read_lock(self.lock_path).get(name)
        if entry is None:
            raise DatasetError(f"dataset {name!r} missing from {self.lock_path}; {FETCH_HINT}")
        return entry

    def path(self, name: DatasetName, filename: str) -> Path:
        entry = self.entry(name)
        locked = entry.files.get(filename)
        if locked is None:
            raise DatasetError(f"{name}/{filename} missing from {self.lock_path}; {FETCH_HINT}")
        path = _download(SPECS[name], filename, entry.revision, self.cache_dir)
        if _sha256(path) != locked.sha256:
            raise DatasetError(f"{path} does not match the locked sha256; delete it and re-fetch")
        return path


# --- parsers (pure) --------------------------------------------------------------------------


def twitter_examples(texts: Sequence[str], labels: Sequence[int], split: TwitterSplit) -> list[Example]:
    return [
        Example(f"tfns:{split}:{row}", text.strip(), TWITTER_LABELS[label], "tfns", split)
        for row, (text, label) in enumerate(zip(texts, labels, strict=True))
    ]


def parse_fpb(raw: str) -> list[Example]:
    """`sentence@label` lines; exact duplicates collapse, a sentence seen with two labels is dropped."""
    by_text: dict[str, Sentiment] = {}
    conflicted: set[str] = set()
    for lineno, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.rsplit("@", 1)
        if len(parts) != 2 or not parts[0].strip() or parts[1].strip() not in FPB_LABELS:
            raise DatasetError(f"FPB line {lineno} is not `sentence@label`")
        text, label = parts[0].strip(), FPB_LABELS[parts[1].strip()]
        if by_text.setdefault(text, label) != label:
            conflicted.add(text)
    examples = [
        Example(
            f"fpb:{hashlib.sha1(text.encode('utf-8'), usedforsecurity=False).hexdigest()[:12]}",
            text,
            label,
            "fpb",
            "all",
        )
        for text, label in by_text.items()
        if text not in conflicted
    ]
    _require_unique_ids(examples)
    return examples


def read_fpb_zip(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        return zf.read(FPB_MEMBER).decode("iso-8859-1")


def finentity_examples(rows: Sequence[Mapping[str, Any]]) -> list[Example]:
    """Paragraphs whose entity annotations all agree; mixed-sentiment paragraphs have no single gold label."""
    examples: list[Example] = []
    for idx, row in enumerate(rows):
        labels = {ann["label"] for ann in row.get("annotations") or ()}
        if len(labels) == 1:
            (label,) = labels
            examples.append(
                Example(
                    f"fin:{idx}", str(row["content"]).strip(), FINENTITY_LABELS[label], "finentity", "all"
                )
            )
    return examples


def _require_unique_ids(examples: Sequence[Example]) -> None:
    if len({e.id for e in examples}) != len(examples):
        raise DatasetError("duplicate example ids")


# --- loaders ---------------------------------------------------------------------------------


def twitter_fns(split: TwitterSplit, *, store: Store | None = None) -> list[Example]:
    path = (store or Store()).path("tfns", f"default/{split}/0000.parquet")
    table = pq.read_table(path, columns=["text", "label"])
    return twitter_examples(table.column("text").to_pylist(), table.column("label").to_pylist(), split)


def fpb_allagree(*, store: Store | None = None) -> list[Example]:
    return parse_fpb(read_fpb_zip((store or Store()).path("fpb", "data/FinancialPhraseBank-v1.0.zip")))


def finentity_single(*, store: Store | None = None) -> list[Example]:
    path = (store or Store()).path("finentity", "FinEntity.json")
    return finentity_examples(json.loads(path.read_text(encoding="utf-8")))
