from __future__ import annotations

from pathlib import Path

import pytest

from veritrade_evals import datasets
from veritrade_evals.datasets import SPECS, DatasetSpec, LockedFile, LockEntry, Store, write_lock

from .synthetic import (
    FINENTITY_ROWS,
    FPB_LINES,
    REVISION,
    twitter_train_rows,
    twitter_validation_rows,
    write_fpb_zip,
    write_json,
    write_parquet,
)


@pytest.fixture
def hub_files(tmp_path: Path) -> dict[str, Path]:
    hub = tmp_path / "hub"
    return {
        "default/train/0000.parquet": write_parquet(hub / "train.parquet", twitter_train_rows()),
        "default/validation/0000.parquet": write_parquet(
            hub / "validation.parquet", twitter_validation_rows()
        ),
        "data/FinancialPhraseBank-v1.0.zip": write_fpb_zip(hub / "fpb.zip", FPB_LINES),
        "FinEntity.json": write_json(hub / "FinEntity.json", FINENTITY_ROWS),
    }


@pytest.fixture
def store(tmp_path: Path, hub_files: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> Store:
    """A Store whose lock pins synthetic files and whose downloads never leave the temp dir."""

    def fake_download(spec: DatasetSpec, filename: str, revision: str, cache_dir: Path) -> Path:
        assert revision == REVISION
        return hub_files[filename]

    monkeypatch.setattr(datasets, "_download", fake_download)
    lock = {
        name: LockEntry(
            repo_id=spec.repo_id,
            ref=spec.ref,
            revision=REVISION,
            license=spec.license,
            files={
                f: LockedFile(datasets._sha256(hub_files[f]), hub_files[f].stat().st_size) for f in spec.files
            },
        )
        for name, spec in SPECS.items()
    }
    lock_path = tmp_path / "datasets.lock.json"
    write_lock(lock, lock_path)
    return Store(lock_path=lock_path, cache_dir=tmp_path / "cache")
