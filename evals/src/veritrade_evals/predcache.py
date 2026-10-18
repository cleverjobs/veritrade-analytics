"""Resumable JSONL prediction cache, one file per (run config, model digest, prompt, harness) hash."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Self

from veritrade_evals.paths import PREDS_DIR

# Bump when harness changes alter predictions for an unchanged config, so old caches stop matching.
HARNESS_VERSION = "1"
RESERVED_KEYS = frozenset({"id", "gold", "error", "t"})
# Publisher and dataset licences: a record may describe a prediction, never carry its input.
FORBIDDEN_KEYS = frozenset({"text", "input", "input_text", "prompt", "messages", "article"})


class PredCacheError(RuntimeError):
    pass


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def run_hash(
    run_config: Mapping[str, Any], model_digest: str, prompt_sha: str, harness_version: str = HARNESS_VERSION
) -> str:
    payload = {
        "config": run_config,
        "model_digest": model_digest,
        "prompt_sha": prompt_sha,
        "harness_version": harness_version,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


@dataclass(frozen=True, slots=True)
class PredError:
    code: str
    message: str
    retryable: bool


def _terminal(record: Mapping[str, Any]) -> bool:
    error = record.get("error")
    return error is None or not error.get("retryable", False)


class PredCache:
    """Single writer per file (flock); every append is flushed so a killed run resumes from the last line."""

    def __init__(
        self,
        model: str,
        run_config: Mapping[str, Any],
        *,
        model_digest: str,
        prompt_sha: str,
        root: Path = PREDS_DIR,
        harness_version: str = HARNESS_VERSION,
    ) -> None:
        self.key = run_hash(run_config, model_digest, prompt_sha, harness_version)
        self.sidecar = {
            "key": self.key,
            "model": model,
            "model_digest": model_digest,
            "prompt_sha": prompt_sha,
            "harness_version": harness_version,
            "run_config": json.loads(canonical_json(run_config)),
        }
        directory = root / model_slug(model)
        self.path = directory / f"{self.key}.jsonl"
        self.config_path = directory / f"{self.key}.config.json"
        self._fh: IO[str] | None = None

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def open(self) -> Self:
        if self._fh is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_or_check_sidecar()
        fh = self.path.open("a", encoding="utf-8")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            fh.close()
            raise PredCacheError(f"{self.path} is already open for writing by another process") from exc
        self._drop_partial_tail()
        self._fh = fh
        return self

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def append(
        self,
        example_id: str,
        gold: str,
        *,
        pred: Mapping[str, Any] | None = None,
        error: PredError | None = None,
        t: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        if self._fh is None:
            raise PredCacheError("cache is not open; use `with PredCache(...) as cache`")
        fields = dict(pred or {})
        clash = fields.keys() & (RESERVED_KEYS | FORBIDDEN_KEYS)
        if clash:
            raise ValueError(f"prediction fields may not use {sorted(clash)}")
        record = {
            "id": example_id,
            "gold": gold,
            **fields,
            "error": asdict(error) if error else None,
            "t": dict(t or {}),
        }
        self._fh.write(canonical_json(record) + "\n")
        self._fh.flush()
        return record

    def iter_records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.endswith("\n"):
                    yield json.loads(line)

    def records(self) -> dict[str, dict[str, Any]]:
        """Per id: the last terminal record, else the last (retryable) failure."""
        latest: dict[str, dict[str, Any]] = {}
        for record in self.iter_records():
            current = latest.get(record["id"])
            if current is None or _terminal(record) or not _terminal(current):
                latest[record["id"]] = record
        return latest

    def done_ids(self) -> set[str]:
        """Ids to skip on resume: answered or failed for good; retryable errors run again."""
        return {rid for rid, record in self.records().items() if _terminal(record)}

    def _write_or_check_sidecar(self) -> None:
        if self.config_path.exists():
            existing = json.loads(self.config_path.read_text(encoding="utf-8"))
            existing.pop("created_at", None)
            if existing != self.sidecar:
                raise PredCacheError(f"{self.config_path} describes a different run for key {self.key}")
            return
        payload = {**self.sidecar, "created_at": datetime.now(UTC).isoformat(timespec="seconds")}
        tmp = self.config_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.config_path)

    def _drop_partial_tail(self) -> None:
        # A crash mid-write leaves a line without "\n"; cut it so the next append starts a clean line.
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            os.truncate(self.path, data.rfind(b"\n") + 1)
