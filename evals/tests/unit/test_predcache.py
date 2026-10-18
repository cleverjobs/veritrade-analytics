from __future__ import annotations

import json
from pathlib import Path

import pytest

from veritrade_evals.predcache import PredCache, PredCacheError, PredError, model_slug, run_hash

CONFIG = {"endpoint": "signal", "scope": "lede", "prompt_version": "v1", "k": 4}
DIGEST = "sha256:" + "d" * 64
PROMPT = "p" * 64


def _cache(root: Path, **overrides: str) -> PredCache:
    return PredCache(
        "qwen3.5:0.8b-q8_0",
        CONFIG,
        model_digest=overrides.get("digest", DIGEST),
        prompt_sha=overrides.get("prompt", PROMPT),
        root=root,
    )


def test_run_hash_is_canonical_and_pinned() -> None:
    shuffled = dict(reversed(list(CONFIG.items())))
    assert run_hash(CONFIG, DIGEST, PROMPT) == run_hash(shuffled, DIGEST, PROMPT)
    # Pinned: a change here silently orphans every existing prediction cache.
    assert run_hash(CONFIG, DIGEST, PROMPT) == "d7a4f121234e1792"
    variants = {
        run_hash(CONFIG, DIGEST, PROMPT),
        run_hash({**CONFIG, "k": 0}, DIGEST, PROMPT),
        run_hash(CONFIG, "sha256:" + "e" * 64, PROMPT),
        run_hash(CONFIG, DIGEST, "q" * 64),
        run_hash(CONFIG, DIGEST, PROMPT, harness_version="2"),
    }
    assert len(variants) == 5


def test_paths_follow_model_slug_and_hash(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    assert model_slug("qwen3.5:0.8b-q8_0") == "qwen3.5_0.8b-q8_0"
    assert cache.path == tmp_path / "qwen3.5_0.8b-q8_0" / f"{cache.key}.jsonl"
    assert cache.config_path.name == f"{cache.key}.config.json"
    assert len(cache.key) == 16


def test_resume_skips_done_and_retries_retryable_errors(tmp_path: Path) -> None:
    with _cache(tmp_path) as cache:
        cache.append("a", "bullish", pred={"label": "bullish", "confidence": 0.9}, t={"wall_ms": 12.5})
        cache.append("b", "bearish", error=PredError("llm_unreachable", "connection refused", retryable=True))
        cache.append("c", "neutral", error=PredError("schema_violation", "bad json", retryable=False))

    with _cache(tmp_path) as resumed:
        assert resumed.done_ids() == {"a", "c"}
        resumed.append("b", "bearish", pred={"label": "neutral", "confidence": 0.4})
        assert resumed.done_ids() == {"a", "b", "c"}
        assert resumed.records()["b"]["label"] == "neutral"


def test_terminal_record_wins_over_a_later_retryable_failure(tmp_path: Path) -> None:
    with _cache(tmp_path) as cache:
        cache.append("a", "bullish", pred={"label": "bullish"})
        cache.append("a", "bullish", error=PredError("llm_unreachable", "down", retryable=True))
        assert cache.records()["a"]["label"] == "bullish"
        assert cache.done_ids() == {"a"}


def test_record_format_has_no_input_text(tmp_path: Path) -> None:
    with _cache(tmp_path) as cache:
        cache.append(
            "a", "bullish", pred={"label": "bullish", "score": 2}, t={"wall_ms": 3.0, "ttft_ms": 1.0}
        )
    (line,) = cache.path.read_text(encoding="utf-8").splitlines()
    assert json.loads(line) == {
        "id": "a",
        "gold": "bullish",
        "label": "bullish",
        "score": 2,
        "error": None,
        "t": {"wall_ms": 3.0, "ttft_ms": 1.0},
    }


@pytest.mark.parametrize("key", ["text", "prompt", "messages", "id", "error"])
def test_append_rejects_input_text_and_reserved_fields(tmp_path: Path, key: str) -> None:
    with _cache(tmp_path) as cache, pytest.raises(ValueError, match=key):
        cache.append("a", "bullish", pred={key: "x"})


def test_partial_last_line_is_cut_on_open(tmp_path: Path) -> None:
    with _cache(tmp_path) as cache:
        cache.append("a", "bullish", pred={"label": "bullish"})
    with cache.path.open("a", encoding="utf-8") as fh:
        fh.write('{"id": "b", "gold": "bea')
    assert set(_cache(tmp_path).records()) == {"a"}
    with _cache(tmp_path) as resumed:
        resumed.append("b", "bearish", pred={"label": "bearish"})
    lines = cache.path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["a", "b"]


def test_second_writer_is_refused(tmp_path: Path) -> None:
    with _cache(tmp_path), pytest.raises(PredCacheError, match="already open"):
        _cache(tmp_path).open()


def test_sidecar_describes_the_run_and_must_match(tmp_path: Path) -> None:
    with _cache(tmp_path) as cache:
        pass
    sidecar = json.loads(cache.config_path.read_text(encoding="utf-8"))
    assert sidecar["model"] == "qwen3.5:0.8b-q8_0"
    assert sidecar["run_config"] == CONFIG
    assert "created_at" in sidecar
    sidecar["prompt_sha"] = "tampered"
    cache.config_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(PredCacheError, match="different run"):
        _cache(tmp_path).open()


def test_append_requires_open(tmp_path: Path) -> None:
    with pytest.raises(PredCacheError, match="not open"):
        _cache(tmp_path).append("a", "bullish")
