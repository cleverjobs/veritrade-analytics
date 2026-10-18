from __future__ import annotations

import pytest

from veritrade.services.runtime import tag_name
from veritrade.services.system import MIN_OLLAMA_VERSION, parse_version


@pytest.mark.parametrize(
    ("text", "parsed"),
    [
        ("0.40.6", (0, 40, 6)),
        ("v0.40.4", (0, 40, 4)),
        ("0.41.0-rc1", (0, 41, 0)),
        (" 1.2.3 ", (1, 2, 3)),
        ("0.40", None),
        ("unknown", None),
    ],
)
def test_parse_version(text: str, parsed: tuple[int, ...] | None) -> None:
    assert parse_version(text) == parsed


def test_minimum_version_is_compared_numerically() -> None:
    assert (parse_version("0.100.0") or ()) >= MIN_OLLAMA_VERSION
    assert (parse_version("0.9.9") or ()) < MIN_OLLAMA_VERSION


def test_untagged_models_are_listed_as_latest() -> None:
    assert tag_name("llama3") == "llama3:latest"
    assert tag_name("qwen3.5:0.8b-q8_0") == "qwen3.5:0.8b-q8_0"
