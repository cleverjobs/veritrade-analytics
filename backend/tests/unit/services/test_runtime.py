from __future__ import annotations

from typing import Any

import pytest

from veritrade.models_registry import ModelSpec
from veritrade.services.runtime import needs_load

GENERATIVE = ModelSpec(
    name="qwen3.5:0.8b-q8_0",
    kind="generative",
    endpoints=("summary",),
    num_ctx=8192,
    max_input_tokens=100,
    num_predict={"summary": 160},
    policy="ordinal",
)
DECISION = ModelSpec(
    name="clef-flash:9b-q8_0",
    kind="decision",
    endpoints=("signal",),
    max_input_tokens=100,
    policy="probability",
    min_p=0.8,
)


def _entry(name: str, context: int = 8192) -> dict[str, Any]:
    return {"name": name, "model": name, "context_length": context}


@pytest.mark.parametrize(
    ("spec", "entries", "cold"),
    [
        pytest.param(GENERATIVE, [], True, id="nothing-loaded"),
        pytest.param(GENERATIVE, [_entry("qwen3.5:4b-q4_K_M")], True, id="another-model-loaded"),
        pytest.param(GENERATIVE, [_entry("qwen3.5:0.8b-q8_0")], False, id="resident-same-context"),
        pytest.param(GENERATIVE, [_entry("qwen3.5:0.8b-q8_0", 4096)], True, id="resident-other-context"),
        pytest.param(GENERATIVE, [{"model": "qwen3.5:0.8b-q8_0"}], False, id="context-unreported"),
        pytest.param(DECISION, [_entry("clef-flash:9b-q8_0", 16384)], False, id="decision-any-context"),
        pytest.param(DECISION, [], True, id="decision-not-loaded"),
        pytest.param(
            GENERATIVE.model_copy(update={"name": "llama3"}),
            [_entry("llama3:latest")],
            False,
            id="untagged-is-latest",
        ),
    ],
)
def test_needs_load(spec: ModelSpec, entries: list[dict[str, Any]], cold: bool) -> None:
    assert needs_load(spec, entries) is cold
