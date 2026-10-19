from __future__ import annotations

import pytest

from veritrade.api.contracts import SignalRule
from veritrade.inference.types import Endpoint
from veritrade.models_registry import ModelSpec
from veritrade.services.runtime import tag_name
from veritrade.services.system import MIN_OLLAMA_VERSION, parse_version, signal_rules

LABEL = ModelSpec(
    name="qwen3.5:0.8b-q8_0",
    kind="generative",
    endpoints=("signal", "summary", "analyze"),
    num_ctx=8192,
    max_input_tokens=100,
    num_predict={"signal": 16, "summary": 160, "analyze": 200},
    signal_schema="label",
    policy="label-logprob",
    min_p=0.0,
    directional_tau=0.28,
)
ORDINAL = ModelSpec(
    name="qwen3.5:4b-q4_K_M",
    kind="generative",
    endpoints=("summary", "analyze"),
    num_ctx=8192,
    max_input_tokens=100,
    num_predict={"summary": 160, "analyze": 200},
    policy="ordinal",
    ordinal_t=2,
)
DECISION = ModelSpec(
    name="clef-flash:9b-q8_0",
    kind="decision",
    endpoints=("signal",),
    max_input_tokens=100,
    policy="probability",
    min_p=0.8,
)
SUMMARY_ONLY = ORDINAL.model_copy(update={"endpoints": ("summary",)})


@pytest.mark.parametrize(
    ("spec", "rules"),
    [
        pytest.param(
            LABEL,
            {
                "signal": SignalRule(policy="label-logprob", threshold=0.0, directional_tau=0.28),
                "analyze": SignalRule(policy="ordinal", threshold=1.0),
            },
            id="label-logprob-signal-ordinal-analyze",
        ),
        pytest.param(ORDINAL, {"analyze": SignalRule(policy="ordinal", threshold=2.0)}, id="ordinal"),
        pytest.param(DECISION, {"signal": SignalRule(policy="probability", threshold=0.8)}, id="decision"),
        pytest.param(SUMMARY_ONLY, {}, id="no-signal-endpoint"),
    ],
)
def test_signal_rules_follow_the_policy_each_endpoint_uses(
    spec: ModelSpec, rules: dict[Endpoint, SignalRule]
) -> None:
    assert signal_rules(spec) == rules


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
