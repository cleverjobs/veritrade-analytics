from __future__ import annotations

import pytest
from pydantic import ValidationError

from veritrade.config import Settings
from veritrade.errors import ModelNotAllowed
from veritrade.models_registry import (
    ENDPOINTS,
    ModelSpec,
    Registry,
    allowlist,
    parse_models_toml,
    read_models_toml,
)


def _settings(**overrides: str) -> Settings:
    base: dict[str, str | None] = {
        "models": "",
        "default_model_signal": None,
        "default_model_summary": None,
        "default_model_analyze": None,
    }
    return Settings(**(base | overrides))  # type: ignore[arg-type]


def test_models_toml_parses_and_defaults_are_allowlisted() -> None:
    specs, defaults = parse_models_toml(read_models_toml())
    assert specs == dict(allowlist())
    for endpoint, name in defaults.items():
        assert specs[name].serves(endpoint)
    assert set(defaults) == set(ENDPOINTS)


def test_every_endpoint_has_an_allowed_model_and_a_default() -> None:
    registry = Registry.load(_settings())
    for endpoint in ENDPOINTS:
        assert registry.allowed(endpoint)
        assert registry.resolve(endpoint, None).name == registry.defaults[endpoint]


def test_resolve_explicit_model() -> None:
    registry = Registry.load(_settings())
    assert registry.resolve("signal", "tev1:0.8b-q8_0").kind == "decision"
    assert registry.resolve("analyze", "ministral-3:3b").field_order == "sentiment-first"


@pytest.mark.parametrize(
    ("endpoint", "model", "message"),
    [
        ("signal", "llama3:8b", "not on the allowlist"),
        ("summary", "nimble:9b-q4_K_M", "does not serve /summary"),
        ("signal", "qwen3.5:2b-q4_K_M", "does not serve /signal"),
    ],
)
def test_resolve_rejects(endpoint: str, model: str, message: str) -> None:
    registry = Registry.load(_settings())
    with pytest.raises(ModelNotAllowed, match=message) as info:
        registry.resolve(endpoint, model)  # type: ignore[arg-type]
    assert info.value.http_status == 400
    assert info.value.hint
    assert "Allowed for" in info.value.hint


def test_restriction_filters_allowlist_and_moves_defaults() -> None:
    registry = Registry.load(_settings(models="qwen3.5:0.8b-q8_0, tev1:0.8b-q8_0"))
    assert [s.name for s in registry.allowed("signal")] == ["tev1:0.8b-q8_0", "qwen3.5:0.8b-q8_0"]
    assert registry.defaults["signal"] == "qwen3.5:0.8b-q8_0"
    assert registry.defaults["summary"] == "qwen3.5:0.8b-q8_0"
    assert registry.defaults["analyze"] == "qwen3.5:0.8b-q8_0"
    with pytest.raises(ModelNotAllowed, match="disabled by VT_MODELS"):
        registry.resolve("signal", "nimble:9b-q4_K_M")


def test_restriction_can_leave_an_endpoint_without_models() -> None:
    registry = Registry.load(_settings(models="tev1:0.8b-q8_0"))
    assert "summary" not in registry.defaults
    with pytest.raises(ModelNotAllowed, match="No model is configured"):
        registry.resolve("summary", None)


def test_restriction_with_unknown_model_fails_fast() -> None:
    with pytest.raises(ValueError, match=r"outside models\.toml"):
        Registry.load(_settings(models="tev1:0.8b-q8_0,typo:1b"))


def test_default_overrides() -> None:
    registry = Registry.load(_settings(default_model_signal="qwen3.5:0.8b-q8_0"))
    assert registry.resolve("signal", None).name == "qwen3.5:0.8b-q8_0"
    with pytest.raises(ValueError, match="not allowed for /summary"):
        Registry.load(_settings(default_model_summary="nimble:9b-q4_K_M"))


def test_analyze_models_default_to_sentiment_first() -> None:
    assert allowlist()["qwen3.5:0.8b-q8_0"].field_order == "sentiment-first"
    assert allowlist()["nimble:9b-q4_K_M"].field_order is None


def test_specs_are_frozen() -> None:
    with pytest.raises(ValidationError):
        allowlist()["tev1:0.8b-q8_0"].min_p = 0.9  # type: ignore[misc]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"kind": "decision", "endpoints": ["signal"], "policy": "ordinal"}, "decision models"),
        ({"kind": "decision", "endpoints": ["summary"], "policy": "probability", "min_p": 0.5}, "decision"),
        ({"kind": "generative", "endpoints": ["summary"], "policy": "probability", "min_p": 0.5}, "native"),
        ({"kind": "generative", "endpoints": ["summary"], "policy": "ordinal"}, "num_ctx"),
        (
            {"kind": "generative", "endpoints": ["summary"], "policy": "ordinal", "num_ctx": 8192},
            "num_predict missing",
        ),
        (
            {
                "kind": "generative",
                "endpoints": ["signal"],
                "policy": "ordinal",
                "num_ctx": 8192,
                "num_predict": {"signal": 16},
                "signal_schema": "label",
            },
            "signal_schema = 'ordinal'",
        ),
        (
            {
                "kind": "generative",
                "endpoints": ["signal"],
                "policy": "label-logprob",
                "num_ctx": 8192,
                "num_predict": {"signal": 16},
                "signal_schema": "label",
            },
            "needs min_p",
        ),
        (
            {
                "kind": "generative",
                "endpoints": ["analyze"],
                "policy": "ordinal",
                "num_ctx": 8192,
                "num_predict": {"analyze": 200},
                "directional_tau": 0.3,
            },
            "directional_tau needs class probabilities",
        ),
        (
            {
                "kind": "decision",
                "endpoints": ["signal"],
                "policy": "probability",
                "min_p": 0.5,
                "prompt": {"summary": "v1"},
            },
            "endpoints it does not serve",
        ),
        (
            {
                "kind": "decision",
                "endpoints": ["signal"],
                "policy": "probability",
                "min_p": 0.5,
                "directional_tau": 1,
            },
            "less than 1",
        ),
    ],
)
def test_inconsistent_specs_are_rejected(body: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ModelSpec.model_validate({"name": "m", "max_input_tokens": 100} | body)


def test_frozen_fields_parse() -> None:
    spec = ModelSpec.model_validate(
        {
            "name": "m",
            "max_input_tokens": 100,
            "kind": "decision",
            "endpoints": ["signal"],
            "policy": "probability",
            "min_p": 0.0,
            "directional_tau": 0.28,
            "prompt": {"signal": "v4"},
        }
    )
    assert (spec.min_p, spec.directional_tau, spec.prompt) == (0.0, 0.28, {"signal": "v4"})
