from __future__ import annotations

import tomllib
from typing import Any

import pytest

from veritrade.models_registry import parse_models_toml, read_models_toml
from veritrade.prompts.registry import PromptRegistry
from veritrade_evals import freeze
from veritrade_evals.freeze import FreezeError, Frozen

TABLE = """# header comment
[defaults]
signal = "a"

[models."a"]
kind = "decision"
policy = "probability"
min_p = 0.5
ordinal_t = 1
evaluated = false  # kept

[models."b"]
kind = "generative"
"""


def test_set_keys_touches_only_the_named_table_and_keys() -> None:
    out = freeze.set_keys(
        TABLE, "a", {"min_p": 0.0, "directional_tau": 0.28, "prompt": {"signal": "v2"}, "evaluated": None}
    )
    assert out.splitlines() == [
        "# header comment",
        "[defaults]",
        'signal = "a"',
        "",
        '[models."a"]',
        'kind = "decision"',
        'policy = "probability"',
        "min_p = 0.0",
        "ordinal_t = 1",
        "directional_tau = 0.28",
        'prompt = { signal = "v2" }',
        "",
        '[models."b"]',
        'kind = "generative"',
    ]
    assert tomllib.loads(out)["models"]["a"]["prompt"] == {"signal": "v2"}
    kept = freeze.set_keys(TABLE, "a", {"field_order": "sentiment-first"})
    assert "evaluated = false  # kept" in kept
    with pytest.raises(FreezeError, match="no table"):
        freeze.set_keys(TABLE, "c", {"min_p": 0.5})


def _frozen() -> dict[str, Frozen]:
    return {
        "qwen3.5:0.8b-q8_0": Frozen(
            "qwen3.5:0.8b-q8_0",
            {"signal_schema": "label", "policy": "label-logprob", "min_p": 0.0, "directional_tau": 0.28},
            {"signal": "v2"},
        ),
        "clef-flash:9b-q8_0": Frozen(
            "clef-flash:9b-q8_0",
            {"signal_schema": None, "policy": "probability", "min_p": 0.8, "directional_tau": None},
            {"signal": "v4"},
        ),
        "qwen3.5:4b-q4_K_M": Frozen(
            "qwen3.5:4b-q4_K_M",
            {"policy": "ordinal", "ordinal_t": 1, "field_order": "sentiment-first"},
            {"analyze": "v5", "summary": "v2"},
        ),
    }


def test_apply_round_trips_through_the_registry_and_keeps_defaults() -> None:
    text = read_models_toml()
    out = freeze.apply(text, _frozen())
    specs, defaults = parse_models_toml(out)
    assert defaults == parse_models_toml(text)[1]
    qwen = specs["qwen3.5:0.8b-q8_0"]
    assert (qwen.min_p, qwen.directional_tau, qwen.prompt) == (0.0, 0.28, {"signal": "v2"})
    assert specs["qwen3.5:4b-q4_K_M"].prompt == {"analyze": "v5", "summary": "v2"}
    assert specs["clef-flash:9b-q8_0"].min_p == 0.8
    assert freeze.apply(out, _frozen()) == out
    untouched = ("tev1:0.8b-q8_0", "qwen3.5:9b-q4_K_M")
    assert all(specs[m] == parse_models_toml(text)[0][m] for m in untouched)


def _best(policy: str, threshold: float, variant: str = "label") -> dict[str, Any]:
    prompt = PromptRegistry().get("signal", "qwen3.5:0.8b-q8_0", version="v2").version
    cis = {"f05_macro": {"point": 0.7, "low": 0.6, "high": 0.8}}
    return {
        "best": {
            "config": "v2-label",
            "variant": variant,
            "policy": policy,
            "threshold": threshold,
            "prompt_version": prompt,
            "cis": cis,
        }
    }


def _article_row(scope: str, policy: str, failure: list[str], repick: Any = None) -> dict[str, Any]:
    row = {
        "model": "qwen3.5:0.8b-q8_0",
        "config": "v2-label",
        "scope": scope,
        "policy": policy,
        "threshold": 0.28,
        "articles": {"cis": {"f05_macro": {}}, "inversions": 0, "directional_calls": 5, "coverage": 1.0},
        "clear_failure": failure,
    }
    return row | ({"repick": repick} if failure else {})


@pytest.mark.parametrize(
    ("policy", "threshold", "expected"),
    [
        ("label-logprob+tau", 0.28, {"policy": "label-logprob", "min_p": 0.0, "directional_tau": 0.28}),
        ("label-logprob", 0.0, {"policy": "label-logprob", "min_p": 0.0, "directional_tau": None}),
        ("label-logprob", 0.62, {"policy": "label-logprob", "min_p": 0.62, "directional_tau": None}),
    ],
)
def test_signal_freeze_maps_the_dev300_rule(
    policy: str, threshold: float, expected: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(freeze, "grid_report", lambda endpoint, model: _best(policy, threshold))
    articles = {"rows": [_article_row("lede", policy, [])], "comparisons": []}
    frozen = freeze.freeze_signal("qwen3.5:0.8b-q8_0", articles, PromptRegistry())
    assert {k: frozen.values[k] for k in expected} == expected
    assert frozen.values["signal_schema"] == "label"
    assert frozen.prompt == {"signal": "v2"}


def test_signal_freeze_re_picks_only_on_a_clear_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(freeze, "grid_report", lambda endpoint, model: _best("label-logprob+tau", 0.28))
    failing = _article_row(
        "lede", "label-logprob+tau", ["inversion CI low 0.05 > cap 0.02"], {"threshold": 0.4}
    )
    frozen = freeze.freeze_signal(
        "qwen3.5:0.8b-q8_0", {"rows": [failing], "comparisons": []}, PromptRegistry()
    )
    assert frozen.values["directional_tau"] == 0.4
    assert frozen.evidence["signal"]["repicked_on_articles"]["threshold"] == 0.4
    hopeless = _article_row("lede", "label-logprob+tau", ["coverage collapse"], None)
    with pytest.raises(FreezeError, match="no re-pick"):
        freeze.freeze_signal("qwen3.5:0.8b-q8_0", {"rows": [hopeless], "comparisons": []}, PromptRegistry())
    # The full-body scope is reported, but /signal serves the lede.
    full = _article_row("full", "label-logprob+tau", ["inversion CI low 0.05 > cap 0.02"], None)
    frozen = freeze.freeze_signal("qwen3.5:0.8b-q8_0", {"rows": [full], "comparisons": []}, PromptRegistry())
    assert frozen.values["directional_tau"] == 0.28


def _analyze_report(v5_verdict: str, order_verdict: str | None = None) -> dict[str, Any]:
    registry = PromptRegistry()
    gap = {"point": 0.02, "low": -0.01, "high": 0.05}

    def follow_up(config: str, version: str, verdict: str) -> dict[str, Any]:
        prompt = registry.get("analyze", "ministral-3:3b", version=version).version
        boot = {"follow_up": gap, "against": gap, "gap": gap}
        return {
            "config": config,
            "threshold": 1.0,
            "prompt_version": prompt,
            "verdict": verdict,
            "paired_bootstrap": boot,
        }

    rows = [follow_up("v5", "v5", v5_verdict)]
    if order_verdict is not None:
        rows.append(follow_up("v1-summary-first", "v1", order_verdict))
    return {"best": follow_up("v1", "v1", "tie") | {"threshold": 2.0}, "follow_ups": rows}


def _comparison(what: str, a: str, verdict: str) -> dict[str, Any]:
    return {"comparison": what, "a": a, "verdict": verdict, "gap": {"point": 0.0, "low": -0.1, "high": 0.1}}


@pytest.mark.parametrize(
    ("v5", "articles_v5", "order", "articles_order", "expected"),
    [
        ("tie", "tie", "tie", "tie", ("v5", 1, "sentiment-first")),
        ("follow-up", "a", "follow-up", "tie", ("v5", 1, "summary-first")),
        ("tie", "tie", "follow-up", "b", ("v5", 1, "sentiment-first")),
        ("against", "tie", "tie", "tie", ("v1", 2, "sentiment-first")),
        ("tie", "b", "tie", "tie", ("v1", 2, "sentiment-first")),
    ],
)
def test_analyze_freeze_gates_the_revision_and_the_field_order(
    v5: str,
    articles_v5: str,
    order: str,
    articles_order: str,
    expected: tuple[str, int, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(freeze, "grid_report", lambda endpoint, model: _analyze_report(v5, order))
    articles = {
        "rows": [],
        "comparisons": [
            _comparison("prompt: v5 - v1", "ministral-3:3b/v5/full (ordinal)", articles_v5),
            _comparison(
                "field order: summary-first - sentiment-first",
                "ministral-3:3b/v1-summary-first/full (ordinal)",
                articles_order,
            ),
        ],
    }
    frozen = freeze.freeze_analyze("ministral-3:3b", articles, PromptRegistry())
    version, t, order = expected
    assert frozen.values == {"policy": "ordinal", "ordinal_t": t, "field_order": order}
    assert frozen.prompt == {"analyze": version}
    assert (frozen.evidence["analyze"]["revision_rejected"] is None) == (version == "v5")


def test_a_stale_prompt_hash_refuses_to_freeze() -> None:
    with pytest.raises(FreezeError, match="changed since it was scored"):
        freeze.version_id(
            "signal/qwen3.5:0.8b-q8_0/v2@00000000", "signal", "qwen3.5:0.8b-q8_0", PromptRegistry()
        )
