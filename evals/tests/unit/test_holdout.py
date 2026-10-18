from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from veritrade.inference.types import Sentiment
from veritrade.prompts.registry import PromptRegistry
from veritrade_evals import holdout as ho
from veritrade_evals.datasets import Example
from veritrade_evals.holdout import Gold, HCell, Rule


@pytest.fixture(scope="module")
def frozen() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(ho.FROZEN_JSON.read_text(encoding="utf-8"))
    return payload


@pytest.fixture(scope="module")
def rules(frozen: dict[str, Any]) -> list[Rule]:
    return ho.rules_from_frozen(frozen, PromptRegistry())


# --- frozen rules and the plan ---------------------------------------------------------------


def test_rules_reuse_the_frozen_thresholds_and_prompts(frozen: dict[str, Any], rules: Sequence[Rule]) -> None:
    for rule in rules:
        entry = frozen["models"][rule.model][rule.endpoint]
        assert rule.run.prompt.version == entry["prompt_version"]
        if rule.endpoint == "signal":
            assert rule.threshold == entry["evidence"]["dev300"]["threshold"]
            assert rule.policy == entry["evidence"]["dev300"]["policy"]
        else:
            assert (rule.policy, rule.threshold) == ("ordinal", float(entry["ordinal_t"]))
        assert rule.dev["dev300"] > 0


def test_plan_runs_finalists_everywhere_and_winners_plus_the_ceiling_cross_domain(
    rules: Sequence[Rule],
) -> None:
    ceiling = ho.ceiling_rule(PromptRegistry())
    signal_winner = next(r.model for r in rules if r.endpoint == "signal")
    analyze_winner = next(r.model for r in rules if r.endpoint == "analyze")
    cells = ho.plan_cells(rules, ceiling, {"signal": signal_winner, "analyze": analyze_winner})
    keys = {c.key for c in cells}
    for rule in rules:
        assert any(k.startswith(f"tfns_val/{rule.endpoint}/{rule.model}/") for k in keys)
        cross = {c.split for c in cells if c.rule is rule and c.split in ho.CROSS}
        assert cross == (set(ho.CROSS) if rule.model in (signal_winner, analyze_winner) else set())
    assert {c.split for c in cells if c.rule is ceiling} == {ho.ARTICLES, *ho.CROSS}
    signal_scopes = {c.scope for c in cells if c.split == ho.ARTICLES and c.rule.endpoint == "signal"}
    assert signal_scopes == {"lede", "full"}
    assert "dev300" not in {c.split for c in cells}
    # Grouped by model so each model loads once.
    blocks = [model for model, _ in itertools.groupby(c.model for c in cells)]
    assert len(blocks) == len(set(blocks))


def test_balanced_head_keeps_every_class() -> None:
    labels: list[Sentiment] = ["bearish"] * 5 + ["bullish"] * 5 + ["neutral"] * 5
    items = [Example(f"x:{i}", "t", label, "tfns", "train") for i, label in enumerate(labels)]
    head = ho.balanced_head(items, 6)
    assert [e.label for e in head] == ["bearish", "bearish", "bullish", "bullish", "neutral", "neutral"]


def test_score_cell_applies_the_frozen_threshold_without_retuning(rules: Sequence[Rule]) -> None:
    rule = next(r for r in rules if r.run.variant == "decision")
    gold = Gold(("a", "b", "c", "d"), ("bullish", "bearish", "neutral", "bullish"))

    def record(probs: dict[str, float]) -> dict[str, Any]:
        return {"kind": "decision", "probs": probs, "error": None, "t": {"wall_ms": 5.0}}

    sure = rule.threshold + (1 - rule.threshold) / 2
    records = {
        "a": record({"bullish": sure, "bearish": 0.0, "neutral": 1 - sure}),
        "b": record({"bullish": 0.0, "bearish": sure, "neutral": 1 - sure}),
        "c": record({"bullish": 0.0, "bearish": 0.0, "neutral": 1.0}),
        # Below the frozen min_p: abstains instead of calling bullish.
        "d": record({"bullish": rule.threshold - 0.1, "bearish": 0.0, "neutral": 1.1 - rule.threshold}),
    }
    row = ho.score_cell(HCell(rule, "tfns_val"), records, gold, n_boot=50)
    assert row["metrics"]["coverage"] == 0.75
    assert row["metrics"]["inversion_rate"] == 0.0
    assert row["dataset"] == "twitter-fns"
    assert row["gap"] == pytest.approx(row["metrics"]["f05_macro"] - row["dev"]["f05Macro"])


# --- the holdout-once guard ------------------------------------------------------------------


def test_guard_refuses_a_second_run_resumes_an_interrupted_one_and_logs_force(
    capsys: pytest.CaptureFixture[str],
) -> None:
    guard: dict[str, Any] = {"runs": {}}
    assert ho.guard_mode(guard, "sha", force=False) == "fresh"
    ho.guard_start(guard, "sha", mode="fresh", selection_sha="sel", cells=3, reason=None)
    assert ho.guard_mode(guard, "sha", force=False) == "resume"
    ho.guard_complete(guard, "sha")
    with pytest.raises(ho.HoldoutError, match="runs once per frozen config"):
        ho.guard_mode(guard, "sha", force=False)
    assert ho.guard_mode(guard, "other-sha", force=False) == "fresh"
    assert ho.guard_mode(guard, "sha", force=True) == "forced"
    ho.guard_start(guard, "sha", mode="forced", selection_sha="sel", cells=3, reason="threshold bug fixed")
    entry = guard["runs"]["sha"]
    assert entry["status"] == "started"
    assert entry["forced"][0]["reason"] == "threshold bug fixed"
    assert "HOLDOUT FORCED" in capsys.readouterr().err


def _args(**overrides: Any) -> argparse.Namespace:
    base = {
        "dry_run": False,
        "out_dir": None,
        "limit": 0,
        "model": None,
        "force": False,
        "reason": None,
        "allow_provisional": False,
        "skip_summaries": False,
        "concurrency": 1,
        "shared": True,
        "base_url": None,
        "n_boot": 50,
    }
    return argparse.Namespace(**(base | overrides))


def test_dry_run_needs_a_scratch_folder_and_model_filters_only_apply_to_it() -> None:
    with pytest.raises(ho.HoldoutError, match="--out-dir"):
        ho.cmd_run(_args(dry_run=True))
    with pytest.raises(ho.HoldoutError, match="--model only narrows"):
        ho.cmd_run(_args(model=["qwen3.5:0.8b-q8_0"]))


def test_dry_run_never_reads_the_guard_or_a_holdout_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("the dry run touched the guard")

    seen: list[str] = []

    async def fake_run(
        args: argparse.Namespace, s: ho.Setup, inputs: Any, preds: Path, manifest: Path
    ) -> None:
        seen.extend(c.split for c in s.cells)
        assert preds.parent == tmp_path
        assert inputs.limit == ho.DRY_LIMIT

    monkeypatch.setattr(ho, "read_guard", forbidden)
    monkeypatch.setattr(ho, "guard_start", forbidden)
    monkeypatch.setattr(ho, "_run", fake_run)
    monkeypatch.setattr(ho, "analyze_cells", lambda *a, **k: ([], []))
    assert ho.cmd_run(_args(dry_run=True, out_dir=tmp_path, model=["ministral-3:3b"])) == 0
    assert seen == [ho.DRY_SPLIT]
    report = json.loads((tmp_path / "holdout.json").read_text(encoding="utf-8"))
    assert report["dryRun"]
    assert report["guard"] is None
    assert "(dry run)" in (tmp_path / "holdout.md").read_text(encoding="utf-8")


def test_real_run_refuses_without_a_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ho, "SELECTION_JSON", tmp_path / "missing.json")
    with pytest.raises(ho.HoldoutError, match="run `select` first"):
        ho.cmd_run(_args())


def test_estimate_uses_dev_timings_and_doubles_qwen35_4b_for_the_ceiling(rules: Sequence[Rule]) -> None:
    ceiling = ho.ceiling_rule(PromptRegistry())
    qwen = next(r for r in rules if r.model == "qwen3.5:4b-q4_K_M" and r.endpoint == "analyze")
    cells = [HCell(qwen, "tfns_val"), HCell(ceiling, "fpb_allagree")]
    est = ho.estimate(cells, {"tfns_val": 100, "fpb_allagree": 100}, summaries=0)
    per_call = {r["key"]: r["perCallMs"] for r in est["cells"]}
    assert per_call[cells[1].key] == 2 * per_call[cells[0].key]
