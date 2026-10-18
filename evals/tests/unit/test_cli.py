from __future__ import annotations

import pytest

from veritrade_evals import __main__ as cli
from veritrade_evals import extraction_bench, finalists, freeze, grid, summary_eval
from veritrade_evals.datasets import Example
from veritrade_evals.splits import Split, SplitError, SplitName


@pytest.mark.parametrize(
    ("argv", "module"),
    [
        (["grid", "list"], grid),
        (["finalists", "list"], finalists),
        (["freeze", "--dry-run"], freeze),
        (["extraction", "list"], extraction_bench),
        (["summary", "report", "--split", "dev"], summary_eval),
    ],
)
def test_wired_commands_dispatch_to_their_module(argv: list[str], module: cli.Command) -> None:
    assert cli.build_parser().parse_args(argv).func is module.run
    assert argv[0] not in cli.PLANNED


@pytest.mark.parametrize("command", sorted(cli.PLANNED))
def test_planned_commands_exit_cleanly(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([command, "--model", "x"]) == 2
    assert "not implemented yet" in capsys.readouterr().err


def test_splits_verify_fails_when_any_split_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_load(name: SplitName) -> Split:
        if name == "fpb_allagree":
            raise SplitError("fpb_allagree: 1 bad rows")
        return Split(name, (Example("x:0", "t", "bullish", "tfns", "train"),))

    monkeypatch.setattr(cli, "load_split", fake_load)
    assert cli.main(["splits", "verify"]) == 1
    out = capsys.readouterr()
    assert "ok dev300" in out.out
    assert "FAIL fpb_allagree" in out.err


@pytest.mark.parametrize("argv", [[], ["splits", "verify", "--bogus"]])
def test_bad_invocations_exit_with_usage(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
