"""Tests for scripts/export_transcript.py on synthetic session fixtures."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_PATH = Path(__file__).resolve().parents[1] / "export_transcript.py"
_SPEC = importlib.util.spec_from_file_location("export_transcript", _PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
et = importlib.util.module_from_spec(_SPEC)
sys.modules["export_transcript"] = et
_SPEC.loader.exec_module(et)

# Leak-shaped values are concatenated so this file stays legible in the redacted transcript it ends up in.
EMAIL = "someone" + "@example.com"
HOME = "/Users" + "/someone"
OPEN, CLOSE = f"<{et.SR}>", f"</{et.SR}>"
SID = "11111111-2222-3333-4444-555555555555"


def rec(kind: str, uuid: str, parent: str | None = None, **extra: Any) -> dict[str, Any]:
    base = {"type": kind, "uuid": uuid, "parentUuid": parent, "isSidechain": False, "userType": "external"}
    return (
        base
        | {"cwd": f"{HOME}/repo", "sessionId": SID, "version": "9.9.9", "timestamp": "2026-01-01T00:00:00Z"}
        | extra
    )


def attachment(uuid: str, parent: str | None, kind: str, **payload: Any) -> dict[str, Any]:
    return rec(
        "attachment", uuid, parent, attachment={"type": kind, **payload}, rendered=[f"{OPEN}\nx\n{CLOSE}"]
    )


def assistant(
    uuid: str, parent: str | None, mid: str, blocks: list[dict[str, Any]], **usage: int
) -> dict[str, Any]:
    msg = {"id": mid, "type": "message", "role": "assistant", "model": "claude-test", "content": blocks}
    return rec("assistant", uuid, parent, message=msg | {"usage": usage}, effort="high")


def user(uuid: str, parent: str | None, content: Any, **extra: Any) -> dict[str, Any]:
    return rec("user", uuid, parent, message={"role": "user", "content": content}, **extra)


def tool_use(tid: str, name: str, **inp: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def tool_result(tid: str, content: Any) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tid, "content": content}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def stage(
    tmp_path: Path, records: list[dict[str, Any]], denylist: list[str] | None = None
) -> tuple[Any, Any, Path]:
    project = tmp_path / "proj"
    write_jsonl(project / f"{SID}.jsonl", records)
    san, harvest = et.Sanitizer(denylist=denylist or []), et.Harvest()
    et.stage_session(project, SID, tmp_path / "staged", san, harvest)
    return san, harvest, tmp_path / "staged" / f"{SID}.jsonl"


def test_allowlist_keeps_conversation_and_drops_plumbing(tmp_path: Path) -> None:
    records = [
        user("u1", None, "hello"),
        attachment("a1", "u1", "instructions", content="private global rules"),
        attachment("a2", "a1", et.SESSION_CTX, content=f"mail {EMAIL}"),
        attachment("a3", "a2", "plan_mode", planFilePath=f"{HOME}/.claude/plans/p.md"),
        attachment("a4", "a3", "hook_success", content="hook out"),
        rec("system", "s1", "a4", subtype="turn_duration", durationMs=5),
        rec("system", "s2", "s1", subtype="stop_hook_summary"),
        {"type": et.BLOB, "message": {"content": "request body"}},
        {
            "type": "api-request",
            "params": {"thinking": {"type": "adaptive", "display": "updates", "budget": 1}},
        },
        {"type": "file-history-snapshot", "snapshot": {}},
        {"type": "permission-mode", "permissionMode": "plan", "sessionId": SID},
        {"type": "custom-title", "customTitle": "t", "sessionId": SID},
        assistant("x1", "s2", "m1", [{"type": "text", "text": "hi"}], output_tokens=1),
    ]
    _, harvest, out = stage(tmp_path, records)
    kept = read_jsonl(out)
    kinds = [et.record_key(r) for r in kept]
    assert kinds == [
        "user",
        "attachment:plan_mode",
        "system:turn_duration",
        "permission-mode",
        "custom-title",
        "assistant",
    ]
    assert all("rendered" not in r for r in kept)
    assert harvest.dropped["attachment:instructions"] == 1
    assert harvest.dropped[et.BLOB] == 1
    assert harvest.thinking_params == {(("type", "adaptive"), ("display", "updates")): 1}
    assert kept[1]["attachment"]["planFilePath"] == "~/.claude/plans/p.md"
    # u1 <- a3 and s1 <- x1 chains skip the dropped records in between.
    assert kept[1]["parentUuid"] == "u1"
    assert kept[-1]["parentUuid"] == "s1"


def test_reminders_stripped_from_nested_tool_result_content(tmp_path: Path) -> None:
    nested = [
        {"type": "text", "text": f"file body\n{OPEN}\nsecret rule\n{CLOSE}\ntrailer"},
        {"type": "text", "text": f'escaped json: "{OPEN}\\nsecret rule\\n{CLOSE}" end'},
        {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
    ]
    records = [
        assistant("x1", None, "m1", [tool_use("t1", "Bash", command="cat notes.txt")]),
        user("u1", "x1", [tool_result("t1", nested)]),
        user("u2", "u1", f"we strip {OPEN} tags in prose"),
    ]
    san, _, out = stage(tmp_path, records)
    result, prose = read_jsonl(out)[1:]
    texts = [i["text"] for i in result["message"]["content"][0]["content"]]
    assert texts == ["file body\n\ntrailer", 'escaped json: "" end', "[omitted: image]"]
    assert san.counts["system_reminder"] == 2
    assert et.SR not in json.dumps(prose)
    assert "[system reminder] tags" in prose["message"]["content"]


def test_thinking_signature_and_redacted_thinking(tmp_path: Path) -> None:
    blocks = [
        {"type": "thinking", "thinking": "", "signature": "Eq" + "x" * 80},
        {"type": "redacted_thinking", "data": "opaque"},
    ]
    _, _, out = stage(tmp_path, [assistant("x1", None, "m1", blocks)])
    content = read_jsonl(out)[0]["message"]["content"]
    assert content == [
        {"type": "thinking", "thinking": ""},
        {"type": "text", "text": "[omitted: redacted_thinking]"},
    ]


def test_redaction_rules_and_counts(tmp_path: Path) -> None:
    secrets = [
        "sk-" + "a1B2" * 6,
        "ghp_" + "Z" * 36,
        "AKIA" + "ABCDEFGHIJKLMNOP",
        "eyJ" + "a" * 10 + ".eyJ" + "b" * 10 + "." + "c" * 10,
        "-----BEGIN " + "OPENSSH PRIVATE KEY-----\nabc\n-----END OPENSSH PRIVATE KEY-----",
    ]
    text = f"{EMAIL} kris.dimitrov92@gmail.com {HOME}/repo/x.py Acme Corp " + " ".join(secrets)
    records = [
        user("u1", None, text),
        user("u2", "u1", "ok", toolUseResult={"answers": {f"ask {EMAIL}?": "yes"}}),
    ]
    san, _, out = stage(tmp_path, records, denylist=["acme corp"])
    first, second = read_jsonl(out)
    body = first["message"]["content"]
    assert body.startswith("[redacted-email] kris.dimitrov92@gmail.com ~/repo/x.py [redacted] ")
    assert body.count("[redacted-secret]") == len(secrets)
    assert first["cwd"] == "~/repo"
    assert second["toolUseResult"] == {"answers": {"ask [redacted-email]?": "yes"}}
    assert san.counts["email"] == 2
    assert san.counts["denylist"] == 1
    assert san.counts["home_path"] == 3
    assert sum(v for k, v in san.counts.items() if k.startswith("secret:")) == len(secrets)


def test_usage_summed_once_per_message_id(tmp_path: Path) -> None:
    text = [{"type": "text", "text": "a"}]
    records = [
        assistant("x1", None, "m1", text, input_tokens=10, cache_read_input_tokens=100, output_tokens=5),
        assistant("x2", "x1", "m1", text, input_tokens=10, cache_read_input_tokens=100, output_tokens=7),
        assistant("x3", "x2", "m2", text, input_tokens=1, cache_creation_input_tokens=50, output_tokens=2),
    ]
    _, harvest, _ = stage(tmp_path, records)
    summary = harvest.usage_summary()
    assert summary["total"] == {
        "input": 11,
        "cache_creation": 50,
        "cache_read": 100,
        "output": 9,
        "thinking": 0,
    }
    assert harvest.models["main"] == {"claude-test": 2}


def test_publisher_and_private_config_content_omitted(tmp_path: Path) -> None:
    page = "<!DOCTYPE html><html><body>" + "story " * 600 + "</body></html>"
    records = [
        assistant(
            "x1",
            None,
            "m1",
            [
                tool_use("t1", "Read", file_path=f"{et.PRIVATE_DATA}/site/a.html"),
                tool_use("t2", "Write", file_path=f"{et.PRIVATE_DATA}/b.txt", content="article text " * 40),
                tool_use("t3", "Bash", command="cat fixture"),
                tool_use("t4", "WebFetch", url="https://news.example/a", prompt="summarize"),
                tool_use("t5", "WebFetch", url="https://news.example/b", prompt="summarize"),
                tool_use("t6", "Read", file_path=f"{HOME}/.ssh/config"),
                tool_use("t7", "Write", file_path="backend/x.py", content=f'PRIVATE = "{et.PRIVATE_DATA}"'),
                tool_use("t8", "Write", file_path="backend/tests/test_page.py", content=f"PAGE = {page!r}"),
                tool_use("t9", "Write", file_path="backend/tests/fixtures/page.html", content=page),
                tool_use("t10", "Read", file_path="backend/tests/test_page.py"),
                tool_use("t11", "Bash", command="cd frontend && cat src/app/layout.tsx src/app/page.tsx"),
            ],
        ),
        user(
            "u1",
            "x1",
            [
                tool_result("t1", "publisher paragraph"),
                tool_result("t2", "ok"),
                tool_result("t3", page),
                tool_result("t4", "w" * (et.WEB_MIN_CHARS + 1)),
                tool_result("t5", "short summary"),
                tool_result("t6", "Host box\n  HostName 10.0.0.1"),
                tool_result("t7", "ok"),
                tool_result("t10", f"1\tPAGE = {page!r}"),
                tool_result("t11", page),
            ],
        ),
    ]
    san, _, out = stage(tmp_path, records)
    uses, results = (r["message"]["content"] for r in read_jsonl(out))
    contents = [r["content"] for r in results]
    assert contents[0].startswith("[omitted: publisher/web content, sha256=")
    assert contents[0].endswith(f", {len('publisher paragraph')} chars]")
    assert contents[2].startswith("[omitted: publisher/web content")
    assert contents[3].startswith("[omitted: publisher/web content")
    assert contents[4] == "short summary"
    assert contents[5].startswith("[omitted: private local config, sha256=")
    assert contents[6] == "ok"
    assert contents[7] == f"1\tPAGE = {page!r}"
    assert contents[8] == page
    assert uses[1]["input"]["content"].startswith("[omitted: publisher/web content")
    assert uses[1]["input"]["file_path"] == f"{et.PRIVATE_DATA}/b.txt"
    assert uses[6]["input"]["content"] == f'PRIVATE = "{et.PRIVATE_DATA}"'
    assert uses[7]["input"]["content"] == f"PAGE = {page!r}"
    assert uses[8]["input"]["content"].startswith("[omitted: publisher/web content")
    assert san.counts["omitted:private_data"] == 3
    assert san.counts["omitted:html"] == 2
    assert san.counts["omitted:web_fetch"] == 1
    assert san.counts["omitted:private_config"] == 1


def test_private_dir_content_omitted_but_mentions_kept(tmp_path: Path) -> None:
    doc = f"Raw pages are cached under {et.PRIVATE_DATA}/ and never committed.\n" * 40
    records = [
        assistant(
            "x1",
            None,
            "m1",
            [
                tool_use("t1", "Bash", command="grep -rn shares evals"),
                tool_use("t2", "Bash", command="cat plan.md"),
            ],
        ),
        user(
            "u1",
            "x1",
            [
                tool_result("t1", f"{et.PRIVATE_DATA}/site/a.html:12: shares rose 3%"),
                tool_result("t2", doc),
            ],
        ),
    ]
    san, _, out = stage(tmp_path, records)
    first, second = (r["content"] for r in read_jsonl(out)[1]["message"]["content"])
    assert first.startswith("[omitted: publisher/web content")
    assert second == doc
    assert san.counts["omitted:private_mention"] == 1


def test_skill_body_replaced_on_meta_user_records(tmp_path: Path) -> None:
    records = [
        user(
            "u1",
            None,
            [{"type": "text", "text": "Base directory for this skill\n" + "rule " * 200}],
            isMeta=True,
        )
    ]
    san, _, out = stage(tmp_path, records)
    assert read_jsonl(out)[0]["message"]["content"][0]["text"].startswith(
        "[omitted: skill/command instructions"
    )
    assert san.counts["omitted:skill_body"] == 1


def test_session_layout_collected_and_others_excluded(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    session = project / SID
    write_jsonl(project / "other-session.jsonl", [user("o1", None, "other project")])
    write_jsonl(session / "subagents" / "agent-a1.jsonl", [user("s1", None, "sub")])
    (session / "subagents" / "agent-a1.meta.json").write_text(
        json.dumps({"agentType": "Explore", "model": "opus"})
    )
    run = session / "subagents" / "workflows" / "wf_abc"
    write_jsonl(run / "journal.jsonl", [{"type": "started", "agentId": "b1", "label": "probe", "phase": "P"}])
    write_jsonl(run / "agent-b1.jsonl", [user("w1", None, "wf agent")])
    (run / "agent-b1.meta.json").write_text(json.dumps({"agentType": "workflow-subagent", "model": "opus"}))
    (session / "workflows" / "scripts").mkdir(parents=True)
    (session / "workflows" / "scripts" / "build-it-wf_abc.js").write_text(f"const repo = '{HOME}/repo'\n")
    (session / "tool-results").mkdir()
    (session / "tool-results" / "big.txt").write_text("fetched page")
    _, harvest, _ = stage(tmp_path, [user("u1", None, "main")])
    staged = sorted(
        str(p.relative_to(tmp_path / "staged")) for p in (tmp_path / "staged").rglob("*") if p.is_file()
    )
    assert staged == [
        f"{SID}.jsonl",
        f"{SID}/subagents/agent-a1.jsonl",
        f"{SID}/subagents/agent-a1.meta.json",
        f"{SID}/subagents/workflows/wf_abc/agent-b1.jsonl",
        f"{SID}/subagents/workflows/wf_abc/agent-b1.meta.json",
        f"{SID}/subagents/workflows/wf_abc/journal.jsonl",
        f"{SID}/workflows/scripts/build-it-wf_abc.js",
    ]
    assert (tmp_path / "staged" / SID / "workflows" / "scripts" / "build-it-wf_abc.js").read_text() == (
        "const repo = '~/repo'\n"
    )
    agent = harvest.workflows["wf_abc"]["agents"]["b1"]
    assert agent == {"label": "probe", "phase": "P", "agentType": "workflow-subagent", "model": "opus"}
    assert harvest.workflows["wf_abc"]["workflowName"] == "build-it"


def test_gate_flags_leaks_and_passes_clean_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    leaky = tmp_path / "leaky"
    leaky.mkdir()
    (leaky / "a.md").write_text(f"ok\npath {HOME}/x\nmail {EMAIL}\n{OPEN}\nnote acme corp\n")
    (leaky / "b.jsonl").write_text(
        json.dumps({"type": "attachment", "attachment": dict(type="instructions")})
    )
    (leaky / "c.html").write_text("<style>.x ." + et.SR + " {color: red}</style>\n" + "y" * 300)
    monkeypatch.setattr(et, "MAX_FILE_BYTES", 200)
    hits = et.scan_paths(leaky, ["acme corp"])
    assert hits == [
        "a.md:2: home-path",
        "a.md:3: email",
        "a.md:4: marker:reminder",
        "a.md:5: denylist",
        "b.jsonl:1: marker:instructions-attachment",
        "c.html:0: size>50MB",
    ]
    assert et.report_gate(hits) == 1

    clean = tmp_path / "clean"
    clean.mkdir()
    code = json.dumps({"text": "@router.post\n" * 3 + "kris.dimitrov92@gmail.com ~/repo [redacted-email]"})
    (clean / "raw.jsonl").write_text(code + "\n")
    (clean / "rendered.md").write_text(json.dumps({"attachment": dict(type="instructions")}) + "\n")
    assert et.scan_paths(clean, ["acme corp"]) == []
    assert et.report_gate([]) == 0


def test_export_publishes_only_when_gate_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_jsonl(tmp_path / "proj" / f"{SID}.jsonl", [user("u1", None, f"see {HOME}/repo")])
    args = argparse.Namespace(
        out=tmp_path / "out",
        staging=tmp_path / "staging",
        projects_root=tmp_path / "proj",
        session=SID[:8],
        denylist=tmp_path / "missing.txt",
        no_render=True,
        html_split=[],
    )
    monkeypatch.setattr(et, "scan_paths", lambda *_: ["raw/x.jsonl:1: email"])
    assert et.export(args) == 1
    assert not args.out.exists()

    monkeypatch.undo()
    assert et.export(args) == 0
    assert sorted(p.name for p in args.out.iterdir()) == ["README.md", "meta.json", "raw", "rendered"]
    meta = json.loads((args.out / "meta.json").read_text())
    assert meta["session_id"] == SID
    assert meta["records"]["kept"] == {"user": 1}
    assert et.scan_paths(args.out, []) == []
    assert "NOTE.md" not in (args.out / "README.md").read_text()


def test_reexport_keeps_the_note_and_points_to_it(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "proj" / f"{SID}.jsonl", [user("u1", None, "hello")])
    out = tmp_path / "out"
    out.mkdir()
    (out / "NOTE.md").write_text("Dear reviewers\n")
    (out / "README.md").write_text("stale\n")
    args = argparse.Namespace(
        out=out,
        staging=tmp_path / "staging",
        projects_root=tmp_path / "proj",
        session=SID[:8],
        denylist=tmp_path / "missing.txt",
        no_render=True,
        html_split=[],
    )
    assert et.export(args) == 0
    assert (out / "NOTE.md").read_text() == "Dear reviewers\n"
    assert "[`NOTE.md`](NOTE.md)" in (out / "README.md").read_text()


def test_html_split_renders_time_ordered_parts_and_lists_them() -> None:
    assert et.render_jobs([]) == [("html", "transcript.html", []), ("md", "transcript.md", [])]
    assert et.render_jobs(["2026-10-19 01:00"]) == [
        ("html", "transcript-1.html", ["--to-date", "2026-10-19 01:00"]),
        ("html", "transcript-2.html", ["--from-date", "2026-10-19 01:00"]),
        ("md", "transcript.md", []),
    ]
    meta = {"session_id": SID, "date_range": {"start": "a", "end": "b"}}
    text = et.readme(meta, html_parts=2)
    assert "`rendered/transcript-1.html`, `rendered/transcript-2.html`" in text
    assert "transcript.html" not in text.replace("transcript-", "")
