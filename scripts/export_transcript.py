"""Export this repo's Claude Code session as a redacted transcript: raw JSONL, HTML/Markdown, meta.json.

Everything is built in a staging dir; outputs reach --out only after a leak gate scans every file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGING = ROOT / ".temp" / "transcript" / "staging"
DEFAULT_OUT = ROOT / ".context" / "transcript"
DEFAULT_DENYLIST = ROOT / ".temp" / "transcript" / "denylist.txt"
RENDERER = "claude-code-log@1.7.0"
RENDER_FLAGS = ("--depth", "tool", "--image-export-mode", "placeholder", "--no-cache")
PROJECT_NAME = "veritrade-analytics"
STAGING_MARKER = ".transcript-staging"
OUTPUT_NAMES = ("raw", "rendered", "meta.json", "README.md")
# Hand-written and committed beside the outputs; publish() never touches it.
NOTE_NAME = "NOTE.md"
MAX_FILE_BYTES = 50 * 1024 * 1024

ALLOWED_EMAILS = frozenset({"kris.dimitrov92@gmail.com", "noreply@anthropic.com"})
PRIVATE_DATA = "evals/data/private"
HTML_MIN_CHARS = 2048
WEB_MIN_CHARS = 4096
FLAGGED_INPUT_MAX_CHARS = 300
SKILL_BODY_MIN_CHARS = 500

# Assembled at runtime so this file's source, once it appears in the transcript it exports,
# carries none of the literals the gate hunts for.
SR = "system" + "-reminder"
BLOB = "api-request" + "-blob"
SESSION_CTX = "session" + "_context"

KEEP_TYPES = frozenset(
    {"user", "assistant", "system", "attachment", "mode", "permission-mode", "ai-title", "custom-title"}
    | {"agent-name"}
)
KEEP_SYSTEM = frozenset({"informational", "turn_duration", "away_summary"})
KEEP_ATTACHMENTS = frozenset(
    {"plan_mode", "ultra_effort_enter", "model", "date", "skill_mention", "queued_command"}
)
BASE_KEYS = frozenset(
    {"type", "uuid", "parentUuid", "logicalParentUuid", "isSidechain", "userType", "cwd", "sessionId"}
    | {"version", "gitBranch", "agentId", "timestamp", "isMeta", "entrypoint"}
)
TYPE_KEYS: dict[str, frozenset[str]] = {
    "user": frozenset(
        {"message", "sourceToolUseID", "sourceToolAssistantUUID", "toolUseResult", "permissionMode"}
        | {"origin", "isCompactSummary", "isVisibleInTranscriptOnly"}
    ),
    "assistant": frozenset({"message", "requestId", "isApiErrorMessage", "error", "effort"}),
    "system": frozenset({"subtype", "content", "level", "durationMs", "messageCount", "compactMetadata"}),
    "attachment": frozenset({"attachment"}),
    "mode": frozenset({"mode"}),
    "permission-mode": frozenset({"permissionMode"}),
    "ai-title": frozenset({"aiTitle"}),
    "custom-title": frozenset({"customTitle"}),
    "agent-name": frozenset({"agentName"}),
}
MESSAGE_KEYS = frozenset({"id", "type", "role", "model", "content", "stop_reason", "stop_sequence", "usage"})
# toolUseResult duplicates tool output (files, pages, stdout); only spawn/workflow/answer links survive.
LINK_RESULT_KEYS = frozenset({"agentId", "taskId", "runId", "answers"})
SLIM_RESULT_KEYS = frozenset(
    {"agentId", "status", "description", "isAsync", "resolvedModel", "taskId", "taskType", "runId"}
    | {"workflowName", "summary", "answers", "questions", "annotations"}
)
MODEL_AUTHORED_TOOLS = frozenset(
    {"Agent", "Task", "Workflow", "SendMessage", "AskUserQuestion", "ExitPlanMode", "EnterPlanMode"}
    | {"Skill", "TodoWrite", "ToolSearch", "TaskOutput", "SubagentHandback"}
)
TARGET_KEYS = frozenset({"file_path", "path", "notebook_path", "pattern", "glob", "command", "url"})
_HOME = str(Path.home())
_ANCESTORS = [p for p in ROOT.parents if str(p).startswith(_HOME)]
_ANCESTOR_ALT = "|".join(
    re.escape(form)
    for p in _ANCESTORS
    for form in (str(p), "~" + str(p)[len(_HOME) :], "$HOME" + str(p)[len(_HOME) :])
)
# Local config, other sessions and dir listings above the repo expose the author's private setup;
# this session's own plan file is the one config-dir file kept.
PRIVATE_CONFIG_RES = [
    re.compile(r"/\.claude[\w.-]*(?:/(?!plans/[\w.-]+\.md)|(?=[\s'\"*]|$))"),
    re.compile(r"/\.(?:ssh|aws|gnupg|docker|kube|netrc|gitconfig|npmrc|pypirc)\b|/\.config/(?:gh|git)\b"),
    re.compile(r"\bsettings(?:\.local)?\.json\b|\bCLAUDE(?:\.local)?\.md\b|/\.vscode\b|\.code-workspace\b"),
    re.compile(r"(?:^|[\s/'\"=])\.env(?:\.[\w-]+)?(?![\w-])"),
    re.compile(r"(?:^|[\s;&|(])(?:env|printenv)(?=\s*(?:$|[|;&>)]))|/Applications\b"),
    re.compile(r"/claude-\d+/\S*/tasks/"),
]
if _ANCESTOR_ALT:
    PRIVATE_CONFIG_RES.append(
        re.compile(r"\b(?:ls|tree|du|find)\b[^;&|]*?\s(?:" + _ANCESTOR_ALT + r")/?(?=[\s;&|)]|$)")
    )
WEB_TOOL_RE = re.compile(r"fetch|get_page_text|read_page|scrape", re.IGNORECASE)
FETCH_CMD_RE = re.compile(r"\b(?:curl|wget)\b|requests\.get|urlopen|httpx\.get")
# Output that carries private file content shows it as grep hits or head/tail headers; prose and code
# that merely name the directory stay readable.
PRIVATE_CONTENT_RE = re.compile(
    re.escape(PRIVATE_DATA)
    + r"/\S*?(?::\d+:|\.(?:html?|txt|jsonl?|xml|md)[:-]\s)|==> \S*"
    + re.escape(PRIVATE_DATA)
)
COMMAND_NAME_RE = re.compile(r"<command-name>/?([\w:.-]+)</command-name>")

REMINDER_BLOCK_RE = re.compile("<" + SR + r">(?:\r?\n|\\+n)[\s\S]*?(?:</" + SR + r">|\Z)")
REMINDER_TAG_RE = re.compile("</?" + SR + ">")
MARKER_WORDS = ((SR, "system reminder"), (BLOB, "api request blob"), (SESSION_CTX, "session context"))
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
HOME_RE = re.compile(r"/Users/[^/\s\"'`<>|:;,)\]}]*")
ENCODED_HOME_RE = re.compile(r"(?<![A-Za-z0-9])-Users-[A-Za-z0-9_.]+")
# Short account names collide with real path segments (/dev, /bin), so only distinctive ones count.
_USER = Path.home().name
USER_FRAGMENT_RE = re.compile("/" + re.escape(_USER) + r"(?![A-Za-z0-9])") if len(_USER) >= 5 else None
SECRET_RES: dict[str, re.Pattern[str]] = {
    "sk_key": re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{30,}"),
    "github_pat": re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    "aws_key": re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])"),
    "private_key": re.compile(r"-----BEGIN [A-Z0-9 ]*KEY-----(?:[\s\S]*?-----END [A-Z0-9 ]*KEY-----)?"),
    "slack_token": re.compile(r"(?<![A-Za-z0-9])xox[abposr]-[A-Za-z0-9-]{10,}"),
    "jwt": re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    "hf_token": re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}"),
    "google_key": re.compile(r"AIza[0-9A-Za-z_-]{35}"),
}
_Q = r"(?:\"|'|&quot;|&#34;|&#39;|&#x27;)"
INSTRUCTIONS_TYPE_RE = re.compile(_Q + r"type" + _Q + r"\s*[:=]\s*" + _Q + r"instructions" + _Q)
# A CSS selector in the renderer's stylesheet is the only legitimate leftover of the reminder word.
SR_LEAK_RE = re.compile(r"(?<!\.)" + re.escape(SR))
JSON_ESCAPE_RE = re.compile(r"\\(?:[nrtbf\"\\/]|u[0-9a-fA-F]{4})")
HTML_DOC_RE = re.compile(r"<!doctype\s+html[\s>]|<html[\s>]", re.IGNORECASE)
# Source files that merely contain an HTML snippet are code, not fetched pages.
CODE_SUFFIXES = frozenset(
    {".py", ".pyi", ".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs", ".json", ".jsonl", ".toml"}
    | {".md"}
    | {".yml", ".yaml", ".css", ".scss", ".sh", ".cfg", ".ini", ".lock", ".sql"}
)
VIEW_CMD_RE = re.compile(r"^\s*(?:cd\s+\S+\s*&&\s*)?(?:cat|head|tail|nl|sed\s+-n)\b")
FILE_TOKEN_RE = re.compile(r"[\w~./-]+\.[A-Za-z0-9]+\b")


def load_denylist(path: Path) -> list[str]:
    if not path.is_file():
        return []
    terms = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [t for t in terms if t and not t.startswith("#")]


def is_html(text: str) -> bool:
    return bool(HTML_DOC_RE.search(text))


@dataclass
class Sanitizer:
    denylist: list[str] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        terms = sorted(self.denylist, key=len, reverse=True)
        alternation = "|".join(re.escape(t) for t in terms)
        self._deny = re.compile(alternation, re.IGNORECASE) if terms else None

    def _sub(self, rule: str, regex: re.Pattern[str], repl: str, text: str) -> str:
        text, n = regex.subn(repl, text)
        if n:
            self.counts[rule] += n
        return text

    def _email(self, m: re.Match[str]) -> str:
        if m.group(0).lower() in ALLOWED_EMAILS:
            return m.group(0)
        self.counts["email"] += 1
        return "[redacted-email]"

    def redact(self, text: str) -> str:
        for name, regex in SECRET_RES.items():
            text = self._sub(f"secret:{name}", regex, "[redacted-secret]", text)
        text = EMAIL_RE.sub(self._email, text)
        text = self._sub("home_path", HOME_RE, "~", text)
        text = self._sub("home_path", ENCODED_HOME_RE, "-~", text)
        if USER_FRAGMENT_RE is not None:
            text = self._sub("home_path", USER_FRAGMENT_RE, "/[user]", text)
        if self._deny is not None:
            text = self._sub("denylist", self._deny, "[redacted]", text)
        return text

    def strip_reminders(self, text: str) -> str:
        text = self._sub("system_reminder", REMINDER_BLOCK_RE, "", text)
        text = REMINDER_TAG_RE.sub(
            lambda m: "[/system reminder]" if m[0][1] == "/" else "[system reminder]", text
        )
        for word, repl in MARKER_WORDS:
            if word in text:
                self.counts["marker_neutralized"] += text.count(word)
                text = text.replace(word, repl)
        return text

    def text(self, text: str) -> str:
        return self.redact(self.strip_reminders(text))

    def omit(self, text: str, reason: str) -> str:
        self.counts[f"omitted:{reason}"] += 1
        digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]
        label = "private local config" if reason == "private_config" else "publisher/web content"
        return f"[omitted: {label}, sha256={digest}, {len(text)} chars]"

    def any(self, value: Any, html: bool = True) -> Any:
        """Clean every string in a JSON value; `html` treats long raw HTML as publisher content."""
        if isinstance(value, str):
            stripped = self.strip_reminders(value)
            if html and len(stripped) > HTML_MIN_CHARS and is_html(stripped):
                return self.omit(stripped, "html")
            return self.redact(stripped)
        if isinstance(value, list):
            return [self.any(v, html) for v in value]
        if isinstance(value, dict):
            # Keys can be prose too: AskUserQuestion answers are keyed by the question text.
            return {self.text(k) if isinstance(k, str) else k: self.any(v, html) for k, v in value.items()}
        return value


@dataclass
class ToolInfo:
    name: str
    flag: str | None = None  # omission reason that covers both the call's long inputs and its result
    fetch_like: bool = False
    code_file: bool = False


def tool_info(block: dict[str, Any]) -> ToolInfo:
    name = str(block.get("name", ""))
    raw = block.get("input")
    inp: dict[str, Any] = raw if isinstance(raw, dict) else {}
    targets = [str(v) for k, v in inp.items() if k in TARGET_KEYS and isinstance(v, str)]
    flag = None
    if any(PRIVATE_DATA in t for t in targets):
        flag = "private_data"
    elif any(rx.search(t) for t in targets for rx in PRIVATE_CONFIG_RES):
        flag = "private_config"
    command = str(inp.get("command", ""))
    file_path = inp.get("file_path") or inp.get("notebook_path")
    code_file = isinstance(file_path, str) and Path(file_path).suffix.lower() in CODE_SUFFIXES
    return ToolInfo(
        name=name,
        flag=flag,
        fetch_like=name == "Bash" and bool(FETCH_CMD_RE.search(command)),
        code_file=code_file or (name == "Bash" and views_code_only(command)),
    )


def views_code_only(command: str) -> bool:
    """A read-only `cat`/`head`/`sed -n` over source files shows code, whatever tags it contains."""
    if not VIEW_CMD_RE.match(command) or "http" in command:
        return False
    files = [Path(f) for f in FILE_TOKEN_RE.findall(command)]
    return bool(files) and all(f.suffix.lower() in CODE_SUFFIXES or f.name.startswith(".") for f in files)


def result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(i.get("text", "")) for i in content if isinstance(i, dict))
    return ""


def omission_reason(info: ToolInfo | None, text: str) -> str | None:
    """Why a tool result must not be published, or None to keep it."""
    name = info.name if info else ""
    if info and info.flag:
        return info.flag
    if len(text) > HTML_MIN_CHARS and is_html(text) and not (info and info.code_file):
        return "html"
    if WEB_TOOL_RE.search(name) and len(text) > WEB_MIN_CHARS:
        return "web_fetch"
    if info and info.fetch_like and len(text) > WEB_MIN_CHARS:
        return "web_fetch"
    if name not in MODEL_AUTHORED_TOOLS and PRIVATE_CONTENT_RE.search(text):
        return "private_mention"
    return None


@dataclass
class FileContext:
    san: Sanitizer
    tools: dict[str, ToolInfo]


def clean_tool_input(block: dict[str, Any], ctx: FileContext) -> dict[str, Any]:
    info = ctx.tools.get(str(block.get("id")))
    inp = block.get("input")
    if not isinstance(inp, dict):
        return {**block, "input": ctx.san.any(inp)}
    flag = info.flag if info else None
    code_file = bool(info and info.code_file)
    out: dict[str, Any] = {}
    for key, value in inp.items():
        if isinstance(value, str):
            stripped = ctx.san.strip_reminders(value)
            if flag and len(stripped) > FLAGGED_INPUT_MAX_CHARS:
                out[key] = ctx.san.omit(stripped, flag)
            elif len(stripped) > HTML_MIN_CHARS and is_html(stripped) and not code_file:
                out[key] = ctx.san.omit(stripped, "html")
            else:
                out[key] = ctx.san.redact(stripped)
        elif flag and len(json.dumps(value)) > FLAGGED_INPUT_MAX_CHARS:
            out[key] = ctx.san.omit(json.dumps(value), flag)
        else:
            out[key] = ctx.san.any(value, html=not code_file)
    return {**block, "input": out}


def strip_nested(value: Any, san: Sanitizer) -> Any:
    if isinstance(value, str):
        return san.strip_reminders(value)
    if isinstance(value, list):
        return [strip_nested(v, san) for v in value]
    if isinstance(value, dict):
        return {k: strip_nested(v, san) for k, v in value.items()}
    return value


def clean_tool_result(block: dict[str, Any], ctx: FileContext) -> dict[str, Any]:
    stripped = strip_nested(block.get("content"), ctx.san)
    text = result_text(stripped)
    reason = omission_reason(ctx.tools.get(str(block.get("tool_use_id"))), text)
    if reason:
        return {**block, "content": ctx.san.omit(text, reason)}
    # omission_reason already weighed HTML with the tool context; the generic net would second-guess it.
    if isinstance(stripped, list):
        items = [omitted_media(i) if is_media(i) else i for i in stripped]
        return {**block, "content": [ctx.san.any(i, html=False) for i in items]}
    return {**block, "content": ctx.san.any(stripped, html=False)}


def is_media(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") in ("image", "document")


def omitted_media(block: dict[str, Any]) -> dict[str, Any]:
    return {"type": "text", "text": f"[omitted: {block.get('type')}]"}


def clean_text_block(text: str, ctx: FileContext, meta: bool) -> str:
    stripped = ctx.san.strip_reminders(text)
    if meta and len(stripped) > SKILL_BODY_MIN_CHARS:
        ctx.san.counts["omitted:skill_body"] += 1
        return f"[omitted: skill/command instructions, {len(stripped)} chars]"
    if not stripped.strip() and text.strip():
        return "[removed: harness reminder]"
    return ctx.san.redact(stripped)


def clean_content(content: Any, ctx: FileContext, meta: bool) -> Any:
    if isinstance(content, str):
        return clean_text_block(content, ctx, meta)
    if not isinstance(content, list):
        return ctx.san.any(content)
    out: list[Any] = []
    for block in content:
        if not isinstance(block, dict):
            out.append(ctx.san.any(block))
            continue
        kind = block.get("type")
        if is_media(block):
            out.append(omitted_media(block))
        elif kind == "text":
            out.append({**block, "text": clean_text_block(str(block.get("text", "")), ctx, meta)})
        elif kind == "thinking":
            kept = {k: v for k, v in block.items() if k != "signature"}
            out.append({**kept, "thinking": ctx.san.text(str(block.get("thinking", "")))})
        elif kind == "redacted_thinking":
            out.append({"type": "text", "text": "[omitted: redacted_thinking]"})
        elif kind == "tool_use":
            out.append(clean_tool_input(block, ctx))
        elif kind == "tool_result":
            out.append(clean_tool_result(block, ctx))
        else:
            out.append(ctx.san.any(block))
    return out


def record_key(rec: dict[str, Any]) -> str:
    kind = str(rec.get("type", "<none>"))
    if kind == "attachment":
        return f"attachment:{(rec.get('attachment') or {}).get('type')}"
    if kind == "system":
        return f"system:{rec.get('subtype')}"
    return kind


def filter_record(rec: dict[str, Any], ctx: FileContext) -> dict[str, Any] | None:
    """Allowlist one transcript record and sanitize what survives; None means drop."""
    kind = rec.get("type")
    if kind not in KEEP_TYPES:
        return None
    if kind == "system" and rec.get("subtype") not in KEEP_SYSTEM:
        return None
    if kind == "attachment" and (rec.get("attachment") or {}).get("type") not in KEEP_ATTACHMENTS:
        return None
    allowed = BASE_KEYS | TYPE_KEYS[str(kind)]
    out: dict[str, Any] = {}
    for key, value in rec.items():
        if key not in allowed:
            continue
        if key == "message" and isinstance(value, dict):
            msg = {k: ctx.san.any(v) for k, v in value.items() if k in MESSAGE_KEYS and k != "content"}
            msg["content"] = clean_content(value.get("content"), ctx, bool(rec.get("isMeta")))
            out[key] = msg
        elif key == "toolUseResult":
            if isinstance(value, dict) and LINK_RESULT_KEYS & value.keys():
                out[key] = ctx.san.any({k: v for k, v in value.items() if k in SLIM_RESULT_KEYS})
        else:
            out[key] = ctx.san.any(value)
    return out


@dataclass
class Harvest:
    """Metadata collected from every record before the allowlist drops anything."""

    versions: Counter[str] = field(default_factory=Counter)
    entrypoints: Counter[str] = field(default_factory=Counter)
    models: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    usage: dict[tuple[str, str, str], dict[str, int]] = field(default_factory=dict)
    usage_model: dict[tuple[str, str, str], str] = field(default_factory=dict)
    efforts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    thinking_params: Counter[tuple[tuple[str, Any], ...]] = field(default_factory=Counter)
    thinking_display: Counter[str] = field(default_factory=Counter)
    permission_modes: Counter[str] = field(default_factory=Counter)
    modes: Counter[str] = field(default_factory=Counter)
    attachments_seen: Counter[str] = field(default_factory=Counter)
    skills: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    tools: Counter[str] = field(default_factory=Counter)
    subagents: list[dict[str, Any]] = field(default_factory=list)
    workflows: dict[str, dict[str, Any]] = field(default_factory=dict)
    titles: list[str] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)
    kept: Counter[str] = field(default_factory=Counter)
    dropped: Counter[str] = field(default_factory=Counter)
    bad_lines: int = 0

    def record(self, rec: dict[str, Any], scope: str, file_key: str) -> None:
        kind = rec.get("type")
        if isinstance(rec.get("version"), str):
            self.versions[rec["version"]] += 1
        if isinstance(rec.get("entrypoint"), str):
            self.entrypoints[rec["entrypoint"]] += 1
        if isinstance(rec.get("timestamp"), str):
            self.timestamps.append(rec["timestamp"])
        if kind == "assistant":
            self._assistant(rec, scope, file_key)
        elif kind == "user":
            text = json.dumps((rec.get("message") or {}).get("content"))
            self.skills["slash_command"].update(COMMAND_NAME_RE.findall(text))
        elif kind == "api-request":
            thinking = (rec.get("params") or {}).get("thinking")
            if isinstance(thinking, dict):
                self.thinking_params[
                    tuple((k, thinking[k]) for k in ("type", "display") if k in thinking)
                ] += 1
        elif kind == "permission-mode":
            self.permission_modes[str(rec.get("permissionMode"))] += 1
        elif kind == "mode":
            self.modes[str(rec.get("mode"))] += 1
        elif kind == "attachment":
            att = rec.get("attachment") or {}
            self.attachments_seen[str(att.get("type"))] += 1
            if att.get("type") == "skill_mention" and att.get("skillName"):
                self.skills["mention"][str(att["skillName"])] += 1
        elif kind in ("ai-title", "custom-title"):
            title = rec.get("customTitle") or rec.get("aiTitle")
            if isinstance(title, str):
                self.titles.append(title)

    def _assistant(self, rec: dict[str, Any], scope: str, file_key: str) -> None:
        msg = rec.get("message") or {}
        mid = str(msg.get("id", rec.get("uuid")))
        model = str(msg.get("model", "unknown"))
        key = (scope, file_key, mid)
        if key not in self.usage:
            self.models[scope][model] += 1
            self.usage[key] = {}
        self.usage_model[key] = model
        usage = msg.get("usage") or {}
        details = usage.get("output_tokens_details") or {}
        fields = {
            "input": usage.get("input_tokens"),
            "cache_creation": usage.get("cache_creation_input_tokens"),
            "cache_read": usage.get("cache_read_input_tokens"),
            "output": usage.get("output_tokens"),
            "thinking": details.get("thinking_tokens"),
        }
        # Content blocks of one response repeat its usage; streaming makes later copies larger, so keep max.
        for name, value in fields.items():
            if isinstance(value, int):
                self.usage[key][name] = max(self.usage[key].get(name, 0), value)
        for name in ("effort", "perTurnEffort"):
            if isinstance(rec.get(name), str):
                self.efforts[name][rec[name]] += 1
        if isinstance(rec.get("thinkingDisplay"), str):
            self.thinking_display[rec["thinkingDisplay"]] += 1
        for block in msg.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                self.tools[str(block.get("name"))] += 1
                if block.get("name") == "Skill":
                    self.skills["skill_tool"][str((block.get("input") or {}).get("skill"))] += 1

    def usage_summary(self) -> dict[str, Any]:
        def zero() -> dict[str, int]:
            return dict.fromkeys(("input", "cache_creation", "cache_read", "output", "thinking"), 0)

        by_scope: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(zero))
        total = zero()
        for key, fields in self.usage.items():
            scope, model = key[0], self.usage_model[key]
            for name, value in fields.items():
                by_scope[scope][model][name] += value
                total[name] += value
        return {
            "note": "summed once per (file, message.id); thinking is a subset of output",
            "total": total,
            "by_scope_and_model": {s: dict(m) for s, m in by_scope.items()},
        }


def read_jsonl(path: Path, harvest: Harvest) -> Iterator[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            harvest.bad_lines += 1  # the live session's last line can be mid-write
            continue
        if isinstance(rec, dict):
            yield rec


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps(r, ensure_ascii=False) for r in records)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def stage_transcript(src: Path, dest: Path, scope: str, san: Sanitizer, harvest: Harvest) -> None:
    records = list(read_jsonl(src, harvest))
    tools: dict[str, ToolInfo] = {}
    for rec in records:
        harvest.record(rec, scope, src.name)
        content = (rec.get("message") or {}).get("content") if rec.get("type") == "assistant" else None
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tools[str(block.get("id"))] = tool_info(block)
    ctx = FileContext(san=san, tools=tools)
    kept: list[dict[str, Any]] = []
    skipped: dict[str, Any] = {}
    for rec in records:
        cleaned = filter_record(rec, ctx)
        if cleaned is None:
            harvest.dropped[record_key(rec)] += 1
            if isinstance(rec.get("uuid"), str):
                skipped[rec["uuid"]] = rec.get("parentUuid")
        else:
            harvest.kept[record_key(rec)] += 1
            kept.append(cleaned)
    write_jsonl(dest, [reparent(rec, skipped) for rec in kept])


def reparent(rec: dict[str, Any], skipped: dict[str, Any]) -> dict[str, Any]:
    """Point past dropped records to the nearest kept ancestor so the renderer keeps the thread."""
    for key in ("parentUuid", "logicalParentUuid"):
        parent, seen = rec.get(key), set()
        while isinstance(parent, str) and parent in skipped and parent not in seen:
            seen.add(parent)
            parent = skipped[parent]
        if key in rec:
            rec[key] = parent
    return rec


def stage_journal(src: Path, dest: Path, san: Sanitizer, harvest: Harvest) -> None:
    run = harvest.workflows.setdefault(src.parent.name, {"run_id": src.parent.name, "agents": {}})
    records = list(read_jsonl(src, harvest))
    for rec in records:
        harvest.kept[f"journal:{rec.get('type')}"] += 1
        if rec.get("type") == "started" and isinstance(rec.get("agentId"), str):
            agent = run["agents"].setdefault(rec["agentId"], {})
            agent.update({"label": rec.get("label"), "phase": rec.get("phase")})
    write_jsonl(dest, [san.any(r) for r in records])


def stage_json(src: Path, dest: Path, san: Sanitizer) -> dict[str, Any]:
    data = json.loads(src.read_text(encoding="utf-8"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(san.any(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data if isinstance(data, dict) else {}


def harvest_snapshot(data: dict[str, Any], harvest: Harvest) -> None:
    run = harvest.workflows.setdefault(str(data.get("runId")), {"run_id": data.get("runId"), "agents": {}})
    run.update(
        {k: data.get(k) for k in ("workflowName", "status", "agentCount", "totalTokens", "durationMs")}
    )
    for node in data.get("workflowProgress") or []:
        if isinstance(node, dict) and node.get("type") == "workflow_agent":
            agent = run["agents"].setdefault(str(node.get("agentId")), {})
            for key in ("label", "agentType", "model", "state", "tokens"):
                if node.get(key) is not None:
                    agent[key] = node[key]
            agent.setdefault("phase", node.get("phaseTitle"))


def stage_session(project_dir: Path, sid: str, dest: Path, san: Sanitizer, harvest: Harvest) -> None:
    """Mirror the session's main JSONL, subagents/** and workflows/** into dest, filtered and redacted."""
    stage_transcript(project_dir / f"{sid}.jsonl", dest / f"{sid}.jsonl", "main", san, harvest)
    session_dir = project_dir / sid
    for src in sorted((session_dir / "subagents").rglob("*")):
        rel = src.relative_to(project_dir)
        if not src.is_file():
            continue
        if src.name == "journal.jsonl":
            stage_journal(src, dest / rel, san, harvest)
        elif src.suffix == ".jsonl":
            stage_transcript(src, dest / rel, "subagents", san, harvest)
        elif src.name.endswith(".meta.json"):
            meta = stage_json(src, dest / rel, san)
            run_id = src.parent.name if src.parent.parent.name == "workflows" else None
            harvest.subagents.append({"agentType": meta.get("agentType"), "model": meta.get("model")})
            if run_id:
                run = harvest.workflows.setdefault(run_id, {"run_id": run_id, "agents": {}})
                agent = run["agents"].setdefault(
                    src.name.removeprefix("agent-").removesuffix(".meta.json"), {}
                )
                agent.setdefault("agentType", meta.get("agentType"))
                agent.setdefault("model", meta.get("model"))
    for src in sorted((session_dir / "workflows").glob("*.json")):
        harvest_snapshot(stage_json(src, dest / src.relative_to(project_dir), san), harvest)
    for src in sorted((session_dir / "workflows" / "scripts").glob("*.js")):
        target = dest / src.relative_to(project_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(san.text(src.read_text(encoding="utf-8")), encoding="utf-8")
        name, sep, tail = src.stem.rpartition("-wf_")
        run_id = f"wf_{tail}" if sep else src.stem
        run = harvest.workflows.setdefault(run_id, {"run_id": run_id, "agents": {}})
        run["script"] = src.name
        if sep and not run.get("workflowName"):
            run["workflowName"] = name


def build_meta(harvest: Harvest, san: Sanitizer, sid: str) -> dict[str, Any]:
    by_type: Counter[str] = Counter(str(s.get("agentType")) for s in harvest.subagents)
    by_model: Counter[str] = Counter(str(s.get("model")) for s in harvest.subagents)
    workflows = [
        {**{k: v for k, v in run.items() if k != "agents"}, "agents": list(run["agents"].values())}
        for run in sorted(harvest.workflows.values(), key=lambda r: str(r.get("run_id")))
    ]
    stamps = sorted(harvest.timestamps)
    models: dict[str, Any] = {"note": "responses per model, deduped by message.id"}
    models.update({scope: dict(c) for scope, c in harvest.models.items()})
    meta: dict[str, Any] = {
        "session_id": sid,
        "title": harvest.titles[-1] if harvest.titles else None,
        "harness": {
            "name": "Claude Code",
            "version": harvest.versions.most_common(1)[0][0] if harvest.versions else None,
            "versions_seen": dict(harvest.versions),
            "entrypoint": harvest.entrypoints.most_common(1)[0][0] if harvest.entrypoints else None,
        },
        "models": models,
        "effort": {"note": "assistant records per value"} | {k: dict(c) for k, c in harvest.efforts.items()},
        "thinking": {
            "api_params": [dict(pairs) | {"requests": n} for pairs, n in harvest.thinking_params.items()],
            "display_on_records": dict(harvest.thinking_display),
        },
        "modes": {
            "permission_modes": dict(harvest.permission_modes),
            "modes": dict(harvest.modes),
            "plan_mode": harvest.permission_modes["plan"] > 0 or harvest.attachments_seen["plan_mode"] > 0,
            "ultracode": harvest.attachments_seen["ultra_effort_enter"] > 0,
            "auto_mode": harvest.permission_modes["auto"] > 0 or harvest.attachments_seen["auto_mode"] > 0,
        },
        "skills_invoked": {k: dict(c) for k, c in harvest.skills.items() if c},
        "tools_used": dict(harvest.tools.most_common()),
        "workflows": workflows,
        "subagents": {"total": len(harvest.subagents), "by_type": dict(by_type), "by_model": dict(by_model)},
        "usage_tokens": harvest.usage_summary(),
        "date_range": {"start": stamps[0] if stamps else None, "end": stamps[-1] if stamps else None},
        "records": {
            "kept_total": sum(harvest.kept.values()),
            "dropped_total": sum(harvest.dropped.values()),
            "kept": dict(sorted(harvest.kept.items())),
            "dropped": dict(sorted(harvest.dropped.items())),
            "unparseable_lines": harvest.bad_lines,
        },
        "redactions": dict(sorted(san.counts.items())),
    }
    # Dropped-record tallies are keyed by marker type names, so keys go through the sanitizer too.
    cleaned: dict[str, Any] = Sanitizer(denylist=san.denylist).any(meta)
    return cleaned


def html_names(parts: int) -> list[str]:
    return ["transcript.html"] if parts <= 1 else [f"transcript-{i}.html" for i in range(1, parts + 1)]


def readme(meta: dict[str, Any], note: bool = False, html_parts: int = 1) -> str:
    dates = meta["date_range"]
    pointer = f"\nStart with [`{NOTE_NAME}`]({NOTE_NAME}), a note from Kris to the reviewers.\n"
    pointer = pointer if note else ""
    html = ", ".join(f"`rendered/{name}`" for name in html_names(html_parts))
    split = " (HTML in time-ordered parts, each under GitHub's 50 MB file limit)" if html_parts > 1 else ""
    return f"""# AI conversation transcript

Exported from the Claude Code session that built this repo (`{meta["session_id"]}`,
{dates["start"]} to {dates["end"]}), including every subagent and workflow run it spawned.
{pointer}
## Contents

- {html}, `rendered/transcript.md`: readable render ({RENDERER}, depth `tool`){split}.
- `raw/`: the filtered, redacted JSONL in the original session layout (main thread, `subagents/**`,
  `workflows/**`); re-renderable with claude-code-log.
- `meta.json`: harness version, models, effort and thinking settings, modes, skills, workflows,
  subagent counts, token usage, record counts and redaction counts.

## What was removed, and why

- Harness plumbing: API request records, file-history snapshots, prompt snapshots, hook output,
  environment, listing and delta attachments. Noise for a reader, and some of it holds private configuration.
- Private context: the author's global instruction file, account and session context, credential data,
  harness reminder blocks inside messages and tool results, skill instruction bodies, and the output of
  tool calls that read local configuration (Claude config dirs other than this session's plan file,
  SSH/git/editor/env files, other session logs, listings above the repo) which becomes
  `[omitted: private local config, sha256=<prefix>, <n> chars]`.
- Thinking-block signatures, images and documents.
- Third-party content: tool calls touching `{PRIVATE_DATA}/`, raw HTML pages over 2 KB and fetched web
  content over 4 KB become `[omitted: publisher/web content, sha256=<prefix>, <n> chars]`.
- Redactions: home paths become `~`; emails other than the author's public address become
  `[redacted-email]`; credential-shaped strings become `[redacted-secret]`; private terms become `[redacted]`.

## How it was produced

`uv run python scripts/export_transcript.py`: allowlist filter and redaction into a staging dir,
render with `uvx {RENDERER}`, then a leak gate over every output file (paths, emails, secrets,
private markers, size). Only a passing export is moved here. The final turns that ran this export
and pushed it are not included.
"""


def render_jobs(splits: list[str]) -> list[tuple[str, str, list[str]]]:
    """(format, file name, date filters) per output; HTML parts cover the gaps between split times."""
    bounds: list[str | None] = [None, *splits, None]
    jobs = []
    for name, start, end in zip(html_names(len(splits) + 1), bounds[:-1], bounds[1:], strict=True):
        dates = (["--from-date", start] if start else []) + (["--to-date", end] if end else [])
        jobs.append(("html", name, dates))
    return [*jobs, ("md", "transcript.md", [])]


def render(session_file: Path, out_dir: Path, cache: Path, splits: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CLAUDE_CODE_LOG_CACHE_PATH": str(cache)}
    for fmt, name, dates in render_jobs(splits):
        cmd = [
            "uvx",
            RENDERER,
            str(session_file),
            "-o",
            str(out_dir / name),
            "-f",
            fmt,
            *RENDER_FLAGS,
            *dates,
        ]
        try:
            subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError) as exc:
            sys.exit(f"render failed ({' '.join(cmd[:2])} -f {fmt}): {exc}; staging kept for inspection")


def scrub_rendered(out_dir: Path, san: Sanitizer) -> None:
    """The renderer may echo staging paths into its output; run the redactor over it once more."""
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.suffix in (".html", ".md", ".json", ".txt"):
            text = path.read_text(encoding="utf-8", errors="replace")
            cleaned = san.redact(text)
            if cleaned != text:
                path.write_text(cleaned, encoding="utf-8")


def scan_line(line: str, deny: re.Pattern[str] | None, structural: bool = True) -> list[str]:
    """Rules hit by one line; `structural` enables record-shape checks that only mean a leak in JSON files."""
    # A JSON escape glued to code reads as an address ("\n@router.post"); splitting on it hides nothing.
    line = JSON_ESCAPE_RE.sub(" ", line)
    hits: list[str] = []
    if "/Users/" in line:
        hits.append("home-path")
    if USER_FRAGMENT_RE is not None and USER_FRAGMENT_RE.search(line):
        hits.append("user-path-fragment")
    if any(m.group(0).lower() not in ALLOWED_EMAILS for m in EMAIL_RE.finditer(line)):
        hits.append("email")
    hits += [f"secret:{name}" for name, regex in SECRET_RES.items() if regex.search(line)]
    if deny is not None and deny.search(line):
        hits.append("denylist")
    if SR_LEAK_RE.search(line):
        hits.append("marker:reminder")
    if BLOB in line:
        hits.append("marker:request-blob")
    if SESSION_CTX in line:
        hits.append("marker:session-context")
    if structural and INSTRUCTIONS_TYPE_RE.search(line):
        hits.append("marker:instructions-attachment")
    return hits


def scan_paths(target: Path, denylist: list[str]) -> list[str]:
    """Leak gate: one `file:line: rule` entry per hit across every file under target."""
    terms = sorted(denylist, key=len, reverse=True)
    deny = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE) if terms else None
    files = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.is_file())
    hits: list[str] = []
    for path in files:
        rel = path.relative_to(target) if target.is_dir() else path.name
        if path.stat().st_size > MAX_FILE_BYTES:
            hits.append(f"{rel}:0: size>50MB")
        text = path.read_text(encoding="utf-8", errors="replace")
        structural = path.suffix in (".jsonl", ".json")
        for lineno, line in enumerate(text.splitlines(), 1):
            hits += [f"{rel}:{lineno}: {rule}" for rule in scan_line(line, deny, structural)]
    return hits


def report_gate(hits: list[str]) -> int:
    if not hits:
        print("gate: PASS")
        return 0
    print(f"gate: FAIL ({len(hits)} hits)", file=sys.stderr)
    for hit in hits[:200]:
        print(f"  {hit}", file=sys.stderr)
    if len(hits) > 200:
        print(f"  ... {len(hits) - 200} more", file=sys.stderr)
    return 1


def resolve_session(projects_root: Path, session: str | None) -> tuple[Path, str]:
    encoded = re.sub(r"[^A-Za-z0-9]", "-", str(ROOT))
    project_dir = projects_root / encoded if (projects_root / encoded).is_dir() else projects_root
    candidates = sorted(project_dir.glob(f"{session or ''}*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        sys.exit(f"no session JSONL matching {session or '*'!r} in {project_dir}")
    if session and len(candidates) > 1:
        sys.exit(f"session prefix {session!r} is ambiguous in {project_dir}")
    return project_dir, candidates[-1].stem


def reset_staging(staging: Path) -> None:
    if staging.exists():
        if any(staging.iterdir()) and not (staging / STAGING_MARKER).exists():
            sys.exit(f"refusing to clear {staging}: not an export staging dir (no {STAGING_MARKER})")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    (staging / STAGING_MARKER).touch()


def publish(export_dir: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        target = out / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        shutil.move(str(export_dir / name), str(target))


def export(args: argparse.Namespace) -> int:
    denylist = load_denylist(args.denylist)
    project_dir, sid = resolve_session(args.projects_root, args.session)
    print(f"session {sid}")
    staging: Path = args.staging
    reset_staging(staging)
    session_root = staging / "projects" / PROJECT_NAME
    export_dir = staging / "export"
    san, harvest = Sanitizer(denylist=denylist), Harvest()
    stage_session(project_dir, sid, session_root, san, harvest)
    meta = build_meta(harvest, san, sid)
    shutil.copytree(session_root, export_dir / "raw")
    if not args.no_render:
        render(
            session_root / f"{sid}.jsonl", export_dir / "rendered", staging / "ccl-cache.db", args.html_split
        )
        scrub_rendered(export_dir / "rendered", Sanitizer(denylist=denylist))
    else:
        (export_dir / "rendered").mkdir(parents=True)
    (export_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", "utf-8")
    note = (args.out / NOTE_NAME).exists()
    readme_text = readme(meta, note=note, html_parts=len(args.html_split) + 1)
    (export_dir / "README.md").write_text(readme_text, encoding="utf-8")
    records = meta["records"]
    print(f"records kept {records['kept_total']}, dropped {records['dropped_total']}")
    print("redactions " + json.dumps(meta["redactions"]))
    for path in sorted((export_dir / "rendered").glob("*")):
        print(f"rendered {path.name}: {path.stat().st_size / 1_048_576:.1f} MB")
    if report_gate(scan_paths(export_dir, denylist)):
        print(f"export kept in {export_dir}; nothing moved to {args.out}", file=sys.stderr)
        return 1
    publish(export_dir, args.out)
    print(f"published to {args.out}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config_dir = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--scan-only", type=Path, help="only run the leak gate over this file or dir")
    parser.add_argument("--projects-root", type=Path, default=config_dir / "projects")
    parser.add_argument("--session", help="session id or prefix (default: most recently written)")
    parser.add_argument("--denylist", type=Path, default=DEFAULT_DENYLIST)
    parser.add_argument("--no-render", action="store_true", help="skip claude-code-log (raw + meta only)")
    parser.add_argument(
        "--html-split",
        action="append",
        default=[],
        metavar="TIME",
        help="split the HTML render at this time (repeatable, claude-code-log date syntax); pick times "
        "inside idle gaps so no message sits on a boundary",
    )
    args = parser.parse_args()
    if args.scan_only:
        if not args.scan_only.exists():
            sys.exit(f"--scan-only target {args.scan_only} does not exist")
        sys.exit(report_gate(scan_paths(args.scan_only, load_denylist(args.denylist))))
    sys.exit(export(args))


if __name__ == "__main__":
    main()
