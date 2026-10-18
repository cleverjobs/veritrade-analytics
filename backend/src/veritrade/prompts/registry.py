"""Versioned prompts per (endpoint, model) from TOML; the version string carries a content hash."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from importlib import resources
from importlib.resources.abc import Traversable
from typing import Any

from veritrade.inference.types import Endpoint, ModelKind, Sentiment
from veritrade.models_registry import allowlist

# Output variants each (endpoint, kind) must define, and the JSON fields each variant emits in order.
REQUIRED_VARIANTS: dict[tuple[Endpoint, ModelKind], tuple[str, ...]] = {
    ("signal", "generative"): ("label", "ordinal"),
    ("signal", "decision"): (),
    ("summary", "generative"): ("default",),
    ("analyze", "generative"): ("sentiment-first", "summary-first"),
}
OUTPUT_FIELDS: dict[str, tuple[str, ...]] = {
    "label": ("label",),
    "ordinal": ("score",),
    "default": ("summary",),
    "sentiment-first": ("score", "entity", "summary"),
    "summary-first": ("summary", "entity", "score"),
}
CRITERIA_ORDER: tuple[Sentiment, ...] = ("neutral", "bullish", "bearish")
_FENCE_TAG = re.compile(r"<\s*/?\s*article\b[^>]*>", re.IGNORECASE)
_VERSION_KEYS = frozenset(
    {"extends", "shots", "system", "user_template", "output", "instructions", "criteria"}
)


@dataclass(frozen=True, slots=True)
class FewShot:
    text: str
    label: Sentiment
    score: int
    entity: str
    summary: str

    def answer(self, fields: tuple[str, ...]) -> str:
        return json.dumps({name: getattr(self, name) for name in fields}, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class PromptSpec:
    endpoint: Endpoint
    model: str
    kind: ModelKind
    version_id: str
    version: str
    source: str
    user_template: str
    system: str = ""
    output: Mapping[str, str] = field(default_factory=dict)
    few_shot: tuple[FewShot, ...] = ()
    instructions: str = ""
    criteria: tuple[tuple[Sentiment, str | None], ...] = ()

    def render_input(self, text: str) -> str:
        # Strip fence tags from the article so it cannot close the untrusted-data block early.
        return self.user_template.replace("{text}", _FENCE_TAG.sub("", text))

    def messages(self, text: str, variant: str) -> list[dict[str, str]]:
        if variant not in self.output:
            raise ValueError(f"{self.version} has no output variant {variant!r}")
        out: list[dict[str, str]] = [
            {"role": "system", "content": self.system.replace("{output}", self.output[variant])}
        ]
        for shot in self.few_shot:
            out.append({"role": "user", "content": self.render_input(shot.text)})
            out.append({"role": "assistant", "content": shot.answer(OUTPUT_FIELDS[variant])})
        out.append({"role": "user", "content": self.render_input(text)})
        return out

    def state(self, text: str) -> str:
        return self.render_input(text)

    @property
    def criteria_map(self) -> dict[Sentiment, str | None]:
        return dict(self.criteria)


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def _content_hash(body: Mapping[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def _resolve_version(doc: Mapping[str, Any], version_id: str, source: str) -> dict[str, Any]:
    chain: list[str] = []
    merged: dict[str, Any] = {}
    current: str | None = version_id
    while current is not None:
        if current in chain:
            raise ValueError(f"{source}: extends cycle {' -> '.join([*chain, current])}")
        table = doc.get(current)
        if not isinstance(table, dict):
            raise ValueError(f"{source}: no version table [{current}]")
        unknown = set(table) - _VERSION_KEYS
        if unknown:
            raise ValueError(f"{source}: [{current}] has unknown keys {sorted(unknown)}")
        chain.append(current)
        merged = {**table, **merged}
        current = table.get("extends")
    merged.pop("extends", None)
    return merged


def _criteria(raw: Any, source: str) -> tuple[tuple[Sentiment, str | None], ...]:
    if not isinstance(raw, dict) or set(raw) != set(CRITERIA_ORDER):
        raise ValueError(f"{source}: criteria must define exactly {CRITERIA_ORDER}")
    # TOML has no null; an empty description sends a bare label.
    return tuple((s, raw[s] or None) for s in CRITERIA_ORDER)


class PromptRegistry:
    def __init__(self, root: Traversable | None = None) -> None:
        self._root = root or resources.files("veritrade.prompts").joinpath("data")
        self._docs: dict[str, dict[str, Any]] = {}

    def _source(self, endpoint: Endpoint, model: str, kind: ModelKind) -> Traversable:
        folder = self._root.joinpath(endpoint)
        for name in (f"{model_slug(model)}.toml", f"default-{kind}.toml", "default.toml"):
            candidate = folder.joinpath(name)
            if candidate.is_file():
                return candidate
        raise LookupError(f"No prompt file for /{endpoint} and {model} under prompts/data/{endpoint}/")

    def _doc(self, source: Traversable, label: str) -> dict[str, Any]:
        if label not in self._docs:
            self._docs[label] = tomllib.loads(source.read_text(encoding="utf-8"))
        return self._docs[label]

    def versions(self, endpoint: Endpoint, model: str, *, kind: ModelKind | None = None) -> list[str]:
        kind = kind or _kind_of(model)
        source = self._source(endpoint, model, kind)
        doc = self._doc(source, f"{endpoint}/{source.name}")
        return [key for key, value in doc.items() if isinstance(value, dict)]

    def get(
        self, endpoint: Endpoint, model: str, *, kind: ModelKind | None = None, version: str | None = None
    ) -> PromptSpec:
        kind = kind or _kind_of(model)
        source = self._source(endpoint, model, kind)
        label = f"{endpoint}/{source.name}"
        doc = self._doc(source, label)
        version_id = version or doc.get("active")
        if not isinstance(version_id, str):
            raise ValueError(f"{label}: missing `active` version")
        body = _resolve_version(doc, version_id, label)
        where = f"{label} [{version_id}]"
        user_template = str(body.get("user_template", ""))
        if "{text}" not in user_template:
            raise ValueError(f"{where}: user_template needs a {{text}} placeholder")
        system, instructions = str(body.get("system", "")), str(body.get("instructions", ""))
        output: dict[str, str] = {}
        few_shot: tuple[FewShot, ...] = ()
        criteria: tuple[tuple[Sentiment, str | None], ...] = ()
        if kind == "decision":
            criteria = _criteria(body.get("criteria"), where)
            if not instructions:
                raise ValueError(f"{where}: decision prompts need instructions")
        else:
            output = {str(k): str(v) for k, v in body.get("output", {}).items()}
            missing = [v for v in REQUIRED_VARIANTS[(endpoint, kind)] if v not in output]
            if missing or "{output}" not in system:
                raise ValueError(f"{where}: needs a system with {{output}} and output variants {missing}")
            shots = int(body.get("shots", 0))
            pool = [FewShot(**shot) for shot in doc.get("few_shot", [])]
            if shots > len(pool):
                raise ValueError(f"{where}: shots={shots} but the pool has {len(pool)}")
            few_shot = tuple(pool[:shots])
        digest = _content_hash(
            {
                "kind": kind,
                "user_template": user_template,
                "system": system,
                "output": output,
                "few_shot": [asdict(s) for s in few_shot],
                "instructions": instructions,
                "criteria": criteria,
            }
        )
        return PromptSpec(
            endpoint=endpoint,
            model=model,
            kind=kind,
            version_id=version_id,
            version=f"{endpoint}/{model}/{version_id}@{digest}",
            source=label,
            user_template=user_template,
            system=system,
            output=output,
            few_shot=few_shot,
            instructions=instructions,
            criteria=criteria,
        )


def _kind_of(model: str) -> ModelKind:
    spec = allowlist().get(model)
    return spec.kind if spec else "generative"


_default_registry = PromptRegistry()


def get(
    endpoint: Endpoint, model: str, *, kind: ModelKind | None = None, version: str | None = None
) -> PromptSpec:
    return _default_registry.get(endpoint, model, kind=kind, version=version)
