"""Skeleton fixtures: real publisher DOM, synthetic text (written by scripts/capture_skeletons.py)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

SKELETONS = Path(__file__).resolve().parents[2] / "fixtures" / "skeletons"
BODY_MARKER = re.compile(r"VTB-\d+")
OTHER_MARKER = re.compile(r"VTX-\d+")


@dataclass(frozen=True)
class Skeleton:
    source: str
    html: str
    expected: dict[str, Any]

    @property
    def url(self) -> str:
        return str(self.expected["fixture_url"])

    @property
    def body_markers(self) -> list[str]:
        return list(self.expected["body_markers"])

    def recall(self, text: str) -> float:
        found = set(BODY_MARKER.findall(text))
        return len(found & set(self.body_markers)) / len(self.body_markers)


@cache
def skeletons() -> tuple[Skeleton, ...]:
    out = []
    for expected_path in sorted(SKELETONS.glob("*/*.expected.json")):
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        html = expected_path.with_name(f"{expected['slug']}.html").read_text(encoding="utf-8")
        out.append(Skeleton(expected["source"], html, expected))
    return tuple(out)


@cache
def vocabulary() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((SKELETONS / "vocabulary.json").read_text(encoding="utf-8"))
    return data
