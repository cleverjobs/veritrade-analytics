"""/models and /health: both answer 200 whatever state Ollama is in."""

from __future__ import annotations

import asyncio
import re
from typing import get_args

from veritrade import __version__
from veritrade.api.contracts import HealthResponse, ModelInfo, ModelsResponse, SignalRule
from veritrade.errors import VtError
from veritrade.extract.types import EXTRACTORS, PARSERS, Scope
from veritrade.inference.confidence import OrdinalPolicy, policy_for
from veritrade.inference.types import Endpoint
from veritrade.models_registry import ModelSpec
from veritrade.services.runtime import InstalledTags, Runtime, tag_name

MIN_OLLAMA_VERSION = (0, 40, 4)
SIGNAL_ENDPOINTS: tuple[Endpoint, ...] = ("signal", "analyze")
_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def parse_version(text: str) -> tuple[int, ...] | None:
    match = _VERSION.match(text.strip().removeprefix("v"))
    return tuple(int(part) for part in match.groups()) if match else None


def signal_rules(spec: ModelSpec) -> dict[Endpoint, SignalRule]:
    rules: dict[Endpoint, SignalRule] = {}
    for endpoint in SIGNAL_ENDPOINTS:
        if not spec.serves(endpoint):
            continue
        policy = policy_for(spec, endpoint=endpoint)
        if isinstance(policy, OrdinalPolicy):
            rules[endpoint] = SignalRule(policy=policy.name, threshold=float(policy.t))
        else:
            rules[endpoint] = SignalRule(
                policy=policy.name, threshold=policy.min_p, directional_tau=policy.directional_tau
            )
    return rules


async def list_models(runtime: Runtime) -> ModelsResponse:
    installed: InstalledTags | None
    try:
        installed = await runtime.installed_tags()
    except VtError:
        installed = None
    models = []
    for spec in runtime.registry.models:
        tag = tag_name(spec.name)
        models.append(
            ModelInfo(
                name=spec.name,
                kind=spec.kind,
                endpoints=list(spec.endpoints),
                installed=installed is not None and tag in installed,
                size_bytes=installed.get(tag) if installed else None,
                evaluated=spec.evaluated,
                signal_rules=signal_rules(spec),
            )
        )
    return ModelsResponse(
        models=models,
        defaults=runtime.registry.defaults,
        parsers=list(PARSERS),
        extractors=list(EXTRACTORS),
        scopes=list(get_args(Scope)),
    )


async def check_health(runtime: Runtime) -> HealthResponse:
    version: str | None
    try:
        async with asyncio.timeout(runtime.settings.ollama_probe_timeout_s):
            version = await runtime.ollama.version()
    except (VtError, TimeoutError):
        version = None
    parsed = parse_version(version) if version else None
    return HealthResponse(
        status="ok" if parsed is not None and parsed >= MIN_OLLAMA_VERSION else "degraded",
        ollama_reachable=version is not None,
        ollama_version=version or None,
        backend=runtime.settings.ollama_backend_label,
        version=__version__,
    )
