"""Process-wide objects, built once by the app lifespan and shared by every request."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from veritrade.config import Settings
from veritrade.errors import LlmUnreachable
from veritrade.fetch import Fetcher, FetchResult, SingleFlightCache
from veritrade.inference.ollama import OllamaClient
from veritrade.inference.types import Endpoint
from veritrade.models_registry import ModelSpec, Registry
from veritrade.prompts.registry import PromptRegistry, PromptSpec
from veritrade.services.cache import CachedResponse, ResultKey, TtlCache

# Installed Ollama tags -> size in bytes (None when Ollama does not report one).
InstalledTags = dict[str, int | None]
_TAGS_KEY = "tags"


def tag_name(name: str) -> str:
    """Ollama lists an untagged model as name:latest."""
    return name if ":" in name else f"{name}:latest"


@dataclass(frozen=True, slots=True)
class Runtime:
    settings: Settings
    fetcher: Fetcher
    ollama: OllamaClient
    registry: Registry
    prompts: Mapping[tuple[Endpoint, str], PromptSpec]
    fetch_cache: SingleFlightCache[str, FetchResult]
    results: TtlCache[ResultKey, CachedResponse]
    tags: SingleFlightCache[str, InstalledTags]

    @classmethod
    def build(
        cls,
        settings: Settings,
        *,
        fetcher: Fetcher,
        ollama: OllamaClient,
        registry: Registry | None = None,
    ) -> Runtime:
        registry = registry or Registry.load(settings)
        prompt_registry = PromptRegistry()
        # Resolved up front so a broken prompt file fails startup instead of a request.
        prompts = {
            (endpoint, spec.name): prompt_registry.get(endpoint, spec.name, kind=spec.kind)
            for spec in registry.models
            for endpoint in spec.endpoints
        }
        return cls(
            settings=settings,
            fetcher=fetcher,
            ollama=ollama,
            registry=registry,
            prompts=prompts,
            fetch_cache=SingleFlightCache(
                max_entries=settings.cache_max_entries,
                ttl_s=settings.fetch_cache_ttl_s,
                error_ttl_s=settings.fetch_error_ttl_s,
                weigh=lambda page: len(page.html),
                max_weight=settings.fetch_cache_max_bytes,
            ),
            results=TtlCache(max_entries=settings.cache_max_entries, ttl_s=settings.result_cache_ttl_s),
            tags=SingleFlightCache(
                max_entries=1, ttl_s=settings.models_cache_ttl_s, error_ttl_s=settings.models_cache_ttl_s
            ),
        )

    def prompt(self, endpoint: Endpoint, spec: ModelSpec) -> PromptSpec:
        return self.prompts[(endpoint, spec.name)]

    async def installed_tags(self) -> InstalledTags:
        """Cached for models_cache_ttl_s, failures included, so a down Ollama is not probed per request."""
        result = await self.tags.get_or_create(_TAGS_KEY, self._list_tags)
        return result.value

    async def _list_tags(self) -> InstalledTags:
        try:
            async with asyncio.timeout(self.settings.ollama_probe_timeout_s):
                entries = await self.ollama.tags()
        except TimeoutError as exc:
            raise LlmUnreachable("Ollama did not list its models in time") from exc
        return {tag_name(str(entry.get("name", ""))): _size(entry) for entry in entries}


def _size(entry: Mapping[str, Any]) -> int | None:
    size = entry.get("size")
    return size if isinstance(size, int) else None
