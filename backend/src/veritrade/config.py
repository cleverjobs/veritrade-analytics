from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VT_", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"

    ollama_base_url: str = Field(
        "http://localhost:11434",
        validation_alias=AliasChoices("VT_OLLAMA_BASE_URL", "OLLAMA_BASE_URL"),
    )
    ollama_backend_label: str = "native-metal"
    ollama_keep_alive: str = "30m"
    inference_timeout_s: float = 90.0
    # Comma-separated restriction of the models.toml allowlist; empty keeps the full allowlist.
    models: str = ""
    default_model_signal: str | None = None
    default_model_summary: str | None = None
    default_model_analyze: str | None = None

    sec_user_agent: str = Field(
        "VeriTrade Analytics PoC kris.dimitrov92@gmail.com",
        validation_alias=AliasChoices("VT_SEC_USER_AGENT", "SEC_USER_AGENT"),
    )
    fetch_connect_timeout_s: float = 3.0
    fetch_deadline_s: float = 8.0
    fetch_max_bytes: int = 5_000_000
    fetch_max_redirects: int = 5
    fetch_cache_ttl_s: float = 600.0
    fetch_error_ttl_s: float = 60.0
    result_cache_ttl_s: float = 3600.0
    cache_max_entries: int = 512

    cors_origins: str = ""
    static_dir: Path | None = None
    # Lets e2e reach the fixture container; refused outside VT_ENV=test so prod keeps the full SSRF guard.
    ssrf_test_allow_hosts: str = ""

    @model_validator(mode="after")
    def _guard_test_only_settings(self) -> Settings:
        if self.ssrf_test_allow_hosts and self.env != "test":
            raise ValueError("VT_SSRF_TEST_ALLOW_HOSTS is only allowed when VT_ENV=test")
        return self

    @property
    def model_restriction(self) -> list[str]:
        return _csv(self.models)

    @property
    def cors_origin_list(self) -> list[str]:
        return _csv(self.cors_origins)

    @property
    def ssrf_allow_hosts(self) -> frozenset[str]:
        return frozenset(host.lower() for host in _csv(self.ssrf_test_allow_hosts))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
