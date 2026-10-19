from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from veritrade import __version__
from veritrade.api.contracts import HealthResponse, ModelsResponse, SignalRule
from veritrade.models_registry import allowlist

from .conftest import AppFactory
from .fakes import FakeOllama


async def test_models_marks_installed_tags(api: httpx.AsyncClient, ollama: FakeOllama) -> None:
    response = await api.get("/api/models")
    assert response.status_code == 200
    body = ModelsResponse.model_validate(response.json())
    by_name = {model.name: model for model in body.models}
    assert set(by_name) == {"fake-decision:1b", "fake-label:1b", "fake-ordinal:1b", "fake-plumbing:1b"}
    assert by_name["fake-label:1b"].installed is True
    assert by_name["fake-label:1b"].size_bytes == 1_000_000
    assert by_name["fake-ordinal:1b"].installed is False
    assert by_name["fake-ordinal:1b"].size_bytes is None
    assert by_name["fake-plumbing:1b"].evaluated is False
    assert by_name["fake-decision:1b"].endpoints == ["signal"]
    assert by_name["fake-label:1b"].signal_rules == {
        "signal": SignalRule(policy="label-logprob", threshold=0.5),
        "analyze": SignalRule(policy="ordinal", threshold=1.0),
    }
    assert body.defaults == {
        "signal": "fake-label:1b",
        "summary": "fake-label:1b",
        "analyze": "fake-label:1b",
    }
    assert body.parsers[0] == "selectolax"
    assert "trafilatura" in body.extractors
    assert body.scopes == ["lede", "full"]


async def test_models_reuses_the_tag_list(api: httpx.AsyncClient, ollama: FakeOllama) -> None:
    await api.get("/api/models")
    await api.get("/api/models")
    assert ollama.tags.call_count == 1


async def test_models_answers_when_ollama_is_down(api: httpx.AsyncClient, ollama: FakeOllama) -> None:
    ollama.down = True
    response = await api.get("/api/models")
    assert response.status_code == 200
    assert not any(model.installed for model in ModelsResponse.model_validate(response.json()).models)


async def test_models_serves_the_shipped_roster(app_factory: AppFactory) -> None:
    async with app_factory(registry=None) as client:
        body = ModelsResponse.model_validate((await client.get("/api/models")).json())
    assert {model.name for model in body.models} == set(allowlist())


@pytest.mark.parametrize(
    ("version", "down", "status", "reachable"),
    [
        ("0.40.6", False, "ok", True),
        ("0.40.4", False, "ok", True),
        ("0.40.3", False, "degraded", True),
        ("", False, "degraded", True),
        ("0.40.6", True, "degraded", False),
    ],
)
async def test_health(
    api: httpx.AsyncClient, ollama: FakeOllama, version: str, down: bool, status: str, reachable: bool
) -> None:
    ollama.version, ollama.down = version, down
    response = await api.get("/api/health")
    assert response.status_code == 200
    body = HealthResponse.model_validate(response.json())
    assert (body.status, body.ollama_reachable) == (status, reachable)
    assert body.ollama_version == (version if reachable and version else None)
    assert (body.backend, body.version) == ("test-backend", __version__)


async def test_static_export_is_served_after_the_api(app_factory: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>VeriTrade</title>", encoding="utf-8")
    async with app_factory(static_dir=tmp_path) as client:
        index = await client.get("/")
        health = await client.get("/api/health")
    assert index.status_code == 200
    assert "<title>VeriTrade</title>" in index.text
    assert health.headers["content-type"] == "application/json"


@pytest.mark.parametrize(("origins", "allowed"), [("http://localhost:3000", True), ("", False)])
async def test_cors_only_when_configured(app_factory: AppFactory, origins: str, allowed: bool) -> None:
    async with app_factory(cors_origins=origins) as client:
        response = await client.options(
            "/api/signal",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
        )
    assert ("access-control-allow-origin" in response.headers) is allowed
