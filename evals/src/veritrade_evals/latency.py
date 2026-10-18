"""Latency suite (R7): `uv run python -m veritrade_evals latency {preflight,run,docker-cpu,report}`.
In-process app over the stored dev-article HTML (local fetcher, no_cache, concurrency 1), timed by the
server from request receipt; raw samples hold article ids and numbers only, never text."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TextIO

import httpx
import numpy as np

from veritrade.api.app import create_app
from veritrade.config import Settings, get_settings
from veritrade.errors import UpstreamHttp, VtError
from veritrade.extract.types import Scope
from veritrade.fetch import FetchResult, normalize_url
from veritrade.inference.ollama import OllamaClient
from veritrade.inference.types import Endpoint
from veritrade.models_registry import allowlist
from veritrade_evals.articles import ArticleExample, ArticlePaths, load_articles, read_fetched
from veritrade_evals.paths import DATA_DIR, REPO_ROOT, RESULTS_DIR

LATENCY_DIR = RESULTS_DIR / "latency"
RAW_DIR = LATENCY_DIR / "raw"
PREFLIGHT_JSON = LATENCY_DIR / "preflight.json"
LATENCY_JSON = LATENCY_DIR / "latency.json"
LATENCY_MD = LATENCY_DIR / "latency.md"
SOURCE_PROBE = DATA_DIR / "source_probe.json"
BUNDLED_ENV = REPO_ROOT / "deploy" / "bundled.env"

COLD_N = 3
WARM_N = 30
DOCKER_WARM_N = 10
HTTP_TIMEOUT_S = 600.0
UNLOAD_WAIT_S = 30.0
NATIVE_BACKEND = "native-metal"
DOCKER_BACKEND = "docker-cpu"
# Fields a raw sample may carry; anything else (titles, summaries, entities) is dropped before writing.
SAMPLE_FIELDS = (
    "session", "cell", "endpoint", "model", "scope", "streaming", "backend", "phase", "article", "ok",
    "status", "error", "total_ms", "fetch_ms", "consent_ms", "parse_ms", "extract_ms", "wall_ms", "ttft_ms",
    "signal_ms", "load_ms", "prompt_eval_ms", "eval_ms", "prompt_tokens", "eval_tokens", "client_ms",
    "input_chars", "truncated", "deltas", "order_ok", "at",
)  # fmt: skip
STAGES = ("fetch_ms", "parse_ms", "extract_ms", "wall_ms", "load_ms", "prompt_eval_ms", "eval_ms")

Phase = Literal["cold", "warm"]
Sample = dict[str, Any]


class LatencyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Budgets:
    """R7: native, warm, excluding fetch."""

    signal_lede_p50_ms: float = 500.0
    ttft_p50_ms: float = 1500.0
    complete_p50_ms: float = 4000.0


BUDGETS = Budgets()


@dataclass(frozen=True, slots=True)
class Cell:
    endpoint: Endpoint
    model: str
    scope: Scope
    streaming: bool

    @property
    def key(self) -> str:
        return f"{self.endpoint}/{self.model}/{self.scope}/{'stream' if self.streaming else 'json'}"

    @property
    def path(self) -> str:
        return f"/api/{self.endpoint}" + ("/stream" if self.streaming else "")

    @classmethod
    def from_sample(cls, sample: Mapping[str, Any]) -> Cell:
        return cls(sample["endpoint"], sample["model"], sample["scope"], bool(sample["streaming"]))

    def body(self, url: str) -> dict[str, Any]:
        body: dict[str, Any] = {"url": url, "model": self.model, "no_cache": True}
        if self.endpoint != "summary":
            body["scope"] = self.scope
        return body


SIGNAL_MODELS = ("qwen3.5:0.8b-q8_0", "clef-flash:9b-q8_0", "nimble:9b-q4_K_M")
SUMMARY_MODELS = ("qwen3.5:2b-q4_K_M", "qwen3.5:4b-q4_K_M")
ANALYZE_MODELS = ("ministral-3:3b", "qwen3:4b-instruct-2507-q4_K_M", "qwen3.5:4b-q4_K_M")


def matrix() -> list[Cell]:
    cells = [Cell("signal", m, s, False) for m in SIGNAL_MODELS for s in ("lede", "full")]
    cells += [Cell("summary", m, "full", st) for m in SUMMARY_MODELS for st in (True, False)]
    cells += [Cell("analyze", m, "full", st) for m in ANALYZE_MODELS for st in (True, False)]
    return cells


def by_model(cells: Sequence[Cell]) -> list[tuple[str, list[Cell]]]:
    """Grouped so each model loads once; first-appearance order keeps the fast models first."""
    groups: dict[str, list[Cell]] = {}
    for cell in cells:
        groups.setdefault(cell.model, []).append(cell)
    return list(groups.items())


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --- preflight -------------------------------------------------------------------------------


def _cmd(args: Sequence[str], timeout: float = 20.0) -> str | None:
    try:
        done = subprocess.run(list(args), capture_output=True, text=True, timeout=timeout, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def running_containers() -> list[str] | None:
    """Names of running containers; None when Docker is absent or its daemon is down (nothing can run)."""
    out = _cmd(["docker", "ps", "--format", "{{.Names}}"])
    return None if out is None else [line for line in out.splitlines() if line.strip()]


_ENV_ASSIGN = re.compile(r"\b(OLLAMA_[A-Z_]+)=\"?([^\"\s]*)\"?")


def ollama_service_env() -> dict[str, Any]:
    """OLLAMA_* settings of the brew service: its launch command, then the live process environment."""
    raw = _cmd(["brew", "services", "info", "ollama", "--json"])
    info: dict[str, Any] = {}
    if raw:
        try:
            entries = json.loads(raw)
            info = entries[0] if isinstance(entries, list) and entries else {}
        except json.JSONDecodeError:
            info = {}
    env = dict(_ENV_ASSIGN.findall(str(info.get("command") or "")))
    pid = info.get("pid")
    if isinstance(pid, int):
        proc = _cmd(["ps", "eww", "-o", "command=", "-p", str(pid)]) or ""
        env.update(_ENV_ASSIGN.findall(proc))
    return {
        "service": info.get("service_name"),
        "running": info.get("running"),
        "env": dict(sorted(env.items())),
    }


def _sysctl(key: str) -> str | None:
    out = _cmd(["sysctl", "-n", key])
    return out.strip() if out else None


def memory_state() -> dict[str, Any]:
    state: dict[str, Any] = {"freePct": None, "swapUsedMb": None, "swapTotalMb": None}
    out = _cmd(["memory_pressure", "-Q"]) or ""
    if m := re.search(r"free percentage:\s*(\d+)%", out):
        state["freePct"] = int(m.group(1))
    if swap := _sysctl("vm.swapusage"):
        total = re.search(r"total = ([\d.]+)M", swap)
        used = re.search(r"used = ([\d.]+)M", swap)
        state["swapTotalMb"] = float(total.group(1)) if total else None
        state["swapUsedMb"] = float(used.group(1)) if used else None
    return state


def machine() -> dict[str, Any]:
    mem = _sysctl("hw.memsize")
    return {
        "chip": _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown",
        "cpus": os.cpu_count(),
        "ramGb": round(int(mem) / 2**30) if mem and mem.isdigit() else None,
        "os": platform.platform(),
        "python": platform.python_version(),
    }


async def ollama_state(base_url: str) -> dict[str, Any]:
    """Version, resident models and installed sizes; names and numbers only."""
    async with OllamaClient(base_url, timeout=10.0) as client:
        try:
            version = await client.version()
            resident = sorted(str(e.get("name") or e.get("model")) for e in await client.ps())
            sizes = {
                str(e.get("name")): int(e["size"])
                for e in await client.tags()
                if isinstance(e.get("size"), int)
            }
        except VtError as exc:
            return {"reachable": False, "error": exc.code.value}
    return {
        "reachable": True,
        "version": version,
        "resident": resident,
        "modelSizes": dict(sorted(sizes.items())),
    }


def preflight_checks(
    containers: list[str] | None,
    ollama: Mapping[str, Any],
    memory: Mapping[str, Any],
    load1: float,
    cpus: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Hard checks fail the preflight; warnings only go on record."""
    checks = [
        {
            "name": "no running Docker containers",
            "ok": not containers,
            "detail": "docker not reachable" if containers is None else f"{len(containers)} running",
        },
        {
            "name": "Ollama reachable",
            "ok": bool(ollama.get("reachable")),
            "detail": ollama.get("version") or ollama.get("error"),
        },
    ]
    warnings = []
    if load1 > cpus / 2:
        warnings.append(f"1-min load average {load1:.1f} is above half the {cpus} cores")
    if ollama.get("resident"):
        warnings.append(f"{len(ollama['resident'])} model(s) resident before the run; `run` unloads them")
    used, total = memory.get("swapUsedMb"), memory.get("swapTotalMb")
    if used and total and used > 0.25 * total:
        warnings.append(f"swap in use: {used:.0f} of {total:.0f} MB")
    return checks, warnings


def new_session() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def preflight(base_url: str, *, resume: bool, log: TextIO = sys.stderr) -> dict[str, Any]:
    previous = read_json(PREFLIGHT_JSON)
    session = previous["session"] if resume and previous else new_session()
    containers = running_containers()
    if containers:
        # Names go to the terminal only: they belong to other projects and stay out of the results.
        print(f"running containers ({len(containers)}): {', '.join(containers)}", file=log)
    ollama = asyncio.run(ollama_state(base_url))
    memory = memory_state()
    host = machine()
    load = os.getloadavg()
    checks, warnings = preflight_checks(containers, ollama, memory, load[0], host["cpus"] or 1)
    settings = get_settings()
    return {
        "version": 1,
        "session": session,
        "ok": all(c["ok"] for c in checks),
        "at": _now(),
        "checks": checks,
        "warnings": warnings,
        "environment": {
            **host,
            "ollamaVersion": ollama.get("version"),
            "backend": settings.ollama_backend_label,
            "ollamaBaseUrl": base_url,
            "ollamaService": ollama_service_env(),
            "ollamaResident": ollama.get("resident", []),
            "modelSizes": ollama.get("modelSizes", {}),
            "loadAverage": [round(x, 2) for x in load],
            "memory": memory,
            "dockerRunning": None if containers is None else len(containers),
        },
    }


# --- local fetcher ---------------------------------------------------------------------------


# Synthetic warm-up page: measuring the warm-up's own article again would hit Ollama's prompt cache.
WARMUP_URL = "https://latency.local/warm-up"
WARMUP_PARAGRAPHS = (
    "Ostrava Logistics said on Monday that it has appointed Jana Novak as chief financial officer, effective "
    "January 1, succeeding Petr Dvorak, who will retire after eleven years with the freight company.",
    "Novak joins from a regional rail operator, where she led finance and procurement. The company said the "
    "change is part of a planned succession and that its full-year outlook is unchanged.",
    "Ostrava Logistics runs warehouses and trucking routes across Central Europe and employs about four "
    "thousand people. It reports third-quarter results next month and holds its investor day in spring.",
    "The board thanked Dvorak for steering the group through two acquisitions and a refinancing of its bank "
    "debt, and said he will advise the new finance chief until the end of March.",
)
WARMUP_HTML = (
    "<html><head><title>Ostrava Logistics names a new finance chief</title></head><body><article>"
    "<h1>Ostrava Logistics names a new finance chief</h1>"
    + "".join(f"<p>{p}</p>" for p in WARMUP_PARAGRAPHS)
    + "</article></body></html>"
)


class LocalFetcher:
    """Serves stored article HTML from memory, so network variance stays out of the timings."""

    def __init__(self, pages: Mapping[str, tuple[str, str]]) -> None:
        self._pages = {normalize_url(url): page for url, page in pages.items()}
        self._pages.setdefault(normalize_url(WARMUP_URL), (WARMUP_URL, WARMUP_HTML))

    async def get(self, url: str) -> FetchResult:
        t0 = time.perf_counter()
        page = self._pages.get(normalize_url(url))
        if page is None:
            raise UpstreamHttp(f"{url} is not in the local latency set", upstream_status=404, retryable=False)
        final_url, html = page
        return FetchResult(
            requested_url=url,
            final_url=final_url,
            status=200,
            html=html,
            content_type="text/html; charset=utf-8",
            fetch_ms=(time.perf_counter() - t0) * 1000,
        )

    async def aclose(self) -> None:
        return None


def local_pages(examples: Sequence[ArticleExample], paths: ArticlePaths) -> dict[str, tuple[str, str]]:
    pages = {}
    for example in examples:
        fetched = read_fetched(paths, example.id)
        final_url = fetched.final_url if fetched and fetched.final_url else example.url
        pages[example.url] = (final_url, example.html_path.read_text(encoding="utf-8"))
    return pages


# --- one request -----------------------------------------------------------------------------


def sse_events(text: str) -> list[tuple[str, dict[str, Any]]]:
    """(event, data) pairs from a text/event-stream body; comments and unnamed frames are skipped."""
    events = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        name, data = None, []
        for line in frame.splitlines():
            if line.startswith("event:"):
                name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data.append(line.removeprefix("data:").lstrip())
        if name and data:
            events.append((name, json.loads("\n".join(data))))
    return events


def _error_code(text: str) -> str | None:
    try:
        code = json.loads(text).get("error", {}).get("code")
    except (json.JSONDecodeError, AttributeError):
        return None
    return str(code) if code else None


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def parse_reply(cell: Cell, status: int, text: str) -> dict[str, Any]:
    """Server-side timings of one response; streams also yield TTFT, time to signal and event order."""
    if status != 200:
        return {"ok": False, "error": _error_code(text) or f"http_{status}"}
    signal_ms: float | None = None
    order_ok: bool | None = None
    deltas: int | None = None
    try:
        if cell.streaming:
            events = sse_events(text)
            names = [name for name, _ in events]
            if "error" in names:
                payload = events[names.index("error")][1]
                return {"ok": False, "error": str(payload.get("error", {}).get("code") or "stream_error")}
            if names.count("done") != 1:
                return {"ok": False, "error": "no_done_event"}
            result = events[names.index("done")][1]["result"]
            deltas = names.count("summary_delta")
            if cell.endpoint == "analyze":
                order_ok = names.count("sentiment") == 1 and names.index("sentiment") < names.index("done")
                if "sentiment" in names:
                    signal_ms = _num(events[names.index("sentiment")][1].get("elapsed_ms"))
            else:
                order_ok = "sentiment" not in names
            order_ok = order_ok and names[0] == "meta" and names[-1] == "done"
        else:
            result = json.loads(text)
        timings = result["timings"]
    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        return {"ok": False, "error": "unparsable_response"}
    inference = timings.get("inference") or {}
    total = _num(timings.get("total_ms"))
    if not cell.streaming and cell.endpoint != "summary":
        signal_ms = total
    return {
        "ok": True,
        "error": None,
        "total_ms": total,
        "fetch_ms": _num(timings.get("fetch_ms")),
        "consent_ms": _num(timings.get("consent_ms")),
        "parse_ms": _num(timings.get("parse_ms")),
        "extract_ms": _num(timings.get("extract_ms")),
        "wall_ms": _num(inference.get("wall_ms")),
        "ttft_ms": _num(inference.get("ttft_ms")),
        "signal_ms": signal_ms,
        "load_ms": _num(inference.get("load_ms")),
        "prompt_eval_ms": _num(inference.get("prompt_eval_ms")),
        "eval_ms": _num(inference.get("eval_ms")),
        "prompt_tokens": inference.get("prompt_tokens"),
        "eval_tokens": inference.get("eval_tokens"),
        "input_chars": (result.get("meta") or {}).get("input_chars"),
        "truncated": result.get("truncated"),
        "deltas": deltas,
        "order_ok": order_ok,
    }


def make_sample(cell: Cell, **values: Any) -> Sample:
    base = {
        "cell": cell.key,
        "endpoint": cell.endpoint,
        "model": cell.model,
        "scope": cell.scope,
        "streaming": cell.streaming,
        "at": _now(),
    }
    merged = {**base, **values}
    return {k: merged.get(k) for k in SAMPLE_FIELDS}


async def post(http: httpx.AsyncClient, cell: Cell, url: str) -> tuple[dict[str, Any], int, float]:
    t0 = time.perf_counter()
    try:
        response = await http.post(cell.path, json=cell.body(url))
    except httpx.HTTPError as exc:
        return {"ok": False, "error": f"client_{type(exc).__name__}"}, 0, (time.perf_counter() - t0) * 1000
    client_ms = (time.perf_counter() - t0) * 1000
    return parse_reply(cell, response.status_code, response.text), response.status_code, client_ms


async def measure(
    http: httpx.AsyncClient, cell: Cell, article: ArticleExample, *, phase: Phase, session: str, backend: str
) -> Sample:
    parsed, status, client_ms = await post(http, cell, article.url)
    return make_sample(
        cell,
        session=session,
        backend=backend,
        phase=phase,
        article=article.id,
        status=status,
        client_ms=client_ms,
        **parsed,
    )


# --- raw store -------------------------------------------------------------------------------


class RawStore:
    """Append-only JSONL per session; a recorded (cell, phase, article) is never measured again."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._done: set[tuple[str, str, str]] = set()
        for sample in self.samples():
            self._done.add((sample["cell"], sample["phase"], sample["article"]))

    def samples(self) -> list[Sample]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def has(self, cell: Cell, phase: Phase, article: str) -> bool:
        return (cell.key, phase, article) in self._done

    def append(self, sample: Sample) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({k: sample.get(k) for k in SAMPLE_FIELDS}, sort_keys=True) + "\n")
        self._done.add((sample["cell"], sample["phase"], sample["article"]))


def raw_path(session: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"{session}.jsonl"


# --- suite -----------------------------------------------------------------------------------


async def unload(ollama: OllamaClient, model: str, wait_s: float = UNLOAD_WAIT_S) -> None:
    """Returns once Ollama no longer lists the model as resident, so the next call is a true cold load."""
    await ollama.unload(model)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if model not in {str(e.get("name") or e.get("model")) for e in await ollama.ps()}:
            return
        await asyncio.sleep(0.2)
    raise LatencyError(f"{model} is still resident {wait_s:.0f} s after the unload request")


async def unload_others(ollama: OllamaClient, model: str, log: TextIO) -> None:
    for entry in await ollama.ps():
        name = str(entry.get("name") or entry.get("model") or "")
        if name and name != model:
            print(f"unloading {name}", file=log, flush=True)
            await unload(ollama, name)


def _p50(samples: Sequence[Sample], key: str) -> str:
    values = [s[key] for s in samples if s.get("ok") and isinstance(s.get(key), int | float)]
    return f"{np.median(values):.0f} ms" if values else "-"


async def run_suite(
    http: httpx.AsyncClient,
    cells: Sequence[Cell],
    articles: Sequence[ArticleExample],
    store: RawStore,
    *,
    session: str,
    backend: str,
    cold_n: int,
    ollama: OllamaClient | None,
    shared: bool,
    warmup_url: str = WARMUP_URL,
    log: TextIO = sys.stderr,
) -> int:
    """Cold samples (model unloaded first), then a reload through one warm-up on an unmeasured page and the
    warm samples, cell by cell; returns the number of new samples. An unreachable Ollama aborts the run."""
    cold = list(articles[:cold_n]) if ollama is not None else []
    added = 0
    for model, group in by_model(cells):
        if ollama is not None and not shared:
            await unload_others(ollama, model, log)
        for cell in group:
            new: list[Sample] = []
            for article in cold:
                if store.has(cell, "cold", article.id):
                    continue
                assert ollama is not None
                await unload(ollama, model)
                new.append(await measure(http, cell, article, phase="cold", session=session, backend=backend))
                store.append(new[-1])
            todo = [a for a in articles if not store.has(cell, "warm", a.id)]
            if todo:
                # A reload empties the prompt cache, so no warm sample reuses a prompt an earlier cell sent.
                if ollama is not None:
                    await unload(ollama, model)
                warmed, _, _ = await post(http, cell, warmup_url)
                if not warmed["ok"]:
                    print(f"[{cell.key}] warm-up failed: {warmed['error']}", file=log, flush=True)
            for article in todo:
                new.append(await measure(http, cell, article, phase="warm", session=session, backend=backend))
                store.append(new[-1])
                if new[-1]["error"] == "llm_unreachable":
                    raise LatencyError(f"{cell.key}: Ollama is unreachable; fix it and re-run to resume")
            added += len(new)
            errors = Counter(str(s["error"]) for s in new if not s["ok"])
            warm = [s for s in new if s["phase"] == "warm"]
            print(
                f"[{cell.key}] new {len(new)} (cold {len(new) - len(warm)}, warm {len(warm)}), "
                f"warm p50 {_p50(warm, 'total_ms')}, errors {dict(errors) or 0}",
                file=log,
                flush=True,
            )
    return added


def select_cells(
    cells: Sequence[Cell], models: Sequence[str] | None, endpoints: Sequence[str] | None
) -> list[Cell]:
    unknown = set(models or ()) - {c.model for c in cells}
    if unknown:
        raise LatencyError(f"not in the latency matrix: {sorted(unknown)}")
    return [
        c for c in cells if (not models or c.model in models) and (not endpoints or c.endpoint in endpoints)
    ]


def serving_settings(base_url: str | None) -> Settings:
    """The full allowlist and no static files, whatever the shell exports."""
    update: dict[str, Any] = {
        "models": "",
        "default_model_signal": None,
        "default_model_summary": None,
        "default_model_analyze": None,
        "static_dir": None,
        "cors_origins": "",
    }
    if base_url:
        update["ollama_base_url"] = base_url
    return get_settings().model_copy(update=update)


async def run_native(
    cells: Sequence[Cell],
    articles: Sequence[ArticleExample],
    pages: Mapping[str, tuple[str, str]],
    *,
    session: str,
    settings: Settings,
    cold_n: int,
    shared: bool,
    raw_dir: Path = RAW_DIR,
    log: TextIO = sys.stderr,
) -> int:
    store = RawStore(raw_path(session, raw_dir))
    ollama = OllamaClient.from_settings(settings)
    app = create_app(settings, fetcher=LocalFetcher(pages), ollama=ollama)
    async with ollama, app.router.lifespan_context(app):
        installed = {str(e.get("name")) for e in await ollama.tags()}
        missing = sorted({c.model for c in cells} - installed)
        if missing:
            print(f"skipping models that are not installed: {', '.join(missing)}", file=log)
        runnable = [c for c in cells if c.model not in missing]
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://latency.local", timeout=HTTP_TIMEOUT_S
        ) as http:
            return await run_suite(
                http,
                runnable,
                articles,
                store,
                session=session,
                backend=settings.ollama_backend_label,
                cold_n=cold_n,
                ollama=ollama,
                shared=shared,
                log=log,
            )


# --- aggregation -----------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float | None:
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), q)), 1) if values else None


def _excl(sample: Sample, key: str) -> float | None:
    value = sample.get(key)
    if not isinstance(value, int | float):
        return None
    return float(value) - float(sample.get("fetch_ms") or 0.0) - float(sample.get("consent_ms") or 0.0)


def _dist(values: Iterable[float | None]) -> dict[str, float | None] | None:
    vals = [v for v in values if v is not None]
    return {"p50": percentile(vals, 50), "p95": percentile(vals, 95)} if vals else None


def aggregate(samples: Sequence[Sample]) -> dict[str, Any]:
    """Totals, TTFT and time to signal exclude fetch and consent (the R7 definition); stages are p50s."""
    ok = [s for s in samples if s.get("ok")]
    return {
        "n": len(ok),
        "errors": len(samples) - len(ok),
        "errorCodes": dict(Counter(str(s.get("error")) for s in samples if not s.get("ok"))),
        "totalMs": _dist(_excl(s, "total_ms") for s in ok),
        "totalInclFetchMs": _dist(_num(s.get("total_ms")) for s in ok),
        "ttftMs": _dist(_excl(s, "ttft_ms") for s in ok),
        "signalMs": _dist(_excl(s, "signal_ms") for s in ok),
        "clientMs": _dist(_num(s.get("client_ms")) for s in ok),
        "stagesP50Ms": {
            k.removesuffix("_ms"): percentile([float(s[k]) for s in ok if _num(s.get(k)) is not None], 50)
            for k in STAGES
        },
        "tokensP50": {
            k: percentile([float(s[f"{k}_tokens"]) for s in ok if _num(s.get(f"{k}_tokens")) is not None], 50)
            for k in ("prompt", "eval")
        },
        "truncated": sum(bool(s.get("truncated")) for s in ok),
        "orderViolations": sum(s.get("order_ok") is False for s in ok),
    }


def budget_status(cell: Cell, warm: Mapping[str, Any], budgets: Budgets = BUDGETS) -> dict[str, Any]:
    """R7 per row: /signal lede p50; /summary and /analyze complete p50, plus TTFT p50 on streams.
    `utilisation` is the worst metric over its limit, the latency axis for a Pareto knee."""
    if not warm.get("n"):
        return {"status": "unmeasured", "checks": [], "utilisation": None}
    checks: list[dict[str, Any]] = []
    if cell.endpoint == "signal":
        if cell.scope != "lede":
            return {"status": "n/a", "checks": [], "utilisation": None}
        checks.append(_check("p50 excl. fetch", warm["totalMs"], budgets.signal_lede_p50_ms))
    else:
        checks.append(_check("complete p50", warm["totalMs"], budgets.complete_p50_ms))
        if cell.streaming:
            checks.append(_check("TTFT p50", warm["ttftMs"], budgets.ttft_p50_ms))
    if any(c["value"] is None for c in checks):
        return {"status": "unmeasured", "checks": checks, "utilisation": None}
    return {
        "status": "met" if all(c["met"] for c in checks) else "missed",
        "checks": checks,
        "utilisation": round(max(c["value"] / c["limit"] for c in checks), 3),
    }


def _check(metric: str, dist: Mapping[str, Any] | None, limit: float) -> dict[str, Any]:
    value = dist.get("p50") if dist else None
    return {"metric": metric, "value": value, "limit": limit, "met": value is not None and value <= limit}


def fetch_reference(path: Path = SOURCE_PROBE) -> dict[str, Any] | None:
    """Live fetch cost from the source probe, reported beside the suite because the suite fetches locally."""
    probe = read_json(path)
    if not probe:
        return None
    per_source = {
        name: entry["summary"]["fetch_ms_median"]
        for name, entry in probe.get("sources", {}).items()
        if entry.get("summary", {}).get("verdict") == "usable"
        and isinstance(entry["summary"].get("fetch_ms_median"), int | float)
    }
    values = sorted(per_source.values())
    return {
        "source": _rel(path),
        "vantage": probe.get("vantage"),
        "generatedAt": probe.get("generated_at"),
        "perSourceMedianMs": dict(sorted(per_source.items())),
        "medianMs": percentile(values, 50),
        "maxMs": max(values) if values else None,
    }


def cell_report(
    cell: Cell, samples: Sequence[Sample], backend: str, targets: tuple[int, int]
) -> dict[str, Any]:
    mine = [s for s in samples if s["cell"] == cell.key]
    cold = aggregate([s for s in mine if s["phase"] == "cold"])
    warm = aggregate([s for s in mine if s["phase"] == "warm"])
    return {
        "key": cell.key,
        "endpoint": cell.endpoint,
        "model": cell.model,
        "scope": cell.scope,
        "streaming": cell.streaming,
        "backend": backend,
        "targets": {"cold": targets[0], "warm": targets[1]},
        "partial": cold["n"] < targets[0] or warm["n"] < targets[1],
        "cold": cold,
        "warm": warm,
        "budget": budget_status(cell, warm) if backend == NATIVE_BACKEND else {"status": "n/a"},
    }


def docker_sessions(raw_dir: Path = RAW_DIR) -> list[Path]:
    return sorted(raw_dir.glob(f"{DOCKER_BACKEND}-*.jsonl"))


def build_report(
    preflight_payload: Mapping[str, Any] | None,
    native: Sequence[Sample],
    docker: Sequence[Sample],
    *,
    docker_session: str | None = None,
) -> dict[str, Any]:
    cells = [cell_report(c, native, NATIVE_BACKEND, (COLD_N, WARM_N)) for c in matrix()]
    docker_cells = [Cell.from_sample(s) for s in {s["cell"]: s for s in docker}.values()]
    cells += [cell_report(c, docker, DOCKER_BACKEND, (0, DOCKER_WARM_N)) for c in docker_cells]
    native_cells = [c for c in cells if c["backend"] == NATIVE_BACKEND]
    missing = [c["key"] for c in native_cells if not c["warm"]["n"]]
    partial = [c["key"] for c in native_cells if c["partial"] and c["warm"]["n"]]
    pf = preflight_payload or {}
    return {
        "version": 1,
        "generatedAt": _now(),
        "generatedBy": "uv run python -m veritrade_evals latency report",
        "session": pf.get("session"),
        "clean": bool(pf.get("ok")),
        "complete": not missing and not partial,
        "missing": missing,
        "partial": partial,
        "dockerSession": docker_session,
        "budgets": {
            "signalLedeP50Ms": BUDGETS.signal_lede_p50_ms,
            "ttftP50Ms": BUDGETS.ttft_p50_ms,
            "completeP50Ms": BUDGETS.complete_p50_ms,
        },
        "method": {
            "native": "in-process app (create_app) with a local fetcher serving the stored HTML of the 30 "
            "dev articles; no_cache=true; client concurrency 1; parser selectolax, extractor cascade",
            "cold": f"n={COLD_N} per cell on the first dev articles, the model unloaded before each request",
            "warm": f"n={WARM_N} per cell, one per dev article, after one unrecorded warm-up request",
            "docker": f"warm n={DOCKER_WARM_N} over HTTP against the bundled app; it fetches live, so fetch "
            "and consent time are subtracted from every metric",
            "clock": "server-side from request receipt; totals, TTFT and time to signal exclude fetch and "
            "consent; TTFT = first summary_delta, time to signal = the sentiment event (JSON: total)",
        },
        "environment": pf.get("environment"),
        "preflight": {k: pf.get(k) for k in ("ok", "at", "checks", "warnings")} if pf else None,
        "fetchReference": fetch_reference(),
        "cells": cells,
    }


# --- markdown --------------------------------------------------------------------------------


def _ms(value: Any) -> str:
    return "-" if value is None else f"{value:,.0f}"


def _p(dist: Mapping[str, Any] | None, key: str = "p50") -> str:
    return _ms(dist.get(key)) if dist else "-"


LATENCY_HEADER = (
    "endpoint", "model", "scope", "transport", "n warm", "p50 ms", "p95 ms", "TTFT p50", "to signal p50",
    "cold p50", "parse", "extract", "inference", "prompt eval", "eval", "budget",
)  # fmt: skip


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _budget_text(budget: Mapping[str, Any]) -> str:
    checks = budget.get("checks") or []
    detail = ", ".join(f"{k['metric']} {_ms(k['value'])}/{_ms(k['limit'])}" for k in checks)
    return str(budget.get("status", "n/a")) + (f" ({detail})" if detail else "")


def _latency_row(c: Mapping[str, Any]) -> list[str]:
    warm, cold, stages = c["warm"], c["cold"], c["warm"]["stagesP50Ms"]
    return [
        f"/{c['endpoint']}",
        f"`{c['model']}`",
        c["scope"],
        "SSE" if c["streaming"] else "JSON",
        f"{warm['n']}" + (f" ({warm['errors']} err)" if warm["errors"] else ""),
        _p(warm["totalMs"]),
        _p(warm["totalMs"], "p95"),
        _p(warm["ttftMs"]),
        _p(warm["signalMs"]),
        _p(cold["totalMs"]),
        *(_ms(stages.get(k)) for k in ("parse", "extract", "wall", "prompt_eval", "eval")),
        _budget_text(c["budget"]),
    ]


def render(report: Mapping[str, Any]) -> str:
    env = report.get("environment") or {}
    service = env.get("ollamaService") or {}
    lines = [
        "# Latency suite",
        "",
        f"Generated by `{report['generatedBy']}`; ids and numbers only, no article text.",
        "",
    ]
    if not report["clean"]:
        lines += [
            "> **Not a clean run.** The preflight failed (another workload shared the machine), so these "
            "numbers are indicative only. Re-run in an exclusive window.",
            "",
        ]
    if report["missing"] or report["partial"]:
        lines += [
            f"> **Incomplete.** {len(report['missing'])} cells unmeasured, {len(report['partial'])} below "
            f"their sample targets.",
            "",
        ]
    lines += [
        f"- Session `{report['session']}`; machine {env.get('chip', '?')}, {env.get('ramGb', '?')} GB, "
        f"Ollama {env.get('ollamaVersion', '?')} (`{env.get('backend', '?')}`), service env "
        + (", ".join(f"`{k}={v}`" for k, v in (service.get("env") or {}).items()) or "none")
        + ".",
        f"- Load average at preflight {env.get('loadAverage', '?')}; memory {env.get('memory', '?')}.",
        *(f"- {k}: {v}" for k, v in report["method"].items()),
        "- Budgets (R7, native, warm, excl. fetch): /signal lede p50 <= "
        f"{_ms(report['budgets']['signalLedeP50Ms'])} ms; /summary and /analyze TTFT p50 <= "
        f"{_ms(report['budgets']['ttftP50Ms'])} ms and complete p50 <= "
        f"{_ms(report['budgets']['completeP50Ms'])} ms. A missed budget is reported, never moved.",
        "",
    ]
    for backend in (NATIVE_BACKEND, DOCKER_BACKEND):
        rows = [c for c in report["cells"] if c["backend"] == backend]
        if not rows:
            continue
        lines += [f"## {backend}", ""]
        lines += _table(LATENCY_HEADER, [_latency_row(c) for c in rows])
        lines.append("")
    missed = [c for c in report["cells"] if c["budget"].get("status") == "missed"]
    if missed:
        lines += ["## Budgets missed", ""]
        lines += [
            f"- /{c['endpoint']} `{c['model']}` {c['scope']} {'SSE' if c['streaming'] else 'JSON'}: "
            + ", ".join(
                f"{k['metric']} {_ms(k['value'])} ms > {_ms(k['limit'])} ms"
                for k in c["budget"]["checks"]
                if not k["met"]
            )
            for c in missed
        ]
        lines.append("")
    if ref := report.get("fetchReference"):
        lines += [
            "## Live fetch cost (reference)",
            "",
            f"The suite fetches locally, so the network is reported here from `{ref['source']}` "
            f"({ref['vantage']}): median of the per-source medians {_ms(ref['medianMs'])} ms, slowest source "
            f"{_ms(ref['maxMs'])} ms.",
            "",
            "| source | fetch ms (median) |",
            "|---|---|",
            *(f"| {k} | {_ms(v)} |" for k, v in ref["perSourceMedianMs"].items()),
            "",
        ]
    return "\n".join(lines)


def write_report(raw_dir: Path = RAW_DIR, out_dir: Path = LATENCY_DIR) -> dict[str, Any]:
    pf = read_json(PREFLIGHT_JSON)
    native = RawStore(raw_path(pf["session"], raw_dir)).samples() if pf else []
    sessions = docker_sessions(raw_dir)
    docker = RawStore(sessions[-1]).samples() if sessions else []
    report = build_report(pf, native, docker, docker_session=sessions[-1].stem if sessions else None)
    write_json(out_dir / LATENCY_JSON.name, report)
    (out_dir / LATENCY_MD.name).write_text(render(report), encoding="utf-8")
    return report


# --- docker-cpu ------------------------------------------------------------------------------


def bundled_defaults(path: Path = BUNDLED_ENV) -> dict[Endpoint, str]:
    keys: dict[str, Endpoint] = {
        "VT_DEFAULT_MODEL_SIGNAL": "signal",
        "VT_DEFAULT_MODEL_SUMMARY": "summary",
        "VT_DEFAULT_MODEL_ANALYZE": "analyze",
    }
    out: dict[Endpoint, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() in keys and value.strip():
            out[keys[key.strip()]] = value.strip()
    return out


def docker_cells(models: Sequence[str] | None, defaults: Mapping[Endpoint, str]) -> list[Cell]:
    """The bundled default per endpoint (signal at lede), or every matrix cell of the given models."""
    if models:
        return select_cells(matrix(), models, None)
    cells: list[Cell] = []
    for endpoint, model in defaults.items():
        if endpoint == "signal":
            cells.append(Cell("signal", model, "lede", False))
        else:
            cells += [Cell(endpoint, model, "full", True), Cell(endpoint, model, "full", False)]
    return cells


async def run_docker(
    base_url: str,
    cells: Sequence[Cell],
    articles: Sequence[ArticleExample],
    *,
    warmup: ArticleExample,
    session: str,
    log: TextIO,
) -> int:
    """The bundled app fetches live, so its warm-up is a dev article outside the measured slice."""
    async with httpx.AsyncClient(base_url=base_url, timeout=HTTP_TIMEOUT_S) as http:
        health = (await http.get("/api/health")).json()
        print(f"health: backend {health.get('backend')}, Ollama {health.get('ollama_version')}", file=log)
        if health.get("backend") != DOCKER_BACKEND:
            raise LatencyError(
                f"{base_url} reports backend {health.get('backend')!r}, not {DOCKER_BACKEND!r}"
            )
        store = RawStore(raw_path(f"{DOCKER_BACKEND}-{session}"))
        return await run_suite(
            http,
            cells,
            articles,
            store,
            session=session,
            backend=DOCKER_BACKEND,
            cold_n=0,
            ollama=None,
            shared=True,
            warmup_url=warmup.url,
            log=log,
        )


# --- CLI -------------------------------------------------------------------------------------


def _dev_articles(limit: int) -> tuple[list[ArticleExample], ArticlePaths]:
    paths = ArticlePaths()
    articles = load_articles("dev", paths=paths)
    return (articles[:limit] if limit else articles), paths


def cmd_preflight(args: argparse.Namespace) -> int:
    payload = preflight(args.base_url or get_settings().ollama_base_url, resume=args.resume)
    write_json(PREFLIGHT_JSON, payload)
    for check in payload["checks"]:
        print(f"{'ok  ' if check['ok'] else 'FAIL'} {check['name']}: {check['detail']}")
    for warning in payload["warnings"]:
        print(f"warn {warning}")
    env = payload["environment"]
    print(f"session {payload['session']}; {env['chip']}, {env['ramGb']} GB; Ollama {env['ollamaVersion']}")
    print(f"wrote {_rel(PREFLIGHT_JSON)}")
    return 0 if payload["ok"] else 1


def cmd_run(args: argparse.Namespace) -> int:
    pf = read_json(PREFLIGHT_JSON)
    if pf is None:
        raise LatencyError("no preflight on record; run `latency preflight` first")
    if not pf["ok"] and not args.allow_unclean:
        raise LatencyError(
            "the last preflight failed; fix it and re-run `latency preflight`, or pass --allow-unclean "
            "for an indicative (non-exclusive) run"
        )
    articles, paths = _dev_articles(args.limit)
    allowed = allowlist()
    cells = select_cells(matrix(), args.model, args.endpoint)
    for cell in cells:
        if cell.model not in allowed or not allowed[cell.model].serves(cell.endpoint):
            raise LatencyError(f"{cell.key}: {cell.model} is not allowlisted for /{cell.endpoint}")
    n = len(cells) * (min(args.cold, len(articles)) + len(articles) + 1)
    print(f"session {pf['session']}: {len(cells)} cells, up to {n} requests", file=sys.stderr)
    added = asyncio.run(
        run_native(
            cells,
            articles,
            local_pages(articles, paths),
            session=pf["session"],
            settings=serving_settings(args.base_url),
            cold_n=min(args.cold, len(articles)),
            shared=args.shared,
        )
    )
    report = write_report()
    print(f"{added} new samples; wrote {_rel(LATENCY_JSON)} and {_rel(LATENCY_MD)}")
    print(f"clean={report['clean']} missing={len(report['missing'])} partial={len(report['partial'])}")
    return 0


def cmd_docker(args: argparse.Namespace) -> int:
    pf = read_json(PREFLIGHT_JSON)
    session = args.session or (pf["session"] if pf else new_session())
    every, _ = _dev_articles(0)
    if not 0 < args.n < len(every):
        raise LatencyError(f"--n must be between 1 and {len(every) - 1} (one dev article warms up)")
    articles, warmup = every[: args.n], every[args.n]
    cells = docker_cells(args.model, bundled_defaults())
    print(f"{DOCKER_BACKEND} session {session}: {len(cells)} cells x {len(articles)} warm", file=sys.stderr)
    added = asyncio.run(
        run_docker(args.base_url, cells, articles, warmup=warmup, session=session, log=sys.stderr)
    )
    write_report()
    print(f"{added} new samples; wrote {_rel(LATENCY_JSON)}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report = write_report()
    print(f"wrote {_rel(LATENCY_JSON)} and {_rel(LATENCY_MD)}; complete={report['complete']}")
    return 0


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Registers `latency` on a parent CLI; dispatch with `run(args)`."""
    parser: argparse.ArgumentParser = subparsers.add_parser("latency", help="latency suite (R7 budgets)")
    actions = parser.add_subparsers(dest="latency_action", required=True)

    pre = actions.add_parser("preflight", help="record the machine; fail while Docker containers run")
    pre.add_argument("--base-url", default=None, help="Ollama URL (default from settings)")
    pre.add_argument("--resume", action="store_true", help="keep the recorded session id (resume a run)")
    pre.set_defaults(latency_func=cmd_preflight)

    run_p = actions.add_parser("run", help="native suite through the in-process app (resumable)")
    run_p.add_argument(
        "--model", "--models", dest="model", action="append", help="restrict to this model (repeatable)"
    )
    run_p.add_argument("--endpoint", action="append", choices=("signal", "summary", "analyze"))
    run_p.add_argument("--limit", type=int, default=0, help="first N dev articles only (smoke runs)")
    run_p.add_argument("--cold", type=int, default=COLD_N, help="cold samples per cell")
    run_p.add_argument("--shared", action="store_true", help="never unload other models (shared Ollama)")
    run_p.add_argument("--allow-unclean", action="store_true", help="run after a failed preflight")
    run_p.add_argument("--base-url", default=None, help="Ollama URL (default from settings)")
    run_p.set_defaults(latency_func=cmd_run)

    docker = actions.add_parser("docker-cpu", help="warm pass over HTTP against the bundled app")
    docker.add_argument("--base-url", required=True, help="the bundled app, e.g. http://localhost:8000")
    docker.add_argument(
        "--model",
        "--models",
        dest="model",
        action="append",
        help="models to measure (default: the bundled.env defaults)",
    )
    docker.add_argument("--n", type=int, default=DOCKER_WARM_N, help="warm samples per cell")
    docker.add_argument("--session", default=None, help="session id (default: the preflight session)")
    docker.set_defaults(latency_func=cmd_docker)

    rep = actions.add_parser("report", help="re-aggregate raw samples into latency.{json,md}")
    rep.set_defaults(latency_func=cmd_report)
    return parser


def run(args: argparse.Namespace) -> int:
    try:
        code: int = args.latency_func(args)
    except (LatencyError, VtError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="veritrade_evals.latency", description=__doc__)
    add_parser(parser.add_subparsers(dest="command", required=True))
    return run(parser.parse_args(["latency", *(argv if argv is not None else sys.argv[1:])]))


if __name__ == "__main__":
    sys.exit(main())
