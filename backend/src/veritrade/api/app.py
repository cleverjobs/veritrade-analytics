from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from veritrade import __version__
from veritrade.api.contracts import ErrorResponse
from veritrade.api.routes import router
from veritrade.api.sse import error_body
from veritrade.config import Settings, get_settings
from veritrade.errors import InvalidUrl, VtError
from veritrade.fetch import Fetcher, HttpFetcher
from veritrade.inference.ollama import OllamaClient
from veritrade.models_registry import Registry
from veritrade.services.runtime import Runtime

URL_HINT = "Send the full article address, starting with http:// or https://."


def error_response(exc: VtError) -> JSONResponse:
    body = ErrorResponse(error=error_body(exc))
    return JSONResponse(status_code=exc.http_status, content=body.model_dump(mode="json"))


def create_app(
    settings: Settings | None = None,
    *,
    fetcher: Fetcher | None = None,
    ollama: OllamaClient | None = None,
    registry: Registry | None = None,
) -> FastAPI:
    """Injected clients stay owned by the caller; the lifespan closes only what it builds."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            page_fetcher = fetcher
            if page_fetcher is None:
                page_fetcher = HttpFetcher(settings)
                stack.push_async_callback(page_fetcher.aclose)
            llm = ollama
            if llm is None:
                llm = OllamaClient.from_settings(settings)
                stack.push_async_callback(llm.aclose)
            app.state.runtime = Runtime.build(settings, fetcher=page_fetcher, ollama=llm, registry=registry)
            yield

    app = FastAPI(title="VeriTrade Analytics", version=__version__, lifespan=lifespan)
    app.state.settings = settings

    @app.exception_handler(VtError)
    async def _vt_error(_: Request, exc: VtError) -> JSONResponse:
        return error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def _invalid_request(request: Request, exc: RequestValidationError) -> Response:
        # A malformed article URL gets the documented invalid_url code; other shape errors keep FastAPI's 422.
        errors = exc.errors()
        if errors and all(tuple(error["loc"][:2]) == ("body", "url") for error in errors):
            return error_response(InvalidUrl(f"url: {errors[0]['msg']}", hint=URL_HINT))
        return await request_validation_exception_handler(request, exc)

    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    app.include_router(router)
    # Mounted last so /api routes win; html=True serves the static export's index.html files.
    if settings.static_dir and settings.static_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="static")
    return app
