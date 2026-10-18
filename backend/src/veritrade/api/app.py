from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from veritrade import __version__
from veritrade.api.contracts import ErrorBody, ErrorResponse
from veritrade.api.routes import router
from veritrade.config import Settings, get_settings
from veritrade.errors import VtError


def error_response(exc: VtError) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(
            code=exc.code,
            message=exc.message,
            hint=exc.hint,
            retryable=exc.retryable,
            upstream_status=exc.upstream_status,
        )
    )
    return JSONResponse(status_code=exc.http_status, content=body.model_dump(mode="json"))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="VeriTrade Analytics", version=__version__)
    app.state.settings = settings

    @app.exception_handler(VtError)
    async def _vt_error(_: Request, exc: VtError) -> JSONResponse:
        return error_response(exc)

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
