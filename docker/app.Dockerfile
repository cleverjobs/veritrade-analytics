ARG PYTHON_IMAGE=python:3.13-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.10.9

FROM ${UV_IMAGE} AS uv

# Static Next.js export: FastAPI serves it from the same origin, so no Node runs in production.
FROM node:24-slim AS frontend
ENV NEXT_TELEMETRY_DISABLED=1 COREPACK_ENABLE_DOWNLOAD_PROMPT=0
RUN corepack enable
WORKDIR /src/frontend
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./
RUN --mount=type=cache,id=pnpm-store,target=/pnpm/store \
    pnpm install --frozen-lockfile --store-dir /pnpm/store
COPY frontend/ ./
# The context holds evals/results/benchmarks.json only once evals publish it; until then the sample is used.
RUN --mount=type=bind,source=evals,target=/src/evals pnpm build

# Third-party dependencies only, so backend edits do not invalidate this layer.
FROM ${PYTHON_IMAGE} AS deps
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
COPY backend/pyproject.toml backend/
COPY evals/pyproject.toml evals/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package veritrade-backend

FROM deps AS wheel
COPY backend/src backend/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv build --wheel --package veritrade-backend --out-dir /dist

FROM ${PYTHON_IMAGE} AS runtime
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=deps /app/.venv /app/.venv
# A small layer of its own: only the app wheel changes between backend edits.
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=from=wheel,source=/dist,target=/dist \
    uv pip install --python /app/.venv/bin/python --no-deps --compile-bytecode --no-cache /dist/*.whl
COPY --from=frontend /src/frontend/out /app/static
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VT_ENV=prod \
    VT_STATIC_DIR=/app/static
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 CMD ["veritrade-healthcheck"]
CMD ["uvicorn", "veritrade.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
