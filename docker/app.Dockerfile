# syntax=docker/dockerfile:1.7
# Stage 1: static Next.js export (FastAPI serves it from the same origin, so no Node at runtime).
FROM node:24-slim AS fe
WORKDIR /src/frontend
RUN corepack enable
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./
RUN --mount=type=cache,target=/root/.local/share/pnpm/store pnpm install --frozen-lockfile
COPY frontend/ ./
COPY evals/results/ /src/evals/results/
RUN pnpm build

# Stage 2: locked, non-editable backend install (runtime image carries no source tree).
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS py
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
COPY backend/pyproject.toml backend/pyproject.toml
COPY evals/pyproject.toml evals/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package veritrade-backend
COPY backend/ backend/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package veritrade-backend

# Stage 3: slim runtime.
FROM python:3.13-slim-bookworm AS runtime
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=py /app/.venv /app/.venv
COPY --from=fe /src/frontend/out /app/static
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    VT_STATIC_DIR=/app/static
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=4s --start-period=20s --retries=5 CMD ["veritrade-healthcheck"]
CMD ["uvicorn", "veritrade.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
