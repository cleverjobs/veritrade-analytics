"""Single quality gate for local runs and CI: lint, types, tests, build and contract drift."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


def backend_steps(full: bool) -> list[tuple[str, list[str], Path]]:
    tests = ["backend/tests", "evals/tests"] if full else ["backend/tests/unit", "evals/tests"]
    steps = [
        ("ruff", ["uv", "run", "ruff", "check", "."], ROOT),
        ("ruff-format", ["uv", "run", "ruff", "format", "--check", "."], ROOT),
        ("mypy", ["uv", "run", "mypy"], ROOT),
        ("openapi-drift", ["uv", "run", "python", "scripts/export_openapi.py", "--check"], ROOT),
        ("pytest", ["uv", "run", "pytest", "-q", *[t for t in tests if (ROOT / t).exists()]], ROOT),
    ]
    # The floor only means something once the API is wired and integration tests exist.
    if full and (ROOT / "backend" / "tests" / "integration").exists():
        steps[-1][1].extend(["--cov", "--cov-report=term-missing:skip-covered", "--cov-fail-under=75"])
    return steps


def frontend_steps(full: bool) -> list[tuple[str, list[str], Path]]:
    steps = [
        ("fe-lint", ["pnpm", "lint"], FRONTEND),
        ("fe-format", ["pnpm", "format:check"], FRONTEND),
        ("fe-typecheck", ["pnpm", "typecheck"], FRONTEND),
        ("fe-test", ["pnpm", "test"], FRONTEND),
    ]
    if full:
        steps.append(("fe-build", ["pnpm", "build"], FRONTEND))
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", help="unit tests only, no build (default)")
    mode.add_argument("--full", action="store_true", help="integration tests, coverage floor, build")
    parser.add_argument("--backend", action="store_true", help="only backend + evals")
    parser.add_argument("--frontend", action="store_true", help="only frontend")
    args = parser.parse_args()

    run_be = not args.frontend or args.backend
    run_fe = not args.backend or args.frontend
    steps = (backend_steps(args.full) if run_be else []) + (frontend_steps(args.full) if run_fe else [])

    failed = []
    for name, cmd, cwd in steps:
        print(f"\n== {name}: {' '.join(cmd)}", flush=True)
        if subprocess.run(cmd, cwd=cwd).returncode != 0:
            failed.append(name)
    if failed:
        sys.exit(f"\nFAILED: {', '.join(failed)}")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
