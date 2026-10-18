"""Write backend/openapi.json from the FastAPI app; --check fails on drift (CI contract gate)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from veritrade.api.app import create_app
from veritrade.config import Settings

TARGET = Path(__file__).resolve().parents[1] / "backend" / "openapi.json"


def render() -> str:
    app = create_app(Settings(static_dir=None))
    return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render()
    if args.check:
        if not TARGET.exists() or TARGET.read_text() != rendered:
            sys.exit("backend/openapi.json is stale: run `uv run python scripts/export_openapi.py`")
        print("openapi.json up to date")
        return
    TARGET.write_text(rendered)
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
