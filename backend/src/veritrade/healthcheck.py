"""Container HEALTHCHECK entrypoint; avoids shipping curl or inline python in the image."""

from __future__ import annotations

import os
import sys
import urllib.request


def main() -> None:
    port = os.environ.get("PORT", "8000")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=3) as resp:
            sys.exit(0 if resp.status == 200 else 1)
    except OSError:
        sys.exit(1)


if __name__ == "__main__":
    main()
