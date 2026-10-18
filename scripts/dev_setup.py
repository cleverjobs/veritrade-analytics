"""Local machine setup: free Docker resources, install native Ollama, pull model sets."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOPPED_FILE = ROOT / ".temp" / "setup" / "stopped-containers.txt"
MODEL_SETS = {
    "eval": ROOT / "deploy" / "models-eval.txt",
    "ci": ROOT / "deploy" / "ci-models.txt",
}
OLLAMA_URL = "http://localhost:11434"
# brew services only reads launchd env, so these must go through launchctl.
LAUNCHD_ENV = {"OLLAMA_MAX_LOADED_MODELS": "3", "OLLAMA_KEEP_ALIVE": "30m"}


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def stop_containers() -> None:
    names = run(["docker", "ps", "--format", "{{.Names}}"]).stdout.split()
    if not names:
        print("no running containers")
        return
    STOPPED_FILE.parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    with STOPPED_FILE.open("a") as fh:
        fh.write(f"# {stamp}\n" + "\n".join(names) + "\n")
    run(["docker", "stop", *names])
    print(f"stopped {len(names)} containers; restart with:\n  docker start {' '.join(names)}")


def restart_hint() -> None:
    if not STOPPED_FILE.exists():
        print("nothing recorded")
        return
    lines = STOPPED_FILE.read_text().splitlines()
    names = sorted({ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")})
    print(f"docker start {' '.join(names)}")


def ollama_version() -> str | None:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/version", timeout=2) as resp:
            return str(json.load(resp)["version"])
    except OSError:
        return None


def native() -> None:
    if shutil.which("ollama") is None:
        run(["brew", "install", "ollama"])
    for key, value in LAUNCHD_ENV.items():
        run(["launchctl", "setenv", key, value])
    run(["brew", "services", "restart", "ollama"])
    for _ in range(60):
        version = ollama_version()
        if version:
            print(f"ollama {version} up at {OLLAMA_URL}")
            return
        time.sleep(1)
    sys.exit("ollama did not come up within 60 s; check /opt/homebrew/var/log/ollama.log")


def pull(model_set: str) -> None:
    models = [
        ln.strip()
        for ln in MODEL_SETS[model_set].read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    failed = []
    for model in models:
        result = subprocess.run(["ollama", "pull", model], text=True, capture_output=True)
        status = "ok" if result.returncode == 0 else f"FAILED: {result.stderr.strip()[-300:]}"
        print(f"{model}: {status}", flush=True)
        if result.returncode != 0:
            failed.append(model)
    print(run(["ollama", "list"]).stdout)
    if failed:
        sys.exit(f"failed pulls: {', '.join(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stop-containers")
    sub.add_parser("restart-hint")
    sub.add_parser("native")
    pull_p = sub.add_parser("pull")
    pull_p.add_argument("--set", choices=sorted(MODEL_SETS), default="eval")
    args = parser.parse_args()
    match args.cmd:
        case "stop-containers":
            stop_containers()
        case "restart-hint":
            restart_hint()
        case "native":
            native()
        case "pull":
            pull(args.set)


if __name__ == "__main__":
    main()
