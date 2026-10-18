from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALS_DIR = REPO_ROOT / "evals"
DATA_DIR = EVALS_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
DATASETS_LOCK = DATA_DIR / "datasets.lock.json"
RESULTS_DIR = EVALS_DIR / "results"
# Committed article set: URLs, hashes, labels and model-written facts only.
ARTICLES_DIR = DATA_DIR / "articles"
OPUS_PROMPTS_DIR = EVALS_DIR / "opus_prompts"

# Gitignored: dataset files, publisher HTML and prediction caches never leave the machine.
PRIVATE_DIR = DATA_DIR / "private"
HF_CACHE_DIR = PRIVATE_DIR / "hf"
PREDS_DIR = PRIVATE_DIR / "preds"
PRIVATE_ARTICLES_DIR = PRIVATE_DIR / "articles"
