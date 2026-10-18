#!/bin/sh
# One-shot init for the bundled profile: pulls VT_PULL_MODELS (comma-separated) plus the lines of
# VT_PULL_MODELS_FILE into the ollama service's model store, skipping models it already holds.
set -eu

models=$(printf '%s\n' "${VT_PULL_MODELS:-}" | tr ',' '\n')
if [ -n "${VT_PULL_MODELS_FILE:-}" ]; then
  models="${models}
$(sed 's/#.*//' "${VT_PULL_MODELS_FILE}")"
fi

count=0
for model in $(printf '%s\n' "${models}" | tr -d ' \r' | grep -v '^$' | sort -u); do
  # A model already in the store needs no registry round-trip, so cached re-runs work offline.
  if ollama show "${model}" > /dev/null 2>&1; then
    echo "present ${model}"
  else
    echo "pulling ${model}"
    ollama pull "${model}"
  fi
  count=$((count + 1))
done

if [ "${count}" -eq 0 ]; then
  echo "no models requested: set VT_PULL_MODELS or VT_PULL_MODELS_FILE" >&2
  exit 1
fi
echo "models ready (${count})"
