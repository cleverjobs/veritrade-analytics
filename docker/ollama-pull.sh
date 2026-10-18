#!/bin/sh
# One-shot init: pull the comma-separated VT_PULL_MODELS into the shared ollama volume.
set -eu
old_ifs=$IFS
IFS=','
for model in $VT_PULL_MODELS; do
  IFS=$old_ifs
  echo "pulling ${model}"
  ollama pull "${model}"
done
echo "models ready"
