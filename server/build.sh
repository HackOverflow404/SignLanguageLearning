#!/usr/bin/env bash
# Assemble the server's Docker build context into server/build/app/: the server, plus the
# gloss engine and sentence-prompt modules copied UNCHANGED from src/ with
# empty package __init__s, so the server never imports the heavy parts of the
# package (torch, OpenCV, the clip cache). Same relative layout as the repo, so
# gloss_rules.py still finds curriculum.yaml at its REPO root.
set -euo pipefail
server=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$server/.." && pwd)
out="$server/build/app"
rm -rf "$out"
mkdir -p "$out/src/aslcv/production" "$out/src/aslcv/generator"
cp "$server/"{app.py,llm.py,eval_generator.py,Dockerfile,requirements.txt,test_app.py} "$out/"
cp "$repo/curriculum.yaml" "$out/"
cp "$repo/src/aslcv/production/gloss_rules.py" "$out/src/aslcv/production/"
cp "$repo/src/aslcv/generator/"{sentence_prompts.py,_hf_client.py} "$out/src/aslcv/generator/"
touch "$out/src/aslcv/__init__.py" "$out/src/aslcv/production/__init__.py" "$out/src/aslcv/generator/__init__.py"
echo "assembled $(find "$out" -type f | wc -l) files -> ${out#$repo/}"
