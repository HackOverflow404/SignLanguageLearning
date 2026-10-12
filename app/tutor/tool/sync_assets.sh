#!/usr/bin/env bash
# Copy the app's model assets in from the repo's models/ (all gitignored there
# and here): the exported grading bundle, the bundle's reference clips, the
# MediaPipe landmarker models, and the still frame + Python landmarks used for
# the on-device GPU parity check. Re-run after scripts/export_mobile.py,
# scripts/export_reference_clips.py, scripts/export_learner_text.py,
# scripts/export_sentence_references.py, or a model change.
set -euo pipefail
app=$(cd "$(dirname "$0")/.." && pwd)
models=$(cd "$app/../../models" && pwd)
spike="$app/../flutter_spike/assets"
a="$app/assets"
rm -rf "$a/bundle" "$a/clips" "$a/mediapipe" "$a/parity"
mkdir -p "$a/bundle" "$a/clips" "$a/mediapipe" "$a/parity"
cp "$models/mobile/"{grader.onnx,bundle.json,skeleton.json,references.f32,text.json,sentence_refs.f32,sentence_refs.json} "$a/bundle/"
cp "$models/mobile/clips/"* "$a/clips/"
cp "$models/"{pose_landmarker_full.task,hand_landmarker.task} "$a/mediapipe/"
cp "$spike/"{parity_frame.png,parity_expected.json} "$a/parity/"
du -sh "$a"
