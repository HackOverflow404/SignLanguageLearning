# ASL Tutor (Flutter)

The phone app (Phase 8). Single-sign mode: pick a sign, watch its real reference
clip, sign it, and get a verdict for each phonological parameter (handshape,
location, exact spot, movement, repeats) against that known target. It always
measures distance to the target sign, never guesses "which sign was that".

The grading pipeline itself lives in [`../aslcv_core`](../aslcv_core) (pure Dart,
parity-tested against the Python implementation) and the landmark detection in
[`../mediapipe_ios`](../mediapipe_ios). This package is the UI and the wiring:

- `lib/engine.dart`: loads the ONNX grader bundle, skeleton, reference-clip index
  and MediaPipe landmarkers; converts landmarker output into the bundle's
  keypoint layout.
- `lib/main.dart`: camera, landmarks, `CaptureBuffer`, grading, and the result
  screen; the ⓘ diagnostics sheet (delegate, fps, hand-slot test, landmark parity
  vs Python).
- `lib/delegate_check.dart`: grades the bundled reference clips with GPU and with
  CPU landmarks and compares the verdicts.

**The front camera feed is selfie-mirrored.** Frames are un-mirrored natively before
detection so left/right match the unflipped training video. The preview stays
mirrored.

## Build and install (iOS, from Linux)

Prerequisites: the [omarchy-apple-dev](https://github.com/joshuaswarren/omarchy-apple-dev)
toolchain installed with an Xcode SDK, Flutter 3.47 (via mise), `xtool auth` done
once, and the phone plugged in, unlocked and trusted.

```bash
# from the repo root, once per model/bundle change
.venv/bin/python scripts/export_mobile.py
.venv/bin/python scripts/export_reference_clips.py
app/mediapipe_ios/tool/fetch_frameworks.sh     # first checkout only
app/tutor/tool/sync_assets.sh

# build + install
cd ~/.local/share/omarchy-apple-dev
PATH=$(mise where flutter@3.47.7)/bin:/usr/lib/swift/usr/bin:$HOME/.local/bin:$PATH \
  flutter/build.sh ~/Documents/Projects/SignLanguageLearning/app/tutor
xtool install ~/Documents/Projects/SignLanguageLearning/app/tutor/build/ios-linux/Runner.ipa
```

With a free Apple ID, the installed app expires after 7 days; reinstall to renew.
Close the app on the phone before installing. If the install fails with
`deadlineExceeded`, Apple's provisioning server timed out; retry it.

## Not yet here

Mastery tracking and the next-sign scheduler, coaching text, sentence mode,
attempt logging, Android.
