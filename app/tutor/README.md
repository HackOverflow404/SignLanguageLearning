# ASL Tutor (Flutter)

The phone app (Phase 8). Single-sign mode: the scheduler picks a sign (or pick one
yourself), you watch its real reference clip, sign it, and get a coaching line plus a
verdict for each phonological parameter (handshape, location, exact spot, movement,
repeats) against that known target. It always measures distance to the target sign,
never guesses "which sign was that". Each graded attempt updates per-sign mastery,
saved on the phone; "Next sign" picks what to drill next (weak and due signs first,
a contrast drill when a minimal-pair partner exists for your last mistake).

The grading pipeline itself lives in [`../aslcv_core`](../aslcv_core) (pure Dart,
parity-tested against the Python implementation) and the landmark detection in
[`../mediapipe_ios`](../mediapipe_ios). This package is the UI and the wiring:

- `lib/engine.dart`: loads the ONNX grader bundle, skeleton, reference-clip index
  and MediaPipe landmarkers; converts landmarker output into the bundle's
  keypoint layout.
- `lib/main.dart`: camera, landmarks, `CaptureBuffer`, grading, coaching, the
  scheduler, and the result screen; the ⓘ diagnostics sheet (delegate, fps,
  hand-slot test, landmark parity vs Python, reset progress).
- `lib/progress.dart`: saves `MasteryState` as JSON in the app's documents
  directory (same keys as the desktop's `data/learner_state.json`).
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
.venv/bin/python scripts/export_learner_text.py
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

## Sentence mode

The notes icon in the top bar switches to sentence mode: the app asks the sentence
server (`server/` in the repo) for a sentence using the scheduler's target sign,
or glosses one you type (pencil icon). The reference card plays each word's clip in
order; sign the whole sentence, then each word is graded on its own after forced
alignment, and mastery updates per word. Build with the server's address:

```bash
flutter/build.sh ~/Documents/Projects/SignLanguageLearning/app/tutor SENTENCE_SERVER=https://<ip-with-dashes>.sslip.io/
```

Without it the mode says no server is set up; single-sign mode works offline.

## Attempt logging

Diagnostics (ⓘ) → "Log attempts for capture tuning" saves each attempt's
keypoints (never video) to the app's Documents/attempts folder, visible in the
Files app. To replay them on the desktop, plug in and unlock the phone:

```bash
.venv/bin/python scripts/import_phone_attempts.py --pull
.venv/bin/python scripts/replay_attempts.py data/live_sessions/phone-*.npz
```

## Not yet here

Android.
