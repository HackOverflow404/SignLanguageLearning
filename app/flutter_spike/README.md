# Flutter iOS feasibility spike

Kept for reference. This app answered one question before Phase 8 committed to
Flutter: can a Flutter app built on Linux (no Mac, free Apple ID) run the camera,
ONNX Runtime and MediaPipe together on an iPhone, matching Python?

Measured on an iPhone 15 Plus (iOS 26), 2026-10-10:

- Camera stream: ~24 fps.
- ONNX grader on a golden fixture: max difference 9.5e-7 vs Python, 2-4 ms.
- MediaPipe (CPU, IMAGE mode) on a still frame vs Python: pose within 0.15 px,
  hands within 0.04 px, identical handedness scores.
- Live pose + hands on the GPU delegate, run in parallel: ~23.5 fps.

The real app is [`../tutor`](../tutor). The MediaPipe plugin both apps use is
[`../mediapipe_ios`](../mediapipe_ios). The spike's assets (`grader.onnx`, `.task`
files) are gitignored copies from `models/`.
