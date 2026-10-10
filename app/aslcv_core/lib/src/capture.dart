/// Port of src/aslcv/capture.py's CaptureBuffer: motion-aware live capture
/// boundaries instead of a fixed trailing window (which would silently drop
/// the start of a slow attempt).
///
///   idle    -- no motion; buffer trimmed to `preroll` frames of rest lead-in
///   active  -- motion seen; accumulate up to `maxFrames`
///   settled -- `settleFrames` consecutive quiet frames (or the cap): a
///              complete attempt, frozen until reset()
///
/// `energyFn` is injected (handMotionEnergy bound to a skeleton, in the app)
/// so the state machine is testable with a synthetic signal. If it ever
/// throws, the buffer degrades to a plain capped trailing window instead of
/// crashing the loop.
library;

import 'dart:math' as math;

enum CaptureState { idle, active, settled }

class CaptureBuffer<T> {
  final List<double> Function(List<T> frames) _energyFn;
  final double motionThreshold;
  int settleFrames;
  final int preroll;
  final int maxFrames;

  List<T> frames = [];
  CaptureState state = CaptureState.idle;
  bool _energyOk = true;

  CaptureBuffer(
    this._energyFn,
    this.motionThreshold, {
    this.settleFrames = 10,
    this.preroll = 15,
    this.maxFrames = 150,
  });

  bool get degraded => !_energyOk;

  void reset() {
    frames = [];
    state = CaptureState.idle;
  }

  List<T> _tail(int n) => frames.length > n ? frames.sublist(frames.length - n) : frames;

  void append(T frame) {
    if (state == CaptureState.settled) return; // frozen until reset()
    frames.add(frame);

    if (!_energyOk) {
      frames = _tail(maxFrames);
      state = frames.length >= preroll ? CaptureState.active : CaptureState.idle;
      return;
    }

    final List<double> energy;
    try {
      energy = _energyFn(frames);
    } catch (_) {
      _energyOk = false;
      return;
    }
    final n = energy.length;
    if (n < 2) return; // not enough frames for a velocity-based signal yet

    if (state == CaptureState.idle) {
      var moving = false;
      for (var i = math.max(0, n - 3); i < n; i++) {
        if (energy[i] > motionThreshold) moving = true;
      }
      if (moving) {
        state = CaptureState.active;
      } else {
        frames = _tail(preroll);
      }
      return;
    }

    // active
    if (frames.length >= maxFrames) {
      state = CaptureState.settled;
      return;
    }
    if (n < settleFrames) return;
    for (var i = n - settleFrames; i < n; i++) {
      if (energy[i] > motionThreshold) return;
    }
    state = CaptureState.settled;
  }
}
