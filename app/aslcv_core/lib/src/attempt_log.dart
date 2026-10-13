/// Live-attempt logging on the phone: the same record Python's
/// `aslcv.attempt_log.AttemptRecorder` keeps on the desktop, so capture
/// settings can be tuned against real phone recordings
/// (`scripts/import_phone_attempts.py`, then `scripts/replay_attempts.py`).
///
/// Every pose fed to the CaptureBuffer since the last reset is kept,
/// INCLUDING frames after the buffer settles, which the buffer itself
/// discards: they are the evidence of a capture that settled while the
/// learner was still signing. Keypoints only, never video.
///
/// File layout (`rawFrames`; the app gzips it), little-endian, n frames of K
/// points, in this order so every block stays aligned:
///   timestamps float64[n] (seconds), keypoints float32[n*K*2] (x, y),
///   scores float32[n*K], states int8[n] (0 idle, 1 active, 2 settled).
library;

import 'dart:typed_data';

import 'capture.dart';
import 'grader.dart';
import 'skeleton.dart';

const attemptFormat = 'aslcv-phone-attempt/1';

/// Python's STATE_CODES.
int stateCode(CaptureState s) => switch (s) {
      CaptureState.idle => 0,
      CaptureState.active => 1,
      CaptureState.settled => 2,
    };

class AttemptRecorder {
  /// Bounded so a session left open can't grow without limit; the dropped
  /// prefix is idle rest, which replay only needs `preroll` frames of.
  final int maxFrames;

  final List<Pose> _poses = [];
  final List<int> _states = [];
  final List<double> _times = [];
  int dropped = 0; // frames evicted from the front since reset

  AttemptRecorder({this.maxFrames = 1800});

  void reset() {
    _poses.clear();
    _states.clear();
    _times.clear();
    dropped = 0;
  }

  /// `state` is the buffer's state AFTER appending this frame, as on the
  /// desktop (the settling frame itself is recorded as settled).
  void record(Pose pose, CaptureState state, double seconds) {
    if (_poses.length == maxFrames) {
      _poses.removeAt(0);
      _states.removeAt(0);
      _times.removeAt(0);
      dropped++;
    }
    _poses.add(pose);
    _states.add(stateCode(state));
    _times.add(seconds);
  }

  int get length => _poses.length;

  bool get everActive => _states.any((s) => s != 0);

  /// [start, stop) of the frames the buffer held when it settled, as
  /// indices into this recording; null if it never settled.
  (int, int)? capturedSpan(int nCaptured) {
    final first = _states.indexOf(2);
    if (first < 0) return null;
    final stop = first + 1;
    return (stop - nCaptured < 0 ? 0 : stop - nCaptured, stop);
  }

  double? get streamFps {
    if (_times.length < 2 || _times.last <= _times.first) return null;
    return (_times.length - 1) / (_times.last - _times.first);
  }

  /// The frame data (layout in the library doc).
  Uint8List rawFrames() {
    final n = _poses.length;
    final k = n == 0 ? 0 : _poses.first.scores.length;
    final out = BytesBuilder(copy: false)
      ..add(Float64List.fromList(_times).buffer.asUint8List());
    final kp = Float32List(n * k * 2), sc = Float32List(n * k);
    for (var t = 0; t < n; t++) {
      kp.setAll(t * k * 2, _poses[t].keypoints);
      sc.setAll(t * k, _poses[t].scores);
    }
    out
      ..add(kp.buffer.asUint8List())
      ..add(sc.buffer.asUint8List())
      ..add(Int8List.fromList(_states).buffer.asUint8List());
    return out.takeBytes();
  }

  /// The sidecar's recording fields; the caller adds what it knows about the
  /// attempt (mode, target, capture parameters, grade).
  Map<String, Object?> describe() => {
        'format': attemptFormat,
        'n_frames': _poses.length,
        'n_keypoints': _poses.isEmpty ? 0 : _poses.first.scores.length,
        'dropped_frames': dropped,
        'stream_fps': streamFps,
      };
}

/// Inverse of `rawFrames`, for tests and tools.
({Float64List times, Float32List keypoints, Float32List scores, Int8List states}) decodeFrames(
    Uint8List raw, int n, int k) {
  final b = ByteData.sublistView(raw);
  var o = 0;
  final times = Float64List(n);
  for (var i = 0; i < n; i++, o += 8) {
    times[i] = b.getFloat64(o, Endian.little);
  }
  final kp = Float32List(n * k * 2);
  for (var i = 0; i < kp.length; i++, o += 4) {
    kp[i] = b.getFloat32(o, Endian.little);
  }
  final sc = Float32List(n * k);
  for (var i = 0; i < sc.length; i++, o += 4) {
    sc[i] = b.getFloat32(o, Endian.little);
  }
  final st = Int8List(n);
  for (var i = 0; i < n; i++, o++) {
    st[i] = b.getInt8(o);
  }
  if (o != raw.length) throw FormatException('expected $o bytes, got ${raw.length}');
  return (times: times, keypoints: kp, scores: sc, states: st);
}

/// One graded segment in the desktop sidecar's `segments` shape.
Map<String, Object?> gradeJson(Grade g, {(int, int)? frameRange}) => {
      'target_sign': g.targetSign,
      'frame_range': frameRange == null ? null : [frameRange.$1, frameRange.$2],
      'fidelity': g.fidelity,
      'parameters': {
        for (final v in g.parameters.values)
          v.parameter: {'predicted': v.predicted, 'target': v.target, 'correct': v.correct, 'confidence': v.confidence},
      },
    };
