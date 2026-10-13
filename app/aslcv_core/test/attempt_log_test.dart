import 'dart:typed_data';

import 'package:aslcv_core/aslcv_core.dart';
import 'package:test/test.dart';

Pose _pose(int t, int k) => Pose(
      Float32List.fromList([for (var i = 0; i < 2 * k; i++) t + i / 10]),
      Float32List.fromList([for (var i = 0; i < k; i++) i.isEven ? 1.0 : 0.0]),
    );

void main() {
  const k = 4;

  test('round trip through the file layout', () {
    final rec = AttemptRecorder();
    final states = [CaptureState.idle, CaptureState.active, CaptureState.active, CaptureState.settled, CaptureState.settled];
    for (var t = 0; t < states.length; t++) {
      rec.record(_pose(t, k), states[t], 10 + t / 25);
    }
    final raw = rec.rawFrames();
    expect(raw.length, 5 * 8 + 5 * k * 2 * 4 + 5 * k * 4 + 5);
    final d = decodeFrames(raw, 5, k);
    expect(d.states, [0, 1, 1, 2, 2]);
    expect(d.times[4], closeTo(10.16, 1e-12));
    expect(d.keypoints.sublist(3 * 2 * k, 4 * 2 * k), _pose(3, k).keypoints);
    expect(d.scores.sublist(0, k), [1, 0, 1, 0]);
    expect(rec.describe(), containsPair('n_keypoints', k));
    expect(rec.streamFps, closeTo(25, 1e-9));
  });

  test('captured span ends one past the settling frame, as in Python', () {
    final rec = AttemptRecorder();
    for (final (t, s) in [(0, CaptureState.idle), (1, CaptureState.active), (2, CaptureState.active),
                          (3, CaptureState.settled), (4, CaptureState.settled)]) {
      rec.record(_pose(t, k), s, t.toDouble());
    }
    expect(rec.capturedSpan(3), (1, 4));
    expect(rec.capturedSpan(10), (0, 4));
    expect(rec.everActive, isTrue);
  });

  test('never settled and never active', () {
    final rec = AttemptRecorder()..record(_pose(0, k), CaptureState.idle, 0);
    expect(rec.capturedSpan(1), isNull);
    expect(rec.everActive, isFalse);
  });

  test('bounded: oldest frames dropped and counted', () {
    final rec = AttemptRecorder(maxFrames: 3);
    for (var t = 0; t < 5; t++) {
      rec.record(_pose(t, k), CaptureState.active, t.toDouble());
    }
    expect(rec.length, 3);
    expect(rec.dropped, 2);
    final d = decodeFrames(rec.rawFrames(), 3, k);
    expect(d.times, [2, 3, 4]);
    rec.reset();
    expect((rec.length, rec.dropped), (0, 0));
  });
}
