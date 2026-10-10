// CaptureBuffer state machine, mirroring tests/test_capture.py case for case.
// Frames are plain ints indexing a synthetic per-frame energy list, so idle
// trimming never breaks the frame -> energy correspondence.
import 'package:aslcv_core/aslcv_core.dart';
import 'package:test/test.dart';

List<double> Function(List<int>) energyFn(List<double> perFrame) =>
    (frames) => [for (final i in frames) perFrame[i]];
List<double> zeros(int n) => List.filled(n, 0.0);
List<double> ones(int n) => List.filled(n, 1.0);
final sentence = [...zeros(5), ...ones(10), ...zeros(5), ...ones(10), ...zeros(5), ...ones(10), ...zeros(20)];

CaptureBuffer<int> buffer(List<double> e, {required int settle, required int preroll, required int max}) =>
    CaptureBuffer<int>(energyFn(e), 0.5, settleFrames: settle, preroll: preroll, maxFrames: max);

void main() {
  test('stays idle while quiet, keeping only the preroll', () {
    final buf = buffer(zeros(20), settle: 5, preroll: 3, max: 50);
    for (var i = 0; i < 10; i++) {
      buf.append(i);
    }
    expect(buf.state, CaptureState.idle);
    expect(buf.frames.length, lessThanOrEqualTo(3));
  });

  test('goes active on a motion spike', () {
    final buf = buffer([...zeros(5), ...ones(20)], settle: 5, preroll: 3, max: 50);
    for (var i = 0; i < 10; i++) {
      buf.append(i);
    }
    expect(buf.state, CaptureState.active);
  });

  test('settles only once the full settle window is quiet', () {
    final buf = buffer([...zeros(5), ...ones(5), ...zeros(20)], settle: 4, preroll: 3, max: 100);
    for (var i = 0; i < 13; i++) {
      buf.append(i);
    }
    expect(buf.state, CaptureState.active);
    buf.append(13);
    expect(buf.state, CaptureState.settled);
  });

  test('forces a settle at the frame cap', () {
    final buf = buffer([...zeros(3), ...ones(100)], settle: 5, preroll: 3, max: 20);
    for (var i = 0; i < 50 && buf.state != CaptureState.settled; i++) {
      buf.append(i);
    }
    expect(buf.state, CaptureState.settled);
    expect(buf.frames.length, lessThanOrEqualTo(20));
  });

  test('reset clears state', () {
    final buf = buffer(ones(10), settle: 3, preroll: 3, max: 50);
    for (var i = 0; i < 6; i++) {
      buf.append(i);
    }
    buf.reset();
    expect(buf.state, CaptureState.idle);
    expect(buf.frames, isEmpty);
  });

  test('is frozen once settled', () {
    final buf = buffer([...zeros(3), ...ones(5), ...zeros(10)], settle: 4, preroll: 3, max: 50);
    for (var i = 0; i < 13; i++) {
      buf.append(i);
    }
    expect(buf.state, CaptureState.settled);
    final len = buf.frames.length;
    buf.append(999); // out of range: would break energyFn if not ignored
    expect(buf.frames.length, len);
  });

  test('sentence mode rides through brief inter-sign pauses', () {
    const lastMotion = 44;
    final buf = buffer(sentence, settle: 15, preroll: 3, max: 200);
    for (var i = 0; i <= lastMotion + 14; i++) {
      buf.append(i);
      expect(buf.state, isNot(CaptureState.settled), reason: 'frame $i');
    }
    buf.append(lastMotion + 15);
    expect(buf.state, CaptureState.settled);
  });

  test('a single-sign settle window ends a sentence mid-way', () {
    final buf = buffer(sentence, settle: 4, preroll: 3, max: 200);
    for (var i = 0; i < 25 && buf.state != CaptureState.settled; i++) {
      buf.append(i);
    }
    expect(buf.state, CaptureState.settled);
  });

  test('degrades to a capped trailing window if energyFn throws', () {
    final buf = CaptureBuffer<int>((_) => throw StateError('no hand blocks'), 0.5,
        settleFrames: 4, preroll: 3, maxFrames: 10);
    for (var i = 0; i < 15; i++) {
      buf.append(i);
    }
    expect(buf.degraded, isTrue);
    expect(buf.state, CaptureState.active);
    expect(buf.frames.length, lessThanOrEqualTo(10));
  });
}
