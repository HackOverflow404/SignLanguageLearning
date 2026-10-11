// Cross-language parity: every stage of the Dart port against the golden
// fixtures exported by scripts/export_mobile.py (Python's own outputs on real
// ASL Citizen val clips). The ONNX graph itself is not run here -- `dart test`
// has no ONNX Runtime -- so the model runner replays Python's recorded
// outputs; the graph's own parity is checked on the phone. Everything around
// it (feature math, standardization, distance, verdict logic) is checked.
//
// Needs the exported bundle (models/mobile/, or $ASLCV_BUNDLE); skipped if absent.
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:aslcv_core/aslcv_core.dart';
import 'package:test/test.dart';

final bundle = Platform.environment['ASLCV_BUNDLE'] ?? '../../models/mobile';

Float32List f32(String b64) => Float32List.sublistView(Uint8List.fromList(base64.decode(b64)));
Float32List f32list(List v) => Float32List.fromList([for (final x in v) (x as num).toDouble()]);

/// |a - b| <= atol + rtol * |b|, elementwise; reports the worst offender.
void expectClose(List<double> actual, List<double> expected, double atol, [double rtol = 1e-5]) {
  expect(actual.length, expected.length);
  var worst = 0.0;
  var at = -1;
  for (var i = 0; i < expected.length; i++) {
    final excess = (actual[i] - expected[i]).abs() - (atol + rtol * expected[i].abs());
    if (excess > worst) {
      worst = excess;
      at = i;
    }
  }
  expect(at, -1, reason: at < 0 ? null : 'index $at: ${actual[at]} vs ${expected[at]}');
}

List<Pose> posesOf(Map<String, dynamic> input) {
  final kp = f32(input['keypoints'] as String);
  final sc = f32(input['scores'] as String);
  final k = input['points'] as int;
  return [
    for (var t = 0; t < (input['frames'] as int); t++)
      Pose(Float32List.sublistView(kp, t * 2 * k, (t + 1) * 2 * k), Float32List.sublistView(sc, t * k, (t + 1) * k)),
  ];
}

void main() {
  final available = File('$bundle/bundle.json').existsSync() && Directory('$bundle/fixtures').existsSync();
  if (!available) {
    test('golden-fixture parity', () {}, skip: 'no bundle at $bundle');
    return;
  }

  final skeleton = Skeleton.fromJson(jsonDecode(File('$bundle/skeleton.json').readAsStringSync()));
  final meta = BundleMeta.fromJson(jsonDecode(File('$bundle/bundle.json').readAsStringSync()));
  final references = Float32List.sublistView(File('$bundle/${meta.referencesFile}').readAsBytesSync());
  final fixtures = (Directory('$bundle/fixtures').listSync().whereType<File>().where((f) => f.uri.pathSegments.last.startsWith('clip_') && f.path.endsWith('.json')).toList()
        ..sort((a, b) => a.path.compareTo(b.path)))
      .map((f) => jsonDecode(f.readAsStringSync()) as Map<String, dynamic>)
      .toList();

  test('has fixtures to check', () => expect(fixtures, isNotEmpty));

  for (final fx in fixtures) {
    group('${fx['sign']} (${fx['video_id']})', () {
      final exp = fx['expected'] as Map<String, dynamic>;
      final poses = posesOf(fx['input'] as Map<String, dynamic>);
      final fullEnergy = handMotionEnergy(poses, skeleton);
      final span = liveCaptureSpan(fullEnergy, meta.motionThreshold, meta.livePreroll, meta.liveSettleFrames);
      final trimmed = poses.sublist(span.$1, span.$2);
      final outputs = (exp['outputs'] as Map).map((k, v) => MapEntry(k as String, f32list(v as List)));

      test('motion energy over the raw clip', () => expectClose(fullEnergy, f32(exp['full_energy']), 1e-6));
      test('live capture trim span', () => expect([span.$1, span.$2], exp['live_capture_span']));

      test('assembled features', () {
        final clip = assemble(trimmed, skeleton, meta.pipelineArgs);
        expect([clip.frames, clip.dim], exp['feature_shape']);
        expectClose(clip.features, f32(exp['features']), 1e-5);
      });

      test('tempo features', () {
        final energy = handMotionEnergy(trimmed, skeleton);
        expectClose(energy, f32(exp['energy']), 1e-6);
        final (lag, height) = tempoFeatures(energy);
        expectClose([lag, height], f32list(exp['tempo']), 1e-5);
      });

      test('end-to-end verdicts: raw keypoints -> grade', () async {
        // Replays Python's recorded model outputs, after checking the runner
        // received exactly the standardized features Python would feed it.
        final expectedRaw = f32(exp['features']);
        final grader = BundleGrader(meta, references, (x, frames, tempo) async {
          final dim = meta.featureDim;
          expectClose(x, [for (var i = 0; i < x.length; i++) (expectedRaw[i] - meta.mean[i % dim]) / meta.std[i % dim]], 1e-4);
          expectClose(tempo, f32list(exp['tempo']), 1e-5);
          return outputs;
        });
        for (final MapEntry(key: target, value: want) in (exp['grades'] as Map<String, dynamic>).entries) {
          final got = await gradeAttempt(grader, skeleton, trimmed, target);
          expect(got.fidelity, closeTo((want['fidelity'] as num).toDouble(), 1e-4), reason: target);
          for (final MapEntry(key: p, value: w) in (want['parameters'] as Map<String, dynamic>).entries) {
            final g = got.parameters[p]!;
            expect([g.predicted, g.target, g.correct], [w['predicted'], w['target'], w['correct']], reason: '$target/$p');
            expect(g.confidence, closeTo((w['confidence'] as num).toDouble(), 1e-4), reason: '$target/$p');
          }
        }
      });
    });
  }
}
