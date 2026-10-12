// Sentence grading's Dart port: dtwAlign on hand-checkable cases, then the
// whole alignAndGrade chain against Python's align_and_grade on synthetic
// continuous attempts (models/mobile/fixtures/alignment.json, from
// scripts/export_sentence_references.py), replaying the recorded ONNX outputs
// per segment as parity_test.dart does for single clips.
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:aslcv_core/aslcv_core.dart';
import 'package:test/test.dart';

final bundle = Platform.environment['ASLCV_BUNDLE'] ?? '../../models/mobile';

Float32List f32(String b64) => Float32List.sublistView(Uint8List.fromList(base64.decode(b64)));

List<Pose> posesOf(Map<String, dynamic> input) {
  final kp = f32(input['keypoints'] as String), sc = f32(input['scores'] as String);
  final k = input['points'] as int;
  return [
    for (var t = 0; t < (input['frames'] as int); t++)
      Pose(Float32List.sublistView(kp, t * 2 * k, (t + 1) * 2 * k), Float32List.sublistView(sc, t * k, (t + 1) * k)),
  ];
}

void main() {
  group('dtwAlign', () {
    test('identical sequences align on the diagonal at zero cost', () {
      final a = Float32List.fromList([0, 1, 2, 3]);
      final (d, path) = dtwAlign(a, 4, a, 4, 1);
      expect(d, 0);
      expect(path, [(0, 0), (1, 1), (2, 2), (3, 3)]);
    });

    test('a held frame warps onto one reference frame', () {
      final (d, path) = dtwAlign(Float32List.fromList([0, 0, 0, 5]), 4, Float32List.fromList([0, 5]), 2, 1);
      expect(d, 0);
      expect(path, [(0, 0), (1, 0), (2, 0), (3, 1)]);
    });

    test('segment ranges cover every gloss', () {
      final ranges = segmentRanges([(0, 0), (1, 0), (2, 1), (3, 2), (4, 2)], [0, 0, 1], 2);
      expect(ranges, [(0, 3), (3, 5)]);
    });
  });

  final fixture = File('$bundle/fixtures/alignment.json');
  if (!fixture.existsSync()) {
    test('alignment parity', () {}, skip: 'no fixture at ${fixture.path}');
    return;
  }
  final skeleton = Skeleton.fromJson(jsonDecode(File('$bundle/skeleton.json').readAsStringSync()));
  final meta = BundleMeta.fromJson(jsonDecode(File('$bundle/bundle.json').readAsStringSync()));
  final references = Float32List.sublistView(File('$bundle/${meta.referencesFile}').readAsBytesSync());
  final refIndex = jsonDecode(File('$bundle/sentence_refs.json').readAsStringSync()) as Map<String, dynamic>;
  final refs = SentenceReferences(
      refIndex, Float32List.sublistView(File('$bundle/${refIndex['file']}').readAsBytesSync()));

  for (final s in (jsonDecode(fixture.readAsStringSync())['sentences'] as List).cast<Map<String, dynamic>>()) {
    final signs = (s['signs'] as List).cast<String>();
    test('align and grade: ${signs.join(' ')}', () async {
      final exp = s['expected'] as Map<String, dynamic>;
      final segments = (exp['segments'] as List).cast<Map<String, dynamic>>();
      var call = 0;
      final grader = BundleGrader(meta, references, (x, frames, tempo) async {
        final outputs = segments[call++]['outputs'] as Map;
        return outputs.map((k, v) => MapEntry(k as String,
            Float32List.fromList([for (final e in v as List) (e as num).toDouble()])));
      });
      final got = await alignAndGrade(grader, skeleton, refs, posesOf(s['input']), signs);

      expect(got.distance, closeTo((exp['distance'] as num).toDouble(), 1e-4));
      expect([for (final (a, b) in got.rawRanges) [a, b]], exp['raw_ranges']);
      expect(call, signs.length);
      for (var g = 0; g < signs.length; g++) {
        final want = segments[g], word = got.words[g];
        expect(word.targetSign, want['sign']);
        expect([word.frameRange.$1, word.frameRange.$2], want['range']);
        expect(word.grade.fidelity, closeTo((want['fidelity'] as num).toDouble(), 1e-4));
        (want['parameters'] as Map).forEach((p, w) {
          final v = word.grade.parameters[p]!;
          expect([v.predicted, v.target, v.correct], [w['predicted'], w['target'], w['correct']], reason: '${want['sign']}/$p');
        });
      }
    });
  }
}
