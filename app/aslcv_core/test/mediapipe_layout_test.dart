// MediaPipe landmarker output -> combined keypoint layout, against
// MediaPipePoseExtractor(mirrored=False) on the same real frames
// (scripts/export_layout_fixture.py), plus the duplicate-label rule.
//
// Needs the exported bundle (models/mobile/, or $ASLCV_BUNDLE); the golden
// cases are skipped if it is absent.
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:aslcv_core/aslcv_core.dart';
import 'package:test/test.dart';

final bundle = Platform.environment['ASLCV_BUNDLE'] ?? '../../models/mobile';

Float32List f32(String b64) => Float32List.sublistView(Uint8List.fromList(base64.decode(b64)));
List<double> doubles(List v) => [for (final x in v) (x as num).toDouble()];

MediaPipeHand handOf(Map h) => MediaPipeHand(h['label'] as String, (h['score'] as num).toDouble(), doubles(h['points'] as List));

void main() {
  final skeletonFile = File('$bundle/skeleton.json');
  final fixtureFile = File('$bundle/fixtures/mediapipe_layout.json');
  if (!skeletonFile.existsSync() || !fixtureFile.existsSync()) {
    test('mediapipe layout', () {}, skip: 'no bundle/layout fixture at $bundle');
    return;
  }
  final skeleton = Skeleton.fromJson(jsonDecode(skeletonFile.readAsStringSync()));
  final layout = MediaPipeLayout(skeleton);
  final cases = (jsonDecode(fixtureFile.readAsStringSync())['cases'] as List).cast<Map<String, dynamic>>();

  test('has cases covering one and two hands', () {
    final counts = {for (final c in cases) (c['input']['hands'] as List).length};
    expect(counts, containsAll([1, 2]));
  });

  for (final c in cases) {
    test('matches the Python extractor on ${c['video_id']}', () {
      final input = c['input'] as Map<String, dynamic>;
      final pose = layout.toPose(
        width: input['width'] as int,
        height: input['height'] as int,
        pose: input['pose'] == null ? null : doubles(input['pose'] as List),
        hands: [for (final h in input['hands'] as List) handOf(h as Map)],
      );
      final wantKp = f32(c['expected']['keypoints'] as String);
      final wantSc = f32(c['expected']['scores'] as String);
      // Python computes x * width in float32 from float32 landmarks; the
      // plugin's landmarks arrive as doubles, so allow float32 rounding.
      for (var i = 0; i < wantKp.length; i++) {
        expect(pose.keypoints[i], closeTo(wantKp[i], 1e-3 + 1e-6 * wantKp[i].abs()), reason: 'keypoint value $i');
      }
      for (var i = 0; i < wantSc.length; i++) {
        expect(pose.scores[i], closeTo(wantSc[i], 1e-6), reason: 'score $i');
      }
    });
  }

  test('a duplicated handedness label keeps the higher-scoring hand', () {
    final low = MediaPipeHand('Left', 0.6, List.filled(42, 0.1));
    final high = MediaPipeHand('Left', 0.9, List.filled(42, 0.2));
    for (final order in [[low, high], [high, low]]) {
      final p = layout.toPose(width: 100, height: 100, hands: order);
      final wrist = skeleton.anchor('left_hand_wrist');
      expect(p.scores[wrist], closeTo(0.9, 1e-6));
      expect(p.keypoints[2 * wrist], closeTo(20, 1e-4));
      expect(p.scores[skeleton.anchor('right_hand_wrist')], 0);
    }
  });

  test('the face mesh stays absent', () {
    final c = cases.first['input'] as Map<String, dynamic>;
    final p = layout.toPose(width: 640, height: 480, pose: doubles(c['pose'] as List),
        hands: [for (final h in c['hands'] as List) handOf(h as Map)]);
    final poseNames = {for (var i = 0; i < 33; i++) 'pose_$i'};
    for (final i in skeleton.region('face')) {
      if (!poseNames.contains(skeleton.names[i])) expect(p.scores[i], 0);
    }
  });
}
