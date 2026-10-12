/// Everything the tutor loads once at startup: the exported grading bundle
/// (ONNX graph + references + metadata + skeleton + grounded text), the
/// MediaPipe landmarkers, and the reference-clip index.
library;

import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:aslcv_core/aslcv_core.dart';
import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:mediapipe_ios/mediapipe_ios.dart';

class ReferenceClip {
  final String asset;
  final String videoId;
  final String signerId;

  const ReferenceClip(this.asset, this.videoId, this.signerId);
}

class Engine {
  final BundleGrader grader;
  final Skeleton skeleton;
  final MediaPipeLayout layout;
  final Map<String, ReferenceClip> clips;
  final String delegate; // what the live landmarkers actually run on
  final LearnerText text;
  final SentenceReferences sentenceRefs;
  late final Map<String, List<(String, String)>> minimalPairs = findMinimalPairs(meta.signs, signs);

  Engine._(this.grader, this.skeleton, this.layout, this.clips, this.delegate, this.text, this.sentenceRefs);

  BundleMeta get meta => grader.meta;

  /// Every sign with both a grading target and a reference clip to imitate --
  /// a sign is never offered without real footage to copy.
  List<String> get signs => [for (final s in meta.signs.keys) if (clips.containsKey(s)) s]..sort();

  static Future<Engine> load() async {
    final meta = BundleMeta.fromJson(jsonDecode(await rootBundle.loadString('assets/bundle/bundle.json')));
    final skeleton = Skeleton.fromJson(jsonDecode(await rootBundle.loadString('assets/bundle/skeleton.json')));
    final refBytes = await rootBundle.load('assets/bundle/${meta.referencesFile}');
    // Copy into a fresh buffer: an asset's ByteData need not be 4-byte aligned.
    final references = Float32List.sublistView(Uint8List.fromList(refBytes.buffer.asUint8List(
        refBytes.offsetInBytes, refBytes.lengthInBytes)));

    final session = await OnnxRuntime().createSessionFromAsset('assets/bundle/${meta.model}');
    Future<Map<String, Float32List>> run(Float32List x, int frames, Float32List tempo) async {
      final inputs = {
        'features': await OrtValue.fromList(x, [1, frames, meta.featureDim]),
        'tempo': await OrtValue.fromList(tempo, [1, 2]),
      };
      final out = await session.run(inputs);
      final result = <String, Float32List>{
        for (final name in meta.outputs)
          name: Float32List.fromList([for (final v in await out[name]!.asFlattenedList()) (v as num).toDouble()]),
      };
      for (final v in [...inputs.values, ...out.values]) {
        await v.dispose();
      }
      return result;
    }

    final index = jsonDecode(await rootBundle.loadString('assets/clips/clips.json')) as Map<String, dynamic>;
    final clips = index.map((sign, c) => MapEntry(
        sign, ReferenceClip('assets/clips/${c['file']}', c['video_id'] as String, c['signer_id'] as String)));

    final text = LearnerText.fromJson(jsonDecode(await rootBundle.loadString('assets/bundle/text.json')));
    final refIndex = jsonDecode(await rootBundle.loadString('assets/bundle/sentence_refs.json')) as Map<String, dynamic>;
    final refBlob = await rootBundle.load('assets/bundle/${refIndex['file']}');
    final sentenceRefs = SentenceReferences(refIndex, Float32List.sublistView(Uint8List.fromList(
        refBlob.buffer.asUint8List(refBlob.offsetInBytes, refBlob.lengthInBytes))));
    final delegate = await MediapipeIos.init(
        poseAsset: 'assets/mediapipe/pose_landmarker_full.task', handAsset: 'assets/mediapipe/hand_landmarker.task');
    return Engine._(BundleGrader(meta, references, run), skeleton, MediaPipeLayout(skeleton), clips, delegate, text, sentenceRefs);
  }

  /// The plugin's detection result -> one raw pose in the bundle's layout.
  Pose poseOf(Map<String, dynamic> r) => layout.toPose(
        width: r['width'] as int,
        height: r['height'] as int,
        pose: r['pose'] == null ? null : [for (final v in r['pose'] as List) (v as num).toDouble()],
        hands: [
          for (final h in (r['hands'] as List).cast<Map>())
            MediaPipeHand(h['label'] as String, (h['score'] as num).toDouble(),
                [for (final v in h['points'] as List) (v as num).toDouble()]),
        ],
      );
}

/// Worst landmark disagreement (pixels) between the phone's landmarkers and
/// Python's, on the same still frame. `gpu` picks the GPU-delegate
/// landmarkers -- the ones live grading uses.
Future<String> landmarkParity({required bool gpu}) async {
  final png = (await rootBundle.load('assets/parity/parity_frame.png')).buffer.asUint8List();
  final want = jsonDecode(await rootBundle.loadString('assets/parity/parity_expected.json'));
  final got = await MediapipeIos.detectImage(png, gpu: gpu);
  final w = (want['width'] as num).toDouble(), h = (want['height'] as num).toDouble();

  double worstPx(List a, List b, int stride) {
    var m = 0.0;
    for (var i = 0; i + 1 < b.length; i += stride) {
      final dx = ((a[i] as num) - (b[i] as num)) * w, dy = ((a[i + 1] as num) - (b[i + 1] as num)) * h;
      m = math.max(m, dx * dx + dy * dy);
    }
    return math.sqrt(m);
  }

  final lines = <String>[];
  final poseGot = got['pose'] as List?;
  lines.add(poseGot == null ? 'pose MISSING' : 'pose ${worstPx(poseGot, want['pose'] as List, 3).toStringAsFixed(2)} px');
  final handsGot = (got['hands'] as List).cast<Map>();
  for (final hw in (want['hands'] as List).cast<Map>()) {
    final match = handsGot.where((g) => g['label'] == hw['label']).toList();
    lines.add(match.isEmpty
        ? '${hw['label']} hand MISSING'
        : '${hw['label']} hand ${worstPx(match.first['points'] as List, hw['points'] as List, 2).toStringAsFixed(2)} px, '
            'score ${(match.first['score'] as num).toStringAsFixed(3)} vs ${(hw['score'] as num).toStringAsFixed(3)}');
  }
  return lines.join('\n');
}
