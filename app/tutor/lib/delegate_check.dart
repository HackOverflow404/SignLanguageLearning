/// Does the GPU landmark delegate change the GRADES, not just the pixels?
/// Training data was extracted on CPU; live grading runs on GPU for speed.
/// For a sample of the bundled reference clips (real ASL Citizen footage,
/// unflipped), this runs fresh VIDEO-mode landmarkers over every frame on
/// both delegates, grades each against the clip's own sign, and reports:
///   * landmark deviation on the points the features read (upper body, arms,
///     both hands), in pixels;
///   * how often each parameter's predicted value differs between delegates;
///   * each delegate's own match rate against the clip's target labels.
library;

import 'dart:math' as math;

import 'package:aslcv_core/aslcv_core.dart';
import 'package:mediapipe_ios/mediapipe_ios.dart';

import 'engine.dart';

const checkedParameters = ['handshape', 'major_location', 'minor_location', 'movement', 'repeated_movement'];

class DelegateReport {
  int clips = 0;
  final List<double> bodyPx = [], handPx = [];
  final Map<String, int> differ = {for (final p in checkedParameters) p: 0};
  final Map<String, (int, int)> cpuMatch = {}, gpuMatch = {}; // parameter -> (matched, judged)
  double fidelityDiff = 0;

  List<String> describe() {
    String px(List<double> v) {
      if (v.isEmpty) return 'no shared points';
      final s = [...v]..sort();
      final mean = v.reduce((a, b) => a + b) / v.length;
      return 'mean ${mean.toStringAsFixed(1)} px, 95th pct ${s[(0.95 * (s.length - 1)).round()].toStringAsFixed(1)} px';
    }

    String rate((int, int)? m) => m == null || m.$2 == 0 ? '-' : '${m.$1}/${m.$2}';
    return [
      '$clips clips, CPU vs GPU:',
      'Upper body + arms: ${px(bodyPx)}',
      'Hands: ${px(handPx)}',
      'Mean distance-to-reference change: ${(clips == 0 ? 0 : fidelityDiff / clips).toStringAsFixed(3)}',
      for (final p in checkedParameters)
        '$p: predictions differ on ${differ[p]}/$clips; matches target CPU ${rate(cpuMatch[p])}, GPU ${rate(gpuMatch[p])}',
    ];
  }
}

Future<DelegateReport> runDelegateCheck(Engine engine, {int every = 3, void Function(String)? progress}) async {
  final sk = engine.skeleton;
  final body = [...sk.region('body_upper'), ...sk.region('arms')];
  final hands = [...sk.region('left_hand'), ...sk.region('right_hand')];
  final report = DelegateReport();
  final signs = engine.signs;

  for (var i = 0; i < signs.length; i += every) {
    final sign = signs[i];
    progress?.call('clip ${report.clips + 1} of ${(signs.length / every).ceil()}: $sign');
    final asset = engine.clips[sign]!.asset;
    final cpu = [for (final r in await MediapipeIos.detectVideo(asset, gpu: false)) engine.poseOf(r)];
    final gpu = [for (final r in await MediapipeIos.detectVideo(asset, gpu: true)) engine.poseOf(r)];
    if (cpu.length != gpu.length || cpu.isEmpty) continue;

    for (var t = 0; t < cpu.length; t++) {
      for (final (points, out) in [(body, report.bodyPx), (hands, report.handPx)]) {
        for (final k in points) {
          if (cpu[t].scores[k] <= 0 || gpu[t].scores[k] <= 0) continue;
          final dx = cpu[t].keypoints[2 * k] - gpu[t].keypoints[2 * k];
          final dy = cpu[t].keypoints[2 * k + 1] - gpu[t].keypoints[2 * k + 1];
          out.add(math.sqrt(dx * dx + dy * dy));
        }
      }
    }

    final gc = await gradeAttempt(engine.grader, sk, cpu, sign);
    final gg = await gradeAttempt(engine.grader, sk, gpu, sign);
    report.clips++;
    report.fidelityDiff += (gc.fidelity - gg.fidelity).abs();
    for (final p in checkedParameters) {
      final vc = gc.parameters[p]!, vg = gg.parameters[p]!;
      if (vc.predicted != vg.predicted) report.differ[p] = report.differ[p]! + 1;
      for (final (v, tally) in [(vc, report.cpuMatch), (vg, report.gpuMatch)]) {
        if (v.correct == null) continue;
        final (m, n) = tally[p] ?? (0, 0);
        tally[p] = (m + (v.correct! ? 1 : 0), n + 1);
      }
    }
  }
  return report;
}
