/// Port of src/aslcv/features.py: FeaturePipeline.assemble,
/// hand_motion_energy, motion_active_span and live_capture_span.
///
/// Division of labor matches the Python: the normalizer decides which frame
/// each point is in; this module only selects points, attaches confidence,
/// appends velocity, and stacks frames. Presence is `score > 0` -- never a
/// coordinate test.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'f32.dart';
import 'normalizer.dart';
import 'skeleton.dart';

const _blockOrder = [Block.global, Block.leftHand, Block.rightHand];

/// The bundle's pipeline_args (the grader checkpoint's training-time config).
class PipelineArgs {
  final bool face;
  final bool legsFeet;
  final String confidence; // "graded" | "binary"
  final double binaryThreshold;
  final bool velocity;
  final bool depthProxies;
  final bool trimToMotion;
  final bool localHand;

  const PipelineArgs({
    required this.face,
    required this.legsFeet,
    required this.confidence,
    required this.binaryThreshold,
    required this.velocity,
    required this.depthProxies,
    required this.trimToMotion,
    required this.localHand,
  });

  factory PipelineArgs.fromJson(Map<String, dynamic> j) => PipelineArgs(
        face: j['face'] as bool,
        legsFeet: j['legs_feet'] as bool,
        confidence: j['confidence'] as String,
        binaryThreshold: (j['binary_threshold'] as num).toDouble(),
        velocity: j['velocity'] as bool,
        depthProxies: j['depth_proxies'] as bool,
        trimToMotion: j['trim_to_motion'] as bool,
        localHand: j['local_hand'] as bool,
      );
}

class FeatureClip {
  final Float32List features; // row-major (frames, dim)
  final int frames;
  final int dim;

  const FeatureClip(this.features, this.frames, this.dim);
}

void _checkSupported(PipelineArgs args) {
  // Fail closed on any config this port does not implement, rather than
  // silently producing features the grader was never trained on.
  if (args.depthProxies) throw UnsupportedError('depth_proxies=true is not supported by this port');
  if (args.trimToMotion) throw UnsupportedError('trim_to_motion=true is not supported by this port');
  if (!args.localHand) throw UnsupportedError('local_hand=false is not supported by this port');
}

/// Rows kept within a block: the global block drops face / legs_feet unless enabled.
List<int> _keptRows(Block block, int nPoints, Skeleton skeleton, PipelineArgs args) {
  final all = List<int>.generate(nPoints, (i) => i);
  if (block != Block.global) return all;
  final drop = <int>{
    if (!args.face) ...skeleton.region('face'),
    if (!args.legsFeet) ...skeleton.region('legs_feet'),
  };
  return all.where((i) => !drop.contains(i)).toList();
}

FeatureClip assemble(List<Pose> poses, Skeleton skeleton, PipelineArgs args) {
  _checkSupported(args);
  if (poses.isEmpty) throw ArgumentError('assemble() needs at least one frame');
  final t = poses.length;
  final channels = args.velocity ? 5 : 3;

  final normalized = [for (final p in poses) normalize(p, skeleton)];
  final first = normalized.first.blocks;
  final selected = <int>[
    for (final b in _blockOrder)
      if (first[b] case (final start, final stop))
        for (final r in _keptRows(b, stop - start, skeleton, args)) start + r,
  ]; // absolute rows into each NormalizedPose
  final p = selected.length;
  final dim = p * channels;

  final out = Float32List(t * dim);
  final posX = Float32List(t * p);
  final posY = Float32List(t * p);
  final presence = Uint8List(t * p);

  for (var ti = 0; ti < t; ti++) {
    final np = normalized[ti];
    for (var i = 0; i < p; i++) {
      final row = selected[i];
      final score = np.scores[row];
      if (!(score > 0)) continue; // absent: all channels stay 0
      final x = np.keypoints[2 * row];
      final y = np.keypoints[2 * row + 1];
      final conf = args.confidence == 'graded' ? score : (score > args.binaryThreshold ? 1.0 : 0.0);
      final o = ti * dim + i * channels;
      out[o] = x;
      out[o + 1] = y;
      out[o + 2] = conf;
      posX[ti * p + i] = x;
      posY[ti * p + i] = y;
      presence[ti * p + i] = 1;
    }
  }

  if (args.velocity) {
    // first-frame delta stays 0; a delta touching an absence is zeroed, not
    // differenced against the zero-fill (real -> 0 -> real is not motion)
    for (var ti = 1; ti < t; ti++) {
      for (var i = 0; i < p; i++) {
        if (presence[ti * p + i] == 0 || presence[(ti - 1) * p + i] == 0) continue;
        final o = ti * dim + i * channels;
        out[o + 3] = f(posX[ti * p + i] - posX[(ti - 1) * p + i]);
        out[o + 4] = f(posY[ti * p + i] - posY[(ti - 1) * p + i]);
      }
    }
  }
  return FeatureClip(out, t, dim);
}

/// Per-frame mean velocity magnitude over both hands' points in the
/// local-hand frame, averaged over PRESENT points only (an undetected hand is
/// not a resting one). Same signal the live capture buffer and the
/// training-data trim use.
Float32List handMotionEnergy(List<Pose> poses, Skeleton skeleton) {
  final t = poses.length;
  if (t == 0) throw ArgumentError('handMotionEnergy() needs at least one frame');
  final normalized = [for (final p in poses) normalize(p, skeleton)];
  final handRows = <int>[
    for (final b in const [Block.leftHand, Block.rightHand])
      if (normalized.first.blocks[b] case (final start, final stop))
        for (var r = start; r < stop; r++) r,
  ];
  if (handRows.isEmpty) throw ArgumentError('handMotionEnergy() needs left_hand/right_hand blocks');

  final energy = Float32List(t);
  for (var ti = 1; ti < t; ti++) {
    final cur = normalized[ti];
    final prev = normalized[ti - 1];
    var sum = 0.0;
    var count = 0;
    for (final r in handRows) {
      final here = cur.scores[r] > 0;
      if (here) count++;
      if (!here || !(prev.scores[r] > 0)) continue;
      final dx = f(cur.keypoints[2 * r] - prev.keypoints[2 * r]);
      final dy = f(cur.keypoints[2 * r + 1] - prev.keypoints[2 * r + 1]);
      sum = f(sum + f(math.sqrt(f(f(dx * dx) + f(dy * dy)))));
    }
    if (count > 0) energy[ti] = f(sum / count);
  }
  return energy;
}

/// [start, stop) covering every frame with energy > threshold, padded; the
/// full range if none.
(int, int) motionActiveSpan(Float32List energy, double threshold, [int padFrames = 0]) {
  var first = -1;
  var last = -1;
  for (var i = 0; i < energy.length; i++) {
    if (energy[i] > threshold) {
      if (first < 0) first = i;
      last = i;
    }
  }
  if (first < 0) return (0, energy.length);
  return (math.max(0, first - padFrames), math.min(energy.length, last + 1 + padFrames));
}

/// Approximation of what the live capture buffer keeps for a clean
/// rest -> sign -> rest clip: the motion-active span extended by `preroll`
/// frames before and `settleFrames` after.
(int, int) liveCaptureSpan(Float32List energy, double threshold, int preroll, int settleFrames) {
  final (start, stop) = motionActiveSpan(energy, threshold);
  return (math.max(0, start - preroll), math.min(energy.length, stop + settleFrames));
}
