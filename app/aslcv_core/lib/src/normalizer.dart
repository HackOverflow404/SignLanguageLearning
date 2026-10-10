/// Port of src/aslcv/normalizer/shoulder.py's ShoulderNormalizer (local_hand=True).
///
/// Emits a GLOBAL block (every point, shoulder-midpoint origin, shoulder-width
/// scale) followed by per-hand LOCAL blocks (each hand's points, wrist origin,
/// wrist->middle-knuckle scale). A degenerate frame (missing anchor or a scale
/// at or below EPS) zero-fills its block instead of dividing. Rotation is
/// deliberately NOT normalized -- a tilted head is grammatical in ASL.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'f32.dart';
import 'skeleton.dart';

const _eps = 1e-6;

enum Block { global, leftHand, rightHand }

class NormalizedPose {
  final Float32List keypoints; // interleaved (x, y), length 2N
  final Float32List scores; // length N
  final Map<Block, (int, int)> blocks; // [start, stop) point rows

  const NormalizedPose(this.keypoints, this.scores, this.blocks);
}

/// float32 Euclidean distance between points i and j of `kp`.
double _dist32(Float32List kp, int i, int j) {
  final dx = f(kp[2 * i] - kp[2 * j]);
  final dy = f(kp[2 * i + 1] - kp[2 * j + 1]);
  return f(math.sqrt(f(f(dx * dx) + f(dy * dy))));
}

NormalizedPose normalize(Pose pose, Skeleton skeleton) {
  final kp = pose.keypoints;
  final sc = pose.scores;
  final nTotal = sc.length;
  final hands = [
    for (final (block, side) in [(Block.leftHand, 'left'), (Block.rightHand, 'right')])
      (
        block: block,
        idx: skeleton.region('${side}_hand'),
        wrist: skeleton.anchor('${side}_hand_wrist'),
        mcp: skeleton.anchor('${side}_hand_middle_mcp'),
      ),
  ];
  final nRows = nTotal + hands.fold<int>(0, (n, h) => n + h.idx.length);
  final outKp = Float32List(2 * nRows);
  final outSc = Float32List(nRows);
  final blocks = <Block, (int, int)>{};

  // -- global block: every point in the shoulder frame
  final ls = skeleton.anchor('left_shoulder');
  final rs = skeleton.anchor('right_shoulder');
  final width = _dist32(kp, ls, rs);
  if (sc[ls] > 0 && sc[rs] > 0 && width > _eps) {
    final ox = f(f(kp[2 * ls] + kp[2 * rs]) / 2);
    final oy = f(f(kp[2 * ls + 1] + kp[2 * rs + 1]) / 2);
    for (var i = 0; i < nTotal; i++) {
      outKp[2 * i] = f(f(kp[2 * i] - ox) / width);
      outKp[2 * i + 1] = f(f(kp[2 * i + 1] - oy) / width);
      outSc[i] = sc[i];
    }
  }
  blocks[Block.global] = (0, nTotal);

  // -- local hand blocks: each hand in its own wrist frame
  var cursor = nTotal;
  for (final h in hands) {
    if (h.idx.isEmpty) continue;
    final scale = _dist32(kp, h.wrist, h.mcp);
    if (sc[h.wrist] > 0 && sc[h.mcp] > 0 && scale > _eps) {
      final wx = kp[2 * h.wrist];
      final wy = kp[2 * h.wrist + 1];
      for (var r = 0; r < h.idx.length; r++) {
        final p = h.idx[r];
        outKp[2 * (cursor + r)] = f(f(kp[2 * p] - wx) / scale);
        outKp[2 * (cursor + r) + 1] = f(f(kp[2 * p + 1] - wy) / scale);
        outSc[cursor + r] = sc[p];
      }
    }
    blocks[h.block] = (cursor, cursor + h.idx.length);
    cursor += h.idx.length;
  }
  return NormalizedPose(outKp, outSc, blocks);
}
