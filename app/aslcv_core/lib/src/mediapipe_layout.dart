/// MediaPipe pose + hand landmarker output -> one raw [Pose] in the bundle
/// skeleton's combined layout, the same placement MediaPipePoseExtractor
/// (mirrored=False) does in Python:
///
///   * coordinates in pixels (normalized x * width, y * height);
///   * pose score = BlazePose visibility, per point;
///   * hand score = that hand's handedness confidence, on every point;
///   * MediaPipe's "Left"/"Right" label used as-is -- anatomically correct on
///     an UN-mirrored frame, so the caller must un-mirror a selfie camera
///     frame before detection, never after;
///   * two hands claiming the same label keep the higher-scoring one;
///   * the face block stays absent (score 0): the phone runs no face model,
///     and the features never read the face region.
///
/// Slots are found by point NAME in skeleton.json, never by hardcoded offset.
library;

import 'dart:typed_data';

import 'skeleton.dart';

class MediaPipeHand {
  final String label; // MediaPipe's raw handedness category, "Left" or "Right"
  final double score;
  final List<double> points; // 21 x (x, y), normalized

  const MediaPipeHand(this.label, this.score, this.points);
}

class MediaPipeLayout {
  static const poseCount = 33;
  static const handCount = 21;

  final Skeleton skeleton;
  final int _pose;
  final int _leftHand;
  final int _rightHand;

  MediaPipeLayout._(this.skeleton, this._pose, this._leftHand, this._rightHand);

  /// Fails closed if the skeleton does not carry MediaPipe's pose and hand
  /// blocks as contiguous runs of named points.
  factory MediaPipeLayout(Skeleton skeleton) {
    int block(String prefix, int count) {
      final start = skeleton.names.indexOf('${prefix}_0');
      if (start < 0) throw ArgumentError('skeleton has no "${prefix}_0" point');
      for (var i = 0; i < count; i++) {
        if (start + i >= skeleton.names.length || skeleton.names[start + i] != '${prefix}_$i') {
          throw ArgumentError('skeleton\'s "$prefix" points are not a contiguous block of $count');
        }
      }
      return start;
    }

    return MediaPipeLayout._(
        skeleton, block('pose', poseCount), block('left_hand', handCount), block('right_hand', handCount));
  }

  /// `pose` is 33 x (x, y, visibility) normalized, or null when no body was found.
  Pose toPose({required int width, required int height, List<double>? pose, List<MediaPipeHand> hands = const []}) {
    final k = skeleton.names.length;
    final keypoints = Float32List(2 * k);
    final scores = Float32List(k);

    if (pose != null) {
      if (pose.length != 3 * poseCount) throw ArgumentError('pose has ${pose.length} values, expected ${3 * poseCount}');
      for (var i = 0; i < poseCount; i++) {
        keypoints[2 * (_pose + i)] = pose[3 * i] * width;
        keypoints[2 * (_pose + i) + 1] = pose[3 * i + 1] * height;
        scores[_pose + i] = pose[3 * i + 2];
      }
    }

    final best = <int, double>{};
    for (final h in hands) {
      if (h.points.length != 2 * handCount) {
        throw ArgumentError('hand has ${h.points.length} values, expected ${2 * handCount}');
      }
      final offset = h.label == 'Left' ? _leftHand : _rightHand;
      if ((best[offset] ?? -1.0) >= h.score) continue;
      best[offset] = h.score;
      for (var i = 0; i < handCount; i++) {
        keypoints[2 * (offset + i)] = h.points[2 * i] * width;
        keypoints[2 * (offset + i) + 1] = h.points[2 * i + 1] * height;
        scores[offset + i] = h.score;
      }
    }
    return Pose(keypoints, scores);
  }
}
