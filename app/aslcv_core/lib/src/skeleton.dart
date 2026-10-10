/// Keypoint topology, loaded from the bundle's skeleton.json (exported from
/// the Python `Skeleton`). Points are always located by MEANING -- an anchor
/// name or a region name -- never by a hardcoded index, so this code stays
/// correct if the extractor's point layout ever changes.
library;

import 'dart:typed_data';

class Skeleton {
  final List<String> names;
  final Map<String, int> anchors;
  final Map<String, List<int>> regions;

  const Skeleton({required this.names, required this.anchors, required this.regions});

  factory Skeleton.fromJson(Map<String, dynamic> json) => Skeleton(
        names: (json['names'] as List).cast<String>(),
        anchors: (json['anchors'] as Map).map((k, v) => MapEntry(k as String, v as int)),
        regions: (json['regions'] as Map).map((k, v) => MapEntry(k as String, (v as List).cast<int>())),
      );

  int anchor(String name) {
    final index = anchors[name];
    if (index == null) throw ArgumentError('skeleton has no anchor "$name"');
    return index;
  }

  /// A region's point indices; an absent region is an empty selection, not an error.
  List<int> region(String name) => regions[name] ?? const [];
}

/// One frame of raw extractor output: K points as interleaved (x, y) pixels,
/// plus K scores. Presence = score > 0.
class Pose {
  final Float32List keypoints; // length 2K
  final Float32List scores; // length K

  const Pose(this.keypoints, this.scores);
}
