/// Port of dtw_align (grading/dtw_grader.py) and align_and_grade
/// (grading/alignment.py): forced alignment of one continuous attempt
/// against a KNOWN gloss sequence, then each word graded against its own
/// target exactly as an isolated attempt is. The target sequence always
/// comes from the sentence engine, so this never has to recognise which
/// signs were made -- only where each expected one is.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../aslcv_core.dart';

/// Per-sign standardized reference features (scripts/export_sentence_references.py):
/// a sentence's reference is these concatenated in gloss order.
class SentenceReferences {
  final int dim;
  final Float32List _rows; // (frames, dim) row-major
  final Map<String, (int, int)> spans; // sign -> (start, count)

  SentenceReferences(Map<String, dynamic> json, this._rows)
      : dim = json['dim'] as int,
        spans = {
          for (final s in json['signs'] as List) s['sign'] as String: (s['start'] as int, s['count'] as int),
        } {
    if (_rows.length != (json['rows'] as int) * dim) throw StateError('sentence_refs size does not match its index');
  }

  /// (features, frames, per-frame gloss index) for the glosses in order.
  (Float32List, int, List<int>) compose(List<String> glosses) {
    final parts = <Float32List>[];
    final glossIndex = <int>[];
    for (var g = 0; g < glosses.length; g++) {
      final span = spans[glosses[g]];
      if (span == null) throw ArgumentError('no reference for "${glosses[g]}"');
      parts.add(Float32List.sublistView(_rows, span.$1 * dim, (span.$1 + span.$2) * dim));
      glossIndex.addAll(List.filled(span.$2, g));
    }
    final out = Float32List(glossIndex.length * dim);
    var o = 0;
    for (final p in parts) {
      out.setAll(o, p);
      o += p.length;
    }
    return (out, glossIndex.length, glossIndex);
  }
}

/// Length-normalized DTW distance and the optimal warp path ((i, j) frame
/// pairs, increasing) between (n, dim) `a` and (m, dim) `b`. Same recurrence,
/// float64 local cost and tie-breaking as dtw_align; no band.
(double, List<(int, int)>) dtwAlign(Float32List a, int n, Float32List b, int m, int dim) {
  if (n == 0 || m == 0) return (double.infinity, const []);
  final cost = Float64List(n * m);
  for (var i = 0; i < n; i++) {
    for (var j = 0; j < m; j++) {
      var s = 0.0;
      for (var k = 0; k < dim; k++) {
        final d = a[i * dim + k] - b[j * dim + k];
        s += d * d;
      }
      cost[i * m + j] = math.sqrt(s);
    }
  }
  final w = m + 1;
  final dp = Float64List((n + 1) * w)..fillRange(0, (n + 1) * w, double.infinity);
  dp[0] = 0;
  for (var i = 1; i <= n; i++) {
    for (var j = 1; j <= m; j++) {
      var best = dp[(i - 1) * w + j];
      if (dp[(i - 1) * w + j - 1] < best) best = dp[(i - 1) * w + j - 1];
      if (dp[i * w + j - 1] < best) best = dp[i * w + j - 1];
      dp[i * w + j] = cost[(i - 1) * m + j - 1] + best;
    }
  }
  final total = dp[n * w + m];
  if (!total.isFinite) return (double.infinity, const []);
  final path = <(int, int)>[];
  var i = n, j = m;
  while (i > 0 && j > 0) {
    path.add((i - 1, j - 1));
    final diag = dp[(i - 1) * w + j - 1], up = dp[(i - 1) * w + j], left = dp[i * w + j - 1];
    final best = math.min(diag, math.min(up, left));
    if (diag <= best) {
      i--;
      j--;
    } else if (up <= best) {
      i--;
    } else {
      j--;
    }
  }
  return (total / (n + m), path.reversed.toList());
}

/// Each gloss's [start, stop) in the attempt: the min/max attempt frame the
/// warp path aligns to any of that gloss's reference frames.
List<(int, int)> segmentRanges(List<(int, int)> path, List<int> frameGlossIndex, int nGlosses) {
  final lo = List<int?>.filled(nGlosses, null), hi = List<int?>.filled(nGlosses, null);
  for (final (i, j) in path) {
    final g = frameGlossIndex[j];
    if (lo[g] == null || i < lo[g]!) lo[g] = i;
    if (hi[g] == null || i > hi[g]!) hi[g] = i;
  }
  if (lo.contains(null)) throw StateError('forced alignment left a gloss with no attempt frames');
  return [for (var g = 0; g < nGlosses; g++) (lo[g]!, hi[g]! + 1)];
}

class AlignedGrade {
  final String targetSign;
  final (int, int) frameRange; // [start, stop) into the attempt, after the segment's live trim
  final Grade grade;

  const AlignedGrade(this.targetSign, this.frameRange, this.grade);
}

class SentenceGrade {
  final double distance;
  final List<(int, int)> rawRanges; // before each segment's live trim
  final List<AlignedGrade> words;

  const SentenceGrade(this.distance, this.rawRanges, this.words);
}

/// Forced-align a continuous attempt against `glosses`' composed reference,
/// re-trim each segment to its own live capture span (a warp-path boundary
/// carries inter-word rest the model never saw in training), and grade each
/// segment against its own target.
Future<SentenceGrade> alignAndGrade(BundleGrader grader, Skeleton skeleton, SentenceReferences refs,
    List<Pose> attempt, List<String> glosses) async {
  final meta = grader.meta;
  final (reference, m, glossIndex) = refs.compose(glosses);
  final clip = assemble(attempt, skeleton, meta.pipelineArgs);
  final (distance, path) = dtwAlign(grader.standardize(clip.features), clip.frames, reference, m, refs.dim);
  if (path.isEmpty) throw ArgumentError('forced alignment failed -- empty attempt');
  final ranges = segmentRanges(path, glossIndex, glosses.length);
  final words = <AlignedGrade>[];
  for (var g = 0; g < glosses.length; g++) {
    final (start, stop) = ranges[g];
    final raw = attempt.sublist(start, stop);
    final (ts, te) = liveCaptureSpan(
        handMotionEnergy(raw, skeleton), meta.motionThreshold, meta.livePreroll, meta.liveSettleFrames);
    final grade = await gradeAttempt(grader, skeleton, raw.sublist(ts, te), glosses[g]);
    words.add(AlignedGrade(glosses[g], (start + ts, start + te), grade));
  }
  return SentenceGrade(distance, ranges, words);
}
