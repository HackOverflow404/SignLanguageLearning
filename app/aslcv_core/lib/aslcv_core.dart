/// The phone-side grading pipeline: raw keypoints -> features -> tempo ->
/// bundle model -> per-parameter verdicts against a known target sign.
library;

import 'src/features.dart';
import 'src/grader.dart';
import 'src/skeleton.dart';
import 'src/tempo.dart';

export 'src/alignment.dart';
export 'src/capture.dart';
export 'src/coaching.dart';
export 'src/features.dart';
export 'src/grader.dart';
export 'src/learner.dart';
export 'src/mediapipe_layout.dart';
export 'src/normalizer.dart';
export 'src/skeleton.dart';
export 'src/tempo.dart';

/// Grade one captured attempt (already bounded by live capture -- not
/// re-trimmed here) against a known target sign: the same chain as
/// EmbeddingGrader's in-memory path, assemble -> motion energy -> tempo ->
/// bundle model.
Future<Grade> gradeAttempt(BundleGrader grader, Skeleton skeleton, List<Pose> poses, String targetSign) {
  final clip = assemble(poses, skeleton, grader.meta.pipelineArgs);
  final tempo = tempoFeatures(handMotionEnergy(poses, skeleton));
  return grader.grade(clip.features, clip.frames, tempo, targetSign);
}
