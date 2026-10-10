import { assemble, handMotionEnergy } from "./features";
import type { BundleGrader, Grade } from "./grader";
import type { Pose, Skeleton } from "./skeleton";
import { tempoFeatures } from "./tempo";

export * from "./capture";
export * from "./features";
export * from "./grader";
export * from "./normalizer";
export * from "./skeleton";
export * from "./tempo";

/**
 * Grade one captured attempt (already bounded by live capture -- not re-trimmed
 * here) against a known target sign: the same chain as EmbeddingGrader's
 * in-memory path, assemble -> motion energy -> tempo -> bundle model.
 */
export async function gradeAttempt(
  grader: BundleGrader,
  skeleton: Skeleton,
  poses: Pose[],
  targetSign: string,
): Promise<Grade> {
  const clip = assemble(poses, skeleton, grader.meta.pipeline_args);
  const tempo = tempoFeatures(handMotionEnergy(poses, skeleton));
  return grader.grade(clip.features, clip.frames, tempo, targetSign);
}
