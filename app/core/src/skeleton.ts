/**
 * Keypoint topology, loaded from the bundle's skeleton.json (exported from the
 * Python `Skeleton`). Points are always located by MEANING -- an anchor name or
 * a region name -- never by a hardcoded index, so this code stays correct if the
 * extractor's point layout ever changes.
 */
export interface Skeleton {
  names: string[];
  anchors: Record<string, number>;
  regions: Record<string, number[]>;
}

/** One frame of raw extractor output: K points as interleaved (x, y) pixels, plus K scores. */
export interface Pose {
  keypoints: Float32Array; // length 2K
  scores: Float32Array; // length K; presence = score > 0
}

export function anchor(skeleton: Skeleton, name: string): number {
  const index = skeleton.anchors[name];
  if (index === undefined) throw new Error(`skeleton has no anchor ${JSON.stringify(name)}`);
  return index;
}

/** A region's point indices; an absent region is an empty selection, not an error. */
export function region(skeleton: Skeleton, name: string): number[] {
  return skeleton.regions[name] ?? [];
}
