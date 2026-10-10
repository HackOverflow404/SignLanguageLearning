/**
 * Port of src/aslcv/normalizer/shoulder.py's ShoulderNormalizer (local_hand=True).
 *
 * Emits a GLOBAL block (every point, shoulder-midpoint origin, shoulder-width
 * scale) followed by per-hand LOCAL blocks (each hand's points, wrist origin,
 * wrist->middle-knuckle scale). A degenerate frame (missing anchor or a scale at
 * or below EPS) zero-fills its block instead of dividing. Rotation is
 * deliberately NOT normalized -- a tilted head is grammatical in ASL.
 *
 * All arithmetic is rounded to float32 at each step (Math.fround) to reproduce
 * numpy's float32 results, not just approximate them.
 */
import { anchor, region, type Pose, type Skeleton } from "./skeleton";

const f = Math.fround;
const EPS = 1e-6;

export type BlockName = "global" | "left_hand" | "right_hand";

export interface NormalizedPose {
  keypoints: Float32Array; // interleaved (x, y), length 2N
  scores: Float32Array; // length N
  blocks: Partial<Record<BlockName, [number, number]>>; // [start, stop) point rows
}

/** float32 Euclidean distance between points i and j of `kp`. */
function dist32(kp: Float32Array, i: number, j: number): number {
  const dx = f(kp[2 * i]! - kp[2 * j]!);
  const dy = f(kp[2 * i + 1]! - kp[2 * j + 1]!);
  return f(Math.sqrt(f(f(dx * dx) + f(dy * dy))));
}

export function normalize(pose: Pose, skeleton: Skeleton): NormalizedPose {
  const { keypoints: kp, scores: sc } = pose;
  const nTotal = sc.length;
  const hands = (["left", "right"] as const).map((side) => ({
    name: `${side}_hand` as BlockName,
    idx: region(skeleton, `${side}_hand`),
    wrist: anchor(skeleton, `${side}_hand_wrist`),
    mcp: anchor(skeleton, `${side}_hand_middle_mcp`),
  }));
  const nRows = nTotal + hands.reduce((n, h) => n + h.idx.length, 0);
  const outKp = new Float32Array(2 * nRows);
  const outSc = new Float32Array(nRows);
  const blocks: NormalizedPose["blocks"] = {};

  // -- global block: every point in the shoulder frame
  const ls = anchor(skeleton, "left_shoulder");
  const rs = anchor(skeleton, "right_shoulder");
  const width = dist32(kp, ls, rs);
  if (sc[ls]! > 0 && sc[rs]! > 0 && width > EPS) {
    const ox = f(f(kp[2 * ls]! + kp[2 * rs]!) / 2);
    const oy = f(f(kp[2 * ls + 1]! + kp[2 * rs + 1]!) / 2);
    for (let i = 0; i < nTotal; i++) {
      outKp[2 * i] = f(f(kp[2 * i]! - ox) / width);
      outKp[2 * i + 1] = f(f(kp[2 * i + 1]! - oy) / width);
      outSc[i] = sc[i]!;
    }
  }
  blocks.global = [0, nTotal];

  // -- local hand blocks: each hand in its own wrist frame
  let cursor = nTotal;
  for (const h of hands) {
    if (h.idx.length === 0) continue;
    const scale = dist32(kp, h.wrist, h.mcp);
    if (sc[h.wrist]! > 0 && sc[h.mcp]! > 0 && scale > EPS) {
      const wx = kp[2 * h.wrist]!;
      const wy = kp[2 * h.wrist + 1]!;
      h.idx.forEach((p, r) => {
        outKp[2 * (cursor + r)] = f(f(kp[2 * p]! - wx) / scale);
        outKp[2 * (cursor + r) + 1] = f(f(kp[2 * p + 1]! - wy) / scale);
        outSc[cursor + r] = sc[p]!;
      });
    }
    blocks[h.name] = [cursor, cursor + h.idx.length];
    cursor += h.idx.length;
  }
  return { keypoints: outKp, scores: outSc, blocks };
}
