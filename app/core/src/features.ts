/**
 * Port of src/aslcv/features.py: FeaturePipeline.assemble, hand_motion_energy,
 * motion_active_span and live_capture_span.
 *
 * Division of labor matches the Python: the normalizer decides which frame each
 * point is in; this module only selects points, attaches confidence, appends
 * velocity, and stacks frames. Presence is `score > 0` -- never a coordinate test.
 *
 * Arithmetic is rounded to float32 at each step (Math.fround) to track numpy's
 * float32 results.
 */
import { normalize, type BlockName, type NormalizedPose } from "./normalizer";
import { region, type Pose, type Skeleton } from "./skeleton";

const f = Math.fround;
const BLOCK_ORDER: BlockName[] = ["global", "left_hand", "right_hand"];

/** The bundle's pipeline_args (the grader checkpoint's training-time config). */
export interface PipelineArgs {
  face: boolean;
  legs_feet: boolean;
  confidence: "graded" | "binary";
  binary_threshold: number;
  velocity: boolean;
  depth_proxies: boolean;
  trim_to_motion: boolean;
  local_hand: boolean;
}

export interface FeatureClip {
  features: Float32Array; // row-major (frames, dim)
  frames: number;
  dim: number;
}

function checkSupported(args: PipelineArgs): void {
  // Fail closed on any config this port does not implement, rather than
  // silently producing features the grader was never trained on.
  if (args.depth_proxies) throw new Error("depth_proxies=true is not supported by this port");
  if (args.trim_to_motion) throw new Error("trim_to_motion=true is not supported by this port");
  if (!args.local_hand) throw new Error("local_hand=false is not supported by this port");
}

/** Rows kept within a block: the global block drops face / legs_feet unless enabled. */
function keptRows(block: BlockName, nPoints: number, skeleton: Skeleton, args: PipelineArgs): number[] {
  const all = Array.from({ length: nPoints }, (_, i) => i);
  if (block !== "global") return all;
  const drop = new Set<number>();
  if (!args.face) region(skeleton, "face").forEach((i) => drop.add(i));
  if (!args.legs_feet) region(skeleton, "legs_feet").forEach((i) => drop.add(i));
  return all.filter((i) => !drop.has(i));
}

export function assemble(poses: Pose[], skeleton: Skeleton, args: PipelineArgs): FeatureClip {
  checkSupported(args);
  if (poses.length === 0) throw new Error("assemble() needs at least one frame");
  const T = poses.length;
  const channels = args.velocity ? 5 : 3;

  const normalized: NormalizedPose[] = poses.map((p) => normalize(p, skeleton));
  const present = BLOCK_ORDER.filter((b) => normalized[0]!.blocks[b] !== undefined);
  const rows = present.map((b) => {
    const [start, stop] = normalized[0]!.blocks[b]!;
    return keptRows(b, stop - start, skeleton, args).map((r) => start + r);
  });
  const selected = rows.flat(); // absolute rows into each NormalizedPose
  const P = selected.length;
  const dim = P * channels;

  const out = new Float32Array(T * dim);
  const posX = new Float32Array(T * P);
  const posY = new Float32Array(T * P);
  const presence = new Uint8Array(T * P);

  for (let t = 0; t < T; t++) {
    const np = normalized[t]!;
    for (let i = 0; i < P; i++) {
      const row = selected[i]!;
      const score = np.scores[row]!;
      if (!(score > 0)) continue; // absent: all channels stay 0
      const x = np.keypoints[2 * row]!;
      const y = np.keypoints[2 * row + 1]!;
      const conf = args.confidence === "graded" ? score : score > args.binary_threshold ? 1 : 0;
      const o = t * dim + i * channels;
      out[o] = x;
      out[o + 1] = y;
      out[o + 2] = conf;
      posX[t * P + i] = x;
      posY[t * P + i] = y;
      presence[t * P + i] = 1;
    }
  }

  if (args.velocity) {
    // first-frame delta stays 0; a delta touching an absence is zeroed, not
    // differenced against the zero-fill (real -> 0 -> real is not motion)
    for (let t = 1; t < T; t++) {
      for (let i = 0; i < P; i++) {
        if (!presence[t * P + i] || !presence[(t - 1) * P + i]) continue;
        const o = t * dim + i * channels;
        out[o + 3] = f(posX[t * P + i]! - posX[(t - 1) * P + i]!);
        out[o + 4] = f(posY[t * P + i]! - posY[(t - 1) * P + i]!);
      }
    }
  }
  return { features: out, frames: T, dim };
}

/**
 * Per-frame mean velocity magnitude over both hands' points in the local-hand
 * frame, averaged over PRESENT points only (an undetected hand is not a resting
 * one). Same signal the live capture buffer and the training-data trim use.
 */
export function handMotionEnergy(poses: Pose[], skeleton: Skeleton): Float32Array {
  const T = poses.length;
  if (T === 0) throw new Error("handMotionEnergy() needs at least one frame");
  const normalized = poses.map((p) => normalize(p, skeleton));
  const handRows: number[] = [];
  for (const b of ["left_hand", "right_hand"] as const) {
    const span = normalized[0]!.blocks[b];
    if (span) for (let r = span[0]; r < span[1]; r++) handRows.push(r);
  }
  if (handRows.length === 0) throw new Error("handMotionEnergy() needs left_hand/right_hand blocks");

  const energy = new Float32Array(T);
  for (let t = 1; t < T; t++) {
    const cur = normalized[t]!;
    const prev = normalized[t - 1]!;
    let sum = 0;
    let count = 0;
    for (const r of handRows) {
      const here = cur.scores[r]! > 0;
      if (here) count++;
      if (!here || !(prev.scores[r]! > 0)) continue;
      const dx = f(cur.keypoints[2 * r]! - prev.keypoints[2 * r]!);
      const dy = f(cur.keypoints[2 * r + 1]! - prev.keypoints[2 * r + 1]!);
      sum = f(sum + f(Math.sqrt(f(f(dx * dx) + f(dy * dy)))));
    }
    if (count > 0) energy[t] = f(sum / count);
  }
  return energy;
}

/** [start, stop) covering every frame with energy > threshold, padded; the full range if none. */
export function motionActiveSpan(energy: Float32Array, threshold: number, padFrames = 0): [number, number] {
  let first = -1;
  let last = -1;
  energy.forEach((e, i) => {
    if (e > threshold) {
      if (first < 0) first = i;
      last = i;
    }
  });
  if (first < 0) return [0, energy.length];
  return [Math.max(0, first - padFrames), Math.min(energy.length, last + 1 + padFrames)];
}

/**
 * Approximation of what the live capture buffer keeps for a clean
 * rest -> sign -> rest clip: the motion-active span extended by `preroll`
 * frames before and `settleFrames` after.
 */
export function liveCaptureSpan(
  energy: Float32Array,
  threshold: number,
  preroll: number,
  settleFrames: number,
): [number, number] {
  const [start, stop] = motionActiveSpan(energy, threshold, 0);
  return [Math.max(0, start - preroll), Math.min(energy.length, stop + settleFrames)];
}
