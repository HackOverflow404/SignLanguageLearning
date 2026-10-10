/**
 * Cross-language parity: every stage of the TypeScript port against the golden
 * fixtures exported by scripts/export_mobile.py (Python's own outputs on real
 * ASL Citizen val clips), then the full raw-keypoints -> verdict chain through
 * the real ONNX graph on onnxruntime-node (CPU, like the phone).
 *
 * Needs the exported bundle (models/mobile/, or $ASLCV_BUNDLE); skipped if absent.
 */
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";
import { describe, expect, it } from "vitest";
import {
  BundleGrader,
  assemble,
  gradeAttempt,
  handMotionEnergy,
  liveCaptureSpan,
  tempoFeatures,
  type BundleMeta,
  type ModelRunner,
  type Pose,
  type Skeleton,
} from "../src/index";

const here = dirname(fileURLToPath(import.meta.url));
const BUNDLE = process.env.ASLCV_BUNDLE ?? join(here, "../../../models/mobile");
const FIXTURES = join(BUNDLE, "fixtures");
const available = existsSync(join(BUNDLE, "grader.onnx")) && existsSync(FIXTURES);

const f32 = (b64: string) => {
  const buf = Buffer.from(b64, "base64");
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
};

/** |a - b| <= atol + rtol * |b|, elementwise; reports the worst offender. */
function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>, atol: number, rtol = 1e-5) {
  expect(actual.length).toBe(expected.length);
  let worst = 0;
  let at = -1;
  for (let i = 0; i < expected.length; i++) {
    const excess = Math.abs(actual[i]! - expected[i]!) - (atol + rtol * Math.abs(expected[i]!));
    if (excess > worst) {
      worst = excess;
      at = i;
    }
  }
  expect(at, `index ${at}: ${actual[at]} vs ${expected[at]}`).toBe(-1);
}

interface Fixture {
  video_id: string;
  sign: string;
  input: { frames: number; points: number; keypoints: string; scores: string };
  expected: {
    full_energy: string;
    live_capture_span: [number, number];
    features: string;
    feature_shape: [number, number];
    energy: string;
    tempo: [number, number];
    outputs: Record<string, number[]>;
    grades: Record<string, { fidelity: number; parameters: Record<string, { predicted: string; target: string; correct: boolean | null; confidence: number }> }>;
  };
}

function posesOf(fx: Fixture): Pose[] {
  const kp = f32(fx.input.keypoints);
  const sc = f32(fx.input.scores);
  const K = fx.input.points;
  return Array.from({ length: fx.input.frames }, (_, t) => ({
    keypoints: kp.subarray(t * 2 * K, (t + 1) * 2 * K),
    scores: sc.subarray(t * K, (t + 1) * K),
  }));
}

describe.skipIf(!available)("golden-fixture parity with the Python pipeline", async () => {
  const skeleton: Skeleton = JSON.parse(readFileSync(join(BUNDLE, "skeleton.json"), "utf8"));
  const meta: BundleMeta = JSON.parse(readFileSync(join(BUNDLE, "bundle.json"), "utf8"));
  const refBuf = readFileSync(join(BUNDLE, meta.references.file));
  const references = new Float32Array(refBuf.buffer, refBuf.byteOffset, refBuf.byteLength / 4);
  const session = await ort.InferenceSession.create(join(BUNDLE, "grader.onnx"), { executionProviders: ["cpu"] });
  const runModel: ModelRunner = async (features, frames, tempo) => {
    const out = await session.run({
      features: new ort.Tensor("float32", features, [1, frames, meta.feature_dim]),
      tempo: new ort.Tensor("float32", tempo, [1, 2]),
    });
    return Object.fromEntries(Object.entries(out).map(([k, v]) => [k, v.data as Float32Array]));
  };
  const grader = new BundleGrader(meta, references, runModel);
  const fixtures: Fixture[] = readdirSync(FIXTURES)
    .filter((n) => n.endsWith(".json"))
    .sort()
    .map((n) => JSON.parse(readFileSync(join(FIXTURES, n), "utf8")));

  it("has fixtures to check", () => expect(fixtures.length).toBeGreaterThan(0));

  for (const fx of fixtures) {
    describe(`${fx.sign} (${fx.video_id})`, () => {
      const poses = posesOf(fx);
      const { motion_threshold, live_preroll, live_settle_frames } = meta.capture;
      const fullEnergy = handMotionEnergy(poses, skeleton);
      const span = liveCaptureSpan(fullEnergy, motion_threshold, live_preroll, live_settle_frames);
      const trimmed = poses.slice(span[0], span[1]);

      it("motion energy over the raw clip", () => expectClose(fullEnergy, f32(fx.expected.full_energy), 1e-6));
      it("live capture trim span", () => expect(span).toEqual(fx.expected.live_capture_span));

      it("assembled features", () => {
        const clip = assemble(trimmed, skeleton, meta.pipeline_args);
        expect([clip.frames, clip.dim]).toEqual(fx.expected.feature_shape);
        expectClose(clip.features, f32(fx.expected.features), 1e-5);
      });

      it("tempo features", () => {
        expectClose(handMotionEnergy(trimmed, skeleton), f32(fx.expected.energy), 1e-6);
        expectClose(tempoFeatures(handMotionEnergy(trimmed, skeleton)), fx.expected.tempo, 1e-5);
      });

      it("model outputs from TS-computed inputs", async () => {
        const clip = assemble(trimmed, skeleton, meta.pipeline_args);
        const out = await grader.run(clip.features, clip.frames, tempoFeatures(handMotionEnergy(trimmed, skeleton)));
        for (const [name, want] of Object.entries(fx.expected.outputs)) expectClose(out[name]!, want, 1e-4);
      });

      it("end-to-end verdicts: raw keypoints -> grade", async () => {
        for (const [target, want] of Object.entries(fx.expected.grades)) {
          const got = await gradeAttempt(grader, skeleton, trimmed, target);
          expect(got.fidelity).toBeCloseTo(want.fidelity, 4);
          for (const [p, w] of Object.entries(want.parameters)) {
            const g = got.parameters[p as keyof typeof got.parameters];
            expect({ predicted: g.predicted, target: g.target, correct: g.correct }, `${target}/${p}`).toEqual({
              predicted: w.predicted,
              target: w.target,
              correct: w.correct,
            });
            expect(g.confidence).toBeCloseTo(w.confidence, 4);
          }
        }
      });
    });
  }
});
