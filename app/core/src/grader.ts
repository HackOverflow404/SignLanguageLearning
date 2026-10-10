/**
 * Port of src/aslcv/grading/bundle_grader.py: grade one attempt against a KNOWN
 * target sign from the exported mobile bundle -- distance to the target's
 * nearest reference clip plus a per-parameter verdict, never an N-way "which
 * sign was that" label.
 *
 * The ONNX runtime is injected (`ModelRunner`) so the same code runs on
 * onnxruntime-node in tests and onnxruntime-react-native in the app.
 */
const f = Math.fround;

export const CATEGORICAL_PARAMETERS = ["handshape", "major_location", "minor_location", "movement"] as const;
export type Parameter = (typeof CATEGORICAL_PARAMETERS)[number] | "repeated_movement";

interface TargetLabel<T> {
  target: T;
  judged: boolean; // false when the label has too few curriculum signs to grade (MIN_SUPPORT)
}

export interface BundleMeta {
  format_version: number;
  outputs: string[];
  feature_dim: number;
  embed_dim: number;
  pipeline_args: import("./features").PipelineArgs;
  capture: { motion_threshold: number; live_preroll: number; live_settle_frames: number };
  standardizer: { mean: number[]; std: number[] };
  classes: Record<(typeof CATEGORICAL_PARAMETERS)[number], string[]>;
  references: { file: string; rows: number; dim: number; signs: { sign: string; start: number; count: number }[] };
  signs: Record<
    string,
    Record<(typeof CATEGORICAL_PARAMETERS)[number], TargetLabel<string>> & {
      repeated_movement: TargetLabel<boolean>;
    }
  >;
}

/** Runs the exported graph: standardized features (1, frames, dim) + tempo (1, 2) -> named outputs. */
export type ModelRunner = (
  features: Float32Array,
  frames: number,
  tempo: Float32Array,
) => Promise<Record<string, Float32Array>>;

export interface Verdict {
  parameter: Parameter;
  predicted: string;
  target: string;
  correct: boolean | null; // null = not judged (insufficient data), never a guess
  confidence: number;
}

export interface Grade {
  targetSign: string;
  fidelity: number; // embedding distance to the target's nearest reference clip; lower = closer
  parameters: Record<Parameter, Verdict>;
}

export const FORMAT_VERSION = 1;

/** "True"/"False", matching the Python verdict's str(bool). */
const pyBool = (b: boolean) => (b ? "True" : "False");

export class BundleGrader {
  private readonly mean: Float32Array;
  private readonly std: Float32Array;
  private readonly spans: Map<string, [number, number]>;

  constructor(
    readonly meta: BundleMeta,
    private readonly references: Float32Array, // (rows, dim) row-major
    private readonly runModel: ModelRunner,
  ) {
    if (meta.format_version !== FORMAT_VERSION) {
      throw new Error(`bundle format ${meta.format_version}, expected ${FORMAT_VERSION}`);
    }
    if (references.length !== meta.references.rows * meta.references.dim) {
      throw new Error("references size does not match bundle.json");
    }
    this.mean = Float32Array.from(meta.standardizer.mean);
    this.std = Float32Array.from(meta.standardizer.std);
    this.spans = new Map(meta.references.signs.map((s) => [s.sign, [s.start, s.count]]));
  }

  /** Raw (unstandardized) row-major (frames, dim) features + tempo -> every named model output. */
  async run(raw: Float32Array, frames: number, tempo: [number, number]): Promise<Record<string, Float32Array>> {
    const dim = this.meta.feature_dim;
    if (raw.length !== frames * dim) throw new Error(`expected ${frames}x${dim} features`);
    const x = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) x[i] = f(f(raw[i]! - this.mean[i % dim]!) / this.std[i % dim]!);
    return this.runModel(x, frames, Float32Array.from(tempo));
  }

  async grade(raw: Float32Array, frames: number, tempo: [number, number], targetSign: string): Promise<Grade> {
    const targets = this.meta.signs[targetSign];
    const span = this.spans.get(targetSign);
    if (!targets || !span) throw new Error(`unknown target sign ${JSON.stringify(targetSign)}`);
    const out = await this.run(raw, frames, tempo);

    const embed = out.embed!;
    const dim = this.meta.references.dim;
    let fidelity = Infinity;
    for (let r = span[0]; r < span[0] + span[1]; r++) {
      let s = 0;
      for (let j = 0; j < dim; j++) {
        const d = this.references[r * dim + j]! - embed[j]!;
        s += d * d;
      }
      fidelity = Math.min(fidelity, Math.sqrt(s));
    }

    const parameters = {} as Record<Parameter, Verdict>;
    for (const p of CATEGORICAL_PARAMETERS) {
      const logits = out[p]!;
      let idx = 0;
      for (let i = 1; i < logits.length; i++) if (logits[i]! > logits[idx]!) idx = i;
      let z = 0;
      for (const l of logits) z += Math.exp(l - logits[idx]!);
      const predicted = this.meta.classes[p][idx]!;
      const t = targets[p];
      parameters[p] = {
        parameter: p,
        predicted,
        target: t.target,
        correct: t.judged ? predicted === t.target : null,
        confidence: 1 / z,
      };
    }
    const prob = 1 / (1 + Math.exp(-out.repeated![0]!));
    const pred = prob > 0.5;
    const t = targets.repeated_movement;
    parameters.repeated_movement = {
      parameter: "repeated_movement",
      predicted: pyBool(pred),
      target: pyBool(t.target),
      correct: t.judged ? pred === t.target : null,
      confidence: pred ? prob : 1 - prob,
    };
    return { targetSign, fidelity, parameters };
  }
}
