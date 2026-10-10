/**
 * Port of src/aslcv/grading/embedding_dataset.py's _tempo_features: the 2-scalar
 * periodicity signal the repeated-movement head reads -- (peak autocorrelation
 * lag / clip length, peak height) of the hand motion-energy signal.
 *
 * Python computes the autocorrelation by FFT; at these lengths (tens to a few
 * hundred frames) the direct O(n^2) sum is cheap and equal to it within float64
 * rounding. Lags 0-1 are excluded from the peak search: energy is smooth
 * frame-to-frame whether or not the motion repeats.
 */
const f = Math.fround;

export function tempoFeatures(energy: Float32Array): [number, number] {
  const n = energy.length;
  if (n < 4 || energy.every((e) => e === 0)) return [0, 0];

  let sum = 0;
  for (const e of energy) sum = f(sum + e);
  const mean = f(sum / n); // numpy's float32 mean
  const x = Float64Array.from(energy, (e) => e - mean);

  const acf = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let s = 0;
    for (let i = 0; i + k < n; i++) s += x[i]! * x[i + k]!;
    acf[k] = s;
  }
  if (acf[0]! <= 1e-8) return [0, 0];

  let peakIdx = 2;
  for (let k = 3; k < n; k++) if (acf[k]! > acf[peakIdx]!) peakIdx = k;
  const height = Math.min(1, Math.max(-1, acf[peakIdx]! / acf[0]!));
  return [f(peakIdx / n), f(height)];
}
