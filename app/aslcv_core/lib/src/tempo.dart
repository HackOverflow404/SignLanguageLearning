/// Port of src/aslcv/grading/embedding_dataset.py's _tempo_features: the
/// 2-scalar periodicity signal the repeated-movement head reads -- (peak
/// autocorrelation lag / clip length, peak height) of the hand motion-energy
/// signal.
///
/// Python computes the autocorrelation by FFT; at these lengths (tens to a few
/// hundred frames) the direct O(n^2) sum is cheap and equal to it within
/// float64 rounding. Lags 0-1 are excluded from the peak search: energy is
/// smooth frame-to-frame whether or not the motion repeats.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'f32.dart';

(double, double) tempoFeatures(Float32List energy) {
  final n = energy.length;
  if (n < 4 || energy.every((e) => e == 0)) return (0, 0);

  var sum = 0.0;
  for (final e in energy) {
    sum = f(sum + e);
  }
  final mean = f(sum / n); // numpy's float32 mean
  final x = Float64List.fromList([for (final e in energy) e - mean]);

  final acf = Float64List(n);
  for (var k = 0; k < n; k++) {
    var s = 0.0;
    for (var i = 0; i + k < n; i++) {
      s += x[i] * x[i + k];
    }
    acf[k] = s;
  }
  if (acf[0] <= 1e-8) return (0, 0);

  var peakIdx = 2;
  for (var k = 3; k < n; k++) {
    if (acf[k] > acf[peakIdx]) peakIdx = k;
  }
  final height = math.min(1.0, math.max(-1.0, acf[peakIdx] / acf[0]));
  return (f(peakIdx / n), f(height));
}
