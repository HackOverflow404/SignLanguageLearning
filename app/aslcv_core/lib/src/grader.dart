/// Port of src/aslcv/grading/bundle_grader.py: grade one attempt against a
/// KNOWN target sign from the exported mobile bundle -- distance to the
/// target's nearest reference clip plus a per-parameter verdict, never an
/// N-way "which sign was that" label.
///
/// The ONNX runtime is injected (`ModelRunner`) so the same code runs under
/// `dart test` with recorded outputs and on the phone with onnxruntime.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'f32.dart';
import 'features.dart';

const categoricalParameters = ['handshape', 'major_location', 'minor_location', 'movement'];
const formatVersion = 1;

/// Runs the exported graph: standardized features (1, frames, dim) + tempo
/// (1, 2) -> named outputs.
typedef ModelRunner = Future<Map<String, Float32List>> Function(
    Float32List features, int frames, Float32List tempo);

/// A sign's target label for one parameter; `judged` is false when the label
/// has too few curriculum signs to grade (MIN_SUPPORT).
class TargetLabel<T> {
  final T target;
  final bool judged;

  const TargetLabel(this.target, this.judged);
}

class SignTargets {
  final Map<String, TargetLabel<String>> categorical;
  final TargetLabel<bool> repeatedMovement;

  const SignTargets(this.categorical, this.repeatedMovement);
}

class BundleMeta {
  final int formatVersion;
  final List<String> outputs;
  final int featureDim;
  final int embedDim;
  final PipelineArgs pipelineArgs;
  final double motionThreshold;
  final int livePreroll;
  final int liveSettleFrames;
  final Float32List mean;
  final Float32List std;
  final Map<String, List<String>> classes;
  final String referencesFile;
  final int referenceRows;
  final int referenceDim;
  final Map<String, (int, int)> referenceSpans; // sign -> (start, count)
  final Map<String, SignTargets> signs;

  BundleMeta.fromJson(Map<String, dynamic> j)
      : formatVersion = j['format_version'] as int,
        outputs = (j['outputs'] as List).cast<String>(),
        featureDim = j['feature_dim'] as int,
        embedDim = j['embed_dim'] as int,
        pipelineArgs = PipelineArgs.fromJson(j['pipeline_args'] as Map<String, dynamic>),
        motionThreshold = (j['capture']['motion_threshold'] as num).toDouble(),
        livePreroll = j['capture']['live_preroll'] as int,
        liveSettleFrames = j['capture']['live_settle_frames'] as int,
        mean = Float32List.fromList([for (final v in j['standardizer']['mean'] as List) (v as num).toDouble()]),
        std = Float32List.fromList([for (final v in j['standardizer']['std'] as List) (v as num).toDouble()]),
        classes = (j['classes'] as Map).map((k, v) => MapEntry(k as String, (v as List).cast<String>())),
        referencesFile = j['references']['file'] as String,
        referenceRows = j['references']['rows'] as int,
        referenceDim = j['references']['dim'] as int,
        referenceSpans = {
          for (final s in j['references']['signs'] as List) s['sign'] as String: (s['start'] as int, s['count'] as int),
        },
        signs = (j['signs'] as Map).map((sign, labels) => MapEntry(
              sign as String,
              SignTargets(
                {
                  for (final p in categoricalParameters)
                    p: TargetLabel(labels[p]['target'] as String, labels[p]['judged'] as bool),
                },
                TargetLabel(labels['repeated_movement']['target'] as bool,
                    labels['repeated_movement']['judged'] as bool),
              ),
            ));
}

class Verdict {
  final String parameter;
  final String predicted;
  final String target;
  final bool? correct; // null = not judged (insufficient data), never a guess
  final double confidence;

  const Verdict(this.parameter, this.predicted, this.target, this.correct, this.confidence);
}

class Grade {
  final String targetSign;
  final double fidelity; // embedding distance to the target's nearest reference clip; lower = closer
  final Map<String, Verdict> parameters;

  const Grade(this.targetSign, this.fidelity, this.parameters);
}

/// "True"/"False", matching the Python verdict's str(bool).
String _pyBool(bool b) => b ? 'True' : 'False';

class BundleGrader {
  final BundleMeta meta;
  final Float32List _references; // (rows, dim) row-major
  final ModelRunner _runModel;

  BundleGrader(this.meta, this._references, this._runModel) {
    if (meta.formatVersion != formatVersion) {
      throw StateError('bundle format ${meta.formatVersion}, expected $formatVersion');
    }
    if (_references.length != meta.referenceRows * meta.referenceDim) {
      throw StateError('references size does not match bundle.json');
    }
  }

  /// Raw (unstandardized) row-major (frames, dim) features + tempo -> every
  /// named model output.
  Future<Map<String, Float32List>> run(Float32List raw, int frames, (double, double) tempo) {
    final dim = meta.featureDim;
    if (raw.length != frames * dim) throw ArgumentError('expected ${frames}x$dim features');
    final x = Float32List(raw.length);
    for (var i = 0; i < raw.length; i++) {
      x[i] = f(f(raw[i] - meta.mean[i % dim]) / meta.std[i % dim]);
    }
    return _runModel(x, frames, Float32List.fromList([tempo.$1, tempo.$2]));
  }

  Future<Grade> grade(Float32List raw, int frames, (double, double) tempo, String targetSign) async {
    final targets = meta.signs[targetSign];
    final span = meta.referenceSpans[targetSign];
    if (targets == null || span == null) throw ArgumentError('unknown target sign "$targetSign"');
    final out = await run(raw, frames, tempo);

    final embed = out['embed']!;
    final dim = meta.referenceDim;
    var fidelity = double.infinity;
    for (var r = span.$1; r < span.$1 + span.$2; r++) {
      var s = 0.0;
      for (var j = 0; j < dim; j++) {
        final d = _references[r * dim + j] - embed[j];
        s += d * d;
      }
      fidelity = math.min(fidelity, math.sqrt(s));
    }

    final parameters = <String, Verdict>{};
    for (final p in categoricalParameters) {
      final logits = out[p]!;
      var idx = 0;
      for (var i = 1; i < logits.length; i++) {
        if (logits[i] > logits[idx]) idx = i;
      }
      var z = 0.0;
      for (final l in logits) {
        z += math.exp(l - logits[idx]);
      }
      final predicted = meta.classes[p]![idx];
      final t = targets.categorical[p]!;
      parameters[p] = Verdict(p, predicted, t.target, t.judged ? predicted == t.target : null, 1 / z);
    }
    final prob = 1 / (1 + math.exp(-out['repeated']![0]));
    final pred = prob > 0.5;
    final t = targets.repeatedMovement;
    parameters['repeated_movement'] = Verdict('repeated_movement', _pyBool(pred), _pyBool(t.target),
        t.judged ? pred == t.target : null, pred ? prob : 1 - prob);
    return Grade(targetSign, fidelity, parameters);
  }
}
