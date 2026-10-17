/// Port of src/aslcv/generator/feedback.py: templated coaching from a graded
/// attempt's verdicts. Every value shown is a readout of the grader's own
/// predicted/target labels; the only grounded descriptive text (handshape and
/// whole-sign descriptions) is written by Python into the bundle's text.json
/// (scripts/export_learner_text.py) and only looked up here.
library;

import 'grader.dart';

class LearnerText {
  final Map<String, String> signs; // sign -> describe_sign()
  final Map<String, String> handshapes; // code -> describe_handshape(), grounded codes only

  LearnerText.fromJson(Map<String, dynamic> j)
      : signs = (j['signs'] as Map).cast<String, String>(),
        handshapes = (j['handshapes'] as Map).cast<String, String>();
}

final _camelBoundary = RegExp(r'(?<=[a-z0-9])(?=[A-Z])');
const _repeatedValues = {'True': 'repeats', 'False': 'does not repeat'};

/// A raw phonology code, formatted for reading: "closed_b" -> "Closed B",
/// "HeadAway" -> "Head Away"; repeated movement -> "repeats"/"does not repeat".
String readableValue(String parameter, String value) {
  if (parameter == 'repeated_movement') return _repeatedValues[value] ?? value;
  if (value.isEmpty) return value;
  final spaced = value.replaceAll('_', ' ').replaceAll(_camelBoundary, ' ');
  return spaced
      .split(RegExp(r'\s+'))
      .where((w) => w.isNotEmpty)
      .map((w) => w.length <= 2 ? w.toUpperCase() : w[0].toUpperCase() + w.substring(1).toLowerCase())
      .join(' ');
}

const paramName = {
  'handshape': 'handshape',
  'major_location': 'general location',
  'minor_location': 'exact location',
  'movement': 'movement path',
  'repeated_movement': 'movement repetition',
};

const paramTip = {
  'handshape': 'check your finger position and shape',
  'major_location': "check the broad area you're signing in (e.g. face vs. body vs. neutral space)",
  'minor_location': "you're in the right general area, but not quite the exact spot",
  'movement': 'check the direction and shape of the motion',
  'repeated_movement': 'check whether this sign repeats its movement or not',
};

const _praise = 'Nice -- every parameter matched the reference.';
const _noVerdict = 'This sign isn\'t scored part by part -- compare your signing with the reference video.';

/// The wrong verdict the grader is most confident about (first wins a tie).
Verdict? _focus(Iterable<Verdict> verdicts) {
  Verdict? best;
  for (final v in verdicts) {
    if (v.correct == false && (best == null || v.confidence > best.confidence)) best = v;
  }
  return best;
}

/// The parameter coachText tells the learner to focus on; null when nothing
/// was wrong. Feeds the scheduler's contrastive drill.
String? focusParameter(Iterable<Verdict> verdicts) => _focus(verdicts)?.parameter;

/// One correction at a time: praise if everything judged matched, otherwise
/// the most-confidently-wrong parameter with a tip, other misses named briefly.
String coachText(Iterable<Verdict> verdicts, Map<String, String> handshapes) {
  final judged = verdicts.where((v) => v.correct != null).toList();
  if (judged.isEmpty) return _noVerdict;
  final focus = _focus(judged);
  if (focus == null) return _praise;

  final p = focus.parameter;
  var msg = 'Focus on your ${paramName[p]}: you signed '
      "'${readableValue(p, focus.predicted)}', the target is '${readableValue(p, focus.target)}' -- ${paramTip[p]}.";
  if (p == 'handshape') {
    final signed = handshapes[focus.predicted], target = handshapes[focus.target];
    if (signed != null) msg += ' You made: $signed.';
    if (target != null) msg += ' Target is: $target.';
  }
  final others = [for (final v in judged) if (v.correct == false && !identical(v, focus)) paramName[v.parameter]];
  if (others.isNotEmpty) msg += ' Also off: ${others.join(', ')}.';
  return msg;
}
