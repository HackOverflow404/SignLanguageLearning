/// Port of src/aslcv/learner/mastery.py and scheduler.py: per-sign,
/// per-parameter mastery (EMA) with SM2-style spacing on a logical attempt
/// clock, and the next-sign picker (due-gate, weakness weighting, contrastive
/// minimal-pair drills). Mastery only remembers verdicts the grader already
/// made -- it never judges an attempt itself.
library;

import 'dart:math' as math;

import 'grader.dart';

const allParameters = [...categoricalParameters, 'repeated_movement'];

const initialMastery = 0.5;
const learningRate = 0.3;
const initialEase = 2.5;
const minEase = 1.3;
const maxEase = 3.0;
const easeDeltaGood = 0.05;
const easeDeltaBad = 0.2;
const secondRepInterval = 3;
const maxInterval = 40;

const weaknessFloor = 0.05;
const highMasteryThreshold = 0.8;
const proactiveContrastiveProb = 0.5;

/// Python's round(): halves go to the even neighbour, so intervals match.
int _pyRound(double x) {
  final f = x.floorToDouble();
  final d = x - f;
  if (d > 0.5 || (d == 0.5 && f % 2 != 0)) return f.toInt() + 1;
  return f.toInt();
}

class MasteryState {
  final Map<String, Map<String, double>> mastery = {};
  final Map<String, Map<String, int>> attempts = {};
  final Map<String, int> lastSeen = {};
  int clock = 0;
  final Map<String, int> interval = {};
  final Map<String, double> ease = {};
  final Map<String, int> repetitions = {};

  MasteryState();

  /// Mean over this sign's attempted parameters; never attempted reads as the
  /// neutral prior, not as known-weak.
  double signMastery(String sign) {
    final params = mastery[sign];
    if (params == null || params.isEmpty) return initialMastery;
    return params.values.reduce((a, b) => a + b) / params.length;
  }

  double parameterMastery(String sign, String parameter) => mastery[sign]?[parameter] ?? initialMastery;

  int attemptsFor(String sign) => (attempts[sign]?.values ?? const <int>[]).fold(0, (a, b) => a + b);

  void _sm2Advance(String sign, bool good) {
    var e = ease[sign] ?? initialEase;
    var reps = repetitions[sign] ?? 0;
    int next;
    if (good) {
      reps += 1;
      next = reps == 1
          ? 1
          : reps == 2
              ? secondRepInterval
              : math.min(maxInterval, _pyRound((interval[sign] ?? 1) * e));
      e = math.min(maxEase, e + easeDeltaGood);
    } else {
      reps = 0;
      next = 1;
      e = math.max(minEase, e - easeDeltaBad);
    }
    repetitions[sign] = reps;
    interval[sign] = next;
    ease[sign] = e;
  }

  /// The tick at which `sign` next becomes due; null if never attempted.
  int? dueAt(String sign) {
    final last = lastSeen[sign];
    return last == null ? null : last + (interval[sign] ?? 1);
  }

  bool isDue(String sign) {
    final due = dueAt(sign);
    return due == null || clock >= due;
  }

  /// One graded attempt. A null verdict (insufficient data) is skipped, never
  /// averaged in. "Good" for spacing means every JUDGED parameter matched; an
  /// attempt with nothing judged leaves the interval alone.
  void update(String sign, Map<String, bool?> correctByParameter) {
    clock += 1;
    lastSeen[sign] = clock;
    final m = mastery.putIfAbsent(sign, () => {});
    final a = attempts.putIfAbsent(sign, () => {});
    final judged = <bool>[];
    for (final MapEntry(key: p, value: correct) in correctByParameter.entries) {
      if (correct == null) continue;
      judged.add(correct);
      final old = m[p] ?? initialMastery;
      m[p] = old + learningRate * ((correct ? 1.0 : 0.0) - old);
      a[p] = (a[p] ?? 0) + 1;
    }
    if (judged.isNotEmpty) {
      _sm2Advance(sign, judged.every((c) => c));
    } else if (correctByParameter.isNotEmpty) {
      _sm2Advance(sign, true); // nothing judgeable (multimorphemic): spaced like a pass
    }
  }

  Map<String, dynamic> toJson() => {
        'mastery': mastery,
        'attempts': attempts,
        'last_seen': lastSeen,
        'clock': clock,
        'interval': interval,
        'ease': ease,
        'repetitions': repetitions,
      };

  /// Same keys as Python's to_dict(), so a state file moves between the two.
  factory MasteryState.fromJson(Map<String, dynamic> j) {
    Map<String, T> flat<T>(Object? v, T Function(num) conv) =>
        {for (final e in ((v ?? {}) as Map).entries) e.key as String: conv(e.value as num)};
    Map<String, Map<String, T>> nested<T>(Object? v, T Function(num) conv) =>
        {for (final e in ((v ?? {}) as Map).entries) e.key as String: flat(e.value, conv)};
    final s = MasteryState()
      ..mastery.addAll(nested(j['mastery'], (n) => n.toDouble()))
      ..attempts.addAll(nested(j['attempts'], (n) => n.toInt()))
      ..lastSeen.addAll(flat(j['last_seen'], (n) => n.toInt()))
      ..interval.addAll(flat(j['interval'], (n) => n.toInt()))
      ..ease.addAll(flat(j['ease'], (n) => n.toDouble()))
      ..repetitions.addAll(flat(j['repetitions'], (n) => n.toInt()));
    s.clock = (j['clock'] as num? ?? 0).toInt();
    return s;
  }
}

/// The scheduler's randomness, injectable so tests can pin each branch.
abstract class SchedulerRandom {
  String choose(List<String> pool, List<double> weights);
  double nextDouble();
}

class DefaultSchedulerRandom implements SchedulerRandom {
  final math.Random _r;

  DefaultSchedulerRandom([math.Random? r]) : _r = r ?? math.Random();

  @override
  String choose(List<String> pool, List<double> weights) {
    var x = _r.nextDouble() * weights.reduce((a, b) => a + b);
    for (var i = 0; i < pool.length; i++) {
      x -= weights[i];
      if (x < 0) return pool[i];
    }
    return pool.last;
  }

  @override
  double nextDouble() => _r.nextDouble();
}

/// The value every parameter is compared on: categorical targets, plus
/// repeated movement as a bool.
Object _label(SignTargets t, String p) => p == 'repeated_movement' ? t.repeatedMovement.target : t.categorical[p]!.target;

/// parameter -> every pair of `signs` differing in exactly that parameter,
/// leaving out multimorphemic signs (their labels cover the first morpheme only).
Map<String, List<(String, String)>> findMinimalPairs(Map<String, SignTargets> targets, List<String> allSigns) {
  final pairs = {for (final p in allParameters) p: <(String, String)>[]};
  final signs = [for (final s in allSigns) if (!targets[s]!.multimorphemic) s];
  for (var i = 0; i < signs.length; i++) {
    for (final b in signs.sublist(i + 1)) {
      final a = signs[i];
      final diffs = [for (final p in allParameters) if (_label(targets[a]!, p) != _label(targets[b]!, p)) p];
      if (diffs.length == 1) pairs[diffs.single]!.add((a, b));
    }
  }
  return pairs;
}

String? _partner(Iterable<(String, String)> pairs, String sign, List<String> candidates) {
  for (final (a, b) in pairs) {
    if (a == sign && candidates.contains(b)) return b;
    if (b == sign && candidates.contains(a)) return a;
  }
  return null;
}

/// Next sign to drill. A contrastive drill fires first when the last
/// attempt's focus mistake has a minimal-pair partner in the pool; otherwise
/// weighted-random by weakness over the due signs (all signs if none are
/// due), and a pick on a well-mastered sign may be redirected to any of its
/// minimal-pair partners.
String pickNext(
  MasteryState mastery,
  List<String> signs, {
  String? lastSign,
  String? lastWrongParameter,
  Map<String, List<(String, String)>>? minimalPairs,
  SchedulerRandom? rng,
}) {
  if (signs.isEmpty) throw ArgumentError('pickNext() needs at least one candidate sign');
  if (minimalPairs != null && lastSign != null && lastWrongParameter != null) {
    final partner = _partner(minimalPairs[lastWrongParameter] ?? const [], lastSign, signs);
    if (partner != null) return partner;
  }
  rng ??= DefaultSchedulerRandom();
  final due = [for (final s in signs) if (mastery.isDue(s)) s];
  final pool = due.isEmpty ? signs : due;
  final weights = [for (final s in pool) math.max(weaknessFloor, 1.05 - mastery.signMastery(s))];
  final chosen = rng.choose(pool, weights);
  if (mastery.signMastery(chosen) >= highMasteryThreshold && rng.nextDouble() < proactiveContrastiveProb) {
    final partner = minimalPairs == null ? null : _partner(minimalPairs.values.expand((p) => p), chosen, signs);
    if (partner != null) return partner;
  }
  return chosen;
}
