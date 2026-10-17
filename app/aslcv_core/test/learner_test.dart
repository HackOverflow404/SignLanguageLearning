// The adaptive loop's Dart port: golden parity against Python's own outputs
// (models/mobile/fixtures/learner.json, from scripts/export_learner_text.py)
// plus the scheduler's branch tests, which need no bundle.
import 'dart:convert';
import 'dart:io';

import 'package:aslcv_core/aslcv_core.dart';
import 'package:test/test.dart';

final bundle = Platform.environment['ASLCV_BUNDLE'] ?? '../../models/mobile';

/// Always picks `choice`, always rolls `roll` -- pins each scheduler branch.
class FakeRandom implements SchedulerRandom {
  final String choice;
  final double roll;

  FakeRandom(this.choice, this.roll);

  @override
  String choose(List<String> pool, List<double> weights) => choice;

  @override
  double nextDouble() => roll;
}

SignTargets targets(String hs, String loc, {bool repeats = false, bool multimorphemic = false}) => SignTargets({
      'handshape': TargetLabel(hs, true),
      'major_location': TargetLabel('Head', true),
      'minor_location': TargetLabel(loc, true),
      'movement': TargetLabel('Straight', true),
    }, TargetLabel(repeats, true), multimorphemic: multimorphemic);

void main() {
  group('mastery', () {
    test('null verdicts are skipped, not averaged in', () {
      final m = MasteryState()..update('mother', {'handshape': null, 'movement': true});
      expect(m.mastery['mother']!.keys, ['movement']);
      expect(m.signMastery('mother'), closeTo(0.65, 1e-12));
      expect(m.signMastery('never'), initialMastery);
    });

    test('a pass spaces the sign out, a miss brings it straight back', () {
      final m = MasteryState()..update('a', {'handshape': true});
      expect(m.isDue('a'), isFalse);
      m.update('b', {'handshape': true});
      expect(m.isDue('a'), isTrue);
      m.update('a', {'handshape': true});
      expect(m.interval['a'], secondRepInterval);
      m.update('a', {'handshape': false});
      expect(m.interval['a'], 1);
      expect(m.repetitions['a'], 0);
    });

    test('an attempt with nothing judgeable is spaced like a pass, mastery untouched', () {
      final m = MasteryState()
        ..update('home', {'handshape': null, 'movement': null})
        ..update('home', {'handshape': null, 'movement': null});
      expect(m.interval['home'], secondRepInterval);
      expect(m.signMastery('home'), initialMastery);
      expect(m.isDue('home'), isFalse);
    });

    test('json round trip', () {
      final m = MasteryState()
        ..update('a', {'handshape': true, 'movement': false})
        ..update('b', {'handshape': true});
      final back = MasteryState.fromJson(jsonDecode(jsonEncode(m.toJson())));
      expect(back.toJson(), m.toJson());
    });
  });

  group('scheduler', () {
    final t = {'mother': targets('5', 'Chin'), 'father': targets('5', 'Forehead'), 'water': targets('w', 'Chin')};
    final pairs = findMinimalPairs(t, ['mother', 'father', 'water']);

    test('minimal pairs differ in exactly one parameter', () {
      expect(pairs['minor_location'], [('mother', 'father')]);
      expect(pairs['handshape'], [('mother', 'water')]);
    });

    test('multimorphemic signs are left out of minimal pairs', () {
      final withCompound = {...t, 'home': targets('5', 'Cheek', multimorphemic: true)};
      final got = findMinimalPairs(withCompound, ['mother', 'father', 'water', 'home']);
      expect(got.values.expand((p) => p).where((p) => p.$1 == 'home' || p.$2 == 'home'), isEmpty);
    });

    test('a wrong parameter with a partner fires a contrastive drill', () {
      expect(
          pickNext(MasteryState(), ['mother', 'father', 'water'],
              lastSign: 'father', lastWrongParameter: 'minor_location', minimalPairs: pairs, rng: FakeRandom('water', 1)),
          'mother');
    });

    test('a just-passed sign is not due and is skipped', () {
      final m = MasteryState()..update('mother', {'handshape': true});
      final seen = <List<String>>[];
      final rng = _Recording(seen);
      pickNext(m, ['mother', 'father'], rng: rng);
      expect(seen.single, ['father']);
    });

    test('a well-mastered pick is proactively redirected to a partner', () {
      final m = MasteryState();
      for (var i = 0; i < 10; i++) {
        m.update('mother', {'minor_location': true});
      }
      final signs = ['mother', 'father', 'water'];
      expect(pickNext(m, signs, minimalPairs: pairs, rng: FakeRandom('mother', 0)), isIn(['father', 'water']));
      expect(pickNext(m, signs, minimalPairs: pairs, rng: FakeRandom('mother', 0.999)), 'mother');
    });
  });

  final fixture = File('$bundle/fixtures/learner.json');
  if (!fixture.existsSync()) {
    test('learner parity', () {}, skip: 'no fixture at ${fixture.path}');
    return;
  }
  final fx = jsonDecode(fixture.readAsStringSync()) as Map<String, dynamic>;
  final text = LearnerText.fromJson(jsonDecode(File('$bundle/text.json').readAsStringSync()));
  final meta = BundleMeta.fromJson(jsonDecode(File('$bundle/bundle.json').readAsStringSync()));

  group('parity with Python', () {
    test('readable values', () {
      (fx['readable'] as Map).forEach((p, values) {
        (values as Map).forEach((raw, want) => expect(readableValue(p, raw), want, reason: '$p/$raw'));
      });
    });

    test('coaching text and focus parameter', () {
      for (final c in fx['coach'] as List) {
        final verdicts = [
          for (final v in c['verdicts'] as List)
            Verdict(v['parameter'], v['predicted'], v['target'], v['correct'], (v['confidence'] as num).toDouble()),
        ];
        expect(coachText(verdicts, text.handshapes), c['text']);
        expect(focusParameter(verdicts), c['focus']);
      }
    });

    test('minimal pairs over the bundle signs', () {
      final signs = meta.signs.keys.toList()..sort();
      final got = findMinimalPairs(meta.signs, signs);
      (fx['minimal_pairs'] as Map).forEach((p, want) {
        expect([for (final (a, b) in got[p]!) [a, b]], want, reason: p);
      });
    });

    test('mastery after a fixed update sequence', () {
      final m = MasteryState();
      final mx = fx['mastery'] as Map<String, dynamic>;
      for (final u in mx['updates'] as List) {
        m.update(u['sign'], (u['verdicts'] as Map).cast<String, bool?>());
      }
      final want = mx['state'] as Map<String, dynamic>;
      final got = jsonDecode(jsonEncode(m.toJson())) as Map<String, dynamic>;
      for (final key in ['attempts', 'last_seen', 'clock', 'interval', 'repetitions']) {
        expect(got[key], want[key], reason: key);
      }
      for (final key in ['mastery', 'ease']) {
        void close(Object? a, Object? b) => a is Map
            ? (b as Map).forEach((k, v) => close(a[k], v))
            : expect((a as num).toDouble(), closeTo((b as num).toDouble(), 1e-12));
        close(got[key], want[key]);
      }
      (mx['due'] as Map).forEach((s, d) => expect(m.isDue(s), d, reason: s));
      (mx['sign_mastery'] as Map).forEach((s, v) => expect(m.signMastery(s), closeTo((v as num).toDouble(), 1e-12)));
    });

    test('every sign has a description', () {
      expect(text.signs.keys.toSet(), meta.signs.keys.toSet());
    });
  });
}

class _Recording implements SchedulerRandom {
  final List<List<String>> pools;

  _Recording(this.pools);

  @override
  String choose(List<String> pool, List<double> weights) {
    pools.add(pool);
    return pool.first;
  }

  @override
  double nextDouble() => 1;
}
