/// The learner's mastery, kept in the app's documents directory across
/// launches. Same JSON as Python's data/learner_state.json.
library;

import 'dart:convert';
import 'dart:io';

import 'package:aslcv_core/aslcv_core.dart';
import 'package:path_provider/path_provider.dart';

class Progress {
  final File _file;
  final MasteryState state;

  Progress._(this._file, this.state);

  /// A missing or unreadable file is a fresh learner.
  static Future<Progress> load() async {
    final file = File('${(await getApplicationDocumentsDirectory()).path}/learner_state.json');
    var state = MasteryState();
    try {
      if (await file.exists()) state = MasteryState.fromJson(jsonDecode(await file.readAsString()));
    } on FormatException {
      // keep the fresh state; the next save replaces the bad file
    }
    return Progress._(file, state);
  }

  /// Temp file + rename, so a kill mid-write never leaves a truncated file.
  Future<void> save() async {
    final tmp = File('${_file.path}.tmp');
    await tmp.writeAsString(jsonEncode(state.toJson()), flush: true);
    await tmp.rename(_file.path);
  }

  Future<void> reset() async {
    state.mastery.clear();
    state.attempts.clear();
    state.lastSeen.clear();
    state.interval.clear();
    state.ease.clear();
    state.repetitions.clear();
    state.clock = 0;
    await save();
  }
}
