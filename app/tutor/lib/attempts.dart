/// Opt-in attempt logging (Diagnostics sheet): each attempt's pose stream
/// and capture states go to Documents/attempts/ as `<stamp>.bin.gz` +
/// `<stamp>.json`, keypoints only. The folder shows in the Files app, and
/// scripts/import_phone_attempts.py pulls it over USB and converts it for
/// scripts/replay_attempts.py.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';

class AttemptLog {
  final Directory _dir;
  final File _settings;
  bool enabled;

  AttemptLog._(this._dir, this._settings, this.enabled);

  /// Off unless the learner turned it on; the choice survives launches.
  static Future<AttemptLog> load() async {
    final docs = (await getApplicationDocumentsDirectory()).path;
    final settings = File('$docs/attempt_log.json');
    var enabled = false;
    try {
      if (await settings.exists()) enabled = jsonDecode(await settings.readAsString())['enabled'] == true;
    } on FormatException {
      // unreadable: treat as off
    }
    return AttemptLog._(Directory('$docs/attempts'), settings, enabled);
  }

  Future<void> setEnabled(bool on) async {
    enabled = on;
    await _writeAtomic(_settings, jsonEncode({'enabled': on}));
  }

  Future<int> count() async =>
      await _dir.exists() ? await _dir.list().where((e) => e.path.endsWith('.json')).length : 0;

  Future<void> clear() async {
    if (await _dir.exists()) await _dir.delete(recursive: true);
  }

  /// Writes the recording (gzipped off the UI isolate) and its sidecar;
  /// the sidecar goes last, so a JSON file always has its frames.
  Future<void> save(Uint8List rawFrames, Map<String, Object?> meta) async {
    await _dir.create(recursive: true);
    final now = DateTime.now();
    String two(int v) => v.toString().padLeft(2, '0');
    final stamp = '${now.year}${two(now.month)}${two(now.day)}-${two(now.hour)}${two(now.minute)}${two(now.second)}'
        '-${now.millisecond.toString().padLeft(3, '0')}';
    final packed = await Isolate.run(() => gzip.encode(rawFrames));
    await _writeAtomic(File('${_dir.path}/$stamp.bin.gz'), packed);
    await _writeAtomic(File('${_dir.path}/$stamp.json'), const JsonEncoder.withIndent('  ').convert(meta));
  }
}

/// Temp file + rename, so a kill mid-write never leaves a truncated file.
Future<void> _writeAtomic(File file, Object contents) async {
  final tmp = File('${file.path}.tmp');
  if (contents is String) {
    await tmp.writeAsString(contents, flush: true);
  } else {
    await tmp.writeAsBytes(contents as List<int>, flush: true);
  }
  await tmp.rename(file.path);
}
