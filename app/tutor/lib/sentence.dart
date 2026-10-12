/// Sentence mode's pieces: the client for the sentence engine server
/// (server/ in the repo, hosted on an Oracle Cloud VM) and the player that
/// shows a sentence's bundled reference clips back to back.
///
/// The server's rule engine decides every gloss sequence; the app only
/// checks that each gloss has a bundled clip and reference features, and
/// refuses the sentence otherwise -- it never patches one up.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import 'engine.dart';

/// The deployed server (server/deploy/deploy.sh prints it): the Oracle VM's
/// IP as an sslip.io name, served over HTTPS by Caddy.
const sentenceServer = String.fromEnvironment('SENTENCE_SERVER', defaultValue: '');

/// The desktop's sentence-mode capture (diagnose_demo.py): a settle window
/// long enough to ride through pauses between words, measured in seconds on
/// real recordings and converted at the live frame rate.
const sentenceSettleSeconds = 1.2;
const sentenceCaptureMax = 400;

class SentenceGloss {
  final String text; // display gloss, e.g. WANT
  final String sign; // asllex id, e.g. want_2

  const SentenceGloss(this.text, this.sign);
}

class SentenceMarker {
  final String marker; // brow_raise | brow_furrow | headshake
  final int start, end; // gloss indices, end exclusive

  const SentenceMarker(this.marker, this.start, this.end);
}

class SentencePrompt {
  final String english;
  final List<SentenceGloss> glosses;
  final List<SentenceMarker> markers;

  const SentencePrompt(this.english, this.glosses, this.markers);

  List<String> get signs => [for (final g in glosses) g.sign];

  /// A GlossSequence.to_dict() from the server; null if out of scope.
  static SentencePrompt? fromJson(Map<String, dynamic> j) {
    if (j['in_scope'] != true) return null;
    return SentencePrompt(
      j['english'] as String,
      [for (final g in (j['glosses'] as List).cast<Map>()) SentenceGloss(g['text'] as String, g['asllex_id'] as String)],
      [
        for (final m in (j['nmm_tags'] as List).cast<Map>())
          SentenceMarker(m['marker'] as String, m['start'] as int, m['end'] as int),
      ],
    );
  }

  String markerLine(SentenceMarker m) {
    final name = switch (m.marker) {
      'brow_raise' => 'Raise your eyebrows',
      'brow_furrow' => 'Furrow your eyebrows',
      'headshake' => 'Shake your head',
      _ => m.marker,
    };
    return '$name during ${[for (final g in glosses.sublist(m.start, m.end)) g.text].join(' ')}';
  }
}

class SentenceError implements Exception {
  final String message;

  const SentenceError(this.message);

  @override
  String toString() => message;
}

class SentenceClient {
  final Uri base;
  final HttpClient _http = HttpClient()..connectionTimeout = const Duration(seconds: 20);

  SentenceClient([String server = sentenceServer]) : base = Uri.parse(server);

  /// The server usually answers from its pool of ready sentences; when a
  /// sign's pool is empty it writes and scores candidates on demand.
  static const timeout = Duration(seconds: 120);

  Future<Map<String, dynamic>> _post(String path, Map<String, dynamic> body) async {
    if (!base.hasScheme) throw const SentenceError('No sentence server is set up for this build.');
    try {
      final req = await _http.postUrl(base.resolve(path));
      req.headers.contentType = ContentType.json;
      req.write(jsonEncode(body));
      final res = await req.close().timeout(timeout);
      final text = await res.transform(utf8.decoder).join().timeout(timeout);
      if (res.statusCode == 200) return jsonDecode(text) as Map<String, dynamic>;
      String detail;
      try {
        detail = '${(jsonDecode(text) as Map)['detail']}';
      } on FormatException {
        detail = 'server error ${res.statusCode}';
      }
      throw SentenceError(res.statusCode == 503 ? 'No sentence this time ($detail). Try again.' : detail);
    } on SocketException {
      throw const SentenceError('Sentence mode needs an internet connection.');
    } on TimeoutException {
      throw const SentenceError('The sentence server took too long. It may be starting up; try again.');
    } on HandshakeException {
      throw const SentenceError('Could not make a secure connection to the sentence server.');
    }
  }

  /// An engine-accepted sentence that uses `target`.
  Future<SentencePrompt> forTarget(String target) async {
    final p = SentencePrompt.fromJson(await _post('sentence', {'target': target}));
    if (p == null) throw const SentenceError('The server returned an out-of-scope sentence.');
    return p;
  }

  /// The engine's reading of a typed sentence; throws its refusal reason.
  Future<SentencePrompt> gloss(String english) async {
    final j = await _post('gloss', {'english': english});
    final p = SentencePrompt.fromJson(j);
    if (p == null) throw SentenceError("Can't sign that one yet: ${j['reason']}");
    return p;
  }
}

/// Every gloss must have a bundled clip and reference features; a sentence
/// with any gap is refused whole.
String? missingReference(SentencePrompt p, Engine engine) {
  final missing = [for (final g in p.glosses) if (!engine.clips.containsKey(g.sign) || !engine.sentenceRefs.spans.containsKey(g.sign)) g.text];
  return missing.isEmpty ? null : 'No reference clip for ${missing.join(', ')}';
}

/// The sentence's reference clips played in order, looping -- real citation
/// clips cut together, so the joins show; never interpolated.
class SequencePlayer extends StatefulWidget {
  final Engine engine;
  final SentencePrompt prompt;

  const SequencePlayer({super.key, required this.engine, required this.prompt});

  @override
  State<SequencePlayer> createState() => _SequencePlayerState();
}

class _SequencePlayerState extends State<SequencePlayer> {
  final List<VideoPlayerController> _clips = [];
  int _index = 0;
  bool _ready = false;
  bool _advancing = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    for (final g in widget.prompt.glosses) {
      final c = VideoPlayerController.asset(widget.engine.clips[g.sign]!.asset);
      await c.initialize();
      await c.setVolume(0);
      if (!mounted) return c.dispose();
      _clips.add(c);
    }
    for (final c in _clips) {
      c.addListener(() => _onTick(c));
    }
    setState(() => _ready = true);
    await _clips.first.play();
  }

  void _onTick(VideoPlayerController c) {
    if (_advancing || c != _clips[_index]) return;
    final v = c.value;
    if (v.isInitialized && !v.isPlaying && v.position >= v.duration) {
      _advancing = true;
      final next = (_index + 1) % _clips.length;
      _clips[next].seekTo(Duration.zero).then((_) {
        if (!mounted) return;
        setState(() => _index = next);
        _advancing = false;
        _clips[next].play();
      });
    }
  }

  @override
  void dispose() {
    for (final c in _clips) {
      c.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_ready) return const AspectRatio(aspectRatio: 4 / 3, child: Center(child: CircularProgressIndicator()));
    final c = _clips[_index];
    return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      AspectRatio(aspectRatio: c.value.aspectRatio, child: VideoPlayer(c)),
      Padding(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        child: Text.rich(TextSpan(children: [
          for (var i = 0; i < widget.prompt.glosses.length; i++)
            TextSpan(
              text: '${widget.prompt.glosses[i].text} ',
              style: TextStyle(
                fontWeight: i == _index ? FontWeight.bold : FontWeight.normal,
                color: i == _index ? Theme.of(context).colorScheme.primary : null,
              ),
            ),
        ])),
      ),
    ]);
  }
}
