// ASL tutor, single-sign mode: watch the target's real reference clip, sign
// it, and get a verdict per phonological parameter -- distance to the known
// target, never "which sign was that".
//
// Camera frames: the front camera's stream is selfie-mirrored (the camera
// plugin sets isVideoMirrored on it). Grading must see the camera's TRUE view,
// like the unflipped ASL Citizen training video (CLAUDE.md issue #6), so every
// front-camera frame is un-mirrored natively BEFORE detection; the preview
// stays mirrored, which is what a learner expects to see.
import 'dart:async';
import 'dart:math' as math;

import 'package:aslcv_core/aslcv_core.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:mediapipe_ios/mediapipe_ios.dart';
import 'package:video_player/video_player.dart';

import 'delegate_check.dart';
import 'engine.dart';

const _maxCaptureFrames = 150; // diagnose_demo.py's single-sign cap

void main() => runApp(MaterialApp(
      title: 'ASL Tutor',
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      darkTheme: ThemeData(colorSchemeSeed: Colors.indigo, brightness: Brightness.dark, useMaterial3: true),
      home: const LoadingPage(),
    ));

class LoadingPage extends StatefulWidget {
  const LoadingPage({super.key});

  @override
  State<LoadingPage> createState() => _LoadingPageState();
}

class _LoadingPageState extends State<LoadingPage> {
  String? _error;

  @override
  void initState() {
    super.initState();
    Engine.load().then((engine) {
      if (!mounted) return;
      Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (_) => TutorPage(engine: engine)));
    }).catchError((Object e) {
      if (mounted) setState(() => _error = '$e');
    });
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        body: Center(
          child: _error == null
              ? const CircularProgressIndicator()
              : Padding(padding: const EdgeInsets.all(24), child: Text('Could not start:\n$_error')),
        ),
      );
}

class TutorPage extends StatefulWidget {
  final Engine engine;

  const TutorPage({super.key, required this.engine});

  @override
  State<TutorPage> createState() => _TutorPageState();
}

class _TutorPageState extends State<TutorPage> {
  Engine get engine => widget.engine;

  late String _target = engine.signs.contains('mother') ? 'mother' : engine.signs.first;
  late final CaptureBuffer<Pose> _capture = CaptureBuffer<Pose>(
    (frames) => handMotionEnergy(frames, engine.skeleton),
    engine.meta.motionThreshold,
    settleFrames: engine.meta.liveSettleFrames,
    preroll: engine.meta.livePreroll,
    maxFrames: _maxCaptureFrames,
  );
  final List<int> _captureTimes = []; // ms timestamps, parallel to _capture.frames

  CameraController? _camera;
  bool _unmirror = false;
  VideoPlayerController? _reference;
  String? _cameraError;

  bool _busy = false;
  final Stopwatch _clock = Stopwatch();
  int _processed = 0;
  double _fps = 0;
  bool _leftHand = false, _rightHand = false;
  final HandStats _handStats = HandStats();

  Grade? _grade;
  double _gradeFps = 0;
  int _gradeFrames = 0;
  bool _grading = false;
  String? _gradeError;

  @override
  void initState() {
    super.initState();
    _loadReference();
    _startCamera();
  }

  @override
  void dispose() {
    _camera?.dispose();
    _reference?.dispose();
    super.dispose();
  }

  Future<void> _loadReference() async {
    final old = _reference;
    final ctl = VideoPlayerController.asset(engine.clips[_target]!.asset);
    setState(() => _reference = null);
    await old?.dispose();
    await ctl.initialize();
    await ctl.setLooping(true);
    await ctl.setVolume(0);
    await ctl.play();
    if (mounted) setState(() => _reference = ctl);
  }

  Future<void> _startCamera() async {
    try {
      final cams = await availableCameras();
      final cam = cams.firstWhere((c) => c.lensDirection == CameraLensDirection.front, orElse: () => cams.first);
      final ctl = CameraController(cam, ResolutionPreset.medium,
          enableAudio: false, imageFormatGroup: ImageFormatGroup.bgra8888);
      await ctl.initialize();
      _unmirror = cam.lensDirection == CameraLensDirection.front;
      _clock.start();
      await ctl.startImageStream(_onFrame);
      if (mounted) setState(() => _camera = ctl);
    } catch (e) {
      if (mounted) setState(() => _cameraError = '$e');
    }
  }

  /// One camera frame -> landmarks -> capture buffer. Frames arriving while
  /// one is still in flight are dropped, so the buffer only ever holds frames
  /// in capture order.
  void _onFrame(CameraImage img) {
    if (_busy) return;
    _busy = true;
    final t = _clock.elapsedMilliseconds;
    final plane = img.planes.first;
    MediapipeIos.detectFrame(plane.bytes, img.width, img.height, plane.bytesPerRow, t, unmirror: _unmirror)
        .then((r) {
      final pose = engine.poseOf(r);
      _processed++;
      if (_processed % 15 == 0) _fps = _processed / (_clock.elapsedMilliseconds / 1000);
      final wasSettled = _capture.state == CaptureState.settled;
      _capture.append(pose);
      if (!wasSettled) {
        _captureTimes.add(t);
        // The buffer trims itself while idle; keep the timestamps aligned.
        if (_captureTimes.length > _capture.frames.length) {
          _captureTimes.removeRange(0, _captureTimes.length - _capture.frames.length);
        }
      }
      final sk = engine.skeleton;
      _leftHand = pose.scores[sk.anchor('left_hand_wrist')] > 0;
      _rightHand = pose.scores[sk.anchor('right_hand_wrist')] > 0;
      _handStats.add(pose, sk);
      if (!wasSettled && _capture.state == CaptureState.settled) _gradeCapture();
      if (mounted) setState(() {});
    }).catchError((Object e) {
      if (mounted) setState(() => _cameraError = 'landmarks: $e');
    }).whenComplete(() => _busy = false);
  }

  Future<void> _gradeCapture() async {
    final frames = List<Pose>.of(_capture.frames);
    final times = List<int>.of(_captureTimes);
    setState(() {
      _grading = true;
      _gradeError = null;
    });
    try {
      final grade = await gradeAttempt(engine.grader, engine.skeleton, frames, _target);
      if (!mounted) return;
      setState(() {
        _grade = grade;
        _gradeFrames = frames.length;
        _gradeFps = times.length > 1 ? 1000 * (times.length - 1) / (times.last - times.first) : 0;
      });
    } catch (e) {
      if (mounted) setState(() => _gradeError = '$e');
    } finally {
      if (mounted) setState(() => _grading = false);
    }
  }

  void _tryAgain() => setState(() {
        _capture.reset();
        _captureTimes.clear();
        _grade = null;
        _gradeError = null;
      });

  void _setTarget(String sign) {
    if (sign == _target) return;
    _target = sign;
    _tryAgain();
    _loadReference();
  }

  void _nextSign() {
    final signs = engine.signs;
    _setTarget(signs[(signs.indexOf(_target) + 1) % signs.length]);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: DropdownButton<String>(
          value: _target,
          underline: const SizedBox(),
          items: [for (final s in engine.signs) DropdownMenuItem(value: s, child: Text(glossLabel(s)))],
          onChanged: (s) => s == null ? null : _setTarget(s),
        ),
        actions: [
          IconButton(icon: const Icon(Icons.info_outline), onPressed: () => _showDiagnostics(context)),
        ],
      ),
      body: SafeArea(
        child: LayoutBuilder(builder: (context, box) {
          final panel = SingleChildScrollView(child: _bottomPanel(context));
          if (box.maxWidth > box.maxHeight) {
            return Row(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
              Expanded(child: _liveView(box.maxWidth * 0.6)),
              SizedBox(width: box.maxWidth * 0.4, child: panel),
            ]);
          }
          return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            Expanded(child: _liveView(box.maxWidth)),
            ConstrainedBox(constraints: BoxConstraints(maxHeight: box.maxHeight * 0.5), child: panel),
          ]);
        }),
      ),
    );
  }

  Widget _liveView(double width) {
    final cam = _camera;
    final ref = _reference;
    return Stack(
      fit: StackFit.expand,
      children: [
        ColoredBox(
          color: Colors.black,
          child: cam != null && cam.value.isInitialized
              ? Center(child: CameraPreview(cam))
              : Center(
                  child: Text(_cameraError ?? 'starting camera...', style: const TextStyle(color: Colors.white))),
        ),
        Positioned(
          left: 8,
          top: 8,
          width: math.min(240, width * 0.4),
          child: Card(
            clipBehavior: Clip.antiAlias,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                AspectRatio(
                  aspectRatio: ref != null && ref.value.isInitialized ? ref.value.aspectRatio : 4 / 3,
                  child: ref != null && ref.value.isInitialized
                      ? VideoPlayer(ref)
                      : const Center(child: CircularProgressIndicator()),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  child: Text('Copy this: ${glossLabel(_target)}', style: const TextStyle(fontWeight: FontWeight.bold)),
                ),
              ],
            ),
          ),
        ),
        Positioned(right: 8, top: 8, child: _stateBadge()),
        Positioned(right: 8, top: 52, child: _handIndicator()),
      ],
    );
  }

  Widget _stateBadge() {
    final (label, color) = switch (_capture.state) {
      CaptureState.idle => ('READY', Colors.blueGrey),
      CaptureState.active => ('CAPTURING', Colors.orange),
      CaptureState.settled => ('CAPTURED', Colors.green),
    };
    return Chip(
      label: Text(label, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
      backgroundColor: color,
      side: BorderSide.none,
    );
  }

  /// Which anatomical hand slots are filled right now: the on-device check
  /// that un-mirroring is right (raise your RIGHT hand -> R lights).
  Widget _handIndicator() {
    Widget dot(String s, bool on) => Container(
          margin: const EdgeInsets.only(left: 4),
          width: 30,
          height: 30,
          alignment: Alignment.center,
          decoration: BoxDecoration(shape: BoxShape.circle, color: on ? Colors.lightGreenAccent : Colors.black45),
          child: Text(s, style: TextStyle(fontWeight: FontWeight.bold, color: on ? Colors.black : Colors.white70)),
        );
    return Row(children: [dot('L', _leftHand), dot('R', _rightHand)]);
  }

  Widget _bottomPanel(BuildContext context) {
    final theme = Theme.of(context);
    final grade = _grade;
    final children = <Widget>[];
    if (_gradeError != null) {
      children.add(Text('Grading failed: $_gradeError', style: TextStyle(color: theme.colorScheme.error)));
    } else if (_grading) {
      children.add(const Row(children: [
        SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2)),
        SizedBox(width: 12),
        Text('Grading...'),
      ]));
    } else if (grade == null) {
      children.add(Text(
        switch (_capture.state) {
          CaptureState.idle => 'Watch the clip, then sign ${glossLabel(_target)}. Start from rest, '
              'hands down; capture starts when you move and stops when you rest again.',
          CaptureState.active => 'Capturing... return your hands to rest when you finish.',
          CaptureState.settled => 'Captured.',
        },
        style: theme.textTheme.titleMedium,
      ));
    } else {
      final judged = grade.parameters.values.where((v) => v.correct != null).toList();
      final right = judged.where((v) => v.correct == true).length;
      children.add(Text('$right of ${judged.length} graded aspects match ${glossLabel(_target)}',
          style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)));
      children.add(const SizedBox(height: 4));
      for (final p in _parameterOrder) {
        final v = grade.parameters[p];
        if (v != null) children.add(_verdictRow(v, theme));
      }
      children.add(const SizedBox(height: 4));
      children.add(Text(
        'Distance to reference ${grade.fidelity.toStringAsFixed(2)} (lower is closer) · '
        '$_gradeFrames frames at ${_gradeFps.toStringAsFixed(0)} fps',
        style: theme.textTheme.bodySmall,
      ));
      children.add(Text(
        'Model feedback, not a Deaf reviewer: a wrong verdict is often the model, not you.',
        style: theme.textTheme.bodySmall?.copyWith(fontStyle: FontStyle.italic),
      ));
    }
    children.add(const SizedBox(height: 8));
    children.add(Row(children: [
      Expanded(
        child: OutlinedButton.icon(
          onPressed: _capture.state == CaptureState.idle && _grade == null ? null : _tryAgain,
          icon: const Icon(Icons.replay),
          label: const Text('Try again'),
        ),
      ),
      const SizedBox(width: 12),
      Expanded(
        child: FilledButton.icon(onPressed: _nextSign, icon: const Icon(Icons.skip_next), label: const Text('Next sign')),
      ),
    ]));
    return Material(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: children),
      ),
    );
  }

  Widget _verdictRow(Verdict v, ThemeData theme) {
    final (icon, color) = switch (v.correct) {
      true => (Icons.check_circle, Colors.green),
      false => (Icons.cancel, Colors.red),
      null => (Icons.remove_circle_outline, Colors.grey),
    };
    final detail = v.correct == null
        ? 'not graded: too few curriculum signs share this target'
        : v.correct!
            ? 'matches: ${valueLabel(v.parameter, v.target)}'
            : 'you: ${valueLabel(v.parameter, v.predicted)} · target: ${valueLabel(v.parameter, v.target)}';
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color, size: 22),
          const SizedBox(width: 8),
          Expanded(
            child: Text.rich(TextSpan(children: [
              TextSpan(text: '${_parameterNames[v.parameter]}  ', style: const TextStyle(fontWeight: FontWeight.bold)),
              TextSpan(text: detail),
            ])),
          ),
          Text('${(v.confidence * 100).round()}%', style: theme.textTheme.bodySmall),
        ],
      ),
    );
  }

  void _showDiagnostics(BuildContext context) {
    final size = _camera?.value.previewSize;
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      builder: (_) => _Diagnostics(
        engine: engine,
        lines: () => [
          'Landmarks: ${engine.delegate}, ${_fps.toStringAsFixed(1)} fps processed',
          'Camera: ${size == null ? '-' : '${size.width.round()}x${size.height.round()}'}, '
              'un-mirrored before detection: $_unmirror',
          'Reference clip: ${engine.clips[_target]!.videoId} (signer ${engine.clips[_target]!.signerId})',
          '',
          'Hand test: hold only your RIGHT hand up, still, for a few seconds.',
          ..._handStats.describe(),
        ],
        onReset: _handStats.reset,
      ),
    );
  }
}

/// Live counts behind the on-device mirroring check. Frames with only one
/// hand slot filled show which slot a raised hand lands in; independently,
/// each filled slot is checked against BlazePose's own whole-body wrist
/// labels -- the training cache disagrees on 1-3% of frames, so a much
/// higher rate here means MediaPipe's handedness is flipping, not a fixed
/// mirroring error (a mirroring error would flip both labels together).
class HandStats {
  int onlyLeft = 0, onlyRight = 0, both = 0;
  final _checked = [0, 0], _disagree = [0, 0];

  void reset() {
    onlyLeft = onlyRight = both = 0;
    _checked.fillRange(0, 2, 0);
    _disagree.fillRange(0, 2, 0);
  }

  void add(Pose p, Skeleton sk) {
    final l = sk.anchor('left_hand_wrist'), r = sk.anchor('right_hand_wrist');
    final pl = sk.anchor('left_body_wrist'), pr = sk.anchor('right_body_wrist');
    final hasL = p.scores[l] > 0, hasR = p.scores[r] > 0;
    if (hasL && hasR) both++;
    if (hasL && !hasR) onlyLeft++;
    if (hasR && !hasL) onlyRight++;
    if (p.scores[pl] < 0.5 || p.scores[pr] < 0.5) return;
    double d(int a, int b) =>
        math.pow(p.keypoints[2 * a] - p.keypoints[2 * b], 2).toDouble() +
        math.pow(p.keypoints[2 * a + 1] - p.keypoints[2 * b + 1], 2).toDouble();
    for (final (i, hand, own, other) in [(0, l, pl, pr), (1, r, pr, pl)]) {
      if (p.scores[hand] <= 0) continue;
      _checked[i]++;
      if (d(hand, other) < d(hand, own)) _disagree[i]++;
    }
  }

  List<String> describe() {
    String rate(int i) => _checked[i] == 0
        ? 'no frames yet'
        : '${_disagree[i]}/${_checked[i]} (${(100 * _disagree[i] / _checked[i]).toStringAsFixed(1)}%)';
    return [
      'Frames with only R: $onlyRight, only L: $onlyLeft, both: $both',
      'Slot disagrees with body wrist: left ${rate(0)}, right ${rate(1)} (training data: 1-3%)',
    ];
  }
}

class _Diagnostics extends StatefulWidget {
  final Engine engine;
  final List<String> Function() lines;
  final VoidCallback onReset;

  const _Diagnostics({required this.engine, required this.lines, required this.onReset});

  @override
  State<_Diagnostics> createState() => _DiagnosticsState();
}

class _DiagnosticsState extends State<_Diagnostics> {
  String _cpu = 'running...', _gpu = 'running...';
  String? _delegateCheck;
  bool _checking = false;

  Future<void> _runDelegateCheck() async {
    setState(() {
      _checking = true;
      _delegateCheck = 'starting...';
    });
    try {
      final report = await runDelegateCheck(widget.engine,
          progress: (s) => mounted ? setState(() => _delegateCheck = s) : null);
      if (mounted) setState(() => _delegateCheck = report.describe().join('\n'));
    } catch (e) {
      if (mounted) setState(() => _delegateCheck = 'FAILED: $e');
    } finally {
      if (mounted) setState(() => _checking = false);
    }
  }
  late final Timer _refresh = Timer.periodic(const Duration(milliseconds: 500), (_) => setState(() {}));

  @override
  void dispose() {
    _refresh.cancel();
    super.dispose();
  }

  @override
  void initState() {
    super.initState();
    _refresh;
    landmarkParity(gpu: false)
        .then((s) => setState(() => _cpu = s))
        .catchError((Object e) => setState(() => _cpu = 'FAILED: $e'));
    landmarkParity(gpu: true)
        .then((s) => setState(() => _gpu = s))
        .catchError((Object e) => setState(() => _gpu = 'FAILED: $e'));
  }

  @override
  Widget build(BuildContext context) => SafeArea(
        child: ConstrainedBox(
          constraints: BoxConstraints(maxHeight: MediaQuery.of(context).size.height * 0.85),
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              SelectableText(
                '${widget.lines().join('\n')}\n\n'
                'Landmarks vs Python on a still frame, worst point:\n'
                'CPU:\n$_cpu\n\nGPU (what live grading uses):\n$_gpu',
              ),
              const SizedBox(height: 8),
              OutlinedButton(onPressed: widget.onReset, child: const Text('Reset hand counts')),
              const SizedBox(height: 16),
              const Text('Grades on GPU vs CPU landmarks, over 20 reference clips (about 2 minutes; '
                  'keep this sheet open):'),
              const SizedBox(height: 4),
              if (_delegateCheck != null) SelectableText(_delegateCheck!),
              const SizedBox(height: 4),
              FilledButton(
                onPressed: _checking ? null : _runDelegateCheck,
                child: Text(_checking ? 'Checking...' : 'Run GPU vs CPU check'),
              ),
            ]),
          ),
        ),
      );
}

const _parameterOrder = ['handshape', 'major_location', 'minor_location', 'movement', 'repeated_movement'];
const _parameterNames = {
  'handshape': 'Handshape',
  'major_location': 'Location',
  'minor_location': 'Exact spot',
  'movement': 'Movement',
  'repeated_movement': 'Repeats',
};

/// "want_2" -> "WANT (2)": the curriculum's id-gloss, readable.
String glossLabel(String sign) {
  final m = RegExp(r'^(.*)_(\d+)$').firstMatch(sign);
  return m == null ? sign.toUpperCase() : '${m[1]!.toUpperCase()} (${m[2]})';
}

/// Model class names -> words: "open_b" -> "open B", "CheekNose" -> "cheek nose".
String valueLabel(String parameter, String value) {
  if (parameter == 'repeated_movement') return value == 'True' ? 'repeated' : 'not repeated';
  if (parameter == 'handshape') {
    return value.split('_').map((w) => w.length == 1 ? w.toUpperCase() : w).join(' ');
  }
  return value.replaceAllMapped(RegExp(r'(?<=[a-z])([A-Z])'), (m) => ' ${m[1]}').toLowerCase();
}
