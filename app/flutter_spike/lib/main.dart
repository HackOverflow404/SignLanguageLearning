// Feasibility spike: does a Linux-built Flutter iOS app get (1) the camera and
// (2) ONNX Runtime running the real grader graph and (3) MediaPipe pose + hand
// landmarkers on the phone? The ORT check
// feeds one golden fixture's standardized features through grader.onnx and
// compares every output head against Python's recorded values.
import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:mediapipe_ios/mediapipe_ios.dart';

void main() => runApp(const MaterialApp(home: SpikePage()));

class SpikePage extends StatefulWidget {
  const SpikePage({super.key});

  @override
  State<SpikePage> createState() => _SpikePageState();
}

class _SpikePageState extends State<SpikePage> {
  CameraController? _camera;
  String _cameraStatus = 'starting...';
  String _ortStatus = 'running...';
  String _mpParity = 'running...';
  String _mpLive = 'waiting for camera...';
  bool _mpReady = false;
  String _mpDelegate = '?';
  bool _mpBusy = false;
  int _mpFrames = 0;
  int _mpHands = 0;
  final Stopwatch _mpClock = Stopwatch();

  @override
  void initState() {
    super.initState();
    _runGrader();
    _initMediaPipe().then((_) => _startCamera());
  }

  Future<void> _startCamera() async {
    try {
      final cams = await availableCameras();
      final front = cams.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cams.first,
      );
      final ctl = CameraController(front, ResolutionPreset.medium, enableAudio: false);
      await ctl.initialize();
      var frames = 0;
      final t0 = DateTime.now();
      await ctl.startImageStream((img) {
        frames++;
        _feedMediaPipe(img);
        if (frames % 30 == 0 && mounted) {
          final fps = frames / (DateTime.now().difference(t0).inMilliseconds / 1000);
          setState(() => _cameraStatus =
              'OK ${img.width}x${img.height} ${img.format.group.name}, ${fps.toStringAsFixed(1)} fps');
        }
      });
      setState(() => _camera = ctl);
    } catch (e) {
      setState(() => _cameraStatus = 'FAILED: $e');
    }
  }

  Future<void> _initMediaPipe() async {
    try {
      _mpDelegate = await MediapipeIos.init(
          poseAsset: 'assets/pose_landmarker_full.task', handAsset: 'assets/hand_landmarker.task');
      _mpReady = true;
      final png = (await rootBundle.load('assets/parity_frame.png')).buffer.asUint8List();
      final want = jsonDecode(await rootBundle.loadString('assets/parity_expected.json'));
      final got = await MediapipeIos.detectImage(png);
      final w = (want['width'] as num).toDouble(), h = (want['height'] as num).toDouble();

      // Worst landmark disagreement in pixels; pose visibility separately.
      double worstPx(List a, List b, int stride) {
        var m = 0.0;
        for (var i = 0; i + 1 < b.length; i += stride) {
          final dx = ((a[i] as num) - (b[i] as num)) * w, dy = ((a[i + 1] as num) - (b[i + 1] as num)) * h;
          final d = (dx * dx + dy * dy);
          if (d > m) m = d;
        }
        return m == 0 ? 0 : sqrt(m);
      }

      final poseGot = got['pose'] as List?;
      final poseWant = want['pose'] as List;
      var lines = <String>[];
      if (poseGot == null) {
        lines.add('pose MISSING');
      } else {
        var vis = 0.0;
        for (var i = 2; i < poseWant.length; i += 3) {
          vis = max(vis, ((poseGot[i] as num) - (poseWant[i] as num)).abs().toDouble());
        }
        lines.add('pose ${worstPx(poseGot, poseWant, 3).toStringAsFixed(2)} px, visibility diff ${vis.toStringAsFixed(3)}');
      }
      final handsGot = (got['hands'] as List).cast<Map>();
      for (final hw in (want['hands'] as List).cast<Map>()) {
        final match = handsGot.where((g) => g['label'] == hw['label']).toList();
        if (match.isEmpty) {
          lines.add('${hw['label']} hand MISSING');
        } else {
          final hg = match.first;
          lines.add('${hw['label']} hand ${worstPx(hg['points'] as List, hw['points'] as List, 2).toStringAsFixed(2)} px, '
              'score ${(hg['score'] as num).toStringAsFixed(3)} vs ${(hw['score'] as num).toStringAsFixed(3)}');
        }
      }
      setState(() => _mpParity = lines.join('\n  '));
    } catch (e) {
      setState(() => _mpParity = 'FAILED: $e');
    }
  }

  void _feedMediaPipe(CameraImage img) {
    if (!_mpReady || _mpBusy) return; // drop frames while one is in flight
    _mpBusy = true;
    if (!_mpClock.isRunning) _mpClock.start();
    final plane = img.planes.first;
    MediapipeIos.detectFrame(plane.bytes, img.width, img.height, plane.bytesPerRow, _mpClock.elapsedMilliseconds)
        .then((r) {
      _mpFrames++;
      _mpHands = (r['hands'] as List).length;
      if (_mpFrames % 10 == 0 && mounted) {
        final fps = _mpFrames / (_mpClock.elapsedMilliseconds / 1000);
        String ms(String k) => (r[k] as num).toStringAsFixed(0);
        setState(() => _mpLive = '${fps.toStringAsFixed(1)} fps on $_mpDelegate, pose ${r['pose'] != null ? 'yes' : 'no'}, '
            'hands $_mpHands\n  per frame: copy ${ms('copyMs')} ms, pose ${ms('poseMs')} ms, '
            'hands ${ms('handMs')} ms, total ${ms('totalMs')} ms');
      }
    }).catchError((Object e) {
      setState(() => _mpLive = 'FAILED: $e');
    }).whenComplete(() => _mpBusy = false);
  }

  Future<void> _runGrader() async {
    try {
      final fx = jsonDecode(await rootBundle.loadString('assets/spike_fixture.json'));
      final shape = (fx['shape'] as List).cast<int>();
      final features = Float32List.fromList((fx['features'] as List).map((v) => (v as num).toDouble()).toList());
      final tempo = Float32List.fromList((fx['tempo'] as List).map((v) => (v as num).toDouble()).toList());

      final session = await OnnxRuntime().createSessionFromAsset('assets/grader.onnx');
      final sw = Stopwatch()..start();
      final out = await session.run({
        'features': await OrtValue.fromList(features, [1, shape[0], shape[1]]),
        'tempo': await OrtValue.fromList(tempo, [1, 2]),
      });
      final ms = sw.elapsedMilliseconds;

      var worst = 0.0;
      final expected = fx['expected'] as Map<String, dynamic>;
      for (final entry in expected.entries) {
        final got = (await out[entry.key]!.asFlattenedList()).cast<num>();
        final want = (entry.value as List).cast<num>();
        for (var i = 0; i < want.length; i++) {
          final d = (got[i] - want[i]).abs().toDouble();
          if (d > worst) worst = d;
        }
      }
      final pass = worst < 1e-4;
      setState(() => _ortStatus =
          '${pass ? "PASS" : "MISMATCH"} on "${fx['sign']}" (${shape[0]} frames): '
          'max diff ${worst.toStringAsExponential(2)}, ${ms} ms');
    } catch (e) {
      setState(() => _ortStatus = 'FAILED: $e');
    }
  }

  @override
  void dispose() {
    _camera?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cam = _camera;
    return Scaffold(
      appBar: AppBar(title: const Text('ASL tutor: iOS spike')),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Expanded(
            child: cam != null && cam.value.isInitialized
                ? Center(child: CameraPreview(cam))
                : const Center(child: Text('no camera yet')),
          ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: Text(
                'Camera: $_cameraStatus\n\nONNX grader: $_ortStatus\n\n'
                'MediaPipe parity vs Python:\n  $_mpParity\n\nMediaPipe live: $_mpLive',
                style: const TextStyle(fontSize: 14)),
          ),
        ],
      ),
    );
  }
}
