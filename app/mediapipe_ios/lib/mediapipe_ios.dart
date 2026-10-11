import 'package:flutter/services.dart';

/// Thin Dart side of the mediapipe_ios method channel.
class MediapipeIos {
  static const _channel = MethodChannel('mediapipe_ios');

  /// Returns which delegate the live (VIDEO-mode) landmarkers ended up on.
  static Future<String> init({required String poseAsset, required String handAsset}) async {
    final r = await _channel.invokeMethod('init', {'pose': poseAsset, 'hand': handAsset});
    return (r as Map)['videoDelegate'] as String;
  }

  /// IMAGE-mode detection on an encoded image; `gpu` uses GPU-delegate
  /// landmarkers (created on first use) instead of the CPU ones.
  static Future<Map<String, dynamic>> detectImage(Uint8List encoded, {bool gpu = false}) async =>
      Map<String, dynamic>.from(await _channel.invokeMethod('detectImage', {'bytes': encoded, 'gpu': gpu}));

  /// VIDEO-mode detection on a BGRA frame. `unmirror` reflects a mirrored
  /// (selfie) frame back to the camera's true view before detection.
  static Future<Map<String, dynamic>> detectFrame(
          Uint8List bgra, int width, int height, int rowBytes, int timestampMs, {bool unmirror = false}) async =>
      Map<String, dynamic>.from(await _channel.invokeMethod('detectFrame', {
        'bytes': bgra,
        'width': width,
        'height': height,
        'rowBytes': rowBytes,
        'timestampMs': timestampMs,
        'unmirror': unmirror,
      }));

  /// Every frame of a bundled video asset through fresh VIDEO-mode
  /// landmarkers on the chosen delegate; one detection result per frame.
  static Future<List<Map<String, dynamic>>> detectVideo(String asset, {required bool gpu}) async {
    final r = await _channel.invokeMethod('detectVideo', {'asset': asset, 'gpu': gpu});
    return [for (final f in (r as Map)['frames'] as List) Map<String, dynamic>.from(f as Map)];
  }
}
