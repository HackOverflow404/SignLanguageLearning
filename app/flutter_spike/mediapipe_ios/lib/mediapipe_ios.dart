import 'package:flutter/services.dart';

/// Thin Dart side of the mediapipe_ios method channel.
class MediapipeIos {
  static const _channel = MethodChannel('mediapipe_ios');

  /// Returns which delegate the live (VIDEO-mode) landmarkers ended up on.
  static Future<String> init({required String poseAsset, required String handAsset}) async {
    final r = await _channel.invokeMethod('init', {'pose': poseAsset, 'hand': handAsset});
    return (r as Map)['videoDelegate'] as String;
  }

  static Future<Map<String, dynamic>> detectImage(Uint8List encoded) async =>
      Map<String, dynamic>.from(await _channel.invokeMethod('detectImage', {'bytes': encoded}));

  static Future<Map<String, dynamic>> detectFrame(
          Uint8List bgra, int width, int height, int rowBytes, int timestampMs) async =>
      Map<String, dynamic>.from(await _channel.invokeMethod('detectFrame', {
        'bytes': bgra,
        'width': width,
        'height': height,
        'rowBytes': rowBytes,
        'timestampMs': timestampMs,
      }));
}
