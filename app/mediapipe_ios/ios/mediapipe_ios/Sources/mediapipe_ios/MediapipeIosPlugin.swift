import AVFoundation
import Accelerate
import CoreVideo
import Flutter
import MediaPipeTasksVision
import UIKit

/// Pose + hand landmarkers behind a method channel. `detectImage` runs IMAGE
/// mode on an encoded image (parity checks, on CPU or GPU); `detectFrame` runs
/// VIDEO mode on raw BGRA camera frames (live use), optionally reflecting a
/// mirrored selfie frame back to the camera's true view first. Landmarks come
/// back normalized to [0,1], as MediaPipe reports them, with MediaPipe's raw
/// handedness labels. `detectVideo` runs fresh VIDEO-mode landmarkers over
/// every frame of a bundled video asset, on CPU or GPU (delegate checks).
public class MediapipeIosPlugin: NSObject, FlutterPlugin {
  private let registrar: FlutterPluginRegistrar
  private var imagePose: PoseLandmarker?
  private var imageHands: HandLandmarker?
  private var gpuImagePose: PoseLandmarker?
  private var gpuImageHands: HandLandmarker?
  private var assets: (pose: String, hand: String)?
  private var videoPose: PoseLandmarker?
  private var videoHands: HandLandmarker?
  private let queue = DispatchQueue(label: "mediapipe_ios")
  private let poseQueue = DispatchQueue(label: "mediapipe_ios.pose")
  private let handQueue = DispatchQueue(label: "mediapipe_ios.hand")
  private var videoDelegate = "cpu"

  init(registrar: FlutterPluginRegistrar) { self.registrar = registrar }

  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(name: "mediapipe_ios", binaryMessenger: registrar.messenger())
    registrar.addMethodCallDelegate(MediapipeIosPlugin(registrar: registrar), channel: channel)
  }

  private func assetPath(_ asset: String) throws -> String {
    let key = registrar.lookupKey(forAsset: asset)
    guard let path = Bundle.main.path(forResource: key, ofType: nil) else {
      throw NSError(domain: "mediapipe_ios", code: 1, userInfo: [NSLocalizedDescriptionKey: "missing asset \(asset)"])
    }
    return path
  }

  private func makePose(_ asset: String, _ mode: RunningMode, gpu: Bool = false) throws -> PoseLandmarker {
    let o = PoseLandmarkerOptions()
    o.baseOptions.modelAssetPath = try assetPath(asset)
    o.baseOptions.delegate = gpu ? .GPU : .CPU
    o.runningMode = mode
    return try PoseLandmarker(options: o)
  }

  private func makeHands(_ asset: String, _ mode: RunningMode, gpu: Bool = false) throws -> HandLandmarker {
    let o = HandLandmarkerOptions()
    o.baseOptions.modelAssetPath = try assetPath(asset)
    o.baseOptions.delegate = gpu ? .GPU : .CPU
    o.runningMode = mode
    o.numHands = 2
    return try HandLandmarker(options: o)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    let args = call.arguments as? [String: Any] ?? [:]
    queue.async {
      do {
        let out: [String: Any]
        switch call.method {
        case "init":
          let pose = args["pose"] as! String
          let hand = args["hand"] as! String
          self.assets = (pose, hand)
          self.imagePose = try self.makePose(pose, .image)
          self.imageHands = try self.makeHands(hand, .image)
          // Parity checks stay on CPU (what Python ran); the live path tries
          // the GPU delegate and falls back to CPU if it cannot be created.
          do {
            self.videoPose = try self.makePose(pose, .video, gpu: true)
            self.videoHands = try self.makeHands(hand, .video, gpu: true)
            self.videoDelegate = "gpu"
          } catch {
            self.videoPose = try self.makePose(pose, .video)
            self.videoHands = try self.makeHands(hand, .video)
            self.videoDelegate = "cpu (gpu failed: \(error.localizedDescription))"
          }
          out = ["videoDelegate": self.videoDelegate]
        case "detectImage":
          let data = (args["bytes"] as! FlutterStandardTypedData).data
          guard let ui = UIImage(data: data) else { throw NSError(domain: "mediapipe_ios", code: 2) }
          let img = try MPImage(uiImage: ui)
          var pose = self.imagePose!, hands = self.imageHands!
          if args["gpu"] as? Bool ?? false {
            if self.gpuImagePose == nil, let a = self.assets {
              self.gpuImagePose = try self.makePose(a.pose, .image, gpu: true)
              self.gpuImageHands = try self.makeHands(a.hand, .image, gpu: true)
            }
            pose = self.gpuImagePose!
            hands = self.gpuImageHands!
          }
          out = Self.pack(try pose.detect(image: img), try hands.detect(image: img), Int(img.width), Int(img.height))
        case "detectFrame":
          let data = (args["bytes"] as! FlutterStandardTypedData).data
          let w = args["width"] as! Int, h = args["height"] as! Int, rowBytes = args["rowBytes"] as! Int
          let ts = args["timestampMs"] as! Int
          let unmirror = args["unmirror"] as? Bool ?? false
          let t0 = CFAbsoluteTimeGetCurrent()
          let img = try MPImage(pixelBuffer: try Self.pixelBuffer(data, w, h, rowBytes, unmirror: unmirror))
          let t1 = CFAbsoluteTimeGetCurrent()
          // Pose and hands are independent graphs: run them concurrently.
          var poseRes: Result<PoseLandmarkerResult, Error>!
          var handRes: Result<HandLandmarkerResult, Error>!
          var poseMs = 0.0, handMs = 0.0
          let group = DispatchGroup()
          self.poseQueue.async(group: group) {
            let s = CFAbsoluteTimeGetCurrent()
            poseRes = Result { try self.videoPose!.detect(videoFrame: img, timestampInMilliseconds: ts) }
            poseMs = (CFAbsoluteTimeGetCurrent() - s) * 1000
          }
          self.handQueue.async(group: group) {
            let s = CFAbsoluteTimeGetCurrent()
            handRes = Result { try self.videoHands!.detect(videoFrame: img, timestampInMilliseconds: ts) }
            handMs = (CFAbsoluteTimeGetCurrent() - s) * 1000
          }
          group.wait()
          var packed = Self.pack(try poseRes.get(), try handRes.get(), w, h)
          packed["copyMs"] = (t1 - t0) * 1000
          packed["poseMs"] = poseMs
          packed["handMs"] = handMs
          packed["totalMs"] = (CFAbsoluteTimeGetCurrent() - t0) * 1000
          out = packed
        case "detectVideo":
          out = try self.detectVideo(asset: args["asset"] as! String, gpu: args["gpu"] as? Bool ?? false)
        default:
          DispatchQueue.main.async { result(FlutterMethodNotImplemented) }
          return
        }
        DispatchQueue.main.async { result(out) }
      } catch {
        DispatchQueue.main.async {
          result(FlutterError(code: "mediapipe", message: "\(error)", details: nil))
        }
      }
    }
  }

  /// Copy a BGRA frame into a fresh pixel buffer, reflecting it horizontally
  /// when `unmirror` is set (a selfie-mirrored frame back to what the camera saw).
  /// Every frame of a video asset through new VIDEO-mode landmarkers (so no
  /// tracking state leaks in from the live camera), timestamped by the
  /// video's own presentation times.
  private func detectVideo(asset: String, gpu: Bool) throws -> [String: Any] {
    guard let a = assets else { throw NSError(domain: "mediapipe_ios", code: 4) }
    let pose = try makePose(a.pose, .video, gpu: gpu)
    let hands = try makeHands(a.hand, .video, gpu: gpu)
    let video = AVURLAsset(url: URL(fileURLWithPath: try assetPath(asset)))
    guard let track = video.tracks(withMediaType: .video).first else { throw NSError(domain: "mediapipe_ios", code: 5) }
    let reader = try AVAssetReader(asset: video)
    let output = AVAssetReaderTrackOutput(
      track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    reader.add(output)
    reader.startReading()
    var frames: [[String: Any]] = []
    var lastTs = -1
    while let sample = output.copyNextSampleBuffer() {
      guard let pb = CMSampleBufferGetImageBuffer(sample) else { continue }
      let ts = max(lastTs + 1, Int(CMSampleBufferGetPresentationTimeStamp(sample).seconds * 1000))
      lastTs = ts
      let img = try MPImage(pixelBuffer: pb)
      frames.append(Self.pack(try pose.detect(videoFrame: img, timestampInMilliseconds: ts),
                              try hands.detect(videoFrame: img, timestampInMilliseconds: ts),
                              CVPixelBufferGetWidth(pb), CVPixelBufferGetHeight(pb)))
    }
    if reader.status == .failed { throw reader.error ?? NSError(domain: "mediapipe_ios", code: 6) }
    return ["frames": frames]
  }

  private static func pixelBuffer(_ data: Data, _ w: Int, _ h: Int, _ rowBytes: Int, unmirror: Bool) throws -> CVPixelBuffer {
    var pb: CVPixelBuffer?
    let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
    guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs, &pb) == kCVReturnSuccess, let buf = pb else {
      throw NSError(domain: "mediapipe_ios", code: 3)
    }
    CVPixelBufferLockBaseAddress(buf, [])
    let dst = CVPixelBufferGetBaseAddress(buf)!
    let dstRow = CVPixelBufferGetBytesPerRow(buf)
    data.withUnsafeBytes { src in
      if unmirror {
        var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src.baseAddress!), height: vImagePixelCount(h),
                              width: vImagePixelCount(w), rowBytes: rowBytes)
        var d = vImage_Buffer(data: dst, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: dstRow)
        vImageHorizontalReflect_ARGB8888(&s, &d, vImage_Flags(kvImageNoFlags))
      } else {
        for y in 0..<h {
          memcpy(dst + y * dstRow, src.baseAddress! + y * rowBytes, min(rowBytes, dstRow))
        }
      }
    }
    CVPixelBufferUnlockBaseAddress(buf, [])
    return buf
  }

  /// Flat lists: pose = 33 x (x, y, visibility); each hand = 21 x (x, y) plus
  /// MediaPipe's raw handedness label and score.
  private static func pack(_ pose: PoseLandmarkerResult, _ hands: HandLandmarkerResult, _ w: Int, _ h: Int) -> [String: Any] {
    var out: [String: Any] = ["width": w, "height": h]
    if let p = pose.landmarks.first {
      let flat: [Double] = p.flatMap { [Double($0.x), Double($0.y), $0.visibility?.doubleValue ?? 0] }
      out["pose"] = flat
    }
    out["hands"] = zip(hands.landmarks, hands.handedness).map { lm, cat -> [String: Any] in
      let points: [Double] = lm.flatMap { [Double($0.x), Double($0.y)] }
      return ["label": cat.first?.categoryName ?? "", "score": Double(cat.first?.score ?? 0), "points": points]
    }
    return out
  }
}
