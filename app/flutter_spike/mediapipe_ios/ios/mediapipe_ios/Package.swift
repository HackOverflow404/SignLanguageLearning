// swift-tools-version: 5.9
import PackageDescription

// MediaPipe's iOS release is CocoaPods-only; this wraps the same 0.10.35
// device frameworks as SwiftPM binary targets. The graph library registers its
// calculators from static initializers, so it must be force-loaded (the
// toolchain links with -all_load).
let package = Package(
  name: "mediapipe_ios",
  platforms: [.iOS("16.0")],
  products: [.library(name: "mediapipe-ios", targets: ["mediapipe_ios"])],
  targets: [
    .binaryTarget(name: "MediaPipeTasksCommon", path: "Frameworks/MediaPipeTasksCommon.xcframework"),
    .binaryTarget(name: "MediaPipeTasksVision", path: "Frameworks/MediaPipeTasksVision.xcframework"),
    .binaryTarget(name: "MediaPipeTasksGraph", path: "Frameworks/MediaPipeTasksGraph.xcframework"),
    .target(
      name: "mediapipe_ios",
      dependencies: ["MediaPipeTasksCommon", "MediaPipeTasksVision", "MediaPipeTasksGraph"],
      linkerSettings: [
        .linkedLibrary("c++"),
        .linkedFramework("Accelerate"),
        .linkedFramework("AVFoundation"),
        .linkedFramework("CoreGraphics"),
        .linkedFramework("CoreImage"),
        .linkedFramework("CoreMedia"),
        .linkedFramework("CoreVideo"),
        .linkedFramework("QuartzCore"),
      ]
    ),
  ]
)
