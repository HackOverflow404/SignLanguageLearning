#!/usr/bin/env bash
# Download MediaPipe Tasks 0.10.35 for iOS (CocoaPods-only upstream) and lay it
# out as the SwiftPM binary targets Package.swift expects: device slices only,
# plus the graph library wrapped as a static-library xcframework.
#
# MediaPipe's archives bundle XNNPACK, pthreadpool, cpuinfo and KleidiAI, which
# ONNX Runtime's prelinked library also defines; under the toolchain's
# -all_load both copies load and the link fails on duplicate symbols. Every
# clashing symbol (xnnpack_rename.txt, from comparing the two libraries'
# exported symbols) is renamed with an mpx_ prefix, so each runtime keeps its
# own private copy.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
F="$here/ios/mediapipe_ios/Frameworks"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

curl -fsSL https://dl.google.com/cpdc/20260427-204214/MediaPipeTasksCommon-0.10.35.tar.gz | tar -xz -C "$tmp"
curl -fsSL https://dl.google.com/cpdc/20260427-204221/MediaPipeTasksVision-0.10.35.tar.gz | tar -xz -C "$tmp"

rm -rf "$F"
mkdir -p "$F/MediaPipeTasksGraph.xcframework/ios-arm64"
for fw in MediaPipeTasksCommon MediaPipeTasksVision; do
  mkdir -p "$F/$fw.xcframework"
  cp -R "$tmp/frameworks/$fw.xcframework/ios-arm64" "$F/$fw.xcframework/"
done
cp "$tmp/frameworks/graph_libraries/libMediaPipeTasksCommon_device_graph.a" "$F/MediaPipeTasksGraph.xcframework/ios-arm64/"
cp "$tmp/LICENSE" "$F/LICENSE.mediapipe"

python3 - "$F" <<'PY'
import os, plistlib, sys
F = sys.argv[1]
for name, lib in [("MediaPipeTasksCommon", "MediaPipeTasksCommon.framework"),
                  ("MediaPipeTasksVision", "MediaPipeTasksVision.framework"),
                  ("MediaPipeTasksGraph", "libMediaPipeTasksCommon_device_graph.a")]:
    with open(os.path.join(F, f"{name}.xcframework", "Info.plist"), "wb") as f:
        plistlib.dump({"AvailableLibraries": [{"LibraryIdentifier": "ios-arm64", "LibraryPath": lib,
                       "SupportedArchitectures": ["arm64"], "SupportedPlatform": "ios"}],
                       "CFBundlePackageType": "XFWK", "XCFrameworkFormatVersion": "1.0"}, f)
PY

for bin in MediaPipeTasksGraph.xcframework/ios-arm64/libMediaPipeTasksCommon_device_graph.a \
           MediaPipeTasksCommon.xcframework/ios-arm64/MediaPipeTasksCommon.framework/MediaPipeTasksCommon \
           MediaPipeTasksVision.xcframework/ios-arm64/MediaPipeTasksVision.framework/MediaPipeTasksVision; do
  llvm-objcopy --redefine-syms="$here/tool/xnnpack_rename.txt" "$F/$bin"
done
echo "MediaPipe frameworks ready in $F"
