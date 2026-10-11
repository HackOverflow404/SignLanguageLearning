"""Phase 8: the reference clip for every bundle sign, as phone-playable video.

Each sign's clip is the one retrieval.fetch_reference picks (the same rule
diagnose_demo.py shows), cut to its motion-active span exactly as Phase 5a's
stitched sentence video cuts it -- real Deaf-signed footage, retrieved and
trimmed, never generated. Written as H.264 mp4 (no audio, scaled to --height)
to models/mobile/clips/<sign>.mp4, plus clips.json (sign -> file, source video
id, signer, frame count) so the app never guesses a filename.

    .venv/bin/python scripts/export_reference_clips.py
"""
import argparse
import json
import subprocess
from pathlib import Path

from aslcv.production.retrieval import _trimmed_frames, fetch_reference, skeleton_for

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = REPO / "models" / "mobile"


def encode(frames, fps, out_path: Path, height: int):
    """Raw BGR frames -> H.264 baseline mp4 via ffmpeg (what AVPlayer plays)."""
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-",
           "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-profile:v", "baseline", "-pix_fmt", "yuv420p",
           "-crf", "26", "-movflags", "+faststart", "-an", str(out_path)]
    subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    meta = json.loads((args.bundle / "bundle.json").read_text())
    extractor = meta["extractor"]
    out_dir = args.bundle / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    skeleton = skeleton_for(extractor)
    index = {}
    for sign in sorted(meta["signs"]):
        clip = fetch_reference(sign, extractor)
        if clip.npz_path is None:
            raise SystemExit(f"{sign}: no cached {extractor} poses to trim its reference clip by")
        frames, fps = _trimmed_frames(clip, skeleton)
        if not frames:
            raise SystemExit(f"{sign}: trimmed reference clip is empty")
        name = f"{sign}.mp4"
        encode(frames, fps, out_dir / name, args.height)
        index[sign] = {"file": name, "video_id": clip.video_id, "signer_id": clip.signer_id,
                       "frames": len(frames), "fps": round(fps, 3)}
    (out_dir / "clips.json").write_text(json.dumps(index, indent=1))
    size = sum(f.stat().st_size for f in out_dir.iterdir()) / 1e6
    print(f"wrote {len(index)} reference clips -> {out_dir.relative_to(REPO)}/ ({size:.1f} MB)")


if __name__ == "__main__":
    main()
