"""Import attempts logged on the phone (Diagnostics -> "Log attempts") into
data/live_sessions/ in the desktop's format, so scripts/replay_attempts.py
replays them through the real CaptureBuffer unchanged.

The phone writes Documents/attempts/<stamp>.bin.gz (frame data, layout in
app/aslcv_core/lib/src/attempt_log.dart) + <stamp>.json (sidecar). Either
pull that folder over USB (the app shares its Documents folder; needs
libimobiledevice's afcclient and the phone plugged in and unlocked), or copy
it off by hand (Files app -> On My iPhone -> ASL Tutor -> attempts):

    .venv/bin/python scripts/import_phone_attempts.py --pull
    .venv/bin/python scripts/import_phone_attempts.py path/to/attempts
    .venv/bin/python scripts/replay_attempts.py data/live_sessions/phone-*.npz
"""
import argparse
import gzip
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "data" / "live_sessions"
FORMAT = "aslcv-phone-attempt/1"
# xtool registers the app under a prefixed ID on a free account (`xtool ds
# identifiers list` shows it); this is the one on the account that built it.
BUNDLE_ID = os.environ.get("ASLCV_BUNDLE_ID", "XTL-2Z38689LTQ.dev.aslcv.aslcvTutor")


def decode(raw: bytes, n: int, k: int) -> dict:
    """The phone's frame layout: timestamps f64[n], keypoints f32[n,k,2],
    scores f32[n,k], states i8[n], little-endian, back to back."""
    sizes = [("timestamps", "<f8", (n,)), ("keypoints", "<f4", (n, k, 2)),
             ("scores", "<f4", (n, k)), ("states", "i1", (n,))]
    out, offset = {}, 0
    for name, dtype, shape in sizes:
        count = int(np.prod(shape))
        out[name] = np.frombuffer(raw, dtype, count, offset).reshape(shape).copy()
        offset += count * np.dtype(dtype).itemsize
    if offset != len(raw):
        raise ValueError(f"expected {offset} bytes for {n} frames x {k} points, got {len(raw)}")
    return out


def convert(sidecar: Path, out_dir: Path) -> "Path | None":
    """One phone attempt -> phone-<stamp>.npz + .json; None if already imported."""
    meta = json.loads(sidecar.read_text())
    if meta.get("format") != FORMAT:
        raise ValueError(f"{sidecar.name}: unknown format {meta.get('format')!r}")
    npz_path = out_dir / f"phone-{sidecar.stem}.npz"
    if npz_path.exists():
        return None
    arrays = decode(gzip.decompress(sidecar.with_suffix(".bin.gz").read_bytes()),
                    meta["n_frames"], meta["n_keypoints"])
    # replay_attempts.py titles an attempt by its sentence; a single sign has none.
    meta = dict(meta, source="phone", sentence=meta.get("sentence") or meta.get("target"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(npz_path) + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, npz_path)
    npz_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return npz_path


def pull(bundle_id: str, dest: Path) -> Path:
    try:
        subprocess.run(["afcclient", "--documents", bundle_id, "--", "get", "-rf", "Documents/attempts", str(dest)],
                       check=True)
    except subprocess.CalledProcessError:
        raise SystemExit("could not pull Documents/attempts: is the phone plugged in and unlocked, and has it "
                         "logged an attempt yet (Diagnostics -> Log attempts)? Bundle ID: " + bundle_id)
    return dest / "attempts"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", type=Path, help="a copied attempts/ folder")
    ap.add_argument("--pull", action="store_true", help="pull the folder from the phone over USB")
    ap.add_argument("--bundle-id", default=BUNDLE_ID)
    ap.add_argument("--out", type=Path, default=LOG_DIR)
    args = ap.parse_args()
    if args.pull == (args.folder is not None):
        ap.error("give a folder or --pull, not both")

    with tempfile.TemporaryDirectory() as tmp:
        folder = pull(args.bundle_id, Path(tmp)) if args.pull else args.folder
        sidecars = sorted(folder.glob("*.json"))
        if not sidecars:
            raise SystemExit(f"no logged attempts in {folder}")
        new = 0
        for sidecar in sidecars:
            path = convert(sidecar, args.out)
            if path is not None:
                new += 1
                meta = json.loads(path.with_suffix(".json").read_text())
                print(f"  {path.name}: {meta['mode']} {meta['sentence']!r} {meta['event']}, "
                      f"{meta['n_frames']} frames @ {meta['stream_fps'] or 0:.1f} fps")
        print(f"imported {new} new of {len(sidecars)} attempt(s) into {args.out}")


if __name__ == "__main__":
    main()
