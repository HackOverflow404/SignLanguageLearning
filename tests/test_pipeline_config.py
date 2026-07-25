"""pipeline_config.py: the shared FeaturePipeline construction eval_slice.py,
eval_minimal_pairs.py, and live_demo.py all route through.

Runs under pytest OR as a plain script (`python tests/test_pipeline_config.py`).

The property under test is the one this module exists to guarantee: given the
SAME resolved args, every script builds an IDENTICALLY CONFIGURED pipeline,
regardless of which skeleton/extractor it's for -- so a default changing in one
place can't silently diverge from another script that forgot to pass a flag.
"""
import argparse

from aslcv.extractor.coco_wholebody import COCO_WHOLEBODY
from aslcv.extractor.mediapipe import MEDIAPIPE_HOLISTIC
from aslcv.pipeline_config import add_pipeline_args, build_pipeline, describe_pipeline_config

SKELETONS = {"coco_wholebody": COCO_WHOLEBODY, "mediapipe_holistic": MEDIAPIPE_HOLISTIC}


def _parse(argv):
    ap = argparse.ArgumentParser()
    add_pipeline_args(ap)
    return ap.parse_args(argv)


def test_defaults_match_featurepipeline_defaults():
    """No flags passed -> the built pipeline's config matches FeaturePipeline's
    OWN constructor defaults, so add_pipeline_args isn't silently overriding them."""
    args = _parse([])
    for name, sk in SKELETONS.items():
        pipe = build_pipeline(args, sk, quiet=True)
        assert pipe.face is False, f"{name}: face should default False"
        assert pipe.legs_feet is False, f"{name}: legs_feet should default False"
        assert pipe.confidence == "graded", f"{name}: confidence should default graded"
        assert pipe.velocity is True, f"{name}: velocity should default True"
        assert pipe.normalizer.local_hand is True, f"{name}: local_hand should default True"
        assert pipe.depth_proxies is False, f"{name}: depth_proxies should default False"
        assert pipe.trim_to_motion is False, f"{name}: trim_to_motion should default False"


def test_every_toggle_actually_toggles():
    args = _parse(["--face", "--legs-feet", "--confidence", "binary",
                    "--binary-threshold", "0.7", "--no-velocity", "--no-local-hand",
                    "--depth-proxies", "--trim-to-motion", "--motion-threshold", "0.05",
                    "--motion-pad-frames", "7"])
    for name, sk in SKELETONS.items():
        pipe = build_pipeline(args, sk, quiet=True)
        assert pipe.face is True, f"{name}: --face did not take effect"
        assert pipe.legs_feet is True, f"{name}: --legs-feet did not take effect"
        assert pipe.confidence == "binary", f"{name}: --confidence binary did not take effect"
        assert pipe.binary_threshold == 0.7, f"{name}: --binary-threshold did not take effect"
        assert pipe.velocity is False, f"{name}: --no-velocity did not take effect"
        assert pipe.normalizer.local_hand is False, f"{name}: --no-local-hand did not take effect"
        assert pipe.depth_proxies is True, f"{name}: --depth-proxies did not take effect"
        assert pipe.trim_to_motion is True, f"{name}: --trim-to-motion did not take effect"
        assert pipe.motion_threshold == 0.05, f"{name}: --motion-threshold did not take effect"
        assert pipe.motion_pad_frames == 7, f"{name}: --motion-pad-frames did not take effect"


def test_same_args_build_identical_config_across_skeletons():
    """The core guarantee: two different skeletons (standing in for two different
    extractors/scripts), same args -> same pipeline CONFIG. Only K (keypoint
    count) may differ; every toggle must be identical."""
    for argv in ([], ["--face"], ["--confidence", "binary"], ["--legs-feet", "--no-velocity"],
                 ["--depth-proxies"], ["--trim-to-motion", "--motion-threshold", "0.1"]):
        args = _parse(argv)
        pipes = {name: build_pipeline(args, sk, quiet=True) for name, sk in SKELETONS.items()}
        configs = {
            name: (p.face, p.legs_feet, p.confidence, p.binary_threshold, p.velocity,
                   p.normalizer.local_hand, p.depth_proxies, p.trim_to_motion,
                   p.motion_threshold, p.motion_pad_frames)
            for name, p in pipes.items()
        }
        assert len(set(configs.values())) == 1, (
            f"argv={argv}: pipeline config diverged across skeletons: {configs}")


def test_confidence_rejects_invalid_choice_at_parse_time():
    ap = argparse.ArgumentParser()
    add_pipeline_args(ap)
    try:
        ap.parse_args(["--confidence", "not_a_mode"])
        assert False, "argparse should reject an invalid --confidence choice"
    except SystemExit:
        pass  # argparse exits on a bad choice; that's the expected behavior


def test_describe_pipeline_config_reports_every_toggle():
    """The printed config line is what makes a reported number reproducible from
    its own log -- every toggle's actual value must appear in it."""
    args = _parse(["--face", "--confidence", "binary", "--binary-threshold", "0.9",
                    "--depth-proxies", "--trim-to-motion", "--motion-threshold", "0.03"])
    desc = describe_pipeline_config(args, extractor_name="dwpose", skeleton=COCO_WHOLEBODY)
    for token in ("extractor=dwpose", "face=True", "legs_feet=False",
                   "confidence=binary", "threshold=0.9", "velocity=True",
                   "local_hand=True", "depth_proxies=True",
                   "trim_to_motion=True", "threshold=0.03"):
        assert token in desc, f"config description missing {token!r}: {desc}"


def test_build_pipeline_quiet_suppresses_print(capsys):
    args = _parse([])
    build_pipeline(args, COCO_WHOLEBODY, quiet=True)
    out = capsys.readouterr().out
    assert out == "", f"quiet=True should print nothing, got: {out!r}"
    build_pipeline(args, COCO_WHOLEBODY, quiet=False)
    out = capsys.readouterr().out
    assert "pipeline config:" in out


class _FakeCapsys:
    """Minimal capsys.readouterr() stand-in for plain-script mode (no pytest):
    resets the buffer on each read, matching pytest's real semantics."""

    def __init__(self, buf):
        self._buf = buf

    def readouterr(self):
        text = self._buf.getvalue()
        self._buf.seek(0)
        self._buf.truncate(0)
        return type("R", (), {"out": text})()


if __name__ == "__main__":
    import io
    import contextlib

    passed = failed = 0
    for _name, fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(fn):
            try:
                if _name == "test_build_pipeline_quiet_suppresses_print":
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        fn(_FakeCapsys(buf))
                else:
                    fn()
                print(f"  PASS {_name}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL {_name}: {e!r}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
