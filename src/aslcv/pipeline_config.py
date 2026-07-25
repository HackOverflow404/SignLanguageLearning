"""pipeline_config.py -- the ONE place a FeaturePipeline gets built from CLI flags.

WHY THIS EXISTS: eval_slice.py, eval_minimal_pairs.py, and live_demo.py each used to
construct their own `FeaturePipeline(...)` call by hand -- eval_slice.py exposed
--face/--legs-feet, eval_minimal_pairs.py exposed NEITHER (silently inheriting
FeaturePipeline's defaults), and live_demo.py hardcoded `velocity=True,
confidence="graded"` straight into the constructor with no flag at all. Today these
happen to agree. The moment ONE of them changes -- a new flag, a different default --
the others silently diverge and whatever numbers they report stop being comparable,
with nothing to warn you. This module is the single source of truth for what pipeline
config flags exist and what a FeaturePipeline looks like given them; every script
that grades or features a clip should route through here, not build its own.

    add_pipeline_args(ap)              -- adds the shared flags to an argparse parser
    build_pipeline(args, skeleton, ...) -- FeaturePipeline built from those args;
                                           prints the resolved config so any number
                                           reported downstream is reproducible
                                           straight from its own log
"""
import argparse

from .features import FeaturePipeline
from .normalizer.shoulder import ShoulderNormalizer


def add_pipeline_args(ap: argparse.ArgumentParser) -> None:
    """Register the shared feature-pipeline flags on an existing parser.

    Every flag here maps 1:1 to a FeaturePipeline / ShoulderNormalizer constructor
    argument (see build_pipeline) -- this function and that one must be edited
    together, which is the whole point of centralizing them.
    """
    ap.add_argument("--face", action="store_true",
                     help="keep FINE face-mesh points in the global block "
                          "(default: dropped). Coarse head location -- nose/eyes/"
                          "ears -- is always kept regardless, on every extractor.")
    ap.add_argument("--legs-feet", action="store_true",
                     help="keep leg/foot keypoints in the global block "
                          "(default: dropped -- most likely to be hallucinated "
                          "under seated, upper-body framing)")
    ap.add_argument("--confidence", default="graded", choices=("graded", "binary"),
                     help="confidence channel mode (default: graded). 'graded' keeps "
                          "each backend's raw per-point score; 'binary' thresholds "
                          "every score to {0,1} at --binary-threshold. MediaPipe's "
                          "hand scores are HANDEDNESS confidence (~0.5-1.0, which "
                          "hand, not keypoint quality) and its face scores a "
                          "hardcoded 1.0, while the rtmlib backends (dwpose/rtmw/"
                          "vitpose) give a graded per-point score -- 'graded' mode "
                          "is comparing different SEMANTICS across backends, not "
                          "just different accuracy. Use 'binary' to compare "
                          "backends on equal footing.")
    ap.add_argument("--binary-threshold", type=float, default=0.5,
                     help="score threshold for --confidence binary (default: 0.5)")
    ap.add_argument("--no-velocity", dest="velocity", action="store_false", default=True,
                     help="disable the (dx, dy) velocity channel (default: on)")
    ap.add_argument("--no-local-hand", dest="local_hand", action="store_false", default=True,
                     help="disable ShoulderNormalizer's per-hand local (handshape) "
                          "blocks, leaving only the shoulder-frame global block "
                          "(default: on)")
    ap.add_argument("--depth-proxies", action="store_true",
                     help="append 4 trailing scalar columns -- left/right apparent "
                          "hand size and left/right arm foreshortening, each "
                          "relative to shoulder width -- as a monocular depth cue "
                          "2D (x, y) keypoints are otherwise blind to (default: "
                          "off until measured). Derived from distances "
                          "ShoulderNormalizer already computes; needs no "
                          "re-extraction and works on all four backends.")
    ap.add_argument("--trim-to-motion", action="store_true",
                     help="drop leading/trailing frames whose hand motion energy "
                          "never clears --motion-threshold before assembly (default: "
                          "off until measured -- see scripts/measure_motion_energy.py). "
                          "ASL Citizen clips are rest -> sign -> rest; DTW normalizes "
                          "cost by path length, so rest frames align cheaply against "
                          "every candidate and dilute the gap between right and wrong "
                          "answers. Measured effect is backend-dependent: real and "
                          "large for MediaPipe (VIDEO-mode tracking -> genuinely "
                          "near-zero jitter at rest), much smaller for the rtmlib "
                          "backends (frame-independent IMAGE-mode detection has a "
                          "higher per-frame noise floor that rarely dips below "
                          "threshold even at rest).")
    ap.add_argument("--motion-threshold", type=float, default=0.02,
                     help="hand motion-energy threshold for --trim-to-motion, in "
                          "normalized local-hand-frame units/frame (default: 0.02, "
                          "calibrated against MediaPipe's signal -- see "
                          "scripts/measure_motion_energy.py)")
    ap.add_argument("--motion-pad-frames", type=int, default=3,
                     help="frames of padding kept on each side of the detected "
                          "active span for --trim-to-motion (default: 3)")


def describe_pipeline_config(args, *, extractor_name: str | None = None,
                              skeleton=None) -> str:
    """The exact one-line config string build_pipeline prints -- exposed separately
    so a caller building several pipelines per run (e.g. one per extractor in a
    loop) can log this without re-constructing anything, or a caller can fold it
    into its own startup banner instead of a separate print."""
    parts = []
    if extractor_name is not None:
        parts.append(f"extractor={extractor_name}")
    if skeleton is not None:
        parts.append(f"K={len(skeleton.names)}")
    parts.append(f"normalizer=ShoulderNormalizer(local_hand={args.local_hand})")
    parts.append(f"face={args.face}")
    parts.append(f"legs_feet={args.legs_feet}")
    conf = f"confidence={args.confidence}"
    if args.confidence == "binary":
        conf += f"(threshold={args.binary_threshold})"
    parts.append(conf)
    parts.append(f"velocity={args.velocity}")
    parts.append(f"depth_proxies={args.depth_proxies}")
    trim = f"trim_to_motion={args.trim_to_motion}"
    if args.trim_to_motion:
        trim += f"(threshold={args.motion_threshold}, pad={args.motion_pad_frames})"
    parts.append(trim)
    return "pipeline config: " + "  ".join(parts)


def build_pipeline(args, skeleton, *, extractor_name: str | None = None,
                    standardizer=None, quiet: bool = False) -> FeaturePipeline:
    """FeaturePipeline built identically from `args` (populated via
    add_pipeline_args) for any skeleton/extractor. Prints the resolved config
    (every toggle) unless `quiet` -- so any number reported downstream is
    reproducible straight from its own log, per-extractor if called once per
    extractor in a loop.

    `standardizer` defaults to None: every current caller (DTWGrader.build,
    eval_minimal_pairs.py's LOOBank.build) fits its OWN standardizer on the
    reference bank's train clips and applies it separately, so the pipeline
    passed in stays raw. Exposed here for a future caller that wants a
    pre-fit standardizer baked into assembly itself.
    """
    if not quiet:
        print(describe_pipeline_config(args, extractor_name=extractor_name, skeleton=skeleton))
    return FeaturePipeline(
        ShoulderNormalizer(local_hand=args.local_hand),
        skeleton,
        face=args.face,
        legs_feet=args.legs_feet,
        confidence=args.confidence,
        binary_threshold=args.binary_threshold,
        velocity=args.velocity,
        depth_proxies=args.depth_proxies,
        trim_to_motion=args.trim_to_motion,
        motion_threshold=args.motion_threshold,
        motion_pad_frames=args.motion_pad_frames,
        standardizer=standardizer,
    )
