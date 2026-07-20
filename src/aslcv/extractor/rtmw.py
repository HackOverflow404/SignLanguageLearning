from .rtmlib_base import RTMLibWholebodyExtractor


class RTMWExtractor(RTMLibWholebodyExtractor):
    """RTMW-x: YOLOX + RTMW-x (cocktail14), producing a 133-keypoint COCO-WholeBody
    `Pose` -- the exact same topology as DWPose, so nothing downstream changes.

    RTMW is a newer architecture (FPN + Hierarchical Encoding Module so hands and
    face aren't drowned out by the torso), but it is still an RTMPose-family SimCC
    model, so it loads through rtmlib's `RTMPose` class. Uses the 384x288
    checkpoint (the accuracy contender; a 256x192 variant exists in models/ for a
    faster/coarser option). See `RTMLibWholebodyExtractor` for the shared machinery.
    """

    POSE_CLASS = "RTMPose"
    POSE_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
    POSE_INPUT_SIZE = (288, 384)  # (W, H) for a 384x288 (H×W) model
