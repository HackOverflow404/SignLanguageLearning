from .rtmlib_base import RTMLibWholebodyExtractor


class DWPoseExtractor(RTMLibWholebodyExtractor):
    """DWPose: YOLOX + RTMPose-l distilled on ucoco (the 2023 classic), producing
    a 133-keypoint COCO-WholeBody `Pose`.

    "DWPose" names the two-stage distillation *training recipe*, not an
    architecture -- the network is RTMPose-large with a SimCC head, so it loads
    through rtmlib's `RTMPose` class. See `RTMLibWholebodyExtractor` for the
    detector, running modes, and the rest of the machinery shared with the other
    rtmlib whole-body backends.
    """

    POSE_CLASS = "RTMPose"
    POSE_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip"
    POSE_INPUT_SIZE = (288, 384)  # (W, H) for a 384x288 (H×W) model
