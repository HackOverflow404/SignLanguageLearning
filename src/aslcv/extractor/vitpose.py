from .rtmlib_base import RTMLibWholebodyExtractor


class ViTPoseExtractor(RTMLibWholebodyExtractor):
    """ViTPose-l whole-body: YOLOX + ViTPose-large, producing a 133-keypoint
    COCO-WholeBody `Pose` -- same topology as DWPose/RTMW, so nothing downstream
    changes.

    The checkpoint is the easy_ViTPose (JunkyByte) whole-body export -- a direct
    ~1.2 GB `.onnx` pulled from HuggingFace into models/ on first use (no zip).
    Smaller variants exist (swap `-l-` for `-b-`/`-s-` in POSE_URL) if the large
    model is too heavy.

    NOTE: confirm the 133-keypoint index order matches `COCO_WHOLEBODY` the first
    time you draw a real frame -- easy_ViTPose trains on COCO-WholeBody so it
    should align, but this is a benchmarking candidate to validate in Phase 3, not
    a wired-in default.
    """

    POSE_CLASS = "ViTPose"
    POSE_URL = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-l-wholebody.onnx"
    POSE_INPUT_SIZE = (192, 256)  # (W, H); easy_ViTPose default
