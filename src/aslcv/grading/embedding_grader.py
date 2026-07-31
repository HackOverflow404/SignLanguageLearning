"""EmbeddingGrader -- the Phase 4 learned grader, wrapping a trained PoseGraderNet.

Mirrors DTWGrader's interface (`grade` / `grade_against`), but "distance" is a
metric-learned embedding distance rather than a DTW alignment cost, and
`grade_against` is extended to return per-parameter phonological VERDICTS -- the
actual diagnosis this phase exists to produce ("handshape right, location wrong"),
not just a ranked list of signs. `grade` (rank every sign) stays the secondary path,
same as DTWGrader.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .. import dataset as dataset_mod
from ..extractor.coco_wholebody import COCO_WHOLEBODY
from ..extractor.mediapipe import MEDIAPIPE_HOLISTIC
from ..features import FeaturePipeline, Standardizer, hand_motion_energy
from ..normalizer.shoulder import ShoulderNormalizer
from .embedding_dataset import EmbeddingClipDataset, _load_poses_npz, _tempo_features, collate_fn
from .embedding_model import PoseGraderNet
from .phonology_labels import CATEGORICAL_PARAMETERS, MIN_SUPPORT, PhonologyLabels


def skeleton_for(extractor: str):
    return MEDIAPIPE_HOLISTIC if extractor == "mediapipe" else COCO_WHOLEBODY


@torch.no_grad()
def embed_dataset(model: PoseGraderNet, ds: EmbeddingClipDataset, device, batch_size: int = 32) -> dict:
    """Forward-pass every clip in a dataset once (eval mode, no grad).

    Shared by the training script's per-epoch eval and EmbeddingGrader.build's
    reference-bank construction, so there is exactly one "run the model over a
    dataset" implementation rather than two copies drifting apart.
    """
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    embeds, gloss_idx = [], []
    logits = {p: [] for p in CATEGORICAL_PARAMETERS}
    labels = {p: [] for p in CATEGORICAL_PARAMETERS}
    repeated_logits, repeated_labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["features"], batch["lengths"], batch["tempo"])
        embeds.append(out["embed"].cpu())
        gloss_idx.append(batch["id_gloss"].cpu())
        for p in CATEGORICAL_PARAMETERS:
            logits[p].append(out[p].cpu())
            labels[p].append(batch[p].cpu())
        repeated_logits.append(out["repeated"].cpu())
        repeated_labels.append(batch["repeated"].cpu())
    return {
        "embeds": torch.cat(embeds),
        "gloss_idx": torch.cat(gloss_idx),
        "logits": {p: torch.cat(v) for p, v in logits.items()},
        "labels": {p: torch.cat(v) for p, v in labels.items()},
        "repeated_logits": torch.cat(repeated_logits),
        "repeated_labels": torch.cat(repeated_labels),
    }


@dataclass
class ParameterVerdict:
    """One phonological parameter's diagnosis for a single graded attempt.

    `correct` is None -- not False -- when the TARGET sign's true label value is
    backed by fewer than phonology_labels.MIN_SUPPORT distinct curriculum signs: the
    head cannot be shown to have learned a generalizable notion of that label value
    (it may just recognize the one sign that happened to carry it), so reporting a
    confident correct/incorrect there would overstate what was actually verified.
    See phonology_labels.py's module docstring for the full reasoning.
    """

    parameter: str
    predicted: str
    target: str
    correct: "bool | None"
    support: int


@dataclass
class GradeResult:
    target_sign: str
    fidelity: float  # embedding distance to the target's nearest reference clip (lower = closer)
    parameters: dict[str, ParameterVerdict]


def _pipeline_from_config(config: dict) -> FeaturePipeline:
    pa = config["pipeline_args"]
    return FeaturePipeline(
        ShoulderNormalizer(local_hand=pa["local_hand"]),
        skeleton_for(config["extractor"]),
        face=pa["face"], legs_feet=pa["legs_feet"], confidence=pa["confidence"],
        binary_threshold=pa["binary_threshold"], velocity=pa["velocity"],
        depth_proxies=pa["depth_proxies"], trim_to_motion=pa["trim_to_motion"],
        motion_threshold=pa["motion_threshold"], motion_pad_frames=pa["motion_pad_frames"],
    )


class EmbeddingGrader:
    def __init__(self, model, pipeline, standardizer, references, id_gloss_encoder, phon_labels, device, extractor):
        self.model = model
        self.extractor = extractor
        self.pipeline = pipeline
        self.standardizer = standardizer
        self.references = references  # {sign: (N_sign, D) tensor of reference embeddings}
        self.id_gloss_encoder = id_gloss_encoder
        self.phon_labels = phon_labels
        self.device = device
        self.signs = list(references.keys())

    @classmethod
    def build(cls, checkpoint_dir, *, signs=None, which: str = "best", device=None) -> "EmbeddingGrader":
        """Load a checkpoint written by scripts/train_embedding_grader.py and build
        the reference bank (TRAIN split embeddings, grouped by sign) it grades
        against -- the learned-embedding analogue of DTWGrader.build's reference bank.

        `which`: "best" (default -- the best-VAL-top1 checkpoint, see the training
        script's overfitting note) or "final" (the last epoch, for comparison only).
        """
        checkpoint_dir = Path(checkpoint_dir)
        with open(checkpoint_dir / "config.json") as f:
            config = json.load(f)

        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        pipeline = _pipeline_from_config(config)
        standardizer = Standardizer.load(checkpoint_dir / "standardizer.npz")

        phon_labels = PhonologyLabels()
        for p in CATEGORICAL_PARAMETERS:
            if phon_labels.encoders[p].classes_ != config["phonology_classes"][p]:
                raise ValueError(
                    f"phonology.csv's current {p!r} classes don't match this checkpoint's "
                    f"training-time classes -- the checkpoint was trained against a different "
                    f"curriculum.yaml/phonology.csv than what's on disk now. Re-train, or check "
                    f"out the phonology.csv this checkpoint was trained with.")

        id_gloss_encoder = dataset_mod.LabelEncoder(config["id_gloss_classes"])
        blocks = {k: slice(v[0], v[1]) for k, v in config["blocks"].items()}
        model = PoseGraderNet(blocks, config["head_classes"], hidden=config["hidden"], embed_dim=config["embed_dim"])
        model.load_state_dict(torch.load(checkpoint_dir / f"model_{which}.pt", map_location=device))
        model.to(device)

        signs = sorted(phon_labels.by_gloss) if signs is None else list(signs)
        train_ds = EmbeddingClipDataset(config["extractor"], "train", pipeline, signs,
                                         id_gloss_encoder, phon_labels, standardizer)
        out = embed_dataset(model, train_ds, device)
        references: dict[str, torch.Tensor] = {}
        for sign in signs:
            idx = id_gloss_encoder.encode(sign)
            mask = out["gloss_idx"] == idx
            references[sign] = out["embeds"][mask]

        return cls(model, pipeline, standardizer, references, id_gloss_encoder, phon_labels, device, config["extractor"])

    # -- embedding a single attempt clip --------------------------------------------

    @torch.no_grad()
    def _forward_npz(self, npz_path) -> dict:
        """Full model output (embed + every head's logits) for one cached clip."""
        clip = self.pipeline.assemble_npz(npz_path)
        feats = self.standardizer.transform(clip.features)
        poses = _load_poses_npz(npz_path)
        energy = hand_motion_energy(self.pipeline.normalizer, self.pipeline.skeleton, poses)
        tempo = _tempo_features(energy)

        features = torch.from_numpy(feats).unsqueeze(0).to(self.device)
        lengths = torch.tensor([feats.shape[0]], dtype=torch.int64).to(self.device)
        tempo_t = torch.from_numpy(tempo).unsqueeze(0).to(self.device)

        self.model.eval()
        out = self.model(features, lengths, tempo_t)
        return {k: v.squeeze(0).cpu() for k, v in out.items()}

    # -- grading ---------------------------------------------------------------------

    def _nearest_clip_distance(self, embed: torch.Tensor, sign: str) -> float:
        refs = self.references[sign]
        return float(torch.cdist(embed.unsqueeze(0), refs).min())

    def grade(self, npz_path) -> list[tuple[str, float]]:
        """Rank every sign by nearest-reference-clip embedding distance -- the
        open-set path, secondary to grade_against (CLAUDE.md: the tutor always knows
        the target it prompted, so grade_against is primary)."""
        out = self._forward_npz(npz_path)
        ranked = [(sign, self._nearest_clip_distance(out["embed"], sign)) for sign in self.signs]
        ranked.sort(key=lambda sd: sd[1])
        return ranked

    def grade_against(self, npz_path, target_sign: str) -> GradeResult:
        """Closed-set grading: distance to the KNOWN target sign, plus a per-parameter
        diagnosis from the attempt's own head predictions vs. the target's true
        phonology label (gated by phonology_labels.MIN_SUPPORT -- see ParameterVerdict)."""
        if target_sign not in self.references:
            raise KeyError(f"unknown target sign {target_sign!r}; have {self.signs}")
        out = self._forward_npz(npz_path)
        fidelity = self._nearest_clip_distance(out["embed"], target_sign)

        parameters: dict[str, ParameterVerdict] = {}
        for p in CATEGORICAL_PARAMETERS:
            predicted = self.phon_labels.encoders[p].decode(int(out[p].argmax()))
            target_label = self.phon_labels.label_for(target_sign, p)
            support = self.phon_labels.support(p, target_label)
            correct = (predicted == target_label) if support >= MIN_SUPPORT else None
            parameters[p] = ParameterVerdict(p, predicted, target_label, correct, support)

        predicted_repeated = bool(out["repeated"] > 0)
        target_repeated = self.phon_labels.repeated_bool(target_sign)
        target_repeated_raw = "1" if target_repeated else "0"
        rep_support = self.phon_labels.support("repeated_movement", target_repeated_raw)
        rep_correct = (predicted_repeated == target_repeated) if rep_support >= MIN_SUPPORT else None
        parameters["repeated_movement"] = ParameterVerdict(
            "repeated_movement", str(predicted_repeated), str(target_repeated), rep_correct, rep_support)

        return GradeResult(target_sign, fidelity, parameters)
