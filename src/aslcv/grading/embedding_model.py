"""Phase 4 learned grader: multi-stream pose encoder + phonological heads.

Pure torch here -- no I/O, no cache/manifest access (that's embedding_dataset.py). This
module owns exactly two things: the network architecture and the metric-learning loss
it trains with. Both exist to satisfy the same non-negotiable (CLAUDE.md): grade by
DISTANCE TO A KNOWN TARGET, never N-way classification, with phonological heads that
can genuinely disagree rather than co-firing because they share a bottleneck.

Multi-stream, not one shared trunk
----------------------------------
`features.py` concatenates ``[global | left_hand | right_hand]`` into one per-frame
vector, but by ``normalizer/shoulder.py``'s own design the GLOBAL block carries
location + movement (shoulder-anchored positions) and the HAND blocks carry handshape
(each hand re-centred on its own wrist, position-invariant). Feeding the concatenated
vector through one shared encoder would let every downstream head read a
representation where those signals are already mixed -- head independence would then
depend on hoping the network kept them separable, not on anything structural. So:

  - a `global_encoder` (BiGRU) runs ONLY over the global block's columns
  - a `hand_encoder` (BiGRU, WEIGHT-SHARED) runs once over left_hand's columns and
    once over right_hand's, and the two outputs combine via an elementwise max --
    ASL has no fixed dominant side (a left-handed signer executes a one-handed sign
    with their left hand, mirrored relative to a right-handed signer executing the
    same sign; 40 of the 60 curriculum signs are one-handed), so a POSITIONAL
    concat of (left, right) would partly teach the handshape head "which physical
    hand had content" -- signer handedness, not the sign. Max-combining the two
    hand-stream embeddings from a SHARED encoder is order-invariant by construction.
  - the phonological heads then read ONLY their relevant stream's output:
    `handshape_head` <- hand stream only; `major_location_head` / `minor_location_head`
    <- global stream only; `movement_head` / `repeated_head` <- global stream + an
    explicit tempo feature (see embedding_dataset.py's autocorrelation-of-
    hand_motion_energy() computation). No head has a gradient path into a stream it
    isn't supposed to see -- verified directly in tests/test_embedding_model.py by
    backpropagating one head's loss and asserting zero gradient reaches the other
    stream's parameters, not just observed as a correlation.

The primary embedding (`embed`, L2-normalized) concatenates BOTH streams and is
trained with a metric-learning loss (batch-hard triplet, below) -- overall fidelity is
meant to reflect all parameters at once, so it alone is allowed to mix the streams.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """(B, max_len) bool -- True where a timestep is real, False where it's padding."""
    ar = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return ar < lengths.unsqueeze(1)


class StreamEncoder(nn.Module):
    """BiGRU over one input stream + masked mean+max temporal pooling to a fixed vector.

    Deliberately small (a single BiGRU layer, hidden=128 by default) -- the reference
    bank is ~15 train clips per sign across 60 signs, so a larger temporal model would
    have far more capacity than data to learn signs rather than signers.

    Mean+max pooling (rather than just the final hidden state) gives the pooled vector
    both "typical" and "most extreme" signal per feature -- useful here because a
    phonological parameter can show up as a sustained posture (mean-friendly) or a
    brief peak (e.g. a handshape held only at contact; max-friendly).
    """

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.out_dim = hidden * 2 * 2  # bidirectional * (mean + max)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x: (B, T, in_dim) zero-padded. lengths: (B,) int64 true frame counts."""
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])  # (B,T,2H)

        mask = _length_mask(lengths, x.shape[1]).unsqueeze(-1)  # (B,T,1)
        summed = (out * mask).sum(dim=1)
        mean = summed / lengths.clamp(min=1).unsqueeze(1).to(out.dtype)
        masked_for_max = out.masked_fill(~mask, float("-inf"))
        max_, _ = masked_for_max.max(dim=1)
        return torch.cat([mean, max_], dim=-1)


class PoseGraderNet(nn.Module):
    """Multi-stream encoder -> a metric-learned embedding + disjoint phonological heads.

    Parameters
    ----------
    blocks : the FeatureClip.blocks slice map for the pipeline config this model is
        trained on (must contain "global", "left_hand", "right_hand" -- the default
        `DEFAULT_BLOCK_ORDER`). Fixed at construction time: the model is tied to one
        pipeline config, same as a DTWGrader's reference bank is tied to one.
    head_classes : {parameter: n_classes} for the four categorical heads (handshape,
        major_location, minor_location, movement) -- sizes come from
        PhonologyLabelEncoder, never hardcoded here.
    """

    def __init__(self, blocks: dict, head_classes: dict, *, hidden: int = 128,
                 embed_dim: int = 128, tempo_dim: int = 2, tempo_hidden: int = 16):
        super().__init__()
        for name in ("global", "left_hand", "right_hand"):
            if name not in blocks:
                raise ValueError(f"PoseGraderNet needs a {name!r} block; got {list(blocks)}")
        self.blocks = dict(blocks)

        global_dim = blocks["global"].stop - blocks["global"].start
        left_dim = blocks["left_hand"].stop - blocks["left_hand"].start
        right_dim = blocks["right_hand"].stop - blocks["right_hand"].start
        if left_dim != right_dim:
            raise ValueError(f"left_hand/right_hand widths must match, got {left_dim} vs {right_dim}")

        self.global_encoder = StreamEncoder(global_dim, hidden)
        self.hand_encoder = StreamEncoder(left_dim, hidden)  # shared weights for both hands

        self.tempo_mlp = nn.Sequential(nn.Linear(tempo_dim, tempo_hidden), nn.ReLU())

        self.embed_proj = nn.Linear(self.global_encoder.out_dim + self.hand_encoder.out_dim, embed_dim)

        self.handshape_head = nn.Linear(self.hand_encoder.out_dim, head_classes["handshape"])
        self.major_location_head = nn.Linear(self.global_encoder.out_dim, head_classes["major_location"])
        self.minor_location_head = nn.Linear(self.global_encoder.out_dim, head_classes["minor_location"])

        movement_in_dim = self.global_encoder.out_dim + tempo_hidden
        self.movement_head = nn.Linear(movement_in_dim, head_classes["movement"])
        self.repeated_head = nn.Linear(movement_in_dim, 1)  # binary logit

    def forward(self, features: torch.Tensor, lengths: torch.Tensor, tempo: torch.Tensor) -> dict:
        """features: (B, T, F) zero-padded, F matching this model's `blocks`.
        lengths: (B,) true frame counts (shared across streams -- one clip, one T).
        tempo: (B, tempo_dim) autocorrelation-derived periodicity features.
        """
        g = self.global_encoder(features[..., self.blocks["global"]], lengths)
        lh = self.hand_encoder(features[..., self.blocks["left_hand"]], lengths)
        rh = self.hand_encoder(features[..., self.blocks["right_hand"]], lengths)
        hand = torch.maximum(lh, rh)  # order-invariant: no fixed "dominant hand" assumption

        embed = F.normalize(self.embed_proj(torch.cat([g, hand], dim=-1)), dim=-1)

        tempo_feat = self.tempo_mlp(tempo)
        move_in = torch.cat([g, tempo_feat], dim=-1)

        return {
            "embed": embed,
            "handshape": self.handshape_head(hand),
            "major_location": self.major_location_head(g),
            "minor_location": self.minor_location_head(g),
            "movement": self.movement_head(move_in),
            "repeated": self.repeated_head(move_in).squeeze(-1),
        }


def pairwise_euclidean(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """(B, B) Euclidean distance between every pair of rows in x.

    Floors squared distance at `eps`, not 0, before the sqrt. sqrt's local gradient is
    1/(2*sqrt(d2)) -- at d2 == 0 (always true on the diagonal, i.e. self-distance) that
    is literally infinite, and even though the diagonal is never SELECTED as a hardest
    pos/neg pair downstream, autograd still computes that local derivative as part of
    backpropagating through this op; 0 (unselected upstream grad) * inf (local grad) =
    NaN, which then corrupts every parameter in the graph from the first batch. A tiny
    epsilon keeps the gradient finite everywhere without changing any real distance by
    more than ~1e-6.
    """
    sq = (x * x).sum(dim=1)
    d2 = sq.unsqueeze(0) + sq.unsqueeze(1) - 2.0 * (x @ x.t())
    return d2.clamp(min=eps).sqrt()


def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """FaceNet-style batch-hard triplet loss over one PK-sampled batch.

    For each anchor, the hardest positive (same label, MAX distance, excluding self)
    and hardest negative (different label, MIN distance) are chosen from THIS BATCH
    only -- there is no learned per-class weight vector anywhere in this path, so this
    stays genuine metric learning between actual clip embeddings, never a
    classification head with a margin bolted on (which is why ArcFace/CosFace-style
    losses were ruled out for this project -- they DO use a per-class weight matrix).

    Requires every label in the batch to have >=2 members (guaranteed by the PK
    sampler in embedding_dataset.py: P signs x K>=2 clips each) so every anchor has at
    least one real positive to compare against.
    """
    dist = pairwise_euclidean(embeddings)
    labels = labels.view(-1, 1)
    same = labels == labels.t()
    n = embeddings.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=embeddings.device)

    pos_mask = same & ~self_mask
    if not bool(pos_mask.any(dim=1).all()):
        raise ValueError("batch_hard_triplet_loss: every label needs >=2 members in the batch")
    hardest_pos = (dist * pos_mask.float()).max(dim=1).values

    neg_mask = ~same
    big = dist.max().detach() + 1.0
    hardest_neg = (dist + (~neg_mask).float() * big).min(dim=1).values

    return F.relu(hardest_pos - hardest_neg + margin).mean()
