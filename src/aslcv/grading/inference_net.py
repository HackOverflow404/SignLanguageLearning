"""Export-friendly single-attempt view of a trained PoseGraderNet (Phase 8).

Live grading always runs on ONE unpadded attempt (batch size 1, every frame
real), so PoseGraderNet's pack_padded_sequence / length-mask machinery -- which
exists for padded training batches and does not export cleanly to mobile
runtimes -- reduces to a plain GRU over the whole sequence plus mean/max
pooling over time. `InferenceGraderNet` wraps the SAME trained modules (shared
parameters, no copies) with that reduced forward, so an exported graph is
numerically the model `EmbeddingGrader` already serves, not a re-implementation.

Outputs are a fixed-order tuple (`OUTPUT_NAMES`) rather than a dict, since an
exported graph's outputs are positional.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding_model import PoseGraderNet, StreamEncoder

OUTPUT_NAMES = ("embed", "handshape", "major_location", "minor_location", "movement", "repeated")


def _encode_unpadded(enc: StreamEncoder, x: torch.Tensor) -> torch.Tensor:
    """StreamEncoder.forward for an unpadded (1, T, in_dim) sequence: with every
    timestep real, packing is a no-op and the length mask is all-True."""
    out, _ = enc.gru(x)
    return torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=-1)


class InferenceGraderNet(nn.Module):
    def __init__(self, model: PoseGraderNet):
        super().__init__()
        self.model = model
        self.g_lo, self.g_hi = model.blocks["global"].start, model.blocks["global"].stop
        self.l_lo, self.l_hi = model.blocks["left_hand"].start, model.blocks["left_hand"].stop
        self.r_lo, self.r_hi = model.blocks["right_hand"].start, model.blocks["right_hand"].stop

    def forward(self, features: torch.Tensor, tempo: torch.Tensor):
        """features: (1, T, F) standardized feature frames. tempo: (1, 2)."""
        m = self.model
        g = _encode_unpadded(m.global_encoder, features[:, :, self.g_lo:self.g_hi])
        lh = _encode_unpadded(m.hand_encoder, features[:, :, self.l_lo:self.l_hi])
        rh = _encode_unpadded(m.hand_encoder, features[:, :, self.r_lo:self.r_hi])
        hand = torch.maximum(lh, rh)

        embed = F.normalize(m.embed_proj(torch.cat([g, hand], dim=-1)), dim=-1)
        move_in = torch.cat([g, m.tempo_mlp(tempo)], dim=-1)
        return (embed, m.handshape_head(hand), m.major_location_head(g), m.minor_location_head(g),
                m.movement_head(move_in), m.repeated_head(move_in).squeeze(-1))
