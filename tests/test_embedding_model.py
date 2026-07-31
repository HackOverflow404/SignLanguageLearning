"""Phase 4 encoder/head architecture tests -- synthetic tensors only, no cache/model
checkpoint needed. Runs under pytest OR as a plain script.

The most important test here is NOT an accuracy number: it's
`test_heads_are_structurally_disjoint`, which proves by backprop that a phonological
head cannot see the stream it isn't supposed to. If the heads just co-fire because
they secretly share a bottleneck, per-parameter diagnosis is a lie regardless of how
good the reported accuracy looks -- see embedding_model.py's module docstring.

Runs on CUDA when available, NOT for speed: tests/test_dwpose_running_mode.py (which
runs earlier in the suite) imports rtmlib/onnxruntime-gpu, which installs an NVBLAS
hook that breaks torch's native CPU RNN kernel process-wide -- a CPU nn.GRU forward
pass later in the same pytest process segfaults inside torch/nn/modules/rnn.py. CUDA
execution doesn't touch that broken codepath. See test_embedding_grader.py's `grader`
fixture docstring for the full root-cause writeup (same fix, same reason).
"""
import torch

from aslcv.grading.embedding_model import (
    PoseGraderNet,
    batch_hard_triplet_loss,
    pairwise_euclidean,
)

BLOCKS = {"global": slice(0, 20), "left_hand": slice(20, 32), "right_hand": slice(32, 44)}
HEAD_CLASSES = {"handshape": 6, "major_location": 5, "minor_location": 9, "movement": 4}
F_DIM = 44
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model(**kw):
    torch.manual_seed(0)
    return PoseGraderNet(BLOCKS, HEAD_CLASSES, hidden=8, embed_dim=10, tempo_hidden=4, **kw).to(_DEVICE)


def _batch(batch_size=6, max_len=15, seed=1):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(4, max_len + 1, (batch_size,), generator=g).to(_DEVICE)
    features = torch.randn(batch_size, max_len, F_DIM, generator=g).to(_DEVICE)
    # zero out padding so it doesn't look like data (matches the real pipeline's convention)
    for i, L in enumerate(lengths.cpu()):
        features[i, L:] = 0.0
    tempo = torch.randn(batch_size, 2, generator=g).to(_DEVICE)
    return features, lengths, tempo


# -- shapes / basic sanity ---------------------------------------------------


def test_forward_shapes():
    model = _model()
    features, lengths, tempo = _batch()
    out = model(features, lengths, tempo)
    B = features.shape[0]
    assert out["embed"].shape == (B, 10)
    assert out["handshape"].shape == (B, HEAD_CLASSES["handshape"])
    assert out["major_location"].shape == (B, HEAD_CLASSES["major_location"])
    assert out["minor_location"].shape == (B, HEAD_CLASSES["minor_location"])
    assert out["movement"].shape == (B, HEAD_CLASSES["movement"])
    assert out["repeated"].shape == (B,)


def test_embedding_is_l2_normalized():
    model = _model()
    features, lengths, tempo = _batch()
    embed = model(features, lengths, tempo)["embed"]
    norms = embed.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_mismatched_hand_widths_rejected():
    bad_blocks = {"global": slice(0, 20), "left_hand": slice(20, 32), "right_hand": slice(32, 43)}
    try:
        PoseGraderNet(bad_blocks, HEAD_CLASSES)
        assert False, "expected ValueError for mismatched left/right hand widths"
    except ValueError:
        pass


# -- the property that matters most ------------------------------------------


def _grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.abs().sum().item()
    return total


def test_heads_are_structurally_disjoint():
    """Backprop each head's loss ALONE and check which streams actually received
    gradient. A head must have EXACTLY zero gradient in the stream(s) it isn't
    supposed to see -- not "small," zero -- because there is no data path there at
    all (see PoseGraderNet.forward: each head's input is a disjoint slice of the
    stream outputs, never the full concatenation)."""
    model = _model()
    features, lengths, tempo = _batch()

    global_params = list(model.global_encoder.parameters())
    hand_params = list(model.hand_encoder.parameters())
    tempo_params = list(model.tempo_mlp.parameters())

    def grads_for(loss_key):
        model.zero_grad()
        out = model(features, lengths, tempo)
        loss = out[loss_key].sum()
        loss.backward()
        return _grad_norm(global_params), _grad_norm(hand_params), _grad_norm(tempo_params)

    # handshape: hand stream only -- zero gradient must reach global_encoder/tempo_mlp
    g, h, t = grads_for("handshape")
    assert g == 0.0, f"handshape head leaked gradient into global_encoder (norm={g})"
    assert t == 0.0, f"handshape head leaked gradient into tempo_mlp (norm={t})"
    assert h > 0.0, "handshape head should receive real gradient from hand_encoder"

    # major_location / minor_location: global stream only -- zero gradient into hand_encoder/tempo_mlp
    for key in ("major_location", "minor_location"):
        g, h, t = grads_for(key)
        assert h == 0.0, f"{key} head leaked gradient into hand_encoder (norm={h})"
        assert t == 0.0, f"{key} head leaked gradient into tempo_mlp (norm={t})"
        assert g > 0.0, f"{key} head should receive real gradient from global_encoder"

    # movement / repeated: global stream + tempo -- zero gradient into hand_encoder
    for key in ("movement", "repeated"):
        g, h, t = grads_for(key)
        assert h == 0.0, f"{key} head leaked gradient into hand_encoder (norm={h})"
        assert g > 0.0, f"{key} head should receive real gradient from global_encoder"
        assert t > 0.0, f"{key} head should receive real gradient from tempo_mlp"


def test_embedding_mixes_both_streams_deliberately():
    """The OPPOSITE property, checked as a control: the primary embed IS allowed (and
    expected) to mix both streams, since overall fidelity should reflect every
    parameter at once. If this ever went to zero for one stream, grade_against's
    distance would silently stop reflecting that stream's correctness."""
    model = _model()
    features, lengths, tempo = _batch()
    global_params = list(model.global_encoder.parameters())
    hand_params = list(model.hand_encoder.parameters())

    model.zero_grad()
    out = model(features, lengths, tempo)
    out["embed"].sum().backward()
    assert _grad_norm(global_params) > 0.0
    assert _grad_norm(hand_params) > 0.0


# -- batch-hard triplet loss --------------------------------------------------


def test_pairwise_euclidean_zero_on_self():
    x = torch.randn(5, 4, device=_DEVICE)
    d = pairwise_euclidean(x)
    assert torch.allclose(d.diag(), torch.zeros(5, device=_DEVICE), atol=1e-5)
    assert torch.allclose(d, d.t(), atol=1e-5)


def test_triplet_loss_requires_two_per_label():
    embeds = torch.randn(4, 8, device=_DEVICE)
    labels = torch.tensor([0, 1, 2, 3], device=_DEVICE)  # every label unique -- no positive exists
    try:
        batch_hard_triplet_loss(embeds, labels)
        assert False, "expected ValueError when a label has < 2 members"
    except ValueError:
        pass


def test_triplet_loss_lower_for_well_separated_clusters():
    torch.manual_seed(0)
    labels = torch.tensor([0, 0, 1, 1, 2, 2], device=_DEVICE)
    # tight clusters, far apart -- should incur ~0 loss at a modest margin
    centers = (torch.eye(3) * 10.0).to(_DEVICE)
    tight = centers[labels] + 0.01 * torch.randn(6, 3, device=_DEVICE)
    tight = torch.nn.functional.normalize(tight, dim=1)
    loss_tight = batch_hard_triplet_loss(tight, labels, margin=0.2)

    # same labels, but embeddings placed randomly (positives far, negatives close)
    loose = torch.nn.functional.normalize(torch.randn(6, 3, device=_DEVICE), dim=1)
    loss_loose = batch_hard_triplet_loss(loose, labels, margin=0.2)

    assert loss_tight < loss_loose


if __name__ == "__main__":
    passed = failed = 0
    for _name, fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {_name}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL {_name}: {e!r}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
