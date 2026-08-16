# align_and_grade vs. CTC-CSLR -- research comparison

100 trials run, 0 skipped, 2-4 signs/sentence, seed=1, extractor=mediapipe.

**This is a research comparison, not a product decision re-litigation.**
Both systems trained/built entirely on this project's own 60-sign curriculum;
"CTC" here means an architecture matching the industry-standard approach, NOT
a claim of matching published SOTA numbers (e.g. DeepMind's SL2T trained on
100k+ hours across 50+ languages -- this trains on ~1,270 synthetic sentences
built from this curriculum's own train-split clips).

## 1. Open-set recognition (CTC's actual normal use case)

CTC free-decode word error rate: 34.5% mean, 33.3% median.

align_and_grade has NO analogous number -- it is never given an unknown
sequence to recognize; the target is always known in advance (this project's
core framing, see project_workflow.md's Phase 7 section). This asymmetry is
the actual finding, not a gap in the benchmark.

## 2. Forced alignment given the TRUE sequence (fair, apples-to-apples)

| system | mean boundary error | median boundary error |
|---|---|---|
| align_and_grade (dtw_align) | 12.8% | 4.0% |
| CTC forced-align | 96.6% | 97.0% |

**Why CTC forced-align is so much worse (verified, not assumed):** manual
inspection of individual trials (see the diagnostic trace this report's
writeup is based on) shows the trained model has learned extremely "PEAKY"
posteriors -- a well-documented, real property of CTC training, not a bug in
this comparison's `forced_align` implementation (independently correctness-
checked against `nn.CTCLoss`'s own forward-algorithm marginal in
`tests/test_ctc_cslr.py`). On one representative 77-frame trial ('where my'),
the model predicted BLANK on 74/77 frames with a huge margin (mean blank
log-prob -0.22 vs. mean best-real-label log-prob -11.8), spiking on the real
label for only 1-2 frames each. Greedy decoding still gets the SEQUENCE right
from a spike that brief (hence a reasonable WER above) -- but Viterbi forced
alignment correctly finds exactly where that brief spike is, producing a
1-2-frame segment for a sign whose true length was 35-42 frames. This is CTC's
loss function rewarding ANY confident spike per label, with no pressure
toward temporally well-calibrated boundaries -- entropy regularization or a
dedicated alignment objective are known mitigations in the literature, neither
attempted here since the goal was a fair VANILLA-CTC comparison on this
project's own data, not a maximally-optimized CTC pipeline.

## 3. Grading agreement on the resulting segments (aligned vs. isolated grade)

| parameter | align_and_grade | CTC forced-align | n (align / ctc) |
|---|---|---|---|
| handshape | 94.1% | 33.5% | 236 / 236 |
| major_location | 92.4% | 48.0% | 302 / 302 |
| minor_location | 94.6% | 56.2% | 240 / 240 |
| movement | 93.6% | 58.9% | 297 / 297 |
| repeated_movement | 93.8% | 51.1% | 305 / 305 |

## Practical tradeoff (not measured above, stated plainly)

- **Training data**: align_and_grade needed ZERO sequence-labeled sentences --
  it's training-free DTW over an EmbeddingGrader trained only on ISOLATED
  single-sign clips (Phase 4). CTC needed synthetic multi-sign sentences
  specifically manufactured for this comparison, since this project's dataset
  has no native continuous-signing footage at all.
- **Failure mode on a malformed attempt**: CTC, an N-way classifier, will
  emit SOME decoded sequence regardless of input quality -- confidently wrong
  on a bad attempt. align_and_grade always compares to the ONE known target and
  reports a real distance, never a false best-guess identity. This is
  CLAUDE.md's actual stated reason for excluding CTC/classification from the
  product, not just measured indirectly here.