# Phase 7 step 4 -- forced-alignment validation (mediapipe, seed=1)

100 trials run, 0 skipped (insufficient sign pool or a failed alignment), 2-4 signs/sentence, band=None.

## Boundary error (predicted vs. true segment length, per gloss)

- mean relative error: 13.3%
- median relative error: 4.3%
- mean absolute error: 4.9 frames
- worst relative error: 131.4% (21/305 segments exceed 50% relative error)

## Grading agreement (aligned segment vs. isolated grade of the same clip)

Excludes thin/insufficient-support target parameters (MIN_SUPPORT gate) --
same convention every other report in this project uses.

| parameter | correct-flag agreement | predicted-label agreement | n |
|---|---|---|---|
| handshape | 90.7% | 89.0% | 236 |
| major_location | 95.4% | 95.4% | 302 |
| minor_location | 91.7% | 91.7% | 240 |
| movement | 95.3% | 93.9% | 297 |
| repeated_movement | 96.1% | 96.1% | 305 |

## Honest caveat

Concatenated real clips have a hard cut and no coarticulation -- this measures
an EASIER problem than genuine fluent continuous signing (see CLAUDE.md /
project_workflow.md's Phase 7 section). A good number here is necessary, not
sufficient, before building live capture (step 5) and the sentence-mode UI
(step 6).