# Phase 4 learned grader -- evaluation report

Checkpoint: `/home/hackoverflow/Documents/Projects/SignLanguageLearning/models/embedding_grader` (`best`), extractor `mediapipe`, 60-sign val split (229 clips).

## grade() / grade_against(): learned vs. DTW baseline

| | top-1 | top-5 |
|---|---|---|
| DTW baseline (re-measured, 60-sign val) | 31.9% | 64.2% |
| Learned embedding grader | 81.2% | 96.5% |

**Beats the DTW baseline** on this split.

## Per-parameter diagnosis accuracy (val, well-supported target classes only, N>=3 signs)

```
  handshape            well-supported: 138/171 = 80.7%   thin/insufficient-data (excluded): 58 clips
  major_location       well-supported: 199/225 = 88.4%   thin/insufficient-data (excluded): 4 clips
  minor_location       well-supported: 158/184 = 85.9%   thin/insufficient-data (excluded): 45 clips
  movement             well-supported: 191/226 = 84.5%   thin/insufficient-data (excluded): 3 clips
  repeated_movement    well-supported: 188/229 = 82.1%   thin/insufficient-data (excluded): 0 clips
```

Thin/singleton-class clips are excluded from the accuracy figures above by design
(see `phonology_labels.py`'s `MIN_SUPPORT` gate) -- their verdicts are reported to a
user as "insufficient data," not folded into an average that would overstate
confidence.

## Head-independence demonstration (mother/father, a real minimal pair)

mother/father differ ONLY in `minor_location` (Forehead vs Chin) -- same handshape,
same movement. Grading a real attempt of one against the other as target:

```
  attempt=father target=mother  fidelity(embedding dist)=0.859
    handshape            predicted='5'          target='5'          [MATCH]
    major_location       predicted='Head'       target='Head'       [MATCH]
    minor_location       predicted='Forehead'   target='Chin'       [DISAGREE]
    movement             predicted='Straight'   target='Straight'   [MATCH]
    repeated_movement    predicted='True'       target='True'       [MATCH]
  attempt=mother target=father  fidelity(embedding dist)=0.742
    handshape            predicted='5'          target='5'          [MATCH]
    major_location       predicted='Neutral'    target='Head'       [DISAGREE]
    minor_location       predicted='Neutral'    target='Forehead'   [DISAGREE]
    movement             predicted='Straight'   target='Straight'   [MATCH]
    repeated_movement    predicted='True'       target='True'       [MATCH]
```

## Overfitting check

See `models/embedding_grader/history.json` for the full per-epoch train/val curve;
the training script tracks and saves the best-val-top1 checkpoint separately from
the final epoch specifically because train top-1 saturates well before val does on
this dataset size (~15 clips/sign) -- the gap itself is the signal, not hidden.
