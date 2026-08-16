"""Research comparisons -- explicitly NOT part of the shipped product
(Phases 0-8). Nothing in this package is imported by `diagnose_demo.py` or
any other learner-facing script, and nothing here is shown to a learner.

Exists to answer a specific question empirically rather than by assertion:
how does this project's chosen approach (retrieve a real reference, grade by
distance to it -- CLAUDE.md's core non-negotiable) compare to the industry-
standard alternative for continuous sign recognition (CTC-based CSLR, an
open-vocabulary classifier)? See project_workflow.md's "CTC-CSLR comparison"
section for the full writeup, including why CTC itself was deliberately kept
out of the shipped product: it is exactly the kind of N-way classifier
CLAUDE.md's non-negotiables rule out (a classifier must emit SOME label, and
will confidently mislabel a malformed attempt as a real sign -- useless for
grading a learner who frequently gets it wrong), and "free-form translation"
is explicitly out of scope. Building it here, isolated, lets that comparison
happen honestly without importing that risk into the product.

Also explicitly not "SOTA" despite using a SOTA-shaped architecture (CTC):
real SOTA (DeepMind's SL2T) trains on 100k+ hours across 50+ languages. This
trains on the ~1,270 clips of this project's own 60-sign curriculum train
split -- a fair architecture comparison on this project's own data, not a
claim of matching published state-of-the-art numbers.
"""
