# AutoPlait — validation status

This documents what the shipped tests **do** and **do not** prove for
`AutoPlaitSegmenter` (`aeon/segmentation/_autoplait.py`), and the deferred
human/off-CI validation steps. It is deliberately candid: AutoPlait is a
correctness-critical from-scratch numeric algorithm and "it runs and returns
plausible cut points" is **not** evidence of correctness.

## What this is

A **clean-room, first reviewable draft** implemented **from the SIGMOD 2014
paper only** (Matsubara, Sakurai & Faloutsos, *AutoPlait: Automatic Mining of
Co-evolving Time Sequences*, DOI 10.1145/2588555.2588556). No reference
implementation was read, copied, adapted or paraphrased. The only public
references (`kokikwbt/autoplait`, `ShinichiTokumochi/Autoplait`) are unlicensed
and depend on `hmmlearn`, which aeon does not use; nothing was ported from them.

Dependencies: `numpy`/`scipy` only. No `hmmlearn`, no new dependency. The
Viterbi-only univariate `aeon/segmentation/_hmm.py` was **not** reused.

## CI-runnable validation (emitted + run — `aeon/segmentation/tests/test_autoplait.py`)

1. **Gaussian HMM component tests (the load-bearing correctness floor).**
   The from-scratch multivariate Gaussian HMM (Baum-Welch fit, Viterbi decode,
   log-space score) is tested **on its own, before anything is layered on top**:
   - `test_gaussian_hmm_recovers_known_model` — EM does not decrease the
     achievable log-likelihood.
   - `test_gaussian_hmm_viterbi_recovers_states` — on data sampled from a known
     2-state HMM, Viterbi recovers the ground-truth state path to >90 % accuracy
     up to label permutation.
   These are the highest-risk component; if the HMM were wrong nothing above it
   could be trusted.

2. **Multi-regime segmentation test.** `test_autoplait_recovers_regimes_and_changepoints`
   builds synthetic data by concatenating segments from **two distinct 2-state
   Gaussian HMM regimes with one recurrence (A, B, A)** and asserts that
   AutoPlait recovers the **number of regimes (2)**, that the recurring regime
   **shares a label** across both occurrences, and that the **change points**
   land within tolerance of the known boundaries.

3. **Output-shape / label-type checks** mirroring `tests/test_clasp.py`
   (`test_autoplait_output_shape_and_type`, `test_autoplait_univariate_and_multivariate`,
   `test_autoplait_single_regime`).

4. **aeon conformance suite** — `test_autoplait_conformance` runs
   `check_estimator(AutoPlaitSegmenter, raise_exceptions=True)`. The segmenter is
   registered in `aeon/segmentation/__init__.py` and supplies fast
   `_get_test_params`. As with the other transductive segmenters (ClaSP, FLUSS,
   HMM, GGS, IGTS), it is listed in `EXCLUDED_TESTS` for
   `check_non_state_changing_method` because it fits within `predict`.

## What the CI tests DO prove

- The from-scratch Gaussian HMM fits (LL non-decreasing under EM) and decodes
  (Viterbi recovers a known state path).
- On synthetic multi-regime data, the MDL-driven solver recovers the correct
  **number of regimes**, **recurrence** and **change-point locations** within
  tolerance — i.e. it produces a *valid multi-regime segmentation that recovers
  synthetic ground truth*.
- The estimator conforms to aeon's estimator API.

## What the CI tests DO NOT prove (and is explicitly deferred)

- **Published-dataset parity with the SIGMOD paper is UNVERIFIED.** No parity
  against the paper's reported results was measured, and none is claimed. The
  maintainers' stated acceptance bar ("comparable in performance to the
  original") has **not** been demonstrated here.
- **The exact MDL coding constants are approximated from the paper's described
  scheme, not verified line-by-line against the SIGMOD text.** The cost model
  captures the paper's *structure* — `Cost = CostM + CostC`, `CostC` = negative
  data log-likelihood in bits, `CostM` = `log*`-coded integers (regimes,
  segments, states) plus a fixed per-float cost for HMM parameters — and this
  structure is what drives parameter-free regime selection. But the precise
  float-precision constant (`_COST_FLOAT = 32`), the exact segment/switch
  encoding terms, and the regime-transition (`Δ`) coding were **not**
  cross-checked against the paper's equations and may differ from the original.
- **CutPointSearch** is implemented as a nested Viterbi over an augmented
  `(regime, hidden-state)` state space with a constant per-switch penalty
  derived from the segment-encoding cost. This is faithful *in spirit* to the
  paper's regime-switching dynamic program but the switch-penalty calibration is
  a modelling choice, not a verified transcription.

## Deferred (human, off-CI) validation plan

1. **Behavioural oracle (human, post-hoc).** Run one of the unlicensed
   reference implementations *locally* on **identical synthetic inputs** and
   compare segmentations (number of regimes, boundaries, assignments). This is a
   human comparison of **outputs only**; the reference code must **never** enter
   this repository or the diff (license: all-rights-reserved).
2. **Published-dataset parity.** Reproduce the paper's experiments on its
   datasets and compare regime/segment recovery against the reported SIGMOD
   results. This is the maintainers' stated bar and remains **future work**.

Until (1) and (2) are done, the correctness ceiling of this draft is exactly:
*"produces a valid multi-regime segmentation that recovers synthetic ground
truth"* — not *"matches the paper."*
