# PreMortemML interview guide

## Your 45-second introduction

“PreMortemML is a Python prototype that stress-tests numeric binary classifiers before deployment. It searches bounded combinations of measurement bias, noise, missing values and outliers. I separate training, scenario search and final testing to reduce leakage. It diagnoses the discovered scenario through component ablations, evaluates a targeted augmentation repair and exports a reproducible interactive report. Current results use synthetic data, so they demonstrate the method rather than certify deployment safety.”

Say “I implemented” only for work you have studied, reproduced and can defend. You can honestly explain using an AI coding assistant and then validating and extending the code.

## Step-by-step architecture

1. Generate or load labelled data.
2. Stratify into 60% training, 20% search/validation, 20% final test.
3. Fit imputer, scaler and logistic regression on training data only.
4. Calculate training-only feature standard deviations.
5. Search fault parameters on validation; average three corruption seeds.
6. Train a candidate on clean training rows and three corrupted copies.
7. Select candidate using validation stress gain and clean-score guardrail.
8. Evaluate both models on held-out rows and five fresh corruption seeds.
9. Calculate ablations, intensity curves and conditional bootstrap interval.
10. Export all settings and measurements.

## Likely questions

**What problem does it solve?** Clean evaluation can hide dependence on reliable measurements. This tool tests explicit assumptions about degraded input quality.

**What is unique?** The auditable engineering workflow is the portfolio contribution. Stress testing, TPE search and augmentation are established ideas. It is not a world-first algorithm.

**Why logistic regression?** Fast, interpretable baseline, allowing us to debug the evaluation protocol before adding model complexity.

**What is balanced accuracy?** Average recall across the two classes. Always predicting the majority class gives 50% balanced accuracy when both classes are present, even when ordinary accuracy looks high.

**How does the search work?** Each trial selects a feature and fault parameters, minimizing validation balanced accuracy. Optuna TPE adapts proposals from past trials. Random search provides a simpler baseline. Neither guarantees a global minimum.

**Why constrain it?** Unrestricted corruption can trivially destroy information. Real applications need domain-approved bounds and feature-dependency constraints. Our bounds are generic demo assumptions.

**Is this real distribution shift?** It simulates recorded-measurement corruption and weighted population mixtures. It does not simulate arbitrary concept drift. Fixed labels are defensible for sensor faults, but not necessarily when changing a true causal attribute.

**How is population change represented?** Class importance weights impose a hypothetical positive prevalence. This affects accuracy and precision, but not balanced accuracy, so prevalence is not driving the search objective.

**How is leakage prevented?** Training-only preprocessing and scales; validation-only scenario discovery and repair selection; final test results never select a model. Repeated tuning against reports would invalidate the holdout.

**How do explanations work?** Remove one fault component at a time with common random draws and measure recovery. These are ablations, not SHAP or causal proof. They do not fully disentangle interactions.

**How does repair work?** Add three corrupted copies of training rows, fit a fresh pipeline, and select on validation only if stress improves and clean balanced accuracy loses no more than two points. Targeted repair may not generalize to other scenarios.

**What uncertainty is measured?** A paired stratified bootstrap resamples test rows for both models together. The interval conditions on one corruption seed and the selected scenario; it excludes search and fitting uncertainty.

**What does HIGH mean?** Heuristic review priority: stressed balanced accuracy below 65% or a drop of at least 15 points. It is not a validated risk probability or deployment approval.

**Why no pickle upload?** Untrusted serialized models can execute code. This version trains a known model from numeric CSVs. Future model integration needs an explicit trust boundary.

**What did you test?** Disjoint splits, deterministic and isolated perturbations, no input mutation, zero-corruption identity, invalid parameters, prevalence weights and end-to-end report generation. Local tests ran; the supplied CI workflow has not yet run on GitHub.

**Main limitation?** Synthetic data, generic bounds and one-feature-at-a-time search. Next, use a real domain benchmark with independently justified faults and reserve new evaluation data.

**Next improvements?** Multi-feature constraints, categorical handling, time/group splits, repeated-training evaluation, independent repair scenario suites and only then service infrastructure.

## Five-minute demo

| Minute | Show |
|---|---|
| 0–1 | Problem and synthetic-data disclosure |
| 1–2 | Clean vs stressed balanced accuracy and exact scenario |
| 2–3 | Trial ledger, intensity curve and ablations |
| 3–4 | Repair selection and held-out comparison |
| 4–5 | Reproducibility, uncertainty and limitations |

## Resume wording after you understand the implementation

“Built a reproducible numeric-classifier stress-testing prototype with bounded corruption search, held-out evaluation, component ablations and validation-selected augmentation repair using Python, scikit-learn and Optuna.”

When mentioning measured improvements, say synthetic benchmark and disclose trial budget/seed context. Do not claim production deployment, causality, guaranteed salary, global worst-case discovery or tools absent from the implementation.
