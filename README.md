# PreMortemML

## Web workspace (v0.2)

### Extra analysis tools

- **Insights:** a downloadable, rule-based decision brief with measured findings and suggested investigations; no external AI service.
- **Feature vulnerability ranking:** observed worst and mean validation scores with per-feature trial counts. Untested features are explicitly marked. Unequal search coverage is not feature importance.
- **Stress intensity explorer:** a slider over the 11 measured failure-curve points, displaying baseline and candidate scores without inventing new simulations.
- **Compare runs:** side-by-side saved experiment scores with warnings for differing datasets, seeds, environments, and discovered scenarios. Comparisons do not select models using held-out results.

Check derived insight logic with `node tests/test_insights.cjs`.

This project now includes a working local web application around the original ML engine.

```powershell
cd 'C:\min sih\premortem-ml'
.\start.ps1
```

Open **http://127.0.0.1:8765**. The supplied reference experiment appears on first launch. Choose **New experiment** to run the synthetic benchmark or upload a CSV. You can select random or Optuna search, set trials and seed, browse persisted experiments, inspect failure curves, compare model metrics, sort search history, and download HTML, JSON, and CSV evidence.

Experiments run in a single background worker; up to three jobs can be queued/running. SQLite metadata and generated reports persist in `workspace/`. Uploaded datasets are held in memory for execution; they are not saved as raw CSVs. The data hash, settings, package versions, and results are saved. Retain your original CSV for reproduction. Interrupted jobs are marked failed at restart and can be rerun. This is a single-user local application bound to loopback, not a public multi-user service.

The web app accepts numeric binary-classification datasets with up to 50,000 rows, 100 features, and an 8 MB upload. Existing scientific limitations below still apply. The app trains its own logistic classifier; it does not ingest external model files. Font loading is optional; system fonts work without internet.

Run checks: `.venv\Scripts\python.exe -m pytest -q`.

Application files: `premortem/server.py` (HTTP API, queue, persistence), `premortem/web/` (responsive frontend), and `tests/test_server.py` (API integration checks). The original CLI and portable report remain available.

---

## Original prototype reference

Pre-deployment stress simulation for numeric binary classifiers. **Working research prototype, v0.1.0.**

Open `reports/demo/dashboard.html` first: a completed, offline interactive experiment. Then read `docs/START_HERE.md` and `docs/INTERVIEW_GUIDE.md`.

## Run (Python 3.11 or 3.12)

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev,search]"
python -m premortem --backend optuna --trials 80 --output reports/latest
python -m pytest -q
```

Open `reports/latest/dashboard.html`. On Windows, if activation is blocked, replace `python` with `.venv\Scripts\python.exe` after creating the environment.

## Own dataset

```bash
python -m premortem --csv data.csv --target churn --backend optuna --trials 80
```

CSV: at least 200 rows, numeric features, target exactly 0/1, at least 30 examples per class. No infinity or entirely missing feature. Partial feature missingness is supported. Remove IDs and outcome-leaking columns. Rows must be independent: grouped/temporal splitting is not implemented. No arbitrary pickle uploads.

## Implemented

- Seeded synthetic data; CSV input; CLI.
- Stratified 60/20/20 training/search/final-test split.
- Training-only median imputation, scaling and logistic regression.
- Feature measurement bias, Gaussian noise, random missingness and outliers.
- Label-prevalence reweighting; random or Optuna TPE search.
- Three corruption seeds per search trial; five seeds in final assessment.
- Targeted augmentation repair selected on validation, with a clean-score guardrail.
- Accuracy, balanced accuracy, precision, recall, F1, ROC AUC, Brier score and confusion matrices.
- Component ablations, intensity curves and paired stratified bootstrap interval.
- Offline interactive HTML report, JSON evidence, CSV trial ledger and reproducibility manifest.
- Automated tests and GitHub Actions configuration.

## Protocol

Only training rows fit the baseline pipeline and feature scales. Search chooses a scenario on validation rows. Repair fits original training rows plus three corrupted copies. Choose repair when validation stress balanced accuracy improves and clean validation balanced accuracy drops by no more than 2 percentage points. Final held-out results never alter that selection. Repeatedly tuning against the final report invalidates that holdout.

The search changes **one selected feature with several corruption types**, not several features jointly. Bounds: offset ±1.5 training SD; noise 0–1 SD; missingness 0–40%; ±5 SD outliers affecting 0–10% of rows; hypothetical positive-class prevalence 20–80%. These generic bounds need domain validation; bounded does not automatically mean realistic.

Labels remain fixed because these are recorded-measurement faults. This is not arbitrary concept drift or causal intervention. Prevalence is class weighting, not resampling: it changes accuracy and precision but not the balanced-accuracy objective. Its sampled value must not be interpreted as an optimized prevalence vulnerability.

Ablations remove one corruption with common random draws; they are diagnostics, not causal proof. The bootstrap interval conditions on a selected scenario and one fixed corruption realization. It excludes fitting/search uncertainty. Repair can overfit the discovered scenario; wider robustness is not established.

Review priority: HIGH when stressed balanced accuracy <65% or drop ≥15 percentage points; MEDIUM when drop ≥5 points; LOW otherwise. This transparent policy example is not a calibrated deployment safety score.

## Scope and future work

Not implemented: multi-feature constrained shifts, categorical pipelines, time/group splits, arbitrary external models, SHAP, MLflow, FastAPI, PostgreSQL, Docker, CV/LLMs or certified reliability scoring. The dashboard inspects generated results; new runs use the CLI. Next: a domain-specific benchmark, justified perturbation bounds and an independent scenario suite for repairs.

## Files

- `premortem/engine.py`: data, models, corruption, search, repair, evaluation.
- `premortem/__main__.py`: command line.
- `premortem/report.py` and `dashboard.html`: portable report.
- `reports/demo/`: actual measured example and trial ledger.
- `tests/`: automated checks.
- `docs/`: learning and interview guides.
- `.github/workflows/tests.yml`: CI configuration, not yet executed on GitHub.

## Reproduce

The supplied demo uses 2,400 synthetic rows, eight abstract signals, seed 42 and 80 Optuna trials. `reports/demo/results.json` records package versions, data hash, split sizes and metrics. `requirements-tested.txt` records exact top-level package versions from the delivered run. For closer reproduction, install it before running. Results can vary across dependency versions/platforms.

## References

[Scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html) · [Pipeline documentation](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) · [Optuna documentation](https://optuna.readthedocs.io/)

An original portfolio implementation of established techniques, not a claim of world-first research.
