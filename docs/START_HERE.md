# Manoj, start here

## First see the result

Extract the ZIP and double-click `reports/demo/dashboard.html`. This step needs no Python. Inspect the cards, move the intensity slider and sort the search ledger. The slider shows existing measurements; it does not rerun training.

## Run it on Windows

Install Python 3.11 or 3.12. Open the extracted project in VS Code. Open a terminal in the folder containing `pyproject.toml`:

```powershell
python --version
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,search]"
.venv\Scripts\python.exe -m premortem --backend optuna --trials 80 --output reports/latest
.venv\Scripts\python.exe -m pytest -q
```

Open `reports/latest/dashboard.html`. Installation needs internet; later experiments do not. If `python` is missing but the Windows launcher exists, use `py -3.11` for the first two commands.

## Understand the flow

A row is an example. Features are inputs. The target is the answer (0/1). Our generated signals are not real customer ages or incomes. Logistic regression estimates the probability of class 1; probabilities at least 0.5 predict class 1.

Read `engine.py` in this order: `demo_data`, `split_data`, `model`, `Scenario`, `stress`, `metrics`, `search`, `run`. Then read `report.py`.

1. Train using one set of rows.
2. Find damaging faults using another set.
3. Try retraining with faulty copies of training rows.
4. Choose repair on validation.
5. Test the frozen choice on untouched rows.

## Your first six hours

| Time | Work | Evidence |
|---|---|---|
| 1 hour | Run the project | Produce your own report |
| 1 hour | Python functions, dictionaries, DataFrames | Explain scenario parameters |
| 1 hour | Features, labels and classifiers | Trace one prediction |
| 1 hour | Confusion matrix and balanced accuracy | Calculate a small example manually |
| 1.5 hours | Read and test corruptions | Explain each transformation |
| 30 minutes | Notes and Git | Commit a change you understand |

Exercises: compare equal-budget random and Optuna search; repeat with another seed without cherry-picking; explain why missingness is probabilistic; explain why lower Brier score is better; explain percentage points versus relative percentage change.

## GitHub

No repository was created or modified for this delivery. Create an empty `premortem-ml` repository in your own account and choose its visibility. Then:

```bash
git init
git add .
git commit -m "Build reproducible ML stress testing prototype"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/premortem-ml.git
git push -u origin main
```

Use your actual username. Do not commit private datasets, credentials or the virtual environment. GitHub Actions runs after pushing.

## Troubleshooting

- No module named premortem: enter the folder containing `pyproject.toml`; install and run with the same interpreter.
- No module named optuna: install `.[search]`, or use `--backend random`.
- CSV error: ensure numeric features and 0/1 labels; do not silently discard failing rows.
- Different results: compare seed, trial count and recorded package versions.
