# MASSIVE German Intent Classification: Hyperparameter Study

Code for the paper *"How do different hyperparameters affect the performance of a
Transformer model for German intent classification on the MASSIVE dataset?"*

It fine-tunes a German Transformer encoder (`deepset/gbert-base`) on the German
subset (`de-DE`) of [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive)
and measures the effect of individual hyperparameters in a controlled,
one-factor-at-a-time design.

## Data flow

```
configs/*.yaml
  -> data.py          load MASSIVE de-DE (official train/val/test), keep text + intent
  -> tokenization.py  tokenize utterances to fixed max length
  -> model.py         gbert-base encoder + classification head
  -> trainer.py       fine-tune, keep best epoch by validation accuracy
  -> pipeline.py      orchestrates the above for one config
  -> evaluate          validation + test accuracy / macro-F1 (+ test predictions)
  -> results/*.json   one file per run (config + scores + predictions)
  -> aggregate_results.py  -> LaTeX tables + scaling figure for the report
```

## Setup

Requires Python 3.10+ and an NVIDIA GPU for reasonable runtimes (developed on an
RTX 3060, 12 GB).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# GPU build of PyTorch (CUDA 12.4); use the CPU index-url if you have no GPU.
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Obtaining the data

No manual download is needed: `datasets` fetches MASSIVE `de-DE` from the Hugging
Face Hub on first run and caches it locally.

## Running

All commands are run from the repository root with the package on the path.

```powershell
# Smoke test (tiny subset, 1 epoch) to verify the environment:
python -m scripts.run_experiment --config configs/baseline.yaml --smoke

# A single configuration:
python -m scripts.run_experiment --config configs/baseline.yaml

# The full controlled matrix (baseline repeated over extra seeds):
python -m scripts.run_matrix --repeat baseline

# Extension: low-resource data scaling (10/25/50/100 %):
python -m scripts.run_extension --config configs/baseline.yaml

# Aggregate all results into CSVs, LaTeX tables and the scaling figure:
python -m scripts.aggregate_results

# Qualitative error analysis (top intent confusions) for one run:
python -m scripts.error_analysis --result results/baseline_seed42.json
```

## Reproducibility

- Every run is fully specified by a config file plus a fixed `seed`.
- `src/seeding.py` seeds Python, NumPy and PyTorch.
- The baseline and the strongest configuration are repeated over additional
  seeds (`scripts/run_matrix.py`) to estimate seed noise.
- Pinned dependency versions are in `requirements.txt`.

## Layout

```
configs/    one YAML per run; non-baseline configs change exactly one factor
src/        single-responsibility modules (data, tokenization, model, trainer, ...)
scripts/    thin CLIs that parse arguments and call src/
results/    run JSONs, aggregated CSV/LaTeX tables, figures
```
