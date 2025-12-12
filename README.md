# Trustworthy Project — Reproducibility Guide

This folder captures the executed notebooks, scripts, and plots needed to reproduce the key figures and metrics for sycophancy steering. To keep the repository pushable to GitHub, large activation caches (.npz) are **not** tracked; instead, scripts are provided to regenerate them. Follow the steps below to rebuild the data and reproduce every figure/table.

## Contents
- `notebooks/`
  - `template_controlling_final_fulltrain.executed.ipynb`: Generates per-layer raw/centered PCA, template-classification metrics, and the raw-vs-centered accuracy table.
  - `sycophancy_supervised_vector_remote.executed.ipynb`: Trains the supervised vector `v_sup`, logs projection/ROC metrics, and plots validation projection histograms (`llama2_layer26_val_projection_hist.png`).
  - `template_generalization_improved.executed.ipynb`: Template-generalization probe and projection diagnostics; saves `template_generalization_projection_hist.png` and stats CSV.
- `scripts/`
  - `combine_layer_pca_grid.py`: Assembles per-layer PCA PNGs into the combined grid `layer_raw_vs_centered_grid_clean.png`.
  - `compute_sycophancy_cache.py`: Runs the model to produce hidden-state caches and deterministic labels for train/val (requires GPU).
- `labeler.py`: Deterministic parsing function (`improved_label`) used by the notebooks/scripts.
- `artifacts/`
  - Steering vectors (small): `llama2_layer20/26/28_supervised_vector.npy`.
  - PCA panels: `layer_20/26/28_raw_pca.png`, `layer_20/26/28_centered_pca.png`, combined `layer_raw_vs_centered_grid.png` and `_clean.png`.
  - Projection diagnostics: `llama2_layer26_val_projection_hist.png`, `llama2_layer26_projection_stats.json`, `template_generalization_projection_hist.png`, `template_generalization_projection_stats.csv`.
  - NOTE: Large activation caches (`llama2_hidden_states_{train,val}.npz`) are intentionally **omitted** to satisfy GitHub size limits—regenerate them via `scripts/compute_sycophancy_cache.py`.
- `data/splits/`: Deterministic JSONL splits (`sycophancy_eval_answer_train.jsonl`, `..._val.jsonl`).
- `beam_cache/metadata.json`: Template/question metadata (small) from the Beam run; the large hidden-state file is removed.

## Environment
Python 3.9+ with `transformers`, `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, `Pillow`. SentencePiece is required for the Llama tokenizer.

Example setup (CPU/MPS, no new forward passes needed):
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers sentencepiece numpy pandas scikit-learn matplotlib tqdm pillow
```

## Reproducing Data, Figures, and Tables

### 0) Regenerate activation caches (required before notebooks)
- Requires GPU and a valid Hugging Face token (`HF_TOKEN` env var).
- Defaults to layers `20,26,28` and model `meta-llama/Llama-2-7b-hf` (override via env).
```bash
export HF_TOKEN=YOUR_HF_TOKEN
export MODEL_NAME=meta-llama/Llama-2-7b-hf
export TARGET_LAYERS=20,26,28
python scripts/compute_sycophancy_cache.py
```
Outputs:
- `artifacts/sycophancy_supervised_train.npz`, `artifacts/sycophancy_supervised_val.npz` (prompt/response activations + labels)
- `artifacts/sycophancy_supervised_train.jsonl`, `artifacts/sycophancy_supervised_val.jsonl` (labels + metadata)
If you need the merged hidden-state format used by older notebooks, rename/copy these to `artifacts/llama2_hidden_states_{train,val}.npz` or adjust the notebook paths accordingly.
The script uses `labeler.deterministic_label` to assign sycophancy labels deterministically.

1) **Raw vs. centered accuracy table & per-layer PCA panels**
   - Ensure caches from step 0 exist (or point the notebook to `artifacts/sycophancy_supervised_{train,val}.npz`).
   - Open `notebooks/template_controlling_final_fulltrain.executed.ipynb` and re-run the metric/report cells, or convert via nbconvert:
   ```bash
   jupyter nbconvert --to notebook --execute notebooks/template_controlling_final_fulltrain.executed.ipynb \
     --output template_controlling_final_fulltrain.rerun.ipynb --ExecutePreprocessor.timeout=0
   ```
   - Outputs: `artifacts/layer_*_raw_pca.png`, `artifacts/layer_*_centered_pca.png`, and the accuracy table inside the notebook.

2) **Combined PCA grid**
   - Requires the per-layer PCA PNGs from step 1.
   - Run:
   ```bash
   python scripts/combine_layer_pca_grid.py
   ```
   - Output: `artifacts/layer_raw_vs_centered_grid_clean.png`.

3) **Projection histogram/statistics (validation, layer 26)**
   - Ensure caches from step 0 exist.
   - Open `notebooks/sycophancy_supervised_vector_remote.executed.ipynb` and run the projection/ROC cells. They read:
     - `artifacts/sycophancy_supervised_{train,val}.npz` (or the renamed `llama2_hidden_states_{train,val}.npz`)
     - `artifacts/llama2_layer26_supervised_vector.npy` (small, already included)
   - Outputs: `artifacts/llama2_layer26_val_projection_hist.png`, `artifacts/llama2_layer26_projection_stats.json`.

4) **Template generalization projection histogram**
   - Open `notebooks/template_generalization_improved.executed.ipynb` and run the template-generalization section (requires caches from step 0).
   - Outputs: `artifacts/template_generalization_projection_hist.png`, `artifacts/template_generalization_projection_stats.csv`.

5) **Data dependencies**
   - Notebooks expect: `data/splits/*.jsonl` (included), regenerated caches in `artifacts/` from step 0, steering vectors (included), and the deterministic parser in `labeler.py`.

## Notes
- The `.executed.ipynb` notebooks already contain outputs; re-run only if you regenerate caches.
- Large `.npz` activation caches are intentionally omitted to satisfy GitHub limits; regenerate them as described in step 0.
- Steering/probe analyses operate on cached activations after you regenerate them; no external API calls occur during notebook execution.
