# Trustworthy Project — Reproducibility Guide

This folder captures the executed notebooks, cached activations, vectors, and scripts needed to regenerate the key figures and metrics used in the sycophancy steering analysis. Everything below is self-contained and operates only on cached artifacts (no new forward passes required).

## Contents
- `notebooks/`
  - `template_controlling_final_fulltrain.executed.ipynb`: Generates per-layer raw/centered PCA, template-classification metrics, and the raw-vs-centered accuracy table.
  - `sycophancy_supervised_vector_remote.executed.ipynb`: Trains the supervised vector `v_sup`, logs projection/ROC metrics, and plots validation projection histograms (`llama2_layer26_val_projection_hist.png`).
  - `template_generalization_improved.executed.ipynb`: Template-generalization probe and projection diagnostics; saves `template_generalization_projection_hist.png` and stats CSV.
- `scripts/`
  - `combine_layer_pca_grid.py`: Assembles per-layer PCA PNGs into the combined grid `layer_raw_vs_centered_grid_clean.png`.
- `artifacts/`
  - Cached activations: `llama2_hidden_states_train.npz`, `llama2_hidden_states_val.npz` (layers 20/26/28, 4096 dims).
  - Steering vectors: `llama2_layer20/26/28_supervised_vector.npy` (unit-norm centered vectors).
  - PCA panels: `layer_20/26/28_raw_pca.png`, `layer_20/26/28_centered_pca.png`, combined `layer_raw_vs_centered_grid.png` and `_clean.png`.
  - Projection diagnostics: `llama2_layer26_val_projection_hist.png`, `llama2_layer26_projection_stats.json`, `template_generalization_projection_hist.png`, `template_generalization_projection_stats.csv`.
  - Hidden-state derivatives for binary probes: `llama2_layer*_affirm_vs_question.npy`, `llama2_layer*_deny_vs_question.npy` (generated in the fulltrain notebook).
- `data/splits/`: Deterministic JSONL splits used throughout (`sycophancy_eval_answer_train.jsonl`, `..._val.jsonl`).
- `beam_cache/`: Original Beam run caches (`hidden_states.npz`, `metadata.json`) used by the fulltrain notebook.

## Environment
Python 3.9+ with `transformers`, `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, `Pillow`. SentencePiece is required for the Llama tokenizer.

Example setup (CPU/MPS, no new forward passes needed):
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers sentencepiece numpy pandas scikit-learn matplotlib tqdm pillow
```

## Reproducing Figures and Tables

1) **Raw vs. centered accuracy table & per-layer PCA panels**
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
   - Open `notebooks/sycophancy_supervised_vector_remote.executed.ipynb` and run the projection/ROC cells. They read:
     - `artifacts/llama2_hidden_states_{train,val}.npz`
     - `artifacts/llama2_layer26_supervised_vector.npy`
   - Outputs: `artifacts/llama2_layer26_val_projection_hist.png`, `artifacts/llama2_layer26_projection_stats.json`.

4) **Template generalization projection histogram**
   - Open `notebooks/template_generalization_improved.executed.ipynb` and run the template-generalization section.
   - Outputs: `artifacts/template_generalization_projection_hist.png`, `artifacts/template_generalization_projection_stats.csv`.

5) **Data dependencies**
   - All notebooks expect the JSONL splits in `data/splits/` and the cached activations/vectors in `artifacts/`. No additional downloads or forward passes are required to regenerate the reported figures.

## Notes
- The `.executed.ipynb` notebooks already contain outputs. Re-running them is optional unless you want to regenerate plots from scratch.
- All steering/probe analyses in this folder operate on cached activations; they will not contact external APIs.

