# Automatic Concept Bottleneck Models

## 🚀 Getting Started
1. Set-up conda environment
    ```
    conda create --name bottleneck --file requirements.txt python=3.10.8
    ```
2. Download public datasets
    ```
    make download_data
    ```
3. Pre-process datasets
    ```
    # CUB-200
    make preprocess_cub

    # MIMIC-CXR
    make preprocess_mimic
    ```
4. You are ready to run your first experiment!
    ```
    python main.py dataset.batch_size=64 seed=42 +experiment={XXX}
    ```

## 🤔 FAQ

1. How can I retrieve explanations?
    > Our pipeline saves explanations in `results.json`, which can be found in experiment folder. For visualization you can use [`inspect.ipynb`](./autoconcept/inspect.ipynb) notebook.

2. How can I assess my model in terms of DCI?
    > Measuring disentanglement, completeness, informativeness is moved outside the default pipeline and can be performed via [`metrics.ipynb`](./autoconcept/metrics.ipynb) notebook.


## 🧬 Experiments

| Model     | EID         | activation | norm_fn  | slot_norm | reg_dist | tie_loss   |  F1-score     | Disentanglement | Completeness    | Directory      |
|:------------|:-----------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:-------------:|:---------------:|:---------------:|:---------------|
|  |  |  |  |  |  |  |  |  |  | -  |
| Baseline    | EXX-SHP     | sigmoid | entmax   |  ✗ | ✓ | KL (w.r.t. $c$) | 0.000 ± 0.0   | 0.000 ± 0.0     | 0.000 ± 0.0     | -              |
