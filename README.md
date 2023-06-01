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

| Model     | EID-DATASET         | act_fn | norm_fn  | slot_norm | reg_dist | tie_loss   |  Performance (F1-score)   | Disentanglement | Completeness    | Directory      |
|:------------|:-----------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:-------------:|:---------------:|:---------------:|:---------------|
| Baseline | E35-SHP | relu | - | - | - | - | 0.994 ± 0.0 | 0.520 ± 0.0 | 0.542 ± 0.0 | `outputs/2023-06-01/20-42-08` |
| Baseline | E36-SHP | sigmoid | - | - | - | - | X | X | X | X  |
| Baseline | E37-SHP | gumbel | - | - | - | - | X | X | X | X  |
| Framework | E38-SHP | sigmoid | softmax | ✗ | ✗ | JS | X | X | X | X  |
| Framework | E39-SHP | gumbel | softmax | ✗ | ✗ | JS | X | X | X | X  |
| Framework | E40-SHP | gumbel | softmax | ✗ | ✗ | KL($f$, $c$) | X | X | X | X  |
| Framework | E41-SHP | gumbel | softmax | ✗ | ✗ | KL($c$, $f$) | X | X | X | X  |
| Framework | E42-SHP | gumbel | entmax | ✗ | ✗ | JS | X | X | X | X  |
| Framework | E43-SHP | gumbel | softmax | ✓ | ✗ | JS | X | X | X | X  |
| Framework | E44-SHP | gumbel | entmax | ✓ | ✗ | JS | X | X | X | X  |
| Framework | E45-SHP | gumbel | softmax | ✗ | ✓ | JS | X | X | X | X  |
| Framework | E46-SHP | gumbel | entmax | ✗ | ✓ | JS | X | X | X | X  |
|  |  |  |  |  |  |  |  |  |  | - |
| Baseline    | EXX-SHP     | sigmoid | entmax   |  ✗ | ✓ | KL (w.r.t. $c$) | 0.000 ± 0.0   | 0.000 ± 0.0     | 0.000 ± 0.0     | -              |
