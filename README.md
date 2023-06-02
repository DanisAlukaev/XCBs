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

| Model     | EID-DATASET         | act_fn | pretrain | norm_fn  | slot_norm | dummy_concept | dummy_tokens | reg_dist | tie_loss   |  Performance (F1-score)   | Disentanglement | Completeness    | Directory      |
|:------------|:-----------:|:-----------:|:--------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-------------:|:---------------:|:---------------:|:---------------|
| Baseline | E35-SHP | relu | ✓ | - | - | -| - | - | - | 0.994 ± 0.0 | 0.605 ± 0.0 | 0.726 ± 0.0 | `outputs/2023-06-02/06-01-57` |
| Baseline | E36-SHP | sigmoid | ✓ | - | - | - | - | - | - | 0.998 ± 0.0 | 0.572 ± 0.0 | 0.660 ± 0.0 | `outputs/2023-06-02/05-51-13` |
| Baseline | E37-SHP | gumbel | ✓ | - | - | - | - | - | - | 0.992 ± 0.0  | 0.505 ± 0.0 | 0.579 ± 0.0 | `outputs/2023-06-02/05-38-06` |
| Framework | E38-SHP | sigmoid | ✓ | softmax | ✗ | - | - | ✗ | JS | 0.996 ± 0.0  | 0.587 ± 0.0 | 0.712 ± 0.0 | `outputs/2023-06-02/06-14-39`  |
| Framework | E39-SHP | gumbel | ✓ | softmax | ✗ | - | - | ✗ | JS | X | X | X | X  |
| Framework | E40-SHP | gumbel | ✓ | softmax | ✗ | - | -  | ✗ | KL($f$, $c$) | X | X | X | X  |
| Framework | E41-SHP | gumbel | ✓ | softmax | ✗ | - | - | ✗ | KL($c$, $f$) | X | X | X | X  |
| Framework | E42-SHP | gumbel | ✓ | entmax | ✗ | - | - | ✗ | JS | X | X | X | X  |
| Framework | E43-SHP | gumbel | ✓ | softmax | ✓ | ✓ | ✗ | ✗ | JS | X | X | X | X  |
| Framework | E44-SHP | gumbel | ✓ | softmax | ✓ | ✓ | ✓ | ✗ | JS | X | X | X | X  |
| Framework | E45-SHP | gumbel | ✓ | entmax | ✓ | ✓ | ✗ | ✗ | JS | X | X | X | X  |
| Framework | E46-SHP | gumbel | ✓ | entmax | ✓ | ✓ | ✓ | ✗ | JS | X | X | X | X  |
| Framework | E47-SHP | gumbel | ✓ | softmax | ✗ | - | - | ✓ | JS | X | X | X | X  |
| Framework | E48-SHP | gumbel | ✓ | entmax | ✗ | - | - | ✓ | JS | X | X | X | X  |
