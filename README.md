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

### 1. Shapes Dataset

| Model     | EID-DATASET         | act_fn | pretrain | norm_fn  | slot_norm | dummy_concept | dummy_tokens | reg_dist | tie_loss   |  Performance (F1-score)   | Disentanglement | Completeness    | Directory      |
|:------------|:-----------:|:-----------:|:--------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-------------:|:---------------:|:---------------:|:---------------|
| Baseline | E35-SHP | relu | ✓ | - | - | -| - | - | - | 0.994 ± 0.0 | 0.605 ± 0.0 | 0.726 ± 0.0 | `outputs/2023-06-02/06-31-54` |
| Baseline | E36-SHP | sigmoid | ✓ | - | - | - | - | - | - |  0.998 ± 0.0 | 0.572 ± 0.0 | 0.660 ± 0.0 | `outputs/2023-06-02/06-44-03` |
| Baseline | E37-SHP | gumbel | ✓ | - | - | - | - | - | - |  0.992 ± 0.0 | 0.505 ± 0.0 | 0.579 ± 0.0 | `outputs/2023-06-02/06-54-19` |
| Framework | E38-SHP | sigmoid | ✓ | softmax | ✗ | - | - | ✗ | JS |  0.992 ± 0.0 | 0.510 ± 0.0 | 0.658 ± 0.0 | `outputs/2023-06-02/07-04-49` |
| Framework | E39-SHP | gumbel | ✓ | softmax | ✗ | - | - | ✗ | JS | 0.913 ± 0.0 | 0.730 ± 0.0 | 0.727 ± 0.0 | `outputs/2023-06-02/07-18-28` |
| Framework | E40-SHP | gumbel | ✓ | softmax | ✗ | - | -  | ✗ | KL($f$, $c$) | 0.586 ± 0.0 | 0.695 ± 0.0 | 0.624 ± 0.0 | `outputs/2023-06-02/07-29-29` |
| Framework | E41-SHP | gumbel | ✓ | softmax | ✗ | - | - | ✗ | KL($c$, $f$) | 0.602 ± 0.0 | 0.764 ± 0.0 | 0.701 ± 0.0 | `outputs/2023-06-02/07-41-38` |
| Framework | E42-SHP | gumbel | ✓ | entmax | ✗ | - | - | ✗ | JS | 0.888 ± 0.0 | 0.763 ± 0.0 | 0.827 ± 0.0 | `outputs/2023-06-02/07-52-20`  |
| Framework | E43-SHP | gumbel | ✓ | softmax | ✓ | ✓ | ✗ | ✗ | JS | 0.730 ± 0.0 | 0.733 ± 0.0 | 0.705 ± 0.0 | `outputs/2023-06-02/08-02-59` |
| Framework | E44-SHP | gumbel | ✓ | softmax | ✓ | ✓ | ✓ | ✗ | JS | 0.792 ± 0.0 | 0.662 ± 0.0 | 0.773 ± 0.0 | `outputs/2023-06-02/08-13-11` |
| Framework | E45-SHP | gumbel | ✓ | entmax | ✓ | ✓ | ✗ | ✗ | JS | 0.673 ± 0.0 | 0.739 ± 0.0 | 0.748 ± 0.0 | `outputs/2023-06-02/08-31-29` |
| Framework | E46-SHP | gumbel | ✓ | entmax | ✓ | ✓ | ✓ | ✗ | JS | 0.712 ± 0.0 | 0.739 ± 0.0 | 0.748 ± 0.0 | `outputs/2023-06-02/08-31-29` |
| Framework | E47-SHP | gumbel | ✓ | softmax | ✗ | - | - | ✓ | JS | 0.912 ± 0.0 | 0.730 ± 0.0 | 0.727 ± 0.0 | `outputs/2023-06-02/08-52-26`  |
| Framework | E48-SHP | gumbel | ✓ | entmax | ✗ | - | - | ✓ | JS | 0.888 ± 0.0 | 0.763 ± 0.0 | 0.827 ± 0.0 | `outputs/2023-06-02/09-02-55` |
