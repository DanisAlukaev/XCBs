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

### 1. Shapes Dataset (ablation study)

**a. Fine-tuned feature extractor.**

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

**b. Training of feature extractor from scratch.**

| Model     | EID-DATASET         | act_fn | pretrain | norm_fn  | slot_norm | dummy_concept | dummy_tokens | reg_dist | tie_loss   |  Performance (F1-score)   | Disentanglement | Completeness    | Directory      |
|:------------|:-----------:|:-----------:|:--------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-------------:|:---------------:|:---------------:|:---------------|
| Baseline | E49-SHP | relu | ✗ | - | - | -| - | - | - | 0.972 ± 0.0 | 0.574 ± 0.0 | 0.647 ± 0.0 | `outputs/2023-06-02/09-57-26` |
| Baseline | E50-SHP | sigmoid | ✗ | - | - | - | - | - | - | 0.983 ± 0.0 | 0.495 ± 0.0 | 0.394 ± 0.0 | `outputs/2023-06-02/10-08-34` |
| Baseline | E51-SHP | gumbel | ✗ | - | - | - | - | - | - | 0.490 ± 0.0 | 0.529 ± 0.0 | 0.435 ± 0.0 | `outputs/2023-06-02/10-21-00` |
| Framework | E52-SHP | sigmoid | ✗ | softmax | ✗ | - | - | ✗ | JS | 0.989 ± 0.0 | 0.509 ± 0.0 | 0.422 ± 0.0 | `outputs/2023-06-02/10-32-52` |
| Framework | E53-SHP | gumbel | ✗ | softmax | ✗ | - | - | ✗ | JS | 0.682 ± 0.0 | 0.536 ± 0.0 | 0.514 ± 0.0 | `outputs/2023-06-02/11-22-21` |
| Framework | E54-SHP | gumbel | ✗ | softmax | ✗ | - | -  | ✗ | KL($f$, $c$) | 0.377 ± 0.0 | 0.578 ± 0.0 | 0.605 ± 0.0 | `outputs/2023-06-02/11-34-23` |
| Framework | E55-SHP | gumbel | ✗ | softmax | ✗ | - | - | ✗ | KL($c$, $f$) | 0.394 ± 0.0 | 0.579 ± 0.0 | 0.566 ± 0.0 | `outputs/2023-06-02/11-44-17` |
| Framework | E56-SHP | gumbel | ✗ | entmax | ✗ | - | - | ✗ | JS | 0.740 ± 0.0 | 0.525 ± 0.0 | 0.506 ± 0.0 | `outputs/2023-06-02/11-54-53` |
| Framework | E57-SHP | gumbel | ✗ | softmax | ✓ | ✓ | ✗ | ✗ | JS | 0.509 ± 0.0 | 0.602 ± 0.0 | 0.642 ± 0.0 | `outputs/2023-06-02/12-05-44` |
| Framework | E58-SHP | gumbel | ✗ | softmax | ✓ | ✓ | ✓ | ✗ | JS | 0.513 ± 0.0 | 0.540 ± 0.0 | 0.564 ± 0.0 | `outputs/2023-06-02/12-16-04` |
| Framework | E59-SHP | gumbel | ✗ | entmax | ✓ | ✓ | ✗ | ✗ | JS | 0.673 ± 0.0 | 0.739 ± 0.0 | 0.748 ± 0.0 | `outputs/2023-06-02/12-26-31` |
| Framework | E60-SHP | gumbel | ✗ | entmax | ✓ | ✓ | ✓ | ✗ | JS | 0.471 ± 0.0 | 0.680 ± 0.0 | 0.669 ± 0.0 | `outputs/2023-06-02/12-37-21` |
| Framework | E61-SHP | gumbel | ✗ | softmax | ✗ | - | - | ✓ | JS | 0.682 ± 0.0 | 0.534 ± 0.0 | 0.511 ± 0.0 | `outputs/2023-06-02/12-52-49` |
| Framework | E62-SHP | gumbel | ✗ | entmax | ✗ | - | - | ✓ | JS | 0.740 ± 0.0 | 0.525 ± 0.0 | 0.506 ± 0.0 | `outputs/2023-06-02/13-02-58` |
