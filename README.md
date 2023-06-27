# Cross-Modal Conceptualization in Bottleneck Models

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

| Model     | EID-DATASET  | Performance (F1-score)   | Disentanglement | Completeness   | Informativeness | act_fn  | norm_fn  | slot_norm | dummy_concept | dummy_tokens | reg_dist | tie_loss   |
|:------------|:-----------:|:-----------:|:--------:|:--------:|:--------:|:----------:|:----------:|:----------:|:-------------:|:---------------:|:---------------:|:---------------:|
| Baseline | E35-SHP | 1.00 ± 0.00 | 0.60 ± 0.05 | 0.64 ± 0.10 | 0.07 ± 0.01 | relu | - | - | -| - | - | - |
| Baseline | E36-SHP | 1.00 ± 0.00 | 0.68 ± 0.10 | 0.72 ± 0.05 | 0.08 ± 0.01 | sigmoid  | - | - | - | - | - | - |
| Baseline | E37-SHP | 0.99 ± 0.00 | 0.51 ± 0.01 | 0.56 ± 0.06 | 0.08 ± 0.02  | gumbel  | - | - | - | - | - | - |
| Framework | E38-SHP |  0.99 ± 0.00 | 0.64 ± 0.09 | 0.68 ± 0.04| 0.08 ± 0.03 | sigmoid | softmax | ✗ | - | - | ✗ | JS |
| Framework | E39-SHP | 0.93 ± 0.01 | 0.78 ± 0.03 | 0.74 ± 0.12 | 0.07 ± 0.02 | gumbel | softmax | ✗ | - | - | ✗ | JS |
| Framework | E40-SHP | 0.62 ± 0.04 | 0.67 ± 0.03 | 0.71 ± 0.05 | 0.09 ± 0.03 | gumbel  | softmax | ✗ | - | -  | ✗ | KL($f$, $c$) |
| Framework | E41-SHP | 0.65 ± 0.06 | 0.69 ± 0.06 | 0.72 ± 0.05 | 0.09 ± 0.02 | gumbel | softmax | ✗ | - | - | ✗ | KL($c$, $f$) |
| Framework | E42-SHP | 0.91 ± 0.01 | 0.75 ± 0.03 | 0.73 ± 0.10 | 0.08 ± 0.01 | gumbel | entmax | ✗ | - | - | ✗ | JS |
| Framework | E43-SHP | 0.75 ± 0.04 | 0.71 ± 0.06 | 0.71 ± 0.07 | 0.09 ± 0.01 | gumbel | softmax | ✓ | ✓ | ✗ | ✗ | JS |
| Framework | E44-SHP | 0.72 ± 0.04 | 0.74 ± 0.06 | 0.79 ± 0.14 | 0.07 ± 0.02 | gumbel | softmax | ✓ | ✓ | ✓ | ✗ | JS |
| Framework | E45-SHP | 0.64 ± 0.04 | 0.75 ± 0.03 | 0.75 ± 0.02 | 0.08 ± 0.02 | gumbel | entmax | ✓ | ✓ | ✗ | ✗ | JS |
| Framework | E46-SHP | 0.66 ± 0.05 | 0.69 ± 0.05 | 0.73 ± 0.08 | 0.08 ± 0.03 | gumbel | entmax | ✓ | ✓ | ✓ | ✗ | JS |
| Framework | E47-SHP | 0.93 ± 0.01 | 0.78 ± 0.03 | 0.74 ± 0.12 | 0.07 ± 0.02 | gumbel  | softmax | ✗ | - | - | ✓ | JS |
| Framework | E48-SHP | 0.93 ± 0.01 | 0.75 ± 0.03 | 0.73 ± 0.10 | 0.08 ± 0.01 | gumbel  | entmax | ✗ | - | - | ✓ | JS |

**b. Training of feature extractor from scratch.**

| Model     | EID-DATASET  | Performance (F1-score)   | Disentanglement | Completeness   | Informativeness | act_fn  | norm_fn  | slot_norm | dummy_concept | dummy_tokens | reg_dist | tie_loss   |
|:------------|:-----------:|:-----------:|:--------:|:--------:|:--------:|:----------:|:----------:|:----------:|:-------------:|:---------------:|:---------------:|:---------------:|
| Baseline | E49-SHP | 0.94 ± 0.04 | 0.52 ± 0.11 | 0.47 ± 0.10 | 0.20 ± 0.04  | relu |  - | - | -| - | - | - |
| Baseline | E50-SHP | 0.98 ± 0.01 | 0.55 ± 0.04 | 0.49 ± 0.08 | 0.16 ± 0.01 | sigmoid | - | - | - | - | - | - |
| Baseline | E51-SHP | 0.49 ± 0.09 | 0.45 ± 0.05 | 0.42 ± 0.03 | 0.45 ± 0.04 | gumbel | - | - | - | - | - | - |
| Framework | E52-SHP | 0.97 ± 0.02 | 0.52 ± 0.06 | 0.48 ± 0.05 | 0.16 ± 0.03 | sigmoid | softmax | ✗ | - | - | ✗ | JS |
| Framework | E53-SHP | 0.83 ± 0.08 | 0.66 ± 0.11 | 0.56 ± 0.07 | 0.16 ± 0.03 |  gumbel | softmax | ✗ | - | - | ✗ | JS |
| Framework | E54-SHP | 0.47 ± 0.10 | 0.56 ± 0.08 | 0.52 ± 0.07 | 0.23 ± 0.08 | gumbel | softmax | ✗ | - | -  | ✗ | KL($f$, $c$) |
| Framework | E55-SHP | 0.50 ± 0.09 | 0.57 ± 0.06| 0.51 ± 0.06 | 0.20 ± 0.04 |  gumbel  | softmax | ✗ | - | - | ✗ | KL($c$, $f$) |
| Framework | E56-SHP | 0.83 ± 0.05 | 0.60 ± 0.08 | 0.52 ± 0.07 | 0.16 ± 0.02 |  gumbel  | entmax | ✗ | - | - | ✗ | JS |
| Framework | E57-SHP | 0.54 ± 0.06 | 0.57 ± 0.15 | 0.52 ± 0.09 | 0.25 ± 0.07 | gumbel | softmax | ✓ | ✓ | ✗ | ✗ | JS |
| Framework | E58-SHP | 0.49 ± 0.07 | 0.57 ± 0.12 | 0.48 ± 0.08 | 0.25 ± 0.07 |  gumbel  | softmax | ✓ | ✓ | ✓ | ✗ | JS |
| Framework | E59-SHP | 0.64 ± 0.04 | 0.75 ± 0.03 | 0.75 ± 0.02 | 0.08 ± 0.02 | gumbel | entmax | ✓ | ✓ | ✗ | ✗ | JS |
| Framework | E60-SHP | 0.49 ± 0.02 | 0.61 ± 0.05 | 0.52 ± 0.09 | 0.23 ± 0.04 |  gumbel | entmax | ✓ | ✓ | ✓ | ✗ | JS |
| Framework | E61-SHP | 0.83 ± 0.08 | 0.66 ± 0.11 | 0.56 ± 0.07 | 0.16 ± 0.03 |  gumbel | softmax | ✗ | - | - | ✓ | JS |
| Framework | E62-SHP | 0.83 ± 0.05 | 0.60 ± 0.08 | 0.52 ± 0.07 | 0.16 ± 0.02 | gumbel | entmax | ✗ | - | - | ✓ | JS |
