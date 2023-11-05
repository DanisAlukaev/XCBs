# Cross-Modal Conceptualization in Bottleneck Models

![architecture-overview](./docs/overview.png)

Concept Bottleneck Models (CBMs) assume that training examples (e.g., x-ray images) are annotated with high-level concepts (e.g., types of abnormalities), and perform classification by first predicting the concepts, followed by predicting the label relying on these concepts. The main difficulty in using CBMs comes from having to choose concepts that are predictive of the label and then having to label training examples with these concepts. In our approach, we adopt a more moderate assumption and instead use text descriptions (e.g., radiology reports), accompanying the images in training, to guide the induction of concepts. Our cross-modal approach treats concepts as discrete latent variables and promotes concepts that (1) are predictive of the label, and (2) can be predicted reliably from both the image and text. Through experiments conducted on datasets ranging from synthetic datasets (e.g., synthetic images with generated descriptions) to realistic medical imaging datasets, we demonstrate that cross-modal learning encourages the induction of interpretable concepts while also facilitating disentanglement. Our results also suggest that this guidance leads to increased robustness by suppressing the reliance on shortcut features.


## 🚀 Getting Started
1. Set-up conda environment
    ```
    make set-up-env
    ```
2. Create `.env` file from `.env.example`
3. Set-up experiment registry in clear-ml
    ```
    clearml-init
    ```
4. Download data for Shapes, CUB-200, MSCOCO, MIMIC-CXR
    ```
    make download_data
    ```
5. Pre-process CUB-200
    ```
    make preprocess_cub
    ```
6. Pre-process MIMIC-CXR
    ```
    make preprocess_mimic
    ```
7. You are ready to run your first experiment! Checkpoints and explanations will appear in hydra `outputs/` directory.
    ```
    python main.py dataset.batch_size=64 seed=42 +experiment={XXX}
    ```


## 🤔 FAQ

1. How can I retrieve explanations?
    > Our pipeline saves explanations in `results.json`, which can be found in experiment folder. For visualization you can use [`inspect.ipynb`](./autoconcept/inspect.ipynb) notebook.

2. How can I assess my model in terms of DCI?
    > Measuring disentanglement, completeness, informativeness is moved outside the default pipeline and can be performed via [`metrics.ipynb`](./autoconcept/metrics.ipynb) notebook.


## 🧬 Experiments

Following table helps you to navigate through all experimental setups. The configurations files are located in [this directory](autoconcept/config/conf/). However, some of them are outdated and left to revise our hypotheses later (for clarity we omit them in navigation table below). Feel free to reuse our setups and add new ones [here](autoconcept/config/conf/experiment).

| Model     | EID-DATASET  | pretrained | act_fn  | norm_fn  | slot_norm | dummy_concept | dummy_tokens | reg_dist | tie_loss   |
|:------------|:-----------|:-----------:|:----------|:----------|:----------:|:-------------:|:---------------:|:---------------:|:---------------|
| Baseline | E35-SHP | ✓ | relu | - | - | -| - | - | - |
| Baseline | E36-SHP | ✓ | sigmoid  | - | - | - | - | - | - |
| Baseline | E36-CUB | ✓ | sigmoid  | - | - | - | - | - | - |
| Baseline | E36-MIM | ✓ | sigmoid  | - | - | - | - | - | - |
| Baseline | E36-SHP-NOROBUST | ✓ | sigmoid  | - | - | - | - | - | - |
| Baseline | E37-SHP | ✓ | gumbel  | - | - | - | - | - | - |
| Framework | E38-SHP | ✓ | sigmoid | softmax | ✗ | - | - | ✗ | JS |
| Framework | E39-SHP | ✓ | gumbel | softmax | ✗ | - | - | ✗ | JS |
| Framework | E39-CUB | ✓ | gumbel | softmax | ✗ | - | - | ✗ | JS |
| Framework | E39-MIM | ✓ | gumbel | softmax | ✗ | - | - | ✗ | JS |
| Framework | E39-SHP-NOROBUST | ✓ | gumbel | softmax | ✗ | - | - | ✗ | JS |
| Framework | E40-SHP | ✓| gumbel  | softmax | ✗ | - | -  | ✗ | KL($f$, $c$) |
| Framework | E41-SHP | ✓ | gumbel | softmax | ✗ | - | - | ✗ | KL($c$, $f$) |
| Framework | E42-SHP | ✓ | gumbel | entmax | ✗ | - | - | ✗ | JS |
| Framework | E43-SHP | ✓ | gumbel | softmax | ✓ | ✓ | ✗ | ✗ | JS |
| Framework | E44-SHP | ✓ | gumbel | softmax | ✓ | ✓ | ✓ | ✗ | JS |
| Framework | E45-SHP | ✓ | gumbel | entmax | ✓ | ✓ | ✗ | ✗ | JS |
| Framework | E46-SHP | ✓ | gumbel | entmax | ✓ | ✓ | ✓ | ✗ | JS |
| Framework | E47-SHP | ✓ | gumbel  | softmax | ✗ | - | - | ✓ | JS |
| Framework | E47-MIM | ✓ | gumbel  | softmax | ✗ | - | - | ✓ | JS |
| Framework | E47-SHP-NOISE | ✓ | gumbel  | softmax | ✗ | - | - | ✓ | JS |
| Framework | E47-SHP-REDUNDANCY | ✓ | gumbel  | softmax | ✗ | - | - | ✓ | JS |
| Framework | E48-SHP | ✓ | gumbel  | entmax | ✗ | - | - | ✓ | JS |
| Baseline | E49-SHP | ✗ | relu |  - | - | -| - | - | - |
| Baseline | E50-SHP | ✗ | sigmoid | - | - | - | - | - | - |
| Baseline | E51-SHP | ✗ | gumbel | - | - | - | - | - | - |
| Framework | E52-SHP | ✗ | sigmoid | softmax | ✗ | - | - | ✗ | JS |
| Framework | E53-SHP | ✗ |  gumbel | softmax | ✗ | - | - | ✗ | JS |
| Framework | E54-SHP | ✗ | gumbel | softmax | ✗ | - | -  | ✗ | KL($f$, $c$) |
| Framework | E55-SHP | ✗ |  gumbel  | softmax | ✗ | - | - | ✗ | KL($c$, $f$) |
| Framework | E56-SHP | ✗ |  gumbel  | entmax | ✗ | - | - | ✗ | JS |
| Framework | E57-SHP | ✗ | gumbel | softmax | ✓ | ✓ | ✗ | ✗ | JS |
| Framework | E58-SHP | ✗ |  gumbel  | softmax | ✓ | ✓ | ✓ | ✗ | JS |
| Framework | E59-SHP | ✗ | gumbel | entmax | ✓ | ✓ | ✗ | ✗ | JS |
| Framework | E60-SHP | ✗ |  gumbel | entmax | ✓ | ✓ | ✓ | ✗ | JS |
| Framework | E61-SHP | ✗ |  gumbel | softmax | ✗ | - | - | ✓ | JS |
| Framework | E62-SHP | ✗ | gumbel | entmax | ✗ | - | - | ✓ | JS |

## 📖 Citation

Our research paper was accepted to The 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023), which will take place in December. Until the proceedings are published, you can use this bibtex for [our pre-print](https://arxiv.org/abs/2310.14805)!
```
@article{alukaev2023cross,
  title={Cross-Modal Conceptualization in Bottleneck Models},
  author={Alukaev, Danis and Kiselev, Semen and Pershin, Ilya and Ibragimov, Bulat and Ivanov, Vladimir and Kornaev, Alexey and Titov, Ivan},
  journal={arXiv preprint arXiv:2310.14805},
  year={2023}
}
```
