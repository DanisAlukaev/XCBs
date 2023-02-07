# Automatic Concept Bottleneck Models
| [📈 ClearML](http://10.100.11.149:8080/projects/747cd2ee35374486acb675187990cf67/experiments) |

## Getting Started
1. Set-up conda environment
```
conda create --name bottleneck --file requirements.txt python=3.10
```
2. Download public datasets
```
make download_data
```
3. Prerocess datasets
```
make preprocess_data
```
## Experiments
| EID   | MID | Model name | Description | Task | Runs |
| ----:| :--- | :---  | :----       | :--- |:--- |
| E01 | M01 | Unrestricted Bottleneck Model | Visual feature extractor + predictor | CLF CUB200 | [[1]](http://10.100.11.149:8080/projects/747cd2ee35374486acb675187990cf67/experiments/45290810b6594c90bd67599f9a9eb948), [[2]](http://10.100.11.149:8080/projects/747cd2ee35374486acb675187990cf67/experiments/7acaef594d8e4785b0259341ed68d619), [[3]](http://10.100.11.149:8080/projects/747cd2ee35374486acb675187990cf67/experiments/46375d4209234e33abb4c9db98fee285/info-output/metrics/scalar) |
| E02 | M02 | Concept Bottleneck Model | Visual feature extractor + predictor | CLF CUB200 | [[1]](http://10.100.11.149:8080/projects/747cd2ee35374486acb675187990cf67/experiments/0651aab925ed46b4bde3574cf523bbd1) |
| E03 | M03 | Binary Attributes Classifier | Predictor | CLF CUB200 | [[1]](http://10.100.11.149:8080/projects/747cd2ee35374486acb675187990cf67/experiments/60fddd0c2a0f46f384e597c5e33d1b2e) |
| E04 | M04 | BOW Classifier | Predictor | CLF CUB200 | [[1]](http://10.100.11.149:8080/projects/747cd2ee35374486acb675187990cf67/experiments/4f04a00d9b9a418fb69ac44dd77d4fbf) |




## How to run?
```
python main.py dataset.batch_size=64 seed=42 +experiment={XXX}
```
