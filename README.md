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

## How to run?
```
python main.py dataset.batch_size=64 seed=42 +experiment={XXX}
```
