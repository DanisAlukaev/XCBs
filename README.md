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
| ID   | Model | Description |
| ----:| :---  | :----       |
| E01 | Unrestricted Bottleneck Model | Visual feature extractor + predictor |
