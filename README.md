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
# CUB-200
make preprocess_cub

# MIMIC-CXR
make preprocess_mimic
```


## Experiments
| EID   | MID | Model name | Description | Task | Runs |
| ----:| :--- | :---  | :----       | :--- |:--- |
| [E01](autoconcept/config/conf/experiment/E01.yaml) | [M01](autoconcept/config/conf/model/M01.yaml) | Unrestricted Bottleneck Model | Visual feature extractor + predictor | CLF CUB200 | [[1]](http://10.100.11.149:8080/projects/747cd2ee35374486acb675187990cf67/experiments/45290810b6594c90bd67599f9a9eb948), [[2]](http://10.100.11.149:8080/projects/747cd2ee35374486acb675187990cf67/experiments/7acaef594d8e4785b0259341ed68d619), [[3]](http://10.100.11.149:8080/projects/747cd2ee35374486acb675187990cf67/experiments/46375d4209234e33abb4c9db98fee285/info-output/metrics/scalar) |
| [E02](autoconcept/config/conf/experiment/E02.yaml) | [M02](autoconcept/config/conf/model/M02.yaml) | Concept Bottleneck Model | Visual feature extractor + predictor | CLF CUB200 | [[1]](http://10.100.11.149:8080/projects/747cd2ee35374486acb675187990cf67/experiments/0651aab925ed46b4bde3574cf523bbd1) |
| [E03](autoconcept/config/conf/experiment/E03.yaml) | [M03](autoconcept/config/conf/model/M03.yaml) | Binary Attributes Classifier | Predictor | CLF CUB200 | [[1]](http://10.100.11.149:8080/projects/747cd2ee35374486acb675187990cf67/experiments/60fddd0c2a0f46f384e597c5e33d1b2e) |
| [E04](autoconcept/config/conf/experiment/E04.yaml) | [M04](autoconcept/config/conf/model/M04.yaml) | BOW Classifier | Predictor | CLF CUB200 | [[1]](http://10.100.11.149:8080/projects/747cd2ee35374486acb675187990cf67/experiments/4f04a00d9b9a418fb69ac44dd77d4fbf) |
| [E05](autoconcept/config/conf/experiment/E05.yaml) | [M05](autoconcept/config/conf/model/M05.yaml) | LSTM Classifier | LSTM concept extractor + predictor | CLF CUB200 | [[1]](http://10.100.11.149:8080/projects/747cd2ee35374486acb675187990cf67/experiments/e1e2454081d04be7aaac8a0b5597c583) |
| [E06](autoconcept/config/conf/experiment/E06.yaml) | [M06](autoconcept/config/conf/model/M06.yaml) | Single CNN Classifier | Single CNN concept extractor + predictor (see issue [#14](https://github.com/DanisAlukaev/AutoConceptBottleneck/issues/14)) | CLF CUB200 | [[1]](http://10.100.11.149:8080/projects/747cd2ee35374486acb675187990cf67/experiments/61fe6ef5e7594f0586d1b814402d173b) |
| [E07](autoconcept/config/conf/experiment/E07.yaml) | [M07](autoconcept/config/conf/model/M07.yaml) | Multiple CNN Classifier | Multiple CNN concept extractor + predictor (see issue [#16](https://github.com/DanisAlukaev/AutoConceptBottleneck/issues/16)) | CLF CUB200 | [[1]](http://10.100.11.149:8080/projects/747cd2ee35374486acb675187990cf67/experiments/06197c1b3a1c4352850bb181dd1bd564) |
| [E08](autoconcept/config/conf/experiment/E08.yaml) | [M08](autoconcept/config/conf/model/M08.yaml) | Simplified Attention Classifier | Simplified Attention concept extractor + predictor (see issue [#20](https://github.com/DanisAlukaev/AutoConceptBottleneck/issues/20)) | CLF CUB200 | [[1]](http://10.100.11.149:8080/projects/747cd2ee35374486acb675187990cf67/experiments/1ac635b4173240928a2a72ea14c497d1) |
| [E09](autoconcept/config/conf/experiment/E09.yaml) | [M09](autoconcept/config/conf/model/M09.yaml) | Transformer Classifier | Transformer concept extractor + predictor (see issue [#22](https://github.com/DanisAlukaev/AutoConceptBottleneck/issues/22)) | CLF CUB200 | [[1]](http://10.100.11.149:8080/projects/747cd2ee35374486acb675187990cf67/experiments/1586fc7eb35b43e9a20123eac86d394d) |
| [E10](autoconcept/config/conf/experiment/E10.yaml) | [M10](autoconcept/config/conf/model/M10.yaml) | Automatic Concept Bottleneck | Visual feature extractor + transformer concept extractor + predictor (see issue [#24](https://github.com/DanisAlukaev/AutoConceptBottleneck/issues/24)) | CLF CUB200 | [[1]](http://10.100.11.149:8080/projects/747cd2ee35374486acb675187990cf67/experiments/b7b4c971f26b4ca0b931c2bd286da8cf) |
| [E10-1](autoconcept/config/conf/experiment/E10-1.yaml) | [M10](autoconcept/config/conf/model/M10.yaml) | Automatic Concept Bottleneck | Visual feature extractor + transformer concept extractor + predictor + freezing callback (see issue [#26](https://github.com/DanisAlukaev/AutoConceptBottleneck/issues/26)) | CLF CUB200 | [[1]](http://10.100.11.149:8080/projects/747cd2ee35374486acb675187990cf67/experiments/6f6609392f404f0080bbfc189ceb62a8) |

## How to run?
```
python main.py dataset.batch_size=64 seed=42 +experiment={XXX}
```
