# Automatic Concept Bottleneck Models

## Getting Started
1. Set-up conda environment
```
conda create --name bottleneck --file requirements.txt python=3.10
```
2. Download public datasets
```
make download_data
```
3. Fill in paths in `.env` file (see `.env.example`)
4. Prerocess datasets
```
make preprocess_data
```
