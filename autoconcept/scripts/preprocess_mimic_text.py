import pandas as pd
from datasets.utils import Preprocess
from tqdm import tqdm

MIMIC_ANNOTATION_PATH = "data/mimic-cxr/annotation.csv"


def main():
    annotations = pd.read_csv(MIMIC_ANNOTATION_PATH)
    preprocess_obj = Preprocess()

    processed = list()
    for _, row in tqdm(annotations.iterrows(), total=annotations.shape[0]):
        text = row.caption
        new_text = preprocess_obj(text)
        processed.append(new_text)

    annotations.caption = processed
    annotations.to_csv(MIMIC_ANNOTATION_PATH)


if __name__ == "__main__":
    main()
