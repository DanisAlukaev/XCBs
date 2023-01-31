import numpy as np
import pandas as pd
from tqdm import tqdm

IMAGE_ATTRIBUTES_PATH = "../data/CUB_200_2011/attributes/image_attribute_labels.txt"
ATTRIBUTES_PATH = "../data/attributes.txt"
IMAGE_NAMES_PATH = "../data/CUB_200_2011/images.txt"
ATTRIBUTES_NUM = 312


def get_attribute_offsets():
    with open(ATTRIBUTES_PATH) as f:
        lines = f.readlines()
    attributes = []
    for line in lines:
        attributes.append(line.split("::")[0].split(" ")[1])
    indexes_a = list()
    prev_a = None
    for idx, a in enumerate(attributes):
        if a != prev_a:
            prev_a = a
            indexes_a.append(idx)
    return indexes_a


def get_image_names():
    images_text = list()
    with open(IMAGE_NAMES_PATH, "r") as f:
        images_text = f.readlines()
    file_names = []
    for val in images_text:
        img = val.split(" ")
        file_names.append(img[1].split(".")[1].split(
            "/")[1]+"."+img[1].split(".")[-1].replace("\n", ""))
    return file_names


def get_attribute_encoding(file_names):
    attributes_text = list()
    with open(IMAGE_ATTRIBUTES_PATH, "r") as f:
        attributes_text = f.readlines()
    attributes_encoding = list()
    image_names = list()
    cur_img_id = -1
    for line in tqdm(attributes_text):
        tokens = line.split(" ")
        img_id = int(tokens[0]) - 1
        attr_id = int(tokens[1]) - 1
        is_present = int(tokens[2])
        image_names.append(file_names[img_id])
        if cur_img_id < img_id:
            attributes_encoding.append([0 for _ in range(ATTRIBUTES_NUM)])
            cur_img_id = img_id
        attributes_encoding[-1][attr_id] = is_present
    return attributes_encoding


def infer_class(x):
    x = x.replace(".jpg", "")
    x = "_".join(x.split("_")[:-2])
    return x


def main():
    image_names = get_image_names()
    attributes_encoding = get_attribute_encoding(image_names)
    attribute_offsets = get_attribute_offsets()
    df = pd.DataFrame(list(zip(image_names, attributes_encoding)), columns=[
                      "image", "attributes"])
    df["class_name"] = df["image"].apply(infer_class)
    class_names = list(set(df.class_name))

    def encode_class(x):
        return class_names.index(x)
    df["class"] = df["class_name"].apply(encode_class)

    def voting(x):
        attributes = np.array(x["attributes"])
        attribute_series = df[df["class"] == x["class"]].attributes
        attributes_np = np.array([m for m in attribute_series.tolist()])
        for i in range(1, len(attribute_offsets)):
            prev_idx, cur_idx = attribute_offsets[i - 1], attribute_offsets[i]
            for idx in range(prev_idx, cur_idx):
                attribute_slice = attributes_np[:, idx]
                share = len(
                    attribute_slice[attribute_slice == 1]) / len(attribute_slice)
                if share > 0.5:
                    attributes[prev_idx: cur_idx] = 0
                    attributes[idx] = 1
                    break

        return list(attributes)

    tqdm.pandas()
    df["attributes_mv"] = df.progress_apply(voting, axis=1)

    attr_flatten = list()
    for a in df.attributes_mv:
        attr_flatten.extend(a)

    with open(IMAGE_ATTRIBUTES_PATH, "r") as f:
        lines = f.readlines()

    new_lines = list()
    for a, line in zip(attr_flatten, lines):
        tokens = line.split()
        tokens[2] = str(a)
        new_lines.append(" ".join(tokens))

    with open(IMAGE_ATTRIBUTES_PATH, "w") as f:
        for new_line in new_lines:
            f.write(f"{new_line}\n")


if __name__ == "__main__":
    main()
