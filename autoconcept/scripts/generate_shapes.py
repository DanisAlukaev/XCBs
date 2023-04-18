from random import choice

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_position(pt, size=299):
    position = ""
    if pt[1] < size // 2:
        position += "up "
    else:
        position += "bottom "

    if pt[0] < size // 2:
        position += "left"
    else:
        position += "right"

    return position


def create_sample(color, shape, size=299):
    # color: "red", "green", "blue"
    if color == "red":
        color = [255, 0, 0]
    elif color == "green":
        color = [0, 255, 0]
    elif color == "blue":
        color = [0, 0, 255]
    else:
        raise ValueError

    image = 255 * np.ones(shape=[size, size, 3], dtype=np.uint8)
    measure = size // 10
    width = np.random.randint(measure, measure * 4)
    start_pt = np.random.randint(0, size - width - 1, size=2)

    position = None

    if shape == "square":
        end_pt = np.array([pt + width for pt in start_pt])
        image = cv2.rectangle(image, start_pt, end_pt,
                              color=color, thickness=-1)
        position = get_position([pt + width // 2 for pt in start_pt], size)

    elif shape == "triangle":
        pt2 = [start_pt[0] + width, start_pt[1]]
        pt3 = [start_pt[0] + width // 2, start_pt[1] + width]
        triangle_cnt = np.array([start_pt, pt2, pt3])
        image = cv2.drawContours(
            image, [triangle_cnt], 0, color=color, thickness=-1)
        image = np.rot90(np.rot90(image))
        center = [size - (start_pt[0] + width // 2),
                  size - (start_pt[1] + width // 2)]
        position = get_position(center, size)
    elif shape == "circle":
        pt1 = np.array([pt + width // 2 for pt in start_pt])
        image = cv2.circle(image, pt1, width // 2, color=color, thickness=-1)
        position = get_position(pt1, size)
    else:
        raise ValueError

    return image, position


def generate_caption(color, shape, position, easy=False):

    color_synonyms = {
        "red": ["red", "scarlet", "rouge"],
        "green": ["green", "lime", "emerald"],
        "blue": ["blue", "lapis", "cobalt"]
    }

    caption = str()
    caption += "this shape "

    if shape == "circle":
        caption += "is round "
    elif shape == "triangle":
        caption += "has three angle "
    elif shape == "square":
        caption += "has four angle "
    else:
        raise ValueError

    if easy:
        caption += f"{color} color "
    else:
        caption += f"{choice(color_synonyms[color])} color "

    caption += f"{position} position "

    rotation = choice(["rotate ", "unrotate "])
    caption += rotation

    background = choice(["blank", "white"])
    caption += f"{background} canvas"

    return caption


def main():
    shapes = ["square", "triangle", "circle"]
    colors = ["red", "green", "blue"]
    N_SAMPLES = 300

    train_test_split = 0.8
    train_val_split = 0.15

    PATH_SAVE = "/home/danis/Projects/AlphaCaption/AutoConceptBottleneck/data/shapes/"
    IMAGE_PATH = PATH_SAVE + "images/"

    import os
    if not os.path.exists(PATH_SAVE):
        os.makedirs(PATH_SAVE)

    if not os.path.exists(IMAGE_PATH):
        os.makedirs(IMAGE_PATH)

    combinations = list()

    dict_df = {
        "filepath": list(),
        "caption": list(),
        "one_hot": list(),
        "class_id": list(),
        "split": list()
    }

    for shape in shapes:
        for color in colors:
            combinations.append((color, shape))

    for idx, (color, shape) in enumerate(combinations):
        for sample_id in range(N_SAMPLES):
            x, pos = create_sample(color, shape)
            y = np.zeros(len(combinations))
            y[idx] = 1.
            caption = generate_caption(color, shape, pos)
            class_id = idx

            img_name = f"{color}_{shape}_{sample_id}.png"
            filepath = IMAGE_PATH + img_name

            dict_df["filepath"].append(filepath)
            dict_df["class_id"].append(class_id)
            dict_df["caption"].append(caption)
            dict_df["one_hot"].append(list(y))

            if sample_id < train_test_split * (1 - train_val_split) * N_SAMPLES:
                # train
                dict_df["split"].append(0)
            elif sample_id > train_test_split * N_SAMPLES:
                # test
                dict_df["split"].append(2)
            else:
                # val
                dict_df["split"].append(1)

            plt.imsave(filepath, x)

    df = pd.DataFrame.from_dict(dict_df)

    df_path = PATH_SAVE + "captions.csv"
    df.to_csv(df_path)


if __name__ == "__main__":
    main()
