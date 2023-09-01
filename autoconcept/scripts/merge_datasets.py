import json
import os

import pandas as pd
import torchvision.datasets as dset
from datasets.utils import Preprocess
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

CUB200_IMG_PATH = "data/CUB_200_2011/images/"
CUB200_IMG_TXT = "data/CUB_200_2011/images.txt"
CUB200_ATTR_TXT = "data/attributes.txt"
CUB200_ATTR_LABELS = "data/CUB_200_2011/attributes/image_attribute_labels.txt"

COCO_IMG_PATH = "data/train2017"
COCO_CAPTIONS = "data/annotations/captions_train2017.json"

MERGED_PATH = "data/merged_files/"
CAPTIONS_MERGED_CSV = "data/captions_merged.csv"
CAPTIONS_MERGED_TXT = "data/captions_merged.txt"


def start_merging() -> None:
    pre_process_obj = Preprocess()
    coco_train = dset.CocoDetection(root=COCO_IMG_PATH,
                                    annFile=COCO_CAPTIONS)

    files = []
    with open(CUB200_IMG_TXT, "r") as f:
        files = f.readlines()

    attributs_baseline_text = []
    with open(CUB200_ATTR_TXT, "r") as f:
        attributs_baseline_text = f.readlines()

    image_attributes_text = []
    with open(CUB200_ATTR_LABELS, "r") as f:
        image_attributes_text = f.readlines()

    images_text = []
    with open(CUB200_IMG_TXT, "r") as f:
        images_text = f.readlines()

    images_text_filtered = []
    file_names = []
    for val in tqdm(images_text):
        img = val.split(" ")

        # uncomment to use species of the bird in caption
        # caption = "This "+img[1].split(".")[1].split("/")[0].replace("_", " ")
        caption = "This bird"

        file_name = img[1].split(".")[1].split(
            "/")[1]+"."+img[1].split(".")[-1].replace("\n", "")
        file_names.append(file_name)
        images_text_filtered.append(caption)

    certainities_text_filtered = [
        "not visibily", "maybe", "probably", "definitely"]

    filterd_attributes_text = []
    for atrib in tqdm(attributs_baseline_text):
        if "150 has_bill_length::about_the_same_as_head" in atrib:
            atrib = "150 has_bill::medium"
        if "151 has_bill_length::longer_than_head" in atrib:
            atrib = "151 has_bill::long"
        if "152 has_bill_length::shorter_than_head" in atrib:
            atrib = "152 has_bill::short"
        atrib = atrib.replace("bill", "beak")
        atri = atrib.split(" ")[1]
        parts = atri.split("::")
        first_part = parts[0].split("_")
        second_part = parts[1].replace("_", " ").replace("\n", "") + " "
        sentence = first_part[0] + " " + second_part
        p = " ".join(first_part[1:])
        p = p.replace('color', '')
        if p != "wing shape" and p != "tail shape":
            sentence += p
        filterd_attributes_text.append(sentence)

    def get_concat_v(im1, im2):
        dst = Image.new(
            'RGB', (im1.width + im2.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, im1.height))
        return dst

    def start_merging_images():
        csv_coco = dict()
        coco_list = []
        for id in tqdm(range(len(files))):
            cub_file_name = files[id].split(" ")[1].replace("\n", "")
            cub_imagepath = CUB200_IMG_PATH + cub_file_name
            cub_image = Image.open(cub_imagepath)
            coco_image = coco_train[id][0]
            image_name = file_names[id]
            merged_path = MERGED_PATH + cub_file_name
            folder_path = os.path.dirname(os.path.abspath(merged_path))
            isExist = os.path.exists(folder_path)

            if not isExist:
                os.makedirs(folder_path)

            get_concat_v(coco_image, cub_image).save(merged_path)
            if image_name not in csv_coco:
                csv_coco[image_name] = [image_name]

            captions = []
            tags = []

            for _, captions_dicts in enumerate(coco_train[id][1]):
                processed_caption = pre_process_obj(captions_dicts['caption'])
                captions.append(processed_caption)
                tags.append("coco")

            csv_coco[image_name].append(captions)
            csv_coco[image_name].append(tags)
            coco_list.append([image_name, captions, tags, []])

        return csv_coco, coco_list

    coco_dict, coco_lst = start_merging_images()

    cur_im_id = -1
    attributes_one_hot = list()
    for text in tqdm(image_attributes_text):
        text_split = text.split(" ")
        im_id = int(text_split[0]) - 1
        attribute_id = int(text_split[1]) - 1
        is_present = int(text_split[2])
        certainity_id = int(text_split[3]) - 1

        if cur_im_id < im_id:
            attributes_one_hot.append([])
            cur_im_id = im_id
        attributes_one_hot[-1].append(is_present)

        if is_present == 1:
            caption = images_text_filtered[im_id] + " " + \
                certainities_text_filtered[certainity_id] + \
                " " + filterd_attributes_text[attribute_id]

            caption = pre_process_obj(caption)
            image_name = file_names[im_id]

            if caption not in coco_lst[im_id][1]:
                coco_lst[im_id][1].extend([caption])
                coco_lst[im_id][2].extend(["cub"])
            if image_name not in coco_dict:
                coco_dict[image_name] = [image_name, [], []]
            if caption not in coco_dict[image_name][1]:
                coco_dict[image_name][1].append(caption)
                coco_dict[image_name][2].append("cub")

    for idx in range(len(coco_lst)):
        coco_lst[idx][3] = attributes_one_hot[idx]

    my_df = pd.DataFrame(coco_lst)
    my_df.to_csv(CAPTIONS_MERGED_CSV, index=False, header=False)
    with open(CAPTIONS_MERGED_TXT, 'w') as convert_file:
        convert_file.write(json.dumps(coco_dict))


if __name__ == '__main__':
    start_merging()
