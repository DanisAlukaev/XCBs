from pathlib import Path
from typing import List, Optional, Union

import albumentations as A
import cv2
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch.transforms import ToTensorV2
from datasets.collators import CollateEmulator
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class JointDataset(Dataset):

    def __init__(
        self,
        annotations_file: Union[str, Path] = "data/captions_merged.csv",
        img_dir: Union[str, Path] = "data/merged_files",
        CUB_dir: Union[str, Path] = "data/CUB_200_2011",
        transforms: List[Union[ImageOnlyTransform, nn.Module]] = None,
        phase: Optional[str] = 'train',
        debug_sample: Optional[int] = None,
        mix_with_mscoco: Optional[bool] = True,
    ) -> None:
        """Constructor of joint CUB & MSCOCO dataset.

        CUB: https://www.vision.caltech.edu/datasets/cub_200_2011/
        MSCOCO: https://cocodataset.org/#home

        Can be set-up to work only with CUB dataset (for easier training).

        :param annotations_file: path to .csv file with annotations.
        :param img_dir: path to directory with dataset of merged images.
        :param CUB_dir: path to directory with CUB data.
        :param transforms: list of albumentation / torchvision trasforms.
        :param phase: either 'train', 'val', 'test'.
        :param debug_sample: index of sample to use as a debug batch.
        :param mix_with_mscoco: bool whether to join CUB and MSCOCO.
        """
        self.annotations_file = hydra.utils.get_original_cwd() / annotations_file
        self.img_dir = hydra.utils.get_original_cwd() / Path(img_dir)
        self.CUB_dir = hydra.utils.get_original_cwd() / Path(CUB_dir)
        self.transforms = transforms
        self.phase = phase
        self.mix_with_mscoco = mix_with_mscoco

        self.read_images_txt()
        self.read_classes_txt()
        self.read_image_class_labels_txt()
        self.read_train_test_split_txt()
        self.read_annotations_file()

        if debug_sample is not None:
            self.annotations = self.annotations.iloc[:debug_sample]

        if self.transforms:
            self.transforms = A.Compose(transforms,)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        row = self.annotations.iloc[idx]

        img_path = self.CUB_dir / 'images' / row.path
        if self.mix_with_mscoco:
            img_path = self.img_dir / row.path
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        mask_captions = row.mask_source_captions
        source_captions = row.source_captions
        mask_captions = [str(l)for l in eval(mask_captions)]
        source_captions = [str(c) for c in eval(source_captions)]

        captions_zip = zip(source_captions, mask_captions)
        captions_cub = [c for c, l in captions_zip if l == 'cub']
        captions_mscoco = [c for c, l in captions_zip if l == 'coco']

        mask_cub = ['cub'] * len(captions_cub)
        mask_mscoco = ['coco'] * len(captions_mscoco)
        if self.mix_with_mscoco:
            source_captions = captions_mscoco + captions_cub
            mask_captions = mask_mscoco + mask_cub
        else:
            source_captions = captions_cub
            mask_captions = mask_cub

        caption = " ".join(source_captions)
        attributes = torch.tensor(eval(row.attributes)).float()
        class_id = self.image_id2class_id[row.name]
        target_one_hot = torch.zeros(len(self.class_id2class_name))
        target_one_hot[class_id - 1] = 1
        target = torch.tensor(class_id - 1)

        sample = dict(
            image=image,
            img_path=str(img_path),
            report=caption,
            target_one_hot=target_one_hot,
            attributes=attributes,
            class_id=class_id,
            class_name=self.class_id2class_name[class_id],
            source_captions=source_captions,
            mask_source_captions=mask_captions,
            target=target,
        )
        self.dict_to_float32(sample)
        sample['target'] = target
        return sample

    @staticmethod
    def dict_to_float32(d) -> None:

        def array_to_float32(a):
            if isinstance(a, np.ndarray):
                return a.astype(np.float32)
            elif isinstance(a, torch.Tensor):
                return a.float()
            else:
                return a

        for k, v in d.items():
            d[k] = array_to_float32(v)

    def read_images_txt(self) -> None:
        filename = self.CUB_dir / 'images.txt'
        df_images = pd.read_csv(filename, sep=' ', names=['image_id', 'path'])
        df_images['filename'] = df_images.path.apply(lambda s: s.split('/')[1])
        self.image_id2filename = df_images.set_index(
            'image_id').filename.to_dict()
        self.image_id2path = df_images.set_index('image_id').path.to_dict()
        self.filename2image_id = df_images.set_index(
            'filename').image_id.to_dict()
        self.df_images = df_images.set_index('image_id')

    def read_annotations_file(self):
        filename = self.annotations_file
        self.annotations = pd.read_csv(
            filename, names=['filename', 'source_captions', 'mask_source_captions', 'attributes'])
        self.annotations['image_id'] = self.annotations.filename.apply(
            lambda s: self.filename2image_id[s])
        self.annotations['path'] = self.annotations.image_id.apply(
            lambda i: self.image_id2path[i])
        self.annotations = self.annotations.set_index('image_id')
        self.annotations = self.annotations.loc[self.get_set_of_img_ids_by_phase(
            self.phase)]

    def read_classes_txt(self):
        filename = self.CUB_dir / 'classes.txt'
        df = pd.read_csv(filename, sep=' ', names=['class_id', 'class_name'])
        self.class_name2class_id = df.set_index(
            'class_name').class_id.to_dict()
        self.class_id2class_name = df.set_index(
            'class_id').class_name.to_dict()
        self.f_classes = df

    def read_image_class_labels_txt(self):
        filename = self.CUB_dir / 'image_class_labels.txt'
        df_classes = pd.read_csv(filename, sep=' ', names=[
                                 'image_id', 'class_id'])
        self.image_id2class_id = df_classes.set_index(
            'image_id').class_id.to_dict()

    def get_filename_by_img_id(self, img_id):
        return self.df_images.loc[img_id, 'filename']

    def get_set_of_img_ids_by_phase(self, phase):
        phase2num = dict(train=1, test=0, val=2)
        return self.df_train_test_split[self.df_train_test_split.is_training_image == phase2num[phase]].index.values

    def read_train_test_split_txt(self, test_size=0.2):
        filename = self.CUB_dir / 'train_test_split.txt'
        df = pd.read_csv(
            filename,
            sep=' ',
            names=['image_id', 'is_training_image']
        )
        df['class_id'] = df.image_id.apply(lambda x: self.image_id2class_id[x])
        self.df_train_test_split = df.set_index('image_id')

        df_train = self.df_train_test_split.loc[self.get_set_of_img_ids_by_phase(
            'train')]

        _, df_val = train_test_split(
            df_train, test_size=test_size, random_state=42, stratify=df_train.class_id)

        self.df_train_test_split.loc[df_val.index, 'is_training_image'] = 2


class JointDataModule(LightningDataModule):
    def __init__(
        self,
        img_size,
        augmentations_train=None,
        augmentations_test=None,
        annotation_path="data/captions_merged.csv",
        img_root="data/merged_files/",
        CUB_dir="data/CUB_200_2011",
        debug_sample=None,
        batch_size=64,
        num_workers=4,
        collate_fn=None,
        shuffle_train=True,
        mix_with_mscoco=False,
        use_val_for_train=False
    ):
        super().__init__()
        self.img_size = img_size
        self.annotation_path = hydra.utils.get_original_cwd() / Path(annotation_path)
        self.img_root = hydra.utils.get_original_cwd() / Path(img_root)
        self.CUB_dir = hydra.utils.get_original_cwd() / Path(CUB_dir)
        self.debug_sample = debug_sample
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.mix_with_mscoco = mix_with_mscoco
        self.use_val_for_train = use_val_for_train

        self.dataset_kwargs = dict(
            annotations_file=self.annotation_path,
            img_dir=self.img_root,
            CUB_dir=self.CUB_dir,
            debug_sample=self.debug_sample,
            mix_with_mscoco=mix_with_mscoco,
        )

        self.dataloader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        if collate_fn is not None:
            self.dataloader_kwargs['collate_fn'] = collate_fn

        # default augmentations were taken from the paper "Concept Bottleneck Models"
        self.pre_transforms = []
        self.augmentations_train = augmentations_train or [
            A.augmentations.transforms.ColorJitter(
                brightness=32/255, saturation=(0.5, 1.5)),
            A.augmentations.crops.transforms.RandomResizedCrop(
                self.img_size, self.img_size),
            A.HorizontalFlip()
        ]
        self.augmentations_test = augmentations_test or [
            A.PadIfNeeded(min_height=self.img_size,
                          min_width=self.img_size, p=1),
            A.CenterCrop(self.img_size, self.img_size, always_apply=True, p=1),
        ]
        self.post_transforms = [A.Normalize(), ToTensorV2()]

    def prepare_data_per_node(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = JointDataset(
                phase='train',
                transforms=self.pre_transforms + self.augmentations_train + self.post_transforms,
                **self.dataset_kwargs
            )

            augmentations_val = self.augmentations_train if self.use_val_for_train else self.augmentations_test
            self.val_dataset = JointDataset(
                phase='val',
                transforms=self.pre_transforms + augmentations_val + self.post_transforms,
                **self.dataset_kwargs
            )

        if stage == "test" or stage is None:
            self.test_dataset = JointDataset(
                phase='test',
                transforms=self.pre_transforms + self.augmentations_test + self.post_transforms,
                **self.dataset_kwargs
            )

    def train_dataloader(self):
        train_dataset = self.train_dataset
        if self.use_val_for_train:
            train_dataset = torch.utils.data.ConcatDataset(
                [self.train_dataset, self.val_dataset])
        return DataLoader(train_dataset, shuffle=self.shuffle_train, **self.dataloader_kwargs,
                          pin_memory=True)

    def val_dataloader(self):
        val_dataset = self.val_dataset
        if self.use_val_for_train:
            val_dataset = torch.utils.data.Subset(
                self.test_dataset, list(range(0, len(self.val_dataset))))
        return DataLoader(val_dataset, **self.dataloader_kwargs,
                          pin_memory=True)

    def test_dataloader(self):
        test_dataset = self.test_dataset
        if self.use_val_for_train:
            test_dataset = torch.utils.data.Subset(self.test_dataset, list(
                range(len(self.val_dataset), len(self.test_dataset))))
        return DataLoader(test_dataset, **self.dataloader_kwargs,
                          pin_memory=True)


if __name__ == '__main__':
    img_root = "data/merged_files"
    annotation_path = "data/captions_merged.csv"
    CUB_dir = 'data/CUB_200_2011'

    collate_fn = CollateEmulator()

    dm = JointDataModule(299,
                         img_root=img_root,
                         annotation_path=annotation_path,
                         CUB_dir=CUB_dir,
                         collate_fn=collate_fn)
    dm.setup()

    training_data = dm.train_dataloader()
    print('\nFirst iteration of data set: ', next(iter(training_data)), '\n')
    print('Length of data set: ', len(training_data), '\n')
