
import os
import random
from pathlib import Path

import albumentations as A
import cv2
import hydra
import numpy
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2
from datasets.collators import CollateIndices
from datasets.utils import VocabularyShapes
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

attribute_mapping = {
    0: [1, 0, 0, 1, 0, 0,  # 1, 0, 0, 0, 0, 0, 0, 0, 0
        ],
    1: [0, 1, 0, 1, 0, 0,  # 0, 1, 0, 0, 0, 0, 0, 0, 0
        ],
    2: [0, 0, 1, 1, 0, 0,  # 0, 0, 1, 0, 0, 0, 0, 0, 0
        ],
    3: [1, 0, 0, 0, 1, 0,  # 0, 0, 0, 1, 0, 0, 0, 0, 0
        ],
    4: [0, 1, 0, 0, 1, 0,  # 0, 0, 0, 0, 1, 0, 0, 0, 0
        ],
    5: [0, 0, 1, 0, 1, 0,  # 0, 0, 0, 0, 0, 1, 0, 0, 0
        ],
    6: [1, 0, 0, 0, 0, 1,  # 0, 0, 0, 0, 0, 0, 1, 0, 0
        ],
    7: [0, 1, 0, 0, 0, 1,  # 0, 0, 0, 0, 0, 0, 0, 1, 0
        ],
    8: [0, 0, 1, 0, 0, 1,  # 0, 0, 0, 0, 0, 0, 0, 0, 1
        ],
}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


class JointDataset(Dataset):

    def __init__(
        self,
        annotations_file,
        transforms=None,
        phase='train',
        debug_sample=None,
    ):
        self.annotations_file = annotations_file
        self.phase = phase
        self.transforms = transforms

        self.read_annotations_file()

        if debug_sample is not None:
            self.annotations = self.annotations.sample(n=debug_sample)

        if self.transforms:
            self.transforms = A.Compose(transforms,)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        try:
            img_path = hydra.utils.get_original_cwd() / Path(row.filepath)
        except:
            img_path = os.getcwd() / Path(row.filepath)
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        caption = row.caption

        target = torch.tensor(row.class_id)
        target_one_hot = torch.tensor(eval(row.one_hot))

        sample = dict(
            image=image,
            img_path=str(img_path),
            report=caption,
            target_one_hot=target_one_hot,
            attributes=attribute_mapping[row.class_id]
        )
        self.dict_to_float32(sample)
        sample['target'] = target
        return sample

    @staticmethod
    def dict_to_float32(d):
        def array_to_float32(a):
            if isinstance(a, np.ndarray):
                return a.astype(np.float32)
            elif isinstance(a, torch.Tensor):
                return a.float()
            else:
                return a

        for k, v in d.items():
            d[k] = array_to_float32(v)

    def read_annotations_file(self):
        filename = self.annotations_file
        self.annotations = pd.read_csv(filename)
        phases = {"train": 0, "val": 1, "test": 2}
        self.annotations = self.annotations[self.annotations.split ==
                                            phases[self.phase]]


class JointDataModule(LightningDataModule):
    def __init__(
        self,
        img_size,
        augmentations_train=None,
        augmentations_test=None,
        annotation_path="data/shapes/captions.csv",
        debug_sample=None,
        batch_size=64,
        num_workers=4,
        collate_fn=None,
        shuffle_train=True,
        use_val_for_train=False
    ):
        super().__init__()

        self.img_size = img_size
        self.annotation_path = Path(annotation_path)
        try:
            self.annotation_path = hydra.utils.get_original_cwd() / self.annotation_path
        except:
            self.annotation_path = os.getcwd() / self.annotation_path
        self.debug_sample = debug_sample
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.use_val_for_train = use_val_for_train

        self.dataset_kwargs = dict(
            annotations_file=self.annotation_path,
            debug_sample=self.debug_sample,
        )

        self.dataloader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        if collate_fn is not None:
            self.dataloader_kwargs['collate_fn'] = collate_fn

        # default augmentations were taken from the paper "Concept Bottleneck Models"
        self.pre_transforms = []
        self.post_transforms = [A.Normalize(), ToTensorV2()]

    def prepare_data_per_node(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = JointDataset(
                phase='train',
                transforms=self.pre_transforms + self.post_transforms,
                **self.dataset_kwargs
            )

            self.val_dataset = JointDataset(
                phase='val',
                transforms=self.pre_transforms + self.post_transforms,
                **self.dataset_kwargs
            )

        if stage == "test" or stage is None:
            self.test_dataset = JointDataset(
                phase='test',
                transforms=self.pre_transforms + self.post_transforms,
                **self.dataset_kwargs
            )

    def train_dataloader(self):
        train_dataset = self.train_dataset
        g = torch.Generator()
        g.manual_seed(0)
        return DataLoader(
            train_dataset,
            shuffle=self.shuffle_train,
            **self.dataloader_kwargs,
            # pin_memory=True,
            # generator=g,
            # worker_init_fn=seed_worker,
        )

    def val_dataloader(self):
        val_dataset = self.val_dataset
        g = torch.Generator()
        g.manual_seed(0)
        return DataLoader(
            val_dataset,
            **self.dataloader_kwargs,
            # pin_memory=True,
            # generator=g,
            # worker_init_fn=seed_worker,
        )

    def test_dataloader(self):
        test_dataset = self.test_dataset
        g = torch.Generator()
        g.manual_seed(0)
        return DataLoader(
            test_dataset,
            **self.dataloader_kwargs,
            # pin_memory=True,
            # generator=g,
            # worker_init_fn=seed_worker,
        )


if __name__ == '__main__':
    annotation_path = "data/shapes/captions.csv"

    vocab = VocabularyShapes(annotation_path=annotation_path)
    collate_fn = CollateIndices(vocabulary=vocab)

    dm = JointDataModule(299,
                         annotation_path=annotation_path,
                         collate_fn=collate_fn)

    dm.setup()

    training_data = dm.train_dataloader()
    print('\nFirst iteration of data set: ', next(iter(training_data)), '\n')
    print('Length of data set: ', len(training_data), '\n')

    print(training_data["target"])
