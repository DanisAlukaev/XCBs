from pathlib import Path

import albumentations as A
import cv2
import hydra
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2
from datasets.collators import CollateIndices
from datasets.utils import VocabularyMimic
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class MimicDataset(Dataset):
    def __init__(
        self,
        annotation_path=None,
        img_dir=None,
        transforms=None,
        phase='train',
        debug_sample=None,
    ):
        self.annotation_path = annotation_path
        self.phase = phase
        self.transforms = transforms
        self.img_dir = Path(img_dir)

        self.read_annotations_file()

        if debug_sample is not None:
            self.annotations = self.annotations.sample(n=debug_sample)

        if self.transforms:
            self.transforms = A.Compose(transforms,)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        img_path = self.img_dir / f"{row.dicom_id}.jpg"
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        report = row.caption
        target = torch.tensor(row.is_pathology)
        attributes = self.get_attributes(idx)

        sample = dict(
            image=image,
            img_path=str(img_path),
            report=report,
            target_one_hot=target,
            attributes=attributes,
            target=target,
        )

        return sample

    def read_annotations_file(self):
        self.annotations = pd.read_csv(self.annotation_path)
        phases = {"train": 0, "val": 1, "test": 2}
        self.annotations = self.annotations[self.annotations.split ==
                                            phases[self.phase]]

    def get_attributes(self, idx):
        row = self.annotations.iloc[idx]

        attribute_cols = ['is_erect', 'atelectasis', 'cardiomegaly', 'consolidation',
                          'edema', 'enlarged_cardiomediastinum', 'fracture', 'lung_lesion',
                          'lung_opacity', 'pleural_effusion', 'pleural_other', 'pneumonia',
                          'pneumothorax', 'support_devices']

        attributes = torch.tensor(list(row[attribute_cols].to_list())).float()

        return attributes


class MimicDataModule(LightningDataModule):
    def __init__(
        self,
        img_size,
        img_dir="data/images/",
        annotation_path="data/mimic-cxr/annotation.csv",
        debug_sample=None,
        batch_size=64,
        num_workers=4,
        collate_fn=None,
        shuffle_train=True,
        use_val_for_train=False
    ):
        super().__init__()
        self.img_size = img_size
        self.img_dir = hydra.utils.get_original_cwd() / Path(img_dir)
        self.annotation_path = hydra.utils.get_original_cwd() / Path(annotation_path)
        self.debug_sample = debug_sample
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.use_val_for_train = use_val_for_train

        self.dataset_kwargs = dict(
            annotation_path=self.annotation_path,
            img_dir=self.img_dir,
            debug_sample=self.debug_sample,
        )

        self.dataloader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        if collate_fn is not None:
            self.dataloader_kwargs['collate_fn'] = collate_fn

        # default augmentations were taken from the paper "Concept Bottleneck Models"
        self.pre_transforms = [
            # A.augmentations.geometric.rotate.Rotate(limit=45),
            # A.augmentations.geometric.transforms.Affine(scale=(0, 10), translate_percent=(0, 0.15)),

        ]
        self.post_transforms = [A.augmentations.geometric.resize.Resize(
            self.img_size, self.img_size), A.Normalize(), ToTensorV2()]

    def prepare_data_per_node(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = MimicDataset(
                phase='train',
                transforms=self.pre_transforms + self.post_transforms,
                **self.dataset_kwargs
            )

            self.val_dataset = MimicDataset(
                phase='val',
                transforms=self.post_transforms,
                **self.dataset_kwargs
            )

        if stage == "test" or stage is None:
            self.test_dataset = MimicDataset(
                phase='test',
                transforms=self.post_transforms,
                **self.dataset_kwargs
            )

    def train_dataloader(self):
        train_dataset = self.train_dataset
        return DataLoader(
            train_dataset,
            shuffle=self.shuffle_train,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        val_dataset = self.val_dataset
        return DataLoader(
            val_dataset,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        test_dataset = self.test_dataset
        return DataLoader(
            test_dataset,
            **self.dataloader_kwargs,
        )


if __name__ == '__main__':
    annotation_path = "data/mimic-cxr/annotation.csv"
    img_dir = "data/mimic-cxr/images"

    vocab = VocabularyMimic(annotation_path=annotation_path)
    collate_fn = CollateIndices(vocabulary=vocab)

    dm = MimicDataModule(
        img_size=299,
        annotation_path=annotation_path,
        img_dir=img_dir,
        collate_fn=collate_fn
    )

    dm.setup()

    training_data = dm.train_dataloader()
    print('\nFirst iteration of data set: ', next(iter(training_data)), '\n')
    print('Length of dataloader: ', len(training_data), '\n')
