from pathlib import Path

import albumentations as A
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class MimicDataset(Dataset):
    def __init__(
        self,
        annotations_path,
        img_dir,
        transforms=None,
        phase='train',
        debug_sample=None,
    ):
        self.annotations_path = annotations_path
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
        )

        return sample

    def read_annotations_file(self):
        self.annotations = pd.read_csv(self.annotations_path)
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


if __name__ == '__main__':
    annotation_path = "/home/danis/Projects/AlphaCaption/AutoConceptBottleneck/data/mimic-cxr/annotation.csv"
    img_dir = "/home/danis/Projects/AlphaCaption/AutoConceptBottleneck/data/mimic-cxr/images"

    # vocab = VocabularyShapes(annotation_path=annotation_path)
    # collate_fn = CollateIndices(vocabulary=vocab)

    # dm = JointDataModule(299,
    #                      annotation_path=annotation_path,
    #                      collate_fn=collate_fn)

    # dm.setup()

    # training_data = dm.train_dataloader()
    # print('\nFirst iteration of data set: ', next(iter(training_data)), '\n')
    # print('Length of data set: ', len(training_data), '\n')

    # print(training_data["target"])
