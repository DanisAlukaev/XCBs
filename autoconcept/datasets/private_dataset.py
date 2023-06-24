import albumentations as A
# from data.utils import CollateZoo
import numpy as np
import pytorch_lightning as pl
from dataloader_factory import DataloaderFactory
from datasets.collators import CollateIndices
from datasets.utils import VocabularyMimic
from torch.utils.data import DataLoader, Subset


class GrayToRGB:
    """Implementation of Albumentations-integrable grayscale to RGB augmentation."""

    def __init__(self, layout="CWH"):
        if layout not in ["CWH", "WHC"]:
            raise ValueError(f"Layout is either 'CWH' or 'WHC'.")
        self.layout = layout

    def __call__(self, image, bbox=None, mask=None, keypoints=None, force_apply=False, *args, **kwargs):
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Expected type 'numpy.ndarray', got {type(image).__name__}.")
        aug_dict = {'image': image, 'bbox': bbox,
                    'mask': mask, 'keypoints': keypoints}
        n_dim = len(image.shape)
        if n_dim < 2 or n_dim > 3:
            raise ValueError(
                f"Expected 2D or 3D array, got {n_dim}D array instead.")
        elif n_dim == 2:
            new_image = np.stack([image] * 3, axis=0)
            if self.layout == "WHC":
                new_image = new_image.transpose((1, 2, 0))
            aug_dict['image'] = new_image
        elif n_dim == 3:
            channels_dim = np.argmin(image.shape)
            n_channels = image.shape[channels_dim]
            if image.shape[channels_dim] != 1:
                raise ValueError(
                    f"Expected 1 channel, got {n_channels} instead.")
            new_image = np.concatenate([image] * 3, axis=channels_dim)
            channels_pos = self.layout.find('C')
            if channels_dim != channels_pos:
                new_image = np.moveaxis(new_image, channels_dim, channels_pos)
            aug_dict['image'] = new_image
        return aug_dict


class MedicalDataModule(pl.LightningDataModule):

    def __init__(
        self,
        img_size,
        augmentations=None,
        group_list="report",
        batch_size=32,
        num_workers=4,
        collate_fn=None,
        shuffle_train=True,
        debug_sample=None
    ):
        super().__init__()
        self.img_size = img_size
        self.augmentations = augmentations or []
        self.group_list = group_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.shuffle_train = shuffle_train
        self.debug_sample = debug_sample

        self.pre_transforms = A.Compose([
            A.Resize(img_size, img_size),
            GrayToRGB()
        ])

        self.factory_kwargs = dict(
            transforms=self.pre_transforms,
            augmentations=self.augmentations
        )

        self.dataloader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

        if collate_fn is not None:
            self.dataloader_kwargs['collate_fn'] = collate_fn

    def setup(self, stage=None):
        factory = DataloaderFactory(**self.factory_kwargs)

        def get_subset(x): return Subset(
            x, list(range(0, min(len(x), self.debug_sample))))

        if stage == "fit" or stage is None:
            self.train_dataset = factory.create_dataset(
                phase="train", group_list=self.group_list)
            self.val_dataset = factory.create_dataset(
                phase="val", group_list=self.group_list)

            if self.debug_sample:
                self.train_dataset = get_subset(self.train_dataset)
                self.val_dataset = get_subset(self.val_dataset)

        if stage == "test" or stage is None:
            self.test_dataset = factory.create_dataset(
                phase="test", group_list=self.group_list)

            if self.debug_sample:
                self.test_dataset = get_subset(self.test_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=self.shuffle_train, **self.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.dataloader_kwargs)


if __name__ == '__main__':

    annotation_path = "/home/danis/Projects/AlphaCaption/AutoConceptBottleneck/autoconcept/datasets/annotations_public.csv"
    vocab = VocabularyMimic(annotation_path=annotation_path)
    collate_fn = CollateIndices(vocabulary=vocab)

    dm = MedicalDataModule(
        299,
        batch_size=64,
        collate_fn=collate_fn
    )
    dm.setup()

    train_dataloader = dm.train_dataloader()
    # # val_dataloader = dm.val_dataloader()
    # # test_dataloader = dm.test_dataloader()

    # reports = list()
    # for i in tqdm(range(len(train_dataloader.dataset))):
    #     sample = train_dataloader.dataset[i]
    #     report = sample['report']
    #     reports.append(report)

    # ann = pd.DataFrame.from_dict(dict(caption=reports))
    # ann.to_csv("./annotations_public.csv")

    # print(train_dataloader.dataset[0])

    # print()
    # print('Length of train dataset: ', len(train_dataloader.dataset))
    # print('Length of val dataset: ', len(val_dataloader.dataset))
    # print('Length of test dataset: ', len(test_dataloader.dataset), '\n')

    print('\nFirst iteration of train dataset: ',
          next(iter(train_dataloader)), '\n')
