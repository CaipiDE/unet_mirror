import os

import numpy as np
import torchvision.transforms as tfs
import albumentations as A

from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from utils import mirrow_extrapolate

resize = tfs.Compose([
    tfs.Resize(564),
    tfs.CenterCrop(564)
])

toTensorNormalizeAndMaxPixel = A.Compose(
    [
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ]
)


class AerialDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = resize(Image.open(img_path).convert("RGB"))
        mask = resize(Image.open(mask_path).convert("L"))

        image = mirrow_extrapolate(image, int(198 / 2))

        image = np.array(image.convert("RGB"))
        mask = np.array(mask, dtype=np.float32)
        mask[mask == 255.0] = 1.0

        augmentations = toTensorNormalizeAndMaxPixel(image=image, mask=mask)
        image = augmentations["image"]
        mask = augmentations["mask"]

        return image, mask


class TestDataSet(Dataset):
    def __init__(self, test_dir):
        self.image_dir = test_dir
        self.images = os.listdir(test_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])

        image = resize(Image.open(img_path).convert("RGB"))
        image = mirrow_extrapolate(image, int(198 / 2))

        image = np.array(image.convert("RGB"))

        augmentations = toTensorNormalizeAndMaxPixel(image=image)
        image = augmentations["image"]

        return image


def get_loaders(train_dir, train_maskdir,val_dir, val_maskdir, batch_size, num_workers=2, pin_memory=True):

    train_ds = AerialDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_ds = AerialDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader


def getTestLoader(test_dir):
    test_ds = TestDataSet(
        test_dir=test_dir
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False
    )

    return test_loader
