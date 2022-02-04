# Common imports
import numpy as np
from pathlib import Path
import pandas as pd
import cv2

# Torch imports
import torch
from torch.utils.data import Dataset

# Load Path
from Faster_RCNN.parameter import ROOT_DIR, DATA_DIR


class WheatDataset(Dataset):
    """A dataset example for GWC 2021 competition."""
    def __init__(self, transform=None):
        """
        Args:
            transform (callable, optional): Optional data augmentation to be applied
                on a sample.
        """
        self.root_dir = Path(f"{DATA_DIR}/train")
        annotations = pd.read_csv(f"{DATA_DIR}/train.csv")
        self.image_list = annotations["image_name"].values
        self.domain_list = annotations["domain"].values
        self.boxes = [self.decodeString(item) for item in annotations["BoxesString"]]

        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        imgp = str(self.root_dir / (self.image_list[idx] + ".png"))
        domain = self.domain_list[idx]  # We don't use the domain information but you could !
        bboxes = self.boxes[idx]
        img = cv2.imread(imgp)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Opencv open images in BGR mode by default

        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=["wheat_head"] * len(
                bboxes))  # Albumentations can transform images and boxes
            image = transformed["image"]
            bboxes = transformed["bboxes"]

        if len(bboxes) > 0:
            bboxes = torch.stack([torch.tensor(item) for item in bboxes])
        else:
            bboxes = torch.zeros((0, 4))
        return image, bboxes, domain

    def decodeString(self, BoxesString):
        """
        Small method to decode the BoxesString
        """
        if BoxesString == "no_box":
            return np.zeros((0, 4))
        else:
            try:
                boxes = np.array([np.array([int(i) for i in box.split(" ")])
                                  for box in BoxesString.split(";")])
                return boxes
            except:
                print(BoxesString)
                print("Submission is not well formatted. empty boxes will be returned")
                return np.zeros((0, 4))


def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """

    images = list()
    targets=list()
    metadatas = list()

    for i, t, m in batch:
        images.append(i)
        targets.append(t)
        metadatas.append(m)
    images = torch.stack(images, dim=0)

    return images, targets, metadatas


class WheatDatasetPredict(Dataset):
    """A dataset example for GWC 2021 competition."""

    def __init__(self, transform):
        """
        Args:
             transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(f'{ROOT_DIR}/Faster_RCNN/data/test')
        annotations = pd.read_csv(f'{DATA_DIR}/submission.csv')

        self.image_list = annotations["image_name"].values
        self.domain_list = annotations["domain"].values
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        imgp = str(self.root_dir / (self.image_list[idx] + ".png"))
        domain = self.domain_list[idx]
        img = cv2.imread(imgp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=img)
            image = transformed["image"]
        return image, img, self.image_list[idx], domain