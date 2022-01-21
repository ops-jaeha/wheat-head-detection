# Import Library
import os

# Torch imports
import torch
from torch.utils.data import Dataset

# Import Python File
from Faster_RCNN.transform import train_transform
from Faster_RCNN.dataset import WheatDataset, collate_fn
from Faster_RCNN.model.faster_rcnn import FasterRCNN
from Faster_RCNN.parameter import ROOT_DIR

# Pytorch import
from pytorch_lightning import Trainer


def train():
    dataset = WheatDataset(transform=train_transform)
    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset)-train_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=False, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=False, collate_fn=collate_fn)

    detector = FasterRCNN(n_classes=2)

    # run learning rate finder, results override hparams.learning_rate
    trainer = Trainer(gpus=1, progress_bar_refresh_rate=1, max_epochs=30, deterministic=False)

    # call tune to find the lr
    # trainer.tune(classifier,train_dataloader,val_dataloader) # we already did it once = 1e-4
    trainer.fit(detector, train_dataloader, val_dataloader)

    return detector

if __name__ == "__main__":
    model = train()
    torch.save(model, f"{ROOT_DIR}/models/Faster_RCNN.pth")