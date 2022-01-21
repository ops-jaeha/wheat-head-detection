# Import Library
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.LongestMaxSize(1024,p=1),
    A.PadIfNeeded(min_height=1024, min_width=1024,p=1,border_mode=1,value=0),
    A.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2(),
],bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_area=20))

test_transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])