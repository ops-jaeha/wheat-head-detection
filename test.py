# Common imports
from tqdm.notebook import tqdm
import pandas as pd

# Torch imports
import torch
from torch.utils.data import Dataset
# Import Python File
from Faster_RCNN.transform import test_transform
from Faster_RCNN.dataset import WheatDatasetPredict
from Faster_RCNN.parameter import DRIVE_DIR, DATA_DIR


detector = torch.load(f"{DRIVE_DIR}/model/Faster_RCNN.pth")
detector.freeze()
test_dataset = WheatDatasetPredict(transform=test_transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


def encode_boxes(boxes):
    if len(boxes) >0:
        boxes = [" ".join([str(int(i)) for i in item]) for item in boxes]
        BoxesString = ";".join(boxes)
    else:
        BoxesString = "no_box"
    return BoxesString


def test():
    results = []
    for batch in tqdm(test_dataloader):
        norm_img, img , img_names , metadata = batch

        predictions = detector.detector(norm_img)

        for img_name, pred, domain in zip(img_names, predictions, metadata):
            boxes = pred["boxes"]
            scores = pred["scores"]
            boxes = boxes[scores > 0.5].cpu().numpy()
            PredString = encode_boxes(boxes)
            results.append([img_name,PredString,domain.item()])

    results = pd.DataFrame(results, columns=["image_name", "PredString", "domain"])
    results.to_csv(f"{DRIVE_DIR}/model/submission_faster_rcnn.csv")


if __name__ == '__main__':
    test()