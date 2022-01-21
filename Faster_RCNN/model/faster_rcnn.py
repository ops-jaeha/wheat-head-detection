# Torch imports
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.boxes import box_iou
from torchvision.models.detection._utils import Matcher

# Pytorch import
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import seed_everything
seed_everything(25081992)

class FasterRCNN(LightningModule):
    def __init__(self, n_classes):
        super().__init__()
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
        self.lr = 1e-4

    def forward(self, imgs, targets=None):
        # Torchvision FasterRCNN returns the loss during training
        # and the boxes during eval
        self.detector.eval()
        return self.detector(imgs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        imgs = batch[0]

        targets = []
        for boxes in batch[1]:
            target = {}
            target["boxes"] = boxes.cuda()
            target["labels"] = torch.ones(len(target["boxes"])).long().cuda()
            targets.append(target)

        # fasterrcnn takes both images and targets for training, returns
        loss_dict = self.detector(imgs, targets)
        loss = sum(loss for loss in loss_dict.values())
        return {"loss": loss, "log": loss_dict}

    def validation_step(self, batch, batch_idx):
        img, boxes, metadata = batch
        pred_boxes = self.forward(img)

        self.val_loss = torch.mean(
            torch.stack([self.accuracy(b, pb["boxes"], iou_threshold=0.7) for b, pb in zip(boxes, pred_boxes)]))
        return self.val_loss

    def test_step(self, batch, batch_idx):
        img, boxes, metadata = batch
        pred_boxes = self.forward(img)  # in validation, faster rcnn return the boxes
        self.test_loss = torch.mean(
            torch.stack([self.accuracy(b, pb["boxes"], iou_threshold=0.7) for b, pb in zip(boxes, pred_boxes)]))
        return self.test_loss

    def accuracy(self, src_boxes, pred_boxes, iou_threshold=1.):
        """
        The accuracy method is not the one used in the evaluator but very similar
        """
        total_gt = len(src_boxes)
        total_pred = len(pred_boxes)
        if total_gt > 0 and total_pred > 0:

            # Define the matcher and distance matrix based on iou
            matcher = Matcher(iou_threshold, iou_threshold, allow_low_quality_matches=False)
            match_quality_matrix = box_iou(src_boxes, pred_boxes)

            results = matcher(match_quality_matrix)

            true_positive = torch.count_nonzero(results.unique() != -1)
            matched_elements = results[results > -1]

            # in Matcher, a pred element can be matched only twice
            false_positive = torch.count_nonzero(results == -1) + (
                        len(matched_elements) - len(matched_elements.unique()))
            false_negative = total_gt - true_positive

            return true_positive / (true_positive + false_positive + false_negative)

        elif total_gt == 0:
            if total_pred > 0:
                return torch.tensor(0.).cuda()
            else:
                return torch.tensor(1.).cuda()
        elif total_gt > 0 and total_pred == 0:
            return torch.tensor(0.).cuda()