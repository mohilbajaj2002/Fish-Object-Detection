
import os
import torch
import math
import shutil
import numpy as np
import pandas as pd
import torchvision
import torchvision as tv
import transforms as T
import utils as torch_utils
import pytorch_lightning as pl
from PIL import Image, ImageEnhance
from torch.nn import functional as F
from engine import train_one_epoch, evaluate
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pytorch_lightning import Trainer, LightningModule, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.ssd import SSDClassificationHead
#from torchvision.models.detection.fcos import FCOSClassificationHead # commented for windows
from torchvision.models.detection.roi_heads import fastrcnn_loss
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class CreateModel(LightningModule):
    def __init__(self, arch_name, opt_name, sched_name, learning_rate, num_classes, map_metric):
        super().__init__()
        self.arch_name = arch_name
        self.num_classes = num_classes
        # init a pretrained model and fine-tune it
        self.model = self.fine_tune_model()
        self.opt_name = opt_name
        self.sched_name = sched_name
        self.learning_rate = learning_rate
        self.eval_metric = MeanAveragePrecision(iou_type="bbox")
        self.map_metric = map_metric

    def fine_tune_model(self):
        if(self.arch_name == 'fasterrcnn_with_resnet50_fpn_backbone_v1'):
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
            self.in_features = model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, self.num_classes)
            return model
        elif(self.arch_name == 'fasterrcnn_with_resnet50_fpn_backbone_v2'):
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
            self.in_features = model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, self.num_classes)
            return model
        elif(self.arch_name == 'fasterrcnn_with_mobilenet_backbone'):
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
            self.in_features = model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, self.num_classes)
            return model
        elif(self.arch_name == 'fcos_with_resnet50_fpn_backbone'):
            model = torchvision.models.detection.fcos_resnet50_fpn(weights="DEFAULT")
            self.in_channels = model.head.classification_head.conv[0].in_channels
            #self.out_channels = model.head.classification_head.conv[3].out_channels
            self.num_anchors = model.head.classification_head.num_anchors
            #model.head.classification_head.num_classes = self.num_classes
            model.head.classification_head = FCOSClassificationHead(self.in_channels, self.num_anchors, self.num_classes)
            return model
        elif(self.arch_name == 'retinanet_with_resnet50_fpn_backbone_v1'):
            model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")
            self.in_features = model.head.classification_head.conv[0][0].in_channels
            self.out_channels = model.head.classification_head.conv[3][0].out_channels
            self.num_anchors = model.head.classification_head.num_anchors
            model.head.classification_head.num_classes = self.num_classes
            self.cls_logits = torch.nn.Conv2d(self.out_channels, self.num_anchors * self.num_classes, kernel_size = 3, stride=1, padding=1)
            torch.nn.init.normal_(self.cls_logits.weight, std=0.01)  # as per pytorch code
            torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code
            # assign cls head to model
            model.head.classification_head.cls_logits = self.cls_logits
            # RetinaNetClassificationHead
            return model
        elif(self.arch_name == 'retinanet_with_resnet50_fpn_backbone_v2'):
            model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights="DEFAULT")
            self.in_channels = model.head.classification_head.conv[0][0].in_channels
            self.out_channels = model.head.classification_head.conv[3][0].out_channels
            self.num_anchors = model.head.classification_head.num_anchors
            model.head.classification_head = RetinaNetClassificationHead(self.in_channels, self.num_anchors, self.num_classes)
            return model
        elif(self.arch_name == 'ssd300_with_vgg16_backbone'):
            model = torchvision.models.detection.ssd300_vgg16(weights="DEFAULT")
            self.in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320)) #model.head.classification_head.module_list[0].in_channels
            #self.out_channels = model.head.classification_head.conv[3].out_channels
            self.num_anchors = model.anchor_generator.num_anchors_per_location()
            model.head.classification_head = SSDClassificationHead(self.in_channels, self.num_anchors, self.num_classes)
            return model

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.model(x, y)
        #print(y_hat)
        arch = self.arch_name.split('_')[0]
        if(arch == 'fasterrcnn'):
          loss = y_hat['loss_classifier'] + y_hat['loss_box_reg']
        else:
          loss = y_hat['classification'] + y_hat['bbox_regression']
        return loss

    def validation_step(self, batch, batch_nb):
        map = self._shared_eval_step(batch, batch_nb)
        metrics = {"val_map": map}
        self.log_dict(metrics)

    def test_step(self, batch, batch_nb):
        map = self._shared_eval_step(batch, batch_nb)
        metrics = {"test_map": map}
        self.log_dict(metrics)

    def _shared_eval_step(self, batch, batch_nb):
        x, y = batch
        #y = list(y)
        y_hat = self.model(x)
        map = self.eval_metric(y_hat, y)
        #print(map)
        return map[self.map_metric] # map['map'], map['map_50'], map['map_75']

    def predict_step(self, batch, batch_nb, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def get_optimizer(self, params):
      if(self.opt_name == 'SGD'):
        return torch.optim.SGD(params, lr=self.learning_rate, momentum=0.9, weight_decay=0.0005)
      elif(self.opt_name == 'Adam'):
        return torch.optim.Adam(params, lr=self.learning_rate, weight_decay=0.0005)
      elif(self.opt_name == 'RMSProp'):
        return torch.optim.RMSprop(params, lr=self.learning_rate, momentum=0.9, weight_decay=0.0005)

    def get_scheduler(self, optimizer):
      if(self.sched_name == 'Step'):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
      elif(self.sched_name == 'Multi-Step'):
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)
      elif(self.sched_name == 'Exponential'):
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        #optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        optimizer = self.get_optimizer(params)
        scheduler = self.get_scheduler(optimizer)
        return [optimizer], [scheduler]
