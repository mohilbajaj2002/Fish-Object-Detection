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


import config
import utilities as fish_utils
import architectures as arch
import data_generation as datagen

# Find Data Augmentation Factor 
model_to_be_used = config.best_model
datatype = model_to_be_used.split('_')[2]
factor = int(datatype[0]) if len(datatype) > 0 else 0       #int(datatype[0])

# Make Prediction
model_path = os.path.join(config.saved_model_root_path, config.best_model)
result = fish_utils.make_prediction(model_path, config.prediction_image_path, factor)

# Visualize Prediction
bboxes = result[0][0]['boxes']
classes = result[0][0]['labels']
fish_utils.visualize_data(config.prediction_image_path, bboxes, classes)
