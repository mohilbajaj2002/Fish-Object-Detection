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

def data_check(base_path, folder, df):
  folder_path = os.path.join(base_path, folder)
  file_list = list(os.listdir(folder_path))
  if(len(file_list) != len(df)):
    print(f'Check 1: Number of files mismatch in {folder} folder and dataframe!')
  else:
    print(f'Check 1 successful for {folder} folder!')
    indi_flag = 0
    overall_flag = 0
    for i in range(len(df)):
      for j in file_list:
        if(df.loc[i, 'filename'] == j):
          indi_flag = 1
          break
      if(indi_flag != 1):
        print(f'{j} not found in {folder} folder!')
        indi_flag = 0
        overall_flag = overall_flag + 1
      else:
        indi_flag = 0
    if(overall_flag != 0):
      print(f'{overall_flag} files not found in {folder} folder!')
    else:
      print(f'Check 2 successful for {folder} folder!')
