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

import tests
import config
import utilities as fish_utils
import architectures as arch
import data_generation as datagen


# Creating validation dataloader
df_validation_processed = pd.read_csv(config.test_annotation_file_path)
data_loader_validation = datagen.generate_dataloader_for_validation(config.data_root_path, config.test_folder_path, df_validation_processed, config.classes, config.evaluation_batch_size)
# Load model(s)
for model_path in os.listdir(config.saved_model_root_path):
    print(f'Model Name: {model_path}')
    file_path = os.listdir(os.path.join(base_path, saved_model_folder, model_path))[0]
    checkpoint = torch.load(os.path.join(base_path, saved_model_folder, model_path, file_path), map_location=torch.device('cpu'))
    #print(checkpoint)
    model_name = model_path.split('_')[1: ]
    arch_name_list = model_path.split('_')[3: -7]
    sched_name = model_path.split('_')[-5]
    opt_name = model_path.split('_')[-7]
    learning_rate = float(model_path.split('_')[-3])
    arch_name = '_'.join(arch_name_list)
    # Create model and load checkpoint
    model = arch.CreateModel(arch_name, opt_name, sched_name, learning_rate, config.num_classes, config.eval_metric)
    model.load_state_dict(checkpoint['state_dict'])
    # Perform Model Evaluation
    trainer = Trainer()
    result = trainer.validate(model, dataloaders=data_loader_validation)
    #print(result)
    print('..................')
