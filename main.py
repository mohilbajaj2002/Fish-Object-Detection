
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


# Creating Data and Model Artifact folders
os.makedirs(config.annotations_root_path, exist_ok=True)
os.makedirs(config.project_artifacts_root_path, exist_ok=True)
os.makedirs(config.saved_model_root_path, exist_ok=True)
os.makedirs(config.saved_history_root_path, exist_ok=True)
os.makedirs(config.plots_root_path, exist_ok=True)

# Moving annotation files to the appropriate folder
original_filename = '_annotations.csv'
data_folders = [config.train_folder, config.validation_folder, config.test_folder]
for folder in data_folders:
    new_filename = folder + original_filename
    source = os.path.join(config.data_root_path, folder, original_filename)
    destination = os.path.join(config.annotations_root_path, new_filename)
    os.rename(source, destination)


# Preprocessing raw data (images and annotations)
for factor_list in config.factor_list_of_list:
    print(f'Performing data augmentation: {factor_list}...')
    df_train_processed, df_validation_processed = datagen.process_label_data(config.data_root_path, config.annotation_folder, config.train_folder, config.validation_folder, factor_list)

    # Performing a quick test to check data validity
    factor_list_str = ''.join([str(i) for i in factor_list])
    new_train_folder = config.train_folder + f'_{factor_list_str}'
    new_validation_folder = config.validation_folder + f'_{factor_list_str}'
    tests.data_check(config.data_root_path, new_train_folder, df_train_processed)
    tests.data_check(config.data_root_path, new_validation_folder, df_validation_processed)

    # Creating train and validation dataloaders
    data_loader, data_loader_test = datagen.generate_dataloaders(config.data_root_path, new_train_folder, new_validation_folder, df_train_processed, df_validation_processed, config.classes, config.batch_size_list)
    # Starting model training
    for arch_name in config.architecture_list:
        for opt_name in config.optimizer_list:
            for sched_name in config.scheduler_list:
                for epoch in config.epoch_list:
                    for learning_rate in config.learning_rate_list:
                        model_no = len([i for i in os.listdir(config.saved_model_root_path)]) + 1
                        model_name = f'Attempt{model_no}_datatype_{factor_list_str}_{arch_name}_{opt_name}_scheduler_{sched_name}_lr_{learning_rate}_epochs_{epoch}'
                        saved_model_path = os.path.join(config.saved_model_root_path, model_name)
                        early_stopping = EarlyStopping(monitor='val_map', mode='max', patience=5)
                        model_checkpoint = ModelCheckpoint(monitor='val_map', mode='max', dirpath=saved_model_path)
                        logger = pl.loggers.TensorBoardLogger(name=model_name, save_dir=config.saved_history_root_path)
                        # Create model
                        model = arch.CreateModel(arch_name, opt_name, sched_name, learning_rate, config.num_classes, config.eval_metric)
                        trainer = Trainer(callbacks=[early_stopping, model_checkpoint],
                                          devices="auto", accelerator="auto",
                                          max_epochs=epoch,
                                          logger = logger)
                        trainer.fit(model, data_loader, data_loader_test)
                        # Model Performance Evaluation
                        trainer.test(model, data_loader_test)
                        # Generate Log Plots
                        save_toggle = True # True saves plots, maintaining continuous looping. False will display plot, pausing execution
                        log_path = os.path.join(config.saved_history_root_path, model_name)
                        fish_utils.visualize_logs(log_path, config.plots_root_path, save_toggle)
