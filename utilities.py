
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
import architectures as arch
import data_generation as datagen


def visualize_data(img_path, bboxes, classes):
  convert_tensor = tv.transforms.ToTensor()
  img = Image.open(img_path).convert("RGB")
  img = convert_tensor(img)
  img = img * 255
  img = torch.as_tensor(img, dtype=torch.uint8) #torch.tensor(img, dtype=torch.uint8)
  bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
  labels = [get_class_name(i) for i in classes]
  pic = tv.utils.draw_bounding_boxes(img, bboxes, labels, width=5, colors=(0,0,255))
  pic = pic.permute(1,2,0)
  plt.imshow(pic)
  plt.show()

def visualize_logs(log_path, plot_path, save_toggle):
  model_name = log_path.split('/')[-1]
  event_acc = EventAccumulator(log_path)
  event_acc.Reload()
  tags = event_acc.Tags()["scalars"]
  for tag in tags:
    if(tag == 'epoch'):
      #epoch_list = []
      event_list = event_acc.Scalars(tag)
      epoch_list = list(map(lambda x: x.value, event_list))
      epoch_list.pop()
    if(tag == 'val_map'):
      #val_map_list = []
      event_list = event_acc.Scalars(tag)
      val_map_list = list(map(lambda x: x.value, event_list))
  plt.plot(epoch_list, val_map_list)
  plt.title(f'Results for {model_name}', fontsize=15)
  plt.xlabel("Epochs")
  plt.ylabel('Val_MAP')
  if(save_toggle):
      image_name = model_name + '.png'
      image_path = os.path.join(plot_path, image_name)
      plt.savefig(image_path, bbox_inches='tight')
  else:
      plt.show()


def make_prediction(model_path, image_path, factor):
    image_list = []
    file_path = os.listdir(os.path.join(model_path))[0]
    checkpoint = torch.load(os.path.join(model_path, file_path), map_location=torch.device('cpu'))
    #print(checkpoint)
    model_name = model_path.split('/')[-1]
    arch_name_list = model_name.split('_')[3: -7]
    sched_name = model_name.split('_')[-5]
    opt_name = model_name.split('_')[-7]
    learning_rate = float(model_name.split('_')[-3])
    arch_name = '_'.join(arch_name_list)
    # Load model
    model = arch.CreateModel(arch_name, opt_name, sched_name, learning_rate, config.num_classes, config.eval_metric)
    model.load_state_dict(checkpoint['state_dict'])
    # Preprocess image
    input_image = Image.open(image_path).convert("RGB")
    enhanced_img = datagen.preprocess_image_input_prediction(input_image, factor)
    #print(enhanced_img.shape)
    enhanced_img = enhanced_img.permute(2, 1, 0)
    #print(enhanced_img.shape)
    image_list.append(enhanced_img)
    # Create dataloader and make prediction
    data_loader_prediction = torch.utils.data.DataLoader(image_list, batch_size=1, shuffle=False, num_workers=2)
    trainer = Trainer()
    result = trainer.predict(model, data_loader_prediction)
    return result
