
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


def get_class_value(x):
  if(x == 'starfish'):
    return 0
  elif(x == 'shark'):
    return 1
  elif(x == 'fish'):
    return 2
  elif(x == 'puffin'):
    return 3
  elif(x == 'stingray'):
    return 4
  elif(x == 'penguin'):
    return 5
  elif(x == 'jellyfish'):
    return 6

def get_class_name(x):
  if(x == 0):
    return 'starfish'
  elif(x == 1):
    return 'shark'
  elif(x == 2):
    return 'fish'
  elif(x == 3):
    return 'puffin'
  elif(x == 4):
    return 'stingray'
  elif(x == 5):
    return 'penguin'
  elif(x == 6):
    return 'jellyfish'


def preprocess_image_input(input_image, factor):
  factor_contrast = factor
  factor_brightness = factor

  enhancer_contrast = ImageEnhance.Contrast(input_image)
  eq_contrast = enhancer_contrast.enhance(factor_contrast)

  enhancer_brightness = ImageEnhance.Brightness(eq_contrast)
  eq_brightness = enhancer_brightness.enhance(factor_brightness)

  enhanced_img = eq_brightness
  return enhanced_img


def preprocess_image_input_prediction(input_image, factor=0):
  factor_contrast = factor
  factor_brightness = factor
  #input_image = Image.fromarray((input_image * 255).astype(np.uint8)) # PIL.Image.fromarray(np.uint8(input_image))

  enhancer_contrast = ImageEnhance.Contrast(input_image)
  eq_contrast = enhancer_contrast.enhance(factor_contrast)

  enhancer_brightness = ImageEnhance.Brightness(eq_contrast)
  eq_brightness = enhancer_brightness.enhance(factor_brightness)

  enhanced_img = np.array(eq_brightness).astype(np.float32)
  #enhanced_img = np.resize(enhanced_img, new_shape)
  #enhanced_img = np.expand_dims(enhanced_img, axis=0)
  #enhanced_img = enhanced_img.unsqueeze_(0)
  enhanced_img = torch.as_tensor(enhanced_img, dtype=torch.float32)
  return enhanced_img


def create_image(data, image_name, final_path):
  path = os.path.join(final_path, image_name)
  data.save(path)


def create_dataframe_from_processsed_data(data_list):
  df = pd.DataFrame(columns=['filename', 'bbox', 'class'])
  for data in data_list:
    df_new = pd.DataFrame(data, columns=['filename', 'bbox', 'class'])
    df = pd.concat([df, df_new], ignore_index=True)
  return df


def process_label_data(base_path, annotation_folder, train_folder, validation_folder, factor_list):
  train_processed_list = []
  validation_processed_list = []
  factor_list_str = ''.join([str(i) for i in factor_list])
  filetype = [train_folder, validation_folder]
  for fil in filetype:
    # create new folder for augmented data
    new_folder_name = fil + f'_{factor_list_str}'
    new_folder_path = os.path.join(base_path, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)
    # read data from file and process
    anno_file_name = f'{fil}_annotations.csv'
    anno_file_path = os.path.join(base_path, annotation_folder, anno_file_name)
    df = pd.read_csv(anno_file_path)
    uni_files = df['filename'].unique()
    for uni in uni_files:
      class_list = []
      bbox_list = []
      df_new = df[df['filename'] == uni]
      df_new = df_new.reset_index(drop=True)
      width = df_new.loc[0, 'width']
      height = df_new.loc[0, 'height']
      df_new['class_value'] = df_new['class'].apply(lambda x: get_class_value(x))
      class_list = df_new['class_value'].values
      xmin = df_new['xmin'].values
      xmax = df_new['xmax'].values
      ymin = df_new['ymin'].values
      ymax = df_new['ymax'].values
      for i in range(len(xmin)):
        bbox = [xmin[i], ymin[i], xmax[i], ymax[i]]
        bbox_list.append(bbox)
      base_img_file_path = os.path.join(base_path, fil, uni)
      input_image = Image.open(base_img_file_path).convert("RGB")
      df_row_list = []
      for factor in factor_list:
        new_image_name = f'{uni[:-4]}_preprocess_factor_{factor}.jpg'
        if(factor != 0):
          enhanced_img = preprocess_image_input(input_image, factor)
        else:
          enhanced_img = input_image
        create_image(enhanced_img, new_image_name, new_folder_path)
        df_row = {'filename': new_image_name, 'bbox': bbox_list, 'class': class_list}
        df_row_list.append(df_row)
      #print(df_row_list)
      if(fil == 'train'):
        train_processed_list.append(df_row_list)
      else:
        validation_processed_list.append(df_row_list)
      df_row_list = []
  #print(train_processed_list)
  df_train = create_dataframe_from_processsed_data(train_processed_list)
  df_validation = create_dataframe_from_processsed_data(validation_processed_list)
  return df_train, df_validation


class FishDataset(torch.utils.data.Dataset):
    def __init__(self, root, folder, df, classes, transforms):
        self.root = root
        self.folder = folder
        self.transforms = transforms
        self.df = df
        self.imgs = list(sorted(df.filename.values))


    def __getitem__(self, idx):
        # load images
        # note that we have converted the images to RGB
        for i in range(len(self.df)):
          if(self.df.loc[i, 'filename'] == self.imgs[idx]):
            img_path = os.path.join(self.root, self.folder, self.imgs[idx])
            img = Image.open(img_path).convert("RGB")
            labels = self.df.loc[i, 'class']
            boxes = self.df.loc[i, 'bbox']
        obj_ids = labels
        num_objs = len(labels)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            #print(idx, type(img), type(target))
        return img, target

    def __len__(self):
        return len(self.imgs)

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def generate_dataloaders(base_path, new_train_folder, new_validation_folder, df_train_processed, df_validation_processed, classes, batch_size_list):
    seed_everything(42, workers=True)
    train_batch_size = batch_size_list[0]
    test_batch_size = batch_size_list[1]

    dataset = FishDataset(base_path, new_train_folder, df_train_processed, classes, get_transform(train=True))
    dataset_test = FishDataset(base_path, new_validation_folder, df_validation_processed, classes, get_transform(train=False))


    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=4,
                                              collate_fn=torch_utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=4,
                                                   collate_fn=torch_utils.collate_fn)
    return data_loader, data_loader_test


def generate_dataloader_for_validation(base_path, new_validation_folder, df_validation_processed, classes, eval_batch_size):
    dataset_validation = FishDataset(base_path, new_validation_folder, df_validation_processed, classes, get_transform(train=False))
    data_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=eval_batch_size, shuffle=False, num_workers=4,
                                                   collate_fn=torch_utils.collate_fn)
    return data_loader_validation
