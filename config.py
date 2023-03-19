import os
import sys

# File to enter basic data about the project

project_name = 'fish_object_detection'

classes = ['starfish', 'shark', 'fish', 'puffin', 'stingray', 'penguin', 'jellyfish', 'background']

# Folders & Paths
train_folder = 'train'
validation_folder = 'valid'
test_folder = 'test'
annotation_folder = 'annotations'
artifacts_folder = 'project_artifacts'
logging_folder = 'logs'
saved_model_folder = 'saved_model'
plots_folder = 'plots'

root_path = os.path.dirname(os.path.realpath(sys.argv[0]))
data_root_path = os.path.join(root_path, 'data')
annotations_root_path = os.path.join(data_root_path, annotation_folder)
project_artifacts_root_path = os.path.join(root_path, artifacts_folder)
saved_model_root_path = os.path.join(project_artifacts_root_path, saved_model_folder)
saved_history_root_path = os.path.join(project_artifacts_root_path, logging_folder)
plots_root_path = os.path.join(project_artifacts_root_path, plots_folder)

# For data aumentation and pre-processing
factor_list_of_list = [[1, 2], [1, 3]] #[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]

# Model parameters & other training options
num_classes = len(classes)
# 'fasterrcnn_with_resnet50_fpn_backbone_v1', 'fasterrcnn_with_resnet50_fpn_backbone_v2', 'fasterrcnn_with_mobilenet_backbone',
# 'fcos_with_resnet50_fpn_backbone', 'retinanet_with_resnet50_fpn_backbone_v1', 'retinanet_with_resnet50_fpn_backbone_v2',
# 'ssd300_with_vgg16_backbone'
architecture_list = ['retinanet_with_resnet50_fpn_backbone_v2', 'ssd300_with_vgg16_backbone']
batch_size_list = [2, 1] # [2, 1], [4, 2]  [train_batch_size, test_batch_size]
epoch_list = [2] # 50, 100, 200
optimizer_list = ['Adam'] # 'SGD', 'Adam', 'RMSProp'
scheduler_list = ['Step'] # 'Step', 'Multi-Step', 'Exponential'
learning_rate_list = [0.0005] # 0.0005, 0.001, 0.0015 etc.

# Model evaluation
test_image_folder_path = 'validation_12'
test_annotation_file_path = os.path.join(annotations_root_path, 'validation_processed_12.csv')
evaluation_batch_size = 16
eval_metric = 'map' # 'map', 'map_50', 'map_75'

# For prediction
best_model = ''
prediction_image_path = os.path.join(data_root_path, 'validation_12', 'IMG_2277_jpeg_jpg.rf.86c72d6192da48d941ffa957f4780665_preprocess_factor_1.jpg')
