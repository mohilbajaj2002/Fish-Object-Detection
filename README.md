"# Fish-Object-Detection"

This project aims to perform object detection on Roboflow.ai's Fish Dataset using Pytorch and its ecosystem. Model architectures like Faster-RCNN, FCOS, RetinaNet and SSD with a variety of backbones like Resnet-50-FPN, MobileNet and VGG-16 were explored. Data pre-processing steps included image resizing and normalization. Data Augmentation was done using image contrast enhancement & brightness rectification, and horizontal flip.

The project consists of 10 files and 2 folders:
- Requirements.txt : This file contains the libraries needed to run the project.
- Requirements_windows.txt : This file contains the libraries to run the project on Windows.
- References : This folder contains some basic data processing scripts and was downloaded from torch's github repo.
- Data : This folder holds the project data i.e. images and annotations.
- Config.py : This file contains all the project variables and parameters.
- Architectures.py : This file contains different model architectures discussed above bundled into a single class.
- Data_generation.py : This file contains functions related to the train and validation dataloaders.
- Tests.py : This file contains a few quick tests to validate data after pre-processing.
- Utilities.py : This file contains code for performing performance evaluation, graph plotting and enhancements related to code readability.
- Main.py : This is the main file. It pre-processes the data and performs model training & evaluation.
- Model_evaluation.py : This file allows standalone model evaluation once training has been performed.
- Prediction.py : This is the file for making prediction(s).

The project has been designed to be a one-shot solution for model training and evaluation (after your preferences have been added to the config file) and thus does not require human intervention or interaction once initiated. This makes it most suited for High Performance Computing (HPC) environments. For environments like single GPU/Google Colab, the main.py file will need to be modified to break processing time into smaller chunks.

To initiate, download the data from Roboflow and extract it to the data directory. Next, copy the files from the references folder to the root folder. Finally, add your preferences to the config.py file and then run main.py. Once model(s) have been trained, update the evaluation/prediction section in the config file and use the appropriate file to perform standalone model performance evaluation or to make prediction(s).
