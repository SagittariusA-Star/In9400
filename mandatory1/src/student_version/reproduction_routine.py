

from __future__ import division
from pydoc import classname
from turtle import Turtle
from matplotlib import image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch import Tensor
import time
import os
import numpy as np
import PIL.Image
import sklearn.metrics
from typing import Callable, Optional
from RainforestDataset import RainforestDataset, ChannelSelect, get_classes_list
from YourNetwork import SingleNetwork, TwoNetworks
import sys 
import h5py 


def load_data(filename):
    """Function loading saved diagnostics and hyperparameters from HDF5 file.

    Parameters
    ----------
    filename : str
        File name ending for datafile to load.

    Returns
    -------
    best model state dict, dictionary with diagnostics data, dictionary with hyperparameters.
        _description_
    """
    with h5py.File("diagnostics_" + filename + ".h5", "r") as infile: 
        # Dictionary saving diagnostics data
        data = {
        "best_epoch": infile["best_epoch"][()],
        "best_measure": infile["best_measure"][()],
        "trainlosses": infile["trainlosses"][()],
        "testlosses": infile["testlosses"][()],
        "testperfs": infile["testperfs"][()],
        "concat_labels": infile["concat_labels"][()],
        "concat_pred": infile["concat_pred"][()],
        "classwise_perf": infile["classwise_perf"][()],
        "filenames": infile["filenames"][()].astype(str)
        }

        # Dictionary saving hyperparameters
        hyperparams = {
        "use_gpu": bool(infile["hyperparameters/use_gpu"][()]),
        "lr":       infile["hyperparameters/learning_rate"][()],
        "batchsize_train": int(infile["hyperparameters/batchsize_train"][()]),
        "batchsize_val":   int(infile["hyperparameters/batchsize_val"][()]),
        "maxnumepochs":    int(infile["hyperparameters/maxnumepochs"][()]),
        "scheduler_stepsize": int(infile["hyperparameters/scheduler_stepsize"][()]),
        "scheduler_factor":   infile["hyperparameters/scheduler_factor"][()],
        "numworkers":         int(infile["hyperparameters/num_workers"][()]),
        "seed"      :         int(infile["hyperparameters/seed"][()]),
        "freeze_layers":      bool(infile["hyperparameters/freeze_layers"][()])
        }
    # State dict of best model
    weights = torch.load("bestweights_" + filename + ".pt")

    return weights, data, hyperparams

def evaluate_meanavgprecision(model, dataloader, device, loaded_data, hyperparams):
    """Function evaluating best model on validation dataset. The routine also compares
       (prints) the computed prediction scores of the model to the scores saved during 
       the best validation epoch.

    Parameters
    ----------
    model : nn.Module
        Model saved during best training and validation epoch
        to be evaluted and compared to saved scores.
    dataloader : torch.utils.data.DataLoader
      Validation data loader instance.
    device : str
        Device to run on
    loaded_data : dict
        Dictionary of saved data from the best epoch to compare
        model evaluations to for reproduction purposes.
    hyperparams : dict
        Dictionary of hyperparameters used to make the best model.
    """

    model.eval()

    start = 0   # Starting intex of batch
    stop  = hyperparams["batchsize_val"]    # Stopping index of batch

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):       # Looping through validation dataset
            if (batch_idx%100==0) and (batch_idx>=100):
               print('at val batchindex: ', batch_idx)
      
            images = data['image'].to(device)               # Sending images to device
            labels = data['label']

            if isinstance(model, TwoNetworks):
                """If we use TwoNetworks model we separate RGB and Ir parts of images"""
                IRchannel  = images[:, 3, ...].unsqueeze(1)   # selecting IR channel
                RGBchannel = images[:, :3, ...]               # selecting RGB channel
                outputs = model.forward(RGBchannel, IRchannel)
            elif isinstance(model, SingleNetwork) and not model.weight_init:
                """If we use SingleNetworks model with no weight argument
                we only call the model with RGB images"""
                images = images[:, :3, ...]
                outputs = model.forward(images)
            else: 
                """If we use SingleNetworks model with weight argument
                we call the model with RGB and Ir images"""
                outputs = model.forward(images)
            

            cpu_out = outputs.to('cpu')       # Sending model output to CPU
            scores = torch.sigmoid(cpu_out)   # Transform model output from real numbers to probability space [0, 1]
            scores = scores.numpy()           # Transforming scores to numpy array
            labels = labels.float()           # Ensuring labels are float datatype

            size = scores.shape              
            stop = start + size[0]            # Update stop index

            # Printing and comparing scores from model to scores saved in file.
            print("----------------------------------------------------")
            print("From model: ")
            print(scores[:, :])
            print("From file: ")
            from_file = loaded_data["concat_pred"][start:stop, :]
            print(from_file[:, :])
            print("Relative difference:", 100 * np.abs((scores - from_file) / scores)[:, :], "%")
            print("----------------------------------------------------")
            start = stop    # Update starting index



# Loading SingleNetwork RGB, the data (scores etc.) and the hyperparameters from best epoch

filename1 = "task1"

weights_1, data_1, hyperparams_1 = load_data(filename1)

# Loading TwoNetworks RGBIr, the data (scores etc.) and the hyperparameters from best epoch

filename3 = "task3"

weights_3, data_3, hyperparams_3 = load_data(filename3)

# Loading SingleNetwork RGBIr, the data (scores etc.) and the hyperparameters from best epoch

filename4 = "task4"

weights_4, data_4, hyperparams_4 = load_data(filename4)


# Setting random seeds for reproducability sake.
torch.manual_seed(hyperparams_1["seed"])
torch.cuda.manual_seed(hyperparams_1["seed"])
np.random.seed(hyperparams_1["seed"])

num_workers = 1

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ChannelSelect(),        # Selecting only RGB channels for first model
        transforms.Normalize([0.7476, 0.6534, 0.4757], [0.1677, 0.1828, 0.2137])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        #transforms.RandomHorizontalFlip(), # Uncommented this for reproducability sake
        transforms.ToTensor(),
        ChannelSelect(),    # Selecting only RGB channels for first model
        transforms.Normalize([0.7476, 0.6534, 0.4757], [0.1677, 0.1828, 0.2137])
    ]),
}

#========================================================
#=== Update this path when you run it on your system ====
#========================================================
data_root_dir = "../../data/" 
#========================================================

###########################################################################################
#####                                     Task 1                                      #####
###########################################################################################


# Defining training and validation data sets used for training SingleNetwork RGB network (task 1)
image_datasets={}
image_datasets['train'] = RainforestDataset(root_dir = data_root_dir, trvaltest=0, transform = data_transforms['train'])
image_datasets['val']   = RainforestDataset(root_dir = data_root_dir, trvaltest=1, transform = data_transforms['val'])

# Defining training and validation data loaders
dataloaders = {}
dataloaders['train'] = DataLoader(image_datasets['train'], batch_size = hyperparams_1["batchsize_train"],   shuffle = True, num_workers = num_workers) 
dataloaders['val']   = DataLoader(image_datasets['val'],   batch_size = hyperparams_1["batchsize_val"],     shuffle = False, num_workers = num_workers) # Shuffle to False to ensure reproducability by TAs 

# Sending to device to be used.
if True == hyperparams_1['use_gpu']:
    device= torch.device('cuda:0')
else:
    device= torch.device('cpu')


pretrained_resnet18 = models.resnet18(pretrained = True)    # Defining model instance of pretrained ResNet18 to build SingleNetwork of off.
model = SingleNetwork(pretrained_resnet18, freeze = hyperparams_1["freeze_layers"]) # Defining an instance of SingleNetwork RGB

model.load_state_dict(torch.load("bestweights_" + filename1 + ".pt"))   # Updating parameters of SingleNetwork instance with loaded state dict.
model = model.to(device)    # Sending model to device.

print("Evaluating SingleNetwork RGB (task 1) and compare scores to scores from file:")
time.sleep(10)
evaluate_meanavgprecision(model, dataloaders["val"], device, data_1, hyperparams_1) # Run reproducability routine on SingleNetwork RGB.


###########################################################################################
#####                                     Task 3                                      #####
###########################################################################################


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ChannelSelect([0, 1, 2, 3]),    # Selecting only RGB and Ir channels for first model
        transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        #transforms.RandomHorizontalFlip(),  # Uncommented this for reproducability sake
        transforms.ToTensor(),
        ChannelSelect([0, 1, 2, 3]),    # Selecting only RGB and Ir channels for first model
        transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284])
    ]),
}
  
# Defining training and validation data sets used for training TwoNetworks RGBIr network (task 1)
image_datasets={}
image_datasets['train'] = RainforestDataset(root_dir = data_root_dir, trvaltest=0, transform = data_transforms['train'])
image_datasets['val']   = RainforestDataset(root_dir = data_root_dir, trvaltest=1, transform = data_transforms['val'])

# Defining training and validation data loaders
dataloaders = {}
dataloaders['train'] = DataLoader(image_datasets['train'], batch_size = hyperparams_3["batchsize_train"],   shuffle = True, num_workers = num_workers) # Shuffle to False to ensure reproducability by TAs
dataloaders['val']   = DataLoader(image_datasets['val'],   batch_size = hyperparams_3["batchsize_val"],     shuffle = False, num_workers = num_workers) 

pretrained_resnet18_1 = models.resnet18()   # Defining the two ResNet18 branches, both RGB and Ir, for TwoNetworks instance
pretrained_resnet18_2 = models.resnet18()

model = TwoNetworks(pretrained_resnet18_1, pretrained_resnet18_2, freeze = hyperparams_3["freeze_layers"])  # Making instance of TwoNetworks instances
model.load_state_dict(torch.load("bestweights_" + filename3 + ".pt"))       # Updating parameters of TwoNetworks instance with loaded state dict.
model = model.to(device) # Sending model to device

print("Evaluating TwoNetworks RGBIr (task 3) and compare scores to scores from file:")
time.sleep(10)
evaluate_meanavgprecision(model, dataloaders["val"], device, data_3, hyperparams_3) # Run reproducability routine on TwoNetworks RGBIr.



###########################################################################################
#####                                     Task 4                                      #####
###########################################################################################


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ChannelSelect([0, 1, 2, 3]),     # Selecting only RGB and Ir channels for first model
        transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        #transforms.RandomHorizontalFlip(),  # Uncommented this for reproducability sake
        transforms.ToTensor(),
        ChannelSelect([0, 1, 2, 3]),     # Selecting only RGB and Ir channels for first model
        transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284])
    ]),
}
  
# Defining training and validation data sets used for training SingleNetwork RGBIr network (task 1)
image_datasets={}
image_datasets['train'] = RainforestDataset(root_dir = data_root_dir, trvaltest=0, transform = data_transforms['train'])
image_datasets['val']   = RainforestDataset(root_dir = data_root_dir, trvaltest=1, transform = data_transforms['val'])

# Defining training and validation data loaders
dataloaders = {}
dataloaders['train'] = DataLoader(image_datasets['train'], batch_size = hyperparams_4["batchsize_train"],   shuffle = True, num_workers = num_workers) # Shuffle to False to ensure reproducability by TAs
dataloaders['val']   = DataLoader(image_datasets['val'],   batch_size = hyperparams_4["batchsize_val"],     shuffle = False, num_workers = num_workers) 

pretrained_resnet18 = models.resnet18(pretrained = True)    # Making instance of ResNet18 to build SingleNetwork RGBIr of off
model = SingleNetwork(pretrained_resnet18, weight_init = "kaiminghe", freeze = hyperparams_4["freeze_layers"])  # Making instance of SingleNetwork RBGIr
model.load_state_dict(torch.load("bestweights_" + filename4 + ".pt"))   # Updating parameters of SingleNetwork instance with loaded state dict.
model = model.to(device)    # Sending model to device

print("Evaluating SingleNetwork RGBIr (task 4) and compare scores to scores from file:")
time.sleep(10)
evaluate_meanavgprecision(model, dataloaders["val"], device, data_4, hyperparams_4) # Run reproducability routine on SingleNetwork RGBIr.

