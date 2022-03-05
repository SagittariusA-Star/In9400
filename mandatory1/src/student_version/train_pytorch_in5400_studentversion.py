

from __future__ import division
from pydoc import classname
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
import copy

def train_epoch(model, trainloader, criterion, device, optimizer):
    """Function that runs a training epoch of a given model.

    Parameters
    ----------
    model : nn.Module
        Model instance to be trained.
    trainloader : torch.utils.data.DataLoader
        Data loader for training dataset.
    criterion nn.modules.loss._Loss: 
        Loss function instance.
    device : str
        Device to run training on.
    optimizer : torch.optim.Optimizer 
        Optimizer to use in training. 
    Returns
    -------
    float
        Averaged loss for training epoch.
    """
    model.train() # This is a training function
 
    losses = []   # Empty list to save training losses in
    for batch_idx, data in enumerate(trainloader):        # Looping through all batches of training dataset
        if (batch_idx % 100 == 0) and (batch_idx >= 100):
          print('at batchidx',batch_idx)

        images = data["image"].to(device)   # Moving image and labels to device in use
        labels = data["label"].to(device)

        if isinstance(model, TwoNetworks):
          """If we use TwoNetworks model we separate RGB and Ir parts of images"""
          IRchannel  = images[:, 3, ...].unsqueeze(1)   # selecting IR channel
          RGBchannel = images[:, :3, ...]               # selecting RGB channel
          output = model(RGBchannel, IRchannel)
        elif isinstance(model, SingleNetwork) and not model.weight_init:
          """If we use SingleNetworks model with no weight argument
             we only call the model with RGB images"""
          images = images[:, :3, ...] # selecting RGB only
          output = model(images)
        else: 
          """If we use SingleNetworks model with weight argument
             we call the model with RGB and Ir images"""
          output = model(images)
        
        loss = criterion(output.float(), labels.float())  # Computing loss from model output and labels
        losses.append(loss.item())  # Save losses to list
        
        optimizer.zero_grad() # Zeroing out gradient before back propagation
        loss.backward()       # Back propagating
        optimizer.step()      # Updating model parameters.

    return np.mean(losses)


def evaluate_meanavgprecision(model, dataloader, criterion, device, numcl):
    """Function running a validation on given model for a given epoch.

    Parameters
    ----------
    model : nn.Module
        Model instance to be trained.
    trainloader : torch.utils.data.DataLoader
        Data loader for training dataset.
    criterion : nn.modules.loss._Loss
        Loss function instance.
    device : str
        Device to run training on.
    numcl : int
        Number of classes to classify.

    Returns
    -------
      ndarray, float, ndarray, ndarray, ndarray
        Returning array of average precision per class, mean validation loss,
        concatinated validation dataset labels and corresponding probability scores
        and lastrly the filenames of images in validation data set used.
    """
    model.eval()  # This is a validation function    
    
    concat_pred   = np.empty((0, numcl)) #prediction scores for each class. each numpy array is a list of scores. one score per image
    concat_labels = np.empty((0, numcl)) #labels scores for each class. each numpy array is a list of labels. one label per image
    avgprecs      = np.zeros(numcl)  # average precision for each class
    fnames        = [] # filenames as they come out of the dataloader

    with torch.no_grad():
      losses = []         # Empty list to save validation losses in
      for batch_idx, data in enumerate(dataloader):
          if (batch_idx%100==0) and (batch_idx>=100):
            print('at val batchindex: ', batch_idx)
      
          images = data['image'].to(device)        # Moving images to used device.
          labels = data['label']
         
          ######################################
          # This was an accuracy computation

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
          
          loss = criterion(outputs, labels.to(device))  # Computing loss from model output and labels
          losses.append(loss.item())    # Saving validation loss to list

          cpu_out = outputs.to('cpu')   # Moving model output to CPU

          scores = torch.sigmoid(cpu_out)   # Transform model output from real numbers to probability space [0, 1]
        
          labels = labels.float()           # Ensuring that dtype is float
        
          concat_pred   = np.concatenate((concat_pred,   scores.float()),  axis = 0)   # Collecting prediction scores, labels and filenames
          concat_labels = np.concatenate((concat_labels, labels.float()), axis = 0)
          fnames.append(data["filename"])

    fnames = [name for i in fnames for name in i] # Nested list to continuus list
          
    for c in range(numcl): 
      avgprecs[c] = sklearn.metrics.average_precision_score(concat_labels[:, c], concat_pred[:, c]) # Computing average precision measure of off the prediction
                                                                                                    # scores and labels.
      
    return avgprecs, np.mean(losses), concat_labels, concat_pred, fnames


def traineval2_model_nocv(dataloader_train, dataloader_test, model,  
                          criterion, optimizer, scheduler, num_epochs, device, numcl):
  """Function looping over training epochs, both training and validating model.

  Parameters
  ----------
  dataloader_train : torch.utils.data.DataLoader
      Training data loader instance.
  dataloader_test : torch.utils.data.DataLoader
      Validation data loader instance.
  model : nn.Module
      Instance of model to train and validate
  criterion : nn.modules.loss._Loss
        Loss function instance.
  optimizer : torch.optim.Optimizer 
        Optimizer to use in training.
  scheduler : torch.optim.lr_scheduler
      Learning rate scheduler to use during training.
  num_epochs : int
      Number of epochs to run traing and validation over
  device : str
      Device to run on.
  numcl : int
      Number of classes to run classification of.

  Returns
  -------
  int, float, dict, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray 

      Returning best epoch, the best mean average precision measure, the best model state dict, the training and test losses, the average 
      precision per class and epoch, the labels, probability scores and image names at the best epoch as well as the classwise average
      percision at the best epoch.
  """

  best_measure = 0  # Best average precision
  best_epoch = -1   # Best epoch number

  trainlosses = []  # Empty lists for training and test losses as well as the average precision for each class and epoch.
  testlosses  = []
  testperfs   = []
  
  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    avgloss = train_epoch(model,  dataloader_train,  criterion,  device , optimizer)  # Run training routine per epoch
    trainlosses.append(avgloss) # Save training losses
    
    if scheduler is not None: 
      scheduler.step() # Updating learning rate with learning rate scheduler

    perfmeasure, testloss, concat_labels, concat_pred, fnames  = evaluate_meanavgprecision(model, 
                                                                dataloader_test, 
                                                                criterion, 
                                                                device, 
                                                                numcl)    # Run validation routine per epoch
    testlosses.append(testloss)   # Save test losses
    testperfs.append(perfmeasure) # Save average precision measure per class
    
    print('at epoch: ', epoch,' classwise perfmeasure ', perfmeasure)
    
    avgperfmeasure = np.mean(perfmeasure) # Computing mean average precision over classes
    print('at epoch: ', epoch,' avgperfmeasure: ', avgperfmeasure, "test loss:", testloss)

    if avgperfmeasure > best_measure: # If the mean average precision is higher than in previous epoch 
                                      # the best model and important diagnostics are saved.

      bestweights = copy.deepcopy(model.state_dict()) # Saving a deep copy of the best model parameters.

      best_measure = avgperfmeasure # Updating best mean average precision, best epoch 
      best_epoch   = epoch          # and best classwise average precision
      best_perfmeasure = perfmeasure 

      best_scores = concat_pred     # Saving prediction scores, labels and filenames of best epoch
      best_labels = concat_labels
      best_fnames = fnames


  return best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs, \
         best_labels, best_scores, best_fnames, best_perfmeasure

class yourloss(nn.modules.loss._Loss):

    def __init__(self, reduction: str = 'mean') -> None:
        super(yourloss, self).__init__(reduction)
        self.reduction = reduction
        
    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
      """Evaluation of binary cross-entropy loss with sigmoid logits, 
         to accomodate multi-class and multi-label outputs.

      Parameters
      ----------
      input_ : Tensor
          Output from last linear layer of network
      target : Tensor
          Tensor of labels to compare output of last linear layer to
      Returns
      -------
      Tensor
          Binary cross-entropy loss with sigmoid logits of the multi-class, multi-label, dataset.
      """
      loss_criterion = nn.BCEWithLogitsLoss(reduction = self.reduction) # Setting up instance of binary cross-entropy loss with logits.
      loss = loss_criterion(input_, target)                             # Calling loss function instance.
      return loss


def runstuff():
  """Function containing the main block of the script. 
  """


  config = dict()
  config['use_gpu'] = True    # Whether to use GPU or not
  config['lr'] = 0.005        # Initial learning rate 
  config['batchsize_train'] = 16  
  config['batchsize_val'] = 64
  config['maxnumepochs'] = 35       # Maximum epochs to train over.
  config['scheduler_stepsize'] = 10 # How many epochs to run over before updating learning rate.
  config['scheduler_factor'] = 0.3  # Which factor to use to update learning rate.
  config["num_workers"]      = 1    # Number of workers used in data loader
  config["seed"]             = 774663 # Which seed to use for reproducability later on.
  config["freeze_layers"]    = False  # Whether to freeze middle layers of pretrained networks. Only used for debugging, not for
                                      # "production runs".

  # Setting random seeds for reproducability sake.
  torch.manual_seed(config["seed"])
  torch.cuda.manual_seed(config["seed"])
  np.random.seed(config["seed"])

  # This is a dataset property.
  config['numcl'] = 17          # Number of classes in dataset

  # Data augmentations.
  data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          ChannelSelect(),        # By default selectiong RGB channels only
          transforms.Normalize([0.7476, 0.6534, 0.4757], [0.1677, 0.1828, 0.2137])
      ]),
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          #transforms.RandomHorizontalFlip(),   # Commented out random flip for reproducability sake.
          transforms.ToTensor(),
          ChannelSelect(), # By default selectiong RGB channels only
          transforms.Normalize([0.7476, 0.6534, 0.4757], [0.1677, 0.1828, 0.2137])
      ]),
  }

  #========================================================
  #=== Update this path when you run it on your system ====
  #========================================================
  data_root_dir = "../../data/" 
  #========================================================
  

  # Defining training and validation data sets used for training SingleNetwork RGB network (task 1)
  image_datasets={}
  image_datasets['train'] = RainforestDataset(root_dir = data_root_dir, trvaltest=0, transform = data_transforms['train'])
  image_datasets['val']   = RainforestDataset(root_dir = data_root_dir, trvaltest=1, transform = data_transforms['val'])

  # Defining training and validation data loaders
  dataloaders = {}
  dataloaders['train'] = DataLoader(image_datasets['train'], batch_size = config["batchsize_train"],   shuffle = True, num_workers = config["num_workers"]) 
  dataloaders['val']   = DataLoader(image_datasets['val'],   batch_size = config["batchsize_val"],     shuffle = False, num_workers = config["num_workers"]) # Shuffle to False to ensure reproducability by TAs 

  # Sending to device to be used.
  if True == config['use_gpu']:
      device= torch.device('cuda:0')
  else:
      device= torch.device('cpu')

  pretrained_resnet18 = models.resnet18(pretrained = True)   # Making instance of pretrained 
                                                             # ResNet18 to base SingleNetwork RGB on

  model = SingleNetwork(pretrained_resnet18, freeze = config["freeze_layers"])  # Making instance of Single Network RGB network.

  model = model.to(device) # Sending model to device

  lossfct = yourloss()     # Defining loss function instance
  
  someoptimizer = optim.SGD(model.parameters(), lr = config['lr'])  # Defining optimizer to be used. No momentum was used.

  somelr_scheduler = lr_scheduler.StepLR(someoptimizer, 
                                         gamma = config['scheduler_factor'], 
                                         step_size = config['scheduler_stepsize'])  # Defining step-learning rate scheduler


  ###########################################################################################
  #####                                     Task 1                                      #####
  ###########################################################################################
  
  # Training and validating SingleNetwork RGB
  print("SingleNetwork RGB:"); t0 = time.time()
  best_epoch, best_measure, bestweights, trainlosses, \
  testlosses, testperfs, concat_labels, concat_pred, fnames, classwiseperf \
  = traineval2_model_nocv(dataloaders['train'], dataloaders['val'], model,  lossfct, 
                          someoptimizer, somelr_scheduler, num_epochs= config['maxnumepochs'], 
                          device = device , numcl = config['numcl'] )
  
  # Saving score data etc. from best epoch and hyperparameter used 
  with h5py.File("diagnostics_task1.h5", "w") as outfile: 
    outfile.create_dataset("best_epoch", data = best_epoch)
    outfile.create_dataset("best_measure", data = best_measure)
    outfile.create_dataset("trainlosses", data = trainlosses)
    outfile.create_dataset("testlosses", data = testlosses)
    outfile.create_dataset("testperfs", data = np.array(testperfs))
    outfile.create_dataset("concat_labels", data = concat_labels)
    outfile.create_dataset("concat_pred", data = concat_pred)
    outfile.create_dataset("classwise_perf", data = classwiseperf)
    dt = h5py.special_dtype(vlen=str)
    outfile.create_dataset("filenames", data = np.array(fnames, dtype = dt))
    outfile.create_dataset("hyperparameters/use_gpu", data = config["use_gpu"])
    outfile.create_dataset("hyperparameters/learning_rate", data = config["lr"])
    outfile.create_dataset("hyperparameters/batchsize_train", data = config["batchsize_train"])
    outfile.create_dataset("hyperparameters/batchsize_val", data = config["batchsize_val"])
    outfile.create_dataset("hyperparameters/maxnumepochs", data = config["maxnumepochs"])
    outfile.create_dataset("hyperparameters/scheduler_stepsize", data = config["scheduler_stepsize"])
    outfile.create_dataset("hyperparameters/scheduler_factor", data = config["scheduler_factor"])
    outfile.create_dataset("hyperparameters/num_workers", data = config["num_workers"])
    outfile.create_dataset("hyperparameters/seed", data = config["seed"])

    outfile.create_dataset("hyperparameters/freeze_layers", data = config["freeze_layers"])

  # Saving best model state dict SingleNetwork RGB.
  torch.save(bestweights, "bestweights_task1.pt")
  print("Time:", time.time() - t0, "sec"); t0 = time.time()
  
  ###########################################################################################
  #####                                    Task 3                                       #####
  ###########################################################################################
  
  # Setting random seeds for reproducability sake.
  torch.manual_seed(config["seed"])
  torch.cuda.manual_seed(config["seed"])
  np.random.seed(config["seed"])
  
  print("TwoNetworks:"); t0 = time.time()

  # Data augmentations.
  data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          ChannelSelect([0, 1, 2, 3]),  # Selectiong RGB and Ir channels
          transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284])  # Normalizing all channels
      ]),
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          #transforms.RandomHorizontalFlip(), # Commented random flip out to ensure reproducability
          transforms.ToTensor(),
          ChannelSelect([0, 1, 2, 3]),  # Selectiong RGB and Ir channels
          transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284]) # Normalizing all channels
      ]),
  }
  
  # Defining training and validation data sets used for training TwoNetworks RGBIr network (task 3)
  image_datasets={}
  image_datasets['train'] = RainforestDataset(root_dir = data_root_dir, trvaltest=0, transform = data_transforms['train'])
  image_datasets['val']   = RainforestDataset(root_dir = data_root_dir, trvaltest=1, transform = data_transforms['val'])

  # Defining training and validation data loaders
  dataloaders = {}
  dataloaders['train'] = DataLoader(image_datasets['train'], batch_size = config["batchsize_train"],   shuffle = True, num_workers = config["num_workers"]) # Shuffle to False to ensure reproducability by TAs
  dataloaders['val']   = DataLoader(image_datasets['val'],   batch_size = config["batchsize_val"],     shuffle = False, num_workers = config["num_workers"]) 

  pretrained_resnet18_1 = models.resnet18(pretrained = True)  # Making instances of pretrained ResNet18, one each
  pretrained_resnet18_2 = models.resnet18(pretrained = True)  # for the RGB and Ir branch of TwoNetworks.
  model = TwoNetworks(pretrained_resnet18_1, pretrained_resnet18_2, freeze = config["freeze_layers"]) # Making instance of TwoNetworks class.
  model = model.to(device)  # Sending model to device


  someoptimizer = optim.SGD(model.parameters(), lr = config['lr']) # Defining optimizer to be used. No momentum was used.

  somelr_scheduler = lr_scheduler.StepLR(someoptimizer, 
                                         gamma = config['scheduler_factor'], 
                                         step_size = config['scheduler_stepsize']) # Defining step-learning rate scheduler.

  # Training and validating TwoNetworks RGBIr
  best_epoch, best_measure, bestweights, trainlosses, \
  testlosses, testperfs, concat_labels, concat_pred, fnames, classwiseperf \
  = traineval2_model_nocv(dataloaders['train'], dataloaders['val'] ,  model ,  lossfct, 
                          someoptimizer, somelr_scheduler, num_epochs= config['maxnumepochs'], 
                          device = device , numcl = config['numcl'] )

  # Saving score data etc. from best epoch and hyperparameter used 
  with h5py.File("diagnostics_task3.h5", "w") as outfile: 
    outfile.create_dataset("best_epoch", data = best_epoch)
    outfile.create_dataset("best_measure", data = best_measure)
    outfile.create_dataset("trainlosses", data = trainlosses)
    outfile.create_dataset("testlosses", data = testlosses)
    outfile.create_dataset("testperfs", data = np.array(testperfs))
    outfile.create_dataset("concat_labels", data = concat_labels)
    outfile.create_dataset("concat_pred", data = concat_pred)
    outfile.create_dataset("classwise_perf", data = classwiseperf)
    dt = h5py.special_dtype(vlen=str)
    outfile.create_dataset("filenames", data = np.array(fnames, dtype = dt))
    outfile.create_dataset("hyperparameters/use_gpu", data = config["use_gpu"])
    outfile.create_dataset("hyperparameters/learning_rate", data = config["lr"])
    outfile.create_dataset("hyperparameters/batchsize_train", data = config["batchsize_train"])
    outfile.create_dataset("hyperparameters/batchsize_val", data = config["batchsize_val"])
    outfile.create_dataset("hyperparameters/maxnumepochs", data = config["maxnumepochs"])
    outfile.create_dataset("hyperparameters/scheduler_stepsize", data = config["scheduler_stepsize"])
    outfile.create_dataset("hyperparameters/scheduler_factor", data = config["scheduler_factor"])
    outfile.create_dataset("hyperparameters/num_workers", data = config["num_workers"])
    outfile.create_dataset("hyperparameters/seed", data = config["seed"])
    outfile.create_dataset("hyperparameters/freeze_layers", data = config["freeze_layers"])
  
  # Saving best model state dict of TwoNetworks RGBIr.
  torch.save(bestweights, "bestweights_task3.pt")
  print("Time:", time.time() - t0, "sec"); t0 = time.time()
  
  ###########################################################################################
  #####                                    Task 4                                       #####
  ###########################################################################################

  # Setting random seeds for reproducability sake.
  torch.manual_seed(config["seed"])
  torch.cuda.manual_seed(config["seed"])
  np.random.seed(config["seed"])

  print("SingleNetwork RGBIr:"); t0 = time.time()

  # Data augmentations.
  data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          ChannelSelect([0, 1, 2, 3]),  # Selectiong RGB and Ir channels
          transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284])  # Normalizing all channels
      ]),
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          #transforms.RandomHorizontalFlip(), # Commented random flip out to ensure reproducability
          transforms.ToTensor(),
          ChannelSelect([0, 1, 2, 3]),  # Selectiong RGB and Ir channels
          transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284])  # Normalizing all channels
      ]),
  }
  
  # Defining training and validation data sets used for training SingleNetwork RGBIr network (task 4)
  image_datasets={}
  image_datasets['train'] = RainforestDataset(root_dir = data_root_dir, trvaltest=0, transform = data_transforms['train'])
  image_datasets['val']   = RainforestDataset(root_dir = data_root_dir, trvaltest=1, transform = data_transforms['val'])

  # Defining training and validation data loaders
  dataloaders = {}
  dataloaders['train'] = DataLoader(image_datasets['train'], batch_size = config["batchsize_train"],   shuffle = True, num_workers = config["num_workers"]) # Shuffle to False to ensure reproducability by TAs
  dataloaders['val']   = DataLoader(image_datasets['val'],   batch_size = config["batchsize_val"],     shuffle = False, num_workers = config["num_workers"]) 


  pretrained_resnet18 = models.resnet18(pretrained = True)  # Defining instance of pretrained ResNet18 that SingleNetwork RGBIr is build upon
  model = SingleNetwork(pretrained_resnet18, weight_init = "kaiminghe", freeze = config["freeze_layers"]) # Making instance of SingleNetworks RGBIr (as we give in "kaiminghe" weight_init)
  model = model.to(device)  # Sending model to device

  someoptimizer = optim.SGD(model.parameters(), lr = config['lr'])  # Defining optimizer to be used. No momentum was used.

  somelr_scheduler = lr_scheduler.StepLR(someoptimizer, gamma = config['scheduler_factor'], step_size = config['scheduler_stepsize']) # Defining step-learning rate scheduler

  # Training and validating SingleNetwork RGBIr
  best_epoch, best_measure, bestweights, trainlosses, \
  testlosses, testperfs, concat_labels, concat_pred, fnames, classwiseperf \
  = traineval2_model_nocv(dataloaders['train'], dataloaders['val'] ,  model ,  
                          lossfct, someoptimizer, somelr_scheduler, num_epochs= config['maxnumepochs'], 
                          device = device , numcl = config['numcl'] )

  # Saving score data etc. from best epoch and hyperparameter used 
  with h5py.File("diagnostics_task4.h5", "w") as outfile: 
    outfile.create_dataset("best_epoch", data = best_epoch)
    outfile.create_dataset("best_measure", data = best_measure)
    outfile.create_dataset("trainlosses", data = trainlosses)
    outfile.create_dataset("testlosses", data = testlosses)
    outfile.create_dataset("testperfs", data = np.array(testperfs))
    outfile.create_dataset("concat_labels", data = concat_labels)
    outfile.create_dataset("concat_pred", data = concat_pred)
    outfile.create_dataset("classwise_perf", data = classwiseperf)
    dt = h5py.special_dtype(vlen=str)
    outfile.create_dataset("filenames", data = np.array(fnames, dtype = dt))
    outfile.create_dataset("hyperparameters/use_gpu", data = config["use_gpu"])
    outfile.create_dataset("hyperparameters/learning_rate", data = config["lr"])
    outfile.create_dataset("hyperparameters/batchsize_train", data = config["batchsize_train"])
    outfile.create_dataset("hyperparameters/batchsize_val", data = config["batchsize_val"])
    outfile.create_dataset("hyperparameters/maxnumepochs", data = config["maxnumepochs"])
    outfile.create_dataset("hyperparameters/scheduler_stepsize", data = config["scheduler_stepsize"])
    outfile.create_dataset("hyperparameters/scheduler_factor", data = config["scheduler_factor"])
    outfile.create_dataset("hyperparameters/num_workers", data = config["num_workers"])
    outfile.create_dataset("hyperparameters/seed", data = config["seed"])
    outfile.create_dataset("hyperparameters/freeze_layers", data = config["freeze_layers"])

  # Saving best model state dict of SingleNetwork RGBIr.
  torch.save(bestweights, "bestweights_task4.pt")
  print("Time:", time.time() - t0, "sec"); t0 = time.time()


if __name__=='__main__':
  runstuff()  # Run main block
  

