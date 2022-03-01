

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

def train_epoch(model, trainloader, criterion, device, optimizer):

    #TODO model.train() or model.eval()? 
    ###############
    model.train()
    ###############
 
    losses = []
    for batch_idx, data in enumerate(trainloader):
        if (batch_idx % 100 == 0) and (batch_idx >= 100):
          print('at batchidx',batch_idx)

        # TODO calculate the loss from your minibatch.
        # If you are using the TwoNetworks class you will need to copy the infrared
        # channel before feeding it into your model. 
        
        ######################################
        images = data["image"].to(device)
        labels = data["label"].to(device)

        if isinstance(model, TwoNetworks):
          IRchannel  = images[:, 3, ...].unsqueeze(1)   # selecting IR channel
          RGBchannel = images[:, :3, ...]               # selecting RGB channel
          output = model(RGBchannel, IRchannel)
        elif isinstance(model, SingleNetwork) and not model.weight_init:
          images = images[:, :3, ...]
          output = model(images)
        else: 
          output = model(images)
        
        #print("print train", output.cpu())
        
        loss = criterion(output.float(), labels.float())
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ######################################

    return np.mean(losses)


def evaluate_meanavgprecision(model, dataloader, criterion, device, numcl):

    #TODO model.train() or model.eval()?

    ###################
    model.eval()
    ###################

    #curcount = 0
    #accuracy = 0 
    
    
    concat_pred   = np.empty((0, numcl)) #prediction scores for each class. each numpy array is a list of scores. one score per image
    concat_labels = np.empty((0, numcl)) #labels scores for each class. each numpy array is a list of labels. one label per image
    avgprecs      = np.zeros(numcl)  # average precision for each class
    fnames        = [] # filenames as they come out of the dataloader

    with torch.no_grad():
      losses = []
      for batch_idx, data in enumerate(dataloader):
          if (batch_idx%100==0) and (batch_idx>=100):
            print('at val batchindex: ', batch_idx)
      
          images = data['image'].to(device)        
          labels = data['label']

          ######################################
          # This was an accuracy computation

          if isinstance(model, TwoNetworks):
            IRchannel  = images[:, 3, ...].unsqueeze(1)   # selecting IR channel
            RGBchannel = images[:, :3, ...]               # selecting RGB channel
            outputs = model.forward(RGBchannel, IRchannel)
          elif isinstance(model, SingleNetwork) and not model.weight_init:
            images = images[:, :3, ...]
            outputs = model.forward(images)
          else: 
            outputs = model.forward(images)
          
          loss = criterion(outputs, labels.to(device))
          losses.append(loss.item())

          cpu_out = outputs.to('cpu')
          scores = torch.sigmoid(cpu_out)   # Transform model output from real numbers to probability space [0, 1]
          
          #_, preds = torch.max(cpuout, 1)
          #preds = torch.gt(scores, 0.5).float()  # Check when output probability is greater than 50 % for each class (50 % is a hyperparameter)

          labels = labels.float()
          #corrects = torch.sum(preds == labels.data)
          #accuracy = accuracy * (curcount / float(curcount + labels.shape[0])) + corrects.float() * (curcount / float(curcount + labels.shape[0]))
          #curcount += labels.shape[0]
          ######################################

          # TODO: collect scores, labels, filenames
          
          ######################################
          concat_pred   = np.concatenate((concat_pred,   scores.float()),  axis = 0)   # sklearn.metrics.average_precision_score takes confidence scores, i.e. network confidence output not thresholded prediction (?)
          concat_labels = np.concatenate((concat_labels, labels.float()), axis = 0)
          fnames.append(data["filename"])

          ######################################

    fnames = [name for i in fnames for name in i] # Nested list to continuus list
          
    for c in range(numcl): 
      #######################################  
      avgprecs[c] = sklearn.metrics.average_precision_score(concat_labels[:, c], concat_pred[:, c]) # TODO, nope it is not sklearn.metrics.precision_score
      
      #######################################  
      
    return avgprecs, np.mean(losses), concat_labels, concat_pred, fnames


def traineval2_model_nocv(dataloader_train, dataloader_test, model,  criterion, optimizer, scheduler, num_epochs, device, numcl):

  best_measure = 0
  best_epoch = -1

  trainlosses = []
  testlosses  = []
  testperfs   = []
  
  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)


    avgloss = train_epoch(model,  dataloader_train,  criterion,  device , optimizer )
    trainlosses.append(avgloss)
    
    if scheduler is not None:
      scheduler.step()
      ##################################
      print("Current learning rate:", scheduler.get_last_lr()[0])
      ##################################

    perfmeasure, testloss, concat_labels, concat_pred, fnames  = evaluate_meanavgprecision(model, dataloader_test, criterion, device, numcl)
    testlosses.append(testloss)
    testperfs.append(perfmeasure)
    
    print('at epoch: ', epoch,' classwise perfmeasure ', perfmeasure)
    
    avgperfmeasure = np.mean(perfmeasure)
    print('at epoch: ', epoch,' avgperfmeasure: ', avgperfmeasure, "test loss:", testloss)

    if avgperfmeasure > best_measure: #higher is better or lower is better?
      bestweights = model.state_dict()
      ####################################
      #TODO track current best performance measure and epoch

      best_measure = avgperfmeasure
      best_epoch   = epoch
      best_perfmeasure = perfmeasure 

      #TODO save your scores
      ####################################
      best_scores = concat_pred
      best_labels = concat_labels
      best_fnames = fnames


  return best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs, best_labels, best_scores, best_fnames, best_perfmeasure


class yourloss(nn.modules.loss._Loss):

    def __init__(self, reduction: str = 'mean') -> None:
        #TODO
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
      #TODO
      ################################################################
      loss_criterion = nn.BCEWithLogitsLoss(reduction = self.reduction)
      loss = loss_criterion(input_, target)
      ################################################################
      return loss


def runstuff():
  config = dict()
  config['use_gpu'] = True #True #TODO change this to True for training on the cluster
  config['lr'] = 0.05
  config['batchsize_train'] = 16
  config['batchsize_val'] = 64
  config['maxnumepochs'] = 35
  config['scheduler_stepsize'] = 10
  config['scheduler_factor'] = 0.3
  config["num_workers"]      = 1
  config["seed"]             = 774663
  config["freeze_layers"]    = False # Whether to freeze middle layers of pretrained networks

  ########################################
  torch.manual_seed(config["seed"])
  torch.cuda.manual_seed(config["seed"])
  ########################################

  # This is a dataset property.
  config['numcl'] = 17

  # Data augmentations.
  data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          ChannelSelect(),
          transforms.Normalize([0.7476, 0.6534, 0.4757], [0.1677, 0.1828, 0.2137])
      ]),
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          ChannelSelect(),
          transforms.Normalize([0.7476, 0.6534, 0.4757], [0.1677, 0.1828, 0.2137])
      ]),
  }


  # Datasets
  
  ##########################
  data_root_dir = "../../data/" 
  ##########################
  
  image_datasets={}
  image_datasets['train'] = RainforestDataset(root_dir = data_root_dir, trvaltest=0, transform = data_transforms['train'])
  image_datasets['val']   = RainforestDataset(root_dir = data_root_dir, trvaltest=1, transform = data_transforms['val'])

  
  # Dataloaders
  #TODO use num_workers=1

  ###################################
  dataloaders = {}
  dataloaders['train'] = DataLoader(image_datasets['train'], batch_size = config["batchsize_train"],   shuffle = True, num_workers = config["num_workers"]) # Shuffle to False to ensure reproducability by TAs
  dataloaders['val']   = DataLoader(image_datasets['val'],   batch_size = config["batchsize_val"],     shuffle = True, num_workers = config["num_workers"]) 

  ###################################
  # Device
  if True == config['use_gpu']:
      device= torch.device('cuda:0')
  else:
      device= torch.device('cpu')

  # Model
  # TODO create an instance of the network that you want to use.
  
  ####################################
  pretrained_resnet18 = models.resnet18(pretrained = True)

  model = SingleNetwork(pretrained_resnet18, freeze = config["freeze_layers"])
  ####################################

  model = model.to(device)

  lossfct = yourloss()
  
  #######################################################################
  #TODO
  # Observe that all parameters are being optimized

  someoptimizer = optim.SGD(model.parameters(), lr = config['lr'])

  # Decay LR by a factor of 0.3 every X epochs
  #TODO

  somelr_scheduler = lr_scheduler.StepLR(someoptimizer, gamma = config['scheduler_factor'], step_size = config['scheduler_stepsize'])

  #######################################################################


  ###########################################################################################
  #####                                     Task 1                                      #####
  ###########################################################################################
  
  print("SingleNetwork RGB:"); t0 = time.time()
  best_epoch, best_measure, bestweights, trainlosses, \
  testlosses, testperfs, concat_labels, concat_pred, fnames, classwiseperf \
  = traineval2_model_nocv(dataloaders['train'], dataloaders['val'], model,  lossfct, someoptimizer, somelr_scheduler, num_epochs= config['maxnumepochs'], device = device , numcl = config['numcl'] )
  
  
  with h5py.File("diagnostics_task1_higherlr.h5", "w") as outfile: 
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

  torch.save(bestweights, "bestweights_task1_higherlr.pt")
  print("Time:", time.time() - t0, "sec"); t0 = time.time()
  
  
  ###########################################################################################
  #####                                    Task 3                                       #####
  ###########################################################################################
  
  
  print("TwoNetworks:"); t0 = time.time()

  data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          ChannelSelect([0, 1, 2, 3]),
          transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284])
      ]),
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          ChannelSelect([0, 1, 2, 3]),
          transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284])
      ]),
  }
  
  image_datasets={}
  image_datasets['train'] = RainforestDataset(root_dir = data_root_dir, trvaltest=0, transform = data_transforms['train'])
  image_datasets['val']   = RainforestDataset(root_dir = data_root_dir, trvaltest=1, transform = data_transforms['val'])

  
  # Dataloaders
  #TODO use num_workers=1

  ###################################
  dataloaders = {}
  dataloaders['train'] = DataLoader(image_datasets['train'], batch_size = config["batchsize_train"],   shuffle = True, num_workers = config["num_workers"]) # Shuffle to False to ensure reproducability by TAs
  dataloaders['val']   = DataLoader(image_datasets['val'],   batch_size = config["batchsize_val"],     shuffle = True, num_workers = config["num_workers"]) 

  pretrained_resnet18_1 = models.resnet18(pretrained = True)
  pretrained_resnet18_2 = models.resnet18(pretrained = True)
  model = TwoNetworks(pretrained_resnet18_1, pretrained_resnet18_2, freeze = config["freeze_layers"])
  model = model.to(device)

  someoptimizer = optim.SGD(model.parameters(), lr = config['lr'])

  somelr_scheduler = lr_scheduler.StepLR(someoptimizer, gamma = config['scheduler_factor'], step_size = config['scheduler_stepsize'])

  best_epoch, best_measure, bestweights, trainlosses, \
  testlosses, testperfs, concat_labels, concat_pred, fnames, classwiseperf \
  = traineval2_model_nocv(dataloaders['train'], dataloaders['val'] ,  model ,  lossfct, someoptimizer, somelr_scheduler, num_epochs= config['maxnumepochs'], device = device , numcl = config['numcl'] )

  with h5py.File("diagnostics_task3_higherlr.h5", "w") as outfile: 
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
  

  torch.save(bestweights, "bestweights_task3_higherlr.pt")
  print("Time:", time.time() - t0, "sec"); t0 = time.time()
  
  ###########################################################################################
  #####                                    Task 4                                       #####
  ###########################################################################################

  print("SingleNetwork RGBIr:"); t0 = time.time()

  data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          ChannelSelect([0, 1, 2, 3]),
          transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284])
      ]),
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          ChannelSelect([0, 1, 2, 3]),
          transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284])
      ]),
  }
  
  image_datasets={}
  image_datasets['train'] = RainforestDataset(root_dir = data_root_dir, trvaltest=0, transform = data_transforms['train'])
  image_datasets['val']   = RainforestDataset(root_dir = data_root_dir, trvaltest=1, transform = data_transforms['val'])

  
  # Dataloaders
  #TODO use num_workers=1

  ###################################
  dataloaders = {}
  dataloaders['train'] = DataLoader(image_datasets['train'], batch_size = config["batchsize_train"],   shuffle = True, num_workers = config["num_workers"]) # Shuffle to False to ensure reproducability by TAs
  dataloaders['val']   = DataLoader(image_datasets['val'],   batch_size = config["batchsize_val"],     shuffle = True, num_workers = config["num_workers"]) 


  pretrained_resnet18 = models.resnet18(pretrained = True)
  model = SingleNetwork(pretrained_resnet18, weight_init = "kaiminghe", freeze = config["freeze_layers"])
  model = model.to(device)

  someoptimizer = optim.SGD(model.parameters(), lr = config['lr'])

  somelr_scheduler = lr_scheduler.StepLR(someoptimizer, gamma = config['scheduler_factor'], step_size = config['scheduler_stepsize'])

  best_epoch, best_measure, bestweights, trainlosses, \
  testlosses, testperfs, concat_labels, concat_pred, fnames, classwiseperf \
  = traineval2_model_nocv(dataloaders['train'], dataloaders['val'] ,  model ,  lossfct, someoptimizer, somelr_scheduler, num_epochs= config['maxnumepochs'], device = device , numcl = config['numcl'] )

  with h5py.File("diagnostics_task4_higherlr.h5", "w") as outfile: 
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


  torch.save(bestweights, "bestweights_task4_higherlr.pt")
  print("Time:", time.time() - t0, "sec"); t0 = time.time()


if __name__=='__main__':

  runstuff()
  

