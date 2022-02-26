

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

    curcount = 0
    accuracy = 0 
    
    
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

          loss = criterion(outputs, labels.to(device))
          losses.append(loss.item())

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
            
          cpu_out = outputs.to('cpu')
          scores = torch.sigmoid(cpu_out)
          #_, preds = torch.max(cpuout, 1)
          preds = torch.gt(scores, 0.5).float()  # Check when output probability is greater than 50 % for each class (50 % is a hyperparameter)

          labels = labels.float()
          corrects = torch.sum(preds == labels.data)
          accuracy = accuracy * (curcount / float(curcount + labels.shape[0])) + corrects.float() * (curcount / float(curcount + labels.shape[0]))
          curcount += labels.shape[0]
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


def traineval2_model_nocv(dataloader_train, dataloader_test ,  model ,  criterion, optimizer, scheduler, num_epochs, device, numcl):

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

    perfmeasure, testloss, concat_labels, concat_pred, fnames  = evaluate_meanavgprecision(model, dataloader_test, criterion, device, numcl)
    testlosses.append(testloss)
    testperfs.append(perfmeasure)
    
    print('at epoch: ', epoch,' classwise perfmeasure ', perfmeasure)
    
    avgperfmeasure = np.mean(perfmeasure)
    print('at epoch: ', epoch,' avgperfmeasure ', avgperfmeasure)

    if avgperfmeasure > best_measure: #higher is better or lower is better?
      bestweights= model.state_dict()
      ####################################
      #TODO track current best performance measure and epoch

      best_measure = avgperfmeasure
      best_epoch   = epoch

      #TODO save your scores
      ####################################
      best_scores = concat_labels
      best_labels = concat_labels
      best_fnames = fnames

  return best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs, best_labels, best_scores, best_fnames, perfmeasure


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


def evaluate_diagnostics(classwise_perf, concat_labels, concat_pred, root_dir, \
                        fnames, trainlosses, testlosses, testperfs, num_epochs):
  
  idx_highAP = np.argmax(classwise_perf)
  pred   = concat_pred[:, idx_highAP] 
  label  = concat_labels[:, idx_highAP]

  idx_sorted = np.argsort(pred)

  pred  = pred[idx_sorted]
  label = label[idx_sorted]
  fnames = fnames[idx_sorted]

  label_names, ncls = get_classes_list()
  class_num = np.arange(ncls)
  
  print("------------------------------------------------")
  print(f"Highest AP class: {label_names[idx_highAP]}")
  print(f"Top-10: {pred[:10]}") 
  print(f"Bottom-10: {pred[:10]}") 
  print("------------------------------------------------")


  fonts = {
    "font.family": "serif",
    "axes.labelsize": 20,
    "font.size": 20,
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20
    }
  plt.rcParams.update(fonts)

  fig, ax = plt.subplots(figsize = (16, 10))
  
  ax.plot(class_num, classwise_perf)
  ax.scatter(idx_highAP, np.max(classwise_perf), label = "Max AP")
  ax.set_xticks(class_num)
  ax.set_xticklabels(label_names, rotation = "vertical")
  ax.set_xlabel("Class labels")
  ax.set_ylabel("Average precision")
  ax.grid()
  ax.legend()


  ts = np.linspace(0, np.max(pred), 10)

  tailac = tail_accuracy(ts, concat_pred, concat_labels)

  fig, ax = plt.subplots(figsize = (16, 10))
  ax.plot(ts, tailac)
  ax.set_xlabel("Acceptance threshold t")
  ax.set_ylabel("Tail accuracy")
  ax.grid()
  ax.legend([f"AP({label_names[i]}) = {classwise_perf[i]}" for i in range(ncls)])



  figt, axt = plt.subplots(2, 5, figsize = (16, 10))  # Figure and axis for top-10 images
  figb, axb = plt.subplots(2, 5, figsize = (16, 10))  # Figure and axis for bottom-10 images
  
  figt.suptitle(f"Top-10 | Label: {label_names[idx_highAP]}")
  figb.suptitle(f"Bottom-10 | Label: {label_names[idx_highAP]}")
  for i in range(2):
    for j in range(5):
      idx = i * 2 + j 

      with PIL.Image.open(root_dir + "train-tif-v2/" + fnames[idx] + ".tif") as img:
        img = np.asarray(img)
        axt[i, j].imshow(img[..., :3])

        classname = label_names[label[idx].astype(bool)]
        axt[i, j].set_title(f"Score: {pred[idx]} | Label: {classname}")
        plt.axis("off")

      with PIL.Image.open(root_dir + "train-tif-v2/" + fnames[-idx] + ".tif") as img:
        img = np.asarray(img)
        axb[i, j].imshow(img[..., :3])

        classname = label_names[label[-idx].astype(bool)]
        axb[i, j].set_title(f"Score: {pred[-idx]} | Label: {classname}")
        plt.axis("off")

  fig, ax = plt.subplots(2, 1, figsize = (16, 10))
  ax[0].plot(np.arange(num_epochs), trainlosses, label = "Train")
  ax[0].plot(np.arange(num_epochs), testlosses,  label = "Test")
  ax[0].set_xlabel("Epoch")
  ax[0].set_ylabel("Loss")
  ax[0].grid()
  ax[0].legend()

  ax[1].plot(np.arange(num_epochs), np.mean(testperfs, axis = 1))
  ax[1].plot(np.arange(num_epochs), testperfs)
  ax[1].set_xlabel("Epoch")
  ax[1].set_ylabel("AP")
  ax[1].grid()
  legend = ["mAP"]
  ax[1].legend(legend + [f"AP({label_names[i]})" for i in range(ncls)])

  plt.show()



def tail_accuracy(ts, pred, label):

  correct = pred[:, :, np.newaxis] > ts[np.newaxis, np.newaxis, :]
  correct *= pred == label[:, :, np.newaxis]

  tailac = np.sum(correct , axis = 1)
  tailac /= np.sum(pred[:, :, np.newaxis] > ts[np.newaxis, np.newaxis, :], axis = 1)

  return tailac



def runstuff():
  config = dict()
  config['use_gpu'] = True #True #TODO change this to True for training on the cluster
  config['lr'] = 0.005
  config['batchsize_train'] = 16
  config['batchsize_val'] = 64
  config['maxnumepochs'] = 35
  config['scheduler_stepsize'] = 10
  config['scheduler_factor'] = 0.3
  config["num_workers"]      = 10

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
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          ChannelSelect(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
  dataloaders['train'] = DataLoader(image_datasets['train'], batch_size = config["batchsize_train"],   shuffle = False, num_workers = config["num_workers"]) # Shuffle to False to ensure reproducability by TAs
  dataloaders['val']   = DataLoader(image_datasets['val'],   batch_size = config["batchsize_val"],     shuffle = False, num_workers = config["num_workers"]) 

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

  model = SingleNetwork(pretrained_resnet18)#, weight_init = "kaiminghe")
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
  """
  best_epoch, best_measure, bestweights, trainlosses, \
  testlosses, testperfs, concat_labels, concat_pred, fnames, classwiseperf \
  = traineval2_model_nocv(dataloaders['train'], dataloaders['val'] ,  model ,  lossfct, someoptimizer, somelr_scheduler, num_epochs= config['maxnumepochs'], device = device , numcl = config['numcl'] )

  with h5py.File("diagnostics_task1.h5", "w") as outfile: 
    outfile.create_dataset("best_epoch", data = best_epoch)
    outfile.create_dataset("best_measure", data = best_measure)
    outfile.create_dataset("trainlosses", data = trainlosses)
    outfile.create_dataset("testlosses", data = testlosses)
    outfile.create_dataset("testperfs", data = np.array(testperfs))
    outfile.create_dataset("concat_labels", data = concat_labels)
    outfile.create_dataset("concat_pred", data = concat_pred)
    outfile.create_dataset("classwise_perf", data = classwiseperf)

  np.save("fnames_task1", np.array(fnames))
  

  torch.save(bestweights, "bestweights_task1.pt")
  """
  ###########################################################################################
  #####                                    Task 3                                       #####
  ###########################################################################################


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
  dataloaders['train'] = DataLoader(image_datasets['train'], batch_size = config["batchsize_train"],   shuffle = False, num_workers = config["num_workers"]) # Shuffle to False to ensure reproducability by TAs
  dataloaders['val']   = DataLoader(image_datasets['val'],   batch_size = config["batchsize_val"],     shuffle = False, num_workers = config["num_workers"]) 


  model = TwoNetworks(pretrained_resnet18, pretrained_resnet18)
  model = model.to(device)

  best_epoch, best_measure, bestweights, trainlosses, \
  testlosses, testperfs, concat_labels, concat_pred, fnames, classwiseperf \
  = traineval2_model_nocv(dataloaders['train'], dataloaders['val'] ,  model ,  lossfct, someoptimizer, somelr_scheduler, num_epochs= config['maxnumepochs'], device = device , numcl = config['numcl'] )

  with h5py.File("diagnostics_task3.h5", "w") as outfile: 
    outfile.create_dataset("best_epoch", data = best_epoch)
    outfile.create_dataset("best_measure", data = best_measure)
    outfile.create_dataset("trainlosses", data = trainlosses)
    outfile.create_dataset("testlosses", data = testlosses)
    outfile.create_dataset("testperfs", data = np.array(testperfs))
    outfile.create_dataset("concat_labels", data = concat_labels)
    outfile.create_dataset("concat_pred", data = concat_pred)
    outfile.create_dataset("classwise_perf", data = classwiseperf)

  np.save("fnames_task3", np.array(fnames))
  

  torch.save(bestweights, "bestweights_task3.pt")

  ###########################################################################################
  #####                                    Task 4                                       #####
  ###########################################################################################


  model = SingleNetwork(pretrained_resnet18, weight_init = "kaiminghe")
  model = model.to(device)

  best_epoch, best_measure, bestweights, trainlosses, \
  testlosses, testperfs, concat_labels, concat_pred, fnames, classwiseperf \
  = traineval2_model_nocv(dataloaders['train'], dataloaders['val'] ,  model ,  lossfct, someoptimizer, somelr_scheduler, num_epochs= config['maxnumepochs'], device = device , numcl = config['numcl'] )

  with h5py.File("diagnostics_task4.h5", "w") as outfile: 
    outfile.create_dataset("best_epoch", data = best_epoch)
    outfile.create_dataset("best_measure", data = best_measure)
    outfile.create_dataset("trainlosses", data = trainlosses)
    outfile.create_dataset("testlosses", data = testlosses)
    outfile.create_dataset("testperfs", data = np.array(testperfs))
    outfile.create_dataset("concat_labels", data = concat_labels)
    outfile.create_dataset("concat_pred", data = concat_pred)
    outfile.create_dataset("classwise_perf", data = classwiseperf)

  np.save("fnames_task4", np.array(fnames))
  

  torch.save(bestweights, "bestweights_task4.pt")


if __name__=='__main__':

  runstuff()
  

