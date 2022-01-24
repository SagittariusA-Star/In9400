import os,sys,numpy as np
from turtle import shape
import matplotlib.pyplot as plt
import torch

import time

def forloopdists(feats, protos):

  #YOUR implementation here
  N, D = feats.shape
  P, D = protos.shape

  result = np.zeros((N, P))
  for i in range(N):
    for j in range(P):
      result[i, j] = np.linalg.norm(feats[i, :] - protos[j, :]) ** 2
  return result

def numpydists(feats, protos):
  #YOUR implementation here
  
  diff = feats[:, np.newaxis, :] - protos[np.newaxis, :, :]
  #norm_sq = np.sum(diff ** 2, axis = -1)
  norm_sq = np.linalg.norm(diff, axis = -1) ** 2

  return norm_sq

def pytorchdists(feats0, protos0, device):
  #YOUR implementation here

  X = torch.from_numpy(feats0)
  T = torch.from_numpy(protos0)
  diff = torch.sub(X.unsqueeze(1), T.unsqueeze(0))
  #norm_sq = torch.sum(diff * diff, -1)
  norm_sq = torch.norm(diff, p = 2, dim = 2) ** 2
  return norm_sq.numpy()

def run():

  ########
  ##
  ## if you have less than 8 gbyte, then reduce from 250k
  ##
  ###############
  
  #feats=np.random.normal(size=(250000,300)) #5000 instead of 250k for forloopdists
  #protos=np.random.normal(size=(500,300))

  feats=np.random.normal(size=(2500,300)) #5000 instead of 250k for forloopdists
  protos=np.random.normal(size=(500,300))


  since = time.time()
  dists0 = forloopdists(feats, protos)
  time_elapsed=float(time.time()) - float(since)
  print('Loop comp complete in {:.3f}s'.format( time_elapsed ))
  print(dists0.shape)

  #device=torch.device('cpu')
  device=torch.device('cuda:0')
  since = time.time()

  dists1=pytorchdists(feats,protos,device)


  time_elapsed=float(time.time()) - float(since)

  print('Pytorch comp complete in {:.3f}s'.format( time_elapsed ))
  print(dists1.shape)

  #print('df0',np.max(np.abs(dists1-dists0)))

  since = time.time()

  dists2=numpydists(feats,protos)


  time_elapsed=float(time.time()) - float(since)

  print('Numpy comp complete in {:.3f}s'.format( time_elapsed ))

  print(dists2.shape)

  print('df', np.max(np.abs(dists1-dists2)))
  print(np.allclose(dists0, dists1))
  print(np.allclose(dists0, dists2))


def kmeans():
  M = 100
  N = 50
  D = 2
  P = 5

  np.random.seed(4877)
  means = np.random.uniform(0, 10, size = (5, D))
  stds  = np.random.uniform(0, 0.5, size = 5)

  print(means)
  print(stds)

  X = np.zeros((N, D))

  for i in range(N):
    idx = np.random.randint(0, 5)
    X[i, :] = np.random.normal(means[idx, :], stds[idx], D)

  fig, ax = plt.subplots()
  ax.scatter(X[:, 0], X[:, 1], s = 5, label = "Data")
  ax.scatter(means[:, 0], means[:, 1], label = "True means")
  
  plt.show()
  


if __name__=='__main__':
  #run()
  kmeans()
