import os,sys,numpy as np
from turtle import shape
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
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

  #X = torch.from_numpy(feats0)
  #T = torch.from_numpy(protos0)
  diff = torch.sub(feats0.unsqueeze(1), protos0.unsqueeze(0))
  #norm_sq = torch.sum(diff * diff, -1)
  norm_sq = torch.norm(diff, p = 2, dim = 2) ** 2
  return norm_sq

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

def get_mean_and_std(P, D, seed = 4877):
  np.random.seed(seed)
  means = np.random.uniform(0, 10, size = (P, D))
  stds  = np.random.uniform(0, 1, size = P)
  return means, stds

def get_data(N = 50, P = 5, seed = 4877):
  """Generate Gaussian clusters

  Parameters
  ----------
  N : int, optional
      Number of samples, by default 50
  P : int, optional
      Number of clusters, by default 2
  Returns
  -------
  numpy.ndarray
      Data matrix of shape (N, 2), i.e. N samples of (x, y)-coordinates.
  """
  D = 2
  means, stds = get_mean_and_std(P, D, seed)

  X = np.zeros((N, D))

  for i in range(N):
    idx = np.random.randint(0, P)
    X[i, :] = np.random.normal(means[idx, :], stds[idx], D)

  return X
  
def kmeans(X, P = 5, M = 10, eps = 0.1, seed = 12345):
  X = torch.from_numpy(X)
  N, D = X.shape
  torch.manual_seed(seed)
  
  #T    = torch.empty((P, D)).uniform_(-12, 12)
  T    = torch.empty((P, D))
  T = X[torch.randint(0, N, (P,)), :]
  Ts = []
  T_init = T.clone()
  device = torch.device('cpu')
  #device = torch.device('cuda:0')
  
  for i in tqdm(range(M)):
    T_old = T.clone()
    dist = pytorchdists(X, T, device)
    idx_j = torch.argmin(dist, dim = 1)

    for j in range(P):
      idx_i = idx_j == j

      if torch.any(idx_i):
        T[j, :] = torch.mean(X[idx_i, :], 0)
      else:
        print("No update!")
      
        
      #print(T[j, :])
    #print(torch.abs(T - T_old))
    Ts.append(T.numpy())
    if torch.all(torch.abs(T - T_old) <= eps):
      print(torch.abs(T - T_old))
      break

  return T, T_init, np.array(Ts)

  

if __name__=='__main__':
  #run()
  N = 1000
  P1 = 5
  D = 2
  seed = 4877
  X = get_data(N, P1, seed)

  means, stds = get_mean_and_std(P1, D, seed)

  P2 = 5
  T, T_init, Ts = kmeans(X, P2, M = 50, eps = 0.001, seed = 662552)

  T = T.numpy()
  T_init = T_init.numpy()


  fig, ax = plt.subplots()
  ax.scatter(X[:, 0], X[:, 1], label = "Data", s = 1, alpha = 0.8)
  ax.scatter(means[:, 0], means[:, 1], label = "True means", s = 10, alpha = 0.8)
  ax.scatter(T_init[:, 0], T_init[:, 1], label = "k-means init", s = 5, alpha = 0.8)
  ax.scatter(T[:, 0], T[:, 1], label = "k-means final", s = 5, alpha = 0.8)

  ax.legend(loc = 2)

  #for i in range(Ts.shape[0]):
  #  ax.scatter(Ts[i,:, 0], Ts[i, :, 1], label = "step", s = 5, alpha = 0.8, c = "m")



  plt.show()
