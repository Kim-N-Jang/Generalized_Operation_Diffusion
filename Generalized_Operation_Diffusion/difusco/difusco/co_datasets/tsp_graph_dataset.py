"""TSP (Traveling Salesman Problem) Graph Dataset"""

import numpy as np
import torch

from sklearn.neighbors import KDTree
from torch_geometric.data import Data as GraphData


class TSPGraphDataset(torch.utils.data.Dataset):
  def __init__(self, data_file, sparse_factor=-1):
    self.data_file = data_file
    self.sparse_factor = sparse_factor
    self.data = np.load(data_file , allow_pickle=True)
    print(f'Loaded "{data_file}" with {len(self.data)} lines')

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    if self.sparse_factor <= 0:
      time, machine, adj = self.data[idx]
      idx = torch.tensor([idx], dtype=torch.long)
      time = torch.from_numpy(time).float().flatten() # JM (Node수) 
      machine = torch.from_numpy(machine).long().flatten() # JM
      adj = torch.from_numpy(adj).float() # JM x JM

      features = torch.stack((time, machine), dim=1) 
      return (idx, features, adj, machine)
