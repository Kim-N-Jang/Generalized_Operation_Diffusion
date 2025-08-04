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
      pt, machine, adj = self.data[idx]
      idx = torch.tensor([idx], dtype=torch.long)
      pt = torch.from_numpy(pt).float().flatten().unsqueeze(1)  # JM (Node수)
      machine = torch.from_numpy(machine).long().flatten() # JM
      adj = torch.from_numpy(adj).float() # JM x JM      
      return (idx, pt, adj, machine) 
    # idx, Node 정보 (Duration pt, Machine), 정답 Adj ,  Machine (Decoded Solution) 
    # To do : Machine -> Operation Sequence 변경, Features (pt만 1차원으로 변경)
