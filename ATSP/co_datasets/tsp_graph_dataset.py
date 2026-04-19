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
      Pt, Machine, JobAdj, MachineAdj, Makespan = self.data[idx]
      idx = torch.tensor([idx], dtype=torch.long)
      Pt = torch.from_numpy(Pt).float().flatten().unsqueeze(1)  # JM (Node수)
      Machine = torch.from_numpy(Machine).long().flatten() # JM
      JobAdj = torch.from_numpy(JobAdj).float() # JM x JM
      MachineAdj = torch.from_numpy(MachineAdj).float() # JM x JM
      Makespan = torch.tensor(Makespan)
      return (idx, Pt, JobAdj, MachineAdj, Machine, Makespan)
    # idx, Node 정보 (Pt), 정답 Adj ,  Machine 
    # To do : Operation Sequence(Decoded Solution) 추가
