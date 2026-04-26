"""ATSP (Asymmetric TSP) Graph Dataset"""                                                           
                                                                                                    
import numpy as np                                                                                  
import torch                                                                                        

                                                                                                    
class ATSPGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, sparse_factor=-1):                                                
        self.data_file = data_file
        self.sparse_factor = sparse_factor
        self.data = np.load(data_file, allow_pickle=True)
        print(f'Loaded ATSP dataset "{data_file}" with {len(self.data)} samples')                   

    def __len__(self):                                                                              
        return len(self.data)

    def __getitem__(self, idx):
        node_cnt, _, EdgeFeature, SolutionAdj, Objective = self.data[idx]                                                                                                    
        idx_tensor = torch.tensor([idx], dtype=torch.long)
        EdgeFeature = torch.from_numpy(EdgeFeature).float()                                         
        SolutionAdj = torch.from_numpy(SolutionAdj).float()                                         
        Objective = torch.tensor(Objective, dtype=torch.float32)
                                                                                                    
        return (idx_tensor, EdgeFeature, SolutionAdj, node_cnt, Objective) 