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
        node_cnt, node_feature, edge_feature, solution_adj, objective = self.data[idx]                      
                                                                                          
        solution_adj = torch.from_numpy(solution_adj).float()
        objective = torch.tensor(objective, dtype=torch.float32)                                            
                        
        if node_feature is not None:
            node_feature = torch.from_numpy(node_feature).float()
        else:
            # node_feature = torch.full((node_cnt, 1), float('nan'), dtype=torch.float32)
            node_feature = torch.ones((node_cnt, 1), dtype=torch.float32)    
        if edge_feature is not None:                                                                        
            edge_feature = torch.from_numpy(edge_feature).float()
        else:
            # edge_feature = torch.full((node_cnt, node_cnt), float('nan'), dtype=torch.float32)
            edge_feature = torch.zeros((node_cnt, node_cnt), dtype=torch.float32)

        return node_cnt, node_feature, edge_feature, solution_adj, objective
