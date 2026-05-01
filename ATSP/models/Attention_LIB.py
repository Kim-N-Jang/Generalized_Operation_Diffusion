import torch
import torch.nn as nn

class MixedScore_MultiHeadAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']
        hidden_dim = model_params['hidden_dim']

        self.Wz1 = nn.Linear(hidden_dim, head_num * qkv_dim, bias=False)
        self.Wz2 = nn.Linear(hidden_dim, head_num * qkv_dim, bias=False)
        
    def reshape_by_heads(self, edge, head_num):

        batch_s = edge.size(0)
        n = edge.size(1)

        edge_reshaped = edge.reshape(batch_s, n, n, head_num, -1)
        # shape: (batch, n, head_num, key_dim)

        edge_transposed = edge_reshaped.permute(0,3,1,2,4)
        # shape: (batch, head_num, n, key_dim)

        return edge_transposed

    def forward(self, q, k, v, z, rank2_ninf_mask=None, rank3_ninf_mask=None):
        # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
        # k,v shape: (batch, head_num, problem, key_dim)
        # z shape: (batch, problem, problem, emb_dim)
        # rank2_ninf_mask.shape: (batch, problem)
        # rank3_ninf_mask.shape: (batch, group, problem)

        batch_s = q.size(0)
        head_num = q.size(1)
        n = q.size(2)
        key_dim = q.size(3)

        input_s = k.size(2)

        ###################################################################
        q_exp = q.unsqueeze(3)  # (batch, head_num, problem, 1, key_dim)      
        k_exp = k.unsqueeze(2)  # (batch, head_num, 1, problem, key_dim)

        z1 = self.reshape_by_heads(self.Wz1(z), head_num) # shape: (batch, head_num, problem, problem, key_dim)
        z2 = self.reshape_by_heads(self.Wz2(z), head_num)

        qz = q_exp + z1  # (B, H, N, N, D)
        kz = k_exp + z2  # (B, H, N, N, D)
        ###################################################################

        #score
        score = torch.einsum('bhnmd,bhnmd->bhnm', qz, kz)

        score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

        if rank2_ninf_mask is not None:
            score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
        if rank3_ninf_mask is not None:
            score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

        weights = nn.Softmax(dim=3)(score_scaled)
        # shape: (batch, head_num, n, problem)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, n, key_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, n, head_num, key_dim)

        out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
        # shape: (batch, n, head_num*key_dim)

        return out_concat



