import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.nn import timestep_embedding

from .Attention_LIB import MixedScore_MultiHeadAttention


class Transformer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.encoder = Encoder(**model_params)
        self.decoder = Decoder(**model_params)

    def pre_forward(self, node_input, edge_input):
        return self.encoder(node_input, edge_input)

    def forward(self, Node, NoisedGraph, MachineGraph, timesteps):
        return self.decoder(Node, NoisedGraph, MachineGraph, timesteps)

        
class Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        encoder_layer_num = model_params['encoder_layer_num']
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])
        hidden_dim = model_params['hidden_dim']

        self.node_idx_projection = nn.Linear(1, hidden_dim)
        self.edge_mtrx_projection = nn.Linear(1, hidden_dim)

    def compute_normalized_matrices(self, data):

        B, N, _ = data.shape

        # 배치마다 min, max 계산 (dim=(1,2)로 전체 N x N에서)
        min_vals = data.view(B, -1).min(dim=1)[0].view(B, 1, 1)
        max_vals = data.view(B, -1).max(dim=1)[0].view(B, 1, 1)

        # 0으로 나눔 방지 (max == min일 경우)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0

        # 정규화
        scaled_data = (data - min_vals) / range_vals

        return scaled_data

    def forward(self, node_input, edge_input):
        # node_input: (batch, cnt, info)
        # edge_input: (batch, cnt, cnt)
        
        if node_input is None:
            batch_size, num_nodes, _ = edge_input.shape
        # elif edge_input is None:
        else:
            batch_size, num_nodes, _ = node_input.shape
        
        # 일단 노드 정보 없음만 가정
        out = self.node_idx_projection(torch.rand((batch_size, num_nodes, 1),device=edge_input.device))

        scaled_data = self.compute_normalized_matrices(edge_input)

        edge_emb = self.edge_mtrx_projection(scaled_data.float().unsqueeze(-1))

        for layer in self.layers:
            out = layer(out, edge_emb)

        return out

class Decoder(nn.Module):
    """Configurable Decoder
  """
    def __init__(self, **model_params):
        super().__init__()

        self.sparse = model_params['sparse']
        self.node_feature_only = model_params['node_feature_only']
        hidden_dim = model_params['hidden_dim']
        use_activation_checkpoint = model_params['use_activation_checkpoint']
        self.hidden_dim = hidden_dim
        self.head_num = model_params['head_num']
        time_embed_dim = hidden_dim // 2
        n_layers = model_params['n_layers']
        self.node_embed = nn.Linear(hidden_dim, hidden_dim)
        self.edge_embed = nn.Linear(1, hidden_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.layers = nn.ModuleList([
            EncoderLayer(**model_params)
            for _ in range(n_layers)
        ])

        self.time_embed_layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    time_embed_dim,
                    hidden_dim,
                ),
            ) for _ in range(n_layers)
        ])

        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_logit_scale = model_params.get('final_logit_scale', 1.0)
        self.final_softcap = model_params.get('final_softcap', 50.0)
        self.use_activation_checkpoint = use_activation_checkpoint

    def forward(self, encoded_nodes, NoisedGraph, solution_adj, timesteps):
        """
    Args:
        Node(x): Input node coordinates (B x V x H)
        NoisedGraph: Noised graph adjacency matrices (B x V x V)
        solution_adj: solutions graph adjacency matrices (B x V x V)
        timesteps: Input node timesteps (B)
        edge_index: Edge indices (2 x E)
    Returns:
        Edge : Updated edge features (B x V x V)
    """
        Node = self.node_embed(encoded_nodes)
        Edge = self.edge_embed(NoisedGraph.unsqueeze(-1))
        time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))

        for layer, time_layer in zip(self.layers, self.time_embed_layers):
            if self.use_activation_checkpoint:
                raise NotImplementedError

            Node = layer(Node, Edge)
            Node = Node + time_layer(time_emb)[:, None, :]

        # GenSCO-style decoder
        features = self.final_proj(Node)
        features = self.final_norm(features)
        
        logits = torch.matmul(features, features.transpose(1, 2))
        
        if self.final_logit_scale != 1.0:
            logits = logits * self.final_logit_scale
        if self.final_softcap is not None:
            logits = self.final_softcap * torch.tanh(logits / self.final_softcap)

        mask_value = 0.5 * torch.finfo(logits.dtype).min
        _arange = torch.arange(logits.shape[2], device=logits.device)
        logits = logits.clone()
        logits[:, _arange, _arange] = mask_value

        Edge = torch.stack([torch.zeros_like(logits), logits], dim=1)
        return Edge


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.encoding_block = EncodingBlock(**model_params)

    def forward(self, node_emb, edge_emb=None, mtrx=None):
        emb_out = self.encoding_block(node_emb, edge_emb, mtrx=mtrx)
        return emb_out


class EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        hidden_dim = self.model_params['hidden_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(hidden_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(hidden_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(hidden_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, hidden_dim)

        self.add_n_normalization_1 = Add_And_Normalization_Module(**model_params)
        self.feed_forward = Feed_Forward_Module(**model_params)
        self.add_n_normalization_2 = Add_And_Normalization_Module(**model_params)

        self.mixed_score_MHA = MixedScore_MultiHeadAttention(**model_params)

    def forward(self, node_emb, edge_emb=None, mtrx=None):
        # NOTE: row and col can be exchanged, if cost_mat.transpose(1,2) is used
        # input1.shape: (batch, row_cnt, embedding)
        # input2.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(node_emb), head_num=head_num)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        k = reshape_by_heads(self.Wk(node_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(node_emb), head_num=head_num)

        if mtrx is not None:
            out_concat = multi_head_attention(q, k, v, mtrx=mtrx)
        else:
            out_concat = self.mixed_score_MHA(q, k, v, edge_emb)

        # shape: (batch, row_cnt, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt, embedding)

        out1 = self.add_n_normalization_1(node_emb, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, row_cnt, embedding)

########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None, mtrx=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    if mtrx is not None:
        score_scaled += mtrx.permute(0, 3, 1, 2)


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

class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['hidden_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['hidden_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))