import functools
import math

import torch
import torch.nn.functional as F
from torch import nn
from models.nn import (
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)
from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum
from torch_sparse import mean as sparse_mean
from torch_sparse import max as sparse_max
import torch.utils.checkpoint as activation_checkpoint


class GNNEmbeddingLayer(nn.Module):
    """
    Difusco의 백본과 동일한 모델 
    jssp의 인스턴스 정보를 노드에 함축시키기 위해 사용
    """

    def __init__(self, hidden_dim, aggregation="sum", norm="batch", learn_norm=True, track_norm=False, gated=True):

        super(GNNEmbeddingLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.norm = norm
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        self.gated = gated
        assert self.gated, "Use gating with GCN, pass the `--gated` flag"

        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.norm_h = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)

        self.norm_e = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)

    def forward(self, h, e, graph, mode="residual", edge_index=None, sparse=False):
        """
    Args:
      h: Input node features (B x V x H)
      e: Input edge features (B x V x V x H)
      graph: Graph adjacency matrices (B x V x V)
      mode: str

    Returns:
        Updated node and edge features
    """
        h_in = h
        e_in = e

        batch_size, num_nodes, hidden_dim = h.shape

        # Linear transformations for node update
        Uh = self.U(h)  # B x V x H

        Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H

        # Linear transformations for edge update and gating
        Ah = self.A(h)  # B x V x H, source
        Bh = self.B(h)  # B x V x H, target
        Ce = self.C(e)  # B x V x V x H / E x H

        # Update edge features and compute edge gates
        e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H

        gates = torch.sigmoid(e)  # B x V x V x H / E x H

        # Update node features
        h = Uh + self.aggregate(Vh, graph, gates, edge_index=edge_index, sparse=sparse)  # B x V x H

        # Normalize node features
        h = self.norm_h(
            h.view(batch_size * num_nodes, hidden_dim)
        ).view(batch_size, num_nodes, hidden_dim) if self.norm_h else h

        # Normalize edge features
        e = self.norm_e(
            e.view(batch_size * num_nodes * num_nodes, hidden_dim)
        ).view(batch_size, num_nodes, num_nodes, hidden_dim) if self.norm_e else e

        # Apply non-linearity
        h = F.relu(h)
        e = F.relu(e)

        # Make residual connection
        if mode == "residual":
            h = h_in + h
            e = e_in + e

        return h, e


    def aggregate(self, Vh, graph, gates, mode=None, edge_index=None, sparse=False):
        """
    Args:
        In Dense version:
          Vh: Neighborhood features (B x V x V x H)
          graph: Graph adjacency matrices (B x V x V)
          gates: Edge gates (B x V x V x H)
          mode: str
        In Sparse version:
          Vh: Neighborhood features (E x H)
          graph: torch_sparse.SparseTensor (E edges for V x V adjacency matrix)
          gates: Edge gates (E x H)
          mode: str
          edge_index: Edge indices (2 x E)
        sparse: Whether to use sparse tensors (True/False)
    Returns:
        Aggregated neighborhood features (B x V x H)
    """
        # Perform feature-wise gating mechanism
        Vh = gates * Vh  # B x V x V x H

        # Enforce graph structure through masking
        # Vh[graph.unsqueeze(-1).expand_as(Vh)] = 0

        # Aggregate neighborhood features
        if (mode or self.aggregation) == "mean":
            return torch.sum(Vh, dim=2) / (torch.sum(graph, dim=2).unsqueeze(-1).type_as(Vh))
        elif (mode or self.aggregation) == "max":
            return torch.max(Vh, dim=2)[0]
        else:
            return torch.sum(Vh, dim=2)


class PositionEmbeddingSine(nn.Module):
    """
  This is a more standard version of the position embedding, very similar to the one
  used by the Attention is all you need paper, generalized to work on images.
  """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        y_embed = x[:, :, 0]
        x_embed = x[:, :, 1]
        if self.normalize:
            # eps = 1e-6
            y_embed = y_embed * self.scale
            x_embed = x_embed * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2.0 * (torch.div(dim_t, 2, rounding_mode='trunc')) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).contiguous()
        return pos


class ScalarEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        x_embed = x
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return pos_x


class GNNInstanceEncoder(nn.Module):
    """Configurable GNN Encoder
  """

    def __init__(self, n_layers, hidden_dim, out_channels=1, aggregation="sum", norm="layer",
                 learn_norm=True, track_norm=False, gated=True,
                 use_activation_checkpoint=False,
                 *args, **kwargs):
        super(GNNInstanceEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        # self.node_embed = nn.Linear(1, hidden_dim)

        self.job_proj  = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(inplace=True))  
        self.mach_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(inplace=True))  
        self.fuse_norm = nn.LayerNorm(hidden_dim, elementwise_affine=True)               
        self.fuse_drop = nn.Dropout(p=0.1)                                               
        self.node_embed = nn.Identity()

        self.edge_embed = nn.Linear(hidden_dim, hidden_dim)

        # 왜 포시셔널 임베딩이 필요한가
        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.edge_pos_embed = ScalarEmbeddingSine(hidden_dim, normalize=False)

        # node 임베딩을 한번 정제해서 출력해야 함 원래 edge에  쓰이나 임베딩에서는 노드에 적용
        # self.out = nn.Sequential(
        #     normalization(hidden_dim),
        #     nn.ReLU(),
        #     # zero_module(
        #     nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)
        #     # ),
        # )

        self.out_node = nn.Sequential(
            nn.LayerNorm(hidden_dim),  # ← 마지막 차원(H)에 대해 정규화
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)  # 노드당 out_channels
        )

        self.layers = nn.ModuleList([
            GNNEmbeddingLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
            for _ in range(n_layers)
        ])


        # 얘가 왜 필요한가 왜 엣지에만 해주는가
        self.per_layer_out = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
                nn.SiLU(),
                zero_module(
                    nn.Linear(hidden_dim, hidden_dim)
                ),
            ) for _ in range(n_layers)
        ])

        self.use_activation_checkpoint = use_activation_checkpoint

    def forward(self, Node, CombinedGraph, edge_index=None):
        """
    jssp의 정보를 함축할 때 timestep은 필요치 않음
    또한 엣지로 표현되는 인스턴스가 아니라면 이 임베딩을 쓸 필요가 없기에 노드 피쳐 온리도 필요 x
    """

        del edge_index
        # Node = self.node_embed(Node)
        job_norm  = Node[..., 0:1].float()                       # (B,V,1)  
        mach_norm = Node[..., 1:2].float()                       # (B,V,1)  
        Node = self.job_proj(job_norm) + self.mach_proj(mach_norm)          
        Node = self.fuse_drop(self.fuse_norm(Node))     
        Edge = self.edge_embed(self.edge_pos_embed(CombinedGraph))

        ## Abilation Study : Aggregate전파 범위 설정 
        # AggregateGraph = torch.ones_like(CombinedGraph).long() # FullyConnectedGraph
        AggregateGraph = CombinedGraph
        
        for layer, out_layer in zip(self.layers, self.per_layer_out):
            Node_in, Edge_in = Node, Edge

            if self.use_activation_checkpoint:
                raise NotImplementedError

            Node, Edge = layer(Node, Edge, AggregateGraph, mode="direct")
            Node = Node_in + out_layer(Node)
            # 왜 엣지는 한번 더 레이어를 거칠까?
            Edge = Edge_in + Edge
        # 노드에 인스턴스를 함축하므로 node를 줘야함
        Node = self.out_node(Node)
        return Node

