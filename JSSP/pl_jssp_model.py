"""Lightning module for training the DIFUSCO TSP model."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pytorch_lightning as pl

from pytorch_lightning.utilities import rank_zero_info
from torch_geometric.data import DataLoader as GraphDataLoader
from models.gnn_encoder import GNNEncoder
from models.gnn_instance_embedding import GNNInstanceEncoder
from utils.lr_schedulers import get_schedule_fn
from utils.diffusion_schedulers import CategoricalDiffusion, InferenceSchedule
from co_datasets.tsp_graph_dataset import TSPGraphDataset
from utils.JSSP_utils import JSSPEvaluator


class JSSPModel(pl.LightningModule):
  def __init__(self,
               data_params=None,
               model_params=None,
               trainer_params=None,
               optimizer_params=None,
               node_feature_only=False):
    super(JSSPModel, self).__init__()
    self.model_params = model_params
    self.data_params = data_params
    self.trainer_params = trainer_params
    self.optimizer_params = optimizer_params
    self.diffusion_schedule = self.trainer_params['diffusion_schedule']
    self.diffusion_steps = self.trainer_params['diffusion_steps']
    self.sparse = self.model_params['sparse_factor'] > 0 or node_feature_only

    out_channels = 2
    self.diffusion = CategoricalDiffusion(
      T=self.diffusion_steps, schedule=self.diffusion_schedule)

    self.model = GNNEncoder(
        n_layers=self.model_params['n_layers'],
        hidden_dim=self.model_params['hidden_dim'],
        out_channels=out_channels,
        aggregation=self.model_params['aggregation'],
        sparse=self.sparse,
        use_activation_checkpoint=self.model_params['use_activation_checkpoint'],
        node_feature_only=node_feature_only,
    )

    self.premodel = GNNInstanceEncoder(
        n_layers=self.model_params['n_layers'],
        hidden_dim=self.model_params['hidden_dim'],
        out_channels=out_channels,
        aggregation=self.model_params['aggregation'],
        use_activation_checkpoint=self.model_params['use_activation_checkpoint'],
    )


    self.num_training_steps_cached = None


    self.train_dataset = TSPGraphDataset(
        data_file=os.path.join(self.data_params['storage_path'], self.data_params['training_split']),
        sparse_factor=self.model_params['sparse_factor'],
    )
    self.test_dataset = TSPGraphDataset(
        data_file=os.path.join(self.data_params['storage_path'], self.data_params['test_split']),
        sparse_factor=self.model_params['sparse_factor'],
    )
    self.validation_dataset = TSPGraphDataset(
        data_file=os.path.join(self.data_params['storage_path'], self.data_params['validation_split']),
        sparse_factor=self.model_params['sparse_factor'],
    )

  def test_epoch_end(self, outputs):
    unmerged_metrics = {}
    for metrics in outputs:
      for k, v in metrics.items():
        if k not in unmerged_metrics:
          unmerged_metrics[k] = []
        unmerged_metrics[k].append(v)

    merged_metrics = {}
    for k, v in unmerged_metrics.items():
      merged_metrics[k] = float(np.mean(v))
    self.logger.log_metrics(merged_metrics, step=self.global_step)

  def get_total_num_training_steps(self) -> int:
    """Total training steps inferred from datamodule and devices."""
    if self.num_training_steps_cached is not None:
      return self.num_training_steps_cached
    dataset = self.train_dataloader()
    if self.trainer.max_steps and self.trainer.max_steps > 0:
      return self.trainer.max_steps

    dataset_size = (
        self.trainer.limit_train_batches * len(dataset)
        if self.trainer.limit_train_batches != 0
        else len(dataset)
    )

    num_devices = max(1, self.trainer.num_devices)
    effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
    self.num_training_steps_cached = (dataset_size // effective_batch_size) * self.trainer.max_epochs

    return self.num_training_steps_cached

  def configure_optimizers(self):
    rank_zero_info('Parameters: %d' % sum([p.numel() for p in self.model.parameters()]))
    rank_zero_info('Training steps: %d' % self.get_total_num_training_steps())

    if self.optimizer_params['lr_scheduler'] == "constant":
      return torch.optim.AdamW(
          self.model.parameters(), lr=self.optimizer_params['optimizer']['lr'], weight_decay=self.optimizer_params['optimizer']['weight_decay'])

    else:
      optimizer = torch.optim.AdamW(
          self.model.parameters(), lr=self.optimizer_params['optimizer']['lr'], weight_decay=self.optimizer_params['optimizer']['weight_decay'])
      scheduler = get_schedule_fn(self.optimizer_params['lr_scheduler'], self.get_total_num_training_steps())(optimizer)

      return {
          "optimizer": optimizer,
          "lr_scheduler": {
              "scheduler": scheduler,
              "interval": "step",
          },
      }

  def categorical_posterior(self, target_t, t, x0_pred_prob, NoisedGraph):
    """Sample from the categorical posterior for a given time step.
       See https://arxiv.org/pdf/2107.03006.pdf for details.
    """
    diffusion = self.diffusion

    if target_t is None:
      target_t = t - 1
    else:
      target_t = torch.from_numpy(target_t).view(1)

    # Thanks to Daniyar and Shengyu, who found the "target_t == 0" branch is not needed :)
    # if target_t > 0:
    Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
    Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)
    # else:
    #   Q_t = torch.eye(2).float().to(x0_pred_prob.device)
    Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device)
    Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device)

    NoisedGraph = F.one_hot(NoisedGraph.long(), num_classes=2).float()
    NoisedGraph = NoisedGraph.reshape(x0_pred_prob.shape)

    x_t_target_prob_part_1 = torch.matmul(NoisedGraph, Q_t.permute((1, 0)).contiguous())
    x_t_target_prob_part_2 = Q_bar_t_target[0]
    x_t_target_prob_part_3 = (Q_bar_t_source[0] * NoisedGraph).sum(dim=-1, keepdim=True)

    x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3

    sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]
    x_t_target_prob_part_2_new = Q_bar_t_target[1]
    x_t_target_prob_part_3_new = (Q_bar_t_source[1] * NoisedGraph).sum(dim=-1, keepdim=True)

    x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / x_t_target_prob_part_3_new

    sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]

    if target_t > 0:
      NoisedGraph = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
    else:
      NoisedGraph = sum_x_t_target_prob.clamp(min=0)

    return NoisedGraph

  def duplicate_edge_index(self, edge_index, num_nodes, device):
    """Duplicate the edge index (in sparse graphs) for parallel sampling."""
    edge_index = edge_index.reshape((2, 1, -1))
    edge_index_indent = torch.arange(0, self.trainer_params['parallel_sampling']).view(1, -1, 1).to(device)
    edge_index_indent = edge_index_indent * num_nodes
    edge_index = edge_index + edge_index_indent
    edge_index = edge_index.reshape((2, -1))
    return edge_index

  def train_dataloader(self):
    batch_size = self.trainer_params['batch_size']
    train_dataloader = GraphDataLoader(
        self.train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=self.data_params['num_workers'], pin_memory=True,
        persistent_workers=True, drop_last=False)
  
    return train_dataloader

  def test_dataloader(self):
    batch_size = 1
    print("Test dataset size:", len(self.test_dataset))
    test_dataloader = GraphDataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
    return test_dataloader

  def val_dataloader(self):
    batch_size = 1
    val_dataset = torch.utils.data.Subset(self.validation_dataset, range(self.data_params['validation_examples']))
    print("Validation dataset size:", len(val_dataset))
    val_dataloader = GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_dataloader

  def forward(self, Node, NoisedGraph, t, edge_index):
    return self.model(Node, NoisedGraph, t, edge_index)

  def pre_forward(self, Node, JobAdj, MachineAdj, edge_index):
    # JobAdj, MachineAdj 는 torch.Tensor (dtype=bool 또는 float) 라고 가정
    CombinedAdj = torch.logical_or(JobAdj.bool(), MachineAdj.bool()).float()
    B, V, _ = CombinedAdj.shape
    idx = torch.arange(V, device=CombinedAdj.device)
    CombinedAdj[:, idx, idx] = 1.0
    return self.premodel(Node, CombinedAdj, edge_index)

  def categorical_training_step(self, batch, batch_idx):
    edge_index = None

    _, Pt, JobAdj, MachineAdj, Assigned_machine, _ = batch
    adj_matrix = torch.logical_or(JobAdj.bool(), MachineAdj.bool()).float()
    adj_matrix.diagonal(dim1=-2, dim2=-1).fill_(1)

    # Node = Pt
    
    Node = self.pre_forward(Pt, JobAdj, MachineAdj, edge_index)

    t = np.random.randint(1, self.diffusion.T + 1, Node.shape[0]).astype(int)
   
    # Sample from diffusion
    adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()

    # valid_noise = self.invalid_noise_label(Assigned_machine, JobAdj)
    
    NoisedGraph = self.diffusion.sample(adj_matrix_onehot, t)

    NoisedGraph = NoisedGraph * 2 - 1
    NoisedGraph = NoisedGraph * (1.0 + 0.05 * torch.rand_like(NoisedGraph))

    t = torch.from_numpy(t).float().view(adj_matrix.shape[0]) 

    # Denoise
    x0_pred = self.forward(
        Node.float().to(adj_matrix.device),
        NoisedGraph.float().to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        edge_index,
    )

    # Compute loss
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(x0_pred, JobAdj.long())
    self.log("train/loss", loss, on_epoch=True)
    return loss
    
  def invalid_noise_label(self, Assigned_machine, JobAdj):
    batch_size, num_ops = Assigned_machine.shape
    device = Assigned_machine.device  # cuda 번호 가져오기

    # 머신 번호 비교 (같으면 1, 다르면 0)
    valid_edge = (Assigned_machine[:, :, None] == Assigned_machine[:, None, :]).long().to(device)
    
    # 선행 작업비교 (선행 1, 비선행 0)
    valid_edge = valid_edge + JobAdj.long().to(device)

    return valid_edge  # shape: (batch_size, num_ops, num_ops)

  def training_step(self, batch, batch_idx):
    return self.categorical_training_step(batch, batch_idx)

  def categorical_denoise_step(self, Pt, NoisedGraph, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      x0_pred = self.forward(
          Pt.float().to(device),
          NoisedGraph.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )
      x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
      NoisedGraph = self.categorical_posterior(target_t, t, x0_pred_prob, NoisedGraph)
      return NoisedGraph

  def test_step(self, batch, batch_idx, split='test'):
    edge_index = None
    np_edge_index = None
    device = batch[-1].device

    real_batch_idx, Pt, JobAdj, MachineAdj, machine, GA_makespan = batch
    adj_matrix = torch.logical_or(JobAdj.bool(), MachineAdj.bool()).float()
    adj_matrix.diagonal(dim1=-2, dim2=-1).fill_(1)
    np_pt = Pt.cpu().numpy()[0]
    np_machine = machine.cpu().numpy()[0]
    GA_makespan = GA_makespan.float()
    

    num_m = len(set(np_machine.tolist()))
    num_j = len(np_pt) // num_m  

    # Node = Pt
    # JobAdj, MachineAdj 는 torch.Tensor (dtype=bool 또는 float) 라고 가정
    Node = self.pre_forward(Pt, JobAdj, MachineAdj, edge_index)

    Node = Node.repeat(self.trainer_params['parallel_sampling'], 1, 1)

    for _ in range(self.trainer_params['sequential_sampling']):
      NoisedGraph = torch.randn_like(adj_matrix.float())
      NoisedGraph = NoisedGraph.repeat(self.trainer_params['parallel_sampling'], 1, 1)
      NoisedGraph = torch.randn_like(NoisedGraph)
      NoisedGraph = (NoisedGraph > 0).long()

      steps = self.trainer_params['inference_diffusion_steps']
      time_schedule = InferenceSchedule(inference_schedule=self.trainer_params['inference_schedule'],
                                        T=self.diffusion.T, inference_T=steps)
      # Diffusion iterations
      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1]).astype(int)
        t2 = np.array([t2]).astype(int)

        NoisedGraph = self.categorical_denoise_step(
          Node, NoisedGraph, t1, device, edge_index, target_t=t2)

      DenoisedGraph = NoisedGraph.float().cpu().detach().numpy() + 1e-6
      if self.model_params['save_numpy_heatmap']:
        self.run_save_numpy_heatmap(DenoisedGraph, np_pt, real_batch_idx, split)

    makespan, schedule = JSSPEvaluator(
        DenoisedGraph,
        np_pt,
        np_machine,
        num_j
    )

    metrics = {
        f"{split}/GA": GA_makespan,
        f"{split}/Diffusion": makespan,
    }
    for k, v in metrics.items():
        self.log(k, v, on_epoch=True, sync_dist=True)
    RPD = (makespan - GA_makespan) / GA_makespan * 100        
    self.log(f"{split}/RPD", RPD, prog_bar=True, on_epoch=True, sync_dist=True)
    # Terminal Log
    # print("RPD",RPD)
    return metrics

  def run_save_numpy_heatmap(self, adj_mat, np_pt, real_batch_idx, split):
    if self.trainer_params['parallel_sampling'] > 1 or self.trainer_params['sequential_sampling'] > 1:
      raise NotImplementedError("Save numpy heatmap only support single sampling")
    exp_save_dir = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version)
    heatmap_path = os.path.join(exp_save_dir, 'numpy_heatmap')
    rank_zero_info(f"Saving heatmap to {heatmap_path}")
    os.makedirs(heatmap_path, exist_ok=True)
    real_batch_idx = real_batch_idx.cpu().numpy().reshape(-1)[0]
    np.save(os.path.join(heatmap_path, f"{split}-heatmap-{real_batch_idx}.npy"), adj_mat)
    np.save(os.path.join(heatmap_path, f"{split}-points-{real_batch_idx}.npy"), np_pt)

  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx, split='val')
