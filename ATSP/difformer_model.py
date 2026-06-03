"""Lightning module for training the DIFUSCO TSP model."""
import math
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
from models.transformer_encoder import TransformerEncoder
from utils.lr_schedulers import get_schedule_fn
from utils.diffusion_schedulers import CategoricalDiffusion, InferenceSchedule
from co_datasets.atsp_graph_dataset import ATSPGraphDataset
from utils.ATSP_utils import ATSPEvaluator


class Difformer_Model(pl.LightningModule):
    def __init__(self,
                 data_params=None,
                 model_params=None,
                 trainer_params=None,
                 optimizer_params=None):
        super(Difformer_Model, self).__init__()
        self.model_params = model_params
        self.data_params = data_params
        self.trainer_params = trainer_params
        self.optimizer_params = optimizer_params
        self.diffusion = CategoricalDiffusion(**self.trainer_params)
        self.sparse = model_params['sparse']

        # 나중에 얘처럼 다 리펙토링 할것 너무 지저분함
        self.premodel = TransformerEncoder(**self.model_params)

        self.model = GNNEncoder(**self.model_params)

        self.num_training_steps_cached = None

        self.train_dataset = ATSPGraphDataset(
            data_file=os.path.join(self.data_params['storage_path'], self.data_params['training_split']),
            sparse_factor=self.model_params['sparse_factor'],
        )
        self.test_dataset = ATSPGraphDataset(
            data_file=os.path.join(self.data_params['storage_path'], self.data_params['test_split']),
            sparse_factor=self.model_params['sparse_factor'],
        )
        self.validation_dataset = ATSPGraphDataset(
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
                list(self.model.parameters()) + list(self.premodel.parameters()),                                                 
                lr=self.optimizer_params['optimizer']['lr'],                     
                weight_decay=self.optimizer_params['optimizer']['weight_decay']                                                   
)   

        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.optimizer_params['optimizer']['lr'],
                weight_decay=self.optimizer_params['optimizer']['weight_decay'])
            scheduler = get_schedule_fn(self.optimizer_params['lr_scheduler'], self.get_total_num_training_steps())(
                optimizer)

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
        batch_size = self.trainer_params['test_batch_size']
        print("Test dataset size:", len(self.test_dataset))
        test_dataloader = GraphDataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return test_dataloader

    def val_dataloader(self):
        batch_size = self.trainer_params['valid_batch_size']
        val_dataset = torch.utils.data.Subset(self.validation_dataset, range(self.data_params['validation_examples']))
        print("Validation dataset size:", len(val_dataset))
        val_dataloader = GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return val_dataloader


    def forward(self, points, xt, solution_adj, t, device, edge_index=None):                            
        return self.model(points, xt, solution_adj, t, edge_index) 

    def pre_forward(self, node_input, edge_input):
        return self.premodel(node_input, edge_input)

    def categorical_training_step(self, batch, batch_idx):
        edge_index = None

        node_cnt, node_feature, edge_feature, solution_adj, objective = batch
        t = np.random.randint(1, self.diffusion.T + 1, node_cnt.shape[0]).astype(int)
        solution_adj_onehot = F.one_hot(solution_adj.long(), num_classes=2).float()

        # 인코더 입력 및 엣지, 노드 정보 저장
        points = self.premodel(node_feature, edge_feature)

        xt = self.diffusion.sample(solution_adj_onehot, t)
        xt = xt * 2 - 1
        xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

        t = torch.from_numpy(t).float().view(solution_adj.shape[0])

        # Denoise
        # xt:노이즈
        # t: 시간 스텝
        # solution_adj: 정답
        x0_pred = self.forward(
            points.float().to(solution_adj.device),
            xt.float().to(solution_adj.device),
            solution_adj.float().to(solution_adj.device),
            t.float().to(solution_adj.device),
            edge_index,
        )

        # Compute loss
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(x0_pred, solution_adj.long())
        self.log("train/loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.categorical_training_step(batch, batch_idx)

    def categorical_denoise_step(self, points, xt, solution_adj, t, device, edge_index=None, target_t=None):  
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            x0_pred = self.forward(
                points.float().to(device),
                xt.float().to(device),
                solution_adj.float().to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
            )
            if not self.sparse:
                x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
            else:
                x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)

            xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
            return xt

    def test_step(self, batch, batch_idx, split='test'):
        edge_index = None
        device = batch[-1].device
        node_cnt, node_feature, edge_feature, solution_adj, objective = batch
        B, N, _ = solution_adj.shape
        S = self.trainer_params['parallel_sampling']

        # points: [B, N, D] → [B*S, N, D]
        points = self.premodel(node_feature, edge_feature)
        if S > 1:
            points = points.repeat_interleave(S, dim=0)  # [B*S, N, D]

        # xt 초기화: binary categorical noise [B*S, N, N]
        xt = (torch.rand(B * S, N, N, device=device) > 0.5).long()

        steps = self.trainer_params['inference_diffusion_steps']
        time_schedule = InferenceSchedule(
            inference_schedule=self.trainer_params['inference_schedule'],
            T=self.diffusion.T,
            inference_T=steps,
        )

        # solution_adj: [B, N, N] → [B*S, N, N]
        solution_adj_rep = solution_adj.repeat_interleave(S, dim=0)

        for i in range(steps):
            t1, t2 = time_schedule(i)
            t1 = np.array([t1]).astype(int)
            t2 = np.array([t2]).astype(int)
            xt = self.categorical_denoise_step(
                points, xt, solution_adj_rep, t1, device, edge_index, target_t=t2
            )

        # xt: [B*S, N, N] → adj_mat: [B, S, N, N]
        # repeat_interleave로 [A,A,B,B,C,C] 순서이므로 view(B,S)가 배치별로 정확히 대응
        adj_mat = xt.float().cpu().detach() + 1e-6
        adj_mat = adj_mat.view(B, S, N, N)

        # tour_len: [B, S]
        tour_len, _ = ATSPEvaluator(adj_mat, edge_feature)
        assert tour_len.shape == (B, S), \
            f"ATSPEvaluator tour_len shape 불일치: {tour_len.shape} != ({B}, {S})"

        # parallel 중 best 선택 → [B]
        best_tour_len = tour_len.min(dim=-1).values.cpu()

        # -------------------------------------------------------------------------
        # sequential_sampling 미사용 버전 (아래는 사용 시 참고용 주석)
        # for _ in range(self.trainer_params['sequential_sampling']):
        #     xt = (torch.rand(B * S, N, N, device=device) > 0.5).long()
        #     ...denoise loop...
        #     cur_best = tour_len.min(dim=-1).values.cpu()
        #     if best_tour_len is None:
        #         best_tour_len = cur_best
        #     else:
        #         improved = cur_best < best_tour_len
        #         best_tour_len = torch.where(improved, cur_best, best_tour_len)
        # -------------------------------------------------------------------------

        if self.model_params['save_numpy_heatmap']:
            self.run_save_numpy_heatmap(
                adj_mat.numpy(), points.cpu().numpy()[0], batch_idx, split
            )

        # best_tour_len: [B] cpu, objective: [B] gpu → cpu로 통일
        opt_gaps = ((best_tour_len - objective.cpu()) / objective.cpu() * 100).clamp(min=0)

        metrics = {
            f"{split}/Heuristic": objective.mean().item(),
            f"{split}/Diffusion":  best_tour_len.mean().item(),
            f"{split}/RPD":        opt_gaps.mean().item(),
        }
        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, sync_dist=True)

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
        np.save(os.path.join(heatmap_path, f"{split}-points-{real_batcQh_idx}.npy"), np_pt)

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, split='val')
