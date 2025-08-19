##########################################################################################
# import

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  
from argparse import ArgumentParser

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info

from pl_jssp_model import JSSPModel

##########################################################################################
# parameters

data_params = {
    'training_split': 'GeneratedData_Job3Machine5Seed42Size10.npy',
    'training_split_label_dir': 'GeneratedData_Job3Machine5Seed42Size10.npy',
    'validation_split': 'GeneratedData_Job3Machine5Seed42Size10.npy',
    'test_split': 'GeneratedData_Job3Machine5Seed42Size10.npy',
    'validation_examples': 10,
    'num_workers': 16,
    'storage_path': './Data/TrainData'}

wandb_params = {
    'project_name': 'jssp_diffusion',
    'wandb_entity': '8chris8',
    'wandb_logger_name': 'difusco',
    'resume_weight_only': False,
    'storage_path': './results/',
    'resume_id': None
}


model_params = {
    'n_layers': 12,
    'hidden_dim': 256,
    'sparse_factor': -1,
    'aggregation': 'sum',
    'two_opt_iterations': 1000,
    'save_numpy_heatmap': True,
    'use_activation_checkpoint': False,
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 0.0
    },
    'lr_scheduler': 'constant'
}

trainer_params = {
    'fp16': True,
    'diffusion_steps': 1000,
    'diffusion_schedule': 'linear',
    'inference_diffusion_steps': 1000,
    'inference_schedule': 'linear',
    'inference_trick': 'ddim',
    'sequential_sampling': 1,
    'parallel_sampling': 1,
    'epochs': 10,
    'batch_size': 5,
    'ckpt_path': None,
    'saving_mode': 'min'
}

##########################################################################################
# main

def main():
    epochs = trainer_params['epochs']
    project_name = wandb_params['project_name']

    model_class = JSSPModel
    saving_mode = trainer_params['saving_mode']
        
    model = model_class(data_params=data_params,
        model_params=model_params,
        trainer_params=trainer_params,
        optimizer_params = optimizer_params)

    wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
    wandb_logger = WandbLogger(
        name=wandb_params['wandb_logger_name'],
        project=wandb_params['project_name'],
        entity=wandb_params['wandb_entity'],
        save_dir=os.path.join(wandb_params['storage_path'], f'models'),
        id=wandb_id,
    )

    print(wandb_logger.save_dir)
    print(wandb_params['wandb_logger_name'])
    print(wandb_logger._id)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val/solved_cost', mode=saving_mode,
        save_top_k=3, save_last=True,
        dirpath=os.path.join(wandb_logger.save_dir,
                            wandb_params['wandb_logger_name'],
                            wandb_logger._id,
                            'checkpoints'),
    )
    lr_callback = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
    accelerator="auto",
    devices=1,
    # devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
    max_epochs=epochs,
    callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback, lr_callback],
    logger=wandb_logger,
    check_val_every_n_epoch=1,
    #strategy=DDPStrategy(static_graph=True),
    precision=16 if trainer_params['fp16'] else 32,
    )

    ckpt_path = trainer_params['ckpt_path']

    if wandb_params['resume_weight_only']:
        model = model_class.load_from_checkpoint(ckpt_path, param_args=model_params)
        trainer.fit(model)
    else:
        trainer.fit(model, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()
