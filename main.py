import torch
import os

import argparse
import wandb

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from model.model import LPTNMmodel
from trainer import LPTNMTrainer
from data.prepare import Data_prepare
from model.foundation import get_foundation_model


def load_data_module(args, num_process):
    if args.dataset == 'N-imagenet':
        return Data_prepare(args.dataset, num_process)
    elif args.dataset == 'N-imagenet-1000':
        return Data_prepare(args.dataset, num_process)
    else:
        raise ValueError('Dataset not found')


def main(args):
    wandb.init(project='event_vit', name=args.exp_name)
    f_model, f_preprocess = get_foundation_model(args)
    
    num_process = args.gpus * args.num_nodes
    model = LPTNMmodel(
        f_model=f_model,
        f_preprocess=f_preprocess,
    )
    model = LPTNMTrainer(model, args)

    sync_batchnorm = True
    trainer = pl.Trainer(
        max_epochs=args.epoch,
        num_nodes=args.num_nodes,
        precision=32,
        gpus=args.gpus,
        strategy=DDPPlugin(find_unused_parameters=True),
        checkpoint_callback=False,
        logger=False,
        sync_batchnorm=sync_batchnorm,
        replace_sampler_ddp=False,
        check_val_every_n_epoch=1,
    )
    data_module = load_data_module(args, num_process)
    trainer.fit(model, data_module)
    torch.save(
        model.state_dict(), 
        os.path.join('checkpoints', args.exp_name + '_last.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--acce", default='ddp', type=str)
    parser.add_argument('--num_nodes', default=1, type=int)
    # wandb
    parser.add_argument('--exp_name', default='exp', type=str)
    # dataset
    parser.add_argument('--dataset', default='N-imagenet', type=str, choices=['N-imagenet', 'N-imagenet-1000'])
    # train
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--warmup_epoch', default=40, type=int)
    parser.add_argument('--base_lr', default=1e-6, type=float)
    parser.add_argument('--b1', default=0.9, type=float)
    parser.add_argument('--b2', default=0.999, type=float)
    parser.add_argument('--weight_decay', default=0.03, type=float)
    parser.add_argument('--zscl_scale', default=0.1, type=float)
    parser.add_argument('--zscl_temperature', default=2, type=float)
    parser.add_argument('--foundation', default='ViT-B/32', type=str, choices=['ViT-B/32', 'ViT-L/14'])

    args = parser.parse_args()
    args.save_path = os.path.join('checkpoints', args.exp_name)
    main(args)    