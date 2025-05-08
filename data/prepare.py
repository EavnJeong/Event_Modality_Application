from copy import deepcopy

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .imagenet_dataset import PretrainImageNetDataset


dataset_config = {
    'N-imagenet': {
        'reshape': True, 
        'reshape_method': 'no_sample', 
        'loader_type': 'reshape_then_acc_count_pol', 
        'slice': {
            'slice_events': True, 
            'slice_length': 30000, 
            'slice_method': 'random', 
            'slice_augment': False, 
            'slice_augment_width': 0, 
            'slice_start': 0, 
            'slice_end': 30000
        }, 
        'height': 224, 
        'width': 224, 
        'augment': True, 
        'augment_type': 'base_augment', 
        'persistent_workers': True, 
        'pin_memory': True, 
        'train': {
            'type': 'PretrainImageNetDataset', 
            'root': '/mnt/Data_3/event/', 
            'file': './configs/train_list_zero.txt', 
            'label_map': '/mnt/Data_3/event/N_ImageNet/extracted_train', 
        }, 
        'eval': {
            'type': 'PretrainImageNetDataset', 
            'root': '/mnt/Data_3/event/', 
            'file': './configs/val_list_zero.txt', 
            'label_map': '/mnt/Data_3/event/N_ImageNet/extracted_val', 
        }, 
        'num_workers': 7, 
        'batch_size': 32, 
        'point_level_aug': False, 
        'view_augmentation': {
            'name': 'Ours', 
            'view1': {
                'crop_min': 0.45
            }, 
            'view2': {
                'crop_min': 0.45
            }
        }
    }
}

train_text_file_lst = {
    '1-shot': './configs/train_list_1_shot.txt',
    '2-shot': './configs/train_list_2_shot.txt',
    '5-shot': './configs/train_list_5_shot.txt',
    'all': './configs/train_list_all_cls.txt'
}

train_batch_size = {
    '1-shot': 20,
    '2-shot': 40,
    '5-shot': 100,
    'all': 4
}


def create_dataloader(
        dataset,
        dataset_opt,
        num_process
    ):
    phase = dataset_opt['phase']
    
    collate_fn = None
    if phase == 'train':
        sampler = torch.utils.data.DistributedSampler(
            dataset, shuffle=True
        )
        TrainLoader = DataLoader(
            dataset, 
            batch_size=dataset_opt["batch_size"], 
            num_workers=dataset_opt["num_workers"], 
            drop_last=True, 
            pin_memory=dataset_opt["pin_memory"], 
            persistent_workers=dataset_opt["persistent_workers"], 
            collate_fn = collate_fn, 
            sampler = sampler
        )
        return TrainLoader
    elif phase== 'test' or phase== 'eval' :
        if len(dataset) % num_process == 0:
            sampler = torch.utils.data.DistributedSampler(
                dataset, shuffle=False,
            )
        else:
           sampler = torch.utils.data.RandomSampler(dataset)
        TestLoader = DataLoader(
            dataset,
            batch_size=dataset_opt["batch_size"], 
            num_workers=dataset_opt["num_workers"], 
            drop_last=False, 
            pin_memory= dataset_opt["pin_memory"],
            persistent_workers=dataset_opt["persistent_workers"], 
            collate_fn = collate_fn, 
            sampler = sampler
        )
        return TestLoader
    else:
        raise AttributeError('Mode not provided')


class Data_prepare(pl.LightningDataModule):
    def __init__(self, ds, num_process, ft=None):
        if ds == 'N-imagenet':
            self.args = dataset_config[ds]
            if ft is not None:
                self.args['train']['file'] = train_text_file_lst[ft]
                self.args['batch_size'] = train_batch_size[ft]
                if ft == 'all':
                    self.args['eval']['file'] = './configs/val_list_all_cls.txt'
        elif ds == 'N-imagenet-1000':
            self.args = dataset_config['N-imagenet']
            self.args['train']['file'] = '/mnt/Data_3/event/N_ImageNet/train_1000_zero.txt'
            self.args['train']['label_map'] = '/mnt/Data_3/event/tmp/extracted_train'
            self.args['train']['emb_path'] = '/mnt/Data_3/event/ImageNet_CLIP/emb_train'
            self.args['eval']['file'] = '/mnt/Data_3/event/N_ImageNet/val_1000_zero.txt'
            self.args['eval']['label_map'] = '/mnt/Data_3/event/tmp/extracted_val'
            self.args['eval']['emb_path'] = '/mnt/Data_3/event/ImageNet_CLIP/emb_val'
            self.args['batch_size'] = 32
        else:
            NotImplementedError
        self.num_process = num_process
        print(f'Load file: {self.args["train"]["file"]}')
        print(f'Load file: {self.args["eval"]["file"]}')

    def setup(self, stage):
        args = self.args
        dataset_args = deepcopy(args)
        dataset_args.update(dataset_args["train"])
        dataset_args.update(dataset_args["slice"])
        dataset_args['phase'] = 'train'
        del dataset_args["train"]
        del dataset_args["slice"]

        self.train_args = dataset_args
        self.train_dataset = PretrainImageNetDataset(dataset_args)

        dataset_args = deepcopy(args)
        dataset_args.update(dataset_args["eval"])
        dataset_args.update(dataset_args["slice"])
        dataset_args['phase'] = 'eval'
        del dataset_args["eval"]
        del dataset_args["slice"]

        self.val_args = dataset_args
        self.val_dataset = PretrainImageNetDataset(dataset_args)

    def train_dataloader(self):
        if hasattr(self, "train_loader"):
            return self.train_loader
        args = self.args
        train_loader = create_dataloader(
            self.train_dataset,
            self.train_args,
            self.num_process
        )
        self.train_loader = train_loader
        return train_loader

    def val_dataloader(self):
        if hasattr(self, "eval_loader"):
            return self.eval_loader
        args = self.args
        eval_loader = create_dataloader(
            self.val_dataset,
            self.val_args,
            self.num_process
        )
        self.eval_loader = eval_loader
        return eval_loader