import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import math
import wandb
import os

from utils import MetricLogger


class LPTNMTrainer(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = vars(args)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.zscl_scale = args.zscl_scale
        self.zscl_temperature = args.zscl_temperature
        self.automatic_optimization = False

        from data.imagenet_dataset import labels, text_inputs
        self.labels = labels
        self.text_inputs = text_inputs

    def num_training_steps(self):
        dataset =  self.trainer._data_connector._train_dataloader_source.dataloader()         
        return len(dataset)

    def num_batch_size(self):
        dataset =  self.trainer._data_connector._train_dataloader_source.dataloader()         
        return dataset.batch_size

    def adjust_learning_rate(self, idx):
        optimizer = self.optimizers()
        warmup_epochs = self.args["warmup_epoch"]
        lr = self.args["learning_rate"]
        minlr = self.args.get("min_lr", 0.0)
        epochs = self.args.get("cycle_epoch", self.args["epoch"])
        epoch = self.current_epoch % epochs + idx / self.num_training_steps()

        if epoch < warmup_epochs:
            lr = minlr + (lr - minlr) * epoch / warmup_epochs
        else:
            lr = minlr + (lr - minlr) * 0.5 * (1. + math.cos(
                math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        if "max_grad_norm" in self.args:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.args["max_grad_norm"]
            )

    def get_acc(self, emb, pred, label):
        self.model.foundation_model.to(emb.device)
        texts = self.text_inputs.to(emb.device)
        texts = self.model.foundation_model.encode_text(texts)

        cos = F.cosine_similarity(
            pred.unsqueeze(1), texts.unsqueeze(0), dim=-1
        )
        
        prediction = torch.argmax(cos, dim=-1)
        prediction_label = [
            list(self.labels.values())[predi] for predi in prediction
        ]
        _, top5_indices = torch.topk(cos, 5, dim=-1)
        pred_list = [
            [list(self.labels.values())[idx] for idx in top5_indices[i]]
                for i in range(top5_indices.size(0))
        ]

        total, correct, correct_5 = 0, 0, 0
        for predi, predi5, true_label in zip(prediction_label, pred_list, label):
            if self.args['dataset'] == "N-imagenet" or self.args['dataset'] == "N-imagenet-1000":
                if self.labels[true_label] == predi:
                    correct += 1
                if self.labels[true_label] in predi5:
                    correct_5 += 1
            elif self.args['dataset'] == "N-caltech":
                if true_label == predi:
                    correct += 1
                if true_label in predi5:
                    correct_5 += 1
            total += 1
        acc1 = correct / total
        acc5 = correct_5 / total
        return acc1, acc5

    def get_text_embedding(self, pred):
        texts = self.text_inputs.to(pred.device)
        with torch.no_grad():
            texts = self.model.foundation_model.encode_text(texts)
        return texts.float()

    def get_pred_loss(self, pred, label):
        texts = self.text_inputs.to(pred.device)
        with torch.no_grad():
            texts = self.model.foundation_model.encode_text(texts)
        label_indices = torch.tensor(
            [list(self.labels.keys()).index(true_label) 
                for true_label in label]).to(pred.device)
        target = torch.ones(pred.size(0)).to(pred.device)
        pred_loss = torch.nn.CosineEmbeddingLoss()(
            pred, texts[label_indices], target)
        return pred_loss

    def distillation(self, t, s, T=2):
        p = F.softmax(t / T, dim=1)
        loss = F.cross_entropy(s / T, p, reduction="mean") * (T ** 2)
        return loss
   
    def get_zscl_loss(self, pred_q, pred_k, text_embedding):
        ref_logits = self.logit_scale.exp() * pred_q @ text_embedding.t()
        cur_logits = self.logit_scale.exp() * pred_k @ text_embedding.t()
        loss_zscl = self.distillation(ref_logits, cur_logits, self.zscl_temperature)
        loss_zscl2 = self.distillation(cur_logits.t(), ref_logits.t(), self.zscl_temperature)        
        zscl_loss = self.zscl_scale * (loss_zscl + loss_zscl2)
        return zscl_loss
   
    def training_step(self, data, idx):
        img = data["img"]
        event = (data["event"] - 0.5) * 2
        label = data["label"]

        optimizer = self.optimizers()
        self.adjust_learning_rate(idx)
        
        moco_ct_loss, kl_loss, pred_k, pred_q, emb = self.model(img, event)        
        acc1, acc5 = self.get_acc(emb, pred_k, label)    
        text_embedding = self.get_text_embedding(pred_k)
        zscl_loss = self.get_zscl_loss(pred_q, pred_k, text_embedding)

        self.manual_backward(moco_ct_loss + kl_loss + zscl_loss)
        optimizer.step()
        optimizer.zero_grad()
        self.produce_log(
            moco_ct_loss,
            kl_loss,
            zscl_loss,
            idx, 
            acc1,
            acc5
        )

    def training_epoch_end(self, training_step_outputs):
        if self.trainer.is_global_zero and \
           (self.current_epoch + 1) % self.args["save_every"] == 0:
                self.save()

    def on_validation_epoch_start(self):
        self.metric_logger = MetricLogger()

    def validation_step(self, data, idx):
        img = data["img"]
        event = (data["event"] - 0.5) * 2
        label = data["label"]

        moco_ct_loss, kl_loss, pred_k, pred_q, emb  = self.model(
            img, event)
        acc1, acc5 = self.get_acc(emb, pred_k, label)
        text_embedding = self.get_text_embedding(pred_k)
        zscl_loss = self.get_zscl_loss(pred_q, pred_k, text_embedding)

        batch_size = event.size(0)
        self.metric_logger.meters["acc1"].update(acc1, n=batch_size)
        self.metric_logger.meters["acc5"].update(acc5, n=batch_size)
        self.metric_logger.meters["moco_ct_loss"].update(moco_ct_loss.item(), n=batch_size)
        self.metric_logger.meters['kl_loss'].update(kl_loss.item(), n=batch_size)
        self.metric_logger.meters['zscl_loss'].update(zscl_loss.item(), n=batch_size)

    def validation_epoch_end(self, outputs):
        self.metric_logger.synchronize_between_processes()
        acc1 = self.metric_logger.acc1.global_avg
        acc5 = self.metric_logger.acc5.global_avg
        moco_ct_loss = self.metric_logger.moco_ct_loss.global_avg
        kl_loss = self.metric_logger.kl_loss.global_avg
        zscl_loss = self.metric_logger.zscl_loss.global_avg

        if not os.path.exists(self.args["save_path"]):
            os.makedirs(self.args["save_path"]) 

        torch.save(
            {
                "checkpoint": self.model.state_dict(),
                "optimizer": self.optimizers().state_dict(),
            }, 
            os.path.join(self.args["save_path"], "best.pt")
        )

        if self.trainer.is_global_zero and self.trainer.num_gpus != 0:
            wandb.log(
                {
                    "val_acc_1": acc1,
                    "val_acc_5": acc5,
                    "val_moco_ct_loss": moco_ct_loss,
                    "val_kl_loss": kl_loss,
                    "val_zscl_loss": zscl_loss,
                }
            )
 
    def produce_log(
            self, 
            moco_ct_loss,
            kl_loss,
            zscl_loss,
            idx, 
            acc_1,
            acc_5,
        ):
        moco_ct_loss = self.all_gather(moco_ct_loss).mean().item()
        kl_loss = self.all_gather(kl_loss).mean().item()
        zscl_loss = self.all_gather(zscl_loss).mean().item()
        
        if self.trainer.is_global_zero and idx % 100 == 0:
            len_loader = self.num_training_steps()
            
            batches_done = self.current_epoch  * len_loader + idx + 1
            batches_left = self.trainer.max_epochs * len_loader - batches_done
            lr = self.optimizers().param_groups[0]['lr']
            wandb.log(
                {
                    "tr_moco_ct_loss": moco_ct_loss,
                    "tr_kl_loss": kl_loss,
                    "tr_zscl_loss": zscl_loss,
                    "tr_acc_1": acc_1,
                    "tr_acc_5": acc_5,
                    "tr_lr": lr,
                    "epochs": self.current_epoch,
                }
            )
    
    def save(self):
        if not os.path.exists(self.args["save_path"]):
            os.makedirs(self.args["save_path"])
        output_path = os.path.join(
            self.args["save_path"], 
            f"{self.current_epoch + 1}.pt")
        torch.save(
            {
                "checkpoint": self.model.state_dict(),
                "optimizer": self.optimizers().state_dict(),
            }, 
            output_path
        )

    def configure_optimizers(self):
        self.args["learning_rate"] = self.trainer.num_gpus \
                                   * self.trainer.num_nodes \
                                   * self.num_batch_size() / 256 \
                                   * self.args["base_lr"] 
        b1 = self.args.get("b1", 0.9)
        b2 = self.args.get("b2", 0.999)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr = self.args["learning_rate"],
            betas = (b1, b2),
            weight_decay = self.args["weight_decay"]
        )
        return optimizer