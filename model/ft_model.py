import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist

import math
import numpy as np
import copy


class FTModel(nn.Module):
    def __init__(
        self, 
        f_model=None,
        f_preprocess=None,
        temp_event=0.2,
        temp_image=0.1,
    ):
        super(FTModel, self).__init__()

        self.encoder_q = copy.deepcopy(f_model)
        self.encoder_k = copy.deepcopy(f_model)

        for param_q, param_k in zip(
            self.encoder_q.parameters(),
            self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = True
            param_q.requires_grad = False

        self.T_image = temp_image
        self.T_event = temp_event
        self.foundation_model = copy.deepcopy(f_model)
        self.foundation_preprocess = copy.deepcopy(f_preprocess)

    def sinkhorn(self, out):
        Q = torch.exp(out).t()
        B = Q.shape[1]
        K = Q.shape[0]

        sum_Q = torch.sum(Q)
        dist.all_reduce(sum_Q)
        Q = Q / sum_Q.detach()

        for it in range(3):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            dist.all_reduce(sum_of_rows)
            Q = Q / sum_of_rows.detach()
            Q = Q / K

            Q = Q / torch.sum(Q, dim=0, keepdim=True)
            Q = Q / B
        Q = Q * B
        return Q.t()

    def get_kl_loss(self, q1, q2, k1):
        q1 = q1.view(q1.size(0), -1)
        q2 = q2.view(q2.size(0), -1)
        k1 = k1.view(k1.size(0), -1)
        
        q1 = nn.functional.normalize(q1, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)
        k1 = nn.functional.normalize(k1, dim=1)
        k2 = k1

        f = nn.LogSoftmax(dim=-1)
        q = torch.einsum('nc,mc->nm', [q1, q2]) / self.T_image
        k = torch.einsum('nc,mc->nm', [k1, k2]) / self.T_image
        return nn.KLDivLoss(
            reduction='batchmean',
            log_target=False
        )(f(q), self.sinkhorn(k))

    def contrastive_loss(self, q, k, T, l2_norm=True):
        if l2_norm:
            q = nn.functional.normalize(q, dim=1)
            k = nn.functional.normalize(k, dim=1)
        k = concat_all_gather(k)
        logits = torch.einsum('nc,mc->nm', [q, k]) / T
        N = logits.shape[0]
        labels = (
            torch.arange(N, dtype=torch.long) + \
            N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * T)

    def get_event(self, img, event):
        event = event.mean(1)
        event = torch.stack([event, event, event], 1)
        ratio2 = 1
        ratio1 = 0
        
        event1 = event * (1 - ratio1) + img * ratio1
        event2 = event * (1 - ratio2) + img * ratio2

        return event, event2, event1

    def forward(self, img, event_frame):
        _, event_frame1, event_frame2 = self.get_event(img, event_frame)
        
        with torch.no_grad():
            pred_q = self.encoder_q.encode_image(event_frame1)
            emb = self.foundation_model.encode_image(img)
        
        pred_k = self.encoder_k.encode_image(event_frame2).float()
            
        kl_loss = self.get_kl_loss(pred_q, pred_k, emb)
        moco_ct_loss = self.contrastive_loss(pred_q, pred_k, self.T_event)
        
        return moco_ct_loss, \
               kl_loss, \
               pred_k, \
               pred_q, \
               emb 
        

@torch.no_grad()
def concat_all_gather(tensor):
    tensors = [torch.ones_like(tensor)
               for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors, tensor, async_op=False)
    output = torch.cat(tensors, dim=0)
    return output