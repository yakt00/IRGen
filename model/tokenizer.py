import math
from functools import partial
import pickle
import os
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from time import time

from model.model import Encoder

class vit_classify(nn.Module):
    
    def __init__(self,img_size=[224], patch_size=16, in_chans=3, num_classes=100, embed_dim=768, enc_depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.encoder = Encoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=enc_depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        self.fc = nn.Linear(embed_dim, num_classes)
        self._reset_parameters()
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        # B, C, H, W = src.shape
        # B, N = tgt.shape  
        enc_output = self.encoder(src)
        # tgt = torch.mean(enc_output,1)
        tgt = enc_output[:,0,:]
        tgt = self.fc(tgt)
        return tgt


class vitrqfc(nn.Module):
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=100, embed_dim=768, enc_depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.encoder = Encoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=enc_depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.fc_rq = nn.Linear(embed_dim, num_classes)
        self._reset_parameters()
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, x_q):
        x = self.encoder(src)[:,0,:]
        pred = self.fc(x)

        loss_quant = []
        pred_rq = []
        for i in range(len(x_q)):
            x_q[i] = x + (x_q[i] - x).detach()
            loss_quant.append((x - x_q[i].detach()).pow(2.0).mean())
            pred_rq.append(self.fc_rq(x_q[i]))

        return x, loss_quant, pred, pred_rq