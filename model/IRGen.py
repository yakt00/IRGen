# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial
import pickle
import os
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from time import time

from model.model import Encoder, Decoder
from model.utils import trunc_normal_
from model.DictTree import TreeNode

def rrank(args):
    in_pre = args[0]
    v = args[1]
    out = args[2]
    k = args[4]
    clusters = args[5]
    k_tree = args[6]
    pred = np.ones(clusters*k)*-1
    for i in range(k):
        tmp = k_tree
        for p in in_pre[i]:
            tmp = tmp.nodes[int(p)]
        pr = v[i]*out[i,-1]
        for j in tmp.nodes.keys():
            pred[i*clusters+j] = pr[j]
    return pred

class IRGen(nn.Module):
    
    def __init__(self, id_len=4, img_size=[224], patch_size=16, in_chans=3, num_classes=100, embed_dim=768, enc_depth=12, dec_depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.encoder = Encoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=enc_depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                 qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        self.decoder = Decoder(embed_dim=embed_dim, num_classes=num_classes, id_len=id_len, depth=dec_depth, num_heads=num_heads,
                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate, norm_layer=norm_layer, **kwargs)
        self._reset_parameters()
        self.id_len = id_len
        self.num_classes = num_classes
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt=None):
        # B, C, H, W = src.shape
        # B, N = tgt.shape  
        enc_output = self.encoder(src)
        dec_output = self.decoder(tgt, enc_output)
        return dec_output

    def beam_search(self, src, k=8, clusters=256, k_tree=None, ids=None):
        enc_output = self.encoder(src)
        B, _, embed_dim = enc_output.shape
        enc_output = enc_output.reshape(B,-1).repeat(1,k).reshape(B,k,-1,embed_dim)
        enc_output = enc_output.reshape(B*k,-1,embed_dim)

        ids = ids.type(torch.LongTensor).cuda()

        sos = self.decoder(None, enc_output)
        sos = F.softmax(sos, dim=-1)

        pred = torch.ones(B,clusters)*-1
        for i in range(B):
            for j in k_tree.nodes.keys():
                pred[i,j] = sos[i,0,j]
        v, i = torch.topk(pred.cuda(), k=k, dim=-1)

        for m in range(1,ids.shape[-1]):
            
            in_cur = torch.zeros(B,k,m)
            if m==1:
                in_cur = i
            else:
                for p in range(B):    
                    for j in range(k):
                        in_cur[p,j,:-1] = in_pre[p*k+int(i[p,j]/clusters)]
                        in_cur[p,j,-1] = i[p,j]%clusters
            
            in_cur = in_cur.reshape(-1,m).type(torch.LongTensor).cuda()
            out = self.decoder(in_cur.cuda(),enc_output)
            out = F.softmax(out, dim=-1)
            v = v.reshape(-1,1)
            for n in range(B):
                if n==0:
                    pred = torch.tensor(rrank((in_cur[n*k:n*k+k],v[n*k:n*k+k],out[n*k:n*k+k],B,k,clusters,k_tree))).reshape(1,-1)
                else:
                    pred = torch.cat((pred, torch.tensor(rrank((in_cur[n*k:n*k+k],v[n*k:n*k+k],out[n*k:n*k+k],B,k,clusters,k_tree))).reshape(1,-1)),0)
            v, i = torch.topk(pred.cuda(), k=k, dim=-1)
            in_pre = in_cur
        in_cur = torch.zeros(B,k,ids.shape[-1]).cuda()
        ans = []
        for p in range(B):
            anss=[]    
            for j in range(k):
                in_cur[p,j,:-1] = in_pre[p*k+int(i[p,j]/clusters)]
                in_cur[p,j,-1] = i[p,j]%clusters
                in_cur[p,j] = in_cur[p,j].type(torch.LongTensor)
                pred = torch.where(torch.sum(in_cur[p,j]==ids,dim=1)==in_cur[p,j].shape[-1])
                if len(pred[0]) >0:
                    for jj in range(len(pred[0])):
                        anss.append(int(pred[0][jj]))
                else:
                    anss.append(0) 
            ans.append(anss[:k])
        return np.array(ans)

    def rerank(self, src, clusters=256, ids=None):
        enc_output = self.encoder(src)
        dec_output = torch.ones(len(ids)).cuda()
        batch=400
        for i in range(0,len(ids),batch):
            end = min(i+batch, len(ids))
            inputs = ids[i:end,:]
            dec_out = self.decoder(inputs.long(), enc_output)
            dec_out = F.softmax(dec_out, dim=-1)
            dec_out = dec_out.cuda()
            # dec_out[:,1:,:][torch.where(inputs==self.num_classes-1)]=torch.tensor([0]*(self.num_classes-1)+[1.]).cuda()
            for k in range(end-i):
                # dec_output[i+k] = dec_out[k,0,i0[k]]*dec_out[k,1,i1[k]]
                for p in range(inputs.shape[-1]):
                    dec_output[i+k] *= dec_out[k,p,int(inputs[k,p])]
        dec_output = dec_output.reshape(1,-1)
        rank = torch.argsort(-dec_output, dim=-1)
        return rank.cpu().numpy()