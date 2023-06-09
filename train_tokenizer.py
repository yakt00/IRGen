import argparse
import datetime
import json
import random
from sched import scheduler
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import pytorch_warmup as warmup
from dataset.dataset_inshop import Inshop
from dataset.dataset_cub200 import CUB
from dataset.dataset_cars196 import Cars
from dataset.dataset_imagenet_val500 import ImageNet
from dataset.dataset_places365 import Places
from model.tokenizer import vitrqfc
from model.DictTree import TreeNode
from utils.utils import train_transform, get_trainable_params, residualquantizer
from utils.logger import get_logger
import os
import pickle
from model.CLIP.clip import clip
import scheduler
import tqdm


@torch.no_grad()
def get_features(Dataset, model):
    loader = DataLoader(Dataset, batch_size=32)
    with torch.no_grad():
        for i, (images,_,_) in enumerate(loader):
            features = model(images.cuda())[:,0,:]
            if i==0:
                feats = features.cpu()
            else:
                feats = torch.cat((feats, features.cpu()), 0)
    del loader
    return feats.numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Tokenizer')
    parser.add_argument('--data_dir', default='../data/isc/Img', type=str, help='datasets path')
    parser.add_argument('--data_name', default='isc', type=str, choices=['car', 'cub', 'isc', 'imagenet', 'places'],
                        help='dataset name')
    parser.add_argument('--pretrained_model', default=None, type=str)
    parser.add_argument('--feats', default='inshop_clip_trainval.npy', type=str)
    parser.add_argument('--output_dir', default='./', type=str)
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=200, type=int, help='train epoch number')
    parser.add_argument('--rq_weight', default=1e-7, type=float, help='tokenizer loss weight')
    parser.add_argument('--local_rank', default=0, type=int)
    opt = parser.parse_args()

    data_dir, data_name = opt.data_dir, opt.data_name
    num_epochs = opt.num_epochs
    batch_size = opt.batch_size
    lr = opt.lr
    rq_weight = opt.rq_weight
    output_dir = opt.output_dir

    if data_name == 'isc':
        num_classes = 7982
        Dataset = Inshop(data_dir, 'db',transform=train_transform(256,224))
    elif data_name == 'cub':
        num_classes = 200
        Dataset = CUB(data_dir, 'db',transform=train_transform(256,224))
    elif data_name == 'car':
        num_classes = 196
        Dataset = Cars(data_dir, 'db',transform=train_transform(256,224))  
    elif data_name == 'imagenet':
        num_classes = 1000
        Dataset = ImageNet(data_dir, 'db',transform=train_transform(256,224))
    elif data_name == 'places':
        num_classes = 365
        Dataset = Places(data_dir, 'db',transform=train_transform(256,224))


    model = vitrqfc(dec_depth=12, num_classes=num_classes)
    mm, preprocess = clip.load('ViT-B-16.pt')
    mm=mm.type(torch.float32)
    model.encoder = mm.visual
    model = model.cuda()

    if opt.feats != '':
        feats = np.load(opt.feats)
    else:
        feats = get_features(Dataset, model.encoder)


    train_loader = DataLoader(Dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    optimizer = AdamW([{'params': get_trainable_params(model.fc)},
                    {'params': get_trainable_params(model.fc_rq), 'lr':1*lr}, 
                    {'params':get_trainable_params(model.encoder), 'lr': 0.01*lr}
                    ], lr=lr, betas=(0.9, 0.96), eps=1e-08, weight_decay=0.05)
    scheduler = scheduler.CosineLRScheduler(optimizer=optimizer,
                                                t_initial=50,
                                                lr_min=1e-6,
                                                warmup_t=20,
                                                warmup_lr_init=1e-7,
                                                cycle_decay=0.5
                                                )
    criterion = nn.CrossEntropyLoss() 
    logger = get_logger('exp_vitrqfc.log')
    for i in range(num_epochs):
        print('start epoch',i)
        x_q, rq_code = residualquantizer(feats,4,8)
        for j,(img,target,idx) in enumerate(train_loader):
            img = img.cuda()
            target = target.unsqueeze(-1).cuda()
            z_q = [torch.tensor(np.array([x_q[l][k] for k in idx])).float().cuda() for l in range(len(x_q))]
            z, loss_quant, output, output_rq = model(img, z_q)
            import pdb
            pdb.set_trace()
            
            output_rq = [output_rq[k].reshape(-1,num_classes) for k in range(len(output_rq))]
            output = output.reshape(-1,num_classes)
            target = target.reshape(-1)

            loss_ce_rq = []
            for i in range(len(output_rq)):
                loss_ce_rq.append(criterion(output_rq[i], target))
            loss_ce = criterion(output, target)

            loss = 1e-7*(sum(loss_ce_rq)+ sum(loss_quant)) + loss_ce
            running_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j%10 == 9:
                logger.info('Epoch:[{}/{}]\t step={}\t loss={:.4f}\t loss_ce={:.2f}\t loss_ce_rq_0={:.2f}\tloss_ce_rq_1={:.2f}\tloss_ce_rq_2={:.2f}\tloss_ce_rq_3={:.2f}\t loss_quant={:.2f}\t lr={:.8f}'.format(i , num_epochs, j+1, loss, loss_ce, loss_ce_rq[0], loss_ce_rq[1], loss_ce_rq[2], loss_ce_rq[3], sum(loss_quant), optimizer.param_groups[0]['lr'] ))
            percent=(j+1)/len(train_loader)
            scheduler.step(i+percent)
        
        scheduler.step(i+1)
        feats = get_features(Dataset, model.encoder)
        
        print('epoch:', i, 'loss:', loss)
        if i%50 == 49:
            state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':i+1, 'loss':running_loss}
            torch.save(state, os.path.join(output_dir, '{}_vitrqfc_{}.pkl').format(data_name, i+1))
            np.save(os.path.join(output_dir, '{}_vitrqfc_e{}_trainval.npy').format(data_name, i+1),feats)