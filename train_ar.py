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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import pytorch_warmup as warmup
from dataset.dataset_inshop import Inshop
from dataset.dataset_cub200 import CUB
from dataset.dataset_cars196 import Cars
from dataset.dataset_imagenet_val500 import ImageNet
from dataset.dataset_places365 import Places
from model.CLIP.clip import clip
from model.IRGen import IRGen
from model.DictTree import TreeNode
from utils.utils import LabelSmoothingCrossEntropy, train_transform, get_trainable_params
from utils.logger import get_logger
import os
import pickle
import scheduler

if __name__ == '__main__':
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()

    parser = argparse.ArgumentParser(description='Train IRGen')
    parser.add_argument('--data_dir', default='/mnt/default/data/isc/isc/Img', type=str, help='datasets path')
    parser.add_argument('--data_name', default='car', type=str, choices=['car', 'cub', 'isc', 'imagenet', 'places'],
                        help='dataset name')
    parser.add_argument('--file_name', default='in-shop_clothes_retrieval_trainval.pkl', type=str)
    parser.add_argument('--codes', default='isc_256rq4_ids.pkl', type=str, help='image identifier')
    parser.add_argument('--pretrained_model', default=None, type=str)
    parser.add_argument('--output_dir', default='./', type=str)
    parser.add_argument('--lr', default=8e-5, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=400, type=int, help='train epoch number')
    parser.add_argument('--smoothing', default=0.1, type=int, help='labelsmoothce smoothing')
    parser.add_argument('--local_rank', default=0, type=int)

    opt = parser.parse_args()

    # args parse
    data_name, data_dir = opt.data_name, opt.data_dir
    file_name = opt.file_name
    codes = opt.codes
    pretrained_model = opt.pretrained_model
    output_dir = opt.output_dir

    num_epochs, batch_size, lr = opt.num_epochs, opt.batch_size, opt.lr

    with open(os.path.join(data_dir, codes),'rb')as f:
        gnd = pickle.load(f)
    mapping = gnd['mapping']
    id_length = mapping.shape[-1]
    num_classes = np.unique(mapping).shape[0]

    if data_name == 'isc':
        Dataset = Inshop(data_dir, 'db',transform=train_transform(256,224))
    elif data_name == 'cub':
        Dataset = CUB(data_dir, 'db',transform=train_transform(256,224))
    elif data_name == 'car':
        Dataset = Cars(data_dir, 'db',transform=train_transform(256,224))
    elif data_name == 'imagenet':
        Dataset = ImageNet(data_dir, 'db',transform=train_transform(256,224))
    elif data_name == 'places':
        Dataset = Places(data_dir, 'db',transform=train_transform(256,224))

    train_sampler = DistributedSampler(Dataset, shuffle=True)

    if data_name in ['isc', 'cub', 'car']:
        model = IRGen(dec_depth=12, num_classes=num_classes, id_len=id_length)
    else:
        model = IRGen(dec_depth=24, num_classes=num_classes, id_len=id_length)
    mm, preprocess = clip.load('ViT-B-16.pt')
    mm = mm.to('cpu')
    mm=mm.type(torch.float32)
    model.encoder = mm.visual
    if pretrained_model:
        checkpoint = torch.load(model_dir)
        state_dict = checkpoint['net']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], 
                                    output_device=local_rank)


    train_loader = DataLoader(Dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8)

    optimizer = AdamW([{'params': get_trainable_params(model.module.decoder)},
                    {'params':get_trainable_params(model.module.encoder), 'lr': 0.01*lr}
                    ], lr=lr, betas=(0.9, 0.96), eps=1e-08, weight_decay=0.05)
    scheduler = scheduler.CosineLRScheduler(optimizer=optimizer,
                                                t_initial=100,
                                                lr_min=5e-7,
                                                warmup_t=20,
                                                warmup_lr_init=5e-8,
                                                cycle_decay=1,
                                                cycle_limit = 4,
                                                )

    # criterion = nn.CrossEntropyLoss() 
    criterion = LabelSmoothingCrossEntropy()
    logger = get_logger('train.log')
    with open(os.path.join(data_dir, file_name), 'rb') as fin:
        gnd = pickle.load(fin)
    class_idx = gnd['class_idx']

    Loss = []
    LR = []
    for i in range(num_epochs):
        train_loader.sampler.set_epoch(num_epochs)
        if dist.get_rank() == 0:
            print('start epoch',i)
        for j,(img,clss,idx) in enumerate(train_loader):
            idx = [np.random.rand(1)*len(class_idx[int(clss[k])]) for k in range(len(img))]
            tgt = np.array([mapping[class_idx[int(clss[k])][int(idx[k])]] for k in range(len(img))])
            tgt[np.where(tgt==-1)]=num_classes
            target = torch.tensor(tgt,dtype=torch.int64)

            img = img.cuda()
            target = target.cuda()
            
            output = model(img, tgt=target)
            output = output[:,:-1,:]
            output = output.reshape(-1,num_classes)
            target = target.reshape(-1)
            loss = criterion(output, target, args.smoothing)
            
            running_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j%10 == 9:
                if dist.get_rank() == 0:
                    logger.info('Epoch:[{}/{}]\t step={}\t loss={:.10f}\t lr={:.8f}'.format(i , num_epochshs, j+1, loss, optimizer.param_groups[0]['lr'] ))
            percent=(j+1)/len(train_loader)
            scheduler.step(i+percent)
        scheduler.step(i+1)

        if dist.get_rank() == 0:
            print('epoch:', i, 'loss:', loss)
            if i>90 and i%50 == 49:
                state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':i+1, 'loss':running_loss}
                torch.save(state, os.path.join(output_dir,'isc_rq_{}.pkl').format(i+1))

        dist.barrier()
