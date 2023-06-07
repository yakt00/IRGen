import argparse
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from model.IRGen import IRGen
from model.DictTree import TreeNode
from dataset.dataset_inshop import Inshop
from dataset.dataset_cub200 import CUB
from dataset.dataset_cars196 import Cars
from dataset.config import config_gnd
from utils.evaluate import compute_map
from utils.logger import get_logger
from utils.utils import test_transform
from model.CLIP.clip import clip
import pickle

@torch.no_grad()
def test(cfg, ks, ranks):
    
    gnd = cfg['gnd']

    map, aps, mpr, prs = compute_map(ranks, gnd, ks)

    for k in ks:
        cnt = 0
        for i in range(len(ranks[0])):
            flag = 0
            for j in range(k):
                if ranks[j][i] in gnd[i]:
                    flag = 1
            cnt += flag
        print('rank@{}:'.format(k), cnt/len(ranks[0]))
    return map, aps, mpr, prs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test IRGen')
    parser.add_argument('--data_dir', default='../data/isc/Img', type=str, help='datasets path')
    parser.add_argument('--data_name', default='isc', type=str, choices=['car', 'cub', 'isc'],
                        help='dataset name')
    parser.add_argument('--file_name', default='in-shop_clothes_retrieval.pkl', type=str)
    parser.add_argument('--codes', default='isc_256rq4_1fc_ids.pkl', type=str, help='image identifier')
    parser.add_argument('--model_dir', default='/home/tinzhan/amlt/isc_256rq4_1fc/isc_rq_400.pkl', type=str)
    parser.add_argument('--beam_size', default=30, type=int, help='test beam size')
    parser.add_argument('--ks', default=[1,10,20,30])

    opt = parser.parse_args()


    # data_dir = os.environ["AMLT_DATA_DIR"]
    data_dir, data_name = opt.data_dir, opt.data_name
    file_name = opt.file_name
    codes = opt.codes
    beam_size, ks = opt.beam_size, opt.ks
    model_dir = opt.model_dir

    # data_dir = '../data/cub/images'
    # file_name = 'CUB200_retrieval.pkl'
    # codes = 'cub_256rq4_rqfc10_ids.pkl'
    # beam_size = 8
    # ks = [1,2,4,8]
    # model_dir = '/home/tinzhan/amlt/cub_256rq4_rqfc10_lm1_le4e-5/cub/cub_rq_120.pkl'
    
    with torch.no_grad():
        with open(os.path.join(data_dir, codes),'rb')as f:
            gnd = pickle.load(f)
        k_tree = gnd['dict_tree']
        mapping = gnd['mapping']
        ids = torch.tensor(mapping).cuda()
        if data_name == 'isc':
            ids = ids[-12612:]
        id_length = ids.shape[-1]

        model = IRGen(dec_depth=12, num_classes=256, id_len=id_length)
        mm, preprocess = clip.load('ViT-B-16.pt')
        mm = mm.to('cpu')
        mm=mm.type(torch.float32)
        model.encoder = mm.visual
        checkpoint = torch.load(model_dir)
        state_dict = checkpoint['net']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model = model.cuda()

        if data_name == 'isc':
            Dataset = Inshop(data_dir, 'query',transform=test_transform(256,224))
        elif data_name == 'cub':
            Dataset = CUB(data_dir, 'query',transform=test_transform(256,224))
        elif data_name == 'car':
            Dataset = Cars(data_dir, 'query',transform=test_transform(256,224))
        test_loader = DataLoader(Dataset, batch_size=1)
        cfg = config_gnd(data_dir, file_name)
        gnd = cfg['gnd']
        logger = get_logger('test.log')

        for i, img in enumerate(test_loader):
            img = img.cuda()

            # out = model.rerank(img,ids=ids)
            out = model.beam_search(img,k=beam_size,k_tree=k_tree,ids=ids)
            if i==0:
                ranks = np.array(out)
            else:
                ranks = np.concatenate((ranks,out),axis=0)
    
            logger.info('number:{} image \t preds{}'.format(i, out))
        ranks = np.asarray(ranks).T
        map, aps, mpr, prs = test(cfg, ks, ranks)
        logger.info('map:{}, mpr:{}'.format(map,mpr))


