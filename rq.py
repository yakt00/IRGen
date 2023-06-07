import argparse
import faiss
import numpy as np
import os
import pickle
from model.DictTree import TreeNode


parser = argparse.ArgumentParser(description='tokenizer')
parser.add_argument('--data_name', default='car', type=str, choices=['car', 'cub', 'isc'],
                    help='dataset name')
parser.add_argument('--features', default='inshop_vitrqfc2_e200_trainval.npy', type=str)
parser.add_argument('--file_name', default='isc_256rq2_1fc_ids.pkl', type=str)
parser.add_argument('--data_dir', default='../data/isc', type=str)

opt = parser.parse_args()

data_name = opt.data_name
features = opt.features
file_name = opt.file_name
data_dir = opt.data_dir

data = np.load('features')

dim = data.shape[1]
m = 2
k = 8
pq = faiss.ProductQuantizer(dim, 1, k)
x_q=[]
for i in range(m):
    print(i)
    pq.train(data)
    codes = pq.compute_codes(data)
    if i == 0:
        rq_codes = codes
        codebook = faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)
        datarec = pq.decode(codes)
    else:
        rq_codes = np.concatenate((rq_codes,codes),axis=1)
        codebook = np.concatenate((codebook,faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)),axis=0)
        datarec += pq.decode(codes)
    x_q.append(datarec.copy())
    data -= pq.decode(codes)

print(rq_codes.shape)

kmeans_tree = TreeNode()
if data_name == 'isc':
    kmeans_tree.insert_many(rq_codes[-12612:,:])
else:
    kmeans_tree.insert_many(rq_codes)

gnd = {'mapping':rq_codes, 'codebook':codebook, 'dict_tree': kmeans_tree}
with open(os.path.join(data_dir, file_name),'wb')as f:
    pickle.dump(gnd,f)
