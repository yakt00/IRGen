import os
import re

import numpy as np
import torch.utils.data
from PIL import Image

import pickle as pkl
# Per-channel mean and SD values in BGR order
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

class ImageNet(torch.utils.data.Dataset):
    """Common dataset."""

    def __init__(self, data_path, split, transform):
        assert os.path.exists(
            data_path), "Data path '{}' not found".format(data_path)
        self._data_path, self._split, self._transform = data_path, split, transform
        self._construct_db()

    def _construct_db(self):
        """Constructs the db."""
        # Compile the split data path
        self._db = []
        with open(os.path.join(self._data_path, 'gnd_imagenet1k_v5k.pkl'), 'rb') as fin:
        # with open(os.path.join(self._data_path, 'gnd_imagenet_v500.pkl'), 'rb') as fin:
            gnd = pkl.load(fin)
            if self._split == 'query':
                for i in range(len(gnd["qimlist"])):
                    im_fn = gnd["qimlist"][i]
                    im_path = os.path.join(
                        self._data_path, im_fn)
                    self._db.append({"im_path": im_path, "gt_ids": gnd["gnd"][i], "qcls_idx": gnd["qcls_idx"][i]})
            elif self._split == 'db':
                for i in range(len(gnd["imlist"])):
                    im_fn = gnd["imlist"][i]
                    im_path = os.path.join(
                        self._data_path, im_fn)
                    self._db.append({"im_path": im_path, "id": gnd["id"][i], "class": gnd["classes"][i], "cls_idx": gnd["cls_idx"][i], "idx":i})
            

    def __getitem__(self, index):
        # Load the image
        try:
            im = Image.open(self._db[index]["im_path"])
            
        except:
            print('error: ', self._db[index]["im_path"])
        im = self._transform(im)
        if "id" in self._db[index].keys() and "class" in self._db[index].keys():
            return im, self._db[index]['idx'], self._db[index]["class"], self._db[index]["cls_idx"]
        
        return im

    def __len__(self):
        return len(self._db)
