import os
import re

import numpy as np
import torch.utils.data
from PIL import Image

import pickle as pkl
# Per-channel mean and SD values in BGR order
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

class Cars(torch.utils.data.Dataset):
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
        with open(os.path.join(self._data_path, 'car_retrieval.pkl'), 'rb') as fin:
            gnd = pkl.load(fin)
            if self._split == 'query':
                for i in range(len(gnd["qimlist"])):
                    im_fn = gnd["qimlist"][i]
                    im_path = os.path.join(
                        self._data_path, im_fn)
                    self._db.append({"im_path": im_path, "qclss": gnd['qclasses'][i]})
            elif self._split == 'db':
                for i in range(len(gnd["imlist"])):
                    im_fn = gnd["imlist"][i]
                    im_path = os.path.join(
                        self._data_path, im_fn)
                    self._db.append({"im_path": im_path, "class": gnd["classes"][i],  "idx": i})
            elif self._split == 'gallery':
                for i in range(len(gnd["gimlist"])):
                    im_fn = gnd["gimlist"][i]
                    im_path = os.path.join(
                        self._data_path, im_fn)
                    self._db.append({"im_path": im_path, "class": gnd["gclasses"][i],  "idx": i})

    def __getitem__(self, index):
        # Load the image
        try:
            im = Image.open(self._db[index]["im_path"])
            
        except:
            print('error: ', self._db[index]["im_path"])
        # x1, y1, x2, y2 = self._db[index]["crop"]
        # im = im.convert('RGB').crop((x1, y1, x2, y2))
        im = self._transform(im)
        if "class" in self._db[index].keys():
            return im, self._db[index]['class'], self._db[index]['idx']
        
        return im, self._db[index]['qclss']


    def __len__(self):
        return len(self._db)
