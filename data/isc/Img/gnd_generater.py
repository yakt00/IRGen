import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch

class TreeNode(object):
    def __init__(self):
        self.nodes = {}       
        self.is_leaf = False  
        self.count = 0        

    def insert(self,word):
        curr = self
        for c in word:
            if not curr.nodes.get(c,None):
                new_node = TreeNode()
                curr.nodes[c] = new_node
            curr = curr.nodes[c]
            curr.is_leaf = True
        self.count += 1
        return

    def insert_many(self,words):
        for word in words:
            self.insert(word)
        return

    def search(self,word):
        curr = self
        try:
            for c in word:
                curr = curr.nodes[c]
        except:
            return False
        return curr.is_leaf


imlist = []
qimlist = []
classes = []
qclasses = []
with open('../Eval/list_eval_partition.txt','r') as f:
    line = f.readline()
    line = f.readline()
    line = f.readline()
    while line:
        info =line.split()
        if info[2] == 'query':
            qimlist.append(info[0])
            qclasses.append(info[1])
        elif info[2] == 'gallery':
            imlist.append(info[0])
            classes.append(info[1])
        line = f.readline()

gnd = []
for i in range(len(qclasses)):
    gt = []
    for j in range(len(classes)):
        if classes[j]==qclasses[i]:
            gt.append(j)
    gnd.append(gt)


class_idx = {}
for i in range(len(classes)):
    if classes[i] not in class_idx.keys():
        for j in range(len(classes)):
            if classes[j] not in class_idx.keys():
                class_idx[classes[j]] = [j]
            else:
                class_idx[classes[j]].append(j)

gnd_file = {'gnd':gnd, 'imlist':imlist, 'qimlist':qimlist, 'classes':np.asarray(classes), 'qclasses':np.asarray(qclasses), 'class_idx':class_idx}
with open('in-shop_clothes_retrieval.pkl', 'wb') as f:
    pickle.dump(gnd_file, f)