import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

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
categories = {}
with open('categories_places365.txt', 'r') as f:
    line = f.readline()
    while line:
        info = line.split()
        clss = ('-').join(info[0].split('/')[2:])
        categories[clss] = info[1]
        line = f.readline()

with open('train.txt', 'r') as f:
    line = f.readline()
    while line:
        imlist.append(line)
        classes.append(categories[line.split('/')[1]])
        line = f.readline()

with open('val.txt', 'r') as f:
    line = f.readline()
    while line:
        qimlist.append(line)
        qclasses.append(categories[line.split('/')[1]])
        line = f.readline()



class_idx = {}
for i in range(len(classes)):
    if classes[i] not in class_idx.keys():
        for j in range(len(classes)):
            if classes[j] not in class_idx.keys():
                class_idx[classes[j]] = [j]
            else:
                class_idx[classes[j]].append(j)


gnd = []
for i in range(len(qclasses)):
    gt = []
    for j in range(len(classes)):
        if classes[j]==qclasses[i]:
            gt.append(j)
    gnd.append(gt)

gnd_file = {'gnd':gnd, 'imlist':imlist, 'qimlist':qimlist, 'classes':np.asarray(classes), 'qclasses':np.asarray(qclasses), 'class_idx': class_idx}
with open('places365_retrieval.pkl', 'wb') as f:
    pickle.dump(gnd_file, f)

