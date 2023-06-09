import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import glob




imlist = []
qimlist = []
imclass = []
qimclass = []
with open('ImageNet_order', 'rb') as f:
    imgs = pickle.load(f)
for i in range(len(imgs)):
    for j in range(len(imgs[i])):
        imlist.append('train/'+imgs[i][j])
        imclass.append(imgs[i][j].split('/')[0])

with open('ILSVRC2012_name_train.txt','r') as f:
    line = f.readline()
    while line:
        imlist.append('train/'+line[:-1])
        imclass.append(line[:9])
        line = f.readline()

classes = [i.split('/')[-1] for i in glob.glob('val/*')]
for cls in classes:
    qimlist += glob.glob('val/'+cls+'/*')
    qimclass += [cls]*len(glob.glob('val/'+cls+'/*'))

gnd_file['qimlist']=qimlist
gnd_file['qclasses']=qimclass
with open('gnd_imagenet1k_v50k.pkl','wb')as f:
    pickle.dump(gnd_file,f)
pdb.set_trace()
qimlist = qimlist[0:50000:10]
qimclass = qimclass[0:50000:10]


gnd = []
for i in range(len(qimclass)):
    gt = []
    for j in range(len(imclass)):
        if imclass[j]==qimclass[i]:
            gt.append(j)
    gnd.append(gt)

classes = np.unique(imclass)
cls_idx = []
for i in range(len(imclass)):
    cls_idx.append(classes.index(imclass[i]))

for i in range(len(cls_idx)):
    if cls_idx[i] not in class_idx.keys():
        class_idx[cls_idx[i]] = [i]
    else:
        class_idx[cls_idx[i]].append(i)


gnd_file = {'gnd':gnd, 'imlist':imlist, 'qimlist':qimlist, 'classes':np.asarray(imclass), 'qclasses':np.asarray(qimclass), 'cls_idx':np.array(cls_idx), 'class_idx': class_idx}
with open('gnd_imagenet1k_v5k.pkl', 'wb') as f:
    pickle.dump(gnd_file, f)