import os
import numpy as np
import pickle
from collections import OrderedDict

import xml.etree.ElementTree
import xmltodict

seq_home = 'datasets/ILSVRC/'
output_path = 'pretrain/data/imagenet_vid.pkl'

train_list = [p for p in os.listdir(seq_home + 'Data/VID/train')]
seq_list = []
for num, cur_dir in enumerate(train_list):
    seq_list += [os.path.join(cur_dir, p) for p in os.listdir(seq_home + 'Data/VID/train/' + cur_dir)]

data = {}
completeNum = 0
for i, seqname in enumerate(seq_list):
    print('{}/{}: {}'.format(i, len(seq_list), seqname))
    seq_path = seq_home + 'Data/VID/train/' + seqname
    gt_path = seq_home +'Annotations/VID/train/' + seqname
    img_list = sorted([p for p in os.listdir(seq_path) if os.path.splitext(p)[1] == '.JPEG'])

    enable_gt = []
    enable_img_list = []
    save_enable = True
    gt_list = sorted([os.path.join(gt_path, p) for p in os.listdir(gt_path) if os.path.splitext(p)[1] == '.xml'])
    
    for gidx in range(0, len(img_list)):
        with open(gt_list[gidx]) as fd:
            doc = xmltodict.parse(fd.read())
        try:
            try:
                object_ = doc['annotation']['object'][0]
            except:
                object_ = doc['annotation']['object']
        except:
            ## no object, occlusion and hidden etc.
            continue

        if int(object_['trackid']) != 0:
            continue

        xmin = float(object_['bndbox']['xmin'])
        xmax = float(object_['bndbox']['xmax'])
        ymin = float(object_['bndbox']['ymin'])
        ymax = float(object_['bndbox']['ymax'])

        ## discard too big object
        if (float(doc['annotation']['size']['width']) / 2. < xmax - xmin ) and \
            (float(doc['annotation']['size']['height']) / 2. < ymax - ymin ):
            continue

        cur_gt = np.zeros((4))
        cur_gt[0] = xmin
        cur_gt[1] = ymin
        cur_gt[2] = xmax - xmin
        cur_gt[3] = ymax - ymin
        
        enable_gt.append(cur_gt)
        enable_img_list.append(img_list[gidx])

    if len(enable_img_list) == 0:
        save_enable = False
        
    if save_enable:
        assert len(enable_img_list) == len(enable_gt), "Lengths do not match!!"
        enable_img_list = [os.path.join(seq_path, p) for p in enable_img_list]
        data[seqname] = {'images':enable_img_list, 'gt':np.asarray(enable_gt)}
        completeNum += 1
        print('Complete!')

# Save db
output_dir = os.path.dirname(output_path)
os.makedirs(output_dir, exist_ok=True)
with open(output_path, 'wb') as fp:
    pickle.dump(data, fp, -1)

print('complete {} videos'.format(completeNum))
