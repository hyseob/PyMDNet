import os
import pickle
from os import listdir
from os import path as osp
import numpy as np
import xml.etree.ElementTree as EL
from PIL import Image
import matplotlib.pyplot as plt

output_path = 'data/imagenet.pkl'

imagenet_home = '/media/hdd1/datasets/ImageNet'
imagenet_vid_home = '/media/hdd1/datasets/ImageNetVideo/ILSVRC'

data = {}

# Treat each class in ImageNet as a class.
imagenet_img_home = osp.join(imagenet_home, 'Image')
imagenet_anno_home = osp.join(imagenet_home, 'Annotation')
for seq in listdir(imagenet_anno_home):
    img_list = []
    anno_list = sorted([p for p in listdir(osp.join(imagenet_anno_home, seq))
                        if p.endswith('.xml')])
    gt = []
    for anno in anno_list:
        e = EL.parse(osp.join(imagenet_anno_home, seq, anno)).getroot()

        folder = e.find('folder').text
        filename = e.find('filename').text
        img_path = osp.abspath(osp.join(imagenet_img_home, folder, filename + '.JPEG'))
        if not osp.exists(img_path):
            continue

        bndbox = e.find('object').find('bndbox')
        x_min = int(bndbox.find('xmin').text)
        y_min = int(bndbox.find('ymin').text)
        x_max = int(bndbox.find('xmax').text)
        y_max = int(bndbox.find('ymax').text)
        gt.append([x_min, y_min, x_max - x_min, y_max - y_min])
        img_list.append(img_path)

    print('Found {}/{} images in sequence {}!'.format(len(img_list), len(anno_list), seq))
    data[seq] = {'images': img_list, 'gt': np.array(gt)}

# Load data from ImageNet Video.
imagenet_vid_video_home = osp.join(imagenet_vid_home, 'Data/VID/train')
imagenet_vid_anno_home = osp.join(imagenet_vid_home, 'Annotations/VID/train')
for group in listdir(imagenet_vid_anno_home):
    for seq in listdir(osp.join(imagenet_vid_anno_home, group)):
        img_list = []
        anno_list = sorted([p for p in listdir(osp.join(imagenet_vid_anno_home, group, seq))
                            if p.endswith('.xml')])
        gt = []
        for anno in anno_list:
            e = EL.parse(osp.join(imagenet_anno_home, seq, anno)).getroot()
            if int(e.find('object').find('occluded').text):
                continue

            folder = e.find('folder').text
            filename = e.find('filename').text
            img_path = osp.abspath(osp.join(imagenet_vid_video_home, folder, filename + '.JPEG'))
            if not osp.exists(img_path):
                continue

            bndbox = e.find('object').find('bndbox')
            x_min = int(bndbox.find('xmin').text)
            y_min = int(bndbox.find('ymin').text)
            x_max = int(bndbox.find('xmax').text)
            y_max = int(bndbox.find('ymax').text)
            gt.append([x_min, y_min, x_max - x_min, y_max - y_min])
            img_list.append(img_path)

        print('Found {}/{} images in sequence {}!'.format(len(img_list), len(anno_list), seq))
        data[seq] = {'images': img_list, 'gt': np.array(gt)}

# Dump everything.
with open(output_path, 'wb') as fp:
    pickle.dump(data, fp, -1)
