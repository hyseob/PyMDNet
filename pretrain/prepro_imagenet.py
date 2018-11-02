import pickle
import xml.etree.ElementTree as EL
from os import listdir
from os import path as osp
from threading import Thread, Lock, Semaphore

from PIL import Image
import numpy as np

output_path = 'data/imagenet.pkl'

imagenet_home = '/media/hdd1/datasets/ImageNet'
imagenet_vid_home = '/media/hdd1/datasets/ImageNetVideo/ILSVRC'

data = {}
data_mutex = Lock()
sem = Semaphore(64)
threads = []


def process_seq(seq, anno_dir, data_dir):
    img_list = []
    anno_list = sorted([p for p in listdir(osp.join(anno_dir, seq))
                        if p.endswith('.xml')])
    gt = []
    for anno in anno_list:
        e = EL.parse(osp.join(anno_dir, seq, anno)).getroot()
        if e.find('object') is None:
            continue
        if e.find('object').find('occluded') is not None and \
                int(e.find('object').find('occluded').text):
            continue

        folder = e.find('folder').text
        filename = e.find('filename').text
        img_path = osp.abspath(osp.join(data_dir, folder, filename + '.JPEG'))
        if not osp.exists(img_path):
            continue
        try:
            img = Image.open(img_path, 'r')
            img.verify()
        except (IOError, SyntaxError) as e:
            print('Bad file:', filename, '({})'.format(e))
            continue

        bndbox = e.find('object').find('bndbox')
        x_min = int(bndbox.find('xmin').text)
        y_min = int(bndbox.find('ymin').text)
        x_max = int(bndbox.find('xmax').text)
        y_max = int(bndbox.find('ymax').text)
        gt.append([x_min, y_min, x_max - x_min, y_max - y_min])
        img_list.append(img_path)

    print('Found {}/{} images in sequence {}!'.format(len(img_list), len(anno_list), seq))
    if len(img_list):
        data[seq] = {'images': img_list, 'gt': np.array(gt)}
    sem.release()


# # Treat each class in ImageNet as a class.
# imagenet_img_home = osp.join(imagenet_home, 'Image')
# imagenet_anno_home = osp.join(imagenet_home, 'Annotation')
#
# for seq in listdir(imagenet_anno_home):
#     t = Thread(target=process_seq, args=(seq, imagenet_anno_home, imagenet_img_home))
#     sem.acquire()
#     threads.append(t)
#     t.start()

# Load data from ImageNet Video.
imagenet_vid_video_home = osp.join(imagenet_vid_home, 'Data/VID/train')
imagenet_vid_anno_home = osp.join(imagenet_vid_home, 'Annotations/VID/train')
for group in listdir(imagenet_vid_anno_home):
    anno_dir = osp.join(imagenet_vid_anno_home, group)
    if not osp.isdir(anno_dir):
        continue
    for seq in listdir(anno_dir):
        t = Thread(target=process_seq, args=(seq, anno_dir, imagenet_vid_video_home))
        sem.acquire()
        threads.append(t)
        t.start()

# Dump everything when all threads have finished.
print('Waiting for threads to join...')
for t in threads:
    t.join()
print('Dumping {} sequences...'.format(len(data)))
with open(output_path, 'wb') as fp:
    pickle.dump(data, fp, -1)
