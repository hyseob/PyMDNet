from os import path as osp
from collections import OrderedDict

opts = OrderedDict()
opts['use_gpu'] = True

opts['random'] = False

# opts['data_path'] = 'data/vot-otb.pkl'
opts['data_path'] = 'data/imagenet.pkl'

# opts['model_type'] = 'VGG-M'
# opts['init_model_path'] = '../models/imagenet-vgg-m.mat'
# opts['model_path'] = '../models/mdnet_vot-otb_new.pth'
opts['model_type'] = 'ResNet18'
# opts['init_model_path'] = None
opts['init_model_path'] = '../models/mdnet_resnet_imagenet_new.pth'
opts['model_path'] = '../models/mdnet_resnet_imagenet_new.pth'

opts['batch_frames'] = 8
opts['batch_pos'] = 32
opts['batch_neg'] = 96

opts['overlap_pos'] = [0.7, 1]
opts['overlap_neg'] = [0, 0.5]

# opts['img_size'] = 107
# opts['padding'] = 16
opts['img_size'] = 48
opts['padding'] = 4

opts['lr'] = 0.0001
opts['w_decay'] = 0.0005
opts['momentum'] = 0.9
opts['grad_clip'] = 10
opts['ft_layers'] = ['conv', 'fc']
opts['lr_mult'] = {'conv4': 10, 'fc': 10}
opts['n_cycles'] = 50
opts['inst_emb_loss_weight'] = 0.1
