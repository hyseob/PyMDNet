import os
from collections import OrderedDict

opts = OrderedDict()
opts['use_gpu'] = True

opts['random'] = False

opts['model_type'] = 'VGG-M'
opts['model_path'] = os.path.join(os.path.dirname(os.path.join(os.path.realpath(__file__))),
                                  '../models/mdnet_vot-otb.pth')
# opts['model_type'] = 'ResNet18'
#opts['model_path'] = None
# opts['model_path'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/mdnet_resnet_imagenet.pth')

opts['img_size'] = 107
opts['padding'] = 16
# opts['img_size'] = 48
# opts['padding'] = 4

opts['batch_pos'] = 32
opts['batch_neg'] = 96
opts['batch_neg_cand'] = 1024
opts['batch_test'] = 256

opts['n_samples'] = 256
opts['trans_f'] = 0.6
opts['scale_f'] = 1.05
opts['trans_f_expand'] = 1.5

opts['n_bbreg'] = 1000
opts['overlap_bbreg'] = [0.6, 1]
opts['scale_bbreg'] = [1, 2]

opts['lr_init'] = 0.0001
opts['maxiter_init'] = 30
opts['n_pos_init'] = 500
opts['n_neg_init'] = 5000
opts['overlap_pos_init'] = [0.7, 1]
opts['overlap_neg_init'] = [0, 0.5]

opts['lr_update'] = 0.0002
opts['maxiter_update'] = 15
opts['n_pos_update'] = 50
opts['n_neg_update'] = 200
opts['overlap_pos_update'] = [0.7, 1]
opts['overlap_neg_update'] = [0, 0.3]

opts['success_thr'] = 0
opts['n_frames_short'] = 20
opts['n_frames_long'] = 100
opts['long_interval'] = 10

opts['w_decay'] = 0.0005
opts['momentum'] = 0.9
opts['grad_clip'] = 10
opts['lr_mult'] = {'conv4': 10, 'fc_ds': 10}
# opts['ft_layers'] = ['fc']
opts['ft_layers'] = ['conv4', 'fc']

opts['bbreg_layer'] = 'conv3'
# opts['bbreg_layer'] = 'conv4'

# Filter evolution options.
opts['enable_fe'] = True
opts['fe_layers'] = ['fc4', 'fc5']
opts['tr_loss_ratio'] = 1
# opts['fe_layers'] = ['conv4', 'fc_ds']
opts['grad_ratio_thresh'] = 0.01
opts['converge_loss_thresh'] = 0
opts['grad_dampen_factor'] = 0.9
