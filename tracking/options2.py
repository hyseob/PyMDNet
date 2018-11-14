from collections import OrderedDict


opts = OrderedDict()
opts['use_gpu'] = True

opts['model_path'] = 'models/mdnet_vot-otb.pth'

# input size
opts['img_size'] = 107
opts['padding'] = 16

# batch size
opts['batch_pos'] = 32
opts['batch_neg'] = 96
opts['batch_neg_cand'] = 1024
opts['batch_test'] = 256

# candidates sampling
opts['n_samples'] = 256
opts['trans_f'] = 0.6
opts['scale_f'] = 1.05
opts['trans_f_expand'] = 1.2

# training examples sampling
opts['trans_f_pos'] = 0.1
opts['scale_f_pos'] = 1.1
opts['trans_f_neg'] = 1.5
opts['scale_f_neg'] = 1.2
opts['aspect_f'] = 1.05

# bounding box regression
opts['n_bbreg'] = 1000
opts['overlap_bbreg'] = [0.6, 1]
opts['scale_bbreg'] = [1, 2]
opts['trans_f_bbreg'] = 0.3
opts['scale_f_bbreg'] = 1.5

# initial training
opts['lr_init'] = 0.01
opts['maxiter_init'] = 30
opts['n_pos_init'] = 500
opts['n_neg_init'] = 5000
opts['overlap_pos_init'] = [0.7, 1]
opts['overlap_neg_init'] = [0, 0.5]

# online training
opts['lr_update'] = 0.01
opts['maxiter_update'] = 10
opts['n_pos_update'] = 50
opts['n_neg_update'] = 200
opts['overlap_pos_update'] = [0.7, 1]
opts['overlap_neg_update'] = [0, 0.3]

# update criteria
opts['success_thr'] = 0.4
opts['n_frames_short'] = 20
opts['n_frames_long'] = 100
opts['long_interval'] = 10

# training 
opts['w_decay'] = 0.0005
opts['momentum'] = 0.9
opts['grad_clip'] = 10
opts['lr_mult'] = {'fc6':10}
opts['ft_layers'] = ['fc']
