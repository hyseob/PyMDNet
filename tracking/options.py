from collections import OrderedDict


opts = OrderedDict()
opts['use_gpu'] = True

opts['model_path'] = 'models2/mdnet_vot-otb_e50_b128_c1_d2_l5.pth'

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
opts['trans_f_limit'] = 1.5
opts['topk'] = 5

# training examples sampling
opts['trans_f_pos'] = 0.1
opts['scale_f_pos'] = 1.2
opts['trans_f_neg'] = 1.5
opts['scale_f_neg'] = 1.2
opts['aspect_f'] = 1.1

# bounding box regression
opts['n_bbreg'] = 1000
opts['overlap_bbreg'] = [0.6, 1]
opts['scale_bbreg'] = [1, 2]
opts['trans_f_bbreg'] = 0.3
opts['scale_f_bbreg'] = 1.5
opts['aspect_f_bbreg'] = 1.2

# initial training
opts['lr_init'] = 0.001
opts['maxiter_init'] = 100
opts['n_pos_init'] = 500
opts['n_neg_init'] = 5000
opts['overlap_pos_init'] = [0.7, 1]
opts['overlap_neg_init'] = [0, 0.5]

# online training
opts['lr_update'] = 0.001
opts['maxiter_update_short'] = 20
opts['maxiter_update_long'] = 20
opts['n_pos_update'] = 50
opts['n_neg_update'] = 200
opts['overlap_pos_update'] = [0.7, 1]
opts['overlap_neg_update'] = [0, 0.3]

# update criteria
opts['success_thr'] = 0
opts['n_frames_long'] = 100
opts['n_frames_short'] = 20
opts['n_frames_neg'] = 20
opts['long_interval'] = 10

# training 
opts['grad_clip'] = 1
opts['lr_mult'] = {'fc6': 10}
opts['ft_layers'] = ['fc']
