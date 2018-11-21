from collections import OrderedDict


opts = OrderedDict()
opts['use_gpu'] = True

# data path
opts['data_path'] = 'pretrain/data/vot-otb.pkl'

# model path
opts['init_model_path'] = 'models/imagenet-vgg-m.mat'
opts['model_path'] = 'models/mdnet_vot-otb_e50_b128_c1_d2_hnm.pth'

# input size
opts['img_size'] = 107
opts['padding'] = 16

# batch size
opts['batch_frames'] = 8
opts['batch_pos'] = 32
opts['batch_neg'] = 96
opts['batch_neg_cand'] = 1024
opts['batch_test'] = 256

# training examples sampling
opts['trans_f_pos'] = 0.1
opts['scale_f_pos'] = 1.2
opts['trans_f_neg'] = 1.5
opts['scale_f_neg'] = 1.2
opts['aspect_f'] = 1.1

# training examples
opts['overlap_pos'] = [0.7, 1]
opts['overlap_neg'] = [0, 0.5]

# training
opts['lr'] = 0.001
opts['grad_clip'] = 1
opts['ft_layers'] = ['conv', 'fc']
opts['lr_mult'] = {'fc': 10}
opts['n_cycles'] = 50
opts['lr_decay'] = []
opts['gamma'] = 0.1
