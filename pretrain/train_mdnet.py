from __future__ import print_function

import pickle
import time

import torch.optim as optim

from data_prov import *
from model import *
from options import *

data_path = 'data/vot-otb.pkl'


def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    lr = None
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


def train_mdnet():
    # Init dataset.
    with open(data_path, 'rb') as fp:
        training_data = pickle.load(fp)

    K = len(training_data)
    datasets = []
    for seqpath, seq in training_data.items():
        img_list = seq['images']
        gt = seq['gt']
        datasets.append(RegionDataset(seqpath, img_list, gt, opts))

    # Init model.
    model = MDNet(opts['init_model_path'], K)
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])

    # Init criterion and optimizer.
    criterion = BinaryLoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, opts['lr'])

    best_prec = 0.
    for i in range(opts['n_cycles']):
        print("==== Start Cycle %d ====" % i)
        k_list = np.random.permutation(K)
        prec = np.zeros(K)
        for j, k in enumerate(k_list):
            tic = time.time()
            pos_regions, neg_regions = datasets[k].next()

            pos_regions = Variable(pos_regions)
            neg_regions = Variable(neg_regions)

            if opts['use_gpu']:
                pos_regions = pos_regions.cuda()
                neg_regions = neg_regions.cuda()

            pos_score = model(pos_regions, k)
            neg_score = model(neg_regions, k)

            loss = criterion(pos_score, neg_score)
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
            optimizer.step()

            prec[k] = evaluator(pos_score, neg_score)

            toc = time.time() - tic
            print("Cycle %2d, K %2d (%2d), Loss %.3f, Prec %.3f, Time %.3f" %
                  (i, j, k, loss.data[0], prec[k], toc))

        cur_prec = prec.mean()
        print("Mean Precision: %.3f" % cur_prec)
        if cur_prec > best_prec:
            best_prec = cur_prec
            if opts['use_gpu']:
                model = model.cpu()
            states = {'shared_layers': model.layers.state_dict()}
            print("Save model to %s" % opts['model_path'])
            torch.save(states, opts['model_path'])
            if opts['use_gpu']:
                model = model.cuda()


if __name__ == "__main__":
    train_mdnet()
