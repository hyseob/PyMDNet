import sys
import pickle
import yaml
import time
import numpy as np

import torch

sys.path.insert(0,'.')
from data_prov import RegionDataset
from modules.model import MDNet, set_optimizer, BCELoss, Precision

opts = yaml.safe_load(open('pretrain/options.yaml','r'))


def train_mdnet():

    # Init dataset
    with open(opts['data_path'], 'rb') as fp:
        data = pickle.load(fp)
    K = len(data)
    dataset = [None] * K
    for k, seq in enumerate(data.values()):
        dataset[k] = RegionDataset(seq['images'], seq['gt'], opts)

    # Init model
    model = MDNet(opts['init_model_path'], K)
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])

    # Init criterion and optimizer
    criterion = BCELoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, opts['lr'], opts['lr_mult'])

    # Main trainig loop
    for i in range(opts['n_cycles']):
        print('==== Start Cycle {:d}/{:d} ===='.format(i + 1, opts['n_cycles']))

        if i in opts.get('lr_decay', []):
            print('decay learning rate')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opts.get('gamma', 0.1)

        # Training
        model.train()
        prec = np.zeros(K)
        k_list = np.random.permutation(K)
        for j, k in enumerate(k_list):
            tic = time.time()
            # training
            pos_regions, neg_regions = dataset[k].next()
            if opts['use_gpu']:
                pos_regions = pos_regions.cuda()
                neg_regions = neg_regions.cuda()
            pos_score = model(pos_regions, k)
            neg_score = model(neg_regions, k)

            loss = criterion(pos_score, neg_score)
            model.zero_grad()
            loss.backward()
            if 'grad_clip' in opts:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
            optimizer.step()

            prec[k] = evaluator(pos_score, neg_score)

            toc = time.time()-tic
            print('Iter {:2d} (Domain {:2d}), Loss {:.3f}, Precision {:.3f}, Time {:.3f}'
                    .format(j, k, loss.item(), prec[k], toc))

        print('Mean Precision: {:.3f}'.format(prec.mean()))
        print('Save model to {:s}'.format(opts['model_path']))
        if opts['use_gpu']:
            model = model.cpu()
        states = {'shared_layers': model.layers.state_dict()}
        torch.save(states, opts['model_path'])
        if opts['use_gpu']:
            model = model.cuda()


if __name__ == "__main__":
    train_mdnet()
