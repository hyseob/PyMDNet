import sys
import pickle
import time
import numpy as np

import torch

sys.path.insert(0,'.')
from data_prov import RegionDataset
from options import opts
from modules.model import MDNet, set_optimizer, BinaryLoss, Accuracy

np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)


def train_mdnet():

    # Init dataset
    with open(opts['data_path'], 'rb') as fp:
        data = pickle.load(fp)
    K = len(data)
    dataset = [None] * K
    dataset_val = [None] * K
    for k, seq in enumerate(data.values()):
        n_val = opts['batch_frames']
        dataset[k] = RegionDataset(seq['images'][:-n_val], seq['gt'][:-n_val], opts)
        # Validate using last frames
        dataset_val[k] = RegionDataset(seq['images'][-n_val:], seq['gt'][-n_val:], opts)

    # Init model
    model = MDNet(opts['init_model_path'], K)
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])

    # Init criterion and optimizer
    criterion = BinaryLoss()
    evaluator = Accuracy()
    optimizer = set_optimizer(model, opts['lr'], opts['lr_mult'])
    best_acc = 0.

    # Main trainig loop
    for i in range(opts['n_cycles']):
        print('==== Start Cycle {:d}/{:d} ===='.format(i + 1, opts['n_cycles']))

        if i in opts['lr_decay']:
            print('decay learning rate')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opts['gamma']

        # Training
        model.train()
        train_acc = np.zeros(K)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
            optimizer.step()

            train_acc[k] = evaluator(pos_score, neg_score)

            toc = time.time()-tic
            print('Train: Iter {:2d} (Domain {:2d}), Loss {:.3f}, Acc {:.3f}, Time {:.3f}'
                    .format(j, k, loss.item(), train_acc[k], toc))

        # Validation
        model.eval()
        val_acc = np.zeros(K)
        for k in range(K):
            pos_regions, neg_regions = dataset_val[k].next()
            if opts['use_gpu']:
                pos_regions = pos_regions.cuda()
                neg_regions = neg_regions.cuda()
            with torch.no_grad():
                pos_score = model(pos_regions, k)
                neg_score = model(neg_regions, k)
            val_acc[k] = evaluator(pos_score, neg_score)
            print('Val: Domain {:2d}, Acc {:.3f}'.format(k, val_acc[k]))

        cur_acc = val_acc.mean()
        print('Mean Train Accuracy: {:.3f}'.format(train_acc.mean()))
        print('Mean Val Accuracy: {:.3f}'.format(cur_acc))
        if cur_acc > best_acc:
            best_acc = cur_acc
            if opts['use_gpu']:
                model = model.cpu()
            states = {'shared_layers': model.layers.state_dict()}
            print('Save model to {:s}'.format(opts['model_path']))
            torch.save(states, opts['model_path'])
            if opts['use_gpu']:
                model = model.cuda()


if __name__ == "__main__":
    train_mdnet()
