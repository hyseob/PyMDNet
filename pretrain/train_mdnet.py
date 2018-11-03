from __future__ import print_function

import argparse
import pickle
import time

import torch.optim as optim

from data_prov import *
from model import *
from options import *

if not opts['random']:
    np.random.seed(123)
    torch.manual_seed(456)
    torch.cuda.manual_seed(789)


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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def dampen(self, ratio):
        self.sum *= ratio
        self.count *= ratio

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_mdnet(gpu):
    # Init dataset.
    print('Loading training data...')
    with open(opts['data_path'], 'rb') as fp:
        training_data = pickle.load(fp)
    K = len(training_data)
    print('Training MDNet with {} domains...'.format(K))

    # Init model.
    print('Initializing model...')
    if opts['model_type'].lower() == 'ResNet18'.lower():
        model = MDNetResNet18(opts['init_model_path'], K)
    else:
        model = MDNetVGGM(opts['init_model_path'], K)
    if opts['use_gpu']:
        torch.cuda.set_device(int(gpu))
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])

    print('Creating datasets...')
    datasets = []
    for seqpath, seq in training_data.items():
        img_list = seq['images']
        gt = seq['gt']
        datasets.append(RegionDataset(seqpath, img_list, gt, opts))

    # Recover training information.
    if opts['init_model_path'] is None or not opts['init_model_path'].endswith('.pth'):
        best_prec = 0.
    else:
        states = torch.load(opts['init_model_path'])
        best_prec = states['best_prec']

    # Init criterion and optimizer.
    cls_criterion = ClassificationLoss()
    inst_emb_criterion = InstanceEmbeddingLoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, opts['lr'])

    loss_meter = AverageMeter()
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

            pos_score = model(pos_regions, range(K))
            neg_score = model(neg_regions, k)
            cls_loss = cls_criterion(pos_score[k], neg_score)

            inst_emb_loss = inst_emb_criterion(pos_score, k)

            loss = cls_loss + opts['inst_emb_loss_weight'] * inst_emb_loss

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
            optimizer.step()

            prec[k] = evaluator(pos_score[k], neg_score)

            toc = time.time() - tic
            loss_meter.update(loss.data[0])
            print("Cycle %2d, K %2d (%2d), Loss %.3f (%.3f), Prec %.3f, Time %.3f" %
                  (i, j, k, loss_meter.avg, loss.data[0], prec[k], toc))
        loss_meter.dampen(0.1)

        cur_prec = prec.mean()
        print("Mean Precision: %.3f" % cur_prec)
        if cur_prec > best_prec:
            best_prec = cur_prec
            if opts['use_gpu']:
                model = model.cpu()
            states = {'shared_layers': model.layers.state_dict(),
                      'best_prec': cur_prec}
            print("Saving model to %s" % opts['model_path'])
            torch.save(states, opts['model_path'])
            if opts['use_gpu']:
                model = model.cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=str, help='id of GPU to use, -1 for cpu', default='0')
    args = parser.parse_args()
    train_mdnet(gpu=args.gpu)
