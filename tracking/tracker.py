import os
import sys

import numpy as np
import torch
from torch import optim as optim
from torch.autograd import Variable

modules_path = os.path.join(os.path.dirname(os.path.join(os.path.realpath(__file__))),
                            '../modules')
sys.path.insert(0, modules_path)
from bbreg import BBRegressor
from data_prov import RegionExtractor
from model import MDNet, BinaryLoss
from options import *
from sample_generator import gen_samples, SampleGenerator

np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)


class Tracker:
    def __init__(self, init_bbox, first_frame):
        self.frame_idx = 0

        self.target_bbox = np.array(init_bbox)
        self.bbreg_bbox = self.target_bbox

        # Init model
        self.model = MDNet(opts['model_path'])
        if opts['use_gpu']:
            self.model = self.model.cuda()
        self.model.set_learnable_params(opts['ft_layers'])

        # Init criterion and optimizer
        self.criterion = BinaryLoss()
        self.init_optimizer = set_optimizer(self.model, opts['lr_init'])
        self.update_optimizer = set_optimizer(self.model, opts['lr_update'])

        # Train bbox regressor
        bbreg_examples = gen_samples(SampleGenerator('uniform', first_frame.size, 0.3, 1.5, 1.1),
                                     self.target_bbox, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])
        bbreg_feats = forward_samples(self.model, first_frame, bbreg_examples)
        self.bbreg = BBRegressor(first_frame.size)
        self.bbreg.train(bbreg_feats, bbreg_examples, self.target_bbox)

        # Draw pos/neg samples
        pos_examples = gen_samples(SampleGenerator('gaussian', first_frame.size, 0.1, 1.2),
                                   self.target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

        neg_examples = np.concatenate([
            gen_samples(SampleGenerator('uniform', first_frame.size, 1, 2, 1.1),
                        self.target_bbox, opts['n_neg_init'] // 2, opts['overlap_neg_init']),
            gen_samples(SampleGenerator('whole', first_frame.size, 0, 1.2, 1.1),
                        self.target_bbox, opts['n_neg_init'] // 2, opts['overlap_neg_init'])])
        neg_examples = np.random.permutation(neg_examples)

        # Extract pos/neg features
        pos_feats = forward_samples(self.model, first_frame, pos_examples)
        neg_feats = forward_samples(self.model, first_frame, neg_examples)
        self.feat_dim = pos_feats.size(-1)

        # Initial training
        train(self.model, self.criterion, self.init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])

        # Init sample generators
        self.sample_generator = SampleGenerator('gaussian', first_frame.size, opts['trans_f'], opts['scale_f'],
                                                valid=True)
        self.pos_generator = SampleGenerator('gaussian', first_frame.size, 0.1, 1.2)
        self.neg_generator = SampleGenerator('uniform', first_frame.size, 1.5, 1.2)

        # Init pos/neg features for update
        self.pos_feats_all = [pos_feats[:opts['n_pos_update']]]
        self.neg_feats_all = [neg_feats[:opts['n_neg_update']]]

    def track(self, image):
        self.frame_idx += 1

        # Estimate target bbox
        samples = gen_samples(self.sample_generator, self.target_bbox, opts['n_samples'])
        sample_scores = forward_samples(self.model, image, samples, out_layer='fc6')
        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu().numpy()
        target_score = top_scores.mean()

        success = target_score > opts['success_thr']

        # Expand search area at failure
        if success:
            self.sample_generator.set_trans_f(opts['trans_f'])
        else:
            self.sample_generator.set_trans_f(opts['trans_f_expand'])

        # Save result at success.
        if success:
            self.target_bbox = samples[top_idx].mean(axis=0)

            # Bbox regression
            bbreg_samples = samples[top_idx]
            bbreg_feats = forward_samples(self.model, image, bbreg_samples)
            bbreg_samples = self.bbreg.predict(bbreg_feats, bbreg_samples)
            self.bbreg_bbox = bbreg_samples.mean(axis=0)

        # Data collect
        if success:
            # Draw pos/neg samples
            pos_examples = gen_samples(self.pos_generator, self.target_bbox,
                                       opts['n_pos_update'],
                                       opts['overlap_pos_update'])
            neg_examples = gen_samples(self.neg_generator, self.target_bbox,
                                       opts['n_neg_update'],
                                       opts['overlap_neg_update'])

            # Extract pos/neg features
            pos_feats = forward_samples(self.model, image, pos_examples)
            neg_feats = forward_samples(self.model, image, neg_examples)
            self.pos_feats_all.append(pos_feats)
            self.neg_feats_all.append(neg_feats)
            if len(self.pos_feats_all) > opts['n_frames_long']:
                del self.pos_feats_all[0]
            if len(self.neg_feats_all) > opts['n_frames_short']:
                del self.neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(self.pos_feats_all))
            pos_data = torch.stack(self.pos_feats_all[-nframes:], 0).view(-1, self.feat_dim)
            neg_data = torch.stack(self.neg_feats_all, 0).view(-1, self.feat_dim)
            train(self.model, self.criterion, self.update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        # Long term update
        elif self.frame_idx % opts['long_interval'] == 0:
            pos_data = torch.stack(self.pos_feats_all, 0).view(-1, self.feat_dim)
            neg_data = torch.stack(self.neg_feats_all, 0).view(-1, self.feat_dim)
            train(self.model, self.criterion, self.update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        return self.bbreg_bbox, target_score


def forward_samples(model, image, samples, out_layer='conv3'):
    model.eval()
    extractor = RegionExtractor(image, samples, opts['img_size'], opts['padding'], opts['batch_test'])
    for i, regions in enumerate(extractor):
        regions = Variable(regions)
        if opts['use_gpu']:
            regions = regions.cuda()
        feat = model(regions, out_layer=out_layer)
        if i == 0:
            feats = feat.data.clone()
        else:
            feats = torch.cat((feats, feat.data.clone()), 0)
    return feats


def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while len(pos_idx) < batch_pos * maxiter:
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while len(neg_idx) < batch_neg_cand * maxiter:
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for iter in range(maxiter):

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx))
        batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx))

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start == 0:
                    neg_cand_score = score.data[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
            model.train()

        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
        optimizer.step()

        # print "Iter %d, Loss %.4f" % (iter, loss.data[0])
