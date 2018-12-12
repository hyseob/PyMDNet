import os
import sys

from torch import optim as optim

modules_path = os.path.join(os.path.dirname(os.path.join(os.path.realpath(__file__))),
                            '../modules')
sys.path.insert(0, modules_path)
from bbreg import BBRegressor
from data_prov import RegionExtractor
from model import *
from options import *
from sample_generator import gen_samples, SampleGenerator

if not opts['random']:
    np.random.seed(123)
    torch.manual_seed(456)
    torch.cuda.manual_seed(789)


class Tracker:
    def __init__(self, init_bbox, first_frame, gpu, verbose=False):
        self.verbose = verbose

        self.frame_idx = 0

        self.target_bbox = np.array(init_bbox)
        self.bbreg_bbox = self.target_bbox

        # Init model
        if verbose:
            print('Loading model from {}...'.format(opts['model_path']))
        if opts['model_type'].lower() == 'ResNet18'.lower():
            print('Using ResNet-18')
            self.model = MDNetResNet18(opts['model_path'])
        else:
            print('Using VGG-M')
            self.model = MDNetVGGM(opts['model_path'])

        # Probe the structure of the network.
        self.first_learnable_layer, self.last_fixed_layer = \
            self.model.set_learnable_params(opts['ft_layers'])

        # Use GPU.
        self.use_gpu = opts['use_gpu'] and gpu >= 0
        if self.use_gpu:
            torch.cuda.set_device(gpu)
            self.model = self.model.cuda()

        # Init criterion and optimizer
        self.criterion = ClassificationLoss()
        self.init_optimizer = self.set_optimizer(opts['lr_init'])
        self.update_optimizer = self.set_optimizer(opts['lr_update'])

        # Train bbox regressor
        self.bbreg = None
        bbreg_examples = gen_samples(SampleGenerator('uniform', first_frame.size, 0.3, 1.5, 1.1),
                                     self.target_bbox, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'],
                                     force_nonempty=False)
        if len(bbreg_examples) > 0:
            bbreg_feats = self.forward_samples(first_frame, bbreg_examples, out_layer=opts['bbreg_layer'])
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
        pos_feats = self.forward_samples(first_frame, pos_examples, out_layer=self.last_fixed_layer)
        neg_feats = self.forward_samples(first_frame, neg_examples, out_layer=self.last_fixed_layer)
        self.feat_dim = pos_feats.size()[1:]

        # Initial training
        final_loss = self.train(self.criterion, self.init_optimizer, pos_feats, neg_feats,
                                opts['maxiter_init'],
                                in_layer=self.first_learnable_layer)
        while opts['converge_loss_thresh'] != 0 and final_loss >= opts['converge_loss_thresh']:
            final_loss = self.train(self.criterion, self.init_optimizer, pos_feats, neg_feats,
                                    opts['maxiter_init'],
                                    in_layer=self.first_learnable_layer)

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
        sample_scores = self.forward_samples(image, samples, out_layer='')
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
            if self.bbreg is None:
                self.bbreg_bbox = self.target_bbox
            else:
                bbreg_samples = samples[top_idx]
                bbreg_feats = self.forward_samples(image, bbreg_samples, out_layer=opts['bbreg_layer'])
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
            pos_feats = self.forward_samples(image, pos_examples, out_layer=self.last_fixed_layer)
            neg_feats = self.forward_samples(image, neg_examples, out_layer=self.last_fixed_layer)
            self.pos_feats_all.append(pos_feats)
            self.neg_feats_all.append(neg_feats)
            if len(self.pos_feats_all) > opts['n_frames_long']:
                del self.pos_feats_all[0]
            if len(self.neg_feats_all) > opts['n_frames_short']:
                del self.neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(self.pos_feats_all))
            # Stack the samples from the selected frames together.
            pos_data = torch.stack(self.pos_feats_all[-nframes:], 0).view(-1, *self.feat_dim)
            neg_data = torch.stack(self.neg_feats_all, 0).view(-1, *self.feat_dim)
            final_loss = self.train(self.criterion, self.update_optimizer, pos_data, neg_data,
                                    opts['maxiter_update'],
                                    in_layer=self.first_learnable_layer)
            while opts['converge_loss_thresh'] != 0 and final_loss >= opts['converge_loss_thresh']:
                final_loss = self.train(self.criterion, self.update_optimizer, pos_data, neg_data,
                                        opts['maxiter_update'],
                                        in_layer=self.first_learnable_layer)

        # Long term update
        elif self.frame_idx % opts['long_interval'] == 0:
            # Stack the samples from all frames together.
            pos_data = torch.stack(self.pos_feats_all, 0).view(-1, *self.feat_dim)
            neg_data = torch.stack(self.neg_feats_all, 0).view(-1, *self.feat_dim)
            final_loss = self.train(self.criterion, self.update_optimizer, pos_data, neg_data,
                                    opts['maxiter_update'],
                                    in_layer=self.first_learnable_layer)
            while opts['converge_loss_thresh'] != 0 and final_loss >= opts['converge_loss_thresh']:
                final_loss = self.train(self.criterion, self.update_optimizer, pos_data, neg_data,
                                        opts['maxiter_update'],
                                        in_layer=self.first_learnable_layer)

        return self.bbreg_bbox, target_score

    def forward_samples(self, image, samples, out_layer):
        assert len(samples) > 0

        self.model.eval()
        extractor = RegionExtractor(image, samples, opts['img_size'], opts['padding'], opts['batch_test'])

        feats = None
        for i, regions in enumerate(extractor):
            regions = Variable(regions)
            if self.use_gpu:
                regions = regions.cuda()
            feat = self.model(regions, out_layer=out_layer)
            if feats is None:
                feats = feat.data.clone()
            else:
                feats = torch.cat((feats, feat.data.clone()), 0)
        return feats

    def set_optimizer(self, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'],
                      w_decay=opts['w_decay']):
        params = self.model.get_learnable_params()
        param_list = []
        lr = lr_base
        for k, p in params.items():
            lr = lr_base
            for l, m in lr_mult.items():
                if k.startswith(l):
                    lr = lr_base * m
            param_list.append({'params': [p], 'lr': lr})
        optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
        return optimizer

    def train(self, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
        self.model.train()

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

        final_loss = 0

        fe_layers = []
        for layer in self.model.ft_layers:
            for pattern in opts['fe_layers']:
                if pattern in layer:
                    fe_layers.append(layer)
                    break

        for iteration in range(maxiter):
            # select pos filter_idx
            pos_next = pos_pointer + batch_pos
            pos_cur_idx = pos_idx[pos_pointer:pos_next]
            pos_cur_idx = pos_feats.new(pos_cur_idx).long()
            pos_pointer = pos_next

            # select neg filter_idx
            neg_next = neg_pointer + batch_neg_cand
            neg_cur_idx = neg_idx[neg_pointer:neg_next]
            neg_cur_idx = neg_feats.new(neg_cur_idx).long()
            neg_pointer = neg_next

            # create batch
            batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx))
            batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx))

            # hard negative mining
            if batch_neg_cand > batch_neg:
                self.model.eval()
                neg_cand_score = None
                for start in range(0, batch_neg_cand, batch_test):
                    end = min(start + batch_test, batch_neg_cand)
                    score = self.model(batch_neg_feats[start:end], in_layer=in_layer)
                    if neg_cand_score is None:
                        neg_cand_score = score.data[:, 1].clone()
                    else:
                        neg_cand_score = torch.cat((neg_cand_score, score.data[:, 1].clone()), 0)

                _, top_idx = neg_cand_score.topk(batch_neg)
                batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
                self.model.train()

            # forward
            pos_outputs = self.model(batch_pos_feats, in_layer=in_layer, out_layer=fe_layers)
            neg_outputs = self.model(batch_neg_feats, in_layer=in_layer, out_layer=fe_layers)
            pos_score = pos_outputs['score']
            neg_score = neg_outputs['score']

            # compute classification loss
            cls_loss = criterion(pos_score, neg_score)

            # compute target-relevance loss
            if opts['enable_fe']:
                tr_loss = sum([
                    torch.sum(
                        torch.exp(-torch.max(pos_outputs[layer], dim=0)[0])
                        + torch.mean(torch.topk(neg_outputs[layer],
                                                int(neg_outputs[layer].shape[0] * 0.9),
                                                dim=0,
                                                largest=False,
                                                sorted=False)[0],
                                     dim=0))
                    for layer in fe_layers
                ])
                tr_loss_ratio = np.exp(-cls_loss.data[0]) * (1 - iteration / (maxiter - 1))
                loss = cls_loss + tr_loss * opts['tr_loss_base_ratio'] * tr_loss_ratio
            else:
                loss = cls_loss

            # optimize
            self.model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), opts['grad_clip'])
            optimizer.step()

            final_loss = cls_loss.data[0]
            # print("Iter %d, Loss %.4f" % (iteration, final_loss))

        return final_loss
