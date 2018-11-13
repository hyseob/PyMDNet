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


class FilterMeta:
    def __init__(self):
        self.evolved = False
        self.gradient_sq_acc = 0
        self.gradient_sq_potential = 0

    def gradient_norm(self):
        return self.gradient_sq_potential and np.sqrt(self.gradient_sq_acc) / self.gradient_sq_potential

    def report_gradient_sq(self, grad_sq):
        self.gradient_sq_potential += 1
        self.gradient_sq_acc += grad_sq

    def report_evolution(self):
        self.evolved = True

    def dampen_gradient_rec(self, factor):
        self.gradient_sq_potential *= factor
        self.gradient_sq_potential *= factor


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
        self.filters_meta = {
            param_name: [FilterMeta() for _ in range(self.model.get_num_filters(param_name))]
            for param_name in self.model.ft_layers
        }

        # Initialize the target-relevance loss mask.
        self.tr_loss_mask = {
            param_name: torch.autograd.Variable(torch.zeros(self.model.get_num_filters(param_name)),
                                                requires_grad=False)
            for param_name in self.model.ft_layers
        }

        # Use GPU.
        self.use_gpu = opts['use_gpu'] and gpu >= 0
        if self.use_gpu:
            torch.cuda.set_device(gpu)
            self.model = self.model.cuda()
            for layer_name in self.tr_loss_mask:
                self.tr_loss_mask[layer_name] = self.tr_loss_mask[layer_name].cuda()

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

        grad_ratio_thresh = opts['grad_ratio_thresh']

        # dampen previous recorded gradients
        dampen_factor = opts['grad_dampen_factor']
        for layer in fe_layers:
            for filter_meta in self.filters_meta[layer]:
                filter_meta.dampen_gradient_rec(dampen_factor)

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

            # compute classificaiton loss
            cls_loss = criterion(pos_score, neg_score)

            # record gradients
            self.model.zero_grad()
            cls_loss.backward(retain_graph=True)
            for layer in fe_layers:
                gradients_sq = torch.pow(self.model.probe_filters_gradients(layer), 2).cpu()
                filters_in_layer = self.filters_meta[layer]
                for idx, gradient_sq in enumerate(gradients_sq):
                    filters_in_layer[idx].report_gradient_sq(gradient_sq)

            # compute target-relevance loss
            if opts['enable_fe']:
                tr_loss = opts['tr_loss_ratio'] * sum([
                    torch.sum(torch.sum(torch.cat([-pos_outputs[layer], neg_outputs[layer]],
                                                  dim=0),
                                        dim=0)
                              * self.tr_loss_mask[layer])
                    for layer in fe_layers
                ])
                tr_loss.backward()

            # optimize
            torch.nn.utils.clip_grad_norm(self.model.parameters(), opts['grad_clip'])
            optimizer.step()

            # evolve filters
            if opts['enable_fe']:
                for layer in fe_layers:
                    filters_in_layer = self.filters_meta[layer]
                    gradient_norm_sum = 0
                    for filter_meta in filters_in_layer:
                        gradient_norm_sum += filter_meta.gradient_norm()
                    mean_gradient_norm = gradient_norm_sum / len(filters_in_layer)
                    filters_to_evolve = list(
                        filter(lambda filter_idx:
                               not filters_in_layer[filter_idx].evolved and
                               filters_in_layer[filter_idx].gradient_norm() < mean_gradient_norm * grad_ratio_thresh,
                               range(len(filters_in_layer)))
                    )

                    if len(filters_to_evolve) > 0:
                        self.tr_loss_mask[layer][filters_to_evolve] = 1
                        for filter_idx in filters_to_evolve:
                            filters_in_layer[filter_idx].report_evolution()
                        if self.verbose:
                            print('Evolved {} filters in {}: {}'.format(len(filters_to_evolve),
                                                                        layer,
                                                                        filters_to_evolve))

            final_loss = cls_loss.data[0]
            # print("Iter %d, Loss %.4f" % (iter, final_loss))

        return final_loss
