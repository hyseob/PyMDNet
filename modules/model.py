import os
from collections import OrderedDict, Iterable

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributions as tdist
from torchvision.models import resnet18


class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        #
        # x: N x C x H x W
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x ** 2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq, pad, pad, pad, pad), 2),
                            torch.cat((pad, x_sq, pad, pad, pad), 2),
                            torch.cat((pad, pad, x_sq, pad, pad), 2),
                            torch.cat((pad, pad, pad, x_sq, pad), 2),
                            torch.cat((pad, pad, pad, pad, x_sq), 2)), 1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:, 2:-2, :, :]
        x = x / ((2. + 0.0001 * x_sumsq) ** 0.75)
        return x


class LinView(nn.Module):
    def __init__(self):
        super(LinView, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet, self).__init__()
        self.K = K

        # To be filled by the build_layers method.
        self.layers = None
        self.branches = None

        self.build_layers(model_path)

        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError("Unkown model format: %s" % (model_path))

        self.params = None
        self.layer_names = set()
        self.next_layer_names = {}
        self.build_param_dict()

    def append_params(self, params, module, prefix):
        last_name = None
        for child_name, child in module.named_children():
            for k, p in child.named_parameters():
                if p is None:
                    continue

                if isinstance(child, nn.BatchNorm2d):
                    name = prefix + '.' + child_name + '_bn.' + k
                else:
                    name = prefix + '.' + child_name + '.' + k

                if 'bn' not in name and 'weight' in name:
                    layer_name = name[:-7]
                    if last_name is not None:
                        self.next_layer_names[last_name] = layer_name
                    last_name = layer_name

                if name not in params:
                    params[name] = p
                else:
                    raise RuntimeError("Duplicated param name: %s" % (name))

    def build_layers(self, model_path):
        raise NotImplementedError

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            self.append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            self.append_params(self.params, module, 'fc_ds_%d' % (k))

    def set_learnable_params(self, layers):
        last_fixed_layer = None
        first_learnable_layer = None
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                if first_learnable_layer is None:
                    first_learnable_layer = k
                p.requires_grad = True
            else:
                p.requires_grad = False
                last_fixed_layer = k
        return first_learnable_layer is not None and first_learnable_layer.split('.')[0], \
               last_fixed_layer is not None and last_fixed_layer.split('.')[0]

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params

    def forward(self, x, k=0, in_layer='conv1', out_layer=''):
        #
        # forward model from in_layer to out_layer

        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x = module(x)
                if name == out_layer:
                    return x

        if isinstance(k, Iterable):
            x = [self.branches[i](x) for i in k]
        else:
            x = self.branches[k](x)
        return x

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers)

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])

    # def probe_filters_gradients(self, block_idx, layer_idx):
    #     layer = self.layers[block_idx][layer_idx]
    #     layer_params = layer.weight
    #     grad = layer_params.grad
    #     return torch.norm(grad.view((grad.shape[0], len(grad.view(-1)) / grad.shape[0])), dim=1)

    def probe_filters_gradients(self, layer_name):
        return self.params[layer_name + '.bias'].grad.data

    def get_num_filters(self, layer_name):
        return self.params[layer_name + '.bias'].shape[0]

    def probe_filter_weight_norms(self, layer_name):
        weights = self.params[layer_name + '.weight'].data
        return torch.norm(weights.view((weights.shape[0], len(weights.view(-1)) / weights.shape[0])), dim=1)

    def evolve_filters(self, optimizer, layer_name, filters_to_evolve, init_bias):
        bias_params = self.params[layer_name + '.bias']
        weight_params = self.params[layer_name + '.weight']
        bias_params.data[filters_to_evolve] = init_bias
        weight_params.data[filters_to_evolve, ...] = 0
        optimizer.state[bias_params]['momentum_buffer'][filters_to_evolve] = 0
        optimizer.state[weight_params]['momentum_buffer'][filters_to_evolve, ...] = 0

        # Set the weights of units following the evolved filters to have the same distribution as others.
        filters_not_to_evolve = list(set(range(len(bias_params))).difference(set(filters_to_evolve)))
        weight_params = self.params[self.next_layer_names[layer_name] + '.weight']
        weights_not_evolved = weight_params.data[:, filters_not_to_evolve, ...].view(weight_params.shape[0], -1)
        mean = torch.mean(weights_not_evolved, dim=1)
        std = torch.std(weights_not_evolved, dim=1)
        dest_shape = weight_params.data[:, filters_to_evolve, ...].shape
        num_params_per_filter = int(np.prod(dest_shape[1:]))
        mean_tensor = mean.unsqueeze(1).repeat(1, num_params_per_filter).view(dest_shape)
        std_tensor = std.unsqueeze(1).repeat(1, num_params_per_filter).view(dest_shape)
        rand_weights = tdist.Normal(mean_tensor, std_tensor).sample()
        if weight_params.data.is_cuda:
            rand_weights = rand_weights.cuda()
        weight_params.data[:, filters_to_evolve, ...] = rand_weights
        # Clear the momentum.
        optimizer.state[weight_params]['momentum_buffer'][:, filters_to_evolve, ...] = 0

    def boost_gradients(self, layer_name, filter_indices, rate):
        bias_params = self.params[layer_name + '.bias']
        weight_params = self.params[layer_name + '.weight']
        bias_params.grad[filter_indices] *= rate
        weight_params.grad[filter_indices, ...] *= rate


class MDNetVGGM(MDNet):
    def __init__(self, model_path=None, K=1):
        super(MDNetVGGM, self).__init__(model_path, K)

    def build_layers(self, model_path):
        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    LinView())),
            ('fc4', nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(512 * 3 * 3, 512),
                                  nn.ReLU())),
            ('fc5', nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(512, 512),
                                  nn.ReLU()))]))

        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 2)) for _ in range(self.K)])


class MDNetResNet18(MDNet):
    def __init__(self, model_path=None, K=1):
        super(MDNetResNet18, self).__init__(model_path, K)

    def build_layers(self, model_path):
        resnet = resnet18(pretrained=model_path is None)
        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(resnet.conv1,
                                    resnet.bn1,
                                    resnet.relu,
                                    resnet.maxpool)),
            ('conv2', resnet.layer1),
            ('conv3', resnet.layer2),
            ('conv4', nn.Sequential(resnet.layer3,
                                    nn.AdaptiveAvgPool2d((1, 1)),
                                    LinView())),
        ]))

        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(256, 2)) for _ in range(self.K)])


class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score, dim=1)[:, 1]
        neg_loss = -F.log_softmax(neg_score, dim=1)[:, 0]

        loss = pos_loss.sum() + neg_loss.sum()
        return loss


class InstanceEmbeddingLoss(nn.Module):
    def __init__(self):
        super(InstanceEmbeddingLoss, self).__init__()

    def forward(self, K_scores, k):
        scores = torch.stack([scores[:, 1] for scores in K_scores], dim=1)
        loss = -F.log_softmax(scores, dim=1)[:, k]
        return loss.sum()


class Accuracy:
    def __init__(self):
        pass

    def __call__(self, pos_score, neg_score):
        pos_correct = (pos_score[:, 1] > pos_score[:, 0]).sum().float()
        neg_correct = (neg_score[:, 1] < neg_score[:, 0]).sum().float()

        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0]


class Precision:
    def __init__(self):
        pass

    def __call__(self, pos_score, neg_score):
        scores = torch.cat((pos_score[:, 1], neg_score[:, 1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8)

        return prec.data[0]
