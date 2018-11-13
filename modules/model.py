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

# Solve the problem of loading models from PyTorch > 0.4.0.
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


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
        self.ft_layers = set()
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
        for param_name, param in self.params.items():
            if any([param_name.startswith(l) for l in layers]):
                layer_name = param_name.split('.')[0]
                if first_learnable_layer is None:
                    first_learnable_layer = layer_name
                self.ft_layers.add(layer_name)
                param.requires_grad = True
            else:
                last_fixed_layer = param_name.split('.')[0]
                param.requires_grad = False
        return first_learnable_layer, last_fixed_layer

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params

    def forward(self, x, k=0, in_layer='conv1', out_layer=''):
        #
        # forward model from in_layer to out_layer
        outputs = {}
        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x = module(x)
                if name == out_layer:
                    return x
                if isinstance(out_layer, list) and name in out_layer:
                    outputs[name] = x

        if isinstance(k, Iterable):
            x = [self.branches[i](x) for i in k]
        else:
            x = self.branches[k](x)
        if isinstance(out_layer, list):
            outputs['score'] = x
            return outputs
        else:
            return x

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers, strict=False)

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])

    def probe_filters_gradients(self, layer_name):
        return self.params[self.find_last_param_of_layer(layer_name, 'bias')].grad.data

    def find_last_param_of_layer(self, layer_name, spec):
        last_param_name = None
        for param_name in self.params:
            if layer_name in param_name and spec in param_name:
                last_param_name = param_name
        return last_param_name

    def get_num_filters(self, layer_name):
        return self.params[self.find_last_param_of_layer(layer_name, 'bias')].shape[0]


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
