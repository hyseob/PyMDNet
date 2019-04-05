import sys
from sklearn.linear_model import Ridge
import numpy as np

from modules.utils import overlap_ratio


class BBRegressor():
    def __init__(self, img_size, alpha=1000, overlap=[0.6, 1], scale=[1, 2]):
        self.img_size = img_size
        self.alpha = alpha
        self.overlap_range = overlap
        self.scale_range = scale
        self.model = Ridge(alpha=self.alpha)

    def train(self, X, bbox, gt):
        X = X.cpu().numpy()
        bbox = np.copy(bbox)
        gt = np.copy(gt)

        if gt.ndim==1:
            gt = gt[None,:]

        r = overlap_ratio(bbox, gt)
        s = np.prod(bbox[:,2:], axis=1) / np.prod(gt[0,2:])
        idx = (r >= self.overlap_range[0]) * (r <= self.overlap_range[1]) * \
              (s >= self.scale_range[0]) * (s <= self.scale_range[1])

        X = X[idx]
        bbox = bbox[idx]

        Y = self.get_examples(bbox, gt)
        self.model.fit(X, Y)

    def predict(self, X, bbox):
        X = X.cpu().numpy()
        bbox_ = np.copy(bbox)

        Y = self.model.predict(X)

        bbox_[:,:2] = bbox_[:,:2] + bbox_[:,2:]/2
        bbox_[:,:2] = Y[:,:2] * bbox_[:,2:] + bbox_[:,:2]
        bbox_[:,2:] = np.exp(Y[:,2:]) * bbox_[:,2:]
        bbox_[:,:2] = bbox_[:,:2] - bbox_[:,2:]/2

        bbox_[:,:2] = np.maximum(bbox_[:,:2], 0)
        bbox_[:,2:] = np.minimum(bbox_[:,2:], self.img_size - bbox[:,:2])
        return bbox_

    def get_examples(self, bbox, gt):
        bbox[:,:2] = bbox[:,:2] + bbox[:,2:]/2
        gt[:,:2] = gt[:,:2] + gt[:,2:]/2

        dst_xy = (gt[:,:2] - bbox[:,:2]) / bbox[:,2:]
        dst_wh = np.log(gt[:,2:] / bbox[:,2:])

        Y = np.concatenate((dst_xy, dst_wh), axis=1)
        return Y

