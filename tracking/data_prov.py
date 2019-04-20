import sys
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

from modules.utils import crop_image2


class RegionExtractor():
    def __init__(self, image, samples, opts):
        self.image = np.asarray(image)
        self.samples = samples

        self.crop_size = opts['img_size']
        self.padding = opts['padding']
        self.batch_size = opts['batch_test']

        self.index = np.arange(len(samples))
        self.pointer = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples):
            self.pointer = 0
            raise StopIteration
        else:
            next_pointer = min(self.pointer + self.batch_size, len(self.samples))
            index = self.index[self.pointer:next_pointer]
            self.pointer = next_pointer
            regions = self.extract_regions(index)
            regions = torch.from_numpy(regions)
            return regions
    next = __next__

    def extract_regions(self, index):
        regions = np.zeros((len(index), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(self.samples[index]):
            regions[i] = crop_image2(self.image, sample, self.crop_size, self.padding)
        regions = regions.transpose(0, 3, 1, 2)
        regions = regions.astype('float32') - 128.
        return regions
