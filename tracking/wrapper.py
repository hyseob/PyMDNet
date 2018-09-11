import os
import sys
import time
import numpy as np

from PIL import Image
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

modules_path = os.path.join(os.path.dirname(os.path.join(os.path.realpath(__file__))),
                            '../modules')
sys.path.insert(0, modules_path)

from tracker import Tracker
from utils import overlap_ratio


def run_mdnet(img_list, init_bbox, gt=None, savefig_dir='', display=False):
    # Init bbox
    target_bbox = np.array(init_bbox)
    result_bb = np.zeros((len(img_list), 4))
    result_bb[0] = target_bbox

    tic = time.time()
    # Load first image
    image = Image.open(img_list[0]).convert('RGB')

    # Initialize the tracker
    tracker = Tracker(init_bbox, image)

    spf_total = time.time() - tic

    # Display
    savefig = savefig_dir != ''
    if display or savefig:
        dpi = 80.0
        figsize = (image.size[0] / dpi, image.size[1] / dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=int(dpi))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image, aspect='auto')

        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3],
                                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)

        rect = plt.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3],
                             linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir, '0000.jpg'), dpi=dpi)

    # Main loop
    for i in range(1, len(img_list)):
        tic = time.time()
        # Load image
        image = Image.open(img_list[i]).convert('RGB')

        # Track and save result
        result_bb[i], target_score = tracker.track(image)

        spf = time.time() - tic
        spf_total += spf

        # Display
        if display or savefig:
            im.set_data(image)

            if gt is not None:
                gt_rect.set_xy(gt[i, :2])
                gt_rect.set_width(gt[i, 2])
                gt_rect.set_height(gt[i, 3])

            rect.set_xy(result_bb[i, :2])
            rect.set_width(result_bb[i, 2])
            rect.set_height(result_bb[i, 3])

            if display:
                plt.pause(.01)
                plt.draw()
            if savefig:
                fig.savefig(os.path.join(savefig_dir, '%04d.jpg' % (i)), dpi=dpi)

        if gt is None:
            print("Frame %d/%d, Score %.3f, Time %.3f" % \
                  (i, len(img_list), target_score, spf))
        else:
            print("Frame %d/%d, Overlap %.3f, Score %.3f, Time %.3f" %
                  (i, len(img_list), overlap_ratio(gt[i], result_bb[i])[0], target_score, spf))

    fps = len(img_list) / spf_total
    return result_bb, fps