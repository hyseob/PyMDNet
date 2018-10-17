import os
import sys
import time

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use('agg')
from matplotlib import pyplot as plt
import cv2
import torch

modules_path = os.path.join(os.path.dirname(os.path.join(os.path.realpath(__file__))),
                            '../modules')
sys.path.insert(0, modules_path)

modules_path = os.path.join(os.path.dirname(os.path.join(os.path.realpath(__file__))),
                            '../modules')
sys.path.insert(0, modules_path)

from tracker import Tracker
from utils import overlap_ratio
from options import *


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())


def pil2cv2(pil_image):
    return np.array(pil_image)[:, :, ::-1]


def fig2cv2(figure):
    return pil2cv2(fig2img(figure).convert('RGB'))


def run_mdnet(img_list, init_bbox, gt=None,
              savefig_dir='', savevideo_dir='',
              display=False,
              seq_name='unknown',
              gpu='0',
              verbose=False):
    # Init bbox
    target_bbox = np.array(init_bbox)
    result_bb = np.zeros((len(img_list), 4))
    result_bb[0] = target_bbox

    tic = time.time()
    # Load first image
    image = Image.open(img_list[0]).convert('RGB')

    # Initialize the tracker
    tracker = Tracker(init_bbox, image, int(gpu), verbose=verbose)

    spf_total = time.time() - tic

    # Display
    savefig = savefig_dir != ''
    savevideo = savevideo_dir != ''
    if savevideo:
        fn = os.path.join(savevideo_dir, seq_name + '.avi')
        video_writer = cv2.VideoWriter(fn,
                                       cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'),
                                       25,
                                       image.size)
        print('Saving result video to {}.'.format(fn))
    else:
        video_writer = None
    if display or savefig or savevideo:
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
        if savevideo:
            video_writer.write(fig2cv2(fig))

    overlap_ratios = []

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
        if display or savefig or savevideo:
            im.set_data(image)

            rect.set_xy(result_bb[i, :2])
            rect.set_width(result_bb[i, 2])
            rect.set_height(result_bb[i, 3])

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
            if savevideo:
                video_writer.write(fig2cv2(fig))

        if gt is not None:
            ratio = overlap_ratio(gt[i], result_bb[i])[0]
            overlap_ratios.append(ratio)
            print("Frame %d/%d, Overlap %.3f, Score %.3f, Time %.3f" %
                  (i, len(img_list), ratio, target_score, spf))
        else:
            print("Frame %d/%d, Score %.3f, Time %.3f" % \
                  (i, len(img_list), target_score, spf))

    save_PATH = '../models/latest.pth'
    torch.save(tracker.model.state_dict(), save_PATH)

    if savevideo:
        video_writer.release()

    if gt is not None:
        dir = os.path.join('analysis', 'data', seq_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        overlap_ratio_fn = os.path.join(dir, 'overlap_ratio.csv')
        print('Average overlap: {}'.format(np.average(overlap_ratios)))
        print('Writing overlap ratios to {}'.format(overlap_ratio_fn))
        with open(overlap_ratio_fn, 'w') as f:
            f.write(','.join(map(str, overlap_ratios)))

    if verbose:
        for layer_name in opts['fe_layers']:
            weight_norms = tracker.model.probe_filter_weight_norms(layer_name)

            weight_norm_sum_evolved = 0
            weight_norm_sum_not_evolved = 0
            grad_norm_sum_evolved = 0
            grad_norm_sum_not_evolved = 0
            filter_evolved_cnt = 0
            filter_not_evolved_cnt = 0

            for idx, filter_meta in enumerate(tracker.filters_meta[layer_name]):
                if filter_meta.evolution_cnt > 0:
                    weight_norm_sum_evolved += weight_norms[idx]
                    grad_norm_sum_evolved += filter_meta.gradient_norm()
                    filter_evolved_cnt += 1
                else:
                    weight_norm_sum_not_evolved += weight_norms[idx]
                    grad_norm_sum_not_evolved += filter_meta.gradient_norm()
                    filter_not_evolved_cnt += 1

            print('Average weights of {} filters with/without evolution: {}/{}'.format(
                layer_name,
                filter_evolved_cnt and weight_norm_sum_evolved / filter_evolved_cnt,
                filter_not_evolved_cnt and weight_norm_sum_not_evolved / filter_not_evolved_cnt
            ))

            print('Average gradient norm of {} filters with/without evolution: {}/{}'.format(
                layer_name,
                filter_evolved_cnt and grad_norm_sum_evolved / filter_evolved_cnt,
                filter_not_evolved_cnt and grad_norm_sum_not_evolved / filter_not_evolved_cnt
            ))

    fps = len(img_list) / spf_total
    return result_bb, fps
