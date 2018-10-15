from __future__ import print_function

import argparse
import json
import os

import numpy as np

from wrapper import run_mdnet


def poly2rec(poly):
    left = min(poly[::2])
    right = max(poly[::2])
    top = min(poly[1::2])
    bottom = max(poly[1::2])
    return [left, top, right - left, bottom - top]


def gen_config(args):
    savevideo_home = '../result_video'
    savefig_home = '../result_fig'
    result_home = '../result'

    seq_path = args.seq_path
    seq_name = os.path.basename(seq_path)
    img_dir = os.path.join(seq_path, 'color')
    gt_path = os.path.join(seq_path, 'groundtruth.txt')

    img_list = os.listdir(img_dir)
    img_list.sort()
    img_list = [os.path.join(img_dir, x) for x in img_list]

    gt_poly = np.loadtxt(gt_path, delimiter=',')
    gt = np.array(list(map(poly2rec, gt_poly)))
    init_bbox = gt[0]

    result_dir = os.path.join(result_home, seq_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = os.path.join(result_dir, 'result.json')

    if args.savefig:
        savefig_dir = os.path.join(savefig_home, seq_name)
        if not os.path.exists(savefig_dir):
            os.makedirs(savefig_dir)
    else:
        savefig_dir = ''

    if args.savevideo:
        savevideo_dir = os.path.join(savevideo_home, seq_name)
        if not os.path.exists(savevideo_dir):
            os.makedirs(savevideo_dir)
    else:
        savevideo_dir = ''

    return img_list, init_bbox, gt, savefig_dir, savevideo_dir, args.display, result_path, seq_name, args.gpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('seq_path', type=str, help='path to a sequence folder in VOT')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-v', '--savevideo', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')
    parser.add_argument('-g', '--gpu', type=str, help='id of GPU to use, -1 for cpu', default='0')
    parser.add_argument('-v', '--verbose', action='store_false', help='print verbose logs')

    args = parser.parse_args()

    # Generate sequence config
    img_list, init_bbox, gt, savefig_dir, savevideo_dir, display, result_path, seq_name, gpu = gen_config(args)

    # Run tracker
    result_bb, fps = run_mdnet(img_list, init_bbox, gt=gt,
                               seq_name=seq_name,
                               savefig_dir=savefig_dir,
                               savevideo_dir=savevideo_dir,
                               display=display,
                               gpu=gpu,
                               verbose=args.verbose)

    # Save result
    res = {'res': result_bb.round().tolist(), 'type': 'rect', 'fps': fps}
    json.dump(res, open(result_path, 'w'), indent=2)
