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
    save_home = '../result_fig'
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
        savefig_dir = os.path.join(save_home, seq_name)
        if not os.path.exists(savefig_dir):
            os.makedirs(savefig_dir)
    else:
        savefig_dir = ''

    return img_list, init_bbox, gt, savefig_dir, args.display, result_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('seq_path', type=str, help='path to a sequence folder in VOT')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')

    args = parser.parse_args()

    # Generate sequence config
    img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)

    # Run tracker
    result_bb, fps = run_mdnet(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display)

    # Save result
    res = {'res': result_bb.round().tolist(), 'type': 'rect', 'fps': fps}
    json.dump(res, open(result_path, 'w'), indent=2)
