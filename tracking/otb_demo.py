from __future__ import print_function

import argparse
import json
import os

import numpy as np

from wrapper import run_mdnet


def gen_config(args):
    if args.seq != '':
        # generate config from a sequence name

        seq_home = '../dataset/OTB'
        save_home = '../result_fig'
        result_home = '../result'

        seq_name = args.seq
        img_dir = os.path.join(seq_home, seq_name, 'img')
        gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt')

        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list = [os.path.join(img_dir, x) for x in img_list]

        gt = np.loadtxt(gt_path, delimiter=',')
        init_bbox = gt[0]

        savefig_dir = os.path.join(save_home, seq_name)
        result_dir = os.path.join(result_home, seq_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir, 'result.json')

    elif args.json != '':
        # load config from a json file

        param = json.load(open(args.json, 'r'))
        seq_name = param['seq_name']
        img_list = param['img_list']
        init_bbox = param['init_bbox']
        savefig_dir = param['savefig_dir']
        result_path = param['result_path']
        gt = None

    if args.savefig:
        if not os.path.exists(savefig_dir):
            os.makedirs(savefig_dir)
    else:
        savefig_dir = ''

    return img_list, init_bbox, gt, savefig_dir, args.display, result_path, seq_name, args.gpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='', help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')
    parser.add_argument('-g', '--gpu', type=str, help='id of GPU to use, -1 for cpu', default='0')
    parser.add_argument('-v', '--verbose', action='store_false', help='print verbose logs')

    args = parser.parse_args()
    assert (args.seq != '' or args.json != '')

    # Generate sequence config
    img_list, init_bbox, gt, savefig_dir, display, result_path, seq_name, gpu = gen_config(args)

    # Run tracker
    result_bb, fps = run_mdnet(img_list, init_bbox, gt=gt,
                               savefig_dir=savefig_dir, display=display, seq_name=seq_name,
                               gpu=gpu,
                               verbose=args.verbose)

    # Save result
    res = {'res': result_bb.round().tolist(), 'type': 'rect', 'fps': fps}
    json.dump(res, open(result_path, 'w'), indent=2)
