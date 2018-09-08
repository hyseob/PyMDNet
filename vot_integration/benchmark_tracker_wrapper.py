from __future__ import print_function

import os
import sys

import vot
from PIL import Image
from trax.region import Rectangle


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


tracking_module_path = os.path.join(os.path.dirname(os.path.join(os.path.realpath(__file__))),
                                    '../tracking')
eprint('Loading tracker from', tracking_module_path)
sys.path.insert(0, tracking_module_path)
from tracker import Tracker

handle = vot.VOT("rectangle")
selection = handle.region()

# Process the first frame
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)
first_frame = Image.open(imagefile).convert('RGB')

mdnet = Tracker((selection.x, selection.y, selection.width, selection.height), first_frame)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    frame = Image.open(imagefile).convert('RGB')

    pred_bbox, confidence = mdnet.track(frame)

    handle.report(Rectangle(pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]), confidence)
