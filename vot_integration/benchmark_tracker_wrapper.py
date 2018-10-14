from __future__ import print_function

import os
import sys

import vot
from PIL import Image
from trax.region import Rectangle

script_dir = os.path.dirname(os.path.join(os.path.realpath(__file__)))
tracking_module_path = os.path.join(script_dir, '../tracking')
sys.path.insert(0, tracking_module_path)
from tracker import Tracker

print('Creating VOT handle...')
handle = vot.VOT("rectangle")
print('Retrieving the target selection...')
selection = handle.region()

# Process the first frame
print('Retrieving the first frame...')
imagefile = handle.frame()
if not imagefile:
    print('Cannot retrieve the first frame!')
    sys.exit(0)
first_frame = Image.open(imagefile).convert('RGB')

print('Initializing tracker...')
mdnet = Tracker((selection.x, selection.y, selection.width, selection.height),
                first_frame,
                gpu=1)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    frame = Image.open(imagefile).convert('RGB')

    pred_bbox, confidence = mdnet.track(frame)

    handle.report(Rectangle(pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]), confidence)

print('Exiting...')
