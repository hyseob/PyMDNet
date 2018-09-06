import sys
import os

from PIL import Image

import vot
from trax.region import Rectangle

sys.path.insert(0, os.path.join(os.getcwd(), '../tracking'))
from tracker import Tracker

handle = vot.VOT("rectangle")
selection = handle.region()

# Process the first frame
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)
first_frame = Image.open(imagefile).convert('RGB')

mdnet = Tracker(first_frame, (selection.x, selection.y, selection.width, selection.height))

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    frame = Image.open(imagefile).convert('RGB')

    pred_bbox, confidence = mdnet.track(frame)

    handle.report(Rectangle(pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]), confidence)
