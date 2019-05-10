import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import ipdb;pdb=ipdb.set_trace
from collections import OrderedDict

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    print(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    print('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    print(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )

gt_anns = 'data/coco/annotations/person_keypoints_val2017.json'
#  dt_anns = '/home/xyliu/2D_pose/deep-high-resolution-net.pytorch/person_keypoints.json'
dt_anns = '/home/xyliu/2D_pose/simple-pose-estimation/person_keypoints.json'

annType = 'keypoints'
cocoGt=COCO(gt_anns)
cocoDt=cocoGt.loadRes(dt_anns)

cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

info_str = []
for ind, name in enumerate(stats_names):
    info_str.append((name, cocoEval.stats[ind]))

name_values = OrderedDict(info_str)
model_name = 'openpose'

if isinstance(name_values, list):
    for name_value in name_values:
        _print_name_value(name_value, model_name)
else:
    _print_name_value(name_values, model_name)
