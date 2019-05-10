'''
使用mmdetection作为pose net模型的前处理
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import ipdb;pdb=ipdb.set_trace
import numpy as np
from tqdm import tqdm
from utilitys import plot_keypoint, PreProcess
import time

import torch
import _init_paths
from config import cfg
import config
from config import update_config

from utils.transforms import *
from lib.core.inference import get_final_preds
import cv2
import models
#  from lib.detector.yolo.human_detector import human_bbox_get as yolo_det
from lib.detector.mmdetection.high_api import human_boxes_get as mm_det
from collections import OrderedDict
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml',
                        #  default='experiments/coco/hrnet/w48_256x192_adam_lr1e-3.yaml',
                        #  default='experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    parser.add_argument("-i", "--video_input", help="input video file name", default="/home/xyliu/Videos/sports/dance.mp4")
    parser.add_argument("-o", "--video_output", help="output video file name", default="output/output.mp4")

    parser.add_argument('--camera', action='store_true')
    parser.add_argument('--display', action='store_true')

    args = parser.parse_args()
    return args



##### load model
def model_load(config):
    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )
    model_file_name  = 'models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth'
    #  model_file_name  = 'models/pytorch/pose_coco/pose_hrnet_w48_256x192.pth'
    #  model_file_name  = 'models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth'
    state_dict = torch.load(model_file_name)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k # remove module.
        #  print(name,'\t')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def ckpt_time(t0=None, display=None):
    if not t0:
        return time.time()
    else:
        t1 = time.time()
        if display:
            print('consume {:2f} second'.format(t1-t0))
        return t1-t0, t1


###### 加载human detecotor model
#  from lib.detector.yolo.human_detector import load_model as yolo_model
#  human_model = yolo_model()

from lib.detector.mmdetection.high_api import load_model as mm_model
human_model = mm_model() 

def main():
    args = parse_args()
    update_config(cfg, args)

    #### load pose-hrnet MODEL
    pose_model = model_load(cfg)
    #  pose_model = torch.nn.DataParallel(pose_model, device_ids=[0,1]).cuda()
    pose_model.cuda()

    from pycocotools.coco import COCO
    annFile = '/ssd/xyliu/data/coco/annotations/instances_val2017.json'
    im_root = '/ssd/xyliu/data/coco/images/val2017/'
    coco = COCO(annFile)
    catIds = coco.getCatIds(catNms=['person'])
    # 所有人体图片的id
    imgIds = coco.getImgIds(catIds=catIds )
    kpts_result = []
    detected_image_num = 0
    box_num = 0
    for imgId in tqdm(imgIds[:]):
        img = coco.loadImgs(imgId)[0]
        im_name = img['file_name']
        img = im_root + im_name
        img_input = plt.imread(img)

        try:
            bboxs, scores = mm_det(human_model, img_input, 0.3)
            inputs, origin_img, center, scale = PreProcess(img_input, bboxs, scores, cfg)

        except Exception as e:
            print(e)
            continue

        detected_image_num += 1
        with torch.no_grad():
            output = pose_model(inputs.cuda())
            preds, maxvals = get_final_preds(
                cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

            #  vis = np.ones(shape=maxvals.shape,)
            vis = maxvals
            preds = preds.astype(np.float16)
            keypoints = np.concatenate((preds, vis), -1)
            for k, s in zip(keypoints, scores.tolist()):
                box_num += 1
                k = k.flatten().tolist()
                item = {"image_id": imgId, "category_id": 1, "keypoints": k, "score":s}
                kpts_result.append(item)



    num_joints = 17
    in_vis_thre = 0.2
    oks_thre = 0.5
    oks_nmsed_kpts = []
    for i in range(len(kpts_result)):
        img_kpts = kpts_result[i]['keypoints']
        kpt = np.array(img_kpts).reshape(17,3)
        box_score = kpts_result[i]['score']
        kpt_score = 0
        valid_num = 0
        # each joint for bbox
        for n_jt in range(0, num_joints):
            # score
            t_s = kpt[n_jt][2]
            if t_s > in_vis_thre:
                kpt_score = kpt_score + t_s
                valid_num = valid_num + 1
        if valid_num != 0:
            kpt_score = kpt_score / valid_num

        # rescoring 关节点的置信度 与 box的置信度的乘积
        kpts_result[i]['score'] = kpt_score * box_score




    import json
    data = json.dumps(kpts_result)
    print('image num is {} \tdetected_image num is {}\t person num is {}'.format(len(imgIds), detected_image_num, box_num)),
    #  data = json.dumps(str(kpts_result))
    with open('person_keypoints.json', 'wt') as f:
        #  pass
        f.write(data)


if __name__ == '__main__':
    main()
