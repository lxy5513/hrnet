'''
使用yolov3作为pose net模型的前处理
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
from pose_utils import plot_keypoint, PreProcess
import time

import torch
import _init_paths
from config import cfg
import config
from config import update_config

from utils.transforms import *
from lib.core.inference import get_final_preds
import cv2
#  import ataset
import models
from lib.detector.yolo.human_detector import human_bbox_get as yolo_det
from lib.detector.mmdetection.high_api import human_boxes_get as mm_det


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
    from collections import OrderedDict
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
from lib.detector.yolo.human_detector import load_model as yolo_model
#  human_model = yolo_model()

from lib.detector.mmdetection.high_api import load_model as mm_model
human_model = mm_model()

def main():
    args = parse_args()
    update_config(cfg, args)

    if not args.camera:
        # handle video
        cam = cv2.VideoCapture(args.video_input)
        video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        cam = cv2.VideoCapture(0)
        video_length = 30000

    ret_val, input_image = cam.read()
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(args.video_output,fourcc, input_fps, (input_image.shape[1],input_image.shape[0]))


    #### load pose-hrnet MODEL
    pose_model = model_load(cfg)
    #  pose_model = torch.nn.DataParallel(pose_model, device_ids=[0,1]).cuda()
    pose_model.cuda()

    item = 0
    for i in tqdm(range(video_length-1)):

        x0 = ckpt_time()
        ret_val, input_image = cam.read()

        #  if args.camera:
            #  #  为取得实时速度，每两帧取一帧预测
            #  if item == 0:
                #  item = 1
                #  continue

        item = 0
        try:
            bboxs, scores = mm_det(human_model, input_image)
            # bbox is coordinate location
            inputs, origin_img, center, scale = PreProcess(input_image, bboxs, scores, cfg)
        except:
            out.write(input_image)
            cv2.namedWindow("enhanced",0);
            cv2.resizeWindow("enhanced", 960, 480);
            cv2.imshow('enhanced', input_image)
            cv2.waitKey(2)
            continue

        with torch.no_grad():
            # compute output heatmap
            inputs = inputs[:,[2,1,0]]
            output = pose_model(inputs.cuda())
            # compute coordinate
            preds, maxvals = get_final_preds(
                cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

        image = plot_keypoint(origin_img, preds, maxvals, 0.1)
        out.write(image)
        if args.display:
            ######### 全屏
            #  out_win = "output_style_full_screen"
            #  cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
            #  cv2.setWindowProperty(out_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            #  cv2.imshow(out_win, image)

            ########### 指定屏幕大小
            cv2.namedWindow("enhanced", cv2.WINDOW_GUI_NORMAL);
            cv2.resizeWindow("enhanced", 960, 480);
            cv2.imshow('enhanced', image)
            cv2.waitKey(1)

if __name__ == '__main__':
    main()
