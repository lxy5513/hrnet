'''
SIMPLE POSE TRACKING
use boxes similarity as OKS similarity for pose tracking
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
from pose_utils import plot_keypoint, PreProcess, plot_keypoint_track, bipartite_matching_greedy, compute_pairwise_oks, boxes_similarity

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
from lib.detector.yolo.human_detector import human_bbox_get as yolo_det

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml',
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

    parser.add_argument("-i", "--video_input", help="input video file name", default="/home/xyliu/Videos/sports/track_test.mp4")
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
    state_dict = torch.load(model_file_name)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k # remove module.
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
human_model = yolo_model()

def main():
    previous_ids = 0
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(args.video_output,fourcc, input_fps, (input_image.shape[1],input_image.shape[0]))

    #### load pose-hrnet MODEL
    pose_model = model_load(cfg)
    pose_model.cuda()

    item = 0
    prev_max_id = 0
    for i in tqdm(range(video_length-1)):
        x0 = ckpt_time()
        ret_val, input_image = cam.read()
        item = 0
        try:
            bboxs, scores = yolo_det(input_image, human_model, 1, 0.9)
            inputs, origin_img, center, scale = PreProcess(input_image, bboxs, scores, cfg)
        except Exception as e:
            print(e)
            out.write(input_image)
            cv2.namedWindow("enhanced",0);
            cv2.resizeWindow("enhanced", 960, 480);
            cv2.imshow('enhanced', input_image)
            cv2.waitKey(2)
            continue

        try:
            with torch.no_grad():
                # compute output heatmap
                inputs = inputs[:,[2,1,0]]
                output = pose_model(inputs.cuda())
                # compute coordinate
                preds, maxvals = get_final_preds(
                    cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))
        except Exception as e:
            print(e)
            continue

        kps_b = np.concatenate((preds, maxvals), 2)
        box_b = bboxs[:preds.shape[0]]

        if previous_ids == 0:
            previous_ids = [j for j in range(len(preds))]

        if i>0:
            pose_similarity_matrix = compute_pairwise_oks(kps_a, box_a, kps_b)
            box_similarity_matrix = boxes_similarity(box_a, box_b)
            # pose similarity ratio
            ratio = 0.8
            similarity_matrix = pose_similarity_matrix*ratio + box_similarity_matrix*(1-ratio)
            # prev_filter_ids: 经过筛选之后的上一帧ids序列
            prev_filter_ids, cur_ids = bipartite_matching_greedy(similarity_matrix)

            print('previous frame boxes: ',previous_ids)
            print(prev_filter_ids, cur_ids)

            cur_len = len(box_b)
            cur_maps = -np.ones(shape=(cur_len,))

            for pos, num in enumerate(cur_ids):
                cur_maps[num] = previous_ids[prev_filter_ids[pos]]

            prev_max_id = max(max(previous_ids), prev_max_id)

            for j in range(cur_len):
                if cur_maps[j] == -1.:
                    prev_max_id += 1
                    cur_maps[j] = prev_max_id

            # 作为下一次循环的上一帧ids序列
            previous_ids = cur_maps.astype(np.uint8).tolist()
            print('after map: ', previous_ids)


        # 作为下一次循环的上一帧
        kps_a = kps_b.copy()
        box_a = box_b.copy()

        if i>0:
            image = plot_keypoint_track(origin_img, preds, maxvals, box_a, previous_ids, 0.1)
            out.write(image)
        if args.display and i>0:
            winname = 'image'
            cv2.namedWindow(winname)        # Create a named window
            cv2.moveWindow(winname, 1000,850)  # Move it to (40,30)
            cv2.imshow(winname, image)
            cv2.waitKey(100)

if __name__ == '__main__':
    main()
