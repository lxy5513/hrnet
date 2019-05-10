'''
使用yolov3作为pose net模型的前处理
使用flownet2产生shift-kpts, shift-boxs,来平滑关节点.
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
from utilitys import plot_keypoint, PreProcess, plot_keypoint_track
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
from lib.detector.yolo.human_detector import main as yolo_det
from flow_utils import *
from track_u import bipartite_matching_greedy, compute_pairwise_oks, boxes_similarity


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

    parser.add_argument("-i", "--video_input", help="input video file name", default="/home/xyliu/Videos/sports/2dance.mp4")
    parser.add_argument("-o", "--video_output", help="output video file name", default="output/output.mp4")

    parser.add_argument('--camera', action='store_true')
    parser.add_argument('--display', action='store_true')

    args = parser.parse_args()
    return args

def match_ids(previous_ids, flow_filter_ids, cur_ids, cur_len):
    '''
    先添加匹配的， 再添加previous_ids中未匹配的， 再添加cur_boxes中剩余的
    '''
    global prev_max_id
    cur_maps = -np.ones(shape=(cur_len,))

    for pos, num in enumerate(cur_ids):
        cur_maps[num] = previous_ids[flow_filter_ids[pos]]

    prev_max_id = max(max(previous_ids), prev_max_id)


    for i in range(cur_len):
        if cur_maps[i] == -1.:
            prev_max_id += 1
            cur_maps[i] = prev_max_id

    nms_ids = cur_maps.astype(np.uint8).tolist()
    return nms_ids

def boxes_nms(flow_boxes, cur_boxes, previous_ids):
    '''
    flow_boxes: (N, 4)
    cur_boxes: (M, 4)
    flow_scores: (N)
    cur_scores: (M)

    返回筛选之后的boxes和ids
    '''
    boxes_similarity_matrix = boxes_similarity(flow_boxes, cur_boxes)
    flow_filter_ids, cur_ids = bipartite_matching_greedy(boxes_similarity_matrix)
    tmp1_boxes = flow_boxes[flow_filter_ids,]
    tmp2_boxes = cur_boxes[cur_ids,]
    cur_len = len(flow_boxes) + len(cur_boxes) - len(cur_ids)
    nms_ids = match_ids(previous_ids, flow_filter_ids, cur_ids, cur_len)

    if len(np.setdiff1d(flow_boxes, tmp1_boxes)) != 0:
        tmp1_boxes = np.concatenate((tmp1_boxes , np.setdiff1d(flow_boxes, tmp1_boxes).reshape(-1, 4)), 0)

    if len(np.setdiff1d(cur_boxes, tmp2_boxes)) != 0:
        tmp1_boxes = np.concatenate((tmp1_boxes , np.setdiff1d(cur_boxes, tmp2_boxes).reshape(-1, 4)), 0)

    nms_boxes = tmp1_boxes

    return nms_boxes, nms_ids


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
sys.path.remove('/home/xyliu/2D_pose/deep-high-resolution-net.pytorch/flow_net')
human_model = yolo_model()

prev_max_id = 0
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
    # 保持长宽都是64的倍数
    resize_W = int(input_image.shape[1] / 64) * 64
    resize_H = int((input_image.shape[0] / input_image.shape[1] * resize_W) / 64 ) * 64
    print(resize_W, resize_H)
    input_image = cv2.resize(input_image, (resize_W, resize_H))
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(args.video_output,fourcc, input_fps, (input_image.shape[1],input_image.shape[0]))

    #### load optical flow model
    flow_model = load_model()

    #### load pose-hrnet MODEL
    pose_model = model_load(cfg)
    pose_model.cuda()

    flow_boxs = 0
    flow_kpts = 0

    previous_ids = 0
    pdb()
    for i in tqdm(range(video_length-1)):
        ret_val, input_image = cam.read()
        input_image = cv2.resize(input_image, (resize_W, resize_H))

        if i > 0:
            try:
                flow_result = flow_net(pre_image, input_image, flow_model)
                flow_boxs, flow_kpts = flow_propagation(pre_keypoints, flow_result)
                flow_kpts = np.concatenate((flow_kpts, flow_pose_scores), -1)
            except Exception as e:
                print(e)
                continue

        pre_image = input_image

        try:
            # boxes_threthold is 0.6
            bboxs, scores = yolo_det(input_image, human_model) # bbox is coordinate location

            # 第一帧
            if i == 0:
                inputs, origin_img, center, scale = PreProcess(input_image, bboxs, scores, cfg)
                # 初始IDs, 和 socres map
                previous_ids = [i for i in range(len(bboxs))]
                #  id_scores_map = {}
                #  for i in range(len(bboxs)): id_scores_map.update({previous_ids[i]: scores[i]})
            else:
                # 本帧、上一帧 边框置信度NMS
                #  new_boxs, new_ids = boxes_nms(flow_boxs, bboxs, previous_ids)
                inputs, origin_img, center, scale = PreProcess(input_image, bboxs, scores, cfg)

        except Exception as e:
            print(e)
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
            keypoints = np.concatenate((preds, maxvals), 2)

        # 当前帧边框置信度, 作为下一帧流边框的置信度
        #  flow_bbox_scores = scores.copy()

        #  if i != 1:
            #  preds = (preds + flow_kpts) / 2

        # shift-kpts, shift-boxes, cur_kpts ------> TRACK

        if i>0:
            kps_b = keypoints.copy()
            box_b = bboxs[:preds.shape[0]]
            kps_a = flow_kpts  # (N, 17, 3)
            box_a = flow_boxs

            pose_similarity_matrix = compute_pairwise_oks(kps_a, box_a, kps_b)
            box_similarity_matrix = boxs_similarity(box_a, box_b)
            ratio = 0.5
            similarity_matrix = pose_similarity_matrix*ratio + box_similarity_matrix*(1-ratio)
            prev_filter_ids, cur_ids = bipartite_matching_greedy(similarity_matrix)

            print('previous frame boxes: ',previous_ids)
            print(prev_filter_ids, cur_ids)

            cur_len = len(box_b) + len(box_a) - len(cur_ids)
            cur_maps = -np.ones(shape=(cur_len,))

            new_boxes = []
            new_kpts = []

            for pos, num in enumerate(cur_ids):
                cur_maps[pos] = previous_ids[prev_filter_ids[pos]]
                new_boxes.append(bo)

            prev_max_id = max(max(previous_ids), prev_max_id)

            for i in range(cur_len):
                if cur_maps[i] == -1.:
                    prev_max_id += 1
                    cur_maps[i] = prev_max_id

            previous_ids = cur_maps.astype(np.uint8).tolist()
            print('after map: ', previous_ids)



        # 整理好传给下一帧flownet的关键点, ids,
        if i==0:
            pre_flow_keypoints = keypoints
            pre_flow_pkt_scores = scores.copy()
        # 根据映射结果
        else:
            pre_flow_keypoints = tracked_keypoints
            pre_flow_pkt_scores = tracked_scores



        if i>1:
            image = plot_keypoint_track(origin_img, preds, maxvals, box_b, previous_ids, 0.1)


        if args.display and i>1:
            ########### 指定屏幕大小
            cv2.namedWindow("enhanced", cv2.WINDOW_GUI_NORMAL);
            cv2.resizeWindow("enhanced", 960, 480);
            cv2.imshow('enhanced', image)
            cv2.waitKey(1)

if __name__ == '__main__':
    main()
