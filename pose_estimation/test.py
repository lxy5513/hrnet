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
from utilitys import plot_keypoint, PreProcess, plot_keypoint_track, plot_boxes
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
from track_u import box_area


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

def match_ids(previous_ids, map_ids, cur_ids, cur_len):
    '''
    先添加匹配的， 再添加previous_ids中未匹配的， 再添加cur_boxes中剩余的
    '''
    global prev_max_id
    cur_maps = -np.ones(shape=(cur_len,))

    for pos,num in enumerate(cur_ids):
        cur_maps[num] = previous_ids[map_ids[pos]]

    # previous中不含有cur_map得数
    tmp_nums = np.setdiff1d(previous_ids, cur_maps).tolist()
    for i in tmp_nums:
        for j in range(len(cur_maps)):
            if cur_maps[j] == -1.:
                cur_maps[j] = i
                break

    max_id = max(max(previous_ids), max_id)

    for i in range(cur_len):
        if cur_maps[i] == -1.:
            prev_max_id += 1
            cur_maps[i] = prev_max_id

    nms_ids = cur_maps.astype(np.uint8).tolist()
    return nms_ids



def pose_match_ids(previous_ids, flow_filter_ids, cur_ids, cur_len):
    '''
    先添加匹配的， 再添加previous_ids中未匹配的
    '''
    global max_id
    cur_maps = -np.ones(shape=(cur_len,))

    for pos,num in enumerate(cur_ids):
        cur_maps[num] = previous_ids[flow_filter_ids[pos]]

    max_id = max(max(previous_ids), max_id)

    for i in range(cur_len):
        if cur_maps[i] == -1.:
            max_id += 1
            cur_maps[i] = max_id

    nms_ids = cur_maps.astype(np.uint8).tolist()
    return nms_ids




def boxes_nms_test(flow_boxes, cur_boxes, previous_ids, image_resolution):
    '''
    flow_boxes: (N, 5) -> 5: x0,y0,x1,y1,score
    cur_boxes: (M, 5)
    previous_ids: flow_boxes的排列顺序
    image_resolution (W, H)
    返回筛选之后的boxes和相应的ids
    '''
    global max_id

    boxes_similarity_matrix = boxes_similarity(flow_boxes[...,:4], cur_boxes[...,:4])
    flow_filter_ids, cur_ids = bipartite_matching_greedy(boxes_similarity_matrix)

    # 匹配之后剩下的
    tmp1_boxes = flow_boxes[flow_filter_ids,]
    tmp2_boxes = cur_boxes[cur_ids,]

    # 如果多出边框的话，表示新出现人体, 添加进boxes
    if len(np.setdiff1d(flow_boxes, tmp1_boxes)) != 0:
        tmp1_boxes = np.concatenate((tmp1_boxes , np.setdiff1d(flow_boxes, tmp1_boxes).reshape(-1, 5)), 0)

    # 如果flow-boxes有未检测到人体 P1. 确实是消失了(丢弃) P2. YOLO检测器没检测到（放入boxes)
    if len(np.setdiff1d(cur_boxes, tmp2_boxes)) != 0:
        remained_boxes = np.setdiff1d(cur_boxes, tmp2_boxes).reshape(-1, 5)
        for item_box in remained_boxes:
            # P1
            box_area_value = box_area(item_box)
            if box_area_value < 1/10 * image_resolution[0] * image_resolution[1]:
                print('flow box dispear...')
                pdb()

            # P2
            else:
                tmp1_boxes = np.concatenate((tmp1_boxes , np.expand_dims(item_box, 0)), 0)

    nms_boxes = tmp1_boxes
    cur_len = len(nms_boxes)
    cur_maps = -np.ones(shape=(cur_len,))

    for pos, num in enumerate(cur_ids):
        cur_maps[pos] = previous_ids[flow_filter_ids[pos]]

    for i in range(cur_len):
        if cur_maps[i] == -1.:
            max_id += 1
            cur_maps[i] = max_id

    nms_ids = cur_maps.astype(np.uint8).tolist()
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

# 目前id最大值
max_id = 0
def main():
    global max_id
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
    # 保持长宽都是64的倍数，用于flownet2
    resize_W = int(input_image.shape[1] / 64) * 64
    resize_H = int((input_image.shape[0] / input_image.shape[1] * resize_W) / 64 ) * 64
    image_resolution = (resize_W, resize_H)
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

    for i in tqdm(range(video_length-1)):
        ret_val, input_image = cam.read()
        input_image = cv2.resize(input_image, (resize_W, resize_H))

        try:
            if i > 0:
                pdb()
                flow_result = flow_net(pre_image, input_image, flow_model)
                flow_boxes, flow_kpts = flow_propagation(prev_kpts, flow_result)
                flow_boxes = np.concatenate((flow_boxes, np.expand_dims(prev_boxes[...,4], -1)), -1) # flow_boxes + previous boxes scores
                flow_kpts = np.concatenate((flow_kpts,prev_kpts_scores), -1)

            # boxes_threthold is 0.9
            detected_boxes, detected_scores = yolo_det(input_image, human_model) # bbox is coordinate location
            detected_scores = np.expand_dims(detected_scores.flatten(), -1)
            detected_boxes = np.concatenate((detected_boxes, detected_scores), -1) # (N, 17, 3)

            if i == 0:
                inputs, origin_img, center, scale = PreProcess(input_image, detected_boxes[...,:4],  detected_boxes[...,4], cfg)
                #  ploted_image = plot_boxes(input_image, detected_boxes, [i for i in range(len(detected_boxes))])
                #  cv2.imshow('image', ploted_image)
                #  cv2.waitKey(100)
            else:
                # 最难！ 会重新给pose net一个输入顺序, 并且给出相应的ids
                print('before mapping: ', previous_ids)
                new_boxes, new_ids = boxes_nms_test(flow_boxes, detected_boxes, previous_ids, image_resolution)
                print('after mapping: ', new_ids)
                print(flow_boxes[:, 1], detected_boxes[:, 1])
                #  ploted_image = plot_boxes(input_image, new_boxes, new_ids)
                #  cv2.imshow('image', ploted_image)
                #  cv2.waitKey(100)
                inputs, origin_img, center, scale = PreProcess(input_image, new_boxes[..., :4], new_boxes[...,4], cfg)

        except Exception as e:
            print(e)
            out.write(input_image)
            cv2.namedWindow("enhanced",0);
            cv2.resizeWindow("enhanced", 960, 480);
            cv2.imshow('enhanced', input_image)
            cv2.waitKey(2)
            continue

        # 姿态检测
        with torch.no_grad():
            # compute output heatmap
            inputs = inputs[:,[2,1,0]]
            output = pose_model(inputs.cuda())
            # compute coordinate
            detected_kpts, detected_kpts_scores = get_final_preds(
                cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))
            detected_kpts = np.concatenate((detected_kpts, detected_kpts_scores), 2)


        # TRACK Assign IDs. flow_boxes; detected_boxes, new_ids
        if i>0:
            pose_similarity_matrix = compute_pairwise_oks(flow_kpts, flow_boxes[...,:4], detected_kpts)
            box_similarity_matrix = boxes_similarity(flow_boxes[...,:4], detected_boxes[...,:4])
            ratio = 0.5
            similarity_matrix = pose_similarity_matrix*ratio + box_similarity_matrix*(1-ratio)
            prev_filter_ids, cur_ids = bipartite_matching_greedy(similarity_matrix)

            print('previous frame boxes: ', prev_pose_ids)
            cur_len = len(detected_kpts)
            new_pose_ids = pose_match_ids(prev_pose_ids, prev_filter_ids, cur_ids, cur_len)

            #  detected_kpts = detected_kpts[ [i-1 for i in new_ids],:]
            #  detected_kpts_scores = detected_kpts_scores[[i-1 for i in new_ids],:]
            print(prev_filter_ids, cur_ids)
            print('after map: ', new_pose_ids)

        # 为下一帧处理做准备
        pre_image = input_image.copy()
        prev_kpts = detected_kpts
        prev_kpts_scores = detected_kpts_scores
        if i == 0:
            prev_boxes = detected_boxes
            previous_ids = [j for j in range(len(detected_boxes))]
            prev_pose_ids = previous_ids

        else:
            previous_ids = new_ids
            prev_boxes = new_boxes
            prev_pose_ids = new_pose_ids
        if i>1:
            image = plot_keypoint_track(origin_img, detected_kpts, detected_kpts_scores, new_boxes[...,:4], new_pose_ids, 0.1)
        else:
            image = plot_keypoint_track(origin_img, detected_kpts, detected_kpts_scores, detected_boxes[...,:4], previous_ids, 0.1)


        if args.display :
            ########### 指定屏幕大小
            cv2.namedWindow("enhanced", cv2.WINDOW_GUI_NORMAL);
            cv2.resizeWindow("enhanced", 960, 480);
            cv2.imshow('enhanced', image)
            cv2.waitKey(1)

if __name__ == '__main__':
    main()
