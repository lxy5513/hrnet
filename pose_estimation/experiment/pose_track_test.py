'''
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
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
from flow_utils import flow_propagation, load_model, flow_net
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

def filter_small_boxes(boxes, min_size):
    assert boxes.shape[1] == 4, 'Func doesnot support tubes yet'
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = np.where((w >= min_size) & (h > min_size))[0]
    return keep


def boxes_nms_test(flow_boxes, cur_boxes, image_resolution):
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
    if len(np.setdiff1d(cur_boxes, tmp2_boxes)) != 0:
        a = cur_boxes
        b = tmp2_boxes
        l = []#用l存储a中b的每一行的索引位置
        for i in range(len(b)):
            for j in range(len(a)):

                import operator as op
                if(op.eq(list(a[j]), list(b[i]))):#op.eq比较两个list，相同返回Ture
                    l.append(j)

        #delete函数删除a中对应行
        re = np.delete(a, l, 0)
        tmp2_boxes = np.concatenate((tmp2_boxes ,re), 0)

    # 如果flow-boxes有未检测到人体 P1. 确实是消失了(丢弃) P2. YOLO检测器没检测到（放入boxes)
    if len(np.setdiff1d(flow_boxes, tmp1_boxes)) != 0:
        a = flow_boxes
        b = tmp1_boxes
        l = []#用l存储a中b的每一行的索引位置
        for i in range(len(b)):
            for j in range(len(a)):

                import operator as op
                if(op.eq(list(a[j]), list(b[i]))):#op.eq比较两个list，相同返回Ture
                    l.append(j)

        #delete函数删除a中对应行
        remained_boxes = np.delete(a, l, 0)
        for item_box in remained_boxes:
            # P1
            box_area_value = box_area(item_box[1:])
            if box_area_value < 1/10 * image_resolution[0] * image_resolution[1]:
                print('flow box dispear...')

            # P2
            else:
                tmp2_boxes = np.concatenate((tmp2_boxes , np.expand_dims(item_box, 0)), 0)

    nms_boxes = tmp2_boxes
    return nms_boxes


###### 加载human detecotor model
from lib.detector.yolo.human_detector import load_model as yolo_model
sys.path.remove('/home/xyliu/2D_pose/deep-high-resolution-net.pytorch/flow_net')
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
    resize_W = int(input_image.shape[1] / 64) * 64
    resize_H = int((input_image.shape[0] / input_image.shape[1] * resize_W) / 64 ) * 64
    input_image = cv2.resize(input_image, (resize_W, resize_H))
    image_resolution = (resize_W, resize_H)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(args.video_output,fourcc, input_fps, (input_image.shape[1],input_image.shape[0]))


    #### load pose-hrnet MODEL
    pose_model = model_load(cfg)
    pose_model.cuda()


    #### load optical flow model
    flow_model = load_model()

    item = 0
    prev_max_id = 0
    for i in tqdm(range(video_length-1)):

        x0 = ckpt_time()
        ret_val, input_image = cam.read()
        input_image = cv2.resize(input_image, (resize_W, resize_H))

        item = 0
        try:
            bboxs, scores = yolo_det(input_image, human_model)
            bboxs = np.concatenate((bboxs, scores.transpose(1,0)), -1)


            # 加入flownet 模块
            if i>0:
                flow_result = flow_net(pre_image, input_image, flow_model)
                flow_boxes, flow_kpts = flow_propagation(prev_kpts, flow_result)
                flow_boxes = np.concatenate((flow_boxes, np.expand_dims(prev_boxes[...,4], -1)), -1)
                flow_kpts = np.concatenate((flow_kpts,prev_kpts_scores), -1)
                detected_boxes = bboxs.copy()
                #  plot_boxes(input_image.copy(), flow_boxes, [i for i in range(len(flow_boxes))], '{}_flow.png'.format(1000+i))
                #  plot_boxes(input_image.copy(), detected_boxes, [i for i in range(len(detected_boxes))], '{}_detected.png'.format(1000+i))
                bboxs = boxes_nms_test(flow_boxes, bboxs, image_resolution)
                #  plot_boxes(input_image.copy(), bboxs, [i for i in range(len(bboxs))], 'nms_{}.png'.format(100+i))

            inputs, origin_img, center, scale = PreProcess(input_image, bboxs[..., :4], bboxs[...,4], cfg)


        except Exception as e:
            print(e)
            pdb()
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
        prev_kpts = kps_b.copy()
        box_b = bboxs[:preds.shape[0]]

        if previous_ids == 0:
            previous_ids = [j for j in range(len(preds))]

        if i>0:
            # kps_a是前一帧的 kps_b是当前hrnet检测出来的
            kps_a = flow_kpts
            box_a = flow_boxes
            pose_similarity_matrix = compute_pairwise_oks(kps_a, box_a, kps_b)
            box_similarity_matrix = boxes_similarity(box_a, box_b)
            # pose similarity ratio
            ratio = 0.5
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
        prev_kpts = kps_b
        prev_kpts_scores = maxvals
        pre_image = input_image
        prev_boxes = bboxs

        if i>0:
            image = plot_keypoint_track(origin_img, preds, maxvals, box_a, previous_ids, 0.1)
            out.write(image)
        if args.display and i>0:
            ########### 指定屏幕大小
            winname = 'image'
            cv2.namedWindow(winname)        # Create a named window
            cv2.moveWindow(winname, 1000,850)  # Move it to (40,30)
            cv2.imshow(winname, image)
            cv2.waitKey(100)

if __name__ == '__main__':
    main()
