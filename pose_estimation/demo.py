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
from utilitys import plot_keypoint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
import config
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

from utils.transforms import *
from lib.core.inference import get_final_preds
import cv2
#  import dataset
import models



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

    args = parser.parse_args()
    return args


def _box2cs(box, image_width, image_height):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, image_width, image_height)

def _xywh2cs(x, y, w, h, image_width, image_height):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    aspect_ratio = image_width * 1.0 / image_height
    pixel_std = 200

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    args.flip_test = True
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file

###### Pre-process
def PreProcess(image_path, bbox):
    try:
        x1,y1,x2,y2 = bbox
    except:
        x1,y1,x2,y2 = eval(bbox)
    box = [x1, y1, x2-x1, y2-y1]

    data_numpy = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    # data_numpy = cv2.resize(data_numpy, (512, 512))
    if data_numpy is None:
        logger.error('=> fail to read {}'.format(image_file))
        raise ValueError('Fail to read {}'.format(image_file))

    # 截取 box fron image  --> return center, scale
    c, s = _box2cs(box, data_numpy.shape[0], data_numpy.shape[1])
    r = 0

    trans = get_affine_transform(c, s, r, cfg.MODEL.IMAGE_SIZE)
    input = cv2.warpAffine(
        data_numpy,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    input = transform(input).unsqueeze(0)
    return input, data_numpy, c, s



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
        print(name,'\t')
        new_state_dict[name] = v
    #  import ipdb;ipdb.set_trace()
    model.load_state_dict(new_state_dict)
    return model

def get_data(path, item=0):
    import json
    data = json.load(open(path, 'rt'))
    image_path = data[item]['name']
    bbox = data[item]['bbox']
    return image_path, bbox


def main():
    args = parse_args()
    update_config(cfg, args)

    image_path = '/home/xyliu/Pictures/pose/one_person.jpg'

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


    ########## 加载human detecotor model
    from lib.detector.yolo.human_detector import load_model as yolo_model
    human_model = yolo_model()

    from lib.detector.yolo.human_detector import main as yolo_det
    bbox = yolo_det(image_path, human_model)

    # bbox is coordinate location
    inputs, origin_img, center, scale = PreProcess(image_path, bbox)

    # load MODEL
    model = model_load(cfg)

    model.eval()

    with torch.no_grad():
        # compute output heatmap
        output = model(inputs)
        # compute coordinate
        preds, maxvals = get_final_preds(
            cfg, output.clone().cpu().numpy(), np.asarray([center]), np.asarray([scale]))

    image = plot_keypoint(origin_img, preds, maxvals, 0.1)
    cv2.imwrite('output.jpg', image)

if __name__ == '__main__':
    main()
