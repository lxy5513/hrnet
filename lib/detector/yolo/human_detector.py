from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import os
import sys

# scipt dirctory
yolo_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(yolo_dir)

from util import *
import argparse
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import pickle as pkl
import itertools
import ipdb;pdb=ipdb.set_trace
num_classes = 80


class args():
    bs = 1
    nms_thresh = 0.4
    cfgfile = 'cfg/yolov3.cfg'
    weightsfile = 'yolov3.weights'
    reso = '416'
    scales='1,2,3'
    confidence = 0.5

def filter_small(box, im_shape, ratio):
    '''
    box: (x0, y0, x1, y1)
    im_shape: (H, W, channel)
    ratio: 过滤的比例 0.1 就是小于图片大小十分之一的都过滤掉
    '''
    H, W = im_shape[:2]
    x0, y0, x1, y1 = box
    box_area = (x1-x0) * (y1-y0)
    print('human box volumn is {:0.3f} of the image'.format(box_area/H/W))
    if box_area < ratio * W * H:
        # filte the box
        return True
    else:
        return False

def ckpt_time(t0=None, display=1):
    if not t0:
        return time.time()
    else:
        ckpt = time.time() - t0
        if display:
            print('consume time {:3f}s'.format(ckpt))
        return ckpt, time.time()


class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers= num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5,5) for x in range(num_layers)])
        self.output = nn.Linear(5,2)

    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)



def load_model():
    CUDA = torch.cuda.is_available()
    classes = load_classes('data/coco.names')

    #Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    #Set the model in evaluation mode
    model.eval()
    return model

def human_bbox_get(images, model=None, is_filter=None, confidence=None):
    '''
    images: array or name
    is_filter: filter small box or not
    confidence: bbox threshold
    '''

    scales = args.scales
    batch_size = int(args.bs)
    if not confidence:
        confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()
    classes = load_classes('data/coco.names')

    if not model:
        #Set up the neural network
        print("Loading YOLO network.....")
        model = load_model()
        print("Network successfully loaded")

    read_dir = time.time()
    #Detection phase
    if type(images) == str:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    else:
        imlist = []
        imlist.append(images)


    load_batch = time.time()

    inp_dim = int(model.net_info["height"])
    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)


    if CUDA:
        im_dim_list = im_dim_list.cuda()

    leftover = 0

    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                            len(im_batches))]))  for i in range(num_batches)]


    i = 0
    objs = {}
    for batch in im_batches:
        #load the image
        start = time.time()
        if CUDA:
            batch = batch.cuda()

        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes)
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)

        prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)

        if type(prediction) == int:
            i += 1
            continue

        prediction[:,0] += i*batch_size
        output = prediction


        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        i += 1

        if CUDA:
            torch.cuda.synchronize()
    try:
        output
    except NameError:
        print("No detections were made")

    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
    output[:,1:5] /= scaling_factor

    #################################
    # select human and export bbox
    #################################
    bboxs = []
    scores = []
    for i in range(len(output)):
        item = output[i]
        im_id = item[-1]
        if int(im_id) == 0:
            bbox = item[1:5].cpu().numpy()
            # conver float32 to .3f data
            bbox = [round(i, 3) for i in list(bbox)]
            score = item[5]

            # 只检测足够大的边框
            if is_filter:
                if filter_small(bbox, batch.squeeze().transpose(-1,0).shape, 0.04):
                    continue

            bboxs.append(bbox)
            scores.append(score)
    scores = np.expand_dims(np.array(scores), 0)
    bboxs = np.array(bboxs)

    return bboxs, scores
