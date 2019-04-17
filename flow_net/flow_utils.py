from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def box_propagation(keypoints, flow):
    """Box propagation from previous frame.
    Arguments:
        keypoints (ndarray): [num_people, num_keypoints, 3] (x, y, score)
                             keypoints detection of previous frame.
        flow (ndarray): [2, H, W]
                        optical flow between previous frame and current frame.
    Returns:
        boxes (ndarray): [num_people, 4]
                         boxes propagated from previous frame.
    """
    extend_factor = 0.15
    H = flow.shape[0]
    W = flow.shape[1]
    num_kpts = keypoints.shape[1]
    flow = flow.transpose((2,1,0))  # [W, H, 2]
    pos = keypoints[:,:,:2].reshape(-1,2).T.astype(int).tolist()
    offset = flow[pos].reshape(-1,num_kpts,2)
    shift_keypoints = keypoints[:,:,:2] + offset
    mask = keypoints[:,:,2] > 0
    mask = mask[:,:,np.newaxis]
    min_ = np.min(shift_keypoints+(1-mask)*max(H,W), axis=1)  # [N,2]
    max_ = np.max(shift_keypoints*mask, axis=1)  # [N,2]
    extend = (max_ - min_) * extend_factor / 2
    up_left = np.fmax(min_-extend, 0)
    bottom_right = np.fmin(max_+extend, np.array([W-1,H-1]))
    boxes = np.concatenate((up_left, bottom_right), axis=1)

    return boxes
