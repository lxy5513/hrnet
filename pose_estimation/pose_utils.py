import cv2
import torch
import torchvision.transforms as transforms
import _init_paths
from utils.transforms import *
import numpy as np
import ipdb;pdb=ipdb.set_trace

joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
                [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                [5, 11], [6, 12], [11, 12],
                [11, 13], [12, 14], [13, 15], [14, 16]]

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255]]


font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
lineType               = 2
def plot_keypoint_track(image, coordinates, confidence, boxes, cur_ids, keypoint_thresh):
    # USE cv2
    joint_visible = confidence[:, :, 0] > keypoint_thresh

    for i in range(coordinates.shape[0]):
        box = boxes[i]
        pts = coordinates[i]
        for color_i, jp in zip(colors, joint_pairs):
            if joint_visible[i, jp[0]] and joint_visible[i, jp[1]]:
                pt0 = pts[jp, 0];pt1 = pts[jp, 1]
                pt0_0, pt0_1, pt1_0, pt1_1 = int(pt0[0]), int(pt0[1]), int(pt1[0]), int(pt1[1])

                cv2.line(image, (pt0_0, pt1_0), (pt0_1, pt1_1), color_i, 6)

                bpt0, bpt1 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

        cv2.rectangle(image, bpt0, bpt1, colors[i], 3)
        cv2.putText(image, str(cur_ids[i]), bpt0, fontFace=font, fontScale=fontScale, color=colors[i], thickness=5)
    return image

def plot_boxes(image, boxes, IDs, im_name=None):
    for i in range(len(boxes)):
        box = boxes[i]
        ID = IDs[i]
        bpt0, bpt1 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, bpt0, bpt1, colors[i], 3)
        cv2.putText(image, str(ID), bpt0, fontFace=font, fontScale=fontScale, color=colors[i], thickness=5)

    if not im_name:
        winname = 'image'
    else:
        winname = im_name
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, 1000,800)  # Move it to (40,30)
    cv2.imshow(winname, image)
    cv2.waitKey(950)
    if im_name:
        cv2.imwrite('test_imgs/' + im_name, image)
    #  cv2.destroyAllWindows()



def plot_keypoint(image, coordinates, confidence, keypoint_thresh):
    # USE cv2
    joint_visible = confidence[:, :, 0] > keypoint_thresh

    for i in range(coordinates.shape[0]):
        pts = coordinates[i]
        for color_i, jp in zip(colors, joint_pairs):
            if joint_visible[i, jp[0]] and joint_visible[i, jp[1]]:
                pt0 = pts[jp, 0];pt1 = pts[jp, 1]
                pt0_0, pt0_1, pt1_0, pt1_1 = int(pt0[0]), int(pt0[1]), int(pt1[0]), int(pt1[1])

                cv2.line(image, (pt0_0, pt1_0), (pt0_1, pt1_1), color_i, 6)
                #  cv2.circle(image,(pt0_0, pt0_1), 2, color_i, thickness=-1)
                #  cv2.circle(image,(pt1_0, pt1_1), 2, color_i, thickness=-1)
    return image



def upscale_bbox_fn(bbox, img, scale=1.25):
    new_bbox = []
    x0 = bbox[0]
    y0 = bbox[1]
    x1 = bbox[2]
    y1 = bbox[3]
    w = (x1 - x0) / 2
    h = (y1 - y0) / 2
    center = [x0 + w, y0 + h]
    new_x0 = max(center[0] - w * scale, 0)
    new_y0 = max(center[1] - h * scale, 0)
    new_x1 = min(center[0] + w * scale, img.shape[1])
    new_y1 = min(center[1] + h * scale, img.shape[0])
    new_bbox = [new_x0, new_y0, new_x1, new_y1]
    return new_bbox


def detector_to_simple_pose(img, class_IDs, scores, bounding_boxs, output_shape=(256, 192), scale=1.25):
    L = class_IDs.shape[1]
    thr = 0.5
    upscale_bbox = []
    for i in range(L):
        if class_IDs[0][i].asscalar() != 0:
            continue
        if scores[0][i].asscalar() < thr:
            continue
        bbox = bounding_boxs[0][i]
        upscale_bbox.append(upscale_bbox_fn(bbox.asnumpy().tolist(), img, scale=scale))
    if len(upscale_bbox) > 0:
        pose_input = crop_resize_normalize(img, upscale_bbox, output_shape)
        pose_input = pose_input.as_in_context(ctx)
    else:
        pose_input = None
    return pose_input, upscale_bbox




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

    return center, scale #(宽、高)

###### Pre-process
def PreProcess(image, bboxs, scores, cfg, thred_score=0.6):

    if type(image) == str:
        data_numpy = cv2.imread(image, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    else:
        data_numpy = image

    inputs = []
    centers = []
    scales = []

    score_num = np.sum(scores>thred_score)
    max_box = min(100, score_num)
    for bbox in bboxs[:max_box]:
        x1,y1,x2,y2 = bbox
        box = [x1, y1, x2-x1, y2-y1]

        # 截取 box from image  --> return center, scale
        c, s = _box2cs(box, data_numpy.shape[0], data_numpy.shape[1])
        centers.append(c)
        scales.append(s)
        r = 0

        trans = get_affine_transform(c, s, r, cfg.MODEL.IMAGE_SIZE)
        # 通过仿射变换截取人体图片
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
        inputs.append(input)


    inputs = torch.cat(inputs)
    return inputs, data_numpy, centers, scales







# --------------------------FOR TRACKING ------------------------------------ 

def box_area(box):
    # box = (x0, y0, x1, y1)
    x0, y0, x1, y1 = box
    # y0 > y1 or x0 > x1 会变成负数
    area = (y1-y0) * (x1-x0)

    area = max(area, 0)
    return area

def box_iou(box1, box2):
    S1 = box_area(box1)
    S2 = box_area(box2)
    x0 = max(box1[0], box2[0])
    y0 = max(box1[1], box2[1])
    x1 = min(box1[2], box2[2])
    y1 = min(box1[3], box2[3])
    box_intersection = [x0, y0, x1, y1]
    S_intesection = box_area(box_intersection)
    return S_intesection/(S1+S2 - S_intesection)


def boxes_similarity(boxs1, boxs2):
    '''
    boxs1: (N, 5)
    boxs2: (M, 5)
    '''
    boxs1, boxs2 =boxs1[...,:4], boxs2[...,:4]
    matrix = np.zeros(shape=(len(boxs1), len(boxs2)))
    for i in range(len(boxs1)):
        for j in range(len(boxs2)):
            box_iou_value = box_iou(boxs1[i], boxs2[j])
            matrix[i,j] = box_iou_value
    return matrix



sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
vars = (sigmas * 2)**2


def compute_oks_fb(src_keypoints, src_roi, dst_keypoints, dst_roi):
    """ Compute OKS for predicted keypoints wrt gt_keypoints.
    src_keypoints: 4xK
    src_roi: 4x1
    dst_keypoints: Nx4xK
    dst_roi: Nx4
    """

    sigmas = np.array([
        .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87,
        .87, .89, .89]) / 10.0
    vars = (sigmas * 2)**2

    # area
    src_area = (src_roi[2] - src_roi[0] + 1) * (src_roi[3] - src_roi[1] + 1)

    # measure the per-keypoint distance if keypoints visible
    '''
    src_keypoints confidence 的阀值 0.4
    dst_keypoints 如果为检测到对应的点 距离怎么算
    '''
    pose_threshold = 0.4
    filter_threshold = src_keypoints[2, :] > pose_threshold

    dx = dst_keypoints[:, 0, :] - src_keypoints[0, :]
    dy = dst_keypoints[:, 1, :] - src_keypoints[1, :]

    dx = dx * filter_threshold
    dy = dy * filter_threshold

    e = (dx**2 + dy**2) / vars / (src_area + np.spacing(1)) / 2
    e = np.sum(np.exp(-e), axis=1) / np.count_nonzero(filter_threshold)

    return e

def compute_pairwise_oks(kps_prev, box_prev, kps_cur):
    '''
    kps_prev: N, 17, 3(X-axis,Y-axis, score)  #上一帧关节点坐标集合
    box_prev: N, 4(x0,y0,x1,y1)
    kps_cur: M, 17, 3(X-axis,Y-axis, score)
    '''
    # to shape (N, 3, 17)
    kps_prev = kps_prev.transpose(0, 2, 1)
    kps_cur = kps_cur.transpose(0, 2, 1)
    similarity_martrix = np.zeros(shape=(len(kps_prev), len(kps_cur)))
    for i in range(len(kps_prev)):
        re = compute_oks_fb(kps_prev[i], box_prev[i], kps_cur, 0)
        similarity_martrix[i] = re
    return similarity_martrix


def compute_oks(pose_a, b_a, pose_b, b_b):

    # 假设b_a bbox 是由两点坐标组成
    a_area = (b_a[2] - b_a[0]) * (b_a[3] - b_a[1])
    # 对角坐标
    x0 = b_b[2] - b_b[0]; x1 = b_b[2] + b_b[0] * 2
    y0 = b_b[3] - b_b[1]; y1 = b_b[3] + b_b[1] * 2

    xg = pose_a[0]
    yg = pose_a[1]
    vg = pose_a[2]

    # 可见 或者分数不为0
    k1 = np.count_nonzero(vg > 0.2)

    xd = pose_b[0]
    yd = pose_b[1]
    cd = pose_b[2]

    dx = (xd - xg) * (vg > 0.2)
    dy = (yd - yg) * (vg > 0.2)

    e = (dx**2 + dy**2) / vars / (a_area+np.spacing(1)) / 2
    #  if k1 > 0:
        #  e=e[vg > 0]
    ious = np.sum(np.exp(-e)) / k1
    #  ious = np.sum(np.exp(-e)) / e.shape[0]

    return ious

def compute_pairwise_kps_oks_distance(kps_a, box_a, kps_b, box_b):
    # 相似度越大越好

    res = np.zeros((len(kps_a), len(kps_b)))
    kps_a = np.array(kps_a)
    kps_b = np.array(kps_b)
    print ('kps_ab.shape', kps_a.shape, kps_b.shape)
    # print (res.shape)
    for i in range(len(kps_a)):
        temp_list = []
        for j in range(len(kps_b)):
            pose_a = kps_a[i].transpose(1,0)
            pose_b = kps_b[j].transpose(1,0)

            # print (pose_a, pose_b)
            b_a = box_a[i]
            b_b = box_b[j]
            res[i, j] = compute_oks(pose_a, b_a, pose_b, b_b)
            temp_list.append(res[i,j])

        temp_list = np.array(temp_list)
        # print ('{} => {}'.format(i, temp_list.argmax()))

    return res



def bipartite_matching_greedy(C):
    """
    Computes the bipartite matching between the rows and columns, given the
    cost matrix, C.
    行代表上一帧的joints
    列代表当前帧的joints
    依次找到它们代价最大的值，将对应的行、列索引添加到row_ids, col_ids, 并删除该行、该列。
    """
    C = C.copy()  # to avoid affecting the original matrix
    prev_ids = []
    cur_ids = []
    row_ids = np.arange(C.shape[0])
    col_ids = np.arange(C.shape[1])

    if isinstance(C, list):
        C = np.array(C)

    # output boxes number
    boxes_num = len(col_ids)

    while C.size > 0:
        # Find the lowest cost element
        #  i, j = np.unravel_index(C.argmin(), C.shape)
        i, j = np.unravel_index(C.argmax(), C.shape)
        # Add to results and remove from the cost matrix
        row_id = row_ids[i]
        col_id = col_ids[j]

        if C[i][j] < 0.01:
            return prev_ids, cur_ids

        prev_ids.append(row_id)
        cur_ids.append(col_id)


        print('上一帧的第{}个box，匹配当前帧的{}个box, 匹配的相似度是{:0.3f}'.format(row_id, col_id, C[i][j]))


        C = np.delete(C, i, 0)
        C = np.delete(C, j, 1)
        # 为了保持原row_ids, col_ids的序列顺序
        row_ids = np.delete(row_ids, i, 0)
        col_ids = np.delete(col_ids, j, 0)

    return prev_ids, cur_ids


