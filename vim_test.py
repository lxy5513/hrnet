import numpy as np

import ipdb;pdb=ipdb.set_trace


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
    return S_intesection/(S1+S2)


def boxs_similarity(boxs1, boxs2):
    '''
    boxs1: (N, 4)
    boxs2: (M, 4)
    '''
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

    # print (res)
    # for i in range(res.shape[0]):
    #     for j in range(res.shape[1]):
    #         if i==j:
    #             print (res[i, j])
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

        if C[i][j] < 0.0001:
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
