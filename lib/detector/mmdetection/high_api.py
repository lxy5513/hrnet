import sys 
import os
main_path = os.path.dirname(os.path.abspath(__file__)) + '/mmd'
sys.path.insert(0, main_path)
#  import ipdb;ipdb.set_trace()
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
from mmdet.apis import re_result
sys.path.remove(main_path)


# 模型配置及其参数
model_cfgs =[
        (main_path + '/configs/cascade_rcnn_r50_fpn_1x.py', '/ssd/xyliu/models/mmdetection/cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth'),  #7 it/s
        (main_path + '/configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py',  '/ssd/xyliu/models/mmdetection/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth'), # 2 it/s
        (main_path + '/configs/htc/htc_r50_fpn_20e.py', '/ssd/xyliu/models/mmdetection/htc_r50_fpn_20e_20190408-c03b7015.pth') #2.8 it/s box AP 43.9

        ]

cfg = mmcv.Config.fromfile(model_cfgs[0][0])
cfg.model.pretrained = None

def load_model():
    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model, model_cfgs[0][1]) # 7 it/s
    return model

def human_boxes_get(model, img, score_thr=0.5):
    if isinstance(img, str):
        img = mmcv.imread(img)
    result = inference_detector(model, img, cfg, device='cuda:0')
    bboxes, scores = re_result(result, score_thr=score_thr)
    return bboxes, scores


if __name__ == '__main__':
    imgs = [main_path + '/demo/test2.jpg', main_path + '/demo/test1.jpg']
    model = load_model()
    for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
        print(i, imgs[i])
        bboxes, scores = re_result(result)

