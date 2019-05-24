import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
from mmdet.apis import re_result

#  cfg = mmcv.Config.fromfile('configs/cascade_rcnn_r50_fpn_1x.py')
cfg = mmcv.Config.fromfile('configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
#  _ = load_checkpoint(model, '/ssd/xyliu/models/mmdetection/cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth')
_ = load_checkpoint(model, '/ssd/xyliu/models/mmdetection/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth')

# test a single image
img = mmcv.imread('/home/xyliu/Pictures/pose/soccer.png')
import ipdb;ipdb.set_trace()
result = inference_detector(model, img, cfg)
show_result(img, result, wait_time=10000)

# test a list of images
imgs = ['demo/test2.jpg', 'demo/test1.jpg']
for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
    print(i, imgs[i])
    bboxes, scores = re_result(result)

