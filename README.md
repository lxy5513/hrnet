## PoseTrack by hrnet   

https://github.com/lxy5513/cvToolBox

<br>

## DEMO
```
python pose_estimation/video.py
-o output_name
-i /path/to/video -o output_name
--display # 显示出来(display in screen)
--camera # 通过网络摄像头作为视频的输入(open web camera by video input)

```

<br>

## ENV CONFIGURE 
```
conda env create -f env_info_file.yml 
cd lib && make
```

<br> 

## Model Download 
pose model(pose_hrnet_w48_256x192.pth) address: https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA)   
>save in $hrnet/models/pytorch/pose_coco/pose_hrnet_w48_256x192.pth   


yolov3 model download: wget https://pjreddie.com/media/files/yolov3.weights
>save in $hrnet/lib/detector/yolo/yolov3.weights


<br>

## UPDATE 2019-05-21
add high mAP mmdetection
`python pose_estimation/demo_mmd.py`
<br>

do RP accuracy  test
`cd tools && ./eval_coco.sh`



---

## imporove from origin code

```
# 通过flow net2来平滑视频(smooth pose joints by flownet2)
python pose_estimation/smooth.py

# 通过SGfilter, 最小二乘法和低次多项式平滑视频 (smooth pose joints by SG-filter)
python pose_estimation/SGfilter.py

```

[todo]
 > 使用flownet2来实现视频姿态track(add tracking module by flownet2)

 > 添加R-FCN、SSD (add other human bounding-box detector like R-FCN SSD)




---


`original code`
clone from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
