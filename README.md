## 改进HRnet视频处理方面
```
python pose_estimation/video.py
-o output_name
-i /path/to/video -o output_name
--display # 显示出来
--camera # 通过网络摄像头作为视频的输入

```

```
# 通过flow net2来平滑视频
python pose_estimation/smooth.py

# 通过SGfilter, 最小二乘法和低次多项式平滑视频  
python pose_estimation/SGfilter.py

```  

[todo]    
 > 使用flownet2来实现视频姿态track

 > 添加R-FCN、SSD  

 > do RP and speed description



---


`original code`   
clone from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch 
