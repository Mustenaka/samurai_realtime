# 自己编写的模块介绍

## 调用层
根目录下：

| 代码名称      | 说明                         | 技术栈                    |
| ------------- | ---------------------------- | ------------------------- |
| cam_point.py  | 摄像头点击标定 测试          | opencv                    |
| cam_save.py   | 摄像头数据流保存             | opencv                    |
| test_cam.py   | 摄像头跟踪标定               | sam2  \| camera_predictor |
| test_img.py   | 照片测试标定                 | sam2 \| pillow            |
| test_video.py | 视频跟踪标定                 | sam2 \| video_predcitor   |
| torch_test.py | gpu测试                      |                           |
| video_seg.py  | 视频分割，test_video前置工作 | opencv                    |

改写sam2底层，路径为./sam2/：

| 代码名称                 | 说明                                                         |
| ------------------------ | ------------------------------------------------------------ |
| build_sam.py             | 创建sam2构造器，增加了camera支持，取消了huggingface网络侧下载（有墙不好下载，直接走BT更好） |
| sam2_camera_predictor.py | realtime配置设置的参数修改                                   |


