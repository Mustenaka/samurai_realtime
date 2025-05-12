# SAMURAI + REALTIME

项目改自Facebook团队推出的sam2，合并samurai版本（基于sam2添加卡尔曼滤波器，效果更好），通过openCV支持摄像头视频流处理方案。

samurai：https://github.com/yangchris11/samurai

sam2：https://github.com/facebookresearch/sam2

sam2_realtime:

### 安装

安装环境：https://github.com/facebookresearch/sam2/blob/main/INSTALL.md

额外添加安装OpenCV

### 运行

> python run_cam.py

鼠标左键：选中新目标

效果如下，追踪物体打上白色蒙版：

![1747038594951](/assets/tracking.png)

### 其他

test_cam.py 测试摄像头追踪

test_img.py 测试图片脚本

test_video.py 测试视频追踪脚本

can_point.py 标记点处理脚本

cam_save.py 摄像头视频保存模块

### 项目缺陷

暂时无法支持多路追踪