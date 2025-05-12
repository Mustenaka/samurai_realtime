import cv2

# 定义鼠标点击回调函数
def click_event(event, x, y, flags, param):
    # 当鼠标左键单击时
    if event == cv2.EVENT_LBUTTONDOWN:
        # 输出点击位置的坐标
        print(f"Clicked at: ({x}, {y})")

# 打开视频流（你可以替换为0来使用默认摄像头，或者替换为视频文件路径）
cap = cv2.VideoCapture(0)

# 检查视频流是否打开成功
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    # 从视频流中读取每一帧
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # 显示视频帧
    cv2.imshow("Video Stream", frame)

    # 绑定鼠标回调函数
    cv2.setMouseCallback("Video Stream", click_event)

    # 持续播放视频流直到用户按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频流并关闭所有窗口
cap.release()
cv2.destroyAllWindows()