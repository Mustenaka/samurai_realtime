import cv2
import os

def capture_camera_frames(output_folder):
    # 创建输出文件夹（如果文件夹不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开摄像头（0表示默认摄像头）
    cap = cv2.VideoCapture(0)
    judge = cap.isOpened()
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    fps = cap.get(5)
    video_out = cv2.VideoWriter('webcam.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, frame_size, isColor=True)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_count = 0

    while True:
        # 读取摄像头帧
        ret, frame = cap.read()

        if not ret:
            break

        # 显示当前帧（可以选择性地显示摄像头画面）
        cv2.imshow("Webcam", frame)

        # 构建文件名，使用五位数序号命名
        filename = f"{frame_count:05d}.jpg"
        file_path = os.path.join(output_folder, filename)

        # 保存当前帧为图片
        cv2.imwrite(file_path, frame)

        # 增加帧计数
        frame_count += 1

        # 保存视频
        video_out.write(frame)

        # 按键按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()
    print(f"Frames saved to {output_folder}")

# 使用示例
output_folder = './notebooks/videos/webcam'
capture_camera_frames(output_folder)