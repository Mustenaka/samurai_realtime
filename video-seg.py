import cv2
import os

def extract_frames(video_path, output_folder):
    # 创建输出文件夹（如果文件夹不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的帧率
    frame_count = 0

    # 读取视频帧
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 构建文件名，使用五位数序号命名
        filename = f"{frame_count:05d}.jpg"
        file_path = os.path.join(output_folder, filename)
        
        # 保存当前帧为图片
        cv2.imwrite(file_path, frame)
        
        # 增加帧计数
        frame_count += 1

    # 释放视频对象
    cap.release()
    print(f"Frames extracted and saved to {output_folder}")

# 使用示例
# video_path = './notebooks/videos/cai.mp4'
# output_folder = './notebooks/videos/cai'

video_path = './assets/dogs.mp4'
output_folder = './assets/dogs'
extract_frames(video_path, output_folder)