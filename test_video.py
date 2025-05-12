import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageChops

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
    
    
from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
print("successful build predictor")

color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)   # 随机颜色 - 颜色一致性

def show_mask(mask, image, show=False):
    # ImageDraw
    draw = ImageDraw.Draw(image)
    # 获取 mask 的高和宽
    h, w = mask.shape[-2:]
    # 创建一个带颜色的mask图像
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # 将 mask_image 转换为 Pillow 图像
    mask_image = (mask_image * 255).astype(np.uint8)  # Pillow 图像需要 0-255 的整数值
    mask_pil_image = Image.fromarray(mask_image)
    # convert RBGA
    image = image.convert("RGBA")
    # blender
    # blend_image = Image.blend(image, mask_pil_image, 0.5) # 灰度 混合，显示效果不行
    mask_with_image_background = Image.composite(mask_pil_image, image, mask_pil_image.split()[3])  # 使用 alpha 通道进行合成
    blend_image = ImageChops.multiply(image, mask_with_image_background)
    
    if show:
        blend_image.show()
        
    return blend_image

def show_points(coords, labels, image, marker_size=200,show=False):
    # 创建一个 ImageDraw 对象
    copy_image = image.copy()
    draw = ImageDraw.Draw(copy_image)

    # 设置点的大小
    radius = int(marker_size ** 0.5)  # 根据marker_size计算圆点的半径
    
    # 通过标签分别提取正负类别的点
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    
    # 绘制正类别点（绿色星形点）
    for point in pos_points:
        x, y = int(point[0]), int(point[1])
        # 使用 Pillow 绘制绿色圆点，边缘为白色
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='green', outline='white')

    # 绘制负类别点（红色星形点）
    for point in neg_points:
        x, y = int(point[0]), int(point[1])
        # 使用 Pillow 绘制红色圆点，边缘为白色
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red', outline='white')

    # 显示图像
    if show:
        copy_image.show()
    
    return copy_image
    
# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "./notebooks/videos/webcam"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0
Image.open(os.path.join(video_dir, frame_names[frame_idx]))

inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Let's add a positive click at (x, y) = (210, 350) to get started
# points = np.array([[210, 350]], dtype=np.float32)   # boy
# points = np.array([[1310, 1512]], dtype=np.float32)   # cai
points = np.array([[300, 150]], dtype=np.float32)   # webcom

# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
img = Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))
show_points(points, labels, img, show=True)
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), img)
         
# # Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
# # sending all clicks (and their labels) to `add_new_points_or_box`
# points = np.array([[210, 350], [250, 220]], dtype=np.float32)
# # for labels, `1` means positive click and `0` means negative click
# labels = np.array([1, 1], np.int32)
# _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     points=points,
#     labels=labels,
# )

# # show the results on the current (interacted) frame
# img = Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))
# show_points(points, labels, img)
# show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), img, True)

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
frames = []

for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

max_frame = 200

# render the segmentation results every few frames
for out_frame_idx in range(0, len(frame_names)):
    rend_image = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        msk = show_mask(out_mask, rend_image)
        frames.append(msk)

import cv2

# 设置视频的输出文件名和帧的尺寸（注意：视频帧的尺寸应与图像尺寸一致）
video_filename = "output_video2.mp4"
frame_width, frame_height = frames[0].size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码方式
fps = 20  # 设置帧率

# 创建 VideoWriter 对象
out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

# 将每一帧图像写入视频
for frame in frames:
    frame_cv2 = np.array(frame)  # 将 PIL 图像转换为 NumPy 数组
    frame_cv2 = cv2.cvtColor(frame_cv2, cv2.COLOR_RGBA2BGR)  # 转换为 BGR 色彩空间
    out.write(frame_cv2)

# 释放 VideoWriter 对象
out.release()

print(f"视频已保存为 {video_filename}")