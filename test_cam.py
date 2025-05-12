import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import time
import cv2

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
    
from sam2.build_sam import build_sam2_camera_predictor

# small model 6~9 fps
# sam2_checkpoint = "checkpoints/sam2.1_hiera_small.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

# large model only 5 fps (and have some delay)
sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint, device=device)
print("successful build predictor")

if_init = False     # 初始化确认
if_first_click = False  # 初次点击确认
prev_frame_time = 0 # 第0帧指定

cap = cv2.VideoCapture(0)

# 定义鼠标点击回调函数
def click_event(event, x, y, flags, param):
    global if_init
    frame = param
    # 当鼠标左键单击时
    if event == cv2.EVENT_LBUTTONDOWN:
        predictor.load_first_frame(frame)

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        
        # 输出点击位置的坐标
        print(f"Clicked at: ({x}, {y})")
        
        #! add points, `1` means positive click and `0` means negative click
        points = np.array([[x, y]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
        )
        
        print("add points, project begining." + str(out_obj_ids) + " mask:" + str(out_mask_logits))
        if_init = True

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    width, height = frame.shape[:2][::-1]
    if not if_init:
        pass
        # predictor.load_first_frame(frame)

        # ann_frame_idx = 0  # the frame index we interact with
        # ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        # Let's add a positive click at (x, y) = (210, 350) to get started


        ##! add points, `1` means positive click and `0` means negative click
        # points = np.array([[660, 267]], dtype=np.float32)
        # labels = np.array([1], dtype=np.int32)

        # _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
        # )

        ## ! add bbox
        # bbox = np.array([[600, 214], [765, 286]], dtype=np.float32)
        # _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
        # )

        ##! add mask
        # mask_img_path="../notebooks/masks/aquarium/aquarium_mask.png"
        # mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        # mask = mask / 255

        # _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, mask=mask
        # )

    else:
        out_obj_ids, out_mask_logits = predictor.track(frame)

        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
        # print(all_mask.shape)
        for i in range(0, len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                np.uint8
            ) * 255

            all_mask = cv2.bitwise_or(all_mask, out_mask)

        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
        frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
        
        # print("跟踪中 id:" + str(out_obj_ids) + " mask:" + str(out_mask_logits))
    
    # count FPS for display
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 计算FPS
    current_frame_time = cv2.getTickCount()
    time_diff = (current_frame_time - prev_frame_time) / cv2.getTickFrequency() 
    fps_display = 1 / time_diff if time_diff > 0 else 0
    prev_frame_time = cv2.getTickCount()
    
    # put FPS on frame
    cv2.putText(frame, f"FPS: {int(fps_display)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    cv2.imshow("frame", frame)
    # 绑定鼠标回调函数
    cv2.setMouseCallback("frame", click_event, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
# gif = imageio.mimsave("./result.gif", frame_list, "GIF", duration=0.00085)
