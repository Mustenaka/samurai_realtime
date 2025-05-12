from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from PIL import Image

import numpy as np
import torch

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    image_path = "./assets/dog/00000.jpg"
    prompts = ""
    
    image = Image.open(image_path)
    np_image = np.array(image)
    
    input_point = np.array([[200,100]])
    input_label = np.array([1])
    
    predictor.set_image(np_image)
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    sorted_ind = np.argsort(scores)[::-1]
    best_mask = masks[sorted_ind[0]]
    
    print(masks)    