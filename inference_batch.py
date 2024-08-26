import argparse
import os 

import torch
from torch import Tensor

from pathlib import Path
import cv2
import numpy as np

from time import time
from PIL import Image
from torchvision.transforms import transforms

from models.model import ViTPose
from utils.visualization import draw_points_and_skeleton, joints_dict
from utils.dist_util import get_dist_info, init_dist
from utils.top_down_eval import keypoints_from_heatmaps

from tqdm import tqdm
import pickle

@torch.no_grad()
def inference(img_dir , img_size, model_cfg: dict, ckpt_path: Path, device: torch.device, save_result: bool=True):
    
    # Prepare model
    vit_pose = ViTPose(model_cfg)
    
    ckpt = torch.load(ckpt_path)
    if 'state_dict' in ckpt:
        vit_pose.load_state_dict(ckpt['state_dict'])
    else:
        vit_pose.load_state_dict(ckpt)

    vit_pose.eval() # 补充上这个 否则 predict会计算BN dropout之类的  
    vit_pose.to(device)
    
    print(f">>> Model loaded: {ckpt_path}")
    
    files = os.listdir(img_dir)
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

    # output_video = None 

    for p in tqdm(range(len(files))):
 
        img_path = os.path.join(img_dir, files[p]) 

        # Prepare input data
        img = Image.open(img_path)
        org_w, org_h = img.size
        img_tensor = transforms.Compose (
            [transforms.Resize((img_size[1], img_size[0])),
            transforms.ToTensor()]
        )(img).unsqueeze(0).to(device)


        # if output_video is None:
        #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        #     OUTPUT_VIDEO_PATH = "~/work/ViTPose_pytorch/result/visual_pose.mp4"
        #     output_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 30.0, (image_ori.shape[1], image_ori.shape[0])) 
    
    
        # Feed to model
        heatmaps = vit_pose(img_tensor).detach().cpu().numpy() # N, 17, h/4, w/4
        points, prob = keypoints_from_heatmaps( heatmaps = heatmaps, 
                                                center   = np.array([[org_w//2, org_h//2]]), 
                                                scale    = np.array([[org_w, org_h]]),
                                                unbiased = True, 
                                                use_udp  = True )
        points = np.concatenate([points[:, :, ::-1], prob], axis=2)
    

        # 序列化到文件 
        if True:
            dict_to_pkl = {"num":1 }
            dict_to_pkl['0'] = {}
            dict_to_pkl['0']['obj_conf']   = 1.0
            dict_to_pkl['0']['class_conf'] = 1.0
            dict_to_pkl['0']['box']        = [0, 0, org_w, org_h]  
            dict_to_pkl['0']['kp']         = points

            directory = os.path.dirname(img_path) 
            file_name_without_ext = os.path.splitext(os.path.basename(img_path))[0]
            directory_vitpose = f"{directory}_vitpose_pkl"  
            if not os.path.exists(directory_vitpose):  
                os.makedirs(directory_vitpose)
            with open(f"{directory_vitpose}/{file_name_without_ext}.pkl", 'wb') as f:
                pickle.dump(dict_to_pkl, f)


        # Visualization 
        if True:
            for pid, point in enumerate(points):
                img = np.array(img)[:, :, ::-1] # RGB to BGR for cv2 modules
                img = draw_points_and_skeleton(
                        img.copy(), 
                        point, 
                        joints_dict()['coco']['skeleton'], 
                        person_index            = pid,                             
                        points_color_palette    = 'gist_rainbow', 
                        skeleton_color_palette  = 'jet',
                        points_palette_samples  = 10, 
                        confidence_threshold    = 0.4)

            # if output_video is not None:
            #     output_video.write(image_ori)
            directory = os.path.dirname(img_path) # 文件所在目录(包含路径)
            file_name_without_ext = os.path.splitext(os.path.basename(img_path))[0]
            directory_pose = f"{directory}_pose" # 新的目录 
            if not os.path.exists(directory_pose): # 如果目录不存在，则创建目录
                os.makedirs(directory_pose)
            cv2.imwrite(f"{directory_pose}/{file_name_without_ext}.png", img)
    

    print(f"end of inference all")

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', nargs='+', type=str, default=None, help='image path(s)') # nargs='+' 多个参数放到 列表 list 中
    args = parser.parse_args()
    print(f"process dir = {args.image_path}")

    from configs2.ViTPose_huge_coco_256x192 import model as model_cfg
    from configs2.ViTPose_huge_coco_256x192 import data_cfg
    CKPT_PATH = "~/work/ViTPose_pytorch/vitpose-h-single-coco.pth"
    
    img_size = data_cfg['image_size']
    inference(      img_dir    = args.image_path[0], 
                    img_size   = img_size, 
                    model_cfg  = model_cfg, 
                    ckpt_path  = CKPT_PATH,
                    device     = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu'),
                    save_result= True)