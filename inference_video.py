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
import fnmatch


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


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
    
    # 读取目录下所有视频 

    # 生成的png和pkl放到一个目录下 

    base_path  = 'dataset/imo_videos/'
    save_root_dir  = 'hand_over_shoulder/' 

    counter = 0
    vitpose_counter = 0
    for filename in find_files(f'{base_path}', '*.mp4'):

        counter += 1
    
        video_path = filename # 'dataset/imo_videos/kongran.mp4'
        save_sub_dir = video_path[len(base_path):] # e.g kongran.mp4
        save_dir = os.path.join(save_root_dir, save_sub_dir[:-4]) # e.g  {save_root_dir}/kongran

        print(f"counter = {counter}")
        print(f"save_root_dir= {save_root_dir}")
        print(f"save_sub_dir = {save_sub_dir}")
        print(f"save_dir     = {save_dir}")
        print(f"filename = {filename}")
        

        video  = cv2.VideoCapture(video_path)
        width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        cur_frame_num = -1 
        while video.isOpened():
            ret, img = video.read()
            if not ret:
                break 

            cur_frame_num += 1 
            img_to_save = img
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            org_w, org_h = img.shape[1], img.shape[0]


            # 模糊帧检测 ----- 可能模糊的vitpose也做不好 

            img_pil = Image.fromarray(img)
            img_tensor = transforms.Compose (
                [transforms.Resize((img_size[1], img_size[0])),
                transforms.ToTensor()]
            )(img_pil).unsqueeze(0).to(device)
        
            # Feed to model
            heatmaps = vit_pose(img_tensor).detach().cpu().numpy() # N, 17, h/4, w/4
            points, prob = keypoints_from_heatmaps( heatmaps = heatmaps, 
                                                    center   = np.array([[org_w//2, org_h//2]]), 
                                                    scale    = np.array([[org_w, org_h]]),
                                                    unbiased = True, 
                                                    use_udp  = True )
            
            # 判断肢体点 
            # 
            ori_points = points
            ori_prob   = prob
            points = points[0]  # (1, 17, 2)
            prob   = prob[0]    # (1, 17, 1)

            save_to_pkl = False 
            prob_threshold = 0.6 
            wrist_below_shoulder = -50
            elbow_below_shoulder = 20
            offset = 5

            begin_offset_x = offset
            begin_offset_y = offset
            end_offset_x   = org_w - offset
            end_offset_y   = org_h - offset

            # 两个肩膀点 必须存在
            if prob[5] > prob_threshold and prob[6] > prob_threshold:
                continue 
        
            if prob[9] > prob_threshold :
                target_point = 0 #5 
                if (begin_offset_y <= points[9][1] < end_offset_y and begin_offset_x <= points[9][0] < end_offset_x)  and points[9][1] < points[target_point][1]  + wrist_below_shoulder :  #   左手腕(允许低过12个像素,不准的)  左肩膀 
                    print(f"match 0 { (points[9][0] - points[0][0]) }") # 左手腕 和 人头区域   改成手腕要在头上 
                    save_to_pkl = True  
            
            if prob[10] > prob_threshold:
                target_point = 0 #6 
                if (begin_offset_y <= points[10][1] < end_offset_y and begin_offset_x <= points[10][0] < end_offset_x) and points[10][1] < points[target_point][1] + wrist_below_shoulder  : # 
                    print(f"match 1")
                    save_to_pkl = True   

            if prob[7] > prob_threshold :
                if (begin_offset_y <= points[7][1] < end_offset_y and begin_offset_x <= points[7][0] < end_offset_x) and points[7][1] < points[5][1]  + elbow_below_shoulder : 
                    print(f"match 2")
                    save_to_pkl = True   

            if prob[8] > prob_threshold :
                if (begin_offset_y <= points[8][1] < end_offset_y and begin_offset_x <= points[8][0] < end_offset_x) and points[8][1] < points[6][1]  + elbow_below_shoulder :   
                    print(f"match 3")
                    save_to_pkl = True   

            # 序列化到文件 
            if save_to_pkl:

                vitpose_counter += 1 
                print(f"save to pkl {cur_frame_num}")

                # 关键点 保存到文件 
                dict_to_pkl = {"num":1 }
                dict_to_pkl['0'] = {}
                dict_to_pkl['0']['obj_conf']   = 1.0
                dict_to_pkl['0']['class_conf'] = 1.0
                dict_to_pkl['0']['box']        = [0, 0, org_w, org_h]   # points 改为了 y x 
                dict_to_pkl['0']['kp']         = np.concatenate([ori_points[:, :, ::-1], ori_prob], axis=2) # 为了跟之前保持一致 

                directory_vitpose = f"{save_dir}_vitpose_pkl"  
                if not os.path.exists(directory_vitpose):  
                    os.makedirs(directory_vitpose)
                with open(f"{directory_vitpose}/{cur_frame_num}.pkl", 'wb') as f:
                    pickle.dump(dict_to_pkl, f)

                directory_image = f"{save_dir}"  
                if not os.path.exists(directory_image):  
                    os.makedirs(directory_image)


                # color = (0, 255, 0)
                # font = cv2.FONT_HERSHEY_SIMPLEX # 定义字体
                # fontScale = 0.5 # 定义字体比例
                # line_thickness = 1 # 定义线条的厚度

            
                # gray = cv2.cvtColor(img_to_save, cv2.COLOR_BGR2GRAY)
                # fm   = cv2.Laplacian(gray, cv2.CV_64F).var()
                # result = "Not Blurry"
                # if fm < 20: # 很难判断这个阈值 
                #     result = "Blurry"
                # img_to_save = cv2.putText(img_to_save, f'{result}', (50, 50), font, fontScale, color, line_thickness, cv2.LINE_AA)   

                # point_size = 4              # 设置点的大小 
                # point_color = (0, 0, 255)   # 设置点的颜色，格式为：(B, G, R)
                # dot_thickness = -1              # 设置点的厚度，-1 表示填充 
                # for k in range(len(points)):
                #     x_p    = points[k][0]
                #     y_p    = points[k][1]
                #     prod_p = prob[k]  
                #     if (prod_p > prob_threshold):
                #         cv2.circle(img_to_save, (int(x_p), int(y_p)), point_size, point_color, dot_thickness)
                #         img_to_save = cv2.putText(img_to_save, f'{k}:{prod_p}', (int(x_p), int(y_p)), font, fontScale, color, line_thickness, cv2.LINE_AA)


                   



                cv2.imwrite(f"{directory_image}/{cur_frame_num}.png", img_to_save)
                #exit(0)
            # Visualization 
            # if True:
            #     for pid, point in enumerate(points):
            #         img = np.array(img)[:, :, ::-1] # RGB to BGR for cv2 modules
            #         img = draw_points_and_skeleton(
            #                 img.copy(), 
            #                 point, 
            #                 joints_dict()['coco']['skeleton'], 
            #                 person_index            = pid,                             
            #                 points_color_palette    = 'gist_rainbow', 
            #                 skeleton_color_palette  = 'jet',
            #                 points_palette_samples  = 10, 
            #                 confidence_threshold    = 0.4)

            #     # if output_video is not None:
            #     #     output_video.write(image_ori)
            #     directory = os.path.dirname(img_path) # 文件所在目录(包含路径)
            #     file_name_without_ext = os.path.splitext(os.path.basename(img_path))[0]
            #     directory_pose = f"{directory}_pose" # 新的目录 
            #     if not os.path.exists(directory_pose): # 如果目录不存在，则创建目录
            #         os.makedirs(directory_pose)
            #     cv2.imwrite(f"{directory_pose}/{file_name_without_ext}.png", img)
    

    print(f"end of inference all")
    print(f"video files counter = {counter} vitpose_counter = {vitpose_counter}")
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', nargs='+', type=str, default=None, help='image path(s)') # nargs='+' 多个参数放到 列表 list 中
    args = parser.parse_args()
    print(f"process dir = {args.image_path}")

    from configs2.ViTPose_huge_coco_256x192 import model as model_cfg
    from configs2.ViTPose_huge_coco_256x192 import data_cfg
    CKPT_PATH = "~/work/ViTPose_pytorch/vitpose-h-single-coco.pth"
    
    img_size = data_cfg['image_size']
    inference(      img_dir    = None, 
                    img_size   = img_size, 
                    model_cfg  = model_cfg, 
                    ckpt_path  = CKPT_PATH,
                    device     = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu'),
                    save_result= True)