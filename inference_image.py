import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from models.model import ViTPose
from utils.visualization import draw_points_and_skeleton, joints_dict
from utils.top_down_eval import keypoints_from_heatmaps
import os
from loguru import logger
import importlib
import sys
import torchvision
from tqdm import tqdm
import pickle

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

# 注意 这里按照 ratio=192/256 是vitpose关键点的输入 推理分辨率
def crop_img_with_faceBox(img_ori, faceBox, ratio=192/256, expand_factor=0.1):
    # faceBox -> [left, top, right, bottom]
    left     = int(faceBox[0])
    top      = int(faceBox[1])
    right    = int(faceBox[2])
    bottom   = int(faceBox[3])

    # 这里的box是已经原图尺寸上的了  yolox的crop_img_with_faceBox会把推理出来的 (中心点+宽高) 转换为 (左上右下)
    left   = max(left,    0)
    top    = max(top,     0)
    right  = min(right,   img_ori.shape[1])
    bottom = min(bottom,  img_ori.shape[0])

    
    cenyer_x = (left + right) / 2.
    cenyer_y = (top  + bottom) / 2.
    
    face_box_width  = right  - left + 1
    face_box_height = bottom - top  + 1
    
    # print(f"{faceBox[0]},{faceBox[1]},{faceBox[2]},{faceBox[3]}")
    # -1.8508682250976562,435.50384521484375,716.6021118164062,1299.3294677734375
    img_ori[:, :left, :] = 0
    img_ori[:, right:, :] = 0
    img_ori[:top, :, :] = 0
    img_ori[bottom:, :, :] = 0

    crop_img_width = int(round(face_box_width + face_box_width * expand_factor * 2))
    crop_img_height = int(round(face_box_height + face_box_height * expand_factor * 2))
    
    
    if (crop_img_width / crop_img_height) < ratio:
        crop_img_width = int(crop_img_height * ratio)
    else:
        crop_img_height = int(crop_img_width / ratio)
    

    # caculate new facebox
    x_min = int(round((cenyer_x) - crop_img_width / 2.0))
    y_min = int(round((cenyer_y) - crop_img_height / 2.0))
    x_max = x_min + crop_img_width
    y_max = y_min + crop_img_height
    
    # crop image
    #cv2.imwrite("~/work/ViTPose_pytorch/result/img_ori.png", img_ori)

    bounding_box = [max(x_min, 0), max(y_min, 0), min(x_max, img_ori.shape[1]), min(y_max, img_ori.shape[0])]
    img_cropped = img_ori[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]].copy()

    #print(f"box:{bounding_box[1]}:{bounding_box[3]},  {bounding_box[0]}:{bounding_box[2]}")
    # cv2.imwrite("~/work/ViTPose_pytorch/result/img_cropped.png", img_cropped)
    
    ## Fill image to specified aspect ratio
    if img_cropped.shape[0] != (y_max - y_min) or img_cropped.shape[1] != (x_max - x_min):
        img_cropped = cv2.copyMakeBorder(img_cropped,
                                        max(-y_min, 0), #top
                                        max(y_max - img_ori.shape[0], 0), #bottom
                                        max(-x_min, 0), #left
                                        max(x_max - img_ori.shape[1], 0), #right
                                        cv2.BORDER_CONSTANT,
                                        value=[0, 0, 0])
    return img_cropped, [x_min, y_min, x_max, y_max]


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    img_ori = img.copy()
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        if cls_id != 0:
            continue
        
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        
        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img

# num_classes 应该是 80

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):

    #print(f"postprocess prediction.shape = {prediction.shape}")
    # postprocess prediction.shape = torch.Size([1, 8400, 85]) 
    # 1=batch size图片的数目  8400一个图中框的数目  85每个框的位置 目标置信度 各类别概率


    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2  # 将边界框的中心点坐标和宽高 转换为左上角和右下角的坐标
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2  # prediction[:, :, 2] / 2 是宽的一半 
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2  #  prediction[:, :, 3] /2 是高的一半 
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]  
    

    # 前 4 个值代表目标的 "边界框坐标" ，即 [left, top, right, bottom]1。
    # 第 5 个值是 "目标的置信度" ，即 obj_conf1。
    # 第 6 到第 85 个值是目标的 "类别概率" ，即 cls_prob。
    # 这里假设模型被训练来检测 80 个不同的类别，所以有 80 个类别概率。
    # 这些概率值经过 softmax 函数处理，使得它们的总和为 1。

    # 后面不用 box_corner 了
    # prediction的 0~3 不包含4 都会 更新成  左上角和右下角的坐标


    output = [None for _ in range(len(prediction))] # 输入有多少个batchSize(图片数量) output就有多少个
    # print(f"before output = { len(output) }") # before output = 1

    for i, image_pred in enumerate(prediction):

        # print(f"i = {i} image_pred = {image_pred.shape}") # i = 0 image_pred = torch.Size([8400, 85]) 
        # image_pred 对应 batch中的 一个图片 

        # If none are remaining => process next image
        if not image_pred.size(0):
            # 没有8400的框
            continue

        # Get score and class with highest confidence  
        #  image_pred[:, 5: 5 + num_classes] 这个是80个类别的概率 
        #  class_conf class_pred 对每个框 筛选出 最高概率 的 类别 和 该类别的概率 
        # 
        #  torch.max keepdim = True 用于指定是否在结果中保持输入张量的维度 (HHL: 原本减少的一个维度 存放值最大的索引)
        #            keepdim=False(默认值) 在执行最大值操作的维度将不会出现在输出张量中(输出张量的维度会比输入张量少一个)
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True) # 1 在dim=1维度上求最大 

        # print(f"class_conf = {class_conf.shape},  class_pred = {class_pred.shape}")
        #  class_conf = torch.Size([8400, 1]),  class_pred = torch.Size([8400, 1])
        # 
        # 8400个框  '每个框的最大置信度'  和  '取得最大置信度的对应类别'
        # 

        # YOLOX 模型在 "多个尺度" 上进行 "目标检测"，"每个尺度上的特征图" 都会产生 "一定数量的预测边界框"。
        # 这些边界框的数量 取决于 特征图的大小 以及 每个网格单元预测的边界框的数量

        # 如果模型在一个 40x40 的特征图上对每个网格单元预测 3 个边界框，那么这个特征图就会产生 40 * 40 * 3 = 4800 个预测边界框

        # '框置信度'和'类别置信度'的乘积 决定, '类别特定的置信度分数'  提高这个可能避免 误判成人 (而不需要改nms的iou_threshold)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        #  image_pred[:, 4] 是 目标(框)的概率  obj_conf
        #  class_conf.squeeze() 是 目标(框)的 最大的类别概率(Top-1)
        #  目标框 和 类别 两个概率相乘  要高于 阈值  conf_thre -----------------过滤掉 置信度不高的框 
        # print(f"conf_mask {conf_mask.shape}, {torch.sum(conf_mask)}") # conf_mask torch.Size([8400]), 34
        

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)  输出格式变成了[左上右下:0123 目标置信度:4 类别概率 类别 ]
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        #print(f"before mask detections = {detections.shape}") # torch.Size([8400, 7])
        detections = detections[conf_mask]
        #print(f"after  mask detections = {detections.shape}") # torch.Size([34, 7]


        if not detections.size(0): # 这里其实代表config_mask len是0?
            continue

       
        # class_agnostic = True, nms_out_index = tensor([22,  4, 21, 11], device='cuda:0')
        # class_agnostic  类不可知的 ?
        if class_agnostic: 

            # 非极大值抑制
            # torchvision.ops.nms 函数的参数包括：
            # boxes：需要执行 NMS 的检测框，它们应该是以 (x1, y1, x2, y2) 格式表示的，其中 0 <= x1 < x2 且 0 <= y1 < y21。
            # scores：每个检测框的得分1。
            # iou_threshold：用于判断两个检测框是否重叠的阈值1。
            # 这个函数的返回值是一个整数型的 Tensor，包含了被 NMS 保留的元素的索引，这些索引是按照得分的降序排列的1。

            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5], # 是目标概率*类别概率 
                detections[:, 6], # :6是类别 
                nms_thre,
            )

        print(f"nms_thre = {nms_thre} class_agnostic = {class_agnostic}, nms_out_index = {nms_out_index}") 
        # class_agnostic = True, nms_out_index = tensor([22,  4, 21, 11], device='cuda:0') # 这个是列表 表示第 22,4,21,11 的框 是ok的

        detections = detections[nms_out_index]

        # 打印保留下来的 框的类别 和 框   detections的格式   (x1, y1, x2, y2, obj_conf, class_conf, class_pred) 
        print(f"after nms detections  = {detections}")

        #print(f"after nms detections = {detections.shape}")  # after nms detections = torch.Size([4, 7])
        # 经过阈值过滤 剩下 34 个框  经过 非极大值抑制 剩下 4个框  (nms 并没有 合并框)
        #print(f"detections = {detections}")
        if output[i] is None:
            output[i] = detections
            # output[i] 是对应照片i的 可能有多个框  框可能是几个人的 或者其他类别的 
        else:
            output[i] = torch.cat((output[i], detections)) # 如果之前 i 已经有值  就cat起来 ??  现在只有0 
            #print(f"output[{i}],  shape = {output[i].shape} ")  # 不会出现  batch有两个一定的i?

    return output


def get_exp_by_file(exp_file):
    try:
        sys.path.append(os.path.dirname(exp_file))
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        exp = current_exp.Exp()
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp


def get_exp_by_name(exp_name):
    exp = exp_name.replace("-", "_")  # convert string like "yolox-s" to "yolox_s"   yolox-x 转为 yolox_x
    module_name = ".".join(["yolox", "exp", "default", exp]) # 转为 yolox.exp.default.yolox_x ???    # 需要安装YoloX模块 

    # https://github.com/Megvii-BaseDetection/YOLOX/blob/main/exps/default/yolox_x.py
    # 定义了 class Exp(MyExp):

    # yolox.exp.default.yolox_x 是 YOLOX 模型的一种配置，它定义了模型的深度和宽度。
    # 在这种配置下，模型的深度为 1.33，宽度为 1.25
    # [left, top, right, bottom, obj_conf, cls_conf, cls_id]  边界框坐标  目标的置信度  类别的置信度  目标的类别ID
 
    # 首先，importlib.import_module(module_name) 会动态地导入名为 module_name 的模块
    # 然后，.Exp() 是在这个模块上调用 Exp 类或函数  调用这个函数 或者类构造 

    # 在 YOLOX 中，Exp() 是一个类，它在 yolox_base.py 文件中定义，
    # 用于封装模型的配置，创建，数据加载器，前处理，后处理，优化，学习率等 
    # Exp() 类是从 MyExp 类继承的

    exp_object = importlib.import_module(module_name).Exp() 
    return exp_object


def get_exp(exp_file=None, exp_name=None):
    """
    get Exp object by file or name. If exp_file and exp_name
    are both provided, get Exp by exp_file.

    Args:
        exp_file (str): file path of experiment.
        exp_name (str): name of experiment. "yolo-s",
    """
    assert (
        exp_file is not None or exp_name is not None
    ), "plz provide exp file or exp name."
    if exp_file is not None:
        print(f"get_exp_by_file = {exp_file}")
        return get_exp_by_file(exp_file)
    else:
        print(f"get_exp_by_name = {exp_name}")
        return get_exp_by_name(exp_name)
    
    
def preproc(img, input_size, swap=(2, 0, 1)): # input_size是目标尺寸 img会缩放到里面 padding颜色是114
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114  # padding的颜色是114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1]) # 按哪一边比例小的缩放 
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img # resize部分填入 

    padded_img = padded_img.transpose(swap) # 变换顺序 HWC -> CHW  
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) # 内存不连续 改为内存连续 ??? 
    return padded_img, r


class ValTransform:
    def __init__(self, swap=(2, 0, 1)):
        self.swap = swap

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size): # res参数没有使用 
        img, _ = preproc(img, input_size, self.swap)
        return img, np.zeros((1, 5)) # 返回 array([[0., 0., 0., 0., 0.]])?
    

def make_parser(): # 这里是 Yolox的官方代码 解析 https://github.com/Megvii-BaseDetection/YOLOX/blob/main/tools/demo.py
    parser = argparse.ArgumentParser("YOLOX Demo!") 
    # parser.add_argument( # args.demo == "image"  必须 python inference_image.py image
    #     "demo", default="image", help="demo type, eg. image, video and webcam"
    # ) 
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default='yolox-x', help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        default=True,
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf",  default = 0.25, type=float, help="test conf")          # 0.25  # test_conf  0.01
    parser.add_argument("--nms",   default = 0.45, type=float, help="test nms threshold") # 0.45  # nmsthre 0.65 
    parser.add_argument("--tsize", default = None, type=int,   help="test img size")      # 640   # test_size (640, 640) 

    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        device,
        cls_names=COCO_CLASSES,
    ):
        self.model = model
        self.cls_names = cls_names  # 这个来自上面自己写的class list 

        self.num_classes = exp.num_classes # 这个来自于模型里面 import yolox_x 
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.preproc = ValTransform()

        # print(f"cls_names   = {self.cls_names}")
        # print(f"num_classes = {self.num_classes}")
        print(f"confthre  = {self.confthre} ")
        print(f"nmsthre   = {self.nmsthre} ")
        print(f"test_size = {self.test_size}")

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"]  = height
        img_info["width"]   = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1]) # 保证全图 
        img_info["ratio"] = ratio # self.preproc 也是同样缩放的 (yolox推理时候 是等比例缩放过的! 加padding 宽高都是 tsize = 640x640 正方形  )

        img, _ = self.preproc(img, None, self.test_size)

        # https://github.com/Megvii-BaseDetection/YOLOX/issues/1492
        # yoloX用的是BGR cv2.imread返回的 
        
        # img 还是BGR 直接保存 跟cv2一致 
        #cv2.imwrite(f"temp.png", np.transpose(img, (1, 2, 0)))

        img = torch.from_numpy(img).unsqueeze(0) # 加上Batch维度 (实际只有一张图片)
        img = img.float()

        img = img.to(self.device)

        # 输入只有一个图片 但是加了batch维度  输出会带有batch维度 
        outputs = self.model(img)

        # print(f"before postprocess outputs={outputs.shape} ") # before postprocess outputs=torch.Size([1, 8400, 85])
        outputs = postprocess(
            outputs, 
            self.num_classes, 
            self.confthre,
            self.nmsthre, 
            class_agnostic=True
        )
        # print(f"after postprocess outputs={len(outputs)} ")
        # for k in range(len(outputs)):
        #     print(f"after postprocess outputs[{k}] = {outputs[k].shape}")

        return outputs, img_info # ratio 原图缩小的比例 height width 宽高 raw_img 原图 

    # def visual(self, output, img_info, cls_conf=0.35):
    #     ratio = img_info["ratio"]
    #     img = img_info["raw_img"]
    #     if output is None:
    #         return img
    #     output = output.cpu()

    #     bboxes = output[:, 0:4]

    #     # preprocessing: resize
    #     bboxes /= ratio

    #     cls = output[:, 6]
    #     scores = output[:, 4] * output[:, 5]
        
    #     vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)   
        
    #     return vis_res
    
    def crop_with_bbox(self, output, img_info, cls_conf=0.35): # 用的是 predictor.confthre
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        
        if output is None:
            return []
        output = output.cpu()

        #                             4         5           6
        # output = (x1, y1, x2, y2, obj_conf, class_conf, class_pred)

        bboxes = output[:, 0:4] # 每个框的坐标(左上右下 )  

        # preprocessing: resize # 坐标 恢复到原图尺寸下 
        bboxes /= ratio  

        cls = output[:, 6] # 类别 
        scores = output[:, 4] * output[:, 5] # 目标置信度*类别概率(已经是这个框概率最大的了)
        

        #cv2.imwrite("~/work/ViTPose_pytorch/result/image.png", img)


        res = []
        for i in range(len(bboxes)):
            box = bboxes[i]  # 这个是原图尺寸下的 坐标 
            cls_id = int(cls[i])
            score = scores[i]
            if cls_id != 0 or score < cls_conf: # 去掉不是'人类' 
                # print(f"cls_id:{cls_id}, name:{self.cls_names[cls_id]} score:{score}")
                # cls_id:56, name:chair score:0.8473848700523376
                # cls_id:27, name:tie score:0.30106642842292786
                continue

            # print(f"box = {box}") # COCO的结果没有超出图像分辨率 

            #box = torch.tensor([0, 497, 705, 1302])


            crop_img, crop_box = crop_img_with_faceBox(img.copy(), box, expand_factor = 0.1) 
            # , expand_factor=0.1 expand_factor 会扩大yolo框 如果超出原图尺寸 会增加黑边 
            #  expand_factor 是 单独一边 要 增加 height或者width的 百分比   最后的width或者height会增加 w*expand_factor*2 
            #  ratio=192/256 默认参数是按照 vitpose肢体点 推理输入的分辨率的

            # crop_img=(920, 690, 3) 原图大小是738x1312 宽已经超出了  为了对齐 192/256
            # crop_box 这个位置 可能 超出图像分辨率

            obj_conf   = output[i, 4]
            class_conf = output[i, 5]

            print(f"{i}:score={score}, obj_conf={obj_conf}, class_conf={class_conf},",
                  f"crop_img={crop_img.shape},{crop_img.dtype},", 
                  f"crop_box[0]={crop_box[0]} crop_box[1]={crop_box[1]}, crop_box[2]={crop_box[2]} crop_box[3]={crop_box[3]},", 
                  f"w={crop_box[2]-crop_box[0]}, h = {crop_box[3]-crop_box[1]}" ) 
                  # 一张照片中只有一个人 但是有两个框
   
            
            # 0:score=0.9090335965156555,  obj_conf=0.9812582731246948, class_conf=0.926395833492279, cls_conf=0.25   
            # 2:score=0.32688212394714355, obj_conf=0.3788975477218628, class_conf=0.8627190589904785, cls_conf=0.25 框的置信度不高 但是类别的概率比较高

            #cv2.imwrite("~/work/ViTPose_pytorch/result/crop_image.png", crop_img)


            res.append([crop_img, crop_box, obj_conf, class_conf, box.detach().cpu().numpy()]) # 裁剪出原图的一部分 并记录在原图的位置 

        return res

def similar_transform(landmark2d, roi_box):
    roi_box = [int(round(_)) for _ in roi_box]
    left, top, right, bottom = roi_box[:4]

    half_width = (right - left) / 2.0
    half_height = (bottom - top) / 2.0
    
    center_x = right - half_width
    center_y = bottom - half_height

    landmark2d[:, 0] = landmark2d[:, 0] * half_width + center_x
    landmark2d[:, 1] = landmark2d[:, 1] * half_height + center_y

    return landmark2d

def image_demo(predictor, path, save_result, vit_pose, device):
    is_dir = False 
    if os.path.isdir(path):
        #files = get_image_list(path)
        files = os.listdir(path)
        files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
        is_dir = True 
    else:
        files = [path]

    
    output_video = None 
    
    for p in tqdm(range(len(files))):
        
        image_name = os.path.join(path, files[p]) if is_dir else files[p] 
        #print(f"{image_name}")

        # yolo 推理完成  
        # outputs:List [ torch.Size([过滤后框的数目, 7]),  ......] # 实际只有outputs[0] 一个图片 
        # img_info:map "file_name" "height" "width" "raw_img" "radio"
        
        # print(f"image_name = {image_name}") # imge_name是全路径
        outputs, img_info = predictor.inference(image_name) # 这里的crop坐标是缩放后的

        # BGR顺序 
        image_ori = img_info["raw_img"].copy()


        # 显示存留框 
        output = outputs[0]
        ratio = img_info["ratio"]
        # output = torch.Size([2, 7]) 2 = 存留的框  7 = (x1, y1, x2, y2, obj_conf, class_conf, class_pred) 
        for k in range(output.shape[0]):
            box = output[k]
            left, top, right, bottom, obj_conf, class_conf, class_pred = box
            left, top, right, bottom = left/ratio, top/ratio, right/ratio, bottom/ratio  
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)
            left_top = (left, top)
            right_top = (right, top)
            right_bottom = (right, bottom)
            left_bottom = (left, bottom)
            image = image_ori.copy()
            cv2.line(image, left_top, right_top, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.line(image, right_top, right_bottom,  (255, 255, 0), 2, cv2.LINE_AA)
            cv2.line(image, right_bottom, left_bottom,  (255, 255, 0), 2, cv2.LINE_AA)
            cv2.line(image, left_bottom, left_top,  (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"obj_conf={int(obj_conf*100)},class_conf={int(class_conf*100)},class_pred={class_pred}", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), thickness=1)
            cv2.imwrite(f"box_{k}.png", image)
            
            



        # if output_video is None:
        #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        #     OUTPUT_VIDEO_PATH = "~/work/ViTPose_pytorch/result/visual_pose.mp4"
        #     output_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 30.0, (image_ori.shape[1], image_ori.shape[0])) 

        # outputs[0] 是代表 batch 中的第一个图片 
  
        crop_res = predictor.crop_with_bbox(outputs[0], img_info, predictor.confthre) 
        # postprocess已经 过滤出 目标置信度*类别概率 > confthre
        # crop_with_bbox 会筛选出 人 而不是其他物体  crop_with_bbox 会按照推理的分辨率 192,256
        # 返回的box会超出图像 
        
        print(f"len(crop_res) = {len(crop_res)}") # len(crop_res) = 2  这里在过滤一下 4个框变成2个了? 
        if len(crop_res) == 0:
            print("%s can't detect any person" % image_name)
            continue

        # 如果完全没有检测到人的话 就没有pkl文件了
        # {'num':5,  '0':{ 'obj_conf'=0.9, 'class_conf'=0.8, 'box'=[0 1 2 3] 'kp'=numpy(1, 17, 3)  },  '1':{} }
        dict_to_pkl = {"num":0 }

        idx = 0
        ## w = data_cfg['image_size'][0]  h = data_cfg['image_size'][0]  ##  竖屏 
        img_size = (192, 256)  # 这个是模型推理的图像尺寸?
        result = []

        human_id = -1

        for human_info in crop_res: # 图片有多个人 会循环多次 
            crop_img    = human_info[0]
            crop_box    = human_info[1]
            obj_conf    = human_info[2].item()
            class_conf  = human_info[3].item()
            box         = human_info[4]  # yolox 输出 已转换为原始尺寸的 坐标 
            #print(f"box = {box}")

            human_id = human_id + 1

            if (len(crop_box) != 4):
                raise Exception(f"crop_box not 4 = {len(crop_box)}")

            #if idx == 1:
            #    idx = idx + 1 
            #    continue 
            #print(f"obj_conf = {obj_conf}")

            # 0: crop_img=(821, 616, 3) crop_box[0]=0   crop_box[1]=486
            # 1: crop_img=(920, 690, 3) crop_box[0]=176 crop_box[1]=362 
            #print(f"{idx}: crop_img={crop_img.shape} crop_box[0]={crop_box[0]} crop_box[1]={crop_box[1]} ")

            #print(f"type = {type(crop_img)}, shape = {crop_img.shape} dtype = {crop_img.dtype}")
            # type = <class 'numpy.ndarray'>, shape = (1149, 862, 3) dtype = uint8
            #cv2.imwrite(f"~/work/ViTPose_pytorch/result/res_pose_{human_id}_crop.png",  crop_img)


            # Prepare input data
            img = Image.fromarray(cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)) 
            org_w, org_h = img.size
            # print(f"vitpose resize {org_w},{org_h} -> {img_size[0]},{img_size[1]} ") # vitpose resize 739,985 -> 192,256
            img_tensor = transforms.Compose ([
                transforms.Resize((img_size[1], img_size[0])), # 不会形变  直接转成 (192, 256)   --- HHL.2024.07.11 推理输入是 192,256 crop_with_bbox 会按照这个比例保持
                transforms.ToTensor()])(img).unsqueeze(0).to(device) # 变形resize 转Tensor 加Batch维度 提交GPU 

            # Feed to model  推给引擎 
            heatmaps = vit_pose(img_tensor).detach().cpu().numpy() # N, 17, h/4, w/4

            #print(f"heatmaps = {heatmaps.shape}") # (1, 17, 64, 48)

            # 在 VitPose 中，keypoints_from_heatmaps 函数的作用是从热图（heatmaps）中提取关键点
            # 在许多姿态估计任务中，模型的输出通常是一组热图，每个热图对应一个关键点的位置
            # 热图上的每个像素值表示对应位置是关键点的概率。
            # 因此，我们可以通过找到热图上的最大值来确定关键点的位置 

            # keypoints_from_heatmaps 它遍历每个热图，找到最大值的位置，然后返回这些位置作为关键点的坐标12。
            # 需要注意的是，由于热图的分辨率通常低于原始图像，所以提取出的关键点坐标可能需要进行上采样或插值以得到更精确的位
 

            # points = heatmap2coords(heatmaps=heatmaps, original_resolution=(org_h, org_w))
            points, prob = keypoints_from_heatmaps(
                                        heatmaps  = heatmaps, 
                                        center    = np.array([[org_w//2, org_h//2]]), 
                                        scale     = np.array([[org_w,     org_h]]),
                                        unbiased  = True, 
                                        use_udp   = True)

            #print(f"points = {points} prob = {prob}")
            transfom_points = points.copy() # 一个人  或者一个框 
            transfom_points[:, :, 0] = transfom_points[:, :, 0] + crop_box[0]
            transfom_points[:, :, 1] = transfom_points[:, :, 1] + crop_box[1] 
            #print(f"transfom_points = {transfom_points} prob = {prob}")

            # crop_box记录在原图上的左上右下 
            # 这里恢复到原图尺寸下的坐标 
            
            # print(f"transfom_points = {transfom_points.shape}") # numpy transfom_points = (1, 17, 2) # 17个关键点 每个是2为的坐标
            # print(f"prob = f{prob.shape}, {prob}") # prob = f(1, 17, 1) # eval() 模式下 每次执行关键点概率 应该是一致的 

            #  NumPy[::-1] 是一种切片操作，用于生成数组的逆序版本1
            points_draw = np.concatenate([transfom_points[:, :, ::-1], prob], axis=2)  # 为什么 point=(y, x) ?
            # print(f"points_draw = {points_draw.shape}") # points_draw = (1, 17, 3)

            if len(points_draw) == 0:
                print("%s human pose detect fail" % image_name)
                continue # 这里不会 idx+=1 
            
            dict_to_pkl[f'{idx}'] = {}
            dict_to_pkl[f'{idx}']['obj_conf']   = obj_conf
            dict_to_pkl[f'{idx}']['class_conf'] = class_conf
            dict_to_pkl[f'{idx}']['box'] = crop_box
            dict_to_pkl[f'{idx}']['kp']  = points_draw


            if True:
                result.append([idx, np.array(crop_box), points_draw])
                # print(points_draw.shape)
                
                # points_draw[points_draw < 0] = 0
                # Visualization 
                if save_result:

                    colors = [ (0, 0, 255) , (255, 0, 0) , (0, 255, 0) ]

                    for pid, point in enumerate(points_draw):
                        #print(point)
                        image_ori = draw_points_and_skeleton(
                            image_ori, 
                            point, 
                            joints_dict()['coco']['skeleton'], 
                            person_index           = human_id,
                            points_color_palette   = 'gist_rainbow', 
                            skeleton_color_palette = 'jet',
                            points_palette_samples = 10, 
                            confidence_threshold   = 0.4) # 只绘制 >= 0.4的点  改成0.6减少误判
                    
                        for k in range(len(point)):
                            if point[k][2] > 0.4:
                                text = f"{k}:{point[k][2]*100:.2f}"
                                position = (int(point[k][1]), int(point[k][0]))   
                                font = cv2.FONT_HERSHEY_SIMPLEX 
                                font_size = 0.4
                                color = colors[human_id]   
                                thickness = 1
                                #print(f"position = {position} ")
                                cv2.putText(image_ori, text, position, font, font_size, color, thickness)

                    if False:
                        save_name = "~/work/ViTPose_pytorch/result/res_pose_%s" % image_name.split("/")[-1]
                        cv2.imwrite(save_name, image_ori)
                    else:
                        directory = os.path.dirname(image_name) # 文件所在目录(包含路径)
                        file_name_without_ext = os.path.splitext(os.path.basename(image_name))[0]
                        directory_pose = f"{directory}_pose" # 新的目录 
                        if not os.path.exists(directory_pose): # 如果目录不存在，则创建目录
                            os.makedirs(directory_pose)
                        cv2.imwrite(f"{directory_pose}/{file_name_without_ext}.png", image_ori)


                    if output_video is not None:
                        output_video.write(image_ori)
                    
            idx += 1  

        dict_to_pkl['num'] = idx
        #print(f"dict_to_pkl = {dict_to_pkl}")

        # 写入文件中 
        if True:
            directory = os.path.dirname(image_name) 
            file_name_without_ext = os.path.splitext(os.path.basename(image_name))[0]
            directory_vitpose = f"{directory}_vitpose_pkl"  
            if not os.path.exists(directory_vitpose):  
                os.makedirs(directory_vitpose)
            with open(f"{directory_vitpose}/{file_name_without_ext}.pkl", 'wb') as f:
                pickle.dump(dict_to_pkl, f)


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    
    os.makedirs(file_name, exist_ok=True)

    logger.info("Args: {}".format(args))

    # 命令行参数 覆盖掉 原来exp里面的 
    if args.conf is not None:
        print(f"[Replace] test_conf {exp.test_conf} -> {args.conf}")
        exp.test_conf = args.conf  
    if args.nms is not None:
        print(f"[Replace] nmsthre   {exp.nmsthre} -> {args.nms}")
        exp.nmsthre = args.nms
    if args.tsize is not None:
        print(f"[Replace] test_size {args.tsize}  -> {args.tsize}")
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model() # 从Exp中获取模型 

    if args.device == "gpu":
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
        
    # load the model state dict
    ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)
    logger.info("loaded checkpoint done.")
    
    
    # Prepare pose model
    vit_pose = ViTPose(model_cfg)
    ckpt = torch.load(args.ckpt_path)
    if 'state_dict' in ckpt:
        vit_pose.load_state_dict(ckpt['state_dict'])
    else:
        vit_pose.load_state_dict(ckpt)
    vit_pose.eval()
    vit_pose.to(device)
    print(f">>> Model loaded: {args.ckpt_path}")


    # COCO的 Predictor
    predictor = Predictor(model, exp, cls_names=COCO_CLASSES, device=device)
    with torch.no_grad():
        image_demo(predictor, args.path, args.save_result, vit_pose, device)
    

# 用法: 
# CUDA_VISIBLE_DEVICES=1 python inference_image.py  -c  yolox_x.pth  --path /home/hehanlong/work/ViTPose_pytorch/my_test_data/17-10-2_0001_00006_.png
# 
# 
# 
if __name__ == "__main__":
    args = make_parser().parse_args()

    args.conf = 0.75
    args.nms  = 0.45

    # 根据名字  获取yolo-x的模型 
    exp = get_exp(args.exp_file, args.name)

    print(f"{exp}")
    

    # from configs.ViTPose_huge_coco_256x192 import model as model_cfg  # 这个要对应权重文件 
    # from configs.ViTPose_huge_coco_256x192 import data_cfg

    from configs2.ViTPose_huge_coco_256x192 import model as model_cfg  # 
    from configs2.ViTPose_huge_coco_256x192 import data_cfg
    # data_cfg 定义了推理宽高 
    # configs/ViTPose_huge_coco_256x192.py  这个文件定义了模型结构 数据集 
    # configs2/ViTPose_huge_coco_256x192.py  或者自己新增 一个配置  对应 vitpose-h-single-coco.pth"

    # https://github.com/ViTAE-Transformer/ViTPose?tab=readme-ov-file 
    # 这里有对应的 config 和 weight 可以下载

    # multi-task traning 是不同的decoder 对应 不同的数据集 ??  这样 最终上线用哪个?? 

    # multi-task training 
    #   Human datasets (MS COCO, AIC, MPII, CrowdPose) Results on MS COCO val set 的 
    #   ViTPose-H* ViTPose_huge_coco_256x192.py   对应权重是  vitpose-h-multi-coco.pth
    #   
    # single-task training 
    #   ViTPose-H ViTPose_huge_coco_256x192.py  
    # 
    # config是一样的 

    # 但是 Results on MS COCO val set 和 Results on OCHuman test set 是不同的 

    # args.ckpt_path = "~/work/ViTPose_pytorch/vitpose-h-multi-coco.pth" # vitpose的模型 
    args.ckpt_path = "/home/hehanlong/work/ViTPose_pytorch/vitpose-h-single-coco.pth"

    main(exp, args)
    