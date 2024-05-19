import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import argparse
import numpy as np
from yolov3.models.gradcam import YOLOV3GradCAM, YOLOV3GradCAMPP
from yolov3.models.yolo_detector import YOLOV3TorchObjectDetector
import cv2
# 从interface.py调用时,代码执行到这里时会调用项目根路径下的style_transfer导致报错,修改成yolov3包下的style_transfer
from yolov3.style_transfer import load_img
from webapp.utils.CONSTANTS import *

# yolov3网络中的三个detect层
target_layers = ['model_15_act', 'model_22_act', 'model_27_1_cv2_act']
# Arguments
parser = argparse.ArgumentParser(conflict_handler='resolve')

parser.add_argument('--model-path', type=str, default="./yolov3/weights/yolov3.pt", help='Path to the model')

parser.add_argument('--input', type=str, default='./yolov3/data/test', help='input image path')
parser.add_argument('--output', type=str, default='./yolov3/outputs/', help='output dir')
parser.add_argument('--img-size', type=int, default=640, help="input image size")
parser.add_argument('--target-layer', type=str, default='model_15_act',
                    help='The layer hierarchical address to which gradcam will applied,'
                         ' the names should be separated by underline')
parser.add_argument('--method', type=str, default='gradcampp', help='gradcam method')
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
parser.add_argument('--no_text_box', action='store_true',
                    help='do not show label and box on the heatmap')
parser.add_argument('--saliency', type=str, default='stop sign', help='The class in saliency map')
parser.add_argument('--range', type=str, default='220', help='Attack range')
args = parser.parse_args()


def get_res_img(bbox, mask, res_img):
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(
        np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    # n_heatmat = (Box.fill_outer_box(heatmap, bbox) / 255).astype(np.float32)
    n_heatmat = (heatmap / 255).astype(np.float32)
    res_img = res_img / 255
    res_img = cv2.add(res_img, n_heatmat)
    res_img = (res_img / res_img.max())
    return res_img, n_heatmat, mask


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # this is a bug in cv2. It does not put box on a converted image from torch unless it's buffered and read again!
    cv2.imwrite('temp.jpg', (img * 255).astype(np.uint8))
    img = cv2.imread('temp.jpg')

    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # if label:
    #     tf = max(tl - 1, 1)  # font thickness
    #     t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    #     outside = c1[1] - t_size[1] - 3 >= 0  # label fits outside box up
    #     c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3 if outside else c1[1] + t_size[1] + 3
    #     outsize_right = c2[0] - img.shape[:2][1] > 0  # label fits outside box right
    #     c1 = c1[0] - (c2[0] - img.shape[:2][1]) if outsize_right else c1[0], c1[1]
    #     c2 = c2[0] - (c2[0] - img.shape[:2][1]) if outsize_right else c2[0], c2[1]
    #     cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    #     cv2.putText(img, label, (c1[0], c1[1] - 2 if outside else c2[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
    #                 lineType=cv2.LINE_AA)
    return img


# 检测单个图片
def main(img_path, model, SAVE_DIR=None, attack_range=220, saliency_class='stop sign', gradcam_method='gradcam'):
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in CLASS_NAMES]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = (GRADCAM_IMAGE_SIZE, GRADCAM_IMAGE_SIZE)
    # 读入图片
    img = cv2.imread(img_path)  # 读取图像格式：BGR
    img_size = (img.shape[0], img.shape[1])

    torch_img = load_img(img_path, GRADCAM_IMAGE_SIZE).to(device)
    mask_im = np.zeros(img_size)

    # 遍历三层检测层
    for target_layer in target_layers:
        # 获取grad-cam方法
        if gradcam_method == 'gradcam':
            saliency_method = YOLOV3GradCAM(model=model, layer_name=target_layer, img_size=input_size)
        elif gradcam_method == 'gradcampp':
            saliency_method = YOLOV3GradCAMPP(model=model, layer_name=target_layer, img_size=input_size)
        masks, logits, [boxes, _, class_names, conf] = saliency_method(torch_img)  # 得到预测结果
        result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
        result = result[..., ::-1]  # convert to bgr

        # 遍历每张图片中的每个目标
        for i, mask in enumerate(masks):
            # 遍历图片中的每个目标
            res_img = result.copy()
            # 获取目标的位置和类别信息
            bbox, cls_name = boxes[0][i], class_names[0][i]
            if (cls_name == saliency_class):  # 设置哪一类别的热力图
                label = f'{cls_name} {conf[0][i]}'  # 类别+置信分数
                # 获取目标的热力图
                res_img, heat_map, mask = get_res_img(bbox, mask, res_img)
                res_img = plot_one_box(bbox, res_img, label=label, color=colors[int(CLASS_NAMES.index(cls_name))],
                                       line_thickness=3)
                tmp = np.zeros((res_img.shape[0], res_img.shape[1], 1))
                tmp[bbox[1]:bbox[3], bbox[0]:bbox[2], ] = 1
                mask = mask * tmp
                # 缩放到原图片大小
                res_img = cv2.resize(res_img, dsize=(img.shape[:-1][::-1]))
                mask = cv2.resize(mask, dsize=(img.shape[:-1][::-1]))
                output_path = f'{SAVE_DIR}/{target_layer[6:8]}_{i}_{gradcam_method}.jpg'
                cv2.imwrite(output_path, res_img)
                mask_im += mask
    mask_im = np.where(mask_im < attack_range, 0, 255)  # 设置阈值
    num = len(mask_im[mask_im == 255])

    output_path = f'{SAVE_DIR}/mask_{gradcam_method}.jpg'
    cv2.imwrite(output_path, mask_im)

    return num, SAVE_DIR


if __name__ == '__main__':
    from webapp.utils.utils import *

    os.chdir(get_project_root())

    device = args.device
    input_size = (args.img_size, args.img_size)
    model = YOLOV3TorchObjectDetector(args.model_path, device, img_size=input_size, names=CLASS_NAMES)

    main(
        img_path=args.input,
        model=model,
        SAVE_DIR=args.output,
        saliency_class=args.saliency,
        attack_range=int(args.range)
    )

