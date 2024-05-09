import os

from tqdm import tqdm

from webapp.utils.utils import get_project_root

os.chdir(get_project_root())
from PyTorchYOLOv3.detect import DetectorYolov3
from torchvision.utils import save_image
from style_transfer import load_img
from webapp.utils.utils import get_image_paths

from webapp.utils.CONSTANTS import CLASS_NAMES
from webapp.utils.CONSTANTS import IMAGE_SIZE
from webapp.utils.CONSTANTS import ATTACK_ID
from webapp.utils.CONSTANTS import SAVE_FOLDER

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def yolov3Test(image_folder, save_folder):
    image_paths = get_image_paths(image_folder)
    yolov3 = DetectorYolov3(show_detail=False, tiny=True)
    iterable = range(len(image_paths))
    with tqdm(iterable, desc="检测图片中", unit="张") as t:
        for i in t:
            image_path = image_paths[i]
            save_path = os.path.join(save_folder, os.path.basename(image_path))
            loaded_image = load_img(image_path, IMAGE_SIZE).to(device)
            _, _, det, _, _ = yolov3.detect(input_imgs=loaded_image, cls_id_attacked=ATTACK_ID)
            detected_img = yolov3.plot(loaded_image, CLASS_NAMES, det, 0.5)
            save_image(detected_img, save_path)


# def select_Stopsign_From_Dataset():
#     iterable = range(len(image_paths))
#     with tqdm(iterable, desc="分拣中", unit="张") as t:
#         for i in t:
#             image_path = image_paths[i]
#             loaded_image = load_img(image_path, IMAGE_SIZE).to(device)
#             _, _, det, _, _ = detectorYolov3.detect(input_imgs=loaded_image, cls_id_attacked=ATTACK_ID)
#             if det is None:
#                 continue
#             labels = [int(i[6]) for i in det]
#             if 11 not in labels:
#                 continue
#             else:
#                 save_path = os.path.join(STOP_SIGN_TEST_FOLDER, os.path.basename(image_path))
#                 save_image(loaded_image, save_path)


if __name__ == '__main__':
    yolov3Test(image_folder=VAL2017_FOLDER, save_folder=SAVE_FOLDER)
    # select_Stopsign_From_Dataset()
