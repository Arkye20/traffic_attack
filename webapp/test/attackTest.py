import os
import shutil

from tqdm import tqdm

from webapp.utils.utils import get_project_root

os.chdir(get_project_root())
from attack import main as attack_main
from PyTorchYOLOv3.detect import DetectorYolov3
from style_transfer import load_img
from torchvision.utils import save_image

from webapp.utils.CONSTANTS import STOP_SIGN_TRAIN_FOLDER, STOP_SIGN_TEST_FOLDER
from webapp.utils.CONSTANTS import ADV_IMAGE_FOLDER, ADV_DET_IMAGE_FOLDER, PATCH_IMAGE_FOLDER
from webapp.utils.CONSTANTS import MASK_IMAGE_FOLDER, STYLE_IMAGE_TIE1, GRADCAM_SAVE_FOLDER
from webapp.utils.CONSTANTS import IMAGE_SIZE, ATTACK_ID, CLASS_NAMES

from webapp.utils.utils import get_image_paths


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def attack(image_folder, save_folder):
    image_paths = get_image_paths(image_folder)
    yolov3 = DetectorYolov3(show_detail=False, tiny=True)
    iterable = range(len(image_paths))
    failed = []
    with tqdm(iterable, desc="攻击图片中", unit="张") as t:
        for i in t:
            image_path = image_paths[i]
            basename = os.path.basename(image_path)
            try:
                attack_main(
                    item=image_path,
                    detectorYolov3=yolov3,
                    STYLE_IMAGE=STYLE_IMAGE_TIE1,
                    MASK_IMAGE=MASK_IMAGE_FOLDER.format(os.path.splitext(basename)[0]),
                    PATCH_PATH=os.path.join(save_folder, basename),
                    ADV_PATH=os.path.join(save_folder, basename),
                    DET_PATH=os.path.join(save_folder, basename),
                    SAVE_PT=True
                )
            except RuntimeError as e:
                print(e)
                failed.append(i)
                print(f"第{i}张图片出现错误")
    print(failed)

def compare():
    SAVE_PATH = 'webapp/images/compare'

    clear_stop_signs = get_image_paths(STOP_SIGN_TRAIN_FOLDER)
    adv_stop_signs = get_image_paths(ADV_IMAGE_FOLDER)

    yolov3 = DetectorYolov3(show_detail=False, tiny=True)
    for i in range(len(clear_stop_signs)):
        basename = os.path.basename(clear_stop_signs[i])
        filename = os.path.splitext(basename)[0]
        save_folder = os.path.join(SAVE_PATH, filename)
        try:
            os.mkdir(save_folder)
        except OSError as error:
            print(error)

        # 加载干净图片
        loaded_clear = load_img(clear_stop_signs[i], IMAGE_SIZE).to(device)
        # 加载对抗样本
        # loaded_adv = load_img(adv_stop_signs[i], IMAGE_SIZE).to(device)
        loaded_adv = torch.load(adv_stop_signs[i]).to(device)
        loaded_adv.requires_grad_(False)

        _, _, clear_det, _, _ = yolov3.detect(input_imgs=loaded_clear, cls_id_attacked=ATTACK_ID)
        _, _, adv_det, _, _ = yolov3.detect(input_imgs=loaded_adv, cls_id_attacked=ATTACK_ID, clear_imgs=loaded_clear)

        detected_clear = yolov3.plot(loaded_clear, CLASS_NAMES, clear_det, 0.5)
        detected_adv = yolov3.plot(loaded_adv, CLASS_NAMES, adv_det, 0.5)

        save_image(detected_clear, os.path.join(save_folder, "clear_" + basename))
        save_image(detected_adv, os.path.join(save_folder, "adv_" + basename))
        print(f"第{i}张图片完成")





if __name__ == '__main__':
    attack()