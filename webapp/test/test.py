import os.path
import shutil
import subprocess
import sys

from webapp.utils.utils import *
from webapp.utils.CONSTANTS import *

os.chdir(get_project_root())
from PyTorchYOLOv3.detect import DetectorYolov3
from torchvision.utils import save_image
from style_transfer import load_img
from tqdm import tqdm
from attack import main as attack_main
import torch
from PyTorchYOLOv3.utils.utils import non_max_suppression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def selectType(image_folder, save_folder, class_id):
    image_paths = get_image_paths(image_folder)
    yolov3 = DetectorYolov3(show_detail=False, tiny=True)
    iterable = range(len(image_paths))
    with tqdm(iterable, desc="挑选目标图片中", unit="张") as t:
        for i in t:
            image_path = image_paths[i]
            save_path = os.path.join(save_folder, os.path.basename(image_path))
            loaded_image = load_img(image_path, IMAGE_SIZE).to(device)
            _, _, det, _, _ = yolov3.detect(input_imgs=loaded_image, cls_id_attacked=ATTACK_ID)
            if det is None:
                continue
            elif class_id in det[:, 6]:
                shutil.copy(image_path, save_path)


def DetectorYolov3Test(image_folder, save_folder):
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


def GradCamTest(image_folder, save_folder):
    count = 0
    image_paths = get_image_paths(image_folder)
    for image_path in image_paths:
        command = [
            sys.executable, "yolov3/main_gradcam.py",
            "--input", image_path,
            "--output", save_folder,
            "--saliency", CLASS_NAMES[ATTACK_ID],
            "--range", str(DEFAULT_ATTACK_RANGE)
        ]
        try:
            results = subprocess.check_output(command).decode()
            print(results)
        except subprocess.CalledProcessError as e:
            print(e.output.decode())
        print("================================================")
        print(f"第{count}次已完成")
        print("================================================")
        count += 1
        # break


def AttackTest(image_folder, mask_folder, adv_folder, adv_det_folder, patch_folder, style_image, save_pt=False):
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
                    STYLE_IMAGE=style_image,
                    MASK_IMAGE=mask_folder.format(os.path.splitext(basename)[0]),
                    PATCH_PATH=os.path.join(patch_folder, basename),
                    ADV_PATH=os.path.join(adv_folder, basename),
                    DET_PATH=os.path.join(adv_det_folder, basename),
                    SAVE_PT=save_pt
                )
            except RuntimeError as e:
                print(e)
                failed.append(i)
                print(f"第{i}张图片出现错误")
    print(failed)


def remove_files(folder1, folder2):
    # paths1 = get_image_paths(folder1)
    paths1 = [os.path.splitext(os.path.basename(file))[0] for file in get_image_paths(folder1)]
    paths2 = get_image_paths(folder2)
    for i in paths2:
        filename = os.path.basename(i)
        if filename not in paths1:
            print(filename)
            shutil.rmtree(i)


def save_labels(image_folder, save_folder):
    image_paths = get_image_paths(image_folder)
    yolov3 = DetectorYolov3(show_detail=False, tiny=True)
    iterable = range(len(image_paths))
    with tqdm(iterable, desc="检测图片中", unit="张") as t:
        for i in t:
            image_path = image_paths[i]
            # save_path = os.path.join(save_folder, os.path.basename(image_path))
            save_path = os.path.join(save_folder, os.path.splitext(os.path.basename(image_path))[0] + '.pt')
            loaded_image = load_img(image_path, IMAGE_SIZE).to(device)
            _, _, det, _, _ = yolov3.detect(input_imgs=loaded_image, cls_id_attacked=ATTACK_ID)
            if det is None:
                continue
            else:
                torch.save(det[0], save_path)
            # detected_img = yolov3.plot(loaded_image, CLASS_NAMES, det, 0.5)
            # save_image(detected_img, save_path)


if __name__ == '__main__':

    GradCamTest(STOP_SIGN_ALL_FOLDER, GRADCAM_SAVE_FOLDER)


    # AttackTest(
    #     image_folder=STOP_SIGN_ALL_FOLDER,
    #     mask_folder=MASK_IMAGE_FOLDER,
    #     adv_folder=ADV_IMAGE_FOLDER,
    #     adv_det_folder=ADV_DET_IMAGE_FOLDER,
    #     patch_folder=PATCH_IMAGE_FOLDER,
    #     style_image=STYLE_IMAGE_TIE1,
    #     save_pt=True
    # )
