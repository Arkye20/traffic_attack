import os

from webapp.utils.utils import get_project_root

os.chdir(get_project_root())
import subprocess
import sys

from webapp.utils.CONSTANTS import CLASS_NAMES
from webapp.utils.CONSTANTS import GRADCAM_SAVE_FOLDER
from webapp.utils.CONSTANTS import STOP_SIGNS_TRAIN_FOLDER
from webapp.utils.CONSTANTS import YOLOv3_MODEL_WEIGHTS_PATH
from webapp.utils.utils import get_image_paths
from yolov3.main_gradcam import args as gradcam_args
from yolov3.models.yolo_detector import YOLOV3TorchObjectDetector

if __name__ == '__main__':

    input_size = (gradcam_args.img_size, gradcam_args.img_size)
    gradcam_model = YOLOV3TorchObjectDetector(
        YOLOv3_MODEL_WEIGHTS_PATH,
        gradcam_args.device,
        img_size=input_size,
        names=CLASS_NAMES
    )

    attack_range = 220
    image_paths = get_image_paths(STOP_SIGNS_TRAIN_FOLDER)
    count = 1

    for image_path in image_paths:
        command = [
            sys.executable, "yolov3/main_gradcam.py",
            "--input", image_path,
            "--output", GRADCAM_SAVE_FOLDER
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
