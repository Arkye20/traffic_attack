import os
import subprocess
import sys

from webapp.utils.utils import get_project_root

os.chdir(get_project_root())
from webapp.utils.CONSTANTS import STOP_SIGNS_TRAIN_FOLDER
from webapp.utils.CONSTANTS import MASK_IMAGE_FOLDER
from webapp.utils.CONSTANTS import PATCH_IMAGE_FOLDER
from webapp.utils.CONSTANTS import ADV_IMAGE_FOLDER
from webapp.utils.CONSTANTS import ADV_DET_IMAGE_FOLDER
from webapp.utils.CONSTANTS import STYLE_IMAGE_TIE1
from webapp.utils.utils import get_image_paths
from attack import main as attack_main
from PyTorchYOLOv3.detect import DetectorYolov3

if __name__ == '__main__':

    image_paths = get_image_paths(STOP_SIGNS_TRAIN_FOLDER)
    yolov3 = DetectorYolov3(show_detail=False, tiny=True)
    count = 1

    for image_path in image_paths:
        basename = os.path.basename(image_path)
        attack_main(
            item=image_path,
            detectorYolov3=yolov3,
            STYLE_IMAGE=STYLE_IMAGE_TIE1,
            MASK_IMAGE=MASK_IMAGE_FOLDER.format(os.path.splitext(basename)[0]),
            PATCH_PATH=os.path.join(PATCH_IMAGE_FOLDER, basename),
            ADV_PATH=os.path.join(ADV_IMAGE_FOLDER, basename),
            DET_PATH=os.path.join(ADV_DET_IMAGE_FOLDER, basename)
        )

        # command = [
        #     sys.executable, "attack.py",
        #     "--img", image_path,
        #     "--style", "None",
        #     "--mask", MASK_IMAGE_FOLDER.format(os.path.splitext(basename)[0]),
        #     "--patch", os.path.join(PATCH_IMAGE_FOLDER, basename),
        #     "--adv", os.path.join(ADV_IMAGE_FOLDER, basename),
        #     "--det", os.path.join(ADV_DET_IMAGE_FOLDER, basename),
        #     "--call", "1"]
        # try:
        #     results = subprocess.check_output(command).decode()
        #     print("输出结果: ", results)
        #     left_pos = results.find('[', results.index('return'))
        #     right_pos = results.find(']', results.index('return'))
        #     paths = eval(results[left_pos:right_pos + 1])
        #     print(paths)
        # except subprocess.CalledProcessError as e:
        #     print(e.output.decode())

        print("================================================")
        print(f"第{count}次已完成")
        print("================================================")
        count += 1
