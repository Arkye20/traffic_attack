import os
from PyTorchYOLOv3.detect import DetectorYolov3
from torchvision.utils import save_image
from style_transfer import load_img
from webapp.utils.utils import get_project_root
from webapp.utils.utils import get_image_paths

from webapp.utils.CONSTANTS import CLASS_NAMES
from webapp.utils.CONSTANTS import STOP_SIGNS_TRAIN_FOLDER
from webapp.utils.CONSTANTS import DET_IMAGE_FOLDER
from webapp.utils.CONSTANTS import IMAGE_SIZE
from webapp.utils.CONSTANTS import ATTACK_ID

if __name__ == '__main__':
    os.chdir(get_project_root())
    detectorYolov3 = DetectorYolov3(show_detail=False, tiny=True)
    image_paths = get_image_paths(STOP_SIGNS_TRAIN_FOLDER)

    for image_path in image_paths:
        save_path = os.path.join(DET_IMAGE_FOLDER, os.path.basename(image_path))
        loaded_image = load_img(image_path, IMAGE_SIZE)
        _, _, det, _, _ = detectorYolov3.detect(input_imgs=loaded_image, cls_id_attacked=ATTACK_ID)
        detected_img = detectorYolov3.plot(loaded_image, CLASS_NAMES, det, 0.5)
        save_image(detected_img, save_path)
