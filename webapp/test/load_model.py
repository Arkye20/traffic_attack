import os
from webapp.utils.utils import get_project_root
os.chdir(get_project_root())

from PyTorchYOLOv3.detect import DetectorYolov3

yolov3 = DetectorYolov3(show_detail=True, tiny=True)