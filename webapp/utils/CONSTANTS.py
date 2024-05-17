DEFAULT_SAVE_FOLDER = 'webapp/save'
# 原始图片路径
STOP_SIGN_TRAIN_FOLDER = 'webapp/images/stop_signs_train/'
STOP_SIGN_TEST_FOLDER = 'webapp/images/stop_signs_test/'
STOP_SIGN_ALL_FOLDER = 'webapp/images/stop_signs_all/'
STOP_SIGN_ALL_LABELS = 'webapp/images/stop_signs_label'
TRAIN2017_FOLDER = 'webapp/images/train2017'


TRAIN_ADV = 'webapp/images/attack_images/train_adv'
VAL_ADV = 'webapp/images/attack_images/val_adv'
TEST_ADV = 'webapp/images/attack_images/test_adv'

# gradcam保存路径
GRADCAM_SAVE_FOLDER = 'webapp/images/gradcam/'

# 下面路径拼接好图片文件名之后，前端读取展示，或者输入到函数中
MASK_IMAGE_FOLDER = DEFAULT_SAVE_FOLDER + '/{}/mask.jpg'
PATCH_IMAGE_FOLDER = DEFAULT_SAVE_FOLDER
ADV_IMAGE_FOLDER = DEFAULT_SAVE_FOLDER
ADV_PT_FOLDER = 'webapp/images/attack_images/adv_pt'
ADV_DET_IMAGE_FOLDER = DEFAULT_SAVE_FOLDER
DET_IMAGE_FOLDER = 'webapp/images/attack_images/yolov3_clear_det'

# 风格图片路径
STYLE_IMAGE_TIE1 = 'webapp/images/style_images/tie1.jpg'
STYLE_IMAGE_TIE2 = 'webapp/images/style_images/tie2.jpg'

# 图片分类 所有class
CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear',
               'hair drier', 'toothbrush']

IMAGE_SIZE = (416, 416)

ATTACK_ID = 11
DEFAULT_ATTACK_RANGE = 220

# 模型权重文件
YOLOv3_MODEL_WEIGHTS_PATH = 'yolov3/weights/yolov3.pt'


CFG_FILE = "PyTorchYOLOv3/config/yolov3-tiny.cfg"
WEIGHT_FILE = "PyTorchYOLOv3/weights/yolov3-tiny.weights"
