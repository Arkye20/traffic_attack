# 原始图片路径
STOP_SIGNS_TRAIN_FOLDER = 'webapp/images/stop_signs_train/'
CLEAR_IMAGE_FOLDER1 = 'webapp/images/clear_images/'
CLEAR_IMAGE_FOLDER2 = 'webapp/images/clear_images3/'

# gradcam保存路径
GRADCAM_SAVE_FOLDER = 'webapp/images/gradcam_images/'

# 下面路径拼接好图片文件名之后，前端读取展示，或者输入到函数中
MASK_IMAGE_FOLDER = 'webapp/images/gradcam_images/{}/mask.jpg'
PATCH_IMAGE_FOLDER = 'webapp/images/attack_images/patch/'
ADV_IMAGE_FOLDER = 'webapp/images/attack_images/adv_img/'
ADV_DET_IMAGE_FOLDER = 'webapp/images/attack_images/det_img/'
DET_IMAGE_FOLDER = 'webapp/images/attack_images/yolov3_clear_det/'

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

# 图片大小
IMAGE_SIZE = (416, 416)
# 攻击的类别，也就是stop sign
ATTACK_ID = 11

# 模型权重文件
YOLOv3_MODEL_WEIGHTS_PATH = 'yolov3/weights/yolov3.pt'
