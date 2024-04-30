import os

from webapp.utils.utils import get_project_root

os.chdir(get_project_root())
from torchvision.utils import save_image

import gradio as gr
from PyTorchYOLOv3.detect import DetectorYolov3
from attack import main as attack_main
from style_transfer import load_img
from webapp.utils.CONSTANTS import ADV_DET_IMAGE_FOLDER
from webapp.utils.CONSTANTS import ADV_IMAGE_FOLDER
from webapp.utils.CONSTANTS import ATTACK_ID
from webapp.utils.CONSTANTS import CLASS_NAMES
from webapp.utils.CONSTANTS import DET_IMAGE_FOLDER
from webapp.utils.CONSTANTS import GRADCAM_SAVE_FOLDER
from webapp.utils.CONSTANTS import IMAGE_SIZE
from webapp.utils.CONSTANTS import MASK_IMAGE_FOLDER
from webapp.utils.CONSTANTS import PATCH_IMAGE_FOLDER
from webapp.utils.CONSTANTS import STYLE_IMAGE_TIE1
from webapp.utils.CONSTANTS import YOLOv3_MODEL_WEIGHTS_PATH
from webapp.utils.utils import delete_files
from yolov3.main_gradcam import args as gradcam_args
from yolov3.main_gradcam import main as gradcam_main
from yolov3.main_gradcam import names as gradcam_names
from yolov3.models.yolo_detector import YOLOV3TorchObjectDetector

css = """
.column2 {
    padding: 100px 10px 0px;
}
.column34{
    padding: 40px 10px 0px;
}
.column2_btn{
    padding: 112px 0px 0px;
}
"""


# 原图检测 使用PyTorchYOLOv3模型
def wrapped_predict_image(image):
    global yolov3
    delete_files(DET_IMAGE_FOLDER)
    save_path = os.path.join(DET_IMAGE_FOLDER, os.path.basename(image))
    loaded_image = load_img(image, IMAGE_SIZE)
    _, _, det, _, _ = yolov3.detect(input_imgs=loaded_image, cls_id_attacked=ATTACK_ID)
    detected_img = yolov3.plot(loaded_image, CLASS_NAMES, det, 0.5)
    save_image(detected_img, save_path)
    return save_path


# 生成攻击区域gradcam
def wrapped_gradcam_main(image, attack):
    global gradcam_model
    delete_files(GRADCAM_SAVE_FOLDER)
    _, save_dir = gradcam_main(img_path=image, model=gradcam_model, SAVE_DIR=GRADCAM_SAVE_FOLDER, attack_range=attack)
    result_img = os.path.join(save_dir, "15_0.jpg")
    mask = os.path.join(save_dir, "mask.jpg")
    return result_img, mask


def wrapped_attack_main(image, style=STYLE_IMAGE_TIE1):
    delete_files(ADV_IMAGE_FOLDER)
    delete_files(ADV_DET_IMAGE_FOLDER)
    delete_files(PATCH_IMAGE_FOLDER)
    delete_files(DET_IMAGE_FOLDER)

    basename = os.path.basename(image)
    adv_det_image_path, patch_image_path = attack_main(
        item=image,
        detectorYolov3=yolov3,
        STYLE_IMAGE=style,
        MASK_IMAGE=MASK_IMAGE_FOLDER.format(os.path.splitext(basename)[0]),
        PATCH_PATH=os.path.join(PATCH_IMAGE_FOLDER, basename),
        ADV_PATH=os.path.join(ADV_IMAGE_FOLDER, basename),
        DET_PATH=os.path.join(ADV_DET_IMAGE_FOLDER, basename)
    )
    return adv_det_image_path, patch_image_path


# 前端页面代码
with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="上传图像", type='filepath')
            style = gr.Image(label="上传风格图像", type='filepath')
            attack_param = gr.Slider(minimum=0, maximum=250, step=1, label="攻击区域", value=220)
        with gr.Column(elem_classes=["column2"]):
            output_img = gr.Image(interactive=False, show_label=False, height="30vh", type='filepath')
            with gr.Row(elem_classes=["column2_btn"]):
                detect_btn = gr.Button(value="原图检测")
                detect_btn.click(fn=wrapped_predict_image, inputs=input_img, outputs=output_img)
        with gr.Column(elem_classes=["column34"]):
            output_img_1 = gr.Image(interactive=False, show_label=False, type="filepath")
            output_img_2 = gr.Image(interactive=False, show_label=False, type="filepath")
            attack_btn = gr.Button(value="生成攻击区域")
            attack_btn.click(fn=wrapped_gradcam_main, inputs=[input_img, attack_param],
                             outputs=[output_img_1, output_img_2])
        with gr.Column(elem_classes=["column34"]):
            result_img_1 = gr.Image(interactive=False, label="对抗样本检测结果", type="filepath")
            result_img_2 = gr.Image(interactive=False, label="生成的补丁", type="filepath")
            result_btn = gr.Button(value="攻击结果")
            result_btn.click(fn=wrapped_attack_main, inputs=[input_img, style], outputs=[result_img_1, result_img_2])

if __name__ == '__main__':
    # 加载原图检测模型
    yolov3 = DetectorYolov3(show_detail=False, tiny=True)

    # 加载热力图模型
    input_size = (gradcam_args.img_size, gradcam_args.img_size)
    gradcam_model = YOLOV3TorchObjectDetector(
        YOLOv3_MODEL_WEIGHTS_PATH,
        gradcam_args.device,
        img_size=input_size,
        names=gradcam_names
    )

    demo.launch()
