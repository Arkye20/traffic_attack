import os.path
from tkinter import Tk, filedialog

from webapp.utils.CONSTANTS import *
from webapp.utils.utils import *

os.chdir(get_project_root())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision.utils import save_image

import gradio as gr
from PyTorchYOLOv3.detect import DetectorYolov3
from attack import main as attack_main
from style_transfer import load_img
from yolov3.main_gradcam import args as gradcam_args
from yolov3.main_gradcam import main as gradcam_main
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

SAVE_FOLDER = DEFAULT_SAVE_FOLDER
GRADCAM_METHOD = 'gradcam'
SAVE_PT = False


# 打开文件管理器，选择文件夹，返回文件夹路径
def on_browse(save_path):
    global SAVE_FOLDER
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()

    filename = filedialog.askdirectory()
    if filename:
        if os.path.isdir(filename):
            root.destroy()
            SAVE_FOLDER = str(filename)
            return str(filename)
        else:
            root.destroy()
            SAVE_FOLDER = str(filename)
            return str(filename)
    elif save_path is not None:
        return save_path
    else:
        filename = "未选择路径"
        root.destroy()
        return str(filename)


# 原图检测 使用PyTorchYOLOv3模型
def wrapped_predict_image(image):
    global yolov3
    print(SAVE_FOLDER)
    image_save_folder = create_folder(SAVE_FOLDER, image)
    copy_file(image, image_save_folder)

    image_save_path = os.path.join(image_save_folder, os.path.basename(image))
    loaded_image = load_img(image, IMAGE_SIZE).to(device)
    _, _, det, _, _ = yolov3.detect(input_imgs=loaded_image, cls_id_attacked=ATTACK_ID)
    detected_img = yolov3.plot(loaded_image, CLASS_NAMES, det, 0.5)
    save_image(detected_img, image_save_path)
    return image_save_path


# 生成攻击区域gradcam
def wrapped_gradcam_main(image, attack):
    global gradcam_model
    print(GRADCAM_METHOD)
    image_save_folder = create_folder(SAVE_FOLDER, image)
    gradcam_main(
        img_path=image,
        model=gradcam_model,
        SAVE_DIR=image_save_folder,
        attack_range=attack,
        gradcam_method=GRADCAM_METHOD
    )
    result_img = os.path.join(image_save_folder, f"15_0_{GRADCAM_METHOD}.jpg")
    mask = os.path.join(image_save_folder, f"mask_{GRADCAM_METHOD}.jpg")
    return result_img, mask


def wrapped_attack_main(image, style=STYLE_IMAGE_TIE1):
    basename = os.path.basename(image)
    image_save_folder = create_folder(SAVE_FOLDER, image)
    adv_det_image_path, patch_image_path = attack_main(
        item=image,
        detectorYolov3=yolov3,
        STYLE_IMAGE=style,
        MASK_IMAGE=SAVE_FOLDER + f"/{os.path.splitext(basename)[0]}/mask_{GRADCAM_METHOD}.jpg",
        PATCH_PATH=os.path.join(image_save_folder, 'patch.jpg'),
        ADV_PATH=os.path.join(image_save_folder, 'adv.jpg'),
        DET_PATH=os.path.join(image_save_folder, 'adv_det.jpg'),
        SAVE_PT=SAVE_PT
    )
    return adv_det_image_path, patch_image_path


def set_gradcam_method(method):
    global GRADCAM_METHOD
    if method == "Grad-CAM":
        GRADCAM_METHOD = 'gradcam'
    elif method == "Grad-CAM++":
        GRADCAM_METHOD = 'gradcampp'


def set_save_pt(save):
    global SAVE_PT
    SAVE_PT = save


# 前端页面代码
with gr.Blocks(css=css) as demo:
    with gr.Tab("对抗攻击算法"):
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
                result_btn.click(fn=wrapped_attack_main, inputs=[input_img, style],
                                 outputs=[result_img_1, result_img_2])
    with gr.Tab("设置"):
        with gr.Row():
            save_folder = gr.Textbox(value=os.path.abspath(SAVE_FOLDER), label="保存路径", scale=5,
                                     interactive=False)
            save_folder_btn = gr.Button("浏览文件", min_width=1)
            save_folder_btn.click(on_browse, inputs=save_folder, outputs=save_folder, show_progress="hidden")
        with gr.Row():
            with gr.Column():
                gradcam_method = gr.Radio(
                    choices=["Grad-CAM", "Grad-CAM++"],
                    label="Grad-CAM方法",
                    info="生成攻击区域步骤的方法",
                    container=True
                )
                gradcam_method.select(set_gradcam_method, inputs=gradcam_method)
            with gr.Column():
                save_pt = gr.Checkbox(label="是", container=True, info="保存对抗样本的张量")
                save_pt.select(set_save_pt, inputs=save_pt)

if __name__ == '__main__':
    # 加载原图检测模型
    yolov3 = DetectorYolov3(show_detail=False, tiny=True)

    # 加载热力图模型
    input_size = (gradcam_args.img_size, gradcam_args.img_size)
    gradcam_model = YOLOV3TorchObjectDetector(
        YOLOv3_MODEL_WEIGHTS_PATH,
        device,
        img_size=input_size,
        names=CLASS_NAMES
    )

    demo.launch(inbrowser=True)
