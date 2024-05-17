import os
import shutil

import torch


# 从任意py文件中获取当前项目的根路径
def get_project_root():
    current_path = os.path.abspath(os.path.dirname(__file__))
    while True:
        if ".gitignore" in os.listdir(current_path):
            return current_path
        current_path = os.path.dirname(current_path)


# 输入存放图片的文件夹路径，输出一个数组，里面包含文件夹下所有图片的路径
def get_image_paths(image_folder):
    image_paths = [os.path.join(image_folder, file) for file in os.listdir(image_folder)]
    return image_paths


# 输入文件夹路径，删除文件夹下所有的文件
def delete_files(folder):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

# 将模型对对抗样本的预测结果维度增加到正常预测结果，空余行补0
def equal_shape(clear_result, adv_result):
    results = torch.zeros(clear_result.shape).to("cuda" if torch.cuda.is_available() else "cpu")
    for i in adv_result:
        for j in range(clear_result.shape[0]):
            if i[-1] == clear_result[j][-1]:
                results[j] = i
                break
    return results


# 复制文件
def copy_file(src_file, dst_folder):
    file = os.path.basename(src_file)
    dst_file = os.path.join(dst_folder, file)
    shutil.copyfile(src_file, dst_file)

def create_folder(save_folder, src_file):
    file = os.path.splitext(os.path.basename(src_file))[0]
    folder = os.path.join(save_folder, file)
    try:
        os.makedirs(folder)
        return folder
    except FileExistsError:
        print("文件夹已存在，将不进行任何操作")
        return folder

