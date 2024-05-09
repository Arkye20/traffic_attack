import os
import shutil


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


