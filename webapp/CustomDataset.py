import torch
from torch.utils.data import Dataset
import os
from webapp.utils.CONSTANTS import IMAGE_SIZE
from style_transfer import load_img

class AdversarialDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.root_dir, file_name)
        data = torch.load(file_path)
        return data



class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.root_dir, file_name)
        data = load_img(file_path, IMAGE_SIZE)
        return data
