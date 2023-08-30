import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import cv2 
import random


class ImgDataset(Dataset):
    def __init__(self, img_list, transform = None):
        super(ImgDataset, self).__init__()
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = T.ToTensor()(Image.open(img_path).convert('RGB')) #Added convert to RGB
        if self.transform:
            img = self.transform(img)
        return self.transform(img)
