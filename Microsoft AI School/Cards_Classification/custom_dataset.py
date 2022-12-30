import glob
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

random_seed = 77727
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


class custom_dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = glob.glob(os.path.join(file_path, '*', '*.jpg'))
        self.transform = transform
        self.class_names = os.listdir(file_path)
        self.class_names.sort()
        self.file_path.sort()
        self.labels = []

        for path in self.file_path:
            self.labels.append(self.class_names.index(path.split('\\')[2]))
        self.labels = np.array(self.labels)
        # print(self.labels)
    def __getitem__(self, index):
        image_path = self.file_path[index]
        img = cv2.imread(image_path)
        label = self.labels[index]
        label = torch.tensor(label)


        # print(label)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label

    def __len__(self):
        return len(self.file_path)



