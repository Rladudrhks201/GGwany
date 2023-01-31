import torch
from torch.utils.data import Dataset
import os
import glob
import cv2


class custom_dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = glob.glob(os.path.join(file_path, '*', '*.png'))
        self.transform = transform
        self.list_label = os.listdir(file_path)
        self.list_label.sort()

    def __getitem__(self, index):
        img_path = self.file_path[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        temp_label = img_path.split('\\')[-2]
        label = self.list_label.index(temp_label)
        label = torch.tensor(int(label))

        if self.transform:
            img = self.transform(image=img)['image']

        return img, label

    def __len__(self):
        return len(self.file_path)



